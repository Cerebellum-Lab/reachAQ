"""DeepLabCut 3.x PyTorch backend, contract-compatible with DlcPoseModel.

Lives behind the PoseModel seam, so nothing above predict() changes and
CroppedPoseModel composes over it unchanged.

The runner contract here was established by running the installed DeepLabCut
3.0.1 on the rig rather than read out of its docstrings, which disagree with
its own code: inference() documents a "bodypart" key while the implementation
writes "bodyparts". Measured, for a batch of two 256x256 frames:

    runner.inference(images=[ndarray, ...])
      -> [{'bodyparts': ndarray(1, 14, 3)}, {'bodyparts': ndarray(1, 14, 3)}]

One dict per image; the key is plural; the value is a bare array rather than a
{"poses": ...} mapping; the shape is (individuals, bodyparts, 3) float32 with
columns x, y, likelihood. Dropping the individual axis yields the
(num_body_parts, 3) block DlcPoseModel returns, which is the shape
PoseAlgorithm and CroppedPoseModel index directly.

Precision is not assumed: it comes from detect_gpu_capability(), because FP16
measures 3-4x *slower* than FP32 on some cards in this fleet.
"""

import glob
import os
import typing

import numpy

from autotrainer.core.logging import get_verbose_logger

from ..gpu_capability import detect_gpu_capability
from ..pose_model import PoseModel

logger = get_verbose_logger(__name__)


class DlcTorchPoseModel(PoseModel):
    """
    A wrapper for using DeepLabCut pose inference through the PyTorch engine.
    """

    BODY_PARTS_KEY = 'bodyparts'
    PROJECT_PATH_KEY = 'project_path'
    TRAINING_FRACTION_KEY = 'TrainingFraction'
    SNAPSHOT_INDEX_KEY = 'snapshotindex'

    DEFAULT_BODY_PART_CATEGORY = 'default'

    # What the runner puts a pose array under.  Deliberately not the "bodypart"
    # spelling used by the DeepLabCut docstring; see the module note.
    POSE_KEY = 'bodyparts'

    def __init__(self, source: str, shuffle_index=1, training_index=0, batch_size=1,
                 device: typing.Optional[str] = None,
                 precision: typing.Optional[str] = None):
        super().__init__()

        self._source = source
        self._shuffle_index = shuffle_index
        self._training_index = training_index
        self._model_batch_size = batch_size
        self._requested_device = device
        self._requested_precision = precision

        self._training_fraction = 0.0
        self._model_folder = ""
        self._snapshot_path = ""
        self._device = ""
        self._precision = ""

        self._sys_configuration = None
        self._runner = None
        self._body_parts_count = 1

    @classmethod
    def pre_validate(cls, location: str):
        configuration_file = os.path.join(location, "config.yaml")
        if not os.path.isfile(configuration_file):
            raise FileNotFoundError(f"{configuration_file!r} does not exist")

    @classmethod
    def has_trained_snapshot(cls, location: str) -> bool:
        """
        Whether the project holds any trained PyTorch snapshot at all.

        Checked on the filesystem rather than through DeepLabCut, so the
        preflight in the parent process stays cheap and does not import torch
        or DeepLabCut just to find out that it cannot run.

        Deliberately broad: this answers "is this project trained for the
        PyTorch engine", not "is the configured shuffle trained". The precise
        answer needs the project configuration and belongs to load(), which
        raises with the shuffle and folder named. This exists so that the
        common case - the backend switched on a project that has only ever been
        trained under TensorFlow - is refused before a process is spawned,
        rather than surfacing as a child-process crash after preflight passed.
        """
        pattern = os.path.join(
            location, "dlc-models-pytorch", "*", "*", "train", "snapshot-*.pt"
        )
        return bool(glob.glob(pattern))

    def is_valid(self) -> bool:
        try:
            self.pre_validate(self._source)
        except Exception as err:
            logger.error("source: %s, validation failed: %s", self._source, err)
            return False
        if not self.has_trained_snapshot(self._source):
            # Imported here rather than at module scope: backend_selection is
            # this module's caller, and the name is only needed for a message.
            from ..backend_selection import POSE_BACKEND_ENV_VAR

            logger.error(
                "source: %s has no trained PyTorch snapshot. %s selects the PyTorch "
                "engine, but this project has no dlc-models-pytorch/**/train/"
                "snapshot-*.pt. Train a PyTorch shuffle, or unset the variable to "
                "use the TensorFlow engine.",
                self._source, POSE_BACKEND_ENV_VAR,
            )
            return False
        return True

    @property
    def snapshot_path(self) -> str:
        return self._snapshot_path

    @property
    def device(self) -> str:
        return self._device

    @property
    def precision(self) -> str:
        return self._precision

    def _load_body_parts(self, cfg_body_parts) -> None:
        """
        Populate the body part lists from a project configuration value.

        Mirrors DlcPoseModel: the ordering and the category grouping are part of
        the contract, because PoseAlgorithm builds its column MultiIndex from
        `body_parts` and gates on `body_part_categories`.  A project moved from
        the TensorFlow engine keeps the same config.yaml, so both engines have
        to read it the same way.
        """
        if isinstance(cfg_body_parts, dict):
            for cat in cfg_body_parts.keys():
                self._body_part_categories.append(cat)
                self._body_parts_by_category[cat] = list()

            for cat in self._body_part_categories:
                for part in cfg_body_parts[cat]:
                    self._body_parts.append(part)
                    self._body_parts_by_category[cat].append(part)
        elif isinstance(cfg_body_parts, list):
            self._body_parts_by_category[self.DEFAULT_BODY_PART_CATEGORY] = list()
            for part in cfg_body_parts:
                self._body_parts.append(part)
                self._body_parts_by_category[self.DEFAULT_BODY_PART_CATEGORY].append(part)
        else:
            raise TypeError(f"Unhandled cfg_body_parts type: {type(cfg_body_parts)}. value={cfg_body_parts}")

        self._body_parts_count = len(self._body_parts)

    def _find_snapshot(self, train_folder: str) -> str:
        """
        Return the snapshot selected by the project's `snapshotindex`.

        The PyTorch engine writes `snapshot-<epoch>.pt` where the TensorFlow
        engine wrote `snapshot-<iteration>.index`, and it may additionally write
        `snapshot-best-<epoch>.pt`.  The "best" files are excluded so the
        ordering stays numeric, matching the increasing-progress ordering that
        DlcPoseModel indexes into.
        """
        try:
            entries = os.listdir(train_folder)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The dataset for shuffle {self._shuffle_index} has not been trained or does not exist.\n Please "
                f"train it before using it to analyze videos.")

        snapshots = [
            name for name in entries
            if name.startswith("snapshot-") and name.endswith(".pt") and "best" not in name
        ]
        if not snapshots:
            raise FileNotFoundError(
                f"No snapshot-*.pt in {train_folder!r}. The dataset for shuffle {self._shuffle_index} has not been "
                f"trained with the PyTorch engine.")

        def epoch_of(name: str) -> int:
            return int(os.path.splitext(name)[0].split('-')[-1])

        snapshots = sorted(snapshots, key=epoch_of)
        snapshot_index = self._sys_configuration[self.SNAPSHOT_INDEX_KEY]
        return os.path.join(train_folder, snapshots[snapshot_index])

    def _build_runner(self, model_configuration_path: str, snapshot_path: str):
        """
        Construct the DeepLabCut pose inference runner.

        Isolated because this is the one call whose signature has moved across
        DeepLabCut 3.0.x releases.  `snapshot_path` has no default and must be
        passed.  Verified against 3.0.1.
        """
        from deeplabcut.pose_estimation_pytorch.apis.utils import get_pose_inference_runner

        return get_pose_inference_runner(
            model_configuration_path,
            snapshot_path=snapshot_path,
            batch_size=self._model_batch_size,
            device=self._device,
        )

    def load(self):
        from deeplabcut.core.engine import Engine
        from deeplabcut.utils import auxiliaryfunctions

        configuration_file = os.path.join(self._source, "config.yaml")
        logger.debug(f"using {configuration_file}")

        self._sys_configuration = auxiliaryfunctions.read_config(configuration_file)
        self._load_body_parts(self._sys_configuration[self.BODY_PARTS_KEY])
        logger.debug(f"loaded {self._body_parts_count} body parts")

        self._training_fraction = self._sys_configuration[self.TRAINING_FRACTION_KEY][self._training_index]

        # The PyTorch engine keeps its models under dlc-models-pytorch/, a
        # different tree from the TensorFlow dlc-models/, so the engine has to
        # be named here or this resolves to the wrong folder.
        self._model_folder = os.path.join(self._sys_configuration[self.PROJECT_PATH_KEY],
                                          str(auxiliaryfunctions.get_model_folder(self._training_fraction,
                                                                                  self._shuffle_index,
                                                                                  self._sys_configuration,
                                                                                  engine=Engine.PYTORCH)))

        train_folder = os.path.join(self._model_folder, "train")
        model_configuration_path = os.path.join(train_folder, "pytorch_config.yaml")
        if not os.path.isfile(model_configuration_path):
            raise FileNotFoundError(
                f"Model for shuffle {self._shuffle_index} and train fraction {self._training_fraction} does not "
                f"exist for the PyTorch engine ({model_configuration_path!r} is missing).")

        self._snapshot_path = self._find_snapshot(train_folder)
        logger.debug(f"using {self._snapshot_path} for {self._model_folder}")

        capability = detect_gpu_capability()
        self._device = self._requested_device or ("cuda" if capability is not None else "cpu")
        self._precision = self._requested_precision or (
            capability.preferred_precision if capability is not None else "fp32")

        logger.notice("DeepLabCut PyTorch engine: snapshot=%s device=%s precision=%s batch=%s parts=%s",
                      os.path.basename(self._snapshot_path), self._device, self._precision,
                      self._model_batch_size, self._body_parts_count)

        self._runner = self._build_runner(model_configuration_path, self._snapshot_path)

    def predict(self, frames) -> typing.List[numpy.ndarray]:
        if self._runner is None:
            raise RuntimeError("load() must be called before predict()")

        # The runner takes a sequence of individual frames rather than a stacked
        # array, and batches them internally up to batch_size.
        results = self._runner.inference(images=list(frames))

        # reshape rather than squeeze: squeeze would also collapse the body part
        # axis for a single-keypoint project.
        return [
            numpy.asarray(result[self.POSE_KEY], dtype="float").reshape(-1, self._body_parts_count, 3)[0]
            for result in results
        ]

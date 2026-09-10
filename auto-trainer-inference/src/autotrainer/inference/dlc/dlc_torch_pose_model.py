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

import yaml

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
        self._model_configuration = None
        self._runner = None
        self._body_parts_count = 1
        # Top-down nets crop to a detected box before estimating pose,
        # so predict() has to supply one. Set by load().
        self._top_down = False

        self._fast_path_enabled = False
        self._fast_shape = None
        self._fast_host = None
        self._fast_device_u8 = None
        self._fast_input = None
        self._fast_mean = None
        self._fast_std = None

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
    def supports_partial_batch(self) -> bool:
        """The runner batches internally, so a short batch costs less."""
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

        # "detector" is excluded as well as "best": a top-down net writes
        # snapshot-detector-<epoch>.pt beside the pose snapshot, and
        # epoch_of would read its trailing number, so a detector trained
        # longer than the pose model would sort above it and be selected.
        snapshots = [
            name for name in entries
            if name.startswith("snapshot-") and name.endswith(".pt")
            and "best" not in name and "detector" not in name
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

        with open(model_configuration_path) as handle:
            self._model_configuration = yaml.safe_load(handle) or {}
        self._top_down = self._model_configuration.get("method") == "td"
        reason = self._fast_path_reason()
        self._fast_path_enabled = reason is None
        if self._fast_path_enabled:
            logger.notice("live predict uses the lean path")
        else:
            logger.notice("live predict uses the DeepLabCut runner: %s", reason)


    # --- lean live path ---------------------------------------------------
    #
    # DeepLabCut's runner.inference() is built for analysing a video: it
    # transforms each image separately on the CPU, torch.stack()s them, keeps
    # parallel lists of contexts and batch sizes and re-slices them per call,
    # runs a per-image postprocessor and rebuilds a dict of dicts. Measured on
    # this rig that machinery costs 1.2 ms of a 6.9 ms predict() at batch 1.
    #
    # At 6.9 ms that is 16%. With a lighter backbone it is not: if compute
    # falls to ~1.9 ms the same 1.2 ms becomes 39% of the total, so the
    # overhead has to go for a fast backbone to be worth having.
    #
    # Live inference does not need any of it. The batch is a fixed number of
    # fixed-size camera frames, there is one head, there is no context and no
    # dynamic cropper. So this path reuses pinned host and device buffers,
    # normalises on the GPU, and calls the runner's predict() directly.
    #
    # The preprocessing it replaces is exactly one step, read from the trained
    # config rather than assumed: albumentations Normalize with the ImageNet
    # mean and standard deviation and max_pixel_value 255. resize,
    # auto_padding and top_down_crop are all null for this project, so there
    # is nothing else to reproduce. _fast_path_reason() re-checks that on load
    # and refuses the fast path if the config ever says otherwise.

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def _fast_path_reason(self) -> typing.Optional[str]:
        """Return why the lean path cannot be used, or None if it can."""
        if self._runner is None:
            return "model is not loaded"
        if not str(self._device).startswith("cuda"):
            return f"device is {self._device!r}"
        if getattr(self._runner, "dynamic", None) is not None:
            return "a dynamic cropper is configured"

        inference_cfg = (self._model_configuration or {}).get("data", {})
        inference_cfg = inference_cfg.get("inference", {}) or {}
        if not inference_cfg.get("normalize_images", False):
            return "normalize_images is off"
        # Anything that changes geometry or scaling would have to be
        # reproduced here, and silently getting it wrong would move every
        # coordinate. Refuse instead.
        for key in ("resize", "auto_padding", "top_down_crop", "longest_max_size",
                    "crop_sampling", "collate"):
            if inference_cfg.get(key):
                return f"{key} is configured"
        if inference_cfg.get("scale_to_unit_range") or inference_cfg.get("grayscale"):
            return "an unsupported scaling transform is configured"
        heads = (self._model_configuration or {}).get("model", {}).get("heads", {})
        if set(heads) - {"bodypart"}:
            return f"more than the bodypart head is configured: {sorted(heads)}"
        return None

    def _ensure_fast_buffers(self, frames):
        """Allocate the reusable host and device buffers once per shape."""
        import torch

        shape = tuple(frames.shape)
        if self._fast_shape == shape:
            return
        batch, height, width, channels = shape
        # Pinned so the host to device copy can be asynchronous and does not
        # pay a staging copy inside the driver on every frame.
        self._fast_host = torch.empty(
            shape, dtype=torch.uint8, pin_memory=True,
        )
        self._fast_device_u8 = torch.empty(
            shape, dtype=torch.uint8, device=self._device,
        )
        self._fast_input = torch.empty(
            (batch, channels, height, width),
            dtype=torch.float32, device=self._device,
        )
        self._fast_mean = torch.tensor(
            self.IMAGENET_MEAN, dtype=torch.float32, device=self._device,
        ).view(1, channels, 1, 1)
        self._fast_std = torch.tensor(
            self.IMAGENET_STD, dtype=torch.float32, device=self._device,
        ).view(1, channels, 1, 1)
        self._fast_shape = shape

    def _fast_predict(self, frames) -> typing.List[numpy.ndarray]:
        import torch

        self._ensure_fast_buffers(frames)
        self._fast_host.copy_(torch.from_numpy(frames))
        self._fast_device_u8.copy_(self._fast_host, non_blocking=True)

        # NHWC uint8 to NCHW float, normalised, into the preallocated tensor.
        # permute is a view, so the only write is the one into _fast_input.
        staged = self._fast_device_u8.permute(0, 3, 1, 2)
        torch.div(staged.to(torch.float32), 255.0, out=self._fast_input)
        self._fast_input.sub_(self._fast_mean).div_(self._fast_std)

        with torch.inference_mode():
            raw = self._runner.predict(self._fast_input)

        part_count = self._body_parts_count
        poses = []
        for item in raw:
            block = numpy.asarray(item["bodypart"]["poses"], dtype="float")
            poses.append(block.reshape(-1, part_count, 3)[0])
        return poses

    def predict(self, frames) -> typing.List[numpy.ndarray]:
        if self._runner is None:
            raise RuntimeError("load() must be called before predict()")

        if self._fast_path_enabled:
            return self._fast_predict(numpy.ascontiguousarray(frames))

        # The runner takes a sequence of individual frames rather than a stacked
        # array, and batches them internally up to batch_size.
        images = list(frames)
        if self._top_down:
            # A full-frame box, not the trained detector. These frames are
            # already the ROI the camera is cropped to and hold one animal,
            # so the box is the identity crop; running the detector would
            # add its latency and its failure modes to the closed loop.
            images = [
                (frame, {"bboxes": numpy.array(
                    [[0, 0, frame.shape[1], frame.shape[0]]], dtype=float)})
                for frame in images
            ]
        results = self._runner.inference(images=images)

        # reshape rather than squeeze: squeeze would also collapse the body part
        # axis for a single-keypoint project.
        return [
            numpy.asarray(result[self.POSE_KEY], dtype="float").reshape(-1, self._body_parts_count, 3)[0]
            for result in results
        ]

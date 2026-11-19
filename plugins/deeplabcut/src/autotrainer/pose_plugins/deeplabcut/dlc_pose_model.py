import logging
import os
import typing

import numpy

from autotrainer.inference.pose_model import PoseModel

logger = logging.getLogger(__name__)


class DlcPoseModel(PoseModel):
    """
    A wrapper for using DeepLabCut pose inference.
    """

    BODY_PARTS_KEY = 'bodyparts'
    PROJECT_PATH_KEY = 'project_path'
    TRAINING_FRACTION_KEY = 'TrainingFraction'
    SNAPSHOT_INDEX_KEY = 'snapshotindex'
    INIT_WEIGHTS_KEY = 'init_weights'
    BATCH_SIZE_KEY = 'batch_size'

    DEFAULT_BODY_PART_CATEGORY = 'default'

    def __init__(self, source: str, shuffle_index=1, training_index=0, batch_size=1):
        super().__init__()

        self._source = source
        self._shuffle_index = shuffle_index
        self._training_index = training_index
        self._model_batch_size = batch_size
        self._training_fraction = 0.0
        self._model_folder = ""
        self._snapshots = None

        self._sys_configuration = None
        self._model_configuration = None
        self._model_session = None
        self._model_inputs = None
        self._model_outputs = None
        self._body_parts_count = 1

        self._predict = None

    @classmethod
    def pre_validate(cls, location: str):
        configuration_file = os.path.join(location, "config.yaml")
        if not os.path.isfile(configuration_file):
            raise FileNotFoundError(f"{configuration_file!r} does not exist")
        # eventual to-do : could extend check with try load config and further sub-checks,
        # like

    def is_valid(self) -> bool:
        try:
            self.pre_validate(self._source)
        except Exception as err:
            logger.error("source: %s, validation failed: %s", self._source, err)
            return False
        return True

    def load(self):
        from deeplabcut.pose_estimation_tensorflow.config import load_config
        from deeplabcut.pose_estimation_tensorflow.core import predict
        from deeplabcut.utils import auxiliaryfunctions

        configuration_file = os.path.join(self._source, "config.yaml")
        logger.debug(f"using {configuration_file}")

        self._predict = predict
        self._sys_configuration = auxiliaryfunctions.read_config(configuration_file)

        cfg_body_parts = self._sys_configuration[self.BODY_PARTS_KEY]
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
        logger.debug(f"loaded {self._body_parts_count} body parts")

        self._training_fraction = self._sys_configuration[self.TRAINING_FRACTION_KEY][self._training_index]

        self._model_folder = os.path.join(self._sys_configuration[self.PROJECT_PATH_KEY],
                                          str(auxiliaryfunctions.get_model_folder(self._training_fraction,
                                                                                  self._shuffle_index,
                                                                                  self._sys_configuration)))

        model_configuration_path = os.path.join(self._model_folder, "test", "pose_cfg.yaml")

        logger.debug(f"using pose configuration {model_configuration_path}")

        try:
            self._model_configuration = load_config(model_configuration_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model for shuffle {self._shuffle_index} and train fraction {self._training_fraction} does not exist.")

        try:
            self._snapshots = numpy.array(
                [fn.split('.')[0] for fn in os.listdir(os.path.join(self._model_folder, "train")) if "index" in fn])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The dataset for shuffle {self._shuffle_index} has not been trained or does not exist.\n Please "
                f"train it before using it to analyze videos.\n Use the function 'train_network' to train the network "
                f"for this shuffle.")

        snapshot_index = self._sys_configuration[self.SNAPSHOT_INDEX_KEY]
        increasing_indices = numpy.argsort([int(m.split('-')[1]) for m in self._snapshots])
        self._snapshots = self._snapshots[increasing_indices]
        logger.debug(f"using {self._snapshots[snapshot_index]} for {self._model_folder}")

        self._model_configuration[self.INIT_WEIGHTS_KEY] = os.path.join(self._model_folder, 'train',
                                                                        self._snapshots[snapshot_index])

        self._model_configuration[self.BATCH_SIZE_KEY] = self._model_batch_size

        self._model_session, self._model_inputs, self._model_outputs = predict.setup_pose_prediction(
            self._model_configuration)

    def predict(self, frames) -> typing.List[numpy.ndarray]:
        pose_data = self._predict.getposeNP(frames, self._model_configuration, self._model_session, self._model_inputs,
                                            self._model_outputs)
        # NB:
        # we are assuming the `pose_data` array we get is a new one for each getposeNP() call.
        return [
            pose_data[frame, :].reshape(self._body_parts_count, 3)
            for frame in range(pose_data.shape[0])
        ]

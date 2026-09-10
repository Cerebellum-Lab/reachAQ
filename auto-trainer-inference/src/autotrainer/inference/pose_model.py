import typing

import numpy


class PoseModel:
    """
    An implementation-independent interface for the pose inference "model" definition.

    These properties and methods represent the API needed by applications to and this module to perform pose inference
    processing.

    All current implementations are structured in a way that can use this as a base class.  If that changes, this
    interface may be better suited as a Protocol.
    """

    def __init__(self):
        self._body_parts = []
        self._body_part_categories = []
        self._body_parts_by_category = {}

    @classmethod
    def pre_validate(cls, location: str):
        """Class level pre-validation for a model location"""
        raise RuntimeError("Must be implemented by subclass")

    @property
    def body_parts(self):
        return self._body_parts

    @property
    def body_part_categories(self):
        return self._body_part_categories

    @property
    def supports_partial_batch(self) -> bool:
        """
        Whether predict() accepts fewer frames than the model was sized for.

        False by default, which is the safe answer: the DeepLabCut TensorFlow
        graph fixes its batch dimension at build time, so a caller must pad to
        the model batch size. A backend that batches internally can say True
        and let the caller skip the padding, which is not a micro-optimisation:
        live inference pads two real camera frames up to six and then discards
        four results, so it pays batch-six compute for batch-two output.
        """
        return False

    def is_valid(self) -> bool:
        """
        Check if the model is valid.  This is called before loading the model.
        Returns:
              True if the model is valid.
        """
        return True

    def load(self) -> None:
        """
        Load the model and perform any required initialization.  It is assumed that when this method returns, it is
        safe to call `predict()`.  Callers should also consider that this method may take a "long" time to complete
        by user-interface/interaction standards.
        """
        pass

    def predict(self, frames: numpy.ndarray) -> typing.List[numpy.ndarray]:
        """
        Return pose data for the given video frames.  Although there are no restrictions imposed here, the data is
        generally assumed to be interleaved camera frames of "batch_size" (camera count times the number of frames per
        camera) where the batch size is generally something that has been configured for or determined by the pose model
        implementation.

        Args:
            frames: a numpy array of video frame data with shape (frame_count, height, width, 3)

        Returns:
            a list of numpy arrays with shape (num_body_parts, 3) for each frame in the input frames.  The 3 values
            are x, y, and confidence.
        """
        return list(numpy.empty((0, 3), dtype="float"))

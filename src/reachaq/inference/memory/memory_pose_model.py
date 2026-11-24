import typing

import numpy

from ..pose_model import PoseModel


class MemoryPoseModel(PoseModel):
    """
    A "dumb" implementation of a pose model that returns random data.  This is used for testing purposes only.
    """
    def __init__(self, batchsize: int = 2):
        super().__init__()
        self._batchsize = batchsize
        self.use_random = False

    @classmethod
    def pre_validate(cls, location: str):
        if location != "":
            raise ValueError(f"Expected empty location but got {location!r}")

    def is_valid(self) -> bool:
        return True

    def load(self):
        # Note - this must be kept in sync w/real models.
        self.body_parts.append("Pellet")
        self.body_parts.append("RH_flat")
        self.body_parts.append("RH_spread")
        self.body_parts.append("RH_grab")
        self.body_parts.append("LH_flat")
        self.body_parts.append("LH_spread")
        self.body_parts.append("LH_grab")
        self.body_parts.append("Star")
        self.body_parts.append("Tongue_mid")
        self.body_parts.append("Tongue_tip")
        self.body_parts.append("Nose")
        self.body_parts.append("Triangle")
        self.body_parts.append("Mouth")
        self.body_parts.append("Diamond")

    def predict(self, frames: numpy.ndarray) -> typing.List[numpy.ndarray]:
        all_frames = list()

        for idx in range(self._batchsize):
            if self.use_random:
                data = numpy.random.rand(len(self.body_parts), 3)
            else:
                data = numpy.ones((len(self.body_parts), 3))

            # pose_predict normalizes on the assumption these values are in absolute frame size
            data[:, 0] *= frames.shape[2]
            data[:, 1] *= frames.shape[1]

            all_frames.append(data)

        return all_frames

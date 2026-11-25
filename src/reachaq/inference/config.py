import dataclasses
import pickle
from pathlib import Path
from typing import Dict, Any


@dataclasses.dataclass
class StereoParams:
    """Unrestricted parameters for calibration"""

    # not sure if good API,
    # or if we don't better keep the outer dict as well (having only 1 key atm)

    key_name: str
    matrix: Dict[str, Any]

    def as_pickle_dict(self):
        return {self.key_name: self.matrix}


def load_calib_stereo_params(file_path: Path) -> StereoParams:
    """Load calibration stereo params data from given pickle file path
    The pickle should be a dict with one key->value,
    and the returned data is that inner value.
    """
    with file_path.open('rb') as fh:
        data_dict = pickle.load(fh)
    keys = tuple(data_dict)
    if len(keys) != 1:
        raise RuntimeError("not 1 key dict: keys=%s", keys)
    k0 = keys[0]
    return StereoParams(key_name=k0, matrix=data_dict[k0])

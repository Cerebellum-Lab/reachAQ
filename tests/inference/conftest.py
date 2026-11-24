import pickle
from pathlib import Path

import yaml

import pytest

from reachaq.inference import calibration_FLIR
from reachaq.inference.config import load_calib_stereo_params
from reachaq.inference.analysis.prepare_jetson_data import DEFAULT_CAM_OFFSET_FILE_NAME
from reachaq.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from reachaq.inference.pose_algorithm import PoseAlgorithm


this_dir = Path(__file__).parent


@pytest.fixture
def calib_dir_path():
    return this_dir.joinpath(DEFAULT_3D_CALIB_DIR_NAME)


@pytest.fixture
def stereo_params(calib_dir_path):
    params = load_calib_stereo_params(calib_dir_path.joinpath("camera_matrix/stereo_params.pickle"))
    return params


@pytest.fixture
def calib_metadata(calib_dir_path):
    metadata_path = calib_dir_path.joinpath('calibration_userset.yaml')
    with metadata_path.open() as fh:
        return yaml.safe_load(fh)


@pytest.fixture
def cam_offsets(calib_dir_path):
    with calib_dir_path.joinpath(DEFAULT_CAM_OFFSET_FILE_NAME).open('rb') as fh:
        return pickle.load(fh)


@pytest.fixture
def pose_algo(calib_dir_path, stereo_params, calib_metadata, cam_offsets):
    calib_info = calibration_FLIR.get_calibration_info(calib_dir_path.as_posix())
    square_size = calib_info[0]
    return PoseAlgorithm(
        cam_names=["20241029_agx001_session002_left", "20241029_agx001_session002_right"],
        stereo_params=stereo_params,
        calib_metadata=calib_metadata,
        square_size=square_size,
        cam_offsets=cam_offsets,
    )

@pytest.fixture
def initialized_pose_algo(pose_algo):
    parts = ['Pellet', 'RH_flat', 'RH_spread', 'RH_grab', 'LH_flat', 'LH_spread', 'LH_grab',
             'Star', 'Tongue_mid', 'Tongue_tip', 'Nose', 'Triangle', 'Mouth', 'Diamond']
    pose_algo.initialize(parts)
    return pose_algo
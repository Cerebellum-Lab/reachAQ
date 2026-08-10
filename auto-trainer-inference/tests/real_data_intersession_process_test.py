import dataclasses
import functools
import os
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import patch

import h5py.h5f
import numpy
import numpy as np
import pandas as pd
import verboselogs

from autotrainer.core import Offset3DTuple, ProjectInfo, get_verbose_logger
from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME
from autotrainer.inference import PoseResponse, PoseLocation
from autotrainer.inference.analysis import (
    intersession_process,
    IntersessionResponse,
    prepare_jetson_data,
)
from autotrainer.core.reach_event import ReachEvent

import pytest

from top_fixtures import AlmostEqualFloat


logger = get_verbose_logger(__name__)

this_dir = Path(__file__).parent.resolve()
data_dir = this_dir.joinpath("data")
calib_dir = this_dir.joinpath(DEFAULT_3D_CALIB_DIR_NAME)


AEF = AlmostEqualFloat


def _init_t_presented_released(self: ProjectInfo):
    self.first_pellet_delivery_offset = 0
    self.first_pellet_presentation_offset = 0


class _ProjectInfo(ProjectInfo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _init_t_presented_released(self)


def patched_identify_dropped_frames(timestamp_file, frame_rate, *, cam_col_present_name: str) -> np.ndarray:
    """patched for test purpose, to allow reuse previous data"""
    name = "left" if "primary" in cam_col_present_name else "right"
    timestamp_file = timestamp_file.replace("frame_timing.csv", f"{name}_timestamps.txt")
    # Load timestamps from the file
    timestamps_df = pd.read_csv(timestamp_file, header=None,
                                names=['timestamp', 'fps', 'frame_when_ns', 'frame_perf_c', 'frame_id'])
    if len(timestamps_df) == 0:
        return np.asarray([], dtype=int)
    # NB: the timestamp is realtime, fps is fps, frame_when_ns is the camera frame "when/timestamp",
    # and the frame_perf_c is system perf_counter, which is common and the most precise we can use here.
    timestamps_ns = timestamps_df['frame_when_ns'].values  # Extract desired column
    timestamps_s = timestamps_ns / 1e9  # Convert seconds <-> nanoseconds

    # Calculate inter-frame intervals
    intervals = np.diff(timestamps_s)

    # Calculate the expected inter-frame interval
    expected_interval = 1.0 / frame_rate

    # Create a binary vector for the entire video length
    expected_frame_count = 1 + round((timestamps_ns[-1] - timestamps_ns[0]) * frame_rate / 1e9)
    if expected_frame_count != len(timestamps_df):
        logger.warning(
            "Correcting expected_frame_count from %s to %s ; file=%s ; timestamps: min=%s max=%s frame_rate=%s",
            expected_frame_count, len(timestamps_df), timestamp_file, timestamps_s.min(), timestamps_s.max(), frame_rate)
        expected_frame_count = len(timestamps_df)

    dropped_frame_vector = np.zeros(expected_frame_count, dtype=int)

    # Mark dropped frames
    current_frame = 0
    for i, interval in enumerate(intervals):
        # not sure:
        if current_frame > len(dropped_frame_vector) - 1:
            logger.verbose("breaking dropped_frame_vector loop")
            break
        dropped_frame_vector[current_frame] = 0  # Mark current frame as successful
        current_frame += 1
        if interval > 1.5 * expected_interval:  # Dropped frame threshold
            # Calculate how many frames were missed
            missed_count = int(round(interval / expected_interval)) - 1
            dropped_frame_vector[current_frame:current_frame + missed_count] = 1  # Mark missed frames
            current_frame += missed_count
            logger.warning("identified drop frame: i=%s iv=%s", i, interval)

    # Mark the last frame as successful
    if current_frame < len(dropped_frame_vector):
        dropped_frame_vector[current_frame] = 0

    return dropped_frame_vector


@pytest.fixture(autouse=True)
def _patch_identify_dropped_frames_for_previous(monkeypatch):
    monkeypatch.setattr(
        prepare_jetson_data,
        prepare_jetson_data.identify_dropped_frames.__qualname__,
        patched_identify_dropped_frames,
    )


@pytest.fixture(autouse=True)
def _patch_project_info_for_t_presented_released(ProjectInfo=ProjectInfo):  # ensure orig
    original_init = ProjectInfo.__init__

    # Define your custom setup logic
    def mocked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _init_t_presented_released(self)

    # Patch the __init__ method of the target class
    with patch.object(ProjectInfo, "__init__", autospec=True, side_effect=mocked_init):
        yield


ProjectInfo = _ProjectInfo  # for below top level ProjectInfo creation


def assert_deep_almost_equal(obj1, obj2, *, places=3, delta=None):
    if dataclasses.is_dataclass(obj1):
        obj1 = dataclasses.asdict(obj1)
    if dataclasses.is_dataclass(obj2):
        obj2 = dataclasses.asdict(obj2)
    if isinstance(obj1, float) or isinstance(obj2, float):
        if delta is not None:
            assert abs(obj1 - obj2) <= delta
            return True
        else:
            if obj1 == obj2 or obj2 == obj1:
                return True
            assert round(abs(obj1 - obj2), places) == 0
            return True
    elif isinstance(obj1, dict) and isinstance(obj2, dict):
        assert set(obj1) == set(obj2)
        for key in obj1:
            if not assert_deep_almost_equal(obj1[key], obj2[key], places=places, delta=delta):
                return False
        return True
    elif isinstance(obj1, (tuple, list)) and isinstance(obj2, (tuple, list)):
        assert len(obj1) == len(obj2)
        if len(obj1) != len(obj2):
            return False
        for item1, item2 in zip(obj1, obj2):
            if not assert_deep_almost_equal(item1, item2, places=places, delta=delta):
                return False
        return True
    else:
        assert obj1 == obj2
        return True


def assert_pose_response_almost_equal(
    r1: PoseResponse, r2: PoseResponse,
    *,
    loc_delta=5,  # pixels
    loc3d_delta=0.25,  # mm
    off_delta=0.5,  # mm
):
    assert r1.sequence == r2.sequence
    assert r1.parts_flags == r2.parts_flags
    assert_deep_almost_equal(r1.locations, r2.locations, delta=loc_delta)
    assert_deep_almost_equal(r1.locations_3d, r2.locations_3d, delta=loc3d_delta)
    assert_deep_almost_equal(r1.parts_3d_offsets, r2.parts_3d_offsets, delta=off_delta)


@pytest.mark.skipif(os.name != "posix", reason="disabled on non-posix")
def test_fp_and_xp_not_same(project_info, caplog):
    project_info.session = 2
    project_info.root = this_dir.joinpath("fp-and-xp-not-same").as_posix()
    project_info.device_id = "agx001"
    project_info.when = datetime(2025, 6, 23)
    project_info.camera_names = ("left", "right")
    caplog.set_level(verboselogs.VERBOSE)
    res = intersession_process(
        project_info,
        calib_dir=this_dir.joinpath(DEFAULT_3D_CALIB_DIR_NAME),
    )
    assert "Correcting expected_frame_count from " in caplog.text
    assert isinstance(res, IntersessionResponse)
    assert res.pellets_presented == 1
    assert res.food_consumed == 0
    assert res.successful_reaches == 0
    assert res.total_reaches == 0
    assert len(res.reach_events) == len(res.rh_max_vp_list) == 0
    assert len(res.other_events) == 1
    event = res.other_events[0]
    assert event.method == 'other'
    assert event.outcome == 'dropped'


@pytest.mark.skipif(os.name != "posix", reason="disabled on non-posix")
def test_agx001_20250806_59(project_info, caplog):
    project_info.session = 59
    project_info.root = this_dir.joinpath("index_error").as_posix()
    project_info.device_id = "agx001"
    project_info.when = datetime(2025, 8, 6)
    project_info.camera_names = ("left", "right")
    caplog.set_level(verboselogs.VERBOSE)
    # with pytest.raises(IndexError, match="fp and xp are not of the same length"):
    # previously was raising an index error, but is now fixed.
    res = intersession_process(
        project_info,
        calib_dir=calib_dir,
    )
    #
    expected = IntersessionResponse(
        rh_max_vp_list=[Offset3DTuple(AEF(0.54), AEF(-3.72), AEF(1.28))],
        reach_events=[
            ReachEvent(
                init=112,
                end=141,
                max=129,
                method='right_hand',
                outcome='dropped',
                delay_since_presented=AEF(0.746),
            )
        ],
        food_consumed=0,
        successful_reaches=0,
        pellets_presented=1,
        total_reaches=1,
    )
    assert_deep_almost_equal(res, expected)


agx001_20251015_15_expected_result = IntersessionResponse(
    rh_max_vp_list=[Offset3DTuple(AEF(-1.61), AEF(-2.43), AEF(1.25))],
    food_consumed=0, successful_reaches=0, pellets_presented=1, total_reaches=1,
    reach_events=[
        ReachEvent(
            delay_since_presented=AEF(0.69),
            end=164,
            init=104,
            max=147,
            outcome='dropped',
            method='right_hand',
        ),
    ]
)


@pytest.fixture
def agx001_20251015_15(project_info):
    project_info.root = data_dir.as_posix()
    project_info.session = 15
    project_info.device_id = "agx001"
    project_info.when = datetime(2025, 10, 15)
    project_info.camera_names = ("left", "right")
    return project_info


def test_intersession_process_agx001_20251015_15(agx001_20251015_15):
    res = intersession_process(
        agx001_20251015_15,
        calib_dir=calib_dir,
    )
    assert_deep_almost_equal(res, agx001_20251015_15_expected_result)


@pytest.mark.bench
def test_intersession_process_bench_agx001_20251015_15(agx001_20251015_15, benchmark):
    res =  benchmark(lambda: intersession_process(
        agx001_20251015_15,
        calib_dir=calib_dir,
    ))
    assert_deep_almost_equal(res, agx001_20251015_15_expected_result)


agx001_20260205_11_expected_result = IntersessionResponse(
    rh_max_vp_list=[Offset3DTuple(AEF(-0.40), AEF(-5.70), AEF(-0.22))],
    food_consumed=0, successful_reaches=0, pellets_presented=1, total_reaches=1,
    reach_events=[ReachEvent(
        delay_since_presented=0.0,
        end=33,
        init=0,
        max=17,
        method='right_hand',
        outcome='dropped')],
)


agx001_20260205_11_project_info = ProjectInfo(
    root=data_dir.as_posix(),
    device_id="agx001",
    session=11,
    when=datetime(2026, 2, 5),
    camera_names=("left", "right"),
)


def test_agx001_20260205_11():
    res = intersession_process(agx001_20260205_11_project_info, calib_dir=calib_dir)
    assert_deep_almost_equal(res, agx001_20260205_11_expected_result)


@pytest.mark.parametrize("frames_per_batch_per_cam,select_frames_method", [
    # (1, "last_one"),
    (2, "all_most_likely"),  # using most likely looks better here
    pytest.param(2, "last_one", marks=pytest.mark.xfail),
    # so the second pair of frames give missing L_Hand in locations:
    #
    #         elif isinstance(obj1, dict) and isinstance(obj2, dict):
    # >           assert set(obj1) == set(obj2)
    # E           AssertionError: assert {'Diamond', 'LH_flat', 'Triangle', 'Mouth', 'Pellet', 'Star', 'Nose'} \
    #               == {'L_Hand', 'Diamond', 'LH_flat', 'Mouth', 'Star', 'Triangle', 'Nose', 'Pellet'}
    # E             Full diff:
    # E               {
    # E                   'Diamond',
    # E                   'LH_flat',
    # E             -     'L_Hand',
    # E                   'Mouth',
    # E                   'Nose',
    # E                   'Pellet',
    # E                   'Star',
    # E                   'Triangle',
    # E               }

    # (3, "last_one"),
    (3, "all_most_likely"),
    # (5, "last_one"),
    (5, "all_most_likely"),
    (10, "all_most_likely"),
])
def test_pose_algo_process_frames_agx001_20251015_15(initialized_pose_algo, agx001_20251015_15, frames_per_batch_per_cam, select_frames_method):
    # if (frames_per_batch_per_cam, select_frames_method) != (3, "all_most_likely"):
    #     return
    pose_algo = initialized_pose_algo
    pairs_3d = [
        ('Triangle', 'Diamond'),
    ]
    # pose_algo.process_frames_select_frames_method = select_frames_method
    sp = agx001_20251015_15.get_session_path()
    fhs = []
    tables = []
    for cam in ("left", "right"):
        p = Path(sp.location).joinpath(f"{sp.prefix}_{cam}_raw2D.h5")
        fh = h5py.File(p, "r")
        fhs.append(fh)
        tables.append(fh["df_with_missing"]["table"])
    #
    cur_frame_idx = 0
    #
    frames_per_batch = frames_per_batch_per_cam
    all_frames = [[] for _ in range(len(fhs))]
    for table, frames in zip(tables, all_frames):
        for idx in range(cur_frame_idx, cur_frame_idx + frames_per_batch):
            f = table[idx][1]
            assert isinstance(f, numpy.ndarray)
            frames.append(f.reshape((pose_algo.part_count, -1)))
    cur_frame_idx += frames_per_batch

    res = pose_algo.process_frames(*all_frames, pairs_3d_offsets=pairs_3d)
    assert isinstance(res, PoseResponse)

    expected = PoseResponse(
        sequence=1,
        parts_flags=(
        {'Pellet': True, 'RH_flat': False, 'RH_spread': False, 'RH_grab': False, 'LH_flat': False, 'LH_spread': False,
         'LH_grab': False, 'Star': True, 'Tongue_mid': False, 'Tongue_tip': False, 'Nose': True, 'Triangle': True,
         'Mouth': True, 'Diamond': True},
        {'Pellet': True, 'RH_flat': False, 'RH_spread': False, 'RH_grab': False, 'LH_flat': True, 'LH_spread': False,
         'LH_grab': False, 'Star': True, 'Tongue_mid': False, 'Tongue_tip': False, 'Nose': True, 'Triangle': True,
         'Mouth': True, 'Diamond': True},
        {'Pellet': True, 'RH_flat': False, 'RH_spread': False, 'RH_grab': False, 'LH_flat': False, 'LH_spread': False,
         'LH_grab': False, 'Star': True, 'Tongue_mid': False, 'Tongue_tip': False, 'Nose': True, 'Triangle': True,
         'Mouth': True, 'Diamond': True}),
        locations=[
        {'Pellet': PoseLocation(index=0, x=107.19, y=124.12), 'Star': PoseLocation(index=7, x=157.90, y=98.02),
         'Nose': PoseLocation(index=10, x=81.19, y=89.28), 'Triangle': PoseLocation(index=11, x=100.96, y=154.61),
         'Mouth': PoseLocation(index=12, x=61.64, y=103.70), 'Diamond': PoseLocation(index=13, x=30.86, y=71.68)},
        {'Pellet': PoseLocation(index=0, x=84.20, y=127.76), 'LH_flat': PoseLocation(index=4, x=142.50, y=57.30),
         'Star': PoseLocation(index=7, x=136.81, y=121.13), 'Nose': PoseLocation(index=10, x=104.90, y=74.91),
         'Triangle': PoseLocation(index=11, x=65.31, y=152.20), 'Mouth': PoseLocation(index=12, x=106.31, y=91.57),
         'Diamond': PoseLocation(index=13, x=37.37, y=66.15), 'L_Hand': PoseLocation(index=-1, x=142.50, y=57.30)}],
        parts_3d_offsets={'Triangle': {'Diamond': Offset3DTuple(3.787355193687686, -10.697681145835073, 12.135112451315234)}},
        locations_3d={
            'Diamond': Offset3DTuple(4.474496892621784, -17.238484450625435, 9.131630112545981),
            'Triangle': Offset3DTuple(0.6871416989340975, -6.540803304790362, -3.0034823387692535),
            'Pellet': Offset3DTuple(-2.954824682334811, -7.975617408647363, 0.7642527423619532),
            'Star': Offset3DTuple(-11.670798738119466, -3.898139789569786, 3.753484237927467),
        },
        raw_loc_3d={
            'Diamond': Offset3DTuple(7.031130763628839, -2.0526320986506033, 23.23216593822239),
            'Triangle': Offset3DTuple(7.8992932916520004, 1.9640348717618656, 22.627585513107295),
            'Pellet': Offset3DTuple(8.860241076555408, 1.0752954692133985, 22.99038576300894),
            'Star': Offset3DTuple(11.376611800930563, 1.1454202017616602, 22.89788346589098),
        })

    assert_pose_response_almost_equal(res, expected)


@pytest.mark.bench
def test_pose_algo_process_frames_bench_agx001_20251015_15(pose_algo, agx001_20251015_15, benchmark):
    def proc():
        parts = ['Pellet', 'RH_flat', 'RH_spread', 'RH_grab', 'LH_flat', 'LH_spread', 'LH_grab',
                 'Star', 'Tongue_mid', 'Tongue_tip', 'Nose', 'Triangle', 'Mouth', 'Diamond']
        pairs_3d = [
            ('Diamond', 'Triangle'),
        ]
        sp = agx001_20251015_15.get_session_path()
        fhs = []
        tables = []
        for cam in ("left", "right"):
            p = Path(sp.location).joinpath(f"{sp.prefix}_{cam}_raw2D.h5")
            fh = h5py.File(p, "r")
            fhs.append(fh)
            tables.append(fh["df_with_missing"]["table"])
        cur_frame_idx = 0
        frames_per_batch = 3
        while True:
            all_frames = [[] for _ in range(len(fhs))]
            for table, frames in zip(tables, all_frames):
                for idx in range(cur_frame_idx, cur_frame_idx + frames_per_batch):
                    if idx > len(table) - 1:
                        break
                    f = table[idx][1]
                    assert isinstance(f, numpy.ndarray)
                    frames.append(f.reshape((len(parts), -1)))
            if any(len(frames) < frames_per_batch for frames in all_frames):
                break
            cur_frame_idx += frames_per_batch
            pose_algo.process_frames(*all_frames, pairs_3d_offsets=pairs_3d)
        return cur_frame_idx

    res = benchmark(proc)
    assert res == 762

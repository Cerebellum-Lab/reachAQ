"""The live numpy 3D path must agree with the DataFrame reference exactly.

`_compute_3d_locations` replaced a path that built DataFrames and called the
helpers shared with the offline pipeline. On one frame per camera that cost
1.529 ms p50 / 2.488 ms p99, and 85% of it was pandas construction and
per-part Python loops rather than arithmetic.

The rewrite is only legitimate because three reductions are exact at one frame
per camera, not approximations:

  * `triangulate_3d_step1`'s `min_cluster` masking never fires, because only
    parts already gated on the same `p_thresh` are passed in, so `low_conf` is
    all-False before the cluster loop runs and every point comes out `p=1`.
  * `reorient_and_center_step1`'s centering block needs ten frames and sets
    `center_len = 0` below that, so `cam_offsets` keeps its loaded values.
  * its reorient loop rebuilds three rotation matrices per body part from
    angles that vary by neither part, frame nor session, so the whole loop
    collapses to one constant 3x3 matrix.

These tests hold the two implementations against each other on the real
calibration, because that equivalence is the entire justification. They also
pin `p == 1`, since a future change that made it anything else would silently
drop parts the reach gate depends on.

Tolerances are relative rather than exact: the reference rotates, then flips
axes, then scales, while the live path folds all three into one matrix. Matrix
multiplication is not associative in floating point, so the two agree to
rounding, not bit for bit.
"""

import numpy
import pytest

from autotrainer.core import Offset3DTuple
from autotrainer.inference.pose_algorithm import PoseAlgorithm


CONFIDENT = 0.95
REJECTED = 0.0


def _detection(algorithm, rng, confident=None):
    """One frame per camera, shaped as `_compute_3d_locations` expects."""
    count = len(algorithm._measure_offset_parts)
    data = rng.uniform(20.0, 230.0, size=(2, 1, count, 3))
    likelihood = numpy.full(count, CONFIDENT)
    if confident is not None:
        likelihood = numpy.where(confident, CONFIDENT, REJECTED)
    data[:, :, :, 2] = likelihood
    return [data[0], data[1]]


def _assert_same(live, reference):
    assert sorted(map(str, live)) == sorted(map(str, reference))
    for part, value in live.items():
        assert isinstance(value, Offset3DTuple)
        assert tuple(value) == pytest.approx(tuple(reference[part]), rel=1e-9)


# --- the preconditions -------------------------------------------------------


def test_the_live_path_runs_for_the_real_calibration(initialized_pose_algo):
    """If this fails the optimization is silently never used."""
    assert initialized_pose_algo._live_3d_reason() is None


@pytest.mark.parametrize("attribute, expected", [
    ("_stereo_params", "no stereo params"),
    ("_cam_offsets", "no camera offsets"),
    ("_square_size", "no square size"),
])
def test_a_missing_precondition_falls_back(initialized_pose_algo, attribute,
                                           expected):
    setattr(initialized_pose_algo, attribute, None)
    initialized_pose_algo._live_3d_reason_cache = PoseAlgorithm._UNRESOLVED
    assert initialized_pose_algo._live_3d_reason() == expected


def test_a_mismatched_camera_pair_falls_back(initialized_pose_algo):
    initialized_pose_algo._cam_names = ["left", "right"]
    initialized_pose_algo._live_3d_reason_cache = PoseAlgorithm._UNRESOLVED
    assert initialized_pose_algo._live_3d_reason().startswith("no stereo entry")


def test_missing_camera_pos_falls_back(initialized_pose_algo):
    """The reference reads camLele unconditionally, so it must run and raise
    rather than the live path returning a different answer."""
    initialized_pose_algo._calib_metadata = {"camera_pos": None}
    initialized_pose_algo._live_3d_reason_cache = PoseAlgorithm._UNRESOLVED
    assert initialized_pose_algo._live_3d_reason() == (
        "no camera_pos in the calibration metadata")


# --- equivalence -------------------------------------------------------------


@pytest.mark.parametrize("seed", range(25))
def test_live_and_reference_agree(initialized_pose_algo, seed):
    detection = _detection(initialized_pose_algo, numpy.random.default_rng(seed))
    live_raw, live_3d = initialized_pose_algo._compute_3d_locations(detection)
    ref_raw, ref_3d = initialized_pose_algo._reference_3d_locations(detection)
    assert live_raw, "nothing was triangulated, so nothing was compared"
    _assert_same(live_raw, ref_raw)
    _assert_same(live_3d, ref_3d)


@pytest.mark.parametrize("seed", range(15))
def test_they_agree_when_only_some_parts_are_confident(initialized_pose_algo,
                                                       seed):
    """The confident subset changes which columns are triangulated at all."""
    rng = numpy.random.default_rng(seed)
    count = len(initialized_pose_algo._measure_offset_parts)
    confident = rng.integers(0, 2, count).astype(bool)
    if not confident.any():
        confident[0] = True
    detection = _detection(initialized_pose_algo, rng, confident=confident)
    live_raw, live_3d = initialized_pose_algo._compute_3d_locations(detection)
    ref_raw, ref_3d = initialized_pose_algo._reference_3d_locations(detection)
    assert len(live_raw) == int(confident.sum())
    _assert_same(live_raw, ref_raw)
    _assert_same(live_3d, ref_3d)


def test_no_confident_part_gives_no_locations(initialized_pose_algo):
    rng = numpy.random.default_rng(0)
    count = len(initialized_pose_algo._measure_offset_parts)
    detection = _detection(initialized_pose_algo, rng,
                           confident=numpy.zeros(count, dtype=bool))
    assert initialized_pose_algo._compute_3d_locations(detection) == ({}, {})
    assert initialized_pose_algo._reference_3d_locations(detection) == ({}, {})


def test_a_single_camera_uses_the_reference(initialized_pose_algo):
    """Triangulation needs a stereo pair; the reference warns and returns
    empty rather than indexing past the end."""
    detection = _detection(initialized_pose_algo, numpy.random.default_rng(0))
    with pytest.warns(UserWarning):
        assert initialized_pose_algo._compute_3d_locations(
            detection[:1]) == ({}, {})


# --- the assumption the rewrite rests on -------------------------------------


def test_every_triangulated_point_is_p_one(initialized_pose_algo):
    """`min_cluster` cannot fire at one frame per camera, so the reference's
    per-part confidence gate is always satisfied. If this ever stops being
    true the live path drops parts the reference would keep."""
    detection = _detection(initialized_pose_algo, numpy.random.default_rng(3))
    raw_df_3d, df_3d = initialized_pose_algo._handle_3d_triangulate(*detection)
    for frame in (raw_df_3d, df_3d):
        p_values = frame.xs("p", level="coords", axis=1).to_numpy()
        assert p_values.size
        assert (p_values == 1).all()


def test_the_live_path_is_wired_into_process_frames():
    """A revert would be silent: the result is identical, only slower."""
    import inspect

    source = inspect.getsource(PoseAlgorithm.process_frames)
    assert "_compute_3d_locations(" in source
    assert "_handle_3d_triangulate(" not in source, (
        "process_frames builds DataFrames again")

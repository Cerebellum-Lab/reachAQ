"""Waypoint segmentation and board detection, checked on synthetic calibration videos.

The rig's capture pauses the stage at each waypoint: the board sits still for a dwell,
then slides to the next waypoint. These tests render that capture from the synthetic
rig's true cameras at the full 1440x1080 sensor: 12 commanded positions including one
no-op move, 15 still frames per waypoint and 6 moving frames between waypoints, written
as mp4v at 40 fps. They check that every dwell is found and mapped to its distinct
waypoint, that the corners land on the truth in one consistent order, and that a missed
dwell or a board that cannot be found is reported rather than silently mis-tagged. With
logged dwells, the frames are taken by time instead, also where a flicker inside a dwell
splits it for the motion.

The truth is where render_board_frame draws the inner corners: the projection of each
waypoint's world corners with zero distortion.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest

from autotrainer.inference.calibration_FLIR import adjust_gamma
from autotrainer.inference.multires_calibration import extract as extract_module
from autotrainer.inference.multires_calibration.extract import (
    FROM_DWELLS,
    FROM_MOTION,
    WaypointAlignmentError,
    _middle_frames,
    _segments_from_motion,
    detect_board,
    distinct_waypoints,
    dwell_segments,
    extract_observations,
    find_stationary_segments,
    normalize_corner_order,
)
from autotrainer.inference.multires_calibration.fit import BoardSpec
from multires_synthetic import render_board_frame, synthetic_rig

IMAGE_SIZE = (1440, 1080)
FPS = 40
STILL_FRAMES = 15
MOVING_FRAMES = 6
# positions[DUPLICATE] repeats positions[DUPLICATE - 1]: a no-op move, so 12 positions
# hold 11 distinct waypoints.
DUPLICATE = 5
DISTINCT = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
DROPPED = 8  # the waypoint whose dwell the missing-dwell right video leaves out
BLANKED = 7  # the waypoint whose dwell the not-found right video blanks out
JITTERED = 9  # the waypoint whose right detections the inconsistency test perturbs
OUTLIER = 10  # the waypoint whose right detections the median test perturbs
BLANK_GREY = 100
# The "_flicker" videos blank one frame of waypoint FLICKERED's dwell, the FLICKER_AT-th.
# FLICKERED is the no-op move, so the flicker falls 17 frames into the 30-frame still run of
# waypoints 4 and 5 and splits it into two runs the motion finds (17 and 11 frames, both at
# least 8), as a flicker split dwells in the dark left video of the 2026-11-19 capture. The
# centred detection window (frames 12-16 of the run) misses it.
FLICKERED = DUPLICATE
FLICKER_AT = 2
# The wall-clock time of frame 0 in the synthetic timestamps.
T0 = 1_795_134_208.5
# Extra still frames at the start of the right_shifted video's first dwell, which puts
# every later dwell this many frames behind the left video's.
SHIFT = 10
# Grey-level sigma of the sensor-like noise in the "left_noisy" video. A real camera's
# frames are never exactly equal, so the clean videos alone would test segmentation only
# in the regime where p20 of the motion is 0.
SENSOR_NOISE = 2.0


@dataclass
class Capture:
    board: BoardSpec
    positions: List[np.ndarray]
    # "left", "right", "left_noisy", "left_flicker", "right_missing_dwell", "right_blank",
    # "right_shifted", "right_flicker"
    videos: Dict[str, Path]
    frame_count: int  # of the left, right and left_noisy videos
    still: Dict[int, Tuple[int, int]]  # waypoint -> [first, last + 1) still frame in those videos
    truth: Dict[Tuple[int, str], np.ndarray]  # (waypoint, camera) -> (48, 2) px


@pytest.fixture(scope="module")
def capture(tmp_path_factory) -> Capture:
    rig = synthetic_rig()
    board = rig.board
    positions = [o.motor_mm.copy() for o in rig.observations[:11]]
    positions.insert(DUPLICATE, positions[DUPLICATE - 1].copy())
    folder = tmp_path_factory.mktemp("multires_extract")
    names = {
        "left": ["left", "left_noisy", "left_flicker"],
        "right": ["right", "right_missing_dwell", "right_blank", "right_shifted", "right_flicker"],
    }
    videos = {name: folder / f"{name}.mp4" for cam in names for name in names[cam]}
    blank = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0]), BLANK_GREY, np.uint8)
    # A few fixed noise fields, cycled, keep the noisy video cheap to write.
    noise = np.random.default_rng(3).normal(0.0, SENSOR_NOISE, (7, IMAGE_SIZE[1], IMAGE_SIZE[0]))
    noise = noise.round().astype(np.int16)

    def write(name, writer, image, frame):
        if name == "left_noisy":
            image = np.clip(image.astype(np.int16) + noise[frame % len(noise)], 0, 255).astype(np.uint8)
        writer.write(image)

    still, truth = {}, {}
    for cam in ("left", "right"):
        model = rig.truth.cameras[cam]
        writers = {
            name: cv2.VideoWriter(str(videos[name]), cv2.VideoWriter_fourcc(*"mp4v"), FPS, IMAGE_SIZE, isColor=False)
            for name in names[cam]
        }
        assert all(w.isOpened() for w in writers.values())
        frame = 0
        for k, p in enumerate(positions):
            if k > 0:
                q = positions[k - 1]
                for j in range(1, MOVING_FRAMES + 1):
                    between = q + (p - q) * j / (MOVING_FRAMES + 1)
                    image = render_board_frame(model, rig.world_corners(between), IMAGE_SIZE, board)
                    for name, w in writers.items():
                        write(name, w, image, frame)
                    frame += 1
            X = rig.world_corners(p)
            image = render_board_frame(model, X, IMAGE_SIZE, board)
            truth[(k, cam)] = cv2.projectPoints(X, model.rvec, model.tvec, model.K, np.zeros(5))[0].reshape(-1, 2)
            still[k] = (frame, frame + STILL_FRAMES)
            if "right_shifted" in writers and k == 0:
                for _ in range(SHIFT):
                    writers["right_shifted"].write(image)
            for j in range(STILL_FRAMES):
                for name, w in writers.items():
                    if name == "right_missing_dwell" and k == DROPPED:
                        continue
                    blanked = (name == "right_blank" and k == BLANKED) or (
                        name.endswith("_flicker") and k == FLICKERED and j == FLICKER_AT)
                    write(name, w, blank if blanked else image, frame)
                frame += 1
        for w in writers.values():
            w.release()
    return Capture(board=board, positions=positions, videos=videos, frame_count=frame, still=still, truth=truth)


@pytest.fixture(scope="module")
def extraction(capture):
    videos = {"left": capture.videos["left"], "right": capture.videos["right"]}
    return extract_observations(videos, capture.positions, capture.board)


def corner_errors(capture, observations, cameras=("left", "right")) -> np.ndarray:
    return np.concatenate([
        np.linalg.norm(o.corners[cam] - capture.truth[(o.index, cam)], axis=1)
        for o in observations for cam in cameras
    ])


def test_distinct_waypoints_collapses_only_consecutive_repeats():
    a, b, c = np.array([1.0, 2.0, 3.0]), np.array([1.0, 4.0, 3.0]), np.array([0.0, 4.0, 3.0])
    positions = [a, a.copy(), b, b + 5e-7, c, a]
    # The first of a repeated run is kept; a return to an earlier position is a real move.
    assert distinct_waypoints(positions) == [0, 2, 4, 5]
    assert distinct_waypoints(positions, tol=1e-9) == [0, 2, 3, 4, 5]
    assert distinct_waypoints([]) == []


def assert_segments_lie_in_the_dwells(capture, segments, label):
    assert len(segments) == len(DISTINCT), label
    for (start, stop), k in zip(segments, DISTINCT):
        # The no-op move keeps the board still from waypoint 4's dwell through 5's.
        first = capture.still[k][0]
        last = capture.still[DUPLICATE][1] if k == DUPLICATE - 1 else capture.still[k][1]
        assert first <= start < stop <= last, (label, k, (start, stop), (first, last))
        assert stop - start >= STILL_FRAMES - 3, (label, k, (start, stop))
    # The last dwell runs to the end of the video, and is still counted.
    assert segments[-1][1] == capture.frame_count, label


def test_segments_match_distinct_waypoints(capture, extraction):
    assert distinct_waypoints(capture.positions) == DISTINCT
    assert [o.index for o in extraction.observations] == DISTINCT
    for o in extraction.observations:
        np.testing.assert_array_equal(o.motor_mm, capture.positions[o.index])
    for cam in ("left", "right"):
        assert extraction.segments[cam] == find_stationary_segments(capture.videos[cam])
        assert_segments_lie_in_the_dwells(capture, extraction.segments[cam], cam)


def test_segments_and_corners_survive_sensor_noise(capture):
    # Noise makes every frame differ, and mp4v's key frames make some differ more; the
    # dwells must still come out one per distinct waypoint, and the corners on the truth.
    result = extract_observations({"left": capture.videos["left_noisy"]}, capture.positions, capture.board)
    assert_segments_lie_in_the_dwells(capture, result.segments["left"], "left_noisy")
    assert result.rejected == []
    assert np.median(corner_errors(capture, result.observations, cameras=("left",))) < 0.3
    assert result.corner_scatter_px["left"] < 0.3


def test_detected_corners_match_truth(capture, extraction):
    assert extraction.rejected == []
    assert all(set(o.corners) == {"left", "right"} for o in extraction.observations)
    for o in extraction.observations:
        for corners in o.corners.values():
            assert corners.shape == (48, 2) and corners.dtype == np.float64
    errors = corner_errors(capture, extraction.observations)
    assert np.median(errors) < 0.3
    # A single waypoint in the other 180-degree order would be off by hundreds of px.
    assert errors.max() < 0.5
    for cam in ("left", "right"):
        assert 0.0 <= extraction.corner_scatter_px[cam] < 0.3


def test_reversed_corners_are_normalized(capture):
    board = capture.board
    for cam in ("left", "right"):
        c = capture.truth[(0, cam)]
        np.testing.assert_array_equal(normalize_corner_order(c[::-1], board), normalize_corner_order(c, board))
        # The object_points order is already normal here, so it is kept.
        np.testing.assert_array_equal(normalize_corner_order(c, board), c)
    # cv2's (N, 1, 2) float32 layout comes back as (N, 2) float64.
    as_cv2 = capture.truth[(0, "left")][::-1].reshape(-1, 1, 2).astype(np.float32)
    out = normalize_corner_order(as_cv2, board)
    assert out.shape == (48, 2) and out.dtype == np.float64
    np.testing.assert_allclose(out, capture.truth[(0, "left")], atol=1e-4)
    # A reference decides instead of x + y, in whichever order the reference is.
    c, later = capture.truth[(0, "left")], capture.truth[(3, "left")]
    for reference in (c, c[::-1]):
        for detected in (later, later[::-1]):
            out = normalize_corner_order(detected, board, reference=reference)
            assert np.dot(out[-1] - out[0], reference[-1] - reference[0]) > 0
    with pytest.raises(ValueError, match="reference"):
        normalize_corner_order(c, board, reference=c[:10])


def board_near_the_anti_diagonal(board: BoardSpec, w: int) -> np.ndarray:
    """(48, 2) corners of waypoint w, in object_points order, for a camera that sees the board
    turned so the diagonal from corner 0 to the last corner points within 1 degree of the
    image's (1, -1) direction, on alternate sides of it from one waypoint to the next."""
    obj = board.object_points()[:, :2] * 15.0  # 15 px squares
    turn = np.radians(-45.0 + (1.0 if w % 2 else -1.0)) - np.arctan2(obj[-1, 1], obj[-1, 0])
    R = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    return obj @ R.T + np.array([700.0 + 3 * w, 500.0 + 2 * w])


def test_order_stays_consistent_when_the_board_diagonal_is_near_the_anti_diagonal(capture, monkeypatch):
    board = capture.board
    truth = [board_near_the_anti_diagonal(board, w) for w in range(len(DISTINCT))]
    # Precondition: x + y alone gives this camera both orders across its waypoints.
    by_x_plus_y = [np.allclose(normalize_corner_order(t, board), t) for t in truth]
    assert any(by_x_plus_y) and not all(by_x_plus_y)

    calls = {"count": 0}

    def detector(gray, board, *, refine="subpix"):
        # Five frames per segment, in segment order: frame n belongs to segment n // 5.
        w = calls["count"] // 5
        calls["count"] += 1
        return normalize_corner_order(truth[w], board)

    monkeypatch.setattr(extract_module, "detect_board", detector)
    result = extract_observations({"left": capture.videos["left"]}, capture.positions, board)
    assert calls["count"] == 5 * len(DISTINCT) and result.rejected == []
    # The first waypoint's x + y order is kept for every waypoint.
    first_reversed = not by_x_plus_y[0]
    for o, t in zip(result.observations, truth):
        np.testing.assert_allclose(o.corners["left"], t[::-1] if first_reversed else t)


def read_grey_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    try:
        for _ in range(index + 1):
            ok, frame = cap.read()
            assert ok
    finally:
        cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


@pytest.mark.parametrize("refine", ["subpix", "legacy"])
def test_detect_board_normalises_whichever_end_the_detector_starts_from(capture, monkeypatch, refine):
    # The 6x8 board looks the same turned 180 degrees, and which end cv2 starts from
    # depends on the pose. On this pose it starts from the end nearer the image origin, so
    # its output is reversed here to stand in for the other choice.
    gray = read_grey_frame(capture.videos["left"], capture.still[0][0] + 7)
    direct = detect_board(gray, capture.board, refine=refine)
    real = cv2.findChessboardCorners

    def reversed_order(image, pattern_size, *args, **kwargs):
        found, corners = real(image, pattern_size, *args, **kwargs)
        return found, (corners[::-1].copy() if found else corners)

    monkeypatch.setattr(extract_module.cv2, "findChessboardCorners", reversed_order)
    flipped = detect_board(gray, capture.board, refine=refine)
    assert direct is not None and flipped is not None
    np.testing.assert_allclose(flipped, direct, atol=1e-6)
    assert np.median(np.linalg.norm(direct - capture.truth[(0, "left")], axis=1)) < 0.3


@pytest.mark.parametrize(
    "refine, owner, refiner", [("subpix", cv2, "cornerSubPix"), ("legacy", extract_module, "refine_corners")]
)
def test_corners_are_refined_on_the_image_each_refinement_expects(capture, monkeypatch, refine, owner, refiner):
    # subpix refines on the frame as recorded: the stretch's gamma curve reshapes the grey
    # levels across every edge it fits. legacy refines on today's preprocessed image, the
    # stretched copy after adjust_gamma(2), which its fixed Canny thresholds were tuned on.
    gray = read_grey_frame(capture.videos["left"], capture.still[0][0] + 7)
    if refine == "subpix":
        expected = gray
    else:
        expected = adjust_gamma(extract_module._detection_image(gray), gamma=2)
    real = getattr(owner, refiner)
    images = []

    def spy(image, *args, **kwargs):
        images.append(image)
        return real(image, *args, **kwargs)

    monkeypatch.setattr(owner, refiner, spy)
    assert detect_board(gray, capture.board, refine=refine) is not None
    assert images and all(np.array_equal(image, expected) for image in images)


def test_segmentation_stats_are_recorded(capture, extraction):
    for cam in ("left", "right"):
        stats = extraction.segmentation_stats[cam]
        assert set(stats) == {"p20", "p95", "tau", "moving_fraction", "n_frames"}
        assert stats["n_frames"] == capture.frame_count
        still = sum(stop - start for start, stop in extraction.segments[cam])
        assert stats["moving_fraction"] == pytest.approx(1.0 - still / capture.frame_count)
        assert 0.0 < stats["moving_fraction"] < 1.0
        assert 0.0 <= stats["p20"] < stats["tau"] < stats["p95"]
        # p20 of a clean video is 0, so tau is set by the noise floor.
        assert stats["tau"] == pytest.approx(np.sqrt(max(stats["p20"], extract_module._NOISE_FLOOR) * stats["p95"]))


def frame_times(capture) -> np.ndarray:
    """Column 0 of a timestamps file for the left, right and _flicker videos."""
    return T0 + np.arange(capture.frame_count) / FPS


def logged_dwells(capture) -> List[Tuple[float, float]]:
    """Each waypoint's (arrive, leave): its first and last still frame's time, as the capture
    stamps arrive after the settle and leave after the dwell."""
    return [(T0 + capture.still[k][0] / FPS, T0 + (capture.still[k][1] - 1) / FPS)
            for k in range(len(capture.positions))]


def test_logged_dwells_find_the_waypoints_where_motion_cannot(capture):
    # The 2026-11-19 capture: a flicker inside a dwell splits it, so motion finds one segment
    # too many and refuses; the logged dwells still put every waypoint on its own frames.
    videos = {"left": capture.videos["left_flicker"], "right": capture.videos["right_flicker"]}
    with pytest.raises(WaypointAlignmentError) as raised:
        extract_observations(videos, capture.positions, capture.board)
    assert str(raised.value) == f"left: expected {len(DISTINCT)} stationary segments, found {len(DISTINCT) + 1}"

    times = frame_times(capture)
    result = extract_observations(videos, capture.positions, capture.board,
                                  frame_times={"left": times, "right": times}, dwells=logged_dwells(capture))
    assert result.segments_from == FROM_DWELLS
    assert [o.index for o in result.observations] == DISTINCT
    # the no-op move's dwell joins the one before it
    expected = [(capture.still[k][0], capture.still[DUPLICATE if k == DUPLICATE - 1 else k][1]) for k in DISTINCT]
    for cam in ("left", "right"):
        assert result.segments[cam] == expected, cam
        assert result.segmentation_stats[cam]["motion_segments"] == len(DISTINCT) + 1, cam
    assert result.rejected == []
    errors = corner_errors(capture, result.observations)
    assert np.median(errors) < 0.3 and errors.max() < 0.5


def test_without_logged_dwells_the_segments_come_from_motion(extraction):
    assert extraction.segments_from == FROM_MOTION
    assert all("motion_segments" not in stats for stats in extraction.segmentation_stats.values())


def test_dwell_segments_take_frames_by_time():
    # Frame 3 was dropped, so frame times skip a step there; the later segments stay on
    # their own frames instead of shifting by one. Waypoints 1 and 2 are one position.
    times = np.array([0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    positions = [np.zeros(3), np.ones(3), np.ones(3), np.full(3, 2.0)]
    dwells = [(0.0, 0.15), (0.35, 0.5), (0.55, 0.6), (0.8, 1.0)]
    assert dwell_segments(times, positions, dwells) == [(0, 2), (3, 6), (7, 10)]
    with pytest.raises(ValueError, match="3 logged dwells for 4 waypoints"):
        dwell_segments(times, positions, dwells[:3])


@pytest.mark.parametrize("change, message", [
    ("missing", "right: no frame times for the logged dwells"),
    ("short", "left: {n} frame times for {m} frames in the video"),
    ("unordered", "left: the frame times do not increase"),
    ("brief", "left: waypoint 0's logged dwell holds 1 frames; at least 8 are needed"),
])
def test_logged_dwells_that_cannot_place_the_frames_raise(capture, change, message):
    videos = {"left": capture.videos["left"], "right": capture.videos["right"]}
    times = frame_times(capture)
    dwells = logged_dwells(capture)
    frame_times_by_cam = {"left": times, "right": times}
    if change == "missing":
        del frame_times_by_cam["right"]
    elif change == "short":
        frame_times_by_cam["left"] = times[:-1]
    elif change == "unordered":
        frame_times_by_cam["left"] = times[::-1]
    elif change == "brief":
        dwells[0] = (dwells[0][0], dwells[0][0])
    with pytest.raises(WaypointAlignmentError) as raised:
        extract_observations(videos, capture.positions, capture.board, frame_times=frame_times_by_cam, dwells=dwells)
    assert str(raised.value) == message.format(n=capture.frame_count - 1, m=capture.frame_count)


def test_frame_times_and_dwells_go_together(capture):
    videos = {"left": capture.videos["left"]}
    with pytest.raises(ValueError, match="together"):
        extract_observations(videos, capture.positions, capture.board, frame_times={"left": frame_times(capture)})
    with pytest.raises(ValueError, match="together"):
        extract_observations(videos, capture.positions, capture.board, dwells=logged_dwells(capture))


def test_cameras_out_of_step_raise_alignment_error(capture):
    # The cameras are hardware-synchronised, so the same dwell starts at the same frame in
    # both; ten extra frames in one camera's first dwell put every later dwell out of step.
    videos = {"left": capture.videos["left"], "right": capture.videos["right_shifted"]}
    left = find_stationary_segments(capture.videos["left"])
    right = find_stationary_segments(capture.videos["right_shifted"])
    assert len(left) == len(right) == len(DISTINCT)
    assert right[1][0] - left[1][0] == SHIFT
    with pytest.raises(WaypointAlignmentError) as raised:
        extract_observations(videos, capture.positions, capture.board)
    assert str(raised.value) == (
        f"segment 1: left starts at frame {left[1][0]} and right at frame {right[1][0]};"
        " synchronised cameras must agree within 3 frames"
    )


def test_missing_dwell_raises_alignment_error(capture):
    videos = {"left": capture.videos["left"], "right": capture.videos["right_missing_dwell"]}
    with pytest.raises(WaypointAlignmentError) as raised:
        extract_observations(videos, capture.positions, capture.board)
    assert str(raised.value) == f"right: expected {len(DISTINCT)} stationary segments, found {len(DISTINCT) - 1}"


def test_board_not_found_is_rejected_not_fatal(capture):
    videos = {"left": capture.videos["left"], "right": capture.videos["right_blank"]}
    result = extract_observations(videos, capture.positions, capture.board)
    assert [o.index for o in result.observations] == DISTINCT
    by_index = {o.index: o for o in result.observations}
    assert "right" not in by_index[BLANKED].corners
    assert "left" in by_index[BLANKED].corners
    assert result.rejected == [(BLANKED, "right", "not found")]
    assert all(set(o.corners) == {"left", "right"} for o in result.observations if o.index != BLANKED)
    assert np.median(corner_errors(capture, [o for o in result.observations if o.index != BLANKED])) < 0.3


def test_inconsistent_detections_are_rejected(capture, monkeypatch):
    real = extract_module.detect_board
    target = capture.truth[(JITTERED, "right")]
    seen = {"count": 0}

    def jittery(gray, board, *, refine="subpix"):
        corners = real(gray, board, refine=refine)
        # Shift every other detection of one waypoint in the right camera by 3 px in x and y,
        # so its frames disagree by far more than the 1 px allowed.
        if corners is not None and np.median(np.linalg.norm(corners - target, axis=1)) < 1.0:
            seen["count"] += 1
            if seen["count"] % 2:
                corners = corners + 3.0
        return corners

    monkeypatch.setattr(extract_module, "detect_board", jittery)
    videos = {"left": capture.videos["left"], "right": capture.videos["right"]}
    result = extract_observations(videos, capture.positions, capture.board)
    assert seen["count"] == 5
    assert result.rejected == [(JITTERED, "right", "inconsistent")]
    by_index = {o.index: o for o in result.observations}
    assert "right" not in by_index[JITTERED].corners and "left" in by_index[JITTERED].corners
    # The rejected waypoint's scatter is not part of the camera's figure.
    assert result.corner_scatter_px["right"] < 0.3


def test_corners_are_the_per_corner_median_of_the_frames(capture, extraction, monkeypatch):
    real = extract_module.detect_board
    target = capture.truth[(OUTLIER, "right")]
    seen = {"count": 0}

    def one_outlier(gray, board, *, refine="subpix"):
        corners = real(gray, board, refine=refine)
        # Move the first of one waypoint's five detections 0.8 px in x and y. Its per-corner
        # std of 0.45 px stays under the 1 px limit, so the waypoint is kept.
        if corners is not None and np.median(np.linalg.norm(corners - target, axis=1)) < 1.0:
            seen["count"] += 1
            if seen["count"] == 1:
                corners = corners + 0.8
        return corners

    monkeypatch.setattr(extract_module, "detect_board", one_outlier)
    result = extract_observations({"right": capture.videos["right"]}, capture.positions, capture.board)
    assert seen["count"] == 5 and result.rejected == []
    got = {o.index: o for o in result.observations}[OUTLIER].corners["right"]
    unperturbed = {o.index: o for o in extraction.observations}[OUTLIER].corners["right"]
    # A mean would move 0.16 px and the first frame 0.8 px; the median ignores the outlier.
    np.testing.assert_allclose(got, unperturbed, atol=0.02)


def test_legacy_refinement_is_passed_through_and_lands_on_the_truth(capture, extraction):
    videos = {"left": capture.videos["left"], "right": capture.videos["right"]}
    result = extract_observations(videos, capture.positions, capture.board, refine="legacy")
    assert result.rejected == []
    errors = corner_errors(capture, result.observations)
    assert np.median(errors) < 0.3
    assert errors.max() < 1.0
    # It is a different refinement, not the subpix one under another name.
    assert not np.allclose(
        result.observations[0].corners["left"], extraction.observations[0].corners["left"], atol=1e-3
    )


def test_unknown_refinement_is_refused(tmp_path):
    board = BoardSpec()
    with pytest.raises(ValueError, match="refine"):
        detect_board(np.zeros((64, 64), np.uint8), board, refine="bogus")
    # Refused before any video is opened.
    with pytest.raises(ValueError, match="refine"):
        extract_observations({"left": tmp_path / "absent.mp4"}, [np.zeros(3)], board, refine="bogus")


def test_detection_frames_are_the_centred_window_clipped_to_the_segment():
    # The ends of a segment are nearest the moves, where a slow start or stop can still
    # pass the motion threshold, so the frames used are the ones in its middle.
    assert _middle_frames((10, 30), 5) == range(17, 22)
    assert _middle_frames((10, 16), 5) == range(10, 15)
    assert _middle_frames((10, 13), 5) == range(10, 13)


def test_unreadable_video_raises(tmp_path):
    with pytest.raises(OSError, match="absent.mp4"):
        find_stationary_segments(tmp_path / "absent.mp4")


def test_featureless_frame_is_not_a_board():
    board = BoardSpec()
    with warnings.catch_warnings():
        # A uniform frame has no contrast to stretch; it must be refused, not divided by zero.
        warnings.simplefilter("error")
        for grey in (0, BLANK_GREY, 255):
            assert detect_board(np.full((1080, 1440), grey, np.uint8), board) is None


def test_static_video_is_one_segment(tmp_path):
    # A still scene: the 20th and 95th percentiles of the motion are about equal, so there
    # is nothing to separate and the whole video is one segment.
    y, x = np.mgrid[0:240, 0:320]
    scene = (128 + 60 * np.sin(x / 9.0) * np.cos(y / 7.0)).astype(np.uint8)
    path = tmp_path / "still.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 240), isColor=False)
    for _ in range(40):
        w.write(scene)
    w.release()
    assert find_stationary_segments(path) == [(0, 40)]
    assert find_stationary_segments(path, min_frames=41) == []


def _motion(labels: str, still: float = 0.2, moving: float = 10.0) -> np.ndarray:
    """m_t for t = 1 .. n-1 from one label per frame ('s' still, 'm' moving); frame 0's label is unused."""
    return np.array([still if c == "s" else moving for c in labels[1:]])


@pytest.mark.parametrize(
    "labels, expected",
    [
        # Frame 0 has no m_0 and takes frame 1's state, so a dwell from frame 0 starts at 0;
        # a dwell running to the last frame is flushed.
        ("s" * 10 + "m" * 10 + "s" * 12 + "m" * 8 + "s" * 9, [(0, 10), (20, 32), (40, 49)]),
        # Runs shorter than min_frames at either end (the recording started or stopped
        # inside a dwell) are dropped, as is one in the middle.
        ("s" * 5 + "m" * 10 + "s" * 12 + "m" * 5 + "s" * 7 + "m" * 6 + "s" * 6, [(15, 27)]),
        # Frame 1 moving makes frame 0 moving too.
        ("sm" + "s" * 8 + "m" * 10 + "s" * 3, [(2, 10)]),
    ],
)
def test_segments_from_motion_handle_the_ends(labels, expected):
    assert _segments_from_motion(_motion(labels), min_frames=8) == expected


def test_segments_from_motion_with_exactly_repeated_frames():
    # Decoded frames of a still board can repeat exactly (mp4v codes them as skipped blocks),
    # making p20 zero; the dwells must still be found.
    labels = "s" * 15 + ("m" * 6 + "s" * 15) * 5
    assert _segments_from_motion(_motion(labels, still=0.0), min_frames=8) == [(0, 15)] + [
        (15 + 21 * i + 6, 15 + 21 * (i + 1)) for i in range(5)
    ]


@pytest.mark.parametrize("still", [0.0, 0.3])
def test_segments_from_motion_static(still):
    n = 30
    rng = np.random.default_rng(1)
    motion = still * (1.0 + 0.1 * rng.random(n - 1))
    assert _segments_from_motion(motion, min_frames=8) == [(0, n)]
    assert _segments_from_motion(motion, min_frames=n + 1) == []

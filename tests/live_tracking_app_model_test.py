import threading
from types import SimpleNamespace
from unittest import mock

import pytest

from autotrainer.core import Offset3DTuple
from autotrainer.core.pose_elements import SceneElement
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.app_model_status import SessionRecordingStatus
from tools.acquisition.model.live_tracking_buffer import LiveTrackingBuffer, LiveTrackingSample
from tools.acquisition.model.session_boundary import SessionBoundary


# The camera's own frame count when recording began, taken from a real rig
# session (20260924 session002). Pose responses never carry this number: their
# source_frame_ids count from the first recorded frame.
ABSOLUTE_FIRST_RECORDED_FRAME = 17602
START_PERF = 500.0
FPS = 150


def _app():
    app = object.__new__(AppModel)
    # Without declared events, Events.__getattr__ answers any attribute the
    # stub forgot with a callable slot; this makes it raise instead.
    app.__events__ = ()
    app._reach_cameras = [
        SimpleNamespace(
            is_enabled=True,
            is_primary=True,
            active_config=SimpleNamespace(params={"fps": FPS}),
        )
    ]
    app._recording_session = SimpleNamespace(
        boundary=SessionBoundary(
            session_id="session",
            primary_camera="left",
            primary_frame_id=ABSOLUTE_FIRST_RECORDED_FRAME,
            start_perf_time=START_PERF,
            start_wall_time=1_790_000_000.0,
            camera_when=0.0,
        ),
        status=SessionRecordingStatus.RECORDING,
        token=lambda: SimpleNamespace(generation=1, session_id="session"),
    )
    app._behavior = SimpleNamespace(
        system_machine=SimpleNamespace(intersession=SimpleNamespace(frame_rate=30.0)),
        algorithm=SimpleNamespace(
            diamond_triangle_config=None,
            algo_paused=False,
            active_config=SimpleNamespace(
                session_control=SimpleNamespace(intertrial_analysis_enabled=False),
                pellet_delivery=SimpleNamespace(
                    triangle_pellet_expected_distance=7.0,
                    triangle_pellet_diff_too_far_threshold=3.0,
                ),
            ),
        ),
    )
    app._live_tracking = LiveTrackingBuffer(capacity=100)
    app._coordinates = SimpleNamespace(validate_pose=lambda *_args, **_kwargs: None)
    app._p_inference_live_begin = 0.0
    return app


def _pose_response(sequence, recorded_frame_indices, *, exposures=None):
    ids = tuple(recorded_frame_indices)
    rsp = SimpleNamespace(
        sequence=sequence,
        source_frame_ids=(ids, ids),
        perf_c=0.0,
        pellet_seen=True,
        locations_3d={SceneElement.Pellet: Offset3DTuple(1, 2, 3)},
        parts_3d_offsets={
            SceneElement.Triangle: {SceneElement.Pellet: Offset3DTuple(4, 5, 6)},
        },
    )
    if exposures is not None:
        rsp.source_frame_perf_times = (tuple(exposures), tuple(exposures))
    return rsp


def _closed_window(app, window_start, window_end):
    """Close a pellet window over the buffered samples and return what is persisted."""
    persisted = []
    app._pellet_cycles = mock.Mock()
    app._pellet_cycles.finish_active.return_value = SimpleNamespace(
        operation_id="operation",
        trial_id=1,
        attempt_id=1,
        capture_end_wall_time=1.0,
    )
    app._intertrial_lock = threading.Lock()
    app._trial_window_start = ("operation", window_start)
    app._session_data_recorder = mock.Mock()
    app._session_data_recorder.persist_trial_tracking.side_effect = (
        lambda _project, request: persisted.append(request)
    )
    app._session_data_recorder.trial_stream_references.return_value = ((), ())
    app._project_info = None
    app._notify_trial_protocol_state = lambda: None
    app._evaluate_automatic_stop_policy = lambda **_kwargs: None
    app._protocol_runner = SimpleNamespace(protocol_complete=False)

    app._complete_pellet_trial_window(window_end, close_reason="pellet cycle completed")

    (request,) = persisted
    return request.window


def test_pose_samples_are_timed_from_the_first_recorded_frame():
    app = _app()

    app._on_pose_response_ready(_pose_response(1, (1, 2, 3)))

    (sample,) = app._live_tracking.snapshot()
    assert sample.primary_frame_perf_times == pytest.approx(
        (500.0 + 1 / 150, 500.0 + 2 / 150, 500.0 + 3 / 150)
    )


def test_pellet_window_holds_the_frames_captured_during_it():
    app = _app()
    for sequence, first in enumerate((95, 100, 105)):
        app._on_pose_response_ready(_pose_response(sequence, range(first, first + 5)))

    window = _closed_window(app, 500.0 + 100 / 150, 500.0 + 109 / 150)

    assert (window.requested_start_frame_id, window.requested_end_frame_id) == (100, 109)
    assert window.observed_frames == 10
    assert window.missing_frame_ids == ()


# Frames per host second from the rig cameras, 103 ppm under the configured 150.
CAMERA_RATE = 149.9845


def _exposure(recorded_frame_index):
    return START_PERF + recorded_frame_index / CAMERA_RATE


def test_pellet_window_an_hour_into_a_session_holds_its_own_frames():
    # By frame 540000 the nominal rate places frames 0.37 s (56 frames) away
    # from when the camera exposed them.
    app = _app()
    for sequence, first in enumerate(range(539900, 540000, 5)):
        ids = range(first, first + 5)
        app._on_pose_response_ready(
            _pose_response(sequence, ids, exposures=[_exposure(i) for i in ids])
        )

    window = _closed_window(app, _exposure(539950), _exposure(539999))

    assert (window.requested_start_frame_id, window.requested_end_frame_id) == (539950, 539999)
    assert window.observed_frames == 50
    assert window.missing_frame_ids == ()


def test_a_sample_left_from_an_earlier_recording_does_not_set_the_rate():
    # The buffer is cleared when a behaviour session starts, not when a
    # recording stops; a sample timed before this boundary is not this
    # recording's evidence of its rate.
    app = _app()
    app._live_tracking.append(
        LiveTrackingSample(
            sequence=9, primary_frame_ids=(100_000,), primary_frame_perf_times=(400.0,),
            source_start_perf=400.0, source_end_perf=400.0, processing_perf=400.01,
            pellet_seen=False, locations_3d=(), offsets_3d=(),
        )
    )
    assert app._live_tracking_anchor(app._recording_session.boundary).frame_rate == 150.0

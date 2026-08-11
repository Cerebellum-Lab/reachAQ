from tools.acquisition.model.app_model_status import SessionRecordingStatus
from tools.acquisition.model.recording_session_controller import (
    RecordingSessionController,
)
from tools.acquisition.model.session_boundary import SessionBoundary


def test_prepare_record_resets_session_scoped_state():
    controller = RecordingSessionController(
        status=SessionRecordingStatus.READY,
        analysis_finished=True,
        analysis_duration_seconds=3.0,
        pending_end_perf=9.0,
        data_complete=False,
        data_errors=("old",),
        enabled_sources=({"id": "old"},),
    )

    controller.prepare_record({"camera.left": {"state": "ready"}})

    assert controller.analysis_finished is False
    assert controller.analysis_duration_seconds is None
    assert controller.pending_end_perf is None
    assert controller.boundary is None
    assert controller.data_complete is True
    assert controller.data_errors == ()
    assert controller.enabled_sources == ()
    assert controller.hardware_status_at_record["camera.left"]["state"] == "ready"


def test_owns_boundary_stream_completeness_and_abort_reset():
    controller = RecordingSessionController()
    controller.boundary = SessionBoundary(
        session_id="session001",
        primary_camera="left",
        primary_frame_id=0,
        start_perf_time=1.0,
        start_wall_time=2.0,
        camera_when=3.0,
    )
    controller.set_stream_result({
        "sessionComplete": False,
        "incompleteReasons": ("nidaq stopped",),
        "enabledSources": ({"id": "nidaq.barcode"},),
    })
    controller.add_data_error("metadata failed")

    assert controller.data_errors == ("nidaq stopped", "metadata failed")
    assert controller.transition(SessionRecordingStatus.ABORTING) is (
        SessionRecordingStatus.READY
    )
    controller.reset_after_abort()

    assert controller.boundary is None
    assert controller.data_complete is True
    assert controller.data_errors == ()
    assert controller.analysis_finished is True

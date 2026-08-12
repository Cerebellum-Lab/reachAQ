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


def test_abort_reset_invalidates_the_active_generation():
    controller = RecordingSessionController()
    _, token = controller.begin_record("session001", {})
    controller.transition(
        SessionRecordingStatus.ABORTING,
        expected=(SessionRecordingStatus.ARMING,),
        token=token,
    )

    controller.reset_after_abort()

    assert controller.token() is None
    assert not controller.is_current(token)


def test_generation_rejects_stale_transitions_and_boundaries():
    controller = RecordingSessionController()
    first = controller.begin_record("session001", {})
    assert first is not None
    _, first_token = first
    assert controller.transition(
        SessionRecordingStatus.ABORTING,
        expected=(SessionRecordingStatus.ARMING,),
        token=first_token,
    ) is SessionRecordingStatus.ARMING
    assert controller.transition(
        SessionRecordingStatus.READY,
        expected=(SessionRecordingStatus.ABORTING,),
        token=first_token,
    ) is SessionRecordingStatus.ABORTING

    second = controller.begin_record("session001", {})
    assert second is not None
    _, second_token = second
    assert second_token.generation == first_token.generation + 1
    assert not controller.is_current(first_token)
    assert controller.transition(
        SessionRecordingStatus.RECORDING,
        expected=(SessionRecordingStatus.ARMING,),
        token=first_token,
    ) is None
    assert not controller.set_boundary(
        SessionBoundary(
            session_id="session001",
            primary_camera="left",
            primary_frame_id=1,
            start_perf_time=1.0,
            start_wall_time=2.0,
            camera_when=3.0,
        ),
        first_token,
    )
    assert controller.status is SessionRecordingStatus.ARMING
    assert controller.boundary is None

    assert controller.set_boundary(
        SessionBoundary(
            session_id="session001",
            primary_camera="left",
            primary_frame_id=2,
            start_perf_time=4.0,
            start_wall_time=5.0,
            camera_when=6.0,
        ),
        second_token,
    )
    assert controller.boundary.primary_frame_id == 2


def test_begin_record_is_atomic_and_refuses_second_reservation():
    controller = RecordingSessionController()
    first = controller.begin_record("session001", {"camera.left": {}})
    second = controller.begin_record("session002", {"camera.left": {}})

    assert first is not None
    assert second is None
    assert controller.status is SessionRecordingStatus.ARMING
    assert controller.session_id == "session001"
    assert controller.metadata_generation_id == "session001-g1"


def test_end_action_is_reserved_once_and_completed_atomically():
    controller = RecordingSessionController()

    assert controller.reserve_end_action("pellet_home", {"ending": "stop"})
    assert not controller.reserve_end_action("pellet_home", {"ending": "abort"})
    controller.finish_end_action(
        "pellet_home", status="completed", commandToken="token-1",
    )

    assert controller.end_actions == ({
        "name": "pellet_home",
        "status": "completed",
        "ending": "stop",
        "commandToken": "token-1",
    },)

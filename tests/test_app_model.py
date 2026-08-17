import dataclasses
import math
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from autotrainer.core import (
    EventManager,
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
    SystemStatusMessageKind,
    SystemCommandKind,
)
from autotrainer.core.interfaces import RecordingEndingReason
from autotrainer.core.capture import CaptureProcessStatus
from autotrainer.core.configuration.persistence_configuration import PersistenceConfiguration
from autotrainer.behavior.behavior_algorithm import BehaviorAlgoStatus
from autotrainer.behavior.pellet_trial import (
    HardwareErrorKind,
    PelletTrialLedger,
    TrialOutcome,
)
from autotrainer.device import CanFailure, CanFailureKind, Tone, Target
from tools.acquisition.model.app_model import (
    app_status_to_api_app_mode,
    app_status_to_behavior_algo_status,
    protocol_state_to_api_training_mode,
)
from tools.acquisition.model.app_model_status import AppModelStatus, SessionRecordingStatus
from tools.acquisition.model.session_boundary import SessionBoundary
from tools.acquisition.model.intertrial_analysis import (
    IntertrialAnalysisRequest,
    IntertrialAnalysisResult,
    PelletMisplacement,
    PelletPresence,
    PelletStateEvidence,
    ReachTrajectory,
)
from tools.acquisition.model.live_tracking_buffer import (
    LiveTrackingSample,
    TrackingWindow,
)
from tools.acquisition.model.session_stop_policy import (
    SessionStopConfiguration,
    SessionStopDecision,
    SessionStopPolicy,
    SessionStopReason,
)
from tools.acquisition.model.subsystem_status import (
    SubsystemId,
    SubsystemState,
)
from tools.acquisition.model.trial_protocol_repository import (
    TrialProtocolRepository,
)
from tools.acquisition.model.trial_protocol_schedule import (
    ProtocolPatch,
    TrialProtocolDocument,
)

from autotrainer.api import ApiApplicationMode, ApiTrainingMode


class TestStatus:

    @pytest.mark.parametrize("app_model_status", list(AppModelStatus))
    def test_it_can_translate_to_api_app_mode(self, app_model_status: AppModelStatus) -> None:
        api_app_mode = app_status_to_api_app_mode(app_model_status)
        assert isinstance(api_app_mode, ApiApplicationMode)

    @pytest.mark.parametrize("app_model_status", list(AppModelStatus))
    def test_it_can_translate_to_behavior_status(self, app_model_status: AppModelStatus):
        algo_status = app_status_to_behavior_algo_status(app_model_status)
        if app_model_status in {AppModelStatus.CALIBRATION_3D, AppModelStatus.CALIBRATION_DCS}:
            assert algo_status is None
        else:
            assert isinstance(algo_status, BehaviorAlgoStatus)
            expected = (
                BehaviorAlgoStatus.RUNNING
                if app_model_status is AppModelStatus.RUNNING
                else BehaviorAlgoStatus.IDLE
            )
            assert algo_status is expected


@pytest.mark.parametrize(
    ("has_plan", "automatic", "expected"),
    (
        (False, False, ApiTrainingMode.MANUAL),
        (False, True, ApiTrainingMode.MANUAL),
        (True, False, ApiTrainingMode.MANUAL_WITH_PROTOCOL),
        (True, True, ApiTrainingMode.AUTOMATIC),
    ),
)
def test_api_training_mode_is_derived_from_protocol_state(
    has_plan,
    automatic,
    expected,
):
    plan = SimpleNamespace() if has_plan else None
    assert protocol_state_to_api_training_mode(
        plan,
        automatic_advance=automatic,
    ) is expected


def test_public_status_has_no_retired_alarm_tunnel_or_magnet_fields(app_model):
    status = app_model._make_api_system_status_payload()
    payload = dataclasses.asdict(status)

    assert payload["schema_version"] == 1
    assert "alarms" not in payload
    assert "tunnel_device" not in payload
    assert "magnet" not in json.dumps(payload).lower()
    assert payload["recording_state"] == "ready"
    assert set(payload["session_counts"]) == {
        "reaches", "presented", "success", "consumed"
    }


def test_automatic_protocol_advance_updates_live_runner(app_model):
    app_model.set_automatic_protocol_advance_enabled(True)

    session_control = app_model.behavior.algorithm.active_config.session_control
    assert session_control.automatic_protocol_advance_enabled is True
    assert app_model._protocol_runner.automatic_advance is True


def test_ordered_protocol_selection_is_separate_from_training_plan(
    app_model,
    tmp_path,
):
    app_model._trial_protocol_repository = TrialProtocolRepository(
        tmp_path / "trial_protocols"
    )
    document = TrialProtocolDocument(
        protocol_id="ordered-a",
        name="Ordered A",
        trial_count=3,
        defaults=ProtocolPatch.from_mapping({"enabled": True}),
    )

    saved = app_model.save_ordered_protocol(document)

    assert app_model.selected_ordered_protocol == saved
    assert app_model.trial_protocol_state["selected_protocol"] == {
        "protocol_id": "ordered-a",
        "name": "Ordered A",
        "revision": 1,
    }
    assert len(app_model.trial_protocol_rows) == 3
    context = app_model._current_trial_protocol_context(1)
    assert context["protocol_id"] == "ordered-a"
    assert context["protocol_revision"] == 1
    assert "training_plan_id" in context


def test_future_ordered_row_edit_persists_new_revision(app_model, tmp_path):
    app_model._trial_protocol_repository = TrialProtocolRepository(tmp_path)
    saved = app_model.save_ordered_protocol(TrialProtocolDocument(
        protocol_id="ordered-edit",
        name="Ordered Edit",
        trial_count=2,
        defaults=ProtocolPatch.from_mapping({
            "enabled": True,
            "position_mode": "fixed_manual",
        }),
    ))

    assert app_model.update_trial_protocol_row(2, "shift_y_mm", "1.5")

    updated = app_model.selected_ordered_protocol
    assert updated.revision == saved.revision + 1
    assert updated.resolve()[1].row.shift_y_mm == 1.5
    reloaded = TrialProtocolRepository(tmp_path)
    reloaded.reload()
    assert reloaded.get("ordered-edit").revision == updated.revision


def test_no_protocol_mode_has_safe_disabled_rows(app_model):
    app_model._select_ordered_protocol_internal(None, persist_animal=False)

    assert app_model.selected_ordered_protocol is None
    assert all(not row["enabled"] for row in app_model.trial_protocol_rows)
    assert not app_model.update_trial_protocol_row(1, "enabled", True)


def test_scored_trial_limit_is_available_with_live_intertrial_scoring(app_model):
    control = app_model.behavior.algorithm.active_config.session_control
    control.trial_limit = 5
    control.trial_count_basis = "scored"
    control.intertrial_analysis_enabled = True

    assert not any("Scored trials" in blocker for blocker in app_model.recording_blockers)


def test_intertrial_analysis_requires_live_inference_before_record(app_model):
    control = app_model.behavior.algorithm.active_config.session_control
    control.intertrial_analysis_enabled = True
    previous_inference = app_model._inference
    app_model._inference = SimpleNamespace(is_enabled=False)
    try:
        assert any(
            "Live intertrial analysis requires live inference" in blocker
            for blocker in app_model.recording_blockers
        )

        control.intertrial_analysis_enabled = False
        assert not any(
            "Live intertrial analysis requires live inference" in blocker
            for blocker in app_model.recording_blockers
        )
    finally:
        app_model._inference = previous_inference


def test_it_drain_record_stop_sema_on_session_recording_start(app_model):
    app_model._cams_record_start_perf.value = 123.0
    app_model._record_stop_sema.release()
    app_model._record_stop_sema.release()
    app_model.behavior.algorithm.start_session(reason="manual")
    assert app_model._record_stop_sema.acquire(block=False) is False, "cannot acquire after: it should be back to 0"
    assert math.isnan(app_model._cams_record_start_perf.value)


def test_preallocated_session_id_is_not_changed_by_behavior_start(app_model):
    algorithm = app_model.behavior.algorithm
    project = app_model.project
    project.calculate_next_session_index()
    reserved_session_id = project.short_id

    assert algorithm.start_session(
        reason="preallocated-test",
        allocate_project_session=False,
    )

    assert project.short_id == reserved_session_id


def test_record_commands_are_serialized_before_session_allocation(app_model):
    entered_first = threading.Event()
    release_first = threading.Event()
    entered_second = threading.Event()
    state_lock = threading.Lock()
    active = 0
    maximum_active = 0
    calls = 0

    def start_locked():
        nonlocal active, maximum_active, calls
        with state_lock:
            calls += 1
            call_number = calls
            active += 1
            maximum_active = max(maximum_active, active)
        if call_number == 1:
            entered_first.set()
            assert release_first.wait(1)
        else:
            entered_second.set()
        with state_lock:
            active -= 1
        return True

    app_model._start_recording_locked = start_locked
    first = threading.Thread(target=app_model.start_recording)
    second = threading.Thread(target=app_model.start_recording)
    first.start()
    assert entered_first.wait(1)
    second.start()
    assert not entered_second.wait(0.05)
    release_first.set()
    first.join(1)
    second.join(1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert entered_second.is_set()
    assert maximum_active == 1


def test_recording_camera_set_includes_all_recordable_reach_cameras(app_model):
    for camera in app_model.reach_cameras:
        camera.is_enabled = True
        camera.is_recording_enabled = True
    assert app_model._get_recording_cams() == app_model._ordered_reach_cameras(
        enabled_only=True,
    )


def test_recording_requires_a_selected_subject(app_model):
    app_model._acquisition.started = True
    app_model._status = AppModelStatus.RUNNING

    with mock.patch.object(app_model, "on_error") as on_error:
        assert app_model.start_recording() is False

    on_error.assert_called_once_with(
        "Recording unavailable",
        "Scan an RFID tag or select a subject before recording.",
    )


def test_configuration_mutators_reject_active_session(app_model):
    app_model._set_session_recording_status(SessionRecordingStatus.RECORDING)
    try:
        with pytest.raises(RuntimeError, match="output location.*recording"):
            app_model.output_location = app_model.output_location
        with pytest.raises(RuntimeError, match="Loading configuration.*recording"):
            app_model.load_configuration()
        with pytest.raises(RuntimeError, match="DAQ port configuration.*recording"):
            app_model.update_daq_port_configuration(None, None)
        with pytest.raises(RuntimeError, match="intertrial analysis.*recording"):
            app_model.set_intertrial_analysis_enabled(False)
        with pytest.raises(RuntimeError, match="protocol advancement.*recording"):
            app_model.set_automatic_protocol_advance_enabled(False)
        with pytest.raises(RuntimeError, match="session configuration.*recording"):
            app_model.update_session_control_option("trial_limit", 5)
        changed = []
        with pytest.raises(RuntimeError, match="pellet behavior.*recording"):
            app_model.update_behavior_configuration(
                "Changing pellet behavior",
                lambda _algorithm: changed.append(True),
            )
        assert changed == []
        with pytest.raises(RuntimeError, match="live inference.*recording"):
            app_model.set_live_inference_enabled(True)
        with pytest.raises(RuntimeError, match="inference model.*recording"):
            app_model.set_inference_model_location("other-model")
        with pytest.raises(RuntimeError, match="preference serial_number.*recording"):
            app_model.update_user_preference("serial_number", "other-rig")
        with pytest.raises(RuntimeError, match="subject.*recording"):
            app_model.selected_animal = SimpleNamespace(id="other-subject")
        with pytest.raises(RuntimeError, match="selected protocol.*recording"):
            app_model.set_training_plan(SimpleNamespace(plan_id="other-protocol"))
    finally:
        app_model._set_session_recording_status(SessionRecordingStatus.READY)


def test_disabling_analysis_changes_scored_counting_to_completed(app_model):
    control = app_model.behavior.algorithm.active_config.session_control
    control.intertrial_analysis_enabled = True
    control.trial_count_basis = "scored"

    app_model.set_intertrial_analysis_enabled(False)

    assert control.trial_count_basis == "completed"


def test_softmouse_refresh_is_a_temporary_recording_blocker(app_model):
    with app_model._animal_metadata_refresh_lock:
        app_model._animal_metadata_refresh_busy = True

    assert "SoftMouse animal metadata refresh is still running" in (
        app_model.recording_blockers
    )


def test_writer_finalization_waits_for_all_reach_cameras_and_pose(app_model):
    for camera in app_model.reach_cameras:
        camera.is_enabled = True
        camera.is_recording_enabled = True
    recording_cameras = app_model._get_recording_cams()

    class InferenceWithPoseWriterAck:
        def __init__(self):
            self.waited_for = []

        def wait_session_pose_closed(self, project, *, timeout):
            self.waited_for.append((project.short_id, timeout))
            return True

    inference = InferenceWithPoseWriterAck()
    previous_inference = app_model._inference
    app_model._inference = inference
    app_model._abort_had_recording_started = True
    reservation = app_model._recording_session.begin_record(
        app_model.project.short_id,
        {},
    )
    assert reservation is not None
    _, token = reservation
    app_model._recording_session.transition(
        SessionRecordingStatus.STOPPING,
        expected=(SessionRecordingStatus.ARMING,),
        token=token,
    )
    camera_closures = {}
    try:
        with mock.patch.object(
            app_model,
            "_merge_camera_timestamp_files",
        ) as merge, mock.patch.object(
            app_model,
            "_complete_stopped_recording",
        ) as complete:
            for camera in recording_cameras[:-1]:
                app_model._handle_proc_msg(
                    (
                        SystemStatusMessageKind.CAMERA_RECORDING_CLOSED_FINISHED,
                        (
                            camera.camera_index,
                            10,
                            app_model.project,
                            token.generation,
                        ),
                    ),
                    cams_closed_finished=camera_closures,
                )
                complete.assert_not_called()

            camera = recording_cameras[-1]
            app_model._handle_proc_msg(
                (
                    SystemStatusMessageKind.CAMERA_RECORDING_CLOSED_FINISHED,
                    (
                        camera.camera_index,
                        10,
                        app_model.project,
                        token.generation,
                    ),
                ),
                cams_closed_finished=camera_closures,
            )

        merge.assert_called_once_with(
            app_model.project,
            tuple(camera for camera in recording_cameras if camera in app_model.reach_cameras),
        )
        assert inference.waited_for == [(app_model.project.short_id, 10.0)]
        complete.assert_called_once_with(app_model.project, token=token)
    finally:
        app_model._inference = previous_inference


def test_startup_project_exists_before_periodic_status_is_published(app_model):
    assert app_model.project is not None
    assert EventManager.default().project is app_model.project


def test_default_output_path_uses_canonical_lowercase_directory():
    assert PersistenceConfiguration.DEFAULT_OUTPUT_PATH == Path("~/Documents/rawdatalocal")


def test_session_manifest_includes_enabled_streams_independent_of_plot_selection(
    app_model,
):
    channels = (
        NidaqSignalChannelConfiguration(
            "barcode",
            "InputCard/port0/line1",
            kind="digital",
        ),
        NidaqSignalChannelConfiguration(
            "tone1",
            "InputCard/port0/line2",
            kind="digital",
        ),
        NidaqSignalChannelConfiguration(
            "laser_feedback",
            "InputCard/ai0",
        ),
    )
    app_model._nidaq_signal_monitor = SimpleNamespace(
        hardware_enabled=True,
        configuration=NidaqSignalStreamConfiguration(
            channels=channels,
            is_enabled=True,
            display_channels=(),
        ),
    )
    app_model._inference = SimpleNamespace(is_enabled=True)
    app_model._hardware = SimpleNamespace(requires_connection=True)
    app_model._laser = SimpleNamespace(
        configuration=SimpleNamespace(backend="nidaq"),
    )
    app_model._get_recording_cams = lambda: (
        SimpleNamespace(
            name="left",
            camera_source=SimpleNamespace(url="spinnaker://left"),
        ),
    )

    manifest = {
        source["id"]: source
        for source in app_model._build_session_source_manifest()
    }

    assert tuple(manifest) == (
        "camera.left",
        "pose",
        "nidaq.barcode",
        "nidaq.tone1",
        "nidaq.laser_feedback",
        "device",
        "laser_outputs",
        "session_logs",
        "trials",
    )
    assert manifest["nidaq.barcode"]["binding"] == "InputCard/port0/line1"
    assert manifest["nidaq.tone1"]["path"] == "streams/nidaq.h5"
    assert manifest["nidaq.laser_feedback"]["kind"] == "nidaq_analog"


def test_pellet_send_and_ack_create_session_trial_attempt(app_model):
    algorithm = app_model.behavior.algorithm
    assert algorithm.start_session(reason="trial-ledger-test")
    assert app_model._trial_ledger is not None

    app_model._on_pellet_sending(
        perf_c=10.0,
        context="send-context",
    )
    with mock.patch.object(
        app_model._protocol_runner,
        "begin_trial",
    ) as begin_trial:
        app_model._on_pellet_sent(
            perf_c=10.25,
            context="send-context",
        )

    attempt = app_model._trial_ledger.active_attempt
    assert attempt.attempt_label == "1.1"
    assert attempt.operation_id == "send-context"
    assert attempt.send_perf_time == 10.0
    assert attempt.send_ack_perf_time == 10.25
    assert app_model._trial_ledger.summary()["pellets_presented"] == 1
    begin_trial.assert_called_once_with("1.1")


def test_pellet_ack_does_not_erase_live_behavior_counts(app_model):
    algorithm = app_model.behavior.algorithm
    assert algorithm.start_session(reason="live-count-test")
    algorithm.pellet_reaches = 3
    algorithm.successful_reaches = 2
    algorithm.pellets_consumed = 1
    app_model._on_pellet_sending(perf_c=10.0, context="send-context")

    app_model._on_pellet_sent(perf_c=10.25, context="send-context")

    assert algorithm.pellets_presented == 1
    assert algorithm.pellet_reaches == 3
    assert algorithm.successful_reaches == 2
    assert algorithm.pellets_consumed == 1


def test_mismatched_pellet_ack_does_not_present_or_count_trial(app_model):
    algorithm = app_model.behavior.algorithm
    assert algorithm.start_session(reason="trial-ledger-test")
    app_model._on_pellet_sending(perf_c=10.0, context="expected")

    app_model._on_pellet_sent(perf_c=10.25, context="stale")

    assert not app_model._trial_ledger.active_attempt.is_presented
    assert algorithm.pellets_presented == 0


def test_ack_timeout_finalizes_attempt_and_runtime_retry_uses_same_trial(app_model):
    algorithm = app_model.behavior.algorithm
    assert algorithm.start_session(reason="hardware-error-test")
    app_model._on_pellet_sending(perf_c=10.0, context="send-1")

    app_model._on_hardware_command_failed(CanFailure(
        CanFailureKind.ACKNOWLEDGEMENT_TIMEOUT,
        "pellet send acknowledgement timed out",
        command=SystemCommandKind.SEND_PELLET,
        context="send-1",
        perf_time=10.5,
        wall_time=110.5,
    ))

    failed = app_model._trial_ledger.attempts[0]
    assert failed.attempt_label == "1.1"
    assert failed.outcome is TrialOutcome.HARDWARE_ERROR
    assert failed.hardware_error_kind is HardwareErrorKind.ACKNOWLEDGEMENT_TIMEOUT
    assert app_model._trial_ledger.count() == 0

    app_model._on_pellet_sending(perf_c=11.0, context="send-2")
    assert app_model._trial_ledger.active_attempt.attempt_label == "1.2"


def test_failed_send_dispatch_is_persisted_without_counting_trial(app_model):
    algorithm = app_model.behavior.algorithm
    assert algorithm.start_session(reason="dispatch-error-test")

    app_model._on_hardware_command_failed(CanFailure(
        CanFailureKind.COMMAND,
        "command could not be queued",
        command=SystemCommandKind.SEND_PELLET,
        context="send-rejected",
        perf_time=10.0,
        wall_time=110.0,
    ))

    failed = app_model._trial_ledger.attempts[0]
    assert failed.operation_id == "send-rejected"
    assert failed.hardware_error_kind is HardwareErrorKind.MOTOR_FAILURE
    assert app_model._trial_ledger.summary()["hardware_errors"] == 1
    assert app_model._trial_ledger.count() == 0


@pytest.mark.parametrize(
    "failure_kind, command, expected_error_kind",
    (
        (
            CanFailureKind.ACKNOWLEDGEMENT_TIMEOUT,
            SystemCommandKind.SEND_PELLET,
            HardwareErrorKind.ACKNOWLEDGEMENT_TIMEOUT,
        ),
        (
            CanFailureKind.TRANSPORT,
            SystemCommandKind.SEND_PELLET,
            HardwareErrorKind.TRANSPORT_FAILURE,
        ),
        (
            CanFailureKind.COMMAND,
            SystemCommandKind.SEND_PELLET,
            HardwareErrorKind.MOTOR_FAILURE,
        ),
        (
            CanFailureKind.COMMAND,
            SystemCommandKind.PLAY_TONE,
            HardwareErrorKind.COMMAND_FAILURE,
        ),
        (
            CanFailureKind.OPERATION_UNKNOWN,
            SystemCommandKind.SEND_PELLET,
            HardwareErrorKind.OPERATION_UNKNOWN,
        ),
    ),
)
def test_production_can_failure_kinds_finalize_the_active_attempt(
    app_model,
    failure_kind,
    command,
    expected_error_kind,
):
    algorithm = app_model.behavior.algorithm
    assert algorithm.start_session(reason="typed-hardware-error-test")
    app_model._on_pellet_sending(perf_c=10.0, context="send-1")

    app_model._on_hardware_command_failed(CanFailure(
        failure_kind,
        "typed failure",
        command=command,
        context="send-1",
        perf_time=10.5,
        wall_time=110.5,
    ))

    attempt = app_model._trial_ledger.attempts[0]
    assert attempt.outcome is TrialOutcome.HARDWARE_ERROR
    assert attempt.hardware_error_kind is expected_error_kind
    assert app_model._trial_ledger.active_attempt is None
    assert app_model._trial_ledger.count() == 0


def test_automatic_stop_finishes_active_trial_normally_before_stopping(
    app_model,
    monkeypatch,
):
    ledger = PelletTrialLedger(app_model.project.short_id)
    ledger.begin_send(5.0, 105.0, operation_id="send-1")
    ledger.acknowledge_presentation(5.1, 105.1)
    policy = SessionStopPolicy(
        SessionStopConfiguration(
            duration_seconds=10,
            drain_timeout_seconds=15,
        ),
    )
    policy.start(0.0)
    app_model._trial_ledger = ledger
    app_model._session_stop_policy = policy
    app_model._set_session_recording_status(SessionRecordingStatus.RECORDING)
    now = iter((10.0, 11.0, 11.0))
    monkeypatch.setattr(
        "tools.acquisition.model.app_model.get_perf_now",
        lambda: next(now),
    )

    try:
        with mock.patch.object(app_model, "on_error") as on_error, mock.patch.object(
            app_model,
            "_stop_recording_with_reason",
        ) as stop:
            draining = app_model._evaluate_automatic_stop_policy()
            app_model._on_pellet_loading_for_trial()

        assert draining.decision is SessionStopDecision.FINISH_ACTIVE_TRIAL
        assert not draining.is_error
        on_error.assert_not_called()
        stop.assert_called_once_with(
            SessionStopReason.DURATION_LIMIT,
            SessionStopDecision.STOP,
        )
        assert ledger.active_attempt is None
        assert ledger.summary()["trials_completed"] == 1
    finally:
        app_model._cancel_automatic_stop_timers()


def test_pellet_cycle_completion_finishes_trial_before_automatic_stop(
    app_model,
):
    ledger = PelletTrialLedger(app_model.project.short_id)
    ledger.begin_send(5.0, 105.0, operation_id="send-1")
    ledger.acknowledge_presentation(5.1, 105.1)
    policy = SessionStopPolicy(
        SessionStopConfiguration(trial_limit=1),
    )
    policy.start(0.0)
    app_model._trial_ledger = ledger
    app_model._session_stop_policy = policy
    app_model._set_session_recording_status(SessionRecordingStatus.RECORDING)

    with mock.patch.object(
        app_model,
        "_stop_recording_with_reason",
    ) as stop, mock.patch.object(
        app_model._protocol_runner,
        "finish_trial",
    ) as finish_trial:
        app_model._on_pellet_cycle_completed(perf_c=6.0)

    stop.assert_called_once_with(
        SessionStopReason.TRIAL_LIMIT,
        SessionStopDecision.STOP,
    )
    assert ledger.active_attempt is None
    assert ledger.summary()["trials_completed"] == 1
    # Protocol progress waits for the per-attempt offline outcome.
    finish_trial.assert_not_called()


def test_live_intertrial_result_finalizes_attempt_persists_and_syncs_counts(
    app_model,
):
    project = app_model.project
    ledger = PelletTrialLedger(project.short_id)
    ledger.begin_send(100.5, 1000.5, operation_id="send-1")
    ledger.acknowledge_presentation(100.55, 1000.55)
    ledger.close_active_for_analysis(101.5, 1001.5)
    app_model._trial_ledger = ledger
    _, token = app_model._recording_session.begin_record(project.short_id, {})
    app_model._recording_session.transition(
        SessionRecordingStatus.RECORDING,
        expected=(SessionRecordingStatus.ARMING,),
        token=token,
    )
    tracking_window = TrackingWindow(
        100.6, 101.5, (), 0, 0, (), (), True,
    )
    request = IntertrialAnalysisRequest(
        token.generation,
        token.session_id,
        1,
        1,
        "send-1",
        tracking_window,
        PelletStateEvidence(
            PelletPresence.PRESENT,
            PelletMisplacement.NOT_MISPLACED,
            10,
            10,
            10,
            5.0,
        ),
    )
    result = IntertrialAnalysisResult(
        request=request,
        outcome=TrialOutcome.SUCCESS,
        reaches=(ReachTrajectory(100.7, 100.8, 100.9, (1, 2, 3), True),),
        reach_count=1,
        success_count=1,
        consumption_count=1,
        recommended_shift=None,
        interpolated_points=0,
        long_gap_count=0,
        analysis_seconds=0.02,
    )

    with mock.patch.object(
        app_model._session_data_recorder,
        "trial_stream_references",
        return_value=(
            ({"perf_time": 100.8, "channel": "tone1"},),
            ({"perf_time": 100.9, "channel": "left"},),
        ),
    ), mock.patch.object(
        app_model._session_data_recorder,
        "persist_trial_ledger",
    ) as update, mock.patch.object(
        app_model._session_data_recorder,
        "persist_trial_tracking",
    ), mock.patch.object(
        app_model._protocol_runner,
        "record_trial_outcome",
    ) as protocol_outcome:
        app_model._on_intertrial_analysis_result(result)

    attempt = ledger.attempts[0]
    assert attempt.outcome is TrialOutcome.SUCCESS
    assert attempt.reach_count == 1
    assert attempt.success_count == 1
    assert attempt.consumption_count == 1
    assert attempt.tone_references[0]["channel"] == "tone1"
    assert attempt.laser_references[0]["channel"] == "left"
    assert ledger.summary()["pending_analysis_attempts"] == 0
    assert app_model.behavior.algorithm.pellet_reaches == 1
    assert app_model.behavior.algorithm.pellets_presented == 1
    assert app_model.behavior.algorithm.successful_reaches == 1
    assert app_model.behavior.algorithm.pellets_consumed == 1
    protocol_outcome.assert_called_once_with("1.1", TrialOutcome.SUCCESS)
    update.assert_called_once_with(
        project,
        ledger.to_records(),
        ledger.summary(),
    )


def test_live_analysis_reserves_configured_behavioral_retry(app_model):
    project = app_model.project
    ledger = PelletTrialLedger(project.short_id)
    ledger.begin_send(100.5, 1000.5, operation_id="send-1")
    ledger.acknowledge_presentation(100.55, 1000.55)
    ledger.close_active_for_analysis(101.5, 1001.5)
    app_model._trial_ledger = ledger
    control = app_model.behavior.algorithm.active_config.session_control
    control.intertrial_analysis_enabled = True
    control.behavioral_retry_outcomes = (TrialOutcome.NO_REACH.value,)
    _, token = app_model._recording_session.begin_record(project.short_id, {})
    app_model._recording_session.transition(
        SessionRecordingStatus.RECORDING,
        expected=(SessionRecordingStatus.ARMING,),
        token=token,
    )
    window = TrackingWindow(100.6, 101.5, (), 0, 0, (), (), True)
    request = IntertrialAnalysisRequest(
        token.generation, token.session_id, 1, 1, "send-1", window,
        PelletStateEvidence(
            PelletPresence.PRESENT, PelletMisplacement.UNKNOWN, 0, 0, 0, None,
        ),
    )
    result = IntertrialAnalysisResult(
        request, TrialOutcome.NO_REACH, (), 0, 0, 0, None, 0, 0, 0.01,
    )

    with mock.patch.object(
        app_model._session_data_recorder,
        "trial_stream_references",
        return_value=((), ()),
    ), mock.patch.object(
        app_model._session_data_recorder,
        "persist_trial_ledger",
    ), mock.patch.object(
        app_model._session_data_recorder,
        "persist_trial_tracking",
    ), mock.patch.object(
        app_model._protocol_runner,
        "record_trial_outcome",
    ):
        app_model._on_intertrial_analysis_result(result)

    retry = ledger.begin_send(102.0, 1002.0, operation_id="send-2")
    assert ledger.attempts[0].logical_trial_complete is False
    assert retry.attempt_label == "1.2"


def test_missing_pellet_retry_finishes_synchronously_without_analysis(app_model):
    project = app_model.project
    ledger = PelletTrialLedger(project.short_id)
    ledger.begin_send(5.0, 105.0, operation_id="send-1")
    ledger.acknowledge_presentation(5.1, 105.1)
    app_model._trial_ledger = ledger
    control = app_model.behavior.algorithm.active_config.session_control
    control.intertrial_analysis_enabled = False
    control.behavioral_retry_outcomes = (TrialOutcome.PELLET_MISSING.value,)
    _, token = app_model._recording_session.begin_record(project.short_id, {})
    app_model._recording_session.transition(
        SessionRecordingStatus.RECORDING,
        expected=(SessionRecordingStatus.ARMING,),
        token=token,
    )
    samples = tuple(
        LiveTrackingSample(
            sequence=index,
            primary_frame_ids=(index,),
            primary_frame_perf_times=(5.2 + index * 0.01,),
            source_start_perf=5.2 + index * 0.01,
            source_end_perf=5.2 + index * 0.01,
            processing_perf=5.3 + index * 0.01,
            pellet_seen=False,
            locations_3d=(),
            offsets_3d=(),
        )
        for index in range(5)
    )
    tracking_window = TrackingWindow(
        5.2,
        6.0,
        samples,
        5,
        5,
        (),
        (),
        True,
    )
    app_model._trial_window_start = ("send-1", 5.2)

    with mock.patch.object(
        app_model._live_tracking,
        "window",
        return_value=tracking_window,
    ), mock.patch.object(
        app_model._session_data_recorder,
        "persist_trial_tracking",
    ), mock.patch.object(
        app_model._session_data_recorder,
        "trial_stream_references",
        return_value=((), ()),
    ), mock.patch.object(
        app_model._session_data_recorder,
        "persist_trial_ledger",
    ), mock.patch.object(
        app_model._intertrial_analysis,
        "submit",
    ) as submit:
        app_model._complete_pellet_trial_window(
            6.0,
            close_reason="pellet cycle completed",
        )

    attempt = ledger.attempts[0]
    assert attempt.outcome is TrialOutcome.PELLET_MISSING
    assert attempt.logical_trial_complete is False
    assert ledger.begin_send(6.1, 106.1).attempt_label == "1.2"
    submit.assert_not_called()


def test_validated_nidaq_tone2_is_the_authoritative_trial_boundary(app_model):
    ledger = PelletTrialLedger(app_model.project.short_id)
    ledger.begin_send(5.0, 105.0, operation_id="send-1")
    app_model._trial_ledger = ledger
    _, token = app_model._recording_session.begin_record(
        app_model.project.short_id, {},
    )
    app_model._recording_session.transition(
        SessionRecordingStatus.RECORDING,
        expected=(SessionRecordingStatus.ARMING,),
        token=token,
    )
    monitor = app_model._nidaq_signal_monitor
    monitor._is_running = True
    monitor._configuration = NidaqSignalStreamConfiguration(
        is_enabled=True,
        channels=(NidaqSignalChannelConfiguration(
            name="tone2",
            physical_channel="Dev1/port0/line1",
            kind="digital",
        ),),
    )

    app_model._on_intertrial_device_message(
        SystemStatusMessageKind.STIMULUS_INPUTS,
        (False, True, False, False),
        5.110,
        105.110,
    )
    assert app_model._trial_window_start is None

    app_model._on_intertrial_nidaq_tone_edge(
        channel="tone2",
        perf_time=5.0023,
        wall_time=105.0023,
        sample_index=123,
    )

    assert app_model._trial_window_start == ("send-1", 5.0023)


def test_immediate_can_tone_status_is_fallback_without_nidaq(app_model):
    ledger = PelletTrialLedger(app_model.project.short_id)
    ledger.begin_send(5.0, 105.0, operation_id="send-1")
    app_model._trial_ledger = ledger
    _, token = app_model._recording_session.begin_record(
        app_model.project.short_id, {},
    )
    app_model._recording_session.transition(
        SessionRecordingStatus.RECORDING,
        expected=(SessionRecordingStatus.ARMING,),
        token=token,
    )
    tone = Tone(Target.PELLET_DEVICE, time_remaining_ms=300, frequency_hz=6000)
    tone.index = 5_004_000_000

    app_model._on_intertrial_device_message(
        SystemStatusMessageKind.TONE_STATUS,
        tone,
        5.010,
        105.010,
    )

    assert app_model._trial_window_start == ("send-1", 5.004)


def test_failed_retry_dependent_analysis_requires_explicit_resolution(app_model):
    project = app_model.project
    ledger = PelletTrialLedger(project.short_id)
    ledger.begin_send(100.5, 1000.5, operation_id="send-1")
    ledger.acknowledge_presentation(100.55, 1000.55)
    ledger.close_active_for_analysis(101.5, 1001.5)
    app_model._trial_ledger = ledger
    control = app_model.behavior.algorithm.active_config.session_control
    control.intertrial_analysis_enabled = True
    control.behavioral_retry_outcomes = (TrialOutcome.NO_REACH.value,)
    _, token = app_model._recording_session.begin_record(project.short_id, {})
    app_model._recording_session.transition(
        SessionRecordingStatus.RECORDING,
        expected=(SessionRecordingStatus.ARMING,),
        token=token,
    )
    window = TrackingWindow(100.6, 101.5, (), 0, 0, (), (), True)
    request = IntertrialAnalysisRequest(
        token.generation,
        token.session_id,
        1,
        1,
        "send-1",
        window,
        PelletStateEvidence(
            PelletPresence.PRESENT,
            PelletMisplacement.UNKNOWN,
            0,
            0,
            0,
            None,
        ),
    )
    result = IntertrialAnalysisResult(
        request,
        TrialOutcome.INCOMPLETE,
        (),
        0,
        0,
        0,
        None,
        0,
        0,
        0.01,
        "analysis worker failed",
    )

    with mock.patch.object(
        app_model._session_data_recorder,
        "persist_trial_tracking",
    ):
        app_model._on_intertrial_analysis_result(result)

    assert ledger.attempts[0].outcome is TrialOutcome.PENDING_ANALYSIS
    assert app_model.trial_protocol_state["analysis"]["resolution_required"]
    assert app_model.behavior.algorithm.pellet_send_block_reason

    with mock.patch.object(
        app_model._session_data_recorder,
        "trial_stream_references",
        return_value=((), ()),
    ), mock.patch.object(
        app_model._session_data_recorder,
        "persist_trial_ledger",
    ):
        assert app_model.continue_without_pending_intertrial_result()

    assert ledger.attempts[0].outcome is TrialOutcome.INCOMPLETE
    assert not app_model.trial_protocol_state["analysis"]["resolution_required"]
    assert app_model.behavior.algorithm.pellet_send_block_reason == ""


def test_abort_removes_whole_session_and_resets_counts(app_model):
    project = app_model.project
    project.session = 1
    session_path = Path(project.get_session_path().location)
    nested_file = session_path / "streams" / "partial.csv"
    nested_file.parent.mkdir(parents=True, exist_ok=True)
    nested_file.write_text("partial data")

    algorithm = app_model.behavior.algorithm
    algorithm.increase_pellets_presented(2)
    algorithm.increase_pellets_consumed(1)
    algorithm.increase_pellet_total_reaches(3)
    algorithm.increase_successful_reaches(1)

    app_model._aborting_project = project.to_local_value()
    app_model._set_session_recording_status(SessionRecordingStatus.ABORTING)
    app_model._finish_abort_recording()

    assert not session_path.exists()
    assert project.session == 0
    assert app_model.session_recording_status is SessionRecordingStatus.READY
    assert algorithm.pellets_presented == 0
    assert algorithm.pellets_consumed == 0
    assert algorithm.pellet_reaches == 0
    assert algorithm.successful_reaches == 0


def test_tokenized_abort_transitions_ready_before_invalidating_generation(
    app_model,
):
    project = app_model.project
    project.session = 1
    session_path = Path(project.get_session_path().location)
    session_path.mkdir(parents=True, exist_ok=True)
    reservation = app_model._recording_session.begin_record(
        project.short_id,
        {},
    )
    assert reservation is not None
    _, token = reservation
    assert app_model._recording_session.transition(
        SessionRecordingStatus.ABORTING,
        expected=(SessionRecordingStatus.ARMING,),
        token=token,
    ) is SessionRecordingStatus.ARMING
    app_model._aborting_project = project.to_local_value()

    app_model._finish_abort_recording(token=token)

    assert app_model.session_recording_status is SessionRecordingStatus.READY
    assert app_model._recording_session.token() is None
    assert app_model._recording_session.generation > token.generation
    assert not session_path.exists()


def test_failed_arming_uses_abort_cleanup_without_reverting_to_arming(
    app_model,
):
    project = app_model.project
    project.session = 1
    reservation = app_model._recording_session.begin_record(
        project.short_id,
        {},
    )
    assert reservation is not None
    _, token = reservation
    timer = mock.Mock()

    with mock.patch.object(
        app_model.behavior.algorithm,
        "end_capture_session",
        return_value=False,
    ), mock.patch(
        "tools.acquisition.model.app_model.make_daemon_timer",
        return_value=timer,
    ):
        assert app_model.abort_recording(token=token)

    assert app_model.session_recording_status is SessionRecordingStatus.ABORTING
    assert app_model._aborting_project.short_id == project.short_id
    timer.start.assert_called_once_with()


def test_abort_during_analysis_cancels_analysis_and_removes_session(
    app_model,
):
    project = app_model.project
    project.session = 1
    session_path = Path(project.get_session_path().location)
    analysis_file = session_path / "analysis" / "partial.h5"
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    analysis_file.write_text("partial analysis")
    app_model.behavior.algorithm.increase_pellets_presented(2)
    app_model._recording_session.analysis_finished = False
    app_model._acquisition.started = True
    app_model._set_subsystem_status(
        SubsystemId.REACH_SYNCHRONIZATION,
        SubsystemState.READY,
    )
    app_model._set_session_recording_status(SessionRecordingStatus.ANALYZING)

    with mock.patch.object(
        app_model._intertrial_analysis,
        "cancel_session",
    ) as cancel, mock.patch.object(app_model._inference, "stop") as stop:
        assert app_model.abort_recording()

    cancel.assert_called_once_with()
    stop.assert_not_called()
    assert not session_path.exists()
    assert project.session == 0
    assert app_model.behavior.algorithm.pellets_presented == 0
    assert app_model.session_recording_status is SessionRecordingStatus.READY


def test_stop_finishes_auxiliary_data_after_raw_writers_close(app_model):
    _, token = app_model._recording_session.begin_record(
        app_model.project.short_id, {},
    )
    app_model._recording_session.set_pending_end(12.5, token)
    app_model._recording_session.set_boundary(SessionBoundary(
        session_id=app_model.project.short_id,
        primary_camera="left",
        primary_frame_id=42,
        start_perf_time=10.0,
        start_wall_time=100.0,
        camera_when=1_000_000.0,
    ), token)
    app_model._recording_session.transition(
        SessionRecordingStatus.STOPPING,
        expected=(SessionRecordingStatus.ARMING,),
        token=token,
    )

    with mock.patch.object(
        app_model._session_data_recorder, "stop"
    ) as stop, mock.patch.object(
        app_model, "_save_project_metadata"
    ) as save_metadata, mock.patch.object(
        app_model._intertrial_analysis, "is_idle", return_value=False,
    ), mock.patch.object(
        app_model, "_wait_and_finish_intertrial_session",
    ):
        app_model._complete_stopped_recording(app_model.project)

    stop.assert_called_once_with(12.5)
    save_metadata.assert_called_once_with(
        app_model.project,
        caller="raw_writers_closed",
    )
    assert app_model.session_recording_status is SessionRecordingStatus.ANALYZING


def test_stop_snapshots_trial_ledger_with_pending_analysis_outcome(app_model):
    _, token = app_model._recording_session.begin_record(
        app_model.project.short_id, {},
    )
    app_model._recording_session.set_pending_end(12.5, token)
    app_model._recording_session.set_boundary(SessionBoundary(
        session_id=app_model.project.short_id,
        primary_camera="left",
        primary_frame_id=42,
        start_perf_time=10.0,
        start_wall_time=100.0,
        camera_when=1_000_000.0,
    ), token)
    ledger = PelletTrialLedger(app_model.project.short_id)
    ledger.begin_send(10.5, 100.5, operation_id="send-1")
    ledger.acknowledge_presentation(10.75, 100.75)
    app_model._trial_ledger = ledger
    app_model._recording_session.transition(
        SessionRecordingStatus.STOPPING,
        expected=(SessionRecordingStatus.ARMING,),
        token=token,
    )

    with mock.patch.object(
        app_model._session_data_recorder,
        "set_trial_ledger",
    ) as set_trial_ledger, mock.patch.object(
        app_model._session_data_recorder,
        "stop",
    ), mock.patch.object(
        app_model,
        "_save_project_metadata",
    ):
        app_model._complete_stopped_recording(app_model.project)

    attempt = ledger.attempts[0]
    assert attempt.outcome is TrialOutcome.INCOMPLETE
    assert attempt.finalized_perf_time == 12.5
    assert "stopped before the pellet cycle completed" in attempt.error
    set_trial_ledger.assert_called_once_with(
        ledger.to_records(),
        ledger.summary(),
    )


def test_final_metadata_uses_canonical_boundary_not_stale_project_timestamp(
    app_model,
    tmp_path,
):
    project = app_model.project.to_local_value()
    project.start_record_timestamp = math.nan
    app_model._recording_session.boundary = SessionBoundary(
        session_id=project.short_id,
        primary_camera="left",
        primary_frame_id=42,
        start_perf_time=10.0,
        start_wall_time=1_800_000_000.25,
        camera_when=1_000_000.0,
        end_perf_time=12.0,
        end_wall_time=1_800_000_002.25,
    )
    app_model._on_session_capture_ended(RecordingEndingReason.MANUAL_STOP)
    session_dir = tmp_path / "session003"
    streams_dir = session_dir / "streams"
    streams_dir.mkdir(parents=True)
    (streams_dir / "alignment.json").write_text("{}")
    (streams_dir / "trial_summary.json").write_text("{}")
    output = session_dir / "metadata"

    app_model._save_metadata(
        project,
        project.when,
        str(output),
        project.session,
    )

    serialized_json = output.with_suffix(".json").read_text()
    saved = json.loads(serialized_json)
    assert saved["metadataSchemaVersion"] == 2
    assert saved["scope"] == "session"
    assert saved["recording"]["firstPelletDeliveryOffsetSeconds"] is None
    assert saved["recording"]["firstPelletPresentationOffsetSeconds"] is None
    assert "NaN" not in output.with_suffix(".json").read_text()
    assert ".nan" not in output.with_suffix(".yaml").read_text().lower()
    assert saved["boundary"]["startWallTime"] == 1_800_000_000.25
    assert saved["boundary"]["endPerfTime"] == 12.0
    assert saved["boundary"]["durationSeconds"] == 2.0
    assert saved["recording"]["stopReason"] == "ManualStop"
    assert "enabledSources" not in saved
    assert "trialSummary" not in saved
    assert "hardwareRuntimeAtRecord" not in saved
    assert set(saved["configuration"]) >= {
        "version",
        "cameras",
        "hardware",
        "inference",
        "laser",
        "nidaq_ports",
        "nidaq_stream",
        "behavior",
        "persistence",
        "watchdog",
    }
    assert len(serialized_json) < 12_000
    assert saved["artifacts"]["alignment"]["$ref"] == "streams/alignment.json"
    assert saved["artifacts"]["trialSummary"]["$ref"] == "streams/trial_summary.json"
    manifest = json.loads((session_dir / "manifest.json").read_text())
    assert saved["metadataGenerationId"] == manifest["metadataGenerationId"]
    assert manifest["authoritativeMetadata"] == "metadata.json"
    assert {item["path"] for item in manifest["files"]} == {
        "metadata.json",
        "metadata.yaml",
    }


def test_metadata_pair_is_not_replaced_when_yaml_serialization_fails(
    app_model,
    tmp_path,
):
    project = app_model.project.to_local_value()
    output = tmp_path / "metadata"
    json_path = output.with_suffix(".json")
    yaml_path = output.with_suffix(".yaml")
    json_path.write_text("previous json")
    yaml_path.write_text("previous yaml")

    with mock.patch(
        "tools.acquisition.model.app_model.yaml.dump",
        side_effect=RuntimeError("serialization failed"),
    ):
        with pytest.raises(RuntimeError, match="serialization failed"):
            app_model._save_metadata(
                project,
                project.when,
                str(output),
                project.session,
            )

    assert json_path.read_text() == "previous json"
    assert yaml_path.read_text() == "previous yaml"
    assert not (tmp_path / "metadata.json.tmp").exists()
    assert not (tmp_path / "metadata.yaml.tmp").exists()


def test_incomplete_auxiliary_streams_are_not_reported_as_fully_saved(
    app_model,
):
    _, token = app_model._recording_session.begin_record(
        app_model.project.short_id, {},
    )
    app_model._recording_session.set_pending_end(12.0, token)
    app_model._recording_session.set_boundary(SessionBoundary(
        session_id=app_model.project.short_id,
        primary_camera="left",
        primary_frame_id=42,
        start_perf_time=10.0,
        start_wall_time=100.0,
        camera_when=1_000_000.0,
    ), token)
    app_model._recording_session.transition(
        SessionRecordingStatus.STOPPING,
        expected=(SessionRecordingStatus.ARMING,),
        token=token,
    )
    result = {
        "sessionComplete": False,
        "incompleteReasons": ("nidaq.barcode reported 1 acquisition gap(s)",),
        "enabledSources": ({"id": "nidaq.barcode", "sampleCount": 10},),
        "cameraNidaqAlignment": {"matchedSampleIndex": 100},
    }

    with mock.patch.object(
        app_model._session_data_recorder,
        "stop",
        return_value=result,
    ), mock.patch.object(
        app_model,
        "_save_project_metadata",
    ), mock.patch.object(
        app_model,
        "on_error",
    ) as on_error:
        app_model._complete_stopped_recording(app_model.project)

    assert app_model._recording_session.data_complete is False
    assert app_model._recording_session.enabled_sources == result["enabledSources"]
    assert app_model._recording_session.boundary.nidaq_sample_index == 100
    assert (
        app_model.subsystem_statuses[SubsystemId.INTERTRIAL_ANALYSIS.value].state
        is SubsystemState.FAILED
    )
    assert app_model.session_recording_status is SessionRecordingStatus.READY
    on_error.assert_called_once()


def test_record_start_timeout_aborts_partial_session(app_model):
    app_model._set_session_recording_status(SessionRecordingStatus.ARMING)
    with mock.patch.object(app_model, "on_error") as on_error, mock.patch.object(
        app_model, "abort_recording"
    ) as abort:
        app_model._record_start_timed_out()

    on_error.assert_called_once()
    abort.assert_called_once_with()


def test_writer_close_timeout_preserves_session_and_marks_invariant(app_model):
    reservation = app_model._recording_session.begin_record(
        app_model.project.short_id,
        {},
    )
    assert reservation is not None
    _, token = reservation
    app_model._recording_session.transition(
        SessionRecordingStatus.STOPPING,
        expected=(SessionRecordingStatus.ARMING,),
        token=token,
    )

    with mock.patch.object(app_model, "on_error") as on_error:
        app_model._writer_close_timed_out(token)

    assert app_model.session_recording_status is SessionRecordingStatus.STOPPING
    assert app_model._recording_session.data_complete is False
    assert app_model._session_invariant_unknown is True
    on_error.assert_called_once()


def test_stale_session_generation_cannot_timeout_or_close_new_session(app_model):
    first_reservation = app_model._recording_session.begin_record(
        app_model.project.short_id,
        {},
    )
    assert first_reservation is not None
    _, first_token = first_reservation
    app_model._recording_session.transition(
        SessionRecordingStatus.ABORTING,
        expected=(SessionRecordingStatus.ARMING,),
        token=first_token,
    )
    app_model._recording_session.transition(
        SessionRecordingStatus.READY,
        expected=(SessionRecordingStatus.ABORTING,),
        token=first_token,
    )
    second_reservation = app_model._recording_session.begin_record(
        app_model.project.short_id,
        {},
    )
    assert second_reservation is not None
    _, second_token = second_reservation

    with mock.patch.object(app_model, "abort_recording") as abort:
        app_model._record_start_timed_out(first_token)
    abort.assert_not_called()
    assert app_model._recording_session.is_current(
        second_token,
        statuses=(SessionRecordingStatus.ARMING,),
    )
    app_model._recording_session.transition(
        SessionRecordingStatus.STOPPING,
        expected=(SessionRecordingStatus.ARMING,),
        token=second_token,
    )

    camera = app_model.reach_cameras[0]
    camera.is_enabled = True
    camera.is_recording_enabled = True
    camera_closures = {}
    with mock.patch.object(app_model, "_complete_stopped_recording") as complete:
        app_model._handle_proc_msg(
            (
                SystemStatusMessageKind.CAMERA_RECORDING_CLOSED_FINISHED,
                (
                    camera.camera_index,
                    10,
                    app_model.project.to_local_value(),
                ),
            ),
            cams_closed_finished=camera_closures,
        )
        app_model._handle_proc_msg(
            (
                SystemStatusMessageKind.CAMERA_RECORDING_CLOSED_FINISHED,
                (
                    camera.camera_index,
                    10,
                    app_model.project.to_local_value(),
                    first_token.generation,
                ),
            ),
            cams_closed_finished=camera_closures,
        )
    assert camera_closures == {}
    complete.assert_not_called()


def test_required_failed_subsystem_is_exposed_as_recording_blocker(app_model):
    app_model._acquisition.started = True
    app_model._set_subsystem_status(
        SubsystemId.NIDAQ_STREAM,
        SubsystemState.FAILED,
        required_for_recording=True,
        error="configured input device unavailable",
    )

    assert app_model.recording_blockers == (
        "nidaq_stream: configured input device unavailable",
    )


def test_required_nidaq_runtime_loss_preserves_session_and_acquisition(
    app_model,
):
    app_model._acquisition.started = True
    app_model._set_subsystem_status(
        SubsystemId.NIDAQ_STREAM,
        SubsystemState.READY,
        required_for_recording=True,
    )
    app_model._set_session_recording_status(SessionRecordingStatus.RECORDING)

    with mock.patch.object(app_model, "abort_recording") as abort, \
            mock.patch.object(app_model, "_stop_recording") as stop:
        app_model._set_subsystem_status(
            SubsystemId.NIDAQ_STREAM,
            SubsystemState.FAILED,
            error="worker stopped",
        )
        app_model._handle_recording_subsystem_failure(
            SubsystemId.NIDAQ_STREAM,
            "worker stopped",
        )

    abort.assert_not_called()
    stop.assert_not_called()
    assert app_model.session_recording_status is SessionRecordingStatus.RECORDING
    assert app_model.acquisition_started


def test_primary_camera_runtime_loss_requests_preserving_stop(app_model):
    primary = app_model.reach_cameras[0]
    primary.set_runtime_primary(True)
    primary.is_enabled = True
    primary.is_recording_enabled = True
    app_model._set_subsystem_status(
        SubsystemId.camera(primary.name),
        SubsystemState.FAILED,
        required_for_recording=True,
        error="transport lost",
    )
    app_model._set_session_recording_status(SessionRecordingStatus.RECORDING)

    with mock.patch.object(app_model, "_stop_recording") as stop, \
            mock.patch.object(app_model, "abort_recording") as abort:
        app_model._handle_recording_subsystem_failure(
            SubsystemId.camera(primary.name), "transport lost"
        )

    stop.assert_called_once_with(
        RecordingEndingReason.REQUIRED_SOURCE_FAILURE,
        token=None,
    )
    abort.assert_not_called()


def test_secondary_camera_runtime_loss_does_not_stop_recording(app_model):
    primary, secondary = app_model.reach_cameras[:2]
    primary.set_runtime_primary(True)
    for camera in (primary, secondary):
        camera.is_enabled = True
        camera.is_recording_enabled = True
    app_model._set_subsystem_status(
        SubsystemId.camera(secondary.name),
        SubsystemState.FAILED,
        required_for_recording=True,
        error="frame timeout",
    )
    app_model._set_session_recording_status(SessionRecordingStatus.RECORDING)

    with mock.patch.object(app_model, "_stop_recording") as stop, \
            mock.patch.object(app_model, "abort_recording") as abort:
        app_model._handle_recording_subsystem_failure(
            SubsystemId.camera(secondary.name), "frame timeout"
        )

    stop.assert_not_called()
    abort.assert_not_called()


def test_startup_summary_correlates_camera_and_nidaq_without_claiming_cause(
    app_model,
    caplog,
):
    app_model._set_subsystem_status(
        SubsystemId.camera("right"),
        SubsystemState.FAILED,
        error="no frame received from hardware-triggered camera",
    )
    app_model._set_subsystem_status(
        SubsystemId.NIDAQ_STREAM,
        SubsystemState.FAILED,
        error="configured chassis is unavailable",
    )

    with caplog.at_level("INFO"):
        app_model._log_acquisition_startup_summary()

    assert "SUMMARY | camera.right | state=failed" in caplog.text
    assert "SUMMARY | nidaq_stream | state=failed" in caplog.text
    assert "may share a physical power, timing, trigger, or ground dependency" in caplog.text
    assert "NI-DAQ software initialization does not trigger the cameras" in caplog.text


def test_reach_secondaries_are_armed_before_primary_first_frame_validation(
    app_model,
):
    events = []

    def camera(name, *, is_primary):
        result = mock.Mock()
        result.name = name
        result.is_primary = is_primary
        result.last_error = ""
        result.last_captured_frame_index = 0
        result.video_status = CaptureProcessStatus.RUNNING
        result.on_prepare_capture.side_effect = (
            lambda *_args, **_kwargs: events.append(f"prepare:{name}") or True
        )
        result.wait_for_capture_status.return_value = True
        result.on_capture_start.side_effect = (
            lambda: events.append(f"arm:{name}")
        )
        result.wait_for_first_frame.side_effect = (
            lambda **_kwargs: events.append(f"frame:{name}") or True
        )
        return result

    primary = camera("primary", is_primary=True)
    secondary = camera("secondary", is_primary=False)
    app_model._ordered_reach_cameras = lambda **_kwargs: (primary, secondary)
    app_model._inference_queue = None

    assert app_model._start_reach_camera_domains({})
    assert events.index("arm:secondary") < events.index("arm:primary")
    assert events.index("arm:primary") < events.index("frame:secondary")


def test_standalone_camera_role_is_restored_for_synchronized_capture(app_model):
    primary, secondary = app_model.reach_cameras[:2]
    primary._configured_is_primary = True
    secondary._configured_is_primary = False
    primary.is_enabled = False
    secondary.is_enabled = True

    app_model._ensure_reach_primary_camera()

    assert secondary.is_primary

    primary.is_enabled = True
    app_model._ensure_reach_primary_camera()

    assert primary.is_primary
    assert not secondary.is_primary


def test_can_start_failure_is_scoped_to_can_domain(app_model):
    app_model.hardware._can_enabled = True
    app_model.hardware._pellet_controller_enabled = True

    with mock.patch.object(
        app_model,
        "_ensure_pellet_controller_connected",
        side_effect=RuntimeError("PXI/CAN interface unavailable"),
    ):
        with mock.patch.object(
            app_model.hardware,
            "safety_shutdown",
        ) as safety_shutdown:
            with mock.patch.object(app_model, "capture_stop") as capture_stop:
                assert not app_model._start_can_domain(wait_connected=True)

    status = app_model.subsystem_statuses[SubsystemId.CAN_PELLET.value]
    assert status.state is SubsystemState.FAILED
    assert "PXI/CAN interface unavailable" in status.error
    safety_shutdown.assert_called_once()
    capture_stop.assert_not_called()


def test_runtime_can_failure_preserves_session_and_only_can_status_recovers(app_model):
    app_model._acquisition.started = True
    app_model._set_subsystem_status(
        SubsystemId.CAN_PELLET,
        SubsystemState.READY,
        required_for_recording=True,
    )
    monitor = app_model.analysis.watchdog_monitor

    with mock.patch.object(monitor, "unregister_watchdog") as unregister, \
            mock.patch.object(monitor, "register_watchdog") as register:
        app_model._on_can_connection_state_changed({
            "state": "failed",
            "error": "CAN adapter removed",
        })
        failed = app_model.subsystem_statuses[SubsystemId.CAN_PELLET.value]
        assert failed.state is SubsystemState.FAILED
        assert failed.error == "CAN adapter removed"
        assert unregister.call_count == 2

        app_model._on_can_connection_state_changed({"state": "ready", "error": ""})
        ready = app_model.subsystem_statuses[SubsystemId.CAN_PELLET.value]
        assert ready.state is SubsystemState.READY
        assert register.call_count == 2


def test_session_end_home_is_acknowledged_once_and_recorded(app_model):
    app_model._set_subsystem_status(
        SubsystemId.CAN_PELLET,
        SubsystemState.READY,
    )
    with mock.patch.object(
        type(app_model.hardware),
        "connected",
        new_callable=mock.PropertyMock,
        return_value=True,
    ), mock.patch.object(
        app_model.hardware,
        "send_home_and_wait",
        return_value="home-token",
    ) as send_home:
        app_model._request_session_end_home("stop")
        app_model._request_session_end_home("stop")

    send_home.assert_called_once_with(timeout=15.0)
    action = app_model._recording_session.end_actions[0]
    assert action["status"] == "completed"
    assert action["ending"] == "stop"
    assert action["insideRecordedBoundary"] is False
    assert action["commandToken"] == "home-token"


def test_session_end_home_failure_does_not_raise_or_change_can_status(app_model):
    app_model._set_subsystem_status(
        SubsystemId.CAN_PELLET,
        SubsystemState.READY,
    )
    with mock.patch.object(
        type(app_model.hardware),
        "connected",
        new_callable=mock.PropertyMock,
        return_value=True,
    ), mock.patch.object(
        app_model.hardware,
        "send_home_and_wait",
        side_effect=TimeoutError("home timed out"),
    ), mock.patch.object(app_model, "on_error") as on_error:
        app_model._request_session_end_home("abort")

    action = app_model._recording_session.end_actions[0]
    assert action["status"] == "failed"
    assert action["error"] == "home timed out"
    assert app_model.subsystem_statuses[
        SubsystemId.CAN_PELLET.value
    ].state is SubsystemState.READY
    on_error.assert_called_once()

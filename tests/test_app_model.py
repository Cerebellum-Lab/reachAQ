import math
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from autotrainer.core import (
    EventManager,
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
    SystemStatusMessageKind,
)
from autotrainer.core.capture import CaptureProcessStatus
from autotrainer.core.configuration.persistence_configuration import PersistenceConfiguration
from autotrainer.behavior.behavior_algorithm import BehaviorAlgoStatus
from autotrainer.behavior.pellet_trial import PelletTrialLedger, TrialOutcome
from tools.acquisition.model.app_model import app_status_to_api_app_mode, app_status_to_behavior_algo_status
from tools.acquisition.model.app_model_status import AppModelStatus, SessionRecordingStatus
from tools.acquisition.model.session_boundary import SessionBoundary
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

from autotrainer.api import ApiApplicationMode


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
            assert algo_status.name == app_model_status.name


def test_it_drain_record_stop_sema_on_session_recording_start(app_model):
    app_model._cams_record_start_perf.value = 123.0
    app_model._record_stop_sema.release()
    app_model._record_stop_sema.release()
    app_model.behavior.algorithm.start_session(reason="manual")
    assert app_model._record_stop_sema.acquire(block=False) is False, "cannot acquire after: it should be back to 0"
    assert math.isnan(app_model._cams_record_start_perf.value)


def test_recording_camera_set_includes_all_recordable_reach_cameras(app_model):
    for camera in app_model.reach_cameras:
        camera.is_enabled = True
        camera.is_recording_enabled = True
    assert app_model._get_recording_cams() == app_model._ordered_reach_cameras(
        enabled_only=True,
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
    app_model._set_session_recording_status(SessionRecordingStatus.STOPPING)
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
                        (camera.camera_index, 10, app_model.project),
                    ),
                    cams_closed_finished=camera_closures,
                )
                complete.assert_not_called()

            camera = recording_cameras[-1]
            app_model._handle_proc_msg(
                (
                    SystemStatusMessageKind.CAMERA_RECORDING_CLOSED_FINISHED,
                    (camera.camera_index, 10, app_model.project),
                ),
                cams_closed_finished=camera_closures,
            )

        merge.assert_called_once_with(
            app_model.project,
            tuple(camera for camera in recording_cameras if camera in app_model.reach_cameras),
        )
        assert inference.waited_for == [(app_model.project.short_id, 10.0)]
        complete.assert_called_once_with(app_model.project)
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


def test_mismatched_pellet_ack_does_not_present_or_count_trial(app_model):
    algorithm = app_model.behavior.algorithm
    assert algorithm.start_session(reason="trial-ledger-test")
    app_model._on_pellet_sending(perf_c=10.0, context="expected")

    app_model._on_pellet_sent(perf_c=10.25, context="stale")

    assert not app_model._trial_ledger.active_attempt.is_presented
    assert algorithm.pellets_presented == 0


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
    finish_trial.assert_called_once_with("1.1", TrialOutcome.PENDING_ANALYSIS)


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


def test_stop_finishes_auxiliary_data_after_raw_writers_close(app_model):
    app_model._pending_session_end_perf = 12.5
    app_model._session_boundary = SessionBoundary(
        session_id=app_model.project.short_id,
        primary_camera="left",
        primary_frame_id=42,
        start_perf_time=10.0,
        start_wall_time=100.0,
        camera_when=1_000_000.0,
    )
    app_model._session_analysis_finished = False
    app_model._set_session_recording_status(SessionRecordingStatus.STOPPING)

    with mock.patch.object(
        app_model._session_data_recorder, "stop"
    ) as stop, mock.patch.object(
        app_model, "_save_project_metadata"
    ) as save_metadata:
        app_model._complete_stopped_recording(app_model.project)

    stop.assert_called_once_with(12.5)
    save_metadata.assert_called_once_with(
        app_model.project,
        caller="raw_writers_closed",
    )
    assert app_model.session_recording_status is SessionRecordingStatus.ANALYZING


def test_stop_snapshots_trial_ledger_with_pending_analysis_outcome(app_model):
    app_model._pending_session_end_perf = 12.5
    app_model._session_boundary = SessionBoundary(
        session_id=app_model.project.short_id,
        primary_camera="left",
        primary_frame_id=42,
        start_perf_time=10.0,
        start_wall_time=100.0,
        camera_when=1_000_000.0,
    )
    ledger = PelletTrialLedger(app_model.project.short_id)
    ledger.begin_send(10.5, 100.5, operation_id="send-1")
    ledger.acknowledge_presentation(10.75, 100.75)
    app_model._trial_ledger = ledger
    app_model._session_analysis_finished = False
    app_model._set_session_recording_status(SessionRecordingStatus.STOPPING)

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
    assert attempt.outcome is TrialOutcome.PENDING_ANALYSIS
    assert attempt.capture_end_perf_time == 12.5
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
    app_model._session_boundary = SessionBoundary(
        session_id=project.short_id,
        primary_camera="left",
        primary_frame_id=42,
        start_perf_time=10.0,
        start_wall_time=1_800_000_000.25,
        camera_when=1_000_000.0,
        end_perf_time=12.0,
        end_wall_time=1_800_000_002.25,
    )
    output = tmp_path / "metadata"

    app_model._save_metadata(
        project,
        project.when,
        str(output),
        project.session,
    )

    saved = json.loads(output.with_suffix(".json").read_text())
    assert saved["start_record_timestamp"] == 1_800_000_000.25
    assert saved["sessionBoundary"]["startWallTime"] == 1_800_000_000.25
    assert saved["sessionBoundary"]["endPerfTime"] == 12.0


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
    app_model._pending_session_end_perf = 12.0
    app_model._session_boundary = SessionBoundary(
        session_id=app_model.project.short_id,
        primary_camera="left",
        primary_frame_id=42,
        start_perf_time=10.0,
        start_wall_time=100.0,
        camera_when=1_000_000.0,
    )
    app_model._session_analysis_finished = True
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

    assert app_model._session_data_complete is False
    assert app_model._session_enabled_sources == result["enabledSources"]
    assert app_model._session_boundary.nidaq_sample_index == 100
    assert (
        app_model.subsystem_statuses[SubsystemId.OFFLINE_ANALYSIS.value].state
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


def test_required_failed_subsystem_is_exposed_as_recording_blocker(app_model):
    app_model._acquisition_started = True
    app_model._set_subsystem_status(
        SubsystemId.NIDAQ_STREAM,
        SubsystemState.FAILED,
        required_for_recording=True,
        error="configured input device unavailable",
    )

    assert app_model.recording_blockers == (
        "nidaq_stream: configured input device unavailable",
    )


def test_required_runtime_loss_aborts_session_without_stopping_acquisition(
    app_model,
):
    app_model._acquisition_started = True
    app_model._set_subsystem_status(
        SubsystemId.NIDAQ_STREAM,
        SubsystemState.READY,
        required_for_recording=True,
    )
    app_model._set_session_recording_status(SessionRecordingStatus.RECORDING)

    with mock.patch.object(app_model, "abort_recording") as abort:
        app_model._set_subsystem_status(
            SubsystemId.NIDAQ_STREAM,
            SubsystemState.FAILED,
            error="worker stopped",
        )
        app_model._abort_recording_for_required_subsystem(
            SubsystemId.NIDAQ_STREAM,
            "worker stopped",
        )

    abort.assert_called_once_with()
    assert app_model.acquisition_started


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

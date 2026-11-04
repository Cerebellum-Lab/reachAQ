"""Live poses up to the moment inference stopped are kept, and the session says why they end.

The data monitor appends live poses to the session's h5 files every 750 frames
(5 s at 150 FPS) and writes the rest when the end-of-recording marker arrives.
When live inference stopped mid-recording, no marker ever came:
- the unwritten tail, up to 5 s of poses per camera, was lost;
- Stop waited 10 s for files that would never close;
- the session reported "pose persistence failed: Timed out waiting for live
  pose files to close", although most of the poses were on disk.
"""

import ast
from types import SimpleNamespace

import source_contract
from autotrainer.inference import pose_result_process
from tools.acquisition.model import app_model
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.app_model_status import SessionRecordingStatus
from tools.acquisition.model.subsystem_status import SubsystemId


def _app(subsystem_id, recording_status):
    statuses = {subsystem_id: SimpleNamespace(subsystem_id=subsystem_id)}
    return SimpleNamespace(
        # Keyed like the real registry: by the id's string, whichever form is asked.
        _acquisition=SimpleNamespace(subsystems=SimpleNamespace(
            get=lambda key: statuses.get(getattr(key, "value", key)))),
        _recording_session=SimpleNamespace(status=recording_status),
        _live_inference_failed_reason=None,
    )


def test_live_inference_failing_during_a_recording_is_remembered_for_its_finalization():
    app = _app(SubsystemId.LIVE_INFERENCE.value, SessionRecordingStatus.RECORDING)

    AppModel._handle_recording_subsystem_failure(
        app, SubsystemId.LIVE_INFERENCE, "live inference stopped unexpectedly: boom")

    assert app._live_inference_failed_reason == "live inference stopped unexpectedly: boom"


def test_another_subsystem_failing_leaves_the_live_poses_alone():
    app = _app(SubsystemId.CAN_PELLET.value, SessionRecordingStatus.RECORDING)

    AppModel._handle_recording_subsystem_failure(app, SubsystemId.CAN_PELLET, "CAN down")

    assert app._live_inference_failed_reason is None


def test_a_failure_outside_a_recording_is_not_charged_to_one():
    app = _app(SubsystemId.LIVE_INFERENCE.value, SessionRecordingStatus.READY)

    AppModel._handle_recording_subsystem_failure(
        app, SubsystemId.LIVE_INFERENCE, "live inference stopped unexpectedly: boom")

    assert app._live_inference_failed_reason is None


def _resets_in(function_name):
    function = source_contract.function(app_model, function_name)
    return [node for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(source_contract.dotted_name(target) == "self._live_inference_failed_reason"
                    for target in node.targets)
            and isinstance(node.value, ast.Constant) and node.value.value is None]


def test_each_recording_starts_with_no_live_inference_failure():
    """Reset where a recording is armed, beside the other per-recording state."""
    assert _resets_in("_start_recording_locked")


def test_finalization_does_not_wait_for_pose_files_inference_already_closed():
    """The 10 s wait is only for a recording whose inference was still running."""
    tree = source_contract.tree(app_model)
    guarded = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "_live_inference_failed_reason" in ast.dump(node.test):
            if any(source_contract.calls(statement, "wait_pose_closed")
                   for statement in node.orelse):
                guarded = True
    assert guarded, "wait_pose_closed() is not in the else of the live-inference-failed branch"


def test_the_data_monitor_closes_the_live_files_when_its_loop_ends_mid_recording():
    """Pinned on the parse tree: the loop needs live pose workers to run.

    After the main loop, a recording still in progress is closed the way the
    end-of-recording marker closes it."""
    monitor = source_contract.function(pose_result_process, "_monitor_data_queue")
    main_loop = next(node for node in monitor.body
                     if isinstance(node, ast.While)
                     and source_contract.dotted_name(node.test) == "self._is_running")
    after_loop = monitor.body[monitor.body.index(main_loop) + 1:]
    closes = [node for statement in after_loop for node in ast.walk(statement)
              if isinstance(node, ast.If)
              and source_contract.dotted_name(node.test) == "recording_in_progress"
              and any(source_contract.calls(inner, "close_live_recording") for inner in node.body)]
    assert closes, "nothing after the main loop closes a recording still in progress"
    # And the marker path uses the same close, so the two cannot drift apart.
    assert len(source_contract.calls(monitor, "close_live_recording")) >= 2

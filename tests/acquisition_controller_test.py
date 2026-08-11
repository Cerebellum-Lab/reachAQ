from tools.acquisition.model.acquisition_controller import AcquisitionController
from tools.acquisition.model.subsystem_status import SubsystemId, SubsystemState


def test_acquisition_lifecycle_is_idempotent():
    controller = AcquisitionController()

    assert controller.begin_start() is True
    assert controller.begin_start() is False
    controller.mark_started()
    assert controller.started is True
    assert controller.starting is False
    assert controller.begin_stop() is True
    assert controller.begin_stop() is False
    controller.mark_stopped()
    assert controller.started is False
    assert controller.stopping is False


def test_camera_failure_is_separate_from_synchronization_readiness():
    controller = AcquisitionController()
    controller.subsystems.transition(
        SubsystemId.camera("left"),
        SubsystemState.READY,
        required_for_recording=True,
    )
    controller.subsystems.transition(
        SubsystemId.camera("right"),
        SubsystemState.FAILED,
        required_for_recording=True,
        error="right camera unavailable",
    )
    controller.subsystems.transition(
        SubsystemId.REACH_SYNCHRONIZATION,
        SubsystemState.BLOCKED,
        required_for_recording=True,
        reason="enabled reach cameras are not all ready",
    )

    assert controller.subsystems.get(SubsystemId.camera("left")).is_ready
    assert not controller.synchronization_ready
    blockers = controller.subsystems.recording_blockers()
    assert any("camera.right" in blocker for blocker in blockers)
    assert any("reach_synchronization" in blocker for blocker in blockers)

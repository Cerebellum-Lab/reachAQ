from tools.acquisition.model.subsystem_status import (
    SubsystemId,
    SubsystemState,
    SubsystemStatusRegistry,
)


def test_registry_tracks_retry_generation_and_preserves_first_start():
    registry = SubsystemStatusRegistry()

    starting, _ = registry.begin_retry(
        SubsystemId.NIDAQ_STREAM,
        required_for_recording=True,
        reason="initializing",
    )
    failed, _ = registry.transition(
        SubsystemId.NIDAQ_STREAM,
        SubsystemState.FAILED,
        error="device unavailable",
    )
    retrying, previous = registry.begin_retry(SubsystemId.NIDAQ_STREAM)

    assert starting.generation == 1
    assert failed.started_perf_time == starting.started_perf_time
    assert retrying.generation == 2
    assert previous is failed
    assert retrying.required_for_recording


def test_recording_blockers_include_only_required_unready_domains():
    registry = SubsystemStatusRegistry()
    registry.transition(
        SubsystemId.NIDAQ_STREAM,
        SubsystemState.FAILED,
        required_for_recording=True,
        error="PXI chassis unavailable",
    )
    registry.transition(
        SubsystemId.TOP_CAPTURE,
        SubsystemState.FAILED,
        required_for_recording=False,
        error="preview camera missing",
    )
    registry.transition(
        SubsystemId.CAN_PELLET,
        SubsystemState.READY,
        required_for_recording=True,
    )

    assert registry.recording_blockers() == (
        "nidaq_stream: PXI chassis unavailable",
    )


def test_snapshot_serializes_enum_state_and_dynamic_camera_id():
    registry = SubsystemStatusRegistry()
    camera_id = SubsystemId.camera("left")
    registry.transition(
        camera_id,
        SubsystemState.READY,
        required_for_recording=True,
        reason="first frame received",
    )

    snapshot = registry.snapshot()

    assert snapshot[camera_id]["state"] == "ready"
    assert snapshot[camera_id]["required_for_recording"] is True

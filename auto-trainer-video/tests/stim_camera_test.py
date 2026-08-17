import h5py
import numpy
import pytest

from autotrainer.video import (
    StimCameraDetectionConfiguration,
    StimCameraDetector,
    StimEvidenceWriter,
    StimRoiDefinition,
)


def _configuration(**kwargs):
    return StimCameraDetectionConfiguration(
        enabled=True,
        roi=StimRoiDefinition(threshold=50, hysteresis=5),
        evidence_batch_size=2,
        **kwargs,
    )


def _arm():
    return {
        "session_generation": 3,
        "operation_id": "op-1",
        "logical_trial_id": 4,
        "attempt_id": 2,
        "nonce": "once",
    }


def test_first_reach_detector_matches_legacy_metric_and_is_one_shot():
    detector = StimCameraDetector(_configuration())
    detector.arm(_arm())
    below = numpy.ones((10, 10), dtype=numpy.uint8) * 4
    above = numpy.ones((10, 10), dtype=numpy.uint8) * 6

    value, decision = detector.process(below, 10, 1000, 1.0)
    assert value == 40
    assert decision is None
    value, decision = detector.process(above, 11, 1100, 1.1)
    assert value == 60
    assert decision.arm.operation_id == "op-1"
    assert detector.process(above, 12, 1200, 1.2)[1] is None


def test_detector_rejects_stale_disarm():
    detector = StimCameraDetector(_configuration())
    detector.arm(_arm())
    assert not detector.disarm("another")
    assert detector.arm_context.operation_id == "op-1"
    assert detector.disarm("op-1")
    assert detector.arm_context is None


def test_future_roi_roles_are_serializable_but_not_runnable():
    StimRoiDefinition(name="roi_1", runnable=False)
    with pytest.raises(ValueError, match="future"):
        StimRoiDefinition(name="roi_2", runnable=True)


def test_evidence_writer_batches_hdf5_rows(tmp_path):
    configuration = _configuration()
    detector = StimCameraDetector(configuration)
    arm = detector.arm(_arm())
    path = tmp_path / "stim_camera_evidence.h5"
    writer = StimEvidenceWriter(path, configuration)
    writer.append(
        frame_id=1, camera_timestamp_ns=10, frame_perf_time=1.0,
        value=40, threshold=50, arm=arm, decision=None,
    )
    writer.append(
        frame_id=2, camera_timestamp_ns=20, frame_perf_time=1.1,
        value=60, threshold=50, arm=arm, decision=None,
    )
    writer.close()

    with h5py.File(path, "r") as store:
        assert store["evidence"].shape == (2,)
        assert store["evidence"]["frame_id"].tolist() == [1, 2]
        assert store.attrs["dropped_batches"] == 0

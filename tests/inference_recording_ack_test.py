import threading
from types import SimpleNamespace

from autotrainer.inference import InferenceMonitorDataMsg
from tools.acquisition.model.inference_model import InferenceModel


def _make_model():
    model = InferenceModel.__new__(InferenceModel)
    model._is_enabled = True
    model._live_recording_closed_event = threading.Event()
    model._live_recording_closed_session_id = None
    return model


def test_pose_writer_acknowledgement_is_scoped_to_session():
    model = _make_model()
    expected = SimpleNamespace(short_id="trial-2")
    stale = SimpleNamespace(short_id="trial-1")

    model.prepare_session_recording(expected)
    model._handle_monitor_data_proc_msg(
        InferenceMonitorDataMsg.LIVE_RECORDING_CLOSED,
        ((stale,), {}),
    )
    assert not model.wait_session_pose_closed(expected, timeout=0.01)

    model._handle_monitor_data_proc_msg(
        InferenceMonitorDataMsg.LIVE_RECORDING_CLOSED,
        ((expected,), {}),
    )
    assert model.wait_session_pose_closed(expected, timeout=0.01)


def test_disabled_inference_has_no_pose_writer_to_wait_for():
    model = _make_model()
    model._is_enabled = False

    assert model.wait_session_pose_closed(
        SimpleNamespace(short_id="trial-1"),
        timeout=0,
    )

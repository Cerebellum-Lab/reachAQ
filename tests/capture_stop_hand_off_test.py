"""Handing the end of a recording to live inference cannot end the capture loop.

perform_stop_recording pads and marks the pose queue before it reports the
recording's last frame to the app. When that hand-off raised - a 5 s timeout on
a pose queue whose reader had died - the capture loop ended there, the report
was never sent, and the camera went down with the loop.

Checked on the parse tree: the function is a closure inside the capture loop,
which needs a camera to run.
"""

import ast

import pytest

import source_contract
from autotrainer.video import video_capture


def _stop_recording():
    return source_contract.function(video_capture, "perform_stop_recording")


def _try_around(call_name):
    for node in ast.walk(_stop_recording()):
        if isinstance(node, ast.Try) and any(
                source_contract.calls(statement, call_name) for statement in node.body):
            return node
    return None


@pytest.mark.parametrize("call", ["pad_to_batch_size", "put_frame_index_category"])
def test_a_failed_hand_off_is_caught_and_not_raised_again(call):
    guard = _try_around(call)

    assert guard is not None, f"{call}() is not inside a try in perform_stop_recording"
    assert any(source_contract.dotted_name(handler.type) == "Exception"
               for handler in guard.handlers)
    for handler in guard.handlers:
        assert not any(isinstance(node, ast.Raise) for node in ast.walk(handler)), (
            "the handler raises again")


def test_the_recording_end_is_still_reported_after_the_hand_off():
    order = source_contract.call_order(
        _stop_recording(), ["put_frame_index_category", "_set_status"])

    assert order == ["put_frame_index_category", "_set_status"]

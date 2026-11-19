"""The live path's batch size has to be the one a fast path is built for.

The YOLO model's own cases (its graph batch, its runtime detail) are in
plugins/yolo/tests/test_live_batch.py.

One loaded model serves live inference and the offline pass. The offline batch
is padded - two cameras times three frames - while live inference sends two
frames per tick. A CUDA graph captured at the padded size never replays for a
live call, and the fall-back to the ultralytics wrapper is silent: on the rig
that was 7.4 ms per call instead of 3.6 ms, with nothing in the log to say so.
"""


import source_contract

from autotrainer.inference import pose_process
from autotrainer.inference.pose_model import PoseModel


def test_the_base_hook_does_nothing():
    """A backend indifferent to the input shape should not have to know."""
    model = PoseModel()
    model.prepare_live_batch(2)  # must not raise


def test_pose_process_prepares_the_live_batch_before_loading():
    """Order matters: the graph is captured inside load().

    Checked against the parse tree rather than the text, because standing up a
    pose process with a live queue and a real model is a long way to go to
    verify two adjacent statements - and because the text form broke on a
    reformat while missing a genuine reordering.
    """
    order = source_contract.call_order(
        pose_process, ["prepare_live_batch", "load"])
    assert order[:2] == ["prepare_live_batch", "load"], (
        f"prepare_live_batch must run before load(); found {order}")

    call = source_contract.one_call(pose_process, "prepare_live_batch")
    assert source_contract.dotted_name(call.args[0]) == (
        "self._live_input_queue.batch_size"), (
        "the live queue's batch size is what the live path sends")

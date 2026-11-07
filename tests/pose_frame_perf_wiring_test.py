"""A live pose batch has to leave the pose process with its frames' capture times.

The capture stamps every frame with frame_perf_c, the camera's timestamp fitted
to when the host saw the frame (not the exposure instant; see video_capture),
and the live queue hands those times to the pose process with the frames. Live
tracking needs them downstream: timing pose by frame ID and a nominal frame rate
drifts by the camera's rate error, about 100 ppm on the rig, which ages reach
evidence past the cue gate's limit within the first hour of a session.

Checked by reading the wiring rather than by starting the process: a real start
needs a GPU, a trained model and a multiprocessing queue.

The batch's latency stamps ride fourth on the data queue and the capture times
fifth.
"""

import ast

import source_contract

from autotrainer.inference import pose_process


def _pose_batch_puts():
    return [
        call for call in source_contract.calls(pose_process, "d_q_put")
        if call.args and isinstance(call.args[0], ast.Tuple)
        and len(call.args[0].elts) == 5
    ]


def test_the_pose_batch_is_sent_with_its_frames_exposure_times():
    puts = _pose_batch_puts()
    assert len(puts) == 1, "the pose batch no longer carries a fifth item"
    assert source_contract.dotted_name(puts[0].args[0].elts[4]) == "frames_perf_out"


def test_the_exposure_times_sent_are_a_copy_of_the_live_batch():
    """A copy, for the same reason frames_indices is copied: the next batch
    overwrites the array while the queue may still be serializing this one."""
    sent = source_contract.assigned(pose_process, "frames_perf_out")
    assert sent is not None, "frames_perf_out is not assigned"
    assert "frames_perf_c1" in sent and "'copy'" in sent, sent
    assert "live_input" in sent, "offline batches carry no live exposure times"

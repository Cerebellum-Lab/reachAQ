"""The live path's batch size has to be the one a fast path is built for.

One loaded model serves live inference and the offline pass. The offline batch
is padded - two cameras times three frames - while live inference sends two
frames per tick. A CUDA graph captured at the padded size never replays for a
live call, and the fall-back to the ultralytics wrapper is silent: on the rig
that was 7.4 ms per call instead of 3.6 ms, with nothing in the log to say so.
"""

import re
from pathlib import Path

from autotrainer.inference import pose_process
from autotrainer.inference.pose_model import PoseModel
from autotrainer.inference.yolo import YoloPoseModel


def test_the_base_hook_does_nothing():
    """A backend indifferent to the input shape should not have to know."""
    model = PoseModel()
    model.prepare_live_batch(2)  # must not raise


def test_yolo_takes_the_live_batch_for_its_graph():
    model = YoloPoseModel("/unused", batch_size=6)
    assert model._batch_size == 6
    model.prepare_live_batch(2)
    assert model._batch_size == 2


def test_a_meaningless_batch_is_ignored():
    """Nothing usable is better left alone than captured at a zero shape."""
    model = YoloPoseModel("/unused", batch_size=6)
    model.prepare_live_batch(0)
    assert model._batch_size == 6


def test_runtime_detail_names_the_batch_the_graph_is_captured_for():
    """The batch matters as much as the graph, so both belong in the line."""
    model = YoloPoseModel("/unused", batch_size=6)
    model.prepare_live_batch(2)
    model._graph = object()
    detail = model.runtime_detail()
    assert "CUDA graph" in detail
    assert "batch 2" in detail


def test_pose_process_prepares_the_live_batch_before_loading():
    """Order matters: the graph is captured inside load().

    Asserted against the source because the alternative is standing up a pose
    process with a live queue and a real model, which is a long way to go to
    check two adjacent statements.
    """
    source = Path(pose_process.__file__).read_text(encoding="utf-8")
    prepare = source.index("prepare_live_batch(")
    load = source.index("self._pose_model.load()")
    assert prepare < load, "prepare_live_batch must run before load()"
    assert re.search(
        r"prepare_live_batch\(\s*self\._live_input_queue\.batch_size\s*\)",
        source,
    ), "the live queue's batch size is what the live path sends"

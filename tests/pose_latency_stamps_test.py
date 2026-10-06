import numpy as np

from autotrainer.core.fixed_array_multiqueue import FixedArrayMultiQueue
from autotrainer.inference.pose_process import take_newest_live_batch
from autotrainer.inference.pose_result_process import split_pose_item


def _put_pair(queue, cam_frame_id, index):
    frame = np.zeros((4, 4), dtype=np.uint8)
    for camera in range(2):
        queue.put(frame, camera, index, block=False, frame_perf_c=1.0,
                  cam_frame_id=cam_frame_id + camera * 1000)


def test_the_drain_reports_skips_and_the_newest_batches_ids():
    queue = FixedArrayMultiQueue(depth=3, cam_count=2, frames_per_camera=1, shape=(4, 4),
                                 name="pose-latency")
    for number in range(3):
        _put_pair(queue, 10 + number, number)
    buffer = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    indices = np.zeros((2, 1), dtype=np.int64)
    ids = np.zeros((2, 1), dtype=np.int64)
    put_perf = np.zeros((2, 1))
    stats = {}

    took = take_newest_live_batch(queue, buffer, indices, None, drain=True,
                                  cam_frame_ids=ids, put_perf_c=put_perf, stats=stats)

    assert took is True
    assert stats["skipped"] == 2
    assert ids[:, 0].tolist() == [12, 1012]


def test_a_three_element_item_has_no_latency():
    assert split_pose_item(("pose", "live", "idx")) == ("pose", "live", "idx", None)


def test_a_four_element_item_keeps_its_latency():
    assert split_pose_item(("pose", "live", "idx", (1,)))[3] == (1,)

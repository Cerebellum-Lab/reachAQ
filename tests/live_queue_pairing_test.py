"""Every live batch must pair the cameras' frames of the same exposure.

The live queue keeps one ring per camera and the capture loops put with
block=False, so a camera whose ring is full drops that frame on its own. When
the reader frees a slot between the two cameras' puts of the same frame, one
camera keeps a frame the other dropped. Paired by slot position, every later
batch then put left frame N+1 beside right frame N, and the pose was
triangulated from two instants 6.7 ms apart, until the rings next happened to
refill. Measured on the rig on 2026-10-06 with the live model: 0.19% of
recorded batches, in runs of up to 33 consecutive batches (0.22 s). A frame
one camera never received shifted the pairing the same way.

What must hold:

  * a frame only one camera holds is skipped, and later batches pair again;
  * a frame whose partner has not arrived yet waits for it;
  * the end-of-recording marker still arrives as one batch after a skip, with
    the padding that evens out the writers' counts never paired with a frame.
"""

import ast
import threading

import numpy

import source_contract

from autotrainer.core.fixed_array_multiqueue import BufferResult
from autotrainer.core.fixed_array_multiqueue import FixedArrayMultiQueue
from autotrainer.core.frame_index import FrameIndexCategory

SHAPE = (4, 4)
CAMERAS = 2
LIVE = FrameIndexCategory.ONLINE_NO_RECORDING


def _queue():
    # The live queue's geometry: depth 2, one frame per camera.
    return FixedArrayMultiQueue(2, CAMERAS, 1, SHAPE, name="test")


def _put(queue, camera, frame, frame_idx=LIVE):
    """Put camera frame number `frame` the way the capture loop does: the
    pixels carry the frame number so a batch shows which frames it paired."""
    return queue.put(numpy.full(SHAPE, frame, dtype=numpy.uint8), camera, frame_idx,
                     block=False, frame_id=frame)


def _take_batch(queue):
    output = numpy.zeros((CAMERAS, *SHAPE, 3), dtype=numpy.uint8)
    indices = numpy.zeros((CAMERAS, 1), dtype="int64")
    if not queue.get_output(output, indices, timeout=0):
        return None
    return tuple(int(output[cam, 0, 0, 0]) for cam in range(CAMERAS)), indices[:, 0].tolist()


def _take(queue):
    batch = _take_batch(queue)
    return None if batch is None else batch[0]


def _overflow_on_camera_0_only(queue, frame_idx=lambda frame: LIVE):
    """The race seen on the rig: both rings full, frame 12 reaches camera 0
    before the reader frees a slot and camera 1 after it."""
    for frame in (10, 11):
        assert _put(queue, 0, frame, frame_idx(frame)) == BufferResult.Ok
        assert _put(queue, 1, frame, frame_idx(frame)) == BufferResult.Ok
    assert _put(queue, 0, 12, frame_idx(12)) == BufferResult.Overflow
    assert _take(queue) == (10, 10)
    assert _put(queue, 1, 12, frame_idx(12)) == BufferResult.Ok
    assert _take(queue) == (11, 11)


def test_a_frame_one_camera_dropped_on_overflow_does_not_shift_later_pairs():
    queue = _queue()
    _overflow_on_camera_0_only(queue, frame_idx=lambda frame: frame - 10)

    # Paired by position these were (13, 12), (14, 13), ...
    for frame in range(13, 19):
        _put(queue, 0, frame, frame - 10)
        _put(queue, 1, frame, frame - 10)
        assert _take(queue) == (frame, frame)


def test_a_frame_one_camera_never_received_does_not_shift_later_pairs():
    queue = _queue()
    _put(queue, 0, 10)
    _put(queue, 1, 10)
    assert _take(queue) == (10, 10)
    _put(queue, 0, 11)  # camera 1's capture loop never saw frame 11

    # Not recording, so the frame index is the same -1 for every frame and
    # only the camera's frame id can tell them apart.
    for frame in range(12, 18):
        _put(queue, 0, frame)
        _put(queue, 1, frame)
        assert _take(queue) == (frame, frame)


def test_an_unmatched_frame_waits_for_its_partner():
    queue = _queue()
    _put(queue, 0, 15)  # camera 0 dropped 14 and already has 15
    _put(queue, 1, 14)  # camera 1 has 14 and has not delivered 15 yet

    assert _take(queue) is None, "a batch without camera 1's frame 15 was served"
    _put(queue, 1, 15)
    assert _take(queue) == (15, 15)

    # The skipped slot went back to camera 1: its ring holds two frames again.
    for frame in (16, 17):
        assert _put(queue, 0, frame) == BufferResult.Ok
        assert _put(queue, 1, frame) == BufferResult.Ok
    assert _take(queue) == (16, 16)
    assert _take(queue) == (17, 17)


def test_the_end_of_recording_marker_still_arrives_as_one_batch_after_a_skip():
    queue = _queue()
    _overflow_on_camera_0_only(queue, frame_idx=lambda frame: frame - 10)
    _put(queue, 0, 13, 3)
    _put(queue, 1, 13, 3)
    assert _take(queue) == (13, 13)  # camera 1's unmatched 12 was skipped

    # perform_stop_recording, in each capture loop at once: even out the put
    # counts with padding, then send the marker. Camera 0 put 3 frames and
    # camera 1 put 4, so camera 0 pads one.
    empty = numpy.zeros(SHAPE, dtype=numpy.uint8)

    def stop(camera, puts):
        queue.pad_to_batch_size(camera, empty, puts, timeout=5)
        queue.put_frame_index_category(empty, FrameIndexCategory.EOF_RECORDING,
                                       cam_idx=camera, timeout=5)

    writers = [threading.Thread(target=stop, args=(0, 3)),
               threading.Thread(target=stop, args=(1, 4))]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(10)

    batch = _take_batch(queue)
    assert batch is not None
    assert batch[1] == [FrameIndexCategory.EOF_RECORDING] * CAMERAS, (
        "the marker was split across batches or paired with padding")

    _put(queue, 0, 14)
    _put(queue, 1, 14)
    assert _take_batch(queue) == ((14, 14), [LIVE, LIVE]), (
        "a second marker or a stray frame followed the end of recording")


# --- the wiring --------------------------------------------------------------


def test_the_capture_loop_names_each_frame_it_offers():
    """Pairing needs each frame's camera id, and losing it would be silent: the
    queue falls back to pairing by position and every test above still passes.

    Read from the code, like the capture loop's other inference wiring in
    live_queue_freshness_test, because running the loop needs a camera.
    """
    call = source_contract.one_call(
        "auto-trainer-video/src/autotrainer/video/video_capture.py", "net_q_put")
    frame_id = source_contract.keyword(call, "frame_id")
    assert frame_id is not None, "the capture loop no longer passes frame ids"
    names = {node.id for node in ast.walk(frame_id) if isinstance(node, ast.Name)}
    assert "cam_frame_id" in names, (
        "the capture loop passes something other than the camera's frame id")

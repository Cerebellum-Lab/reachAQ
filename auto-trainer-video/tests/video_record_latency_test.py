import queue
import threading

import h5py
import numpy

from autotrainer.core.latency import latency_stream_path
from autotrainer.video import VideoRecord, VideoRecordMode, VideoRecordProperties


def test_each_batch_written_gets_a_latency_row(project_info):
    input_queue = queue.Queue()
    stopped = threading.Semaphore(0)
    recorder = VideoRecord(
        VideoRecordProperties(project_info=project_info, name="left", frame_size=(32, 24),
                              fps=30, record_mode=VideoRecordMode.TRIGGER,
                              video_rotate_interval=0),
        input_queue,
        record_stop_sema=stopped,
    )
    recorder.start()
    try:
        frame = numpy.zeros((24, 32), dtype=numpy.uint8)
        input_queue.put([(idx, frame, idx / 30.0, idx / 30.0) for idx in range(0, 5)])
        input_queue.put([(idx, frame, idx / 30.0, idx / 30.0) for idx in range(5, 8)])
        input_queue.put([])  # end of recording
        assert stopped.acquire(timeout=10), "recorder never closed the recording"
    finally:
        recorder.cancel()
        recorder.join(5)

    with h5py.File(latency_stream_path(project_info, "record_left"), "r") as store:
        writes = store["record_writes"][:]
    assert writes["first_frame_id"].tolist() == [0, 5]
    assert writes["last_frame_id"].tolist() == [4, 7]
    assert writes["frame_count"].tolist() == [5, 3]
    assert (writes["dequeue_perf"] <= writes["write_entry_perf"]).all()
    assert (writes["write_entry_perf"] <= writes["write_return_perf"]).all()

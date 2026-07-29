import numpy as np

from autotrainer.core import (
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
)
from autotrainer.device import NidaqSignalSampleBlock
from tools.acquisition.model.nidaq_sample_ring import SharedNidaqSampleRing


def test_one_epoch_anchor_keeps_times_monotonic_across_jittered_blocks():
    channel = NidaqSignalChannelConfiguration("signal", "Dev1/ai0")
    configuration = NidaqSignalStreamConfiguration(
        channels=(channel,),
        is_enabled=True,
        sample_rate_hz=1000.0,
        read_chunk_size=3,
    )
    ring = SharedNidaqSampleRing(configuration, capacity=12)
    destination = np.empty((1, ring.capacity), dtype=np.float32)

    ring.write_block(NidaqSignalSampleBlock(
        wall_time=100.010,
        perf_time=10.010,
        sample_rate_hz=1000.0,
        sample_index=0,
        channels=(channel,),
        values={"signal": (1.0, 2.0, 3.0)},
        epoch_perf_time=10.0,
        epoch_wall_time=100.0,
    ))
    first = ring.copy_since(None, destination)
    ring.write_block(NidaqSignalSampleBlock(
        # This observation moves backward, reproducing the old failure mode.
        wall_time=100.009,
        perf_time=10.009,
        sample_rate_hz=1000.0,
        sample_index=3,
        channels=(channel,),
        values={"signal": (4.0, 5.0, 6.0)},
        epoch_perf_time=10.0,
        epoch_wall_time=100.0,
    ))
    second = ring.copy_since(first.end_sample_index, destination)

    first_indices = np.arange(first.start_sample_index, first.end_sample_index)
    second_indices = np.arange(second.start_sample_index, second.end_sample_index)
    perf = np.concatenate((
        first.perf_times(first_indices),
        second.perf_times(second_indices),
    ))

    assert np.all(np.diff(perf) > 0)
    assert perf.tolist() == [10.0, 10.001, 10.002, 10.003, 10.004, 10.005]
    assert second.source_perf_time == 10.009

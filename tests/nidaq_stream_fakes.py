"""Stand-ins for the NI-DAQ worker and discovery, for stream lifecycle tests.

A module of its own, with nothing heavy imported, because the stream worker
runs in a forkserver child that imports whatever module its target lives in.
Kept beside a test file, the child would import the whole application to
reach one function, and a slow import there reads as a worker that never
became ready.
"""

from tools.acquisition.model.nidaq_discovery import NidaqDevicePorts


def discover_dev1():
    return (NidaqDevicePorts(name="Dev1"),), None


def idle_worker(_configuration, _timing_plan, message_queue, _sample_ring,
                stop_event, _log_dict_config):
    """Come up, then hold the stream open until asked to stop."""
    message_queue.put(("ready", None))
    stop_event.wait(60.0)
    message_queue.put(("stopped", None))


def failing_worker(_configuration, _timing_plan, message_queue, _sample_ring,
                   _stop_event, _log_dict_config):
    """Fail before becoming ready, as a driver refusing the task does."""
    message_queue.put(("error", "the task was refused"))
    message_queue.put(("stopped", None))

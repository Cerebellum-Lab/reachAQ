"""Each board keeps its own clock model and sequence tracker.

Before this, one model and one tracker served the whole bus: a second board's
trailer looked like a reboot of the first and reset its clock, and a trailer on
a frame the host drops was never counted, so the next one looked like a gap.
"""

from autotrainer.device import CanTransportConfiguration, Target
from autotrainer.device.can_interface import CanInterface
from autotrainer.device.socketcan_jerrycan import (
    JerryCANCmdType,
    JerryCANMsg,
    JerryCANTimestampKind,
    JerryCANTiming,
    SyncClapObserved,
    TimeSyncResponse,
)

PELLET = 0x00
IO_MODULE = 0x08


def _interface():
    interface = CanInterface(
        required_targets=(Target.PELLET_DEVICE,),
        can_transport=CanTransportConfiguration(kind="socketcan", fd=True),
    )
    interface._get_index = lambda item: item.index
    interface._get_timestamp_ns = lambda item: item.timestamp_ns
    return interface


def _timed(message_type, address, *, boot_id, sequence, board_time_us, index=1, kind=JerryCANTimestampKind.TX_QUEUE):
    message = JerryCANMsg()
    message.type = message_type
    message.dst_id = address
    message.index = index
    message.board_timing_valid = True
    message.timing = JerryCANTiming(
        version=1, kind=kind, boot_id=boot_id, sequence=sequence, board_time_us=board_time_us,
    )
    return message


def _sync_pellet(interface, *, boot_id=7):
    response = _timed(
        JerryCANCmdType.TIME_SYNC_RESPONSE, PELLET,
        boot_id=boot_id, sequence=1, board_time_us=1_000_100, index=101_000_300_000,
    )
    response.time_sync_response = TimeSyncResponse(
        request_id=5, request_receive_time_us=1_000_000, response_queue_time_us=1_000_200,
    )
    interface._time_sync_requests[5] = 101_000_000_000
    assert interface._translate(response).clock_model


def test_another_boards_trailer_does_not_reset_the_pellet_clock():
    interface = _interface()
    _sync_pellet(interface)
    clap = _timed(
        JerryCANCmdType.SYNC_CLAP_OBSERVED, IO_MODULE, boot_id=99, sequence=40, board_time_us=5_000,
        kind=JerryCANTimestampKind.OBSERVED,
    )
    clap.sync_clap_observed = SyncClapObserved(tick=3, clapper_node=IO_MODULE)
    interface._translate(clap)

    tone = interface._translate(_timed(
        JerryCANCmdType.TONE, PELLET, boot_id=7, sequence=2, board_time_us=1_500_000,
        kind=JerryCANTimestampKind.PHYSICAL_START,
    ))

    assert tone.timestamp_method == "board_clock_affine"
    assert tone.board_aligned_perf_time is not None


def test_a_trailer_on_a_dropped_frame_is_not_counted_as_lost():
    interface = _interface()
    interface._translate(_timed(JerryCANCmdType.TONE, PELLET, boot_id=7, sequence=1, board_time_us=10))
    # A trailer-bearing frame the host does not turn into a Source.
    interface._translate(_timed(JerryCANCmdType.DOOR_SENSOR, PELLET, boot_id=7, sequence=2, board_time_us=20))

    tone = interface._translate(_timed(JerryCANCmdType.TONE, PELLET, boot_id=7, sequence=3, board_time_us=30))

    assert tone.board_sequence_status["gap"] == 0


def test_a_reboot_invalidates_only_that_boards_clock():
    interface = _interface()
    _sync_pellet(interface, boot_id=7)

    tone = interface._translate(_timed(
        JerryCANCmdType.TONE, PELLET, boot_id=8, sequence=1, board_time_us=1_500_000,
        kind=JerryCANTimestampKind.PHYSICAL_START,
    ))

    assert tone.board_aligned_perf_time is None
    assert tone.timestamp_method == "host_receive_perf_time"

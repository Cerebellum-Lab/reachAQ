"""Before recording, the selected protocol is checked for what would stop it.

Ben, 2026-10-02: "laser 2 profile was never set ... before recording there
needs to be a check to see if the selected protocol will be able to run as
expected (ex. shutters enabled?, profiles selected for each laser, etc)".
Errors block Record; warnings are shown and Record stays available.

check_protocol_readiness is pure: these tests give it christielab10's laser
block, its NI-DAQ stream and timing plan, and a snapshot of the app's state,
and no hardware is touched.
"""

import dataclasses
import io
import json
from pathlib import Path

import pytest

from autotrainer.core import NidaqTimingRoute, SystemConfiguration

from nidaq_timing_test import _pinned_christielab10
from tools.acquisition.model.nidaq_channel_plan import (
    build_nidaq_acquisition_configuration,
)
from tools.acquisition.model.protocol_readiness import (
    STIM_CAMERA_REQUIRED,
    ProtocolRigConfiguration,
    ReadinessLiveState,
    check_protocol_readiness,
    protocol_blocker_lines,
)
from tools.acquisition.model.stimulus_profile_repository import StimulusProfileLibrary
from tools.acquisition.model.trial_action import LaserPulseProfile, StimulusTriggerProfile
from tools.acquisition.model.trial_protocol_schedule import (
    CoverPolicy,
    LaserTriggerRoute,
    TrialProtocolDocument,
    TrialProtocolRow,
    TrialProtocolSchedule,
)

HERE = Path(__file__).parent
#: The protocol Ben ran on 2026-10-01, as exported on christielab10
#: (/tmp/hwprotocol, sha256 bf4ec730...). Trials 3-5 fire laser 1 and trials
#: 6-8 laser 2, each 1.5 s before the reveal.
BENCH_PROTOCOL = HERE / "bench-laser-validation-2026-10-01.json"


#: christielab10's NI-DAQ and laser blocks, as the rig's file holds them.
CHRISTIELAB10 = SystemConfiguration.load_yaml(io.StringIO(
    f"!SystemConfiguration\nversion: {SystemConfiguration.version}\n"
    + (HERE / "christielab10_nidaq_blocks.yaml").read_text(encoding="utf-8")
    + (HERE / "christielab10_laser_block.yaml").read_text(encoding="utf-8")))


def _rig(laser):
    """The rig's configuration with `laser`, and the acquisition plan it gives."""
    return ProtocolRigConfiguration(
        laser=laser,
        nidaq_stream=build_nidaq_acquisition_configuration(
            CHRISTIELAB10.nidaq_stream, CHRISTIELAB10.nidaq_ports, laser),
    )


RIG = _rig(CHRISTIELAB10.laser)
#: The profile the bench protocol names, as the rig's library held it:
#: 2 V, 31 x 23 ms at 29 Hz, no PMT margins.
LASER1 = LaserPulseProfile("laser1", 1, 2.0, 23.0, pulse_count=31, frequency_hz=29.0)
LIBRARY = StimulusProfileLibrary(laser_profiles=(LASER1,))
#: The timing plan the stream runs on christielab10's two boards.
PLAN = _pinned_christielab10()
#: System Mode running on christielab10: the stream running on its shared
#: clock, the controller opened on that plan, stimCam disabled, and a
#: profile picked on both laser tabs.
READY = ReadinessLiveState(
    laser_connected=True,
    nidaq_stream_state="running",
    timing_plan=PLAN,
    laser_timing_plan=PLAN,
    stim_camera_enabled=False,
    laser_tab_profiles={1: "laser1", 2: "laser1"},
)


def _laser_row(trial_id, channel=1, **changes):
    """A pre-reveal laser row as the bench protocol has them."""
    values = {
        "enabled": True,
        "cover_policy": "reveal",
        "laser_profile_id": "laser1",
        "laser_phase": "pellet_presentation",
        "laser_trigger_route": "hardware_stim3",
        "laser_channel_id": channel,
        "stimulus_assignment": "always",
        "stimulus_trigger": "pre_reveal",
        "pre_reveal_ms": 1500,
    }
    values.update(changes)
    return TrialProtocolRow(trial_id=trial_id).with_updates(values)


def _check(*rows, configuration=RIG, library=LIBRARY, live=READY):
    return check_protocol_readiness(
        TrialProtocolSchedule(rows), configuration, library, live)


def _errors(findings):
    return [(item.trial_id, item.message) for item in findings if item.severity == "error"]


def _warnings(findings):
    return [(item.trial_id, item.message) for item in findings if item.severity == "warning"]


def _laser(configuration, channel_id, **changes):
    """`configuration` with one laser channel changed."""
    laser = configuration.laser
    return dataclasses.replace(configuration, laser=dataclasses.replace(laser, channels=tuple(
        dataclasses.replace(channel, **changes) if int(channel.channel_id) == channel_id
        else channel
        for channel in laser.channels)))


# ------------------------------------------------------------------ errors


def test_a_row_whose_profile_is_not_in_the_library_is_an_error():
    findings = _check(_laser_row(4, laser_profile_id="laser9"))

    (trial, message), = _errors(findings)
    assert trial == 4
    assert "laser9" in message and "profile library" in message


def test_a_profile_beyond_the_lasers_command_range_is_an_error():
    # christielab10's lasers take 0..5 V.
    hot = LaserPulseProfile("hot", 1, 6.0, 23.0)
    findings = _check(
        _laser_row(3, laser_profile_id="hot"),
        library=StimulusProfileLibrary(laser_profiles=(LASER1, hot)))

    assert _errors(findings) == [(3, "Profile hot is 6 V; laser 1 accepts 0..5 V")]


def test_a_board_stim_row_on_a_laser_with_no_board_stim_line_is_an_error():
    # Laser 2 with boardStimLine unset, as the configuration could leave it.
    findings = _check(
        _laser_row(6, channel=2), _laser_row(7, channel=2),
        configuration=_laser(RIG, 2, board_stim_line=None))

    assert [trial for trial, _message in _errors(findings)] == [6, 7]
    assert all("Laser 2 has no board STIM line" in message
               for _trial, message in _errors(findings))


def test_a_backplane_trigger_with_nothing_routed_onto_it_is_an_error():
    # Laser 2 arms on PXI_Trig2, which only the PFI1 route drives: without
    # triggerRouteSource the STIM2 pulse never reaches it.
    findings = _check(
        _laser_row(6, channel=2),
        configuration=_laser(RIG, 2, trigger_route_source=None))

    (trial, message), = _errors(findings)
    assert trial == 6
    assert "PXI_Trig2" in message and "triggerRouteSource" in message


def _route_errors(findings):
    return [(trial, message) for trial, message in _errors(findings)
            if "triggerRouteSource" in message]


def test_a_laser_row_that_never_fires_needs_no_route():
    # Assignment disabled: the runtime's compile drops the laser
    # (stimulus_selected False, no firing), and the trial runs without it.
    row = _laser_row(6, channel=2, stimulus_assignment="disabled", stimulus_trigger="none",
                     pre_reveal_ms=0, cover_policy="keep_current")

    findings = _check(row, configuration=_laser(RIG, 2, trigger_route_source=None))

    assert _route_errors(findings) == []


def test_a_backplane_line_another_lasers_route_drives_needs_no_route_of_its_own():
    # Laser 2 armed on PXI_Trig0: the controller routes laser 1's PFI0 onto
    # that line when it opens, for every channel, and holds it until close.
    configuration = _laser(RIG, 2, trigger_source="/PXI1Slot4/PXI_Trig0",
                           trigger_route_source=None, board_stim_line=3)

    assert _route_errors(_check(_laser_row(6, channel=2), configuration=configuration)) == []


def test_a_backplane_line_an_external_timing_route_drives_needs_no_route():
    plan = READY.timing_plan
    plan = dataclasses.replace(plan, routes=plan.routes + (NidaqTimingRoute(
        "laser_2_trigger", "/PXI1Slot5/PFI1", ("/PXI1Slot5/PXI_Trig2",)),))

    findings = _check(_laser_row(6, channel=2),
                      configuration=_laser(RIG, 2, trigger_route_source=None),
                      live=dataclasses.replace(READY, timing_plan=plan))

    assert _route_errors(findings) == []


def test_a_row_on_a_laser_that_is_not_configured_is_an_error():
    findings = _check(_laser_row(3, channel=3))

    assert _errors(findings) == [(3, "Laser 3 is not configured")]


def test_a_pre_reveal_row_without_the_reveal_cover_policy_is_an_error():
    # A saved row cannot hold this (the row rules refuse it), but the
    # runtime refuses it again mid-trial (_configure_protocol_cover), so the
    # check does too, on rows built without those rules.
    covered = dataclasses.replace(_laser_row(5), cover_policy=CoverPolicy.COVER)
    direct = dataclasses.replace(
        _laser_row(6), laser_trigger_route=LaserTriggerRoute.DIRECT_NI_SOFTWARE)

    errors = _errors(_check(covered, direct))

    assert (5, "Pre-reveal stimulation requires the pellet cover policy Reveal") in errors
    assert (6, "Pre-reveal stimulation requires a board STIM laser row") in errors


def test_a_randomized_row_that_can_draw_pre_reveal_is_an_error():
    # The known defect: execution keys on the row's own stimulus_trigger,
    # which a randomized row leaves at none, so a drawn Pre-reveal arms the
    # laser and nothing ever starts it.
    profile = StimulusTriggerProfile("mixed", 1, categories=(
        {"category_id": "pre", "trigger": "pre_reveal", "label": "Pre-reveal",
         "percentage": 50.0, "offset_ms": 1500},
        {"category_id": "tone1", "trigger": "tone_1", "label": "Tone 1",
         "percentage": 50.0},
    ))
    row = _laser_row(
        9, stimulus_assignment="randomized", stimulus_trigger="none", pre_reveal_ms=0,
        stimulus_trigger_profile_id="mixed")

    findings = _check(row, library=dataclasses.replace(
        LIBRARY, stimulus_trigger_profiles=(profile,)))

    (trial, message), = _errors(findings)
    assert trial == 9
    assert "randomized" in message and "Pre-reveal" in message


def test_a_first_reach_row_without_the_stim_camera_reuses_the_existing_blocker():
    row = _laser_row(
        4, cover_policy="keep_current", stimulus_trigger="first_reach", pre_reveal_ms=0)

    findings = _check(row)

    assert _errors(findings) == [(None, STIM_CAMERA_REQUIRED)]
    # recording_blockers already lists it, live; Record shows it once.
    assert protocol_blocker_lines(findings) == ()
    assert _errors(_check(row, live=dataclasses.replace(READY, stim_camera_enabled=True))) == []


def test_a_closed_laser_controller_is_an_error():
    # The controller's plan is not judged until it is open; its error says it.
    findings = _check(_laser_row(3), live=dataclasses.replace(
        READY, laser_connected=False, laser_timing_plan=None))

    (trial, message), = _errors(findings)
    assert trial is None
    assert "laser controller is not open" in message and "trial 3" in message


def test_a_laser_close_still_running_is_an_error():
    refusal = "the laser controller is still closing"
    findings = _check(_laser_row(3), live=dataclasses.replace(
        READY, laser_close_refusal=refusal))

    (trial, message), = _errors(findings)
    assert trial is None and refusal in message


def test_a_stopped_nidaq_stream_is_an_error():
    findings = _check(_laser_row(3), live=dataclasses.replace(
        READY, nidaq_stream_state="stopped"))

    (trial, message), = _errors(findings)
    assert trial is None
    assert "NI-DAQ stream is stopped" in message


def test_a_timing_plan_without_the_lasers_board_is_an_error():
    # "Protocol laser timing is not ready", which failed trials mid-session:
    # the laser's PXI-6713 is not on the stream's shared clock.
    plan = dataclasses.replace(READY.timing_plan, hardware_output_devices=())

    findings = _check(_laser_row(3), _laser_row(6, channel=2),
                      live=dataclasses.replace(READY, timing_plan=plan, laser_timing_plan=plan))

    errors = _errors(findings)
    assert [trial for trial, _message in errors] == [None, None]
    assert all("Protocol laser timing is not ready" in message for _trial, message in errors)
    assert "laser 1" in errors[0][1] and "laser 2" in errors[1][1]
    assert "not in the resolved timing topology" in errors[0][1]


def test_a_laser_opened_without_a_plan_cannot_take_a_board_trigger():
    # The controller keeps the plan it opened with, here none, whatever the
    # stream runs now.
    findings = _check(_laser_row(3), live=dataclasses.replace(READY, laser_timing_plan=None))

    (trial, message), = _errors(findings)
    assert trial is None
    assert "laser 1 (trial 3)" in message
    assert "opened without a valid NI timing plan" in message
    assert "restart System Mode" in message


def test_a_laser_opened_on_an_earlier_plan_than_the_streams_is_an_error():
    earlier = dataclasses.replace(PLAN, sample_clock_source="/PXI1Slot5/te0/SampleClock")

    (trial, message), = _errors(_check(_laser_row(3), live=dataclasses.replace(
        READY, laser_timing_plan=earlier)))

    assert trial is None and "opened on an earlier NI timing plan" in message


def test_a_stream_plan_that_differs_only_where_the_laser_does_not_read_is_no_error():
    # The monitor sets its plan twice in one start, as built and again after
    # the multidevice probe, and the stream restarts after every recording:
    # its bookkeeping can differ from what the laser opened with while the
    # clock and terminals the laser uses do not.
    again = dataclasses.replace(
        PLAN, multidevice_probe_status=f"{PLAN.multidevice_probe_status}, again",
        reason=f"{PLAN.reason} (built again)", task_graph=None)
    assert again != PLAN

    findings = _check(_laser_row(3), _laser_row(6, channel=2),
                      live=dataclasses.replace(READY, timing_plan=again))

    assert _errors(findings) == []


def test_a_laser_without_hardware_timing_cannot_take_a_board_trigger():
    # With hardwareTimed off the controller is given no timing plan, and a
    # board STIM pulse needs it.
    configuration = dataclasses.replace(
        RIG, laser=dataclasses.replace(RIG.laser, hardware_timed=False))

    (trial, message), = _errors(_check(_laser_row(3), configuration=configuration))

    assert trial is None and "hardwareTimed" in message


def test_only_future_enabled_rows_are_checked():
    completed = _laser_row(2, laser_profile_id="laser9")
    future = _laser_row(3, laser_profile_id="laser9")
    # Last, so it is where the protocol ends rather than where it stops.
    disabled = dataclasses.replace(_laser_row(4, laser_profile_id="laser9"), enabled=False)

    findings = _check(completed, future, disabled, live=dataclasses.replace(
        READY, completed_trial_ids=(2,)))

    assert [trial for trial, _message in _errors(findings)] == [3]


def test_a_disabled_row_before_enabled_ones_stops_the_session_there():
    # The ledger takes the next trial in order, never skipping one, and the
    # compile refuses a disabled row ("Protocol row is disabled"): every SEND
    # from trial 4 on is refused, and trials 5-8 never run.
    rows = [_laser_row(trial) for trial in range(1, 9)]
    rows[3] = dataclasses.replace(rows[3], enabled=False)

    (trial, message), = _errors(_check(*rows))

    assert trial == 4
    assert "disabled" in message
    assert "4 enabled trials after it" in message and "from trial 5" in message
    assert "enable trial 4, or disable the trials after it" in message


def test_each_disabled_row_with_enabled_rows_after_it_is_named():
    rows = [_laser_row(trial) for trial in range(1, 7)]
    for index in (1, 3):
        rows[index] = dataclasses.replace(rows[index], enabled=False)

    assert [trial for trial, _message in _errors(_check(*rows))] == [2, 4]


def test_disabled_rows_at_the_end_are_where_the_protocol_ends():
    rows = [_laser_row(1), _laser_row(2)] + [
        dataclasses.replace(_laser_row(trial), enabled=False) for trial in (3, 4)]

    assert _errors(_check(*rows)) == []


def test_a_row_the_runtime_compile_refuses_is_an_error_in_its_words():
    # Anything else the runtime's own compile refuses, here a tone profile
    # that is not saved, is reported with the compiler's message.
    row = TrialProtocolRow(trial_id=2).with_updates({
        "enabled": True, "tone_profile_id": "beep", "tone_phase": "before_send"})

    assert _errors(_check(row)) == [(2, "Unknown tone profile 'beep'")]


# ---------------------------------------------------------------- warnings


def test_pmt_margins_without_a_pmt_shutter_line_are_a_warning():
    margins = LaserPulseProfile("margins", 1, 2.0, 23.0, pmt_open_lead_ms=50.0,
                                pmt_close_lag_ms=20.0)
    findings = _check(
        _laser_row(3, laser_profile_id="margins"),
        library=StimulusProfileLibrary(laser_profiles=(margins,)))

    assert _errors(findings) == []
    pmt = [message for _trial, message in _warnings(findings) if "PMT" in message]
    assert len(pmt) == 1
    assert "margins" in pmt[0] and "pmtShutterOutput" in pmt[0] and "trial 3" in pmt[0]


def test_a_laser_whose_trigger_is_not_read_back_is_a_warning():
    # christielab10's lasers have no triggerMonitorInput.
    findings = _check(_laser_row(3))

    readback = [message for _trial, message in _warnings(findings) if "readback" in message]
    assert len(readback) == 1 and readback[0].startswith("laser 1 ")
    # christielab10 records the edge all the same, as laser1_trigger_readback
    # on ai9, a custom channel the check cannot tie to the laser: the
    # warning says what is missing, and not that nothing is recorded.
    assert "triggerMonitorInput" in readback[0]
    assert "not recorded" not in readback[0]

    # Read back on ai9, which the acquisition plan then acquires: no warning.
    configuration = _rig(_laser(RIG, 1, trigger_monitor_input="PXI1Slot5/ai9").laser)
    findings = _check(_laser_row(3), configuration=configuration)
    assert not [message for _trial, message in _warnings(findings) if "readback" in message]


def test_the_shutters_and_diodes_are_listed_once_to_check_by_hand():
    findings = _check(_laser_row(3), _laser_row(6, channel=2))

    lines = [message for _trial, message in _warnings(findings) if "shutter" in message]
    assert len(lines) == 1
    for pin in ("PXI1Slot5/port0/line4", "PXI1Slot5/ai8",
                "PXI1Slot5/port0/line5", "PXI1Slot5/ai4"):
        assert pin in lines[0]


def test_a_laser_with_no_tab_profile_is_said_to_be_fine():
    # The 2026-10-02 confusion: laser 2's tab had no profile, which a
    # protocol row does not use. Said, informationally, and never an error.
    findings = _check(_laser_row(6, channel=2), live=dataclasses.replace(
        READY, laser_tab_profiles={1: "laser1"}))

    assert _errors(findings) == []
    tab = [message for _trial, message in _warnings(findings) if "Laser Control tab" in message]
    assert len(tab) == 1
    assert "laser 2" in tab[0] and "own profile" in tab[0]


# ------------------------------------------------------------ Record's lines


def test_records_lines_name_the_trials_and_stop_after_five():
    findings = _check(
        _laser_row(6, channel=2), _laser_row(7, channel=2), _laser_row(8, channel=2),
        configuration=_laser(RIG, 2, board_stim_line=None))

    lines = protocol_blocker_lines(findings)
    assert lines == (
        "Protocol check: trials 6-8: Laser 2 has no board STIM line; set "
        "boardStimLine for it in the system configuration",
    )

    many = _check(*(_laser_row(trial, laser_profile_id=f"missing{trial}")
                    for trial in range(1, 9)))
    lines = protocol_blocker_lines(many)
    assert len(lines) == 6
    assert lines[0].startswith("Protocol check: trial 1: ")
    assert lines[-1] == "Protocol check: and 3 more; see Check protocol on the Protocol tab"


# ------------------------------------------------------------------- clean


def test_the_2026_10_01_validation_protocol_is_clean_on_christielab10():
    # The real export, against christielab10's own configuration blocks. Its
    # laser rows were compiled offline the same way on 2026-10-01; nothing
    # here stops it.
    document = TrialProtocolDocument.from_record(
        json.loads(BENCH_PROTOCOL.read_text(encoding="utf-8")))

    findings = check_protocol_readiness(document, RIG, LIBRARY, READY)

    assert _errors(findings) == []
    # What is left to say: the trigger readbacks are not configured, and
    # the shutters and diodes are for a person to check.
    assert {message.split(" ", 2)[1] for _trial, message in _warnings(findings)
            if "readback" in message} == {"1", "2"}
    assert any("shutter" in message for _trial, message in _warnings(findings))

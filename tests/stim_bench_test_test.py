import pytest

from tools.acquisition.model.stim_bench_test import (
    BenchRecipe,
    StimTestResult,
    bench_wait_seconds,
    refuse_reason,
)
from tools.acquisition.model.trial_action import LaserPulseProfile
from tools.acquisition.model.trial_protocol_schedule import LaserTriggerRoute


def make_profile(**overrides):
    values = dict(
        profile_id="stim-a",
        revision=1,
        channel_id=1,
        amplitude_volts=2.0,
        pulse_duration_ms=5.0,
        trigger_route=LaserTriggerRoute.HARDWARE_STIM3,
        trigger_terminal="/Dev1/PFI0",
        trigger_pulse_us=1000,
    )
    values.update(overrides)
    return LaserPulseProfile(**values)


def allowed(**overrides):
    values = dict(
        profile=make_profile(),
        recording_status_value="ready",
        trial_operation_active=False,
        laser_backend="nidaq",
        configured_channel_ids=(1, 2),
        firmware_capabilities=("finite_stim3_pulse",),
    )
    values.update(overrides)
    return values


def test_a_fully_configured_bench_test_is_allowed():
    assert refuse_reason(**allowed()) is None


def test_a_recording_session_refuses_the_test():
    reason = refuse_reason(**allowed(recording_status_value="recording"))

    assert reason is not None
    assert "recording" in reason.lower()


def test_an_active_trial_operation_refuses_the_test():
    reason = refuse_reason(**allowed(trial_operation_active=True))

    assert reason is not None
    assert "trial" in reason.lower()


def test_a_non_nidaq_laser_backend_refuses_the_test():
    reason = refuse_reason(**allowed(laser_backend="null"))

    assert reason is not None
    assert "nidaq" in reason.lower()


def test_an_unconfigured_channel_refuses_the_test():
    reason = refuse_reason(**allowed(configured_channel_ids=(2, 3)))

    assert reason is not None
    assert "channel 1" in reason.lower()


def test_a_board_that_reports_capabilities_without_this_one_is_refused():
    reason = refuse_reason(**allowed(firmware_capabilities=("time_sync",)))

    assert reason is not None
    assert "finite_stim3_pulse" in reason


def test_a_board_that_reports_no_capabilities_at_all_is_allowed():
    # No released pellet firmware answers the capability request, so an empty
    # set means "did not say", not "cannot". Refusing here would block the path
    # the trial route already drives successfully with no capability gate.
    assert refuse_reason(**allowed(firmware_capabilities=())) is None


def test_a_software_start_profile_is_allowed_without_a_terminal_or_the_board():
    # A software start involves neither the trigger terminal nor the board,
    # so neither may refuse it.
    reason = refuse_reason(
        **allowed(
            profile=make_profile(
                trigger_route=LaserTriggerRoute.DIRECT_NI_SOFTWARE,
                trigger_terminal="",
            ),
            firmware_capabilities=("time_sync",),
        )
    )

    assert reason is None


def test_a_board_profile_without_a_terminal_is_still_refused():
    # Built directly, since the profile itself refuses to exist without one.
    profile = make_profile()
    object.__setattr__(profile, "trigger_terminal", "")

    reason = refuse_reason(**allowed(profile=profile))

    assert reason is not None
    assert "terminal" in reason


def test_a_missing_profile_refuses_the_test():
    reason = refuse_reason(**allowed(profile=None))

    assert reason is not None
    assert "profile" in reason.lower()


def test_the_recording_check_is_case_insensitive():
    reason = refuse_reason(**allowed(recording_status_value="RECORDING"))

    assert reason is not None


def test_the_bench_recipe_never_claims_a_session_identity():
    recipe = BenchRecipe()

    assert recipe.session_generation == 0
    assert recipe.logical_trial_id == 0
    assert recipe.session_id == "bench"
    assert "bench" in recipe.protocol_id


def test_each_bench_recipe_gets_its_own_operation_id():
    assert BenchRecipe().operation_id != BenchRecipe().operation_id


def test_the_result_reads_as_a_measurement_when_it_completed():
    result = StimTestResult(
        profile_id="stim-a",
        channel_id=1,
        trigger_terminal="/Dev1/PFI0",
        trigger_pulse_us=1000,
        arm_to_terminal_ms=3.5,
        detail="completed",
    )

    text = str(result)
    assert "stim-a" in text
    assert "3.50 ms" in text


def test_the_result_says_so_when_no_timing_was_measured():
    result = StimTestResult(
        profile_id="stim-a",
        channel_id=1,
        trigger_terminal="/Dev1/PFI0",
        trigger_pulse_us=1000,
        arm_to_terminal_ms=None,
        detail="board acknowledged but the waveform did not report terminal",
    )

    assert "did not report terminal" in str(result)


def test_a_software_start_result_says_it_was_started_from_the_host():
    result = StimTestResult(
        profile_id="stim-s",
        channel_id=2,
        trigger_terminal="",
        trigger_pulse_us=1000,
        arm_to_terminal_ms=5012.0,
        detail="completed",
        trigger_route="direct_ni_software",
    )

    text = str(result)
    assert "software start" in text
    assert "STIM" not in text
    assert "5012.00 ms" in text


def test_a_profile_knows_how_long_its_waveform_runs():
    # burst_100hz_5s on christielab10: 500 pulses of 1 ms at 100 Hz.
    profile = make_profile(pulse_duration_ms=1.0, pulse_count=500,
                           frequency_hz=100.0, baseline_ms=10.0, post_stim_ms=20.0)

    assert profile.waveform_seconds == pytest.approx(0.001 + 4.99 + 0.030)


def test_the_bench_test_waits_for_the_whole_waveform():
    # The wait was max(3 s, trigger pulse + 2 s), so a 5 s burst was reported
    # as never finishing while it was still running.
    profile = make_profile(pulse_duration_ms=1.0, pulse_count=500, frequency_hz=100.0)

    assert bench_wait_seconds(profile) > profile.waveform_seconds + 1.0


def test_a_short_profile_keeps_the_old_minimum_wait():
    assert bench_wait_seconds(make_profile()) == pytest.approx(3.0)

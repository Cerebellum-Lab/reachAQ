import dataclasses

import pytest
import yaml

from autotrainer.core import SystemConfiguration
from autotrainer.core.configuration import (
    SessionControlConfiguration,
    SystemConfigurationDumper,
    SystemConfigurationLoader,
)


def test_defaults_use_grouped_attempts_and_fifteen_second_drain():
    configuration = SessionControlConfiguration()

    assert configuration.attempt_assignment == "retry_within_trial"
    assert configuration.automatic_protocol_advance_enabled is False
    assert configuration.trial_count_basis == "completed"
    assert configuration.intertrial_analysis_enabled is False
    assert configuration.intertrial_progression_mode == "continue"
    assert configuration.behavioral_retry_outcomes == ()
    assert configuration.stop_drain_timeout_seconds == 15.0
    assert configuration.duration_limit_seconds is None
    assert configuration.trial_limit is None


def test_session_control_round_trips_with_system_configuration():
    configuration = SystemConfiguration()
    configuration.behavior.session_control = SessionControlConfiguration(
        automatic_pellet_cycles_enabled=True,
        automatic_protocol_advance_enabled=True,
        intertrial_analysis_enabled=True,
        intertrial_progression_mode="wait",
        behavioral_retry_outcomes=("no_reach", "pellet_missing"),
        attempt_assignment="successful_presentations_only",
        retry_settings="resample",
        trial_count_basis="presented",
        counted_trial_outcomes=("success", "incomplete"),
        duration_limit_seconds=600,
        trial_limit=25,
        stop_on_protocol_complete=True,
    )

    dumped = yaml.dump(configuration, Dumper=SystemConfigurationDumper)
    loaded = yaml.load(dumped, Loader=SystemConfigurationLoader)

    assert dataclasses.asdict(loaded.behavior.session_control) == dataclasses.asdict(
        configuration.behavior.session_control
    )


def test_session_duration_is_capped_at_two_hours():
    SessionControlConfiguration(duration_limit_seconds=7200)
    with pytest.raises(ValueError, match="cannot exceed 2 hours"):
        SessionControlConfiguration(duration_limit_seconds=7200.1)


def test_missing_pellet_retry_does_not_require_intertrial_analysis():
    configuration = SessionControlConfiguration(
        behavioral_retry_outcomes=("pellet_missing",),
    )
    assert configuration.intertrial_analysis_enabled is False
    assert configuration.behavioral_retry_outcomes == ("pellet_missing",)


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"attempt_assignment": "opaque"}, "attempt assignment"),
        ({"retry_settings": "opaque"}, "retry settings"),
        ({"trial_count_basis": "opaque"}, "trial count basis"),
        ({"intertrial_progression_mode": "opaque"}, "progression mode"),
        ({"behavioral_retry_outcomes": ("opaque",), "intertrial_analysis_enabled": True}, "retry outcome"),
        ({"behavioral_retry_outcomes": ("no_reach",)}, "require intertrial analysis"),
        ({"trial_count_basis": "scored"}, "requires intertrial analysis"),
        ({"counted_trial_outcomes": ("hardware_error",)}, "outcome"),
        ({"duration_limit_seconds": 0}, "duration"),
        ({"trial_limit": 0}, "Trial target"),
        ({"stop_drain_timeout_seconds": 0}, "drain timeout"),
    ),
)
def test_invalid_operator_values_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SessionControlConfiguration(**kwargs)

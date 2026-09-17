"""Re-applying the timing already in force is not a change.

Demo mode is entered by reloading the configuration, which re-applies the
NI-DAQ timing it already has. While acquiring, that raised, the toggle
reported "Could not change demo mode" and swallowed the reason, and the clip
never played - so Next/Previous trial stayed disabled with nothing on screen
to say why.
"""

import pytest

from tools.acquisition.model.nidaq_signal_monitor_model import (
    NidaqSignalMonitorModel,
)


def _model():
    model = NidaqSignalMonitorModel.__new__(NidaqSignalMonitorModel)
    model._is_running = False
    model._is_starting = False
    model._timing_configuration = None
    model._hardware_timed_output_devices = ()
    model._hardware_timed_output_channels = ()
    model._device_identities = ()
    model._runtime_device_aliases = {}
    model._timing_plan = None
    model._set_timing_plan = lambda plan: setattr(model, "_timing_plan", plan)
    return model


def test_reapplying_identical_timing_while_running_is_allowed():
    model = _model()
    model.configure_timing("timing-a", hardware_timed_output_devices=["Dev1"])
    model._is_running = True

    # Must not raise: nothing is being changed.
    model.configure_timing("timing-a", hardware_timed_output_devices=["Dev1"])

    assert model._timing_configuration == "timing-a"


def test_a_real_timing_change_while_running_is_still_refused():
    model = _model()
    model.configure_timing("timing-a")
    model._is_running = True

    with pytest.raises(RuntimeError, match="while acquisition is active"):
        model.configure_timing("timing-b")


def test_a_changed_output_device_while_running_is_still_refused():
    model = _model()
    model.configure_timing("timing-a", hardware_timed_output_devices=["Dev1"])
    model._is_running = True

    with pytest.raises(RuntimeError, match="while acquisition is active"):
        model.configure_timing("timing-a", hardware_timed_output_devices=["Dev2"])


def test_duplicate_device_spellings_do_not_count_as_a_change():
    """configure_timing de-duplicates, so the comparison must too."""
    model = _model()
    model.configure_timing("timing-a", hardware_timed_output_devices=["Dev1"])
    model._is_running = True

    model.configure_timing("timing-a",
                           hardware_timed_output_devices=["Dev1", "Dev1", ""])

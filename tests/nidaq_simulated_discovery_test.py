from reachaq.core import (
    NidaqSignalChannelConfiguration,
    NidaqSignalStreamConfiguration,
    NidaqTimingConfiguration,
)
from reachaq.app.model import nidaq_discovery
from reachaq.app.model.nidaq_discovery import (
    discover_nidaq_devices,
    simulated_nidaq_devices,
)
from reachaq.app.model.nidaq_timing import build_nidaq_timing_plan


def test_simulation_discovery_returns_simdaq1_without_a_subprocess(monkeypatch):
    monkeypatch.setenv("REACHAQ_SIMULATION", "1")

    def no_subprocess(*_args, **_kwargs):
        raise AssertionError("simulated discovery must not start a process")

    monkeypatch.setattr(nidaq_discovery.subprocess, "run", no_subprocess)
    devices, error = discover_nidaq_devices()
    assert error is None
    assert devices == simulated_nidaq_devices()
    assert [device.name for device in devices] == ["SimDAQ1"]
    assert devices[0].product_type == "Simulated"


def test_simdaq1_satisfies_a_single_device_timing_plan():
    configuration = NidaqSignalStreamConfiguration(
        (
            NidaqSignalChannelConfiguration("cam_frames", "SimDAQ1/port0/line0", "digital"),
            NidaqSignalChannelConfiguration("tone1", "SimDAQ1/port0/line1", "digital"),
            NidaqSignalChannelConfiguration("tone2", "SimDAQ1/port0/line31", "digital"),
            NidaqSignalChannelConfiguration("laser1_diode", "SimDAQ1/ai15", "analog"),
            NidaqSignalChannelConfiguration(
                "laser1_copy", "SimDAQ1/_ao0_vs_aognd", "analog"),
        ),
        is_enabled=True,
    )

    plan = build_nidaq_timing_plan(
        configuration,
        NidaqTimingConfiguration(),
        simulated_nidaq_devices(),
        hardware_timed_output_channels=("SimDAQ1/ao0",),
    )

    assert plan.is_valid, plan.reason
    assert plan.resolved_mode == "same_device"
    assert plan.master_device == "SimDAQ1"


def test_simdaq1_digital_only_plan_masters_on_its_counter():
    configuration = NidaqSignalStreamConfiguration(
        (NidaqSignalChannelConfiguration("cam_frames", "SimDAQ1/port0/line0", "digital"),),
        is_enabled=True,
    )
    plan = build_nidaq_timing_plan(
        configuration, NidaqTimingConfiguration(), simulated_nidaq_devices())
    assert plan.is_valid, plan.reason
    assert plan.clock_producer == "counter"

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

from autotrainer.core import (
    NidaqDeviceIdentity,
    NidaqSignalStreamConfiguration,
    NidaqTimingConfiguration,
    NidaqTimingPlan,
    NidaqTimingRoute,
)
from tools.acquisition.model.nidaq_discovery import (
    NidaqDevicePorts,
    device_name_from_channel,
)


def build_nidaq_timing_plan(
    configuration: NidaqSignalStreamConfiguration,
    timing: NidaqTimingConfiguration,
    devices: Sequence[NidaqDevicePorts],
    *,
    hardware_timed_output_devices: Iterable[str] = tuple(),
) -> NidaqTimingPlan:
    discovered = {device.name: device for device in devices}
    input_devices = tuple(dict.fromkeys(
        device_name
        for channel in configuration.channels
        for device_name in (device_name_from_channel(channel.physical_channel),)
        if device_name is not None
    ))
    output_devices = tuple(dict.fromkeys(
        device_name
        for device_name in hardware_timed_output_devices
        if device_name
    ))
    active_devices = tuple(dict.fromkeys((*input_devices, *output_devices)))
    missing = tuple(name for name in active_devices if name not in discovered)
    if missing:
        return _invalid_plan(
            timing,
            "Configured NI-DAQ device(s) unavailable: " + ", ".join(missing),
        )
    if not active_devices:
        return _invalid_plan(timing, "No hardware-timed NI-DAQ tasks are configured")

    try:
        master = _select_master(
            timing.timing_master,
            active_devices,
            input_devices,
            configuration,
            discovered,
        )
    except ValueError as exc:
        return _invalid_plan(timing, str(exc))

    slaves = tuple(device for device in active_devices if device != master)
    task_start_order = (*slaves, master)
    if not slaves:
        return NidaqTimingPlan(
            requested_mode=timing.sync_mode,
            resolved_mode="same_device",
            is_valid=True,
            master_device=master,
            task_start_order=task_start_order,
            synchronization_quality="hardware_same_device",
            reason="All sampled tasks use one NI-DAQ device",
        )

    if timing.sync_mode == "independent":
        valid = not timing.require_hardware_synchronization
        return NidaqTimingPlan(
            requested_mode=timing.sync_mode,
            resolved_mode="independent",
            is_valid=valid,
            master_device=master,
            slave_devices=slaves,
            task_start_order=task_start_order,
            synchronization_quality="independent_host_estimated",
            reason=(
                "Independent device clocks are diagnostic-only"
                if not valid
                else "Independent device clocks explicitly allowed"
            ),
        )

    active_capabilities = tuple(discovered[name] for name in active_devices)
    common_pxi_backplane = _has_common_pxi_backplane(active_capabilities)
    requested = timing.sync_mode
    if requested == "backplane" and not common_pxi_backplane:
        return _invalid_plan(
            timing,
            "Backplane synchronization requested, but active devices do not "
            "share a discoverable PXI chassis",
            master=master,
            slaves=slaves,
        )

    use_external = requested == "external"
    if requested == "auto":
        use_external = not common_pxi_backplane
    if use_external and not (
        timing.start_trigger_source and timing.sample_clock_source
    ):
        return _invalid_plan(
            timing,
            "No deterministic shared NI-DAQ timing route is available; configure "
            "external start-trigger and sample-clock terminals",
            master=master,
            slaves=slaves,
        )

    if use_external:
        reference_clock = timing.reference_clock_source
        start_trigger = timing.start_trigger_source
        sample_clock = timing.sample_clock_source
        resolved_mode = "external"
        quality = "hardware_external"
    else:
        reference_clock = timing.reference_clock_source or "PXI_CLK10"
        start_trigger = timing.start_trigger_source or f"/{master}/ai/StartTrigger"
        master_has_analog_input = any(
            device_name_from_channel(channel.physical_channel) == master
            for channel in configuration.analog_channels
        )
        sample_clock = timing.sample_clock_source or (
            f"/{master}/ai/SampleClock"
            if master_has_analog_input
            else f"/{master}/Ctr0InternalOutput"
        )
        resolved_mode = "backplane"
        quality = "hardware_backplane"

    routes = (
        NidaqTimingRoute(
            "reference_clock",
            reference_clock or "",
            tuple(f"{slave}:reference_clock" for slave in slaves),
        ),
        NidaqTimingRoute(
            "start_trigger",
            start_trigger or "",
            tuple(f"{slave}:start_trigger" for slave in slaves),
        ),
        NidaqTimingRoute(
            "sample_clock",
            sample_clock or "",
            tuple(f"{slave}:sample_clock" for slave in slaves),
        ),
    )
    return NidaqTimingPlan(
        requested_mode=requested,
        resolved_mode=resolved_mode,
        is_valid=True,
        master_device=master,
        slave_devices=slaves,
        reference_clock_source=reference_clock,
        reference_clock_rate_hz=10_000_000.0 if reference_clock == "PXI_CLK10" else None,
        sample_clock_source=sample_clock,
        start_trigger_source=start_trigger,
        routes=routes,
        task_start_order=task_start_order,
        synchronization_quality=quality,
        reason=(
            "Resolved shared PXI reference/start/sample timing"
            if resolved_mode == "backplane"
            else "Resolved explicitly configured external start/sample timing"
        ),
    )


def _select_master(
    override: Optional[NidaqDeviceIdentity],
    active_devices: tuple[str, ...],
    input_devices: tuple[str, ...],
    configuration: NidaqSignalStreamConfiguration,
    discovered: Mapping[str, NidaqDevicePorts],
) -> str:
    if override is not None:
        candidates = tuple(
            device
            for device in discovered.values()
            if (
                override.serial_number is None
                or device.serial_number == override.serial_number
            )
            and (
                override.product_type is None
                or device.product_type == override.product_type
            )
            and (
                override.serial_number is not None
                or override.runtime_name is None
                or device.name == override.runtime_name
            )
        )
        active_candidates = tuple(
            device for device in candidates if device.name in active_devices
        )
        if len(active_candidates) != 1:
            raise ValueError(
                f"Timing master {override.logical_name!r} did not resolve to "
                "exactly one active device with the configured identity"
            )
        return active_candidates[0].name

    for channel in configuration.channels:
        if channel.name == "cam_frames":
            device = device_name_from_channel(channel.physical_channel)
            if device in active_devices:
                return device
    if input_devices:
        analog_device = next(
            (
                device_name_from_channel(channel.physical_channel)
                for channel in configuration.analog_channels
                if device_name_from_channel(channel.physical_channel) in active_devices
            ),
            None,
        )
        return analog_device or input_devices[0]
    return sorted(active_devices)[0]


def _has_common_pxi_backplane(
    devices: Sequence[NidaqDevicePorts],
) -> bool:
    if not devices:
        return False
    if not all("PXI" in device.bus_type.upper() for device in devices):
        return False
    chassis = {device.pxi_chassis_number for device in devices}
    return len(chassis) == 1 and None not in chassis


def _invalid_plan(
    timing: NidaqTimingConfiguration,
    reason: str,
    *,
    master: Optional[str] = None,
    slaves: tuple[str, ...] = tuple(),
) -> NidaqTimingPlan:
    return NidaqTimingPlan(
        requested_mode=timing.sync_mode,
        resolved_mode="unavailable",
        is_valid=False,
        master_device=master,
        slave_devices=slaves,
        task_start_order=(*slaves, *((master,) if master else tuple())),
        synchronization_quality="unavailable",
        reason=reason,
    )

from __future__ import annotations

import dataclasses
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
    channel_error = _validate_channels_and_rates(
        configuration,
        discovered,
    )
    if channel_error:
        return _invalid_plan(timing, channel_error)

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
    resolved_devices = _resolved_device_identities(
        active_devices,
        discovered,
    )
    if not slaves:
        return NidaqTimingPlan(
            requested_mode=timing.sync_mode,
            resolved_mode="same_device",
            is_valid=True,
            master_device=master,
            task_start_order=task_start_order,
            resolved_devices=resolved_devices,
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
            resolved_devices=resolved_devices,
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
    if common_pxi_backplane and any(
        device.digital_trigger_supported is False
        for device in active_capabilities
    ):
        return _invalid_plan(
            timing,
            "Backplane synchronization requires digital trigger support on "
            "every active NI-DAQ device",
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

    unavailable_terminal = next(
        (
            terminal
            for terminal in (
                reference_clock,
                start_trigger,
                sample_clock,
            )
            if terminal
            and not _terminal_is_discoverable(
                terminal,
                active_capabilities,
                allow_pxi_clock=common_pxi_backplane,
            )
        ),
        None,
    )
    if unavailable_terminal is not None:
        return _invalid_plan(
            timing,
            f"Configured NI-DAQ timing terminal is not exposed by discovery: "
            f"{unavailable_terminal}",
            master=master,
            slaves=slaves,
        )

    routes = (
        NidaqTimingRoute(
            "reference_clock",
            reference_clock or "",
            tuple(f"/{slave}/ReferenceClock" for slave in slaves),
        ),
        NidaqTimingRoute(
            "start_trigger",
            start_trigger or "",
            tuple(f"/{slave}/StartTrigger" for slave in slaves),
        ),
        NidaqTimingRoute(
            "sample_clock",
            sample_clock or "",
            tuple(f"/{slave}/SampleClock" for slave in slaves),
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
        resolved_devices=resolved_devices,
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
        selected = active_candidates[0].name
        if not _device_can_master(
            selected,
            configuration,
            discovered[selected],
        ):
            raise ValueError(
                f"Timing master {override.logical_name!r} cannot export a "
                "sample clock for the configured task topology"
            )
        return selected

    for channel in configuration.channels:
        if channel.name == "cam_frames":
            device = device_name_from_channel(channel.physical_channel)
            if (
                device in active_devices
                and _device_can_master(
                    device,
                    configuration,
                    discovered[device],
                )
            ):
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
        if analog_device:
            return analog_device
    compatible = tuple(
        device
        for device in active_devices
        if _device_can_master(
            device,
            configuration,
            discovered[device],
        )
    )
    if not compatible:
        raise ValueError(
            "No active NI-DAQ device can export the sample clock required by "
            "the configured task topology"
        )
    return sorted(compatible)[0]


def resolve_nidaq_device_aliases(
    identities: Sequence[NidaqDeviceIdentity],
    devices: Sequence[NidaqDevicePorts],
) -> Mapping[str, str]:
    aliases = {}
    for identity in identities:
        candidates = tuple(
            device
            for device in devices
            if (
                identity.serial_number is None
                or device.serial_number == identity.serial_number
            )
            and (
                identity.product_type is None
                or device.product_type == identity.product_type
            )
            and (
                identity.serial_number is not None
                or identity.runtime_name is None
                or device.name == identity.runtime_name
            )
        )
        if len(candidates) != 1:
            raise ValueError(
                f"Configured NI-DAQ device {identity.logical_name!r} did not "
                "resolve to exactly one discovered device with the expected "
                "model/serial identity"
            )
        runtime_name = candidates[0].name
        aliases[identity.logical_name] = runtime_name
        if identity.runtime_name:
            aliases[identity.runtime_name] = runtime_name
    return aliases


def resolve_nidaq_stream_configuration(
    configuration: NidaqSignalStreamConfiguration,
    aliases: Mapping[str, str],
) -> NidaqSignalStreamConfiguration:
    channels = tuple(
        dataclasses.replace(
            channel,
            physical_channel=remap_nidaq_physical_channel(
                channel.physical_channel,
                aliases,
            ),
        )
        for channel in configuration.channels
    )
    return dataclasses.replace(configuration, channels=channels)


def resolve_nidaq_device_names(
    names: Iterable[str],
    aliases: Mapping[str, str],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(aliases.get(str(name), str(name)) for name in names)
    )


def _has_common_pxi_backplane(
    devices: Sequence[NidaqDevicePorts],
) -> bool:
    if not devices:
        return False
    if not all("PXI" in device.bus_type.upper() for device in devices):
        return False
    chassis = {device.pxi_chassis_number for device in devices}
    return len(chassis) == 1 and None not in chassis


def _validate_channels_and_rates(configuration, discovered) -> Optional[str]:
    analog_counts = {}
    for channel in configuration.channels:
        device_name = device_name_from_channel(channel.physical_channel)
        device = discovered.get(device_name)
        if device is None:
            continue
        normalized = channel.physical_channel.strip("/")
        if channel.kind == "analog":
            available = {
                name.strip("/")
                for name in device.analog_inputs
            }
            if available and normalized not in available:
                return (
                    f"Configured analog input {channel.physical_channel} is not "
                    f"available on {device_name}"
                )
            analog_counts[device_name] = analog_counts.get(device_name, 0) + 1
        else:
            available = {
                name.strip("/")
                for name in device.digital_inputs
            }
            if available and normalized not in available:
                return (
                    f"Configured digital input {channel.physical_channel} is not "
                    f"available on {device_name}"
                )
    for device_name, channel_count in analog_counts.items():
        device = discovered[device_name]
        maximum = (
            device.analog_input_max_single_channel_rate
            if channel_count == 1
            else device.analog_input_max_multi_channel_rate
        )
        if maximum is not None and configuration.sample_rate_hz > maximum:
            return (
                f"Configured sample rate {configuration.sample_rate_hz:g} Hz "
                f"exceeds {device_name} capability {maximum:g} Hz"
            )
    return None


def _resolved_device_identities(active_devices, discovered):
    return tuple(
        NidaqDeviceIdentity(
            logical_name=name,
            runtime_name=name,
            product_type=discovered[name].product_type or None,
            serial_number=discovered[name].serial_number,
        )
        for name in active_devices
    )


def _device_can_master(device_name, configuration, device) -> bool:
    has_analog_input = any(
        channel.kind == "analog"
        and device_name_from_channel(channel.physical_channel) == device_name
        for channel in configuration.channels
    )
    if has_analog_input:
        return True
    has_digital_input = any(
        channel.kind == "digital"
        and device_name_from_channel(channel.physical_channel) == device_name
        for channel in configuration.channels
    )
    if has_digital_input:
        if device.counter_outputs:
            return True
        # Some simulated/older discovery providers do not expose counter
        # capability. Let task construction be the final validation only when
        # the capability inventory is entirely absent.
        return not (
            device.analog_inputs
            or device.analog_outputs
            or device.digital_inputs
            or device.digital_outputs
            or device.counter_inputs
        )
    return device.analog_output_sample_clock_supported is True


def _terminal_is_discoverable(
    terminal: str,
    devices: Sequence[NidaqDevicePorts],
    *,
    allow_pxi_clock: bool,
) -> bool:
    normalized = terminal.strip().lower()
    if normalized in {"pxi_clk10", "pxiclk10", "/pxi_clk10", "/pxiclk10"}:
        return allow_pxi_clock
    device_name = device_name_from_channel(terminal)
    if device_name is None:
        return True
    device = next(
        (candidate for candidate in devices if candidate.name == device_name),
        None,
    )
    if device is None or not device.terminals:
        return True
    available = {item.strip("/").lower() for item in device.terminals}
    return terminal.strip("/").lower() in available


def remap_nidaq_physical_channel(
    physical_channel: str,
    aliases: Mapping[str, str],
) -> str:
    leading_slash = physical_channel.startswith("/")
    parts = physical_channel.strip("/").split("/", 1)
    if len(parts) != 2:
        return physical_channel
    mapped = aliases.get(parts[0], parts[0])
    result = f"{mapped}/{parts[1]}"
    return f"/{result}" if leading_slash else result


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

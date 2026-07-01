from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple


@dataclasses.dataclass(frozen=True)
class NidaqDevicePorts:
    name: str
    analog_outputs: Tuple[str, ...] = tuple()
    analog_inputs: Tuple[str, ...] = tuple()
    digital_outputs: Tuple[str, ...] = tuple()
    digital_inputs: Tuple[str, ...] = tuple()


def discover_nidaq_devices() -> Tuple[Tuple[NidaqDevicePorts, ...], Optional[str]]:
    try:
        from nidaqmx.system import System
        from nidaqmx.errors import DaqNotFoundError
    except ModuleNotFoundError:
        return tuple(), (
            "NI-DAQmx discovery is unavailable because the nidaqmx Python package is not installed."
        )

    try:
        system = System.local()
        devices = []
        for device in system.devices:
            name = getattr(device, "name", None)
            if not name:
                continue
            devices.append(
                NidaqDevicePorts(
                    name=name,
                    analog_outputs=_channel_names(getattr(device, "ao_physical_chans", tuple())),
                    analog_inputs=_channel_names(getattr(device, "ai_physical_chans", tuple())),
                    digital_outputs=_channel_names(
                        getattr(device, "do_lines", tuple()),
                        getattr(device, "do_physical_chans", tuple()),
                    ),
                    digital_inputs=_channel_names(
                        getattr(device, "di_lines", tuple()),
                        getattr(device, "di_physical_chans", tuple()),
                    ),
                )
            )
        return tuple(devices), None
    except DaqNotFoundError as exc:
        return tuple(), (
            "NI-DAQmx discovery is unavailable because the NI-DAQmx runtime/C driver is not installed "
            f"or not loadable by this OS environment: {exc}"
        )
    except Exception as exc:
        return tuple(), f"NI-DAQmx discovery failed: {exc}"


def device_name_from_channel(channel_name: Optional[str]) -> Optional[str]:
    if not channel_name:
        return None
    parts = channel_name.strip().lstrip("/").split("/")
    if len(parts) < 2 or not parts[0]:
        return None
    return parts[0]


def _channel_names(*collections) -> Tuple[str, ...]:
    names: List[str] = []
    for collection in collections:
        try:
            iterator = iter(collection)
        except TypeError:
            continue
        for channel in iterator:
            name = getattr(channel, "name", None)
            if name is None:
                name = str(channel)
            if name and name not in names:
                names.append(name)
    return tuple(names)

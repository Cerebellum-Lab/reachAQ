from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
import sys
import time
from typing import List, Optional, Tuple

from autotrainer.core.logging import get_verbose_logger, log_hardware_initialization


logger = get_verbose_logger(__name__)


@dataclasses.dataclass(frozen=True)
class NidaqDevicePorts:
    name: str
    product_type: str = ""
    product_number: Optional[int] = None
    serial_number: Optional[int] = None
    analog_outputs: Tuple[str, ...] = tuple()
    analog_inputs: Tuple[str, ...] = tuple()
    digital_outputs: Tuple[str, ...] = tuple()
    digital_inputs: Tuple[str, ...] = tuple()
    counter_outputs: Tuple[str, ...] = tuple()
    counter_inputs: Tuple[str, ...] = tuple()
    bus_type: str = ""
    pxi_chassis_number: Optional[int] = None
    terminals: Tuple[str, ...] = tuple()
    analog_output_sample_clock_supported: Optional[bool] = None
    digital_trigger_supported: Optional[bool] = None
    analog_input_max_single_channel_rate: Optional[float] = None
    analog_input_max_multi_channel_rate: Optional[float] = None
    analog_output_max_rate: Optional[float] = None


def discover_nidaq_devices() -> Tuple[Tuple[NidaqDevicePorts, ...], Optional[str]]:
    started = time.perf_counter()
    log_hardware_initialization(
        logger,
        "START | NI-DAQ discovery | isolated_process=true timeout=10s",
    )
    command = [
        sys.executable,
        "-c",
        (
            "import dataclasses, json; "
            "from tools.acquisition.model.nidaq_discovery import _discover_nidaq_devices_direct; "
            "devices, error = _discover_nidaq_devices_direct(); "
            "print(json.dumps({'devices': [dataclasses.asdict(device) for device in devices], 'error': error}))"
        ),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        error = "NI-DAQmx discovery timed out while probing devices."
        _log_discovery_result(tuple(), error, started)
        return tuple(), error

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        detail = (
            f"signal {-completed.returncode}"
            if completed.returncode < 0
            else f"exit code {completed.returncode}"
        )
        if stderr:
            detail = f"{detail}: {stderr}"
        error = (
            "NI-DAQmx discovery failed in an isolated probe process "
            f"({detail}). The NI-DAQmx native runtime may be installed but unusable in this OS environment."
        )
        _log_discovery_result(tuple(), error, started)
        return tuple(), error

    json_line = _last_stdout_line(completed.stdout)
    if not json_line:
        error = "NI-DAQmx discovery returned no device data."
        _log_discovery_result(tuple(), error, started)
        return tuple(), error

    try:
        payload = json.loads(json_line)
    except json.JSONDecodeError as exc:
        error = f"NI-DAQmx discovery returned invalid device data: {exc}"
        _log_discovery_result(tuple(), error, started)
        return tuple(), error

    devices = tuple(
        NidaqDevicePorts(
            name=str(device["name"]),
            product_type=str(device.get("product_type", "")),
            product_number=device.get("product_number"),
            serial_number=device.get("serial_number"),
            analog_outputs=tuple(device.get("analog_outputs", tuple())),
            analog_inputs=tuple(device.get("analog_inputs", tuple())),
            digital_outputs=tuple(device.get("digital_outputs", tuple())),
            digital_inputs=tuple(device.get("digital_inputs", tuple())),
            counter_outputs=tuple(device.get("counter_outputs", tuple())),
            counter_inputs=tuple(device.get("counter_inputs", tuple())),
            bus_type=str(device.get("bus_type", "")),
            pxi_chassis_number=device.get("pxi_chassis_number"),
            terminals=tuple(device.get("terminals", tuple())),
            analog_output_sample_clock_supported=_optional_bool(
                device.get("analog_output_sample_clock_supported")
            ),
            digital_trigger_supported=_optional_bool(
                device.get("digital_trigger_supported")
            ),
            analog_input_max_single_channel_rate=_optional_float(
                device.get("analog_input_max_single_channel_rate")
            ),
            analog_input_max_multi_channel_rate=_optional_float(
                device.get("analog_input_max_multi_channel_rate")
            ),
            analog_output_max_rate=_optional_float(
                device.get("analog_output_max_rate")
            ),
        )
        for device in payload.get("devices", tuple())
        if device.get("name")
    )
    error = payload.get("error")
    error = str(error) if error else None
    _log_discovery_result(devices, error, started)
    return devices, error


def _log_discovery_result(
    devices: Tuple[NidaqDevicePorts, ...],
    error: Optional[str],
    started: float,
) -> None:
    state = "UNAVAILABLE" if error else "READY"
    log_hardware_initialization(
        logger,
        "%s | NI-DAQ discovery | count=%d devices=%s elapsed=%.3fs%s",
        state,
        len(devices),
        tuple(device.name for device in devices),
        time.perf_counter() - started,
        "" if error is None else f" error={error}",
        level=logging.INFO,
    )


def _discover_nidaq_devices_direct() -> Tuple[Tuple[NidaqDevicePorts, ...], Optional[str]]:
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
                    product_type=str(getattr(device, "product_type", "") or ""),
                    product_number=_optional_int(getattr(device, "product_num", None)),
                    serial_number=_optional_int(getattr(device, "serial_num", None)),
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
                    counter_outputs=_channel_names(
                        getattr(device, "co_physical_chans", tuple()),
                    ),
                    counter_inputs=_channel_names(
                        getattr(device, "ci_physical_chans", tuple()),
                    ),
                    bus_type=str(getattr(device, "bus_type", "") or ""),
                    pxi_chassis_number=_optional_int(
                        getattr(device, "pxi_chassis_num", None)
                    ),
                    terminals=tuple(
                        str(terminal)
                        for terminal in (getattr(device, "terminals", tuple()) or tuple())
                    ),
                    analog_output_sample_clock_supported=_optional_bool(
                        getattr(device, "ao_samp_clk_supported", None)
                    ),
                    digital_trigger_supported=_optional_bool(
                        getattr(device, "dig_trig_supported", None)
                    ),
                    analog_input_max_single_channel_rate=_optional_float(
                        getattr(device, "ai_max_single_chan_rate", None)
                    ),
                    analog_input_max_multi_channel_rate=_optional_float(
                        getattr(device, "ai_max_multi_chan_rate", None)
                    ),
                    analog_output_max_rate=_optional_float(
                        getattr(device, "ao_max_rate", None)
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


def _last_stdout_line(stdout: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _optional_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)

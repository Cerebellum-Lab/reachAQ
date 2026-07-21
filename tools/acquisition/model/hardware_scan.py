from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path
from typing import Optional


@dataclasses.dataclass(frozen=True)
class HardwareScanEntry:
    """One row of the most recent hardware refresh result."""

    info: str
    state: str = "idle"


_PEAK_PCI_VENDOR_ID = "0x001c"


def scan_can_adapters(
    *,
    selected_interface: Optional[str] = None,
    selected_backend: Optional[str] = None,
    net_root: Path = Path("/sys/class/net"),
    pci_root: Path = Path("/sys/bus/pci/devices"),
) -> HardwareScanEntry:
    """Report physical CAN adapters without opening or disturbing the bus."""
    interfaces = tuple(
        sorted(
            path
            for path in net_root.glob("can*")
            if path.name[3:].isdigit()
        )
    )
    peak_devices = tuple(
        path
        for path in pci_root.glob("*")
        if _read_text(path / "vendor").lower() == _PEAK_PCI_VENDOR_ID
    )

    interface_details = tuple(
        _can_interface_detail(
            path,
            selected=path.name == selected_interface,
        )
        for path in interfaces
    )
    drivers = tuple(
        dict.fromkeys(
            driver
            for driver in (
                _driver_name(path / "device" / "driver")
                for path in interfaces
            )
            if driver
        )
    )
    if not drivers:
        drivers = tuple(
            dict.fromkeys(
                driver
                for driver in (_driver_name(path / "driver") for path in peak_devices)
                if driver
            )
        )

    if interfaces:
        adapter = "PEAK PCIe adapter" if peak_devices or any("peak" in item for item in drivers) else "CAN adapter"
        driver_text = ", ".join(drivers) if drivers else "unknown driver"
        application_selection = _can_application_selection(selected_backend, selected_interface)
        return HardwareScanEntry(
            f"✓ {adapter} · {driver_text}\n"
            + "\n".join(interface_details)
            + (f"\n{application_selection}" if application_selection else ""),
            "ok",
        )
    if peak_devices:
        driver_text = ", ".join(drivers) if drivers else "no bound driver"
        locations = ", ".join(path.name for path in peak_devices)
        return HardwareScanEntry(
            f"! PEAK PCIe · {locations} · {driver_text}\n→ no CAN interface",
            "warning",
        )
    return HardwareScanEntry("! No physical CAN adapter", "warning")


def scan_gpus() -> HardwareScanEntry:
    """Return compact NVIDIA GPU identity without importing CUDA frameworks."""
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ),
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except FileNotFoundError:
        return HardwareScanEntry("! NVIDIA driver utility not found", "warning")
    except subprocess.TimeoutExpired:
        return HardwareScanEntry("! NVIDIA GPU scan timed out", "error")
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "driver query failed"
        return HardwareScanEntry(f"! NVIDIA GPU unavailable\n→ {detail}", "error")
    rows = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    if not rows:
        return HardwareScanEntry("! No NVIDIA GPU found", "warning")
    details = []
    for row in rows:
        parts = tuple(part.strip() for part in row.split(",", 3))
        if len(parts) == 4:
            index, name, driver, memory_mib = parts
            details.append(f"→ GPU{index} {name} · {memory_mib} MiB · drv {driver}")
        else:
            details.append(f"→ {row}")
    return HardwareScanEntry(f"✓ {len(rows)} NVIDIA GPU" + ("s" if len(rows) != 1 else "") + "\n" + "\n".join(details), "ok")


def _can_interface_detail(
    path: Path,
    *,
    selected: bool = False,
) -> str:
    flags_text = _read_text(path / "flags")
    try:
        is_up = bool(int(flags_text, 16) & 0x1)
    except ValueError:
        is_up = _read_text(path / "operstate").lower() == "up"
    selection = " · selected" if selected else ""
    return f"→ {path.name} {'↑' if is_up else '↓'}{selection}"


def _can_application_selection(
    selected_backend: Optional[str],
    selected_interface: Optional[str],
) -> str:
    if not selected_backend and not selected_interface:
        return ""
    selection = selected_backend or "unspecified backend"
    if selected_interface:
        selection += f" → {selected_interface}"
    return f"  ↳ app: {selection}"


def _driver_name(path: Path) -> str:
    try:
        return path.resolve(strict=True).name
    except OSError:
        return ""


def _read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""

"""Collect the rig evidence a pellet firmware release needs before qualification.

A firmware entry in config/pellet-firmware-compatibility.yaml is only
qualified once the running board has been shown, on a real rig, to do what the
entry claims. Until now that evidence was gathered by hand, differently each
time, and mostly not kept. This runs the same checks for every release and
writes them to one JSON record.

Every claim is checked from two sides where the rig allows it: what the board
says over CAN, and what the NI-DAQ sees on the wire. A board that acknowledges
a pulse it never produced, or produces one it never reports, fails either way.

  * identity: version, wire schema, capabilities and boot id, judged by the
    same compatibility policy the application applies at startup;
  * mapping: which NI terminal each board stimulus output drives, measured by
    holding each output high, rather than trusted from configuration;
  * tones: each mapped frequency raises only its confirmation line, and an
    unmapped one raises none;
  * pulses: a 500 ms finite pulse on STIM2 and STIM3 rises, returns low
    unaided, and lasts as long as asked;
  * refusals: a pulse aimed at a tone confirmation line is refused by the
    board itself with -EPERM, and nothing moves;
  * drops: a run of short pulses under the board's normal status traffic,
    each needing its acknowledgement, both status frames, and one NI edge
    counted in hardware;
  * timing labels: no event claims board time unless the board reports the
    capability that supplies it;
  * reconnect and reboot: the board is found again each time, and a reboot
    changes the boot id and nothing else.

It drives STIM0-STIM3 and plays tones. STIM2 and STIM3 feed the laser trigger
path, so run it only with lasers off or shutters closed. Nothing here moves a
motor or opens a shutter.

    python qualify_pellet_firmware.py --expect-version 2.3.0
    python qualify_pellet_firmware.py --expect-version 2.3.0 --skip-reboot
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    "auto-trainer-core/src",
    "auto-trainer-device/src",
    "auto-trainer-model/src",
    "auto-trainer-pyside/src",
    "auto-trainer-behavior/src",
    "auto-trainer-inference/src",
    "auto-trainer-video/src",
):
    sys.path.insert(0, str(_REPO_ROOT / _path))
sys.path.insert(0, str(_REPO_ROOT))

from autotrainer.core.configuration import SystemConfiguration  # noqa: E402
from tools.acquisition.model.firmware_compatibility import (  # noqa: E402
    CAPABILITY_BITS,
    FirmwareCompatibilityPolicy,
)
from tools.acquisition.model.nidaq_wiring_verification import wiring_points  # noqa: E402
from tools.hardware.verify_nidaq_wiring import (  # noqa: E402
    DIGITAL_PORTS,
    changed_lines,
    check_digital,
    matches,
)

DEFAULT_CONFIG = Path.home() / "Autotrainer" / "system_configuration.yaml"
DEFAULT_RECORD_DIR = Path.home() / "Autotrainer" / "pellet_firmware_qualification"

EPERM = 1
#: jerrycan_gpio_pulse_phase_t in reachAQ-hardware's jerrycan_types.h.
PHASE_REJECTED, PHASE_ASSERTED, PHASE_COMPLETED = 0, 1, 2
#: Generic-GPIO indices on the board. 4 and 5 are STIM0 and STIM1, owned by
#: the tone generator; 6 and 7 are STIM2 and STIM3, the pulseable lines.
TONE_LINE_GPIO = {"STIM0": 4, "STIM1": 5}
PULSE_GPIO = {"STIM2": 6, "STIM3": 7}
LONG_PULSE_US = 500_000
#: Allowed error on a long pulse's measured width: the polling resolution is
#: well under a millisecond, so this is a firmware tolerance, not slack for
#: the measurement.
LONG_PULSE_TOLERANCE_S = 0.002
DROP_PULSES = 50
DROP_PULSE_US = 1_000
DROP_SPACING_S = 0.02
#: Minimum pulse width the edge counter accepts. The smallest setting an
#: M-series counter input offers; see check_drops for why it is needed.
EDGE_FILTER_S = 125e-9
TONE_MS = 400
#: Neither 5 kHz nor 6 kHz, so the board must raise no confirmation for it.
UNMAPPED_TONE_HZ = 8_000
RECONNECTS = 3
REBOOT_WAIT_S = 20.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect qualification evidence for the running pellet firmware.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expect-version", default=None, metavar="X.Y.Z",
                        help="fail unless the board reports exactly this version")
    parser.add_argument("--record", type=Path, default=None,
                        help="where to write the JSON evidence record (default: "
                             f"{DEFAULT_RECORD_DIR}/pellet-<version>-<time>.json)")
    parser.add_argument("--pulses", type=int, default=DROP_PULSES,
                        help=f"short pulses in the drop check (default {DROP_PULSES})")
    parser.add_argument("--skip-reboot", action="store_true",
                        help="do not reboot the board")
    return parser.parse_args()


class Evidence:
    """The checks run, in order, each with what it measured."""

    def __init__(self):
        self.checks = []

    def add(self, name, passed, detail, **measured):
        self.checks.append({"name": name, "passed": bool(passed),
                            "detail": detail, "measured": measured})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    @property
    def passed(self):
        return bool(self.checks) and all(check["passed"] for check in self.checks)

    def run(self, name, check, *args, default=None):
        """Run one check; if it raises, that is its result, and the run goes on.

        A crash in one check used to end the run with nothing written, which
        threw away every result already measured.
        """
        try:
            return check(*args)
        except Exception as error:
            self.add(name, False, f"raised {type(error).__name__}: {error}")
            return default


def can_health(channel="can0"):
    """Controller state and error counters, as the kernel reports them."""
    try:
        text = subprocess.run(["ip", "-details", "-statistics", "link", "show", channel],
                              capture_output=True, text=True, timeout=5).stdout
    except Exception as error:
        return {"error": str(error)}
    health = {}
    state = re.search(r"state (\S+) \(berr-counter tx (\d+) rx (\d+)\)", text)
    if state:
        health.update(state=state.group(1), tx_errors=int(state.group(2)),
                      rx_errors=int(state.group(3)))
    counts = re.search(r"re-started\s+bus-errors\s+arbit-lost\s+error-warn\s+error-pass\s+"
                       r"bus-off\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", text)
    if counts:
        health.update(zip(("restarts", "bus_errors", "arbitration_lost", "error_warning",
                           "error_passive", "bus_off"), map(int, counts.groups())))
    return health


# --- board side --------------------------------------------------------------

def open_interface():
    from autotrainer.device import CanInterface, CanTransportConfiguration, Target

    interface = CanInterface(
        required_targets=(Target.PELLET_DEVICE,),
        can_transport=CanTransportConfiguration(
            kind="socketcan", channel="can0", bitrate=1_000_000,
            data_bitrate=5_000_000, fd=True,
        ),
    )
    if interface.open() and interface.are_addresses_valid():
        return interface
    interface.close()
    return None


def identify(interface):
    """What the board says it is. Capabilities are None when it does not answer."""
    from autotrainer.device import Target
    from autotrainer.device.device_interface import BoardCapabilities, Version

    interface.request_version()
    reply = interface.get_response(Version, Target.PELLET_DEVICE, motor=None, timeout=2.0)
    match = re.search(r"(\d+\.\d+\.\d+)", getattr(reply, "version", "") or "")
    capabilities = None
    if interface.request_capabilities():
        capabilities = interface.get_response(
            BoardCapabilities, Target.PELLET_DEVICE, motor=None, timeout=1.0)
    return {
        "version": match.group(1) if match else None,
        "wire_schema_version": getattr(capabilities, "wire_schema_version", None),
        "capabilities": getattr(capabilities, "capabilities", None),
        "boot_id": getattr(capabilities, "boot_id", None),
    }


def capability_names(bits):
    return sorted(name for name, bit in CAPABILITY_BITS.items() if int(bits or 0) & bit)


def send_raw(interface, method, *args):
    """Send through the transport with a uuid this tool knows.

    CanInterface's own senders keep the uuid to themselves, so their
    acknowledgements cannot be tied to the request. They also refuse a pulse
    on a tone line before it leaves the host, which would hide the board's
    own refusal - the thing being qualified. Hence the transport directly.
    """
    from autotrainer.device import CanInterface

    uuid = CanInterface.next_uuid()
    rc = getattr(interface._jc, method)(interface.pellet_address, *args, uuid)
    return uuid, rc


def collect(interface, seconds, done=None):
    """Everything the board sends for up to `seconds`, or until `done(messages)`."""
    messages = []
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        batch = interface.read(15, collect_ms=5)
        if batch:
            messages.extend(batch)
            if done is not None and done(messages):
                break
    return messages


def acks_for(messages, uuid):
    from autotrainer.device.device_interface import Acknowledge

    return [m for m in messages if isinstance(m, Acknowledge) and int(m.uuid) == int(uuid)]


def pulse_statuses(messages):
    from autotrainer.device.device_interface import DigitalPulseStatus

    return [m for m in messages if isinstance(m, DigitalPulseStatus)]


def phase_of(status):
    try:
        return int(status.phase)
    except (TypeError, ValueError):
        return {"rejected": PHASE_REJECTED, "asserted": PHASE_ASSERTED,
                "completed": PHASE_COMPLETED}.get(str(status.phase).lower())


# --- NI side -----------------------------------------------------------------

def terminal_bit(terminal):
    """(device, port, bit) for `/Dev/PFIn` or `Dev/port0/lineN`."""
    parts = terminal.strip("/").split("/")
    device = parts[0]
    pfi = re.fullmatch(r"PFI(\d+)", parts[-1], re.IGNORECASE)
    if pfi:
        number = int(pfi.group(1))
        return device, "port1" if number < 8 else "port2", number % 8
    line = re.fullmatch(r"line(\d+)", parts[-1], re.IGNORECASE)
    if len(parts) == 3 and line:
        return device, parts[1], int(line.group(1))
    raise ValueError(f"not a static digital terminal: {terminal}")


class PortWatcher:
    """Static reads of every digital port on one device."""

    def __init__(self, nidaqmx, device):
        self.device = device
        self.tasks = {}
        for port in DIGITAL_PORTS:
            try:
                task = nidaqmx.Task(f"qualify_{device}_{port}")
                task.di_channels.add_di_chan(f"{device}/{port}")
                task.read()
                self.tasks[port] = task
            except Exception:
                continue

    def snapshot(self):
        return {port: int(task.read()) for port, task in self.tasks.items()}

    def close(self):
        for task in self.tasks.values():
            try:
                task.close()
            except Exception:
                pass
        self.tasks = {}


def watch(watcher, seconds, before):
    """Every line that rose from `before` at any point in the next `seconds`.

    `before` has to be read ahead of the command that might raise a line: a
    snapshot taken after sending can already show it high, and a line that is
    high in both reads never registers as rising.
    """
    seen = set()
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        seen.update(changed_lines(watcher.device, before, watcher.snapshot()))
    return sorted(seen)


# --- checks ------------------------------------------------------------------

def check_identity(evidence, identity, policy, expected):
    version = identity["version"]
    evidence.add("version reported", version is not None,
                 f"board reports {version or 'nothing'}", **identity)
    if expected is not None:
        evidence.add("version expected", version == expected,
                     f"expected {expected}, board reports {version}")
    result = policy.evaluate(version or "",
                             wire_schema_version=identity["wire_schema_version"],
                             capabilities=identity["capabilities"] or 0)
    evidence.add(
        "compatibility policy", result.commands_allowed,
        (f"{result.reason}; reports {capability_names(identity['capabilities']) or 'no capabilities'}, "
         f"requires {list(result.required_capabilities) or 'none'}"),
        supported=result.supported, missing=list(result.missing_capabilities),
        qualification_status=result.qualification_status)


def map_stim_terminals(evidence, nidaqmx, interface, configuration):
    """Which NI terminal each pulseable board output drives, measured now."""
    observed = {}
    check_digital(nidaqmx, interface, configuration, wiring_points(configuration),
                  lambda label, terminals: observed.__setitem__(label, terminals))
    terminals = {}
    for name in PULSE_GPIO:
        seen = observed.get(f"board {name}", [])
        ok = len(seen) == 1
        evidence.add(f"{name} lands on NI", ok,
                     f"holding board {name} high raised {', '.join(seen) or 'nothing'}",
                     terminals=seen)
        if ok:
            terminals[name] = seen[0]
    return terminals


def check_tones(evidence, interface, watcher, configuration):
    ports = configuration.nidaq_ports
    lines = {"tone1": ports.tone1, "tone2": ports.tone2}
    for frequency, expected in ((5_000, "tone1"), (6_000, "tone2"), (UNMAPPED_TONE_HZ, None)):
        before = watcher.snapshot()
        uuid, rc = send_raw(interface, "ToneWrite", 0, frequency, TONE_MS)
        seen = watch(watcher, TONE_MS / 1000.0 + 0.3, before)
        acks = acks_for(collect(interface, 0.2), uuid)
        raised = sorted(name for name, channel in lines.items()
                        if channel and any(matches(channel, t) for t in seen))
        want = [expected] if expected else []
        evidence.add(
            f"tone {frequency} Hz", rc == 0 and raised == want,
            f"raised {', '.join(raised) or 'no tone line'}, expected {expected or 'none'}",
            send_rc=rc, lines_seen=seen, ack_errors=[int(a.error) for a in acks])


def time_pulse(interface, watcher, terminal, gpio_index, duration_us):
    """Send one pulse and time its rise and fall on the NI terminal."""
    _device, port, bit = terminal_bit(terminal)
    task = watcher.tasks[port]
    uuid, rc = send_raw(interface, "GPIOPulse", 0, gpio_index, duration_us)
    sent = time.perf_counter()
    rise = fall = None
    reads = 0
    deadline = sent + duration_us / 1e6 + 1.0
    while time.perf_counter() < deadline:
        high = bool(int(task.read()) & (1 << bit))
        now = time.perf_counter()
        reads += 1
        if high and rise is None:
            rise = now
        elif rise is not None and not high:
            fall = now
            break
    elapsed = time.perf_counter() - sent
    messages = collect(interface, 0.3)
    return {
        "send_rc": rc,
        "rise_after_send_s": None if rise is None else rise - sent,
        "width_s": None if rise is None or fall is None else fall - rise,
        "poll_interval_s": elapsed / max(reads, 1),
        "ack_errors": [int(a.error) for a in acks_for(messages, uuid)],
        "phases": [phase_of(s) for s in pulse_statuses(messages)],
        "messages": messages,
    }


def check_pulses(evidence, interface, watcher, terminals):
    messages = []
    for name, gpio_index in PULSE_GPIO.items():
        if name not in terminals:
            evidence.add(f"{name} 500 ms pulse", False, "no NI terminal to watch it on")
            continue
        result = time_pulse(interface, watcher, terminals[name], gpio_index, LONG_PULSE_US)
        messages.extend(result.pop("messages"))
        width = result["width_s"]
        ok = (result["send_rc"] == 0 and width is not None
              and abs(width - LONG_PULSE_US / 1e6) <= LONG_PULSE_TOLERANCE_S
              and result["ack_errors"] == [0]
              and result["phases"].count(PHASE_ASSERTED) == 1
              and result["phases"].count(PHASE_COMPLETED) == 1)
        detail = ("never rose" if result["rise_after_send_s"] is None
                  else "rose and never returned low" if width is None
                  else f"rose {result['rise_after_send_s'] * 1e3:.2f} ms after send, "
                       f"returned low after {width * 1e3:.2f} ms")
        evidence.add(f"{name} 500 ms pulse", ok,
                     f"{detail}; ack errors {result['ack_errors']}, phases {result['phases']}",
                     **result)
    return messages


def check_refusals(evidence, interface, watcher):
    messages = []
    for name, gpio_index in TONE_LINE_GPIO.items():
        before = watcher.snapshot()
        uuid, rc = send_raw(interface, "GPIOPulse", 0, gpio_index, 100_000)
        seen = watch(watcher, 0.3, before)
        batch = collect(interface, 0.3)
        messages.extend(batch)
        ack_errors = [int(a.error) for a in acks_for(batch, uuid)]
        rejected = [int(s.error) for s in pulse_statuses(batch)
                    if phase_of(s) == PHASE_REJECTED]
        ok = ack_errors == [-EPERM] and rejected == [-EPERM] and not seen
        evidence.add(f"pulse on {name} refused", ok,
                     f"ack errors {ack_errors}, rejected with {rejected}, "
                     f"lines moved: {', '.join(seen) or 'none'}",
                     send_rc=rc, ack_errors=ack_errors, rejected_errors=rejected,
                     lines_seen=seen)
    return messages


def check_drops(evidence, nidaqmx, interface, terminal, count):
    """Short pulses back to back under normal traffic, each fully accounted for."""
    from nidaqmx.constants import CountDirection, Edge

    device = terminal.strip("/").split("/")[0]
    counter = nidaqmx.Task("qualify_edges")
    messages = []
    tally = {"sent_ok": 0, "acked_ok": 0, "asserted": 0, "completed": 0, "other": 0}
    try:
        channel = counter.ci_channels.add_ci_count_edges_chan(
            f"{device}/ctr0", edge=Edge.RISING, initial_count=0,
            count_direction=CountDirection.COUNT_UP)
        channel.ci_count_edges_term = terminal
        # Unfiltered, christielab10's PFI0 counts about ten rising edges per
        # pulse, at any width, and none with nothing driven: glitches under
        # 125 ns at the transitions (measured 2026-09-23). The smallest filter
        # the 6221 offers rejects them and cannot swallow a 1 ms pulse. The
        # width has to be set before the filter is enabled, or DAQmx refuses.
        channel.ci_count_edges_dig_fltr_min_pulse_width = EDGE_FILTER_S
        channel.ci_count_edges_dig_fltr_enable = True
        counter.start()
        for _ in range(count):
            uuid, rc = send_raw(interface, "GPIOPulse", 0, PULSE_GPIO["STIM3"], DROP_PULSE_US)
            tally["sent_ok"] += rc == 0

            def settled(batch, uuid=uuid):
                return (acks_for(batch, uuid)
                        and any(phase_of(s) == PHASE_COMPLETED for s in pulse_statuses(batch)))

            batch = collect(interface, 0.25, done=settled)
            messages.extend(batch)
            tally["acked_ok"] += [int(a.error) for a in acks_for(batch, uuid)] == [0]
            for status in pulse_statuses(batch):
                phase = phase_of(status)
                if phase == PHASE_ASSERTED and int(status.error) == 0:
                    tally["asserted"] += 1
                elif phase == PHASE_COMPLETED and int(status.error) == 0:
                    tally["completed"] += 1
                else:
                    tally["other"] += 1
            time.sleep(DROP_SPACING_S)
        time.sleep(0.05)
        edges = int(counter.read())
    finally:
        counter.close()
    tally["ni_edges"] = edges
    ok = all(tally[k] == count for k in ("sent_ok", "acked_ok", "asserted", "completed",
                                         "ni_edges")) and tally["other"] == 0
    evidence.add(f"{count} short STIM3 pulses", ok,
                 ", ".join(f"{k} {v}" for k, v in tally.items())
                 + f" (NI edges counted through a {EDGE_FILTER_S * 1e9:g} ns input filter)",
                 expected=count, edge_filter_s=EDGE_FILTER_S, **tally)
    return messages


def check_timing_labels(evidence, messages, capabilities):
    """No event may claim board time the board did not say it can supply."""
    trailer = bool(int(capabilities or 0) & CAPABILITY_BITS["timing_trailer"])
    claimed = [m for m in messages if getattr(m, "board_time_us", None) is not None]
    labels = sorted({str(getattr(m, "timing_confidence", "")) for m in messages} - {""})
    ok = trailer or not claimed
    evidence.add(
        "timing labels", ok and bool(messages),
        (f"{len(messages)} board events labelled {', '.join(labels) or 'nothing'}; "
         f"{len(claimed)} carry board time; timing_trailer "
         f"{'reported' if trailer else 'not reported, so no clock uncertainty exists to measure'}"),
        events=len(messages), board_time_events=len(claimed), labels=labels)


def check_reconnects(evidence, first):
    for attempt in range(1, RECONNECTS + 1):
        interface = open_interface()
        if interface is None:
            evidence.add(f"reconnect {attempt}", False, "board not found")
            continue
        try:
            identity = identify(interface)
        finally:
            interface.close()
        same = (identity["version"] == first["version"]
                and identity["capabilities"] == first["capabilities"]
                and identity["boot_id"] == first["boot_id"])
        evidence.add(f"reconnect {attempt}", same,
                     f"version {identity['version']}, boot id {identity['boot_id']}",
                     **identity)


def check_reboot(evidence, first):
    from autotrainer.device import Target

    interface = open_interface()
    if interface is None:
        evidence.add("reboot", False, "board not found before reboot")
        return
    try:
        sent = interface.board_reboot(Target.PELLET_DEVICE)
    finally:
        interface.close()
    started = time.perf_counter()
    time.sleep(1.0)
    identity = None
    while time.perf_counter() - started < REBOOT_WAIT_S:
        interface = open_interface()
        if interface is not None:
            try:
                identity = identify(interface)
            finally:
                interface.close()
            if identity["version"]:
                break
        time.sleep(1.0)
    back = time.perf_counter() - started
    if not sent or identity is None or not identity["version"]:
        evidence.add("reboot", False, f"board did not answer within {REBOOT_WAIT_S:g} s",
                     reboot_sent=sent)
        return
    unchanged = (identity["version"] == first["version"]
                 and identity["capabilities"] == first["capabilities"]
                 and identity["wire_schema_version"] == first["wire_schema_version"])
    if first["boot_id"] is None:
        # Firmware before 2.3.0 reports no boot id, so a reboot cannot be told
        # apart from a board that ignored the command.
        evidence.add("reboot", unchanged,
                     f"answered after {back:.1f} s with {identity['version']}; this "
                     "firmware reports no boot id, so the reboot itself is unproven",
                     seconds_to_answer=back, **identity)
        return
    evidence.add("reboot", unchanged and identity["boot_id"] != first["boot_id"],
                 f"answered after {back:.1f} s; boot id {first['boot_id']} -> "
                 f"{identity['boot_id']}, everything else unchanged: {unchanged}",
                 seconds_to_answer=back, previous_boot_id=first["boot_id"], **identity)


# --- run ---------------------------------------------------------------------

def repository_revision():
    try:
        return subprocess.run(["git", "-C", str(_REPO_ROOT), "describe", "--always",
                               "--dirty", "--tags"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def write_record(args, evidence, identity, started, can_before, can_after):
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    path = args.record or (
        DEFAULT_RECORD_DIR / f"pellet-{identity.get('version') or 'unknown'}-{stamp}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": 1,
        "generated_on": started.isoformat(),
        "host": socket.gethostname(),
        "reachaq_revision": repository_revision(),
        "board": identity,
        "can_before": can_before,
        "can_after": can_after,
        "passed": evidence.passed,
        "checks": evidence.checks,
    }
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    started = datetime.now(timezone.utc)
    configuration = SystemConfiguration.load_yaml_file(args.config, save_backup=False)
    policy = FirmwareCompatibilityPolicy.load()
    evidence = Evidence()
    can_before = can_health()
    print(f"CAN before: {can_before}")

    import nidaqmx

    print("--- identity ---")
    interface = open_interface()
    if interface is None:
        print("CAN did not come up or the pellet board was not found")
        return 2
    messages = []
    empty = {"version": None, "wire_schema_version": None, "capabilities": None,
             "boot_id": None}
    try:
        identity = evidence.run("identify", identify, interface, default=empty)
        check_identity(evidence, identity, policy, args.expect_version)

        print("--- where the board outputs land ---")
        terminals = evidence.run("map board outputs", map_stim_terminals, evidence,
                                 nidaqmx, interface, configuration, default={})

        device = configuration.nidaq_ports.tone1.strip("/").split("/")[0]
        watcher = PortWatcher(nidaqmx, device)
        try:
            print("--- tones ---")
            evidence.run("tones", check_tones, evidence, interface, watcher, configuration)
            print("--- finite pulses ---")
            messages += evidence.run("pulses", check_pulses, evidence, interface, watcher,
                                     terminals, default=[])
            print("--- refusals ---")
            messages += evidence.run("refusals", check_refusals, evidence, interface,
                                     watcher, default=[])
        finally:
            # The counter below takes the STIM3 terminal as its source, so the
            # static tasks on that port are released first.
            watcher.close()

        print("--- drops under normal traffic ---")
        if "STIM3" in terminals:
            messages += evidence.run(f"{args.pulses} short STIM3 pulses", check_drops,
                                     evidence, nidaqmx, interface, terminals["STIM3"],
                                     args.pulses, default=[])
        else:
            evidence.add(f"{args.pulses} short STIM3 pulses", False,
                         "no NI terminal to count them on")
        check_timing_labels(evidence, messages, identity["capabilities"])
    finally:
        interface.close()

    print("--- reconnect ---")
    evidence.run("reconnect", check_reconnects, evidence, identity)
    if args.skip_reboot:
        print("--- reboot skipped ---")
    else:
        print("--- reboot ---")
        evidence.run("reboot", check_reboot, evidence, identity)

    can_after = can_health()
    print(f"CAN after: {can_after}")
    bus_off = can_after.get("bus_off", 0) - can_before.get("bus_off", 0)
    evidence.add("CAN health", bus_off == 0 and can_after.get("state") != "BUS-OFF",
                 f"state {can_before.get('state')} -> {can_after.get('state')}, "
                 f"tx errors {can_before.get('tx_errors')} -> {can_after.get('tx_errors')}, "
                 f"bus-off events during run: {bus_off}")

    path = write_record(args, evidence, identity, started, can_before, can_after)
    failed = [check["name"] for check in evidence.checks if not check["passed"]]
    print()
    print(f"{len(evidence.checks) - len(failed)} of {len(evidence.checks)} checks passed"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    print(f"evidence written to {path}")
    return 0 if evidence.passed else 1


if __name__ == "__main__":
    sys.exit(main())

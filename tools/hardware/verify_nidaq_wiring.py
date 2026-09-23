"""Check the NI-DAQ channel map against the cables, and record the result.

The configuration asserts which pin carries what. Nothing tested those
assertions, and on this rig several were wrong for weeks while the stream
recorded them happily: inputs that are not connected to what they are named
for still answered to a laser being driven, so they produced clean pulses
rather than silence. That is worse than a dead channel, because it looks like
data. Part of that turned out to be this application's own doing - analog
channels were added without a terminal configuration, which is not uniform
across a 6221 - and part is in the cabling.

Method: drive one thing at a time and watch everything readable.

  * each board stimulus output is held high in turn, with every digital line
    on both boards read as a static level. This needs no timing and no edge
    detection, so it names the pin for each output outright;
  * each laser command output is held at a DC level while the configured
    analog channels are read.

It reports what it saw and where that contradicts the configuration. It does
not diagnose why: a mislabelled cable, a split, and coupling between inputs
are indistinguishable from this side, and an earlier version of this tool
asserted a mechanism - multiplexer ghosting - that a later measurement
refuted. A channel answering to another channel's driver is reported as
exactly that, and left for someone with a meter.

Shutters stay closed unless --open-shutters is given, so the default run
confirms command copies, tone lines and stimulus triggers without emitting
anything. The photodiodes can only be confirmed with light on them, which is
why that is opt-in.

    python verify_nidaq_wiring.py                 # safe: no emission
    python verify_nidaq_wiring.py --open-shutters # also tests the photodiodes
    python verify_nidaq_wiring.py --dry-run       # show the points, touch nothing
"""

from __future__ import annotations

import argparse
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
from tools.acquisition.model.nidaq_wiring_report import (  # noqa: E402
    build_report,
    format_report,
)
from tools.acquisition.model.nidaq_wiring_verification import (  # noqa: E402
    CONFIRMED,
    DRIVER,
    OPAQUE,
    SILENT,
    UNEXPECTED,
    UNTESTED,
    WiringCheck,
    WiringVerification,
    wiring_points,
)

DEFAULT_CONFIG = Path.home() / "Autotrainer" / "system_configuration.yaml"
DEFAULT_RECORD = Path.home() / "Autotrainer" / "nidaq_wiring_verification.json"

#: Ports holding every digital line worth watching. port1 and port2 are the
#: PFI pins on an M-series board, which is where a stimulus trigger lands.
DIGITAL_PORTS = ("port0", "port1", "port2")
SETTLE_SECONDS = 0.15
ANALOG_RATE_HZ = 10_000.0
ANALOG_SAMPLES = 500
#: A response has to clear this to count, which keeps noise and the tail of a
#: decaying ghost from registering as a connection.
ANALOG_THRESHOLD_V = 0.25


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify the NI-DAQ channel map against the cables.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--open-shutters", action="store_true",
                        help="open each shutter while its laser is driven, so "
                             "the photodiodes can be confirmed. This emits "
                             "light.")
    parser.add_argument("--command-volts", type=float, default=1.0,
                        help="level each laser command is held at (default 1.0)")
    parser.add_argument("--observe", type=float, default=0.0, metavar="SECONDS",
                        help="after the driven checks, watch the lines "
                             "nothing here can drive - cam_frames, barcode "
                             "- for this long, and confirm any that change. "
                             "Start the camera or trigger a barcode during "
                             "the window.")
    parser.add_argument("--report", type=Path, default=None,
                        help="also write the formatted report here")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the points the configuration asserts and "
                             "stop, without touching the hardware")
    return parser.parse_args()


def digital_snapshot(tasks):
    return {port: int(task.read()) for port, task in tasks.items()}


def changed_lines(device, before, after):
    """Terminals that went high between two static reads."""
    names = []
    for port, before_bits in before.items():
        rising = (after[port] ^ before_bits) & after[port]
        for bit in range(8):
            if not rising & (1 << bit):
                continue
            if port == "port1":
                names.append(f"/{device}/PFI{bit}")
            elif port == "port2":
                names.append(f"/{device}/PFI{bit + 8}")
            else:
                names.append(f"{device}/port0/line{bit}")
    return names


def observe_undriven(nidaqmx, points, seconds, undriven, report):
    """Watch the lines nothing in this run drives, and see if they move.

    cam_frames and barcode are sourced by a running camera and a session,
    neither of which exists here, so the driven phase can only ever report
    them untested. It cannot drive them; it can watch while somebody else
    does, which turns "nothing is known" into an answer an operator can
    produce in one action.

    What this confirms is weaker than the driven checks, and is recorded as
    such: a line that changes while being watched carries something, which is
    not the same as carrying what the configuration says it does. The
    operator asserts the cause by choosing when to start the camera.
    """
    watched = [point for point in points if point.fingerprint in undriven]
    if not watched:
        print("\n" + "nothing to observe: every point had a driver")
        return
    devices = sorted({point.physical_channel.strip("/").split("/", 1)[0]
                      for point in watched})
    tasks = {}
    try:
        for device in devices:
            for port in DIGITAL_PORTS:
                try:
                    task = nidaqmx.Task(f"observe_{device}_{port}")
                    task.di_channels.add_di_chan(f"{device}/{port}")
                    task.read()
                    tasks[(device, port)] = task
                except Exception:
                    continue
        if not tasks:
            print("\n" + "no digital port could be opened to observe")
            return

        print("\n" + f"--- watching {len(watched)} undriven line(s) "
              f"for {seconds:g}s: {', '.join(p.name for p in watched)} ---")
        print("    start the camera or trigger a barcode now")
        baseline = {key: int(task.read()) for key, task in tasks.items()}
        seen = set()
        deadline = time.time() + seconds
        while time.time() < deadline:
            for (device, port), task in tasks.items():
                try:
                    word = int(task.read())
                except Exception:
                    continue
                for name in changed_lines(
                        device, {port: baseline[(device, port)]}, {port: word}):
                    seen.add(name)
        print(f"    changed: {', '.join(sorted(seen)) if seen else 'nothing'}")
        for point in watched:
            if any(matches(point.physical_channel, name) for name in seen):
                report(point, f"changed while watched for {seconds:g}s; nothing "
                              "in this run drove it, so what caused it is the "
                              "operator's assertion")
    finally:
        for task in tasks.values():
            try:
                task.close()
            except Exception:
                pass


def matches(point_channel, observed):
    """Whether an observed terminal is the one a config point names.

    The two are spelled differently in places - a configuration says
    `PXI1Slot5/port0/line0` and a terminal is `/PXI1Slot5/PFI0` - so compare
    on the normalised tail rather than the exact string.
    """
    return point_channel.strip("/").lower() == observed.strip("/").lower()


def check_digital(nidaqmx, interface, configuration, points, report):
    """Hold each board output high and see which pin follows it."""
    from autotrainer.device import DigitalOutputs
    from autotrainer.device.device_interface import BOARD_STIM_LINE_OUTPUTS

    label_of = {output: f"board STIM{line}"
                for line, output in BOARD_STIM_LINE_OUTPUTS.items()}
    devices = sorted({point.physical_channel.strip("/").split("/", 1)[0]
                      for point in points
                      if "/" in point.physical_channel.strip("/")})

    tasks = {}
    try:
        for device in devices:
            for port in DIGITAL_PORTS:
                try:
                    task = nidaqmx.Task(f"verify_{device}_{port}")
                    task.di_channels.add_di_chan(f"{device}/{port}")
                    task.read()
                    tasks[(device, port)] = task
                except Exception:
                    # A board without this port simply has nothing to watch.
                    continue

        for output in DigitalOutputs:
            label = label_of.get(output, output.name)
            per_device_before = {
                device: {port: int(task.read())
                         for (dev, port), task in tasks.items() if dev == device}
                for device in devices
            }
            interface.set_digital_output(output, True)
            time.sleep(SETTLE_SECONDS)
            per_device_after = {
                device: {port: int(task.read())
                         for (dev, port), task in tasks.items() if dev == device}
                for device in devices
            }
            interface.set_digital_output(output, False)
            time.sleep(SETTLE_SECONDS)

            observed = []
            for device in devices:
                observed.extend(changed_lines(
                    device, per_device_before[device], per_device_after[device]))
            report(label, observed)
    finally:
        for task in tasks.values():
            try:
                task.close()
            except Exception:
                pass


def read_analog(nidaqmx, channels, terminal_config):
    """One short buffered read of these channels, referenced as the stream is.

    Referencing matters more than it looks: left to DAQmx the mode is not
    uniform across a 6221's channel range, and the run would then measure a
    different circuit from the one the application records.
    """
    from nidaqmx.constants import AcquisitionType

    task = nidaqmx.Task("verify_analog")
    try:
        for channel in channels:
            if terminal_config is None:
                task.ai_channels.add_ai_voltage_chan(channel)
            else:
                task.ai_channels.add_ai_voltage_chan(
                    channel, terminal_config=terminal_config)
        task.timing.cfg_samp_clk_timing(
            ANALOG_RATE_HZ, sample_mode=AcquisitionType.FINITE,
            samps_per_chan=ANALOG_SAMPLES)
        raw = task.read(number_of_samples_per_channel=ANALOG_SAMPLES, timeout=5.0)
        rows = raw if isinstance(raw[0], list) else [raw]
        return {channel: sum(row) / len(row) for channel, row in zip(channels, rows)}
    finally:
        try:
            task.close()
        except Exception:
            pass


def inspect_configuration(configuration):
    """What the boards say about this map, before anything is driven.

    Discovery needs the hardware, and this tool is useful without it, so a
    failure to discover is reported as a note rather than swallowed: the
    configuration could not be checked, which is not the same as it passing.
    """
    try:
        from tools.acquisition.model.nidaq_discovery import discover_nidaq_devices
        from tools.acquisition.model.nidaq_validation import (
            chassis_identification_note,
            validate_nidaq_configuration,
        )
    except Exception as error:
        return (), (f"configuration could not be checked: {error}",)

    devices, error = discover_nidaq_devices()
    if error or not devices:
        detail = f": {error}" if error else ""
        return (), (f"no boards to check the configuration against{detail}",)

    issues = validate_nidaq_configuration(
        devices,
        stream=configuration.nidaq_stream,
        ports=configuration.nidaq_ports,
        laser=configuration.laser,
    )
    notes = tuple(n for n in (chassis_identification_note(devices),) if n)
    return tuple(issue.describe() for issue in issues), notes


def emit_report(configuration, verification, issues, notes, path) -> None:
    report = build_report(configuration, verification, issues=issues,
                          notes=notes)
    text = format_report(report)
    print()
    print(text)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + chr(10), encoding="utf-8")
        print()
        print(f"report written to {path}")


def main() -> int:
    args = parse_args()
    configuration = SystemConfiguration.load_yaml_file(args.config,
                                                       save_backup=False)
    points = wiring_points(configuration)
    issues, notes = inspect_configuration(configuration)

    print(f"{len(points)} points asserted by {args.config}")
    for point in points:
        note = f"  ({point.opaque_reason})" if point.kind == OPAQUE else ""
        print(f"  {point.name:<26} {point.physical_channel:<26} "
              f"{point.role}{note}")
    if args.dry_run:
        # Everything that can be said without touching the rig: where
        # each point is on the block, anything the boards already disagree
        # with, and what the last run found. Reading the existing record
        # matters - a dry run that ignored it would report a confirmed rig
        # as entirely unknown.
        emit_report(configuration, WiringVerification.load(args.record),
                    issues, notes, args.report)
        return 0

    import nidaqmx
    from autotrainer.device import (
        CanInterface,
        CanTransportConfiguration,
        LaserChannelId,
        Target,
    )
    from tools.acquisition.model.laser_model import LaserModel

    analog_points = [p for p in points
                     if p.kind != OPAQUE and "/ai" in p.physical_channel]
    observed_by_point = {}
    # Which drivers actually ran, so a point that stayed silent can be told
    # apart from one nothing asked a question of.
    lasers_driven = set()
    board_outputs_exercised = False

    def record_digital(label, observed):
        print(f"\n{label} drives: "
              + (", ".join(observed) if observed else "nothing"))
        for point in points:
            if point.kind == OPAQUE:
                continue
            if any(matches(point.physical_channel, terminal)
                   for terminal in observed):
                observed_by_point[point.fingerprint] = (
                    CONFIRMED, f"held high by {label}")

    interface = CanInterface(
        required_targets=(Target.PELLET_DEVICE,),
        can_transport=CanTransportConfiguration(
            kind="socketcan", channel="can0", bitrate=1_000_000,
            data_bitrate=5_000_000, fd=True,
        ),
    )
    laser = LaserModel()
    try:
        if not interface.open() or not interface.are_addresses_valid():
            print("CAN did not come up; board outputs cannot be checked")
        else:
            print("\n--- board outputs, held high one at a time ---")
            check_digital(nidaqmx, interface, configuration, points,
                          record_digital)
            board_outputs_exercised = True

        if analog_points and configuration.laser.backend == "nidaq":
            from nidaqmx.constants import TerminalConfiguration
            mode_name = getattr(
                configuration.nidaq_stream, "analog_terminal_config", "") or ""
            terminal_config = getattr(
                TerminalConfiguration, mode_name.upper(), None)
            print(f"\n--- laser commands, held at {args.command_volts:g} V, "
                  f"analog referencing {mode_name or 'as DAQmx chooses'} ---")
            laser.load_configuration(configuration.laser,
                                     feedback_reader=lambda _c: 0.0)
            order = [p.physical_channel for p in analog_points]
            for channel in configuration.laser.channels:
                number = channel.channel_id.value
                if args.open_shutters:
                    laser.set_shutter_open(LaserChannelId(number), True)
                laser.set_command_voltage(LaserChannelId(number),
                                          args.command_volts)
                time.sleep(SETTLE_SECONDS)
                levels = read_analog(nidaqmx, order, terminal_config)
                laser.set_command_voltage(
                    LaserChannelId(number), channel.minimum_command_volts)
                if args.open_shutters:
                    laser.set_shutter_open(LaserChannelId(number), False)

                responded = sorted(c for c, v in levels.items()
                                   if abs(v) >= ANALOG_THRESHOLD_V)
                print(f"\nlaser {number} at {args.command_volts:g} V moved: "
                      + (", ".join(f"{c.split('/')[-1]}={levels[c]:.3f}V"
                                   for c in responded) or "nothing"))
                # A point belongs to this laser when the configuration says so
                # by name. Anything else that moved is a contradiction, and
                # the run reports it without deciding what causes it.
                prefix = f"laser{number}_"
                for point in analog_points:
                    if point.physical_channel not in responded:
                        continue
                    if point.name.startswith(prefix):
                        observed_by_point[point.fingerprint] = (
                            CONFIRMED,
                            f"responded to laser {number} at "
                            f"{levels[point.physical_channel]:.3f} V")
                    else:
                        observed_by_point.setdefault(point.fingerprint, (
                            UNEXPECTED,
                            f"responded to laser {number}, which it is not "
                            f"assigned to, at "
                            f"{levels[point.physical_channel]:.3f} V"))
                lasers_driven.add(number)
    finally:
        try:
            laser.close()
        except Exception:
            pass
        try:
            interface.close()
        except Exception:
            pass

    def driver_exercised(point) -> bool:
        """Whether this run drove anything that should have moved this point.

        The distinction decides between "did not respond", which is evidence
        of a missing cable, and "nothing asked it to", which is no evidence at
        all. cam_frames and barcode fall in the second group: their sources
        are a running camera and a session, neither of which exists here.
        """
        for number in lasers_driven:
            if point.name.startswith(f"laser{number}_"):
                return True
        if not board_outputs_exercised:
            return False
        return point.name in {"tone1", "tone2"} or point.name.endswith(
            "_trigger_in")

    if args.observe > 0:
        undriven = {
            point.fingerprint for point in points
            if point.kind != OPAQUE
            and point.fingerprint not in observed_by_point
            and not driver_exercised(point)
        }

        def record_observed(point, detail):
            observed_by_point[point.fingerprint] = (CONFIRMED, detail)

        try:
            observe_undriven(nidaqmx, points, args.observe, undriven,
                             record_observed)
        except Exception as error:
            print(f"observation failed: {error}")

    checks = []
    # Observable points first: a driver is confirmed by what it moved, so its
    # witnesses have to exist before it is judged.
    for point in sorted(points, key=lambda p: p.kind == DRIVER):
        if point.kind == OPAQUE:
            checks.append(WiringCheck.for_point(
                point, UNTESTED, "not observable", point.opaque_reason))
            continue
        method = ("held DC on its laser command"
                  if "/ai" in point.physical_channel
                  else "static level while each board output was held high")
        if point.fingerprint in observed_by_point and not driver_exercised(point):
            method = "watched while the operator drove it"

        if point.fingerprint in observed_by_point:
            status, detail = observed_by_point[point.fingerprint]
        elif point.kind == DRIVER:
            # An output cannot answer to itself. It is confirmed by what it
            # made move: if this laser's command copy responded, the command
            # output reached it.
            witnesses = [c for c in checks
                         if c.status == CONFIRMED
                         and c.name.startswith(point.name + "_")]
            if witnesses:
                status = CONFIRMED
                method = "confirmed through what it drove"
                detail = "drove " + ", ".join(c.name for c in witnesses)
            else:
                status = SILENT
                method = "confirmed through what it drove"
                detail = "driving it moved nothing it is wired to"
        elif driver_exercised(point):
            status = SILENT
            detail = "its driver was exercised and it did not respond"
        else:
            status = UNTESTED
            method = "no driver available"
            detail = "nothing in this run drives it"
        checks.append(WiringCheck.for_point(point, status, method, detail))

    verification = WiringVerification(
        checks=tuple(checks),
        generated_on=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    verification.save(args.record)

    print()
    print(f"--- result, written to {args.record} ---")
    emit_report(configuration, verification, issues, notes, args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

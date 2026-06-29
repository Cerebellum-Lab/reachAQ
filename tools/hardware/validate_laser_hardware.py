from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

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

from autotrainer.core.configuration import SystemConfiguration
from autotrainer.device import LaserCalibrationRamp, LaserChannelId, LaserPulseTrain
from tools.acquisition.model.laser_model import LaserModel


def main() -> int:
    args = _parse_args()
    configuration = SystemConfiguration.load_yaml_file(args.config)
    laser_config = configuration.laser
    if laser_config.backend != "nidaq":
        raise RuntimeError(f"laser backend must be 'nidaq' for hardware validation, got {laser_config.backend!r}")

    model = LaserModel()
    try:
        model.load_configuration(laser_config)
        channel_id = LaserChannelId(args.channel)
        channel = laser_config.get_channel(channel_id)
        _print_laser_summary(laser_config, channel_id)

        if args.action == "connect":
            return 0
        if args.action == "feedback":
            sample = model.read_feedback_sample(channel_id)
            print(
                f"feedback channel={sample.channel_id.value} command={sample.command_volts:.6f} V "
                f"diode={sample.diode_volts:.6f} V command_copy={sample.command_copy_volts}"
            )
            return 0
        if args.action == "shutter":
            print(f"opening shutter for laser {channel_id.value}")
            try:
                model.set_shutter_open(channel_id, True)
                time.sleep(args.hold_seconds)
            finally:
                print(f"closing shutter for laser {channel_id.value}")
                model.set_shutter_open(channel_id, False)
            return 0
        if args.action == "manual-ao":
            try:
                applied = model.set_command_voltage(channel_id, args.volts)
                print(f"applied command voltage {applied:.6f} V on laser {channel_id.value}")
                if args.hold_seconds > 0:
                    time.sleep(args.hold_seconds)
                sample = model.read_feedback_sample(channel_id)
                print(
                    f"feedback after manual AO: diode={sample.diode_volts:.6f} V "
                    f"command_copy={sample.command_copy_volts}"
                )
            finally:
                model.set_command_voltage(channel_id, channel.minimum_command_volts)
            return 0
        if args.action == "pulse":
            pulse = LaserPulseTrain(
                channel_id=channel_id,
                amplitude_volts=args.amplitude,
                duration_ms=args.duration_ms,
                baseline_ms=args.baseline_ms,
                post_stim_ms=args.post_stim_ms,
                pulse_count=args.pulse_count,
                frequency_hz=args.frequency_hz if args.pulse_count > 1 else None,
                trigger_source=_optional(args.trigger_source) or channel.trigger_source,
                trigger_edge=args.trigger_edge,
                enable_pmt_shutter=args.pmt,
                emit_trigger_output=args.trigger_do,
                emit_timing_trigger_output=args.timing_trigger_do,
            )
            model.run_pulse_train(pulse)
            print(f"pulse train completed on laser {channel_id.value}")
            return 0
        if args.action == "ramp":
            ramp = LaserCalibrationRamp(
                channel_id=channel_id,
                start_volts=args.ramp_start,
                stop_volts=args.ramp_stop,
                steps=args.ramp_steps,
                samples_per_step=args.samples_per_step,
                enable_pmt_shutter=args.pmt,
            )
            points = model.run_calibration_ramp(ramp)
            curve = model.make_diode_power_curve(points)
            print(f"calibration ramp completed on laser {channel_id.value}: {len(points)} points")
            for point in points:
                print(
                    f"  command={point.command_volts:.6f} V diode={point.diode_volts:.6f} V "
                    f"command_copy={point.command_copy_volts}"
                )
            print(
                f"diode curve range: {curve.points[0].diode_volts:.6f}.."
                f"{curve.points[-1].diode_volts:.6f} V"
            )
            return 0
        raise RuntimeError(f"Unhandled action: {args.action}")
    finally:
        model.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate reachAQ NI-DAQ laser hardware one path at a time.")
    parser.add_argument("--config", type=Path, required=True, help="SystemConfiguration YAML file.")
    parser.add_argument("--channel", type=int, default=1, choices=(1, 2, 3, 4), help="Laser channel to validate.")
    parser.add_argument(
        "--action",
        choices=("connect", "feedback", "shutter", "manual-ao", "pulse", "ramp"),
        default="connect",
        help="Hardware action to run.",
    )
    parser.add_argument("--hold-seconds", type=float, default=0.25, help="Hold time for shutter/manual AO actions.")
    parser.add_argument("--volts", type=float, default=0.0, help="Manual AO voltage for manual-ao.")
    parser.add_argument("--amplitude", type=float, default=1.0, help="Pulse amplitude in volts.")
    parser.add_argument("--duration-ms", type=float, default=10.0, help="Pulse high duration.")
    parser.add_argument("--baseline-ms", type=float, default=0.0, help="Pulse baseline before stimulation.")
    parser.add_argument("--post-stim-ms", type=float, default=0.0, help="Pulse baseline after stimulation.")
    parser.add_argument("--pulse-count", type=int, default=1, help="Number of pulses in the train.")
    parser.add_argument("--frequency-hz", type=float, default=10.0, help="Pulse frequency when pulse-count > 1.")
    parser.add_argument("--trigger-source", default=None, help="Override configured NI-DAQ start-trigger route.")
    parser.add_argument("--trigger-edge", choices=("rising", "falling"), default="rising")
    parser.add_argument("--pmt", action="store_true", help="Enable configured PMT shutter output during action.")
    parser.add_argument("--trigger-do", action="store_true", help="Emit the configured per-laser trigger DO.")
    parser.add_argument("--timing-trigger-do", action="store_true", help="Emit configured per-laser timing trigger DO.")
    parser.add_argument("--ramp-start", type=float, default=0.0, help="Calibration ramp start voltage.")
    parser.add_argument("--ramp-stop", type=float, default=1.0, help="Calibration ramp stop voltage.")
    parser.add_argument("--ramp-steps", type=int, default=11, help="Calibration ramp point count.")
    parser.add_argument("--samples-per-step", type=int, default=100, help="Samples acquired for each ramp point.")
    return parser.parse_args()


def _print_laser_summary(laser_config, channel_id: LaserChannelId) -> None:
    channel = laser_config.get_channel(channel_id)
    print(
        f"laser backend={laser_config.backend} hardware_timed={laser_config.hardware_timed} "
        f"sample_rate={laser_config.sample_rate_hz}"
    )
    print(
        f"channel {channel_id.value}: AO={channel.analog_output} diode={channel.diode_input} "
        f"command_copy={channel.command_copy_input} shutter={channel.shutter_output} "
        f"trigger={channel.trigger_source}"
    )


def _optional(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    "auto-trainer-core/src",
    "auto-trainer-device/src",
):
    sys.path.insert(0, str(_REPO_ROOT / _path))

from autotrainer.core import Motor
from autotrainer.device import CanInterface, CanTransportConfiguration, Target


def main() -> int:
    args = _parse_args()
    transport = CanTransportConfiguration(
        kind=args.transport,
        channel=args.channel,
        bitrate=args.bitrate,
        data_bitrate=args.data_bitrate,
        fd=args.fd,
        receive_timeout_seconds=args.receive_timeout_seconds,
    )
    interface = CanInterface(
        required_targets=(Target.PELLET_DEVICE,),
        can_transport=transport,
    )
    try:
        print(
            f"opening CAN transport={transport.kind.value} channel={transport.channel} "
            f"bitrate={transport.bitrate} fd={transport.fd}"
        )
        if not interface.open():
            raise RuntimeError("CAN interface failed to open")
        if not interface.are_addresses_valid():
            raise RuntimeError("required pellet CAN board address was not discovered")
        print(f"pellet board address: {interface.pellet_address}")

        if args.action == "discover":
            _listen(interface, args.listen_seconds)
            return 0
        if args.action == "version":
            if not interface.request_version():
                raise RuntimeError("firmware version request failed")
            _listen(interface, args.listen_seconds)
            return 0
        if args.action == "listen":
            _listen(interface, args.listen_seconds)
            return 0
        if args.action == "request-motor-config":
            motor = _motor_from_name(args.motor)
            if not interface.request_motor_config(motor):
                raise RuntimeError(f"motor configuration request failed for {motor}")
            _listen(interface, args.listen_seconds)
            return 0

        _require_motion_allowed(args)
        if args.action == "set-x":
            _require_success(interface.set_motor_x(args.position, relative=args.relative), args.action)
        elif args.action == "set-y":
            _require_success(interface.set_motor_y(args.position, relative=args.relative), args.action)
        elif args.action == "set-z":
            _require_success(interface.set_motor_z(args.position, relative=args.relative), args.action)
        elif args.action == "move-load-servo":
            _require_success(interface.move_load_servo(args.position), args.action)
        elif args.action == "move-cover-servo":
            _require_success(interface.move_cover_servo(args.position), args.action)
        elif args.action == "home":
            _require_success(interface.stepper_home(_motor_from_name(args.motor)), args.action)
        else:
            raise RuntimeError(f"Unhandled action: {args.action}")
        _listen(interface, args.listen_seconds)
        return 0
    finally:
        interface.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate reachAQ SocketCAN/PCAN pellet-board communication.")
    parser.add_argument("--transport", choices=("socketcan", "pcan_basic"), default="socketcan")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--bitrate", type=int, default=1000000)
    parser.add_argument("--data-bitrate", type=int, default=5000000)
    parser.add_argument(
        "--fd",
        action="store_true",
        default=True,
        help="Use CAN FD (required by JerryCAN; enabled by default).",
    )
    parser.add_argument("--receive-timeout-seconds", type=float, default=0.0)
    parser.add_argument(
        "--action",
        choices=(
            "discover",
            "version",
            "listen",
            "request-motor-config",
            "set-x",
            "set-y",
            "set-z",
            "move-load-servo",
            "move-cover-servo",
            "home",
        ),
        default="discover",
    )
    parser.add_argument("--listen-seconds", type=float, default=2.0)
    parser.add_argument("--allow-motion", action="store_true", help="Required for any command that can move hardware.")
    parser.add_argument("--position", type=float, default=0.0, help="Position for set/move actions.")
    parser.add_argument("--relative", action="store_true", help="Use relative movement for set-x/set-y/set-z.")
    parser.add_argument("--motor", choices=("x", "y", "z", "load", "cover"), default="x")
    return parser.parse_args()


def _listen(interface: CanInterface, seconds: float) -> None:
    print(f"listening for {seconds:.3f} seconds")
    end = time.perf_counter() + seconds
    count = 0
    while time.perf_counter() < end:
        messages = interface.read(50, collect_ms=5)
        for message in messages:
            count += 1
            print(f"  {message!r}")
        if not messages:
            time.sleep(0.005)
    print(f"received {count} translated CAN messages")


def _motor_from_name(name: str) -> Motor:
    if name == "x":
        return Motor.PELLET_X_MOTOR
    if name == "y":
        return Motor.PELLET_Y_MOTOR
    if name == "z":
        return Motor.PELLET_Z_MOTOR
    if name == "load":
        return Motor.PELLET_LOAD_SERVO
    if name == "cover":
        return Motor.PELLET_COVER_SERVO
    raise ValueError(f"Unhandled motor name: {name}")


def _require_motion_allowed(args: argparse.Namespace) -> None:
    if not args.allow_motion:
        raise RuntimeError(f"{args.action} can move hardware; rerun with --allow-motion after confirming rig safety")


def _require_success(result: bool, action: str) -> None:
    if not result:
        raise RuntimeError(f"CAN action failed: {action}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

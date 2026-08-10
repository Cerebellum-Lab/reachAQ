import argparse
import ast
import logging
import queue
import sys
import time
from threading import Thread
from copy import copy
from enum import IntEnum

from autotrainer.core import SystemStatusMessageKind, SystemCommandKind, EventManager
from autotrainer.core.logging import setup_logging
from autotrainer.core.message import SystemDataArgsKwargs
from autotrainer.device import (
    CanDevice,
    DeviceConnection,
    Motor,
    StepperConfig,
    ServoConfig,
    motor_to_str,
    target_to_str,
    is_stepper,
    CompoundMovements,
    MotorConfigurationFile,
    StepperStatus,
    is_servo,
    Target,
)

msg_queue_active = True
output_file = None
perf_start = None
perf_count = -1
perf_print = False
print_motor_status = Motor.NONE
print_status = None
get_input = True


class StatusType(IntEnum):
    STIMULUS = 1


def monitor_message_queue(msg_queue):
    global perf_start, perf_count, print_motor_status, print_status
    global get_input

    logger.info("starting message queue thread")

    measurement_count = 0

    perf_end = None

    output_fd = None

    if output_file is not None:
        output_fd = open(output_file, 'w')
        output_fd.write("Time, Index, Weight, Switch, Pressure, Temperature, Humidity\n")

    next_heartbeat = 500

    next_perf_log = time.perf_counter() + 600
    kinds = set()

    while msg_queue_active:

        if __debug__:
            perf_now = time.perf_counter()
            if perf_now > next_perf_log:
                logger.debug("kinds=%s", sorted(kinds))
                kinds = set()
                next_perf_log += 600

        try:
            kind, data = msg_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.001)
            continue

        kinds.add(kind)

        if kind == SystemStatusMessageKind.ACKNOWLEDGE:
            get_input = True

        elif kind == SystemStatusMessageKind.FIRMWARE_VERSION:
            print(data)
            get_input = True

        elif kind == SystemStatusMessageKind.MOTOR_CONFIGURATION:
            if isinstance(data, ServoConfig):
                print(
                    f"SERVO\n"
                    f"- target={target_to_str(data.target)}\n"
                    f"- motor={motor_to_str(data.motor)}\n"
                    f"- max vel (deg/sec)={data.maximum_velocity:.2f}\n"
                    f"- max accel (deg/sec^2)={data.maximum_acceleration:.2f}\n"
                    f"- min pos (deg)={data.minimum_position:.1f}\n"
                    f"- max pos (deg)={data.maximum_position:.1f}\n"
                    f"- min pwm={data.minimum_pwm_duration:.1f}\n"
                    f"- max pwm={data.maximum_pwm_duration:.1f}\n"
                    f"- detach={data.detach}\n"
                )
            elif isinstance(data, StepperConfig):
                print(f"STEPPER\n"
                      f"- target={target_to_str(data.target)}\n"
                      f"- motor={motor_to_str(data.motor)}\n"
                      f"- max vel (mm/sec)={data.maximum_velocity:.2f}\n"
                      f"- max accel (mm/sec^2)={data.maximum_acceleration:.2f}\n"
                      f"- home vel (mm/sec)={data.homing_velocity:.2f}\n"
                      f"- flip limit orientation={data.flip_limit_orientation}\n"
                      f"- microsteps={data.microsteps}\n"
                      f"- step/rev={data.steps_per_revolution:.0f}\n"
                      )
            get_input = True

        elif (kind, print_motor_status) in (
            (SystemStatusMessageKind.PELLET_COVER, Motor.PELLET_COVER_SERVO),
            (SystemStatusMessageKind.PELLET_LOAD, Motor.PELLET_LOAD_SERVO),
        ):
            # TODO deliver full packet. See can_device at or around line 328
            # assert isinstance(data, ServoStatus)
            print(
                f"SERVO:\n"
                # f"- target={target_to_str(data.target)}\n"
                f"- motor={motor_to_str(print_motor_status)}\n"
                f"- position (deg)={data:.2f}\n"
            )
            print_motor_status = Motor.NONE
            get_input = True

        elif ((kind == SystemStatusMessageKind.PELLET_MOTOR_X and print_motor_status == Motor.PELLET_X_MOTOR)
           or (kind == SystemStatusMessageKind.PELLET_MOTOR_Y and print_motor_status == Motor.PELLET_Y_MOTOR)
           or (kind == SystemStatusMessageKind.PELLET_MOTOR_Z and print_motor_status == Motor.PELLET_Z_MOTOR)
        ):
            assert isinstance(data, StepperStatus)
            print(
                f"STEPPER:\n"
                # f"target={target_to_str(data.target)}\n"
                f"- motor={motor_to_str(print_motor_status)}\n"
                f"- position (mm)={data.position:.3f}\n"
                f"- send_pos (mm)={data.send_position:.3f}\n"
                f"- limit={data.is_at_limit}\n"
            )
            print_motor_status = Motor.NONE
            get_input = True

        elif kind == SystemStatusMessageKind.STIMULUS_INPUTS:
            if print_status is StatusType.STIMULUS:
                for i in range(4):
                    print(f"- Stimulus #{i + 1}:     {data[i]}")
                print_status = None
                get_input = True

    if output_fd is not None:
        output_fd.close()

    if perf_print and perf_count > 0 and perf_start is not None:
        logger.info(
            f"{perf_count} samples at {(1.0e9 * perf_count) / (perf_end - perf_start)} samples/s")


def str_to_motor(motor_name: str):
    if motor_name == 'x':
        return Motor.PELLET_X_MOTOR
    elif motor_name == 'y':
        return Motor.PELLET_Y_MOTOR
    elif motor_name == 'z':
        return Motor.PELLET_Z_MOTOR
    elif motor_name == 'load' or motor_name == 'l':
        return Motor.PELLET_LOAD_SERVO
    elif motor_name == 'cover' or motor_name == 'c':
        return Motor.PELLET_COVER_SERVO
    else:
        return None


def write_config(motor: Motor, device_thread):
    if motor is None:
        return

    orig_config = device_thread._interface.get_motor_configuration(motor)

    if is_stepper(motor):
        assert isinstance(orig_config, StepperConfig)
        config = copy(orig_config)

        resp = input(f"Max Velocity (mm/sec) [{orig_config.maximum_velocity:.2f}] = ")
        if resp != '':
            config.maximum_velocity = float(resp)

        resp = input(f"Max Acceleration (mm/sec^2) [{orig_config.maximum_acceleration:.2f}]= ")
        if resp != '':
            config.maximum_acceleration = float(resp)

        resp = input(f"Homing Velocity (mm/sec) [{orig_config.homing_velocity:.2f}] = ")
        if resp != '':
            config.homing_velocity = float(resp)

        resp = input(f"Flip Limit Location [0, 1] [{orig_config.flip_limit_orientation}]= ")
        if resp != '':
            config.flip_limit_orientation = int(resp) == 1

        resp = input(f"Microsteps [2,4,8,16,32,64] [{orig_config.microsteps}]= ")
        if resp != '':
            config.microsteps = int(resp)

        resp = input(f"Steps/Revolution [{orig_config.steps_per_revolution:.0f}]= ")
        if resp != '':
            config.steps_per_revolution = float(resp)

        device_thread.send_message(SystemCommandKind.WRITE_MOTOR_CONFIGURATION, (motor, config),
                                   context="write stepper config")
    else:
        assert isinstance(orig_config, ServoConfig)

        config = copy(orig_config)

        resp = input(f"Max Velocity (deg/sec) [{orig_config.maximum_velocity:.2f}]= ")
        if resp != '':
            config.maximum_velocity = float(resp)

        resp = input(f"Max Acceleration (deg/sec^2) [{orig_config.maximum_acceleration:.2f}]= ")
        if resp != '':
            config.maximum_acceleration = float(resp)

        resp = input(f"Min Position (deg) [{orig_config.minimum_position:.1f}]= ")
        if resp != '':
            config.minimum_position = float(resp)

        resp = input(f"Max Position (deg) [{orig_config.maximum_position:.1f}]= ")
        if resp != '':
            config.maximum_position = float(resp)

        resp = input(f"Min PWM Duration (usec) [{orig_config.minimum_pwm_duration:.1f}]= ")
        if resp != '':
            config.minimum_pwm_duration = float(resp)

        resp = input(f"Max PWM Duration (usec) [{orig_config.maximum_pwm_duration:.1f}]= ")
        if resp != '':
            config.maximum_pwm_duration = float(resp)

        resp = input(f"Detach [{orig_config.detach}]= ")
        if resp != '':
            config.detach = bool(ast.literal_eval(resp))

        device_thread.send_message(SystemCommandKind.WRITE_MOTOR_CONFIGURATION, (motor, config),
                                   context="write servo config")


def round_trip_test(motor: Motor, trips: int, device_thread):
    global print_motor_status

    cmd = motor_to_move_command[motor]

    for i in range(trips):
        device_thread.send_message(cmd, data=22.0 if is_stepper(motor) else 100, context="trip out")
        time.sleep(2)

        device_thread.send_message(cmd, data=0, context="trip in")
        time.sleep(2)


# Generate either just a position value or a tuple of (position, rate)
def move_parameter(params):
    if len(params) == 1:
        return float(params[0])
    else:
        return float(params[0]), float(params[1])


def run_monitor():
    global get_input
    global perf_count
    global print_motor_status
    global print_status, msg_queue_active

    msg_queue = queue.Queue()

    mon_thread = Thread(target=monitor_message_queue, args=(msg_queue,))
    mon_thread.start()

    can_dev = CanDevice(required_targets=(Target.PELLET_DEVICE,))
    device_connection = DeviceConnection(can_dev, msg_queue)

    device_connection.request_connect()

    # Not necessary to ===>>>
    # time.sleep(0.01)  # this is to allows the request connection to be executed by the device thread
    # # and allows it to get/read the devices address at its startup
    # end = time.time() + 1.5
    # while time.time() < end:
    #     if can_dev.device_interface.are_addresses_valid():
    #         break
    #     time.sleep(0.05)
    # if not can_dev.device_interface.are_addresses_valid():
    #     logger.critical("Could not read devices CAN bus addr in time")
    # # only then we are able to set the motor configuration:

    # <<<=== given the load default motor config uses the device queue to send the commands,
    # and that the device thread goes into its main loop after having received the above request_connect(),
    # and that before entering its main loop it already reads and assign the devices addresses,
    # and that then it cannot miss any of the commands put into the queue in its main loop,

    device_connection.load_default_motor_config()
    device_connection.load_default_move_config()

    last_command = ""

    while True:
        if perf_count <= 0:
            while not get_input:
                time.sleep(0.1)
            get_input = False
            line = ""
            while len(line) == 0:
                line = input(f"Enter command (?=help, enter='{last_command}'): ")
                if len(line) == 0 and len(last_command) != 0:
                    break

            if len(line) == 0:
                line = last_command
            else:
                last_command = line

            argv = line.split()

            if len(argv) == 0:
                get_input = True
                print("empty command, please type a command or ?")
                continue

            cmd = argv[0]
            params = argv[1:]

            motor = str_to_motor(cmd)

            # '?' - help
            # 'a' - audio
            # 'c' - cover servo
            # 'd' - delay (sec)
            # 'f' - load-from-files
            # 'h' - home stepper
            # 'k' - stepper known position
            # 'l' - load servo
            # 'o' - set output
            # 'p' - pellet move commands
            # 'r' - RGB LED
            # 's' - System status
            # 'v' - version
            # 'x' - x motor
            # 'y' - y motor
            # 'z' - z motor
            try:
                if cmd == '?':
                    print_help()
                    get_input = True

                elif motor is not None:
                    handle_motor_command(motor, params, device_connection)

                elif cmd == 'a' or cmd == 'audio':
                    device_connection.send_message(SystemCommandKind.PLAY_TONE,
                                               # period from sec to msec
                                               data=(int(params[0]), int(float(params[1]) * 1000)),
                                               context="tone")

                elif cmd == 'd' or cmd == 'delay':
                    device_connection.send_message(SystemCommandKind.DELAY, float(params[0]),
                                               context="delay")

                elif cmd == 'f' or cmd == 'file':
                    if params[0] == 'motor':
                        motors_cfg = MotorConfigurationFile.from_file(params[1])
                        device_connection.use_motor_configurations(motors_cfg)
                    elif params[0] == 'move':
                        movements_cfg = CompoundMovements.from_file(params[1])
                        device_connection.use_compound_movements(movements_cfg)
                    else:
                        logger.error(f"Unknown file request: {params[0]}")
                    get_input = True

                elif cmd == 'h' or cmd == 'home':
                    device_connection.send_message(SystemCommandKind.SEND_HOME, context="home")

                elif cmd == 'k' or cmd == 'known':
                    device_connection.send_message(SystemCommandKind.SEND_FIXED_XYZ, context="known")

                elif cmd == 'o' or cmd == 'output':
                    handle_output_command(params, device_connection)

                elif cmd == 'p' or cmd == 'pellet':
                    handle_pellet_command(params, device_connection)

                elif cmd == 'q' or cmd == 'quit':
                    device_connection.request_disconnect()
                    break

                elif cmd == 'r' or cmd == 'rgb':
                    device_connection.send_message(SystemCommandKind.SET_RGB_LED,
                                               (int(params[0]), int(params[1]), int(params[2])),
                                               context="rgb")

                elif cmd == 's' or cmd == 'status':
                    print_status = StatusType.STIMULUS

                elif cmd == 'v' or cmd == 'version':
                    device_connection.send_message(SystemCommandKind.REQUEST_VERSION)

                elif cmd == 'board_reboot':
                    if len(params) != 1:
                        logger.warning("expected board target name: pellet")
                        continue
                    if params[0] == "pellet":
                        tgt = Target.PELLET_DEVICE
                    else:
                        logger.error("unknown target board: %s", params[0])
                        continue
                    device_connection.send_message(SystemCommandKind.BOARD_REBOOT, tgt, context=f"board_reboot_{tgt}")

                elif cmd == "logger":
                    get_input = True
                    if len(params) == 0:
                        logger.info("handlers=%s", logging.root.handlers)
                        logger.info("level=%s", logging.root.level)
                    elif len(params) >= 1:
                        if len(params) >= 2:
                            name, level = params
                        else:
                            name = logging.root.name
                            level = params[0]
                        log = logging.getLogger(name)
                        if level.isdigit():
                            level = int(level)
                        logger.info("logger %s: level=%s", log.name, log.level)
                        log.setLevel(level)
                else:
                    get_input = True
                    logger.warning("Unknown command: %s", cmd)

            except ValueError as err:
                logger.exception("ValueError: %s", err)
                get_input = True
        else:
            if not mon_thread.is_alive():
                device_connection.request_disconnect()
                break
            else:
                time.sleep(0.1)

    msg_queue_active = False

    mon_thread.join()

    logger.info("waiting for device connection to terminate")

    device_connection.join()

    EventManager.try_close_default()

    logger.info("done")


motor_to_set_command = {
    Motor.PELLET_X_MOTOR: SystemCommandKind.SET_X,
    Motor.PELLET_Y_MOTOR: SystemCommandKind.SET_Y,
    Motor.PELLET_Z_MOTOR: SystemCommandKind.SET_Z,
}

motor_to_move_command = {
    Motor.PELLET_X_MOTOR: SystemCommandKind.MOVE_X,
    Motor.PELLET_Y_MOTOR: SystemCommandKind.MOVE_Y,
    Motor.PELLET_Z_MOTOR: SystemCommandKind.MOVE_Z,
    Motor.PELLET_COVER_SERVO: SystemCommandKind.MOVE_COVER_SERVO,
    Motor.PELLET_LOAD_SERVO: SystemCommandKind.MOVE_LOAD_SERVO,
}


def handle_motor_command(motor: Motor, params, device_connection):
    global print_motor_status, get_input
    try:
        float(params[0])
        numeric = True
    except ValueError:
        numeric = False
    except IndexError:
        numeric = False

    # query stepper status
    if len(params) == 0:
        print_motor_status = motor
        logger.debug("print_motor_status=%s", print_motor_status)
    # set position
    elif params[0] == 'move':
        float_params = move_parameter(params[1:])
        args_kwargs = SystemDataArgsKwargs(float_params)
        if motor in motor_to_set_command:
            # only consider relative move for stepper X/Y/Z ; not for servo motors
            relative = params[1].startswith(('+', '-'))
            args_kwargs.kwargs["relative"] = relative
        device_connection.send_message(
            motor_to_move_command[motor],
            data=args_kwargs,
            context="motor move",
        )

    # set position (no 'move')
    elif numeric:
        device_connection.send_message(motor_to_move_command[motor],
                                       data=move_parameter(params[0:]), context="motor move")

    elif params[0] == 'set':
        device_connection.send_message(motor_to_set_command[motor],
                                       data=move_parameter(params[1:]), context="motor set")

    # sent to home position
    elif params[0] == 'home':
        device_connection.send_message(SystemCommandKind.SEND_TO_LIMITS, data=motor, context="motor "
                                                                                         "home")

    elif params[0] == 'trip':
        round_trip_test(motor, int(params[1]), device_connection)
        get_input = True

    # read or write configuration
    elif params[0] == 'config':
        if params[1] == 'read':
            device_connection.send_message(SystemCommandKind.READ_MOTOR_CONFIGURATION, data=motor,
                                           context="motor read")
        elif params[1] == 'write':
            write_config(motor, device_connection)
        else:
            print(f"Unrecognized configuration request: {params[1]}")
            get_input = True

    elif params[0] == 'step':
        start = int(params[1])
        stop = int(params[2])
        step = int(params[3])
        step = step if start < stop else -step

        for position in range(start, stop + step, step):
            device_connection.send_message(motor_to_move_command[motor],
                                           data=float(position), context="motor step")
            time.sleep(1 + .25 * step)
        get_input = True

    elif params[0] in ('attach', 'detach'):
        if not is_servo(motor):
            print("BAD servo motor")
        else:
            cmd = SystemCommandKind.SERVO_ATTACH if params[0] == "attach" else SystemCommandKind.SERVO_DETACH
            device_connection.send_message(cmd, data=motor)
        get_input = True

    else:
        print(f"Unrecognized motor command: {params[0]}")
        get_input = True


def handle_pellet_command(params, device_connection):
    global get_input
    """
    Handle a pellet control sequence
    """
    cmd = params[0]

    if cmd == 'cover' or cmd == 'c':
        device_connection.send_message(SystemCommandKind.COVER_PELLET, context="pellet cover")
    elif cmd == 'load' or cmd == 'l':
        device_connection.send_message(SystemCommandKind.LOAD_PELLET, context="pellet load")
    elif cmd == 'release' or cmd == 'r':
        device_connection.send_message(SystemCommandKind.RELEASE_PELLET, context="pellet release")
    elif cmd == 'send' or cmd == 's':
        device_connection.send_message(SystemCommandKind.SEND_PELLET, context="pellet send")
    else:
        print(f"Unrecognized pellet command: {cmd}")
        get_input = True


def handle_output_command(params, device_connection):
    cmd = params[0]

    if cmd == 'd' or cmd == 'digital':
        device_connection.send_message(SystemCommandKind.SET_DIGITAL_OUTPUT,
                                       (int(params[1]), int(params[2])), context="dout")
    elif cmd == 'a' or cmd == 'analog':
        device_connection.send_message(SystemCommandKind.SET_ANALOG_OUTPUT,
                                       (int(params[1]), int(params[2])), context="aout")


def print_help():
    print("?                                  "
          " ::help")
    print("For the commands, you can either use the letter or full command name (e.g. q or quit)\n")

    print("<motor>                            "
          " ::Motor status")
    print("<motor> [move] <pos> [<rate>]      "
          " ::Move servo pos [0:120] (deg) rate [0:100] (%)\n"
          "                                   "
          " ::Move stepper pos [0:35] (mm) rate [0:100] (%)")
    print("<motor> set <pos>                  "
          " ::Set stepper send location pos [0:35] (mm)")
    print("<motor> step <start> <end> <step>"
          " ::Step degrees or mms at a time")
    print("<motor> config read                "
          " ::Read Configuration")
    print("<motor> config write               "
          " ::Write Configuration")
    print("<motor> trip <cnt>                 "
          " ::<cnt> Round trips")
    print("<motor> is one of: x, y, z, l[oad], c[over]")
    print()
    print("<servo> attach/detach              "
          " ::Attach or Detach from a servo")
    print()
    print("p[ellet] c[over]                   "
          " ::Cover Pellet Sequence")
    print("p[ellet] l[oad]                    "
          " ::Load Pellet Sequence")
    print("p[ellet] r[elease]                 "
          " ::Release Pellet Sequence")
    print("p[ellet] s[end]                    "
          " ::Send Pellet Sequence")
    print()

    print("a[udio] <freq> <period>            "
          " ::Audio sound (hz) (sec)")
    print("d[elay] <sec>                      "
          " ::Delay")
    print("h[ome]                             "
          " ::Go to Home Position (0, 0, 0)")
    print("k[nown]                            "
          " ::Go to Known/Send Position (X, Y, Z)")
    print("f[ile] motor <file>                "
          " ::Load Motor Configuration")
    print("f[ile] move <file>                 "
          " ::Load Compound Movement Configuration")
    print("o[utput] d[igital] <chan> <state>  "
          " ::Set digital output on pellet chan [1:4] state [0:1]")
    print("o[utput] a[nalog] <chan> <mvolts>  "
          " ::Set analog output on pellet chan [1] mvolts [0:5000]")
    print("q[uit]                             "
          " ::Quit")
    print("r[gb] <red> <green> <blue>         "
          " ::Set RGB LED. Values in %")
    print("s[tatus]                           "
          " ::Show Status")
    print("board_reboot <board>               "
          " ::Reboot the pellet board")
    print("v[ersion]                          "
          " ::Version")
    print("logger [<name>] <level>            "
          " ::Set logger [name] level")
    print()


def parse_log_level(value: str):
    if value.isdigit():
        return int(value)
    return value


def main():
    global output_file, perf_print

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", help="and output file to record measurements")
    parser.add_argument("-p", "--perf",
                        help="performance measurement with specified number of samples",
                        type=int, default=-1)
    parser.add_argument("--log-level", default=logging.INFO, type=parse_log_level)

    args = parser.parse_args()

    logging.getLogger("autotrainer").setLevel(args.log_level)  # can be changed with "logger" cli command
    # logging.root.setLevel(args.log_level)

    output_file = args.output

    perf_print = perf_count != -1

    run_monitor()


if __name__ == '__main__':
    logger = setup_logging(time_precision=4)
    sys.exit(main())

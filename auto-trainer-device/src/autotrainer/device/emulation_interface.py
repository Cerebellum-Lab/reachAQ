import math
import threading
import time
import typing
from pathlib import Path
from random import uniform, random
from typing import Union, Tuple

from autotrainer.core import Offset3DTuple
from autotrainer.core.logging import get_verbose_logger

from .device_interface import (DeviceInterface, ServoConfig, StepperConfig,
                               StepperStatus, ServoStatus, Target, DigitalOutputs,
                               Motor, AnalogOutputs, SensorStatus, MagnetDigitalInputs,
                               AudioData, PressureReading, Version,
                               PelletDigitalInputs, DoorData, Acknowledge, ColorLed
                               )
from .can_interface import motor_to_str

logger = get_verbose_logger(__name__)


# Slower than the real hardware to be more forgiving in emulation.
_STATUS_MESSAGE_INTERVAL = 2.0
_AUDIO_MESSAGE_INTERVAL = 0.5
_DATA_MESSAGE_INTERVAL = 0.2


class _SharedList:
    # just to make life easier for making thread safe previous code below using this.

    def __init__(self, lock):
        self._lock = lock
        self._value = []

    def append(self, item):
        with self._lock:
            self._value.append(item)

    def get_and_reset(self):
        with self._lock:
            cur = self._value
            self._value = []
        return cur


class EmulationInterface(DeviceInterface):

    _uuid: int = 1

    @classmethod
    def next_uuid(cls) -> int:
        cls._uuid = cls._uuid + 1 & 0xFF
        if cls._uuid == 0:  # don't allow 0's
            cls._uuid = 1
        return cls._uuid

    @classmethod
    def uuid(cls) -> int:
        return cls._uuid

    def __init__(self):
        super().__init__()

        self._thread_lock = threading.Lock()
        self._is_open = False

        self._last_status_message = 0.0
        self._last_audio_message = 0.0
        self._last_data_message = 0.0

        self._color_led = ColorLed()

        self._positions = {
            Motor.PELLET_LOAD_SERVO: 0.0,
            Motor.PELLET_X_MOTOR: 0.0,
            Motor.PELLET_Y_MOTOR: 0.0,
            Motor.PELLET_Z_MOTOR: 0.0,
            Motor.TUNNEL_MAGNET_SERVO: 0.0,
            Motor.TUNNEL_GATE_SERVO: 0.0,
            Motor.PELLET_COVER_SERVO: 0.0,
            Motor.TUNNEL_FAN_SERVO: 0.0,
        }
        self._send_pos = {
            Motor.PELLET_X_MOTOR: 0.0,
            Motor.PELLET_Y_MOTOR: 0.0,
            Motor.PELLET_Z_MOTOR: 0.0,
        }

        self._configs = {
            Motor.PELLET_LOAD_SERVO: ServoConfig(Target.PELLET_DEVICE, Motor.PELLET_LOAD_SERVO),
            Motor.PELLET_COVER_SERVO: ServoConfig(Target.PELLET_DEVICE, Motor.PELLET_COVER_SERVO),
            Motor.TUNNEL_MAGNET_SERVO: ServoConfig(Target.MAGNET_DEVICE, Motor.TUNNEL_MAGNET_SERVO),
            Motor.TUNNEL_GATE_SERVO: ServoConfig(Target.MAGNET_DEVICE, Motor.TUNNEL_GATE_SERVO),
            Motor.PELLET_X_MOTOR: StepperConfig(Target.PELLET_DEVICE, Motor.PELLET_X_MOTOR),
            Motor.PELLET_Y_MOTOR: StepperConfig(Target.PELLET_DEVICE, Motor.PELLET_Y_MOTOR),
            Motor.PELLET_Z_MOTOR: StepperConfig(Target.PELLET_DEVICE, Motor.PELLET_Z_MOTOR),
            Motor.TUNNEL_FAN_SERVO: ServoConfig(Target.PELLET_DEVICE, Motor.TUNNEL_FAN_SERVO),
        }

        self._messages = _SharedList(lock=self._thread_lock)
        #
        self._prev_audio_data = self._cur_audio_data = None
        self._audio_replay_fh = None

    def _check_audio_replay(self):
        p = Path("./audio_spectrum_replay.csv")
        if p.exists():
            logger.info("opening %s for replay", p)
            fh = p.open()
            fh.readline()  # skip header
            self._prev_audio_data = None
            cur = self._cur_audio_data = self._read_audio_row(fh)
            if cur is None:
                raise RuntimeError("empty audio replay csv file")
            self._audio_replay_fh = fh
            self._audio_replay_when_start = cur.when
            self._audio_when_diff_start = time.time() - cur.when
        else:
            self._cur_audio_data = self._prev_audio_data = None
            self._audio_replay_fh = None
            self._audio_when_diff_start = None

    @staticmethod
    def _read_audio_row(fh):
        # could/should use csv.DictReader, but previous csv files contains some extra space that below .strip() calls
        # correctly handle easily.
        data = fh.readline()
        if not data:
            return None
        when, index, *data = data.split(",")
        a = AudioData(
            target=Target.MAGNET_DEVICE,
            packet_id=1,
            when=float(when.strip()),
            index=int(index.strip()),
            magnitudes=list(map(lambda v: float(v.strip()), data))
        )
        # logger.debug("read: %s", a)
        return a

    def _set_pellet_address(self, addr):
        pass

    def _set_magnet_address(self, addr):
        pass

    @property
    def motors_position(self):
        return Offset3DTuple(
            self._positions[k]
            for k in (Motor.PELLET_X_MOTOR, Motor.PELLET_Y_MOTOR, Motor.PELLET_Z_MOTOR)
        )

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> bool:
        self._is_open = True
        self._check_audio_replay()
        return self._is_open

    def close(self):
        self._is_open = False

    def can_read(self) -> bool:
        return self._is_open

    def read(self, max_count: int = 1, *, collect_ms: int = 0) -> typing.Any:
        # TODO: handle collect_ms

        messages = self._messages.get_and_reset()

        perf_now = time.perf_counter()

        # Just to do one type, even if all should be updated.  Do not want this to be taking up much time.
        if perf_now - self._last_status_message > _STATUS_MESSAGE_INTERVAL:
            self._last_status_message = perf_now
            for motor in (Motor.PELLET_X_MOTOR, Motor.PELLET_Y_MOTOR, Motor.PELLET_Z_MOTOR):
                messages.append(
                    StepperStatus(Target.PELLET_DEVICE, motor,
                                  self._positions[motor],
                                  self._send_pos[motor],
                                  self._positions[motor] == 0))

            messages.append(
                ServoStatus(Target.PELLET_DEVICE, Motor.PELLET_COVER_SERVO, self._positions[
                    Motor.PELLET_COVER_SERVO]))

            messages.append(
                ServoStatus(Target.PELLET_DEVICE, Motor.PELLET_LOAD_SERVO,
                            self._positions[Motor.PELLET_LOAD_SERVO]))

            messages.append(ServoStatus(Target.MAGNET_DEVICE, Motor.TUNNEL_MAGNET_SERVO,
                                        self._positions[Motor.TUNNEL_MAGNET_SERVO]))

            messages.append(ServoStatus(Target.MAGNET_DEVICE, Motor.TUNNEL_GATE_SERVO,
                                        self._positions[Motor.TUNNEL_GATE_SERVO]))

            messages.append(
                ServoStatus(Target.PELLET_DEVICE, Motor.TUNNEL_FAN_SERVO, self._positions[
                    Motor.TUNNEL_FAN_SERVO]))

            messages.append(
                MagnetDigitalInputs(continuity_0=random() < 0.1, continuity_1=random() < 0.1))
            messages.append(PelletDigitalInputs(
                target=Target.PELLET_DEVICE, stimulus_1=True, stimulus_2=False, stimulus_3=True, stimulus_4=False))
            messages.append(DoorData())
            messages.append(SensorStatus(temperature_c=28.0 + uniform(-2, 2),
                                         humidity_percent=50.0 + uniform(-2, 2)))
            messages.append(self._color_led)

        fh_audio_replay = self._audio_replay_fh
        if fh_audio_replay is not None:
            prev, cur = self._prev_audio_data, self._cur_audio_data
            now = time.time()
            if prev is None or now - cur.when - self._audio_when_diff_start > 0:
                self._last_audio_message = now
                cur.when = now
                self._prev_audio_data = cur
                messages.append(cur)
                cur = self._cur_audio_data = self._read_audio_row(fh_audio_replay)
                if cur is None:
                    self._check_audio_replay()  # loopback
        elif perf_now - self._last_audio_message > _AUDIO_MESSAGE_INTERVAL:
            self._last_audio_message = perf_now
            audio = AudioData(target=Target.MAGNET_DEVICE, packet_id=1, when=time.time(),
                              index=time.perf_counter_ns())
            spectrum = []
            for _ in range(64):
                spectrum.append(uniform(80, 130))
            audio.magnitudes = spectrum
            messages.append(audio)

        if perf_now - self._last_data_message > _DATA_MESSAGE_INTERVAL:
            self._last_data_message = perf_now
            messages.append(PressureReading(pressure=512 + uniform(-10, 10), ))

        return messages

    def write(self, value: typing.Any) -> int:
        if self._is_open:
            return 1

        return 0

    @property
    def cover_config(self):
        return self._configs[Motor.PELLET_COVER_SERVO]

    @property
    def load_config(self):
        return self._configs[Motor.PELLET_LOAD_SERVO]

    @property
    def magnet_config(self):
        return self._configs[Motor.TUNNEL_MAGNET_SERVO]

    @property
    def gate_config(self):
        return self._configs[Motor.TUNNEL_GATE_SERVO]

    @property
    def x_config(self):
        return self._configs[Motor.PELLET_X_MOTOR]

    @property
    def y_config(self):
        return self._configs[Motor.PELLET_Y_MOTOR]

    @property
    def z_config(self):
        return self._configs[Motor.PELLET_Z_MOTOR]

    def set_motor_configuration(self, motor: Motor, config, _write_to_remote: bool = True) -> bool:
        if self._is_open:
            logger.info(f"Set motor configuration %s", motor)
            self._configs[motor] = config
            self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return self._is_open

    def tare_pressure_sensor(self) -> bool:
        if self._is_open:
            logger.info(f"tare pressure sensor")
            self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return self._is_open

    def move_magnet_servo(self, position: float) -> bool:
        if self._is_open:
            logger.info(f"set magnet position {position}")
            self._positions[Motor.TUNNEL_MAGNET_SERVO] = position + 0.00001
            self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return self._is_open

    def move_gate_servo(self, position: float) -> bool:
        if self._is_open:
            logger.info(f"set gate position {position}")
            self._positions[Motor.TUNNEL_GATE_SERVO] = position + 0.00001
            self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return self._is_open

    def set_motor_x(self, position, *, relative: bool=False) -> bool:
        return self.move_motor_x(position, True, relative=relative)

    def move_motor_x(self, position: float, save_as_fixed: bool = False, *, relative: bool=False) -> bool:
        if self._is_open:
            logger.info("set pellet %s x %s", ("absolute", "relative")[relative], position)
            if save_as_fixed:
                if relative:
                    position += self._send_pos[Motor.PELLET_X_MOTOR]
                self._send_pos[Motor.PELLET_X_MOTOR] = position
            else:
                if relative:
                    position += self._positions[Motor.PELLET_X_MOTOR]
                self._positions[Motor.PELLET_X_MOTOR] = position + 0.00001
            self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return self._is_open

    def set_motor_y(self, position, *, relative: bool = False) -> bool:
        return self.move_motor_y(position, True, relative=relative)

    def move_motor_y(self, position: float, save_as_fixed: bool = False, *, relative: bool=False) -> bool:
        if self._is_open:
            logger.info("set pellet %s y %s", ("absolute", "relative")[relative], position)
            if save_as_fixed:
                if relative:
                    position += self._send_pos[Motor.PELLET_Y_MOTOR]
                self._send_pos[Motor.PELLET_Y_MOTOR] = position
            else:
                if relative:
                    position += self._positions[Motor.PELLET_Y_MOTOR]
                self._positions[Motor.PELLET_Y_MOTOR] = position + 0.00001
            self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return self._is_open

    def set_motor_z(self, position, *, relative: bool=False) -> bool:
        return self.move_motor_z(position, True, relative=relative)

    def move_motor_z(self, position: float, save_as_fixed: bool = False, *, relative: bool=False) -> bool:
        if self._is_open:
            logger.info("set pellet %s z %s", ("absolute", "relative")[relative], position)
            if save_as_fixed:
                if relative:
                    position += self._send_pos[Motor.PELLET_Z_MOTOR]
                self._send_pos[Motor.PELLET_Z_MOTOR] = position
            else:
                if relative:
                    position += self._positions[Motor.PELLET_Z_MOTOR]
                self._positions[Motor.PELLET_Z_MOTOR] = position + 0.00001
            self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return self._is_open

    def move_servo_motor(self, motor: Motor, position: Union[float, Tuple[float, float]]):
        if isinstance(position, tuple):
            position = position[0]
        self._positions[motor] = position
        self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return True

    def move_load_servo(self, position: float, _save: bool = False) -> bool:
        if self._is_open:
            logger.info(f"set load arm {position}")
            if isinstance(position, float) or isinstance(position, int):
                velocity = 100  # config.maximum_velocity
            elif isinstance(position, tuple):
                velocity = float(position[1]) / 100.0 * 100  # config.maximum_velocity
                position = float(position[0])
            self._positions[Motor.PELLET_LOAD_SERVO] = position + 0.00001
            self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return self._is_open

    def retrieve_pellet(self) -> bool:
        if self._is_open:
            logger.info("retrieve pellet")
        return self.move_load_servo(self._configs[Motor.PELLET_LOAD_SERVO].maximum_position)

    def scoop_pellet(self) -> bool:
        if self._is_open:
            logger.info("scoop pellet")
        return self.move_load_servo(self._configs[Motor.PELLET_LOAD_SERVO].minimum_position)

    def move_cover_servo(self, position, _save: bool = False) -> bool:
        if self._is_open:
            logger.info(f"set barrier arm {position}")
            self._positions[Motor.PELLET_COVER_SERVO] = position + 0.00001
            self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return self._is_open

    def release_pellet(self) -> bool:
        if self._is_open:
            logger.info("release pellet")
        return self.move_cover_servo(self._configs[Motor.PELLET_COVER_SERVO].minimum_position)

    def cover_pellet(self) -> bool:
        if self._is_open:
            logger.info("cover pellet")
        return self.move_cover_servo(self._configs[Motor.PELLET_COVER_SERVO].maximum_position)

    def fixed_position(self) -> bool:
        for motor in (Motor.PELLET_X_MOTOR, Motor.PELLET_Y_MOTOR, Motor.PELLET_Z_MOTOR):
            self._positions[motor] = self._send_pos[motor]
        self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return self._is_open

    def emit_tone(self, frequency, duration_ms) -> bool:
        if self._is_open:
            logger.info(f"play tone f={frequency} d={duration_ms}")
            self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return self._is_open

    def get_motor_configuration(self, motor: Motor):
        return self._configs[motor]

    def request_motor_config(self, motor: Motor) -> bool:
        if self._is_open:
            add = self._messages.append
            logger.info(f"request motor config {motor_to_str(motor)}")
            if motor == Motor.PELLET_COVER_SERVO:
                add(self._configs[Motor.PELLET_COVER_SERVO])
            elif motor == Motor.PELLET_LOAD_SERVO:
                add(self._configs[Motor.PELLET_LOAD_SERVO])
            elif motor == Motor.TUNNEL_MAGNET_SERVO:
                add(self._configs[Motor.TUNNEL_MAGNET_SERVO])
            elif motor == Motor.TUNNEL_GATE_SERVO:
                add(self._configs[Motor.TUNNEL_GATE_SERVO])
            elif motor == Motor.PELLET_X_MOTOR:
                add(self._configs[Motor.PELLET_X_MOTOR])
            elif motor == Motor.PELLET_Y_MOTOR:
                add(self._configs[Motor.PELLET_Y_MOTOR])
            elif motor == Motor.PELLET_Z_MOTOR:
                add(self._configs[Motor.PELLET_Z_MOTOR])
            elif motor == Motor.TUNNEL_FAN_SERVO:
                add(self._configs[Motor.TUNNEL_FAN_SERVO])

            self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return self._is_open

    def send_heartbeat(self) -> bool:
        self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return self._is_open

    def set_digital_output(self, gpio: DigitalOutputs, state: bool) -> bool:
        if self._is_open:
            logger.info(f"Set digital output {int(gpio.value)} -> {state}")
            self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return self._is_open

    def set_analog_output(self, channel: AnalogOutputs, millivolts: int) -> bool:
        if self._is_open:
            logger.info(f"Set analog output {int(channel.value)} -> {millivolts}")
            self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return self._is_open

    def set_color_led(self, red_percent: int, green_percent: int, blue_percent: int) -> bool:
        if self._is_open:
            color_led = self._color_led = ColorLed(red=red_percent, green=green_percent, blue=blue_percent)
            logger.info(f"Set color LED ({red_percent}, {green_percent}, {blue_percent})")
            self._messages.append(color_led)
            self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return self._is_open

    def stepper_home(self, motor: Motor):
        self._positions[motor] = 0.0
        self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return self._is_open

    def request_version(self):
        if self._is_open:
            self._version_requested = True
            logger.verbose("request version")
            self._messages.append(Version(Target.PELLET_DEVICE, "Pellet Emulator v0.1.0"))
            self._messages.append(Version(Target.MAGNET_DEVICE, "Magnet Emulator v0.1.0"))
        return self._is_open

    def delay(self, delay):
        if self._is_open:
            time.sleep(float(delay))
            self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return self._is_open

    def servo_attach(self, motor: Motor):
        return self._is_open

    def servo_detach(self, motor: Motor):
        return self._is_open

    def set_tunnel_fan_on(self) -> bool:
        logger.verbose("setting tunnel fan ON")
        self._positions[Motor.TUNNEL_FAN_SERVO] = 100
        self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return True

    def set_tunnel_fan_off(self) -> bool:
        logger.verbose("setting tunnel fan OFF")
        self._positions[Motor.TUNNEL_FAN_SERVO] = 0
        self._messages.append(Acknowledge(uuid=EmulationInterface.next_uuid()))
        return True

    def board_reboot(self, target: Target):
        logger.verbose("board_reboot %s", target)
        return True

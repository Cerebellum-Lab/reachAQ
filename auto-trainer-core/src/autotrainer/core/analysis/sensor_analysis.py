from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, IO

import numpy

from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.project import ProjectInfo, ProjectInterval
from autotrainer.core.perf_monitor import PerfMonitor
from autotrainer.core.observable_object import ObservableObject
from autotrainer.core.video_detection import PresenceDetectionAttrs
from .alarm_detector import AlarmDetector
from .animal_evasion_alarm import AnimalEvasionAlarm
from .animal_thrash_alarm import AnimalThrashAlarm
from .autoclamp_evasion_detector import AutoClampEvasionDetector
from .device_comm_alarm import DeviceCommAlarm
from .presence_in_cage_alarm import PresenceInCageAlarm
from .watchdog_monitor import WatchdogMonitor
from ..message.audio_spectrum_message import AudioSpectrumMessage
from .head_fix_measurement import HeadFixMeasurement
from .audio_spectrum_monitor import AudioSpectrumThrashMonitor
from .headbar_pressure_monitor import HeadbarPressureMonitor
from .load_cell_monitor import LoadCellMonitor
from .load_cell_tare_monitor import LoadCellTareMonitor
from .alarm_monitor import EmergencyAlarmMonitor
from .global_animal_presence_monitor import GlobalAnimalPresenceAlarm
from .external_doors_monitor import ExternalDoorsAlarm
from .pellet_position_monitor import PelletMisplacedDetector
from ..configuration.pellet_misplaced_config import PelletMisplacedDetectorConfiguration
from .auto_tunnel_fan_monitor import AutoTunnelSweepMonitor
from ..configuration.tunnel_sweep_config import AutoTunnelSweepConfiguration
from .system_maintenance_alarm import SystemMaintenanceAlarm
from .system_fault_monitor import SystemFaultAlarm

logger = get_verbose_logger(__name__)


def _disable_alarm_config(config):
    for attr in ("use", "is_emergency_condition", "allow_autoresume_on_cleared"):
        if hasattr(config, attr):
            setattr(config, attr, False)


# TODO: Separate true analysis from data recording to file(s) for post-analysis.
class SensorAnalysis(ObservableObject):

    def __init__(self, *, topcam_presence: Optional[PresenceDetectionAttrs] = None):
        super().__init__()

        self._project_info: Optional[ProjectInfo] = None
        self._interval = ProjectInterval.HOUR
        self._have_new_project_audio = True
        self._have_new_project_record_data = True
        # NB: need one have_new_project bool for each file
        self._filter_invalid_weight_started = False

        # The "monitor" CSV file with the bulk of the sensor data.
        self._record_file = None
        self._current_record_interval = -1
        self._had_write_error = False

        # The audio spectrum file.
        self._audio_record_file_writer: Optional[Tuple[IO[str], csv.DictWriter]] = None
        self._current_audio_record_interval = -1
        self._audio_had_write_error = False

        self._is_load_cell_engaged = False
        self._load_cell_monitor = LoadCellMonitor()

        # The analog pressure sensor on the headbar.
        self._is_headbar_pressure_engaged = False
        self._headbar_pressure_monitor = HeadbarPressureMonitor()

        # The digital i/o switch on the headbar.
        self._is_headbar_switch_engaged = False

        self._tare_detector = LoadCellTareMonitor()
        self._tare_callback = None

        self._audio_thrashing_monitor = AudioSpectrumThrashMonitor()

        self._global_animal_presence_alarm = GlobalAnimalPresenceAlarm(
            load_cell_monitor=self._load_cell_monitor,
            topcam_presence=topcam_presence,
        )

        self._external_doors_alarm = ExternalDoorsAlarm()

        self._device_comm_alarm = DeviceCommAlarm()

        self._pellet_misplaced_monitor = PelletMisplacedDetector()
        self._auto_tunnel_sweep_monitor = AutoTunnelSweepMonitor(
            pellet_misplaced_detector=self._pellet_misplaced_monitor,
        )

        self._watchdog_monitor = WatchdogMonitor()

        self._system_maintenance_alarm = SystemMaintenanceAlarm()
        self._system_fault_alarm = SystemFaultAlarm(watchdog_monitor=self._watchdog_monitor)

        self._animal_thrash_alarm = AnimalThrashAlarm(
            load_cell_detector=self._load_cell_monitor,
            audio_thrash_detector=self._audio_thrashing_monitor,
        )

        self._presence_in_cage_alarm = PresenceInCageAlarm(
            load_cell_monitor=self._load_cell_monitor,
            topcam_presence_attrs=topcam_presence,
        )

        self._alarm_monitor = EmergencyAlarmMonitor()

        self._autoclamp_evasion_detector = AutoClampEvasionDetector(
            loadcell_detector=self._load_cell_monitor,
            headbar_detector=self._headbar_pressure_monitor,
        )
        self._animal_evasion_alarm = AnimalEvasionAlarm()

        self._perf_monitor = PerfMonitor(name="<sensor-analysis>", units="mps", report_window=30)

        alarm_candidates = [
            self._animal_thrash_alarm,
            self._animal_evasion_alarm,
            self._system_maintenance_alarm,
            self._system_fault_alarm,
            self._external_doors_alarm,
            self._global_animal_presence_alarm,
            self._presence_in_cage_alarm,
            self._device_comm_alarm,
        ]

        self._detectors = [
            self._load_cell_monitor,
            self._tare_detector,
            self._audio_thrashing_monitor,
            self._pellet_misplaced_monitor,
            self._auto_tunnel_sweep_monitor,
            self._watchdog_monitor,
            self._autoclamp_evasion_detector,
        ]

        logger.info("Alarm and emergency monitors disabled for reachAQ runtime")
        _disable_alarm_config(self._alarm_monitor.config)
        for alarm in alarm_candidates:
            _disable_alarm_config(alarm.config)
        self._alarms = []

    @property
    def alarms(self) -> List[AlarmDetector]:
        return self._alarms

    @property
    def detectors(self):
        return self._detectors

    def start(self):
        logger.notice("Start requested, starting all ..")
        # ensure new check_update() will be done:
        self._have_new_project_audio = True
        self._have_new_project_record_data = True
        for detector in self._detectors:
            detector.start()

    def stop(self):
        logger.notice("Stop requested, stopping all ..")
        for detector in self._detectors:
            detector.stop()
        self._close_record_file()
        self._close_audio_file()

    def restart(self):
        logger.notice("Restart requested")
        cur_project = self._project_info
        self.project_info = None
        for detector in self._detectors:
            detector.restart()
        self.project_info = cur_project

    @property
    def project_info(self) -> Optional[ProjectInfo]:
        return self._project_info

    @project_info.setter
    def project_info(self, value: ProjectInfo) -> None:
        self._project_info = value
        self._have_new_project_audio = True
        self._have_new_project_record_data = True
        self._perf_monitor.reset()

    @property
    def interval(self) -> ProjectInterval:
        return self._interval

    @interval.setter
    def interval(self, value: ProjectInterval) -> None:
        self._interval = value

    @property
    def load_cell_monitor(self) -> LoadCellMonitor:
        return self._load_cell_monitor

    @property
    def headbar_pressure_monitor(self) -> HeadbarPressureMonitor:
        return self._headbar_pressure_monitor

    @property
    def load_cell_tare_monitor(self) -> LoadCellTareMonitor:
        return self._tare_detector

    @property
    def audio_thrashing_monitor(self) -> AudioSpectrumThrashMonitor:
        return self._audio_thrashing_monitor

    @property
    def animal_thrashing_alarm(self) -> AnimalThrashAlarm:
        return self._animal_thrash_alarm

    @property
    def emergency_alarm_monitor(self) -> EmergencyAlarmMonitor:
        return self._alarm_monitor

    @property
    def presence_in_cage_alarm(self) -> PresenceInCageAlarm:
        return self._presence_in_cage_alarm

    @property
    def global_animal_presence_alarm(self) -> GlobalAnimalPresenceAlarm:
        return self._global_animal_presence_alarm

    @property
    def external_doors_alarm(self) -> ExternalDoorsAlarm:
        return self._external_doors_alarm

    @property
    def device_comm_alarm(self) -> DeviceCommAlarm:
        return self._device_comm_alarm

    @property
    def pellet_misplaced_monitor(self) -> PelletMisplacedDetector:
        return self._pellet_misplaced_monitor

    @property
    def auto_tunnel_sweep_monitor(self) -> AutoTunnelSweepMonitor:
        return self._auto_tunnel_sweep_monitor

    @property
    def system_maintenance_alarm(self) -> SystemMaintenanceAlarm:
        return self._system_maintenance_alarm

    @property
    def system_fault_alarm(self) -> SystemFaultAlarm:
        return self._system_fault_alarm

    @property
    def watchdog_monitor(self) -> WatchdogMonitor:
        return self._watchdog_monitor

    @property
    def autoclamp_evasion_detector(self) -> AutoClampEvasionDetector:
        return self._autoclamp_evasion_detector

    @property
    def animal_evasion_alarm(self) -> AnimalEvasionAlarm:
        return self._animal_evasion_alarm

    #

    @property
    def is_headbar_switch_engaged(self):
        # TODO - this signal is not currently encapsulated in an object for whatever analysis is needed like the others.
        #  If it is used in the future, this should probably be refactored similar to the other sub-components.
        return self._is_headbar_switch_engaged

    def stream_start(self):
        logger.verbose("SensorAnalysis: stream_start")
        self._perf_monitor.reset()

    def measurements_received(
        self,
        measurements: List[HeadFixMeasurement]
    ) -> Tuple[List[float], List[bool], List[float], List[float], List[float]] :
        """Return weight_vals, switch_vals, pressure_vals, temperature_vals, humidity_vals"""
        if len(measurements) == 0:
            raise RuntimeError("Expected non-empty measurements list")

        switch_vals: List[bool] = []
        pressure_vals: List[float] = []
        temperature_vals: List[float] = []
        humidity_vals: List[float] = []

        fh = self._record_file
        needs_update = self._have_new_project_record_data
        if fh is not None:
            now = datetime.now()
            needs_update |= (
                now.hour if self._interval == ProjectInterval.HOUR
                else now.minute
            ) != self._current_record_interval
        if needs_update:
            self._update_record_file()
        fh = self._record_file
        #
        load_cell_mon = self._load_cell_monitor
        # Load cell monitor. and Auto-Tare monitor
        weight_vals: List[float] = []
        filtered_weight_vals: List[float] = []
        load_cell_cfg = load_cell_mon.config

        for m in measurements:
            switch_vals.append(m.switch)
            pressure_vals.append(m.pressure)
            temperature_vals.append(m.temperature)
            humidity_vals.append(m.humidity)

            # NB: save all measurements to file
            if fh is not None:
                try:
                    fh.write(
                        f"{m.when}, {m.timestamp}, {m.weight}, {int(m.switch)}, {m.pressure}, "
                        f"{m.temperature}, {m.humidity}, {int(load_cell_mon.is_engaged)}\n")
                except Exception as err:
                    # This could be too much if something major is wrong.  Just output once per file rotation.
                    if not self._had_write_error:
                        logger.exception("<sensor-analysis>: unable to write: %s", err)
                        self._had_write_error = True
            value = m.weight
            weight_vals.append(value)
            if not (load_cell_cfg.weight_min_filter < value < load_cell_cfg.weight_max_filter):
                filtered_weight_vals.append(math.nan)
                if not self._filter_invalid_weight_started:
                    self._filter_invalid_weight_started = True
                    logger.verbose(
                        "starting filter value outside accepted range: %s", value
                    )
                continue
            if self._filter_invalid_weight_started:
                logger.verbose("finished filter value outside accepted range: %s", value)
                self._filter_invalid_weight_started = False
            #
            filtered_weight_vals.append(value)
            load_cell_mon.update(value, m.when, m.timestamp)

        # (Auto-)tare detection.
        self._tare_detector.update(filtered_weight_vals)

        # Headbar analog pressure monitor.
        first_measure = measurements[0]
        self._headbar_pressure_monitor.update(pressure_vals, first_measure.when, first_measure.timestamp)

        # Headbar digital switch - no real implementation at this time.
        self._is_headbar_switch_engaged = numpy.mean(switch_vals) > 0.5

        # Performance monitoring.
        self._perf_monitor.add_cycles(len(measurements))

        return weight_vals, switch_vals, pressure_vals, temperature_vals, humidity_vals

    def audio_spectrum_received(self, spectrum: AudioSpectrumMessage):
        if spectrum is None or not spectrum.magnitudes:
            logger.debug("audio spectrum empty magnitudes")
            return

        self._audio_thrashing_monitor.update(spectrum.magnitudes, spectrum.when, spectrum.index)

        cur = self._audio_record_file_writer
        needs_update = self._have_new_project_audio
        if cur is not None:
            now = datetime.now()
            needs_update |= (
                now.hour if self._interval == ProjectInterval.HOUR else now.minute
            ) != self._current_audio_record_interval
        if needs_update:
            self._update_audio_file()
        # May or may not exist after the above.
        cur = self._audio_record_file_writer
        if cur is not None:
            fh, writer = cur
            try:
                r = dict(
                    Time=spectrum.when,
                    Index=spectrum.index,
                    **{f"Bin {i}": val for i, val in enumerate(spectrum.magnitudes)},
                )
                writer.writerow(r)
                # fh.flush()
            except Exception as err:
                # This could be too much if something major is wrong.  Just output once per file rotation.
                if not self._audio_had_write_error:
                    logger.exception("audio unable to write: %s", err)
                    self._audio_had_write_error = True

    def _close_record_file(self):
        fh = self._record_file
        if fh is not None:
            self._record_file = None
            try:
                fh.flush()
                fh.close()
            except Exception as err:
                logger.warning("Failure closing record file: %s", err)

    def _update_record_file(self):
        self._have_new_project_record_data = False
        self._close_record_file()
        project = self._project_info
        if project is None:
            return
        interval_file_info = project.get_monitor_file(interval=self._interval, when=datetime.now())
        dest_path = Path(interval_file_info.file)
        try:
            file_existed = dest_path.exists()
            fh = dest_path.open("a")
            if not file_existed:
                fh.write("Time, Index, Weight, Switch, Pressure, Temperature, Humidity, LoadCellEngaged\n")
                fh.flush()
            self._current_record_interval = interval_file_info.current_interval
            self._had_write_error = False
            self._record_file = fh  # last
        except Exception as err:
            logger.error("unable to write to %s: %s",dest_path, err)
        logger.info("saving to %s", dest_path)

    def _close_audio_file(self):
        cur = self._audio_record_file_writer
        if cur is None:
            return
        fh, writer = cur
        self._audio_record_file_writer = None
        logger.debug("closing audio %s", fh.name)
        try:
            fh.flush()
            fh.close()
        except Exception as err:
            logger.warning("audio record file close failed: %s", err)

    def _update_audio_file(self) -> None:
        logger.verbose("updating audio file .. cur = %s", self._audio_record_file_writer)
        self._have_new_project_audio = False
        project = self._project_info
        if project is None:
            self._close_audio_file()
            return
        interval_file_info = project.get_audio_spectrum_file(interval=self._interval, when=datetime.now())
        dest_path = Path(interval_file_info.file)
        cur = self._audio_record_file_writer
        if cur is not None:
            fh, writer = cur
            if fh.name == dest_path.as_posix():
                logger.debug("audio already to %s", fh.name)
                return
            self._close_audio_file()

        try:
            file_existed = dest_path.exists() and dest_path.stat().st_size > 0
            fh = dest_path.open("a")
            writer = csv.DictWriter(
                fh,
                fieldnames=('Time', 'Index', *(f'Bin {i}' for i in range(64))),
            )
            if not file_existed:
                logger.debug("writing audio header to %s", dest_path)
                writer.writeheader()
            logger.info("saving audio spectrum to %s", dest_path)
            self._current_audio_record_interval = interval_file_info.current_interval
            self._audio_had_write_error = False
            self._audio_record_file_writer = fh, writer  # last
        except Exception as err:
            logger.error("unable to write to %r: %s", interval_file_info.file, err)

from autotrainer.api import ApiAlarmKind
from autotrainer.core import get_perf_now
from autotrainer.core.analysis.alarm_detector import AlarmDetector
from autotrainer.core.analysis.audio_spectrum_monitor import AudioSpectrumThrashMonitor
from autotrainer.core.configuration.animal_thrash_config import AnimalThrashAlarmConfig


class AnimalThrashAlarm(AlarmDetector[AnimalThrashAlarmConfig]):

    config_cls = AnimalThrashAlarmConfig
    alarm_api_kind = ApiAlarmKind.thrashing

    def __init__(
        self,
        *,
        audio_thrash_detector: AudioSpectrumThrashMonitor,
    ):
        super().__init__()
        self._audio_thrash_det = audio_thrash_detector
        self._audio_thrash_values = []
        audio_thrash_detector.property_changed += self._on_audio_prop_changed

    def _on_audio_prop_changed(self, name, value, _):
        if not self._running:  # keep check to not append values below
            return
        det = self._audio_thrash_det
        if name == det.IS_ENGAGED:
            with self._lock:
                self._audio_thrash_values.append((get_perf_now(), value,
                                                  det.disengaged_age if value
                                                  else det.engaged_age
                                                  ))
            self.check_state()

    def _expire_audio(self, perf_now):
        cfg = self._config
        values = self._audio_thrash_values
        idx = len(values) - 1
        while idx >= 0:
            t_perf = values[idx][0]
            if perf_now - t_perf > cfg.aggregate_delay:
                del values[:idx + 1]
                break
            idx -= 1

    def _check_state(self):
        perf_now = get_perf_now()
        self._expire_audio(perf_now)
        cfg: AnimalThrashAlarmConfig = self._config
        perf_c_start = perf_now - cfg.aggregate_delay
        count_audio_thrash_triggers = 0  # count
        tot_audio_thrash_engaged = 0  # seconds
        v = None
        for idx, v in enumerate(self._audio_thrash_values):
            if v[1]:
                count_audio_thrash_triggers += 1
            else:
                tot_audio_thrash_engaged += (v[0] - perf_c_start) if idx == 0 else v[2]
        if v is not None:
            if v[1]:
                tot_audio_thrash_engaged += perf_now - v[0]
        elif self._audio_thrash_det.is_engaged:
            tot_audio_thrash_engaged += cfg.aggregate_delay
        #
        pc_audio_thrash = (
            100 * tot_audio_thrash_engaged / cfg.aggregate_delay
        )
        self.is_engaged = (
            pc_audio_thrash >= cfg.audio_thrash_percent_on
            or count_audio_thrash_triggers >= cfg.audio_thrash_count
        )

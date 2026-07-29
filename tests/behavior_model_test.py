import datetime as dtm

import pytest

from tools.acquisition.model.app_model import AppModel

from tools.acquisition.model.app_model_status import AppModelStatus
from top_fixtures import MockSystemMachine


def test_save_config_includes_active_analysis_detectors(behavior_model):
    save = behavior_model.save_configuration
    analysis = behavior_model.analysis
    #
    cfg = analysis.auto_tunnel_sweep_monitor.config
    new = cfg.enabled = not cfg.enabled
    assert save().auto_tunnel_sweep.enabled == new
    #
    thresh = analysis.headbar_pressure_monitor.engaged_threshold
    new = analysis.headbar_pressure_monitor.engaged_threshold = thresh + 5
    assert save().headbar_pressure.threshold == new
    #
    val = analysis.auto_tunnel_sweep_monitor.config.enabled
    new = analysis.auto_tunnel_sweep_monitor.config.enabled = not val
    assert save().auto_tunnel_sweep.enabled == new
    #
    val = analysis.audio_thrashing_monitor.config.threshold_percent
    new = analysis.audio_thrashing_monitor.config.threshold_percent = val + 5
    assert save().audio.threshold_percent == new


class TestEmergency(MockSystemMachine):

    @pytest.fixture(autouse=True)
    def _use_app_model(self, app_model):
        self._app_model = app_model
        self._behavior = app_model.behavior

    def test_emergency_stop_is_disabled(self, app_model):
        algo = app_model.behavior.algorithm
        assert not algo.algo_paused
        assert app_model.behavior.source_emergency is None

        with pytest.raises(RuntimeError, match="Emergency stop is disabled in reachAQ"):
            app_model.behavior.emergency_stop(source="testing")

        assert app_model.behavior.source_emergency is None
        assert not algo.algo_paused

    def test_emergency_resume_is_disabled(self, app_model):
        algo = app_model.behavior.algorithm
        assert not algo.algo_paused
        assert app_model.behavior.source_emergency is None

        with pytest.raises(RuntimeError, match="Emergency resume is disabled in reachAQ"):
            app_model.behavior.emergency_resume(source="testing")

        assert app_model.behavior.source_emergency is None
        assert not algo.algo_paused


mid_day = dtm.datetime(2026,1,1, 12, 0)
mid_night = dtm.datetime(2026, 1, 1, 0, 0)


class TestColorLed:

    @pytest.fixture()
    def app_model(self, app_model) -> AppModel:
        self._app_model = app_model
        app_model.status = AppModelStatus.ACQUIRING  # force
        fault_alarm = self.fault_alarm = app_model.analysis.system_fault_alarm
        # ensure used as emergency condition:
        fault_alarm.config.use = True
        fault_alarm.config.is_emergency_condition = True
        self.led_alarm_cfg = app_model.behavior.algorithm.active_config.led_alarm
        alarm_mon = app_model.analysis.emergency_alarm_monitor
        # force not use daemon, so that below set of is_engaged are all handled in this thread.
        alarm_mon.use_daemon = False
        alarm_mon.restart()
        return app_model

    @property
    def get_color(self):
        return self._app_model.behavior.get_led_color

    def test_start_stop_same_day(self, app_model):
        get_color = self.get_color
        fault_alarm = self.fault_alarm
        #
        assert get_color(now=mid_day) == (0, 0, 0)
        assert get_color(now=mid_night) == (0, 100, 0)
        #
        fault_alarm.is_engaged = True
        assert get_color(now=mid_day) == (0, 0, 0)
        assert get_color(now=mid_night) == (0, 100, 0)
        #
        fault_alarm.config.is_emergency_condition = False
        fault_alarm.property_changed(fault_alarm.CONFIG, fault_alarm.config, None)
        assert get_color(now=mid_day) == (0, 0, 0)
        assert get_color(now=mid_night) == (0, 100, 0)

    def test_start_stop_not_same_day(self, app_model: AppModel):
        get_color = self.get_color
        fault_alarm = self.fault_alarm
        #
        self.led_alarm_cfg.start_ignore_hour = dtm.time(22, 0)
        self.led_alarm_cfg.stop_ignore_hour = dtm.time(10, 0)
        #
        assert get_color(now=mid_night) == (0, 0, 0)
        assert get_color(now=mid_day) == (0, 100, 0)
        #
        fault_alarm.is_engaged = True
        assert get_color(now=mid_day) == (0, 100, 0)
        assert get_color(now=mid_night) == (0, 0, 0)
        #
        fault_alarm.config.is_emergency_condition = False
        fault_alarm.property_changed(fault_alarm.CONFIG, fault_alarm.config, None)
        assert get_color(now=mid_night) == (0, 0, 0)
        assert get_color(now=mid_day) == (0, 100, 0)

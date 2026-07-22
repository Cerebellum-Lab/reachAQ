from pathlib import Path

import pytest

from autotrainer.core import EventManager
from autotrainer.core.configuration.persistence_configuration import PersistenceConfiguration
from autotrainer.behavior.behavior_algorithm import BehaviorAlgoStatus
from tools.acquisition.model.app_model import app_status_to_api_app_mode, app_status_to_behavior_algo_status
from tools.acquisition.model.app_model_status import AppModelStatus

from autotrainer.api import ApiApplicationMode


class TestStatus:

    @pytest.mark.parametrize("app_model_status", list(AppModelStatus))
    def test_it_can_translate_to_api_app_mode(self, app_model_status: AppModelStatus) -> None:
        api_app_mode = app_status_to_api_app_mode(app_model_status)
        assert isinstance(api_app_mode, ApiApplicationMode)

    @pytest.mark.parametrize("app_model_status", list(AppModelStatus))
    def test_it_can_translate_to_behavior_status(self, app_model_status: AppModelStatus):
        algo_status = app_status_to_behavior_algo_status(app_model_status)
        if app_model_status in {AppModelStatus.CALIBRATION_3D, AppModelStatus.CALIBRATION_DCS}:
            assert algo_status is None
        else:
            assert isinstance(algo_status, BehaviorAlgoStatus)
            assert algo_status.name == app_model_status.name


def test_it_drain_record_stop_sema_on_session_recording_start(app_model):
    app_model._record_stop_sema.release()
    app_model._record_stop_sema.release()
    app_model.behavior.algorithm.start_session(reason="manual")
    assert app_model._record_stop_sema.acquire(block=False) is False, "cannot acquire after: it should be back to 0"


def test_startup_project_exists_before_periodic_status_is_published(app_model):
    assert app_model.project is not None
    assert EventManager.default().project is app_model.project


def test_default_output_path_uses_canonical_lowercase_directory():
    assert PersistenceConfiguration.DEFAULT_OUTPUT_PATH == Path("~/Documents/rawdatalocal")

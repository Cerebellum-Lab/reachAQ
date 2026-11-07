import contextlib
import datetime
import logging
from unittest import mock

import pytest

from autotrainer.core import ProjectInfo
from transitions import MachineError

from autotrainer.behavior import SegmentationConfiguration, DetectionConfiguration, SystemState
from autotrainer.behavior.intersession import IntersessionState
from autotrainer.inference.analysis import IntersessionResponse

from top_fixtures import MockSystemMachine


def test_intersession(
    mock_system,
    machine,
):
    intersession = machine.intersession

    project = machine.project
    assert project is not None
    assert intersession.state == IntersessionState.idle

    with pytest.raises(MachineError):
        seg_cfg = SegmentationConfiguration(
            project=project,
            complete=lambda s: 1 / 0,  # noqa
        )
        intersession.perform_detection(seg_cfg)

    machine.state = SystemState.intersession

    with mock_system.mock_perform_segmentation() as m_perf_segm:
        intersession.perform_segmentation(project)

    segment_cfg = intersession._segmentation_configuration

    assert m_perf_segm.call_args_list == [
        mock.call(segment_cfg)
    ]

    assert intersession.state == IntersessionState.segmentation

    with mock_system.mock_perform_detection() as m_perf_detect:
        segment_cfg.complete(True)

    assert intersession.state == IntersessionState.detection

    detection_cfg = intersession._detection_configuration
    assert m_perf_detect.call_args_list == [
        mock.call(detection_cfg)
    ]

    assert intersession.state == IntersessionState.detection

    detection_cfg.complete(True)

    assert intersession.state == IntersessionState.idle

    assert mock_system.intersession_state_trans == [
        IntersessionState.segmentation,
        IntersessionState.detection,
        IntersessionState.idle,
    ]


def test_intersession_increase_algo_counts(mock_system):
    algo = mock_system.algo
    algo.intersession_enabled = True
    mock_system.mock_pose_response(pellet_seen=True)
    mock_system.start_recording_session(set_recording_status=True)
    assert algo.is_in_session
    algo.update_mouse_seen(True)
    res = IntersessionResponse(
        pellets_presented=4,
        total_reaches=3,
        food_consumed=2,
        successful_reaches=1,
    )
    with mock_system.mock_intersession_analysis(results=res):
        mock_system.stop_recording_session()
    assert algo.pellets_presented == 0  # pellet-sent owns this count
    assert algo.pellet_reaches == 3
    assert algo.pellets_consumed == 2
    assert algo.successful_reaches == 1


def test_stop_recording_session_when_analysis_ongoing(mock_system, machine, caplog):
    algo = mock_system.algo
    algo.intersession_enabled = True
    def verify_analysis_blocks_exit():
        assert machine.state == SystemState.intersession
        mock_system.stop_recording_session()
        assert machine.state == SystemState.intersession

    mock_system.start_recording_session()
    mock_system.mock_pose_response(pellet_seen=True, mouse_seen=True, triangle_seen=True)

    with caplog.at_level(logging.DEBUG):
        with mock_system.mock_intersession_analysis(concurrent_func=verify_analysis_blocks_exit):
            assert machine.state == SystemState.ready
            mock_system.stop_recording_session()
            assert machine.state == SystemState.intersession

    assert machine.state == SystemState.ready, "Must be ready after intersession analysis"

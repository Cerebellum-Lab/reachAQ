import logging
import math
from unittest import mock

import pytest

import autotrainer.behavior.pellet.pellet_machine
from autotrainer.behavior import SystemState, PelletDeviceProtocol, BehaviorAlgorithm, \
    SystemMachine
from autotrainer.behavior.pellet import PelletState
from autotrainer.behavior.pellet.pellet_machine import PelletDeviceCommandFailed

from top_fixtures import MockSystemMachine, mock_system, get_current_simulate_perf_now, increase_simulate_perf_now


@pytest.mark.parametrize("cover_enabled", [False, True])
def test_cover_or_release_pellet_on_load_pellet(mock_system, machine, cover_enabled):
    pellet_m = machine.pellet
    algo = machine.algorithm

    algo.pellet_cover_enabled = cover_enabled
    pellet_m._covered_state = cover_enabled  # start already covered or released as desired

    # for start:
    pellet_m.state = PelletState.retract  # start from retract
    algo.update_pellet_seen(True)
    algo.update_triangle_seen(True)

    assert machine.state == SystemState.ready
    assert not algo.is_in_session
    assert algo.pellet_recently_seen
    # assert pellet_m.state == PelletState.monitoring

    # 1st pellet missing:
    mock_system.mock_pose_response(pellet_seen=False)
    assert algo.session_pellet_loaded_count == 0
    assert pellet_m.state == PelletState.retract  # still
    assert algo.pellet_recently_seen  # still
    # now:
    mock_system.increment_perf_now(algo.pellet_missing_time)
    assert not algo.pellet_recently_seen  # now not recently seen
    mock_system.mock_pose_response(pellet_seen=False)
    assert not algo.pellet_recently_seen  # still ofc
    assert algo.session_pellet_loaded_count == 0
    assert pellet_m.state == PelletState.loading
    mock_system.mock_pose_response(pellet_seen=True)
    assert algo.pellet_recently_seen  # back !
    assert algo.session_pellet_loaded_count == 0
    mock_system.mock_pellet_ack()  # ack the load
    assert algo.session_pellet_loaded_count == 1  # here it is !
    mock_system.start_recording_session(set_recording_status=True)
    assert algo.session_pellet_loaded_count == 0  # back to 0 given new session/trial
    mock_system.mock_pose_response(pellet_seen=True)
    mock_system.mock_pellet_ack()  # ack the retract
    assert pellet_m.state == PelletState.sending
    mock_system.mock_pose_response(pellet_seen=True)
    mock_system.mock_pellet_ack()  # ack the sending
    assert pellet_m.state == PelletState.monitoring
    assert pellet_m.can_use_pellet_command()
    exp = [
        PelletState.retract,
        PelletState.loading,
    ]
    if cover_enabled:
        exp += [PelletState.covering, PelletState.retract]
    else:
        exp += [PelletState.retract, PelletState.releasing]
    exp += [PelletState.sending, PelletState.monitoring]
    assert mock_system.pellet_state_trans == exp
    mock_system.mock_pellet_ack()  # ack the send
    assert algo.can_cover_pellet() is (True if cover_enabled else False)
    assert algo.can_release_pellet() is (False if cover_enabled else True)
    mock_system.mock_pose_response(pellet_seen=True)
    assert algo.session_pellet_loaded_count == 0
    mock_system.mock_pose_response(pellet_seen=True)
    assert algo.session_pellet_loaded_count == 0  # still ofc, no new load-pellet


@pytest.mark.parametrize("cover_enabled", [False, True])
def test_send_pellet_after_load_when_triangle_not_seen(mock_system, machine, cover_enabled):
    pellet_m = machine.pellet
    algo = machine.algorithm

    algo.pellet_cover_enabled = cover_enabled
    pellet_m._covered_state = cover_enabled  # fake start already covered or released as desired

    # for start:
    algo.update_pellet_seen(True)
    algo.update_triangle_seen(True)

    assert machine.state == SystemState.ready
    assert not algo.is_in_session
    assert algo.pellet_recently_seen
    # A missing pellet should trigger a load/cover cycle while no session is recording.
    mock_system.mock_pose_response(pellet_seen=False, triangle_seen=True)
    mock_system.mock_pellet_ack(until_none=True)
    assert algo.session_pellet_loaded_count == 0
    assert pellet_m.state == PelletState.retract  # still
    assert algo.pellet_recently_seen  # still
    assert algo.triangle_recently_seen

    mock_system.increment_perf_now(algo.pellet_missing_time / 2)
    mock_system.mock_pose_response(pellet_seen=False, triangle_seen=True)
    assert algo.pellet_recently_seen  # still
    assert pellet_m.state == PelletState.retract  # still

    mock_system.increment_perf_now(algo.pellet_missing_time / 2 + 1e-3)
    mock_system.mock_pose_response(pellet_seen=False, triangle_seen=False)
    assert pellet_m.state == PelletState.loading  # Now
    assert not algo.pellet_recently_seen  # now not recently seen
    assert algo.triangle_recently_seen  # but triangle yes
    mock_system.increment_perf_now(algo.pellet_missing_time)
    assert not algo.triangle_recently_seen  # and triangle snot anymore seen as well
    assert pellet_m.state == PelletState.loading  # still
    assert pellet_m._api_status_token is not None  # must be acked
    mock_system.mock_pose_response(pellet_seen=True, triangle_seen=True)
    mock_system.mock_pellet_ack(until_none=True)  # ack the cover
    # assert not algo.triangle_recently_seen  # still ofc
    assert algo.session_pellet_loaded_count == 1
    expected = [
        PelletState.retract,
        PelletState.loading,
    ]
    if cover_enabled:
        expected += [PelletState.covering, PelletState.retract]
    else:
        expected += [PelletState.retract, PelletState.releasing, PelletState.retract]
    assert mock_system.pellet_state_trans == expected
    #
    mock_system.mock_pose_response(pellet_seen=True, triangle_seen=False)
    assert algo.session_pellet_loaded_count == 1
    assert pellet_m.state == PelletState.retract


def test_uncover_when_record_aged_enough_with_no_pellet_hand_uncover_distance(mock_system, machine):
    pellet_m = machine.pellet
    algo = machine.algorithm
    algo.pellet_cover_enabled = True
    uncov_cfg = algo.active_config.pellet_uncover
    # algo.pellet_hand_uncover_distance = None
    uncov_cfg.min_y_dcs = -math.inf
    uncov_cfg.trigger_delay = 0
    pellet_m._covered_state = True  # fake already covered to simplify test
    #
    algo.update_pellet_seen(True)
    algo.update_triangle_seen(True)
    assert pellet_m.state == PelletState.monitoring
    assert not algo.is_in_session

    algo.start_session(reason="manual")
    assert algo.is_in_session
    assert mock_system.pellet_state_trans == []

    mock_system.make_recording_aged_enough()
    algo.project.first_pellet_delivery_offset = 0

    uncov_ctx = algo.uncover_context
    uncov_ctx.y_dcs_valid = True
    uncov_ctx.start_y_dcs_valid_perf_c = get_current_simulate_perf_now()
    increase_simulate_perf_now(algo.active_config.pellet_uncover.trigger_delay)
    algo.update_pellet_seen(True)  # must still be present before env changed
    pellet_m.environment_changed()

    assert pellet_m.state == PelletState.monitoring
    # now:
    assert mock_system.pellet_state_trans == [
        PelletState.releasing,
        PelletState.monitoring,
    ]
    assert pellet_m._api_status_token is not None, "we wait the release"
    assert pellet_m._api_status_token is pellet_m._token_release_pellet


def test_uncover_when_hands_near_pellet_after_recording_aged_enough(mock_system, machine):
    pellet_m = machine.pellet
    algo = machine.algorithm
    algo.pellet_cover_enabled = True
    uncov_cfg = algo.active_config.pellet_uncover
    uncov_cfg.min_y_dcs = 5
    uncov_cfg.trigger_delay = 2.5
    algo.update_pellet_seen(True)
    algo.update_triangle_seen(True)
    pellet_m._covered_state = True  # fake already covered to simplify test
    algo.start_session(reason="manual")
    assert algo.is_in_session
    #
    mock_system.make_recording_aged_enough()
    pellet_m.environment_changed()
    assert pellet_m.state == PelletState.monitoring
    assert mock_system.pellet_state_trans == [], "contrary to test_uncover_when_record_aged_enough"
    #
    # now make pellet-hands-min distance trigger:
    algo.pellet_hands_min_distance = algo.pellet_uncover_y_dcs / 2
    algo.uncover_context.y_dcs_valid = True
    algo.uncover_context.start_y_dcs_valid_perf_c = get_current_simulate_perf_now()
    algo.update_pellet_seen(True)
    pellet_m.environment_changed()
    #
    assert mock_system.pellet_state_trans == []  # not yet
    #
    increase_simulate_perf_now(uncov_cfg.trigger_delay)
    algo.update_pellet_seen(True)
    pellet_m.environment_changed()
    # and :
    assert pellet_m.state == PelletState.monitoring
    assert mock_system.pellet_state_trans == [
        PelletState.releasing,
        PelletState.monitoring,
    ]
    assert pellet_m._api_status_token is not None, "we wait the release ack"
    assert pellet_m._api_status_token is pellet_m._token_release_pellet, "current token must the one of release pellet"


@pytest.mark.skipif(True, reason="Session limits and associated logic currently on hold")
def test_session_limit(mock_system, machine):
    # TODO: Session limits and associated logic currently on hold.
    pass


@pytest.mark.skipif(True, reason="Disabled until day limit is implemented via reach detection.")
def test_day_limit(machine, mock_system):
    # TODO
    pass


@pytest.mark.parametrize("trigger_count", [0, 1, 3])
def test_force_home_with_load_retract_count_triggered(machine, mock_system, monkeypatch, trigger_count, caplog):
    pellet_m = machine.pellet
    algo = machine.algorithm
    monkeypatch.setattr(autotrainer.behavior.pellet.pellet_machine,
                        "DEFAULT_LOAD_RETRACT_COUNT_FORCE_HOME", trigger_count)
    algo.pellet_cover_enabled = True
    pellet_m.state = PelletState.monitoring
    # before:
    assert machine._pellet_device.send_home.call_args_list == []
    caplog.set_level(logging.INFO)
    #
    for _ in range(trigger_count - 1):
        pellet_m.load_pellet(force=True)
        mock_system.mock_pellet_ack(until_none=True)
    pellet_m.load_pellet(force=True)
    mock_system.mock_pellet_ack(until_none=True)
    pellet_m.send_pellet(force=True)
    if trigger_count > 0:
        assert machine._pellet_device.send_home.call_args_list == [mock.call()]
        assert f'Forcing a send_home to reset to limits due to load + retract count greater-or-equal than threshold: {2 * trigger_count} vs {trigger_count}' in caplog.text
    else:
        assert machine._pellet_device.send_home.call_args_list == []
        assert "Forcing a send_home to reset to limits" not in caplog.text, "when 0 it's disabled"


def test_move_home(machine, mock_system):
    pellet_m = machine.pellet
    algo = machine.algorithm
    algo.update_pellet_seen(True)
    assert pellet_m.state == PelletState.monitoring
    pellet_m.move_home()
    assert pellet_m._api_status_token is not None
    assert pellet_m.state == PelletState.home
    mock_system.mock_pellet_ack()
    mock_system.start_recording_session(set_recording_status=True)
    assert algo.is_in_session
    mock_system.mock_pellet_ack()
    mock_system.mock_pose_response(pellet_seen=True)
    assert pellet_m.state == PelletState.sending
    mock_system.mock_pellet_ack()
    assert pellet_m.state == PelletState.monitoring
    assert mock_system.pellet_state_trans == [
        PelletState.home,
        PelletState.covering,
        PelletState.retract,
        PelletState.sending,
        PelletState.monitoring,
    ]
    assert pellet_m._api_status_token is None


def test_can_cover_pellet_in_monitoring(machine, mock_system):
    pellet_m = machine.pellet
    algo = machine.algorithm
    #
    algo.pellet_cover_enabled = False
    assert pellet_m.can_cover_pellet() is False
    #
    algo.pellet_cover_enabled = True
    assert pellet_m.can_cover_pellet() is True
    assert pellet_m._api_status_token is None
    assert pellet_m.covered_state is None  # unknown on start
    assert pellet_m.state == PelletState.monitoring
    #
    pellet_m.cover_pellet()
    assert pellet_m.covered_state is True
    assert pellet_m._api_status_token is not None
    assert mock_system.pellet_state_trans == [
        PelletState.covering,
    ]
    # ensure pellet-seen:
    algo.update_pellet_seen(True)
    algo.update_triangle_seen(True)
    # otherwise there would be a load-pellet executed.
    mock_system.mock_pellet_ack(until_none=True)
    #
    assert mock_system.pellet_state_trans == [
        PelletState.covering,
        PelletState.retract,
    ], "should be now covered + retract given pellet seen"
    assert pellet_m._api_status_token is None


def test_release_pellet(machine, mock_system):
    pellet_m = machine.pellet
    algo = machine.algorithm
    #
    algo.pellet_cover_enabled = False
    assert algo.can_release_pellet() is True
    assert pellet_m.covered_state is None  # on start
    assert pellet_m.state == PelletState.monitoring
    #
    pellet_m.release_pellet()
    assert pellet_m.covered_state is False
    assert pellet_m._api_status_token is not None
    assert mock_system.pellet_state_trans == [PelletState.releasing]
    algo.update_pellet_seen(True)
    algo.update_triangle_seen(True)
    mock_system.mock_pellet_ack()
    assert mock_system.pellet_state_trans == [
        PelletState.releasing, PelletState.retract,
    ]
    mock_system.mock_pellet_ack()
    assert pellet_m._api_status_token is None


@pytest.mark.parametrize("pellet_dev_method", [
    "send_pellet",
    "load_pellet",
    ("move_home", "send_home"),
    ("move_retract", "send_retract"),
    "cover_pellet",
    "release_pellet"
])
def test_on_device_command_failed_get_exception(machine, mock_system, pellet_dev_method):
    algo = machine.algorithm
    pellet_m = machine.pellet
    pellet_dev = pellet_m._pellet_device
    if isinstance(pellet_dev_method, str):
        action = getattr(pellet_m, pellet_dev_method)
        dev_meth = getattr(pellet_dev, pellet_dev_method)
    else:
        action = getattr(pellet_m, pellet_dev_method[0])
        dev_meth = getattr(pellet_dev, pellet_dev_method[1])
    pellet_m.state = PelletState.monitoring  # is normally accepted for all actions
    dev_meth.return_value = None
    algo.update_triangle_seen(True)  # need triangle seen for can_load_pellet
    machine.algorithm.pellet_cover_enabled = action != pellet_m.release_pellet  # need for release_pellet
    with pytest.raises(PelletDeviceCommandFailed):
        if action == pellet_m.send_pellet:
            kw = dict(force=True)
        else:
            kw = {}
        action(**kw)


@pytest.mark.parametrize("before_state", list(PelletState))
@pytest.mark.parametrize("cover_enabled", [False, True])
def test_manual_send_pellet(machine, mock_system, before_state, cover_enabled):
    algo = machine.algorithm
    algo.pellet_delivery_enabled = True  # otherwise cannot send_pellet
    algo.pellet_cover_enabled = cover_enabled
    pellet_m = machine.pellet
    # pellet_m._covered_state = cover_enabled
    #
    pellet_m.state = before_state
    mock_system.pellet_state_trans.clear()
    #
    pellet_m.send_pellet(force=True)
    assert pellet_m._api_status_token is not None
    assert pellet_m.state == PelletState.sending
    mock_system.mock_pellet_ack()
    assert pellet_m._api_status_token is None
    assert pellet_m.state == PelletState.monitoring
    cover_state = PelletState.covering if cover_enabled else PelletState.releasing
    assert mock_system.pellet_state_trans == [
        *((cover_state,) if before_state != cover_state else ()),
        PelletState.sending,
        PelletState.monitoring,
    ]


@pytest.mark.parametrize("before_state", list(PelletState))
def test_manual_send_pellet_when_delivery_not_enabled(machine, mock_system, before_state):
    algo = machine.algorithm
    algo.pellet_delivery_enabled = False
    pellet_m = machine.pellet
    assert mock_system.pellet_state_trans == []
    #
    pellet_m.state = before_state
    mock_system.pellet_state_trans.clear()

    assert mock_system.pellet_state_trans == []
    pellet_m.send_pellet()
    assert pellet_m._api_status_token is None
    assert pellet_m.state == before_state
    assert mock_system.pellet_state_trans == []


def test_protocol_send_guard_blocks_before_hardware_dispatch(machine):
    pellet_m = machine.pellet
    pellet_m.set_send_preparation_callbacks(
        guard=lambda: (_ for _ in ()).throw(RuntimeError("not prepared")),
    )

    with pytest.raises(RuntimeError, match="not prepared"):
        pellet_m.send_pellet(force=True)

    assert pellet_m._token_pellet_send is None


def test_protocol_readiness_can_defer_automatic_send(machine, monkeypatch):
    pellet_m = machine.pellet
    monkeypatch.setattr(machine.algorithm, "can_send_pellet", lambda: True)
    pellet_m.set_send_preparation_callbacks(readiness=lambda: False)

    assert pellet_m.can_send_pellet() is False
    assert pellet_m.can_send_pellet(force=True) is True


def test_prepared_keep_current_cover_policy_is_not_overridden(machine, mock_system):
    pellet_m = machine.pellet
    machine.algorithm.pellet_cover_enabled = True
    pellet_m._covered_state = False
    pellet_m.prepare_cover_policy("keep_current")
    mock_system.pellet_state_trans.clear()

    pellet_m.send_pellet(force=True)

    assert mock_system.pellet_state_trans == [PelletState.sending]

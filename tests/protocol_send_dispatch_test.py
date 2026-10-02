"""A protocol trial's SEND, dispatched off the Qt thread, through the real relay.

christielab10, 2026-10-02: every Send of a recorded protocol session failed with
"Pellet SEND was not accepted after preparation". The pellet machine's triggers
are relayed to the behaviour algorithm's thread without waiting
(relay_transitions(wait=False)), and the relay returns nothing, so the dispatch
saw a still-prepared trial and failed it; the algorithm thread then ran the
SEND and its guard refused it ("Pellet SEND requires Prepared state, found
failed"). The test suite runs the algorithm in place (_no_handler_thread),
which hides this, so these tests start the real handler thread.
"""
import threading
from types import SimpleNamespace

import pytest

from autotrainer.behavior.behavior_algorithm import BehaviorAlgorithm
from tools.acquisition.model.app_model import AppModel
from tools.acquisition.model.trial_action import PreparedState


@pytest.fixture
def algorithm_thread(monkeypatch):
    """The behaviour algorithm's own handler thread, as the app runs it."""
    monkeypatch.setattr(BehaviorAlgorithm, "_no_handler_thread", False)
    monkeypatch.setattr(
        BehaviorAlgorithm, "_handler_thread_queue", (threading.main_thread(), None, [])
    )
    BehaviorAlgorithm._check_start_thread(thread_lock=threading.RLock())
    handler_thread = BehaviorAlgorithm._handler_thread_queue[0]
    assert handler_thread.name == "AlgoHandler" and handler_thread.is_alive()
    yield handler_thread
    BehaviorAlgorithm.close_algorithm_handler()


class _Executor:
    def __init__(self):
        self.operation = SimpleNamespace(state=PreparedState.PREPARED)
        self.failures = []

    def prepare(self, _recipe):
        return self.operation

    def fail(self, error):
        self.failures.append(str(error))
        self.operation.state = PreparedState.FAILED

    def cancel(self, **_kwargs):
        self.operation.state = PreparedState.CANCELLED


def _dispatcher(accept):
    """The attributes _prepare_and_dispatch_protocol_send reads, and a SEND
    relayed exactly as the pellet machine's triggers are."""
    executor = _Executor()
    ran_on = []

    def send_trigger(force=False):
        ran_on.append(threading.current_thread().name)
        # What _before_send_pellet -> pellet_sending -> bind_send does when
        # the SEND is taken; a refused trigger leaves the trial prepared.
        if accept and executor.operation.state is PreparedState.PREPARED:
            executor.operation.state = PreparedState.SEND_ACCEPTED

    pellet = SimpleNamespace(
        send_pellet=BehaviorAlgorithm.relay_func(send_trigger, wait=False),
        cancel_prepared_cover_policy=lambda: None,
    )
    errors = []
    model = SimpleNamespace(
        _compile_protocol_trial=lambda token, trial_id: object(),
        _trial_action_executor=executor,
        _recording_session=SimpleNamespace(is_current=lambda token, statuses: True),
        _behavior=SimpleNamespace(system_machine=SimpleNamespace(pellet=pellet)),
        on_error=lambda title, message: errors.append((title, message)),
        _protocol_action_lock=threading.Lock(),
        _protocol_action_thread=None,
        _notify_trial_protocol_state=lambda: None,
    )
    return model, executor, errors, ran_on


def _dispatch(model):
    # Off the Qt thread, as request_pellet_send starts it, and bounded, so a
    # deadlock fails the test instead of hanging the suite.
    thread = threading.Thread(
        target=AppModel._prepare_and_dispatch_protocol_send,
        args=(model, object(), 3, True),
        name="PreparePelletTrial-3",
    )
    thread.start()
    thread.join(5.0)
    assert not thread.is_alive(), "the dispatch did not return within 5 s"


def test_an_accepted_protocol_send_is_not_failed(algorithm_thread):
    model, executor, errors, ran_on = _dispatcher(accept=True)
    _dispatch(model)
    assert ran_on == ["AlgoHandler"]
    assert executor.operation.state is PreparedState.SEND_ACCEPTED
    assert executor.failures == []
    assert errors == []


def test_a_refused_protocol_send_is_still_failed_and_reported(algorithm_thread):
    model, executor, errors, ran_on = _dispatcher(accept=False)
    _dispatch(model)
    assert ran_on == ["AlgoHandler"]
    assert executor.operation.state is PreparedState.FAILED
    assert executor.failures == ["pellet state machine did not accept SEND"]
    assert [title for title, _ in errors] == ["Pellet trial was not sent"]

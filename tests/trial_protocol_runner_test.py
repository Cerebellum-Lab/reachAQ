from types import SimpleNamespace
from unittest import mock

from autotrainer.behavior.pellet_trial import TrialOutcome
from autotrainer.training.training_phase import SessionResult

from tools.acquisition.model.trial_protocol_runner import TrialProtocolRunner


class Signal:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def __isub__(self, handler):
        self.handlers.remove(handler)
        return self


class FakeProgress:
    def __init__(self):
        self.session_count = 0
        self.resumed = 0
        self.paused = 0

    def progress_resumed(self):
        self.resumed += 1

    def progress_paused(self):
        self.paused += 1


class FakePhase:
    def __init__(self, result=SessionResult.NONE):
        self.progress = FakeProgress()
        self.result = result
        self.capture_end_count = 0
        self.session_end_count = 0

    def capture_ended(self):
        self.capture_end_count += 1
        self.progress.progress_paused()

    def session_ended(self, _context):
        self.session_end_count += 1
        return self.result


class FakePlan:
    def __init__(self, phase):
        self.current_phase = phase
        self._system_context = SimpleNamespace(capture_analysis_result=None)
        self._on_session_starting = lambda: None
        self._on_session_capture_ending = lambda reason: None
        self._on_session_ending = lambda project, result: None
        self.can_advance = False
        self.can_fallback = False
        self.advance = mock.Mock(return_value=True)
        self.fallback = mock.Mock(return_value=True)
        self.progress_state = None
        self.progress_updates = 0
        self.detached = False

    def attach(self, algorithm, _pellet_device, tunnel_device):
        assert tunnel_device is None
        self.algorithm = algorithm
        algorithm.session_starting += self._on_session_starting
        algorithm.session_capture_ending += self._on_session_capture_ending
        algorithm.session_ending += self._on_session_ending

    def detach(self):
        self.algorithm.session_starting -= self._on_session_starting
        self.algorithm.session_capture_ending -= self._on_session_capture_ending
        self.algorithm.session_ending -= self._on_session_ending
        self.detached = True

    def _on_progress_updated(self):
        self.progress_updates += 1


def make_algorithm():
    return SimpleNamespace(
        session_starting=Signal(),
        session_capture_ending=Signal(),
        session_ending=Signal(),
    )


def test_completed_pellet_trial_advances_progress_once():
    phase = FakePhase()
    plan = FakePlan(phase)
    algorithm = make_algorithm()
    runner = TrialProtocolRunner()
    runner.attach(plan, algorithm, object())

    assert algorithm.session_starting.handlers == []
    assert runner.begin_trial("10.1")
    assert not runner.begin_trial("10.1")
    assert runner.finish_trial("10.1", TrialOutcome.PENDING_ANALYSIS)
    assert not runner.finish_trial("10.1", TrialOutcome.PENDING_ANALYSIS)
    assert phase.progress.session_count == 1
    assert phase.capture_end_count == 1
    assert phase.session_end_count == 1
    assert plan.progress_updates == 1


def test_hardware_error_never_counts_or_evaluates_protocol_trial():
    phase = FakePhase()
    plan = FakePlan(phase)
    runner = TrialProtocolRunner()
    runner.attach(plan, make_algorithm(), object())
    runner.begin_trial("3.2")

    assert not runner.finish_trial("3.2", TrialOutcome.HARDWARE_ERROR)
    assert phase.progress.session_count == 0
    assert phase.session_end_count == 0


def test_terminal_advance_marks_protocol_complete():
    completed = mock.Mock()
    phase = FakePhase(SessionResult.ADVANCE)
    plan = FakePlan(phase)
    runner = TrialProtocolRunner(
        automatic_advance=True,
        on_protocol_complete=completed,
    )
    runner.attach(plan, make_algorithm(), object())
    runner.begin_trial("7.1")

    assert runner.finish_trial("7.1", TrialOutcome.SUCCESS)
    assert runner.protocol_complete
    completed.assert_called_once_with()


def test_detach_restores_callbacks_expected_by_legacy_plan():
    phase = FakePhase()
    plan = FakePlan(phase)
    algorithm = make_algorithm()
    runner = TrialProtocolRunner()
    runner.attach(plan, algorithm, object())

    runner.detach()

    assert plan.detached
    assert algorithm.session_starting.handlers == []

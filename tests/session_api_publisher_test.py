from unittest import mock

from autotrainer.api import ApiEventKind
from autotrainer.behavior import PelletTrialLedger, TrialOutcome

from tools.acquisition.model.session_api_publisher import SessionApiPublisher


def test_session_and_pellet_attempt_lifecycle_is_balanced_exactly_once():
    events = mock.Mock()
    publisher = SessionApiPublisher(events)
    ledger = PelletTrialLedger("session001")
    attempt = ledger.begin_send(1.0, 11.0, operation_id="send-1")

    assert publisher.session_started("session001")
    assert not publisher.session_started("session001")
    assert publisher.trial_started(attempt)
    assert not publisher.trial_started(attempt)
    ledger.acknowledge_presentation(1.1, 11.1)
    pending = ledger.close_active_for_analysis(2.0, 12.0)
    assert publisher.trial_capture_ended(pending)
    assert not publisher.trial_capture_ended(pending)
    final = ledger.finalize_pending(
        1, 1, TrialOutcome.SUCCESS, 2.1, 12.1
    )
    assert publisher.trial_ended(final)
    assert not publisher.trial_ended(final)
    assert publisher.session_ended(ledger.summary())
    assert not publisher.session_ended(ledger.summary())

    kinds = [call.args[0] for call in events.post_event_content.call_args_list]
    assert kinds == [
        ApiEventKind.sessionStarted,
        ApiEventKind.trialStarted,
        ApiEventKind.trialCaptureEnded,
        ApiEventKind.trialEnded,
        ApiEventKind.sessionEnded,
    ]


def test_trial_event_can_open_session_if_send_wins_camera_status_race():
    events = mock.Mock()
    publisher = SessionApiPublisher(events)
    attempt = PelletTrialLedger("session002").begin_send(
        1.0, 11.0, operation_id="send-1"
    )

    publisher.trial_started(attempt)

    kinds = [call.args[0] for call in events.post_event_content.call_args_list]
    assert kinds == [ApiEventKind.sessionStarted, ApiEventKind.trialStarted]


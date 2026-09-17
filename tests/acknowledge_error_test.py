"""A board that refuses a command acknowledges it with an errno.

The acknowledgement payload carries an int32 error, and the board sets it when
it refuses: -EPERM for a pulse aimed at a tone confirmation line, -EBUSY for a
second pulse while one is in flight, -EINVAL for a duration out of range,
-ENOTSUP for a board with no pulse counter. Discarding it makes a refusal look
like a success.
"""

import pytest

from autotrainer.device.device_interface import Acknowledge


def test_an_acknowledgement_defaults_to_success():
    assert Acknowledge(uuid=4).error == 0


def test_an_acknowledgement_carries_a_refusal_errno():
    assert Acknowledge(uuid=4, error=-1).error == -1


def test_a_successful_acknowledgement_is_not_a_failure():
    assert Acknowledge(uuid=4, error=0).is_failure is False


def test_a_refusal_is_a_failure():
    assert Acknowledge(uuid=4, error=-1).is_failure is True


def test_the_failure_reason_names_the_errno():
    reason = Acknowledge(uuid=4, error=-1).failure_reason

    assert "-1" in reason
    assert "refused" in reason.lower()


def test_a_successful_acknowledgement_has_no_reason():
    assert Acknowledge(uuid=4, error=0).failure_reason == ""

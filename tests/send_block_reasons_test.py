"""Tests for holding the pellet send under more than one reason at a time.

The failure this prevents is specific: intertrial analysis and inter-trial
timing both want to hold the send, and `pellet_send_block_reason` is one
string. Sharing it directly means one subsystem clearing its hold releases a
pellet the other still wants held.
"""

from tools.acquisition.model.send_block_reasons import SendBlockReasons


def test_no_reasons_means_no_block():
    reasons = SendBlockReasons()
    assert reasons.is_blocked is False
    assert reasons.render() == ""


def test_one_reason_renders_as_itself():
    reasons = SendBlockReasons()
    reasons.set("analysis", "waiting for retried pellet trial analysis")
    assert reasons.is_blocked is True
    assert reasons.render() == "waiting for retried pellet trial analysis"


def test_clearing_one_reason_leaves_the_other_holding():
    """The whole point: releasing one hold must not release the send."""
    reasons = SendBlockReasons()
    reasons.set("analysis", "waiting for analysis")
    reasons.set("intertrial", "waiting 1.2 s for the inter-trial target")

    reasons.clear("analysis")

    assert reasons.is_blocked is True
    assert reasons.render() == "waiting 1.2 s for the inter-trial target"


def test_clearing_the_last_reason_releases_the_send():
    reasons = SendBlockReasons()
    reasons.set("intertrial", "waiting")
    reasons.clear("intertrial")
    assert reasons.is_blocked is False
    assert reasons.render() == ""


def test_setting_an_empty_detail_clears_that_reason():
    reasons = SendBlockReasons()
    reasons.set("intertrial", "waiting")
    reasons.set("intertrial", "")
    assert reasons.is_blocked is False


def test_rendering_is_stable_regardless_of_insertion_order():
    """The same holds must read the same way between sessions."""
    first = SendBlockReasons()
    first.set("analysis", "A")
    first.set("intertrial", "B")

    second = SendBlockReasons()
    second.set("intertrial", "B")
    second.set("analysis", "A")

    assert first.render() == second.render() == "A; B"


def test_resetting_a_reason_replaces_its_text():
    reasons = SendBlockReasons()
    reasons.set("intertrial", "waiting 2.0 s")
    reasons.set("intertrial", "waiting 0.5 s")
    assert reasons.render() == "waiting 0.5 s"


def test_clear_all_releases_everything():
    reasons = SendBlockReasons()
    reasons.set("analysis", "A")
    reasons.set("intertrial", "B")
    assert reasons.clear_all() == ""
    assert reasons.is_blocked is False


def test_every_change_is_published():
    """The rendered string is what BehaviorAlgorithm actually reads."""
    published = []
    reasons = SendBlockReasons(publish=published.append)

    reasons.set("analysis", "A")
    reasons.set("intertrial", "B")
    reasons.clear("analysis")
    reasons.clear_all()

    assert published == ["A", "A; B", "B", ""]


def test_holds_and_names_report_the_current_state():
    reasons = SendBlockReasons()
    reasons.set("intertrial", "waiting")
    assert reasons.holds("intertrial") is True
    assert reasons.holds("analysis") is False
    assert reasons.names == ("intertrial",)

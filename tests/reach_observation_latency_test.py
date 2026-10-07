from types import SimpleNamespace

from tools.acquisition.model.reach_state_source import LiveTrackingReachProvider


class _Buffer:
    def __init__(self, sample):
        self.sample = sample

    def latest(self):
        return self.sample


def _sample(sequence, located):
    return SimpleNamespace(sequence=sequence, processing_perf=4.0, source_perf=3.9,
                           location=lambda name: (1.0, 2.0, 3.0) if located else None)


def test_the_gate_reports_which_pose_it_read():
    seen = []
    provider = LiveTrackingReachProvider(_Buffer(_sample(7, True)),
                                         on_observe=lambda *args: seen.append(args))

    assert provider() == (True, 3.9)
    asked_at, sample, reaching = seen[0]
    assert sample.sequence == 7
    assert reaching is True
    assert asked_at > 0


def test_an_empty_buffer_is_reported_with_no_sample():
    seen = []
    provider = LiveTrackingReachProvider(_Buffer(None),
                                         on_observe=lambda *args: seen.append(args))

    assert provider() is None
    assert seen[0][1] is None and seen[0][2] is None


def test_a_failing_observer_does_not_change_the_answer():
    def broken(*_args):
        raise RuntimeError("recorder down")

    provider = LiveTrackingReachProvider(_Buffer(_sample(1, False)), on_observe=broken)
    assert provider() == (False, 3.9)

from autotrainer.video.record_backlog import RecordBacklog


def test_warns_once_at_a_quarter_full_and_again_only_after_draining():
    backlog = RecordBacklog(128, 60, 512 * 512)

    assert backlog.observe(31) is None
    warning = backlog.observe(32)
    assert warning is not None
    assert "32/128 batches" in warning
    assert "1920 frames" in warning
    assert "503 MB" in warning
    assert backlog.observe(40) is None   # still backed up, already said
    assert backlog.observe(8) is None    # not drained enough to re-arm
    assert backlog.observe(7) is None    # drained: re-armed
    assert backlog.observe(32) is not None


def test_the_peak_is_reported_and_restarts_per_recording():
    backlog = RecordBacklog(128, 60, 1)
    for depth in (1, 5, 3):
        backlog.observe(depth)
    assert backlog.take_peak() == 5
    assert backlog.take_peak() == 0


def test_an_unbounded_queue_never_warns():
    backlog = RecordBacklog(0, 60, 1)
    assert backlog.observe(10_000) is None
    assert backlog.take_peak() == 10_000

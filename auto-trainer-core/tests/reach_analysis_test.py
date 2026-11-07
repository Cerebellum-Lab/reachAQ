from autotrainer.core.analysis import ReachAnalysis


def test_reach_analysis_retains_only_required_detectors(project_info):
    analysis = ReachAnalysis()
    analysis.project_info = project_info

    assert analysis.project_info is project_info
    assert analysis.pellet_misplaced_monitor is not None
    assert analysis.watchdog_monitor is not None
    assert not hasattr(analysis, "alarms")
    assert not hasattr(analysis, "emergency_alarm_monitor")


def test_reach_analysis_detector_lifecycle():
    analysis = ReachAnalysis()

    analysis.start()
    assert analysis.pellet_misplaced_monitor.running
    assert analysis.watchdog_monitor.running

    analysis.restart()
    assert analysis.pellet_misplaced_monitor.running
    assert analysis.watchdog_monitor.running

    analysis.stop()
    assert not analysis.pellet_misplaced_monitor.running
    assert not analysis.watchdog_monitor.running

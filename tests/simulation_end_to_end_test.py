"""A whole simulated session: connect, record a reach protocol, stop, finalise, validate.

The session runs in a fresh interpreter (this file run as a script): the NI
worker is a forkserver child, and the forkserver keeps the environment it
started with, so prepare_simulation has to run before anything starts it.
The pytest process may already have one. Real clocks throughout: the CAN
reader stalls on the fake perf clocks.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD_SECONDS = 20.0
SEND_AT_SECONDS = 1.0
LASER_AT_SECONDS = 8.0
#: The validation rule a simulated session warns on, by design.
PROVENANCE_RULE = "session.provenance"


def _wait(condition, timeout, what):
    """Wait for condition(); `what` is called for the timeout's message, so it says how things stood then."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for {what()}")
        time.sleep(0.1)


def run_simulated_session(home: Path) -> Path:
    """Record one session in a sim home at `home`; returns its directory.

    Only call this in a process of its own, before anything has started a
    multiprocessing forkserver.
    """
    from multiprocessing import set_start_method
    set_start_method("spawn")

    from tools.acquisition.args import make_autotrainer_parser
    from tools.acquisition.model.app_model_status import AppModelStatus
    from tools.simulation.sim_home import prepare_simulation

    args = make_autotrainer_parser(default_start_mode=AppModelStatus.RUNNING).parse_args(
        ["--simulate", "--sim-home", str(home)])
    simulation = prepare_simulation(args)

    import logging
    from autotrainer.core.logging import setup_logging, stop_multiproc_logging
    setup_logging(logger_level=logging.INFO, time_precision=6, multiprocess_enabled=True)

    from autotrainer.behavior import BehaviorAlgorithm
    from autotrainer.core import LaserChannelId
    from autotrainer.core.event.event_manager import EventManager
    from autotrainer.device import LaserPulseTrain
    from tools.acquisition.model.app_model import AppModel
    from tools.acquisition.model.app_model_status import SessionRecordingStatus
    from tools.acquisition.model.helpers import get_config_location
    from tools.acquisition.model.user_preferences import UserPreferences

    preferences = UserPreferences(settings_file_path=args.preferences_file)
    app = AppModel(preferences, calib_dir=simulation.calib_dir)
    try:
        app.load_configuration(get_config_location(preferences, args.configuration))
        assert not app.nidaq_plan_error, app.nidaq_plan_error
        assert app.capture_start(target_status=AppModelStatus.RUNNING), "capture_start refused"
        app.on_activated()

        app.add_animal("sim-mouse", select=True)
        app.save_tone_profile("sim-tone-1", 5000, 200)
        app.create_ordered_protocol("sim-reach", "Simulated reach", trial_count=5)
        app.apply_ordered_protocol_values(
            (), {"enabled": True, "tone_profile_id": "sim-tone-1", "tone_phase": "before_send"},
            scope_kind="protocol")

        _wait(lambda: not [b for b in app.recording_blockers if b != "System Mode is not running"],
              90.0, lambda: f"the recording blockers to clear: {app.recording_blockers}")
        assert app.start_recording(), f"start_recording refused: {app.recording_blockers}"
        _wait(lambda: app.session_recording_status is SessionRecordingStatus.RECORDING,
              30.0, lambda: f"the session to record: {app.session_recording_status}")
        started = time.monotonic()

        # The Send button's call: the protocol's next trial, Tone 1 before its SEND.
        time.sleep(SEND_AT_SECONDS)
        assert app.request_pellet_send(force=True), "the protocol trial was not sent"

        time.sleep(max(0.0, LASER_AT_SECONDS - (time.monotonic() - started)))
        app.laser.run_pulse_train(
            LaserPulseTrain(channel_id=LaserChannelId.LASER_1, amplitude_volts=1.0,
                            duration_ms=10.0, pulse_count=5, frequency_hz=20.0),
            manual_context={"profile_id": "sim-burst", "profile_revision": 1,
                            "trigger_mode": "internal"})

        time.sleep(max(0.0, RECORD_SECONDS - (time.monotonic() - started)))
        assert app.stop_recording(), "stop_recording refused"
        _wait(lambda: (app.session_recording_status is SessionRecordingStatus.READY
                       and app.last_stopped_session_path is not None),
              120.0, lambda: f"the session to finalise: {app.session_recording_status}")
        return Path(app.last_stopped_session_path)
    finally:
        app.capture_stop(force=True)
        app.on_close()
        BehaviorAlgorithm.close_algorithm_handler()
        EventManager.try_close_default()
        stop_multiproc_logging()


@pytest.mark.functional
def test_a_simulated_session_validates_with_only_the_provenance_warning(tmp_path):
    from tools.session_validation.engine import validate_session
    from tools.session_validation.model import ValidationProfile

    home = tmp_path / "sim-home"
    log_path, result_path = tmp_path / "session.log", tmp_path / "session.json"
    with log_path.open("wb") as log:
        # Its own session, so a timeout can take the NI worker and its children down too.
        child = subprocess.Popen(
            [sys.executable, __file__, str(home), str(result_path)],
            cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            child.wait(timeout=240)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            pytest.fail("the simulated session timed out:" + chr(10)
                        + log_path.read_text(errors="replace")[-6000:])
    completed = child
    assert completed.returncode == 0, log_path.read_text(errors="replace")[-6000:]
    session_dir = Path(json.loads(result_path.read_text())["session"])
    assert home in session_dir.parents

    report = validate_session(session_dir, profile=ValidationProfile.FULL)
    results = [(r.rule_id, r.status.value, r.message) for r in report.results]
    failures = [r for r in results if r[1] in {"fail", "tool_error"}]
    warnings = [r for r in results if r[1] == "warning" and r[0] != PROVENANCE_RULE]
    assert not failures, failures
    assert not warnings, warnings
    assert (PROVENANCE_RULE, "warning") in [(r[0], r[1]) for r in results]
    # Not passed vacuously: the trial's tone reached the NI, and the train is in the session.
    alignment = json.loads((session_dir / "streams" / "alignment.json").read_text())
    matched = [m["channel"] for m in alignment["toneConfirmation"]["matched"]]
    assert matched and set(matched) == {"tone1"}, matched
    laser = (session_dir / "streams" / "laser.csv").read_text()
    assert "manual pulse" in laser and "completed" in laser
    assert (session_dir / "streams" / "trials.jsonl").read_text().strip()


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    session = run_simulated_session(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps({"session": session.as_posix()}))

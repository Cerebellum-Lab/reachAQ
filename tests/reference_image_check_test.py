"""The reference-image check: a camera nudged since its calibration is caught at Record.

The images are synthetic but rig-like: a board-free scene of fixed structure (blocks, bars
and a smooth texture) seen through a 256x256 crop, with sensor noise, and moved by warping
the scene about the crop's centre. Nothing here touches hardware: the cameras that record a
reference clip are fakes that write a short video when their recording stops.
"""

import dataclasses
import os
import re
import statistics
import time
from datetime import datetime
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from autotrainer.core.configuration import DEFAULT_3D_CALIB_DIR_NAME  # noqa: E402
from autotrainer.inference.multires_calibration import library, store  # noqa: E402
from calibration_metadata_test import _saved_calibration3d  # noqa: E402
from handle_3d_calibration_waypoints_test import _CaptureCamera, _capture_app_model  # noqa: E402
from reload_calib_multires_test import BASE_PARAMS, _configure_live_pair, _write_multires  # noqa: E402
from tools.acquisition.model import handle_3d_calibration as h3d  # noqa: E402
from tools.acquisition.model import reference_image_check as ric  # noqa: E402
from tools.acquisition.model.app_model import SetupDifference  # noqa: E402
from tools.acquisition.view import calibration_dialog as dialog_module  # noqa: E402
from tools.acquisition.view.behavior_content import BehaviorContent  # noqa: E402

SIZE = 256
ON = {"checks": {"reference_image": {"enabled": True}}}


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _world(seed=0, size=384):
    """A scene larger than the camera's view, so a moved camera still sees scene at its edges."""
    rng = np.random.default_rng(seed)
    texture = cv2.resize(rng.random((48, 48)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    image = 60 + 80 * texture
    for _ in range(14):
        x0, y0 = rng.integers(0, size, 2)
        w, h = rng.integers(20, 120, 2)
        cv2.rectangle(image, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), float(rng.integers(20, 235)), -1)
    for _ in range(8):
        p = rng.integers(0, size, 4)
        cv2.line(image, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), float(rng.integers(0, 255)), 3)
    return np.clip(image, 0, 255).astype(np.float32)


WORLD = _world()


def _view(shift=(0.0, 0.0), rotation=0.0, scale=1.0, noise=3.0, seed=1):
    """The camera's 256x256 view of WORLD, the camera moved by `shift` px, `rotation` degrees
    and `scale` about the view's centre, with Gaussian sensor noise."""
    big = WORLD.shape[0]
    centre = (big - 1) / 2
    warp = cv2.getRotationMatrix2D((centre, centre), rotation, scale)
    warp[:, 2] += shift
    warped = cv2.warpAffine(WORLD, warp, (big, big), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    o = (big - SIZE) // 2
    crop = warped[o:o + SIZE, o:o + SIZE] + np.random.default_rng(seed).normal(0, noise, (SIZE, SIZE))
    return np.clip(crop, 0, 255).astype(np.uint8)


REFERENCE = _view(seed=1)


def _entry(tmp_path, cameras=("left",)):
    """A calibration folder holding REFERENCE as each camera's reference image."""
    folder = tmp_path / "entry"
    (folder / "reference").mkdir(parents=True)
    for camera in cameras:
        cv2.imwrite(str(folder / "reference" / f"{camera}.png"), REFERENCE)
    return folder


def test_identical_frames_are_ok_with_subpixel_shift():
    registration = ric.register(REFERENCE, _view(seed=2))  # the same view, other sensor noise

    assert registration.shift_px < 0.5
    assert abs(registration.rotation_deg) < 0.1 and abs(registration.scale_pct) < 0.1
    assert registration.quality > 0.95
    assert ric.judge("left", registration, ric.Limits()).status == ric.OK


def test_12px_shift_is_moved_and_reports_value(tmp_path):
    verdicts = ric.check_cameras(_entry(tmp_path), {"left": _view(shift=(12.0, 0.0), seed=2)}, ON)

    verdict = verdicts["left"]
    assert verdict.status == ric.MOVED
    assert verdict.registration.shift_px == pytest.approx(12.0, abs=0.3)
    detail = ric.moved_detail(verdict)
    assert re.fullmatch(r"left: shifted 12\.\d px \(limit 4\.0\), rotated 0\.0° \(limit 1\.5\),"
                        r" scale 0\.0% \(limit 2\.0\), match 0\.9\d", detail), detail


def test_the_moved_detail_reads_as_the_confirmation_says():
    verdict = ric.judge("left", ric.Registration(6.3, -0.2, 0.1, 0.93), ric.Limits())

    assert verdict.status == ric.MOVED
    assert ric.moved_detail(verdict) == (
        "left: shifted 6.3 px (limit 4.0), rotated 0.2° (limit 1.5), scale 0.1% (limit 2.0), match 0.93")


def test_rotation_and_scale_measured():
    rotated2 = ric.register(REFERENCE, _view(rotation=2.0, seed=2))
    rotated1 = ric.register(REFERENCE, _view(rotation=1.0, seed=2))
    scaled3 = ric.register(REFERENCE, _view(scale=1.03, seed=2))
    scaled1 = ric.register(REFERENCE, _view(scale=1.01, seed=2))

    assert abs(rotated2.rotation_deg) == pytest.approx(2.0, abs=0.1)
    assert abs(rotated1.rotation_deg) == pytest.approx(1.0, abs=0.1)
    assert scaled3.scale_pct == pytest.approx(3.0, abs=0.2)
    assert scaled1.scale_pct == pytest.approx(1.0, abs=0.2)
    # A rotation about the centre is not counted as a shift.
    assert rotated2.shift_px < 0.5 and scaled3.shift_px < 0.5
    limits = ric.Limits()
    assert ric.judge("left", rotated2, limits).status == ric.MOVED
    assert ric.judge("left", rotated1, limits).status == ric.OK
    assert ric.judge("left", scaled3, limits).status == ric.MOVED
    assert ric.judge("left", scaled1, limits).status == ric.OK


def test_occluded_frame_is_cant_tell(tmp_path):
    # A hand or an animal: a large patch of other texture over a quarter of the view.
    frame = _view(seed=2)
    rng = np.random.default_rng(5)
    patch = cv2.resize(rng.random((20, 20)).astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_CUBIC) * 200 + 30
    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.ellipse(mask, (130, 140), (80, 60), 20, 0, 360, 255, -1)
    frame[mask > 0] = patch[mask > 0].astype(np.uint8)

    verdict = ric.check_cameras(_entry(tmp_path), {"left": frame}, ON)["left"]

    assert verdict.registration.quality < 0.8
    assert verdict.status == ric.CANT_TELL


def test_disabled_by_default(tmp_path):
    settings = ric.reference_check_settings({})
    assert settings.enabled is False
    assert settings.limits == ric.Limits(max_shift_px=4.0, max_rotation_deg=1.5, max_scale_pct=2.0, min_quality=0.80)
    assert ric.reference_check_settings(library.read_settings(tmp_path)).enabled is False

    assert ric.check_cameras(_entry(tmp_path), {"left": _view(shift=(12.0, 0.0))}, {}) == {}


def test_per_camera_override_loosens_one_camera(tmp_path):
    settings = {"checks": {"reference_image": {
        "enabled": True, "max_shift_px": 4.0, "per_camera": {"right": {"max_shift_px": 8.0}}}}}
    frame = _view(shift=(6.0, 0.0), seed=2)

    verdicts = ric.check_cameras(_entry(tmp_path, ("left", "right")), {"left": frame, "right": frame}, settings)

    assert verdicts["left"].status == ric.MOVED
    assert verdicts["right"].status == ric.OK and verdicts["right"].limits.max_shift_px == 8.0


def test_no_reference_no_frame_or_another_size(tmp_path):
    entry = _entry(tmp_path)
    frames = {"left": None, "right": _view()}
    verdicts = ric.check_cameras(entry, frames, ON)
    assert verdicts["left"].status == ric.CANT_TELL and verdicts["left"].reason == "no frame from the camera"
    assert verdicts["right"].status == ric.NO_REFERENCE

    other = ric.check_cameras(entry, {"left": cv2.resize(_view(), (128, 128))}, ON)["left"]
    assert other.status == ric.CANT_TELL and "128x128" in other.reason


def test_register_takes_under_100_ms_on_a_256_frame():
    current = _view(shift=(12.0, 0.0), seed=2)
    ric.register(REFERENCE, current)  # the first call loads OpenCV's code paths
    times = []
    for _ in range(10):
        start = time.perf_counter()
        ric.register(REFERENCE, current)
        times.append(time.perf_counter() - start)
    print(f"register 256x256: median {statistics.median(times) * 1e3:.1f} ms, max {max(times) * 1e3:.1f} ms")
    assert max(times) < 0.100


class _AppModel:
    """Task 7's confirmation fake, with the reference-image check."""

    def __init__(self, difference, verdicts):
        self.difference = difference
        self.verdicts = verdicts
        self.uncalibrated_reasons = {}
        self.recording_camera_names = ["left", "right"]
        self.protocol_check_notice = None
        self.calls = []

    def camera_setup_difference(self, selected):
        return self.difference

    def check_reference_images(self, frames):
        self.calls.append(("check", sorted(frames)))
        return self.verdicts

    def acknowledge_setup_difference(self):
        self.calls.append("acknowledge")

    def start_recording(self):
        self.calls.append("start")
        return True


def _question(monkeypatch, answer):
    asked = []

    def question(parent, title, text, buttons=None, default=None):
        asked.append((title, text))
        return answer

    monkeypatch.setattr(dialog_module, "QMessageBox", SimpleNamespace(
        question=question, StandardButton=QMessageBox.StandardButton))
    return asked


def _behavior(app_model, frames):
    status_bar = SimpleNamespace(messages=[])
    status_bar.showMessage = lambda text, timeout=0: status_bar.messages.append(text)
    window = SimpleNamespace(statusBar=lambda: status_bar)
    return SimpleNamespace(_app_model=app_model, window=lambda: window, frame_source=lambda: frames), status_bar


def test_moved_camera_reaches_the_confirmation(monkeypatch):
    moved = ric.judge("left", ric.Registration(6.3, 0.2, 0.1, 0.93), ric.Limits())
    ok = ric.judge("right", ric.Registration(0.2, 0.0, 0.0, 0.97), ric.Limits())
    app_model = _AppModel(SetupDifference([], [], [], []), {"left": moved, "right": ok})
    behavior, _ = _behavior(app_model, {"left": REFERENCE, "right": REFERENCE})
    asked = _question(monkeypatch, QMessageBox.StandardButton.Yes)

    BehaviorContent._start_recording(behavior)

    (title, text), = asked
    assert title == "Calibration looks needed"
    lines = text.splitlines()
    assert dialog_module.MOVED_HEADER in lines
    assert "left: shifted 6.3 px (limit 4.0), rotated 0.2° (limit 1.5), scale 0.1% (limit 2.0), match 0.93" in lines
    assert not any(line.startswith("right:") for line in lines)
    # The frames the views show are checked, and Yes covers the moved camera, then records.
    assert app_model.calls == [("check", ["left", "right"]), "acknowledge", "start"]


def test_a_moved_camera_answered_no_records_nothing(monkeypatch):
    moved = ric.judge("left", ric.Registration(6.3, 0.2, 0.1, 0.93), ric.Limits())
    app_model = _AppModel(SetupDifference([], [], [], []), {"left": moved})
    behavior, _ = _behavior(app_model, {"left": REFERENCE})
    _question(monkeypatch, QMessageBox.StandardButton.No)

    BehaviorContent._start_recording(behavior)

    assert "start" not in app_model.calls and "acknowledge" not in app_model.calls


def test_cant_tell_is_shown_but_does_not_ask(monkeypatch):
    unsure = ric.ReferenceVerdict("left", ric.CANT_TELL, ric.Registration(0.3, 0.0, 0.0, 0.62), ric.Limits(),
                                  "match 0.62 below 0.80")
    app_model = _AppModel(SetupDifference([], [], [], []), {"left": unsure})
    behavior, status_bar = _behavior(app_model, {"left": REFERENCE})
    asked = _question(monkeypatch, QMessageBox.StandardButton.No)

    BehaviorContent._start_recording(behavior)

    assert asked == [] and app_model.calls[-1] == "start"
    assert status_bar.messages == ["Reference image check can't tell: left (match 0.62 below 0.80)"]


def test_check_results_and_acknowledgement_reach_the_metadata(app_model, tmp_path):
    folder = _write_multires(tmp_path / "Autotrainer-calib" / DEFAULT_3D_CALIB_DIR_NAME)
    (folder / "reference").mkdir()
    for camera in ("left", "right"):
        cv2.imwrite(str(folder / "reference" / f"{camera}.png"), REFERENCE)
    _configure_live_pair(app_model, **BASE_PARAMS)
    app_model.reload_calib(folder)
    for camera in (app_model.left_camera, app_model.right_camera):
        camera.is_recording_enabled = True
    frames = {"left": _view(shift=(12.0, 0.0), seed=2), "right": _view(seed=3)}

    assert app_model.check_reference_images(frames) == {}  # off by default
    assert "movedCameraChecks" not in _saved_calibration3d(app_model, tmp_path)

    settings = library.settings_path(app_model.calibration_library_root)
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(yaml.safe_dump({"checks": {"quality": "warn"}, "live3d": True}), encoding="utf-8")
    app_model.reference_image_check_enabled = True
    assert yaml.safe_load(settings.read_text(encoding="utf-8")) == {
        "checks": {"quality": "warn", "reference_image": {"enabled": True}}, "live3d": True}

    verdicts = app_model.check_reference_images(frames)
    assert {name: v.status for name, v in verdicts.items()} == {"left": ric.MOVED, "right": ric.OK}
    app_model.acknowledge_setup_difference()
    app_model.start_recording()  # refused here (no acquisition), but it is the start the check was for
    record = _saved_calibration3d(app_model, tmp_path)["movedCameraChecks"]["referenceImage"]
    assert record["ran"] is True and record["moved"] == ["left"] and record["acknowledged"] is True
    left = record["cameras"]["left"]
    assert left["status"] == "moved" and left["shiftPx"] == pytest.approx(12.0, abs=0.3)
    assert left["limits"] == {"maxShiftPx": 4.0, "maxRotationDeg": 1.5, "maxScalePct": 2.0, "minQuality": 0.8}
    assert record["cameras"]["right"]["status"] == "ok"

    # The next start was not checked (a headless or API start): no results carried over.
    app_model.start_recording()
    record = _saved_calibration3d(app_model, tmp_path)["movedCameraChecks"]["referenceImage"]
    assert record["ran"] is False and record["cameras"] == {}


class _RecordingCamera(_CaptureCamera):
    """A fake camera that writes a three-frame video (grey 40, 120, 200) when its recording stops."""

    def __init__(self, *args, session, **kwargs):
        super().__init__(*args, **kwargs)
        self._session = session

    def on_trigger_recording(self, on):
        super().on_trigger_recording(on)
        if on:
            return
        video = self._session / f"{self.name}_recording.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        for value in (40, 120, 200):
            writer.write(np.full((48, 64, 3), value, np.uint8))
        writer.release()
        (self._session / f"{self.name}_recording_timestamps.txt").write_text("")


def _recording_app_model(tmp_path, calls):
    app_model = _capture_app_model(tmp_path, calls, stops=True)
    session = tmp_path / "session"
    app_model.left_camera = _RecordingCamera("left", True, calls, stops=True, camera_id=0, session=session)
    app_model.right_camera = _RecordingCamera("right", False, calls, stops=True, camera_id=1, session=session)
    return app_model, session


def test_reference_capture_keeps_each_middle_frame_without_moving_anything(tmp_path):
    calls = []
    app_model, session = _recording_app_model(tmp_path, calls)
    cameras = [app_model.left_camera, app_model.right_camera]
    saved = tuple(camera.save_configuration() for camera in cameras)
    project = app_model.make_project_info()

    written = h3d.capture_reference_images(cameras, saved, project, session, names={"right": "camera2"},
                                           sleep=lambda s: calls.append(("sleep", s)))

    assert sorted(written) == ["left", "right"]
    assert written["right"].name == "camera2.png"
    for path in written.values():
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert image.ndim == 2 and abs(int(image.mean()) - 120) <= 6  # the middle frame, grayscale
    assert not [call for call in calls if call[0] == "motor"]  # the stage is never connected
    for camera in cameras:
        loads = [call[2] for call in calls if call[:2] == (camera.name, "load_configuration")]
        # recorded at the camera's own (live) params, in TRIGGER mode, then given back
        assert loads[0].params == camera.saved.params and loads[0].record_mode == h3d.VideoRecordMode.TRIGGER
        assert loads[-1] is camera.saved


def test_a_reference_recorded_at_a_capture_binning_is_pooled_to_the_live_geometry(tmp_path):
    calls = []
    app_model, session = _recording_app_model(tmp_path, calls)
    cameras = [app_model.left_camera, app_model.right_camera]
    for camera in cameras:  # the clip is 64x48: twice this live geometry, as at a capture-binning preset
        camera.saved = dataclasses.replace(camera.saved, params=dict(camera.saved.params, width=32, height=24))
    saved = tuple(camera.save_configuration() for camera in cameras)

    written = h3d.capture_reference_images(cameras, saved, app_model.make_project_info(), session,
                                           sleep=lambda s: None)

    for path in written.values():
        assert cv2.imread(str(path), cv2.IMREAD_UNCHANGED).shape == (24, 32)


def test_calibration_with_references_puts_them_in_the_capture(tmp_path, monkeypatch):
    calls = []
    app_model, session = _recording_app_model(tmp_path, calls)
    monkeypatch.setattr(h3d, "run_waypoints", lambda *a, **k: SimpleNamespace(verified=[True]))
    monkeypatch.setattr(h3d, "calibrate_waypoint_capture",
                        lambda *a, **k: (session / "built", SimpleNamespace(id="x"), SimpleNamespace(verdict="pass")))
    monkeypatch.setattr(h3d, "REFERENCE_CLIP_S", 0.05)
    asked = []

    h3d.make_3d_calib(app_model, [app_model.left_camera, app_model.right_camera],
                      capture_references=True, confirm_references=lambda: asked.append(1) or True)

    capture = session / DEFAULT_3D_CALIB_DIR_NAME
    assert asked == [1]
    assert sorted(p.name for p in (capture / "reference").glob("*.png")) == ["left.png", "right.png"]
    # The waypoint videos are the calibration's, not overwritten by the clip.
    assert sorted(p.name for p in (capture / "source_videos").glob("*.mp4")) == ["left.mp4", "right.mp4"]


def test_capture_now_copy_inherits_the_fit_under_a_new_id(tmp_path):
    root = library.library_root(tmp_path / "config")
    built = _write_multires(tmp_path / "built" / DEFAULT_3D_CALIB_DIR_NAME)
    source_id = library.add_entry(root, built)
    source = library.entry_path(root, source_id)
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    png = tmp_path / "left.png"
    cv2.imwrite(str(png), REFERENCE)
    now = datetime(2026, 11, 20, 9, 30, 0)

    new_id = h3d.copy_with_references(source, {"left": png}, tmp_path / "copy", now=now)
    added = library.add_entry(root, tmp_path / "copy")

    assert added == new_id != source_id
    entry = library.entry_path(root, new_id)
    index, old = store.load_index(entry), store.load_index(source)
    raw = yaml.safe_load((entry / store.INDEX_FILE).read_text(encoding="utf-8"))
    assert raw["parent_id"] == source_id and raw["references_captured"] == "2026-11-20T09:30:00"
    assert index.created == old.created and index.cameras == old.cameras  # as old as its fit
    assert (entry / "sensor" / "fit.yaml").read_bytes() == (source / "sensor" / "fit.yaml").read_bytes()
    tomls = list(entry.rglob("*.toml"))
    assert tomls and all(new_id in t.read_text() and source_id not in t.read_text() for t in tomls)
    assert (entry / "reference" / "left.png").read_bytes() == png.read_bytes()
    # The source entry is untouched.
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before

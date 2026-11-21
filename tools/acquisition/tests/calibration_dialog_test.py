"""The 3D calibration dialog, the result page and the calibration library view.

The dialog lists every configured camera, ticked, the stimCam among them; a camera that is
disabled or has no hardware is listed with the reason and cannot be ticked. A camera the
calibration left 2D only leads the result page, named with its reason, in red.
"""

import os
from datetime import datetime
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.inference.multires_calibration.library import LibraryEntry  # noqa: E402
from autotrainer.inference.multires_calibration.report import CameraVerdict, QualityReport  # noqa: E402
from tools.acquisition.model.app_model_status import AppModelStatus  # noqa: E402
from tools.acquisition.view import calibration_library_view as library_view_module  # noqa: E402
from tools.acquisition.view import main_window as main_window_module  # noqa: E402
from tools.acquisition.view.calibration_dialog import (  # noqa: E402
    CalibrationDialog,
    CalibrationRequest,
    result_page,
    scout_advice_text,
)
from tools.acquisition.view.calibration_library_view import CalibrationLibraryView  # noqa: E402
from tools.acquisition.view.main_window import MainWindow  # noqa: E402

REASON = "too few board views: 4 of 12 needed"
BANNER = f"camera4 NOT CALIBRATED (2D only): {REASON}"


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _camera(name, serial, *, scheme="spinnaker", enabled=True, primary=False):
    return SimpleNamespace(name=name, is_enabled=enabled, is_primary=primary,
                           active_config=SimpleNamespace(scheme=scheme, host=serial))


def _rig(**stim):
    return [
        _camera("left", "21000001", primary=True),
        _camera("right", "21000002"),
        _camera("stimCam", "21000003", **stim),
    ]


def _report(verdict="warn") -> QualityReport:
    return QualityReport(
        created="2026-11-20T10:00:00-07:00", verdict=verdict, metrics=[], thresholds={}, notes=[],
        cameras={
            "left": CameraVerdict("left", True, ""),
            "right": CameraVerdict("right", True, ""),
            "camera4": CameraVerdict("camera4", False, REASON),
        },
    )


def test_dialog_lists_every_camera_ticked_including_stimcam():
    dialog = CalibrationDialog(_rig())

    boxes = [dialog.checkbox(name) for name in ("left", "right", "stimCam")]
    assert [box.text() for box in boxes] == ["left (21000001)", "right (21000002)", "stimCam (21000003)"]
    assert all(box.isChecked() and box.isEnabled() for box in boxes)
    assert dialog.start_button.isEnabled()
    assert dialog.request() == CalibrationRequest(
        cameras=["left", "right", "stimCam"], move_list="scout", capture_references=False)


def test_the_request_puts_the_configured_primary_first():
    cameras = [_camera("left", "21000001"), _camera("right", "21000002", primary=True), _camera("stimCam", "21000003")]
    assert CalibrationDialog(cameras).request().cameras == ["right", "left", "stimCam"]


@pytest.mark.parametrize("stim, reason", [
    (dict(scheme="random", serial=""), "no hardware (scheme random)"),
    (dict(enabled=False), "disabled"),
], ids=["random_scheme", "disabled"])
def test_non_hardware_camera_cannot_be_ticked(stim, reason):
    cameras = _rig()
    serial = stim.pop("serial", "21000003")
    cameras[2] = _camera("stimCam", serial, **stim)
    dialog = CalibrationDialog(cameras)

    box = dialog.checkbox("stimCam")
    assert not box.isChecked() and not box.isEnabled()
    assert reason in box.toolTip()
    assert reason in dialog.reason_label("stimCam").text()
    assert dialog.request().cameras == ["left", "right"]


def test_start_needs_two_to_six():
    dialog = CalibrationDialog(_rig())
    dialog.checkbox("stimCam").setChecked(False)
    assert dialog.start_button.isEnabled()
    dialog.checkbox("right").setChecked(False)
    assert not dialog.start_button.isEnabled()

    seven = [_camera(f"camera{i}", f"2100001{i}", primary=i == 0) for i in range(7)]
    dialog = CalibrationDialog(seven)
    assert not dialog.start_button.isEnabled()
    dialog.checkbox("camera6").setChecked(False)
    assert dialog.start_button.isEnabled()


def test_the_scout_is_the_default_move_list_and_references_wait_for_later():
    dialog = CalibrationDialog(_rig())
    assert dialog.scout_button.isEnabled() and dialog.scout_button.isChecked()
    assert "hand-written list when fewer than 12" in dialog.scout_button.toolTip()
    dialog.handwritten_button.setChecked(True)
    assert dialog.request().move_list == "handwritten"
    assert not dialog.references_box.isEnabled() and dialog.references_box.toolTip() == "not available yet"
    assert not dialog.references_box.isChecked()


def test_scout_advice_names_each_camera_and_what_continue_does():
    text = scout_advice_text({"camera4": "board never in view: the camera points elsewhere or the scout bounds miss its view"})
    assert text.splitlines()[0] == (
        "camera4: board never in view: the camera points elsewhere or the scout bounds miss its view")
    assert text.splitlines()[-1] == "Continue: the cameras named may end up 2D only."


def test_uncalibrated_banner_names_camera_and_reason():
    page = result_page(_report())

    assert BANNER in page
    # The banner is red, at the top, then the verdict, then the report.
    assert page.index(BANNER) < page.index("3D calibration quality: WARN")
    banner_element = page[:page.index(BANNER)]
    assert "color:" in banner_element.rsplit("<", 1)[-1]
    # Named once, and no calibrated camera is.
    assert page.count("NOT CALIBRATED") == 1
    assert "left NOT CALIBRATED" not in page


def test_the_installed_page_leads_with_the_banner():
    boxes = []
    window = SimpleNamespace(
        _set_calibration_due_label=lambda label: None,
        _show_msg_box=lambda title, text, icon: boxes.append((title, text)),
        _app_model=SimpleNamespace(dcs_state=None),
    )
    MainWindow._show_3d_calib_installed(window, _report(), None)
    (title, text), = boxes
    assert title == "3D calibration installed" and text.startswith(result_page(_report())[:40])
    assert BANNER in text


def test_make_3d_calibration_uses_the_dialog_choice(monkeypatch):
    cameras = {name: SimpleNamespace(name=name) for name in ("left", "right", "stimCam")}
    asked = []

    class _Action:
        def __init__(self):
            self.enabled, self.checked = True, True

        def setEnabled(self, value):
            self.enabled = value

        def setChecked(self, value):
            self.checked = value

    window = SimpleNamespace(
        _app_model=SimpleNamespace(cameras=tuple(cameras.values()), status=AppModelStatus.IDLE),
        run_action=_Action(), make_3d_calib_action=_Action(),
        _ask_calibration_cameras=lambda: CalibrationRequest(["right", "stimCam"], "handwritten", False),
        _finish_3d_calib=lambda error, result, prev: asked.append(("finish", error)),
    )

    class _InlineThread:
        def __init__(self, target, name, args=(), daemon=None):
            self._target, self._args = target, args

        def start(self):
            self._target(*self._args)

        def join(self):
            pass

    monkeypatch.setattr(main_window_module, "make_3d_calib",
                        lambda app_model, chosen, move_list=None, **kwargs: asked.append((list(chosen), move_list)))
    monkeypatch.setattr(main_window_module, "threading", SimpleNamespace(Thread=_InlineThread))
    monkeypatch.setattr(main_window_module, "invoke_method", lambda method: method)

    MainWindow.on_3d_calibrate(window, True)
    assert asked == [([cameras["right"], cameras["stimCam"]], "handwritten"), ("finish", None)]

    # Cancel starts nothing and gives Run and the action back.
    asked.clear()
    window._ask_calibration_cameras = lambda: None
    window._app_model.status = AppModelStatus.IDLE
    MainWindow.on_3d_calibrate(window, True)
    assert asked == []
    assert window.run_action.enabled and not window.make_3d_calib_action.checked
    assert window._app_model.status is AppModelStatus.IDLE


def _entries(tmp_path, active="b" * 32):
    entries = []
    for calibration_id, created, uncalibrated in (
        ("b" * 32, datetime(2026, 11, 20, 10, 0, 0), {"camera4": REASON}),
        ("a" * 32, datetime(2026, 11, 1, 9, 30, 0), {}),
    ):
        path = tmp_path / calibration_id
        path.mkdir(exist_ok=True)
        entries.append(LibraryEntry(
            id=calibration_id, created=created,
            cameras=["left", "right", "camera4"] if uncalibrated else ["left", "right"],
            verdict="warn" if uncalibrated else "pass", active=calibration_id == active,
            uncalibrated=uncalibrated, path=path))
    return entries


def test_library_view_lists_entries_with_their_2d_only_cameras(tmp_path, monkeypatch):
    monkeypatch.setattr(library_view_module, "list_entries", lambda root: _entries(tmp_path))
    view = CalibrationLibraryView(tmp_path, set_active=lambda calibration_id: None)

    table = view.table
    headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    assert headers == ["ID", "Created", "Cameras", "Verdict", "Active", "2D only"]
    rows = [[table.item(r, c).text() for c in range(table.columnCount())] for r in range(table.rowCount())]
    assert rows == [
        ["bbbbbbbb", "2026-11-20 10:00:00", "left, right, camera4", "WARN", "active", f"camera4: {REASON}"],
        ["aaaaaaaa", "2026-11-01 09:30:00", "left, right", "PASS", "", ""],
    ]
    assert table.item(0, 0).toolTip() == "b" * 32
    assert [b.text() for b in view.findChildren(type(view.set_active_button))
            if b.text() in ("Set active", "Show report")] == ["Set active", "Show report"]


def test_library_view_sets_the_selected_entry_active(tmp_path, monkeypatch):
    active = ["b" * 32]
    monkeypatch.setattr(library_view_module, "list_entries", lambda root: _entries(tmp_path, active[0]))
    chosen = []

    def set_active(calibration_id):
        chosen.append(calibration_id)
        active[0] = calibration_id

    view = CalibrationLibraryView(tmp_path, set_active=set_active)
    view.table.selectRow(0)
    assert not view.set_active_button.isEnabled(), "the active entry is active already"
    view.table.selectRow(1)
    assert view.set_active_button.isEnabled()
    view.set_active_button.click()

    assert chosen == ["a" * 32]
    assert view.table.item(1, 4).text() == "active" and view.table.item(0, 4).text() == ""


def test_library_view_refusal_is_shown_and_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(library_view_module, "list_entries", lambda root: _entries(tmp_path))
    warnings = []
    monkeypatch.setattr(library_view_module, "QMessageBox", SimpleNamespace(
        warning=lambda parent, title, text: warnings.append((title, text))))
    view = CalibrationLibraryView(tmp_path, set_active=lambda calibration_id: "Stop the recording first.")
    view.table.selectRow(1)
    view.set_active_button.click()
    assert warnings == [("Calibration not set active", "Stop the recording first.")]
    assert view.table.item(0, 4).text() == "active"


def test_library_report_page_leads_with_the_banner(tmp_path):
    entry, legacy = _entries(tmp_path)
    (entry.path / "quality_report.json").write_text(_report().to_json(), encoding="utf-8")
    page = library_view_module.report_page(entry)
    assert BANNER in page and page.index(BANNER) < page.index("3D calibration quality: WARN")
    assert "no quality report" in library_view_module.report_page(legacy)

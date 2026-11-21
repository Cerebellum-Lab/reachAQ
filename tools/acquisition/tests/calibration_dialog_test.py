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
from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QWidget  # noqa: E402

from autotrainer.inference.multires_calibration.library import LibraryEntry  # noqa: E402
from autotrainer.inference.multires_calibration.report import CameraVerdict, QualityReport  # noqa: E402
from tools.acquisition.model.app_model_status import AppModelStatus  # noqa: E402
from tools.acquisition.view import calibration_library_view as library_view_module  # noqa: E402
from tools.acquisition.view import main_window as main_window_module  # noqa: E402
from tools.acquisition.view.calibration_dialog import (  # noqa: E402
    REFERENCES_TOOLTIP,
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
    assert dialog.references_box.isEnabled() and dialog.references_box.toolTip() == REFERENCES_TOOLTIP
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


def test_library_view_captures_references_for_the_selected_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(library_view_module, "list_entries", lambda root: _entries(tmp_path))
    warnings = []
    monkeypatch.setattr(library_view_module, "QMessageBox", SimpleNamespace(
        warning=lambda parent, title, text: warnings.append((title, text))))
    asked = []
    view = CalibrationLibraryView(tmp_path, set_active=lambda calibration_id: None,
                                  capture_references=lambda calibration_id: asked.append(calibration_id) or "busy")
    button = view.capture_references_button
    assert button.text() == "Capture reference images now" and not button.isEnabled()
    view.table.selectRow(1)
    button.click()
    assert asked == ["a" * 32] and warnings == [("Reference images not captured", "busy")]
    # Without the callback the button is not offered.
    plain = CalibrationLibraryView(tmp_path, set_active=lambda calibration_id: None)
    assert not plain.capture_references_button.isVisibleTo(plain)


def test_library_report_page_leads_with_the_banner(tmp_path):
    entry, legacy = _entries(tmp_path)
    (entry.path / "quality_report.json").write_text(_report().to_json(), encoding="utf-8")
    page = library_view_module.report_page(entry)
    assert BANNER in page and page.index(BANNER) < page.index("3D calibration quality: WARN")
    assert "no quality report" in library_view_module.report_page(legacy)


def test_set_active_from_the_library_goes_through_the_app_model_and_resets_the_due_label(tmp_path, monkeypatch):
    answers = {"a" * 32: None, "c" * 32: "set active is not possible while recording"}
    labels, calls = [], []
    app_model = SimpleNamespace(
        calibration_library_root=tmp_path, dcs_state=None,
        set_active_calibration=lambda calibration_id: calls.append(calibration_id) or answers[calibration_id])
    window = SimpleNamespace(_app_model=app_model, _set_calibration_due_label=labels.append)
    monkeypatch.setattr(main_window_module.library, "pointer_id", lambda root: "b" * 32)

    assert MainWindow._set_active_calibration(window, "a" * 32) is None
    assert labels == [None], "the monthly reminder's label is reset, as after an install"

    assert MainWindow._set_active_calibration(window, "c" * 32) == "set active is not possible while recording"
    assert labels == [None], "a refusal leaves the label of the calibration still active"
    assert calls == ["a" * 32, "c" * 32]


def test_a_refused_set_active_of_the_active_entry_keeps_the_due_label(tmp_path, monkeypatch):
    pointer = ["c" * 32]
    answers = {"c" * 32: "set active is not possible while recording",
               "d" * 32: "Set active, but it could not be loaded: broken. Restart the app."}
    labels = []

    def set_active(calibration_id):
        if calibration_id == "d" * 32:
            pointer[0] = calibration_id  # written, then the load failed
        return answers[calibration_id]

    app_model = SimpleNamespace(calibration_library_root=tmp_path, dcs_state=None, set_active_calibration=set_active)
    window = SimpleNamespace(_app_model=app_model, _set_calibration_due_label=labels.append)
    monkeypatch.setattr(main_window_module.library, "pointer_id", lambda root: pointer[0])

    # Refused while recording: the pointer already named it, and the reminder stays.
    assert MainWindow._set_active_calibration(window, "c" * 32) == answers["c" * 32]
    assert labels == []
    # Set active, then not loaded: the pointer moved, so the reminder of the old one goes.
    assert MainWindow._set_active_calibration(window, "d" * 32) == answers["d" * 32]
    assert labels == [None]


def test_the_count_uses_the_right_plural():
    dialog = CalibrationDialog(_rig())
    assert dialog.count_text() == "3 cameras ticked"
    dialog.checkbox("right").setChecked(False)
    dialog.checkbox("stimCam").setChecked(False)
    assert dialog.count_text() == "1 camera ticked; a calibration takes 2 to 6"
    dialog.checkbox("right").setChecked(True)
    assert dialog.count_text() == "2 cameras ticked"


def test_scout_again_is_offered_only_with_the_scout():
    dialog = CalibrationDialog(_rig())
    box = dialog.rescout_box
    assert box.text() == "Scout again" and not box.isChecked() and box.isEnabled()
    assert dialog.request().rescout is False
    box.setChecked(True)
    assert dialog.request() == CalibrationRequest(
        cameras=["left", "right", "stimCam"], move_list="scout", capture_references=False, rescout=True)
    dialog.handwritten_button.setChecked(True)
    assert not box.isEnabled()
    assert dialog.request().rescout is False, "the hand-written list never scouts"
    dialog.scout_button.setChecked(True)
    assert box.isEnabled() and dialog.request().rescout is True


def _flush_deletes():
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()


class _Host(QWidget):
    """A stand-in MainWindow: what the dialog methods use, as a real parent widget."""

    def __init__(self, app_model):
        super().__init__()
        self._app_model = app_model
        self._library_view = None


def test_the_calibration_dialogs_are_freed_after_use(tmp_path, monkeypatch):
    monkeypatch.setattr(library_view_module, "list_entries", lambda root: _entries(tmp_path))
    host = _Host(SimpleNamespace(cameras=tuple(_rig()), calibration_library_root=tmp_path))
    try:
        monkeypatch.setattr(CalibrationDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
        assert MainWindow._ask_calibration_cameras(host).cameras == ["left", "right", "stimCam"]
        monkeypatch.setattr(CalibrationDialog, "exec", lambda self: QDialog.DialogCode.Rejected)
        assert MainWindow._ask_calibration_cameras(host) is None

        def use(view):
            view.table.selectRow(0)
            view.show_report_button.click()  # one report left open
            view.table.selectRow(1)
            view.show_report_button.click()
            reports = view.findChildren(QMessageBox)
            assert len(reports) == 2
            reports[0].close()
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(CalibrationLibraryView, "exec", use)
        host._set_active_calibration = lambda calibration_id: None
        host._capture_reference_images = lambda calibration_id: None
        MainWindow._on_calibration_library(host)
        _flush_deletes()

        assert host.findChildren(CalibrationDialog) == []
        assert host.findChildren(CalibrationLibraryView) == []
        assert host.findChildren(QMessageBox) == []
        assert host._library_view is None
    finally:
        host.deleteLater()
        _flush_deletes()


def test_a_question_asked_from_a_worker_returns_the_buttons_answer():
    import threading

    answers = []

    def worker(ask):
        answers.append(main_window_module._ask_on_gui_thread(ask, "advice"))

    for ask, expected in ((lambda advice: True, True), (lambda advice: False, False)):
        thread = threading.Thread(target=worker, args=(ask,))
        thread.start()
        while thread.is_alive():
            QApplication.processEvents()
            thread.join(0.01)
        assert answers.pop() is expected
    # On the GUI thread it runs at once.
    assert main_window_module._ask_on_gui_thread(lambda advice: advice == "advice", "advice") is True


def test_a_question_that_raises_reads_as_cancel():
    import threading

    def broken(*args):
        raise RuntimeError("the box could not be shown")

    assert main_window_module._ask_on_gui_thread(broken, "advice") is False
    answers = []
    thread = threading.Thread(target=lambda: answers.append(main_window_module._ask_on_gui_thread(broken)))
    thread.start()
    while thread.is_alive():
        QApplication.processEvents()
        thread.join(0.01)
    assert answers == [False]


def _capture_camera(name, *, primary=False, enabled=True):
    return _camera(name, "", enabled=enabled, primary=primary)


def test_capture_now_records_the_primary_the_entry_did_not_calibrate():
    primary, right, stim = _capture_camera("left", primary=True), _capture_camera("right"), _capture_camera("stimCam")

    # The entry calibrated right and stimCam only: the primary records too, for their trigger.
    cameras, record_only = main_window_module.reference_capture_cameras(
        [primary, right, stim], {"right": "camera2", "stimCam": "camera3"})
    assert cameras == [primary, right, stim] and record_only == ["left"]

    # With the primary calibrated, nothing is added.
    assert main_window_module.reference_capture_cameras(
        [primary, right], {"left": "camera1", "right": "camera2"}) == ([primary, right], [])

    # A primary that cannot record refuses the capture.
    off = _capture_camera("left", primary=True, enabled=False)
    refusal = main_window_module.reference_capture_cameras([off, right], {"right": "camera2"})
    assert refusal == ("The primary camera left triggers the others but cannot record (disabled):"
                       " enable it, or make a new calibration with the cameras that can.")
    assert main_window_module.reference_capture_cameras([primary], {}) == (
        "None of the configured cameras is one this calibration calibrated.")

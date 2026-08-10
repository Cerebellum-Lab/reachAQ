import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from autotrainer.device import Motor  # noqa: E402
from autotrainer.pyside import MotorConfigDialog  # noqa: E402


def test_motor_configuration_only_exposes_pellet_board_motors():
    app = QApplication.instance() or QApplication([])
    dialog = MotorConfigDialog()

    assert set(dialog.motor_mapping.values()) == {
        Motor.PELLET_X_MOTOR,
        Motor.PELLET_Y_MOTOR,
        Motor.PELLET_Z_MOTOR,
        Motor.PELLET_LOAD_SERVO,
        Motor.PELLET_COVER_SERVO,
    }
    assert dialog.motor_combo.count() == 5

    dialog.close()
    app.processEvents()

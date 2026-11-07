from PySide6 import Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QFormLayout, QLabel, QColormap, QColorDialog, QCheckBox
from PySide6.QtGui import QColor

from autotrainer.core import MessageHandler, get_verbose_logger
from autotrainer.device import ColorLed
from autotrainer.pyside import CardWidget, StatusIcon, QSwitch
from autotrainer.pyside.content_widget import invoke_method
from tools.pellet_delivery.model.app_model import AppModel


logger = get_verbose_logger(__name__)


def _create_stimulus_panel():
    layout = QFormLayout()
    layout.setHorizontalSpacing(8)

    inputs = []
    for idx in range(4):
        box = StatusIcon()
        inputs.append(box)
        lbl = f"Stimulus #{idx + 1}"
        layout.addRow(f"{lbl}:", box)

    layout.setContentsMargins(8, 8, 8, 8)

    panel = CardWidget(title="GPIO", content_layout=layout)

    return inputs, panel


def _create_led_panel(app_model: AppModel):
    layout = QFormLayout()
    layout.setHorizontalSpacing(8)
    layout.setContentsMargins(8, 8, 8, 8)
    inputs = []
    for color in ("Red", "Green", "Blue"):
        lbl = QLabel(color)
        value = QLabel()
        layout.addRow(lbl, value)
        inputs.append(value)

    checkbox = QCheckBox()
    inputs.append(checkbox)

    def on_led_color_click(chk=checkbox):
        cl = app_model.color_led
        if cl is None:
            cl = ColorLed(red=0, green=0, blue=0)
        # convert from % to 255:
        c = QColor(*(round(v * 2.55) for v in (cl.red, cl.green, cl.blue)))
        c = QColorDialog.getColor(c, None, "Select Color")
        if c.isValid():
            # convert from 255 to % :
            rgb = tuple(round(v / 2.55) for v in (c.red(), c.green(), c.blue()))
            app_model.set_color_led(*rgb)

    checkbox.clicked.connect(on_led_color_click)
    layout.addRow("Select Color:", checkbox)

    panel = CardWidget(title="LED", content_layout=layout)

    return inputs, panel


class PelletStateWidget(QWidget):

    def __init__(self, app_model: AppModel):
        super().__init__()

        self._app_model = app_model
        app_model.property_changed += self._model_property_changed

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self._stimulus, panel = _create_stimulus_panel()
        layout.addWidget(panel)

        inputs, panel = _create_led_panel(self._app_model)
        self._leds_widgets = inputs[:3]
        self._color_led_widget = inputs[3]
        self._set_color_led(app_model.color_led)

        layout.addWidget(panel)
        self.setEnabled(False)
        self.setLayout(layout)

    def _set_color_led(self, col_led: ColorLed):
        if col_led is None:
            col_led = ColorLed(red=0, green=0, blue=0)
        # convert to 0->255
        rgb = tuple(round(v * 2.55) for v in (col_led.red, col_led.green, col_led.blue))
        logger.verbose("refreshing color led: %s", rgb)
        for c, w in zip(rgb, self._leds_widgets):
            w.setText(f"{c}")
        checkbox = self._color_led_widget
        col_code = "#" + "".join(f"{hex(val)[2:]:0>2}" for val in rgb)
        checkbox.setStyleSheet("""
            QCheckBox {
                spacing: 10px;
                color: white;
            }
            QCheckBox::indicator {
                width: 50px;
                height: 25px;
                border-radius: 12px;
                background-color: """
        + f"{col_code}"
        + """; /* Unchecked background color */
                border: 1px solid #9ca3af;
            }
            QCheckBox::indicator:checked {
                background-color: """
        + f"{col_code}"
        + """; /* Checked background color */
                border: 1px solid #1d4ed8;
            }
            QCheckBox::indicator::branch {
                background-color: transparent;
            }
            /* The sliding circle */
            QCheckBox::indicator::handle {
                width: 21px;
                height: 21px;
                border-radius: 10px;
                background-color: """
        + f"{col_code}"
        + """; /* Checked background color */
                margin: 2px; /* Push inward */
            }
            QCheckBox::indicator:checked::handle {
                margin-left: 26px; /* Push to the right when checked */
            }
        """
        )

    @invoke_method
    def _model_property_changed(self, name: str, value, _old_value):
        app_model = self._app_model
        if name == "is_connected":
            if value:
                reset_prop = self._model_property_changed
                reset_prop(MessageHandler.STIMULI_PROPERTY, app_model.stimuli, None)
            else:
                for box in self._stimulus:
                    box.setStatus(False)
        elif name == MessageHandler.STIMULI_PROPERTY:
            if value is None:
                value = [False] * len(self._stimulus)
            for v, box in zip(value, self._stimulus):
                box.setStatus(bool(v))
        elif name == MessageHandler.COLOR_LED:
            self._set_color_led(value)

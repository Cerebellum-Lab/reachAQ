import inspect
import textwrap
import threading
from io import StringIO
from pprint import pformat
from typing import get_type_hints

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QCheckBox, QSizePolicy, QPushButton


class DebugView(QDialog):
    _sig_output_text_changed = Signal(str)
    _sig_append_output_text = Signal(str)
    _sig_enabled_changed = Signal(bool)
    
    def __init__(self, main_window):
        super().__init__()
        self.setWindowTitle("AutoTrainer Debug View")
        self._main_window = main_window
        layout = QVBoxLayout()
        vlayout = QHBoxLayout()
        box = QPushButton("Clear output")
        def on_clear_output():
            self._output_label.setText("")
        box.clicked.connect(on_clear_output)
        vlayout.addWidget(box)
        layout.addLayout(vlayout)
        widget = self._use_exec_box = QCheckBox()
        widget.setText("Use exec instead of eval")
        layout.addWidget(widget)
        self._cmd_line_edit = widget = QLineEdit()
        self._sig_enabled_changed.connect(widget.setEnabled)
        # only using returnPressed, a lot safer:
        widget.returnPressed.connect(self._on_cmd_return_pressed)
        layout.addWidget(widget)
        label = self._output_label = QLabel()
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self._sig_output_text_changed.connect(label.setText)
        self._sig_append_output_text.connect(self._append_text)
        label.setWordWrap(True)
        layout.addWidget(label)
        #
        self.setLayout(layout)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowFlag(Qt.WindowType.WindowMinMaxButtonsHint, True)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        # self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        # this is what can be used/referred to directly
        app = main_window.app_model
        print = self._print
        self._locals = {
            "w": main_window,
            "app": app,
            "print": print,
        }
        print("You can use 'w' for main_window or 'app' for app_model in command ..\n")
        print(textwrap.dedent(
            "To execute trials/sessions simulations, you can use:\n\n"
            "w._simulate_sessions("
            "n_sessions=1, "
            "n_trials=1, "
            "rand_mouse_seen=1, "
            "rand_hands_near_pellet=1, "
            "print=print,"
            ")\n"
        ))
        print("_simulate_sessions signature:")
        sig = inspect.signature(main_window._simulate_sessions)
        for name, param in sig.parameters.items():
            print(f"{name}: {getattr(param.annotation, '__name__', param.annotation)}={param.default},")

    def _append_text(self, txt: str):
        prev = self._output_label.text()
        if not prev.endswith("\n"):
            prev += "\n"
        self._output_label.setText(prev + txt)

    def _print(self, *args, **kwargs):
        t = StringIO()
        print(*args, **kwargs, file=t)
        self._sig_append_output_text.emit(t.getvalue())

    def _execute_cmd(self, cmd, use_exec):
        try:
            if use_exec:
                exec(cmd, self._locals)
                result = None
            else:
                try:
                    result = eval(cmd, self._locals)
                except SyntaxError:
                    self._on_cmd_return_pressed(use_exec=True)
                    return
        except BaseException as err:
            result = err
        if result is not None:
            self._sig_append_output_text.emit(str(result))
        self._sig_enabled_changed.emit(True)

    def _on_cmd_return_pressed(self, use_exec=None):
        cmd = self._cmd_line_edit.text()
        if use_exec is None:
            use_exec = self._use_exec_box.isChecked()
        if cmd.startswith(("import ", "from ")):
            use_exec = True
        self._cmd_line_edit.setEnabled(False)
        self._output_label.setText(f"Executing {cmd}")
        th = threading.Thread(
            target=self._execute_cmd, daemon=True, name="debug-exec-cmd",
            args=(cmd, use_exec))
        th.start()

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                               QComboBox, QDoubleSpinBox, QCheckBox,
                               QPushButton, QGroupBox, QSpinBox)
from PySide6.QtCore import Qt, QMetaObject, Signal, Q_ARG

from autotrainer.device import Motor, StepperConfig, ServoConfig
from autotrainer.pyside import QSwitch


class MotorConfigDialog(QDialog):
    """Dialog for configuring motor parameters."""

    motor_selected = Signal(Motor, name="motor_selected")
    update_stepper_signal = Signal(object, name="update_stepper")
    update_servo_signal = Signal(object, name="update_servo")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Motor Configuration")
        self.resize(400, 500)

        # Motor types
        self.motor_mapping = {
            "X": Motor.PELLET_X_MOTOR,
            "Y": Motor.PELLET_Y_MOTOR,
            "Z": Motor.PELLET_Z_MOTOR,
            "Load Arm": Motor.PELLET_LOAD_SERVO,
            "Barrier": Motor.PELLET_COVER_SERVO,
        }

        self.stepper_motors = [name for name, motor_id in self.motor_mapping.items()
                               if motor_id in (
                                   Motor.PELLET_X_MOTOR, Motor.PELLET_Y_MOTOR,
                                   Motor.PELLET_Z_MOTOR)]
        self.servo_motors = [name for name, motor_id in self.motor_mapping.items()
                             if motor_id in (
                                 Motor.PELLET_LOAD_SERVO,
                                 Motor.PELLET_COVER_SERVO,
                             )]

        self.all_motors = self.stepper_motors + self.servo_motors

        self.config = None

        self.update_stepper_signal.connect(self._update_stepper_ui)
        self.update_servo_signal.connect(self._update_servo_ui)

        self._init_ui()
        self._init_connections()

    def _init_ui(self):
        """Initialize the UI components."""
        # Create the main layout with self as parent
        main_layout = QVBoxLayout(self)  # This sets the layout to self directly

        # Motor selection
        form_layout = QFormLayout()  # No parent in constructor
        self.motor_combo = QComboBox(self)
        self.motor_combo.addItems(self.all_motors)
        form_layout.addRow("Motor:", self.motor_combo)
        main_layout.addLayout(form_layout)

        # Common fields group
        self.common_group = QGroupBox("Common Parameters", self)
        common_layout = QFormLayout()  # Create layout without parent

        self.max_velocity_spin = QDoubleSpinBox(self)
        self.max_velocity_spin.setRange(0, 10000)
        self.max_velocity_spin.setSuffix(" units/s")
        common_layout.addRow("Maximum Velocity:", self.max_velocity_spin)

        self.max_accel_spin = QDoubleSpinBox(self)
        self.max_accel_spin.setRange(0, 10000)
        self.max_accel_spin.setSuffix(" units/s²")
        common_layout.addRow("Maximum Acceleration:", self.max_accel_spin)

        self.common_group.setLayout(common_layout)  # Set layout to group box
        main_layout.addWidget(self.common_group)

        # Stepper-specific fields group
        self.stepper_group = QGroupBox("Stepper Parameters", self)
        stepper_layout = QFormLayout()  # Create layout without parent

        self.homing_velocity_spin = QDoubleSpinBox(self)
        self.homing_velocity_spin.setRange(0, 80)
        self.homing_velocity_spin.setSuffix(" mm/s")
        stepper_layout.addRow("Homing Velocity:", self.homing_velocity_spin)

        self.flip_orientation_check = QCheckBox(self)
        stepper_layout.addRow("Flip Motor Orientation:", self.flip_orientation_check)

        self.micro_steps_combo = QComboBox(self)
        micro_steps_options = [2, 4, 8, 16, 32, 64]
        self.micro_steps_combo.addItems([str(x) for x in micro_steps_options])
        stepper_layout.addRow("Micro Steps:", self.micro_steps_combo)

        self.steps_per_rev_spin = QSpinBox(self)
        self.steps_per_rev_spin.setRange(1, 10000)
        stepper_layout.addRow("Steps/Revolution:", self.steps_per_rev_spin)

        self.stepper_group.setLayout(stepper_layout)  # Set layout to group box
        main_layout.addWidget(self.stepper_group)

        # Servo-specific fields group
        self.servo_group = QGroupBox("Servo Parameters", self)
        servo_layout = QFormLayout()  # Create layout without parent

        self.min_position_spin = QDoubleSpinBox(self)
        self.min_position_spin.setRange(0, 120)
        self.min_position_spin.setSuffix("°")
        servo_layout.addRow("Minimum Position:", self.min_position_spin)

        self.max_position_spin = QDoubleSpinBox(self)
        self.max_position_spin.setRange(0, 120)
        self.max_position_spin.setSuffix("°")
        servo_layout.addRow("Maximum Position:", self.max_position_spin)

        self.min_pwm_spin = QDoubleSpinBox(self)
        self.min_pwm_spin.setRange(0, 10000)
        self.min_pwm_spin.setSuffix(" usec")
        servo_layout.addRow("Minimum PWM duration:", self.min_pwm_spin)

        self.max_pwm_spin = QDoubleSpinBox(self)
        self.max_pwm_spin.setRange(0, 10000)
        self.max_pwm_spin.setSuffix(" usec")
        servo_layout.addRow("Maximum PWM duration:", self.max_pwm_spin)

        self.servo_detach_toggle = QSwitch()
        servo_layout.addRow("detach:", self.servo_detach_toggle)

        self.servo_group.setLayout(servo_layout)  # Set layout to group box
        main_layout.addWidget(self.servo_group)

        # Buttons
        button_layout = QHBoxLayout()  # No parent
        self.cancel_button = QPushButton("Cancel", self)
        self.save_button = QPushButton("Save", self)
        self.save_button.setDefault(True)

        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)

        main_layout.addStretch()
        main_layout.addLayout(button_layout)

    def _init_connections(self):
        """Initialize signal-slot connections."""
        self.motor_combo.currentIndexChanged.connect(self._query_config)
        self.save_button.clicked.connect(self._on_save)
        self.cancel_button.clicked.connect(self.reject)

    def _adjust_view(self):
        """Show/hide appropriate fields based on motor selection"""
        motor_name = self.motor_combo.currentText()

        if motor_name in self.stepper_motors:
            self.stepper_group.setVisible(True)
            self.servo_group.setVisible(False)
            self.max_velocity_spin.setSuffix(" mm/s")
            self.max_accel_spin.setSuffix(" mm/s²")

        elif motor_name in self.servo_motors:
            self.stepper_group.setVisible(False)
            self.servo_group.setVisible(True)
            self.max_velocity_spin.setSuffix(" deg/s")
            self.max_accel_spin.setSuffix(" deg/s²")

    def _query_config(self):
        motor_name = self.motor_combo.currentText()
        motor = self.motor_mapping[motor_name]

        self._adjust_view()
        self.motor_selected.emit(motor)

    def showEvent(self, event):
        super().showEvent(event)
        self._query_config()

    def _on_save(self):
        """Handle save button click."""
        selected_motor = self.motor_combo.currentText()

        motor = self.motor_mapping[selected_motor]

        if selected_motor in self.stepper_motors:
            config = StepperConfig()
            config.motor = motor
            config.maximum_velocity = self.max_velocity_spin.value()
            config.maximum_acceleration = self.max_accel_spin.value()
            config.homing_velocity = self.homing_velocity_spin.value()
            config.flip_limit_orientation = self.flip_orientation_check.isChecked()
            config.microsteps = int(self.micro_steps_combo.currentText())
            config.steps_per_revolution = self.steps_per_rev_spin.value()

            self.config = config

        elif selected_motor in self.servo_motors:
            config = ServoConfig()
            config.motor = motor
            config.maximum_velocity = self.max_velocity_spin.value()
            config.maximum_acceleration = self.max_accel_spin.value()
            config.minimum_position = self.min_position_spin.value()
            config.maximum_position = self.max_position_spin.value()
            config.minimum_pwm_duration = self.min_pwm_spin.value()
            config.maximum_pwm_duration = self.max_pwm_spin.value()
            config.detach = self.servo_detach_toggle.isChecked()

            self.config = config

        self.accept()

    def _select_motor(self, motor: Motor):
        """
        Select the corresponding motor in the combo box based on its Motor enum value.

        Args:
            motor: The Motor enum value to select in the combo box
        """

        for name, motor_id in self.motor_mapping.items():
            if motor_id == motor:
                index = self.motor_combo.findText(name)
                if index >= 0:
                    self.motor_combo.setCurrentIndex(index)
                    self._adjust_view()
                break

    def update_servo_config(self, servo_config: ServoConfig):
        """
        Update the UI with values from a ServoConfig structure.
        This method is designed to be called from a non-Qt thread.
        
        Args:
            servo_config
        """
        self.update_servo_signal.emit(servo_config)

    def _update_servo_ui(self, servo_config: ServoConfig):
        """
        Update UI with servo config values.
        """
        self._select_motor(servo_config.motor)

        self.max_velocity_spin.setValue(servo_config.maximum_velocity)
        self.max_accel_spin.setValue(servo_config.maximum_acceleration)

        self.min_position_spin.setValue(servo_config.minimum_position)
        self.max_position_spin.setValue(servo_config.maximum_position)
        self.min_pwm_spin.setValue(servo_config.minimum_pwm_duration)
        self.max_pwm_spin.setValue(servo_config.maximum_pwm_duration)
        self.servo_detach_toggle.setChecked(servo_config.detach)

    def update_stepper_config(self, stepper_config: ServoConfig):
        """
        Update the UI with values from a StepperConfig structure.
        This method is designed to be called from a non-Qt thread.
        
        Args:
            stepper_config
        """
        self.update_stepper_signal.emit(stepper_config)

    def _update_stepper_ui(self, stepper_config: StepperConfig):
        """
        Update UI with stepper config values.
        """
        self._select_motor(stepper_config.motor)

        self.max_velocity_spin.setValue(stepper_config.maximum_velocity)
        self.max_accel_spin.setValue(stepper_config.maximum_acceleration)

        self.homing_velocity_spin.setValue(stepper_config.homing_velocity)
        self.flip_orientation_check.setChecked(stepper_config.flip_limit_orientation)
        index = self.micro_steps_combo.findText(str(stepper_config.microsteps))
        self.micro_steps_combo.setCurrentIndex(index)
        self.steps_per_rev_spin.setValue(stepper_config.steps_per_revolution)

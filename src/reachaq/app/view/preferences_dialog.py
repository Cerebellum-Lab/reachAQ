from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QSizePolicy

from reachaq.app.model.app_model import AppModel
from reachaq.app.model.user_preferences import UserPreferences
from reachaq.app.view.preferences_content import PreferencesContent


class PreferencesDialog(QDialog):
    def __init__(self, preferences: UserPreferences, model: AppModel):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.setWindowTitle("Preferences")

        buttons = QDialogButtonBox.StandardButton.Ok

        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        self.layout.addWidget(PreferencesContent(preferences, model))
        self.layout.addWidget(self.buttonBox)

        self.setLayout(self.layout)

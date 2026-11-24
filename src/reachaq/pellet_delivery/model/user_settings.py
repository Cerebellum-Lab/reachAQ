from PySide6.QtCore import QCoreApplication, QSettings


class UserSettings:
    def __init__(self):
        QCoreApplication.setOrganizationName("reachAQ")
        QCoreApplication.setOrganizationDomain("colorado.edu")
        QCoreApplication.setApplicationName("PelletDeliveryUI")

        self._settings = QSettings()

        self._cameras = None

    def save(self):
        self._settings.sync()

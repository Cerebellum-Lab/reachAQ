import logging
import platform
import sys
from pathlib import Path
from typing import Optional, Union

from PySide6.QtCore import QByteArray, QCoreApplication, QRect, QSettings

from autotrainer.core import ObservableObject
from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.configuration import SystemConfiguration
from tools.acquisition.model.animal_metadata_sync import (
    DEFAULT_SOFTMOUSE_MANIFEST_PATH,
)


logger = get_verbose_logger(__name__)


def get_default_configuration_location() -> Path:
    return SystemConfiguration.DEFAULT_CONFIG_DIR.expanduser()


def get_default_animals_location(default_config_location: Path) -> Path:
    return default_config_location.joinpath("animals")


class UserPreferences(ObservableObject):

    CONFIGURATION_LOCATION = "configuration_location"
    SERIAL_NUMBER = "serial_number"
    LIVE_FEED_REFRESH_RATE = "live_feed_refresh_rate"
    LOG_LOCATION = "log_location"
    LOG_LEVEL = "log_level"
    SELECTED_ANIMAL = "selected_animal"
    ANIMAL_LOCATION = "animal_location"
    PELLET_PORT = "pellet_port"
    MEASUREMENT_GRAPH = "measurement_graph"
    SOFTMOUSE_MANIFEST_PATH = "softmouse_manifest_path"
    SOFTMOUSE_NAME_COLUMN = "softmouse_name_column"
    SOFTMOUSE_NIGHTLY_REFRESH = "softmouse_nightly_refresh"
    WINDOW_NORMAL_GEOMETRY = "window_normal_geometry"

    def __init__(self, *, settings_file_path: Optional[Path] = None):
        super().__init__()
        if settings_file_path is None:
            settings_args = ()
        else:
            settings_args = (settings_file_path.as_posix(), QSettings.Format.IniFormat)

        if sys.platform.startswith("win"):
            if settings_file_path is None:
                settings = self._settings = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "Colorado", "Auto Trainer")
            else:
                settings = self._settings = QSettings(*settings_args)
        else:
            QCoreApplication.setOrganizationName("Colorado")
            QCoreApplication.setOrganizationDomain("colorado.edu")
            QCoreApplication.setApplicationName("Auto Trainer")
            # IniFormat is required for the settings_file_path to be effective,
            # otherwise it's prepended with XDG_CONFIG_DIR :
            settings = self._settings = QSettings(*settings_args)

        logger.verbose("Using setting ini file: %r", settings.fileName())

        self._serial_number: str = settings.value("system/serial_number", platform.node(), str)  # noqa
        self._last_configuration: str = settings.value("system/last_configuration", "", str)  # noqa

        self._configuration_location: str = settings.value(  # noqa
            "system/configuration_location",
            get_default_configuration_location().as_posix(),
            str,
        )

        self._animal_location: str = settings.value(
            "system/animal_location",
            get_default_animals_location(Path(self._configuration_location)).as_posix())  # noqa

        # Subject selection is session state. Never restore it when reachAQ starts,
        # and remove the legacy persisted value so older installations also start
        # with an empty subject.
        self._selected_animal: str = ""
        settings.remove("system/selected_animal")

        self._log_location: str = settings.value("system/log_location", "")  # noqa
        self._log_level: int = settings.value("system/log_level", logging.WARNING, int)  # noqa

        self._live_feed_refresh_rate: int = settings.value("display/refresh_rate", 15, int)  # noqa
        self._measurement_graph: str = settings.value("ui/measurement_graph", "")  # noqa

        self._softmouse_manifest_path: str = (
            settings.value(
                "softmouse/manifest_path",
                DEFAULT_SOFTMOUSE_MANIFEST_PATH.as_posix(),
                str,
            ).strip()
            or DEFAULT_SOFTMOUSE_MANIFEST_PATH.as_posix()
        )
        self._softmouse_name_column: str = settings.value(
            "softmouse/new_animal_name_column", "Physical Tag", str
        )
        self._softmouse_nightly_refresh: bool = settings.value(
            "softmouse/nightly_refresh", True, bool
        )
        self._window_normal_geometry = QRect(
            settings.value("ui/window_normal_geometry", QRect(), QRect)
        )
        # Transient values that may come from individual configuration files, but are conveniently accessed from
        # the user preferences.

        self._pellet_port = None

    def save(self):
        logger.verbose("Saving ini config to %s", self._settings.fileName())
        self._settings.sync()

    def splitter_state(self, name: str) -> QByteArray:
        value = self._settings.value(f"ui/splitters/{name}", QByteArray())
        if isinstance(value, QByteArray):
            return value
        return QByteArray(value) if value else QByteArray()

    def set_splitter_state(self, name: str, state: QByteArray) -> None:
        self._settings.setValue(f"ui/splitters/{name}", state)

    @property
    def window_normal_geometry(self) -> QRect:
        return QRect(self._window_normal_geometry)

    @window_normal_geometry.setter
    def window_normal_geometry(self, value: QRect) -> None:
        value = QRect(value)
        if not value.isValid() or value.width() < 1 or value.height() < 1:
            return
        previous = self._window_normal_geometry
        self._window_normal_geometry = value
        self._settings.setValue("ui/window_normal_geometry", value)
        self._on_property_changed(self.WINDOW_NORMAL_GEOMETRY, value, previous)

    @property
    def last_configuration(self) -> str:
        return self._last_configuration

    @property
    def configuration_location(self) -> str:
        return self._configuration_location

    @configuration_location.setter
    def configuration_location(self, value: Union[str, Path]) -> None:
        value = Path(value).as_posix()
        prev, self._configuration_location = self._configuration_location, value
        self._on_property_changed(self.CONFIGURATION_LOCATION, value, prev)
        self._settings.setValue("system/configuration_location", value)

    @property
    def serial_number(self) -> str:
        return self._serial_number

    @serial_number.setter
    def serial_number(self, value: str):
        prev, self._serial_number = self._serial_number, value
        self._settings.setValue("system/serial_number", value)
        self._on_property_changed(self.SERIAL_NUMBER, value, prev)

    @property
    def live_feed_refresh_rate(self) -> int:
        return self._live_feed_refresh_rate

    @live_feed_refresh_rate.setter
    def live_feed_refresh_rate(self, value: int):
        prev, self._live_feed_refresh_rate = self._live_feed_refresh_rate, value
        self._settings.setValue("display/refresh_rate", value)
        self._on_property_changed(self.LIVE_FEED_REFRESH_RATE, value, prev)

    @property
    def log_location(self) -> str:
        return self._log_location

    @log_location.setter
    def log_location(self, value: Union[str, Path]):
        value = Path(value).as_posix()
        prev, self._log_location = self._log_location, value
        self._settings.setValue("system/log_location", value)
        self._on_property_changed(self.LOG_LOCATION, value, prev)

    @property
    def selected_animal(self) -> str:
        return self._selected_animal

    @selected_animal.setter
    def selected_animal(self, value: str):
        # set new value first,
        prev, self._selected_animal = self._selected_animal, value
        # then eventually trigger the on_property_changed event:
        self._on_property_changed(self.SELECTED_ANIMAL, value, prev)

    @property
    def animal_location(self) -> str:
        return self._animal_location

    @animal_location.setter
    def animal_location(self, value: Union[str, Path]):
        value = Path(value).as_posix()
        prev, self._animal_location = self._animal_location, value
        self._on_property_changed(self.ANIMAL_LOCATION, value, prev)
        self._settings.setValue("system/animal_location", value)

    @property
    def log_level(self) -> int:
        return self._log_level

    @log_level.setter
    def log_level(self, value: int):
        prev, self._log_level = self._log_level, value
        self._settings.setValue("system/log_level", value)
        self._on_property_changed(self.LOG_LEVEL, value, prev)

    # Transient Values

    @property
    def pellet_port(self) -> str:
        return self._pellet_port

    @pellet_port.setter
    def pellet_port(self, value: str):
        prev, self._pellet_port = self._pellet_port, value
        self._on_property_changed(self.PELLET_PORT, value, prev)

    @property
    def measurement_graph(self) -> str:
        return self._measurement_graph

    @measurement_graph.setter
    def measurement_graph(self, value: str) -> None:
        prev, self._measurement_graph = self._measurement_graph, value
        self._settings.setValue("ui/measurement_graph", value)
        self._on_property_changed(self.MEASUREMENT_GRAPH, value, prev)

    @property
    def softmouse_manifest_path(self) -> str:
        return self._softmouse_manifest_path

    @softmouse_manifest_path.setter
    def softmouse_manifest_path(self, value: str) -> None:
        value = value.strip() or DEFAULT_SOFTMOUSE_MANIFEST_PATH.as_posix()
        prev, self._softmouse_manifest_path = self._softmouse_manifest_path, value
        self._settings.setValue("softmouse/manifest_path", self._softmouse_manifest_path)
        self._on_property_changed(self.SOFTMOUSE_MANIFEST_PATH, self._softmouse_manifest_path, prev)

    @property
    def softmouse_name_column(self) -> str:
        return self._softmouse_name_column

    @softmouse_name_column.setter
    def softmouse_name_column(self, value: str) -> None:
        value = value.strip() or "Physical Tag"
        prev, self._softmouse_name_column = self._softmouse_name_column, value
        self._settings.setValue("softmouse/new_animal_name_column", value)
        self._on_property_changed(self.SOFTMOUSE_NAME_COLUMN, value, prev)

    @property
    def softmouse_nightly_refresh(self) -> bool:
        return self._softmouse_nightly_refresh

    @softmouse_nightly_refresh.setter
    def softmouse_nightly_refresh(self, value: bool) -> None:
        value = bool(value)
        prev, self._softmouse_nightly_refresh = self._softmouse_nightly_refresh, value
        self._settings.setValue("softmouse/nightly_refresh", value)
        self._on_property_changed(self.SOFTMOUSE_NIGHTLY_REFRESH, value, prev)

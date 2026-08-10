import logging
import platform
import sys
from pathlib import Path
from typing import Optional, Union

from PySide6.QtCore import QByteArray, QCoreApplication, QSettings

from autotrainer.core import ObservableObject
from autotrainer.core.logging import get_verbose_logger
from autotrainer.core.configuration import SystemConfiguration


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
    TUNNEL_PORT = "tunnel_port"
    MEASUREMENT_GRAPH = "measurement_graph"

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

        self._selected_animal: str = settings.value("system/selected_animal", "")  # noqa

        self._log_location: str = settings.value("system/log_location", "")  # noqa
        self._log_level: int = settings.value("system/log_level", logging.WARNING, int)  # noqa

        self._live_feed_refresh_rate: int = settings.value("display/refresh_rate", 15, int)  # noqa
        self._measurement_graph: str = settings.value("ui/measurement_graph", "")  # noqa

        # Transient values that may come from individual configuration files, but are conveniently accessed from
        # the user preferences.

        self._tunnel_port = None
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
        self._settings.setValue("system/selected_animal", value)
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
    def tunnel_port(self) -> str:
        return self._tunnel_port

    @tunnel_port.setter
    def tunnel_port(self, value: str):
        prev, self._tunnel_port = self._tunnel_port, value
        self._on_property_changed(self.TUNNEL_PORT, value, prev)

    @property
    def measurement_graph(self) -> str:
        return self._measurement_graph

    @measurement_graph.setter
    def measurement_graph(self, value: str) -> None:
        prev, self._measurement_graph = self._measurement_graph, value
        self._settings.setValue("ui/measurement_graph", value)
        self._on_property_changed(self.MEASUREMENT_GRAPH, value, prev)

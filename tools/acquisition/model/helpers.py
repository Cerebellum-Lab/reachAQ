
from pathlib import Path
from typing import Optional

from autotrainer.core import get_verbose_logger
from autotrainer.core.configuration import SystemConfiguration
from tools.acquisition.model.user_preferences import get_default_configuration_location, UserPreferences

logger = get_verbose_logger(__name__)


def first_line(text, maximum_length: Optional[int] = None) -> str:
    """The first non-empty line of `text`, cut to `maximum_length` with an ellipsis."""
    line = next((part.strip() for part in str(text).splitlines() if part.strip()), "")
    if maximum_length is not None and len(line) > maximum_length:
        line = line[: maximum_length - 1].rstrip() + "…"
    return line


def get_config_location(preferences: UserPreferences, location: Optional[str] = None) -> Path:
    if location is None:
        # Check to see if there is a file in the new default location.  If so, use it.
        path = Path(preferences.configuration_location)
        logger.info(
            "did not receive explicit configuration file, trying default p_location=%s",
            path,
        )
        default_path = SystemConfiguration.make_default_yaml_config_path(path)
        if default_path.is_file():
            return default_path
        file_path = Path(preferences.last_configuration)
        if file_path.is_file():
            return file_path
        default_path = SystemConfiguration.make_default_yaml_config_path(
            get_default_configuration_location()
        )
        return default_path
    path = Path(location)
    if path.is_dir():
        return SystemConfiguration.make_default_yaml_config_path(path)
    return path

from dataclasses import dataclass
from typing import ClassVar

from pathlib import Path

from autotrainer.core import make_camelize_representer, make_decamelize_constructor


@dataclass
class PersistenceConfiguration:

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path("~/Documents/rawdatalocal")  # must use .expanduser() on it

    output_location: str = DEFAULT_OUTPUT_PATH.expanduser().as_posix()

    @classmethod
    def get_default_output_path(cls) -> Path:
        return cls.DEFAULT_OUTPUT_PATH.expanduser()

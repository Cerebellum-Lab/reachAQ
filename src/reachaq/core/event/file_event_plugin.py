import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO

from ..logging import get_verbose_logger
from ..project import ProjectInfo, ProjectInterval
from .event_info import EventInfo

from .event_manager_plugin import EventManagerPlugin

logger = get_verbose_logger(__name__)


class FileEventPlugin(EventManagerPlugin):
    """
    An event manager plugin directs events to a project file.  This plugin will nore record an event history if
    `project` is set to `None`.
    """

    def __init__(self):
        self._project_info: Optional[ProjectInfo] = None
        self._have_new_project = False
        self._event_file: Optional[TextIO] = None
        self._dict_writer: Optional[csv.DictWriter] = None
        self._write_active = True
        self._current_record_interval = -1
        self._bad_file_attempt = False

    def set_project(self, project: Optional[ProjectInfo]) -> None:
        self._project_info = project
        self._bad_file_attempt = False
        self._have_new_project = True

    def set_enable(self, enable: bool) -> None:
        self._write_active = enable

    def process_event(self, info: EventInfo, repeat_count: int):
        if not self._write_active:
            logger.debug("write not active, skipping %s", info)
            return
        file_timestamp = datetime.now()
        needs_update = (
            self._have_new_project
            or file_timestamp.hour != self._current_record_interval
        )
        if needs_update:
            self._update_event_file()
        event_file = self._event_file
        if event_file is None:
            logger.warning("event_file None, skipping %s", info)
            return
        dict_writer = self._dict_writer
        if dict_writer is not None:
            # ["Time" , "Index", "EventId", "EventName", "Data", "Repeat"]
            dict_writer.writerow(dict(
                Time=info.when,
                Index=info.index,
                EventId=int(info.kind),
                EventName=str(info.kind),
                Data=str(info.context),
                Repeat=repeat_count,
            ))

    def flush(self):
        fh = self._event_file
        if fh is not None:
            try:
                fh.flush()
            except IOError as err:
                # this could happen if another thread is closing the file at the same time
                logger.verbose("flush on %s failed: %s", fh, err)

    def close(self) -> None:
        event_file = self._event_file
        self._event_file = None
        self._dict_writer = None
        if event_file is not None:
            logger.debug("closing %s", event_file.name)
            event_file.close()

    def _update_event_file(self):
        self.close()
        if not self._write_active:
            return
        project = self._project_info
        self._have_new_project = False
        if project is not None:
            event_file_info = project.get_monitor_file(name="events", interval=ProjectInterval.HOUR,
                                                       when=datetime.now())
            if event_file_info is None:
                logger.error("unable to write to expected event file location")
                return

            try:
                file_path = Path(event_file_info.file)
                file_existed = file_path.exists()
                file_path.parent.mkdir(exist_ok=True, parents=True)
                fh = file_path.open("a")
                dict_writer = csv.DictWriter(
                    fh,
                    fieldnames=["Time" , "Index", "EventId", "EventName", "Data", "Repeat"],
                    quotechar='"',
                    escapechar='\\',
                    quoting=csv.QUOTE_NONNUMERIC,
                )
                if not file_existed:
                    dict_writer.writeheader()
                    fh.flush()
                self._current_record_interval = event_file_info.current_interval
                self._event_file = fh
                self._dict_writer = dict_writer
                logger.info(f"event file opened at {event_file_info.file}")
            except Exception as err:
                if not self._bad_file_attempt:
                    # Don't spam log if there are write/access issues with wherever project info is pointing.  Can reset
                    # the flag the next tile the project changes.
                    logger.error("unable to write to %s: %s", event_file_info.file, err)
                    self._bad_file_attempt = True

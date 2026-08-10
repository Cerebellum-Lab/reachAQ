from __future__ import annotations

import ctypes
import dataclasses
import logging
import math
import multiprocessing.managers
import os
import os.path
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from typing import Tuple, NamedTuple, Union, Optional, ClassVar, Sequence

from typing_extensions import Self

from autotrainer.core import (
    ValueHolderDescriptor,
    RawValueHolder,
    get_verbose_logger,
    Offset3DTuple,
)
from autotrainer.core.multiproc import get_mp_ctx

logger = get_verbose_logger(__name__)


DATE_FORMAT = "%Y%m%d"

TIME_FORMAT = "%H%M%S"

DATE_TIME_FORMAT = f"{DATE_FORMAT}_{TIME_FORMAT}"

MINUTE_INTERVAL_FORMAT = "h%Hm%M"

HOUR_INTERVAL_FORMAT = "h%H"

IMAGE_CAPTURE_SUFFIX = "_images"

REACH_EVENT_SUFFIX = "_reach_events.h5"


class ProjectInterval(IntEnum):
    NONE = -1
    MINUTE = 0
    HOUR = 1


class ProjectPath(NamedTuple):
    location: str
    prefix: str
    full_path: str


class IntervalSource(NamedTuple):
    location: str
    prefix: str
    interval: int


class SessionSource(NamedTuple):
    location: str
    prefix: str
    session_index: int


class IntervalFileInfo(NamedTuple):
    location: str
    file: str
    current_interval: int


def _ensure_location(location: str):
    try:
        path = Path(location)
        path.mkdir(parents=True, exist_ok=True)
    except Exception as err:
        logger.error("Could not create dir %r: %s", location, err)
        raise




# NB: so that we can easily patch from test
def _get_datetime_now() -> datetime:
    return datetime.now()


@dataclass
class _ProjectInfo:
    """Contains all details needed to uniquely identify a session project.

    The ProjectInfo is used/shared possibly to multiple processes.
    It will only have this multiprocess capability if-and-only-if it's created without specifying the `when` and
     `session` attributes. Or if they are both set to None.
    If `when` is provided but not `session` then `session` defaults to 1.
    Providing `session` but not `when` is an error.
    """

    # Windows does not like .mp4 extension when opencv is technically saving to an mkv container.
    video_write_ext: ClassVar[str] = (
        "mp4" if sys.platform.startswith("linux") else "mkv"
    )

    root: str = ""
    device_id: str = ""
    _when: Union[Synchronized[ctypes.c_double], RawValueHolder] = None
    when: ClassVar[datetime] = ValueHolderDescriptor(  # noqa
        convert_from=datetime.fromtimestamp,
        convert_to=lambda v: v.timestamp(),
    )
    ensure_exists: bool = False
    camera_1: str = ""
    camera_2: str = ""
    camera_names: Tuple[str, ...] = ()
    _session: Union[Synchronized[ctypes.c_uint32], RawValueHolder] = None
    session: ClassVar[int] = ValueHolderDescriptor()  # noqa
    send_position: Optional[Offset3DTuple] = None
    dcs_send_position: Optional[Offset3DTuple] = None
    start_record_timestamp: float = math.nan  # regular unix timestamp, in seconds
    t_pellet_delivered: float = math.nan  # in seconds (zero-based on start_recording)
    t_pellet_presented: float = math.nan


@dataclass
class ProjectInfo(_ProjectInfo):

    # custom/overloaded init to allow normal/exact same than previous argument names
    def __init__(
        self,
        *,
        root: str = _ProjectInfo.root,
        device_id: str = _ProjectInfo.device_id,
        when: Optional[datetime] = None,
        ensure_exists: bool = _ProjectInfo.ensure_exists,
        camera_1: str = _ProjectInfo.camera_1,
        camera_2: str = _ProjectInfo.camera_2,
        camera_names: Optional[Sequence[str]] = None,
        session: Optional[int] = None,
        send_position: Optional[Offset3DTuple] = _ProjectInfo.send_position,
        dcs_send_position: Optional[Offset3DTuple] = _ProjectInfo.dcs_send_position,
        start_record_timestamp: float = _ProjectInfo.start_record_timestamp,
        t_pellet_delivered: float = _ProjectInfo.t_pellet_delivered,
        t_pellet_presented: float = _ProjectInfo.t_pellet_presented,
        #
        mp_manager: Optional[multiprocessing.managers.BaseManager]=None,
    ):
        super().__init__()
        if when is None:
            _when = None
        else:
            _when = RawValueHolder(when.timestamp())
            if session is None:
                session = 1
        if session is None:
            _session = None
        else:
            _session = RawValueHolder(session)
            if when is None:
                raise ValueError("Cannot create ProjectInfo with session but without when")
        if _session is None and _when is None:
            ctx = get_mp_ctx() if mp_manager is None else mp_manager
            _session = ctx.Value(ctypes.c_uint32, 1)
            # use the same lock for both session and when mp shared values:
            _when = ctx.Value(ctypes.c_double,  # double required, not float !!
                                   _get_datetime_now().timestamp(),
                                   # with mp_manager on 3.8 we would need to access private _getvalue
                                   # lock=(session_shared_obj if mp_manager is None
                                   #       else session_shared_obj._getvalue()).get_lock()
                                   # see:
                                   )
        self.root = root
        self.device_id = device_id
        self._when = _when
        self._session = _session
        self.ensure_exists = ensure_exists
        if camera_names is None:
            camera_names = tuple(name for name in (camera_1, camera_2) if name)
        else:
            camera_names = tuple(camera_names)
            if len(camera_names) > 0 and not camera_1:
                camera_1 = camera_names[0]
            if len(camera_names) > 1 and not camera_2:
                camera_2 = camera_names[1]
        self.camera_1 = camera_1
        self.camera_2 = camera_2
        self.camera_names = tuple(camera_names)
        self.send_position = send_position
        self.dcs_send_position = dcs_send_position
        self.start_record_timestamp = start_record_timestamp
        self.t_pellet_delivered = t_pellet_delivered
        self.t_pellet_presented = t_pellet_presented

    @property
    def short_id(self) -> str:
        return f"{self.when.strftime(DATE_FORMAT)}_{self.device_id}_{self.session:03d}"

    # def __repr__(self):
    #     return (
    #         f"{self.__class__.__name__}(device={self.device_id!r}, session={self.session!r}, when={self.when!r}, "
    #         f"send_pos={self.send_position}, dcs_send_pos={self.dcs_send_position})"
    #     )

    def __eq__(self, other):
        if isinstance(other, ProjectInfo):
            return (
                self.root == other.root
                and self.device_id == other.device_id
                and self.when == other.when
                and self.camera_1 == other.camera_1
                and self.camera_2 == other.camera_2
                and self.camera_names == other.camera_names
                and self.session == other.session
            )
        return super().__eq__(other)

    # for multiprocess capability:
    def __enter__(self):
        if hasattr(self._session, "acquire"):
            self._session.acquire()

    def __exit__(self, ex, ex_type, ex_stack):
        if hasattr(self._session, "acquire"):
            self._session.release()

    def _get_when_or_now(self, when: Optional[datetime] = None) -> datetime:
        when: datetime = self.when if when is None else when
        return when

    def is_valid(self):
        return self.root is not None and len(self.root) > 0

    def get_t_pellet_delivered_or_default(self, *, default: float=0.) -> float:
        t = self.t_pellet_delivered
        return t if math.isfinite(t) else default

    def get_t_pellet_presented_or_default(self, *, default: float=0.):
        t = self.t_pellet_presented
        return t if math.isfinite(t) else self.get_t_pellet_delivered_or_default(default=default)

    def get_day_path(self, skip_ensure: bool = False, when: Optional[datetime]=None) -> Tuple[str, str]:
        """Get the location and related datetime for given arguments. If when is None then self.when is used."""
        when: datetime = self._get_when_or_now(when)
        today = when.strftime(DATE_FORMAT)
        location = os.path.join(self.root, today)
        if self.device_id:
            location = os.path.join(location, self.device_id)

        if not skip_ensure and self.ensure_exists:
            _ensure_location(location)

        return location, today

    def get_interval(self, interval: ProjectInterval = ProjectInterval.NONE, when: Optional[datetime] = None) -> int:
        if interval == ProjectInterval.NONE:
            return -1
        r_when = self._get_when_or_now(when)
        return r_when.hour if interval == ProjectInterval.HOUR else r_when.minute

    def get_interval_path(
        self,
        name: str = "",
        interval: ProjectInterval = ProjectInterval.HOUR,
        skip_ensure: bool = False,
        when: Optional[datetime] = None,
    ) -> IntervalSource:
        when: datetime = self._get_when_or_now(when)
        time_format = HOUR_INTERVAL_FORMAT if interval == ProjectInterval.HOUR else MINUTE_INTERVAL_FORMAT
        when_str = f"_{when.strftime(time_format)}"
        location, today = self.get_day_path(skip_ensure=skip_ensure, when=when)
        s = f"_{name}" if name else ""
        d = f"_{self.device_id}" if self.device_id else ""
        prefix = f"{today}{d}{when_str}{s}"
        return IntervalSource(location, prefix, when.hour if interval == ProjectInterval.HOUR else when.minute)

    def get_session_path(self, name: str = "", session: int = -1, skip_ensure: bool = False,
                         when: Optional[datetime] = None) -> SessionSource:
        (location, today) = self.get_day_path(True, when=when)
        if session < 0:
            session = self.session
        session_str = f"session{session:03}"
        location = os.path.join(location, session_str)
        if not skip_ensure and self.ensure_exists:
            _ensure_location(location)
        d = f"_{self.device_id}" if self.device_id else ""
        prefix = f"{today}{d}_{session_str}"
        s = f"_{name}" if name else ""
        prefix = f"{prefix}{s}"
        return SessionSource(location, prefix, session)

    def get_source_path(
        self,
        name: str = "",
        interval: ProjectInterval = ProjectInterval.NONE,
        session: int = -1,
        skip_ensure: bool = False,
        when: Optional[datetime] = None,
    ) -> ProjectPath:
        if interval is None or interval == ProjectInterval.NONE:
            path = self.get_session_path(name, session=session, skip_ensure=skip_ensure, when=when)
        else:
            path = self.get_interval_path(name, interval=interval, skip_ensure=skip_ensure, when=when)
        return ProjectPath(path.location, path.prefix, os.path.join(path.location, path.prefix))

    def get_metadata_file(self, session: Optional[int] = -1, when: Optional[datetime] = None) -> str:
        """Returns the metadata file path,
            if session is None it's not scoped to a recording session.
            if < -1 then self.session is used
        """
        when: datetime = self._get_when_or_now(when)
        timestamp = when.strftime(TIME_FORMAT)
        if session is None:
            location, prefix = self.get_day_path(when=when)
            d = f"_{self.device_id}" if self.device_id else ""
            return os.path.join(location, f"{prefix}{d}_{timestamp}_metadata")
        else:
            source = self.get_session_path("metadata", session=session)
            return os.path.join(source.location, f"{source.prefix}")

    def get_monitor_file(self, name: str = "monitor", ext: str = "csv",
                         interval: ProjectInterval = ProjectInterval.HOUR,
                         when: Optional[datetime] = None) -> IntervalFileInfo:
        iv_path = self.get_interval_path(name, interval, when=when)
        return IntervalFileInfo(iv_path.location,
                                os.path.join(iv_path.location, f"{iv_path.prefix}.{ext}"),
                                iv_path.interval)

    def get_audio_spectrum_file(
        self,
        name: str = "spectrum",
        ext: str = "csv",
        interval: ProjectInterval = ProjectInterval.HOUR,
        when: Optional[datetime] = None
    ) -> IntervalFileInfo:
        audio_path = self.get_interval_path(name, interval, when=when)
        return IntervalFileInfo(audio_path.location,
                                os.path.join(audio_path.location, f"{audio_path.prefix}.{ext}"),
                                audio_path.interval)

    def get_video_path(
        self,
        name: str = "",
        interval: ProjectInterval = ProjectInterval.NONE,
        session: int = -1,
        allow_overwrite: bool = False,
    ) -> Tuple[str, str, str]:
        """Get the 3-tuple of video paths for given arguments"""
        vid_path = self.get_source_path(name, interval=interval, session=session)
        file_name = f"{vid_path.full_path}.{self.video_write_ext}"
        index = 0

        if not allow_overwrite:
            while os.path.exists(file_name):
                index += 1
                file_name = f"{vid_path.full_path}_{index}.{self.video_write_ext}"

        modifier = "" if index == 0 else "_" + str(index)

        ts_file = f"{vid_path.full_path}_timestamps{modifier}.txt"
        frames_processed_indices_file = f"{vid_path.full_path}_processed_frames.txt"

        return file_name, ts_file, frames_processed_indices_file

    def get_image_capture_path(self,
        name: str = "",
        interval: ProjectInterval = ProjectInterval.NONE,
        session: int = -1,
        when: Optional[datetime] = None,
    ) -> Tuple[Path, str]:
        """Get the 2-tuple of image paths for given arguments"""
        base = self.get_source_path(name, interval=interval, session=session, skip_ensure=True, when=when)
        image_location = os.path.join(base.location, f"{base.prefix}{IMAGE_CAPTURE_SUFFIX}")
        if self.ensure_exists:
            _ensure_location(image_location)
        image_file_format_str = base.prefix + "_{when}" + ".png"
        return Path(image_location), image_file_format_str

    def get_reach_event_path(self) -> Path:
        base = self.get_source_path("")
        return Path(base.location).joinpath(f"{base.prefix}{REACH_EVENT_SUFFIX}")

    def get_intersession_pose_path(
        self,
        name: str = "",
        session: int = -1,
        *,
        suffix: str = "",
        when: Optional[datetime] = None,
    ) -> str:
        source = self.get_source_path(name, session=session, when=when)
        return os.path.join(source.location, f"{source.prefix}_raw2D{suffix}.h5")

    def calculate_next_session_index(self, when: Optional[datetime] = None):
        """Calculate the next session index & date and store it locally"""
        self._calculate_next_session_index(when)
        logger.success("Calculated next session index=%s when=%s",
                       self.session, self.when)

    def _reset_vals(self, when, session):
        with self:
            self.when = when  # noqa
            self.session = session  # noqa
            self.start_record_timestamp = self.t_pellet_delivered = self.t_pellet_presented = math.nan

    def _calculate_next_session_index(self, when: Optional[datetime] = None):
        """Calculate the next session index & date and store it locally"""
        if when is None:
            when = _get_datetime_now()
        assert when is not None
        prev_when = self.when
        assert prev_when is not None
        location, _ = self.get_day_path(when=when)
        logger.debug("calculating next session index in %s", location)
        path = Path(location)
        existed = path.exists()
        was_dir = path.is_dir()
        path.mkdir(parents=True, exist_ok=True)  # this ensures 2 consecutive won't get same
        if not existed or not was_dir:
            tentative_p, _, _ = self.get_session_path(session=1, when=when, skip_ensure=True)
            try:
                Path(tentative_p).mkdir(parents=True)
            except FileExistsError:
                pass
            else:
                self._reset_vals(when, 1)
                return
        if prev_when.date() < when.date():
            # actually the day directory could be already created from possible other writers to it,
            # this ensures that we should get the correct new session nbr
            new_session_nbr = 1
        else:
            new_session_nbr = self.session + 1
        tentative_p, _, _ = self.get_session_path(session=new_session_nbr, when=when, skip_ensure=True)
        tentative_p = Path(tentative_p)
        if not tentative_p.exists():
            try:
                Path(tentative_p).mkdir(parents=True)
            except FileExistsError:
                pass
            else:
                logger.info("found fast next session: %s", tentative_p)
                self._reset_vals(when, new_session_nbr)
                return

        # slower code way
        session_dirs = [
            x.name[len("session"):]
            for x in path.iterdir()
            if x.is_dir() and x.name.startswith("session")
        ]

        def int_map_fcn(value: str):
            try:
                return int(value)
            except (ValueError, TypeError):
                return None

        session_vals = [int(x) for x in session_dirs if int_map_fcn(x) is not None]
        if len(session_vals) == 0:
            logger.debug("no existing sessions found")
            self._reset_vals(when, 1)
        else:
            logger.debug("found %s existing session directories", len(session_vals))
            session_vals.sort(reverse=True)
            greater_val = session_vals[0]
            logger.info("last session index for day: %s", greater_val)
            self._reset_vals(when, greater_val + 1)

    def get_log_file_path(self, when: Optional[datetime] = None, *, auto_new: bool=True) -> Path:
        when = self._get_when_or_now(when)
        loc, today = self.get_day_path(when=when)
        today = when.strftime(DATE_FORMAT)
        dev_id = self.device_id
        s_dev = f"_{dev_id}" if dev_id else ""
        loc = Path(loc)
        fmt = f"{today}{s_dev}_{{idx}}.log"
        tot_prev_log_files = len(tuple(loc.glob(fmt.format(idx="*"))))
        idx = tot_prev_log_files
        if auto_new:
            idx += 1
        return Path(loc).joinpath(fmt.format(idx=f"{idx:03d}"))

    def get_frame_timing_path(self) -> Path:
        source = self.get_session_path("frame_timing.csv", session=self.session)
        return Path(source.location).joinpath(source.prefix)

    def to_local_value(self) -> Self:
        """Detach, if it was, from the possible shared memory values used for `when` & `session`.
        This ensures that the "detached" instance won't have its `when` and `session` values updated
         in the background by any other process attached to the shared values.
        """
        with self:
            dct = {
                field_name: getattr(self, field_name)
                for field_name, field in
                ((field.name.lstrip("_"), field) for field in dataclasses.fields(self))
            }
        return self.__class__(**dct)

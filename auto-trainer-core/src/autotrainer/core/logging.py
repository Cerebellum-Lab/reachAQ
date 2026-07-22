import copy
import dataclasses
import functools
import itertools
import logging
import logging.handlers
import multiprocessing
import operator
import signal
import threading
import time
from pathlib import Path
from logging import LogRecord
from queue import Empty
from multiprocessing import Process
from typing import Optional, Dict, Union, List

import sys
import verboselogs
import coloredlogs
from datetime import datetime

from autotrainer.core.multiproc import get_mp_ctx

logger = logging.getLogger(__name__)

_LogLevelT = Union[str, int]

_orig_logger_set_level = logging.Logger.setLevel

#

_already_setup = False
_base_logger: logging.Logger = logging.root
_multiprocess_log_queue: Optional[multiprocessing.Queue] = None
_queue_listener: Optional[logging.handlers.QueueListener] = None
_queue_handler: Optional[logging.Handler] = None
_console_handler: Optional[logging.StreamHandler] = None
_root_handler = None


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
MULTIPROC_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s[%(processName)s.%(process)d-%(threadName)s.%(thread_id)s] %(message)s"
ALWAYS_CONSOLE_LOG_ATTRIBUTE = "always_console"


# these loggers can be too verbose:
_limit_loggers_level = {
    'botocore': {
        'level': 'INFO'
    },
    'boto3': {
        'level': 'INFO'
    },
    'urllib3': {
        'level': 'INFO'
    },
    'py4j': {
        'level': 'INFO'
    },
    'h5py': {
        'level': 'INFO'
    },
    'watchdog': {
        'level': 'INFO'
    }
}


class DateTimeFormats:
    # could be an enum eventually
    hour_time_precise = "%H:%M:%S.%f"
    month_day_time_precise = f"%m/%d {hour_time_precise}"
    year_precise = f"%Y/%m/%d {hour_time_precise}"


DEFAULT_FIELD_STYLES = dict(
    asctime=dict(color='white', bold=False),
    hostname=dict(color='magenta'),
    levelname=dict(color='blue', bold=True),
    name=dict(color='cyan', bold=False),
    programname=dict(color='cyan'),
    username=dict(color='yellow'),
)


DEFAULT_LEVEL_STYLES = dict(
    spam=dict(color='white', faint=True),
    debug=dict(color='white', bold=False, faint=False),
    verbose=dict(color='white', bold=True),
    info=dict(color='blue', bold=False, faint=True),
    notice=dict(color='magenta', faint=True),
    warning=dict(color='yellow'),
    success=dict(color='green', bold=False),
    error=dict(color='red', bold=False, faint=True),
    critical=dict(color='red', bold=True),
)


@dataclasses.dataclass
class LogConfig:
    base_logger_name: Optional[str] = None  # i.e: "root" logger if None
    logger_level: _LogLevelT = logging.NOTSET
    root_level: _LogLevelT = logging.NOTSET
    log_format: str = MULTIPROC_LOG_FORMAT
    date_format: str = DateTimeFormats.hour_time_precise
    time_precision: int = 3  # for sub seconds precision, nbr of digits after the dot.
    level_styles: Dict[str, Dict[str, str]] = dataclasses.field(default_factory=lambda: copy.deepcopy(DEFAULT_LEVEL_STYLES))
    field_styles: Dict[str, Dict[str, str]] = dataclasses.field(default_factory=lambda: copy.deepcopy(DEFAULT_FIELD_STYLES))
    stream: str = "sys.stdout"


class ConsoleHandler(logging.StreamHandler):
    """A level-filtered console handler with support for explicit progress records.

    Hardware initialization can take long enough that the application otherwise
    appears hung.  Those records need to remain visible even when the user's
    normal terminal threshold is WARNING, while all other INFO/NOTICE records
    should continue to respect that preference.
    """

    def __init__(self, *args, **kwargs):
        self._configured_level = logging.NOTSET
        super().__init__(*args, **kwargs)
        # Keep the handler's actual level open so both Logger.callHandlers and a
        # QueueListener deliver tagged records to handle().
        super().setLevel(logging.NOTSET)

    def setLevel(self, level):
        self._configured_level = logging._checkLevel(level)
        super().setLevel(logging.NOTSET)

    def filter(self, record):
        if (
            record.levelno < self._configured_level
            and not getattr(record, ALWAYS_CONSOLE_LOG_ATTRIBUTE, False)
        ):
            return False
        return super().filter(record)


def make_console_handler(cfg: LogConfig):
    stream = sys.stdout
    console_handler = ConsoleHandler(stream=stream)
    console_handler.name = "console_handler"
    console_handler.addFilter(thread_id_filter)
    fmt = ColoredPreciseTimeFormatter(
        cfg.log_format,
        level_styles=cfg.level_styles,
        field_styles=cfg.field_styles,
        datefmt=cfg.date_format,
        time_precision=cfg.time_precision,
    )
    console_handler.setFormatter(fmt)
    return console_handler


def listener_command(func):
    """Relay the given func to the log queue listener proc side"""
    @functools.wraps(func)
    def wrapped(self: "LogQueueListenerProc", *args, **kwargs):
        self._send_command(func.__name__, (args, kwargs))
    return wrapped


class LogQueueListenerProc(Process):

    def __init__(
        self,
        log_queue,
        log_config: LogConfig,
    ):
        super().__init__(daemon=True)
        self._queue = log_queue
        mp_ctx = get_mp_ctx()
        self._command_queue = mp_ctx.Queue()
        self._command_executed = mp_ctx.Event()
        self._log_config = log_config
        self._listener: Optional[WithThreadIdQueueListener] = None
        self._console_handler = None
        self._file_handler: Optional[logging.FileHandler] = None

    def _send_command(self, cmd, data):
        self._command_executed.clear()
        self._command_queue.put((cmd, data))
        if not self._command_executed.wait(5):
            logger.warning("command %s timeout after 5s, skipping waiting")

    @listener_command
    def set_handler_level(self, name, level):
        """Set handler level"""

    def _set_handler_level(self, name, level):
        if name == "console_handler":
            self._console_handler.setLevel(level)
        else:
            logger.verbose("unhandled handler name: %r", name)

    @listener_command
    def set_logger_level(self, name, level):
        """Set logger level"""

    @staticmethod
    def _set_logger_level(name, level):
        logging.getLogger(name).setLevel(level)
        logger.debug("set_logger_level(%s, %s)", name, level)

    @listener_command
    def add_file_handler(self, path):
        """Add file handler to path"""

    def _add_file_handler(self, path: Path):
        logger.info("Adding file handler to %s", path)
        file_handler = logging.FileHandler(path)
        file_handler.addFilter(thread_id_filter)
        file_handler.setFormatter(
            PreciseTimeFormatter(
                MULTIPROC_LOG_FORMAT,
                datefmt=DateTimeFormats.year_precise,
                time_precision=6,
            )
        )
        file_handler.setLevel(verboselogs.SPAM + 1)  # writes everything up to DEBUG which reaches it
        self._listener.handlers += (file_handler,)
        self._file_handler = file_handler
        logger.info("logging.root.handlers=%s ; listener_handlers=%s",
                    logging.root.handlers, self._listener.handlers)
        return file_handler

    @listener_command
    def switch_file_handler(self, path: Path):
        """Switch current, or set new, file handler to path"""

    def _switch_file_handler(self, path: Path):
        logger.info("Switching file handler to %s", path)
        prev = self._file_handler
        self._add_file_handler(path)
        if prev is not None:
            logger.verbose("removing previous handler %s", prev)
            self._listener.handlers = tuple(
                handler for handler in self._listener.handlers
                if handler != prev
            )
            logger.debug("new handlers: %s", self._listener.handlers)
            prev.close()

    #

    def stop(self):
        self._command_queue.put(None)
        # os.kill(self.pid, signal.SIGINT)
        self.join()

    def run(self):
        # print(f"{logging.root.handlers}")
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        cfg = self._log_config
        #
        console_handler = self._console_handler = make_console_handler(cfg)

        # NB: start the listener as soon as possible
        listener = self._listener = WithThreadIdQueueListener(
            self._queue,
            console_handler,
            respect_handler_level=True,  # False,  # True,
        )
        listener.start()

        base_logger = get_verbose_logger(cfg.base_logger_name)
        base_logger.addHandler(console_handler)

        base_logger.setLevel(logging.INFO)
        base_logger.info("Starting log queue listener. config=%s", self._log_config)
        base_logger.setLevel(cfg.root_level)

        # NB: must be installed AFTER the listener is started
        install_log_exception_hook()

        command_q = self._command_queue
        command_executed = self._command_executed.set
        while True:
            try:
                data = command_q.get()
            except Empty:
                continue
            if data is None:
                break
            cmd = data[0]
            args, kwargs = data[1]
            meth = getattr(self, f"_{cmd}", None)
            if meth is None:
                logger.warning("unknown command: %sr", cmd)
                command_executed()
                continue
            try:
                meth(*args, **kwargs)
            except Exception as err:
                logger.error("Failed executing cmd %r: %s", cmd, err)
            finally:
                command_executed()
        # end while True
        listener.stop()


def main_exception_hook(exc_type, exc_value, exc_traceback):
    logger.exception("Fatal unhandled main-thread exception: %s",
                     exc_value, exc_info=(exc_type, exc_value, exc_traceback))


def thread_exception_hook(arg):
    logger.exception("Fatal unhandled thread exception: %s", arg.exc_value)


def install_log_exception_hook():
    sys.excepthook = main_exception_hook
    threading.excepthook = thread_exception_hook


def get_root_handler():
    return _root_handler


def get_console_handler() -> logging.Handler:
    return _console_handler


def get_multiprocess_log_queue() -> Optional[multiprocessing.Queue]:
    return _multiprocess_log_queue


def get_log_queue_listener() -> Optional[LogQueueListenerProc]:
    return _queue_listener


def get_log_queue_handler():
    return _queue_handler


class ThreadIdFilter(logging.Filter):

    def filter(self, record):
        """Inject thread_id to log records"""
        if not hasattr(record, "thread_id"):
            record.thread_id = threading.get_native_id()
        return True


thread_id_filter = ThreadIdFilter()


class PreciseTimeFormatter(logging.Formatter):
    """A logger formatter with time precision handling"""

    converter = datetime.fromtimestamp

    def __init__(self, *args, time_precision: int = 3, **kwargs):
        self._time_precision = time_precision
        super().__init__(*args, **kwargs)

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if not datefmt:
            datefmt = DateTimeFormats.year_precise  # "%Y-%m-%d %H:%M:%S.%f"
        if self._time_precision > 0 and "%f" in datefmt:
            msec_len = len(str(int(record.msecs)))
            v = str(record.msecs).replace(".", "")
            v0 = "0" * (3 - msec_len)
            v = (v0 + v)[:self._time_precision]
        else:
            v = ""
        with_dot = ".%f" in datefmt
        rep = f".%f" if with_dot and self._time_precision == 0 else "%f"
        datefmt = datefmt.replace(rep, v)
        s = ct.strftime(datefmt)
        return s


class ColoredPreciseTimeFormatter(PreciseTimeFormatter, coloredlogs.ColoredFormatter):
    """A colored logger formatter with time precision handling"""


def stop_multiproc_logging():
    global _multiprocess_log_queue, _queue_listener, _queue_handler, _console_handler, _already_setup

    # always:
    logging.Logger.setLevel = _orig_logger_set_level

    mp_log_queue = _multiprocess_log_queue
    q_listener = _queue_listener
    q_handler = _queue_handler

    if _already_setup and q_listener is not None:
        _already_setup = False
        # Must remove the handler before closing it:
        for handler in _base_logger.handlers:
            if isinstance(handler, WithThreadIdQueueHandler):
                _base_logger.removeHandler(handler)

    if q_listener is not None:
        # must be before following log queue close()
        q_listener.stop()
        _queue_listener = None

    if mp_log_queue is not None:
        mp_log_queue.close()
        mp_log_queue.join_thread()
        _multiprocess_log_queue = None

    if q_handler is not None:
        q_handler.close()
        _queue_handler = None


def repr_all_loggers():
    vals = []
    for k, v in itertools.chain(dict(root=logging.root).items(), logging.Logger.manager.loggerDict.items()):
        if not isinstance(v, logging.PlaceHolder):
            vals.append(
                f"+ [{k.ljust(20)}] {v.__class__.__name__} lvl={v.level} prop={v.propagate} disabled={v.disabled}")
            for h in v.handlers:
                if not isinstance(v, logging.PlaceHolder):
                    vals.append(f" +++ {h.name}[{h.__class__.__name__}] lvl={h.level}")
    return "\n".join(vals)


def repr_logger(obj: logging.Logger):
    return (f"{obj.__class__.__name__}({obj.name} lvl={obj.level} prop={obj.propagate} disabled={obj.disabled} "
            f"handlers={obj.handlers})")


def repr_handler(obj: logging.Handler):
    return f"{obj.__class__.__name__}({obj.name} lvl={obj.level})"


class WithThreadIdQueueListener(logging.handlers.QueueListener):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._2_sorter = []
        self._q_2_sorter_lock = threading.Lock()
        self._delay_buffer = 0.25  # aggregate records up to that age, and only output when older than that, sorted
        self._recheck_delay = 0.1  # delay between re-aggregate/sorts
        self._sorter_thread = threading.Thread(target=self._sorter, daemon=True)
        self._sorter_thread.start()

    def _get_2_sorter(self):
        with self._q_2_sorter_lock:
            buff = self._2_sorter
            self._2_sorter = []
        return buff

    def _sorter(self):
        buffer: List[LogRecord] = []
        recheck_delay = self._recheck_delay
        p_next_sort = time.perf_counter() + recheck_delay
        get_created = operator.attrgetter("created")
        while True:
            new_recs = self._get_2_sorter()
            want_quit = False
            while any(r is None for r in new_recs):
                want_quit = True
                new_recs.remove(None)
            buffer.extend(new_recs)
            p_now = time.perf_counter()
            if p_now < p_next_sort and not want_quit:
                if len(new_recs) == 0:
                    time.sleep(0.005)
            else:
                p_next_sort = p_now + recheck_delay
                buffer = sorted(buffer, key=get_created)
                t_now = time.time()
                idx = 0
                for record in buffer:
                    if t_now - record.created < self._delay_buffer and not want_quit:
                        break
                    self._handle(record)
                    idx += 1
                del buffer[:idx]
            #
            if want_quit:
                break

    def _handle(self, record):
        for handler in self.handlers:
            if not self.respect_handler_level:
                process = True
            else:
                process = record.levelno >= handler.level
            if process:
                handler.handle(record)

    def handle(self, record):
        record = self.prepare(record)
        with self._q_2_sorter_lock:
            self._2_sorter.append(record)

    def prepare(self, record):
        thread_id_filter.filter(record)
        return record

    def stop(self):
        super().stop()
        with self._q_2_sorter_lock:
            self._2_sorter.append(None)
        self._sorter_thread.join(5)


class WithThreadIdQueueHandler(logging.handlers.QueueHandler):

    def prepare(self, record):
        record = super().prepare(record)
        thread_id_filter.filter(record)
        return record


class VerboseLoggerWithThreadId(verboselogs.VerboseLogger):
    def filter(self, record):
        return thread_id_filter.filter(record)


class RelayHandler(logging.Handler):

    def __init__(self, listener: LogQueueListenerProc):
        super().__init__()
        self.listener = listener

    def setLevel(self, level):
        self.listener.set_handler_level(self.name, level)


class LoggerWriter:
    def __init__(self, lvl):
        self._lvl = lvl

    def write(self, msg):
        logging.root.log(self._lvl, msg)

    def flush(self):
        pass


def setup_logging(
    name: str = "autotrainer",
    *,
    base_logger_name: Optional[str] = None,  # i.e: "root" logger if None
    logger_level: _LogLevelT = logging.NOTSET,
    root_level: _LogLevelT = logging.NOTSET,
    log_format: str = MULTIPROC_LOG_FORMAT,
    date_format: str = DateTimeFormats.hour_time_precise,
    time_precision: int = 3,  # for sub seconds precision, nbr of digits after the dot.
    level_styles: Optional[Dict[str, Dict[str, str]]] = None,
    field_styles: Optional[Dict[str, Dict[str, str]]] = None,
    multiprocess_enabled: bool = False,
    fork_method: str = "spawn",
) -> verboselogs.VerboseLogger:
    global _already_setup
    global _multiprocess_log_queue, _queue_listener, _queue_handler, _console_handler, _root_handler

    if _already_setup:
        return get_verbose_logger(name)

    # actually we do not require this:
    # verboselogs.install()  # thx to get_verbose_logger function above.

    if level_styles is None:
        level_styles = DEFAULT_LEVEL_STYLES
    if field_styles is None:
        field_styles = DEFAULT_FIELD_STYLES
    #
    cfg = LogConfig(
        base_logger_name=base_logger_name,
        logger_level=logger_level,
        root_level=root_level,
        log_format=log_format,
        date_format=date_format,
        time_precision=time_precision,
        level_styles=level_styles,
        field_styles=field_styles,
        # stream: TextIO = sys.stdout,
    )
    #
    stop_multiproc_logging()
    #
    # pre-set these too verbose loggers level:
    for _limit_name, v in _limit_loggers_level.items():
        logging.getLogger(_limit_name).setLevel(v["level"])
    #
    base_logger = get_verbose_logger(base_logger_name)
    # set the base logger level before creating possible dedicated subproc log handling:
    base_logger.setLevel(root_level)
    #
    if multiprocess_enabled:
        # using queue created using the desired fork method context:
        multiproc_ctx = multiprocessing.get_context(fork_method)
        _multiprocess_log_queue = log_queue = multiproc_ctx.Queue()

        listener = LogQueueListenerProc(log_queue, cfg)
        listener.start()
        _queue_listener = listener  # keep global ref to ensure it stays alive
        queue_handler = WithThreadIdQueueHandler(log_queue)
        queue_handler.name = "queue_handler"
        _queue_handler = queue_handler  # keep global ref to ensure it stays alive
        root_handler = _root_handler = queue_handler
        _console_handler = RelayHandler(listener)
        _console_handler.name = "console_handler"  # "fake" it so that it will relay to the correct handler
        logging.Logger.setLevel = lambda self, lvl: listener.set_logger_level(self.name, lvl)
    else:
        _console_handler = console_handler = make_console_handler(cfg)
        root_handler = _root_handler = console_handler

    base_logger.addHandler(root_handler)

    #
    # recursion issue atm, could be left todo:
    # sys.stdout = LoggerWriter(logging.root.info)
    # sys.stderr = LoggerWriter(logging.root.warning)

    # logger.debug("Setup logging ; base_logger=%s", repr_logger(base_logger))

    get_verbose_logger("transitions").setLevel(logger_level)
    get_verbose_logger("tools").setLevel(logger_level)
    get_verbose_logger("autotrainer").setLevel(logger_level)
    get_verbose_logger("inference_algorithms").setLevel(logger_level)

    for _limit_name, v in _limit_loggers_level.items():
        logging.getLogger(_limit_name).setLevel(v["level"])

    desired_logger = get_verbose_logger(name)
    desired_logger.setLevel(logger_level)

    # logger.info("all loggers:\n%s", repr_all_loggers())
    # logger.info("top logger: %s", repr_logger(top_logger))
    # logger.info("console: level=%s", c_h.level)
    # logger.info("queue: level=%s", q_h.level)
    # logger.info("autotrainer: level=%s prop=%s handlers=%s",
    #             top_logger.level, top_logger.propagate, top_logger.handlers)

    _already_setup = True

    install_log_exception_hook()

    return desired_logger


def set_logger_level(context: Dict[str, Union[str, int]]):
    for name, value in context.items():
        logging.getLogger(name).setLevel(value)


def make_log_dict_config(
    *,
    root_log_level: int = logging.NOTSET,
    log_queue: Optional[multiprocessing.Queue] = None,
) -> Optional[Dict]:
    # usable by logging.config.dictConfig
    if log_queue is None:
        log_queue = get_multiprocess_log_queue()
        if log_queue is None:
            return None
    dct_cfg = {
        'version': 1,
        'disable_existing_loggers': False,
        'handlers': {
            'queue': {
                'class': 'autotrainer.core.logging.WithThreadIdQueueHandler',
                'queue': log_queue,
                'level': logging.NOTSET,  # pass everything to the listener
            }
        },
        # root logger is here:
        'root': {
            'handlers': ['queue'],
            # with its own level here:
            'level': logging.NOTSET,  # root_log_level,
            # FORCE NOTSET to relay everything so that file handler can properly get DEBUG as well
        },
        # but eventual level of other loggers have to be defined here:
        'loggers': copy.deepcopy(_limit_loggers_level),
    }
    return dct_cfg


def get_verbose_logger(name: Optional[str] = None) -> VerboseLoggerWithThreadId:
    obj = logging.getLogger(name)
    if not isinstance(obj, VerboseLoggerWithThreadId):
        obj.__class__ = VerboseLoggerWithThreadId
    assert isinstance(obj, VerboseLoggerWithThreadId)
    return obj


def log_hardware_initialization(
    target_logger: logging.Logger,
    message: str,
    *args,
    level: int = logging.INFO,
    **kwargs,
) -> None:
    """Log a hardware-init milestone to every configured logging destination.

    The record is tagged so it is always shown by :class:`ConsoleHandler`, but
    it remains a normal logging record for the file logger, UI diagnostics, and
    tests.  Callers should include a state such as START, READY, SKIP, or FAILED
    in ``message`` so slow operations are easy to identify from a partial log.
    """

    extra = dict(kwargs.pop("extra", {}) or {})
    extra[ALWAYS_CONSOLE_LOG_ATTRIBUTE] = True
    target_logger.log(level, "HARDWARE INIT | " + message, *args, extra=extra, **kwargs)


def get_log_file_location(*, log_base_dir: str = "", full_format: str):
    if not log_base_dir:
        log_base_dir = Path.home().joinpath("Documents/rawdatalocal")
    else:
        log_base_dir = Path(log_base_dir)

    date_stamp = datetime.now().strftime("%Y%m%d")
    # pre-format with index=0, and create parent dir
    log_location = Path(full_format.format(
        log_location=log_base_dir,
        date_stamp=date_stamp,
        idx=0,
    ))
    log_dir = log_location.parent
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as err:
        logger.error("Failed to create directory for log location %s: %s", log_location, err)
        raise
    # then check for index:
    tot_prev_log_files = len(tuple(log_dir.glob("*.log")))
    log_location = Path(full_format.format(
        log_location=log_base_dir,
        date_stamp=date_stamp,
        idx=tot_prev_log_files + 1,
    ))

    return log_location


_prev_file_handler: Optional[logging.FileHandler] = None

def set_log_location(log_file: Path):
    global _prev_file_handler
    logger.verbose("Setting log file to %s", log_file)
    #
    q_listener = get_log_queue_listener()
    if q_listener is not None:
        q_listener.switch_file_handler(log_file)
    else:
        file_handler = logging.FileHandler(log_file)
        file_handler.addFilter(thread_id_filter)
        file_handler.setFormatter(
            PreciseTimeFormatter(
                MULTIPROC_LOG_FORMAT,
                datefmt=DateTimeFormats.year_precise,
                time_precision=6,
            )
        )
        file_handler.setLevel(verboselogs.SPAM + 1)  # writes everything up to DEBUG which reaches it
        logging.root.addHandler(file_handler)
        if _prev_file_handler is not None:
            logging.root.removeHandler(_prev_file_handler)
        _prev_file_handler = file_handler


# finally:

logger = get_verbose_logger(__name__)  # noqa

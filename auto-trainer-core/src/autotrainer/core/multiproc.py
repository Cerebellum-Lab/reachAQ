import logging.config
import multiprocessing
import signal
import threading


def get_mp_ctx():
    return multiprocessing.get_context("spawn")


_nidaq_mp_ctx = None


def get_nidaq_mp_ctx():
    """The context for NI-DAQ worker processes, which cannot use spawn.

    Once the NI-DAQmx runtime is resident in this process - which it is as
    soon as the laser loads - starting a child by spawn fails. The exec dies
    with "[Errno 14] Bad address" naming the Python executable; through
    multiprocessing that surfaces only as a child that exits 255 with no
    result, which is what the exact-task preflight reported and how the
    signal stream came to refuse to start at all.

    Measured on christielab10, in one process moments apart: spawn works
    before the laser loads and fails after it, while a forkserver works
    either side, including one first used after the laser is resident. A
    forkserver rather than plain fork because the server is a clean child
    that never loads NI-DAQmx or Qt, so every worker forks from something
    uncontaminated.

    The NI-DAQ path diverges here alone. Cameras and inference keep spawn:
    nothing has shown they need otherwise, and a start method is not
    something to change across an application on one subsystem's evidence.

    Falls back to spawn where forkserver does not exist, which is Windows.
    The fault this avoids is a Linux one, so the fallback hides nothing.
    """
    global _nidaq_mp_ctx
    if _nidaq_mp_ctx is None:
        try:
            context = multiprocessing.get_context("forkserver")
            # The server has to be up before NI-DAQmx is, because starting it
            # is itself an exec and that is the thing that breaks. Calling
            # this early is the whole point: a forkserver first started after
            # the laser has loaded fails with a broken pipe, and one started
            # before it serves fine for the rest of the process's life.
            context._name  # noqa: B018 - fail fast if the context is unusable
            from multiprocessing import forkserver
            forkserver.ensure_running()
            _nidaq_mp_ctx = context
        except Exception:
            _nidaq_mp_ctx = get_mp_ctx()
    return _nidaq_mp_ctx


class EmptyWithContext:

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DaemonTimer(threading.Timer):
    """A Timer that does not block main process exit"""

    finished: threading.Event

    def __init__(self, delay, func, args=None, kwargs=None):
        super().__init__(delay, func, args=args, kwargs=kwargs)
        self.daemon = True


def make_daemon_timer(delay, func, *args, **kwargs):
    return DaemonTimer(delay, func, *args, **kwargs)


def pool_init(log_dict_cfg=None):
    """For process pool"""
    # this is to prevent keyboard interrupted being delivered to all child processes of the main process:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # given by default this is what is done.
    # inner-import on purpose: prevent import loop:
    from autotrainer.core.logging import setup_logging, install_log_exception_hook
    if log_dict_cfg is None:
        setup_logging()
    else:
        logging.config.dictConfig(log_dict_cfg)
        install_log_exception_hook()
    logging.root.info("Initialized pool worker")


no_op_timer = make_daemon_timer(0, lambda: None)
no_op_timer.finished.set()

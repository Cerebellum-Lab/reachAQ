"""Keep the HDF5 files a process has open out of the processes it starts.

h5py opens files through HDF5's sec2 driver, which calls open(2) without
O_CLOEXEC, unlike Python's own open() (PEP 446). So every process forked and
exec'd from this one inherits the descriptor, and with it HDF5's flock on the
file: the lock belongs to the open file description, which the child shares,
and HDF5 does not unlock it at close. A later open of the file is refused with
BlockingIOError (errno 11) for as long as the child lives.

NI-DAQmx's CEIP client does exactly that from the process that uses it: the
first task the process makes and the first it closes each launch
/opt/ni-ceip/niceip under ni-wine, whose helpers live about 4 s. In v3.0.1 they
held the NI spool's lock as Stop read it, and the session lost its nidaq.h5
(christielab10, 2026-11-13).
"""

import logging
import os

logger = logging.getLogger(__name__)


def mark_close_on_exec(store) -> bool:
    """Set close-on-exec on an open h5py File's descriptor; True once it is set.

    Call it straight after the open: a process started between the open and
    this call still inherits the file (0.25-0.43 ms for a "w" open on
    christielab10). Off POSIX it does nothing and returns False. A store with
    no single descriptor, or one the call fails on, is logged and left usable:
    the file is still written, only no longer kept from child processes.
    """
    if os.name != "posix":
        return False
    try:
        os.set_inheritable(int(store.id.get_vfd_handle()), False)
    except Exception as error:
        logger.warning(
            "%s could not be marked close-on-exec, so a process started now would hold it "
            "open and keep its HDF5 lock: %s: %s",
            getattr(store, "filename", store), type(error).__name__, error)
        return False
    return True

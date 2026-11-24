import multiprocessing
import os
import sys
import faulthandler
import signal

# NB: do not put any imports of reachaq* or any module not part from standard python lib.

def _exec_main(args, logger):

    from reachaq.app.run_acquisition import run_acquisition

    exit_val = run_acquisition(args)
    (logger.success if exit_val in (0, None) else logger.error)("application finished ; exit_val=%s", exit_val)
    return exit_val


def enable_fault_dumps():
    """Crash tracebacks, and every thread's stack on SIGUSR1."""
    faulthandler.enable()
    # A frozen app can be diagnosed without a debugger: `kill -USR1 <pid>`
    # prints every thread's stack to stderr (the launcher's terminal).
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)


def main():
    enable_fault_dumps()
    fork_method = "spawn"  # please check python multiprocessing fork method documentation
    multiprocessing.set_start_method(fork_method)  # MUST BE SET VERY EARLY BEFORE MOST IMPORTS

    from reachaq.app.args import make_reachaq_parser, parse_reachaq_args

    args = parse_reachaq_args(allow_dev_mode=True)

    # Before the lock and logging, which would make a fresh ~/reachAQ beside the
    # operator's unmigrated ~/Autotrainer. A simulation keeps its own home.
    if not args.simulate:
        from reachaq.core.settings_migration import UNMIGRATED_SETTINGS_EXIT, unmigrated_settings_message
        refusal = unmigrated_settings_message()
        if refusal:
            print(refusal, file=sys.stderr)
            return UNMIGRATED_SETTINGS_EXIT

    # After argument parsing, so -h still answers while an instance runs, and
    # before logging, so a refused start leaves no log behind.
    from reachaq.app.instance_lock import (
        ALREADY_RUNNING_EXIT, AlreadyRunning, acquire_instance_lock)
    simulation = None
    try:
        if args.simulate:
            # Takes the lock in its own home before writing there, and runs
            # before AppModel starts the forkserver, which keeps the
            # environment it starts with.
            from reachaq.simulation.sim_home import prepare_simulation
            simulation = prepare_simulation(args)
            instance_lock = simulation.instance_lock  # noqa: F841 - held until exit
        else:
            instance_lock = acquire_instance_lock()  # noqa: F841 - held until exit
    except AlreadyRunning as err:
        print(err, file=sys.stderr)
        return ALREADY_RUNNING_EXIT

    # import reachaq only AFTER having set mp start method,
    # otherwise it can be set by some other 3rd party dependency.
    from reachaq.core.logging import setup_logging, stop_multiproc_logging, repr_all_loggers

    app_start_log_level = os.getenv("REACHAQ_LOG_LEVEL", "NOTSET")
    if app_start_log_level.isdigit():
        app_start_log_level = int(app_start_log_level)

    # BEFORE setup_logging: after it, a level set in this process no longer takes effect.
    from reachaq.app.can_frame_logging import limit_can_frame_logging, log_can_frame_logging
    can_frame_logging = limit_can_frame_logging()

    logger = setup_logging(
        "reachaq",
        logger_level=app_start_log_level,
        time_precision=6,
        multiprocess_enabled=True,
        fork_method=fork_method,
    )
    log_can_frame_logging(logger, can_frame_logging)

    try:
        return _exec_main(args, logger)
    except Exception as err:
        logger.exception("Fatal error: %s", err)
        return 1
    finally:
        stop_multiproc_logging()


if __name__ == '__main__':
    sys.exit(main())

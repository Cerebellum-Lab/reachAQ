import os
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent

# This is established before importing any application modules. Individual
# transport codec/backend tests may construct mocked hardware configurations,
# but no pytest process may select or reset a host CAN interface implicitly.
os.environ["AUTOTRAINER_CAN_TRANSPORT"] = "emulation"
os.environ["REACHAQ_PROHIBIT_PRIVILEGED_CAN_RESET"] = "1"
for _source_dir in reversed((
    _REPO_ROOT,
    _REPO_ROOT / "auto-trainer-behavior" / "src",
    _REPO_ROOT / "auto-trainer-core" / "src",
    _REPO_ROOT / "auto-trainer-device" / "src",
    _REPO_ROOT / "auto-trainer-inference" / "src",
    _REPO_ROOT / "auto-trainer-model" / "src",
    _REPO_ROOT / "auto-trainer-pyside" / "src",
    _REPO_ROOT / "auto-trainer-video" / "src",
)):
    _source_text = str(_source_dir)
    if _source_text in sys.path:
        sys.path.remove(_source_text)
    sys.path.insert(0, _source_text)


pytest_plugins = [
    # NB: having to use another name than "fixtures(.py)" given otherwise it's overridden by:
    # ./auto-trainer-core/tests/fixtures
    # notably/already.
    "top_fixtures",
]


def pytest_addoption(parser):
    parser.addoption(
        "--functional", action="store_true", default=False, help="run functional tests"
    )
    parser.addoption(
        "--canbus", action="store_true", default=False, help="run canbus tests"
    )
    parser.addoption(
        "--run-bench", action="store_true", default=False, help="run bench tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "functional: mark test as functional")
    config.addinivalue_line("markers", "canbus: mark test as requiring canbus hardware")
    config.addinivalue_line("markers", "bench: mark test as performance/benchmark only")


def pytest_collection_modifyitems(config, items):
    def has_option(mark_name, with_no_run: bool = False):
        return (
            mark_name in (config.getoption("-m") or "").split(" ")
            or config.getoption(f"--{mark_name}" if with_no_run else f"--run-{mark_name}")
        )
    include_functional = has_option("functional", with_no_run=True)
    include_canbus = has_option("canbus", with_no_run=True)
    include_bench = has_option("bench")

    skip_functional = pytest.mark.skip(reason="need --functional option to run, or -m functional")
    skip_canbus = pytest.mark.skip(reason="need --canbus option to run, or -m canbus")
    skip_bench = pytest.mark.skip(reason="need --run-bench option to run, or -m bench")

    for item in items:
        if "functional" in item.keywords and not include_functional:
            item.add_marker(skip_functional)
        if "canbus" in item.keywords and not include_canbus:
            item.add_marker(skip_canbus)
        if "bench" in item.keywords and not include_bench:
            item.add_marker(skip_bench)


def pytest_unconfigure(config):
    """CI only: report what is alive as pytest hands over to interpreter exit.

    The job once hung after the summary. A hang there is an exit blocked on a
    child that never died, so name the children and the process tree. The dump
    below fires only if this process is still alive 5 minutes from now; a healthy
    exit takes about 10 s and cancels it. Nothing here may fail a healthy run.
    """
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    import faulthandler
    import multiprocessing
    import subprocess
    import threading

    out = sys.__stderr__ or sys.stderr
    try:
        print("\n=== exit diagnostics ===", file=out)
        for proc in multiprocessing.active_children():
            print(f"child {proc.name} pid={proc.pid} daemon={proc.daemon}", file=out)
        print(
            "non-daemon threads:",
            [t.name for t in threading.enumerate() if not t.daemon and t is not threading.main_thread()],
            file=out,
        )
        out.flush()
        subprocess.run(
            ["ps", "--forest", "-u", str(os.getuid()), "-o", "pid,ppid,etimes,stat,wchan:20,comm,args"],
            stdout=out,
            check=False,
            timeout=30,
        )
    except Exception as error:  # a diagnostic must never change the exit status
        print(f"exit diagnostics failed: {error!r}", file=out)
    faulthandler.dump_traceback_later(300, file=out)

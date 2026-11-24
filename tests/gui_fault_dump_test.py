import os
import signal
import subprocess
import sys

import pytest


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="SIGUSR1 is POSIX only")
def test_sigusr1_prints_every_thread_stack_and_keeps_running():
    script = (
        "import os, signal, threading, time\n"
        "from reachaq.app.gui import enable_fault_dumps\n"
        "enable_fault_dumps()\n"
        "threading.Thread(target=time.sleep, args=(5,), name='worker', daemon=True).start()\n"
        "os.kill(os.getpid(), signal.SIGUSR1)\n"
        "time.sleep(0.2)\n"
        "print('still running')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )

    assert result.returncode == 0
    assert "still running" in result.stdout
    assert "Current thread 0x" in result.stderr
    assert result.stderr.count("hread 0x") >= 2  # the worker and the current thread

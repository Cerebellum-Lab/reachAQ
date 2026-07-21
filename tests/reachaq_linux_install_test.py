import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = REPO_ROOT / "tools" / "install" / "reachaq-linux-install.sh"


def _run_installer(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(INSTALL_SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_portable_installer_is_executable_and_rejects_options():
    assert os.access(INSTALL_SCRIPT, os.X_OK)

    completed = _run_installer("--help")

    assert completed.returncode == 2
    assert "does not accept arguments" in completed.stderr
    assert "Run it with no options" in completed.stderr


def test_portable_installer_always_attempts_complete_workflow():
    source = INSTALL_SCRIPT.read_text()

    assert 'run_step "Install Miniconda" install_miniconda' in source
    assert 'begin_category "TensorFlow GPU runtime"' in source
    assert 'run_step "Verify TensorFlow GPU preflight"' in source
    assert 'run_step "Run focused non-hardware tests"' in source
    assert "[options]" not in source
    assert "--skip-" not in source
    assert "--install-tensorflow-gpu" not in source
    assert "grep -q 'git lfs pre-push'" in source


def test_documented_conda_application_launches_stream_terminal_output():
    launch_lines = []
    for document in REPO_ROOT.rglob("*.md"):
        for line_number, line in enumerate(document.read_text().splitlines(), start=1):
            if "conda run" in line and (
                "python -m reachAQ.app" in line or "auto-trainer-headless" in line
            ):
                launch_lines.append((document, line_number, line))

    assert launch_lines
    missing_live_output = [
        f"{document.relative_to(REPO_ROOT)}:{line_number}: {line}"
        for document, line_number, line in launch_lines
        if "--no-capture-output" not in line
    ]
    assert not missing_live_output, (
        "Long-running conda launches must use --no-capture-output for live terminal logs:\n"
        + "\n".join(missing_live_output)
    )

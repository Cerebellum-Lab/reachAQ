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


def test_portable_installer_is_executable_and_has_help():
    assert os.access(INSTALL_SCRIPT, os.X_OK)

    completed = _run_installer("--help")

    assert completed.returncode == 0
    assert "Every operational step continues after failure" in completed.stdout
    assert "--dry-run" in completed.stdout


def test_portable_installer_dry_run_reports_plan_without_changes():
    completed = _run_installer("--dry-run", "--install-miniconda")

    assert completed.returncode == 0
    assert "reachAQ portable install report" in completed.stdout
    assert "PLAN  Preflight | Validate repository checkout" in completed.stdout
    assert "Dry run complete; no changes were made." in completed.stdout


def test_portable_installer_continues_after_failure_and_reports_at_end(tmp_path):
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    completed = _run_installer(
        "--repo",
        str(tmp_path / "missing-repository"),
        "--config-dir",
        str(config_dir),
        "--data-dir",
        str(data_dir),
        "--skip-system-packages",
        "--skip-python-env",
        "--skip-git-lfs",
        "--skip-verification",
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 1
    assert config_dir.is_dir()
    assert data_dir.is_dir()
    assert "FAIL  Preflight | Validate repository checkout" in output
    assert "PASS  Preflight | Create runtime directories" in output
    assert "Completed with failures. Review every FAIL entry above." in output


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

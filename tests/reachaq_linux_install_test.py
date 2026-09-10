import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = REPO_ROOT / "tools" / "install" / "reachaq-linux-install.sh"
DESKTOP_INSTALL_SCRIPT = REPO_ROOT / "tools" / "install" / "install-reachaq-desktop.sh"
TOOLS_PROJECT = REPO_ROOT / "tools" / "pyproject.toml"
DEVICE_PROJECT = REPO_ROOT / "auto-trainer-device" / "pyproject.toml"
SOFTMOUSE_SERVICE = (
    REPO_ROOT
    / "tools"
    / "softmouse_sync"
    / "systemd"
    / "reachaq-softmouse-publisher.service"
)


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


def test_portable_installer_configures_closed_loop_latency():
    """Both settings have to reach every rig, not just the one they were tuned on.

    Measured at 900 Hz under CPU load: without the rtprio allowance the stim
    loop's worst case is 6.05 ms against a 5 ms budget; with SCHED_FIFO it is
    0.155 ms. The governor is worth p50 12.55 -> 10.79 ms on live pose
    inference, and a runtime setting does not survive a reboot.
    """
    source = INSTALL_SCRIPT.read_text()

    assert 'begin_category "Closed-loop latency tuning"' in source
    assert 'run_step "Grant real-time priority to the stim loop"' in source
    assert 'run_step "Install CPU governor unit"' in source
    assert "/etc/security/limits.d/90-reachaq-rtprio.conf" in source
    assert "rtprio   80" in source, "priority must stay below the kernel's own RT threads"


def test_latency_setup_script_is_tracked_and_reversible():
    """The two settings must be applicable without re-running the installer.

    A rig that is already installed should not have to run a full install to
    change a governor, and every action has to have a documented way back.
    """
    script = REPO_ROOT / "tools" / "hardware" / "reachaq-latency-setup.sh"
    assert script.is_file()
    text = script.read_text()

    for action in ("status", "governor", "rtprio-enable", "rtprio-disable"):
        assert action in text, action
    # Both directions, or an operator cannot undo what they applied.
    assert "powersave" in text and "performance" in text
    assert "/etc/security/limits.d/90-reachaq-rtprio.conf" in text
    # status must be readable without root, so it can be inspected first.
    assert 'action="${1:-status}"' in text


def test_cpu_governor_unit_is_tracked_and_installable():
    unit = REPO_ROOT / "tools" / "hardware" / "reachaq-cpu-governor.service"
    assert unit.is_file()
    text = unit.read_text()

    assert "performance" in text
    assert "WantedBy=multi-user.target" in text
    # oneshot + RemainAfterExit, or systemd treats the unit as dead immediately
    # and "systemctl status" reports inactive on a working machine.
    assert "Type=oneshot" in text
    assert "RemainAfterExit=yes" in text
    # Driven through sysfs so it does not depend on the versioned linux-tools
    # package matching the running kernel. Checked on the command lines rather
    # than the whole file, which explains that choice in a comment.
    commands = [line for line in text.splitlines() if line.startswith("Exec")]
    assert commands, "no Exec lines"
    assert all("scaling_governor" in line for line in commands)
    assert not any("cpupower" in line for line in commands)
    # Stopping the unit has to hand the machine back to the distro default.
    assert any(line.startswith("ExecStop") and "powersave" in line for line in commands)


def test_portable_installer_covers_softmouse_rfid_requirements():
    source = INSTALL_SCRIPT.read_text()

    for package in (
        "dbus-user-session",
        "gnome-keyring",
        "libsecret-1-0",
        "libxcb-icccm4",
        "libxcb-xinerama0",
    ):
        assert package in source

    assert 'run_step "Configure RFID serial permissions"' in source
    assert 'run_step "Verify SoftMouse runtime"' in source
    assert 'run_step "Verify RFID runtime"' in source
    assert 'run_step "Verify SoftMouse systemd units"' in source
    assert "tests/softmouse_cli_test.py" in source
    assert "tests/softmouse_https_source_test.py" in source
    assert "auto-trainer-device/tests/rfid_reader_test.py" in source
    assert "resolves outside the current checkout" in source


def test_desktop_launcher_installer_creates_terminal_entry(tmp_path):
    home = tmp_path / "home"
    desktop = home / "Desktop"
    config_dir = home / "Autotrainer"
    fake_conda = tmp_path / "conda"
    home.mkdir()
    desktop.mkdir()
    config_dir.mkdir()
    fake_conda.write_text("#!/bin/sh\nexit 0\n")
    fake_conda.chmod(0o755)
    (config_dir / "system_configuration.yaml").write_text("test: true\n")
    (desktop / "ReachAQ-startup").write_text("obsolete\n")
    env = {
        **os.environ,
        "HOME": str(home),
        "REACHAQ_CONDA_BIN": str(fake_conda),
        "REACHAQ_INSTALL_REPO": str(REPO_ROOT),
        "REACHAQ_INSTALL_CONFIG_DIR": str(config_dir),
        "REACHAQ_DESKTOP_DIR": str(desktop),
    }

    completed = subprocess.run(
        [str(DESKTOP_INSTALL_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    entry = (desktop / "reachAQ.desktop").read_text()
    assert "Terminal=true" in entry
    assert "Icon=reachaq" in entry
    assert str(home / ".local/bin/reachaq-launcher") in entry
    assert not (desktop / "ReachAQ-startup").exists()
    assert os.access(home / ".local/bin/reachaq-launcher", os.X_OK)
    installed_launcher = (home / ".local/bin/reachaq-launcher").read_text()
    assert 'lock_dir="$runtime_root/reachaq"' in installed_launcher
    assert 'exec 9<"$lock_dir"' in installed_launcher
    assert 'exec 9>"$lock_file"' not in installed_launcher
    assert "stat -c '%u'" in installed_launcher
    assert "stat -c '%a'" in installed_launcher
    launcher_config = (home / ".config/reachaq/launcher.conf").read_text()
    assert f"CONDA_BIN={fake_conda}" in launcher_config
    assert f"SYSTEM_CONFIG={config_dir / 'system_configuration.yaml'}" in launcher_config


def test_softmouse_rfid_python_dependencies_are_packaged():
    tools_project = TOOLS_PROJECT.read_text()
    device_project = DEVICE_PROJECT.read_text()

    for dependency in ("openpyxl", "requests", "keyring"):
        assert dependency in tools_project
    assert "pyserial" in device_project


def test_softmouse_service_supports_installer_conda_locations():
    service = SOFTMOUSE_SERVICE.read_text()

    assert "%h/anaconda3/bin" in service
    assert "%h/miniconda3/bin" in service
    assert "%h/mambaforge/bin" in service
    assert "conda run --no-capture-output -n reachaq" in service
    assert "/home/christielab10/anaconda3" not in service


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

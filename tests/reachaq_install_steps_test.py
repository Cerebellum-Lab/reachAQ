"""Installer steps run as shell functions, against fakes.

Each function is cut out of reachaq-linux-install.sh and run in bash after the
shared library, as the plugin tests do. dpkg-query and the helper scripts are
replaced by fakes in a scratch directory; nothing is installed.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from reachaq_plugin_install_test import MAIN_SCRIPT, PLUGIN_LIB, REPO_ROOT, _fake_conda, _make_env


def _function(name):
    source = MAIN_SCRIPT.read_text()
    body = source[source.index(f"{name}() {{"):]
    return body[:body.index("\n}\n") + 3]


PRELUDE = (
    'CURRENT_CATEGORY=Test; RESULT_NAMES=(); RESULT_STATES=(); RESULT_DETAILS=()\n'
    'PASS_COUNT=0; FAIL_COUNT=0; SKIP_COUNT=0; WARN_COUNT=0; REPORT_PRINTED=false\n'
)


def _run(script, env, *, functions=(), conda=None):
    text = (
        (f'CONDA_BIN="{conda}"\n' if conda else 'CONDA_BIN=""\n')
        # The fake conda's python is this one, so the wanted Python is too.
        + f'INSTALL_REPO="{REPO_ROOT}"\nINSTALL_ENV=reachaq\n'
        + f'INSTALL_PYTHON={sys.version_info.major}.{sys.version_info.minor}\n'
        + f'. "{PLUGIN_LIB}" || exit 99\n'
        + PRELUDE
        + "".join(_function(name) for name in functions)
        + script
    )
    return subprocess.run(["bash", "-c", text], env=env, capture_output=True, text=True,
                          check=False)


def _fake_script_dir(tmp_path, **exit_codes):
    """SCRIPT_DIR holding helper scripts that print their arguments and exit as told."""
    script_dir = tmp_path / "script_dir"
    script_dir.mkdir()
    for name, code in exit_codes.items():
        marker = tmp_path / f"{name}.ran"
        (script_dir / f"{name}.py").write_text(
            f"import sys\nopen({str(marker)!r}, 'w').close()\n"
            f"print('{name}', *sys.argv[1:])\nsys.exit({code})\n")
    return script_dir


@pytest.mark.parametrize("code,override,expected,warned", [
    (0, "0", 0, False),
    (1, "0", 1, False),
    (1, "1", 0, True),
    (2, "1", 2, False),   # a broken check is never overridden
    (3, "1", 3, False),
])
def test_platform_check_result_maps_to_the_step(tmp_path, code, override, expected, warned):
    conda, envs, env = _fake_conda(tmp_path)
    script_dir = _fake_script_dir(tmp_path, platform_check=code)

    completed = _run(
        f'SCRIPT_DIR="{script_dir}"; INSTALL_UNSUPPORTED={override}\n'
        'check_supported_platform; echo "status=$? warns=$WARN_COUNT"\n',
        env, functions=("check_supported_platform",))

    assert f"status={expected} warns={1 if warned else 0}" in completed.stdout, completed.stderr
    if expected == 1:
        assert "REACHAQ_INSTALL_UNSUPPORTED=1" in completed.stderr


def _fake_dpkg(tmp_path, env, status_line):
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    dpkg = bin_dir / "dpkg-query"
    if status_line is None:
        dpkg.write_text("#!/bin/sh\nexit 1\n", newline="\n")
    else:
        dpkg.write_text(f"#!/bin/sh\nprintf '%s\\n' '{status_line}'\n", newline="\n")
    dpkg.chmod(0o755)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"


@pytest.mark.parametrize("status_line,version", [
    ("install ok installed 3.2.0.57", "3.2.0.57"),
    ("install ok installed 1:3.2.0.57-2", "3.2.0.57"),
    ("deinstall ok config-files 3.2.0.57", ""),
    (None, ""),
])
def test_installed_sdk_version_is_flirs_number(tmp_path, status_line, version):
    conda, envs, env = _fake_conda(tmp_path)
    _fake_dpkg(tmp_path, env, status_line)

    completed = _run('echo "version=[$(spinnaker_sdk_version)]"\n', env,
                     functions=("spinnaker_sdk_version",))

    assert f"version=[{version}]" in completed.stdout, completed.stderr


@pytest.mark.parametrize("installed,expected,message", [
    ("3.2.0.57", 0, "Spinnaker SDK 3.2.0.57 is already installed."),
    ("4.0.0.116", 1, "Remove Spinnaker 4.0.0.116"),
    ("x.y", 2, "The Spinnaker version check itself failed (exit 2)"),
])
def test_an_installed_sdk_is_kept_refused_or_reported_broken(tmp_path, installed, expected,
                                                             message):
    conda, envs, env = _fake_conda(tmp_path)
    _fake_dpkg(tmp_path, env, f"install ok installed {installed}")

    completed = _run(
        f'SCRIPT_DIR="{REPO_ROOT / "tools" / "install"}"; INSTALL_SYSTEM=1\n'
        'install_spinnaker_sdk; echo "status=$?"\n',
        env, functions=("spinnaker_sdk_version", "install_spinnaker_sdk"))

    assert f"status={expected}" in completed.stdout, completed.stdout + completed.stderr
    assert message in completed.stdout + completed.stderr


def test_a_missing_sdk_without_root_steps_stops_with_the_reason(tmp_path):
    conda, envs, env = _fake_conda(tmp_path)
    _fake_dpkg(tmp_path, env, None)

    completed = _run(
        f'SCRIPT_DIR="{REPO_ROOT / "tools" / "install"}"; INSTALL_SYSTEM=0\n'
        'install_spinnaker_sdk; echo "status=$?"\n',
        env, functions=("spinnaker_sdk_version", "install_spinnaker_sdk"))

    assert "status=1" in completed.stdout
    assert "REACHAQ_INSTALL_SPINNAKER=0" in completed.stderr


def _env_with_binding(envs, version):
    root = _make_env(envs, "reachaq")
    dist = root / f"spinnaker_python-{version}.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: spinnaker-python\nVersion: {version}\n")
    return root


@pytest.mark.parametrize("binding,sdk,matches", [
    ("3.2.0.62", "3.2.0.57", True),   # the reference rig: vendored 3.2.0.62 on SDK 3.2.0.57
    ("3.1.0.79", "3.2.0.57", False),
])
def test_an_installed_binding_of_the_sdk_series_is_kept(tmp_path, binding, sdk, matches):
    conda, envs, env = _fake_conda(tmp_path)
    _env_with_binding(envs, binding)

    completed = _run(
        f'installed_spinnaker_binding_matches {sdk}; echo "status=$?"\n',
        env, conda=conda, functions=("installed_spinnaker_binding_matches",))

    assert f"status={0 if matches else 1}" in completed.stdout, completed.stderr
    log = Path(env["FAKE_CONDA_LOG"]).read_text()
    assert "pip install" not in log


def test_the_binding_check_needs_no_download_on_a_rerun(tmp_path):
    # SDK 3.2.0.57 installed, env on Python 3.10 with binding 3.2.0.62, and no
    # download anywhere: the early check passes without looking for one.
    conda, envs, env = _fake_conda(tmp_path)
    _fake_dpkg(tmp_path, env, "install ok installed 3.2.0.57")
    _env_with_binding(envs, "3.2.0.62")
    env["PATH"] = f"{conda.parent}:{env['PATH']}"
    script_dir = _fake_script_dir(tmp_path, spinnaker_download=1)

    completed = _run(
        f'SCRIPT_DIR="{script_dir}"; SPINNAKER_DOWNLOADS="{tmp_path / "none"}"\n'
        'check_spinnaker_binding_available; echo "status=$?"\n',
        env, functions=("spinnaker_sdk_version", "installed_spinnaker_binding_matches",
                        "spinnaker_python_tag", "extract_spinnaker_binding",
                        "check_spinnaker_binding_available"))

    assert "status=0" in completed.stdout, completed.stdout + completed.stderr
    assert not (tmp_path / "spinnaker_download.ran").exists()


def test_the_binding_check_asks_the_download_on_a_fresh_host(tmp_path):
    conda, envs, env = _fake_conda(tmp_path)   # no environment yet
    _fake_dpkg(tmp_path, env, "install ok installed 3.2.0.57")
    script_dir = _fake_script_dir(tmp_path, spinnaker_download=1)

    completed = _run(
        f'SCRIPT_DIR="{script_dir}"; SPINNAKER_DOWNLOADS="{tmp_path / "none"}"\n'
        'check_spinnaker_binding_available; echo "status=$?"\n',
        env, functions=("spinnaker_sdk_version", "installed_spinnaker_binding_matches",
                        "spinnaker_python_tag", "extract_spinnaker_binding",
                        "check_spinnaker_binding_available"))

    assert "status=1" in completed.stdout, completed.stdout + completed.stderr
    assert (tmp_path / "spinnaker_download.ran").exists()


def test_the_installer_refuses_to_run_under_sudo():
    source = MAIN_SCRIPT.read_text()
    guard = source.index('if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then')

    assert guard < source.index('begin_category "Preflight"')
    assert "Do not run reachaq-linux-install.sh with sudo" in source

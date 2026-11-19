#!/usr/bin/env bash

# Portable reachAQ Linux host setup.
#
# This script intentionally excludes hardware/model-specific drivers and
# configuration (FLIR Spinnaker, NI-DAQ/PXI, PEAK CAN, NVIDIA kernel drivers,
# and channel mappings). The single no-argument workflow installs every
# portable component into a Python 3.10 Conda environment: the application,
# PyTorch for the built-in keypoint backend, and all tracked verification. It
# installs no pose plugin: tools/install/reachaq-plugin-install.sh adds the
# DeepLabCut plugin (with its TensorFlow engine and that engine's CUDA
# user-space runtime, kept apart from PyTorch's) or the YOLO plugin on demand.
#
# Rerunning it is the update path. An existing environment on another Python
# version is kept, renamed <env>-py<version>-<date>, and a fresh one is built.
# Either way the pose plugins the environment had are reinstalled from this
# checkout; on the first update after the plugins existed, an environment with
# deeplabcut or ultralytics installed gets the matching plugin.
# REACHAQ_INSTALL_SYSTEM=0 skips the steps that need root (apt packages, groups,
# limits, systemd), for rerunning on a host that already has them.
#
# Do not enable `set -e`: every step must be attempted independently and the
# complete pass/fail/skip report must be printed at the end.

set -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)
if [ -z "$DEFAULT_REPO" ]; then
    DEFAULT_REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd)
fi

INSTALL_REPO=${REACHAQ_INSTALL_REPO:-$DEFAULT_REPO}
INSTALL_ENV=${REACHAQ_INSTALL_ENV:-reachaq}
INSTALL_PYTHON=${REACHAQ_INSTALL_PYTHON:-3.10}
INSTALL_SYSTEM=${REACHAQ_INSTALL_SYSTEM:-1}
INSTALL_CONFIG_DIR=${REACHAQ_INSTALL_CONFIG_DIR:-$HOME/Autotrainer}
INSTALL_DATA_DIR=${REACHAQ_INSTALL_DATA_DIR:-$HOME/Documents/rawdatalocal}
INSTALL_OPERATOR=${SUDO_USER:-${USER:-$(id -un)}}
REALTIME_GROUP=${REACHAQ_INSTALL_REALTIME_GROUP:-reachaq-rt}
REALTIME_LIMITS_FILE=/etc/security/limits.d/90-reachaq-rtprio.conf

CURRENT_CATEGORY="General"
RESULT_NAMES=()
RESULT_STATES=()
RESULT_DETAILS=()
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
REPORT_PRINTED=false
REPORT_TITLE="reachAQ portable install report"
REPORT_FOOTER="Hardware-specific kernel drivers/configuration are not installed by this script."
CONDA_BIN=""
# Pose plugins the environment had before this run, reinstalled after it.
PLUGINS_TO_RESTORE=()

if [ "$#" -ne 0 ]; then
    printf '%s\n' \
        'reachaq-linux-install.sh does not accept arguments.' \
        'Run it with no options; every install and verification category is attempted.' >&2
    exit 2
fi

# shellcheck source=plugin-install-lib.sh
if ! . "$SCRIPT_DIR/plugin-install-lib.sh"; then
    printf 'Cannot load %s; the checkout is incomplete.\n' "$SCRIPT_DIR/plugin-install-lib.sh" >&2
    exit 1
fi

trap on_exit EXIT
trap on_interrupt INT TERM

run_as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif have_command sudo; then
        sudo "$@"
    else
        printf 'Root privileges are required, but sudo is unavailable.\n' >&2
        return 127
    fi
}

apt_update() {
    run_as_root env DEBIAN_FRONTEND=noninteractive apt-get update
}

apt_install_base() {
    run_as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        can-utils \
        dbus-user-session \
        desktop-file-utils \
        dkms \
        expat \
        ffmpeg \
        git \
        git-lfs \
        gnome-keyring \
        iproute2 \
        libegl1 \
        libfontconfig1 \
        libgl1 \
        libopenal1 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libxkbcommon-x11-0 \
        libsecret-1-0 \
        pkg-config \
        v4l-utils \
        wget
}

check_repo() {
    test -f "$INSTALL_REPO/pyproject.toml" \
        && test -f "$INSTALL_REPO/requirements.txt" \
        && test -d "$INSTALL_REPO/tools/acquisition"
}

make_runtime_directories() {
    mkdir -p "$INSTALL_CONFIG_DIR" "$INSTALL_DATA_DIR"
}

ensure_realtime_priority_limits() {
    # The 900 Hz stim loop asks for SCHED_FIFO so its wake-up latency is
    # bounded. Measured on a rig under CPU load, 60 s at 900 Hz: without this
    # the worst case is 6.05 ms with 0.026% of cycles past the 5 ms budget;
    # with it, 0.155 ms and no misses. A stock install grants no rtprio at all,
    # so without this file the loop silently runs at normal priority.
    if ! getent group "$REALTIME_GROUP" >/dev/null 2>&1; then
        run_as_root groupadd --system "$REALTIME_GROUP" || return
    fi
    if ! id -nG "$INSTALL_OPERATOR" | tr ' ' '\n' | grep -qx "$REALTIME_GROUP"; then
        run_as_root usermod --append --groups "$REALTIME_GROUP" "$INSTALL_OPERATOR" || return
        printf '%s\n' \
            "$INSTALL_OPERATOR was added to $REALTIME_GROUP. Log out and back in before the stim loop can use it."
    fi
    # Priority 80 stays below the kernel's own real-time threads, which run at
    # 99, so a runaway loop cannot lock the machine out.
    printf '%s\n' \
        "# reachAQ: allow the stim-camera capture thread to run SCHED_FIFO." \
        "# Installed by tools/install/reachaq-linux-install.sh." \
        "@${REALTIME_GROUP}   -   rtprio   80" \
        "@${REALTIME_GROUP}   -   memlock  524288" \
        | run_as_root tee "$REALTIME_LIMITS_FILE" >/dev/null || return
    printf 'Wrote %s\n' "$REALTIME_LIMITS_FILE"
}

install_cpu_governor_unit() {
    # intel_pstate defaults to powersave, which on the reference workstation
    # holds the P-cores near 2700 MHz under sustained load against a 5000 MHz
    # ceiling. Measured effect on live pose inference: p50 12.55 -> 10.79 ms,
    # p99 17.58 -> 11.53 ms. Runtime changes do not survive a reboot, hence a
    # unit rather than a one-off command.
    if ! have_command systemctl; then
        printf 'systemd is unavailable; set the governor manually.\n' >&2
        return 1
    fi
    run_as_root install -m 0644 \
        "$INSTALL_REPO/tools/hardware/reachaq-cpu-governor.service" \
        /etc/systemd/system/reachaq-cpu-governor.service || return
    run_as_root systemctl daemon-reload || return
    run_as_root systemctl enable --now reachaq-cpu-governor.service || return
    printf 'Governor is now %s\n' \
        "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
}

ensure_rfid_serial_group() {
    if ! getent group dialout >/dev/null 2>&1; then
        run_as_root groupadd --system dialout || return
    fi
    if id -nG "$INSTALL_OPERATOR" | tr ' ' '\n' | grep -qx dialout; then
        printf '%s already belongs to the dialout group.\n' "$INSTALL_OPERATOR"
        return 0
    fi
    run_as_root usermod --append --groups dialout "$INSTALL_OPERATOR" || return
    printf '%s\n' \
        "$INSTALL_OPERATOR was added to dialout. Log out and back in before using the RFID reader."
}

install_miniconda() {
    local machine
    machine=$(uname -m)
    local installer_arch
    case "$machine" in
        x86_64) installer_arch=x86_64 ;;
        aarch64|arm64) installer_arch=aarch64 ;;
        *)
            printf 'Unsupported Miniconda architecture: %s\n' "$machine" >&2
            return 2
            ;;
    esac
    if [ -e "$HOME/miniconda3" ]; then
        printf '%s already exists; refusing to overwrite it.\n' "$HOME/miniconda3" >&2
        return 3
    fi
    local temporary_dir
    temporary_dir=$(mktemp -d) || return 4
    local installer="$temporary_dir/miniconda.sh"
    local url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${installer_arch}.sh"
    if have_command wget; then
        wget -O "$installer" "$url"
    elif have_command curl; then
        curl -fL -o "$installer" "$url"
    else
        printf 'wget or curl is required to install Miniconda.\n' >&2
        rmdir "$temporary_dir" 2>/dev/null || true
        return 5
    fi
    local download_result=$?
    if [ "$download_result" -eq 0 ]; then
        bash "$installer" -b -p "$HOME/miniconda3"
        download_result=$?
    fi
    rm -f "$installer"
    rmdir "$temporary_dir" 2>/dev/null || true
    return "$download_result"
}

ensure_conda_environment() {
    local wanted existing
    wanted=$(printf '%s' "$INSTALL_PYTHON" | cut -d. -f1-2)
    existing=$("$CONDA_BIN" run -n "$INSTALL_ENV" python -c \
        'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)
    if [ -n "$existing" ] && [ "$existing" = "$wanted" ]; then
        printf 'Conda environment %s already exists with Python %s.\n' \
            "$INSTALL_ENV" "$existing"
        return 0
    fi
    if [ -n "$existing" ]; then
        archive_conda_environment "$existing" || return
    fi
    "$CONDA_BIN" create -y -n "$INSTALL_ENV" "python=$INSTALL_PYTHON"
}

# An environment on another Python cannot be upgraded in place - pip refuses a
# package whose requires-python it does not meet - so keep it, renamed, and
# build fresh. The last working environment stays one `conda run -n` away.
archive_conda_environment() {
    local old_python=$1
    local archive_name
    archive_name="${INSTALL_ENV}-py${old_python//./}-$(date +%Y%m%d)"
    if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$archive_name"; then
        printf 'Cannot archive %s: %s already exists. Rename or remove it, then rerun.\n' \
            "$INSTALL_ENV" "$archive_name" >&2
        return 1
    fi
    printf 'Archiving the Python %s environment %s as %s.\n' \
        "$old_python" "$INSTALL_ENV" "$archive_name"
    "$CONDA_BIN" rename -n "$INSTALL_ENV" "$archive_name" || return
    # conda rewrites the paths inside files its own packages installed, but not
    # the first line of pip's console scripts, which still names the old
    # location - where the new environment is about to be built.
    local archive_prefix old_prefix script
    archive_prefix=$("$CONDA_BIN" run -n "$archive_name" python -c \
        'import sys; print(sys.prefix)') || return
    old_prefix="${archive_prefix%/*}/$INSTALL_ENV"
    for script in "$archive_prefix"/bin/*; do
        [ -f "$script" ] || continue
        if head -n 1 "$script" 2>/dev/null | grep -q "^#!$old_prefix/bin/"; then
            sed -i "1s|^#!$old_prefix/bin/|#!$archive_prefix/bin/|" "$script" || return
        fi
    done
    printf 'Run the archived environment with: conda run --no-capture-output -n %s python -m reachAQ.app\n' \
        "$archive_name"
    # The new environment is built from nothing; it gets the plugins the one
    # it replaces had.
    find_plugins_to_restore "$archive_name"
}

# find_plugins_to_restore <env>: sets PLUGINS_TO_RESTORE to the plugins <env>
# has registered, plus those inferred from engines installed before the
# plugins existed. An update never strips an engine.
find_plugins_to_restore() {
    local env=$1
    local found
    found=$({ installed_pose_plugins "$env"; legacy_pose_engines "$env"; } | known_plugins_in)
    PLUGINS_TO_RESTORE=()
    if [ -n "$found" ]; then
        mapfile -t PLUGINS_TO_RESTORE <<< "$found"
    fi
    printf 'Pose plugins to restore from %s: %s\n' \
        "$env" "${PLUGINS_TO_RESTORE[*]:-none}"
}

# requirements.txt holds `-e .`, which pip resolves against the working
# directory rather than the file, so run from the checkout: started from
# anywhere else, this step failed with "does not appear to be a Python project".
install_python_requirements() {
    (cd "$INSTALL_REPO" && conda_run python -m pip install -r requirements.txt)
}

install_editable_package() {
    conda_run python -m pip install -e "$INSTALL_REPO[test]"
}

# The built-in keypoint backend runs on torch, and the DeepLabCut plugin
# requires it. Installed first, from the `torch` extra of
# auto-trainer-inference: left to resolve it alone pip takes the newest build -
# CUDA 13, which TensorFlow cannot share a process with - only for this step to
# replace it, leaving the CUDA 13 libraries behind.
install_torch_engine() {
    local pyproject="$INSTALL_REPO/auto-trainer-inference/pyproject.toml"
    if [ "$(uname -m)" = x86_64 ]; then
        install_extra "$pyproject" torch --index-url https://download.pytorch.org/whl/cu128
    else
        install_extra "$pyproject" torch
    fi
}

# pip check, with exactly the two TensorFlow caps the installer overrides on
# purpose allowed through. Anything else is a real conflict and fails.
verify_python_dependencies() {
    local report
    report=$(conda_run python -m pip check 2>&1)
    local unexpected
    unexpected=$(printf '%s\n' "$report" \
        | grep -v -E '^tensorflow [0-9.]+ has requirement (numpy|typing-extensions)' \
        | grep -v -E '^No broken requirements found|^ERROR conda|^$')
    printf '%s\n' "$report"
    if [ -n "$unexpected" ]; then
        printf 'Unexpected dependency conflicts:\n%s\n' "$unexpected" >&2
        return 1
    fi
}

install_desktop_launcher() {
    REACHAQ_INSTALL_REPO="$INSTALL_REPO" \
    REACHAQ_INSTALL_ENV="$INSTALL_ENV" \
    REACHAQ_INSTALL_CONFIG_DIR="$INSTALL_CONFIG_DIR" \
    REACHAQ_CONDA_BIN="$CONDA_BIN" \
        "$INSTALL_REPO/tools/install/install-reachaq-desktop.sh"
}

# install-reachaq-desktop.sh installs `reachaq` and `reachaq-sync` beside the
# desktop launcher. Check they landed and run, rather than trusting its exit.
verify_terminal_commands() {
    local bin_dir=${REACHAQ_USER_BIN_DIR:-$HOME/.local/bin}
    local command_name
    for command_name in reachaq reachaq-sync; do
        if [ ! -x "$bin_dir/$command_name" ]; then
            printf 'Missing terminal command: %s\n' "$bin_dir/$command_name" >&2
            return 1
        fi
    done
    "$bin_dir/reachaq-sync" --help >/dev/null
}

# A convolution rather than a matmul, because cuDNN is what the two frameworks
# fought over, and a matmul never touches it.
verify_torch_gpu_runtime() {
    conda_run python - <<'PY'
import torch
from autotrainer.inference import detect_gpu_runtime

status = detect_gpu_runtime(required_backend="torch")
print(status)
if not status.is_available:
    raise SystemExit(status.error)

image = torch.ones((1, 3, 32, 32), device="cuda")
result = torch.nn.functional.conv2d(image, torch.ones((4, 3, 3, 3), device="cuda"))
torch.cuda.synchronize()
print("PyTorch GPU convolution device:", result.device,
      "cuDNN", torch.backends.cudnn.version())
PY
}

# The Spinnaker SDK itself is a vendor install (docs/linux-install/flir-spinnaker.md);
# its Python binding is bundled here per interpreter and architecture, so once
# the SDK is present the matching wheel is installed automatically.
install_spinnaker_binding() {
    conda_run python - "$INSTALL_REPO/vendor/spinnaker" <<'PY'
import json
import pathlib
import platform
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
manifest = json.loads((root / "manifest.json").read_text())
match = [artifact for artifact in manifest["artifacts"]
         if artifact["system"] == "linux"
         and artifact["machine"] == platform.machine()
         and artifact["python"] == tag]
if not match:
    raise SystemExit(f"no bundled Spinnaker wheel for {tag} on {platform.machine()}")
subprocess.check_call([sys.executable, "-m", "pip", "install", str(root / match[0]["path"])])
PY
}

verify_spinnaker_binding() {
    conda_run python - <<'PY'
import PySpin

system = PySpin.System.GetInstance()
try:
    library = system.GetLibraryVersion()
    cameras = system.GetCameras()
    serials = [camera.TLDevice.DeviceSerialNumber.GetValue() for camera in cameras]
    cameras.Clear()
finally:
    system.ReleaseInstance()
print("PySpin uses Spinnaker %d.%d.%d.%d; cameras: %s" % (
    library.major, library.minor, library.type, library.build, serials or "none attached"))
PY
}

git_lfs_install() {
    # Ask git where the hook lives rather than assuming .git/hooks: in a
    # worktree .git is a file and the hooks are in the shared git directory, so
    # the check never matched there and `git lfs install --local` then refused
    # to overwrite the LFS hook already in place (exit 2).
    local pre_push_hook
    pre_push_hook=$(git -C "$INSTALL_REPO" rev-parse --path-format=absolute \
        --git-path hooks/pre-push) || return
    if [ -f "$pre_push_hook" ] && grep -q 'git lfs pre-push' "$pre_push_hook"; then
        printf 'Compatible Git LFS pre-push hook is already installed.\n'
        return 0
    fi
    git -C "$INSTALL_REPO" lfs install --local
}

git_lfs_pull() {
    git -C "$INSTALL_REPO" lfs pull
}

# torch for the built-in keypoint backend, plus the entry-point module of each
# pose plugin this run restored; those resolve from the checkout too.
verify_imports() {
    conda_run python - "$INSTALL_REPO" "${PLUGINS_TO_RESTORE[@]}" <<'PY'
import importlib
import sys
from pathlib import Path

import PySide6
import can
import cv2
import keyring
import nidaqmx
import openpyxl
import requests
import serial
import torch
import autotrainer.core
import autotrainer.device
import autotrainer.video
import reachAQ
import tools

plugins = [importlib.import_module(f"autotrainer.pose_plugins.{name}.plugin")
           for name in sys.argv[2:]]

repository = Path(sys.argv[1]).resolve()
for module in (autotrainer.core, autotrainer.device, autotrainer.video, reachAQ, tools, *plugins):
    module_file = getattr(module, "__file__", None)
    locations = [Path(module_file).resolve()] if module_file else [
        Path(value).resolve() for value in module.__path__ if Path(value).exists()
    ]
    resolves_from_checkout = False
    for location in locations:
        try:
            location.relative_to(repository)
        except ValueError:
            continue
        resolves_from_checkout = True
        break
    if not resolves_from_checkout:
        raise SystemExit(
            f"{module.__name__} resolves outside the current checkout: {locations}"
        )
print(f"generic imports resolve from current checkout: {repository}")
PY
}

verify_softmouse_runtime() {
    conda_run python - <<'PY'
import keyring
import openpyxl
import requests

from tools.softmouse_sync.https_source import SoftMouseHttpsSource
from tools.softmouse_sync.publisher import SoftMouseExportPublisher

backend = keyring.get_keyring()
priority = backend.priority
if priority <= 0:
    raise SystemExit(f"No usable OS keyring backend is available: {backend}")
print(f"SoftMouse HTTPS, spreadsheet, and keyring runtime ok ({backend})")
PY
}

verify_rfid_runtime() {
    conda_run python - <<'PY'
import os
from pathlib import Path

import serial
from autotrainer.device.rfid_reader import RfidReaderService

devices = sorted(Path("/dev/serial/by-id").glob("*"))
if not devices:
    print("RFID runtime ok; no /dev/serial/by-id device is currently attached")
else:
    inaccessible = [path for path in devices if not os.access(path, os.R_OK | os.W_OK)]
    if inaccessible:
        raise SystemExit(
            "RFID serial device is not readable/writable in this login session: "
            + ", ".join(map(str, inaccessible))
            + ". Log out and back in after joining dialout."
        )
    print("RFID runtime and serial permissions ok: " + ", ".join(map(str, devices)))
PY
}

verify_softmouse_systemd_units() {
    systemd-analyze --user verify \
        "$INSTALL_REPO/tools/softmouse_sync/systemd/reachaq-softmouse-publisher.service" \
        "$INSTALL_REPO/tools/softmouse_sync/systemd/reachaq-softmouse-publisher.timer"
}

run_focused_tests() {
    (
        cd "$INSTALL_REPO" || return
        conda_run python -m pytest \
            auto-trainer-core/tests/logging_test.py \
            auto-trainer-core/tests/external_metadata_test.py \
            auto-trainer-device/tests/can_transport_test.py \
            auto-trainer-device/tests/laser_test.py \
            auto-trainer-device/tests/rfid_reader_test.py \
            auto-trainer-inference/tests/gpu_runtime_test.py \
            tests/acquisition_args_test.py \
            tests/animal_metadata_sync_test.py \
            tests/autotrainer_headless_test.py::test_cli_help \
            tests/autotrainer_headless_test.py::test_load_config \
            tests/autotrainer_headless_test.py::test_gpu_preflight_failure_does_not_block_cameras_and_hardware \
            tests/autotrainer_headless_test.py::test_live_inference_override_is_not_persisted_with_other_configuration_changes \
            tests/autotrainer_headless_test.py::test_load_config_extra_reach_camera_slot \
            tests/autotrainer_headless_test.py::test_load_config_random_camera_override \
            tests/autotrainer_headless_test.py::test_load_config_random_camera_override_adds_default_reach_cameras \
            tests/hardware_status_content_test.py \
            tests/inference_dependencies_test.py \
            tests/instance_lock_test.py \
            tests/nidaq_port_configuration_dialog_test.py \
            tests/reachaq_linux_install_test.py \
            tests/reachaq_plugin_install_test.py \
            tests/rfid_app_model_test.py \
            tests/signal_stream_ui_test.py \
            tests/softmouse_cli_test.py \
            tests/softmouse_https_source_test.py \
            tests/softmouse_publication_controller_test.py \
            tests/softmouse_publisher_test.py \
            tests/softmouse_registry_test.py \
            tests/user_preferences_softmouse_test.py \
            -q
    )
}

begin_category "Preflight"
run_step "Validate repository checkout" check_repo
run_step "Create runtime directories" make_runtime_directories

SYSTEM_SKIPPED="REACHAQ_INSTALL_SYSTEM=0; needs root, assumed already in place"

begin_category "Portable Ubuntu packages"
if [ "$INSTALL_SYSTEM" = "0" ]; then
    skip_step "Update apt metadata" "$SYSTEM_SKIPPED"
    skip_step "Install base packages" "$SYSTEM_SKIPPED"
    skip_step "Configure RFID serial permissions" "$SYSTEM_SKIPPED"
else
    if ! have_command apt-get; then
        skip_step "Update apt metadata" "apt-get is unavailable; install equivalent packages manually"
        skip_step "Install base packages" "apt-get is unavailable; install equivalent packages manually"
    else
        run_step "Update apt metadata" apt_update
        run_step "Install base packages" apt_install_base
    fi
    run_step "Configure RFID serial permissions" ensure_rfid_serial_group
fi

begin_category "Closed-loop latency tuning"
if [ "$INSTALL_SYSTEM" = "0" ]; then
    skip_step "Grant real-time priority to the stim loop" "$SYSTEM_SKIPPED"
    skip_step "Install CPU governor unit" "$SYSTEM_SKIPPED"
else
    run_step "Grant real-time priority to the stim loop" ensure_realtime_priority_limits
    run_step "Install CPU governor unit" install_cpu_governor_unit
fi

begin_category "Conda runtime"
CONDA_BIN=$(find_conda)
if [ -z "$CONDA_BIN" ]; then
    run_step "Install Miniconda" install_miniconda
    CONDA_BIN=$(find_conda)
fi
if [ -z "$CONDA_BIN" ]; then
    record_result FAIL "Locate conda" "not found after automatic Miniconda installation attempt"
    printf '\n[FAIL] Locate conda: dependent steps will be skipped\n' >&2
    skip_step "Find pose plugins to restore" "conda unavailable"
    skip_step "Create conda environment" "conda unavailable"
    skip_step "Upgrade Python packaging tools" "conda unavailable"
    skip_step "Install Python requirements" "conda unavailable"
    skip_step "Install PyTorch pose engine" "conda unavailable"
    skip_step "Install reachAQ editable package" "conda unavailable"
else
    record_result PASS "Locate conda" "$CONDA_BIN"
    printf '\n[PASS] Locate conda: %s\n' "$CONDA_BIN"
    # Before the environment can be rebuilt or archived; archiving reads the
    # archive again, which is where the plugins then are.
    run_step "Find pose plugins to restore" find_plugins_to_restore "$INSTALL_ENV"
    run_step "Create conda environment" ensure_conda_environment
    # tomli explicitly: pyproject_extra reads pyproject.toml with it,
    # Python 3.10 has no tomllib, and relying on build to pull it in would break
    # silently the day build stops needing it.
    run_step "Upgrade Python packaging tools" conda_run python -m pip install --upgrade pip setuptools wheel build tomli
    run_step "Install PyTorch pose engine" install_torch_engine
    run_step "Install Python requirements" install_python_requirements
    run_step "Install reachAQ editable package" install_editable_package
    # No plugin is installed by default; only those the environment had. The
    # DeepLabCut plugin brings the TensorFlow engine, its CUDA runtime and the
    # both-engines check as steps of their own.
    for name in "${PLUGINS_TO_RESTORE[@]}"; do
        run_step "Restore $name pose plugin" install_pose_plugin "$name"
    done
    run_step "Install reachAQ desktop launcher and terminal commands" install_desktop_launcher
    run_step "Verify reachaq and reachaq-sync commands" verify_terminal_commands
fi

begin_category "Pose engine GPU runtimes"
if [ -z "$CONDA_BIN" ]; then
    skip_step "Verify PyTorch GPU preflight" "conda unavailable"
else
    run_step "Verify PyTorch GPU preflight" verify_torch_gpu_runtime
fi

begin_category "Git LFS"
if ! have_command git; then
    record_result FAIL "Locate git" "git executable not found"
    printf '\n[FAIL] Locate git: dependent Git LFS steps will be skipped\n' >&2
    skip_step "Initialize Git LFS" "git unavailable"
    skip_step "Pull Git LFS assets" "git unavailable"
elif ! git lfs version >/dev/null 2>&1; then
    record_result FAIL "Locate Git LFS" "git-lfs executable not found"
    printf '\n[FAIL] Locate Git LFS: install git-lfs and rerun this category\n' >&2
    skip_step "Initialize Git LFS" "git-lfs unavailable"
    skip_step "Pull Git LFS assets" "git-lfs unavailable"
else
    record_result PASS "Locate Git LFS" "$(git lfs version 2>/dev/null)"
    run_step "Initialize Git LFS" git_lfs_install
    run_step "Pull Git LFS assets" git_lfs_pull
fi

# After Git LFS: the bundled wheels are LFS objects.
begin_category "Spinnaker camera binding"
if [ -z "$CONDA_BIN" ]; then
    skip_step "Install Spinnaker Python binding" "conda unavailable"
    skip_step "Verify Spinnaker cameras" "conda unavailable"
elif ! ls /opt/spinnaker/lib/libSpinnaker.so* >/dev/null 2>&1; then
    skip_step "Install Spinnaker Python binding" \
        "Spinnaker SDK not installed; see docs/linux-install/flir-spinnaker.md, then rerun"
    skip_step "Verify Spinnaker cameras" "Spinnaker SDK not installed"
else
    run_step "Install Spinnaker Python binding" install_spinnaker_binding
    run_step "Verify Spinnaker cameras" verify_spinnaker_binding
fi

begin_category "Portable verification"
if [ -z "$CONDA_BIN" ]; then
    skip_step "Verify Python version" "conda unavailable"
    skip_step "Verify Python dependencies" "conda unavailable"
    skip_step "Verify generic imports" "conda unavailable"
    skip_step "Verify SoftMouse runtime" "conda unavailable"
    skip_step "Verify RFID runtime" "conda unavailable"
    skip_step "Verify GUI CLI" "conda unavailable"
    skip_step "Verify headless CLI" "conda unavailable"
else
    run_step "Verify Python version" conda_run python --version
    run_step "Verify Python dependencies" verify_python_dependencies
    run_step "Verify generic imports" verify_imports
    run_step "Verify SoftMouse runtime" verify_softmouse_runtime
    run_step "Verify RFID runtime" verify_rfid_runtime
    run_step "Verify GUI CLI" conda_run python -m reachAQ.app -h
    run_step "Verify headless CLI" conda_run auto-trainer-headless -h
fi
run_step "Verify SoftMouse systemd units" verify_softmouse_systemd_units

if [ -z "$CONDA_BIN" ]; then
    skip_step "Run focused non-hardware tests" "conda unavailable"
else
    run_step "Run focused non-hardware tests" run_focused_tests
fi

print_report
if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
exit 0

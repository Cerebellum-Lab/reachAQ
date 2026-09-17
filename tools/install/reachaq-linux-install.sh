#!/usr/bin/env bash

# Portable reachAQ Linux host setup.
#
# This script intentionally excludes hardware/model-specific drivers and
# configuration (FLIR Spinnaker, NI-DAQ/PXI, PEAK CAN, NVIDIA kernel drivers,
# and channel mappings). The single no-argument workflow installs every
# portable component, including the TensorFlow-compatible CUDA user-space
# runtime, and runs all tracked verification.
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
INSTALL_PYTHON=${REACHAQ_INSTALL_PYTHON:-3.8}
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
CONDA_BIN=""

if [ "$#" -ne 0 ]; then
    printf '%s\n' \
        'reachaq-linux-install.sh does not accept arguments.' \
        'Run it with no options; every install and verification category is attempted.' >&2
    exit 2
fi

record_result() {
    local state=$1
    local name=$2
    local detail=${3:-}
    RESULT_STATES+=("$state")
    RESULT_NAMES+=("$CURRENT_CATEGORY | $name")
    RESULT_DETAILS+=("$detail")
    case "$state" in
        PASS) PASS_COUNT=$((PASS_COUNT + 1)) ;;
        FAIL) FAIL_COUNT=$((FAIL_COUNT + 1)) ;;
        SKIP) SKIP_COUNT=$((SKIP_COUNT + 1)) ;;
    esac
}

begin_category() {
    CURRENT_CATEGORY=$1
    printf '\n============================================================\n'
    printf '%s\n' "$CURRENT_CATEGORY"
    printf '============================================================\n'
}

print_command() {
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
}

run_step() {
    local name=$1
    shift
    printf '\n--- %s\n' "$name"
    print_command "$@"
    "$@"
    local result=$?
    if [ "$result" -eq 0 ]; then
        record_result PASS "$name"
        printf '[PASS] %s\n' "$name"
    else
        record_result FAIL "$name" "exit $result"
        printf '[FAIL] %s (exit %d); continuing\n' "$name" "$result" >&2
    fi
    return 0
}

skip_step() {
    local name=$1
    local reason=$2
    record_result SKIP "$name" "$reason"
    printf '\n[SKIP] %s: %s\n' "$name" "$reason"
}

print_report() {
    if $REPORT_PRINTED; then
        return
    fi
    REPORT_PRINTED=true
    printf '\n============================================================\n'
    printf 'reachAQ portable install report\n'
    printf '============================================================\n'
    local index
    for ((index = 0; index < ${#RESULT_NAMES[@]}; index++)); do
        printf '%-4s  %s' "${RESULT_STATES[$index]}" "${RESULT_NAMES[$index]}"
        if [ -n "${RESULT_DETAILS[$index]}" ]; then
            printf ' (%s)' "${RESULT_DETAILS[$index]}"
        fi
        printf '\n'
    done
    printf '%s\n' '------------------------------------------------------------'
    printf 'PASS=%d FAIL=%d SKIP=%d\n' \
        "$PASS_COUNT" "$FAIL_COUNT" "$SKIP_COUNT"
    printf 'Repository: %s\n' "$INSTALL_REPO"
    printf 'Conda environment: %s (Python %s)\n' "$INSTALL_ENV" "$INSTALL_PYTHON"
    printf 'Config directory: %s\n' "$INSTALL_CONFIG_DIR"
    printf 'Data directory: %s\n' "$INSTALL_DATA_DIR"
    if [ "$FAIL_COUNT" -gt 0 ]; then
        printf 'Completed with failures. Review every FAIL entry above.\n'
    else
        printf 'Installation and verification completed without reported failures.\n'
    fi
    printf '%s\n' 'Hardware-specific kernel drivers/configuration are not installed by this script.'
}

on_exit() {
    local result=$?
    if ! $REPORT_PRINTED; then
        if [ "$result" -ne 0 ]; then
            record_result FAIL "Unexpected script termination" "exit $result"
        fi
        print_report
    fi
}

on_interrupt() {
    record_result FAIL "Interrupted" "signal received"
    print_report
    exit 130
}

trap on_exit EXIT
trap on_interrupt INT TERM

have_command() {
    command -v "$1" >/dev/null 2>&1
}

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

find_conda() {
    if have_command conda; then
        command -v conda
        return 0
    fi
    local candidate
    for candidate in \
        "$HOME/anaconda3/bin/conda" \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/mambaforge/bin/conda"; do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
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
    if "$CONDA_BIN" run -n "$INSTALL_ENV" python --version >/dev/null 2>&1; then
        printf 'Conda environment %s already exists.\n' "$INSTALL_ENV"
        return 0
    fi
    "$CONDA_BIN" create -y -n "$INSTALL_ENV" "python=$INSTALL_PYTHON"
}

conda_run() {
    "$CONDA_BIN" run --no-capture-output -n "$INSTALL_ENV" "$@"
}

install_editable_package() {
    conda_run python -m pip install -e "$INSTALL_REPO[test]"
}

install_desktop_launcher() {
    REACHAQ_INSTALL_REPO="$INSTALL_REPO" \
    REACHAQ_INSTALL_ENV="$INSTALL_ENV" \
    REACHAQ_INSTALL_CONFIG_DIR="$INSTALL_CONFIG_DIR" \
    REACHAQ_CONDA_BIN="$CONDA_BIN" \
        "$INSTALL_REPO/tools/install/install-reachaq-desktop.sh"
}

install_tensorflow_gpu_runtime() {
    if [ "$(uname -m)" != "x86_64" ]; then
        printf '%s\n' \
            'The automated TensorFlow CUDA runtime install supports x86_64 only.' \
            'Use the NVIDIA JetPack TensorFlow packages on Jetson/aarch64.' >&2
        return 2
    fi

    local tensorflow_version
    tensorflow_version=$(conda_run python -c \
        'from importlib.metadata import version; print(version("tensorflow"))') || return

    # TensorFlow's tested build table specifies CUDA 11.8 and cuDNN 8.6 for
    # TensorFlow 2.12 and 2.13. Keep this case explicit: silently installing a
    # guessed runtime for a newer TensorFlow version is worse than a clear fail.
    # https://www.tensorflow.org/install/source#gpu
    case "$tensorflow_version" in
        2.12.*|2.13.*)
            ;;
        *)
            printf 'TensorFlow %s is not supported by this automated GPU runtime step.\n' \
                "$tensorflow_version" >&2
            printf '%s\n' \
                'Consult https://www.tensorflow.org/install/source#gpu and install matching CUDA/cuDNN versions.' >&2
            return 3
            ;;
    esac

    printf 'Installing CUDA 11.8 and cuDNN 8.6 libraries for TensorFlow %s.\n' \
        "$tensorflow_version"
    conda_run python -m pip install \
        'nvidia-cuda-runtime-cu11==11.8.89' \
        'nvidia-cuda-cupti-cu11==11.8.87' \
        'nvidia-cuda-nvrtc-cu11==11.8.89' \
        'nvidia-cublas-cu11==11.11.3.6' \
        'nvidia-cufft-cu11==10.9.0.58' \
        'nvidia-curand-cu11==10.3.0.86' \
        'nvidia-cusolver-cu11==11.4.1.48' \
        'nvidia-cusparse-cu11==11.7.5.86' \
        'nvidia-cudnn-cu11==8.6.0.163' || return

    local nvidia_root
    nvidia_root=$(conda_run python -c \
        'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia")') || return
    local tensorflow_library_path
    tensorflow_library_path="$nvidia_root/cublas/lib:$nvidia_root/cuda_cupti/lib"
    tensorflow_library_path="$tensorflow_library_path:$nvidia_root/cuda_nvrtc/lib:$nvidia_root/cuda_runtime/lib"
    tensorflow_library_path="$tensorflow_library_path:$nvidia_root/cudnn/lib:$nvidia_root/cufft/lib"
    tensorflow_library_path="$tensorflow_library_path:$nvidia_root/curand/lib:$nvidia_root/cusolver/lib:$nvidia_root/cusparse/lib"

    local existing_library_path
    existing_library_path=$("$CONDA_BIN" env config vars list -n "$INSTALL_ENV" \
        | sed -n 's/^LD_LIBRARY_PATH = //p')
    if [ -n "$existing_library_path" ]; then
        local preserved_library_path=""
        local library_directory
        local existing_library_directories=()
        IFS=: read -r -a existing_library_directories <<< "$existing_library_path"
        for library_directory in "${existing_library_directories[@]}"; do
            case "$library_directory" in
                "$nvidia_root"/*/lib)
                    ;;
                *)
                    if [ -n "$preserved_library_path" ]; then
                        preserved_library_path="$preserved_library_path:$library_directory"
                    else
                        preserved_library_path=$library_directory
                    fi
                    ;;
            esac
        done
        if [ -n "$preserved_library_path" ]; then
            tensorflow_library_path="$tensorflow_library_path:$preserved_library_path"
        fi
    fi
    "$CONDA_BIN" env config vars set -n "$INSTALL_ENV" \
        "LD_LIBRARY_PATH=$tensorflow_library_path"
}

verify_tensorflow_gpu_runtime() {
    conda_run python - <<'PY'
import tensorflow as tf
from autotrainer.inference import detect_gpu_runtime

status = detect_gpu_runtime(required_backend="tensorflow")
print(status)
if not status.is_available:
    raise SystemExit(status.error)

with tf.device("/GPU:0"):
    result = tf.linalg.matmul(tf.ones((128, 128)), tf.ones((128, 128)))
print("TensorFlow GPU calculation device:", result.device)
if "GPU:0" not in result.device:
    raise SystemExit("TensorFlow calculation did not execute on GPU:0")
PY
}

git_lfs_install() {
    local pre_push_hook="$INSTALL_REPO/.git/hooks/pre-push"
    if [ -f "$pre_push_hook" ] && grep -q 'git lfs pre-push' "$pre_push_hook"; then
        printf 'Compatible Git LFS pre-push hook is already installed.\n'
        return 0
    fi
    git -C "$INSTALL_REPO" lfs install --local
}

git_lfs_pull() {
    git -C "$INSTALL_REPO" lfs pull
}

verify_imports() {
    conda_run python - "$INSTALL_REPO" <<'PY'
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
import autotrainer.core
import autotrainer.device
import autotrainer.video
import reachAQ
import tools

repository = Path(sys.argv[1]).resolve()
for module in (autotrainer.core, autotrainer.device, autotrainer.video, reachAQ, tools):
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
            tests/autotrainer_headless_test.py::test_gpu_preflight_fails_before_cameras_and_hardware \
            tests/autotrainer_headless_test.py::test_live_inference_override_is_not_persisted_with_other_configuration_changes \
            tests/autotrainer_headless_test.py::test_load_config_extra_reach_camera_slot \
            tests/autotrainer_headless_test.py::test_load_config_random_camera_override \
            tests/autotrainer_headless_test.py::test_load_config_random_camera_override_adds_default_reach_cameras \
            tests/hardware_status_content_test.py \
            tests/nidaq_port_configuration_dialog_test.py \
            tests/reachaq_linux_install_test.py \
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

begin_category "Portable Ubuntu packages"
if ! have_command apt-get; then
    skip_step "Update apt metadata" "apt-get is unavailable; install equivalent packages manually"
    skip_step "Install base packages" "apt-get is unavailable; install equivalent packages manually"
else
    run_step "Update apt metadata" apt_update
    run_step "Install base packages" apt_install_base
fi
run_step "Configure RFID serial permissions" ensure_rfid_serial_group

begin_category "Closed-loop latency tuning"
run_step "Grant real-time priority to the stim loop" ensure_realtime_priority_limits
run_step "Install CPU governor unit" install_cpu_governor_unit

begin_category "Conda runtime"
CONDA_BIN=$(find_conda)
if [ -z "$CONDA_BIN" ]; then
    run_step "Install Miniconda" install_miniconda
    CONDA_BIN=$(find_conda)
fi
if [ -z "$CONDA_BIN" ]; then
    record_result FAIL "Locate conda" "not found after automatic Miniconda installation attempt"
    printf '\n[FAIL] Locate conda: dependent steps will be skipped\n' >&2
    skip_step "Create conda environment" "conda unavailable"
    skip_step "Upgrade Python packaging tools" "conda unavailable"
    skip_step "Install Python requirements" "conda unavailable"
    skip_step "Install reachAQ editable package" "conda unavailable"
else
    record_result PASS "Locate conda" "$CONDA_BIN"
    printf '\n[PASS] Locate conda: %s\n' "$CONDA_BIN"
    run_step "Create conda environment" ensure_conda_environment
    run_step "Upgrade Python packaging tools" conda_run python -m pip install --upgrade pip setuptools wheel build
    run_step "Install Python requirements" conda_run python -m pip install -r "$INSTALL_REPO/requirements.txt"
    run_step "Install reachAQ editable package" install_editable_package
    run_step "Install reachAQ desktop launcher" install_desktop_launcher
fi

begin_category "TensorFlow GPU runtime"
if [ -z "$CONDA_BIN" ]; then
    skip_step "Install compatible CUDA user-space runtime" "conda unavailable"
    skip_step "Verify TensorFlow GPU preflight" "conda unavailable"
else
    run_step "Install compatible CUDA user-space runtime" install_tensorflow_gpu_runtime
    run_step "Verify TensorFlow GPU preflight" verify_tensorflow_gpu_runtime
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
    run_step "Verify Python dependencies" conda_run python -m pip check
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

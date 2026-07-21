#!/usr/bin/env bash

# Portable reachAQ Linux host setup.
#
# This script intentionally excludes hardware/model-specific drivers and
# configuration (FLIR Spinnaker, NI-DAQ/PXI, PEAK CAN, NVIDIA/CUDA, and channel
# mappings). Those remain in the focused guides under docs/linux-install/.
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

INSTALL_SYSTEM_PACKAGES=true
INSTALL_PYTHON_ENV=true
INSTALL_GIT_LFS=true
INSTALL_VERIFY=true
INSTALL_MINICONDA=false
INSTALL_TEST_DEPS=true
INSTALL_RUN_TESTS=false
INSTALL_DRY_RUN=false

CURRENT_CATEGORY="General"
RESULT_NAMES=()
RESULT_STATES=()
RESULT_DETAILS=()
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
PLAN_COUNT=0
REPORT_PRINTED=false
CONDA_BIN=""

usage() {
    cat <<'EOF'
Usage: tools/install/reachaq-linux-install.sh [options]

Portable setup only; hardware-specific drivers are intentionally excluded.

Paths and environment:
  --repo PATH                 Repository root (default: detected checkout)
  --env NAME                  Conda environment name (default: reachaq)
  --python VERSION            Conda Python version (default: 3.8)
  --config-dir PATH           Runtime configuration directory
  --data-dir PATH             Acquisition output directory

Optional behavior:
  --install-miniconda         Install Miniconda under $HOME/miniconda3 if conda is absent
  --without-test-deps         Install editable package without the test extra
  --run-tests                 Run the focused non-hardware verification suite
  --skip-system-packages      Do not run apt update/install
  --skip-python-env           Do not create/update the conda environment
  --skip-git-lfs              Do not initialize or pull Git LFS
  --skip-verification         Do not run final import/CLI checks
  --dry-run                   Print and report the planned steps without executing them
  -h, --help                  Show this help

Every operational step continues after failure. The final report lists all
PASS/FAIL/SKIP results, and the script exits nonzero only after the full run if
one or more steps failed.
EOF
}

require_option_value() {
    local option=$1
    local remaining=$2
    if [ "$remaining" -lt 2 ]; then
        printf 'Option %s requires a value.\n\n' "$option" >&2
        usage >&2
        exit 2
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo)
            require_option_value "$1" "$#"
            INSTALL_REPO=$2
            shift 2
            ;;
        --env)
            require_option_value "$1" "$#"
            INSTALL_ENV=$2
            shift 2
            ;;
        --python)
            require_option_value "$1" "$#"
            INSTALL_PYTHON=$2
            shift 2
            ;;
        --config-dir)
            require_option_value "$1" "$#"
            INSTALL_CONFIG_DIR=$2
            shift 2
            ;;
        --data-dir)
            require_option_value "$1" "$#"
            INSTALL_DATA_DIR=$2
            shift 2
            ;;
        --install-miniconda)
            INSTALL_MINICONDA=true
            shift
            ;;
        --without-test-deps)
            INSTALL_TEST_DEPS=false
            shift
            ;;
        --run-tests)
            INSTALL_RUN_TESTS=true
            shift
            ;;
        --skip-system-packages)
            INSTALL_SYSTEM_PACKAGES=false
            shift
            ;;
        --skip-python-env)
            INSTALL_PYTHON_ENV=false
            shift
            ;;
        --skip-git-lfs)
            INSTALL_GIT_LFS=false
            shift
            ;;
        --skip-verification)
            INSTALL_VERIFY=false
            shift
            ;;
        --dry-run)
            INSTALL_DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

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
        PLAN) PLAN_COUNT=$((PLAN_COUNT + 1)) ;;
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
    if $INSTALL_DRY_RUN; then
        record_result PLAN "$name" "dry run"
        printf '[PLAN] %s\n' "$name"
        return 0
    fi

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
    printf 'PASS=%d FAIL=%d SKIP=%d PLAN=%d\n' \
        "$PASS_COUNT" "$FAIL_COUNT" "$SKIP_COUNT" "$PLAN_COUNT"
    printf 'Repository: %s\n' "$INSTALL_REPO"
    printf 'Conda environment: %s (Python %s)\n' "$INSTALL_ENV" "$INSTALL_PYTHON"
    printf 'Config directory: %s\n' "$INSTALL_CONFIG_DIR"
    printf 'Data directory: %s\n' "$INSTALL_DATA_DIR"
    if [ "$FAIL_COUNT" -gt 0 ]; then
        printf 'Completed with failures. Review every FAIL entry above.\n'
    elif $INSTALL_DRY_RUN; then
        printf 'Dry run complete; no changes were made.\n'
    else
        printf 'Portable installation steps completed without reported failures.\n'
    fi
    printf '%s\n' 'Hardware-specific drivers/configuration are not installed by this script.'
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
        dkms \
        expat \
        ffmpeg \
        git \
        git-lfs \
        iproute2 \
        libegl1 \
        libgl1 \
        libopenal1 \
        libxcb-cursor0 \
        libxkbcommon-x11-0 \
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
    if $INSTALL_TEST_DEPS; then
        conda_run python -m pip install -e "$INSTALL_REPO[test]"
    else
        conda_run python -m pip install -e "$INSTALL_REPO"
    fi
}

git_lfs_install() {
    git -C "$INSTALL_REPO" lfs install --local
}

git_lfs_pull() {
    git -C "$INSTALL_REPO" lfs pull
}

verify_imports() {
    conda_run python -c \
        'import autotrainer.core, autotrainer.device, autotrainer.video, PySide6, cv2, can, nidaqmx; print("generic imports ok")'
}

run_focused_tests() {
    (
        cd "$INSTALL_REPO" || return
        conda_run python -m pytest \
            auto-trainer-core/tests/logging_test.py \
            auto-trainer-device/tests/can_transport_test.py \
            auto-trainer-device/tests/laser_test.py \
            auto-trainer-inference/tests/gpu_runtime_test.py \
            tests/acquisition_args_test.py \
            tests/behavior_model_test.py::TestEmergency \
            tests/autotrainer_headless_test.py::test_cli_help \
            tests/autotrainer_headless_test.py::test_load_config \
            tests/autotrainer_headless_test.py::test_gpu_preflight_fails_before_cameras_and_hardware \
            tests/autotrainer_headless_test.py::test_live_inference_override_is_not_persisted_with_other_configuration_changes \
            tests/autotrainer_headless_test.py::test_load_config_extra_reach_camera_slot \
            tests/autotrainer_headless_test.py::test_load_config_random_camera_override \
            tests/autotrainer_headless_test.py::test_load_config_random_camera_override_adds_default_reach_cameras \
            tests/nidaq_port_configuration_dialog_test.py \
            tests/reachaq_linux_install_test.py \
            -q
    )
}

begin_category "Preflight"
run_step "Validate repository checkout" check_repo
run_step "Create runtime directories" make_runtime_directories

begin_category "Portable Ubuntu packages"
if ! $INSTALL_SYSTEM_PACKAGES; then
    skip_step "Update apt metadata" "disabled by --skip-system-packages"
    skip_step "Install base packages" "disabled by --skip-system-packages"
elif ! have_command apt-get; then
    skip_step "Update apt metadata" "apt-get is unavailable; install equivalent packages manually"
    skip_step "Install base packages" "apt-get is unavailable; install equivalent packages manually"
else
    run_step "Update apt metadata" apt_update
    run_step "Install base packages" apt_install_base
fi

begin_category "Conda runtime"
if ! $INSTALL_PYTHON_ENV; then
    CONDA_BIN=$(find_conda)
    if [ -n "$CONDA_BIN" ]; then
        record_result PASS "Locate conda" "$CONDA_BIN"
        printf '\n[PASS] Locate conda: %s\n' "$CONDA_BIN"
    else
        record_result FAIL "Locate conda" "not found; verification cannot use the requested environment"
        printf '\n[FAIL] Locate conda: verification steps will be skipped\n' >&2
    fi
    skip_step "Create conda environment" "disabled by --skip-python-env"
    skip_step "Upgrade Python packaging tools" "disabled by --skip-python-env"
    skip_step "Install Python requirements" "disabled by --skip-python-env"
    skip_step "Install reachAQ editable package" "disabled by --skip-python-env"
else
    CONDA_BIN=$(find_conda)
    if [ -z "$CONDA_BIN" ] && $INSTALL_MINICONDA; then
        run_step "Install Miniconda" install_miniconda
        if $INSTALL_DRY_RUN; then
            CONDA_BIN="$HOME/miniconda3/bin/conda"
        else
            CONDA_BIN=$(find_conda)
        fi
    elif [ -z "$CONDA_BIN" ]; then
        record_result FAIL "Locate conda" "not found; rerun with --install-miniconda or install conda manually"
        printf '\n[FAIL] Locate conda: executable not found; dependent steps will be skipped\n' >&2
    else
        record_result PASS "Locate conda" "$CONDA_BIN"
        printf '\n[PASS] Locate conda: %s\n' "$CONDA_BIN"
    fi

    if [ -z "$CONDA_BIN" ]; then
        skip_step "Create conda environment" "conda unavailable"
        skip_step "Upgrade Python packaging tools" "conda unavailable"
        skip_step "Install Python requirements" "conda unavailable"
        skip_step "Install reachAQ editable package" "conda unavailable"
    else
        run_step "Create conda environment" ensure_conda_environment
        run_step "Upgrade Python packaging tools" conda_run python -m pip install --upgrade pip setuptools wheel build
        run_step "Install Python requirements" conda_run python -m pip install -r "$INSTALL_REPO/requirements.txt"
        run_step "Install reachAQ editable package" install_editable_package
    fi
fi

begin_category "Git LFS"
if ! $INSTALL_GIT_LFS; then
    skip_step "Initialize Git LFS" "disabled by --skip-git-lfs"
    skip_step "Pull Git LFS assets" "disabled by --skip-git-lfs"
elif $INSTALL_DRY_RUN; then
    run_step "Initialize Git LFS" git_lfs_install
    run_step "Pull Git LFS assets" git_lfs_pull
elif ! have_command git; then
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
if ! $INSTALL_VERIFY; then
    skip_step "Verify Python version" "disabled by --skip-verification"
    skip_step "Verify Python dependencies" "disabled by --skip-verification"
    skip_step "Verify generic imports" "disabled by --skip-verification"
    skip_step "Verify GUI CLI" "disabled by --skip-verification"
    skip_step "Verify headless CLI" "disabled by --skip-verification"
elif [ -z "$CONDA_BIN" ]; then
    skip_step "Verify Python version" "conda unavailable"
    skip_step "Verify Python dependencies" "conda unavailable"
    skip_step "Verify generic imports" "conda unavailable"
    skip_step "Verify GUI CLI" "conda unavailable"
    skip_step "Verify headless CLI" "conda unavailable"
else
    run_step "Verify Python version" conda_run python --version
    run_step "Verify Python dependencies" conda_run python -m pip check
    run_step "Verify generic imports" verify_imports
    run_step "Verify GUI CLI" conda_run python -m reachAQ.app -h
    run_step "Verify headless CLI" conda_run auto-trainer-headless -h
fi

if $INSTALL_RUN_TESTS; then
    if [ -z "$CONDA_BIN" ]; then
        skip_step "Run focused non-hardware tests" "conda unavailable"
    elif ! $INSTALL_TEST_DEPS; then
        skip_step "Run focused non-hardware tests" "test dependencies disabled"
    else
        run_step "Run focused non-hardware tests" run_focused_tests
    fi
else
    skip_step "Run focused non-hardware tests" "enable with --run-tests"
fi

print_report
if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
exit 0

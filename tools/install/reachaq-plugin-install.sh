#!/usr/bin/env bash

# Install reachAQ pose backend plugins into an existing environment.
#
#   tools/install/reachaq-plugin-install.sh <plugin>...
#
# Plugins: deeplabcut (DeepLabCut 3, with its TensorFlow engine and that
# engine's CUDA runtime) and yolo (YOLO-pose through ultralytics, AGPL-3.0).
# Nothing else in the environment is rebuilt. The environment must already
# exist; tools/install/reachaq-linux-install.sh builds it, and reinstalls the
# plugins found here whenever it is rerun.
#
# REACHAQ_INSTALL_ENV (default reachaq) and REACHAQ_INSTALL_REPO select the
# environment and checkout, as for the main installer.
#
# A failed step stops the script and the report names it (run_step in
# plugin-install-lib.sh); it does not use `set -e`, so the report is always
# printed.

set -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)
if [ -z "$DEFAULT_REPO" ]; then
    DEFAULT_REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd)
fi

INSTALL_REPO=${REACHAQ_INSTALL_REPO:-$DEFAULT_REPO}
INSTALL_ENV=${REACHAQ_INSTALL_ENV:-reachaq}

CURRENT_CATEGORY="General"
RESULT_NAMES=()
RESULT_STATES=()
RESULT_DETAILS=()
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
WARN_COUNT=0
REPORT_PRINTED=false
REPORT_TITLE="reachAQ pose plugin install report"
RERUN_HINT="tools/install/reachaq-plugin-install.sh $*"
CONDA_BIN=""

# shellcheck source=plugin-install-lib.sh
if ! . "$SCRIPT_DIR/plugin-install-lib.sh"; then
    printf 'Cannot load %s; the checkout is incomplete.\n' "$SCRIPT_DIR/plugin-install-lib.sh" >&2
    exit 1
fi

usage() {
    printf '%s\n' \
        'Usage: reachaq-plugin-install.sh <plugin>...' \
        "Installs pose plugins into the existing conda environment $INSTALL_ENV." \
        "Plugins: ${KNOWN_PLUGINS[*]}" >&2
}

if [ "$#" -eq 0 ]; then
    usage
    exit 2
fi
for name in "$@"; do
    if ! is_known_plugin "$name"; then
        printf 'Unknown pose plugin: %s\n' "$name" >&2
        usage
        exit 2
    fi
done

trap on_exit EXIT
trap on_interrupt INT TERM

begin_category "Conda runtime"
CONDA_BIN=$(find_conda)
if [ -z "$CONDA_BIN" ]; then
    record_result FAIL "Locate conda" "conda not found"
    printf '%s\n' \
        'conda was not found. Build the environment first with tools/install/reachaq-linux-install.sh.' >&2
    stop_install "Locate conda"
fi
record_result PASS "Locate conda" "$CONDA_BIN"
if ! conda_env_exists "$INSTALL_ENV"; then
    record_result FAIL "Locate conda environment" "$INSTALL_ENV does not exist"
    printf '%s\n' \
        "Conda environment $INSTALL_ENV does not exist. This script installs plugins into an existing environment only;" \
        'build it first with tools/install/reachaq-linux-install.sh.' >&2
    stop_install "Locate conda environment"
fi
record_result PASS "Locate conda environment" "$INSTALL_ENV"

begin_category "Pose plugins"
for name in "$@"; do
    run_step "Install $name pose plugin" install_pose_plugin "$name"
done
run_step "Verify one CUDA runtime per engine" conda_run python "$SCRIPT_DIR/cuda_runtime_check.py"

print_report
exit 0

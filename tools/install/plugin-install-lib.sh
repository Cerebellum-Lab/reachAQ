# shellcheck shell=bash
#
# Functions shared by reachaq-linux-install.sh and reachaq-plugin-install.sh.
#
# Definitions only: sourcing this file runs nothing. The caller sets
# INSTALL_REPO, INSTALL_ENV and CONDA_BIN, initialises the RESULT_* arrays and
# counters, and installs the on_exit / on_interrupt traps. REPORT_TITLE and
# REPORT_FOOTER, when set, label the final report.

# Pose backend plugins this checkout carries, each a distribution under
# plugins/<name> that registers <name> in the autotrainer.pose_backends
# entry-point group.
KNOWN_PLUGINS=(deeplabcut yolo)

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
    printf '%s\n' "${REPORT_TITLE:-reachAQ install report}"
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
    if [ -n "${INSTALL_PYTHON:-}" ]; then
        printf 'Conda environment: %s (Python %s)\n' "$INSTALL_ENV" "$INSTALL_PYTHON"
    else
        printf 'Conda environment: %s\n' "$INSTALL_ENV"
    fi
    if [ -n "${INSTALL_CONFIG_DIR:-}" ]; then
        printf 'Config directory: %s\n' "$INSTALL_CONFIG_DIR"
    fi
    if [ -n "${INSTALL_DATA_DIR:-}" ]; then
        printf 'Data directory: %s\n' "$INSTALL_DATA_DIR"
    fi
    if [ "$FAIL_COUNT" -gt 0 ]; then
        printf 'Completed with failures. Review every FAIL entry above.\n'
    else
        printf 'Installation and verification completed without reported failures.\n'
    fi
    if [ -n "${REPORT_FOOTER:-}" ]; then
        printf '%s\n' "$REPORT_FOOTER"
    fi
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

have_command() {
    command -v "$1" >/dev/null 2>&1
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

conda_run() {
    "$CONDA_BIN" run --no-capture-output -n "$INSTALL_ENV" "$@"
}

conda_env_exists() {
    "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$1"
}

is_known_plugin() {
    local known
    for known in "${KNOWN_PLUGINS[@]}"; do
        [ "$known" = "$1" ] && return 0
    done
    return 1
}

# Keeps only the known plugin names found in stdin, one per line, in
# KNOWN_PLUGINS order. conda run can add lines of its own to the output.
known_plugins_in() {
    local found known
    found=$(cat)
    for known in "${KNOWN_PLUGINS[@]}"; do
        if printf '%s\n' "$found" | grep -qx "$known"; then
            printf '%s\n' "$known"
        fi
    done
}

# Plugin names registered in <env>'s autotrainer.pose_backends group; nothing
# when the environment does not exist. The archived environment of an update
# can be on an older Python, where entry_points() returns a dict, hence both
# lookups.
installed_pose_plugins() {
    "$CONDA_BIN" run -n "$1" python -c '
from importlib.metadata import entry_points
found = entry_points()
group = (found.select(group="autotrainer.pose_backends") if hasattr(found, "select")
         else found.get("autotrainer.pose_backends", ()))
print("\n".join(sorted({entry.name for entry in group})))
' 2>/dev/null | known_plugins_in
}

# The first update after DeepLabCut and YOLO left core: no environment has a
# plugin distribution yet, but one that already has deeplabcut or ultralytics
# ran that engine, so it gets the matching plugin and nothing more.
legacy_pose_engines() {
    "$CONDA_BIN" run -n "$1" python -c '
from importlib.util import find_spec
if find_spec("deeplabcut") is not None:
    print("deeplabcut")
if find_spec("ultralytics") is not None:
    print("yolo")
' 2>/dev/null | known_plugins_in
}

# The lines of one optional-dependencies extra of a pyproject.toml. setup.py
# merges only base dependencies, so the pins are read from where they are
# declared rather than copied here.
pyproject_extra() {
    conda_run python - "$1" "$2" <<'PY'
import sys

import tomli

with open(sys.argv[1], "rb") as stream:
    extras = tomli.load(stream)["project"]["optional-dependencies"]
print("\n".join(extras[sys.argv[2]]))
PY
}

# install_extra <pyproject> <extra> [pip arguments]
install_extra() {
    local pyproject=$1
    local extra=$2
    shift 2
    local requirement packages=()
    while IFS= read -r requirement; do
        [ -n "$requirement" ] && packages+=("$requirement")
    done < <(pyproject_extra "$pyproject" "$extra")
    if [ "${#packages[@]}" -eq 0 ]; then
        printf 'No %s requirements were found in %s.\n' "$extra" "$pyproject" >&2
        return 1
    fi
    conda_run python -m pip install "${packages[@]}" "$@"
}

# install_pose_plugin <name>: the plugin from this checkout, then whatever its
# engine needs beyond pip. Both DeepLabCut engines in one environment needs
# the conditions recorded beside the tensorflow extra in
# plugins/deeplabcut/pyproject.toml; in short: torch from the CUDA 12.8 index
# (the main installer's PyTorch step), TensorFlow's CUDA runtime kept apart
# (install_tensorflow_gpu_runtime), and TensorFlow 2.12's stale numpy and
# typing-extensions caps overridden afterwards. The TensorFlow steps are
# reported as steps of their own.
install_pose_plugin() {
    local name=$1
    if ! is_known_plugin "$name"; then
        printf 'Unknown pose plugin: %s (known: %s)\n' "$name" "${KNOWN_PLUGINS[*]}" >&2
        return 2
    fi
    if [ "$name" = yolo ]; then
        printf '%s\n' \
            'auto-trainer-pose-yolo installs ultralytics, which is AGPL-3.0.' \
            'Its terms then cover this environment: distributing it, or letting others use' \
            'a modified copy over a network, requires offering them the source under the' \
            'AGPL. Check that this is acceptable for the site.' \
            'To remove it: python -m pip uninstall auto-trainer-pose-yolo ultralytics' \
            'Continuing.'
    fi
    if ! conda_run python -m pip install -e "$INSTALL_REPO/plugins/$name"; then
        if [ "$name" = deeplabcut ]; then
            skip_step "Install TensorFlow pose engine" "DeepLabCut plugin install failed"
            skip_step "Install TensorFlow CUDA runtime" "DeepLabCut plugin install failed"
            skip_step "Verify TensorFlow GPU preflight" "DeepLabCut plugin install failed"
            skip_step "Verify both engines in one process" "DeepLabCut plugin install failed"
        fi
        return 1
    fi
    if [ "$name" = deeplabcut ]; then
        run_step "Install TensorFlow pose engine" install_tensorflow_engine
        run_step "Install TensorFlow CUDA runtime" install_tensorflow_gpu_runtime
        run_step "Verify TensorFlow GPU preflight" verify_tensorflow_gpu_runtime
        run_step "Verify both engines in one process" verify_pose_engines_together
    fi
}

install_tensorflow_engine() {
    install_extra "$INSTALL_REPO/plugins/deeplabcut/pyproject.toml" tensorflow || return
    # Installing TensorFlow 2.12 pulls numpy down to 1.24.3, typing-extensions
    # to 4.5 and filelock out of DeepLabCut's range. Put all three back.
    conda_run python -m pip install \
        'numpy >= 1.26, < 2' 'typing-extensions >= 4.10' 'filelock >= 3.12, < 3.16'
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

    # TensorFlow 2.12 needs CUDA 11.8 and cuDNN 8.6; PyTorch brings CUDA 12 and
    # cuDNN 9. Both arrive as nvidia-* wheels that install into the same
    # site-packages/nvidia directory, so the second install overwrote the
    # first: on christielab10 libcudnn.so.8 was gone and TensorFlow fell back
    # to the CPU without a word. The libraries themselves differ by soname
    # (libcudnn.so.8 and .9, libcublas.so.11 and .12), so TensorFlow's set goes
    # in a directory of its own, reached only through LD_LIBRARY_PATH, and
    # neither framework can load the other's.
    local env_prefix runtime_dir
    env_prefix=$(conda_run python -c 'import sys; print(sys.prefix)') || return
    runtime_dir="$env_prefix/lib/reachaq-tensorflow-cuda11"
    printf 'Installing CUDA 11.8 and cuDNN 8.6 libraries for TensorFlow %s into %s.\n' \
        "$tensorflow_version" "$runtime_dir"
    conda_run python -m pip install --upgrade --target "$runtime_dir" \
        'nvidia-cuda-runtime-cu11==11.8.89' \
        'nvidia-cuda-cupti-cu11==11.8.87' \
        'nvidia-cuda-nvrtc-cu11==11.8.89' \
        'nvidia-cublas-cu11==11.11.3.6' \
        'nvidia-cufft-cu11==10.9.0.58' \
        'nvidia-curand-cu11==10.3.0.86' \
        'nvidia-cusolver-cu11==11.4.1.48' \
        'nvidia-cusparse-cu11==11.7.5.86' \
        'nvidia-cudnn-cu11==8.6.0.163' || return

    local nvidia_root="$runtime_dir/nvidia"
    local site_nvidia_root
    site_nvidia_root=$(conda_run python -c \
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
            # Drop this step's own entries, and the site-packages ones an
            # earlier version of it set - those now hold PyTorch's CUDA 12.
            case "$library_directory" in
                "$nvidia_root"/*/lib|"$site_nvidia_root"/*/lib)
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

# DeepLabCut 3 imports torch on import, so its TensorFlow engine always runs
# with both frameworks in one process. That aborted with torch's CUDA 13 build;
# check it rather than rely on the pins staying right.
verify_pose_engines_together() {
    conda_run python - <<'PY'
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import torch
import tensorflow as tf

torch_result = torch.nn.functional.conv2d(
    torch.ones((1, 3, 32, 32), device="cuda"), torch.ones((4, 3, 3, 3), device="cuda"))
torch.cuda.synchronize()
with tf.device("/GPU:0"):
    tf_result = tf.nn.conv2d(tf.ones((1, 32, 32, 3)), tf.ones((3, 3, 3, 4)), 1, "SAME")
if "GPU:0" not in tf_result.device:
    raise SystemExit("TensorFlow convolution did not execute on GPU:0")
print("Both engines in one process: PyTorch on", torch_result.device,
      "and TensorFlow on", tf_result.device)
PY
}

"""The pose plugin installer and the functions it shares with the main installer.

Nothing here runs pip or a real conda. The shell functions are exercised
against a fake `conda` that logs each call and runs only the inline Python
snippets, with `-S` and a scratch directory as the environment's sys.path, so
the entry points and packages an environment "has" are the ones a test writes.
"""

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_DIR = REPO_ROOT / "tools" / "install"
PLUGIN_SCRIPT = INSTALL_DIR / "reachaq-plugin-install.sh"
PLUGIN_LIB = INSTALL_DIR / "plugin-install-lib.sh"
MAIN_SCRIPT = INSTALL_DIR / "reachaq-linux-install.sh"

AGPL_NOTICE = "reachaq-pose-yolo installs ultralytics, which is AGPL-3.0."

FAKE_CONDA = r"""#!/usr/bin/env bash
# Logs every call. `env list` lists $FAKE_ENVS; `run -n ENV python ...` runs
# the snippet against $FAKE_ENVS/ENV, except pip, which is only logged.
printf 'FAKE CONDA: %s\n' "$*"
printf '%s\n' "$*" >> "$FAKE_CONDA_LOG"
case "$1" in
    env)
        [ "$2" = list ] && ls -1 "$FAKE_ENVS"
        exit 0
        ;;
    run) shift ;;
    *) exit 0 ;;
esac
env_name=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-capture-output) shift ;;
        -n) env_name=$2; shift 2 ;;
        *) break ;;
    esac
done
if [ ! -d "$FAKE_ENVS/$env_name" ]; then
    printf 'EnvironmentLocationNotFound: %s\n' "$env_name" >&2
    exit 1
fi
[ "$1" = python ] || exit 0
shift
[ "$1 $2" = "-m pip" ] && exit 0
if [ -e "$FAKE_ENVS/$env_name/FAIL_PYTHON" ]; then
    printf 'probe python broke in %s\n' "$env_name" >&2
    exit 3
fi
PYTHONPATH="$FAKE_ENVS/$env_name" exec "$FAKE_PYTHON" -S "$@"
"""


def _fake_conda(tmp_path):
    bin_dir = tmp_path / "bin"
    envs = tmp_path / "envs"
    bin_dir.mkdir()
    envs.mkdir()
    conda = bin_dir / "conda"
    conda.write_text(FAKE_CONDA, newline="\n")
    conda.chmod(0o755)
    inherited = {key: value for key, value in os.environ.items()
                 if not key.startswith("REACHAQ_")}
    env = {
        **inherited,
        "PATH": f"{bin_dir}{os.pathsep}{inherited.get('PATH', '')}",
        "FAKE_ENVS": str(envs),
        "FAKE_PYTHON": sys.executable,
        "FAKE_CONDA_LOG": str(tmp_path / "conda.log"),
    }
    return conda, envs, env


def _make_env(envs, name, *, entry_points=(), packages=()):
    root = envs / name
    root.mkdir()
    if entry_points:
        dist = root / "fake_plugins-1.0.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: fake-plugins\nVersion: 1.0\n")
        (dist / "entry_points.txt").write_text(
            "[reachaq.pose_backends]\n"
            + "".join(f"{name} = fake_module:THING\n" for name in entry_points))
    for package in packages:
        (root / package).mkdir()
        (root / package / "__init__.py").write_text("")
    return root


def _source_lib(script, env, conda):
    """Run `script` in bash after sourcing the library, with CONDA_BIN set."""
    prelude = (
        f'CONDA_BIN="{conda}"\n'
        f'INSTALL_REPO="{REPO_ROOT}"\n'
        'INSTALL_ENV=reachaq\n'
        f'. "{PLUGIN_LIB}" || exit 99\n'
    )
    return subprocess.run(
        ["bash", "-c", prelude + script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_plugin_script(*args, env=None):
    return subprocess.run(
        [str(PLUGIN_SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_plugin_installer_rejects_no_or_unknown_names():
    assert os.access(PLUGIN_SCRIPT, os.X_OK)

    for args in ((), ("onnx",), ("deeplabcut", "onnx")):
        completed = _run_plugin_script(*args)

        assert completed.returncode == 2, (args, completed.stdout, completed.stderr)
        assert "Usage: reachaq-plugin-install.sh" in completed.stderr, args
        assert "deeplabcut" in completed.stderr and "yolo" in completed.stderr, args


def test_plugin_installer_refuses_an_environment_that_does_not_exist(tmp_path):
    conda, envs, env = _fake_conda(tmp_path)
    env["REACHAQ_INSTALL_ENV"] = "missing-env"

    completed = _run_plugin_script("yolo", env=env)

    assert completed.returncode == 1
    assert "missing-env" in completed.stderr
    assert "reachaq-linux-install.sh" in completed.stderr
    assert "pip install" not in Path(env["FAKE_CONDA_LOG"]).read_text()


def test_plugin_installer_warns_before_installing_yolo(tmp_path):
    # The notice is printed in the yolo branch ahead of its pip install, both
    # in the source and in what the operator actually sees.
    source = PLUGIN_LIB.read_text()
    function = source[source.index("install_pose_plugin() {"):]
    function = function[:function.index("\n}\n")]
    assert AGPL_NOTICE in function
    assert function.index(AGPL_NOTICE) < function.index('pip install -e "$INSTALL_REPO/plugins/')

    conda, envs, env = _fake_conda(tmp_path)
    _make_env(envs, "reachaq")
    env["REACHAQ_INSTALL_ENV"] = "reachaq"

    completed = _run_plugin_script("yolo", env=env)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    pip_line = f"pip install -e {REPO_ROOT}/plugins/yolo"
    assert AGPL_NOTICE in completed.stdout
    assert pip_line in completed.stdout
    assert completed.stdout.index(AGPL_NOTICE) < completed.stdout.index(pip_line)
    assert "PASS  Pose plugins | Install yolo pose plugin" in completed.stdout
    # yolo needs no TensorFlow; only deeplabcut brings those steps.
    assert "TensorFlow" not in completed.stdout


def test_deeplabcut_plugin_runs_the_tensorflow_steps(tmp_path):
    conda, envs, env = _fake_conda(tmp_path)
    _make_env(envs, "reachaq")

    completed = _source_lib(
        "run_step() { printf 'STEP %s\\n' \"$1\"; }\n"
        "install_pose_plugin deeplabcut\n",
        env, conda)

    assert completed.returncode == 0, completed.stderr
    log = Path(env["FAKE_CONDA_LOG"]).read_text()
    assert f"pip install -e {REPO_ROOT}/plugins/deeplabcut" in log
    steps = [line[5:] for line in completed.stdout.splitlines() if line.startswith("STEP ")]
    assert steps == [
        "Install TensorFlow pose engine",
        "Install TensorFlow CUDA runtime",
        "Verify TensorFlow GPU preflight",
        "Verify both engines in one process",
    ]
    assert AGPL_NOTICE not in completed.stdout


def test_tensorflow_requirements_come_from_the_deeplabcut_plugin(tmp_path):
    source = PLUGIN_LIB.read_text()

    assert '"$INSTALL_REPO/plugins/deeplabcut/pyproject.toml" tensorflow' in source
    # The overrides TensorFlow 2.12's caps need, and its CUDA runtime kept out
    # of site-packages/nvidia, move with it.
    assert "'numpy >= 1.26, < 2' 'typing-extensions >= 4.10'" in source
    assert '--target "$runtime_dir"' in source


def test_installed_pose_plugins_lists_known_entry_points(tmp_path):
    conda, envs, env = _fake_conda(tmp_path)
    _make_env(envs, "with-plugins", entry_points=("yolo", "onnx", "deeplabcut"))
    _make_env(envs, "bare")

    completed = _source_lib(
        'echo "with: $(installed_pose_plugins with-plugins | tr "\\n" " ")"\n'
        'echo "bare: $(installed_pose_plugins bare | tr "\\n" " ")"\n'
        'echo "missing: $(installed_pose_plugins no-such-env | tr "\\n" " ")"\n',
        env, conda)

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    # Known names only, in KNOWN_PLUGINS order; an environment that does not
    # exist has none.
    assert "with: deeplabcut yolo " in lines
    assert "bare: " in lines
    assert "missing: " in lines


def test_installed_pose_plugins_reads_the_v3_group_too(tmp_path):
    """v3 registered its plugins under autotrainer.pose_backends; the first update
    after the rename must still restore them."""
    conda, envs, env = _fake_conda(tmp_path)
    dist = _make_env(envs, "v3") / "auto_trainer_pose_deeplabcut-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: auto-trainer-pose-deeplabcut\nVersion: 1.0\n")
    (dist / "entry_points.txt").write_text(
        "[autotrainer.pose_backends]\ndeeplabcut = fake_module:THING\n")

    completed = _source_lib('echo "v3: $(installed_pose_plugins v3 | tr "\\n" " ")"\n', env, conda)

    assert completed.returncode == 0, completed.stderr
    assert "v3: deeplabcut " in completed.stdout.splitlines()


def test_legacy_pose_engines_infers_plugins_from_old_packages(tmp_path):
    conda, envs, env = _fake_conda(tmp_path)
    _make_env(envs, "dlc-only", packages=("deeplabcut",))
    _make_env(envs, "both", packages=("deeplabcut", "ultralytics"))
    _make_env(envs, "neither")

    completed = _source_lib(
        'for e in dlc-only both neither no-such-env; do\n'
        '    echo "$e: $(legacy_pose_engines "$e" | tr "\\n" " ")"\n'
        'done\n',
        env, conda)

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert "dlc-only: deeplabcut " in lines
    assert "both: deeplabcut yolo " in lines
    assert "neither: " in lines
    assert "no-such-env: " in lines


def _find_plugins_to_restore_source():
    source = MAIN_SCRIPT.read_text()
    function = source[source.index("find_plugins_to_restore() {"):]
    return function[:function.index("\n}\n") + 3]


REPORT_PRELUDE = (
    'CURRENT_CATEGORY=Test; RESULT_NAMES=(); RESULT_STATES=(); RESULT_DETAILS=()\n'
    'PASS_COUNT=0; FAIL_COUNT=0; SKIP_COUNT=0\n'
)


def test_a_failed_probe_of_an_existing_env_fails_the_step(tmp_path):
    # An empty answer from a broken environment would restore nothing and
    # report PASS; the rebuilt environment would then lose its engines.
    conda, envs, env = _fake_conda(tmp_path)
    (_make_env(envs, "broken", packages=("deeplabcut",)) / "FAIL_PYTHON").write_text("")

    completed = _source_lib(
        REPORT_PRELUDE + _find_plugins_to_restore_source()
        + 'PLUGINS_TO_RESTORE=(yolo)\n'
        'installed_pose_plugins broken; echo "installed status=$?"\n'
        'legacy_pose_engines broken; echo "legacy status=$?"\n'
        'find_plugins_to_restore broken; echo "find status=$?"\n'
        'echo "kept=${PLUGINS_TO_RESTORE[*]}"\n',
        env, conda)

    assert completed.returncode == 0, completed.stderr
    assert "installed status=3" in completed.stdout
    assert "legacy status=3" in completed.stdout
    assert "find status=3" in completed.stdout
    assert "kept=yolo" in completed.stdout
    assert "probe python broke in broken" in completed.stderr
    assert "Probing the pose plugins of conda environment broken failed" in completed.stderr


def test_a_missing_env_has_no_plugins_to_restore(tmp_path):
    conda, envs, env = _fake_conda(tmp_path)

    completed = _source_lib(
        REPORT_PRELUDE + _find_plugins_to_restore_source()
        + 'installed_pose_plugins no-such-env; echo "installed status=$?"\n'
        'legacy_pose_engines no-such-env; echo "legacy status=$?"\n'
        'run_step "Find pose plugins to restore" find_plugins_to_restore no-such-env\n'
        'echo "states=${RESULT_STATES[*]} count=${#PLUGINS_TO_RESTORE[@]}"\n',
        env, conda)

    assert completed.returncode == 0, completed.stderr
    assert "installed status=0" in completed.stdout
    assert "legacy status=0" in completed.stdout
    assert "Pose plugins to restore from no-such-env: none" in completed.stdout
    assert "states=PASS count=0" in completed.stdout


def test_a_failed_archive_probe_is_reported_without_stopping_the_rebuild():
    source = MAIN_SCRIPT.read_text()
    archive = source[source.index("archive_conda_environment() {"):]
    archive = archive[:archive.index("\n}\n")]

    # Stopping here would leave no environment of that name, and the rerun
    # would restore nothing: a WARN, not a FAIL.
    assert 'if ! find_plugins_to_restore "$archive_name"; then' in archive
    assert 'warn_step "Find pose plugins to restore from $archive_name"' in archive
    assert archive.rstrip().endswith("fi")


def test_a_failed_step_stops_and_names_itself(tmp_path):
    conda, envs, env = _fake_conda(tmp_path)

    completed = _source_lib(
        REPORT_PRELUDE + 'REPORT_PRINTED=false; RERUN_HINT=again.sh\n'
        'run_step "First" true\n'
        'run_step "Broken step" sh -c "echo why >&2; exit 7"\n'
        'echo "NOT REACHED"\n',
        env, conda)

    assert completed.returncode == 1
    assert "NOT REACHED" not in completed.stdout
    assert "[FAIL] Broken step (exit 7)" in completed.stderr
    assert "why" in completed.stderr
    report = completed.stdout[completed.stdout.index("reachAQ install report"):]
    assert "PASS  Test | First" in report
    assert "FAIL  Test | Broken step (exit 7)" in report
    assert "STOPPED at step: Test | Broken step" in report
    assert "Failed command: sh -c" in report
    assert "rerun again.sh" in report


def test_a_warning_is_reported_without_stopping(tmp_path):
    conda, envs, env = _fake_conda(tmp_path)

    completed = _source_lib(
        REPORT_PRELUDE + 'REPORT_PRINTED=false; WARN_COUNT=0\n'
        'warn_step "Odd host" "override set"\n'
        'run_step "After" true\n'
        'print_report\n',
        env, conda)

    assert completed.returncode == 0, completed.stderr
    assert "[WARN] Odd host: override set" in completed.stderr
    assert "WARN  Test | Odd host (override set)" in completed.stdout
    assert "PASS=1 FAIL=0 SKIP=0 WARN=1" in completed.stdout
    assert "completed with warnings" in completed.stdout


def test_no_installer_step_carries_on_after_a_failure():
    for script in (MAIN_SCRIPT, PLUGIN_SCRIPT, PLUGIN_LIB):
        source = script.read_text()
        assert "continuing" not in source, script.name
        # A FAIL recorded by hand must stop like run_step does.
        lines = source.splitlines()
        for index, line in enumerate(lines):
            if "record_result FAIL" in line and script is not PLUGIN_LIB:
                following = "\n".join(lines[index:index + 6])
                assert "stop_install" in following, (script.name, line)


def test_plugin_library_has_no_side_effects_when_sourced(tmp_path):
    conda, envs, env = _fake_conda(tmp_path)

    completed = _source_lib('echo "known: ${KNOWN_PLUGINS[*]}"\n', env, conda)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "known: deeplabcut yolo\n"
    assert completed.stderr == ""
    assert not Path(env["FAKE_CONDA_LOG"]).exists()


def test_main_installer_restores_installed_and_legacy_plugins():
    source = MAIN_SCRIPT.read_text()
    flow = source[source.index('begin_category "Preflight"'):]

    create = flow.index('run_step "Create conda environment" ensure_conda_environment')
    find = flow.index('run_step "Find pose plugins to restore" find_plugins_to_restore "$INSTALL_ENV"')
    assert find < create
    editable = flow.index('run_step "Install reachAQ editable package" install_editable_package')
    restore = flow.index('run_step "Restore $name pose plugin" install_pose_plugin "$name"')
    assert editable < restore

    helper = source[source.index("find_plugins_to_restore() {"):]
    helper = helper[:helper.index("\n}\n")]
    assert 'installed_pose_plugins "$env"' in helper
    assert 'legacy_pose_engines "$env"' in helper
    assert '. "$SCRIPT_DIR/plugin-install-lib.sh"' in source


def test_archived_environment_is_the_restore_source():
    source = MAIN_SCRIPT.read_text()
    archive = source[source.index("archive_conda_environment() {"):]
    archive = archive[:archive.index("\n}\n")]

    rename = archive.index('"$CONDA_BIN" rename -n "$INSTALL_ENV" "$archive_name"')
    assert archive.index('find_plugins_to_restore "$archive_name"') > rename


def test_main_installer_installs_no_plugin_by_default():
    source = MAIN_SCRIPT.read_text()

    install_calls = [line.strip() for line in source.splitlines()
                     if "install_pose_plugin" in line and not line.strip().startswith("#")]
    assert install_calls == [
        'run_step "Restore $name pose plugin" install_pose_plugin "$name"'
    ]
    restore_loop = source[source.index('for name in "${PLUGINS_TO_RESTORE[@]}"'):]
    assert restore_loop.index("install_pose_plugin") < restore_loop.index("done")
    for step in (
        "Install TensorFlow pose engine",
        "Verify TensorFlow GPU preflight",
        "Verify both engines in one process",
    ):
        assert step not in source, step


def test_main_installer_verifies_each_restored_plugin():
    source = MAIN_SCRIPT.read_text()
    verify = source[source.index("verify_imports() {"):]
    verify = verify[:verify.index("\nPY\n")]

    assert 'conda_run python - "$INSTALL_REPO" "${PLUGINS_TO_RESTORE[@]}"' in verify
    assert "import torch" in verify
    assert 'f"reachaq.pose_plugins.{name}.plugin"' in verify


def test_installer_scripts_parse():
    for script in (MAIN_SCRIPT, PLUGIN_SCRIPT, PLUGIN_LIB):
        completed = subprocess.run(["bash", "-n", str(script)],
                                   capture_output=True, text=True, check=False)
        assert completed.returncode == 0, (script.name, completed.stderr)

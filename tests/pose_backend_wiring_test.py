"""The selected backend has to reach both the GPU probe and the pose process.

These are the two places where a half-wired switch is dangerous rather than
merely wrong:

  * check_live_inference_runtime used to hardcode "tensorflow". On the reachAQ
    rig that probe passes while every torch convolution aborts in cuDNN, so a
    torch deployment would clear its preflight and then crash the pose process.

  * PoseProcess constructs the model. If it kept constructing DlcPoseModel the
    environment variable would appear to work - the preflight would report
    torch - while inference still ran on TensorFlow.

The pose process is checked by reading the wiring rather than by starting it: a
real start needs a GPU, a trained model and a multiprocessing queue.
"""

import inspect
import pathlib
import re

from autotrainer.inference import pose_process
from autotrainer.inference.backend_selection import (
    POSE_BACKEND_ENV_VAR,
    TENSORFLOW_BACKEND,
    TORCH_BACKEND,
)


def _inference_model_source() -> str:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    path = repo_root / "tools" / "acquisition" / "model" / "inference_model.py"
    return path.read_text(encoding="utf-8")


def test_the_runtime_probe_is_no_longer_pinned_to_tensorflow():
    source = _inference_model_source()
    assert 'required_backend="tensorflow"' not in source
    # Matched loosely on purpose: what matters is that the probe asks the
    # selector rather than naming an engine, not how the call is spelled. The
    # literal form used to be asserted, and adding the model path to the call
    # failed the test while strengthening exactly what it guards.
    assert re.search(r"required_backend=selected_backend\(", source)


def test_the_runtime_probe_tells_the_selector_which_model_is_configured():
    """A YOLO model is torch whatever the environment defaults to.

    The model is chosen in the configuration and the backend only through the
    environment, so a probe that does not pass the model can demand a
    TensorFlow runtime for torch weights and refuse to start.
    """
    source = _inference_model_source()
    assert re.search(
        r"selected_backend\(\s*model_path=self\._model_location\s*\)", source)


def test_the_runtime_probe_imports_the_selector():
    source = _inference_model_source()
    assert "from autotrainer.inference.backend_selection import" in source
    assert "selected_backend" in source


def test_can_start_live_inference_also_validates_the_model():
    """A bad model path must fail in the parent, not inside the child process."""
    source = _inference_model_source()
    assert "_can_load_pose_model" in source
    assert "def can_start_live_inference" in source


def test_an_empty_model_location_stays_valid():
    """An empty location selects MemoryPoseModel for rig checkout."""
    source = _inference_model_source()
    body = source.split("def _can_load_pose_model")[1]
    assert "if not self._model_location:" in body
    assert "return True" in body


def test_the_pose_process_builds_through_the_selector():
    source = inspect.getsource(pose_process)
    assert "build_pose_model(" in source
    # The direct construction is what the selector replaces.
    assert "DlcPoseModel(" not in source


def test_the_pose_process_no_longer_imports_the_tensorflow_model():
    """A torch-only deployment cannot import DlcPoseModel at all."""
    source = inspect.getsource(pose_process)
    assert "import DlcPoseModel" not in source
    assert "MemoryPoseModel" in source, "the in-memory model is still needed"


def test_the_pose_process_logs_which_backend_it_loaded():
    source = inspect.getsource(pose_process)
    assert re.search(r"selected_backend\(", source)
    assert "backend" in source.split("Loading DLC model")[1][:200]


def test_the_pose_process_selects_from_the_model_it_was_given():
    """The child has to reach the same answer as the parent's probe."""
    source = inspect.getsource(pose_process)
    assert re.search(r"selected_backend\(\s*model_path=model_path\s*\)", source)


def test_the_env_var_name_is_stable():
    """Operators put this in the rig's service environment; renaming it breaks them."""
    assert POSE_BACKEND_ENV_VAR == "REACHAQ_POSE_BACKEND"
    assert TENSORFLOW_BACKEND == "tensorflow"
    assert TORCH_BACKEND == "torch"

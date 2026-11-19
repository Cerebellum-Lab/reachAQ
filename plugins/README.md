# Pose backend plugins

A pose engine ships as its own distribution under `plugins/<name>` and
registers itself in the `autotrainer.pose_backends` entry-point group. reachAQ
core installs PyTorch for the built-in keypoint backend and no optional pose
plugin, so TensorFlow arrives only with the DeepLabCut plugin. It finds the
installed plugins, asks each whether it owns a model directory, and runs the model through the one that does. Core
never imports an engine to decide that.

| Plugin | Distribution | Claims a model directory with | Runs on |
|---|---|---|---|
| (built in) `keypoint` | `auto-trainer-inference` | `keypoint_pose.yaml` | torch |
| [`deeplabcut`](deeplabcut/README.md) | `auto-trainer-pose-deeplabcut` | `config.yaml` | tensorflow or torch |
| [`yolo`](yolo/README.md) | `auto-trainer-pose-yolo` (AGPL-3.0) | `yolo_pose.yaml` | torch |

Install one into an existing environment with
`tools/install/reachaq-plugin-install.sh deeplabcut|yolo` (see
[INSTALL.md](../INSTALL.md)). Run with no arguments it prints its usage and
exits 2.

## The contract

`autotrainer.inference.pose_plugins.PoseBackendPlugin`. Every method must be
cheap and import-free except `build()` and `pre_validate()`, which are only
called for the plugin that claimed the model. `missing_requirements()` uses
`find_spec`, not `import`, because importing TensorFlow to ask whether it is
present costs seconds and loads CUDA.

| Member | Meaning |
|---|---|
| `name` | Short unique name. Entry-point plugins are ordered by it. `keypoint` is reserved. |
| `distribution` | The pip distribution to name when requirements are missing. |
| `api_version` | Leave at the default, `POSE_PLUGIN_API_VERSION` (currently 1). A plugin built against another version is skipped, not called. |
| `requires` | Top-level module names the engine imports. |
| `claims(model_path)` | Whether this plugin owns the model at that location. Filesystem only. |
| `missing_requirements()` | The entries of `requires` that are not importable. Implemented for you. |
| `runtime(model_path, environ)` | The GPU runtime to probe for, as `gpu_runtime` names it: `"torch"` or `"tensorflow"`. |
| `runtimes()` | Optional. Every runtime the engine could run on here, for diagnostics. Defaults to the one `runtime(None, os.environ)` answers. |
| `pre_validate(model_path, runtime)` | Cheap filesystem check of the model; raises if unusable. |
| `build(model_path, runtime, *, shuffle_index, training_index, batch_size)` | Construct a `PoseModel`. The engine import happens here. |
| `confidence_threshold(model_path, runtime)` | The confidence a keypoint needs to count as present, on this engine's scale. Measure it on held-out frames; do not borrow another engine's. |

`build()` returns a `PoseModel` (`autotrainer.inference.pose_model`), so
callers never branch on the engine.

### Skeleton

```python
# src/autotrainer/pose_plugins/myengine/plugin.py
import os
import typing

from autotrainer.inference import pose_plugins
from autotrainer.inference.pose_model import PoseModel


class MyEnginePlugin(pose_plugins.PoseBackendPlugin):
    name = "myengine"
    distribution = "auto-trainer-pose-myengine"
    requires = ("myengine",)

    SIDECAR = "myengine_pose.yaml"

    def claims(self, model_path: str) -> bool:
        return bool(model_path) and os.path.isfile(os.path.join(model_path, self.SIDECAR))

    def runtime(self, model_path: typing.Optional[str],
                environ: typing.Mapping[str, str]) -> str:
        return "torch"

    def pre_validate(self, model_path: str, runtime: str) -> None:
        from .myengine_pose_model import MyEnginePoseModel  # engine imports stay in here
        MyEnginePoseModel.pre_validate(model_path)

    def build(self, model_path: str, runtime: str, *, shuffle_index: int,
              training_index: int, batch_size: int) -> PoseModel:
        from .myengine_pose_model import MyEnginePoseModel
        return MyEnginePoseModel(model_path, batch_size=batch_size)

    def confidence_threshold(self, model_path: typing.Optional[str],
                             runtime: str) -> float:
        return 0.1


MYENGINE_PLUGIN = MyEnginePlugin()
```

The package's `__init__.py` must not import the engine either: discovery
imports `plugin.py` on every rig that has the distribution. `plugins/yolo`
exports its model class lazily through a module `__getattr__`; copy that.

## Packaging and the entry point

Start from `plugins/yolo/pyproject.toml`, the smaller of the two. The parts
that matter:

```toml
[project]
name = "auto-trainer-pose-myengine"
license = { text = "..." }          # the distribution's own licence, see below
dependencies = [
    "auto-trainer",                  # the installed application
    "myengine >= 1, < 2",
]

[project.entry-points."autotrainer.pose_backends"]
myengine = "autotrainer.pose_plugins.myengine.plugin:MYENGINE_PLUGIN"
```

- The entry point may name an instance, a plugin class, or a factory function;
  anything callable that is not already a `PoseBackendPlugin` is called once.
- The `auto-trainer` dependency is required: the plugin imports
  `autotrainer.inference`, which is folded into that distribution. Never
  depend on `auto-trainer-inference` on its own.
- The package lives in the `autotrainer.pose_plugins.<name>` namespace;
  `[tool.setuptools.packages.find]` uses `namespaces = true` with an `include`
  for that package only.
- Add the name to `KNOWN_PLUGINS` in `tools/install/plugin-install-lib.sh` so
  the plugin installer accepts it.

A plugin that fails to load (bad entry point, wrong type, wrong `api_version`,
the name `keypoint`, or a name another plugin already provides) is skipped with
a warning and listed by `pose_plugins.broken_plugins()`. It does not stop live
inference.

## Detection rules

`pose_plugins.resolve(model_path)` picks the plugin for a model directory:

1. A `keypoint_pose.yaml` sidecar wins outright (`KEYPOINT_PLUGIN`). It is an
   explicit declaration, where a `config.yaml` beside it is only what
   DeepLabCut happens to look for.
2. Otherwise every installed plugin is asked `claims()`. Exactly one claimant
   runs the model. If its `missing_requirements()` is not empty,
   `PoseBackendNotInstalled` names the missing modules and the install command.
3. More than one claimant raises `AmbiguousPoseModel`, so give your engine a
   marker no other plugin claims. A sidecar of your own, as `keypoint` and
   `yolo` use, is safer than a generic file such as `config.yaml`.
4. No claimant: `KNOWN_LAYOUTS` is checked so the error can name the plugin to
   install; if nothing matches there either, `PoseBackendNotInstalled` is
   raised with `engine=None` and lists the plugins that are installed.

`backend_selection.selected_backend()` never raises. For a model no plugin can
run, or no model at all, it answers a recognised `REACHAQ_POSE_BACKEND`, then
`tensorflow` if it is installed, then `torch`. Otherwise
`REACHAQ_POSE_BACKEND` is read only by the DeepLabCut plugin.

## Adding a `KNOWN_LAYOUTS` entry

`KNOWN_LAYOUTS` in `auto-trainer-inference/src/autotrainer/inference/pose_plugins.py`
is what lets core say "this is a YOLO model, install `yolo`" when the plugin is
not installed, without importing it. Once an adapter exists, add a
`KnownLayout(engine, plugin, distribution, matches, hint="")`:

```python
KnownLayout("MyEngine", "myengine", "auto-trainer-pose-myengine",
            _is_myengine_layout)
```

`matches(model_path)` is a filesystem-only test of the marker the plugin claims
on. It can be stricter than `claims()`: the DeepLabCut plugin claims any
`config.yaml`, but the not-installed hint needs `config.yaml` plus a
`dlc-models*` directory. `hint`, optional, is appended to the install instruction for a layout with
a supported alternative; the YOLO entry offers retraining as a keypoint model.

## Licence rule

Check an engine's licence before writing its adapter. A GPL or AGPL engine
makes the plugin distribution carry the same licence (`plugins/yolo` is
AGPL-3.0-only because `ultralytics` is), and such an engine is never installed
in CI or in the product build: the CI job installs only `plugins/deeplabcut`.
The installer prints a licence notice before it installs one, so a site does so
knowingly.

## APT and SLEAP

No adapter exists for APT or SLEAP. Their licences and their model-directory
markers have not been checked, so none is stated here. Before starting one,
read the engine's licence (it decides whether the engine can ever be built or
tested in CI) and inspect a trained model directory for a marker that no other
plugin claims.

## Tests

Each plugin's tests live in its own `tests` directory, and the directories
share file basenames (`test_plugin.py`, `test_routing.py`), so run them
separately:

```bash
pytest tests auto-trainer-inference/tests -q   # core, with no plugin installed
pytest plugins/deeplabcut/tests -q             # needs the DeepLabCut plugin installed
pytest plugins/yolo/tests -q                   # needs the YOLO plugin installed
```

A plugin's `tests/conftest.py` resets discovery around each test, because
discovery is cached per process.

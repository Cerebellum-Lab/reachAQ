"""The contract a pose engine implements to plug into live inference, and how
the installed ones are found.

DeepLabCut and YOLO drag in TensorFlow, PyTorch and ultralytics, none of which
a rig needs all of. Each engine therefore ships as its own optional
distribution and advertises itself through the "autotrainer.pose_backends"
entry-point group; this module finds them and asks each one whether it owns a
model directory. Core never imports an engine to answer that: a plugin claims
a model from the filesystem and reports what it needs with find_spec, and only
build() imports the engine.

The keypoint backend is built in and always listed first, so a rig with no
plugin distribution installed still has one engine.

A plugin that fails to load must not take live inference down with it: a
half-removed distribution would otherwise stop every rig that merely has it
installed. Such plugins are skipped, logged once, and kept in broken_plugins()
so the operator-facing message can say why a model was refused.
"""

import functools
import importlib.metadata
import importlib.util
import os
import typing

from autotrainer.core.logging import get_verbose_logger

from .pose_model import PoseModel

logger = get_verbose_logger(__name__)

# Bumped when a method is added to, removed from, or changes meaning in
# PoseBackendPlugin. A plugin built against another version is skipped rather
# than called, because a missing method would surface mid-session.
POSE_PLUGIN_API_VERSION = 1
ENTRY_POINT_GROUP = "autotrainer.pose_backends"


def _installed(module: str) -> bool:
    """Whether an engine is importable, without paying to import it.

    find_spec only looks the module up; importing TensorFlow to ask whether
    TensorFlow is present would cost seconds and load CUDA.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # A partially removed distribution can leave a spec that raises.
        return False


class PoseBackendPlugin:
    """One pose engine. Subclass it and expose an instance (or the class) as
    an entry point in the "autotrainer.pose_backends" group.

    Every method must be cheap and import-free except build() and
    pre_validate(), which are only called for the plugin that claimed the model.
    """

    # Short unique name; entry-point plugins are ordered by it.
    name: str = ""
    # The pip distribution to name when its requirements are missing.
    distribution: str = ""
    api_version: int = POSE_PLUGIN_API_VERSION
    # Top-level module names the engine imports.
    requires: typing.Tuple[str, ...] = ()

    def claims(self, model_path: str) -> bool:
        """Whether this plugin owns the model at this location."""
        raise NotImplementedError

    def missing_requirements(self) -> typing.Tuple[str, ...]:
        """The entries of requires that are not importable."""
        return tuple(module for module in self.requires if not _installed(module))

    def runtime(self, model_path: typing.Optional[str],
                environ: typing.Mapping[str, str]) -> str:
        """The GPU runtime to probe for, as gpu_runtime names it."""
        raise NotImplementedError

    def pre_validate(self, model_path: str, runtime: str) -> None:
        """Cheap filesystem check of a model location; raises if unusable."""
        raise NotImplementedError

    def build(self, model_path: str, runtime: str, *, shuffle_index: int,
              training_index: int, batch_size: int) -> PoseModel:
        """Construct the model. The engine import happens here, not earlier."""
        raise NotImplementedError

    def confidence_threshold(self, model_path: typing.Optional[str],
                             runtime: str) -> float:
        """The confidence a keypoint needs to count as present, on this
        engine's scale."""
        raise NotImplementedError


class _KeypointPlugin(PoseBackendPlugin):
    """The built-in keypoint backend.

    The keypoint module and its threshold are looked up inside the methods so
    that importing this module, and discovering plugins, never imports torch.
    """

    name = "keypoint"
    distribution = "auto-trainer-inference"
    requires = ("torch",)

    SIDECAR = "keypoint_pose.yaml"

    def claims(self, model_path: str) -> bool:
        if not model_path:
            return False
        return os.path.isfile(os.path.join(model_path, self.SIDECAR))

    def runtime(self, model_path: typing.Optional[str],
                environ: typing.Mapping[str, str]) -> str:
        # These weights are torch and cannot be anything else.
        return "torch"

    def pre_validate(self, model_path: str, runtime: str) -> None:
        from .keypoint import KeypointPoseModel
        KeypointPoseModel.pre_validate(model_path)

    def build(self, model_path: str, runtime: str, *, shuffle_index: int,
              training_index: int, batch_size: int) -> PoseModel:
        from .keypoint import KeypointPoseModel
        return KeypointPoseModel(model_path, batch_size=batch_size)

    def confidence_threshold(self, model_path: typing.Optional[str],
                             runtime: str) -> float:
        from .backend_selection import KEYPOINT_CONFIDENCE_THRESHOLD
        return KEYPOINT_CONFIDENCE_THRESHOLD


KEYPOINT_PLUGIN = _KeypointPlugin()

# (entry point name, reason) for each entry point that could not be used.
_broken: typing.List[typing.Tuple[str, str]] = []


def _entry_points() -> typing.Iterable[typing.Any]:
    """The group's entry points; a seam for tests to replace."""
    return importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)


def _load(entry_point: typing.Any) -> PoseBackendPlugin:
    """Resolve one entry point to a plugin instance, or raise with the reason.

    An entry point may name an instance, a plugin class, or a factory function,
    so anything callable that is not already an instance is called once.
    """
    loaded = entry_point.load()
    if not isinstance(loaded, PoseBackendPlugin) and callable(loaded):
        loaded = loaded()
    if not isinstance(loaded, PoseBackendPlugin):
        raise TypeError(f"{entry_point.name!r} did not yield a PoseBackendPlugin "
                        f"(got {type(loaded).__name__})")
    if loaded.api_version != POSE_PLUGIN_API_VERSION:
        raise ValueError(f"api_version {loaded.api_version}, this build speaks "
                         f"{POSE_PLUGIN_API_VERSION}")
    return loaded


@functools.cache
def discover_plugins() -> typing.Tuple[PoseBackendPlugin, ...]:
    """The usable plugins: keypoint first, then the installed ones by name."""
    _broken.clear()
    found: typing.List[PoseBackendPlugin] = []
    for entry_point in _entry_points():
        try:
            found.append(_load(entry_point))
        except Exception as err:  # noqa: BLE001 - a third-party plugin can raise anything
            reason = f"{type(err).__name__}: {err}"
            _broken.append((entry_point.name, reason))
            logger.warning("pose backend plugin %r skipped: %s",
                           entry_point.name, reason)
    found.sort(key=lambda plugin: plugin.name)
    return (KEYPOINT_PLUGIN, *found)


def broken_plugins() -> typing.Tuple[typing.Tuple[str, str], ...]:
    """(entry point name, reason) for each plugin discovery had to skip."""
    discover_plugins()
    return tuple(_broken)


def reset_discovery() -> None:
    """Forget the cached discovery, for tests."""
    discover_plugins.cache_clear()
    _broken.clear()

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
plugin distribution installed still has one engine. DeepLabCut ships as the
auto-trainer-pose-deeplabcut distribution and YOLO as auto-trainer-pose-yolo;
each is found only through its entry point.

A plugin that fails to load must not take live inference down with it: a
half-removed distribution would otherwise stop every rig that merely has it
installed. Such plugins are skipped, logged once, and kept in broken_plugins()
so the operator-facing message can say why a model was refused.
"""

import functools
import glob
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

    def runtimes(self) -> typing.Tuple[str, ...]:
        """Every runtime this engine could run on here, for diagnostics.

        Optional: the default is the one runtime() answers without a model,
        which is all an engine with a fixed runtime has.
        """
        return (self.runtime(None, os.environ),)


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


def _skip(entry_point_name: str, reason: str) -> None:
    _broken.append((entry_point_name, reason))
    logger.warning("pose backend plugin %r skipped: %s", entry_point_name, reason)


@functools.cache
def discover_plugins() -> typing.Tuple[PoseBackendPlugin, ...]:
    """The usable plugins: keypoint first, then the installed ones by name."""
    _broken.clear()
    try:
        # Sorted so that which of two same-named plugins wins does not depend
        # on the order the distributions happened to be installed in.
        entry_points = sorted(_entry_points(), key=lambda point: point.name)
    except Exception as err:  # noqa: BLE001 - broken distribution metadata can raise anything
        # Losing the third-party plugins is bad; losing the built-in ones with
        # them would stop every rig that has a damaged environment.
        _skip(ENTRY_POINT_GROUP, f"{type(err).__name__}: {err}")
        entry_points = []

    found: typing.Dict[str, PoseBackendPlugin] = {}
    for entry_point in entry_points:
        try:
            plugin = _load(entry_point)
        except Exception as err:  # noqa: BLE001 - a third-party plugin can raise anything
            _skip(entry_point.name, f"{type(err).__name__}: {err}")
            continue
        if plugin.name == KEYPOINT_PLUGIN.name:
            _skip(entry_point.name,
                  f"plugin name {plugin.name!r} is reserved for the built-in backend")
        elif plugin.name in found:
            _skip(entry_point.name, f"plugin name {plugin.name!r} is already provided")
        else:
            found[plugin.name] = plugin

    return (KEYPOINT_PLUGIN, *sorted(found.values(), key=lambda plugin: plugin.name))


def broken_plugins() -> typing.Tuple[typing.Tuple[str, str], ...]:
    """(entry point name, reason) for each plugin discovery had to skip."""
    discover_plugins()
    return tuple(_broken)


def reset_discovery() -> None:
    """Forget the cached discovery, for tests."""
    discover_plugins.cache_clear()
    _broken.clear()


# --- routing ------------------------------------------------------------------

INSTALL_COMMAND = "tools/install/reachaq-plugin-install.sh"


class PoseBackendNotInstalled(ImportError):
    """No installed plugin can run this model.

    An ImportError because that is what it is, and because callers that
    already guard the engine import catch it. engine is None when the layout is
    not one core recognises either.
    """

    def __init__(self, model_path: typing.Optional[str], engine: typing.Optional[str],
                 distribution: typing.Optional[str] = None,
                 missing: typing.Sequence[str] = (), *,
                 plugin_name: typing.Optional[str] = None,
                 installed: typing.Sequence[str] = (),
                 hint: str = ""):
        self.model_path = model_path
        self.engine = engine
        self.distribution = distribution
        self.missing = tuple(missing)
        self.plugin_name = plugin_name
        self.installed = tuple(installed)
        self.hint = hint
        super().__init__(self._message())

    def _message(self) -> str:
        if self.engine is None:
            return (f"no installed pose backend recognises {self.model_path!r} "
                    f"(installed: {', '.join(self.installed) or 'none'}). If it is "
                    f"a model for another engine, install that engine's plugin "
                    f"with {INSTALL_COMMAND} <name>.")
        install = (f"Install it with {INSTALL_COMMAND} {self.plugin_name} "
                   f"(distribution {self.distribution}).")
        if self.hint:
            install = f"{install} {self.hint}"
        if self.missing:
            return (f"{self.model_path!r} is a {self.engine} model, and the "
                    f"{self.engine} pose backend cannot run here: "
                    f"{', '.join(self.missing)} not installed. {install}")
        return (f"{self.model_path!r} is a {self.engine} model, and the "
                f"{self.engine} pose backend plugin is not installed. {install}")


class AmbiguousPoseModel(ValueError):
    """More than one plugin claims the same model directory."""

    def __init__(self, model_path: str, names: typing.Sequence[str]):
        self.model_path = model_path
        self.names = tuple(names)
        super().__init__(
            f"{model_path!r} is claimed by more than one pose backend "
            f"({', '.join(self.names)}); remove the marker files of all but one")


class KnownLayout(typing.NamedTuple):
    """A model layout core can name the engine of without its plugin."""
    engine: str
    plugin: str
    distribution: str
    matches: typing.Callable[[str], bool]
    # Another way out, appended to the install instruction, for a layout with
    # a supported alternative.
    hint: str = ""


def _is_yolo_layout(model_path: str) -> bool:
    return os.path.isfile(os.path.join(model_path, "yolo_pose.yaml"))


def _is_deeplabcut_layout(model_path: str) -> bool:
    return (os.path.isfile(os.path.join(model_path, "config.yaml"))
            and any(os.path.isdir(path)
                    for path in glob.glob(os.path.join(model_path, "dlc-models*"))))


# ultralytics is AGPL-3.0, so the plugin is optional and the supported way
# off it is to retrain the model as a keypoint model.
YOLO_HINT = ("Alternatively, retrain the model as a keypoint model with "
             "python -m autotrainer.inference.keypoint.train.")


# Used only to word the error when nothing installed claims a model, so an
# operator is told which plugin to install rather than that the model is bad.
KNOWN_LAYOUTS: typing.Tuple[KnownLayout, ...] = (
    KnownLayout("YOLO", "yolo", "auto-trainer-pose-yolo", _is_yolo_layout,
                YOLO_HINT),
    KnownLayout("DeepLabCut", "deeplabcut", "auto-trainer-pose-deeplabcut",
                _is_deeplabcut_layout),
)


def _layout_for(plugin_name: str) -> typing.Optional[KnownLayout]:
    for layout in KNOWN_LAYOUTS:
        if layout.plugin == plugin_name:
            return layout
    return None


def _hint_for(plugin_name: str) -> str:
    layout = _layout_for(plugin_name)
    return layout.hint if layout else ""


def _engine_label(plugin_name: str) -> str:
    layout = _layout_for(plugin_name)
    return layout.engine if layout else plugin_name


def _claims(plugin: PoseBackendPlugin, model_path: str) -> bool:
    try:
        return bool(plugin.claims(model_path))
    except Exception as err:  # noqa: BLE001 - a third-party plugin can raise anything
        logger.warning("pose backend plugin %r failed to answer for %r: %s",
                       plugin.name, model_path, err)
        return False


def resolve(model_path: typing.Optional[str]) -> PoseBackendPlugin:
    """The plugin that runs the model at this location, or raise.

    The keypoint sidecar wins outright: it is an explicit declaration, where a
    config.yaml beside it is only what DeepLabCut happens to look for.

    Raises AmbiguousPoseModel when several plugins claim the directory, and
    PoseBackendNotInstalled when none can run it.
    """
    if KEYPOINT_PLUGIN.claims(model_path or ""):
        return KEYPOINT_PLUGIN
    plugins = discover_plugins()
    claiming = ([plugin for plugin in plugins
                 if plugin is not KEYPOINT_PLUGIN and _claims(plugin, model_path)]
                if model_path else [])
    if len(claiming) > 1:
        raise AmbiguousPoseModel(model_path, [plugin.name for plugin in claiming])
    if claiming:
        (plugin,) = claiming
        missing = plugin.missing_requirements()
        if missing:
            raise PoseBackendNotInstalled(model_path, _engine_label(plugin.name),
                                          plugin.distribution, missing,
                                          plugin_name=plugin.name,
                                          hint=_hint_for(plugin.name))
        return plugin
    if model_path:
        for layout in KNOWN_LAYOUTS:
            if layout.matches(model_path):
                raise PoseBackendNotInstalled(model_path, layout.engine,
                                              layout.distribution,
                                              plugin_name=layout.plugin,
                                              hint=layout.hint)
    raise PoseBackendNotInstalled(model_path, None,
                                  installed=[plugin.name for plugin in plugins])

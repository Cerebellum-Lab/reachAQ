"""YOLO-pose as a reachAQ pose backend plugin.

The model class is exported lazily: discovery imports .plugin on every rig that
has this distribution, and importing this package must not pull in ultralytics
or torch with it.
"""

import importlib

_LAZY = {
    "YoloPoseModel": ".yolo_pose_model",
}

__all__ = list(_LAZY)


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module, __name__), name)

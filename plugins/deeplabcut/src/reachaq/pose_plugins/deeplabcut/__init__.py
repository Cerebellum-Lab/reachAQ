"""DeepLabCut as a reachAQ pose backend plugin.

The model classes are exported lazily: discovery imports .plugin on every rig,
and importing this package must not pull in DeepLabCut, TensorFlow or torch
with them.
"""

import importlib

_LAZY = {
    "DlcPoseModel": ".dlc_pose_model",
    "DlcTorchPoseModel": ".dlc_torch_pose_model",
}

__all__ = list(_LAZY)


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module, __name__), name)

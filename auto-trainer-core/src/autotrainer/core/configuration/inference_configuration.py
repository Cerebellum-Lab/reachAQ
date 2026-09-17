from dataclasses import dataclass, field
from typing import Tuple

from autotrainer.core import make_camelize_representer, make_decamelize_constructor


@dataclass
class _InferenceConfiguration:
    pose_model_location: str = ""
    is_enabled: bool = False
    #: Parts to draw on the live overlay. Empty means every part the model
    #: emits, which is the useful default: the overlay used to be a fixed
    #: list inside the painter, and ten of this model's fourteen keypoints
    #: were silently undrawable. Name parts here only to narrow a cluttered
    #: view - never to make one drawable, which is now automatic.
    overlay_parts: Tuple[str, ...] = field(default_factory=tuple)

@dataclass
class InferenceConfiguration(_InferenceConfiguration):
    pass

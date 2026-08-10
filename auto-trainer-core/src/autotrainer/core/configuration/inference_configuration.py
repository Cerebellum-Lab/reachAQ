from dataclasses import dataclass

from autotrainer.core import make_camelize_representer, make_decamelize_constructor


@dataclass
class _InferenceConfiguration:
    pose_model_location: str = ""
    is_enabled: bool = False

@dataclass
class InferenceConfiguration(_InferenceConfiguration):
    pass

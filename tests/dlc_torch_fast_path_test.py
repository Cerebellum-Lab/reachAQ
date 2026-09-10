"""The lean live predict path must refuse itself whenever it cannot be exact.

It reimplements the runner's preprocessing, which is safe only because that
preprocessing is a single ImageNet Normalize for this project. If a trained
config ever specifies a resize, padding, a top-down crop or a second head, the
lean path would silently produce different coordinates - the failure that does
not announce itself. So the gate is tested exhaustively, and it defaults to
refusing anything it does not recognise.

Verified separately against the real model on the rig: the lean path agrees
with runner.inference() to 1.5e-5 px and 0.0 likelihood at batch 1, 2 and 6.
"""

import pytest

from autotrainer.inference.dlc import DlcTorchPoseModel


GOOD_CONFIG = {
    "data": {
        "inference": {
            "normalize_images": True,
            "resize": None,
            "auto_padding": None,
            "top_down_crop": None,
            "longest_max_size": None,
            "crop_sampling": None,
            "collate": None,
            "scale_to_unit_range": False,
            "grayscale": None,
        }
    },
    "model": {"heads": {"bodypart": {}}},
}


class _Runner:
    dynamic = None


def _model(config=None, device="cuda", runner=_Runner()):
    model = DlcTorchPoseModel("/unused", shuffle_index=2, batch_size=2)
    model._runner = runner
    model._device = device
    model._model_configuration = GOOD_CONFIG if config is None else config
    return model


def _config(**inference_changes):
    import copy

    config = copy.deepcopy(GOOD_CONFIG)
    config["data"]["inference"].update(inference_changes)
    return config


def test_the_known_good_configuration_is_accepted():
    assert _model()._fast_path_reason() is None


def test_an_unloaded_model_refuses():
    model = DlcTorchPoseModel("/unused")
    assert "not loaded" in model._fast_path_reason()


@pytest.mark.parametrize("device", ["cpu", "mps", ""])
def test_a_non_cuda_device_refuses(device):
    """The buffers are pinned host memory and device tensors."""
    assert "device is" in _model(device=device)._fast_path_reason()


def test_cuda_with_an_index_is_accepted():
    assert _model(device="cuda:1")._fast_path_reason() is None


def test_a_dynamic_cropper_refuses():
    class _Dynamic:
        pass

    class _WithDynamic:
        dynamic = _Dynamic()

    reason = _model(runner=_WithDynamic())._fast_path_reason()
    assert "dynamic cropper" in reason


def test_normalisation_being_off_refuses():
    """The lean path always normalises, so it must not run when it should not."""
    reason = _model(_config(normalize_images=False))._fast_path_reason()
    assert "normalize_images" in reason


@pytest.mark.parametrize("key", [
    "resize", "auto_padding", "top_down_crop", "longest_max_size",
    "crop_sampling", "collate",
])
def test_any_geometry_transform_refuses(key):
    """These would move coordinates, so guessing is not acceptable."""
    reason = _model(_config(**{key: {"width": 256, "height": 256}}))._fast_path_reason()
    assert key in reason


@pytest.mark.parametrize("key", ["scale_to_unit_range", "grayscale"])
def test_an_unsupported_scaling_transform_refuses(key):
    reason = _model(_config(**{key: True}))._fast_path_reason()
    assert "scaling transform" in reason


def test_a_second_head_refuses():
    """Only the bodypart head is extracted; another would be dropped silently."""
    import copy

    config = copy.deepcopy(GOOD_CONFIG)
    config["model"]["heads"]["unique_bodypart"] = {}
    reason = _model(config)._fast_path_reason()
    assert "bodypart head" in reason


def test_a_missing_inference_section_refuses():
    """An unrecognised config must not be assumed to need no preprocessing."""
    reason = _model({"model": {"heads": {"bodypart": {}}}})._fast_path_reason()
    assert "normalize_images" in reason


def test_the_normalisation_constants_match_the_trained_config():
    """Read off the project's own pytorch_config: albumentations Normalize with
    the ImageNet statistics and max_pixel_value 255."""
    assert DlcTorchPoseModel.IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert DlcTorchPoseModel.IMAGENET_STD == (0.229, 0.224, 0.225)


def test_the_fast_path_is_off_until_load_decides():
    """A model that never loaded must not take the lean path."""
    assert DlcTorchPoseModel("/unused")._fast_path_enabled is False

from urllib.parse import parse_qs, urlsplit

import numpy
import pytest

from autotrainer.core import FixedArrayMultiQueue
from autotrainer.core.capture import CaptureProcessStatus
from autotrainer.core.multiproc import get_mp_ctx


def _random_base(model, **extra):
    # A RandomCam stands in for the Blackfly: it takes width/height and ignores
    # hbin, offsets and gain (logging "unknown property"), which is all a
    # preset needs to be exercised end to end without hardware.
    conf = model.save_configuration()
    conf.scheme, conf.host, conf.path = "random", "binning", ""
    conf.params.update(width=256, height=256, hbin=4, vbin=4, offsetx=52, offsety=6,
                       gain=1, fps=30, primary="true", **extra)
    model.load_configuration(conf)


def test_a_preset_rewrites_only_the_runtime_url(video_capture_model):
    _random_base(video_capture_model, capture_binning=2)

    query = parse_qs(urlsplit(video_capture_model._runtime_camera_url()).query)

    assert {k: query[k] for k in ("hbin", "vbin", "width", "height", "offsetx", "offsety", "gain")} == {
        "hbin": ["2"], "vbin": ["2"], "width": ["512"], "height": ["512"],
        "offsetx": ["104"], "offsety": ["12"], "gain": ["13.0412"]}
    assert "capture_binning" not in query
    saved = video_capture_model.save_configuration().params
    assert (saved["width"], saved["hbin"], saved["capture_binning"]) == (256, 4, 2)
    assert video_capture_model.shape == (256, 256)
    assert video_capture_model.capture_shape == (512, 512)
    assert video_capture_model.capture_binning == 2
    assert video_capture_model.base_binning == 4
    assert video_capture_model.effective_capture_binning == 2


def test_setting_and_clearing_the_preset_round_trips(video_capture_model):
    _random_base(video_capture_model)
    assert video_capture_model.capture_binning is None
    assert video_capture_model.effective_capture_binning == 4

    video_capture_model.set_capture_binning(1)
    assert video_capture_model.capture_shape == (1024, 1024)

    video_capture_model.set_capture_binning(None)
    assert "capture_binning" not in video_capture_model.save_configuration().params
    assert video_capture_model.capture_shape == (256, 256)


def test_an_impossible_preset_fails_before_the_start_allocates_anything(video_capture_model):
    _random_base(video_capture_model, capture_binning=3)
    with pytest.raises(ValueError, match="whole factor"):
        video_capture_model.on_prepare_capture()
    assert video_capture_model._video_capture is None


def test_a_bin_2_capture_feeds_averaged_256_frames_to_inference(video_capture_model):
    _random_base(video_capture_model, capture_binning=2)
    net_q = FixedArrayMultiQueue(2, 1, 1, shape=video_capture_model.shape, primary=0,
                                 name="binning_q", mp_ctx=get_mp_ctx())
    assert video_capture_model.on_prepare_capture(net_q, inference_index=0) is True
    video_capture_model.on_capture_start()
    try:
        assert video_capture_model.wait_for_capture_status(CaptureProcessStatus.RUNNING, timeout=10)
        assert video_capture_model.wait_for_first_frame(timeout=10)
        out = numpy.zeros((1, 256, 256, 3))
        assert net_q.get_output(out, timeout=3)
        # 2x2 averages of uniform noise: std about half the raw ~73.6.
        assert 25 < out[0, :, :, 0].std() < 50
    finally:
        video_capture_model.on_capture_stop()

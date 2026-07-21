from tools.acquisition.args import make_autotrainer_parser
from tools.acquisition.model.app_model_status import AppModelStatus


def test_gui_parser_defaults_to_idle_with_configured_inference():
    args = make_autotrainer_parser().parse_args([])

    assert args.start_mode is AppModelStatus.IDLE
    assert args.live_inference is None


def test_live_inference_cli_overrides():
    parser = make_autotrainer_parser()

    assert parser.parse_args(["--live-inference"]).live_inference is True
    assert parser.parse_args(["--no-live-inference"]).live_inference is False


def test_headless_can_keep_acquiring_default():
    args = make_autotrainer_parser(default_start_mode=AppModelStatus.ACQUIRING).parse_args([])

    assert args.start_mode is AppModelStatus.ACQUIRING

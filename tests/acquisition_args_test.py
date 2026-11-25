from reachaq.app.args import make_reachaq_parser
from reachaq.app.model.app_model_status import AppModelStatus


def test_gui_parser_defaults_to_idle_with_configured_inference():
    args = make_reachaq_parser().parse_args([])

    assert args.start_mode is AppModelStatus.IDLE
    assert args.live_inference is None


def test_live_inference_cli_overrides():
    parser = make_reachaq_parser()

    assert parser.parse_args(["--live-inference"]).live_inference is True
    assert parser.parse_args(["--no-live-inference"]).live_inference is False


def test_headless_can_keep_acquiring_default():
    args = make_reachaq_parser(default_start_mode=AppModelStatus.RUNNING).parse_args([])

    assert args.start_mode is AppModelStatus.RUNNING

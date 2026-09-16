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
    args = make_autotrainer_parser(default_start_mode=AppModelStatus.RUNNING).parse_args([])

    assert args.start_mode is AppModelStatus.RUNNING


def test_demo_flag_defaults_to_none():
    args = make_autotrainer_parser().parse_args([])

    assert args.demo is None


def test_bare_demo_flag_uses_the_default_spec_path():
    from pathlib import Path

    from autotrainer.core.configuration.demo_sources import DEFAULT_DEMO_SOURCES_PATH

    args = make_autotrainer_parser().parse_args(["--demo"])

    assert Path(args.demo) == DEFAULT_DEMO_SOURCES_PATH


def test_demo_flag_accepts_an_explicit_path():
    from pathlib import Path

    args = make_autotrainer_parser().parse_args(["--demo", "/tmp/custom.yaml"])

    assert Path(args.demo) == Path("/tmp/custom.yaml")


def test_demo_and_random_cameras_are_mutually_exclusive():
    import pytest

    with pytest.raises(SystemExit):
        make_autotrainer_parser().parse_args(["--demo", "--random-cameras"])

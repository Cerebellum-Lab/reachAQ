from urllib.parse import parse_qs, urlsplit


def test_spinnaker_runtime_role_does_not_mutate_configured_role(
    video_capture_model,
):
    configuration = video_capture_model.save_configuration()
    configuration.scheme = "spinnaker"
    configuration.host = "24095781"
    configuration.params["primary"] = "false"
    video_capture_model.load_configuration(configuration)

    assert video_capture_model.configured_is_primary is False
    assert video_capture_model.is_primary is False

    video_capture_model.set_runtime_primary(True)
    runtime_query = parse_qs(
        urlsplit(video_capture_model._runtime_camera_url()).query
    )

    assert runtime_query["primary"] == ["yes"]
    assert video_capture_model.save_configuration().params["primary"] is False

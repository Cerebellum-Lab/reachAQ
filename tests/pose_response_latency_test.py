import math

from autotrainer.inference.pose_algorithm import PoseResponse


def test_a_response_keeps_its_live_stamps_through_round():
    response = PoseResponse(sequence=3, live_recv_perf_c=1.5, live_put_perf_c=1.75)
    rounded = response.round()
    assert rounded.live_recv_perf_c == 1.5
    assert rounded.live_put_perf_c == 1.75


def test_stamps_default_to_nan_off_the_live_path():
    response = PoseResponse()
    assert math.isnan(response.live_recv_perf_c)
    assert math.isnan(response.live_put_perf_c)

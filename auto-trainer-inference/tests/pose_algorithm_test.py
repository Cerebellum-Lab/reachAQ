import typing
from typing import Dict, List

import numpy

from autotrainer.core import Offset3DTuple
from autotrainer.inference import PoseAlgorithm
from autotrainer.inference import PoseResponse, PoseLocation
from autotrainer.core.pose_elements import SceneElement


def verify_common_output(response: PoseResponse, sequence: int, parts: typing.List[str]):
    assert response.sequence == sequence

    num_parts = len(parts)

    assert response.parts_flags and len(response.parts_flags) == 3

    assert len(response.parts_flags[0]) == num_parts
    assert len(response.parts_flags[1]) == num_parts
    assert len(response.parts_flags[2]) == num_parts

    assert response.locations and len(response.locations) == 2

    assert isinstance(response.locations[0], dict)
    assert isinstance(response.locations[1], dict)


def verify_all_false(flags: Dict[str, bool], except_part: str = ""):
    for name, val in flags.items():
        if name == except_part:
            assert val is True
        else:
            assert val is False


def verify_all_empty(
        algo: PoseAlgorithm,
        locations_list: List[Dict[str, PoseLocation]], except_camera: int = -1,
        except_part: str = ""):
    for cdx, locations in enumerate(locations_list):
        for idx, part in enumerate(algo.part_names):
            location = locations.get(part)
            if cdx == except_camera and part == except_part:
                assert location is not None
                assert location.index == algo.get_part_index(part)
                assert location.x != -1
                assert location.y != -1
            else:
                assert location is None


def test_algorithm_output():
    data = list()

    # Ensure all confidence values will be below the default thresholds (/2).
    for idx in range(6):
        data.append(numpy.random.rand(10, 3) / 2)

    parts = ["Part00", "Part01", "Part02", "Part03", "Part04", "Part05", "Part06", "Part07", "Part08", "Part09"]

    algorithm = PoseAlgorithm()

    algorithm.initialize(parts)

    response = algorithm.process(data)

    verify_common_output(response, 1, parts)

    verify_all_false(response.parts_flags[0])
    verify_all_false(response.parts_flags[1])
    verify_all_false(response.parts_flags[2])

    verify_all_empty(algorithm, response.locations)

    # Set the confidence for a part in a middle frame above the plot and seen thresholds.
    data[3][5][2] = 0.95

    response = algorithm.process(data)

    verify_common_output(response, 2, parts)

    verify_all_false(response.parts_flags[0])
    verify_all_false(response.parts_flags[1], "Part05")
    verify_all_false(response.parts_flags[2])

    # Interleaved frame 3 changed above is for the right/second camera.
    verify_all_empty(algorithm, response.locations, 1, "Part05")

    # Trigger dual part flag.
    data[2][5][2] = 0.95

    response = algorithm.process(data)

    verify_all_false(response.parts_flags[0], "Part05")
    verify_all_false(response.parts_flags[1], "Part05")
    verify_all_false(response.parts_flags[2], "Part05")

    # Seen in both, but not in paired frames - should not trigger dual
    data[2][5][2] = 0.00
    data[4][5][2] = 0.95

    response = algorithm.process(data)

    verify_all_false(response.parts_flags[0], "Part05")
    verify_all_false(response.parts_flags[1], "Part05")
    verify_all_false(response.parts_flags[2])


def test_algorithm_output_three_cameras():
    parts = ["Part00", "Part01"]
    algorithm = PoseAlgorithm()
    algorithm.initialize(parts)
    data = [
        numpy.zeros((len(parts), 3))
        for _ in range(6)
    ]

    # Second frame for each of three interleaved cameras sees Part00.
    for frame_index in (3, 4, 5):
        data[frame_index][0][2] = 0.95
    # Only the third camera sees Part01.
    data[5][1][2] = 0.95

    response = algorithm.process(data, camera_count=3)

    assert len(response.parts_flags) == 4
    assert len(response.locations) == 3
    verify_all_false(response.parts_flags[0], "Part00")
    verify_all_false(response.parts_flags[1], "Part00")
    assert response.parts_flags[2]["Part00"] is True
    assert response.parts_flags[2]["Part01"] is True
    verify_all_false(response.parts_flags[-1], "Part00")
    assert response.parts_flags[-1]["Part01"] is False
    assert all("Part00" in locations for locations in response.locations)
    assert "Part01" not in response.locations[0]
    assert "Part01" not in response.locations[1]
    assert "Part01" in response.locations[2]

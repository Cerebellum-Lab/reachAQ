"""src/reachaq/calibration/make_charuco_board: the printable ChArUco board."""

import re

import cv2
import numpy as np
import pytest
from PIL import Image

from reachaq.inference.multires_calibration.fit import RIG_CHARUCO_BOARD, BoardSpec
from reachaq.calibration import make_charuco_board as maker


def test_it_writes_the_2400_dpi_pattern_and_the_letter_pdf(tmp_path):
    png, pdf = maker.make_board_files(tmp_path)

    with Image.open(png) as image:
        # 36 x 28 mm at 2400 dpi.
        assert image.size == (round(36 / 25.4 * 2400), round(28 / 25.4 * 2400)) == (3402, 2646)
        assert tuple(round(v) for v in image.info["dpi"]) == (2400, 2400)
        pixels = np.asarray(image.convert("L"))
    # Exactly the board the rig's was printed from, as cv2 draws it.
    generator = cv2.aruco.CharucoBoard((9, 7), 4.0, 3.0, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    expected = generator.generateImage((3402, 2646), marginSize=0, borderBits=1)
    assert pixels.tobytes() == expected.tobytes()

    data = pdf.read_bytes()
    assert data.startswith(b"%PDF")
    # One US-letter page: 612 x 792 pt.
    box = [float(v) for v in re.search(rb"/MediaBox\s*\[\s*([\d.\s]+)\]", data).group(1).split()]
    assert box[2:] == [612.0, 792.0]
    assert b"/DCTDecode" not in data  # no JPEG: the markers must print sharp


def test_the_page_holds_24_readable_boards():
    page = np.asarray(maker.page(RIG_CHARUCO_BOARD, dpi=300).convert("L"))
    assert page.shape == (3300, 2550)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    _, ids, _ = detector.detectMarkers(page)
    # 31 markers per board, 24 boards.
    assert ids is not None and len(ids) == 24 * 31


def test_only_a_charuco_board_is_printed(tmp_path):
    with pytest.raises(ValueError, match="ChArUco"):
        maker.make_board_files(tmp_path / "out", BoardSpec())
    assert not (tmp_path / "out").exists()

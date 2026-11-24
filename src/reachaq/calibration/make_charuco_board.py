"""Make the printable ChArUco calibration board.

The board is the calibration's own (multires_calibration.scout.configured_board): the rig's
cv2.aruco.CharucoBoard((9, 7), 4.0, 3.0, DICT_4X4_50) unless the library's settings.yaml
names another ChArUco board. Two files are written:

- charuco_<spec>_2400dpi.png: one board at 2400 dpi (36 x 28 mm for the rig's board), the
  cv2 pattern exactly as generateImage(marginSize=0, borderBits=1) draws it;
- charuco_<spec>_letter.pdf: a US-letter page at 1200 dpi with 24 boards in a 4 x 6 grid,
  8 mm apart with faint cut lines midway, a header line with the spec, and a 50 mm ruler
  with 1 mm ticks to check the print scale with calipers.

Print the PDF at 100 % ("Actual size"), never "fit to page", on matte paper, and measure
the ruler before cutting: it must read 50.0 mm. See docs/acquisition/stereo-calibration.md.

    python -m reachaq.calibration.make_charuco_board OUT_DIR [--library CALIBRATIONS_DIR]

Needs only cv2, numpy and PIL.
"""

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from reachaq.inference.multires_calibration.fit import CHARUCO, BoardSpec
from reachaq.inference.multires_calibration.library import read_settings
from reachaq.inference.multires_calibration.scout import configured_board

PATTERN_DPI = 2400
PAGE_DPI = 1200
PAGE_IN = (8.5, 11.0)  # US letter
GRID = (4, 6)  # boards across, down
GAP_MM = 8.0
RULER_MM = 50
_MM_PER_IN = 25.4
# The cut lines are faint: thin and dashed, 1 mm on, 2 mm off. The page is bilevel, so the
# PDF holds it losslessly rather than as a JPEG that would blur the markers.
_CUT_DASH_MM = (1.0, 2.0)
_PAGE_MARGIN_MM = 12.0


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / _MM_PER_IN))


def board_size_mm(board: BoardSpec) -> Tuple[float, float]:
    """The whole board's (width, height) in mm: (cols + 1) x (rows + 1) squares."""
    return (board.cols + 1) * board.square_mm, (board.rows + 1) * board.square_mm


def pattern(board: BoardSpec, dpi: int) -> np.ndarray:
    """The board as uint8 grey at `dpi`, as cv2 draws it (marginSize 0, borderBits 1)."""
    if board.kind != CHARUCO:
        raise ValueError(f"only a ChArUco board is printed here, not a {board.kind}")
    width_mm, height_mm = board_size_mm(board)
    size = (mm_to_px(width_mm, dpi), mm_to_px(height_mm, dpi))
    return board.charuco_board().generateImage(size, marginSize=0, borderBits=1)


def spec_text(board: BoardSpec) -> str:
    width_mm, height_mm = board_size_mm(board)
    return (f"ChArUco {board.cols + 1} x {board.rows + 1} squares, {board.square_mm:g} mm squares,"
            f" {board.marker_mm:g} mm markers, {board.dictionary}, legacy pattern off\n"
            f"{width_mm:g} x {height_mm:g} mm, {board.cols} x {board.rows} inner corners."
            f" Print at 100 % / Actual size, no fit to page; the ruler must measure {RULER_MM} mm.")


def page(board: BoardSpec, dpi: int = PAGE_DPI) -> Image.Image:
    """The US-letter page as a bilevel PIL image ("1") at `dpi`."""
    W, H = int(round(PAGE_IN[0] * dpi)), int(round(PAGE_IN[1] * dpi))
    image = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(image)
    px = lambda mm: mm_to_px(mm, dpi)  # noqa: E731

    tile = Image.fromarray(pattern(board, dpi))
    width_mm, height_mm = board_size_mm(board)
    cols, rows = GRID
    grid_w = cols * width_mm + (cols - 1) * GAP_MM
    left_mm = (PAGE_IN[0] * _MM_PER_IN - grid_w) / 2
    header_mm = _PAGE_MARGIN_MM
    top_mm = header_mm + 14.0

    font = _font(px(3.0))
    draw.multiline_text((px(left_mm), px(header_mm)), spec_text(board), fill=0, font=font, spacing=px(1.5))

    for r in range(rows):
        for c in range(cols):
            x_mm = left_mm + c * (width_mm + GAP_MM)
            y_mm = top_mm + r * (height_mm + GAP_MM)
            image.paste(tile, (px(x_mm), px(y_mm)))
    grid_h = rows * height_mm + (rows - 1) * GAP_MM
    line = max(1, px(0.1))
    y0, y1 = top_mm - GAP_MM / 2, top_mm + grid_h + GAP_MM / 2
    x0, x1 = left_mm - GAP_MM / 2, left_mm + grid_w + GAP_MM / 2
    for c in range(1, cols):  # cut lines midway between the boards
        x = left_mm + c * (width_mm + GAP_MM) - GAP_MM / 2
        _dashed(draw, px, (x, y0), (x, y1), line)
    for r in range(1, rows):
        y = top_mm + r * (height_mm + GAP_MM) - GAP_MM / 2
        _dashed(draw, px, (x0, y), (x1, y), line)

    _ruler(draw, px, left_mm, top_mm + grid_h + 12.0, font)
    return image.point(lambda v: 255 if v >= 128 else 0).convert("1", dither=Image.Dither.NONE)


def _dashed(draw: ImageDraw.ImageDraw, px, start_mm, end_mm, width: int) -> None:
    """A horizontal or vertical dashed line from start_mm to end_mm, in page mm."""
    on, off = _CUT_DASH_MM
    (xa, ya), (xb, yb) = start_mm, end_mm
    length = math.hypot(xb - xa, yb - ya)
    t = 0.0
    while t < length:
        e = min(t + on, length)
        draw.line([(px(xa + (xb - xa) * t / length), px(ya + (yb - ya) * t / length)),
                   (px(xa + (xb - xa) * e / length), px(ya + (yb - ya) * e / length))], fill=0, width=width)
        t += on + off


def _ruler(draw: ImageDraw.ImageDraw, px, x_mm: float, y_mm: float, font) -> None:
    """A RULER_MM ruler with 1 mm ticks, longer every 5 mm and labelled every 10 mm, its
    baseline at y_mm; every tick is centred on its exact mm position."""
    width = max(1, px(0.15))
    draw.line([(px(x_mm), px(y_mm)), (px(x_mm + RULER_MM), px(y_mm))], fill=0, width=width)
    for k in range(RULER_MM + 1):
        length = 4.0 if k % 10 == 0 else 2.5 if k % 5 == 0 else 1.5
        x = px(x_mm + k)
        draw.line([(x, px(y_mm)), (x, px(y_mm - length))], fill=0, width=width)
        if k % 10 == 0:
            draw.text((x, px(y_mm + 1.0)), str(k), fill=0, font=font, anchor="mt")
    draw.text((px(x_mm + RULER_MM + 3.0), px(y_mm - 2.0)), f"{RULER_MM} mm", fill=0, font=font, anchor="lm")


def _font(size_px: int):
    try:
        return ImageFont.load_default(size=size_px)
    except TypeError:  # Pillow < 10.1 has one bitmap size
        return ImageFont.load_default()


def file_stem(board: BoardSpec) -> str:
    return f"charuco_{board.cols + 1}x{board.rows + 1}_{board.square_mm:g}mm_{board.marker_mm:g}mm_{board.dictionary}"


def make_board_files(out_dir: Path, board: Optional[BoardSpec] = None) -> Tuple[Path, Path]:
    """Write the 2400 dpi pattern PNG and the US-letter PDF to `out_dir`; return (png, pdf).
    `board` defaults to configured_board({}), the rig's board."""
    board = configured_board({}) if board is None else board
    if board.kind != CHARUCO:
        raise ValueError(f"only a ChArUco board is printed here, not a {board.kind}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = file_stem(board)
    png = out_dir / f"{stem}_{PATTERN_DPI}dpi.png"
    Image.fromarray(pattern(board, PATTERN_DPI)).save(png, dpi=(PATTERN_DPI, PATTERN_DPI))
    pdf = out_dir / f"{stem}_letter.pdf"
    page(board).save(pdf, "PDF", resolution=float(PAGE_DPI))
    return png, pdf


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--library", type=Path, default=None,
                        help="the calibration library folder whose settings.yaml names the board")
    args = parser.parse_args(argv)
    board = configured_board({} if args.library is None else read_settings(args.library))
    png, pdf = make_board_files(args.out_dir, board)
    print(f"{png}\n{pdf}")


if __name__ == "__main__":
    main()

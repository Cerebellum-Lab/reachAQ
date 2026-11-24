"""A synthetic camera rig for the multi-resolution calibration tests.

Up to six cameras (left and right by default) look at a chessboard that the stage slides
through paused waypoints, as the rig's 3D-calibration move list does. Everything about it is known, so a fit
can be checked against the truth, and render_board_frame draws the board as a
camera sees it for the detection and end-to-end tests.

Conventions are fit.py's: X_cam = R(rvec) @ X_motor + tvec in mm, and the motor
frame's origin is the board's corner 0 at motor position 0 (board_tvec = 0). The
board corners of waypoint w are therefore

    X = R(board_rvec) @ board.object_points() + motor_scale * motor_mm_w

which SyntheticRig.world_corners returns.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from reachaq.inference.multires_calibration.fit import (
    INTRINSIC_NAMES,
    BoardSpec,
    CameraModel,
    MotorFrameFit,
    WaypointObservation,
)

# The move list is fixed, like the rig's, so every synthetic rig has the same waypoints;
# `seed` only draws the corner noise.
_MOVE_LIST_SEED = 2026
_START_MM = (-9.4, 10.0, -35.0)
_RANGE_MM = ((-10.0, 10.0), (10.0, 35.0), (-35.0, -22.0))
_N_WAYPOINTS = 37  # the start plus 36 moves, as on the rig
_MIN_STEP_MM = 2.0

_BOARD_ROTATION_DEG = 20.0  # about the scene's z axis, which is motor z by default
_CAMERA_DISTANCE_MM = 150.0
_HALF_SEPARATION_DEG = 20.0  # left and right are 40 degrees apart
_IMAGE_MARGIN_PX = 20.0

# The camera slots, up to 6. Each is (azimuth, intrinsics): the camera stands
# _HALF_SEPARATION_DEG off the board normal, turned by the azimuth about it from the scene's
# +x, and its intrinsics are (fx, fy/fx, cx offset, cy offset from the image centre, dist).
# The cameras differ slightly so a fit that swaps or shares them fails. "left" always takes
# slot 0 and "right" slot 1, so the pair is the same whatever else is in the rig; any other
# name takes the next free slot in order.
_SLOTS = (
    (180.0, (2300.0, 1.002, 3.0, -2.5, (-0.25, 0.1, 0.0005, -0.0003, 0.0))),  # left: the scene's -x side
    (0.0, (2310.0, 1.002, -3.5, 4.0, (-0.24, 0.09, -0.0004, 0.0002, 0.0))),  # right: the +x side
    (90.0, (2290.0, 0.999, 2.0, 3.0, (-0.23, 0.08, 0.0003, 0.0001, 0.0))),
    (270.0, (2320.0, 1.001, -2.0, -3.5, (-0.26, 0.11, -0.0002, 0.0004, 0.0))),
    (45.0, (2305.0, 1.003, 4.0, 1.5, (-0.245, 0.095, 0.0001, -0.0001, 0.0))),
    (225.0, (2295.0, 0.998, -1.5, -1.0, (-0.255, 0.105, -0.0003, -0.0002, 0.0))),
)
_FIXED_SLOTS = {"left": 0, "right": 1}

_TEXTURE_PX_PER_MM = 50
_SUPERSAMPLE = 3
_BLACK, _WHITE, _BACKGROUND = 20, 235, 100


@dataclass
class SyntheticRig:
    truth: MotorFrameFit
    observations: List[WaypointObservation]
    board: BoardSpec

    def world_corners(self, motor_mm) -> np.ndarray:
        """(rows*cols, 3) motor-frame board corners at the commanded position `motor_mm`."""
        return _world_corners(self.board, self.truth.board_rvec, self.truth.motor_scale, motor_mm)


def synthetic_rig(
    *,
    cameras: Sequence[str] = ("left", "right"),
    seen: Optional[Mapping[str, Sequence[int]]] = None,
    motor_scale: float = 1.0,
    noise_px: float = 0.1,
    seed: int = 0,
    image_size=(1440, 1080),
    frame_rotation_deg: float = 0.0,
    frame_rotation_axis: Tuple[float, float, float] = (0.3, -0.5, 0.81),
) -> SyntheticRig:
    """`cameras` (up to 6, left and right by default), 37 waypoints, and Gaussian corner
    noise of `noise_px` per coordinate.

    By default the scene is aligned with the motor axes: the board faces motor z and the
    left and right cameras stand apart along motor x; the others stand around the board
    normal at the same angle from it (_SLOTS). `frame_rotation_deg` turns the whole scene,
    board and cameras together, by that angle about `frame_rotation_axis` (motor
    coordinates, normalised here). Seen from the scene, the motor axes are then oblique, so
    a frame transform that is transposed or only flips signs no longer looks right. The
    move list does not change.

    `seen` maps a camera to the waypoints it sees, as indices into the move list; a camera
    it does not name sees every waypoint. The observations hold a camera only where it sees
    the board. The default rig draws the same noise as it always has.

    Raises ValueError if any corner of a waypoint a camera sees falls behind it or within
    _IMAGE_MARGIN_PX of the image edge, so every observation is a full board.
    """
    cameras = tuple(cameras)
    if not 1 <= len(cameras) <= len(_SLOTS) or len(set(cameras)) != len(cameras):
        raise ValueError(f"synthetic rig: 1 to {len(_SLOTS)} distinct cameras, got {cameras}")
    seen = {} if seen is None else {cam: set(int(w) for w in waypoints) for cam, waypoints in seen.items()}
    board = BoardSpec()
    width, height = int(image_size[0]), int(image_size[1])
    motor = _move_list()
    axis = np.asarray(frame_rotation_axis, float)
    # The scene's axes as columns, in motor coordinates. At 0 degrees this is exactly the
    # identity, so the default rig is unchanged.
    scene = cv2.Rodrigues(axis / np.linalg.norm(axis) * np.radians(frame_rotation_deg))[0]
    board_rvec = cv2.Rodrigues(scene @ _rotation_z(np.radians(_BOARD_ROTATION_DEG)))[0].ravel()

    # Aim both cameras at the middle of the region the corners sweep, placed from the
    # commanded positions so the cameras do not move with motor_scale.
    centre = np.mean([_world_corners(board, board_rvec, 1.0, m) for m in motor], axis=(0, 1))
    free = iter(i for i in range(len(_SLOTS)) if i not in {_FIXED_SLOTS[c] for c in cameras if c in _FIXED_SLOTS})
    slots = {cam: _FIXED_SLOTS[cam] if cam in _FIXED_SLOTS else next(free) for cam in cameras}
    models = {}
    for cam in cameras:
        azimuth, (fx, aspect, dcx, dcy, dist) = _SLOTS[slots[cam]]
        K = np.array([
            [fx, 0.0, (width - 1) / 2 + dcx],
            [0.0, fx * aspect, (height - 1) / 2 + dcy],
            [0.0, 0.0, 1.0],
        ])
        a, phi = np.radians(_HALF_SEPARATION_DEG), np.radians(azimuth)
        # Every camera stands on the scene's -z side of the board, 20 degrees from its
        # normal (left and right either side of it along the scene's x), with image y along
        # the scene's y. At azimuth 180 and 0, cos and sin are exactly -1, 0 and 1, 0.
        direction = np.array([np.round(np.cos(phi), 15) * np.sin(a), np.round(np.sin(phi), 15) * np.sin(a), -np.cos(a)])
        eye = centre + _CAMERA_DISTANCE_MM * scene @ direction
        rvec, tvec = _look_at(eye, centre, down=scene @ np.array([0.0, 1.0, 0.0]))
        models[cam] = CameraModel(K=K, dist=np.array(dist, float), rvec=rvec, tvec=tvec)

    rng = np.random.default_rng(seed)
    observations, sq_err = [], {cam: {} for cam in models}
    for w, m in enumerate(motor):
        X = _world_corners(board, board_rvec, motor_scale, m)
        corners = {}
        for cam, model in models.items():
            if cam in seen and w not in seen[cam]:
                continue
            depth = (X @ cv2.Rodrigues(model.rvec)[0].T + model.tvec)[:, 2]
            ideal = cv2.projectPoints(X, model.rvec, model.tvec, model.K, model.dist)[0].reshape(-1, 2)
            if depth.min() <= 0 or not _inside(ideal, width, height):
                raise ValueError(f"synthetic rig: waypoint {w} is not fully in view of the {cam} camera")
            noise = rng.normal(0.0, noise_px, ideal.shape)
            corners[cam] = ideal + noise
            sq_err[cam][w] = (noise ** 2).sum(axis=1)
        observations.append(WaypointObservation(index=w, motor_mm=m.copy(), corners=corners))

    truth = MotorFrameFit(
        cameras=models,
        board_rvec=board_rvec,
        board_tvec=np.zeros(3),
        motor_scale=float(motor_scale),
        image_size=(width, height),
        # At the true parameters the residuals are exactly the injected noise.
        rms_px={cam: float(np.sqrt(np.mean(np.concatenate(list(e.values()))))) for cam, e in sq_err.items()},
        per_view_rms_px={cam: {w: float(np.sqrt(v.mean())) for w, v in e.items()} for cam, e in sq_err.items()},
        param_std={cam: {name: 0.0 for name in INTRINSIC_NAMES} for cam in models},
    )
    return SyntheticRig(truth=truth, observations=observations, board=board)


def render_board_frame(
    model: CameraModel, X_board_world: np.ndarray, image_size, board: BoardSpec = BoardSpec()
) -> np.ndarray:
    """A uint8 grey (height, width) image of the board as `model` sees it, without distortion.

    `X_board_world` is the board's (rows*cols, 3) inner corners in the motor frame, in
    object_points order (SyntheticRig.world_corners gives them). The image is the full
    chessboard: (cols+1) x (rows+1) squares, reaching one square beyond the inner
    corners on every side, inside a one-square white margin, on a grey background.

    The board plane maps to the image by the homography fitted from
    board.object_points()[:, :2] to cv2.projectPoints(X_board_world, model.rvec,
    model.tvec, model.K, zeros(5)). Those projections are therefore exactly where the
    inner corners are drawn, which is the truth a detector should be compared with.
    model.dist is ignored.
    """
    width, height = int(image_size[0]), int(image_size[1])
    obj = board.object_points()
    X = np.asarray(X_board_world, float)
    if X.shape != obj.shape:
        raise ValueError(f"expected the {obj.shape} inner corners in the world frame, got shape {X.shape}")
    image_points = cv2.projectPoints(X, model.rvec, model.tvec, model.K, np.zeros(5))[0].reshape(-1, 2)
    H_mm, _ = cv2.findHomography(obj[:, :2], image_points, 0)

    texture, texture_to_mm = _board_texture(board)
    # Draw at _SUPERSAMPLE times the resolution and area-average down, so square edges are
    # anti-aliased. Final pixel i is centred on i and covers super pixels s*i .. s*i + s-1,
    # so a final coordinate x sits at s*x + (s-1)/2 in the supersampled image.
    s = _SUPERSAMPLE
    to_super = np.array([[s, 0.0, (s - 1) / 2], [0.0, s, (s - 1) / 2], [0.0, 0.0, 1.0]])
    big = cv2.warpPerspective(
        texture,
        to_super @ H_mm @ texture_to_mm,
        (width * s, height * s),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_BACKGROUND,
    )
    return cv2.resize(big, (width, height), interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=4)
def _board_texture(board: BoardSpec) -> Tuple[np.ndarray, np.ndarray]:
    """The chessboard texture and the 3x3 map from texture pixels to board-plane mm."""
    sq = board.square_mm
    ppm = _TEXTURE_PX_PER_MM
    # The squares span -sq .. cols*sq in x and -sq .. rows*sq in y; the margin adds one more square.
    x0 = y0 = -2.0 * sq
    nx = int(round((board.cols + 3) * sq * ppm))
    ny = int(round((board.rows + 3) * sq * ppm))
    # Texel i is centred half a texel in from its left edge, so every square edge falls
    # exactly between two texels and no texel straddles one.
    x = x0 + (np.arange(nx) + 0.5) / ppm
    y = y0 + (np.arange(ny) + 0.5) / ppm
    a, b = np.floor(x / sq)[None, :], np.floor(y / sq)[:, None]
    on_board = (a >= -1) & (a <= board.cols - 1) & (b >= -1) & (b <= board.rows - 1)
    # The corner squares are black; with an odd number of squares each way the board then
    # looks the same turned 180 degrees, like the real one.
    black = on_board & ((a + b) % 2 == 0)
    texture = np.where(black, _BLACK, _WHITE).astype(np.uint8)
    texture_to_mm = np.array([[1.0 / ppm, 0.0, x0 + 0.5 / ppm], [0.0, 1.0 / ppm, y0 + 0.5 / ppm], [0.0, 0.0, 1.0]])
    return texture, texture_to_mm


def _move_list() -> np.ndarray:
    """(37, 3) commanded positions: the rig's start, then single-axis moves within the rig's ranges."""
    rng = np.random.default_rng(_MOVE_LIST_SEED)
    position = np.array(_START_MM, float)
    out = [position.copy()]
    while len(out) < _N_WAYPOINTS:
        axis = int(rng.integers(3))
        lo, hi = _RANGE_MM[axis]
        value = round(float(rng.uniform(lo, hi)), 1)
        # A move must go somewhere: a repeated position would be a no-op waypoint.
        if abs(value - position[axis]) < _MIN_STEP_MM:
            continue
        position = position.copy()
        position[axis] = value
        out.append(position)
    return np.array(out)


def _world_corners(board: BoardSpec, board_rvec, motor_scale: float, motor_mm) -> np.ndarray:
    R = cv2.Rodrigues(np.asarray(board_rvec, float))[0]
    return board.object_points() @ R.T + motor_scale * np.asarray(motor_mm, float).reshape(1, 3)


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _look_at(eye: np.ndarray, target: np.ndarray, down: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(rvec, tvec) of a camera at `eye` looking at `target`, image y along `down` as far as possible."""
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(down, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # rows are the camera axes in the motor frame
    return cv2.Rodrigues(R)[0].ravel(), -R @ eye


def _inside(points: np.ndarray, width: int, height: int) -> bool:
    m = _IMAGE_MARGIN_PX
    return bool(
        np.all(points[:, 0] >= m) and np.all(points[:, 0] <= width - 1 - m)
        and np.all(points[:, 1] >= m) and np.all(points[:, 1] <= height - 1 - m)
    )

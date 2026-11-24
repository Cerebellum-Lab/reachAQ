"""Check that the calibration's anipose TOMLs load and project in aniposelib 0.8.0.

Not a pytest module: aniposelib needs opencv-contrib-python, which replaces the reachaq
environment's cv2, so the check runs in a virtualenv of its own and reachaq never imports
aniposelib. It has two steps, each in its own environment:

1. In the reachaq environment, write a calibration of the synthetic 3-camera rig (left,
   right and camera4, each at its own live geometry, plus an uncalibrated stimCam) and, for
   each of its TOMLs, a JSON of the motor-frame points and the pixels the fit puts them at
   in that file's geometry:

       PYTHONPATH=src python multires_anipose_compat_check.py --write-inputs OUT

2. In the check virtualenv (aniposelib==0.8.0 and toml, the reachaq
   "anipose-check" extra), for each pair:

       python multires_anipose_compat_check.py OUT/<name>.toml OUT/<name>.json

   It loads the TOML with aniposelib.cameras.CameraGroup.load, projects the points with
   CameraGroup.project, and exits 1 when a camera is missing, the metadata lost a key, or
   a projection is more than TOLERANCE_PX from the expected pixel.
"""

import argparse
import json
import sys
from pathlib import Path

TOLERANCE_PX = 1e-6
METADATA_KEYS = ("adjusted", "error", "calibration_id", "cameras", "units", "frame")


def write_inputs(out: Path) -> None:
    """Step 1, in the reachaq environment."""
    import cv2
    import numpy as np

    sys.path.insert(0, str(Path(__file__).parent))
    from multires_synthetic import synthetic_rig

    from reachaq.inference.multires_calibration.fit import CaptureCamera
    from reachaq.inference.multires_calibration.geometry import (
        PresetGeometry,
        presets_for_cameras,
        sensor_to_preset_points,
    )
    from reachaq.inference.multires_calibration.report import CameraVerdict, CaptureSummary, build_report
    from reachaq.inference.multires_calibration.store import load_index, write_calibration

    rig = synthetic_rig(noise_px=0.1, frame_rotation_deg=35.0, cameras=("left", "right", "camera4"))
    cameras = [
        CaptureCamera("left", "24152533", 0),
        CaptureCamera("right", None, 1),
        CaptureCamera("stimCam", "20142630", 2),
        CaptureCamera("camera4", "24152535", 3),
    ]
    live = {
        "left": PresetGeometry(4, 256, 256, 52, 6),
        "right": PresetGeometry(4, 256, 256, 52, 6),
        "camera4": PresetGeometry(2, 300, 200, 10, 40),
    }
    n = len(rig.observations)
    names = [c.name for c in cameras]
    summary = CaptureSummary(n, n, 0, {cam: n for cam in names}, {cam: {} for cam in names})
    verdicts = {cam: CameraVerdict(cam, cam != "stimCam", "too few board views: 4 of 12 needed" if cam == "stimCam" else "")
                for cam in names}
    report = build_report(rig.truth, rig.observations, rig.board, live["left"].sensor_roi(), summary, [], None,
                          cameras=verdicts)
    out.mkdir(parents=True, exist_ok=True)
    entry = out / "entry"
    write_calibration(entry, fit=rig.truth, preset_sets=presets_for_cameras(live), board=rig.board, report=report,
                      capture={}, cameras=cameras)
    index = load_index(entry)

    points = np.concatenate([rig.world_corners(o.motor_mm) for o in rig.observations])
    width, height = rig.truth.image_size
    files = {"sensor": (entry / "calibration.toml", {cam: PresetGeometry(1, width, height, 0, 0) for cam in live})}
    for s in index.preset_sets:
        geometries = {cam: PresetGeometry(**g) for cam, g in s["cameras"].items()}
        files[s["name"]] = (entry / "presets" / s["name"] / "calibration.toml", geometries)
    for name, (toml_path, geometries) in files.items():
        projections = {}
        for cam, g in geometries.items():
            m = rig.truth.cameras[cam]
            sensor = cv2.projectPoints(points, m.rvec, m.tvec, m.K, m.dist)[0].reshape(-1, 2)
            projections[cam] = sensor_to_preset_points(sensor, g).tolist()
        (out / f"{name}.toml").write_text(toml_path.read_text(encoding="utf-8"), encoding="utf-8")
        expected = {"calibration_id": index.id, "points": points.tolist(), "projections": projections}
        (out / f"{name}.json").write_text(json.dumps(expected), encoding="utf-8")
        print(f"wrote {out / name}.toml and .json: {', '.join(geometries)}")


def check(toml_path: Path, expected_path: Path) -> int:
    """Step 2, in the check virtualenv. Returns the exit status."""
    import numpy as np
    from aniposelib.cameras import CameraGroup

    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    cgroup = CameraGroup.load(str(toml_path))
    names = cgroup.get_names()
    failures = []
    if sorted(names) != sorted(expected["projections"]):
        failures.append(f"cameras {names}, expected {sorted(expected['projections'])}")
    metadata = getattr(cgroup, "metadata", {}) or {}
    missing = [key for key in METADATA_KEYS if key not in metadata]
    if missing:
        failures.append(f"metadata lost {missing}")
    elif metadata["calibration_id"] != expected["calibration_id"]:
        failures.append(f"metadata calibration_id {metadata['calibration_id']}, expected {expected['calibration_id']}")
    points = np.asarray(expected["points"], float)
    projected = np.asarray(cgroup.project(points), float).reshape(len(names), len(points), 2)
    for cam, pixels in zip(names, projected):
        if cam not in expected["projections"]:
            continue
        error = float(np.abs(pixels - np.asarray(expected["projections"][cam], float)).max())
        print(f"{toml_path.name}: {cam} max |projection - expected| = {error:.3g} px over {len(points)} points")
        if not error <= TOLERANCE_PX:
            failures.append(f"{cam}: {error:.3g} px")
    for failure in failures:
        print(f"FAIL {toml_path}: {failure}")
    if not failures:
        print(f"OK {toml_path}: {len(names)} cameras, metadata {sorted(metadata)}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-inputs", type=Path, metavar="OUT", help="step 1: write the TOMLs and JSONs to OUT")
    parser.add_argument("toml", nargs="?", type=Path)
    parser.add_argument("expected", nargs="?", type=Path)
    args = parser.parse_args()
    if args.write_inputs is not None:
        write_inputs(args.write_inputs)
        return 0
    if args.toml is None or args.expected is None:
        parser.error("give a TOML and its expected-projections JSON, or --write-inputs OUT")
    return check(args.toml, args.expected)


if __name__ == "__main__":
    sys.exit(main())

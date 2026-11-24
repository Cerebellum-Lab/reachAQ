"""Score a pose model on the held-out labelled frames of a dataset.

The held-out set is a few dozen frames, and models report different shares of
keypoints: one that stays silent on hard parts scores a lower raw median than
one that guesses at them. So error is compared at matched coverage - every
labelled keypoint sorted by the model's own confidence, the median error of
the most confident 25, 50 and 100% - with the detection rate at the live gate
beside it. Raw RMSE is not reported at all.

Any installed backend can be scored, not just a keypoint model: the model is
built through the plugin that claims it, so a candidate is compared against
whatever the rig runs now on the same frames.
"""

import math
import os
import time
import typing

import numpy

from ..pose_model import PoseModel

COVERAGES = (25, 50, 100)
# A part with fewer held-out labels than this gets its numbers flagged.
INDICATIVE_BELOW = 5
# Two cameras at 150 Hz: the time one pose batch may take.
FRAME_BUDGET_MS = 1000 / 150


def score(predicted: numpy.ndarray, labels: numpy.ndarray,
          bodyparts: typing.Sequence[str], gate: float) -> dict:
    """
    predicted (N, K, 3) x, y, confidence; labels (N, K, 2), NaN = not labelled.
    """
    labelled = numpy.isfinite(labels).all(axis=2)
    error = numpy.linalg.norm(predicted[..., :2] - labels, axis=2)
    confidence = predicted[..., 2]
    # A part the model does not have is never detected and never close.
    error = numpy.where(numpy.isfinite(error), error, numpy.inf)

    def coverage_medians(err, conf):
        order = numpy.argsort(-conf, kind="stable")
        result = {}
        for coverage in COVERAGES:
            count = math.ceil(len(err) * coverage / 100)
            result[coverage] = float(numpy.median(err[order[:count]])) if count else math.nan
        return result

    flat_error, flat_conf = error[labelled], confidence[labelled]
    parts = {}
    for index, part in enumerate(bodyparts):
        mask = labelled[:, index]
        n = int(mask.sum())
        err, conf = error[mask, index], confidence[mask, index]
        parts[part] = {
            "n": n,
            "detected": float((conf >= gate).mean()) if n else math.nan,
            "median_px": float(numpy.median(err)) if n else math.nan,
            "indicative": 0 < n < INDICATIVE_BELOW,
        }
    n = int(labelled.sum())
    return {
        "n": n,
        "frames": int(labelled.any(axis=1).sum()),
        "gate": float(gate),
        "coverage_median": coverage_medians(flat_error, flat_conf),
        "detected": float((flat_conf >= gate).mean()) if n else math.nan,
        "parts": parts,
    }


def match_parts(predicted: numpy.ndarray, model_parts: typing.Sequence[str],
                bodyparts: typing.Sequence[str]
                ) -> typing.Tuple[numpy.ndarray, typing.List[str]]:
    """Reorder a model's (N, M, 3) output to the dataset's parts by name.

    A dataset part the model does not have comes back at NaN position and zero
    confidence, and is named in the second return value.
    """
    model_parts = list(model_parts)
    out = numpy.zeros((len(predicted), len(bodyparts), 3), "float32")
    out[..., :2] = numpy.nan
    missing = []
    for index, part in enumerate(bodyparts):
        if part in model_parts:
            out[:, index] = predicted[:, model_parts.index(part)]
        else:
            missing.append(part)
    return out, missing


# -- models -----------------------------------------------------------------

def open_model(model_path: str, batch_size: int = 1) -> typing.Tuple[PoseModel, float]:
    """Build and load a model the way live inference would; (model, gate)."""
    from .. import pose_plugins
    from ..backend_selection import build_pose_model, confidence_threshold
    plugin = next((p for p in pose_plugins.discover_plugins() if p.claims(model_path)), None)
    if plugin is not None:
        runtime = plugin.runtime(model_path, os.environ)
        plugin.pre_validate(model_path, runtime)
        model = plugin.build(model_path, runtime, shuffle_index=1, training_index=0,
                             batch_size=batch_size)
        gate = plugin.confidence_threshold(model_path, runtime)
    else:
        model = build_pose_model(model_path, batch_size=batch_size)
        gate = confidence_threshold(model_path=model_path)
    model.load()
    return model, gate


def predict_with(model: PoseModel, frames: numpy.ndarray, batch_size: int = 1) -> numpy.ndarray:
    """(N, S, S) grey uint8 frames -> (N, K, 3), as the live path feeds them."""
    out = []
    for start in range(0, len(frames), batch_size):
        chunk = frames[start:start + batch_size]
        rgb = numpy.repeat(chunk[..., None], 3, axis=3)
        out.extend(model.predict(numpy.ascontiguousarray(rgb)))
    if not out:
        return numpy.zeros((0, len(model.body_parts), 3), "float32")
    return numpy.stack(out).astype("float32")


def evaluate_model(model_path: str, dataset) -> dict:
    """Score a model on the dataset's held-out labelled frames."""
    frames, labels = dataset.arrays("val")
    model, gate = open_model(model_path, batch_size=1)
    predicted = predict_with(model, frames)
    mapped, missing = match_parts(predicted, model.body_parts, dataset.bodyparts)
    result = score(mapped, labels, dataset.bodyparts, gate)
    result["model"] = model_path
    result["missing_parts"] = missing
    return result


def measure_speed(model_path: str, batch: int = 2, iterations: int = 500,
                  warmup: int = 50) -> dict:
    """p50 / p99 of one live pose batch, as the live path calls predict()."""
    model, _ = open_model(model_path, batch_size=batch)
    rng = numpy.random.default_rng(0)
    frames = rng.integers(0, 255, (batch, 256, 256, 3)).astype("uint8")
    for _ in range(warmup):
        model.predict(frames)
    times = []
    for _ in range(iterations):
        started = time.perf_counter()
        model.predict(frames)
        times.append((time.perf_counter() - started) * 1000)
    return {"batch": batch, "iterations": iterations,
            "p50_ms": float(numpy.percentile(times, 50)),
            "p99_ms": float(numpy.percentile(times, 99)),
            "budget_ms": FRAME_BUDGET_MS}

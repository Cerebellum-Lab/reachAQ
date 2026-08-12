from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2


@dataclass(frozen=True)
class ClosedVideoValidation:
    video_path: str
    timestamp_path: str
    decoded_frame_count: int
    writer_frame_count: int
    timestamp_row_count: int
    writer_error_count: int
    writer_first_error: Optional[str]
    counter_backend: str
    warnings: Tuple[str, ...]
    failure: str = ""

    def diagnostics(self) -> dict:
        result = asdict(self)
        result["warnings"] = list(self.warnings)
        return result


def _ffprobe_frame_count(video_path: Path, timeout_seconds: float) -> int:
    completed = subprocess.run(
        (
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            video_path.as_posix(),
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=float(timeout_seconds),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or f"ffprobe exited with status {completed.returncode}"
        )
    value = completed.stdout.strip().splitlines()
    if not value or value[-1].strip().upper() == "N/A":
        raise RuntimeError("ffprobe did not report a decoded frame count")
    return int(value[-1].strip())


def _opencv_frame_count(video_path: Path) -> int:
    capture = cv2.VideoCapture(video_path.as_posix())
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not open the closed video")
    count = 0
    try:
        while capture.grab():
            count += 1
    finally:
        capture.release()
    return count


def _timestamp_row_count(timestamp_path: Path) -> int:
    if not timestamp_path.is_file():
        return 0
    with timestamp_path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def validate_closed_video(
    video_path: Path,
    timestamp_path: Path,
    *,
    writer_frame_count: int,
    writer_diagnostics: Optional[dict] = None,
    ffprobe_timeout_seconds: float = 120.0,
) -> ClosedVideoValidation:
    writer_diagnostics = dict(writer_diagnostics or {})
    warnings = []
    failure = ""
    counter_backend = "ffprobe"

    if not video_path.is_file():
        decoded_count = 0
        failure = f"closed video is missing: {video_path}"
    else:
        try:
            decoded_count = _ffprobe_frame_count(
                video_path,
                ffprobe_timeout_seconds,
            )
        except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as exc:
            warnings.append(f"ffprobe count unavailable: {exc}")
            counter_backend = "opencv_grab"
            try:
                decoded_count = _opencv_frame_count(video_path)
            except Exception as fallback_error:
                decoded_count = 0
                failure = (
                    "closed video is unreadable: "
                    f"{fallback_error.__class__.__name__}: {fallback_error}"
                )

    try:
        timestamp_count = _timestamp_row_count(timestamp_path)
    except (OSError, UnicodeError) as error:
        timestamp_count = 0
        failure = failure or (
            "camera timestamp file is unreadable: "
            f"{error.__class__.__name__}: {error}"
        )
    writer_count = int(writer_frame_count)
    writer_error_count = int(writer_diagnostics.get("errorCount", 0) or 0)
    writer_first_error = writer_diagnostics.get("firstError") or None
    if writer_error_count:
        warnings.append(
            f"writer reported {writer_error_count} error(s); first: "
            f"{writer_first_error or 'not available'}"
        )
    counts = {
        "decoded": decoded_count,
        "writer": writer_count,
        "timestamps": timestamp_count,
    }
    if len(set(counts.values())) > 1:
        warnings.append(
            "camera frame counts disagree: "
            + ", ".join(f"{name}={value}" for name, value in counts.items())
        )
    if decoded_count <= 0 and not failure:
        failure = "closed video contains zero decodable frames"
    if timestamp_count <= 0 and not failure:
        failure = "camera timestamp file contains zero rows"

    return ClosedVideoValidation(
        video_path=video_path.as_posix(),
        timestamp_path=timestamp_path.as_posix(),
        decoded_frame_count=decoded_count,
        writer_frame_count=writer_count,
        timestamp_row_count=timestamp_count,
        writer_error_count=writer_error_count,
        writer_first_error=writer_first_error,
        counter_backend=counter_backend,
        warnings=tuple(warnings),
        failure=failure,
    )

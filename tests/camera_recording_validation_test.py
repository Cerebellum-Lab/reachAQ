import shutil
import subprocess

import cv2
import numpy as np
import pytest

from tools.acquisition.model import camera_recording_validation as validation_module
from tools.acquisition.model.camera_recording_validation import validate_closed_video


def test_closed_video_count_mismatch_is_warning_only(tmp_path, monkeypatch):
    video = tmp_path / "left.mp4"
    timestamps = tmp_path / "left.txt"
    video.write_bytes(b"closed-video")
    timestamps.write_text("a\nb\nc\n", encoding="utf-8")
    monkeypatch.setattr(
        validation_module,
        "_ffprobe_packet_count",
        lambda *_args, **_kwargs: 2,
    )

    result = validate_closed_video(
        video,
        timestamps,
        writer_frame_count=3,
        writer_diagnostics={"errorCount": 1, "firstError": "write failed"},
    )

    assert result.failure == ""
    assert result.decoded_frame_count == 2
    assert any("counts disagree" in warning for warning in result.warnings)
    assert any("write failed" in warning for warning in result.warnings)


def test_missing_or_zero_frame_video_marks_source_incomplete(tmp_path, monkeypatch):
    timestamps = tmp_path / "left.txt"
    timestamps.write_text("row\n", encoding="utf-8")
    missing = validate_closed_video(
        tmp_path / "missing.mp4",
        timestamps,
        writer_frame_count=1,
    )
    assert "missing" in missing.failure

    video = tmp_path / "zero.mp4"
    video.write_bytes(b"closed-video")
    monkeypatch.setattr(
        validation_module,
        "_ffprobe_packet_count",
        lambda *_args, **_kwargs: 0,
    )
    zero = validate_closed_video(
        video,
        timestamps,
        writer_frame_count=0,
    )
    assert zero.failure == "closed video contains zero decodable frames"


def test_unreadable_timestamp_file_marks_source_incomplete(tmp_path, monkeypatch):
    video = tmp_path / "left.mp4"
    timestamps = tmp_path / "left.txt"
    video.write_bytes(b"closed-video")
    timestamps.write_bytes(b"\xff\xfe")
    monkeypatch.setattr(
        validation_module,
        "_ffprobe_packet_count",
        lambda *_args, **_kwargs: 2,
    )

    result = validate_closed_video(video, timestamps, writer_frame_count=2)

    assert "timestamp file is unreadable" in result.failure


def test_ffprobe_timeout_is_bounded_failure_without_unbounded_fallback(
    tmp_path,
    monkeypatch,
):
    video = tmp_path / "left.mp4"
    timestamps = tmp_path / "left.txt"
    video.write_bytes(b"closed-video")
    timestamps.write_text("row\n", encoding="utf-8")
    monkeypatch.setattr(
        validation_module,
        "_ffprobe_packet_count",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("ffprobe", 3.0)
        ),
    )
    monkeypatch.setattr(
        validation_module,
        "_opencv_frame_count",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("timed-out validation must not enter an unbounded fallback")
        ),
    )

    result = validate_closed_video(video, timestamps, writer_frame_count=1)

    assert result.counter_backend == "ffprobe_timeout"
    assert "timed out after 3.0 seconds" in result.failure


def test_the_frame_count_decodes_with_threads(monkeypatch, tmp_path):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"], seen["timeout"] = list(command), kwargs["timeout"]
        return subprocess.CompletedProcess(command, 0, stdout="12\n", stderr="")

    monkeypatch.setattr(validation_module.subprocess, "run", fake_run)

    assert validation_module._ffprobe_frame_count(tmp_path / "left.mp4", 42.0) == 12
    assert seen["command"][seen["command"].index("-threads") + 1] == "8"
    assert seen["timeout"] == 42.0


@pytest.mark.parametrize(("stream", "threads"), (
    ("mpeg4,256,256", "1"),     # the rig's mp4v: frame threads slow its decode down
    ("mpeg4,1024,1024", "8"),
    ("h264,1024,1024", "8"),
    ("", "8"),                  # an unreadable stream decodes as before and reports its own error
))
def test_the_full_decode_picks_its_threads_from_the_stream(monkeypatch, tmp_path, stream, threads):
    decodes = []

    def fake_run(command, **kwargs):
        if "-count_frames" in command:
            decodes.append(list(command))
            return subprocess.CompletedProcess(command, 0, stdout="12\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout=f"{stream}\n", stderr="")

    monkeypatch.setattr(validation_module.subprocess, "run", fake_run)

    assert validation_module._ffprobe_frame_count(tmp_path / "left.mp4", 42.0) == 12
    assert decodes[0][decodes[0].index("-threads") + 1] == threads


def test_the_closed_video_count_reads_packets_without_decoding(monkeypatch, tmp_path):
    # Decoding both cameras' 45-minute videos took 17 s of Stop on christielab10 at
    # 256 px mp4v, and would take about 112 s at 1024 px h264; counting their packets
    # took 0.26 s. --full session validation still decodes every frame.
    video = tmp_path / "left.mp4"
    timestamps = tmp_path / "left.txt"
    video.write_bytes(b"closed-video")
    timestamps.write_text("a\nb\n", encoding="utf-8")
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = list(command)
        return subprocess.CompletedProcess(command, 0, stdout="2\n", stderr="")

    monkeypatch.setattr(validation_module.subprocess, "run", fake_run)

    result = validate_closed_video(video, timestamps, writer_frame_count=2)

    assert "-count_packets" in seen["command"]
    assert "stream=nb_read_packets" in seen["command"]
    assert "-count_frames" not in seen["command"]
    assert (result.decoded_frame_count, result.counter_backend) == (2, "ffprobe_packets")
    assert (result.failure, result.warnings) == ("", ())


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="needs ffprobe")
def test_a_closed_mp4v_video_counts_its_frames(tmp_path):
    video = tmp_path / "left.mp4"
    timestamps = tmp_path / "left.txt"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 150, (64, 64), isColor=False)
    for index in range(5):
        writer.write(np.full((64, 64), index * 40, dtype=np.uint8))
    writer.release()
    timestamps.write_text("".join(f"{index}\n" for index in range(5)), encoding="utf-8")

    result = validate_closed_video(video, timestamps, writer_frame_count=5)

    assert (result.decoded_frame_count, result.counter_backend) == (5, "ffprobe_packets")
    assert (result.failure, result.warnings) == ("", ())


def test_the_validation_deadline_grows_with_the_recording():
    deadline = validation_module.closed_video_timeout_seconds
    assert deadline(0) == 30.0
    assert deadline(18_000) == 30.0     # 2 minutes at 150 fps
    assert deadline(54_052) == 54.052   # 6 minutes; decoded in 7.7 s at 1024x1024 H.264
    assert deadline(270_000) == 270.0   # 30 minutes

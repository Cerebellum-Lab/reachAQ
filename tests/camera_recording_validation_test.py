from tools.acquisition.model import camera_recording_validation as validation_module
from tools.acquisition.model.camera_recording_validation import validate_closed_video


def test_closed_video_count_mismatch_is_warning_only(tmp_path, monkeypatch):
    video = tmp_path / "left.mp4"
    timestamps = tmp_path / "left.txt"
    video.write_bytes(b"closed-video")
    timestamps.write_text("a\nb\nc\n", encoding="utf-8")
    monkeypatch.setattr(
        validation_module,
        "_ffprobe_frame_count",
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
        "_ffprobe_frame_count",
        lambda *_args, **_kwargs: 0,
    )
    zero = validate_closed_video(
        video,
        timestamps,
        writer_frame_count=0,
    )
    assert zero.failure == "closed video contains zero decodable frames"

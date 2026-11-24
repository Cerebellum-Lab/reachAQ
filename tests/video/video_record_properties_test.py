from reachaq.core.project import ProjectInfo
from reachaq.video import VideoRecordProperties, VideoRecordMode


def test_should_record_mode_none():
    properties = VideoRecordProperties()
    assert properties.should_record(False) is False
    assert properties.should_record(True) is False

    project_info = ProjectInfo(root="local")

    properties = VideoRecordProperties(project_info)
    assert properties.should_record(False) is False
    assert properties.should_record(True) is False

    properties = VideoRecordProperties(project_info, video_rotate_interval=3600)
    assert properties.should_record(False) is False
    assert properties.should_record(True) is False

    properties = VideoRecordProperties(project_info, image_interval=1)
    assert properties.should_record(False) is False
    assert properties.should_record(True) is False

    properties = VideoRecordProperties(project_info, video_rotate_interval=3600, image_interval=1)
    assert properties.should_record(False) is False
    assert properties.should_record(True) is False

    properties = VideoRecordProperties(None, video_rotate_interval=3600,  image_interval=1)
    assert properties.should_record(False) is False
    assert properties.should_record(True) is False


def test_should_record_continuous():
    project_info = ProjectInfo(root="local")

    properties = VideoRecordProperties(project_info, record_mode=VideoRecordMode.CONTINUOUS)
    assert properties.should_record(False) is False
    assert properties.should_record(True) is False

    properties = VideoRecordProperties(project_info, record_mode=VideoRecordMode.CONTINUOUS, video_rotate_interval=3600)
    assert properties.should_record(False) is True
    assert properties.should_record(True) is True

    properties = VideoRecordProperties(project_info, record_mode=VideoRecordMode.CONTINUOUS, image_interval=1)
    assert properties.should_record(False) is True
    assert properties.should_record(True) is True

    properties = VideoRecordProperties(project_info, record_mode=VideoRecordMode.CONTINUOUS, video_rotate_interval=3600,
                                       image_interval=1)
    assert properties.should_record(False) is True
    assert properties.should_record(True) is True

    properties = VideoRecordProperties(None, record_mode=VideoRecordMode.CONTINUOUS, video_rotate_interval=3600,
                                       image_interval=1)
    assert properties.should_record(False) is False
    assert properties.should_record(True) is False


def test_should_record_trigger():
    project_info = ProjectInfo(root="local")

    properties = VideoRecordProperties(project_info, record_mode=VideoRecordMode.TRIGGER)
    assert properties.should_record(False) is False
    assert properties.should_record(True) is False

    properties = VideoRecordProperties(project_info, record_mode=VideoRecordMode.TRIGGER, video_rotate_interval=3600)
    assert properties.should_record(False) is False
    assert properties.should_record(True) is True

    properties = VideoRecordProperties(project_info, record_mode=VideoRecordMode.TRIGGER, image_interval=1)
    assert properties.should_record(False) is False
    assert properties.should_record(True) is True

    properties = VideoRecordProperties(project_info, record_mode=VideoRecordMode.TRIGGER, video_rotate_interval=3600,
                                       image_interval=1)
    assert properties.should_record(False) is False
    assert properties.should_record(True) is True

    properties = VideoRecordProperties(None, record_mode=VideoRecordMode.TRIGGER, video_rotate_interval=3600,
                                       image_interval=1)
    assert properties.should_record(False) is False
    assert properties.should_record(True) is False

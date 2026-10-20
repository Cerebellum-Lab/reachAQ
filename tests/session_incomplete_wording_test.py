"""A source that failed with its data on disk is called incomplete, not unpersisted.

The session lists why it is incomplete, one reason per source that was not
cleanly written, and the operator reads that list in the "Session data
incomplete" alert. Every such reason read "<source> persistence failed", even
for a source whose data is on disk: live pose that ends early because
inference stopped mid-recording, or camera frames that are only partly
synchronized. The persistenceStatus field itself is unchanged; only the
sentence says what happened.
"""

from tools.acquisition.model.session_data_recorder import _persistence_reason


def test_a_source_that_failed_with_its_data_on_disk_is_called_incomplete():
    source = {
        "id": "pose",
        "persistenceStatus": "failed",
        "sampleCount": 2955,
        "failure": "Live pose ends early: live inference stopped during the recording",
    }

    assert _persistence_reason(source) == (
        "pose incomplete: Live pose ends early: live inference stopped during the recording")


def test_a_source_that_failed_with_nothing_written_still_says_persistence_failed():
    source = {
        "id": "pose",
        "persistenceStatus": "failed",
        "sampleCount": 0,
        "failure": "Timed out waiting for live pose files to close",
    }

    assert _persistence_reason(source) == (
        "pose persistence failed: Timed out waiting for live pose files to close")


def test_a_missing_source_is_still_called_missing():
    source = {"id": "camera.left", "persistenceStatus": "missing", "sampleCount": 0,
              "failure": None}

    assert _persistence_reason(source) == "camera.left persistence missing"

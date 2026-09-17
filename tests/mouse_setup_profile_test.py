import pytest

from tools.acquisition.model.mouse_setup_profile import (
    CameraGeometry,
    MouseSetupError,
    MouseSetupProfile,
    SetupMarker,
    SetupProfileStatus,
)


MARKERS = (
    {
        "marker_id": "nose",
        "label": "Nose",
        "color": "#e53935",
        "normalized_by_role": {"left": (0.55, 0.35), "right": (0.5, 0.35)},
    },
    {
        "marker_id": "right_paw",
        "label": "Right Paw",
        "color": "#1e88e5",
        "normalized_by_role": {"left": (0.45, 0.7)},
    },
)


def _profile(**changes):
    values = {
        "profile_id": "rig-a",
        "markers": MARKERS,
        "status": SetupProfileStatus.CALIBRATED,
        "provenance": "session 20260904_NTC_003, stereo calibrated",
    }
    values.update(changes)
    return MouseSetupProfile(**values)


def test_geometry_maps_normalized_points_to_pixels():
    geometry = CameraGeometry(width=800, height=600)
    assert geometry.to_pixels((0.5, 0.25)) == (400.0, 150.0)


def test_geometry_offset_accounts_for_a_cropped_capture():
    geometry = CameraGeometry(width=400, height=300, offset_x=100, offset_y=50)
    assert geometry.to_pixels((0.5, 0.5)) == (300.0, 200.0)


@pytest.mark.parametrize("size", [(0, 100), (100, 0), (-1, 10)])
def test_geometry_requires_a_positive_size(size):
    with pytest.raises(ValueError, match="positive width and height"):
        CameraGeometry(width=size[0], height=size[1])


def test_geometry_rejects_negative_offsets():
    with pytest.raises(ValueError, match="offsets cannot be negative"):
        CameraGeometry(width=10, height=10, offset_x=-1)


@pytest.mark.parametrize("point", [(-0.1, 0.5), (0.5, 1.1), (0.5,), (0.5, 0.5, 0.5)])
def test_normalized_positions_are_validated(point):
    with pytest.raises(ValueError):
        CameraGeometry(width=10, height=10).to_pixels(point)


def test_marker_requires_at_least_one_role():
    with pytest.raises(ValueError, match="at least one camera role"):
        SetupMarker(marker_id="nose", normalized_by_role={})


@pytest.mark.parametrize("color", ["red", "#12", "ffffff", ""])
def test_marker_colors_must_be_hex(color):
    with pytest.raises(ValueError, match="invalid color"):
        SetupMarker(
            marker_id="nose",
            color=color,
            normalized_by_role={"left": (0.5, 0.5)},
        )


def test_marker_label_defaults_to_its_id():
    marker = SetupMarker(marker_id="nose", normalized_by_role={"left": (0.5, 0.5)})
    assert marker.label == "nose"


def test_marker_roles_are_normalized_and_sorted():
    marker = SetupMarker(
        marker_id="nose",
        normalized_by_role={" Right ": (0.5, 0.5), "left": (0.4, 0.4)},
    )
    assert marker.roles == ("left", "right")
    assert marker.position_for("RIGHT") == (0.5, 0.5)


def test_profile_reports_every_role_its_markers_cover():
    assert _profile().roles == ("left", "right")


def test_profile_requires_at_least_one_marker():
    with pytest.raises(ValueError, match="at least one marker"):
        _profile(markers=())


def test_profile_rejects_duplicate_markers():
    with pytest.raises(ValueError, match="Duplicate marker ID"):
        _profile(markers=(MARKERS[0], MARKERS[0]))


def test_a_placeholder_profile_is_not_usable():
    profile = MouseSetupProfile(profile_id="draft", markers=MARKERS)
    assert profile.status is SetupProfileStatus.PLACEHOLDER
    assert profile.is_usable is False
    with pytest.raises(MouseSetupError, match="cannot be used for comparison"):
        profile.require_usable()


def test_a_calibrated_profile_must_record_its_provenance():
    with pytest.raises(ValueError, match="must record where its positions"):
        MouseSetupProfile(
            profile_id="rig-a",
            markers=MARKERS,
            status=SetupProfileStatus.CALIBRATED,
        )


def test_overlay_places_every_marker_for_a_role():
    overlay = _profile().overlay_for("left", CameraGeometry(width=800, height=600))
    assert [item["marker_id"] for item in overlay] == ["nose", "right_paw"]
    assert overlay[0]["pixel_x"] == pytest.approx(440.0)
    assert overlay[0]["pixel_y"] == pytest.approx(210.0)
    assert overlay[0]["color"] == "#e53935"


def test_overlay_skips_markers_without_a_position_for_that_role():
    overlay = _profile().overlay_for("right", CameraGeometry(width=800, height=600))
    assert [item["marker_id"] for item in overlay] == ["nose"]


def test_overlay_refuses_a_placeholder_profile():
    profile = MouseSetupProfile(profile_id="draft", markers=MARKERS)
    with pytest.raises(MouseSetupError, match="cannot be used for comparison"):
        profile.overlay_for("left", CameraGeometry(width=800, height=600))


def test_overlay_can_be_previewed_from_a_placeholder_profile():
    profile = MouseSetupProfile(profile_id="draft", markers=MARKERS)
    overlay = profile.overlay_for(
        "left", CameraGeometry(width=800, height=600), require_usable=False
    )
    assert len(overlay) == 2


def test_overlay_rejects_an_unknown_role():
    with pytest.raises(MouseSetupError, match="no positions for camera role"):
        _profile().overlay_for("top", CameraGeometry(width=800, height=600))


def test_record_round_trips():
    profile = _profile()
    restored = MouseSetupProfile.from_record(profile.to_record())
    assert restored == profile
    assert profile.to_record()["status"] == "calibrated"


def test_record_accepts_the_reference_keyed_marker_form():
    profile = MouseSetupProfile.from_record({
        "profile_id": "rig-a",
        "status": "placeholder_not_for_scientific_use",
        "markers": {
            "nose": {
                "label": "Nose",
                "color": "#e53935",
                "normalized_by_role": {"left": [0.55, 0.35]},
            }
        },
    })
    assert profile.markers[0].marker_id == "nose"
    assert profile.is_usable is False


def test_an_unsupported_schema_is_rejected():
    with pytest.raises(ValueError, match="Unsupported mouse setup schema"):
        _profile(schema_version=99)

from tools.acquisition.model.firmware_compatibility import FirmwareCompatibilityPolicy


def test_policy_accepts_exact_legacy_version_with_visible_fallback():
    result = FirmwareCompatibilityPolicy.load().evaluate("2.0.0")

    assert result.commands_allowed
    assert "time_sync" in result.optional_capabilities
    assert result.reported_capabilities == ()


def test_policy_rejects_unknown_or_wrong_connection_type():
    policy = FirmwareCompatibilityPolicy.load()

    assert not policy.evaluate("9.9.9").commands_allowed
    assert not policy.evaluate("0.1.0", emulation=False).commands_allowed
    assert policy.evaluate("0.1.0", emulation=True).commands_allowed


def test_policy_reports_board_timing_capabilities():
    result = FirmwareCompatibilityPolicy.load().evaluate(
        "Pellet: 2.1.0", wire_schema_version=1, capabilities=0b111,
    )

    assert result.commands_allowed
    assert result.reported_capabilities == (
        "timing_trailer", "time_sync", "finite_stim3_pulse",
    )


def test_new_firmware_requires_every_declared_capability():
    result = FirmwareCompatibilityPolicy.load().evaluate(
        "2.1.0", wire_schema_version=1, capabilities=0b011,
    )

    assert not result.commands_allowed
    assert result.missing_capabilities == ("finite_stim3_pulse",)

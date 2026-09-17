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


SYNTHETIC_POLICY = """
schema_version: 1
board_type: pellet_module
versions:
  - version: 9.9.9
    wire_schema_version: 1
    required_capabilities:
      - timing_trailer
      - time_sync
      - finite_stim3_pulse
    optional_capabilities: []
    hardware_release: synthetic
    qualification_status: test_only
"""


def test_requirement_enforcement_still_rejects_a_missing_capability(tmp_path):
    """A version that DOES declare requirements must still reject a short board.

    Written against a synthetic policy rather than a shipped version, because
    no shipped firmware declares required capabilities today: v2.1.0 asked for
    three its firmware never implemented, and requiring them rejected every
    real board. Pointing this at a real version again would delete the only
    coverage of the enforcement path the moment the policy changed - and that
    path is what protects the rig when a firmware finally does report
    capabilities.
    """
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(SYNTHETIC_POLICY, encoding="utf-8")

    result = FirmwareCompatibilityPolicy.load(policy_file).evaluate(
        "9.9.9", wire_schema_version=1, capabilities=0b011,
    )

    assert not result.commands_allowed
    assert result.missing_capabilities == ("finite_stim3_pulse",)


def test_shipped_v2_1_0_runs_without_capability_reporting():
    """The real v2.1.0 firmware answers no capability query at all.

    Verified on the rig and against the release source (commit 161b635): its
    command set has no counterpart to CAPABILITIES_REQUEST (0x23) or
    TIME_SYNC_REQUEST (0x1F). A board reporting nothing must still run, with
    the three timing capabilities simply absent.
    """
    result = FirmwareCompatibilityPolicy.load().evaluate(
        "2.1.0", wire_schema_version=1, capabilities=0,
    )

    assert result.commands_allowed
    assert result.missing_capabilities == ()
    assert "time_sync" in result.optional_capabilities

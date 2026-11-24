import json
import os
import time
import zipfile

import pytest

from tools.acquisition.model import diagnostic_bundle
from tools.acquisition.model.diagnostic_bundle import (
    REDACTED,
    BundleSources,
    SecretLeakError,
    build_diagnostic_bundle,
    redact_text,
)

KEYRING_FALLBACK_SECRET = "hunter2-fallback-Zq9"
PREFERENCES_SECRET = "prefs-token-Xk31"
CONFIG_SECRET = "cfg-api-key-Ww07"
LOG_SECRET = "log-password-Pp55"
URL_SECRET = "url-pass-Uu88"


def _all_text(path):
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name).decode("utf-8", errors="replace") for name in archive.namelist()}


@pytest.fixture
def rig(tmp_path):
    """A rig's files, with a secret planted in every place one could leak from."""
    data = tmp_path / "rawdatalocal"
    day = data / "20261123" / "rig1"
    day.mkdir(parents=True)
    (day / "20261123_rig1_001.log").write_text(
        "2026-11-23 INFO started\n"
        f"2026-11-23 DEBUG softmouse login password={LOG_SECRET} ok\n"
        f"2026-11-23 DEBUG GET https://ben:{URL_SECRET}@softmouse.example/colony\n"
        "2026-11-23 DEBUG header Authorization: Bearer abcdefghijklmnop\n"
        f"2026-11-23 DEBUG cache {KEYRING_FALLBACK_SECRET}\n",
        encoding="utf-8")
    old = day / "20261101_rig1_001.log"
    old.write_text("old log\n", encoding="utf-8")
    stale = time.time() - 30 * 86400
    os.utime(old, (stale, stale))
    for index in range(1, 8):
        session = day / f"session{index:03d}"
        (session / "streams").mkdir(parents=True)
        (session / "streams" / "stream_manifest.json").write_text(json.dumps({"session": index}))
        (session / "left.mp4").write_bytes(b"\0" * 1024)
        (session / "streams" / "nidaq.h5").write_bytes(b"\0" * 1024)
        mtime = time.time() - (10 - index) * 60
        os.utime(session, (mtime, mtime))
    config = tmp_path / "Autotrainer" / "system_configuration.yaml"
    config.parent.mkdir()
    config.write_text(
        "version: 59\npersistence: !PersistenceConfiguration\n"
        f"  outputLocation: {data.as_posix()}\n"
        f"apiPlugin:\n  api_key: {CONFIG_SECRET}\n  keypointCount: 12\n",
        encoding="utf-8")
    preferences = tmp_path / "Auto Trainer.conf"
    preferences.write_text(
        "[system]\nlog_level=20\n[softmouse]\n"
        f"token={PREFERENCES_SECRET}\nname_column=Physical Tag\n",
        encoding="utf-8")
    credential_dir = tmp_path / "config" / "reachAQ"
    credential_dir.mkdir(parents=True)
    credentials = credential_dir / "animal-metadata-credentials.json"
    credentials.write_text(json.dumps({"softmouse/password": KEYRING_FALLBACK_SECRET}))
    return {
        "data": data,
        "config": config,
        "preferences": preferences,
        "credentials": credentials,
        "tmp": tmp_path,
    }


def _sources(rig, **extra):
    return BundleSources(
        configuration_path=rig["config"],
        preferences_path=rig["preferences"],
        output_roots=(rig["data"], rig["credentials"].parent),
        log_dirs=(rig["credentials"].parent,),
        credential_files=(rig["credentials"],),
        **extra,
    )


def _build(rig, **extra):
    return build_diagnostic_bundle(
        rig["tmp"] / "out" / "bundle.zip",
        _sources(rig, **extra),
        versions=lambda: {"reachaq": "test"},
        inventory=lambda: {"gpus": "none"},
    )


def test_no_planted_secret_reaches_the_bundle(rig, monkeypatch):
    import keyring

    def refuse(*args, **kwargs):
        raise AssertionError("the bundle must never read the OS keyring")

    monkeypatch.setattr(keyring, "get_password", refuse)
    monkeypatch.setattr(keyring, "get_keyring", refuse)
    result = _build(rig, alerts=[{"title": "x", "message": f"password={LOG_SECRET}"}])
    raw = rig["tmp"].joinpath("out", "bundle.zip").read_bytes()
    texts = _all_text(result.path)
    blob = "\n".join(texts.values())
    for secret in (KEYRING_FALLBACK_SECRET, PREFERENCES_SECRET, CONFIG_SECRET, LOG_SECRET, URL_SECRET,
                   "abcdefghijklmnop"):
        assert secret not in blob, secret
        assert secret.encode() not in raw, secret
    assert not any("credential" in name for name in texts), "the credential file is never added"
    assert result.redactions >= 6
    # Redaction keeps what support needs.
    assert "keypointCount: 12" in texts["configuration/system_configuration.yaml"]
    assert "name_column=Physical Tag" in texts["preferences/Auto Trainer.conf"]
    assert f"api_key: {REDACTED}" in texts["configuration/system_configuration.yaml"]


def test_bundle_contents_and_limits(rig):
    result = _build(rig, subsystem_snapshot={"laser": {"state": "failed"}}, app_state={"status": "idle"})
    names = set(result.members)
    assert {"versions.json", "hardware_inventory.json", "manifest.json", "subsystem_statuses.json",
            "app_state.json", "configuration/system_configuration.yaml",
            "preferences/Auto Trainer.conf", "logs/20261123/rig1/20261123_rig1_001.log"} <= names
    assert "logs/20261123/rig1/20261101_rig1_001.log" not in names, "older than 7 days"
    sessions = {name.split("/")[3] for name in names if name.startswith("sessions/")}
    assert sessions == {f"session{index:03d}" for index in range(3, 8)}, "the 5 newest sessions"
    assert not any(name.endswith((".mp4", ".h5")) for name in names)
    manifest = json.loads(_all_text(result.path)["manifest.json"])
    assert "keyring was not read" in manifest["secret_handling"]


def test_a_secret_that_escapes_redaction_fails_the_export(rig, monkeypatch):
    monkeypatch.setattr(diagnostic_bundle, "redact_text", lambda text: (text, 0))
    sources = _sources(rig)
    # A secret the scrubber cannot know about by value, hidden from the
    # replace step: the final check must still catch the stored one.
    monkeypatch.setattr(diagnostic_bundle._Writer, "text",
                        lambda self, name, text, redact=True: (self.archive.writestr(name, text),
                                                               self.members.append(name)))
    with pytest.raises(SecretLeakError):
        build_diagnostic_bundle(rig["tmp"] / "leak.zip", sources,
                                versions=lambda: {}, inventory=lambda: {})
    assert not (rig["tmp"] / "leak.zip").exists()
    assert not (rig["tmp"] / "leak.zip.partial").exists()


@pytest.mark.parametrize("line, expected", [
    ("password: abc123", f"password: {REDACTED}"),
    ('"client_secret": "s3cr3t"', f'"client_secret": "{REDACTED}"'),
    ("softmouse/token=xyz", f"softmouse/token={REDACTED}"),
    ("tests passed: 12", "tests passed: 12"),
    ("keypoint_count: 12", "keypoint_count: 12"),
    ("connect postgres://user:pw@db/x", f"connect postgres://{REDACTED}@db/x"),
    ("GET /colony?user=ben&api_key=abc123&page=2", f"GET /colony?user=ben&api_key={REDACTED}&page=2"),
    ("Cookie: a=1; sessionid=zzz", f"Cookie: {REDACTED}"),
    ('{\\"password\\": \\"x1\\"}', '{\\"password\\": \\"' + REDACTED + '\\"}'),
    ("Authorization: Bearer abcdefghijklmnop", f"Authorization: {REDACTED} {REDACTED}"),
])

def test_redact_text(line, expected):
    assert redact_text(line)[0] == expected


def test_no_partial_file_is_left_when_the_build_fails(rig):
    def broken():
        raise OSError("disk full")

    with pytest.raises(OSError):
        build_diagnostic_bundle(rig["tmp"] / "x.zip", _sources(rig), versions=broken, inventory=dict)
    assert not list(rig["tmp"].glob("x.zip*"))


def test_command_line_bundle(rig, tmp_path, capsys):
    out = tmp_path / "cli.zip"
    code = diagnostic_bundle.main([
        "-o", str(out), "-c", str(rig["config"]), "--preferences-file", str(rig["preferences"]),
    ])
    assert code == 0
    names = _all_text(out)
    assert "configuration/system_configuration.yaml" in names
    assert any(name.startswith("logs/") for name in names), "logs found through the config's data folder"
    assert "Wrote" in capsys.readouterr().out

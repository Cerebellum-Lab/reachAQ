from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HARDWARE_DIR = REPO_ROOT / "tools" / "hardware"


def test_can_ownership_lock_uses_dedicated_runtime_directory():
    helper = (HARDWARE_DIR / "reachaq-reset-can.sh").read_text()
    tmpfiles = (HARDWARE_DIR / "reachaq-can.tmpfiles").read_text()

    assert 'lock_root="/run/lock/reachaq"' in helper
    assert "/tmp/reachaq-can-" not in helper
    assert "d /run/lock/reachaq 2770 root reachaq -" in tmpfiles


def test_can_service_creates_runtime_lock_directory():
    service = (HARDWARE_DIR / "reachaq-can.service").read_text()

    assert (
        "ExecStartPre=/usr/bin/systemd-tmpfiles --create "
        "/etc/tmpfiles.d/reachaq-can.conf"
    ) in service

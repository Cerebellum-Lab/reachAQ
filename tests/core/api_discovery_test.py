import json
import os

import pytest

from reachaq.core.api import ApiOptions, create_api_service, create_default_api_options
from reachaq.core.api.client import ApiClient
from reachaq.core.api.discovery import (
    ApiInstanceNotFound,
    find_api_instance,
    list_api_instances,
)


def _start(tmp_path, **changes):
    options = ApiOptions(port=0, discovery_dir=tmp_path, **changes)
    svc = create_api_service(options)
    assert svc.start()
    return svc


def test_two_default_instances_do_not_collide(tmp_path):
    # Two instances on the defaults: neither may lose its API to the other.
    first = create_api_service(create_default_api_options({"REACHAQ_API_DISCOVERY_DIR": str(tmp_path)}))
    second = create_api_service(create_default_api_options({
        "REACHAQ_API_DISCOVERY_DIR": str(tmp_path), "REACHAQ_SIMULATION": "1"}))
    try:
        assert first.start()
        assert second.start()
        assert first.port != second.port
    finally:
        first.stop()
        second.stop()


def test_running_service_publishes_and_stop_removes_its_record(tmp_path):
    svc = _start(tmp_path, simulated=True)
    try:
        (instance,) = list_api_instances(tmp_path)
        assert instance.pid == os.getpid()
        assert instance.port == svc.port
        assert instance.simulated is True
    finally:
        svc.stop()
    assert list_api_instances(tmp_path) == []
    assert list(tmp_path.iterdir()) == []


def test_client_finds_the_instance_by_simulation_flag(tmp_path):
    real = _start(tmp_path)
    try:
        # Both records carry this process's pid, so give the second its own file.
        record = next(tmp_path.glob("*.json"))
        data = json.loads(record.read_text(encoding="utf-8"))
        data["port"], data["simulated"] = 1, True
        (tmp_path / "instance-fake.json").write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ApiInstanceNotFound, match="2 reachAQ instances"):
            find_api_instance(directory=tmp_path)
        assert find_api_instance(simulated=False, directory=tmp_path).port == real.port
        with ApiClient(simulated=False, discovery_dir=tmp_path) as client:
            assert client.port == real.port
            assert client.request("NONE", nonce="n1")["nonce"] == "n1"
    finally:
        real.stop()


def test_stale_record_of_a_dead_process_is_ignored_and_removed(tmp_path):
    stale = tmp_path / "instance-999999.json"
    stale.write_text(json.dumps({
        "schemaVersion": 1, "pid": 2 ** 22 + 12345, "processStartTime": 1.0,
        "host": "127.0.0.1", "port": 1, "simulated": False,
    }), encoding="utf-8")
    assert list_api_instances(tmp_path) == []
    assert not stale.exists()
    with pytest.raises(ApiInstanceNotFound, match="No reachAQ instance"):
        find_api_instance(directory=tmp_path)


def test_record_of_a_reused_pid_is_ignored(tmp_path):
    svc = _start(tmp_path)
    try:
        record = next(tmp_path.glob("*.json"))
        data = json.loads(record.read_text(encoding="utf-8"))
        data["processStartTime"] -= 3600.0  # same pid, a different process
        record.write_text(json.dumps(data), encoding="utf-8")
        assert list_api_instances(tmp_path) == []
    finally:
        svc.stop()


def test_fixed_port_from_environment_is_still_honoured(tmp_path):
    options = create_default_api_options({"REACHAQ_API_PORT": "5000", "REACHAQ_API_DISCOVERY_DIR": str(tmp_path)})
    assert options.port == 5000
    assert options.discovery_dir == tmp_path

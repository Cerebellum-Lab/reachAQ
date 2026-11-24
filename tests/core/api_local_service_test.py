import dataclasses
import datetime as dt
import socket
import threading

import pytest

from reachaq.core.api import (
    ApiCommand,
    ApiCommandRequestErrorKind,
    ApiCommandRequestResponse,
    ApiCommandRequestResult,
    ApiEvent,
    ApiEventKind,
    ApiOptions,
    ApiSystemConfiguration,
    ApiApplicationMode,
    create_api_service,
    create_default_api_options,
)
from reachaq.core.api.client import ApiClient
from reachaq.core.api.options import DEFAULT_API_HOST, DEFAULT_API_PORT


@pytest.fixture
def service():
    svc = create_api_service(ApiOptions(port=0))
    assert svc.start()
    yield svc
    svc.stop()


def _respond(request, result=ApiCommandRequestResult.SUCCESS, data=None):
    return ApiCommandRequestResponse(nonce=request.nonce, command=request.command, result=result, data=data)


def test_request_reaches_delegate_and_response_carries_nonce(service):
    seen = []

    def delegate(request):
        seen.append(request)
        return _respond(request, data=ApiSystemConfiguration(
            application_version="3.1", device_id="rig", configuration_location="/c",
            data_location="/d", animal_location="/a", log_location="/l", inference_model=None))

    service.command_request_delegate = delegate
    with ApiClient(port=service.port) as client:
        rsp = client.request("GET_CONFIGURATION", nonce="n1", data={"x": 1})
    assert rsp == {
        "type": "response", "nonce": "n1", "command": "GET_CONFIGURATION", "result": "success",
        "errorCode": "none", "errorMessage": None,
        "data": {"application_version": "3.1", "device_id": "rig", "configuration_location": "/c",
                 "data_location": "/d", "animal_location": "/a", "log_location": "/l",
                 "inference_model": None},
    }
    assert seen[0].command is ApiCommand.GET_CONFIGURATION
    assert seen[0].data == {"x": 1}


def test_command_accepted_by_integer_id(service):
    service.command_request_delegate = _respond
    with ApiClient(port=service.port) as client:
        assert client.request(int(ApiCommand.GET_STATUS))["command"] == "GET_STATUS"


@pytest.mark.parametrize("line, nonce", [
    (b"not json\n", None),
    (b'{"type":"request","nonce":"n2","command":"FLY"}\n', "n2"),
    (b'{"type":"subscribe","nonce":"n3"}\n', "n3"),
    (b"[1, 2]\n", None),
])
def test_malformed_requests_get_unrecognized(service, line, nonce):
    service.command_request_delegate = _respond
    with ApiClient(port=service.port) as client:
        client.send_line(line)
        rsp = client.wait_response(nonce)
    assert rsp["result"] == "unrecognized"
    assert rsp["errorCode"] == "command_error"


def test_delegate_exception_is_reported(service):
    def delegate(request):
        raise RuntimeError("boom")

    service.command_request_delegate = delegate
    with ApiClient(port=service.port) as client:
        rsp = client.request("GET_STATUS")
    assert rsp["result"] == "exception"
    assert rsp["errorCode"] == "system_error"
    assert "boom" in rsp["errorMessage"]


def test_no_delegate_fails_cleanly(service):
    with ApiClient(port=service.port) as client:
        assert client.request("GET_STATUS")["result"] == "failed"


def test_events_are_pushed_to_every_client(service):
    with ApiClient(port=service.port) as a, ApiClient(port=service.port) as b:
        _wait_clients(service, 2)
        when = dt.datetime(2026, 11, 18, 12, 0, 0)
        service.send_event(ApiEvent(ApiEventKind.applicationModeChanged, when, 7,
                                    {"mode": ApiApplicationMode.RUNNING, "bad": float("nan")}))
        for client in (a, b):
            event = client.events.get(timeout=5)
            assert event == {"type": "event", "kind": 301, "name": "applicationModeChanged",
                             "when": "2026-11-18T12:00:00", "index": 7,
                             "context": {"mode": "RUNNING", "bad": None}}


def test_pending_command_result_follows_its_first_response(service):
    # The final result can be ready before the delegate has returned PENDING.
    def delegate(request):
        final = dataclasses.replace(_respond(request), result=ApiCommandRequestResult.SUCCESS)
        service.send_command_result(final)
        return _respond(request, ApiCommandRequestResult.PENDING_WITH_NOTIFICATION)

    service.command_request_delegate = delegate
    with ApiClient(port=service.port) as client:
        first = client.request("START_ACQUISITION", nonce="s1")
        final = client.wait_response("s1")
    assert first["result"] == "pending_with_notification"
    assert final["result"] == "success"


def test_late_command_result_goes_to_the_requesting_client(service):
    release = threading.Event()

    def delegate(request):
        def later():
            release.wait(5)
            service.send_command_result(dataclasses.replace(
                _respond(request), result=ApiCommandRequestResult.FAILED,
                error_code=ApiCommandRequestErrorKind.COMMAND_ERROR, error_message="no"))
        threading.Thread(target=later, daemon=True).start()
        return _respond(request, ApiCommandRequestResult.PENDING_WITH_NOTIFICATION)

    service.command_request_delegate = delegate
    with ApiClient(port=service.port) as asker, ApiClient(port=service.port) as other:
        assert asker.request("STOP_ACQUISITION", nonce="s2")["result"] == "pending_with_notification"
        release.set()
        assert asker.wait_response("s2")["errorMessage"] == "no"
        with pytest.raises(Exception):
            other.wait_response("s2", timeout=0.3)


def test_busy_port_leaves_the_service_stopped(service):
    second = create_api_service(ApiOptions(port=service.port))
    assert second.start() is False
    assert second.is_running is False
    second.send_event(ApiEvent(ApiEventKind.systemStatus))  # must not raise
    second.stop()


def test_stop_disconnects_clients():
    svc = create_api_service(ApiOptions(port=0))
    assert svc.start()
    port = svc.port
    client = ApiClient(port=port)
    _wait_clients(svc, 1)
    svc.stop()
    client.close()
    with pytest.raises(OSError):
        socket.create_connection((DEFAULT_API_HOST, port), timeout=0.5).close()


def test_disabled_options_create_no_service():
    assert create_api_service(ApiOptions(enabled=False)) is None


def test_default_options_from_environment():
    # Port 0: the OS picks a free port, and the instance publishes it for clients.
    assert DEFAULT_API_PORT == 0
    defaults = create_default_api_options({})
    assert (defaults.enabled, defaults.host, defaults.port, defaults.simulated) == (
        True, DEFAULT_API_HOST, 0, False)
    assert defaults.discovery_dir is not None
    options = create_default_api_options(
        {"REACHAQ_API_ENABLED": "0", "REACHAQ_API_HOST": "0.0.0.0", "REACHAQ_API_PORT": "5000"})
    assert (options.enabled, options.host, options.port) == (False, "0.0.0.0", 5000)
    with pytest.raises(ValueError):
        create_default_api_options({"REACHAQ_API_PORT": "70000"})


def _wait_clients(svc, count, timeout=5.0):
    deadline = dt.datetime.now() + dt.timedelta(seconds=timeout)
    while dt.datetime.now() < deadline:
        with svc._lock:
            if len(svc._clients) >= count:
                return
        threading.Event().wait(0.01)
    raise AssertionError(f"{count} clients did not connect")

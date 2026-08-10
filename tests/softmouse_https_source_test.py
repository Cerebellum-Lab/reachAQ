from pathlib import Path

import pytest

from tools.softmouse_sync.https_source import (
    SoftMouseCredentials,
    SoftMouseHttpsConfiguration,
    SoftMouseHttpsSource,
)


class Response:
    def __init__(self, *, url, content=b"", status=200, json_value=None):
        self.url = url
        self.content = content
        self.status_code = status
        self._json_value = json_value

    @property
    def text(self):
        return self.content.decode("utf-8", errors="replace")

    def json(self):
        if self._json_value is None:
            raise ValueError
        return self._json_value

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class Session:
    def __init__(self):
        self.calls = []
        self.poll_count = 0

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if url.endswith("login.do"):
            return Response(
                url=url,
                content=(
                    b'<form action="login.do?reqCode=doLogin" method="post">'
                    b'<input type="hidden" name="csrf" value="token">'
                    b"</form>"
                ),
            )
        if url.endswith("animals.do"):
            return Response(url=url, content=b"Animals")
        if "downLoadFile" in url:
            self.poll_count += 1
            if self.poll_count == 1:
                return Response(url=url, status=202)
            return Response(
                url=url,
                content=b"Physical Tag,Alt. ID,State\nPT-1,D4D47231005A30010000000000,Stock\n",
            )
        raise AssertionError(url)

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if method == "GET":
            return self.get(url, **kwargs)
        if "reqCode=doLogin" in url:
            assert kwargs["data"] == {
                "csrf": "token",
                "user": "ben",
                "password": "secret",
            }
            return Response(url="https://softmouse.example/animals.do", content=b"ok")
        if url.endswith("start-export"):
            return Response(url=url, json_value={"result": {"task": "task/1"}})
        raise AssertionError(url)


def test_https_source_logs_in_with_csrf_and_polls_encoded_task(tmp_path):
    session = Session()
    config = SoftMouseHttpsConfiguration(
        base_url="https://softmouse.example/",
        login_path="login.do",
        export_start_path="start-export",
        download_path_template="export/downLoadFile?taskid={task_id}",
        login_username_field="user",
        csrf_field="csrf",
        authentication_check_path="animals.do",
        export_suffix=".csv",
        task_id_json_path=("result", "task"),
        poll_interval_seconds=0,
    )
    source = SoftMouseHttpsSource(
        config, SoftMouseCredentials("ben", "secret"), session=session
    )

    path = source.download(tmp_path)

    assert path.suffix == ".csv"
    assert b"Physical Tag" in path.read_bytes()
    poll_urls = [url for method, url, _ in session.calls if "downLoadFile" in url]
    assert poll_urls[-1].endswith("taskid=task%2F1")
    assert "secret" not in repr(source.credentials)


def test_https_source_rejects_interactive_authentication_challenge(tmp_path):
    session = Session()
    original_get = session.get

    def challenged_get(url, **kwargs):
        if url.endswith("animals.do"):
            return Response(url=url, content=b"Enter verification code for two-factor login")
        return original_get(url, **kwargs)

    session.get = challenged_get
    source = SoftMouseHttpsSource(
        SoftMouseHttpsConfiguration(
            base_url="https://softmouse.example/",
            login_path="login.do",
            export_start_path="start-export",
            download_path_template="download/{task_id}",
            authentication_check_path="animals.do",
            csrf_field="csrf",
            login_username_field="user",
        ),
        SoftMouseCredentials("ben", "secret"),
        session=session,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(RuntimeError, match="interactive challenge"):
        source.download(tmp_path)


def test_https_source_requires_expected_colony_marker(tmp_path):
    session = Session()
    source = SoftMouseHttpsSource(
        SoftMouseHttpsConfiguration(
            base_url="https://softmouse.example/",
            login_path="login.do",
            export_start_path="start-export",
            download_path_template="download/{task_id}",
            authentication_check_path="animals.do",
            expected_colony_marker="Christie Lab Colony",
            csrf_field="csrf",
            login_username_field="user",
        ),
        SoftMouseCredentials("ben", "secret"),
        session=session,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(RuntimeError, match="colony marker"):
        source.download(tmp_path)


def test_https_source_retries_transient_network_failure_on_safe_get(tmp_path):
    session = Session()
    original_request = session.request
    failed_once = False

    def transient_request(method, url, **kwargs):
        nonlocal failed_once
        if method == "GET" and url.endswith("login.do") and not failed_once:
            failed_once = True
            raise OSError("temporary network failure")
        return original_request(method, url, **kwargs)

    session.request = transient_request
    source = SoftMouseHttpsSource(
        SoftMouseHttpsConfiguration(
            base_url="https://softmouse.example/",
            login_path="login.do",
            export_start_path="start-export",
            download_path_template="export/downLoadFile?taskid={task_id}",
            authentication_check_path="animals.do",
            csrf_field="csrf",
            login_username_field="user",
            export_suffix=".csv",
            task_id_json_path=("result", "task"),
            poll_interval_seconds=0,
            transient_retry_initial_seconds=0,
        ),
        SoftMouseCredentials("ben", "secret"),
        session=session,
        sleep=lambda _seconds: None,
    )

    path = source.download(tmp_path)

    assert failed_once
    assert b"Physical Tag" in path.read_bytes()

import csv

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
        self.animals_content = b'<a id="exportMouseMenuButton">Export</a>'
        self.home_content = (
            b'<a href="smdb/mouseline/list.do?reqCode=gotoMouselinelist&amp;ownerCode=34867">'
            b"Jason Christie</a>"
        )
        self.list_rows = [
            {
                "id": 1,
                "ownerId": 34867,
                "owner": "Jason Christie",
                "mousestate": "Stock",
                "physicaltag": "PT-1",
                "plateIdPattern": "360002353933099",
            },
            {
                "id": 2,
                "ownerId": 34867,
                "owner": "Jason Christie",
                "mousestate": "Mating",
                "physicaltag": "PT-2",
                "plateIdPattern": "360002353933101",
            },
        ]

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if method == "GET" and url == "https://www.softmouse.net/":
            return Response(
                url=url,
                content=(
                    b'<form action="login.do" method="post">'
                    b'<input type="hidden" name="csrftoken" value="token">'
                    b"</form>"
                ),
            )
        if method == "POST" and "reqCode=doLogin" in url:
            assert kwargs["data"] == {
                "csrftoken": "token",
                "username": "ben",
                "password": "secret",
            }
            return Response(url="https://www.softmouse.net/HomePage.do", content=self.home_content)
        if method == "GET" and "gotoMouselinelist" in url:
            return Response(url=url, content=b"Selected")
        if method == "GET" and url.endswith("smdb/mouse/list.do"):
            return Response(url=url, content=self.animals_content)
        if method == "POST" and url.endswith("mouse/list.json"):
            return Response(
                url=url,
                json_value={
                    "records": 2,
                    "total": 1,
                    "page": 1,
                    "list": self.list_rows,
                },
            )
        raise AssertionError((method, url, kwargs))


def test_https_source_selects_christie_active_scope_and_writes_csv(tmp_path):
    session = Session()
    source = SoftMouseHttpsSource(
        SoftMouseHttpsConfiguration(),
        SoftMouseCredentials("ben", "secret"),
        session=session,
    )

    path = source.download(tmp_path)

    assert path.suffix == ".csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["Physical Tag"] == "PT-1"
    assert rows[0]["Plate ID"] == "360002353933099"
    assert rows[0]["State"] == "Stock"
    list_call = next(call for call in session.calls if call[1].endswith("mouse/list.json"))
    payload = list_call[2]["data"]
    assert ("ownerId", "34867") in payload
    assert [value for name, value in payload if name == "mouseStates"] == [
        "MATING",
        "STOCK",
        "WEANLING",
        "ORDERED",
    ]
    assert "secret" not in repr(source.credentials)


def test_https_source_rejects_interactive_authentication_challenge(tmp_path):
    session = Session()
    session.animals_content = b"Enter verification code for two-factor login"
    source = SoftMouseHttpsSource(
        SoftMouseHttpsConfiguration(),
        SoftMouseCredentials("ben", "secret"),
        session=session,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(RuntimeError, match="interactive challenge"):
        source.download(tmp_path)


def test_https_source_requires_exact_christie_colony(tmp_path):
    session = Session()
    session.home_content = b'<a href="other">Another Lab</a>'
    source = SoftMouseHttpsSource(
        SoftMouseHttpsConfiguration(),
        SoftMouseCredentials("ben", "secret"),
        session=session,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(RuntimeError, match="Jason Christie"):
        source.download(tmp_path)


def test_https_source_rejects_non_christie_export_scope(tmp_path):
    session = Session()
    session.list_rows[0]["ownerId"] = 999
    source = SoftMouseHttpsSource(
        SoftMouseHttpsConfiguration(),
        SoftMouseCredentials("ben", "secret"),
        session=session,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(RuntimeError, match="non-Christie owner"):
        source.download(tmp_path)


def test_https_source_fetches_every_active_page(tmp_path):
    session = Session()
    original_request = session.request

    def paged_request(method, url, **kwargs):
        if method == "POST" and url.endswith("mouse/list.json"):
            payload = dict(kwargs["data"])
            page = int(payload["page"])
            row = dict(session.list_rows[page - 1])
            return Response(
                url=url,
                json_value={
                    "records": 2,
                    "total": 2,
                    "page": page,
                    "list": [row],
                },
            )
        return original_request(method, url, **kwargs)

    session.request = paged_request
    source = SoftMouseHttpsSource(
        SoftMouseHttpsConfiguration(page_size=1),
        SoftMouseCredentials("ben", "secret"),
        session=session,
    )

    path = source.download(tmp_path)

    with path.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_https_source_retries_transient_network_failure_on_safe_get(tmp_path):
    session = Session()
    original_request = session.request
    failed_once = False

    def transient_request(method, url, **kwargs):
        nonlocal failed_once
        if method == "GET" and url == "https://www.softmouse.net/" and not failed_once:
            failed_once = True
            raise OSError("temporary network failure")
        return original_request(method, url, **kwargs)

    session.request = transient_request
    source = SoftMouseHttpsSource(
        SoftMouseHttpsConfiguration(
            transient_retry_initial_seconds=0,
        ),
        SoftMouseCredentials("ben", "secret"),
        session=session,
        sleep=lambda _seconds: None,
    )

    path = source.download(tmp_path)

    assert failed_once
    assert path.read_text(encoding="utf-8").startswith("SoftMouse ID,")

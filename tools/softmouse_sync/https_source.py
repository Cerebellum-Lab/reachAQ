from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import quote, urljoin, urlparse


@dataclass(frozen=True)
class SoftMouseCredentials:
    username: str
    password: str = field(repr=False)

    @classmethod
    def from_keyring(cls, service: str, username: str) -> "SoftMouseCredentials":
        try:
            import keyring
        except ImportError as exc:
            raise RuntimeError("keyring is required for unattended publication") from exc
        password = keyring.get_password(service, username)
        if password is None:
            raise RuntimeError(f"No OS-keyring credential found for {service!r}")
        return cls(username, password)


@dataclass(frozen=True)
class SoftMouseHttpsConfiguration:
    base_url: str
    login_path: str
    export_start_path: str
    download_path_template: str
    login_username_field: str = "username"
    login_password_field: str = "password"
    csrf_field: Optional[str] = None
    export_start_method: str = "POST"
    export_parameters: Mapping[str, str] = field(default_factory=dict)
    task_id_json_path: Tuple[str, ...] = ("taskid",)
    task_id_regex: Optional[str] = None
    authentication_check_path: Optional[str] = None
    authenticated_page_marker: Optional[str] = None
    expected_colony_marker: Optional[str] = None
    login_page_marker: str = "login.do"
    authentication_challenge_markers: Tuple[str, ...] = (
        "captcha",
        "multi-factor",
        "two-factor",
        "verification code",
    )
    request_timeout_seconds: float = 30.0
    export_deadline_seconds: float = 180.0
    poll_interval_seconds: float = 2.0
    transient_request_attempts: int = 3
    transient_retry_initial_seconds: float = 0.5
    transient_retry_max_seconds: float = 5.0
    export_suffix: str = ".xlsx"

    def __post_init__(self) -> None:
        origin = urlparse(self.base_url)
        if origin.scheme != "https" or not origin.netloc:
            raise ValueError("SoftMouse base_url must be an absolute HTTPS URL")
        if self.export_start_method.upper() not in {"GET", "POST"}:
            raise ValueError("export_start_method must be GET or POST")
        if self.export_suffix.casefold() not in {".xlsx", ".csv"}:
            raise ValueError("export_suffix must be .xlsx or .csv")
        if self.transient_request_attempts < 1:
            raise ValueError("transient_request_attempts must be at least 1")


class _LoginFormParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms = []
        self._current = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag.casefold() == "form":
            self._current = {
                "action": attributes.get("action", ""),
                "method": attributes.get("method", "post").upper(),
                "hidden": {},
            }
            self.forms.append(self._current)
        elif tag.casefold() == "input" and self._current is not None:
            if attributes.get("type", "").casefold() == "hidden" and attributes.get("name"):
                self._current["hidden"][attributes["name"]] = attributes.get("value", "")

    def handle_endtag(self, tag):
        if tag.casefold() == "form":
            self._current = None


class SoftMouseHttpsSource:
    """Configurable replay of the authorized login/export HTTP workflow.

    The export-start configuration is intentionally deployment input: it must be
    populated from a locally redacted authorized capture, because SoftMouse does
    not publish a stable contract for this internal endpoint.
    """

    def __init__(
        self,
        configuration: SoftMouseHttpsConfiguration,
        credentials: SoftMouseCredentials,
        *,
        session=None,
        sleep=time.sleep,
    ):
        self.configuration = configuration
        self.credentials = credentials
        if session is None:
            try:
                import requests
            except ImportError as exc:
                raise RuntimeError("requests is required for HTTPS publication") from exc
            session = requests.Session()
        self.session = session
        self._sleep = sleep

    def download(self, destination_directory: Path) -> Path:
        self._login()
        task_id, immediate = self._start_export()
        content = immediate if immediate is not None else self._poll_download(task_id)
        self._validate_download(content)
        destination = Path(destination_directory) / (
            "SoftMouse-AnimalList-download" + self.configuration.export_suffix
        )
        destination.write_bytes(content)
        return destination

    def _url(self, path: str) -> str:
        url = urljoin(self.configuration.base_url.rstrip("/") + "/", path)
        if urlparse(url).netloc != urlparse(self.configuration.base_url).netloc:
            raise ValueError("Refusing a SoftMouse request outside the configured origin")
        return url

    def _login(self) -> None:
        cfg = self.configuration
        login_url = self._url(cfg.login_path)
        response = self._transient_request("GET", login_url)
        response.raise_for_status()
        parser = _LoginFormParser()
        parser.feed(response.text)
        if not parser.forms:
            raise RuntimeError("SoftMouse login form was not found")
        form = next(
            (
                item
                for item in parser.forms
                if "login" in item["action"].casefold()
            ),
            parser.forms[0],
        )
        values = dict(form["hidden"])
        if cfg.csrf_field and cfg.csrf_field not in values:
            raise RuntimeError(f"SoftMouse CSRF field {cfg.csrf_field!r} was not found")
        values[cfg.login_username_field] = self.credentials.username
        values[cfg.login_password_field] = self.credentials.password
        target = self._url(form["action"] or cfg.login_path)
        login_response = self.session.request(
            form["method"],
            target,
            data=values,
            timeout=cfg.request_timeout_seconds,
        )
        login_response.raise_for_status()
        check = login_response
        if cfg.authentication_check_path:
            check = self._transient_request(
                "GET", self._url(cfg.authentication_check_path)
            )
            check.raise_for_status()
        if cfg.login_page_marker.casefold() in check.url.casefold():
            raise RuntimeError("SoftMouse authentication returned to the login page")
        check_text = check.text[:100_000].casefold()
        challenge = next(
            (
                marker
                for marker in cfg.authentication_challenge_markers
                if marker.casefold() in check_text
            ),
            None,
        )
        if challenge is not None:
            raise RuntimeError(
                "SoftMouse authentication requires an interactive challenge "
                f"({challenge}); unattended publication stopped"
            )
        if (
            cfg.authenticated_page_marker
            and cfg.authenticated_page_marker.casefold() not in check_text
        ):
            raise RuntimeError("SoftMouse authenticated-page marker was not found")
        if (
            cfg.expected_colony_marker
            and cfg.expected_colony_marker.casefold() not in check_text
        ):
            raise RuntimeError("SoftMouse authenticated colony marker was not found")

    def _start_export(self):
        cfg = self.configuration
        method = cfg.export_start_method.upper()
        kwargs = {"params" if method == "GET" else "data": dict(cfg.export_parameters)}
        export_url = self._url(cfg.export_start_path)
        if method == "GET":
            response = self._transient_request(method, export_url, **kwargs)
        else:
            # Starting an export may not be idempotent. Do not create duplicate
            # jobs by automatically replaying a timed-out POST.
            response = self.session.request(
                method,
                export_url,
                timeout=cfg.request_timeout_seconds,
                **kwargs,
            )
        response.raise_for_status()
        if self._looks_like_export(response.content):
            return None, response.content
        task_id = None
        try:
            value: Any = response.json()
            for part in cfg.task_id_json_path:
                value = value[part]
            task_id = str(value)
        except (ValueError, KeyError, TypeError):
            if cfg.task_id_regex:
                match = re.search(cfg.task_id_regex, response.text)
                if match:
                    task_id = match.group(1)
        if not task_id:
            raise RuntimeError("SoftMouse export response did not contain a task ID")
        return task_id, None

    def _poll_download(self, task_id: str) -> bytes:
        cfg = self.configuration
        deadline = time.monotonic() + cfg.export_deadline_seconds
        path = cfg.download_path_template.format(task_id=quote(task_id, safe=""))
        while time.monotonic() < deadline:
            response = self._transient_request("GET", self._url(path))
            if response.status_code in {202, 204, 404, 429, 500, 502, 503, 504}:
                self._sleep(cfg.poll_interval_seconds)
                continue
            response.raise_for_status()
            if self._looks_like_export(response.content):
                return response.content
            if "login" in response.text[:1000].casefold():
                raise RuntimeError("SoftMouse session expired while waiting for export")
            self._sleep(cfg.poll_interval_seconds)
        raise TimeoutError("Timed out waiting for the SoftMouse export task")

    def _transient_request(self, method: str, url: str, **kwargs):
        cfg = self.configuration
        delay = cfg.transient_retry_initial_seconds
        last_response = None
        last_error = None
        for attempt in range(cfg.transient_request_attempts):
            try:
                response = self.session.request(
                    method,
                    url,
                    timeout=cfg.request_timeout_seconds,
                    **kwargs,
                )
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= cfg.transient_request_attempts:
                    raise
                self._sleep(delay)
                delay = min(
                    cfg.transient_retry_max_seconds,
                    max(delay * 2, 0.1),
                )
                continue
            last_response = response
            if response.status_code not in {429, 500, 502, 503, 504}:
                return response
            if attempt + 1 < cfg.transient_request_attempts:
                self._sleep(delay)
                delay = min(cfg.transient_retry_max_seconds, max(delay * 2, 0.1))
        if last_response is not None:
            return last_response
        raise last_error

    def _looks_like_export(self, content: bytes) -> bool:
        if self.configuration.export_suffix.casefold() == ".xlsx":
            return content.startswith(b"PK\x03\x04")
        beginning = content[:4096].lstrip().lower()
        return b"physical tag" in beginning and not beginning.startswith(b"<html")

    def _validate_download(self, content: bytes) -> None:
        if len(content) < 32 or not self._looks_like_export(content):
            raise RuntimeError("SoftMouse returned HTML, an error, or an invalid export file")

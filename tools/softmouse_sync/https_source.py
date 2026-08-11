from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Tuple
from urllib.parse import parse_qs, urljoin, urlparse


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
    """Application-owned description of the Christie SoftMouse export flow."""

    base_url: str = "https://www.softmouse.net/"
    login_path: str = ""
    login_submit_path: str = "login.do?reqCode=doLogin"
    login_username_field: str = "username"
    login_password_field: str = "password"
    csrf_field: str = "csrftoken"
    colony_name: str = "Jason Christie"
    colony_owner_parameter: str = "ownerCode"
    animal_list_path: str = "smdb/mouse/list.do"
    animal_list_marker: str = "exportMouseMenuButton"
    animal_data_path: str = "mouse/list.json"
    active_states: Tuple[str, ...] = (
        "MATING",
        "STOCK",
        "WEANLING",
        "ORDERED",
    )
    page_size: int = 100
    maximum_export_rows: int = 10_000
    authentication_challenge_markers: Tuple[str, ...] = (
        "captcha",
        "multi-factor",
        "two-factor",
        "verification code",
    )
    request_timeout_seconds: float = 30.0
    transient_request_attempts: int = 3
    transient_retry_initial_seconds: float = 0.5
    transient_retry_max_seconds: float = 5.0
    export_suffix: str = ".csv"

    def __post_init__(self) -> None:
        origin = urlparse(self.base_url)
        if origin.scheme != "https" or not origin.netloc:
            raise ValueError("SoftMouse base_url must be an absolute HTTPS URL")
        if self.export_suffix.casefold() != ".csv":
            raise ValueError("The normalized SoftMouse Animals export must be .csv")
        if self.transient_request_attempts < 1:
            raise ValueError("transient_request_attempts must be at least 1")
        if self.maximum_export_rows < 1:
            raise ValueError("maximum_export_rows must be positive")
        if self.page_size < 1:
            raise ValueError("page_size must be positive")


class _SoftMousePageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms = []
        self.links = []
        self.inputs = []
        self._current_form = None
        self._current_link = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        name = tag.casefold()
        if name == "form":
            self._current_form = {
                "action": attributes.get("action", ""),
                "method": attributes.get("method", "post").upper(),
                "hidden": {},
            }
            self.forms.append(self._current_form)
        elif name == "input":
            self.inputs.append(attributes)
            if (
                self._current_form is not None
                and attributes.get("type", "").casefold() == "hidden"
                and attributes.get("name")
            ):
                self._current_form["hidden"][attributes["name"]] = attributes.get(
                    "value", ""
                )
        elif name == "a":
            self._current_link = {"href": attributes.get("href", ""), "text": []}
            self.links.append(self._current_link)

    def handle_data(self, data):
        if self._current_link is not None:
            self._current_link["text"].append(data)

    def handle_endtag(self, tag):
        name = tag.casefold()
        if name == "form":
            self._current_form = None
        elif name == "a":
            self._current_link = None


class SoftMouseHttpsSource:
    """Browserless replay of the fixed Christie SoftMouse Animals export flow."""

    _EXPORT_COLUMNS = (
        ("SoftMouse ID", "id"),
        ("Animal SID", "sid"),
        ("Physical Tag", "physicaltag"),
        ("Plate ID", "plateIdPattern"),
        ("State", "mousestate"),
        ("Sex", "sex"),
        ("Date of Birth", "dateofbirth"),
        ("Strain", "strain"),
        ("Mouseline", "mouseline"),
        ("Genotype", "genotype"),
        ("Cage Tag", "cagetag"),
        ("Cage Barcode", "cagebarcode"),
        ("Protocol", "protocol"),
        ("Owner", "owner"),
    )

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
        owner_code = self._login_and_select_colony()
        rows = self._download_and_validate_export_scope(owner_code)
        destination = Path(destination_directory) / (
            "SoftMouse-AnimalList-download" + self.configuration.export_suffix
        )
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[header for header, _ in self._EXPORT_COLUMNS],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        header: "" if row.get(source) is None else row.get(source)
                        for header, source in self._EXPORT_COLUMNS
                    }
                )
        return destination

    def _url(self, path: str) -> str:
        url = urljoin(self.configuration.base_url.rstrip("/") + "/", path)
        if urlparse(url).netloc != urlparse(self.configuration.base_url).netloc:
            raise ValueError("Refusing a SoftMouse request outside the configured origin")
        return url

    @staticmethod
    def _parse_page(text: str) -> _SoftMousePageParser:
        parser = _SoftMousePageParser()
        parser.feed(text)
        return parser

    def _login_and_select_colony(self) -> str:
        cfg = self.configuration
        response = self._transient_request("GET", self._url(cfg.login_path))
        response.raise_for_status()
        parser = self._parse_page(response.text)
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
        if cfg.csrf_field not in values:
            raise RuntimeError(f"SoftMouse CSRF field {cfg.csrf_field!r} was not found")
        values[cfg.login_username_field] = self.credentials.username
        values[cfg.login_password_field] = self.credentials.password

        # SoftMouse changes the form action with JavaScript before submitting.
        # A requests-based client must use that final endpoint explicitly.
        login_response = self.session.request(
            form["method"],
            self._url(cfg.login_submit_path),
            data=values,
            timeout=cfg.request_timeout_seconds,
        )
        login_response.raise_for_status()
        login_parser = self._parse_page(login_response.text)
        if any("login" in item["action"].casefold() for item in login_parser.forms):
            raise RuntimeError("SoftMouse rejected the stored username or password")
        self._reject_interactive_challenge(login_response.text)

        target_name = cfg.colony_name.casefold()
        matching_links = [
            link
            for link in login_parser.links
            if " ".join(link["text"]).strip().casefold() == target_name
        ]
        if len(matching_links) != 1:
            raise RuntimeError(
                f"Expected exactly one SoftMouse colony named {cfg.colony_name!r}; "
                f"found {len(matching_links)}"
            )
        colony_href = matching_links[0]["href"]
        colony_url = self._url(colony_href)
        owner_values = parse_qs(urlparse(colony_url).query).get(
            cfg.colony_owner_parameter, []
        )
        if len(owner_values) != 1 or not owner_values[0]:
            raise RuntimeError("SoftMouse Christie colony link has no owner identifier")
        owner_code = owner_values[0]

        selected = self._transient_request("GET", colony_url)
        selected.raise_for_status()
        animals = self._transient_request("GET", self._url(cfg.animal_list_path))
        animals.raise_for_status()
        self._reject_interactive_challenge(animals.text)
        if cfg.animal_list_marker.casefold() not in animals.text.casefold():
            raise RuntimeError("SoftMouse Animals page marker was not found")
        return owner_code

    def _reject_interactive_challenge(self, text: str) -> None:
        check_text = text[:100_000].casefold()
        challenge = next(
            (
                marker
                for marker in self.configuration.authentication_challenge_markers
                if marker.casefold() in check_text
            ),
            None,
        )
        if challenge is not None:
            raise RuntimeError(
                "SoftMouse authentication requires an interactive challenge "
                f"({challenge}); unattended publication stopped"
            )

    def _download_and_validate_export_scope(self, owner_code: str):
        cfg = self.configuration
        rows = []
        expected_records = None
        expected_pages = None
        page = 1
        while expected_pages is None or page <= expected_pages:
            payload = [
                ("_search", "false"),
                ("rows", str(cfg.page_size)),
                ("page", str(page)),
                ("sidx", "birthDate"),
                ("sord", "asc"),
                ("filterMode", "active"),
                ("ownerId", owner_code),
            ]
            payload.extend(("mouseStates", state) for state in cfg.active_states)
            response = self._transient_request(
                "POST",
                self._url(cfg.animal_data_path),
                data=payload,
                headers={
                    "X-Requested-With": "XMLHttpRequest",
                    "Accept": "application/json, text/javascript, */*; q=0.01",
                    "Referer": self._url(cfg.animal_list_path),
                },
            )
            response.raise_for_status()
            try:
                value = response.json()
                record_count = int(value["records"])
                page_count = int(value["total"])
                returned_page = int(value["page"])
                page_rows = value.get("list", value.get("rows", []))
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise RuntimeError("SoftMouse Animals list did not return valid JSON") from exc
            if expected_records is None:
                expected_records = record_count
                expected_pages = page_count
            if record_count != expected_records or page_count != expected_pages:
                raise RuntimeError("SoftMouse Animals list changed during pagination")
            if returned_page != page:
                raise RuntimeError("SoftMouse returned an unexpected Animals page")
            if not isinstance(page_rows, list):
                raise RuntimeError("SoftMouse returned invalid Animals rows")
            rows.extend(page_rows)
            page += 1

        record_count = expected_records or 0
        if record_count < 1:
            raise RuntimeError("SoftMouse Christie active-animal export is empty")
        if record_count > cfg.maximum_export_rows:
            raise RuntimeError(
                f"SoftMouse reports {record_count} active Christie animals; its "
                f"configured publication safety limit is {cfg.maximum_export_rows}"
            )
        if not rows:
            raise RuntimeError("SoftMouse returned no rows for export-scope validation")
        if len(rows) != record_count:
            raise RuntimeError(
                f"SoftMouse reported {record_count} active animals but returned "
                f"{len(rows)}"
            )
        allowed_states = {state.casefold() for state in cfg.active_states}
        identifiers = set()
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError("SoftMouse returned an invalid Animals row")
            if str(row.get("ownerId", "")) != owner_code:
                raise RuntimeError("SoftMouse export scope includes a non-Christie owner")
            if str(row.get("owner", "")).strip().casefold() != cfg.colony_name.casefold():
                raise RuntimeError("SoftMouse export scope owner name is not Christie")
            if str(row.get("mousestate", "")).strip().casefold() not in allowed_states:
                raise RuntimeError("SoftMouse export scope includes an inactive animal")
            identifier = row.get("id")
            if identifier is None or identifier in identifiers:
                raise RuntimeError("SoftMouse export contains a missing or duplicate animal ID")
            identifiers.add(identifier)
        return rows

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

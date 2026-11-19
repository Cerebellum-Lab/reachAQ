"""A minimal blocking client for the reachAQ local API."""

import json
import queue
import socket
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .discovery import find_api_instance
from .options import DEFAULT_API_HOST


class ApiClient:
    """Connects, sends requests and collects responses and events.

    ``request()`` returns the first response for its nonce; for a
    ``pending_with_notification`` command call ``wait_response(nonce)`` for the
    final one. Events accumulate in ``events`` (a queue of dicts).

    With no ``port`` the client connects to the running instance's published
    port (``discovery.find_api_instance``): the only one, or the one whose
    ``simulated`` flag matches. Several matching instances raise
    ``ApiInstanceNotFound`` rather than guessing.
    """

    def __init__(self, host: str = DEFAULT_API_HOST, port: Optional[int] = None, timeout: float = 5.0,
                 *, simulated: Optional[bool] = None, discovery_dir: Optional[Path] = None):
        if port is None:
            instance = find_api_instance(simulated=simulated, directory=discovery_dir)
            port = instance.port
            # An instance listening on every address is reached on loopback.
            if instance.host not in ("", "0.0.0.0", "::"):
                host = instance.host
        self.host, self.port = host, port
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.settimeout(None)
        self._timeout = timeout
        self._file = self._sock.makefile("rb")
        self._responses: Dict[Optional[str], "queue.Queue[Dict[str, Any]]"] = {}
        self._lock = threading.Lock()
        self.events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True, name="ApiClientReader")
        self._reader.start()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()

    def send_line(self, line: bytes):
        self._sock.sendall(line)

    def request(self, command: Union[str, int], data: Any = None, *, nonce: Optional[str] = None,
                custom_command: Optional[str] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        nonce = uuid.uuid4().hex if nonce is None else nonce
        message = {"type": "request", "nonce": nonce, "command": command,
                   "customCommand": custom_command, "data": data}
        self._queue_for(nonce)
        self.send_line((json.dumps(message) + "\n").encode("utf-8"))
        return self.wait_response(nonce, timeout=timeout)

    def wait_response(self, nonce: Optional[str], timeout: Optional[float] = None) -> Dict[str, Any]:
        return self._queue_for(nonce).get(timeout=self._timeout if timeout is None else timeout)

    def _queue_for(self, nonce: Optional[str]) -> "queue.Queue[Dict[str, Any]]":
        with self._lock:
            return self._responses.setdefault(nonce, queue.Queue())

    def _read_loop(self):
        for line in self._file:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("type") == "event":
                self.events.put(message)
            elif message.get("type") == "response":
                self._queue_for(message.get("nonce")).put(message)

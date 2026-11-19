"""The reachAQ local API: newline-delimited JSON over TCP.

Every connected client receives every published event. A client's request is
answered on its own connection; a command that finishes later
(``PENDING_WITH_NOTIFICATION``) sends its final response to the client that
asked, or to every client if that one has gone.
"""

import queue
import socket
import socketserver
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, Optional, Set

from ..logging import get_verbose_logger
from .commands import (
    ApiCommandRequest,
    ApiCommandRequestErrorKind,
    ApiCommandRequestResponse,
    ApiCommandRequestResult,
)
from .discovery import publish_instance, withdraw_instance
from .event import ApiEvent
from .options import ApiOptions
from .wire import MAX_LINE_BYTES, RequestDecodeError, decode_request, encode_event, encode_response

logger = get_verbose_logger(__name__)

CommandRequestDelegate = Callable[[ApiCommandRequest], ApiCommandRequestResponse]

# Lines queued for one client before it is dropped as too slow. Events are
# published from the event manager's worker thread, which must never block on
# a socket.
CLIENT_QUEUE_LINES = 4096


class _Client:
    def __init__(self, sock: socket.socket, address):
        self.sock = sock
        self.address = address
        self._lines: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=CLIENT_QUEUE_LINES)
        self.closed = threading.Event()
        self._writer = threading.Thread(target=self._write_loop, daemon=True, name=f"ApiClientWriter-{address}")
        self._writer.start()

    def send(self, line: bytes) -> bool:
        if self.closed.is_set():
            return False
        try:
            self._lines.put_nowait(line)
        except queue.Full:
            logger.warning("Local API client %s is not reading; disconnecting it", self.address)
            self.close()
            return False
        return True

    def _write_loop(self):
        while True:
            line = self._lines.get()
            if line is None:
                return
            try:
                self.sock.sendall(line)
            except OSError as err:
                logger.info("Local API client %s write failed: %s", self.address, err)
                self.close()
                return

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._lines.put_nowait(None)
        except queue.Full:
            pass


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    # Rebinding after a restart needs SO_REUSEADDR on Linux; on Windows the same
    # flag lets a second instance bind the port too, so it stays off there.
    allow_reuse_address = sys.platform != "win32"

    def __init__(self, address, service: "RpcService"):
        self.service = service
        super().__init__(address, _Handler)


class _Handler(socketserver.StreamRequestHandler):
    server: _Server

    def handle(self):
        service = self.server.service
        client = _Client(self.request, self.client_address)
        service._add_client(client)
        try:
            while not client.closed.is_set():
                line = self.rfile.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_LINE_BYTES:
                    client.send(encode_response(_error_response(
                        None, None, ApiCommandRequestResult.UNRECOGNIZED,
                        f"message longer than {MAX_LINE_BYTES} bytes")))
                    break
                if not line.strip():
                    continue
                service._handle_line(line, client)
        except OSError:
            pass
        finally:
            service._remove_client(client)
            client.close()


def _error_response(nonce, command, result, message,
                    error_code=ApiCommandRequestErrorKind.COMMAND_ERROR) -> ApiCommandRequestResponse:
    return ApiCommandRequestResponse(
        nonce=nonce, command=command, result=result, error_code=error_code, error_message=message)


class RpcService:
    def __init__(self, options: ApiOptions):
        self._options = options
        self._lock = threading.Lock()
        self._server: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None
        self._clients: Set[_Client] = set()
        # nonce -> [client, lines deferred while its request is still in the delegate]
        self._pending: Dict[str, list] = {}
        self._record: Optional[Path] = None
        self.command_request_delegate: Optional[CommandRequestDelegate] = None

    def __repr__(self):
        return f"RpcService({self._options.host}:{self.port})"

    @property
    def options(self) -> ApiOptions:
        return self._options

    @property
    def port(self) -> Optional[int]:
        """The bound port (useful with port 0), or None when not listening."""
        server = self._server
        return None if server is None else server.server_address[1]

    @property
    def is_running(self) -> bool:
        return self._server is not None

    def start(self) -> bool:
        """Start listening. A port that cannot be bound is logged and leaves the
        application running without the API."""
        with self._lock:
            if self._server is not None:
                return True
            try:
                server = _Server((self._options.host, self._options.port), self)
            except OSError as err:
                logger.error("Local API could not listen on %s:%s: %s",
                             self._options.host, self._options.port, err)
                return False
            self._server = server
            self._thread = threading.Thread(
                target=server.serve_forever, kwargs=dict(poll_interval=0.2), daemon=True, name="LocalApiServer")
            self._thread.start()
            port = server.server_address[1]
            if self._options.discovery_dir is not None:
                try:
                    self._record = publish_instance(
                        self._options.discovery_dir, self._options.host, port, self._options.simulated)
                except OSError as err:
                    logger.warning("Local API on port %s could not publish its record in %s: %s",
                                   port, self._options.discovery_dir, err)
        logger.info("Local API listening on %s:%s%s", self._options.host, port,
                    "" if self._record is None else f" (record {self._record})")
        return True

    def stop(self) -> None:
        with self._lock:
            server, self._server = self._server, None
            thread, self._thread = self._thread, None
            clients, self._clients = self._clients, set()
            self._pending.clear()
            record, self._record = self._record, None
        withdraw_instance(record)
        if server is None:
            return
        server.shutdown()
        server.server_close()
        for client in clients:
            client.close()
        if thread is not None:
            thread.join(timeout=2)
        logger.info("Local API stopped")

    def send_event(self, info: ApiEvent) -> None:
        with self._lock:
            clients = tuple(self._clients)
        if not clients:
            return
        try:
            line = encode_event(info)
        except Exception as err:
            logger.warning("Local API could not encode event %s: %s", info.kind, err)
            return
        for client in clients:
            client.send(line)

    def send_command_result(self, response: ApiCommandRequestResponse) -> None:
        line = encode_response(response)
        with self._lock:
            entry = self._pending.get(response.nonce) if response.nonce is not None else None
            if entry is not None and entry[1] is not None:
                # The request is still being handled: send after its first reply.
                entry[1].append(line)
                return
            if entry is not None:
                del self._pending[response.nonce]
            client = None if entry is None else entry[0]
            targets = (client,) if client is not None and not client.closed.is_set() else tuple(self._clients)
        for target in targets:
            target.send(line)

    def _add_client(self, client: _Client):
        with self._lock:
            self._clients.add(client)
        logger.info("Local API client connected: %s", client.address)

    def _remove_client(self, client: _Client):
        with self._lock:
            self._clients.discard(client)
            for nonce in [n for n, entry in self._pending.items() if entry[0] is client]:
                del self._pending[nonce]
        logger.info("Local API client disconnected: %s", client.address)

    def _handle_line(self, line: bytes, client: _Client) -> None:
        try:
            request = decode_request(line)
        except RequestDecodeError as err:
            client.send(encode_response(_error_response(
                err.nonce, err.command, ApiCommandRequestResult.UNRECOGNIZED, str(err))))
            return
        delegate = self.command_request_delegate
        if delegate is None:
            client.send(encode_response(_error_response(
                request.nonce, request.command, ApiCommandRequestResult.FAILED,
                "the application is not accepting commands", ApiCommandRequestErrorKind.SYSTEM_ERROR)))
            return
        nonce = request.nonce
        if nonce is not None:
            # Registered before the delegate runs: an asynchronous command may
            # finish before the delegate returns, and its final response must
            # still follow the first one.
            with self._lock:
                self._pending[nonce] = [client, []]
        try:
            response = delegate(request)
        except Exception as err:
            logger.exception("Local API command %s failed: %s", request.command, err)
            response = _error_response(
                nonce, request.command, ApiCommandRequestResult.EXCEPTION,
                f"{type(err).__name__}: {err}", ApiCommandRequestErrorKind.SYSTEM_ERROR)
        first = encode_response(response)
        if nonce is None:
            client.send(first)
            return
        # Queued under the lock (send never blocks), so a final response that
        # send_command_result produces meanwhile cannot overtake this one.
        with self._lock:
            client.send(first)
            entry = self._pending.get(nonce)
            if entry is not None and entry[0] is client:
                deferred, entry[1] = entry[1], None
                for line_out in deferred:
                    client.send(line_out)
                if deferred or response.result != ApiCommandRequestResult.PENDING_WITH_NOTIFICATION:
                    del self._pending[nonce]


def create_api_service(options: ApiOptions) -> Optional[RpcService]:
    """The service for ``options``, or None when the API is disabled."""
    if not options.enabled:
        logger.info("Local API disabled")
        return None
    return RpcService(options)

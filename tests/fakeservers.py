"""Tiny local HTTP servers used to prove what the service sends where, and how much it reads."""

from __future__ import annotations

import gzip
import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARTIFACTS = "/api/2.0/mlflow-artifacts/artifacts"


class Attacker:
    """A host that records every request it receives (headers included)."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _record(self) -> None:
                outer.requests.append({"path": self.path, "headers": dict(self.headers.items())})
                body = json.dumps({"files": []}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST = do_PUT = _record

            def log_message(self, *args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_port
        self.netloc = f"127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@contextmanager
def attacker() -> Iterator[Attacker]:
    server = Attacker()
    try:
        yield server
    finally:
        server.close()


class FakeTracking:
    """Fake MLflow artifact proxy: serves a listing and (possibly endless/hostile) bodies."""

    def __init__(
        self,
        listing: list[dict],
        bodies: dict[str, Callable[[BaseHTTPRequestHandler], None]],
        json_routes: dict[str, dict | tuple[int, dict]] | None = None,
    ) -> None:
        self.requests: list[dict] = []
        self.bytes_sent = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer.requests.append({"path": self.path, "headers": dict(self.headers.items())})
                route, _, _query = self.path.partition("?")
                if route == ARTIFACTS or route in (json_routes or {}):
                    status, payload = 200, {"files": listing}
                    if route != ARTIFACTS:
                        found = (json_routes or {})[route]
                        status, payload = found if isinstance(found, tuple) else (200, found)
                    body = json.dumps(payload).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                name = route.rsplit("/", 1)[-1]
                handler = bodies.get(name)
                if handler is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                try:
                    handler(self)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def count(self, handler: BaseHTTPRequestHandler, data: bytes) -> None:
        self.bytes_sent += len(data)
        handler.wfile.write(data)


def fixed_body(data: bytes) -> Callable[[BaseHTTPRequestHandler], None]:
    def respond(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    return respond


def gzip_bomb(decoded_bytes: int) -> Callable[[BaseHTTPRequestHandler], None]:
    payload = gzip.compress(b"\0" * decoded_bytes)

    def respond(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Encoding", "gzip")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    return respond


def redirect_to(location: str) -> Callable[[BaseHTTPRequestHandler], None]:
    def respond(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(302)
        handler.send_header("Location", location)
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    return respond

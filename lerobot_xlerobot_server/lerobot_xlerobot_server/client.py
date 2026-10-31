"""Minimal client for the XLeRobot arm server (no dependency on any harness).

Used by cli.py, and a reference for client implementations elsewhere: one REQ
socket per thread, a timeout on every request, and a fresh socket after a
timeout (a REQ socket that sent without receiving cannot send again).
"""
from __future__ import annotations

import json
from typing import Any

import zmq

from . import PROTOCOL_VERSION


class ServerTimeout(RuntimeError):
    pass


class ServerError(RuntimeError):
    def __init__(self, rep: dict[str, Any]) -> None:
        super().__init__(f"[{rep.get('code')}] {rep.get('error')}")
        self.rep = rep


class XLeRobotClient:
    def __init__(self, addr: str = "tcp://localhost:5560", timeout_s: float = 2.0) -> None:
        self.addr = addr
        self.timeout_ms = int(timeout_s * 1000)
        self.ctx = zmq.Context.instance()
        self.sock = None
        self._open()

    def _open(self) -> None:
        if self.sock is not None:
            self.sock.close(linger=0)
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.connect(self.addr)

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close(linger=0)
            self.sock = None

    def call_raw(self, cmd: str, **kw: Any) -> tuple[dict[str, Any], list[bytes]]:
        """-> (JSON header, binary attachments). Replies are multipart since 0.3."""
        try:
            self.sock.send_json({"cmd": cmd, **kw})
            parts = self.sock.recv_multipart()
        except zmq.Again as e:
            self._open()  # the old REQ socket is stuck in "awaiting reply"
            raise ServerTimeout(f"{cmd}: no reply from {self.addr} within {self.timeout_ms} ms") from e
        rep = json.loads(parts[0])
        if not rep.get("ok", False):
            raise ServerError(rep)
        return rep, parts[1:]

    def call(self, cmd: str, **kw: Any) -> dict[str, Any]:
        return self.call_raw(cmd, **kw)[0]

    def get_obs(self, **kw: Any) -> tuple[dict[str, Any], dict[str, bytes]]:
        """-> (header, {camera name: JPEG bytes})."""
        rep, blobs = self.call_raw("get_obs", **kw)
        names = [im["name"] for im in rep["images"]]
        if len(names) != len(blobs):
            raise RuntimeError(f"get_obs: {len(names)} images in header, {len(blobs)} frames")
        return rep, dict(zip(names, blobs))

    def check_protocol(self) -> dict[str, Any]:
        rep = self.call("ping")
        theirs = str(rep.get("protocol", "?"))
        if theirs.split(".")[0] != PROTOCOL_VERSION.split(".")[0]:
            raise RuntimeError(f"protocol mismatch: server {theirs}, client {PROTOCOL_VERSION}")
        return rep

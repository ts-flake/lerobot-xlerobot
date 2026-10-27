"""XLeVR server: WebSocket + HTTPS + per-arm latch.

Mirrors the browser's *absolute* poses to the teleop. Each VR frame carries
  - position_rb        : EE / head position in robot-base frame (m)
  - quaternion_rb      : R_rb_ee (controllers) or R_rb_head (headset), [x,y,z,w]
  - chest_quaternion_rb: R_rb_chest, [x,y,z,w]
  - analog inputs + source timestamp (browser clock, seconds)

The server does NOT compute deltas — the teleop differences consecutive
absolute poses at its own read rate, which is idempotent to frame-rate
mismatch (no dropped/duplicated motion). No calibration / no B-key handling;
B is consumed by the browser to toggle chest-yaw freeze.

Naming: R_a_b = "frame b in a coords".
"""

import asyncio
import http.server
import json
import logging
import os
import ssl
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Set

import numpy as np

from .config import XLeVRConfig
from .utils import parse_quat, parse_vec3
from .camera_stream import (
    CameraFrameStore,
    build_frame_payloads,
    camera_list_message,
    clamp_subscription,
)

logger = logging.getLogger("xlevr")


# ────────────────────────────  Data types  ────────────────────────────────

_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


@dataclass
class VRState:
    """Latest absolute pose + inputs for one controller/headset (one VR frame)."""
    arm: Literal["left", "right", "headset"]

    position_rb: Optional[np.ndarray] = None         # [x,y,z] m in rb
    quaternion_rb: Optional[np.ndarray] = None        # [x,y,z,w]  R_rb_ee / R_rb_head
    chest_quaternion_rb: np.ndarray = field(default_factory=lambda: _IDENTITY_QUAT.copy())

    trigger_value: float = 0.0
    trigger_active: bool = False
    thumbstick: np.ndarray = field(default_factory=lambda: np.zeros(2))
    buttons: Dict[str, bool] = field(default_factory=dict)

    timestamp: float = field(default_factory=time.time)  # browser clock, seconds


# ────────────────────────────  WS server  ────────────────────────────────

class VRWebSocketServer:
    """Parses raw VR payload and queues absolute VRState per controller."""

    def __init__(self, command_queue: asyncio.Queue, config: XLeVRConfig,
                 store: "CameraFrameStore", debug: bool = False):
        self.command_queue = command_queue
        self.config = config
        self.store = store
        self.debug = debug
        self.clients: Set = set()
        self.subscriptions: Dict = {}   # ws -> list[str] of subscribed camera names
        self.camera_list_dirty = False
        self.server = None
        self.is_running = False

    # ---- SSL ----

    def _setup_ssl(self) -> Optional[ssl.SSLContext]:
        if not self.config.ssl_files_exist and not self.config.ensure_ssl_certificates():
            logger.error("Failed to generate SSL certs")
            return None
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            ctx.load_cert_chain(certfile=self.config.certfile, keyfile=self.config.keyfile)
            return ctx
        except ssl.SSLError as e:
            logger.error(f"SSL load: {e}")
            return None

    # ---- Lifecycle ----

    async def start(self):
        import websockets
        if not self.config.enable_vr:
            return
        ctx = self._setup_ssl()
        if ctx is None:
            return
        try:
            self.server = await websockets.serve(
                self._handle, self.config.host_ip, self.config.websocket_port, ssl=ctx,
            )
            self.is_running = True
            logger.info(f"WS on wss://{self.config.host_ip}:{self.config.websocket_port}")
        except Exception as e:
            logger.error(f"WS start failed: {e}")

    async def stop(self):
        self.is_running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def _handle(self, ws, path=None):
        import websockets
        addr = ws.remote_address
        logger.info(f"Client {addr} connected")
        self.clients.add(ws)
        self.subscriptions[ws] = []
        await self._send_config(ws)
        await self._send_camera_list(ws)
        try:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                    await self._process(data, ws)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.error(f"Processing: {e}", exc_info=True)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            self.subscriptions.pop(ws, None)
            logger.info(f"Client {addr} disconnected")

    async def _send_config(self, ws):
        try:
            await ws.send(json.dumps({
                "type": "xlevr_config",
                "grip_rake_deg": self.config.grip_rake_deg,
                "grip_yaw_deg":  self.config.grip_yaw_deg,
            }))
        except Exception:
            pass

    async def _send_camera_list(self, ws):
        try:
            await ws.send(json.dumps(camera_list_message(self.store.names())))
        except Exception:
            pass

    # ---- Processing ----

    async def _process(self, data: Dict, ws=None):
        if data.get("type") == "camera_subscribe":
            requested = data.get("cameras") or []
            self.subscriptions[ws] = clamp_subscription(
                requested, self.store.names(), self.config.camera_max_feeds
            )
            return

        cq = data.get("chestQuaternion")
        q_rb_chest = (np.array([cq["x"], cq["y"], cq["z"], cq["w"]], float)
                      if cq else _IDENTITY_QUAT.copy())
        ts_ms = data.get("timestamp")
        ts_s = (float(ts_ms) / 1000.0) if ts_ms is not None else time.time()

        for key, hand in (("leftController", "left"),
                          ("rightController", "right"),
                          ("headset", "headset")):
            payload = data.get(key)
            if payload:
                await self._process_one(hand, payload, q_rb_chest, ts_s)

    async def _process_one(self, hand: str, d: Dict, q_rb_chest: np.ndarray, ts_s: float):
        pos  = parse_vec3(d.get("position", {}))
        quat = parse_quat(d.get("quaternion", {}))
        if pos is None or quat is None:
            return

        trigger_value = float(d.get("trigger", 0))
        ts_raw = d.get("thumbstick") or {}
        thumbstick = np.array([float(ts_raw.get("x", 0.0)), float(ts_raw.get("y", 0.0))])
        buttons: Dict[str, bool] = d.get("buttons") or {}

        state = VRState(
            arm=hand,
            position_rb=pos,
            quaternion_rb=quat,
            chest_quaternion_rb=q_rb_chest,
            trigger_value=trigger_value,
            trigger_active=trigger_value > 0.5,
            thumbstick=thumbstick,
            buttons=dict(buttons),
            timestamp=ts_s,
        )
        try:
            await self.command_queue.put(state)
        except Exception:
            pass

    async def camera_broadcast_loop(self):
        """Encode subscribed cameras and push frames to each client at stream fps."""
        period = 1.0 / max(1, self.config.camera_stream_fps)
        while self.is_running:
            t0 = time.time()

            # Re-advertise the camera list when the available set changes.
            if self.camera_list_dirty:
                self.camera_list_dirty = False
                msg = json.dumps(camera_list_message(self.store.names()))
                await asyncio.gather(
                    *(c.send(msg) for c in list(self.clients)), return_exceptions=True
                )

            union = set().union(*self.subscriptions.values()) if self.subscriptions else set()
            if union:
                payloads = build_frame_payloads(
                    self.store, union,
                    quality=self.config.camera_jpeg_quality,
                    max_width=self.config.camera_max_width,
                )
                sends = []
                for ws, subs in list(self.subscriptions.items()):
                    for name in subs:
                        if name in payloads:
                            sends.append(ws.send(json.dumps(payloads[name])))
                if sends:
                    # Concurrent + best-effort: a slow/closed client can't stall others.
                    await asyncio.gather(*sends, return_exceptions=True)

            sleep = period - (time.time() - t0)
            if sleep > 0:
                await asyncio.sleep(sleep)


# ────────────────────────────  HTTPS server  ──────────────────────────────

# web-ui/ lives next to xlevr/, one level up from this file.
_WEB_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "web-ui",
)


class _HTTPHandler(http.server.BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        try:
            super().end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, ssl.SSLError):
            pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args, **kwargs):
        pass

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self._serve('index.html', 'text/html')
        elif self.path.endswith('.css'):
            self._serve(self.path.lstrip('/'), 'text/css')
        elif self.path.endswith('.js'):
            self._serve(self.path.lstrip('/'), 'application/javascript')
        elif self.path.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            t = ('image/jpeg' if self.path.endswith(('.jpg', '.jpeg'))
                 else 'image/png' if self.path.endswith('.png') else 'image/gif')
            self._serve(self.path.lstrip('/'), t)
        else:
            self.send_error(404, "Not found")

    def _serve(self, rel: str, content_type: str):
        try:
            path = os.path.join(_WEB_ROOT, rel)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    body = f.read()
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404, f"Not found: {rel}")
        except Exception as e:
            logger.error(f"serve {rel}: {e}")
            self.send_error(500, "Server error")


class HTTPSServer:
    def __init__(self, config: XLeVRConfig):
        self.config = config
        self.httpd = None
        self.thread = None

    async def start(self):
        try:
            self.httpd = http.server.HTTPServer(
                (self.config.host_ip, self.config.https_port), _HTTPHandler,
            )
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(self.config.certfile, self.config.keyfile)
            self.httpd.socket = ctx.wrap_socket(self.httpd.socket, server_side=True)
            self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.thread.start()
            logger.info(f"HTTPS on {self.config.host_ip}:{self.config.https_port}")
        except Exception as e:
            logger.error(f"HTTPS start: {e}")
            raise

    async def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            if self.thread:
                self.thread.join(timeout=5)


# ────────────────────────────  Monitor  ───────────────────────────────────

def _local_ip() -> str:
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "localhost"


class VRMonitor:
    """Owns the WS + HTTPS servers and a per-arm latch of the latest VRState."""

    def __init__(self, debug: bool = False):
        self.config: Optional[XLeVRConfig] = None
        self.ws_server: Optional[VRWebSocketServer] = None
        self.https_server: Optional[HTTPSServer] = None
        self.queue: Optional[asyncio.Queue] = None
        self.is_running = False
        self.debug = debug

        self._lock = threading.Lock()
        self.left_state: Optional[VRState] = None
        self.right_state: Optional[VRState] = None
        self.headset_state: Optional[VRState] = None
        self.camera_store = CameraFrameStore()

    # ---- Lifecycle ----

    async def start_monitoring(self):
        self.config = XLeVRConfig()
        self.config.enable_vr = True
        self.queue = asyncio.Queue()

        self.ws_server    = VRWebSocketServer(self.queue, self.config, self.camera_store, debug=self.debug)
        self.https_server = HTTPSServer(self.config)

        # Chdir so the SSL cert paths in XLeVRConfig (relative) resolve.
        os.chdir(os.path.dirname(_WEB_ROOT))

        try:
            await self.https_server.start()
            await self.ws_server.start()
            self.is_running = True
            self.ws_server.is_running = True
            asyncio.create_task(self.ws_server.camera_broadcast_loop())
            host = _local_ip() if self.config.host_ip == "0.0.0.0" else self.config.host_ip
            logger.info(f"📱 Open https://{host}:{self.config.https_port} in your VR headset.")
            await self._drain()
        except KeyboardInterrupt:
            logger.info("Stopping VR monitor")
        except Exception as e:
            logger.error(f"Monitor: {e}", exc_info=True)
        finally:
            await self._stop()

    async def _drain(self):
        while self.is_running:
            try:
                st = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                with self._lock:
                    if   st.arm == "left":    self.left_state    = st
                    elif st.arm == "right":   self.right_state   = st
                    elif st.arm == "headset": self.headset_state = st
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Drain: {e}", exc_info=True)

    async def _stop(self):
        self.is_running = False
        if self.ws_server:
            self.ws_server.is_running = False
            await self.ws_server.stop()
        if self.https_server: await self.https_server.stop()

    def update_camera_frames(self, frames: Dict[str, np.ndarray]) -> None:
        """Store the latest camera frames (called from the control-loop thread)."""
        changed = self.camera_store.update(frames)
        if changed and self.ws_server is not None:
            self.ws_server.camera_list_dirty = True

    # ---- Poll API ----

    def get_latest_goal_nowait(self, arm: Optional[str] = None):
        with self._lock:
            if arm == "left":    return self.left_state
            if arm == "right":   return self.right_state
            if arm == "headset": return self.headset_state
            return {
                "left":        self.left_state,
                "right":       self.right_state,
                "headset":     self.headset_state,
                "has_left":    self.left_state    is not None,
                "has_right":   self.right_state   is not None,
                "has_headset": self.headset_state is not None,
            }

    def get_left_goal_nowait(self):    return self.get_latest_goal_nowait("left")
    def get_right_goal_nowait(self):   return self.get_latest_goal_nowait("right")

"""Camera streaming core for XLeVR: frame selection, encoding, store, payloads.

Deliberately free of `websockets`/`asyncio` so it is unit-testable with only
numpy + opencv. The asyncio/websocket glue lives in `server.py`.
"""

import base64
import threading
import time
from typing import Dict, Iterable, List

import cv2
import numpy as np


def select_image_frames(obs: Dict) -> Dict[str, np.ndarray]:
    """Pick the camera-image entries (HWC, ndim==3 ndarrays) from an observation."""
    frames: Dict[str, np.ndarray] = {}
    for key, val in obs.items():
        if isinstance(val, np.ndarray) and val.ndim == 3:
            frames[key] = val
    return frames


def clamp_subscription(requested: Iterable[str], known: Iterable[str], max_feeds: int) -> List[str]:
    """Keep requested cameras that exist, de-duplicated, order-preserving, capped."""
    known_set = set(known)
    out: List[str] = []
    for name in requested:
        if name in known_set and name not in out:
            out.append(name)
        if len(out) >= max_feeds:
            break
    return out


def encode_frame_jpeg(frame: np.ndarray, quality: int = 70, max_width: int = 640) -> str:
    """Downscale (if wider than max_width) and JPEG-encode an RGB frame to base64.

    Robot frames are RGB; OpenCV expects BGR to write correct JPEG colors, so we
    convert before encoding. The browser decodes the JPEG back to RGB for display.
    """
    img = frame
    h, w = img.shape[:2]
    if max_width and w > max_width:
        scale = max_width / float(w)
        img = cv2.resize(img, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf).decode("ascii")


def camera_list_message(names: Iterable[str]) -> Dict:
    return {"type": "camera_list", "cameras": list(names)}


def camera_frame_message(name: str, data_b64: str, ts: float) -> Dict:
    return {"type": "camera_frame", "name": name, "format": "jpeg", "data": data_b64, "ts": ts}


class CameraFrameStore:
    """Thread-safe latest-frame-per-camera store with name-set change detection."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: Dict[str, np.ndarray] = {}
        self._names: List[str] = []

    def update(self, frames: Dict[str, np.ndarray]) -> bool:
        """Replace the current frames. Returns True if the camera name-set changed."""
        with self._lock:
            self._frames = {k: v for k, v in frames.items()}
            new_names = sorted(self._frames.keys())
            changed = new_names != self._names
            self._names = new_names
            return changed

    def names(self) -> List[str]:
        with self._lock:
            return list(self._names)

    def snapshot(self, names: Iterable[str]) -> Dict[str, np.ndarray]:
        with self._lock:
            return {n: self._frames[n] for n in names if n in self._frames}


def build_frame_payloads(
    store: CameraFrameStore,
    subscribed: Iterable[str],
    quality: int = 70,
    max_width: int = 640,
    ts: float | None = None,
) -> Dict[str, Dict]:
    """Encode the subscribed-and-available cameras into camera_frame messages."""
    stamp = time.time() if ts is None else ts
    payloads: Dict[str, Dict] = {}
    for name, frame in store.snapshot(subscribed).items():
        b64 = encode_frame_jpeg(frame, quality=quality, max_width=max_width)
        payloads[name] = camera_frame_message(name, b64, stamp)
    return payloads

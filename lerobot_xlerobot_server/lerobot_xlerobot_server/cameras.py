"""Cameras, owned by the server independently of XLeRobotX.

Built with lerobot's own camera classes (``make_cameras_from_configs``) but kept out of
``XLeRobotXConfig.cameras``: a camera fault then never touches the motor buses, and the
fake mode can use synthetic cameras. Each camera runs lerobot's background read thread;
``grab`` returns its latest frame (age <= about one frame period).

Config (server.yaml), one entry per camera name:

    cameras:
      head:
        path: /dev/v4l/by-path/...-video-index0   # or an integer index
        width: 640
        height: 480
        fps: 30
        fourcc: MJPG       # optional
        rotation: 0        # optional: 0, 90, 180, -90
"""
from __future__ import annotations

import glob
import logging
import os
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CAM_DEFAULTS = {"width": 640, "height": 480, "fps": 30, "fourcc": "MJPG", "rotation": 0}
CAM_KEYS = {"path"} | set(CAM_DEFAULTS)


def normalize_cam_cfgs(cfgs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for name, c in (cfgs or {}).items():
        c = dict(c or {})
        unknown = set(c) - CAM_KEYS
        if unknown:
            raise KeyError(f"cameras.{name}: unknown keys {sorted(unknown)}")
        if "path" not in c:
            raise KeyError(f"cameras.{name}: 'path' is required")
        out[name] = {**CAM_DEFAULTS, **c}
    return out


def encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[..., ::-1]),
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def list_v4l_by_path() -> list[tuple[str, str]]:
    """[(by-path link, /dev/videoN)] for capture nodes (index0)."""
    links = sorted(glob.glob("/dev/v4l/by-path/*-video-index0"))
    return [(p, os.path.realpath(p)) for p in links]


class CameraHub:
    """grab(name) -> (rgb uint8 HxWx3, age_ms)."""

    def __init__(self, cam_cfgs: dict[str, dict[str, Any]], timeout_ms: float = 500.0) -> None:
        self.cfgs = normalize_cam_cfgs(cam_cfgs)
        self.timeout_ms = timeout_ms
        self.cams: dict[str, Any] = {}

    @property
    def names(self) -> list[str]:
        return list(self.cfgs)

    def connect(self) -> None:
        if not self.cfgs:
            return
        from lerobot.cameras import make_cameras_from_configs
        from lerobot.cameras.opencv.configuration_opencv import Cv2Rotation, OpenCVCameraConfig

        lr_cfgs = {}
        for name, c in self.cfgs.items():
            path = c["path"]
            lr_cfgs[name] = OpenCVCameraConfig(
                index_or_path=int(path) if str(path).isdigit() else path,
                width=c["width"], height=c["height"], fps=c["fps"],
                fourcc=c["fourcc"], rotation=Cv2Rotation(int(c["rotation"])),
            )
        cams = make_cameras_from_configs(lr_cfgs)
        try:
            for name, cam in cams.items():
                logger.info("connecting camera %s (%s)", name, self.cfgs[name]["path"])
                cam.connect()
                self.cams[name] = cam
        except Exception as e:
            self.disconnect()
            raise RuntimeError(f"camera {name!r} ({self.cfgs[name]['path']}) failed: {e}") from e

    def grab(self, name: str) -> tuple[np.ndarray, float]:
        cam = self.cams[name]
        frame = cam.async_read(timeout_ms=self.timeout_ms)
        ts = getattr(cam, "latest_timestamp", None)  # perf_counter at capture (lerobot >= 0.4)
        age_ms = (time.perf_counter() - ts) * 1e3 if ts is not None else float("nan")
        return frame, age_ms

    def disconnect(self) -> None:
        for name, cam in list(self.cams.items()):
            try:
                cam.disconnect()
            except Exception:
                logger.exception("camera %s disconnect failed", name)
        self.cams = {}


class FakeCameraHub(CameraHub):
    """Synthetic frames: a gradient with a square that moves on every grab."""

    def __init__(self, cam_cfgs: dict[str, dict[str, Any]] | None = None) -> None:
        cfgs = cam_cfgs or {n: {"path": f"fake:{n}"} for n in ("head", "left_wrist", "right_wrist")}
        super().__init__(cfgs)
        self.count = 0

    def connect(self) -> None:
        self.cams = {n: None for n in self.cfgs}

    def grab(self, name: str) -> tuple[np.ndarray, float]:
        c = self.cfgs[name]
        h, w = c["height"], c["width"]
        img = np.zeros((h, w, 3), np.uint8)
        img[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        img[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        img[..., 2] = 40 * (self.names.index(name) + 1)
        self.count += 1
        x = (self.count * 7) % max(1, w - 40)
        img[h // 2 - 20:h // 2 + 20, x:x + 40] = 255
        return img, 0.0

    def disconnect(self) -> None:
        self.cams = {}

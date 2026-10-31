"""Per-arm motion: a waypoint queue drained by one thread per arm.

Each arm lives on its own bus (left: bus1, right: bus2), so the two threads never
block each other; within a bus the RobotIO lock serialises them with REP-thread reads.
A command REPLACES whatever is still queued for that arm and returns at once, which
keeps the single-threaded REP server from serialising the two arms.

Interpolation starts from the last COMMANDED joints, not the measured ones: with low
servo gains the measured pose sags under gravity, and restarting from it on every
command would make each step jump by the sag. ``resync`` (torque on, hold) resets the
commanded joints to the measured ones.

No watchdog: commands are short bounded trajectories; if the client vanishes the
thread drains what is queued (<= a few hundred ms) and the arm holds there.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from .robot_io import RobotIO, arm_motors, min_jerk

logger = logging.getLogger(__name__)


def linear_waypoints(q0: np.ndarray, q1: np.ndarray, dq_max_deg: float, n_max: int) -> list[np.ndarray]:
    """Intermediate points + the goal. N intermediates = ceil(max|dq| / dq_max) - 1, in [0, n_max]."""
    span = float(np.max(np.abs(q1 - q0))) if len(q0) else 0.0
    n_mid = min(max(int(math.ceil(span / dq_max_deg)) - 1, 0), n_max)
    return [q0 + (q1 - q0) * (k / (n_mid + 1)) for k in range(1, n_mid + 2)]


def min_jerk_waypoints(q0: np.ndarray, q1: np.ndarray, duration_s: float, hz: float) -> list[np.ndarray]:
    n = max(1, int(round(duration_s * hz)))
    return [q0 + (q1 - q0) * min_jerk(k / n) for k in range(1, n + 1)]


class ArmMotion:
    def __init__(self, side: str, io: RobotIO, hz: float = 30.0) -> None:
        self.side = side
        self.io = io
        self.hz = hz
        self.motors = arm_motors(side)[:6]
        self._queue: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.q_cmd: np.ndarray = np.zeros(6)
        self.last_error: Optional[str] = None
        self.resync()

    # -- control ------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"motion-{self.side}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def resync(self) -> None:
        """Drop queued motion; commanded joints := measured."""
        with self._lock:
            self._queue.clear()
            self.q_cmd = self.io.read_arm(self.side)[0]

    def command(self, waypoints: list[np.ndarray]) -> None:
        """Replace the queue. Waypoints are 6-vectors in degrees."""
        with self._lock:
            self._queue.clear()
            self._queue.extend(np.asarray(w, float) for w in waypoints)
        self._wake.set()

    @property
    def moving(self) -> bool:
        with self._lock:
            return bool(self._queue)

    def wait_idle(self, timeout_s: float = 5.0) -> bool:
        t_end = time.monotonic() + timeout_s
        while self.moving:
            if time.monotonic() > t_end:
                return False
            time.sleep(0.005)
        return True

    # -- thread -------------------------------------------------------------
    def _run(self) -> None:
        dt = 1.0 / self.hz
        while not self._stop.is_set():
            with self._lock:
                q = self._queue[0] if self._queue else None
            if q is None:
                self._wake.wait(0.1)
                self._wake.clear()
                continue
            t0 = time.perf_counter()
            try:
                self.io._write_goals(dict(zip(self.motors, q)))
                self.q_cmd = q
                self.last_error = None
            except Exception as e:  # keep the thread alive; drop the rest of the move
                logger.exception("%s arm write failed", self.side)
                self.last_error = f"{type(e).__name__}: {e}"
                with self._lock:
                    self._queue.clear()
                continue
            with self._lock:
                if self._queue and self._queue[0] is q:
                    self._queue.popleft()
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))

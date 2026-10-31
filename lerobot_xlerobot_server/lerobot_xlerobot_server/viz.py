"""meshcat view of both arms (measured joints) and their current targets.

Display only: the arm bases are drawn at y = +-arm_separation_m / 2 (left at +y), base
frames parallel, as mounted on XLeRobot X (config ``viz.arm_separation_m``, default
0.3 m). The server itself never relates the two arms to each other.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from . import ARMS
from .kinematics import URDF_JOINTS

logger = logging.getLogger(__name__)


class Viz:
    def __init__(self, server, hz: float = 10.0) -> None:
        import placo
        from placo_utils.visualization import robot_viz

        self.server = server
        self.hz = hz
        urdf = server.cfg["kinematics"]["urdf"]
        self.offset = {}
        self.robots, self.viz = {}, {}
        half = server.cfg["viz"]["arm_separation_m"] / 2.0
        for side, y in zip(ARMS, (half, -half)):
            T = np.eye(4)
            T[1, 3] = y
            self.offset[side] = T
            r = placo.RobotWrapper(urdf)
            r.set_T_world_fbase(T)
            self.robots[side] = r
            self.viz[side] = robot_viz(r, f"xlerobot_{side}")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="viz", daemon=True)

    def start(self) -> None:
        self._thread.start()
        logger.info("meshcat viz running (arm bases %.2f m apart, left at +y)",
                    self.server.cfg["viz"]["arm_separation_m"])

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        from placo_utils.visualization import frame_viz

        while not self._stop.wait(1.0 / self.hz):
            try:
                for side in ARMS:
                    q, _ = self.server.io.read_arm(side)
                    r = self.robots[side]
                    for name, v in zip(URDF_JOINTS, np.deg2rad(q)):
                        r.set_joint(name, v)
                    r.update_kinematics()
                    self.viz[side].display(r.state.q)
                    T = self.server.target_T.get(side)
                    if T is not None:
                        frame_viz(f"target_{side}", self.offset[side] @ T, scale=0.5)
            except Exception:
                logger.exception("viz update failed")

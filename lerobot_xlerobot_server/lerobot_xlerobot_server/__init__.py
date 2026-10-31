"""XLeRobot X arm server: owns the motor buses (and later the cameras) and serves
end-effector state / commands over ZMQ.

Independent of any client: it depends only on lerobot, lerobot_robot_xlerobot,
placo and pyzmq. The only contract with clients (e.g. Show-Harness) is the
protocol in PROTOCOL.md, versioned by PROTOCOL_VERSION.
"""

# MAJOR.MINOR. A client must refuse a server whose MAJOR differs from its own.
# MINOR grows when requests or reply fields are ADDED (old clients keep working).
PROTOCOL_VERSION = "0.5"

ARMS = ("left", "right")

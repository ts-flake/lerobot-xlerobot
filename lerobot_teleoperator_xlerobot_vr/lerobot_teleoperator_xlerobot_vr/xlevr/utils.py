"""SSL certificate generation + small parse helpers."""

import os
import subprocess
import stat
import logging
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


# ---------- SSL ----------

def generate_ssl_certificates(cert_path: str, key_path: str) -> bool:
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", key_path, "-out", cert_path,
                "-days", "365", "-nodes",
                "-subj", "/CN=localhost",
            ],
            check=True,
            capture_output=True,
        )
        os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)
        os.chmod(cert_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        logger.info(f"SSL certificates generated: {cert_path}, {key_path}")
        return True
    except Exception as e:
        logger.error(f"SSL generation failed: {e}")
        return False


def ensure_ssl_certificates(cert_path: str, key_path: str) -> bool:
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return True
    return generate_ssl_certificates(cert_path, key_path)


# ---------- Parsing helpers ----------

def parse_vec3(d: Dict, keys: tuple = ("x", "y", "z")) -> Optional[np.ndarray]:
    if d and all(k in d for k in keys):
        return np.array([d[k] for k in keys], dtype=float)
    return None


def parse_quat(d: Dict) -> Optional[np.ndarray]:
    """Return [x, y, z, w] quaternion (scipy/scipy scalar-last convention)."""
    return parse_vec3(d, ("x", "y", "z", "w"))


def euler_to_quat(euler_deg: np.ndarray) -> np.ndarray:
    """Convert xyz-extrinsic Euler angles [deg] to quaternion [x, y, z, w]."""
    return R.from_euler("xyz", euler_deg, degrees=True).as_quat(scalar_first=False)

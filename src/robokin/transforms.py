"""Rotations and homogeneous transforms.

Conventions used throughout the package:

* Rotation matrices act on column vectors, ``v_0 = R_01 @ v_1``.
* Homogeneous transforms are 4x4 arrays ``[[R, p], [0, 0, 0, 1]]``.
* Quaternions are unit quaternions stored scalar-first, ``q = (w, x, y, z)``,
  and are returned with ``w >= 0`` so that each rotation has one representative.
* RPY angles are ZYX Euler angles: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
  Read right to left this is roll about the fixed X axis, then pitch about the
  fixed Y axis, then yaw about the fixed Z axis.
* A rotation vector ("axis-angle") is ``angle * unit_axis`` with angle in [0, pi].
"""

from __future__ import annotations

import numpy as np

# Below this angle the Rodrigues formula switches to its first-order expansion.
_SMALL_ANGLE = 1e-12
# cos(pitch) below this value is treated as the RPY gimbal-lock case.
_GIMBAL_EPS = 1e-10


def rotx(angle: float) -> np.ndarray:
    """Rotation about the x axis by ``angle`` radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def roty(angle: float) -> np.ndarray:
    """Rotation about the y axis by ``angle`` radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotz(angle: float) -> np.ndarray:
    """Rotation about the z axis by ``angle`` radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix with ``skew(a) @ b == cross(a, b)``."""
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def homogeneous(R: np.ndarray | None = None, p: np.ndarray | None = None) -> np.ndarray:
    """Build a 4x4 transform from a rotation matrix and a translation vector."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def transl(x: float, y: float, z: float) -> np.ndarray:
    """Pure translation as a 4x4 transform."""
    return homogeneous(p=np.array([x, y, z], dtype=float))


def inv_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform using ``R^T`` instead of a general inverse."""
    R = T[:3, :3]
    p = T[:3, 3]
    return homogeneous(R.T, -R.T @ p)


def is_rotation(R: np.ndarray, tol: float = 1e-9) -> bool:
    """True if ``R`` is orthonormal with determinant +1 (a proper rotation)."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return False
    return bool(np.allclose(R.T @ R, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1.0) < tol)


# --------------------------------------------------------------------------- RPY


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix from ZYX Euler angles, ``Rz(yaw) Ry(pitch) Rx(roll)``."""
    return rotz(yaw) @ roty(pitch) @ rotx(roll)


def rot_to_rpy(R: np.ndarray) -> np.ndarray:
    """ZYX Euler angles ``(roll, pitch, yaw)`` with pitch in [-pi/2, pi/2].

    At pitch = +-pi/2 (gimbal lock) only ``yaw - roll`` (pitch = +pi/2) or
    ``yaw + roll`` (pitch = -pi/2) is determined; roll is then set to zero so the
    result is still an exact decomposition of ``R``.
    """
    R = np.asarray(R, dtype=float)
    cos_pitch = np.hypot(R[0, 0], R[1, 0])
    pitch = np.arctan2(-R[2, 0], cos_pitch)
    if cos_pitch > _GIMBAL_EPS:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # With roll = 0 the first two rows reduce to Rz(yaw) acting on the y axis
        # for both signs of pitch, so yaw can be read from R[0,1] and R[1,1].
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    return np.array([roll, pitch, yaw])


# ------------------------------------------------------------------- axis-angle


def rot_exp(rotvec: np.ndarray) -> np.ndarray:
    """Rotation matrix from a rotation vector (Rodrigues' formula)."""
    w = np.asarray(rotvec, dtype=float)
    angle = np.linalg.norm(w)
    if angle < _SMALL_ANGLE:
        return np.eye(3) + skew(w)
    K = skew(w / angle)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def rot_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector of ``R``; the inverse of :func:`rot_exp`.

    Computed through the quaternion so it stays accurate near 0 and near pi,
    where the textbook ``arccos((trace - 1) / 2)`` formula loses precision.
    """
    q = rot_to_quat(R)
    w, v = q[0], q[1:]
    s = np.linalg.norm(v)
    if s < _SMALL_ANGLE:
        return 2.0 * v / w
    return (2.0 * np.arctan2(s, w) / s) * v


def axis_angle_to_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation by ``angle`` about ``axis`` (need not be normalised)."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n == 0.0:
        raise ValueError("rotation axis must be non-zero")
    return rot_exp(axis / n * angle)


def rot_to_axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit axis and angle in [0, pi]; the axis is (1, 0, 0) for the identity."""
    w = rot_log(R)
    angle = float(np.linalg.norm(w))
    if angle < _SMALL_ANGLE:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return w / angle, angle


def orientation_error(R_goal: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotation vector ``e`` (base frame) with ``rot_exp(e) @ R == R_goal``.

    This is the orientation part of the pose error used by the numeric IK; it
    lives in the same frame as the angular rows of the geometric Jacobian.
    """
    return rot_log(R_goal @ R.T)


# ------------------------------------------------------------------ quaternions


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Unit quaternion with non-negative scalar part."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0.0 else q


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a (not necessarily normalised) quaternion (w, x, y, z)."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Quaternion of a rotation matrix using Shepperd's method.

    The branch is chosen by the largest of (trace, R00, R11, R22) so the square
    root argument is never close to zero, which keeps the result well
    conditioned for every rotation including 180 degree turns.
    """
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    diag = np.diag(R)
    k = int(np.argmax(diag))
    if tr >= diag[k]:
        w = 0.5 * np.sqrt(1.0 + tr)
        f = 0.25 / w
        q = np.array([w, (R[2, 1] - R[1, 2]) * f, (R[0, 2] - R[2, 0]) * f, (R[1, 0] - R[0, 1]) * f])
    elif k == 0:
        x = 0.5 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        f = 0.25 / x
        q = np.array([(R[2, 1] - R[1, 2]) * f, x, (R[0, 1] + R[1, 0]) * f, (R[0, 2] + R[2, 0]) * f])
    elif k == 1:
        y = 0.5 * np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2])
        f = 0.25 / y
        q = np.array([(R[0, 2] - R[2, 0]) * f, (R[0, 1] + R[1, 0]) * f, y, (R[1, 2] + R[2, 1]) * f])
    else:
        z = 0.5 * np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2])
        f = 0.25 / z
        q = np.array([(R[1, 0] - R[0, 1]) * f, (R[0, 2] + R[2, 0]) * f, (R[1, 2] + R[2, 1]) * f, z])
    return quat_normalize(q)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product; ``quat_to_rot(q1 * q2) == quat_to_rot(q1) @ quat_to_rot(q2)``."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate, which is the inverse for unit quaternions."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, s: float | np.ndarray) -> np.ndarray:
    """Spherical linear interpolation along the shorter great-circle arc.

    Returns shape (4,) for scalar ``s`` and (len(s), 4) for an array. The
    rotation angle between the result and ``q0`` grows linearly in ``s``,
    which is what gives LIN motions a constant angular velocity.
    """
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        # q and -q are the same rotation; flipping picks the short way round.
        q1, dot = -q1, -dot
    s_arr = np.atleast_1d(np.asarray(s, dtype=float))[:, None]
    if dot > 1.0 - 1e-12:
        out = q0 + s_arr * (q1 - q0)
        out /= np.linalg.norm(out, axis=1, keepdims=True)
    else:
        theta = np.arccos(dot)
        out = (np.sin((1.0 - s_arr) * theta) * q0 + np.sin(s_arr * theta) * q1) / np.sin(theta)
    return out[0] if np.ndim(s) == 0 else out


def interpolate_pose(T0: np.ndarray, T1: np.ndarray, s: float) -> np.ndarray:
    """Pose at fraction ``s`` of a straight-line move with SLERP orientation."""
    p = (1.0 - s) * T0[:3, 3] + s * T1[:3, 3]
    q = quat_slerp(rot_to_quat(T0[:3, :3]), rot_to_quat(T1[:3, :3]), s)
    return homogeneous(quat_to_rot(q), p)


def wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to the interval (-pi, pi]."""
    wrapped = np.pi - np.mod(np.pi - np.asarray(a, dtype=float), 2.0 * np.pi)
    return float(wrapped) if np.ndim(wrapped) == 0 else wrapped

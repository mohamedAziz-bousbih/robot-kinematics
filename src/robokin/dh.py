"""Standard Denavit-Hartenberg links and serial kinematic chains.

A standard DH link ``i`` describes the transform from frame ``i-1`` to frame
``i`` as

    A_i = Rz(theta_i) * Tz(d_i) * Tx(a_i) * Rx(alpha_i)

For a revolute joint the joint variable is added to ``theta``; for a prismatic
joint it is added to ``d``. Joint ``i`` therefore moves about or along the
z axis of frame ``i-1``, which is what the geometric Jacobian relies on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

import numpy as np

from .transforms import inv_transform


@dataclass(frozen=True)
class DHLink:
    """One row of a standard DH table.

    ``theta`` and ``d`` are the constant offsets; the joint variable is added
    to whichever of them the joint type makes variable. ``qlim`` is in radians
    for revolute joints and metres for prismatic ones.
    """

    a: float
    alpha: float
    d: float = 0.0
    theta: float = 0.0
    prismatic: bool = False
    qlim: tuple[float, float] = (-np.pi, np.pi)

    def __post_init__(self) -> None:
        if not self.qlim[0] < self.qlim[1]:
            raise ValueError(f"invalid joint limits {self.qlim}")

    def transform(self, q: float | np.ndarray) -> np.ndarray:
        """Link transform ``A_i(q)``; shape (4, 4) or (N, 4, 4) for q of shape (N,)."""
        q = np.asarray(q, dtype=float)
        if self.prismatic:
            theta = np.full_like(q, self.theta)
            d = self.d + q
        else:
            theta = self.theta + q
            d = np.full_like(q, self.d)
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(self.alpha), np.sin(self.alpha)
        A = np.zeros(q.shape + (4, 4))
        A[..., 0, 0] = ct
        A[..., 0, 1] = -st * ca
        A[..., 0, 2] = st * sa
        A[..., 0, 3] = self.a * ct
        A[..., 1, 0] = st
        A[..., 1, 1] = ct * ca
        A[..., 1, 2] = -ct * sa
        A[..., 1, 3] = self.a * st
        A[..., 2, 1] = sa
        A[..., 2, 2] = ca
        A[..., 2, 3] = d
        A[..., 3, 3] = 1.0
        return A


class IKSolution(NamedTuple):
    """One inverse-kinematics solution and the name of its configuration branch."""

    q: np.ndarray
    branch: str


@dataclass(eq=False)
class SerialRobot:
    """Serial chain described by standard DH parameters.

    ``base`` maps the world frame to DH frame 0 and ``tool`` maps DH frame n to
    the tool centre point (TCP). ``task_rows`` selects which rows of the 6-D
    twist (vx, vy, vz, wx, wy, wz) the robot can actually control; it is used
    for manipulability, singularity tests and the numeric IK so that a planar
    arm is not reported as permanently singular in the directions it can never
    move.
    """

    name: str
    links: Sequence[DHLink]
    base: np.ndarray = field(default_factory=lambda: np.eye(4))
    tool: np.ndarray = field(default_factory=lambda: np.eye(4))
    task_rows: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

    def __post_init__(self) -> None:
        self.links = tuple(self.links)
        self.base = np.asarray(self.base, dtype=float)
        self.tool = np.asarray(self.tool, dtype=float)

    # ------------------------------------------------------------ properties

    @property
    def n(self) -> int:
        """Number of joints."""
        return len(self.links)

    @property
    def qlim(self) -> np.ndarray:
        """Joint limits as an (n, 2) array of [lower, upper]."""
        return np.array([link.qlim for link in self.links], dtype=float)

    @property
    def revolute(self) -> np.ndarray:
        """Boolean mask of revolute joints."""
        return np.array([not link.prismatic for link in self.links])

    # ------------------------------------------------------ forward kinematics

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Tool pose in the world frame.

        ``q`` of shape (n,) gives a (4, 4) transform; ``q`` of shape (N, n)
        gives (N, 4, 4). The batched form is what makes Monte Carlo workspace
        sampling fast: every link is evaluated for all samples at once.
        """
        q = self._check_q(q)
        T = np.broadcast_to(self.base, q.shape[:-1] + (4, 4))
        for i, link in enumerate(self.links):
            T = T @ link.transform(q[..., i])
        return T @ self.tool

    def fk_frames(self, q: np.ndarray) -> np.ndarray:
        """All frames of one configuration, shape (n + 2, 4, 4).

        Index 0 is DH frame 0 (the base), index i is DH frame i and index n+1
        is the tool frame, all expressed in the world frame.
        """
        q = self._check_q(q)
        if q.ndim != 1:
            raise ValueError("fk_frames expects a single configuration")
        frames = np.empty((self.n + 2, 4, 4))
        frames[0] = self.base
        for i, link in enumerate(self.links):
            frames[i + 1] = frames[i] @ link.transform(q[i])
        frames[-1] = frames[-2] @ self.tool
        return frames

    # ---------------------------------------------------------------- Jacobian

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Geometric Jacobian (6 x n) of the TCP, expressed in the world frame.

        Column i uses the z axis ``z`` and origin ``o`` of the frame the joint
        acts in (DH frame i-1): a revolute joint contributes
        ``[z x (p_tcp - o); z]``, a prismatic joint ``[z; 0]``.
        """
        frames = self.fk_frames(q)
        p_tcp = frames[-1][:3, 3]
        J = np.zeros((6, self.n))
        for i, link in enumerate(self.links):
            z = frames[i][:3, 2]
            if link.prismatic:
                J[:3, i] = z
            else:
                J[:3, i] = np.cross(z, p_tcp - frames[i][:3, 3])
                J[3:, i] = z
        return J

    def task_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Rows of the geometric Jacobian selected by ``task_rows``."""
        return self.jacobian(q)[list(self.task_rows)]

    def singular_values(self, q: np.ndarray) -> np.ndarray:
        """Singular values of the task Jacobian in descending order."""
        return np.linalg.svd(self.task_jacobian(q), compute_uv=False)

    def manipulability(self, q: np.ndarray) -> float:
        """Yoshikawa manipulability ``sqrt(det(J J^T))`` of the task Jacobian.

        Computed as the product of singular values, which equals the
        determinant form for full-rank square or wide Jacobians and stays
        meaningful for tall ones.
        """
        return float(np.prod(self.singular_values(q)))

    def is_singular(self, q: np.ndarray, tol: float = 1e-6) -> bool:
        """True if the smallest task-space singular value is below ``tol``."""
        sv = self.singular_values(q)
        return bool(sv.size < len(self.task_rows) or sv[-1] < tol)

    # ------------------------------------------------------------ joint limits

    def within_limits(self, q: np.ndarray, tol: float = 1e-9) -> bool:
        """True if every joint lies inside its limits (with tolerance ``tol``)."""
        q = np.asarray(q, dtype=float)
        lim = self.qlim
        return bool(np.all(q >= lim[:, 0] - tol) and np.all(q <= lim[:, 1] + tol))

    def wrap_to_limits(
        self,
        q: np.ndarray,
        ref: np.ndarray | None = None,
        tol: float = 1e-9,
        *,
        partial: bool = False,
    ) -> np.ndarray | None:
        """Shift revolute joints by multiples of 2 pi so they fit the limits.

        IK formulas return angles in (-pi, pi], but real joints often have
        ranges such as [-45, 225] degrees, and joints with more than 360 degrees
        of travel admit several equivalent angles. Among the in-range
        candidates the one closest to ``ref`` is taken, or the one with the
        smallest shift if no reference is given. Returns ``None`` if no shift
        brings a joint into range, unless ``partial`` is set: then such joints
        keep their value and only the others are shifted.
        """
        q = np.array(q, dtype=float)
        lim = self.qlim
        for i, link in enumerate(self.links):
            shifts = [0.0] if link.prismatic else [0.0, 2 * np.pi, -2 * np.pi, 4 * np.pi, -4 * np.pi]
            valid = [q[i] + s for s in shifts if lim[i, 0] - tol <= q[i] + s <= lim[i, 1] + tol]
            if not valid:
                if partial:
                    continue
                return None
            best = valid[0] if ref is None else min(valid, key=lambda c: abs(c - ref[i]))
            q[i] = min(max(best, lim[i, 0]), lim[i, 1])
        return q

    def clip_to_limits(self, q: np.ndarray) -> np.ndarray:
        """Clamp every joint to its limits."""
        lim = self.qlim
        return np.clip(q, lim[:, 0], lim[:, 1])

    def random_q(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray:
        """Configurations drawn uniformly from the joint-limit box."""
        lim = self.qlim
        shape = (self.n,) if size is None else (size, self.n)
        return rng.uniform(lim[:, 0], lim[:, 1], size=shape)

    # -------------------------------------------------------------- utilities

    def strip_base_tool(self, T_world_tool: np.ndarray) -> np.ndarray:
        """Convert a world-frame TCP pose into the pose of DH frame n in frame 0."""
        return inv_transform(self.base) @ T_world_tool @ inv_transform(self.tool)

    def dh_table(self) -> list[dict[str, float | str]]:
        """DH parameters as a list of dictionaries (one per joint), for reports."""
        rows = []
        for i, link in enumerate(self.links, start=1):
            rows.append(
                {
                    "joint": i,
                    "type": "P" if link.prismatic else "R",
                    "theta": link.theta,
                    "d": link.d,
                    "a": link.a,
                    "alpha": link.alpha,
                    "qmin": link.qlim[0],
                    "qmax": link.qlim[1],
                }
            )
        return rows

    def _check_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape[-1] != self.n:
            raise ValueError(f"{self.name} has {self.n} joints, got q with shape {q.shape}")
        return q

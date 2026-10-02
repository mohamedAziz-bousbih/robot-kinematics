"""The three robot models: planar 3R arm, SCARA (RRPR) and PUMA 560.

Each model derives from :class:`AnalyticRobot`, which adds closed-form inverse
kinematics on top of the generic DH chain. Every analytic candidate is pushed
through forward kinematics before it is returned, so a formula error or a pose
outside the robot's task space can never produce a wrong "solution".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from .collision import CollisionModel, LinkCapsule
from .dh import DHLink, IKSolution, SerialRobot
from .transforms import orientation_error, wrap_angle

DEG = np.pi / 180.0

# A candidate is accepted only if its FK reproduces the target to this accuracy.
_FK_TOL_POS = 1e-7  # metres
_FK_TOL_ROT = 1e-7  # radians


def _two_link(x: float, y: float, l1: float, l2: float) -> list[tuple[float, float]]:
    """Both (q1, q2) solutions of a planar 2R arm reaching (x, y).

    ``q2 > 0`` puts the elbow on the clockwise side of the line from the
    shoulder to the target. Returns an empty list if the point is out of reach
    and a single solution on the stretched/folded boundary.
    """
    c2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if abs(c2) > 1.0 + 1e-12:
        return []
    c2 = float(np.clip(c2, -1.0, 1.0))
    s2_abs = np.sqrt(1.0 - c2 * c2)
    out = []
    for s2 in (s2_abs, -s2_abs) if s2_abs > 0.0 else (0.0,):
        q2 = np.arctan2(s2, c2)
        q1 = np.arctan2(y, x) - np.arctan2(l2 * s2, l1 + l2 * c2)
        out.append((q1, q2))
    return out


def _admissible(q: np.ndarray, qlim: tuple[float, float], tol: float = 1e-9) -> np.ndarray:
    """Element-wise: does some 2 pi alias of the revolute angle ``q`` lie in ``qlim``?"""
    return np.any(
        [(q + s >= qlim[0] - tol) & (q + s <= qlim[1] + tol) for s in 2.0 * np.pi * np.arange(-2, 3)],
        axis=0,
    )


def _elbow_label(q2: float, positive: str, negative: str) -> str:
    s = np.sin(q2)
    if abs(s) < 1e-12:
        return "singular"
    return positive if s > 0 else negative


class AnalyticRobot(SerialRobot, ABC):
    """Serial robot with a closed-form inverse kinematics solver and a capsule model."""

    def ik(
        self,
        T: np.ndarray,
        *,
        respect_limits: bool = True,
        q_ref: np.ndarray | None = None,
    ) -> list[IKSolution]:
        """All closed-form IK solutions for the world-frame TCP pose ``T``.

        Revolute angles are shifted by multiples of 2 pi where that brings
        them into the joint limits. With ``respect_limits`` the candidates
        that cannot be brought inside are dropped; otherwise they are kept,
        and only the offending joints stay in (-pi, pi]. ``q_ref`` resolves
        the free joint combination at a singularity (q4 + q6 at a PUMA wrist
        singularity) and picks among equivalent in-range angles, so that
        tracked motions stay smooth.
        """
        T = np.asarray(T, dtype=float)
        rev = self.revolute
        solutions: list[IKSolution] = []
        for q, branch in self._candidates(self.strip_base_tool(T), q_ref):
            q = np.array(q, dtype=float)
            q[rev] = wrap_angle(q[rev])
            shifted = self.wrap_to_limits(q, ref=q_ref)
            if shifted is None:
                if respect_limits:
                    continue
                shifted = self.wrap_to_limits(q, ref=q_ref, partial=True)
            q = shifted
            if not self._reproduces(q, T):
                continue
            if any(np.allclose(q, s.q, atol=1e-9) for s in solutions):
                continue
            solutions.append(IKSolution(q, branch))
        return solutions

    def ik_closest(self, T: np.ndarray, q_ref: np.ndarray) -> IKSolution | None:
        """The IK solution nearest to ``q_ref`` (joint-space Euclidean distance).

        Solutions are already wrapped into the joint limits, so ``q - q_ref`` is
        the motion the joints would actually have to make.
        """
        q_ref = np.asarray(q_ref, dtype=float)
        sols = self.ik(T, q_ref=q_ref)
        if not sols:
            return None
        return min(sols, key=lambda s: float(np.linalg.norm(s.q - q_ref)))

    def collision_model(self) -> CollisionModel:
        """Default capsule model of this robot."""
        return CollisionModel(self, self.capsules())

    @abstractmethod
    def capsules(self) -> list[LinkCapsule]:
        """Capsules approximating the links."""

    @abstractmethod
    def singularity_kinds(self, q: np.ndarray, tol: float = 1e-3) -> set[str]:
        """Names of the closed-form singularity conditions that ``q`` is within ``tol`` of."""

    @abstractmethod
    def _candidates(self, T0: np.ndarray, q_ref: np.ndarray | None) -> list[tuple[np.ndarray, str]]:
        """Raw closed-form candidates for the pose of DH frame n in frame 0."""

    def _reproduces(self, q: np.ndarray, T: np.ndarray) -> bool:
        Tq = self.fk(q)
        pos_ok = np.linalg.norm(Tq[:3, 3] - T[:3, 3]) < _FK_TOL_POS
        rot_ok = np.linalg.norm(orientation_error(T[:3, :3], Tq[:3, :3])) < _FK_TOL_ROT
        return bool(pos_ok and rot_ok)


# ------------------------------------------------------------------ planar 3R


class Planar3R(AnalyticRobot):
    """Planar arm with three revolute joints about parallel z axes.

    The task space is (x, y, phi) with ``phi = q1 + q2 + q3`` the angle of the
    last link. The link lengths and limits are illustrative defaults.
    """

    DEFAULT_LENGTHS = (0.5, 0.4, 0.2)
    DEFAULT_QLIM = ((-170 * DEG, 170 * DEG), (-150 * DEG, 150 * DEG), (-180 * DEG, 180 * DEG))

    def __init__(
        self,
        lengths: Sequence[float] = DEFAULT_LENGTHS,
        qlim: Sequence[tuple[float, float]] = DEFAULT_QLIM,
        base: np.ndarray | None = None,
        tool: np.ndarray | None = None,
    ):
        if len(lengths) != 3 or len(qlim) != 3:
            raise ValueError("Planar3R needs exactly three link lengths and three limits")
        links = [DHLink(a=float(l), alpha=0.0, qlim=tuple(lim)) for l, lim in zip(lengths, qlim)]
        super().__init__(
            name="Planar 3R",
            links=links,
            base=np.eye(4) if base is None else base,
            tool=np.eye(4) if tool is None else tool,
            task_rows=(0, 1, 5),
        )

    @property
    def lengths(self) -> tuple[float, ...]:
        return tuple(link.a for link in self.links)

    def _candidates(self, T0, q_ref):
        l1, l2, l3 = self.lengths
        phi = np.arctan2(T0[1, 0], T0[0, 0])
        # Subtracting the last link gives the wrist point, which a 2R arm reaches.
        xw = T0[0, 3] - l3 * np.cos(phi)
        yw = T0[1, 3] - l3 * np.sin(phi)
        return [
            (np.array([q1, q2, phi - q1 - q2]), _elbow_label(q2, "elbow_down", "elbow_up"))
            for q1, q2 in _two_link(xw, yw, l1, l2)
        ]

    def singularity_kinds(self, q: np.ndarray, tol: float = 1e-3) -> set[str]:
        """``elbow`` if links 1 and 2 are aligned: ``det J = l1 l2 sin(q2)`` vanishes."""
        return {"elbow"} if abs(np.sin(q[1])) < tol else set()

    def capsules(self) -> list[LinkCapsule]:
        return [
            LinkCapsule("link1", 0, 1, 0.04),
            LinkCapsule("link2", 1, 2, 0.035),
            LinkCapsule("link3", 2, 4, 0.03),
        ]


# ---------------------------------------------------------------------- SCARA


class Scara(AnalyticRobot):
    """SCARA arm, joint sequence R-R-P-R.

    The DH assignment follows the SCARA example in Spong, Hutchinson and
    Vidyasagar, "Robot Modeling and Control" (Wiley, 2006), with an added
    column height ``d1`` so that z is measured from the floor.

    DH table (joint variables marked with *)::

        i  theta   d      a    alpha
        1  q1*     d1     a1   0
        2  q2*     0      a2   pi
        3  0       q3*    0    0
        4  q4*     d4     0    0

    The twist of pi on link 2 points z2 downwards, so q3 > 0 lowers the quill
    and the tool z axis points at the table. The TCP pose is
    ``p = (a1 c1 + a2 c12, a1 s1 + a2 s12, d1 - q3 - d4)`` and
    ``R = Rz(q1 + q2 - q4) Rx(pi)``. Dimensions and limits are illustrative.
    """

    DEFAULT_QLIM = (
        (-130 * DEG, 130 * DEG),
        (-145 * DEG, 145 * DEG),
        (0.0, 0.20),
        (-360 * DEG, 360 * DEG),
    )

    def __init__(
        self,
        a1: float = 0.325,
        a2: float = 0.275,
        d1: float = 0.40,
        d4: float = 0.05,
        qlim: Sequence[tuple[float, float]] = DEFAULT_QLIM,
        base: np.ndarray | None = None,
        tool: np.ndarray | None = None,
    ):
        links = [
            DHLink(a=a1, alpha=0.0, d=d1, qlim=tuple(qlim[0])),
            DHLink(a=a2, alpha=np.pi, qlim=tuple(qlim[1])),
            DHLink(a=0.0, alpha=0.0, prismatic=True, qlim=tuple(qlim[2])),
            DHLink(a=0.0, alpha=0.0, d=d4, qlim=tuple(qlim[3])),
        ]
        super().__init__(
            name="SCARA (RRPR)",
            links=links,
            base=np.eye(4) if base is None else base,
            tool=np.eye(4) if tool is None else tool,
            task_rows=(0, 1, 2, 5),
        )

    def _candidates(self, T0, q_ref):
        l1, l2, l3, l4 = self.links
        x, y, z = T0[:3, 3]
        psi = np.arctan2(T0[1, 0], T0[0, 0])  # tool yaw, from R = Rz(psi) Rx(pi)
        q3 = l1.d - l4.d - l3.d - z
        return [
            (np.array([q1, q2, q3, q1 + q2 - psi]), _elbow_label(q2, "righty", "lefty"))
            for q1, q2 in _two_link(x, y, l1.a, l2.a)
        ]

    def singularity_kinds(self, q: np.ndarray, tol: float = 1e-3) -> set[str]:
        """``elbow`` if the two arm links are aligned: ``|det J| = a1 a2 |sin(q2)|`` vanishes.

        The prismatic and the last revolute joint never lose rank: they act
        along and about the vertical axis independently of the arm.
        """
        return {"elbow"} if abs(np.sin(q[1])) < tol else set()

    def capsules(self) -> list[LinkCapsule]:
        d1 = self.links[0].d
        return [
            LinkCapsule("column", 0, 0, 0.08, point_b=(0.0, 0.0, d1)),
            LinkCapsule("arm1", 0, 1, 0.06, point_a=(0.0, 0.0, d1)),
            LinkCapsule("arm2", 1, 2, 0.05),
            LinkCapsule("quill", 2, 5, 0.02),
        ]


# ------------------------------------------------------------------ PUMA 560


class Puma560(AnalyticRobot):
    """Unimation PUMA 560 with standard DH parameters.

    Kinematic parameters and joint limits are those of the standard-DH model
    ``mdl_puma560`` in P. Corke's Robotics Toolbox for MATLAB. The toolbox
    takes the kinematic parameters from P. Corke and B. Armstrong-Helouvry,
    "A search for consensus among model parameters reported for the PUMA 560
    robot", Proc. IEEE ICRA 1994, pp. 1608-1613, and the joint limits from
    Wang and Chen, IEEE Trans. Robotics and Automation 7(4), 1991, pp. 489-499.
    Frame 0 sits at the intersection of axes 1 and 2, not at the floor.

    Joints 4-6 form a spherical wrist centred at the origin of frames 4-6, so
    the IK decouples into wrist position (q1..q3) and orientation (q4..q6).
    """

    QZ = np.zeros(6)
    QR = np.array([0.0, np.pi / 2, -np.pi / 2, 0.0, 0.0, 0.0])  # "ready": arm straight up
    # "Nominal" dexterous pose; the toolbox writes q3 = pi, -pi is the same
    # angle expressed inside the joint-3 limits.
    QN = np.array([0.0, np.pi / 4, -np.pi, 0.0, np.pi / 4, 0.0])

    def __init__(self, base: np.ndarray | None = None, tool: np.ndarray | None = None):
        links = [
            DHLink(a=0.0, alpha=np.pi / 2, d=0.0, qlim=(-160 * DEG, 160 * DEG)),
            DHLink(a=0.4318, alpha=0.0, d=0.0, qlim=(-45 * DEG, 225 * DEG)),
            DHLink(a=0.0203, alpha=-np.pi / 2, d=0.15005, qlim=(-225 * DEG, 45 * DEG)),
            DHLink(a=0.0, alpha=np.pi / 2, d=0.4318, qlim=(-110 * DEG, 170 * DEG)),
            DHLink(a=0.0, alpha=-np.pi / 2, d=0.0, qlim=(-100 * DEG, 100 * DEG)),
            DHLink(a=0.0, alpha=0.0, d=0.0, qlim=(-266 * DEG, 266 * DEG)),
        ]
        super().__init__(
            name="PUMA 560",
            links=links,
            base=np.eye(4) if base is None else base,
            tool=np.eye(4) if tool is None else tool,
        )

    @property
    def forearm(self) -> tuple[float, float]:
        """Length and angle of the elbow-to-wrist-centre vector in frame 2.

        With ``a3`` and ``d4`` the wrist centre sits at
        ``L3 * (cos(q3 + beta), sin(q3 + beta))`` relative to the elbow, which
        turns joints 2 and 3 into an ordinary 2R arm with links a2 and L3.
        """
        a3, d4 = self.links[2].a, self.links[3].d
        return float(np.hypot(a3, d4)), float(np.arctan2(d4, a3))

    def arm_plane_reach(self, q: np.ndarray) -> float:
        """Signed horizontal reach ``x1`` of the wrist centre in the arm plane."""
        a2, a3, d4 = self.links[1].a, self.links[2].a, self.links[3].d
        q2, q23 = q[1], q[1] + q[2]
        return float(a2 * np.cos(q2) + a3 * np.cos(q23) - d4 * np.sin(q23))

    def wrist_distance_range(self) -> tuple[float, float]:
        """Smallest and largest distance of the wrist centre from the frame-0 origin.

        The distance is ``sqrt(rho^2 + d3^2)`` with the in-plane shoulder-wrist
        distance ``rho^2 = a2^2 + L3^2 + 2 a2 L3 cos(q3 + beta)``; joint 2 only
        rotates that vector. So the range follows from the extremes of
        ``cos(q3 + beta)`` over the joint-3 limits: an interior extremum where
        the interval contains a multiple of pi, otherwise an end point. This is
        the analytic reference for the Monte Carlo workspace.
        """
        a2, d3 = self.links[1].a, self.links[2].d
        L3, beta = self.forearm
        lo, hi = self.links[2].qlim[0] + beta, self.links[2].qlim[1] + beta
        cos_values = [np.cos(lo), np.cos(hi)]
        for k in range(int(np.ceil(lo / np.pi)), int(np.floor(hi / np.pi)) + 1):
            cos_values.append(np.cos(k * np.pi))
        rho2 = [a2 * a2 + L3 * L3 + 2.0 * a2 * L3 * c for c in (min(cos_values), max(cos_values))]
        return float(np.sqrt(rho2[0] + d3 * d3)), float(np.sqrt(rho2[1] + d3 * d3))

    def wrist_reachable(self, points: np.ndarray) -> np.ndarray:
        """Which wrist-centre positions (frame 0, shape (N, 3)) joints 1-3 can reach.

        This is the position half of the closed-form IK in vectorised form,
        reduced to a yes/no answer: a point is reachable if one of the four
        shoulder/elbow solutions has q1, q2 and q3 inside their limits (up to
        2 pi aliases). The wrist joints do not move the wrist centre, so their
        limits play no role. Used for hit-or-miss volume estimates, where
        millions of points are classified at once.
        """
        p = np.atleast_2d(np.asarray(points, dtype=float))
        l1, l2, l3 = self.links[:3]
        L3, beta = self.forearm
        px, py, pz = p[:, 0], p[:, 1], p[:, 2] - l1.d
        r2 = px * px + py * py - l3.d * l3.d  # squared in-plane reach x1^2
        c = (r2 + pz * pz - l2.a * l2.a - L3 * L3) / (2.0 * l2.a * L3)  # cos(q3 + beta)
        geometric = (r2 >= 0.0) & (np.abs(c) <= 1.0)
        r = np.sqrt(np.maximum(r2, 0.0))
        theta_abs = np.arccos(np.clip(c, -1.0, 1.0))
        reachable = np.zeros(len(p), dtype=bool)
        for x1 in (r, -r):
            q1_ok = _admissible(np.arctan2(py, px) - np.arctan2(-l3.d, x1), l1.qlim)
            for theta in (theta_abs, -theta_abs):
                q2 = np.arctan2(pz, x1) - np.arctan2(L3 * np.sin(theta), l2.a + L3 * np.cos(theta))
                reachable |= q1_ok & _admissible(q2, l2.qlim) & _admissible(theta - beta, l3.qlim)
        return reachable & geometric

    def singularity_kinds(self, q: np.ndarray, tol: float = 1e-3) -> set[str]:
        """Which of the three classic PUMA singularities ``q`` is close to.

        * ``shoulder``: the wrist centre lies on the cylinder of radius d3
          around axis 1 (``x1 = 0``), where the two shoulder solutions merge.
        * ``elbow``: upper arm and forearm are aligned (arm stretched/folded).
        * ``wrist``: axes 4 and 6 are aligned (``q5 = 0``).
        """
        _, beta = self.forearm
        kinds = set()
        if abs(self.arm_plane_reach(q)) < tol:
            kinds.add("shoulder")
        if abs(np.sin(q[2] + beta)) < tol:
            kinds.add("elbow")
        if abs(np.sin(q[4])) < tol:
            kinds.add("wrist")
        return kinds

    def _candidates(self, T0, q_ref):
        l1, l2, l3, _, _, l6 = self.links
        R = T0[:3, :3]
        pw = T0[:3, 3] - l6.d * R[:, 2]  # wrist centre
        px, py, pz = pw[0], pw[1], pw[2] - l1.d
        r2 = px * px + py * py - l3.d * l3.d
        if r2 < -1e-12:
            return []  # inside the cylinder swept by the shoulder offset
        r = np.sqrt(max(r2, 0.0))
        L3, beta = self.forearm
        out = []
        for x1 in (r, -r) if r > 0.0 else (0.0,):
            shoulder = "front" if x1 >= 0.0 else "back"
            # Joint 1 rotates the in-plane vector (x1, -d3) onto (px, py).
            q1 = np.arctan2(py, px) - np.arctan2(-l3.d, x1)
            for q2, theta in _two_link(x1, pz, l2.a, L3):
                q3 = theta - beta
                # Elbow "up" = elbow above the straight line shoulder -> wrist.
                cross = x1 * l2.a * np.sin(q2) - pz * l2.a * np.cos(q2)
                elbow = "up" if cross * x1 > 0.0 else "down"
                R03 = (l1.transform(q1) @ l2.transform(q2) @ l3.transform(q3))[:3, :3]
                for (q4, q5, q6), wrist in self._wrist(R03.T @ R, q_ref):
                    out.append((np.array([q1, q2, q3, q4, q5, q6]), f"{shoulder}/{elbow}/{wrist}"))
        return out

    @staticmethod
    def _wrist(R36: np.ndarray, q_ref: np.ndarray | None) -> list[tuple[tuple[float, float, float], str]]:
        """Solve ``R36 = Rz(q4) Ry(-q5) Rz(q6)``.

        alpha4 = +90 deg and alpha5 = -90 deg turn the wrist into a ZYZ Euler
        chain with a negated middle angle, hence the sign pattern below.
        """
        s5 = np.hypot(R36[0, 2], R36[1, 2])
        c5 = R36[2, 2]
        if s5 > 1e-9:
            out = []
            for sign, label in ((1.0, "noflip"), (-1.0, "flip")):
                q5 = np.arctan2(sign * s5, c5)
                q4 = np.arctan2(-sign * R36[1, 2], -sign * R36[0, 2])
                q6 = np.arctan2(-sign * R36[2, 1], sign * R36[2, 0])
                out.append(((q4, q5, q6), label))
            return out
        # Wrist singularity: only q4 + q6 (q5 = 0) or q6 - q4 (q5 = pi) is
        # defined. Keep q4 from the reference so a tracked motion does not jump.
        q4 = 0.0 if q_ref is None else float(q_ref[3])
        if c5 > 0.0:
            return [((q4, 0.0, np.arctan2(R36[1, 0], R36[0, 0]) - q4), "wrist_singular")]
        return [((q4, np.pi, np.arctan2(R36[1, 0], -R36[0, 0]) + q4), "wrist_singular")]

    def capsules(self) -> list[LinkCapsule]:
        return [
            LinkCapsule("pedestal", 0, 0, 0.10, point_a=(0.0, 0.0, -0.60)),
            LinkCapsule("upper_arm", 1, 2, 0.07),
            LinkCapsule("shoulder_offset", 2, 3, 0.05),
            LinkCapsule("forearm", 3, 4, 0.05),
            LinkCapsule("wrist", 4, 7, 0.04),
        ]

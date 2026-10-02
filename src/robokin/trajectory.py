"""Trajectory planning: PTP in joint space and LIN in Cartesian space.

PTP (point-to-point) moves every joint from ``q0`` to ``q1`` with either a
trapezoidal velocity profile (piecewise constant acceleration) or a quintic
polynomial (continuous acceleration). Three coordination modes are offered:

``async``
    every axis runs its own time-optimal profile and stops when it is done;
``sync``
    all axes finish together: the slowest ("leading") axis sets the duration
    and the others keep their maximum acceleration but cruise slower;
``full``
    fully synchronised trapezoids: all axes also share the acceleration and
    deceleration phases, so the joint path is a straight line in joint space.

LIN moves the TCP along a straight line with the orientation interpolated by
SLERP; the path parameter follows a trapezoidal profile that respects both a
translational and a rotational speed/acceleration limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence

import numpy as np

from .transforms import interpolate_pose, rot_log

if TYPE_CHECKING:
    from .robots import AnalyticRobot


class Profile(Protocol):
    duration: float

    def sample(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


# ----------------------------------------------------------------- trapezoid


@dataclass(frozen=True)
class TrapezoidProfile:
    """Rest-to-rest 1-D motion: accelerate for ``t_acc``, cruise, decelerate for ``t_acc``.

    ``t_acc = duration / 2`` is the triangular case without a cruise phase.
    """

    start: float
    distance: float
    t_acc: float
    duration: float

    def __post_init__(self) -> None:
        if self.distance != 0.0 and not (0.0 < self.t_acc <= 0.5 * self.duration * (1 + 1e-12)):
            raise ValueError("need 0 < t_acc <= duration / 2 for a non-zero move")

    @property
    def velocity(self) -> float:
        """Signed cruise velocity."""
        if self.distance == 0.0:
            return 0.0
        return self.distance / (self.duration - self.t_acc)

    @property
    def acceleration(self) -> float:
        """Signed acceleration during the first phase."""
        if self.distance == 0.0:
            return 0.0
        return self.velocity / self.t_acc

    @classmethod
    def time_optimal(cls, start: float, distance: float, vmax: float, amax: float) -> TrapezoidProfile:
        """Shortest profile with ``|v| <= vmax`` and ``|a| <= amax``."""
        if vmax <= 0.0 or amax <= 0.0:
            raise ValueError("vmax and amax must be positive")
        d = abs(distance)
        if d == 0.0:
            return cls(start, 0.0, 0.0, 0.0)
        if d >= vmax * vmax / amax:
            t_acc = vmax / amax
            return cls(start, distance, t_acc, d / vmax + t_acc)
        t_acc = np.sqrt(d / amax)  # triangular: vmax is never reached
        return cls(start, distance, t_acc, 2.0 * t_acc)

    @classmethod
    def with_duration(cls, start: float, distance: float, duration: float, amax: float) -> TrapezoidProfile:
        """Profile of a given duration that keeps full acceleration ``amax``.

        From ``d = a * t_acc * (T - t_acc)`` the acceleration time is the smaller
        root ``t_acc = (T - sqrt(T^2 - 4 d / a)) / 2``; it exists if ``T`` is at
        least the time-optimal triangular duration ``2 sqrt(d / a)``.
        """
        d = abs(distance)
        if d == 0.0:
            return cls(start, 0.0, 0.0, duration)
        disc = duration * duration - 4.0 * d / amax
        if disc < -1e-9 * duration * duration:
            raise ValueError("duration too short for the acceleration limit")
        t_acc = 0.5 * (duration - np.sqrt(max(disc, 0.0)))
        return cls(start, distance, t_acc, duration)

    def sample(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Position, velocity and acceleration at time(s) ``t``.

        Before 0 and after ``duration`` the axis is at rest.
        """
        t = np.asarray(t, dtype=float)
        if self.distance == 0.0:
            zeros = np.zeros_like(t)
            return zeros + self.start, zeros, zeros
        T, ta, v, a = self.duration, self.t_acc, self.velocity, self.acceleration
        tc = np.clip(t, 0.0, T)
        accel = tc < ta
        decel = tc > T - ta
        pos = np.where(
            accel,
            0.5 * a * tc * tc,
            np.where(decel, self.distance - 0.5 * a * (T - tc) ** 2, 0.5 * v * ta + v * (tc - ta)),
        )
        vel = np.where(accel, a * tc, np.where(decel, a * (T - tc), v))
        acc = np.where(accel, a, np.where(decel, -a, 0.0))
        moving = (t >= 0.0) & (t <= T)
        return self.start + pos, np.where(moving, vel, 0.0), np.where(moving, acc, 0.0)


# ------------------------------------------------------------------- quintic


@dataclass(frozen=True)
class QuinticProfile:
    """Quintic polynomial ``q(t) = sum c_k t^k`` on [0, duration]."""

    coeffs: tuple[float, float, float, float, float, float]
    duration: float

    @classmethod
    def from_boundary(
        cls,
        q0: float,
        q1: float,
        duration: float,
        v0: float = 0.0,
        v1: float = 0.0,
        a0: float = 0.0,
        a1: float = 0.0,
    ) -> QuinticProfile:
        """Unique quintic matching position, velocity and acceleration at both ends."""
        if duration <= 0.0:
            if q0 != q1:
                raise ValueError("a non-zero move needs a positive duration")
            return cls((q0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0)
        T = duration
        c0, c1, c2 = q0, v0, 0.5 * a0
        # The remaining three coefficients follow from the end conditions.
        M = np.array([[T**3, T**4, T**5], [3 * T**2, 4 * T**3, 5 * T**4], [6 * T, 12 * T**2, 20 * T**3]])
        rhs = np.array([q1 - (c0 + c1 * T + c2 * T * T), v1 - (c1 + 2 * c2 * T), a1 - 2 * c2])
        c3, c4, c5 = np.linalg.solve(M, rhs)
        return cls((c0, c1, c2, float(c3), float(c4), float(c5)), T)

    @staticmethod
    def min_duration(distance: float, vmax: float, amax: float) -> float:
        """Shortest rest-to-rest duration within the limits.

        The normalised profile ``10 s^3 - 15 s^4 + 6 s^5`` peaks at velocity
        ``15/8 d/T`` (s = 1/2) and acceleration ``10/sqrt(3) d/T^2``
        (s = 1/2 -+ sqrt(3)/6), which gives the two lower bounds on ``T``.
        """
        d = abs(distance)
        return max(15.0 * d / (8.0 * vmax), np.sqrt(10.0 * d / (np.sqrt(3.0) * amax)))

    def sample(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Position, velocity and acceleration; the end state is held outside [0, T]."""
        t = np.asarray(t, dtype=float)
        tc = np.clip(t, 0.0, self.duration)
        c = self.coeffs
        pos = c[0] + tc * (c[1] + tc * (c[2] + tc * (c[3] + tc * (c[4] + tc * c[5]))))
        vel = c[1] + tc * (2 * c[2] + tc * (3 * c[3] + tc * (4 * c[4] + tc * 5 * c[5])))
        acc = 2 * c[2] + tc * (6 * c[3] + tc * (12 * c[4] + tc * 20 * c[5]))
        moving = (t >= 0.0) & (t <= self.duration)
        return pos, np.where(moving, vel, 0.0), np.where(moving, acc, 0.0)


# -------------------------------------------------------------------- PTP


@dataclass(frozen=True)
class JointTrajectory:
    """Multi-axis trajectory made of independent 1-D profiles."""

    profiles: tuple[Profile, ...]

    @property
    def duration(self) -> float:
        return max(p.duration for p in self.profiles)

    def sample(self, t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Joint positions, velocities, accelerations, each of shape (len(t), n)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        parts = [p.sample(t) for p in self.profiles]
        return tuple(np.stack([part[k] for part in parts], axis=1) for k in range(3))  # type: ignore[return-value]

    def __call__(self, t: float) -> np.ndarray:
        """Joint position at a single time ``t`` (used by the collision checker)."""
        return self.sample(t)[0][0]


def plan_ptp(
    q0: Sequence[float],
    q1: Sequence[float],
    vmax: Sequence[float],
    amax: Sequence[float],
    *,
    profile: str = "trapezoid",
    sync: str = "full",
) -> JointTrajectory:
    """Plan a PTP motion; see the module docstring for the ``sync`` modes.

    ``profile`` is ``"trapezoid"`` or ``"quintic"``. For quintics ``"sync"``
    and ``"full"`` are identical because a common duration already fixes the
    shape of every axis.
    """
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    vmax, amax = np.asarray(vmax, dtype=float), np.asarray(amax, dtype=float)
    dist = q1 - q0
    if not (q0.shape == q1.shape == vmax.shape == amax.shape):
        raise ValueError("q0, q1, vmax and amax must have the same length")
    if np.any(vmax <= 0.0) or np.any(amax <= 0.0):
        raise ValueError("vmax and amax must be positive")
    if sync not in ("async", "sync", "full"):
        raise ValueError(f"unknown sync mode {sync!r}")

    if profile == "quintic":
        durations = [QuinticProfile.min_duration(d, v, a) for d, v, a in zip(dist, vmax, amax)]
        if sync != "async":
            durations = [max(durations)] * len(dist)
        return JointTrajectory(
            tuple(QuinticProfile.from_boundary(a, b, T) for a, b, T in zip(q0, q1, durations))
        )
    if profile != "trapezoid":
        raise ValueError(f"unknown profile {profile!r}")

    fastest = [TrapezoidProfile.time_optimal(s, d, v, a) for s, d, v, a in zip(q0, dist, vmax, amax)]
    if sync == "async":
        return JointTrajectory(tuple(fastest))
    if sync == "sync":
        T = max(p.duration for p in fastest)
        return JointTrajectory(
            tuple(TrapezoidProfile.with_duration(s, d, T, a) for s, d, a in zip(q0, dist, amax))
        )

    # Fully synchronised: shared t_acc and cruise end te = T - t_acc. Axis i
    # needs te >= |d_i| / vmax_i (speed) and te * t_acc >= |d_i| / amax_i
    # (acceleration). Minimising T = t_acc + te under t_acc <= te gives:
    ad = np.abs(dist)
    if not np.any(ad > 0.0):
        return JointTrajectory(tuple(TrapezoidProfile(s, 0.0, 0.0, 0.0) for s in q0))
    E = float(np.max(ad / vmax))
    P = float(np.max(ad / amax))
    if np.sqrt(P) >= E:
        t_acc = te = np.sqrt(P)  # triangular for the leading axis
    else:
        te, t_acc = E, P / E
    T = t_acc + te
    return JointTrajectory(
        tuple(
            TrapezoidProfile(s, d, t_acc if d != 0.0 else 0.0, T)
            for s, d in zip(q0, dist)
        )
    )


# -------------------------------------------------------------------- LIN


@dataclass(frozen=True)
class CartesianTrajectory:
    """Straight-line TCP motion with SLERP orientation, timed by ``timing``.

    ``timing`` maps time to the path parameter s in [0, 1].
    """

    T_start: np.ndarray
    T_end: np.ndarray
    timing: TrapezoidProfile

    @property
    def duration(self) -> float:
        return self.timing.duration

    def pose(self, t: float) -> np.ndarray:
        """TCP pose at time ``t``."""
        s = float(self.timing.sample(t)[0])
        return interpolate_pose(self.T_start, self.T_end, s)


def plan_lin(
    T_start: np.ndarray,
    T_end: np.ndarray,
    v_max: float,
    a_max: float,
    w_max: float,
    alpha_max: float,
) -> CartesianTrajectory:
    """Time a LIN move so that TCP speed/acceleration and angular speed/acceleration stay in bounds.

    With path length ``L`` and rotation angle ``theta`` the path parameter may
    move at most ``min(v_max / L, w_max / theta)`` per second (and likewise
    for acceleration), because position and SLERP angle both grow linearly in s.
    """
    L = float(np.linalg.norm(T_end[:3, 3] - T_start[:3, 3]))
    theta = float(np.linalg.norm(rot_log(T_end[:3, :3] @ T_start[:3, :3].T)))
    v_s = min(v_max / L if L > 0 else np.inf, w_max / theta if theta > 0 else np.inf)
    a_s = min(a_max / L if L > 0 else np.inf, alpha_max / theta if theta > 0 else np.inf)
    if np.isinf(v_s):
        timing = TrapezoidProfile(0.0, 0.0, 0.0, 0.0)
    else:
        timing = TrapezoidProfile.time_optimal(0.0, 1.0, v_s, a_s)
    return CartesianTrajectory(np.array(T_start, float), np.array(T_end, float), timing)


@dataclass(frozen=True)
class SampledJointTrajectory:
    """Joint trajectory given by samples, linearly interpolated in between."""

    t: np.ndarray  # (K,)
    q: np.ndarray  # (K, n)

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def __call__(self, t: float) -> np.ndarray:
        return np.array([np.interp(t, self.t, self.q[:, j]) for j in range(self.q.shape[1])])

    def velocities(self) -> np.ndarray:
        """Joint velocities by finite differences (second order in the interior)."""
        if len(self.t) < 2:
            return np.zeros_like(self.q)  # a single sample is a motion that stands still
        return np.gradient(self.q, self.t, axis=0)

    def max_step(self) -> float:
        """Largest single-joint change between consecutive samples."""
        return float(np.max(np.abs(np.diff(self.q, axis=0)))) if len(self.t) > 1 else 0.0


def track_lin(
    robot: AnalyticRobot, traj: CartesianTrajectory, q_start: np.ndarray, dt: float
) -> SampledJointTrajectory:
    """Convert a LIN move into joint space with the robot's closed-form IK.

    At each sample the IK branch closest to the previous configuration is
    taken, which keeps the arm in one configuration (no elbow or wrist flips)
    along the path. Raises ``ValueError`` if a pose on the line is unreachable
    within the joint limits.
    """
    if traj.duration > 0.0:
        times = np.linspace(0.0, traj.duration, max(int(np.ceil(traj.duration / dt)), 1) + 1)
    else:
        # A LIN to the current pose: one sample, so no zero-length time steps
        # end up in the finite-difference velocities.
        times = np.zeros(1)
    q_prev = np.asarray(q_start, dtype=float)
    qs = []
    for t in times:
        sol = robot.ik_closest(traj.pose(t), q_prev)
        if sol is None:
            raise ValueError(f"LIN pose at t = {t:.3f} s is not reachable within the joint limits")
        q_prev = sol.q
        qs.append(sol.q)
    return SampledJointTrajectory(times, np.array(qs))

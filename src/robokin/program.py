"""A minimal motion program: a sequence of PTP and LIN moves.

This mirrors how industrial robots are taught: the programmer stores target
poses and chooses per move whether the joints interpolate (PTP, fast, path not
straight) or the TCP moves on a straight line (LIN). Every move starts and
ends at rest where the previous one ended, so the composed joint trajectory is
continuous. Blending (fly-by) between moves is not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

from .collision import CollisionModel, PathCheck, Sphere, check_motion
from .robots import AnalyticRobot
from .trajectory import JointTrajectory, SampledJointTrajectory, plan_lin, plan_ptp, track_lin

Motion = Union[JointTrajectory, SampledJointTrajectory]


@dataclass(frozen=True)
class Segment:
    """One move of a program, placed on the program's time axis."""

    kind: str  # "PTP" or "LIN"
    label: str
    t_start: float
    motion: Motion

    @property
    def t_end(self) -> float:
        return self.t_start + self.motion.duration


class Program:
    """Builds a program move by move; call the instance to get q(t).

    ``vmax``/``amax`` are per-joint limits for PTP moves, which are planned
    fully synchronised with the given ``profile`` ("trapezoid" or "quintic");
    ``lin_limits`` is ``(v, a, w, alpha)`` for the TCP and is only needed by
    programs with LIN moves; ``dt`` is the IK sampling period for LIN moves.
    """

    def __init__(
        self,
        robot: AnalyticRobot,
        q_start: Sequence[float],
        vmax: Sequence[float],
        amax: Sequence[float],
        lin_limits: tuple[float, float, float, float] | None = None,
        dt: float = 0.01,
        profile: str = "trapezoid",
    ):
        self.robot = robot
        self.q = np.asarray(q_start, dtype=float)
        self.q_start = self.q.copy()
        self.vmax = np.asarray(vmax, dtype=float)
        self.amax = np.asarray(amax, dtype=float)
        self.lin_limits = lin_limits
        self.dt = dt
        self.profile = profile
        self.segments: list[Segment] = []

    @property
    def duration(self) -> float:
        return self.segments[-1].t_end if self.segments else 0.0

    def ptp(self, target: np.ndarray, label: str = "") -> Program:
        """Joint-interpolated move to a joint vector or a 4x4 TCP pose.

        For a pose the IK branch closest to the current configuration is used,
        so the arm keeps its posture instead of flipping elbow or wrist.
        """
        target = np.asarray(target, dtype=float)
        if target.shape == (4, 4):
            sol = self.robot.ik_closest(target, self.q)
            if sol is None:
                raise ValueError(f"PTP target {label!r} is not reachable within the joint limits")
            target = sol.q
        motion = plan_ptp(self.q, target, self.vmax, self.amax, profile=self.profile, sync="full")
        return self._append("PTP", label, motion, target)

    def lin(self, T_goal: np.ndarray, label: str = "") -> Program:
        """Straight-line TCP move to ``T_goal`` with SLERP orientation."""
        if self.lin_limits is None:
            raise ValueError("LIN moves need lin_limits = (v, a, w, alpha)")
        v, a, w, alpha = self.lin_limits
        cart = plan_lin(self.robot.fk(self.q), np.asarray(T_goal, dtype=float), v, a, w, alpha)
        motion = track_lin(self.robot, cart, self.q, self.dt)
        return self._append("LIN", label, motion, motion.q[-1])

    def __call__(self, t: float) -> np.ndarray:
        """Joint configuration at program time ``t`` (clamped to the program)."""
        if not self.segments:
            return self.q_start.copy()
        for seg in self.segments:
            if t <= seg.t_end:
                return np.asarray(seg.motion(max(t - seg.t_start, 0.0)), dtype=float)
        last = self.segments[-1]
        return np.asarray(last.motion(last.motion.duration), dtype=float)

    def sample(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Times and joint configurations on a uniform grid over the program."""
        times = np.linspace(0.0, self.duration, max(int(np.ceil(self.duration / dt)), 1) + 1)
        return times, np.array([self(t) for t in times])

    def lin_speed_ratio(self) -> float:
        """Largest ``|qd_i| / vmax_i`` over all LIN moves (PTP moves are <= 1 by design).

        A LIN move fixes the TCP path, so the joint speeds follow from the
        inverse Jacobian and can exceed the joint limits near singularities;
        this ratio shows whether a real controller would have to slow down.
        """
        ratios = [
            float(np.max(np.abs(seg.motion.velocities()) / self.vmax))
            for seg in self.segments
            if seg.kind == "LIN" and len(seg.motion.t) > 1
        ]
        return max(ratios, default=0.0)

    def breakpoints(self) -> np.ndarray:
        """Times where a joint may reverse: segment ends and LIN interpolation knots."""
        times = [seg.t_end for seg in self.segments]
        for seg in self.segments:
            if isinstance(seg.motion, SampledJointTrajectory):
                times.extend(seg.t_start + seg.motion.t)
        return np.unique(times)

    def check_collisions(
        self, model: CollisionModel, obstacles: Sequence[Sphere], **kwargs: int
    ) -> PathCheck:
        """Path check of the whole program; see :func:`robokin.check_motion`."""
        return check_motion(
            model, obstacles, self, self.duration, breakpoints=self.breakpoints(), **kwargs
        )

    def _append(self, kind: str, label: str, motion: Motion, q_end: np.ndarray) -> Program:
        self.segments.append(Segment(kind, label, self.duration, motion))
        self.q = np.asarray(q_end, dtype=float)
        return self

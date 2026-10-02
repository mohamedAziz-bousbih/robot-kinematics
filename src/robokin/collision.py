"""Capsule-based collision checking of a serial arm against spherical obstacles.

Each link is approximated by a capsule (a line segment swept by a sphere) whose
end points are fixed in two DH frames. A capsule and a sphere are separated iff
the distance from the sphere centre to the segment exceeds the sum of radii, so
the signed clearance is cheap and exact for this geometry.

A path is checked on samples, and the interval between two samples is
*certified* collision-free with a displacement bound: if no point of the arm can
move further than ``delta`` between the samples, the clearance anywhere in
between is at least ``(c_k + c_{k+1} - delta) / 2``. Intervals where that bound
is not positive are bisected until they are certified, a sample collides, or the
sample budget runs out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .dh import SerialRobot


@dataclass(frozen=True)
class Sphere:
    """Spherical obstacle."""

    center: tuple[float, float, float]
    radius: float


@dataclass(frozen=True)
class LinkCapsule:
    """Capsule attached to the robot.

    ``frame_a``/``frame_b`` index the output of :meth:`SerialRobot.fk_frames`
    (0 = base, i = DH frame i, n + 1 = tool); ``point_a``/``point_b`` are the
    segment end points expressed in those frames.
    """

    name: str
    frame_a: int
    frame_b: int
    radius: float
    point_a: tuple[float, float, float] = (0.0, 0.0, 0.0)
    point_b: tuple[float, float, float] = (0.0, 0.0, 0.0)


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance from point ``p`` to the segment ``[a, b]``."""
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom == 0.0 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def capsule_sphere_clearance(
    a: np.ndarray, b: np.ndarray, capsule_radius: float, sphere: Sphere
) -> float:
    """Signed distance between capsule and sphere surfaces (negative = overlap)."""
    c = np.asarray(sphere.center, dtype=float)
    return point_segment_distance(c, a, b) - capsule_radius - sphere.radius


class CollisionModel:
    """Capsule model of a robot plus the displacement bound used for certification."""

    def __init__(self, robot: SerialRobot, capsules: Sequence[LinkCapsule]):
        self.robot = robot
        self.capsules = tuple(capsules)
        last = robot.n + 1
        for cap in self.capsules:
            if not (0 <= cap.frame_a <= last and 0 <= cap.frame_b <= last):
                raise ValueError(f"capsule {cap.name!r} references a frame outside 0..{last}")
        self._lever = self._lever_arms()

    def segments(self, q: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """World-frame end points and radius of every capsule at configuration ``q``."""
        frames = self.robot.fk_frames(q)
        out = []
        for cap in self.capsules:
            a = frames[cap.frame_a] @ np.append(cap.point_a, 1.0)
            b = frames[cap.frame_b] @ np.append(cap.point_b, 1.0)
            out.append((a[:3], b[:3], cap.radius))
        return out

    def clearance(self, q: np.ndarray, obstacles: Sequence[Sphere]) -> float:
        """Smallest signed capsule-obstacle clearance (``inf`` without obstacles)."""
        best = np.inf
        for a, b, r in self.segments(q):
            for sphere in obstacles:
                best = min(best, capsule_sphere_clearance(a, b, r, sphere))
        return float(best)

    def displacement_bound(self, q0: np.ndarray, q1: np.ndarray) -> float:
        """Upper bound on how far any capsule point moves between ``q0`` and ``q1``.

        A revolute joint rotating by ``dq`` moves a point at distance ``r`` from
        its axis by at most ``r * |dq|``; a prismatic joint moves it by
        ``|dq|``. ``r`` is bounded by the sum of the link offsets between the
        joint and the point, so the bound needs no forward kinematics. Points
        on a capsule axis are convex combinations of its end points, so
        bounding the end points bounds the whole capsule. The bound assumes
        each joint moves monotonically between the two configurations, which
        holds for rest-to-rest PTP profiles and for short sub-intervals of
        smooth motions.
        """
        dq = np.abs(np.asarray(q1, dtype=float) - np.asarray(q0, dtype=float))
        return float(np.max(self._lever @ dq, initial=0.0))

    def _lever_arms(self) -> np.ndarray:
        """Matrix (end points x joints) of metres moved per unit of joint motion."""
        robot = self.robot
        n = robot.n
        # Largest |a| + |d| of each link; a prismatic d takes its extreme value.
        offset = np.array(
            [
                abs(link.a)
                + (max(abs(link.d + link.qlim[0]), abs(link.d + link.qlim[1]))
                   if link.prismatic else abs(link.d))
                for link in robot.links
            ]
        )
        tool_offset = float(np.linalg.norm(robot.tool[:3, 3]))
        rows = []
        for cap in self.capsules:
            for frame, point in ((cap.frame_a, cap.point_a), (cap.frame_b, cap.point_b)):
                row = np.zeros(n)
                moved_by = min(frame, n)  # frame i hangs off joints 1..i
                extra = float(np.linalg.norm(point)) + (tool_offset if frame == n + 1 else 0.0)
                for i in range(moved_by):
                    if robot.links[i].prismatic:
                        row[i] = 1.0
                    else:
                        # Joint i+1 turns about z of frame i, whose origin is on
                        # the axis; offsets add up along the chain to the point.
                        row[i] = offset[i:moved_by].sum() + extra
                rows.append(row)
        return np.array(rows).reshape(len(rows), n)


@dataclass(frozen=True)
class PathCheck:
    """Result of a sampled path check.

    ``collision`` is True if a sample penetrates an obstacle. ``certified`` is
    True if every interval between samples was proven collision-free by the
    displacement bound. ``min_clearance`` is the smallest sampled clearance.
    """

    collision: bool
    certified: bool
    min_clearance: float
    first_collision_time: float | None
    samples: int


def check_samples(
    model: CollisionModel,
    obstacles: Sequence[Sphere],
    q_samples: np.ndarray,
    times: np.ndarray | None = None,
) -> PathCheck:
    """Check a fixed list of configurations without refinement.

    ``times`` labels the samples for ``first_collision_time``; without it the
    sample index is reported instead.
    """
    q_samples = np.asarray(q_samples, dtype=float)
    times = np.arange(len(q_samples), dtype=float) if times is None else np.asarray(times, float)
    c = np.array([model.clearance(q, obstacles) for q in q_samples])
    hits = np.flatnonzero(c < 0.0)
    certified = hits.size == 0 and all(
        c[k] + c[k + 1] > model.displacement_bound(q_samples[k], q_samples[k + 1])
        for k in range(len(q_samples) - 1)
    )
    return PathCheck(
        collision=hits.size > 0,
        certified=bool(certified),
        min_clearance=float(c.min()),
        first_collision_time=float(times[hits[0]]) if hits.size else None,
        samples=len(q_samples),
    )


def check_motion(
    model: CollisionModel,
    obstacles: Sequence[Sphere],
    q_of_t: Callable[[float], np.ndarray],
    duration: float,
    *,
    initial_samples: int = 16,
    max_samples: int = 20000,
    breakpoints: Sequence[float] = (),
) -> PathCheck:
    """Check a continuous motion ``q_of_t`` on [0, duration] with adaptive bisection.

    Starts from ``initial_samples`` uniform samples plus the ``breakpoints``
    and bisects every interval whose displacement bound is not covered by the
    clearances at its ends. The bound needs every joint to move monotonically
    inside an interval, so times where a joint may reverse (segment
    boundaries of a program, knots of a piecewise-linear path) must be given
    as breakpoints; intervals never straddle one. ``first_collision_time``
    reports the earliest colliding sample time found.
    """
    if not obstacles:
        return PathCheck(False, True, np.inf, None, 0)
    grid = np.linspace(0.0, duration, max(initial_samples, 2))
    extra = np.asarray(breakpoints, dtype=float)
    times = list(np.unique(np.concatenate([grid, extra[(extra > 0.0) & (extra < duration)]])))
    qs = [np.asarray(q_of_t(t), dtype=float) for t in times]
    cs = [model.clearance(q, obstacles) for q in qs]
    samples = len(times)
    collision_times = [t for t, c in zip(times, cs) if c < 0.0]

    # Work list of intervals as index pairs into the growing sample lists.
    pending = [(k, k + 1) for k in range(len(times) - 1)]
    certified = True
    while pending:
        i, j = pending.pop()
        if cs[i] < 0.0 or cs[j] < 0.0:
            continue
        if cs[i] + cs[j] > model.displacement_bound(qs[i], qs[j]):
            continue
        if samples >= max_samples or times[j] - times[i] < 1e-12:
            certified = False
            continue
        t_mid = 0.5 * (times[i] + times[j])
        q_mid = np.asarray(q_of_t(t_mid), dtype=float)
        c_mid = model.clearance(q_mid, obstacles)
        times.append(t_mid)
        qs.append(q_mid)
        cs.append(c_mid)
        samples += 1
        m = len(times) - 1
        if c_mid < 0.0:
            collision_times.append(t_mid)
        pending.extend([(i, m), (m, j)])

    collision = bool(collision_times)
    return PathCheck(
        collision=collision,
        certified=certified and not collision,
        min_clearance=float(min(cs)),
        first_collision_time=float(min(collision_times)) if collision else None,
        samples=samples,
    )

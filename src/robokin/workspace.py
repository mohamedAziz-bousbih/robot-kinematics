"""Monte Carlo workspace estimation.

Joint configurations are drawn uniformly from the joint-limit box and pushed
through the batched forward kinematics. The reachable workspace is then the
set of resulting TCP positions; its size is estimated by counting occupied
grid cells (2-D) or voxels (3-D).

Uniform sampling in joint space is *not* uniform in Cartesian space: regions
near the outer boundary are reached only by nearly stretched configurations
and are therefore sampled sparsely. Occupancy counting has two opposing
biases. Cells that no sample happened to hit are missing, which shrinks the
estimate and vanishes as the sample count grows; every cell that is only
partly inside the workspace counts fully, which inflates the estimate and
vanishes only as the cell size shrinks. Where an exact membership test
exists, :func:`hit_or_miss_volume` avoids both, and :func:`two_link_area`
gives an exact reference for 2R arms such as the SCARA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .dh import SerialRobot


@dataclass(frozen=True)
class WorkspaceSamples:
    """Joint samples and the TCP positions they produce."""

    q: np.ndarray  # (N, n)
    points: np.ndarray  # (N, 3)

    @property
    def max_reach(self) -> float:
        """Largest distance of a sampled TCP position from the base origin."""
        return float(np.linalg.norm(self.points, axis=1).max())


def sample_workspace(
    robot: SerialRobot, n_samples: int, rng: np.random.Generator, batch: int = 50_000
) -> WorkspaceSamples:
    """Sample ``n_samples`` configurations and their TCP positions.

    Batches bound the memory used by the (N, 4, 4) intermediate transforms.
    """
    q = robot.random_q(rng, n_samples)
    points = np.empty((n_samples, 3))
    for start in range(0, n_samples, batch):
        stop = min(start + batch, n_samples)
        points[start:stop] = robot.fk(q[start:stop])[:, :3, 3]
    return WorkspaceSamples(q=q, points=points)


def occupied_area(points_2d: np.ndarray, cell: float) -> float:
    """Area covered by the grid cells (edge ``cell``) that contain a point."""
    idx = np.floor(np.asarray(points_2d) / cell).astype(np.int64)
    return float(len(np.unique(idx, axis=0)) * cell * cell)


def occupied_volume(points_3d: np.ndarray, voxel: float) -> float:
    """Volume covered by the voxels (edge ``voxel``) that contain a point."""
    idx = np.floor(np.asarray(points_3d) / voxel).astype(np.int64)
    return float(len(np.unique(idx, axis=0)) * voxel**3)


def two_link_area(
    a1: float,
    a2: float,
    q1lim: tuple[float, float],
    q2lim: tuple[float, float],
    n_radii: int = 20001,
) -> float:
    """Exact reachable area of a planar 2R arm with joint limits (reference for Monte Carlo).

    For a distance ``r`` from the base the elbow angle is fixed up to its sign,
    ``cos q2 = (r^2 - a1^2 - a2^2) / (2 a1 a2)``. Every admissible ``q2`` lets
    joint 1 sweep the polar arc ``[q1min + psi, q1max + psi]`` with
    ``psi = atan2(a2 sin q2, a1 + a2 cos q2)``, so the area is the radial
    integral of ``r`` times the length of the union of at most two such arcs.
    The integrand is piecewise smooth; the trapezoidal rule on ``n_radii``
    points is accurate to well below the Monte Carlo noise it is compared with.
    """
    if not (q1lim[0] < q1lim[1] and q2lim[0] < q2lim[1]):
        raise ValueError("joint limits must be increasing intervals")
    width = q1lim[1] - q1lim[0]
    radii = np.linspace(abs(a1 - a2), a1 + a2, n_radii)
    arc = np.zeros_like(radii)
    for k, r in enumerate(radii):
        c2 = np.clip((r * r - a1 * a1 - a2 * a2) / (2.0 * a1 * a2), -1.0, 1.0)
        elbow = np.arccos(c2)
        offsets = []
        for q2 in (elbow, -elbow) if elbow > 0.0 else (0.0,):
            # Joint ranges may exceed one turn, so test every 2 pi alias of q2.
            if any(q2lim[0] <= q2 + 2.0 * np.pi * m <= q2lim[1] for m in range(-2, 3)):
                offsets.append(np.arctan2(a2 * np.sin(q2), a1 + a2 * np.cos(q2)))
        if not offsets or width >= 2.0 * np.pi:
            arc[k] = 2.0 * np.pi if offsets else 0.0
            continue
        if len(offsets) == 1:
            arc[k] = width
            continue
        # Two arcs of equal length: the second starts `gap` further on.
        gap = np.mod(offsets[1] - offsets[0], 2.0 * np.pi)
        overlap = max(0.0, width - gap) + max(0.0, gap + width - 2.0 * np.pi)
        arc[k] = min(2.0 * np.pi, 2.0 * width - overlap)
    f = radii * arc
    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(radii)))


def hit_or_miss_volume(
    contains: Callable[[np.ndarray], np.ndarray],
    lo: Sequence[float],
    hi: Sequence[float],
    n_samples: int,
    rng: np.random.Generator,
    batch: int = 500_000,
) -> tuple[float, float]:
    """Volume of the set ``{p : contains(p)}`` inside the box ``[lo, hi]``.

    Points are drawn uniformly in Cartesian space and classified by an exact
    membership test (e.g. :meth:`robokin.Puma560.wrist_reachable`). Unlike
    voxel occupancy of joint-space samples, this estimator has no boundary
    bias; with hit fraction ``f`` its standard error is
    ``V_box * sqrt(f (1 - f) / N)``. Returns ``(volume, standard_error)``.
    """
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    hits = 0
    for start in range(0, n_samples, batch):
        size = min(batch, n_samples - start)
        hits += int(np.count_nonzero(contains(rng.uniform(lo, hi, size=(size, len(lo))))))
    box = float(np.prod(hi - lo))
    f = hits / n_samples
    return box * f, box * float(np.sqrt(f * (1.0 - f) / n_samples))


def slab(points: np.ndarray, axis: int, center: float, half_width: float) -> np.ndarray:
    """Points whose coordinate ``axis`` lies within ``center +- half_width``.

    Used for cross-section plots: e.g. the x-z section of a 3-D workspace is
    the slab ``|y - y0| < h`` projected onto the x-z plane.
    """
    mask = np.abs(points[:, axis] - center) < half_width
    return points[mask]

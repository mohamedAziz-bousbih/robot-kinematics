"""Numeric inverse kinematics by damped least squares (Levenberg-Marquardt).

The update step solves

    dq = J^T (J J^T + lambda^2 I)^-1 e

where ``e`` stacks the position error and the orientation error (a rotation
vector, see :func:`robokin.transforms.orientation_error`) restricted to the
robot's task rows. The damping ``lambda`` bounds the step near singularities,
where the plain pseudo-inverse would demand huge joint motions. It is adapted
like in Levenberg-Marquardt: shrunk after a step that reduces the error and
grown (with the step rejected) otherwise, so the solver behaves like
Gauss-Newton far from singularities and like gradient descent near them.

Joint limits are handled in one of two ways (``limits`` argument):

``"clamp"``
    Every iterate stays inside the limits. A joint that sits on a limit and
    whose step would push it further out is locked (its Jacobian column
    removed) and the step is recomputed, so the free joints compensate
    instead of the clipped step being wasted. This is the right mode for a
    seed close to the answer (tracking), but from a far seed the iteration
    often ends pinned on a limit in a local minimum of the pose error.
``"wrap"``
    Revolute joints move freely during the iteration; the converged angles
    are then shifted by multiples of 2 pi into the limits, and the run only
    counts as a success if that is possible. This converges far more often
    from an arbitrary seed, but it may land on a configuration branch that
    the limits exclude. Prismatic joints are clamped in both modes because
    they have no periodicity.

The two failure modes are complementary, so :func:`ik_numeric` alternates
the modes over random restarts. Because a restart is cheap, a run is
abandoned as soon as it stalls (less than 1 % cost reduction over
``stall_window`` iterations) instead of crawling to ``max_iter``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .dh import SerialRobot
from .transforms import orientation_error, wrap_angle


@dataclass(frozen=True)
class IKResult:
    """Outcome of :func:`ik_dls` and :func:`ik_numeric`."""

    q: np.ndarray
    success: bool  # converged and within the joint limits
    iterations: int
    position_error: float
    orientation_error: float
    within_limits: bool


def pose_error(robot: SerialRobot, q: np.ndarray, T_goal: np.ndarray) -> np.ndarray:
    """6-vector ``[p_goal - p; rot_log(R_goal R^T)]`` in the world frame."""
    T = robot.fk(q)
    return np.concatenate([T_goal[:3, 3] - T[:3, 3], orientation_error(T_goal[:3, :3], T[:3, :3])])


def ik_dls(
    robot: SerialRobot,
    T_goal: np.ndarray,
    q0: np.ndarray,
    *,
    limits: str = "clamp",
    tol_pos: float = 1e-10,
    tol_rot: float = 1e-10,
    max_iter: int = 500,
    damping: float = 1e-2,
    min_damping: float = 1e-6,
    max_damping: float = 1e3,
    stall_window: int = 20,
) -> IKResult:
    """Solve ``fk(q) = T_goal`` iteratively, starting from ``q0``.

    ``limits`` is ``"clamp"`` or ``"wrap"`` (see the module docstring). The
    solver reports failure instead of raising if it gets stuck (damping hits
    ``max_damping``, or the cost drops by less than 1 % over
    ``stall_window`` iterations; 0 disables that test), runs out of
    iterations, or converges to a configuration outside the joint limits.
    """
    if limits not in ("clamp", "wrap"):
        raise ValueError(f"unknown limit mode {limits!r}")
    rows = list(robot.task_rows)
    pos_rows = [r for r in rows if r < 3]
    rot_rows = [r for r in rows if r >= 3]
    lim = robot.qlim
    if limits == "wrap":
        lim[robot.revolute] = (-np.inf, np.inf)
    q = np.clip(np.asarray(q0, dtype=float), lim[:, 0], lim[:, 1])
    e = pose_error(robot, q, T_goal)
    cost = float(e[rows] @ e[rows])
    lam = damping
    history = [cost]

    def converged(err: np.ndarray) -> bool:
        return bool(
            np.linalg.norm(err[pos_rows]) <= tol_pos and np.linalg.norm(err[rot_rows]) <= tol_rot
        )

    it = 0
    while it < max_iter and not converged(e):
        it += 1
        dq = _limited_step(robot.jacobian(q)[rows], e[rows], lam, q, lim)
        q_new = np.clip(q + dq, lim[:, 0], lim[:, 1])
        e_new = pose_error(robot, q_new, T_goal)
        cost_new = float(e_new[rows] @ e_new[rows])
        if cost_new < cost:
            q, e, cost = q_new, e_new, cost_new
            lam = max(lam * 0.5, min_damping)
        else:
            lam *= 4.0
            if lam > max_damping:
                break
        history.append(cost)
        if stall_window and it >= stall_window and cost > 0.99 * history[it - stall_window]:
            break

    within = True
    if limits == "wrap":
        q[robot.revolute] = wrap_angle(q[robot.revolute])
        shifted = robot.wrap_to_limits(q, ref=q0)
        within = shifted is not None
        q = shifted if within else q
    ok = converged(e)
    return IKResult(
        q=q,
        success=ok and within,
        iterations=it,
        position_error=float(np.linalg.norm(e[pos_rows])),
        orientation_error=float(np.linalg.norm(e[rot_rows])) if rot_rows else 0.0,
        within_limits=within,
    )


def _limited_step(J: np.ndarray, e: np.ndarray, lam: float, q: np.ndarray, lim: np.ndarray) -> np.ndarray:
    """Damped least-squares step with joints on a limit locked if they push outwards."""
    free = np.ones(J.shape[1], dtype=bool)
    damping = (lam * lam) * np.eye(J.shape[0])
    while True:
        Jf = J * free  # locked joints contribute a zero column
        dq = Jf.T @ np.linalg.solve(Jf @ Jf.T + damping, e)
        blocked = free & (((q <= lim[:, 0]) & (dq < 0.0)) | ((q >= lim[:, 1]) & (dq > 0.0)))
        if not blocked.any():
            return dq
        free &= ~blocked


def ik_numeric(
    robot: SerialRobot,
    T_goal: np.ndarray,
    q0: np.ndarray,
    rng: np.random.Generator,
    *,
    restarts: int = 20,
    **kwargs: float,
) -> IKResult:
    """:func:`ik_dls` from ``q0``, then from random seeds until one succeeds.

    This is the fallback for chains without a closed-form solver. The run
    from ``q0`` clamps the limits, so a good seed keeps its configuration
    branch. Restart seeds are drawn uniformly from the joint limits and the
    restarts alternate between ``"wrap"`` and ``"clamp"``. The first
    successful run is returned; if none succeeds, the in-limit run with the
    smallest remaining position error is returned with ``success=False``.
    ``iterations`` counts the iterations of all runs.
    """
    best = ik_dls(robot, T_goal, q0, limits="clamp", **kwargs)
    total = best.iterations
    for k in range(restarts):
        if best.success:
            break
        mode = "wrap" if k % 2 == 0 else "clamp"
        res = ik_dls(robot, T_goal, robot.random_q(rng), limits=mode, **kwargs)
        total += res.iterations
        if res.success or _rank(res) < _rank(best):
            best = res
    return replace(best, iterations=total)


def _rank(res: IKResult) -> tuple[bool, float]:
    """Sort key for failed runs: in-limit results first, then by position error."""
    return (not res.within_limits, res.position_error)

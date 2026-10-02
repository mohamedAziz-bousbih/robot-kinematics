"""Closed-form versus numeric inverse kinematics on random reachable poses.

For each robot, configurations are drawn uniformly inside the joint limits and
their TCP poses computed by forward kinematics, so every target is reachable.
The closed-form solver must return the source configuration among its
branches. The damped-least-squares (DLS) solver is run once from a seed near
the source, once from a random seed in each joint-limit mode ("clamp" and
"wrap"), and as :func:`robokin.ik_numeric` with random restarts. Prints
success rates, accuracy and wall-clock time per pose. Timings depend on the
machine and are only meant for comparing the methods with each other.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from robokin import Planar3R, Puma560, Scara, ik_dls, ik_numeric
from robokin.transforms import wrap_angle

SEED = 11


def joint_distance(robot, q_a: np.ndarray, q_b: np.ndarray) -> float:
    d = q_a - q_b
    d[robot.revolute] = wrap_angle(d[robot.revolute])
    return float(np.max(np.abs(d)))


def benchmark(robot, n_poses: int, rng: np.random.Generator) -> dict[str, float]:
    qs = robot.random_q(rng, n_poses)
    targets = robot.fk(qs)

    start = time.perf_counter()
    solutions = [robot.ik(T) for T in targets]
    t_closed = (time.perf_counter() - start) / n_poses
    found = [min(joint_distance(robot, s.q, q) for s in sols) < 1e-8 for sols, q in zip(solutions, qs)]
    fk_error = max(np.max(np.abs(robot.fk(s.q) - T)) for sols, T in zip(solutions, targets) for s in sols)

    # Seeds near the source: revolute joints within +-0.1 rad, prismatic within +-1 cm.
    scale = np.where(robot.revolute, 0.1, 0.01)
    near = [ik_dls(robot, T, robot.clip_to_limits(q + rng.uniform(-1, 1, robot.n) * scale))
            for T, q in zip(targets, qs)]
    clamp = [ik_dls(robot, T, robot.random_q(rng), limits="clamp") for T in targets]
    wrap = [ik_dls(robot, T, robot.random_q(rng), limits="wrap") for T in targets]
    start = time.perf_counter()
    restarted = [ik_numeric(robot, T, robot.random_q(rng), rng) for T in targets]
    t_numeric = (time.perf_counter() - start) / n_poses

    return {
        "closed_form_found_source": float(np.mean(found)),
        "closed_form_mean_branches": float(np.mean([len(s) for s in solutions])),
        "closed_form_max_fk_error": float(fk_error),
        "closed_form_ms": 1e3 * t_closed,
        "dls_near_seed_success": float(np.mean([r.success for r in near])),
        "dls_clamp_random_seed_success": float(np.mean([r.success for r in clamp])),
        "dls_wrap_random_seed_success": float(np.mean([r.success for r in wrap])),
        "dls_wrap_converged_outside_limits": float(
            np.mean([not r.within_limits and r.position_error < 1e-9 and r.orientation_error < 1e-9
                     for r in wrap])
        ),
        "dls_random_seed_median_iter": float(
            np.median([r.iterations for r in clamp + wrap if r.success])
        ),
        "restarts_success": float(np.mean([r.success for r in restarted])),
        "restarts_ms": 1e3 * t_numeric,
    }


def main(n_poses: int = 200) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(SEED)
    return {robot.name: benchmark(robot, n_poses, rng) for robot in (Planar3R(), Scara(), Puma560())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poses", type=int, default=200)
    args = parser.parse_args()
    results = main(args.poses)
    print(f"{args.poses} random reachable poses per robot, seed {SEED}")
    for name, r in results.items():
        print(f"{name}:")
        print(f"  closed form: source found {100 * r['closed_form_found_source']:.1f} %, "
              f"{r['closed_form_mean_branches']:.2f} branches in limits on average, "
              f"max |FK(q) - T| {r['closed_form_max_fk_error']:.1e}, {r['closed_form_ms']:.2f} ms/pose")
        print(f"  DLS clamp, seed within 0.1 rad / 1 cm : {100 * r['dls_near_seed_success']:.1f} % solved")
        print(f"  DLS clamp, random seed                : {100 * r['dls_clamp_random_seed_success']:.1f} % solved")
        print(f"  DLS wrap,  random seed                : {100 * r['dls_wrap_random_seed_success']:.1f} % solved, "
              f"{100 * r['dls_wrap_converged_outside_limits']:.1f} % converged to a branch outside the limits")
        print(f"  median iterations of successful random-seed runs: {r['dls_random_seed_median_iter']:.0f}")
        print(f"  ik_numeric, <= 20 restarts            : {100 * r['restarts_success']:.1f} % solved, "
              f"{r['restarts_ms']:.1f} ms/pose")

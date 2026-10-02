"""Path check of two PUMA 560 programs against a spherical obstacle.

A direct PTP move sweeps the wrist through a sphere and the checker reports
the collision. Adding a via point that lifts the forearm avoids the sphere, and
the checker certifies the whole program collision-free. Writes
``collision_check.png`` and prints the checker results next to a dense 1 ms
reference sampling. Joint speed limits are illustrative values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from robokin import Program, Puma560, Sphere  # noqa: E402
from robokin import plotting as rp  # noqa: E402

VMAX = np.array([2.0, 1.5, 2.5, 4.0, 4.0, 5.0])  # rad/s
AMAX = np.array([4.0, 3.0, 5.0, 10.0, 10.0, 12.0])  # rad/s^2
Q_A = np.deg2rad([-60.0, 45.0, -125.0, 0.0, 30.0, 0.0])
Q_B = np.deg2rad([60.0, 45.0, -125.0, 0.0, 30.0, 0.0])
Q_VIA = np.deg2rad([0.0, 45.0, -90.0, 0.0, 30.0, 0.0])  # forearm raised by 35 deg
OBSTACLE = Sphere((0.70, -0.15, 0.30), 0.08)
DENSE_DT = 1e-3


def run(robot: Puma560, program: Program) -> dict:
    model = robot.collision_model()
    check = program.check_collisions(model, [OBSTACLE])
    t, q = program.sample(DENSE_DT)
    clearance = np.array([model.clearance(qk, [OBSTACLE]) for qk in q])
    hits = np.flatnonzero(clearance < 0.0)
    return {
        "check": check,
        "t": t,
        "q": q,
        "clearance": clearance,
        "dense_min_clearance": float(clearance.min()),
        "dense_contact_time": float(t[hits[0]]) if hits.size else None,
    }


def main(out: Path) -> dict[str, dict]:
    rp.apply_style()
    robot = Puma560()
    direct = Program(robot, Q_A, VMAX, AMAX).ptp(Q_B, "direct")
    detour = Program(robot, Q_A, VMAX, AMAX).ptp(Q_VIA, "via").ptp(Q_B, "goal")
    results = {"direct": run(robot, direct), "via point": run(robot, detour)}
    colors = {"direct": rp.CRITICAL, "via point": rp.SERIES[0]}

    fig = plt.figure(figsize=(12.5, 4.8), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    rp.draw_sphere_3d(ax3d, OBSTACLE)
    for name, res in results.items():
        tcp = np.array([robot.fk(qk)[:3, 3] for qk in res["q"][::20]])
        ax3d.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], color=colors[name], lw=1.4, label=f"TCP path, {name}")
    hit_time = results["direct"]["check"].first_collision_time
    rp.draw_arm_3d(ax3d, robot, direct(hit_time), color=rp.CRITICAL, alpha=0.8)
    rp.draw_arm_3d(ax3d, robot, Q_VIA, color=rp.SERIES[0], alpha=0.8)
    for q in (Q_A, Q_B):
        rp.draw_arm_3d(ax3d, robot, q, color=rp.MUTED, alpha=0.6)
    rp.set_equal_3d(ax3d, (0.3, -0.05, 0.25), 0.55)
    rp.tidy_3d(ax3d, (-0.2, 0.2, 0.6))
    ax3d.view_init(elev=28, azim=-35)
    ax3d.legend(loc="upper left")
    ax3d.set_title("grey: start/goal; red: direct PTP at its first colliding sample;\n"
                   "blue: via-point configuration", fontsize=9)

    ax = fig.add_subplot(1, 2, 2)
    for name, res in results.items():
        check = res["check"]
        verdict = "collision" if check.collision else ("certified free" if check.certified else "not certified")
        label = f"{name}: {verdict}, {check.samples} checker samples"
        ax.plot(res["t"], 100.0 * res["clearance"], color=colors[name], label=label)
    ax.axhline(0.0, color=rp.INK_SECONDARY, lw=0.9)
    ax.axvline(hit_time, color=rp.CRITICAL, lw=0.8, ls="--")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("clearance arm to sphere [cm]")
    ax.set_title("clearance on a dense 1 ms grid; dashed: first colliding checker sample", fontsize=9)
    ax.legend(loc="upper right")
    rp.save(fig, out / "collision_check.png")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "img")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, res in main(args.out).items():
        c = res["check"]
        print(f"{name}:")
        print(f"  duration                    : {res['t'][-1]:.3f} s")
        print(f"  collision / certified       : {c.collision} / {c.certified}")
        print(f"  checker samples             : {c.samples}")
        print(f"  first colliding sample      : {c.first_collision_time}")
        print(f"  min sampled clearance       : {c.min_clearance * 100:.2f} cm")
        print(f"  dense 1 ms: min clearance   : {res['dense_min_clearance'] * 100:.2f} cm")
        print(f"  dense 1 ms: contact from    : {res['dense_contact_time']}")

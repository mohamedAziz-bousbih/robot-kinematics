"""All eight closed-form IK solutions of the PUMA 560 for one TCP pose.

Writes ``ik_puma_solutions.png`` and prints the joint angles of every branch.
Branches that exist geometrically but violate a joint limit are drawn grey.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from robokin import Puma560  # noqa: E402
from robokin import plotting as rp  # noqa: E402
from robokin.transforms import homogeneous, rpy_to_rot  # noqa: E402

POSITION = (0.45, -0.30, 0.20)  # m, in frame 0 (shoulder)
RPY_DEG = (0.0, 90.0, 0.0)  # tool z axis pointing along +x


def main(out: Path) -> list[tuple[str, np.ndarray, bool, float]]:
    rp.apply_style()
    robot = Puma560()
    T = homogeneous(rpy_to_rot(*np.deg2rad(RPY_DEG)), POSITION)
    solutions = robot.ik(T, respect_limits=False)
    rows = []
    fig = plt.figure(figsize=(13, 6.8))
    for k, sol in enumerate(solutions):
        within = robot.within_limits(sol.q)
        err = float(np.max(np.abs(robot.fk(sol.q) - T)))
        rows.append((sol.branch, sol.q, within, err))
        ax = fig.add_subplot(2, 4, k + 1, projection="3d")
        color = rp.SERIES[0] if within else rp.MUTED
        ax.plot([0, 0], [0, 0], [-0.6, 0.0], color=rp.MUTED, lw=4)  # pedestal
        rp.draw_arm_3d(ax, robot, sol.q, color=color)
        # The tool sits at the wrist centre, so a fin along x4 is what shows
        # the wrist flip: q4 + pi turns it by half a revolution.
        frame4 = robot.fk_frames(sol.q)[4]
        fin = np.array([frame4[:3, 3], frame4[:3, 3] + 0.1 * frame4[:3, 0]])
        ax.plot(fin[:, 0], fin[:, 1], fin[:, 2], color=color, lw=1.5)
        rp.draw_frame_3d(ax, T, length=0.12)
        rp.set_equal_3d(ax, (0.1, -0.1, 0.0), 0.6)
        rp.tidy_3d(ax, (-0.4, 0.0, 0.4))
        ax.view_init(elev=18, azim=-58)
        q456 = ", ".join(f"{a:.0f}" for a in np.rad2deg(sol.q[3:]))
        title = f"{sol.branch}\nq4, q5, q6 = {q456} deg" + ("" if within else "\noutside joint limits")
        ax.set_title(title, fontsize=8.5, color=rp.INK if within else rp.MUTED)
    fig.suptitle(
        f"PUMA 560: {len(solutions)} IK branches for p = {POSITION} m, RPY = {RPY_DEG} deg. "
        "Target frame: x orange, y aqua, z black; short stub: x axis of frame 4",
        fontsize=10, color=rp.INK,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.02, hspace=0.2)
    rp.save(fig, out / "ik_puma_solutions.png")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "img")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{'branch':<20} {'q1..q6 [deg]':<56} limits  max |FK(q) - T|")
    for branch, q, within, err in main(args.out):
        angles = " ".join(f"{a:8.2f}" for a in np.rad2deg(q))
        print(f"{branch:<20} {angles:<56} {'ok' if within else 'out':<7} {err:.1e}")

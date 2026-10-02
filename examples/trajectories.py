"""PTP and LIN motions of the PUMA 560.

Writes ``ptp_profiles.png`` (fully synchronised trapezoid vs. quintic PTP) and
``lin_motion.png`` (straight-line TCP move with SLERP orientation, tracked
with the closed-form IK) and prints the durations of all PTP variants.
The joint speed and acceleration limits are illustrative values, not
manufacturer data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from robokin import Puma560, plan_lin, plan_ptp, track_lin  # noqa: E402
from robokin import plotting as rp  # noqa: E402
from robokin.transforms import homogeneous, rot_log, rpy_to_rot  # noqa: E402

VMAX = np.array([2.0, 1.5, 2.5, 4.0, 4.0, 5.0])  # rad/s
AMAX = np.array([4.0, 3.0, 5.0, 10.0, 10.0, 12.0])  # rad/s^2
Q_START = Puma560.QN
Q_GOAL = np.deg2rad([90.0, 20.0, -150.0, 60.0, -30.0, 120.0])

LIN_START = homogeneous(rpy_to_rot(*np.deg2rad([180.0, 0.0, 0.0])), [0.45, -0.25, -0.20])
LIN_GOAL = homogeneous(rpy_to_rot(*np.deg2rad([180.0, 20.0, 45.0])), [0.35, 0.25, -0.05])
LIN_LIMITS = dict(v_max=0.5, a_max=1.0, w_max=1.0, alpha_max=2.0)  # m/s, m/s^2, rad/s, rad/s^2
JOINTS = [f"q{i}" for i in range(1, 7)]


def ptp_figure(out: Path) -> dict[str, float]:
    variants = {
        (profile, sync): plan_ptp(Q_START, Q_GOAL, VMAX, AMAX, profile=profile, sync=sync)
        for profile in ("trapezoid", "quintic")
        for sync in ("async", "sync", "full")
    }
    shown = [("trapezoid", "full"), ("quintic", "full")]
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 7.4), sharex=True, constrained_layout=True)
    t_end = max(variants[key].duration for key in shown)
    for col, key in enumerate(shown):
        traj = variants[key]
        t = np.linspace(0.0, traj.duration, 1500)
        q, qd, qdd = traj.sample(t)
        for j in range(6):
            axes[0, col].plot(t, np.rad2deg(q[:, j]), label=JOINTS[j])
            axes[1, col].plot(t, qd[:, j] / VMAX[j])
            axes[2, col].plot(t, qdd[:, j] / AMAX[j])
        axes[0, col].set_title(f"{key[0]}, fully synchronised: T = {traj.duration:.3f} s")
        for row in (1, 2):
            axes[row, col].set_ylim(-1.1, 1.1)
            axes[row, col].axhline(0.0, color=rp.AXIS, lw=0.8)
        axes[2, col].set_xlabel("t [s]")
        axes[2, col].set_xlim(0.0, t_end)
    axes[0, 0].set_ylabel("joint angle [deg]")
    axes[1, 0].set_ylabel("velocity / vmax")
    axes[2, 0].set_ylabel("acceleration / amax")
    axes[0, 1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    rp.save(fig, out / "ptp_profiles.png")
    return {f"{p}/{s}": traj.duration for (p, s), traj in variants.items()}


def lin_figure(out: Path) -> dict[str, float]:
    robot = Puma560()
    cart = plan_lin(LIN_START, LIN_GOAL, **LIN_LIMITS)
    q_start = robot.ik_closest(LIN_START, Q_START).q
    joint = track_lin(robot, cart, q_start, dt=0.005)
    poses = np.array([robot.fk(q) for q in joint.q])

    fig = plt.figure(figsize=(13, 4.4), constrained_layout=True)
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    snapshots = np.linspace(0, len(joint.t) - 1, 6).astype(int)
    for k, alpha in zip(snapshots, np.linspace(0.3, 1.0, len(snapshots))):  # later = more opaque
        rp.draw_arm_3d(ax3d, robot, joint.q[k], color=rp.SERIES[0], alpha=alpha)
        rp.draw_frame_3d(ax3d, poses[k], length=0.07)
    ax3d.plot(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3], color=rp.INK, lw=1.2)
    rp.set_equal_3d(ax3d, (0.25, 0.0, 0.1), 0.5)
    rp.tidy_3d(ax3d, (-0.2, 0.2, 0.6))
    ax3d.view_init(elev=25, azim=-70)
    ax3d.set_title("TCP line (black) and 6 arm snapshots\nframes: x orange, y aqua, z black", fontsize=9)

    # Cartesian speeds from the tracked joint path, i.e. what the arm really does.
    dt = np.diff(joint.t)
    speed = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) / dt
    omega = np.array(
        [np.linalg.norm(rot_log(b[:3, :3] @ a[:3, :3].T)) for a, b in zip(poses[:-1], poses[1:])]
    ) / dt
    tm = 0.5 * (joint.t[1:] + joint.t[:-1])
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(tm, speed / LIN_LIMITS["v_max"], label="|v| / v_max")
    ax.plot(tm, omega / LIN_LIMITS["w_max"], label="|w| / w_max")
    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel("t [s]")
    ax.set_title(f"TCP speed, duration {cart.duration:.3f} s", fontsize=9)
    ax.legend(loc="lower center")

    ax = fig.add_subplot(1, 3, 3)
    qd = joint.velocities()
    for j in range(6):
        ax.plot(joint.t, qd[:, j] / VMAX[j], label=JOINTS[j])
    ratio = float(np.max(np.abs(qd) / VMAX))
    ax.axhline(0.0, color=rp.AXIS, lw=0.8)
    ax.set_xlabel("t [s]")
    ax.set_title(f"joint velocity / vmax (peak {ratio:.2f})", fontsize=9)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    rp.save(fig, out / "lin_motion.png")
    return {
        "lin_duration_s": cart.duration,
        "lin_peak_speed_ratio": float(speed.max() / LIN_LIMITS["v_max"]),
        "lin_peak_omega_ratio": float(omega.max() / LIN_LIMITS["w_max"]),
        "lin_peak_joint_ratio": ratio,
        "lin_samples": len(joint.t),
    }


def main(out: Path) -> dict[str, float]:
    rp.apply_style()
    return {**ptp_figure(out), **lin_figure(out)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "img")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in main(args.out).items():
        print(f"{name:<24} {value:.4f}" if isinstance(value, float) else f"{name:<24} {value}")

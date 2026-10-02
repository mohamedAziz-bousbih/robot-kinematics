"""Command-line interface: DH tables, forward/inverse kinematics and workspace estimates.

Angles on the command line and in the output are in degrees; prismatic joint
values and positions are in metres. Examples::

    robokin dh puma560
    robokin fk puma560 0 45 -180 0 45 0
    robokin ik puma560 0.5 -0.2 -0.1 180 0 0
    robokin workspace scara --samples 1000000
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Sequence

import numpy as np

from .robots import AnalyticRobot, Planar3R, Puma560, Scara
from .transforms import homogeneous, rot_to_quat, rot_to_rpy, rpy_to_rot
from .workspace import occupied_area, occupied_volume, sample_workspace

ROBOTS: dict[str, Callable[[], AnalyticRobot]] = {
    "planar3r": Planar3R,
    "scara": Scara,
    "puma560": Puma560,
}

# Below this mean sample count per occupied cell the CLI warns about undersampling.
MIN_SAMPLES_PER_CELL = 10


def _to_internal(robot: AnalyticRobot, values: Sequence[float]) -> np.ndarray:
    """Degrees to radians for revolute joints; prismatic values stay in metres."""
    q = np.array(values, dtype=float)
    q[robot.revolute] = np.deg2rad(q[robot.revolute])
    return q


def _to_display(robot: AnalyticRobot, q: np.ndarray) -> np.ndarray:
    out = np.array(q, dtype=float)
    out[robot.revolute] = np.rad2deg(out[robot.revolute])
    return out


def _fmt(values: Sequence[float], width: int = 9, prec: int = 3) -> str:
    # Rounding first (and adding 0.0) avoids printing round-off as "-0.000".
    return " ".join(f"{round(float(v), prec) + 0.0:{width}.{prec}f}" for v in values)


def cmd_dh(robot: AnalyticRobot, _args: argparse.Namespace) -> int:
    print(f"{robot.name}: standard DH parameters (* = joint variable)")
    print(f"{'joint':>5} {'type':>4} {'theta [deg]':>12} {'d [m]':>9} {'a [m]':>9} "
          f"{'alpha [deg]':>12} {'limits':>20}")
    for row in robot.dh_table():
        prismatic = row["type"] == "P"
        theta = f"{np.rad2deg(row['theta']):.1f}" + ("" if prismatic else " + q*")
        d = f"{row['d']:.5g}" + (" + q*" if prismatic else "")
        scale = 1.0 if prismatic else 180.0 / np.pi
        unit = "m" if prismatic else "deg"
        limits = f"[{row['qmin'] * scale:.4g}, {row['qmax'] * scale:.4g}] {unit}"
        print(f"{row['joint']:>5} {row['type']:>4} {theta:>12} {d:>9} {row['a']:>9.5g} "
              f"{np.rad2deg(row['alpha']):>12.1f} {limits:>20}")
    return 0


def cmd_fk(robot: AnalyticRobot, args: argparse.Namespace) -> int:
    if len(args.q) != robot.n:
        print(f"error: {robot.name} needs {robot.n} joint values, got {len(args.q)}", file=sys.stderr)
        return 2
    q = _to_internal(robot, args.q)
    if not robot.within_limits(q):
        print("warning: configuration is outside the joint limits", file=sys.stderr)
    T = robot.fk(q)
    print("T =")
    for row in T:
        print("  " + _fmt(row, 10, 5))
    print(f"position [m]          : {_fmt(T[:3, 3])}")
    print(f"roll pitch yaw [deg]  : {_fmt(np.rad2deg(rot_to_rpy(T[:3, :3])))}")
    print(f"quaternion (w x y z)  : {_fmt(rot_to_quat(T[:3, :3]), 9, 5)}")
    print(f"manipulability        : {robot.manipulability(q):.5g}")
    if robot.is_singular(q):
        kinds = ", ".join(sorted(robot.singularity_kinds(q)))
        print(f"singular              : yes ({kinds})")
    else:
        print("singular              : no")
    return 0


def cmd_ik(robot: AnalyticRobot, args: argparse.Namespace) -> int:
    x, y, z, roll, pitch, yaw = args.pose
    T = homogeneous(rpy_to_rot(*np.deg2rad([roll, pitch, yaw])), [x, y, z])
    sols = robot.ik(T, respect_limits=not args.no_limits)
    if not sols:
        print("no solution: pose is outside the workspace or the joint limits")
        return 1
    print(f"{len(sols)} solution(s) for {robot.name}")
    for sol in sols:
        within = "" if robot.within_limits(sol.q) else "  (outside limits)"
        print(f"  {sol.branch:<22} q = {_fmt(_to_display(robot, sol.q), 9, 3)}{within}")
    return 0


def cmd_workspace(robot: AnalyticRobot, args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    ws = sample_workspace(robot, args.samples, rng)
    lo, hi = ws.points.min(axis=0), ws.points.max(axis=0)
    print(f"{robot.name}: {args.samples} joint-space samples (seed {args.seed})")
    print(f"  x range [m]      : {lo[0]: .4f} .. {hi[0]: .4f}")
    print(f"  y range [m]      : {lo[1]: .4f} .. {hi[1]: .4f}")
    print(f"  z range [m]      : {lo[2]: .4f} .. {hi[2]: .4f}")
    print(f"  max reach [m]    : {ws.max_reach:.4f}")
    mm = args.cell * 1000
    if np.ptp(ws.points[:, 2]) < 1e-12:
        size = occupied_area(ws.points[:, :2], args.cell)
        cells = size / args.cell**2
        print(f"  area [m^2]       : {size:.4f}  (occupied {mm:.0f} mm cells)")
    else:
        size = occupied_volume(ws.points, args.cell)
        cells = size / args.cell**3
        print(f"  volume [m^3]     : {size:.4f}  (occupied {mm:.0f} mm voxels)")
    density = args.samples / cells
    print(f"  samples per cell : {density:.1f}")
    # With only a few samples per cell, many interior cells are never hit and
    # the occupancy estimate is far too small (see robokin.workspace).
    if density < MIN_SAMPLES_PER_CELL:
        print(f"warning: fewer than {MIN_SAMPLES_PER_CELL} samples per occupied cell, the estimate is "
              "biased low; increase --samples or --cell", file=sys.stderr)
    return 0


def _positive(kind: type) -> Callable[[str], float]:
    """argparse type that also rejects zero and negative values."""

    def parse(text: str):
        value = kind(text)
        if value <= 0:
            raise argparse.ArgumentTypeError(f"must be positive, got {text}")
        return value

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robokin",
        description="Kinematics of a planar 3R arm, a SCARA and a PUMA 560. "
        "Angles are in degrees, lengths in metres.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    robot_arg = {"choices": sorted(ROBOTS), "help": "robot model"}

    p = sub.add_parser("dh", help="print the DH table")
    p.add_argument("robot", **robot_arg)
    p.set_defaults(func=cmd_dh)

    p = sub.add_parser("fk", help="forward kinematics of one configuration")
    p.add_argument("robot", **robot_arg)
    p.add_argument("q", type=float, nargs="+", help="joint values (deg, or m for prismatic)")
    p.set_defaults(func=cmd_fk)

    p = sub.add_parser("ik", help="all closed-form inverse kinematics solutions")
    p.add_argument("robot", **robot_arg)
    p.add_argument("pose", type=float, nargs=6, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
                   help="TCP position [m] and ZYX roll/pitch/yaw [deg]")
    p.add_argument("--no-limits", action="store_true", help="also list solutions outside the limits")
    p.set_defaults(func=cmd_ik)

    p = sub.add_parser("workspace", help="Monte Carlo workspace estimate")
    p.add_argument("robot", **robot_arg)
    p.add_argument("--samples", type=_positive(int), default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cell", type=_positive(float), default=0.02, help="grid cell / voxel edge [m]")
    p.set_defaults(func=cmd_workspace)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(ROBOTS[args.robot](), args)


if __name__ == "__main__":
    raise SystemExit(main())

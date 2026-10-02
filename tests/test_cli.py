"""Command-line interface: output content and exit codes."""

import re

import numpy as np
import pytest

from robokin import Puma560
from robokin.cli import main
from robokin.transforms import homogeneous, rpy_to_rot


def numbers(line):
    return [float(v) for v in re.findall(r"-?\d+\.\d+", line)]


def test_dh_table_lists_every_joint(capsys):
    assert main(["dh", "puma560"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2 + 6
    assert "0.4318" in lines[3] and "[-45, 225] deg" in lines[3]
    assert main(["dh", "scara"]) == 0
    out = capsys.readouterr().out
    assert "0 + q*" in out and "[0, 0.2] m" in out  # prismatic joint 3


def test_fk_prints_the_zero_pose(capsys):
    assert main(["fk", "puma560", "0", "0", "0", "0", "0", "0"]) == 0
    out = capsys.readouterr().out
    pos = next(line for line in out.splitlines() if line.startswith("position"))
    np.testing.assert_allclose(numbers(pos), [0.452, -0.150, 0.432], atol=1e-3)
    # q5 = 0 aligns axes 4 and 6.
    assert "singular              : yes (wrist)" in out
    assert main(["fk", "planar3r", "10", "30", "-20"]) == 0
    assert "singular              : no" in capsys.readouterr().out


def test_fk_rejects_wrong_joint_count(capsys):
    assert main(["fk", "scara", "10", "20", "0.1"]) == 2
    assert "needs 4 joint values" in capsys.readouterr().err


def test_ik_solutions_reproduce_the_pose(capsys):
    assert main(["ik", "puma560", "0.5", "-0.2", "-0.1", "180", "0", "0", "--no-limits"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "8 solution(s) for PUMA 560"
    robot = Puma560()
    T = homogeneous(rpy_to_rot(np.pi, 0.0, 0.0), [0.5, -0.2, -0.1])
    for line in lines[1:]:
        q = np.deg2rad(numbers(line))
        assert len(q) == 6
        # Printed with 3 decimals of a degree, so compare at the 1e-4 level.
        np.testing.assert_allclose(robot.fk(q), T, atol=1e-4)
        assert ("outside limits" in line) != robot.within_limits(q, tol=1e-4)


def test_ik_out_of_reach_exit_code(capsys):
    assert main(["ik", "planar3r", "5", "0", "0", "0", "0", "0"]) == 1
    assert "no solution" in capsys.readouterr().out


def test_workspace_summary(capsys):
    assert main(["workspace", "planar3r", "--samples", "200000", "--seed", "3"]) == 0
    captured = capsys.readouterr()
    out = captured.out
    reach = numbers(next(line for line in out.splitlines() if "max reach" in line))[0]
    assert 1.0 < reach <= 1.1
    area = numbers(next(line for line in out.splitlines() if "area [m^2]" in line))[0]
    # Bounded by the disc of radius l1 + l2 + l3, up to boundary cells counted in full.
    assert 3.5 < area < np.pi * 1.1**2 + 0.1
    density = numbers(next(line for line in out.splitlines() if "samples per cell" in line))[0]
    assert density == pytest.approx(200000 / (area / 0.02**2), rel=1e-3)
    assert captured.err == ""
    assert main(["workspace", "scara", "--samples", "20000"]) == 0
    assert "volume [m^3]" in capsys.readouterr().out


def test_workspace_warns_when_undersampled(capsys):
    # 10 mm voxels in a 0.19 m^3 SCARA workspace: far fewer samples than voxels.
    assert main(["workspace", "scara", "--samples", "20000", "--cell", "0.01"]) == 0
    assert "biased low" in capsys.readouterr().err


@pytest.mark.parametrize("option", [["--samples", "0"], ["--cell", "-0.01"]])
def test_workspace_rejects_non_positive_options(option, capsys):
    with pytest.raises(SystemExit):
        main(["workspace", "scara", *option])
    assert "must be positive" in capsys.readouterr().err


def test_unknown_robot_is_rejected():
    with pytest.raises(SystemExit):
        main(["dh", "kuka"])

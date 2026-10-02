"""Motion programs: sequencing of PTP and LIN moves, continuity and collision checks."""

import numpy as np
import pytest

from robokin import Program, Puma560, Sphere
from robokin.transforms import homogeneous, rpy_to_rot

from conftest import assert_pose_close

VMAX = np.array([2.0, 1.5, 2.5, 4.0, 4.0, 5.0])
AMAX = np.array([4.0, 3.0, 5.0, 10.0, 10.0, 12.0])
DOWN = rpy_to_rot(np.pi, 0.0, 0.0)  # tool z axis pointing at the floor
ABOVE = homogeneous(DOWN, [0.5, -0.2, -0.1])
GRASP = homogeneous(DOWN, [0.5, -0.2, -0.3])


@pytest.fixture(params=["trapezoid", "quintic"])
def pick(request):
    robot = Puma560()
    prog = Program(robot, Puma560.QN, VMAX, AMAX, lin_limits=(0.5, 1.0, 1.0, 2.0),
                   dt=0.01, profile=request.param)
    prog.ptp(ABOVE, "above").lin(GRASP, "grasp").lin(ABOVE, "retract").ptp(Puma560.QN, "home")
    return robot, prog


def test_segments_are_contiguous(pick):
    _, prog = pick
    assert [s.kind for s in prog.segments] == ["PTP", "LIN", "LIN", "PTP"]
    assert prog.segments[0].t_start == 0.0
    for prev, seg in zip(prog.segments, prog.segments[1:]):
        assert seg.t_start == pytest.approx(prev.t_end)
    assert prog.duration == pytest.approx(prog.segments[-1].t_end)


def test_program_reaches_every_target(pick):
    robot, prog = pick
    np.testing.assert_allclose(prog(0.0), Puma560.QN, atol=1e-12)
    targets = [ABOVE, GRASP, ABOVE]
    for seg, T in zip(prog.segments, targets):
        assert_pose_close(robot.fk(prog(seg.t_end)), T, atol=1e-9)
    np.testing.assert_allclose(prog(prog.duration), Puma560.QN, atol=1e-12)
    # Times outside the program are clamped to its ends.
    np.testing.assert_allclose(prog(-1.0), Puma560.QN, atol=1e-12)
    np.testing.assert_allclose(prog(prog.duration + 1.0), Puma560.QN, atol=1e-12)


def test_joint_path_is_continuous_and_ptp_respects_vmax(pick):
    _, prog = pick
    dt = 1e-3
    t, q = prog.sample(dt)
    step = np.abs(np.diff(q, axis=0))
    assert step.max() < 0.02  # no jumps, also not at segment boundaries
    for seg in prog.segments:
        if seg.kind != "PTP":
            continue
        inside = (t >= seg.t_start) & (t <= seg.t_end)
        qd = np.abs(np.diff(q[inside], axis=0)) / np.diff(t[inside])[:, None]
        assert np.all(qd <= VMAX * (1 + 1e-6))


def test_lin_segments_move_the_tcp_vertically(pick):
    robot, prog = pick
    for seg in prog.segments:
        if seg.kind != "LIN":
            continue
        for tk in seg.t_start + seg.motion.t:
            T = robot.fk(prog(tk))
            np.testing.assert_allclose(T[:2, 3], [0.5, -0.2], atol=1e-9)
            np.testing.assert_allclose(T[:3, :3], DOWN, atol=1e-9)
        assert prog.lin_speed_ratio() > 0.0


def test_breakpoints_cover_segment_ends_and_lin_knots(pick):
    _, prog = pick
    bp = prog.breakpoints()
    for seg in prog.segments:
        assert np.min(np.abs(bp - seg.t_end)) < 1e-12
    lin = prog.segments[1]
    for tk in lin.t_start + lin.motion.t:
        assert np.min(np.abs(bp - tk)) < 1e-12


def test_program_collision_check(pick):
    robot, prog = pick
    model = robot.collision_model()
    # A small sphere on the vertical approach path is hit by the wrist.
    hit = Sphere((0.5, -0.2, -0.25), 0.03)
    res = prog.check_collisions(model, [hit])
    assert res.collision and not res.certified
    grasp = prog.segments[1]
    assert grasp.t_start <= res.first_collision_time <= prog.segments[2].t_end
    far = Sphere((-1.0, 1.0, 1.0), 0.1)
    res = prog.check_collisions(model, [far])
    assert res.certified and not res.collision


def test_lin_without_limits_is_rejected():
    prog = Program(Puma560(), Puma560.QN, VMAX, AMAX)
    prog.ptp(ABOVE)
    with pytest.raises(ValueError, match="lin_limits"):
        prog.lin(GRASP)


def test_unreachable_targets_raise():
    prog = Program(Puma560(), Puma560.QN, VMAX, AMAX, lin_limits=(0.5, 1.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="not reachable"):
        prog.ptp(homogeneous(DOWN, [1.5, 0.0, 0.0]), "far away")
    prog.ptp(ABOVE)
    with pytest.raises(ValueError, match="not reachable"):
        prog.lin(homogeneous(DOWN, [1.5, 0.0, 0.0]))


def test_lin_to_the_current_pose_is_a_zero_length_move():
    """A LIN that does not move must not produce NaN velocities (zero time steps)."""
    robot = Puma560()
    prog = Program(robot, Puma560.QN, VMAX, AMAX, lin_limits=(0.5, 1.0, 1.0, 2.0))
    prog.lin(robot.fk(Puma560.QN), "stay")
    seg = prog.segments[0]
    assert seg.motion.duration == 0.0 and len(seg.motion.t) == 1
    np.testing.assert_allclose(seg.motion.velocities(), 0.0)
    assert prog.lin_speed_ratio() == 0.0
    np.testing.assert_allclose(prog(0.0), Puma560.QN, atol=1e-12)
    # The program goes on normally after the empty move.
    prog.ptp(ABOVE, "above")
    assert prog.duration > 0.0
    assert_pose_close(robot.fk(prog(prog.duration)), ABOVE, atol=1e-9)

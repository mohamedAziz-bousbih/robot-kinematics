"""Capsule geometry, the displacement bound, and sampled path checks."""

import numpy as np
import pytest

from robokin import CollisionModel, LinkCapsule, Planar3R, Puma560, Sphere, check_motion, check_samples, plan_ptp
from robokin.collision import capsule_sphere_clearance, point_segment_distance


def test_point_segment_distance_cases():
    a, b = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    assert point_segment_distance(np.array([0.5, 2.0, 0.0]), a, b) == pytest.approx(2.0)  # interior
    assert point_segment_distance(np.array([-3.0, 4.0, 0.0]), a, b) == pytest.approx(5.0)  # beyond a
    assert point_segment_distance(np.array([2.0, 0.0, 0.0]), a, b) == pytest.approx(1.0)  # beyond b
    assert point_segment_distance(np.array([0.0, 3.0, 4.0]), a, a) == pytest.approx(5.0)  # degenerate


def test_capsule_sphere_clearance_sign():
    a, b = np.zeros(3), np.array([1.0, 0.0, 0.0])
    assert capsule_sphere_clearance(a, b, 0.1, Sphere((0.5, 0.5, 0.0), 0.2)) == pytest.approx(0.2)
    assert capsule_sphere_clearance(a, b, 0.1, Sphere((0.5, 0.25, 0.0), 0.2)) == pytest.approx(-0.05)


def test_capsules_follow_the_links():
    robot = Planar3R()
    q = np.array([0.3, -0.5, 0.9])
    frames = robot.fk_frames(q)
    segs = robot.collision_model().segments(q)
    for (a, b, _), (i, j) in zip(segs, [(0, 1), (1, 2), (2, 4)]):
        np.testing.assert_allclose(a, frames[i][:3, 3])
        np.testing.assert_allclose(b, frames[j][:3, 3])


def test_bad_capsule_frame_rejected():
    with pytest.raises(ValueError):
        CollisionModel(Planar3R(), [LinkCapsule("x", 0, 9, 0.1)])


def test_displacement_bound_holds(robot, rng):
    """No capsule end point moves further than the bound along a monotone joint path."""
    model = robot.collision_model()
    for _ in range(40):
        q0 = robot.random_q(rng)
        q1 = robot.clip_to_limits(q0 + rng.uniform(-0.3, 0.3, robot.n))
        bound = model.displacement_bound(q0, q1)
        start = model.segments(q0)
        for s in np.linspace(0, 1, 25):
            for (a0, b0, _), (a, b, _) in zip(start, model.segments(q0 + s * (q1 - q0))):
                assert np.linalg.norm(a - a0) <= bound + 1e-12
                assert np.linalg.norm(b - b0) <= bound + 1e-12


VMAX = np.full(6, 2.0)
AMAX = np.full(6, 4.0)


def ptp(q0, q1):
    return plan_ptp(q0, q1, VMAX, AMAX, sync="full")


def test_ptp_through_obstacle_is_detected():
    robot = Puma560()
    model = robot.collision_model()
    q0 = np.array([-1.0, 0.8, -2.2, 0.0, 0.5, 0.0])
    q1 = np.array([1.0, 0.8, -2.2, 0.0, 0.5, 0.0])
    traj = ptp(q0, q1)
    mid = robot.fk(traj(traj.duration / 2))[:3, 3]
    obstacle = Sphere(tuple(mid), 0.05)
    res = check_motion(model, [obstacle], traj, traj.duration)
    assert res.collision and not res.certified
    assert model.clearance(traj(res.first_collision_time), [obstacle]) < 0.0
    # Both ends are free: the collision is genuinely in the middle of the move.
    assert model.clearance(q0, [obstacle]) > 0 and model.clearance(q1, [obstacle]) > 0


def test_thin_obstacle_between_coarse_samples_is_found():
    robot = Puma560()
    model = robot.collision_model()
    q0 = np.array([-1.2, 0.8, -2.2, 0.0, 0.5, 0.0])
    q1 = np.array([1.2, 0.8, -2.2, 0.0, 0.5, 0.0])
    traj = ptp(q0, q1)
    # Small sphere touched by the wrist only around 40 % of the motion time.
    p = robot.fk(traj(0.4 * traj.duration))[:3, 3]
    obstacle = Sphere(tuple(p), 0.01)
    coarse = check_samples(model, [obstacle], [traj(t) for t in (0.0, 0.5 * traj.duration, traj.duration)])
    assert not coarse.collision and not coarse.certified  # missed, but honestly not certified
    fine = check_motion(model, [obstacle], traj, traj.duration, initial_samples=3)
    assert fine.collision


def test_far_obstacle_is_certified_with_few_samples():
    robot = Puma560()
    model = robot.collision_model()
    traj = ptp(Puma560.QN, np.array([0.8, 0.9, -2.5, 0.4, 0.6, 1.0]))
    res = check_motion(model, [Sphere((3.0, 3.0, 3.0), 0.5)], traj, traj.duration)
    assert res.certified and not res.collision
    assert res.samples < 100
    assert res.min_clearance > 2.0


def test_near_miss_needs_refinement_but_is_certified():
    robot = Puma560()
    model = robot.collision_model()
    q0 = np.array([-1.0, 0.8, -2.2, 0.0, 0.5, 0.0])
    q1 = np.array([1.0, 0.8, -2.2, 0.0, 0.5, 0.0])
    traj = ptp(q0, q1)
    # Centre 25 cm above the wrist at mid-motion; the radius is chosen so the
    # sphere surface passes 2 cm from the closest capsule on a dense sampling.
    centre = robot.fk(traj(traj.duration / 2))[:3, 3] + [0.0, 0.0, 0.25]
    dense_q = [traj(t) for t in np.linspace(0.0, traj.duration, 2001)]
    gap = min(model.clearance(q, [Sphere(tuple(centre), 0.0)]) for q in dense_q)
    obstacle = Sphere(tuple(centre), gap - 0.02)
    res = check_motion(model, [obstacle], traj, traj.duration)
    assert res.certified and not res.collision
    assert res.samples > 16  # the initial grid alone cannot certify a 2 cm gap
    assert res.min_clearance == pytest.approx(0.02, abs=2e-3)


def test_static_capsules_do_not_inflate_the_bound():
    """The PUMA pedestal is fixed in frame 0; only moving capsules count."""
    robot = Puma560()
    model = robot.collision_model()
    q0 = Puma560.QN
    q1 = q0 + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.3])  # joint 6 only
    # With the tool at the wrist centre, turning joint 6 moves no capsule point.
    assert model.displacement_bound(q0, q1) == 0.0
    # Joint 1 moves the arm: bounded by the longest chain of link offsets.
    q2 = q0 + np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    offsets = 0.4318 + 0.15005 + 0.0203 + 0.4318
    assert model.displacement_bound(q0, q2) == pytest.approx(0.1 * offsets)


def test_no_obstacles_is_trivially_free():
    robot = Puma560()
    traj = ptp(Puma560.QN, Puma560.QR)
    res = check_motion(robot.collision_model(), [], traj, traj.duration)
    assert res.certified and not res.collision and res.samples == 0


def test_breakpoints_prevent_false_certification():
    """An out-and-back motion ends where it started, so the end-point bound is zero.

    Without the turning point as a breakpoint the checker would certify an
    interval in which the arm sweeps through the obstacle.
    """
    robot = Planar3R()
    model = robot.collision_model()
    q_turn = np.array([1.0, 0.0, 0.0])
    tip = robot.fk(q_turn)[:3, 3]
    obstacle = Sphere(tuple(tip), 0.05)

    def out_and_back(t):
        return q_turn * (1.0 - abs(2.0 * t - 1.0))

    blind = check_motion(model, [obstacle], out_and_back, 1.0, initial_samples=2)
    assert blind.certified and not blind.collision  # the documented failure mode
    res = check_motion(model, [obstacle], out_and_back, 1.0, initial_samples=2, breakpoints=[0.5])
    assert res.collision and res.first_collision_time == pytest.approx(0.5)

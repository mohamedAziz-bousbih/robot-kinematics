"""PTP and LIN trajectories: boundary conditions, limit compliance and synchronisation."""

import numpy as np
import pytest

from robokin import Puma560, QuinticProfile, TrapezoidProfile, plan_lin, plan_ptp, track_lin
from robokin.transforms import homogeneous, rot_log, rpy_to_rot

from conftest import assert_pose_close

# Illustrative limits (rad/s, rad/s^2) used across the PTP tests.
VMAX = np.array([2.0, 1.5, 2.5, 4.0, 4.0, 5.0])
AMAX = np.array([4.0, 3.0, 5.0, 10.0, 10.0, 12.0])


def dense(duration, n=4001):
    return np.linspace(0.0, duration, n)


def check_limits(traj, vmax, amax, rtol=1e-9):
    t = dense(traj.duration)
    _, qd, qdd = traj.sample(t)
    assert np.all(np.abs(qd) <= vmax * (1 + rtol))
    assert np.all(np.abs(qdd) <= amax * (1 + rtol))


# ---------------------------------------------------------------- trapezoid


@pytest.mark.parametrize("distance", [3.0, -3.0, 0.2, -0.2])
def test_trapezoid_boundaries_and_limits(distance):
    vmax, amax = 1.5, 2.0
    p = TrapezoidProfile.time_optimal(0.5, distance, vmax, amax)
    pos, vel, _ = p.sample(np.array([0.0, p.duration]))
    np.testing.assert_allclose(pos, [0.5, 0.5 + distance], atol=1e-14)
    np.testing.assert_allclose(vel, [0.0, 0.0], atol=1e-14)
    pos, vel, acc = p.sample(dense(p.duration))
    assert np.all(np.abs(vel) <= vmax + 1e-12) and np.all(np.abs(acc) <= amax + 1e-12)
    # Position and velocity are continuous (no jumps between dense samples).
    dt = p.duration / 4000
    assert np.max(np.abs(np.diff(pos))) <= vmax * dt * 1.001
    assert np.max(np.abs(np.diff(vel))) <= amax * dt * 1.001
    # The velocity is the derivative of the position.
    np.testing.assert_allclose(np.gradient(pos, dense(p.duration))[1:-1], vel[1:-1], atol=2e-3)


def test_trapezoid_durations_closed_form():
    # Long move: v reaches vmax. T = d / vmax + vmax / amax.
    p = TrapezoidProfile.time_optimal(0.0, 3.0, 1.5, 2.0)
    assert p.duration == pytest.approx(3.0 / 1.5 + 1.5 / 2.0)
    assert abs(p.velocity) == pytest.approx(1.5)
    # Short move: triangular, T = 2 sqrt(d / amax), peak speed below vmax.
    p = TrapezoidProfile.time_optimal(0.0, 0.2, 1.5, 2.0)
    assert p.duration == pytest.approx(2 * np.sqrt(0.2 / 2.0))
    assert p.velocity < 1.5
    assert p.t_acc == pytest.approx(p.duration / 2)


def test_trapezoid_zero_distance_and_outside_time():
    p = TrapezoidProfile.time_optimal(1.0, 0.0, 1.0, 1.0)
    assert p.duration == 0.0
    pos, vel, acc = p.sample(np.array([-1.0, 0.0, 5.0]))
    np.testing.assert_array_equal(pos, [1.0, 1.0, 1.0])
    p = TrapezoidProfile.time_optimal(0.0, 1.0, 1.0, 1.0)
    pos, vel, acc = p.sample(np.array([-1.0, p.duration + 1.0]))
    np.testing.assert_array_equal(pos, [0.0, 1.0])
    np.testing.assert_array_equal(vel, [0.0, 0.0])
    np.testing.assert_array_equal(acc, [0.0, 0.0])


def test_trapezoid_with_duration_stretches_cruise():
    fast = TrapezoidProfile.time_optimal(0.0, 1.0, 2.0, 2.0)
    slow = TrapezoidProfile.with_duration(0.0, 1.0, 2.5 * fast.duration, 2.0)
    assert slow.duration == pytest.approx(2.5 * fast.duration)
    assert abs(slow.acceleration) == pytest.approx(2.0)
    assert slow.velocity < fast.velocity
    assert slow.sample(slow.duration)[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        TrapezoidProfile.with_duration(0.0, 1.0, 0.5 * fast.duration, 2.0)


# ------------------------------------------------------------------ quintic


def test_quintic_general_boundary_conditions():
    p = QuinticProfile.from_boundary(0.2, -1.0, 1.7, v0=0.3, v1=-0.4, a0=1.0, a1=-2.0)
    pos, vel, acc = p.sample(np.array([0.0, 1.7]))
    np.testing.assert_allclose(pos, [0.2, -1.0], atol=1e-12)
    np.testing.assert_allclose(vel, [0.3, -0.4], atol=1e-12)
    np.testing.assert_allclose(acc, [1.0, -2.0], atol=1e-12)


@pytest.mark.parametrize("distance, vmax, amax", [(2.0, 1.0, 10.0), (2.0, 10.0, 1.0), (-0.5, 0.7, 0.9)])
def test_quintic_min_duration_is_tight(distance, vmax, amax):
    T = QuinticProfile.min_duration(distance, vmax, amax)
    p = QuinticProfile.from_boundary(0.0, distance, T)
    _, vel, acc = p.sample(dense(T, 20001))
    peak_v, peak_a = np.max(np.abs(vel)), np.max(np.abs(acc))
    assert peak_v <= vmax * (1 + 1e-9) and peak_a <= amax * (1 + 1e-6)
    # One of the two limits is active.
    assert max(peak_v / vmax, peak_a / amax) == pytest.approx(1.0, rel=1e-6)


# ---------------------------------------------------------------------- PTP


Q0 = np.array([0.0, 0.4, -1.0, 0.0, 0.5, 0.0])
Q1 = np.array([1.2, 1.1, -2.5, 1.5, -0.4, 2.0])


@pytest.mark.parametrize("profile", ["trapezoid", "quintic"])
@pytest.mark.parametrize("sync", ["async", "sync", "full"])
def test_ptp_reaches_goal_within_limits(profile, sync):
    traj = plan_ptp(Q0, Q1, VMAX, AMAX, profile=profile, sync=sync)
    q, qd, _ = traj.sample([0.0, traj.duration])
    np.testing.assert_allclose(q, [Q0, Q1], atol=1e-12)
    np.testing.assert_allclose(qd, 0.0, atol=1e-12)
    check_limits(traj, VMAX, AMAX)


def test_ptp_synchronisation_modes():
    asy = plan_ptp(Q0, Q1, VMAX, AMAX, sync="async")
    syn = plan_ptp(Q0, Q1, VMAX, AMAX, sync="sync")
    full = plan_ptp(Q0, Q1, VMAX, AMAX, sync="full")
    durations = [p.duration for p in asy.profiles]
    assert len(set(np.round(durations, 12))) > 1  # axes finish at different times
    assert syn.duration == pytest.approx(max(durations))
    assert all(p.duration == pytest.approx(syn.duration) for p in syn.profiles)
    assert full.duration >= syn.duration - 1e-12
    # Fully synchronised axes share acceleration phase, so the path is a
    # straight line in joint space: (q(t) - q0) is proportional to (q1 - q0).
    assert len({round(p.t_acc, 12) for p in full.profiles}) == 1
    q, _, _ = full.sample(dense(full.duration, 200))
    frac = (q - Q0) / (Q1 - Q0)
    np.testing.assert_allclose(frac, frac[:, :1].repeat(6, axis=1), atol=1e-12)


def test_full_sync_uses_a_limit_of_some_axis():
    full = plan_ptp(Q0, Q1, VMAX, AMAX, sync="full")
    _, qd, qdd = full.sample(dense(full.duration))
    usage = max(np.max(np.abs(qd) / VMAX), np.max(np.abs(qdd) / AMAX))
    assert usage == pytest.approx(1.0, rel=1e-9)


def test_ptp_rejects_bad_input():
    with pytest.raises(ValueError):
        plan_ptp(Q0, Q1[:3], VMAX, AMAX)
    with pytest.raises(ValueError):
        plan_ptp(Q0, Q1, -VMAX, AMAX)
    with pytest.raises(ValueError):
        plan_ptp(Q0, Q1, VMAX, AMAX, sync="sometimes")
    with pytest.raises(ValueError):
        plan_ptp(Q0, Q1, VMAX, AMAX, profile="cubic")


def test_ptp_with_resting_axes():
    q1 = Q0.copy()
    q1[2] += 0.5
    for sync in ("async", "sync", "full"):
        traj = plan_ptp(Q0, q1, VMAX, AMAX, sync=sync)
        q, _, _ = traj.sample(dense(traj.duration, 50))
        np.testing.assert_allclose(q[:, [0, 1, 3, 4, 5]], np.tile(Q0[[0, 1, 3, 4, 5]], (50, 1)))
    still = plan_ptp(Q0, Q0, VMAX, AMAX, sync="full")
    assert still.duration == 0.0


# ---------------------------------------------------------------------- LIN


T_A = homogeneous(rpy_to_rot(np.pi, 0.0, 0.0), [0.45, -0.25, -0.20])
T_B = homogeneous(rpy_to_rot(np.pi, 0.3, 0.8), [0.35, 0.25, -0.05])


def test_lin_path_is_straight_with_slerp_orientation():
    traj = plan_lin(T_A, T_B, v_max=0.5, a_max=1.0, w_max=1.0, alpha_max=2.0)
    assert_pose_close(traj.pose(0.0), T_A, atol=1e-12)
    assert_pose_close(traj.pose(traj.duration), T_B, atol=1e-12)
    d = T_B[:3, 3] - T_A[:3, 3]
    theta = np.linalg.norm(rot_log(T_B[:3, :3] @ T_A[:3, :3].T))
    for t in dense(traj.duration, 101):
        T = traj.pose(t)
        rel = T[:3, 3] - T_A[:3, 3]
        s = rel @ d / (d @ d)
        np.testing.assert_allclose(rel, s * d, atol=1e-12)  # on the line
        angle = np.linalg.norm(rot_log(T[:3, :3] @ T_A[:3, :3].T))
        assert angle == pytest.approx(s * theta, abs=1e-9)  # orientation progresses with s


def test_lin_speed_limits():
    v_max, a_max, w_max, alpha_max = 0.5, 1.0, 0.4, 2.0
    traj = plan_lin(T_A, T_B, v_max, a_max, w_max, alpha_max)
    t = dense(traj.duration, 2001)
    poses = np.array([traj.pose(ti) for ti in t])
    speed = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) / np.diff(t)
    omega = np.array(
        [np.linalg.norm(rot_log(b[:3, :3] @ a[:3, :3].T)) for a, b in zip(poses[:-1], poses[1:])]
    ) / np.diff(t)
    assert speed.max() <= v_max * (1 + 1e-6)
    assert omega.max() <= w_max * (1 + 1e-6)
    # Here the rotation is the binding limit.
    assert omega.max() == pytest.approx(w_max, rel=1e-2)


def test_track_lin_follows_the_cartesian_path():
    robot = Puma560()
    q_start = robot.ik_closest(T_A, Puma560.QN).q
    traj = plan_lin(T_A, T_B, 0.5, 1.0, 1.0, 2.0)
    joint = track_lin(robot, traj, q_start, dt=0.01)
    np.testing.assert_allclose(joint.q[0], q_start, atol=1e-9)
    for t, q in zip(joint.t, joint.q):
        assert_pose_close(robot.fk(q), traj.pose(t), atol=1e-9)
    # Branch tracking: no joint jumps between 10 ms samples.
    assert joint.max_step() < 0.05


def test_track_lin_unreachable_raises():
    robot = Puma560()
    far = homogeneous(T_A[:3, :3], [1.5, 0.0, 0.0])
    traj = plan_lin(T_A, far, 0.5, 1.0, 1.0, 2.0)
    with pytest.raises(ValueError, match="not reachable"):
        track_lin(robot, traj, robot.ik_closest(T_A, Puma560.QN).q, dt=0.02)

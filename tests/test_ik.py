"""Inverse kinematics: closed-form round trips for every branch, and the numeric DLS solver."""

import numpy as np
import pytest

from robokin import Planar3R, Puma560, Scara, ik_dls, ik_numeric
from robokin.transforms import homogeneous, rotx, rotz, rpy_to_rot, transl

from conftest import assert_pose_close, joint_distance


def test_round_trip_all_branches(robot, rng):
    """FK(IK(T)) == T for every returned solution, and the source q is among them."""
    for q in robot.random_q(rng, 300):
        T = robot.fk(q)
        sols = robot.ik(T)
        assert sols, f"no IK solution for a pose produced by FK at q={q}"
        for sol in sols:
            assert_pose_close(robot.fk(sol.q), T, atol=1e-9)
            assert robot.within_limits(sol.q)
        assert min(joint_distance(robot, s.q, q) for s in sols) < 1e-8
        # Branches are distinct configurations.
        for i, a in enumerate(sols):
            for b in sols[i + 1:]:
                assert joint_distance(robot, a.q, b.q) > 1e-9


def test_puma_has_eight_solutions_without_limits(rng):
    robot = Puma560()
    for q in robot.random_q(rng, 200):
        if robot.singularity_kinds(q, tol=1e-6):
            continue
        sols = robot.ik(robot.fk(q), respect_limits=False)
        assert len(sols) == 8
        labels = {s.branch for s in sols}
        assert len(labels) == 8
        assert {lab.split("/")[0] for lab in labels} == {"front", "back"}
        assert {lab.split("/")[1] for lab in labels} == {"up", "down"}
        assert {lab.split("/")[2] for lab in labels} == {"flip", "noflip"}


def test_out_of_limit_solutions_keep_in_range_joints_wrapped():
    """Without limit filtering, joints that can be wrapped into range still are.

    Here the wrist of one branch violates the joint-4 limit; its arm joints
    must still be expressed exactly like those of its in-range twin branch.
    """
    robot = Puma560()
    T = homogeneous(rpy_to_rot(0.0, np.pi / 2, 0.0), [0.45, -0.30, 0.20])
    sols = {s.branch: s for s in robot.ik(T, respect_limits=False)}
    in_range = {b for b, s in sols.items() if robot.within_limits(s.q)}
    assert 0 < len(in_range) < len(sols)
    for branch, sol in sols.items():
        wrist = "flip" if branch.endswith("noflip") else "noflip"
        twin = sols[branch.rsplit("/", 1)[0] + "/" + wrist]
        np.testing.assert_allclose(sol.q[:3], twin.q[:3], atol=1e-12)
        for i in range(6):
            lo, hi = robot.qlim[i]
            if lo <= sol.q[i] <= hi:
                continue
            # A joint left outside its range has no 2 pi alias inside it.
            assert not any(lo <= sol.q[i] + k * 2 * np.pi <= hi for k in (-2, -1, 1, 2))


def test_puma_wrist_flip_relation(rng):
    """The two wrist solutions differ by (q4 + pi, -q5, q6 + pi)."""
    robot = Puma560()
    q = np.array([0.3, 0.6, -1.2, 0.4, 0.8, -0.5])
    sols = {s.branch: s.q for s in robot.ik(robot.fk(q), respect_limits=False)}
    for branch, qa in sols.items():
        if branch.endswith("/noflip"):
            qb = sols[branch.replace("noflip", "flip")]
            np.testing.assert_allclose(qa[:3], qb[:3], atol=1e-12)
            assert joint_distance(robot, qb, qa + [0, 0, 0, np.pi, -2 * qa[4], np.pi]) < 1e-9


def test_planar_branches_and_reach():
    robot = Planar3R()
    l1, l2, l3 = robot.lengths
    T = homogeneous(rotz(0.3), [0.6, 0.3, 0.0])
    sols = robot.ik(T)
    assert sorted(s.branch for s in sols) == ["elbow_down", "elbow_up"]
    for s in sols:
        assert np.sign(s.q[1]) == (1 if s.branch == "elbow_down" else -1)
    # Fully stretched along x: the two branches merge into one.
    stretched = robot.ik(homogeneous(np.eye(3), [l1 + l2 + l3, 0.0, 0.0]))
    assert len(stretched) == 1 and stretched[0].branch == "singular"
    # Out of reach, or outside the plane of motion: no solution.
    assert robot.ik(homogeneous(np.eye(3), [l1 + l2 + l3 + 1e-3, 0.0, 0.0])) == []
    assert robot.ik(homogeneous(rotz(0.3), [0.6, 0.3, 0.05])) == []
    assert robot.ik(homogeneous(rotx(0.3), [0.6, 0.3, 0.0])) == []


def test_scara_branches_and_stroke():
    robot = Scara()
    T = robot.fk([0.4, 1.0, 0.1, -0.7])
    assert sorted(s.branch for s in robot.ik(T)) == ["lefty", "righty"]
    # Below the stroke of the prismatic joint (q3 > 0.2 m) there is no solution.
    too_low = T.copy()
    too_low[2, 3] -= 0.15
    assert robot.ik(too_low) == []
    assert len(robot.ik(too_low, respect_limits=False)) == 2
    # A tilted tool cannot be produced by a SCARA.
    assert robot.ik(T @ homogeneous(rotx(0.1))) == []


def test_puma_unreachable_poses():
    robot = Puma560()
    R = rpy_to_rot(np.pi, 0, 0)
    assert robot.ik(homogeneous(R, [1.2, 0.0, 0.0])) == []  # beyond reach
    assert robot.ik(homogeneous(R, [0.05, 0.05, 0.3])) == []  # inside the shoulder-offset cylinder


def test_puma_wrist_singularity_uses_reference():
    robot = Puma560()
    q = np.array([0.2, 0.5, -1.0, 0.7, 0.0, 0.4])  # q5 = 0: only q4 + q6 is defined
    T = robot.fk(q)
    for q4_ref in (-0.3, 0.7, 1.1):
        sol = robot.ik_closest(T, np.array([0.2, 0.5, -1.0, q4_ref, 0.0, 0.0]))
        assert sol.branch.endswith("wrist_singular")
        assert sol.q[3] == pytest.approx(q4_ref)
        assert sol.q[3] + sol.q[5] == pytest.approx(q[3] + q[5])
        assert_pose_close(robot.fk(sol.q), T, atol=1e-9)


def test_ik_closest_returns_source_configuration(robot, rng):
    for q in robot.random_q(rng, 50):
        sol = robot.ik_closest(robot.fk(q), q)
        assert np.max(np.abs(sol.q - q)) < 1e-8


def test_ik_with_base_and_tool():
    base = homogeneous(rotz(0.4), [0.3, -0.2, 0.8])
    robot = Puma560(base=base, tool=transl(0.0, 0.0, 0.12))
    q = np.array([0.1, 0.9, -2.0, 0.3, 0.6, 1.0])
    T = robot.fk(q)
    assert any(np.allclose(s.q, q, atol=1e-8) for s in robot.ik(T))


@pytest.mark.parametrize("perturbation", [0.1, 0.3])
def test_dls_converges_from_nearby_start(robot, rng, perturbation):
    for q in robot.random_q(rng, 40):
        T = robot.fk(q)
        q0 = robot.clip_to_limits(q + rng.uniform(-perturbation, perturbation, robot.n)
                                  * np.where(robot.revolute, 1.0, 0.1))
        res = ik_dls(robot, T, q0)
        assert res.success, f"DLS failed from {q0} to {q}: {res}"
        assert res.position_error <= 1e-10 and res.orientation_error <= 1e-10
        assert_pose_close(robot.fk(res.q), T, atol=1e-8)
        assert robot.within_limits(res.q)


def test_limit_modes_fail_differently_and_restarts_recover(rng):
    """From the same far seed, "clamp" stalls on the q2 limit and "wrap" converges
    to a branch the limits exclude; the restart wrapper still finds a solution."""
    robot = Puma560()
    q_goal = np.array([-0.382, 3.470, -0.156, 1.376, 1.472, -1.154])
    q0 = np.array([-0.576, 3.924, 0.154, 1.664, 1.606, -0.897])
    T = robot.fk(q_goal)

    clamped = ik_dls(robot, T, q0, limits="clamp")
    assert not clamped.success and clamped.within_limits
    assert clamped.position_error > 0.1  # stuck far from the target, not merely slow
    assert robot.qlim[1, 1] - clamped.q[1] < 1e-12  # ... with joint 2 on its upper limit

    wrapped = ik_dls(robot, T, q0, limits="wrap")
    assert not wrapped.success and not wrapped.within_limits
    assert wrapped.position_error < 1e-10 and wrapped.orientation_error < 1e-10
    # It is a genuine solution, just one the closed form also rejects for its limits.
    assert_pose_close(robot.fk(wrapped.q), T, atol=1e-9)
    assert any(joint_distance(robot, s.q, wrapped.q) < 1e-6 for s in robot.ik(T, respect_limits=False))
    assert all(joint_distance(robot, s.q, wrapped.q) > 1e-3 for s in robot.ik(T))

    res = ik_numeric(robot, T, q0, rng)
    assert res.success and res.within_limits
    assert res.iterations > clamped.iterations
    assert_pose_close(robot.fk(res.q), T, atol=1e-8)
    assert robot.within_limits(res.q)


def test_wrap_mode_shifts_into_limits_near_the_seed():
    """Joint 6 spans 532 deg; of two equivalent in-range angles the one nearer the seed wins."""
    robot = Puma560()
    q_goal = np.array([0.3, 0.6, -1.2, 0.4, 0.8, np.deg2rad(170.0)])
    T = robot.fk(q_goal)
    for seed_q6, expected in ((150.0, 170.0), (-170.0, -190.0)):
        q0 = q_goal.copy()
        q0[5] = np.deg2rad(seed_q6)
        res = ik_dls(robot, T, q0, limits="wrap")
        assert res.success and robot.within_limits(res.q)
        assert np.rad2deg(res.q[5]) == pytest.approx(expected, abs=1e-6)
        assert_pose_close(robot.fk(res.q), T, atol=1e-9)


def test_unknown_limit_mode_rejected():
    robot = Planar3R()
    with pytest.raises(ValueError, match="limit mode"):
        ik_dls(robot, robot.fk(np.zeros(3)), np.zeros(3), limits="ignore")


def test_numeric_fallback_from_arbitrary_seeds(robot, rng):
    for q in robot.random_q(rng, 30):
        T = robot.fk(q)
        res = ik_numeric(robot, T, robot.random_q(rng), rng)
        assert res.success
        assert_pose_close(robot.fk(res.q), T, atol=1e-8)
        assert robot.within_limits(res.q)


@pytest.mark.parametrize("mode", ["clamp", "wrap"])
def test_successful_runs_are_inside_the_limits(robot, rng, mode):
    for q in robot.random_q(rng, 40):
        T = robot.fk(q)
        res = ik_dls(robot, T, robot.random_q(rng), limits=mode)
        assert res.success == (res.within_limits and res.position_error <= 1e-10
                               and res.orientation_error <= 1e-10)
        if res.success:
            assert robot.within_limits(res.q)
            assert_pose_close(robot.fk(res.q), T, atol=1e-8)


def test_dls_reports_failure_for_unreachable_target():
    robot = Puma560()
    T = homogeneous(np.eye(3), [2.0, 0.0, 0.0])
    res = ik_dls(robot, T, Puma560.QN, max_iter=200)
    assert not res.success
    # The closest the arm gets is the stretched wrist centre at distance r_max.
    _, r_max = robot.wrist_distance_range()
    assert res.position_error == pytest.approx(2.0 - r_max, abs=1e-4)
    assert np.all(np.isfinite(res.q))
    # The stall test gives up on the plateau earlier than the damping limit does.
    patient = ik_dls(robot, T, Puma560.QN, max_iter=200, stall_window=0)
    assert not patient.success and res.iterations < patient.iterations
    assert res.position_error == pytest.approx(patient.position_error, abs=1e-6)


def test_active_set_locks_joints_pushing_against_a_limit():
    """A target beyond the joint-2 limit: the solver ends on the limit, never past it."""
    robot = Planar3R()
    q_goal = np.array([0.2, robot.qlim[1, 1] + 0.3, 0.1])
    T = robot.fk(q_goal)
    q0 = np.array([0.2, robot.qlim[1, 1] - 0.05, 0.1])
    res = ik_dls(robot, T, q0)
    assert robot.within_limits(res.q)
    # Either the target was reached through another branch, or joint 2 is parked on its limit.
    assert res.success or abs(res.q[1] - robot.qlim[1, 1]) < 1e-12


def test_dls_starts_from_a_singular_configuration():
    robot = Puma560()
    # qz has q5 = 0 (wrist singularity); the plain pseudo-inverse would blow up.
    target = robot.fk(np.array([0.3, 0.4, -0.9, 0.5, 0.7, -0.2]))
    res = ik_dls(robot, target, Puma560.QZ)
    assert res.success
    assert_pose_close(robot.fk(res.q), target, atol=1e-8)

"""Forward kinematics against hand-derived closed forms and published reference poses."""

import numpy as np
import pytest

from robokin import DHLink, Planar3R, Puma560, Scara, SerialRobot
from robokin.transforms import homogeneous, rotx, rotz, transl

from conftest import assert_pose_close


def dh_by_composition(a, alpha, d, theta):
    """A_i built literally as Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    return (
        homogeneous(rotz(theta)) @ transl(0, 0, d) @ transl(a, 0, 0) @ homogeneous(rotx(alpha))
    )


def test_dh_link_matches_definition(rng):
    for _ in range(50):
        a, alpha, d, theta, q = rng.uniform(-2, 2, size=5)
        rev = DHLink(a=a, alpha=alpha, d=d, theta=theta)
        pri = DHLink(a=a, alpha=alpha, d=d, theta=theta, prismatic=True, qlim=(-3, 3))
        np.testing.assert_allclose(rev.transform(q), dh_by_composition(a, alpha, d, theta + q), atol=1e-14)
        np.testing.assert_allclose(pri.transform(q), dh_by_composition(a, alpha, d + q, theta), atol=1e-14)


def test_invalid_limits_rejected():
    with pytest.raises(ValueError):
        DHLink(a=1.0, alpha=0.0, qlim=(1.0, -1.0))


def test_batched_fk_equals_single(robot, rng):
    Q = robot.random_q(rng, 64)
    batch = robot.fk(Q)
    for q, T in zip(Q, batch):
        np.testing.assert_allclose(robot.fk(q), T, atol=1e-14)


def test_fk_frames_end_with_tool(robot, rng):
    q = robot.random_q(rng)
    frames = robot.fk_frames(q)
    assert frames.shape == (robot.n + 2, 4, 4)
    np.testing.assert_allclose(frames[-1], robot.fk(q), atol=1e-14)


def test_wrong_joint_count_raises(robot):
    with pytest.raises(ValueError):
        robot.fk(np.zeros(robot.n + 1))


def test_planar_closed_form(rng):
    robot = Planar3R()
    l1, l2, l3 = robot.lengths
    for q in robot.random_q(rng, 100):
        q1, q12, q123 = np.cumsum(q)
        T = robot.fk(q)
        assert T[0, 3] == pytest.approx(l1 * np.cos(q1) + l2 * np.cos(q12) + l3 * np.cos(q123))
        assert T[1, 3] == pytest.approx(l1 * np.sin(q1) + l2 * np.sin(q12) + l3 * np.sin(q123))
        assert T[2, 3] == pytest.approx(0.0, abs=1e-15)
        np.testing.assert_allclose(T[:3, :3], rotz(q123), atol=1e-14)


def test_scara_closed_form_spong(rng):
    """Spong, Hutchinson, Vidyasagar: T = [[c,s,0,x],[s,-c,0,y],[0,0,-1,z]] with c = cos(q1+q2-q4)."""
    robot = Scara()
    a1, a2 = robot.links[0].a, robot.links[1].a
    d1, d4 = robot.links[0].d, robot.links[3].d
    for q1, q2, q3, q4 in robot.random_q(rng, 100):
        psi = q1 + q2 - q4
        c, s = np.cos(psi), np.sin(psi)
        expected = np.array(
            [
                [c, s, 0, a1 * np.cos(q1) + a2 * np.cos(q1 + q2)],
                [s, -c, 0, a1 * np.sin(q1) + a2 * np.sin(q1 + q2)],
                [0, 0, -1, d1 - q3 - d4],
                [0, 0, 0, 1],
            ]
        )
        assert_pose_close(robot.fk([q1, q2, q3, q4]), expected, atol=1e-14)


def test_puma_zero_pose_reference():
    # Derived by hand in the README from the DH table (all joint angles zero).
    T = Puma560().fk(Puma560.QZ)
    expected = homogeneous(np.eye(3), [0.4521, -0.15005, 0.4318])
    assert_pose_close(T, expected, atol=1e-12)


def test_puma_ready_pose_reference():
    # qr = [0, 90, -90, 0, 0, 0] deg: upper arm and forearm point straight up.
    T = Puma560().fk(Puma560.QR)
    expected = homogeneous(np.eye(3), [0.0203, -0.15005, 0.8636])
    assert_pose_close(T, expected, atol=1e-12)


def test_puma_wrist_centre_closed_form(rng):
    """p = Rz(q1) (x1, -d3) horizontally and y1 vertically, as used by the IK."""
    robot = Puma560()
    a2, a3, d3, d4 = 0.4318, 0.0203, 0.15005, 0.4318
    for q in robot.random_q(rng, 100):
        q1, q2, q23 = q[0], q[1], q[1] + q[2]
        x1 = a2 * np.cos(q2) + a3 * np.cos(q23) - d4 * np.sin(q23)
        y1 = a2 * np.sin(q2) + a3 * np.sin(q23) + d4 * np.cos(q23)
        expected = [np.cos(q1) * x1 + np.sin(q1) * d3, np.sin(q1) * x1 - np.cos(q1) * d3, y1]
        np.testing.assert_allclose(robot.fk(q)[:3, 3], expected, atol=1e-14)
        assert robot.arm_plane_reach(q) == pytest.approx(x1)


def test_base_and_tool_transforms():
    base = homogeneous(rotz(0.5), [1.0, 2.0, 3.0])
    tool = transl(0.0, 0.0, 0.1)
    plain, mounted = Puma560(), Puma560(base=base, tool=tool)
    q = Puma560.QN
    np.testing.assert_allclose(mounted.fk(q), base @ plain.fk(q) @ tool, atol=1e-14)
    np.testing.assert_allclose(mounted.strip_base_tool(mounted.fk(q)), plain.fk(q), atol=1e-14)


def test_generic_chain_with_prismatic_joint():
    # Cartesian gantry: two prismatic joints along x and y (via alpha twists) and one along z.
    gantry = SerialRobot(
        "gantry",
        [
            DHLink(a=0.0, alpha=-np.pi / 2, theta=-np.pi / 2, prismatic=True, qlim=(0, 1)),
            DHLink(a=0.0, alpha=-np.pi / 2, theta=-np.pi / 2, prismatic=True, qlim=(0, 1)),
            DHLink(a=0.0, alpha=0.0, prismatic=True, qlim=(0, 1)),
        ],
    )
    T = gantry.fk([0.2, 0.3, 0.4])
    p = T[:3, 3]
    # Each prismatic joint moves along a different, mutually orthogonal axis.
    J = gantry.jacobian([0.2, 0.3, 0.4])[:3]
    np.testing.assert_allclose(np.abs(J.T @ J), np.eye(3), atol=1e-14)
    np.testing.assert_allclose(p, J @ [0.2, 0.3, 0.4], atol=1e-14)


def test_wrap_to_limits_shifts_by_full_turns():
    robot = Puma560()
    q = np.deg2rad([0.0, -170.0, 30.0, 0.0, 0.0, 0.0])
    wrapped = robot.wrap_to_limits(q)
    assert np.rad2deg(wrapped[1]) == pytest.approx(190.0)
    assert robot.within_limits(wrapped)
    # Joint 5 has no equivalent angle inside [-100, 100] deg for 150 deg.
    assert robot.wrap_to_limits(np.deg2rad([0, 0, 0, 0, 150, 0])) is None
    # Joint 6 spans 532 deg, so a reference picks between equivalent angles.
    q6 = np.deg2rad([0, 0, 0, 0, 0, 170.0])
    ref = np.deg2rad([0, 0, 0, 0, 0, -180.0])
    assert np.rad2deg(robot.wrap_to_limits(q6, ref=ref)[5]) == pytest.approx(-190.0)

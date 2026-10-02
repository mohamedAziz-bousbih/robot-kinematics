"""Geometric Jacobian, manipulability and singularity detection."""

import numpy as np
import pytest

from robokin import Planar3R, Puma560, Scara
from robokin.transforms import rot_log


def numeric_jacobian(robot, q, h=1e-6):
    """Central differences: position directly, orientation through rot_log(R+ R-^T)."""
    J = np.zeros((6, robot.n))
    for i in range(robot.n):
        dq = np.zeros(robot.n)
        dq[i] = h
        Tp, Tm = robot.fk(q + dq), robot.fk(q - dq)
        J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
        J[3:, i] = rot_log(Tp[:3, :3] @ Tm[:3, :3].T) / (2 * h)
    return J


def test_jacobian_matches_finite_differences(robot, rng):
    for q in robot.random_q(rng, 50):
        np.testing.assert_allclose(robot.jacobian(q), numeric_jacobian(robot, q), atol=1e-8)


def test_jacobian_with_tool_offset(rng):
    from robokin.transforms import transl

    robot = Puma560(tool=transl(0.05, 0.0, 0.15))
    for q in robot.random_q(rng, 20):
        np.testing.assert_allclose(robot.jacobian(q), numeric_jacobian(robot, q), atol=1e-8)


def test_planar_determinant(rng):
    robot = Planar3R()
    l1, l2, _ = robot.lengths
    for q in robot.random_q(rng, 100):
        det = np.linalg.det(robot.task_jacobian(q))
        assert det == pytest.approx(l1 * l2 * np.sin(q[1]), abs=1e-12)
        assert robot.manipulability(q) == pytest.approx(abs(det), abs=1e-12)


def test_scara_determinant(rng):
    robot = Scara()
    a1, a2 = robot.links[0].a, robot.links[1].a
    for q in robot.random_q(rng, 100):
        det = np.linalg.det(robot.task_jacobian(q))
        assert abs(det) == pytest.approx(a1 * a2 * abs(np.sin(q[1])), abs=1e-12)


def test_puma_determinant_factorises(rng):
    """|det J| = |x1| * a2 * L3 * |sin(q3 + beta)| * |sin q5| with the TCP at the wrist centre.

    The three factors are exactly the shoulder, elbow and wrist singularities.
    """
    robot = Puma560()
    L3, beta = robot.forearm
    a2 = robot.links[1].a
    for q in robot.random_q(rng, 100):
        expected = abs(robot.arm_plane_reach(q) * a2 * L3 * np.sin(q[2] + beta) * np.sin(q[4]))
        assert abs(np.linalg.det(robot.jacobian(q))) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "q, kind",
    [
        ([0.3, 0.4, -np.arctan2(0.4318, 0.0203), 0.2, 0.9, 0.1], "elbow"),
        ([0.3, 0.4, -0.8, 0.2, 0.0, 0.1], "wrist"),
    ],
)
def test_puma_named_singularities(q, kind):
    robot = Puma560()
    q = np.asarray(q)
    assert kind in robot.singularity_kinds(q)
    assert robot.is_singular(q)
    assert robot.manipulability(q) < 1e-12


def test_puma_shoulder_singularity():
    robot = Puma560()
    # Solve x1(q2, q3) = 0 for q2 with q3 fixed: a2 c2 + L3 cos(q2 + q3 + beta) = 0.
    L3, beta = robot.forearm
    a2 = robot.links[1].a
    q3 = -0.6
    phi = q3 + beta
    # a2 c2 + L3 (c2 cos(phi) - s2 sin(phi)) = 0  ->  tan(q2) = (a2 + L3 cos(phi)) / (L3 sin(phi))
    q2 = np.arctan2(a2 + L3 * np.cos(phi), L3 * np.sin(phi))
    q = np.array([0.5, q2, q3, 0.3, 0.8, 0.0])
    assert abs(robot.arm_plane_reach(q)) < 1e-12
    assert robot.singularity_kinds(q) == {"shoulder"}
    assert robot.is_singular(q)


def test_generic_configuration_is_not_singular():
    robot = Puma560()
    assert robot.singularity_kinds(Puma560.QN) == set()
    assert not robot.is_singular(Puma560.QN)
    assert robot.manipulability(Puma560.QN) > 1e-3


def test_planar_stretched_arm_is_singular():
    robot = Planar3R()
    assert robot.is_singular([0.4, 0.0, 1.0])
    assert not robot.is_singular([0.4, 0.5, 1.0])


def test_singularity_kinds_agree_with_the_rank_test(robot, rng):
    """Every rank loss of the task Jacobian is explained by a named singularity."""
    qs = list(robot.random_q(rng, 200))
    # Exactly singular configurations: q2 = 0 stretches the planar and SCARA
    # arms, q5 = 0 aligns the PUMA wrist axes 4 and 6.
    joint = 4 if isinstance(robot, Puma560) else 1
    for q in robot.random_q(rng, 10):
        q[joint] = 0.0
        qs.append(q)
    singular = 0
    for q in qs:
        named = bool(robot.singularity_kinds(q, tol=1e-9))
        assert named == robot.is_singular(q, tol=1e-9), q
        singular += named
    assert singular == 10

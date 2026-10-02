import numpy as np
import pytest

from robokin import transforms as tf


def random_rotation(rng):
    q = rng.normal(size=4)
    return tf.quat_to_rot(q / np.linalg.norm(q))


def test_elementary_rotations_follow_right_hand_rule():
    np.testing.assert_allclose(tf.rotz(np.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(tf.rotx(np.pi / 2) @ [0, 1, 0], [0, 0, 1], atol=1e-15)
    np.testing.assert_allclose(tf.roty(np.pi / 2) @ [0, 0, 1], [1, 0, 0], atol=1e-15)


def test_skew_matches_cross_product(rng):
    a, b = rng.normal(size=3), rng.normal(size=3)
    np.testing.assert_allclose(tf.skew(a) @ b, np.cross(a, b), atol=1e-14)


def test_inverse_transform(rng):
    T = tf.homogeneous(random_rotation(rng), rng.normal(size=3))
    np.testing.assert_allclose(T @ tf.inv_transform(T), np.eye(4), atol=1e-14)
    np.testing.assert_allclose(tf.inv_transform(T), np.linalg.inv(T), atol=1e-14)


def test_is_rotation_rejects_reflections_and_scaling(rng):
    R = random_rotation(rng)
    assert tf.is_rotation(R)
    assert not tf.is_rotation(R @ np.diag([1, 1, -1]))
    assert not tf.is_rotation(1.01 * R)
    assert not tf.is_rotation(np.eye(4))


def test_rpy_definition_is_fixed_axis_xyz():
    r, p, y = 0.3, -0.4, 1.1
    expected = tf.rotz(y) @ tf.roty(p) @ tf.rotx(r)
    np.testing.assert_allclose(tf.rpy_to_rot(r, p, y), expected, atol=1e-15)
    np.testing.assert_allclose(tf.rpy_to_rot(r, 0, 0), tf.rotx(r), atol=1e-15)


def test_rpy_round_trip(rng):
    for _ in range(500):
        rpy = rng.uniform([-np.pi, -np.pi / 2 + 1e-3, -np.pi], [np.pi, np.pi / 2 - 1e-3, np.pi])
        np.testing.assert_allclose(tf.rot_to_rpy(tf.rpy_to_rot(*rpy)), rpy, atol=1e-9)


@pytest.mark.parametrize("pitch", [np.pi / 2, -np.pi / 2])
def test_rpy_gimbal_lock_still_reconstructs_rotation(pitch):
    R = tf.rpy_to_rot(0.7, pitch, -0.2)
    rpy = tf.rot_to_rpy(R)
    assert rpy[1] == pytest.approx(pitch)
    assert rpy[0] == 0.0
    np.testing.assert_allclose(tf.rpy_to_rot(*rpy), R, atol=1e-12)


def test_rotation_vector_round_trip(rng):
    for _ in range(500):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0, np.pi)
        w = tf.rot_log(tf.axis_angle_to_rot(axis, angle))
        np.testing.assert_allclose(w, axis * angle, atol=1e-10)


@pytest.mark.parametrize("angle", [0.0, 1e-14, 1e-7, np.pi - 1e-7, np.pi])
def test_axis_angle_edge_angles(angle):
    axis = np.array([2.0, -1.0, 0.5]) / np.linalg.norm([2.0, -1.0, 0.5])
    R = tf.axis_angle_to_rot(axis, angle)
    assert tf.is_rotation(R)
    np.testing.assert_allclose(tf.rot_exp(tf.rot_log(R)), R, atol=1e-12)
    ax, ang = tf.rot_to_axis_angle(R)
    assert ang == pytest.approx(angle, abs=1e-12)
    if angle > 1e-6:
        # At exactly pi, -axis describes the same rotation.
        assert abs(abs(ax @ axis) - 1.0) < 1e-9


def test_rodrigues_matches_elementary_rotation():
    np.testing.assert_allclose(tf.axis_angle_to_rot([0, 0, 3], 0.8), tf.rotz(0.8), atol=1e-15)
    with pytest.raises(ValueError):
        tf.axis_angle_to_rot([0, 0, 0], 1.0)


def test_quaternion_known_value():
    q = tf.rot_to_quat(tf.rotz(np.pi / 2))
    np.testing.assert_allclose(q, [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], atol=1e-15)


def test_quaternion_round_trip_including_half_turns(rng):
    rotations = [random_rotation(rng) for _ in range(500)]
    rotations += [np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]), np.diag([-1.0, -1.0, 1.0])]
    for R in rotations:
        q = tf.rot_to_quat(R)
        assert np.linalg.norm(q) == pytest.approx(1.0)
        assert q[0] >= 0.0
        np.testing.assert_allclose(tf.quat_to_rot(q), R, atol=1e-12)


def test_quaternion_product_composes_rotations(rng):
    for _ in range(100):
        R1, R2 = random_rotation(rng), random_rotation(rng)
        q = tf.quat_multiply(tf.rot_to_quat(R1), tf.rot_to_quat(R2))
        np.testing.assert_allclose(tf.quat_to_rot(q), R1 @ R2, atol=1e-12)
        q1 = tf.rot_to_quat(R1)
        np.testing.assert_allclose(tf.quat_multiply(q1, tf.quat_conjugate(q1)), [1, 0, 0, 0], atol=1e-14)


def test_slerp_endpoints_and_constant_rate(rng):
    q0, q1 = tf.rot_to_quat(random_rotation(rng)), tf.rot_to_quat(random_rotation(rng))
    theta = np.linalg.norm(tf.rot_log(tf.quat_to_rot(q1) @ tf.quat_to_rot(q0).T))
    np.testing.assert_allclose(tf.quat_to_rot(tf.quat_slerp(q0, q1, 0.0)), tf.quat_to_rot(q0), atol=1e-12)
    np.testing.assert_allclose(tf.quat_to_rot(tf.quat_slerp(q0, q1, 1.0)), tf.quat_to_rot(q1), atol=1e-12)
    for s, qs in zip(np.linspace(0, 1, 11), tf.quat_slerp(q0, q1, np.linspace(0, 1, 11))):
        angle = np.linalg.norm(tf.rot_log(tf.quat_to_rot(qs) @ tf.quat_to_rot(q0).T))
        assert angle == pytest.approx(s * theta, abs=1e-10)


def test_slerp_takes_the_short_way():
    q0 = tf.rot_to_quat(np.eye(3))
    q1 = tf.rot_to_quat(tf.rotz(np.deg2rad(170)))
    mid_a = tf.quat_to_rot(tf.quat_slerp(q0, q1, 0.5))
    mid_b = tf.quat_to_rot(tf.quat_slerp(q0, -q1, 0.5))
    np.testing.assert_allclose(mid_a, tf.rotz(np.deg2rad(85)), atol=1e-12)
    np.testing.assert_allclose(mid_a, mid_b, atol=1e-12)


def test_interpolate_pose_midpoint():
    T0 = tf.homogeneous(np.eye(3), [0, 0, 0])
    T1 = tf.homogeneous(tf.rotx(1.0), [1, 2, 3])
    Tm = tf.interpolate_pose(T0, T1, 0.5)
    np.testing.assert_allclose(Tm[:3, 3], [0.5, 1.0, 1.5])
    np.testing.assert_allclose(Tm[:3, :3], tf.rotx(0.5), atol=1e-12)


def test_orientation_error_is_left_rotation(rng):
    R, R_goal = random_rotation(rng), random_rotation(rng)
    e = tf.orientation_error(R_goal, R)
    np.testing.assert_allclose(tf.rot_exp(e) @ R, R_goal, atol=1e-12)


def test_wrap_angle():
    np.testing.assert_allclose(tf.wrap_angle(np.array([0.0, np.pi, -np.pi, 3 * np.pi / 2, -7.0])),
                               [0.0, np.pi, np.pi, -np.pi / 2, -7.0 + 2 * np.pi], atol=1e-15)
    assert isinstance(tf.wrap_angle(4.0), float)

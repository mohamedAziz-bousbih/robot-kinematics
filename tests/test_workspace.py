"""Monte Carlo workspace sampling against analytically known workspaces."""

from dataclasses import replace

import numpy as np
import pytest

from robokin import (
    Planar3R,
    Puma560,
    Scara,
    hit_or_miss_volume,
    occupied_area,
    occupied_volume,
    sample_workspace,
    two_link_area,
)
from robokin.transforms import homogeneous
from robokin.workspace import slab

FULL_TURN = ((-np.pi, np.pi),) * 3


def test_samples_respect_limits_and_match_fk(robot, rng):
    ws = sample_workspace(robot, 2000, rng, batch=512)
    assert ws.q.shape == (2000, robot.n) and ws.points.shape == (2000, 3)
    assert all(robot.within_limits(q) for q in ws.q)
    for q, p in zip(ws.q[:50], ws.points[:50]):
        np.testing.assert_allclose(robot.fk(q)[:3, 3], p, atol=1e-14)


def test_planar_annulus_radii(rng):
    """Without joint limits a planar 3R reaches an annulus r_in <= r <= l1 + l2 + l3."""
    robot = Planar3R(lengths=(0.6, 0.3, 0.1), qlim=FULL_TURN)
    r = np.linalg.norm(sample_workspace(robot, 100_000, rng).points[:, :2], axis=1)
    r_out, r_in = 1.0, 0.6 - 0.3 - 0.1
    assert r.max() <= r_out + 1e-12 and r.max() > 0.995 * r_out
    assert r.min() >= r_in - 1e-12 and r.min() < r_in + 0.005


def test_planar_area_estimate_converges_to_annulus(rng):
    robot = Planar3R(lengths=(0.6, 0.3, 0.1), qlim=FULL_TURN)
    pts = sample_workspace(robot, 400_000, rng).points[:, :2]
    exact = np.pi * (1.0**2 - 0.2**2)
    est = occupied_area(pts, cell=0.01)
    assert est == pytest.approx(exact, rel=0.03)


def test_scara_height_range(rng):
    robot = Scara()
    z = sample_workspace(robot, 20_000, rng).points[:, 2]
    d1, d4 = robot.links[0].d, robot.links[3].d
    assert z.min() >= d1 - 0.20 - d4 - 1e-12 and z.max() <= d1 - d4 + 1e-12


def test_puma_wrist_distance_range(rng):
    """Sampled wrist-centre distances fill exactly the analytic interval.

    The inner bound comes from the joint-3 limit (the elbow cannot fold
    completely), the outer one from the stretched arm.
    """
    robot = Puma560()
    r_min, r_max = robot.wrist_distance_range()
    L3, _ = robot.forearm
    assert r_max == pytest.approx(np.hypot(robot.links[1].a + L3, robot.links[2].d))
    dist = np.linalg.norm(sample_workspace(robot, 200_000, rng).points, axis=1)
    assert dist.min() >= r_min - 1e-12 and dist.min() < r_min + 0.005
    assert dist.max() <= r_max + 1e-12 and dist.max() > r_max - 0.005


def test_puma_wrist_distance_range_without_joint3_limit():
    """With a full joint-3 turn the arm folds and only the shoulder offset d3 remains."""
    robot = Puma560()
    robot.links = robot.links[:2] + (replace(robot.links[2], qlim=(-np.pi, np.pi)),) + robot.links[3:]
    a2, d3 = robot.links[1].a, robot.links[2].d
    L3, _ = robot.forearm
    r_min, _ = robot.wrist_distance_range()
    assert r_min == pytest.approx(np.hypot(L3 - a2, d3))


def test_two_link_area_without_limits_is_an_annulus():
    a1, a2 = 0.325, 0.275
    full = (-np.pi, np.pi)
    exact = np.pi * ((a1 + a2) ** 2 - (a1 - a2) ** 2)
    assert two_link_area(a1, a2, full, full) == pytest.approx(exact, rel=1e-6)
    # One elbow sign still reaches every point if joint 1 turns fully ...
    assert two_link_area(a1, a2, full, (0.0, np.pi)) == pytest.approx(exact, rel=1e-6)
    # ... and covers exactly half of the annulus if joint 1 only turns half way.
    assert two_link_area(a1, a2, (0.0, np.pi), (0.0, np.pi)) == pytest.approx(exact / 2, rel=1e-6)
    # With both elbow signs the two half-turn arcs overlap only partially.
    both = two_link_area(a1, a2, (0.0, np.pi), full)
    assert exact / 2 < both < exact
    with pytest.raises(ValueError):
        two_link_area(a1, a2, (1.0, 0.0), full)


def test_scara_monte_carlo_matches_exact_area(rng):
    """Monte Carlo top-view area of the SCARA against the semi-analytic integral."""
    robot = Scara()
    l1, l2 = robot.links[:2]
    exact = two_link_area(l1.a, l2.a, l1.qlim, l2.qlim)
    pts = sample_workspace(robot, 400_000, rng).points
    assert occupied_area(pts[:, :2], cell=0.005) == pytest.approx(exact, rel=0.03)


def test_wrist_reachable_accepts_every_sampled_wrist_position(rng):
    robot = Puma560()
    points = sample_workspace(robot, 50_000, rng).points
    assert np.all(robot.wrist_reachable(points))


def test_wrist_reachable_agrees_with_closed_form_ik(rng):
    """For random points in the reach box the vectorised test matches the scalar IK."""
    robot = Puma560()
    _, r_max = robot.wrist_distance_range()
    points = rng.uniform(-r_max, r_max, size=(1500, 3))
    verdict = robot.wrist_reachable(points)
    assert 0.2 < verdict.mean() < 0.8  # both classes are well represented
    lim = robot.qlim[:3]
    for p, reachable in zip(points, verdict):
        # The tool sits at the wrist centre, so any orientation will do;
        # only joints 1-3 decide whether the position is reachable.
        sols = robot.ik(homogeneous(np.eye(3), p), respect_limits=False)
        arm_ok = any(np.all((s.q[:3] >= lim[:, 0] - 1e-9) & (s.q[:3] <= lim[:, 1] + 1e-9)) for s in sols)
        assert arm_ok == reachable, p


def test_hit_or_miss_volume_of_a_ball(rng):
    def ball(p):
        return np.linalg.norm(p, axis=1) <= 1.0

    volume, se = hit_or_miss_volume(ball, [-1, -1, -1], [1, 1, 1], 400_000, rng, batch=150_000)
    exact = 4.0 / 3.0 * np.pi
    assert abs(volume - exact) < 4.0 * se
    assert se == pytest.approx(8.0 * np.sqrt(exact / 8 * (1 - exact / 8) / 400_000), rel=0.01)


def test_puma_hit_or_miss_volume_is_below_voxel_occupancy(rng):
    """Voxel occupancy counts partly covered boundary voxels fully and so overestimates."""
    robot = Puma560()
    _, r_max = robot.wrist_distance_range()
    volume, se = hit_or_miss_volume(robot.wrist_reachable, [-r_max] * 3, [r_max] * 3, 400_000, rng)
    # Points inside the reach shell but excluded by the joint limits must be rejected.
    r_min, _ = robot.wrist_distance_range()
    shell = 4.0 / 3.0 * np.pi * (r_max**3 - r_min**3)
    assert volume < shell
    occupancy = occupied_volume(sample_workspace(robot, 400_000, rng).points, 0.04)
    assert occupancy > volume + 5 * se


def test_occupancy_of_known_shapes(rng):
    square = rng.uniform(0, 1, size=(200_000, 2))
    assert occupied_area(square, 0.02) == pytest.approx(1.0, rel=0.02)
    cube = rng.uniform(0, 1, size=(200_000, 3))
    assert occupied_volume(cube, 0.05) == pytest.approx(1.0, rel=0.02)


def test_slab_selects_section():
    pts = np.array([[0, 0.0, 0], [0, 0.05, 1], [0, -0.2, 2]])
    np.testing.assert_array_equal(slab(pts, axis=1, center=0.0, half_width=0.1), pts[:2])

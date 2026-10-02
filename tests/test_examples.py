"""The example scripts run end to end and their printed results hold up."""

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workspace_plots(tmp_path):
    stats = load("workspace_plots").main(tmp_path, samples=20_000)
    assert (tmp_path / "workspace_top.png").is_file()
    assert (tmp_path / "workspace_sections.png").is_file()
    # Sampled extremes can only approach the analytic bounds from inside.
    assert stats["puma_r_min_sampled_m"] >= stats["puma_r_min_exact_m"] - 1e-12
    assert stats["puma_r_max_sampled_m"] <= stats["puma_r_max_exact_m"] + 1e-12
    # The hit-or-miss volume lies inside the spherical shell spanned by the reach range.
    r_min, r_max = stats["puma_r_min_exact_m"], stats["puma_r_max_exact_m"]
    assert 0.0 < stats["puma_volume_hit_or_miss_m3"] < 4.0 / 3.0 * np.pi * (r_max**3 - r_min**3)


def test_ik_solutions(tmp_path):
    rows = load("ik_solutions").main(tmp_path)
    assert (tmp_path / "ik_puma_solutions.png").is_file()
    assert len(rows) == 8
    assert all(err < 1e-9 for _, _, _, err in rows)
    assert any(within for _, _, within, _ in rows)


def test_trajectories(tmp_path):
    stats = load("trajectories").main(tmp_path)
    assert (tmp_path / "ptp_profiles.png").is_file()
    assert (tmp_path / "lin_motion.png").is_file()
    # Synchronising can only make a PTP move longer than its slowest axis.
    assert stats["trapezoid/full"] >= stats["trapezoid/sync"] == pytest.approx(stats["trapezoid/async"])
    assert stats["lin_peak_speed_ratio"] <= 1.0 + 1e-6
    assert stats["lin_peak_omega_ratio"] <= 1.0 + 1e-6


def test_collision_demo(tmp_path):
    results = load("collision_demo").main(tmp_path)
    assert (tmp_path / "collision_check.png").is_file()
    direct, detour = results["direct"], results["via point"]
    assert direct["check"].collision and direct["dense_min_clearance"] < 0.0
    assert detour["check"].certified and detour["dense_min_clearance"] > 0.0


def test_ik_benchmark():
    results = load("ik_benchmark").main(n_poses=10)
    assert len(results) == 3
    for r in results.values():
        assert r["closed_form_found_source"] == 1.0
        assert r["closed_form_max_fk_error"] < 1e-9
        assert r["restarts_success"] >= 0.9
        assert r["dls_near_seed_success"] == 1.0

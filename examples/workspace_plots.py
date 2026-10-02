"""Monte Carlo workspaces of the three robots: top views and vertical sections.

Writes ``workspace_top.png`` and ``workspace_sections.png`` and prints the
Monte Carlo estimates next to the analytic references where one exists.
Run from the repository root::

    python examples/workspace_plots.py                       # figures into docs/img
    python examples/workspace_plots.py --out /tmp/ws --samples 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from robokin import (  # noqa: E402
    Planar3R,
    Puma560,
    Scara,
    hit_or_miss_volume,
    occupied_area,
    occupied_volume,
    sample_workspace,
    two_link_area,
)
from robokin import plotting as rp  # noqa: E402
from robokin.workspace import slab  # noqa: E402

SEED = 7
SECTION_HALF_WIDTH = 0.01  # slabs of +-1 cm around the section plane


def circle(ax, radius: float, **kw) -> None:
    t = np.linspace(0.0, 2.0 * np.pi, 361)
    ax.plot(radius * np.cos(t), radius * np.sin(t), color=rp.INK_SECONDARY, lw=0.9, **kw)


def main(out: Path, samples: int = 1_000_000) -> dict[str, float]:
    rp.apply_style()
    rng = np.random.default_rng(SEED)
    planar, scara, puma = Planar3R(), Scara(), Puma560()
    ws_planar = sample_workspace(planar, samples, rng)
    ws_scara = sample_workspace(scara, samples, rng)
    ws_puma = sample_workspace(puma, 2 * samples, rng)

    l1, l2, l3 = scara.links[:3]
    scara_r_in = np.sqrt(l1.a**2 + l2.a**2 + 2.0 * l1.a * l2.a * np.cos(l2.qlim[1]))
    scara_exact_area = two_link_area(l1.a, l2.a, l1.qlim, l2.qlim)
    puma_r_min, puma_r_max = puma.wrist_distance_range()
    puma_dist = np.linalg.norm(ws_puma.points, axis=1)
    # The PUMA model has its TCP at the wrist centre, so the exact wrist
    # reachability test classifies TCP positions directly.
    box = [puma_r_max] * 3
    puma_hm, puma_hm_se = hit_or_miss_volume(
        puma.wrist_reachable, [-b for b in box], box, 2 * samples, rng
    )
    stats = {
        "planar_area_m2": occupied_area(ws_planar.points[:, :2], 0.01),
        "planar_reach_m": ws_planar.max_reach,
        "scara_area_mc_m2": occupied_area(ws_scara.points[:, :2], 0.005),
        "scara_area_exact_m2": scara_exact_area,
        "scara_volume_mc_m3": occupied_volume(ws_scara.points, 0.02),
        "scara_volume_exact_m3": scara_exact_area * (l3.qlim[1] - l3.qlim[0]),
        "puma_volume_mc_m3": occupied_volume(ws_puma.points, 0.02),
        "puma_volume_hit_or_miss_m3": puma_hm,
        "puma_volume_hit_or_miss_se_m3": puma_hm_se,
        "puma_r_min_sampled_m": float(puma_dist.min()),
        "puma_r_min_exact_m": puma_r_min,
        "puma_r_max_sampled_m": float(puma_dist.max()),
        "puma_r_max_exact_m": puma_r_max,
    }

    # ------------------------------------------------------------ top views
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
    ax = axes[0]
    images = [rp.density_image(ax, ws_planar.points[:, :2], (-1.15, 1.15, -1.15, 1.15), bins=150)]
    circle(ax, sum(planar.lengths))
    for q, color in ((np.deg2rad([30, 60, -40]), rp.SERIES[1]), (np.deg2rad([150, -110, 90]), rp.SERIES[2])):
        rp.draw_arm_2d(ax, planar, q, color=color)
    ax.set_title("Planar 3R: reachable TCP positions\ncircle: l1 + l2 + l3; two sample postures", fontsize=9)

    ax = axes[1]
    images.append(rp.density_image(ax, ws_scara.points[:, :2], (-0.65, 0.65, -0.65, 0.65), bins=150))
    circle(ax, l1.a + l2.a)
    circle(ax, scara_r_in)
    rp.draw_arm_2d(ax, scara, np.array([np.deg2rad(40), np.deg2rad(70), 0.1, 0.0]), color=rp.SERIES[1])
    ax.set_title("SCARA: top view, all heights\ncircles: a1 + a2 and the q2-limit radius", fontsize=9)

    ax = axes[2]
    top = slab(ws_puma.points, axis=2, center=0.0, half_width=SECTION_HALF_WIDTH)
    images.append(rp.density_image(ax, top[:, :2], (-0.95, 0.95, -0.95, 0.95), bins=110))
    circle(ax, puma_r_max)
    circle(ax, puma_r_min)
    ax.set_title(
        f"PUMA 560: section z = 0 (+-{SECTION_HALF_WIDTH * 100:.0f} cm)\n"
        "circles: analytic wrist-centre distance range",
        fontsize=9,
    )

    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
    rp.share_norm(images)
    fig.colorbar(images[0], ax=axes, shrink=0.8, label="samples per cell (log)")
    rp.save(fig, out / "workspace_top.png")

    # ----------------------------------------------------- vertical sections
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    ax = axes[0]
    r = np.hypot(ws_scara.points[:, 0], ws_scara.points[:, 1])
    radial = np.column_stack([r, ws_scara.points[:, 2]])
    images = [rp.density_image(ax, radial, (0.0, 0.65, 0.0, 0.5), bins=130)]
    ax.axhline(l1.d, color=rp.INK_SECONDARY, lw=0.9)
    ax.set_title(
        "SCARA: radial section, r = sqrt(x^2 + y^2)\nline: arm height d1; band: 0.2 m quill stroke",
        fontsize=9,
    )
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")

    ax = axes[1]
    side = slab(ws_puma.points, axis=1, center=0.0, half_width=SECTION_HALF_WIDTH)
    images.append(rp.density_image(ax, side[:, [0, 2]], (-0.95, 0.95, -0.95, 0.95), bins=110))
    ax.plot([0.0], [0.0], "o", color=rp.INK, ms=4)
    circle(ax, puma_r_max)
    circle(ax, puma_r_min)
    d3 = puma.links[2].d
    for x in (-d3, d3):
        ax.axvline(x, color=rp.INK_SECONDARY, lw=0.9, ls="--")
    ax.set_title(
        f"PUMA 560: section y = 0 (+-{SECTION_HALF_WIDTH * 100:.0f} cm)\n"
        "dot: frame 0; dashed: x = +-d3 (shoulder offset)",
        fontsize=9,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    for ax in axes:
        ax.set_aspect("equal")
    rp.share_norm(images)
    fig.colorbar(images[0], ax=axes, shrink=0.8, label="samples per cell (log)")
    rp.save(fig, out / "workspace_sections.png")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "img")
    parser.add_argument("--samples", type=int, default=1_000_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    s = main(args.out, args.samples)
    print(f"samples: {args.samples} (planar 3R, SCARA), {2 * args.samples} (PUMA 560), seed {SEED}")
    print(f"planar 3R  area, 10 mm cells        : {s['planar_area_m2']:.4f} m^2")
    print(f"planar 3R  max sampled reach        : {s['planar_reach_m']:.4f} m")
    print(f"SCARA      area, 5 mm cells         : {s['scara_area_mc_m2']:.4f} m^2 "
          f"(exact {s['scara_area_exact_m2']:.4f})")
    print(f"SCARA      volume, 20 mm voxels     : {s['scara_volume_mc_m3']:.4f} m^3 "
          f"(exact {s['scara_volume_exact_m3']:.4f})")
    print(f"PUMA 560   volume, 20 mm voxels     : {s['puma_volume_mc_m3']:.4f} m^3")
    print(f"PUMA 560   volume, hit-or-miss      : {s['puma_volume_hit_or_miss_m3']:.4f} m^3 "
          f"(standard error {s['puma_volume_hit_or_miss_se_m3']:.4f})")
    print(f"PUMA 560   wrist distance, sampled  : {s['puma_r_min_sampled_m']:.4f} .. "
          f"{s['puma_r_max_sampled_m']:.4f} m")
    print(f"PUMA 560   wrist distance, exact    : {s['puma_r_min_exact_m']:.4f} .. "
          f"{s['puma_r_max_exact_m']:.4f} m")

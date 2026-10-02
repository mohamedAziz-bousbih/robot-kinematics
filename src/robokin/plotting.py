"""Matplotlib helpers for drawing arms, obstacles and workspace densities.

Only this module imports matplotlib, so the kinematics core needs nothing but
numpy. Figures use an opaque light surface so they stay readable when the
PNGs are shown on a dark page.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from .collision import Sphere
from .dh import SerialRobot

# Categorical slots in fixed order (never cycled): series i always gets SERIES[i].
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CRITICAL = "#d03b3b"  # reserved for "collision" / limit violations

# One-hue sequential ramp (light -> dark) for sample densities.
DENSITY = LinearSegmentedColormap.from_list(
    "robokin_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def apply_style() -> None:
    """Set rcParams for all example figures."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "font.family": "sans-serif",
        }
    )


def frame_origins(robot: SerialRobot, q: np.ndarray) -> np.ndarray:
    """World positions of all frame origins (base, DH frames 1..n, tool), shape (n + 2, 3)."""
    return robot.fk_frames(q)[:, :3, 3]


def draw_arm_3d(ax, robot: SerialRobot, q: np.ndarray, color: str = SERIES[0], **kw) -> None:
    """Stick figure through the frame origins on a 3-D axes."""
    p = frame_origins(robot, q)
    ax.plot(p[:, 0], p[:, 1], p[:, 2], "-", color=color, lw=kw.pop("lw", 2.5), **kw)
    ax.scatter(p[:-1, 0], p[:-1, 1], p[:-1, 2], s=12, color=INK_SECONDARY, depthshade=False)


def draw_arm_2d(
    ax, robot: SerialRobot, q: np.ndarray, axes: tuple[int, int] = (0, 1), color: str = SERIES[0], **kw
) -> None:
    """Stick figure projected onto two world axes (x-y by default)."""
    p = frame_origins(robot, q)[:, list(axes)]
    ax.plot(p[:, 0], p[:, 1], "-", color=color, lw=kw.pop("lw", 2.5), **kw)
    ax.plot(p[:-1, 0], p[:-1, 1], "o", color=INK_SECONDARY, ms=3.5)


FRAME_COLORS = (SERIES[1], SERIES[2], INK)  # x orange, y aqua, z (approach) near-black


def draw_frame_3d(ax, T: np.ndarray, length: float = 0.1) -> None:
    """Coordinate axes of a pose, coloured with :data:`FRAME_COLORS`."""
    for k in range(3):
        end = T[:3, 3] + length * T[:3, k]
        ax.plot(*zip(T[:3, 3], end), color=FRAME_COLORS[k], lw=1.5)


def tidy_3d(ax, ticks: Sequence[float]) -> None:
    """Few ticks and a plain background on a 3-D axes."""
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.tick_params(labelsize=6, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(SURFACE)
        axis.pane.set_edgecolor(GRID)


def draw_sphere_3d(ax, sphere: Sphere, color: str = CRITICAL, alpha: float = 0.25) -> None:
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 24), np.linspace(0, np.pi, 12))
    c, r = np.asarray(sphere.center), sphere.radius
    ax.plot_surface(
        c[0] + r * np.cos(u) * np.sin(v),
        c[1] + r * np.sin(u) * np.sin(v),
        c[2] + r * np.cos(v),
        color=color, alpha=alpha, linewidth=0,
    )


def set_equal_3d(ax, center: Sequence[float], half: float) -> None:
    """Equal scaling on a 3-D axes (matplotlib has no reliable 'equal' for 3-D)."""
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(c - half, c + half)
    ax.set_box_aspect((1, 1, 1))


def density_image(ax, points_2d: np.ndarray, extent: Sequence[float], bins: int):
    """2-D histogram of sample points as an image; empty cells show the surface.

    Counts are drawn on a log scale because joint-space sampling is far
    denser in some regions than near the outer boundary. Returns the image;
    panels that share a colour bar must go through :func:`share_norm`.
    """
    x0, x1, y0, y1 = extent
    counts, _, _ = np.histogram2d(points_2d[:, 0], points_2d[:, 1], bins=bins, range=[[x0, x1], [y0, y1]])
    ax.grid(False)
    return ax.imshow(
        np.ma.masked_equal(counts.T, 0), origin="lower", extent=extent, cmap=DENSITY,
        norm=LogNorm(vmin=1, vmax=max(counts.max(), 1)), interpolation="nearest", aspect="equal",
    )


def share_norm(images) -> None:
    """Give several density images one common colour scale."""
    vmax = max(float(img.get_array().max()) for img in images)
    for img in images:
        img.set_norm(LogNorm(vmin=1, vmax=vmax))


def save(fig, path) -> None:
    """Save with the settings used for every committed PNG."""
    fig.savefig(path, dpi=110, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)

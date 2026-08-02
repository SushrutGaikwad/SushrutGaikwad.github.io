# Runs via the root uv environment with uv run (see command at the bottom).
"""Why a network of straight-line units can fence off any region at all.

Left: one convex polygon is the AND of the half-planes cut by its own sides.
Middle: two disjoint polygons are the OR of two such sub-networks, which is what
forces a third layer. Right: a region with a curved boundary is neither, but it
is the OR of many small polygons, and shrinking the polygons shrinks the error
without limit. Three panels, one escalating argument.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import ACCENT, BG, FG, GOOD, MUTED, use_dark_theme

use_dark_theme()


def regular_polygon(n, radius=1.0, centre=(0.0, 0.0), rotation=np.pi / 2):
    """Vertices of a regular n-gon, counter-clockwise."""
    angles = rotation + np.arange(n) * 2 * np.pi / n
    return np.column_stack([centre[0] + radius * np.cos(angles),
                            centre[1] + radius * np.sin(angles)])


def draw_side_lines(ax, verts, span=6.0, color=MUTED, alpha=0.45):
    """Extend each side of the polygon into the full line the unit actually cuts."""
    n = len(verts)
    for i in range(n):
        p, q = verts[i], verts[(i + 1) % n]
        d = q - p
        d = d / np.linalg.norm(d)
        a, b = p - span * d, p + span * d
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=0.9, alpha=alpha,
                zorder=1)


def blob_radius(theta):
    """A closed curve with no straight pieces anywhere on it."""
    return 1.0 + 0.28 * np.sin(3 * theta) + 0.13 * np.cos(5 * theta - 0.7)


def inside_blob(x, y):
    theta = np.arctan2(y, x)
    return np.hypot(x, y) <= blob_radius(theta)


# Kept as 1x3, because the three panels are an escalating argument that has to
# be read left to right. But narrowed from 15in to 11.5in with larger type: at
# 15in the titles rendered at about half the size of the post's body text.
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6))

# ---- Panel 1: one pentagon = AND of five half-planes ----
ax = axes[0]
pent = regular_polygon(5, radius=1.15)
draw_side_lines(ax, pent)
ax.add_patch(Polygon(pent, closed=True, facecolor=ACCENT, alpha=0.35,
                     edgecolor=ACCENT, lw=2.2, zorder=3))
ax.set_title("One convex region\n"
             "5 units, ANDed by a 6th",
             fontsize=16, pad=12)

# ---- Panel 2: two pentagons = OR of two such sub-networks ----
ax = axes[1]
left = regular_polygon(5, radius=0.70, centre=(-1.00, -0.12))
right = regular_polygon(5, radius=0.70, centre=(1.00, 0.12), rotation=-np.pi / 2)
for verts in (left, right):
    draw_side_lines(ax, verts, alpha=0.30)
    ax.add_patch(Polygon(verts, closed=True, facecolor=ACCENT, alpha=0.35,
                         edgecolor=ACCENT, lw=2.2, zorder=3))
ax.annotate("OR", xy=(0.0, 0.0), ha="center", va="center", fontsize=18,
            color=GOOD, zorder=5)
ax.set_title("Two disjoint regions\n"
             "ORed by a third layer",
             fontsize=16, pad=12)

# ---- Panel 3: a curved region = OR of many small polygons ----
ax = axes[2]
step = 0.135
grid = np.arange(-1.6, 1.6 + step, step)
for gx in grid:
    for gy in grid:
        cx, cy = gx + step / 2, gy + step / 2
        if inside_blob(cx, cy):
            ax.add_patch(Rectangle((gx, gy), step, step, facecolor=ACCENT,
                                   alpha=0.32, edgecolor=MUTED, lw=0.35,
                                   zorder=2))
theta = np.linspace(0, 2 * np.pi, 800)
r = blob_radius(theta)
ax.plot(r * np.cos(theta), r * np.sin(theta), color=GOOD, lw=2.4, zorder=4)
ax.set_title("Any region at all\n"
             "tile it, then OR the tiles",
             fontsize=16, pad=12)

for ax in axes:
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.75, 1.75)
    ax.set_aspect("equal")
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_alpha(0.4)
    ax.set_xlabel(r"$x_1$", fontsize=14)
axes[0].set_ylabel(r"$x_2$", fontsize=14)

fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.11, wspace=0.14)

out = Path(__file__).resolve().parent.parent / "images" / "polytope_composition.png"
fig.savefig(out, dpi=200, facecolor=BG)
print(f"wrote {out}")

# Run (from the repo root):
#   uv run python blog/published/neural-networks-history-of-an-idea/_src/polytope_composition.py
# Output: blog/published/neural-networks-history-of-an-idea/images/polytope_composition.png
# Embed with:
#   ![...](images/polytope_composition.png){#fig-polytope_composition fig-align="center" width=100%}

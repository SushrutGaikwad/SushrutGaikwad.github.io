# Runs via the root uv environment with uv run (see command at the bottom).
"""Five straight-line units, and a sixth that counts them.

Each unit fires on one side of one line, so after five of them every point in
the plane carries a number: how many of the five happen to be firing there. The
number is 5 in exactly one place, the pentagon, because that is the only region
on the correct side of all five lines at once. Thresholding the count at 5 is
an AND, and the AND is the sixth unit.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import ACCENT, BG, FG, GOOD, MUTED, QUIET, use_dark_theme

use_dark_theme()

N_SIDES = 5
RADIUS = 1.25
LIM = 3.1

angles = np.pi / 2 + np.arange(N_SIDES) * 2 * np.pi / N_SIDES
verts = np.column_stack([RADIUS * np.cos(angles), RADIUS * np.sin(angles)])

gx, gy = np.meshgrid(np.linspace(-LIM, LIM, 420), np.linspace(-LIM, LIM, 420))


def side_indicator(i, x, y):
    """1 where the i-th unit fires, i.e. on the polygon's side of side i."""
    p, q = verts[i], verts[(i + 1) % N_SIDES]
    edge = q - p
    return ((edge[0] * (y - p[1]) - edge[1] * (x - p[0])) >= 0).astype(float)


counts = [side_indicator(i, gx, gy) for i in range(N_SIDES)]
cum = np.cumsum(np.array(counts), axis=0)          # cum[k] = count after k+1 units

# Points at which to print the count, and the counts actually there.
label_pts = [(0.0, 0.0)]
for i in range(N_SIDES):
    mid = (verts[i] + verts[(i + 1) % N_SIDES]) / 2
    label_pts.append(tuple(mid * 1.85))            # just outside one side
    label_pts.append(tuple(verts[i] * 2.35))       # out past a vertex


def count_at(x, y, k):
    return int(sum(side_indicator(i, np.array([x]), np.array([y]))[0]
                   for i in range(k + 1)))


# A ramp from the page background up to the highlight colour, one step per unit.
cmap = LinearSegmentedColormap.from_list(
    "count", ["#242424", "#3b3a2a", "#585330", "#7d7136", "#ad9a3a", ACCENT], N=6)

# ---- Frame schedule -------------------------------------------------------
BEAT = 42                       # ~1.4 s per unit
frames = []
for k in range(N_SIDES):
    frames += [("add", k)] * BEAT
frames += [("count", N_SIDES - 1)] * 75      # 2.5 s: read the numbers
frames += [("and", N_SIDES - 1)] * 100       # 3.3 s: the AND fires

fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=180)
fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.09)


def draw(frame):
    phase, k = frame
    ax.clear()
    ax.set_facecolor(BG)

    if phase == "and":
        ax.contourf(gx, gy, cum[k], levels=[N_SIDES - 0.5, N_SIDES + 0.5],
                    colors=[ACCENT], alpha=0.8)
        ax.contourf(gx, gy, cum[k], levels=[-0.5, N_SIDES - 0.5],
                    colors=["#242424"], alpha=1.0)
    else:
        ax.contourf(gx, gy, cum[k], levels=np.arange(-0.5, N_SIDES + 1, 1.0),
                    cmap=cmap)

    # Every line placed so far. The newest one is bright.
    for i in range(k + 1):
        p, q = verts[i], verts[(i + 1) % N_SIDES]
        direction = (q - p) / np.linalg.norm(q - p)
        a, b = p - 8 * direction, p + 8 * direction
        newest = (i == k and phase == "add")
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color=GOOD if newest else MUTED,
                lw=2.6 if newest else 1.2,
                alpha=1.0 if newest else 0.65, zorder=3)

    if phase in ("count", "and"):
        for (px, py) in label_pts:
            c = count_at(px, py, k)
            ax.text(px, py, str(c), ha="center", va="center", fontsize=15,
                    color=GOOD if c == N_SIDES else FG, zorder=6,
                    bbox=dict(facecolor=BG, edgecolor="none", alpha=0.55,
                              pad=1.5))

    if phase == "and":
        ax.plot(np.append(verts[:, 0], verts[0, 0]),
                np.append(verts[:, 1], verts[0, 1]),
                color=GOOD, lw=3.0, zorder=5)

    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r"$x_1$", fontsize=13)
    ax.set_ylabel(r"$x_2$", fontsize=13)

    if phase == "add":
        title = f"Unit {k + 1} of 5 fires on one side of one line"
        sub = ("brighter shading means more of the units placed so far "
               "are firing there")
        colour = FG
    elif phase == "count":
        title = "Every point now carries a count"
        sub = "the count reaches 5 in exactly one region, and nowhere else"
        colour = FG
    else:
        title = r"A sixth unit thresholds the count at 5"
        sub = (r"$y_1 + \cdots + y_5 \geq 5$ is an AND, and the network now "
               r"fires only inside the pentagon")
        colour = GOOD

    ax.set_title(title, fontsize=15, pad=40, color=colour)
    ax.text(0.5, 1.045, sub, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11, color=MUTED)
    return ()


ani = FuncAnimation(fig, draw, frames=frames, interval=1000 / 30, blit=False)

out = Path(__file__).resolve().parent.parent / "images" / "pentagon_construction.mp4"
ani.save(out, writer="ffmpeg", fps=30, dpi=180,
         savefig_kwargs={"facecolor": BG},
         extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                     "-crf", "20", "-preset", "slow"])
print(f"wrote {out}")

# Run (from the repo root):
#   uv run python blog/published/neural-networks-history-of-an-idea/_src/pentagon_construction.py
# Output: blog/published/neural-networks-history-of-an-idea/images/pentagon_construction.mp4
# Embed as a numbered figure with:
#   ::: {#fig-pentagon_construction}
#   {{< video images/pentagon_construction.mp4 >}}
#
#   Caption text.
#   :::

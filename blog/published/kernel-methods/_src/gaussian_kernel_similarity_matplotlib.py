# Runs via the root uv environment:
#   uv run python blog/published/kernel-methods/_src/gaussian_kernel_similarity_matplotlib.py
#
# STATIC PNG illustrating the "kernel = similarity" view via the Gaussian kernel
#       K(x,z) = exp(-||x - z||^2 / (2 sigma^2)).
#   LEFT:   a strip showing the 14 points themselves, sitting directly above the
#           14-by-14 kernel matrix they generate, so cell (i, j) can be read straight
#           off the distance between point i and point j. Two cells are called out:
#           a neighbouring pair (bright) and a far-apart pair (dark).
#   RIGHT:  K as a function of the distance ||x - z||, for three bandwidths sigma.
#           Every curve starts at 1 (identical points) and decays toward 0;
#           larger sigma = a wider notion of "close".

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import BG, CURVE, FG, GOOD, GRID, LINE, MUTED, POINT, use_dark_theme

use_dark_theme()


def gaussian_kernel_matrix(points, sigma):
    diff = points[:, None] - points[None, :]
    return np.exp(-(diff ** 2) / (2.0 * sigma ** 2))


n = 14
pts = np.linspace(0.0, 6.0, n)
K = gaussian_kernel_matrix(pts, sigma=1.0)

NEAR = (2, 3)    # a neighbouring pair: bright cell
FAR = (2, 12)    # a far-apart pair: dark cell

fig = plt.figure(figsize=(12.4, 5.6), dpi=300)
gs = fig.add_gridspec(2, 2, height_ratios=[0.20, 1.0], width_ratios=[1.0, 1.15],
                      hspace=0.30, wspace=0.40)

ax_strip = fig.add_subplot(gs[0, 0])
ax_mat = fig.add_subplot(gs[1, 0], sharex=ax_strip)
ax_decay = fig.add_subplot(gs[:, 1])

# ---- STRIP: the 14 points, drawn against the matrix columns ----
ax_strip.scatter(range(n), np.zeros(n), marker="o", s=34, color=MUTED, zorder=3)
for idx, color in ((NEAR[0], GOOD), (NEAR[1], GOOD), (FAR[1], LINE)):
    ax_strip.scatter([idx], [0.0], marker="o", s=90, color=color, zorder=4)
ax_strip.axhline(0.0, color=GRID, lw=1.0, zorder=1)
ax_strip.set_ylim(-0.6, 0.6)
ax_strip.set_facecolor(BG)
ax_strip.set_yticks([])
for side in ("top", "right", "left", "bottom"):
    ax_strip.spines[side].set_visible(False)
ax_strip.tick_params(labelbottom=False, bottom=False)
ax_strip.set_title("the 14 points, evenly spaced on a line",
                   fontsize=11.5, color=MUTED, pad=6)

# ---- MATRIX ----
im = ax_mat.imshow(K, cmap="magma", vmin=0.0, vmax=1.0, origin="upper",
                   interpolation="nearest")
ax_mat.set_xlabel(r"$j$", fontsize=12)
ax_mat.set_ylabel(r"$i$", fontsize=12)
ax_mat.set_xticks(range(0, n, 2))
ax_mat.set_yticks(range(0, n, 2))
# Steal colorbar space from BOTH panels so the strip stays aligned with the columns.
cbar = fig.colorbar(im, ax=[ax_strip, ax_mat], fraction=0.046, pad=0.03)
cbar.set_label(r"similarity $K(x_i, x_j)$", color=FG)
cbar.ax.yaxis.set_tick_params(color=FG)
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=FG)

for (i, j), color, note, tx in ((NEAR, GOOD, "neighbours: bright", -0.4),
                                (FAR, LINE, "far apart: dark", 7.6)):
    ax_mat.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor=color, lw=2.2, zorder=5))
    ax_mat.annotate(note, xy=(j, i - 0.5), xytext=(tx, -1.9),
                    fontsize=10.5, color=color, ha="left", va="bottom",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.1))

# ---- RIGHT: decay of K with distance for three bandwidths ----
d = np.linspace(0.0, 5.0, 400)
for sigma, c in zip([0.5, 1.0, 2.0], [POINT, CURVE, LINE]):
    ax_decay.plot(d, np.exp(-(d ** 2) / (2.0 * sigma ** 2)),
                  color=c, lw=2.4, label=rf"$\sigma = {sigma}$")
ax_decay.set_title(r"$K(x,z) = \exp\!\left(-\,\|x-z\|^2 / 2\sigma^2\right)$",
                   fontsize=12.5, pad=8)
ax_decay.set_xlabel(r"distance $\|x - z\|$", fontsize=12)
ax_decay.set_ylabel(r"$K(x,z)$", fontsize=12)
ax_decay.set_facecolor(BG)
ax_decay.grid(True, color=GRID, linestyle="--", linewidth=0.4, alpha=0.5)
ax_decay.set_xlim(0, 5)
ax_decay.set_ylim(-0.02, 1.05)
ax_decay.text(0.97, 0.55,
              "identical points\nscore 1; distant\npoints score ~0",
              transform=ax_decay.transAxes, fontsize=10.5, color=MUTED,
              ha="right", va="top")
leg = ax_decay.legend(frameon=False, fontsize=11)
for txt in leg.get_texts():
    txt.set_color(FG)

# The colorbar reflows both left-hand axes, so pin the strip to the matrix's exact
# horizontal extent afterwards: every dot must sit over its own matrix column.
fig.canvas.draw()
pos_mat = ax_mat.get_position()
pos_strip = ax_strip.get_position()
ax_strip.set_position([pos_mat.x0, pos_strip.y0, pos_mat.width, pos_strip.height])
ax_strip.set_xlim(ax_mat.get_xlim())

out = Path(__file__).resolve().parent.parent / "images" / "gaussian_kernel_similarity.png"
plt.savefig(out, dpi=300, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# Output: images/gaussian_kernel_similarity.png
# Embed in Quarto (dark figure, so no .invert):
#   ![Caption](images/gaussian_kernel_similarity.png){#fig-gaussian_kernel_similarity fig-align="center" width=100%}

# Runs via the root uv environment:
#   uv run python blog/published/kernel-methods/_src/gaussian_kernel_similarity_matplotlib.py
#
# STATIC PNG (2 panels) illustrating the "kernel = similarity" view via the
# Gaussian kernel  K(x,z) = exp(-||x - z||^2 / (2 sigma^2)).
#   LEFT:   the n-by-n kernel matrix on sorted 1D points. Bright (~1) on the
#           diagonal (a point is maximally similar to itself), fading to dark (~0)
#           away from it -> nearby points are similar, far points are not.
#   RIGHT:  K as a function of the distance ||x - z||, for three bandwidths sigma.
#           Every curve starts at 1 (identical points) and decays toward 0;
#           larger sigma = a wider notion of "close".

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Dark theme matching the darkly Quarto theme
plt.style.use("dark_background")
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.facecolor": "#222222",
    "figure.facecolor": "#222222",
    "savefig.facecolor": "#222222",
    "axes.edgecolor": "#e6e6e6",
    "axes.labelcolor": "#e6e6e6",
    "xtick.color": "#e6e6e6",
    "ytick.color": "#e6e6e6",
    "axes.titlecolor": "#e6e6e6",
    "text.color": "#e6e6e6",
})


def gaussian_kernel_matrix(points, sigma):
    diff = points[:, None] - points[None, :]
    return np.exp(-(diff ** 2) / (2.0 * sigma ** 2))


fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

# ---- LEFT: kernel matrix heatmap on sorted 1D points ----
n = 14
pts = np.linspace(0.0, 6.0, n)
K = gaussian_kernel_matrix(pts, sigma=1.0)

im = axes[0].imshow(K, cmap="magma", vmin=0.0, vmax=1.0, origin="upper")
axes[0].set_title(r"Kernel matrix $K_{ij} = K(x_i, x_j)$"
                  "\n(Gaussian kernel, sorted points)", fontsize=12.5, pad=8)
axes[0].set_xlabel(r"$j$", fontsize=12)
axes[0].set_ylabel(r"$i$", fontsize=12)
axes[0].set_xticks(range(0, n, 2))
axes[0].set_yticks(range(0, n, 2))
cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
cbar.set_label(r"similarity $K(x_i, x_j)$", color="#e6e6e6")
cbar.ax.yaxis.set_tick_params(color="#e6e6e6")
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#e6e6e6")

# ---- RIGHT: decay of K with distance for three bandwidths ----
d = np.linspace(0.0, 5.0, 400)
sigmas = [0.5, 1.0, 2.0]
colors = ["#e74c3c", "#f1c40f", "#5dade2"]
for sigma, c in zip(sigmas, colors):
    axes[1].plot(d, np.exp(-(d ** 2) / (2.0 * sigma ** 2)),
                 color=c, lw=2.4, label=rf"$\sigma = {sigma}$")
axes[1].set_title(r"$K(x,z) = \exp\!\left(-\,\|x-z\|^2 / 2\sigma^2\right)$",
                  fontsize=12.5, pad=8)
axes[1].set_xlabel(r"distance $\|x - z\|$", fontsize=12)
axes[1].set_ylabel(r"$K(x,z)$", fontsize=12)
axes[1].grid(True, color="#444444", linestyle="--", linewidth=0.4, alpha=0.5)
axes[1].set_xlim(0, 5)
axes[1].set_ylim(-0.02, 1.05)
leg = axes[1].legend(frameon=False, fontsize=11)
for txt in leg.get_texts():
    txt.set_color("#e6e6e6")

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "images" / "gaussian_kernel_similarity.png"
plt.savefig(out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# Output: images/gaussian_kernel_similarity.png
# Embed in Quarto (dark figure, so no .invert):
#   ![Caption](images/gaussian_kernel_similarity.png){#fig-gaussian_kernel_similarity fig-align="center" width=100%}

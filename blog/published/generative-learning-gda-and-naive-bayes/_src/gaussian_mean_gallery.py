# Runs via the root uv environment: uv run python blog/published/generative-learning-gda-and-naive-bayes/_src/gaussian_mean_gallery.py
#
# STATIC PNG: a row of 2D contour plots of bivariate Gaussians, all with
# identity covariance (so each is circular), but with the mean vector shifted
# to different locations. Shows that the mean simply moves the center of the
# density around the plane.

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---- Dark theme matching the darkly Quarto theme ----
# usetex gives genuine LaTeX bold for the mean vector symbol in the titles
# (a vector, so shown bold); requires a LaTeX install on PATH.
plt.style.use("dark_background")
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
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


def gaussian_density(grid_x, grid_y, mean):
    """Identity-covariance bivariate Gaussian density on a meshgrid."""
    norm = 1.0 / (2.0 * np.pi)
    quad = (grid_x - mean[0]) ** 2 + (grid_y - mean[1]) ** 2
    return norm * np.exp(-0.5 * quad)


means = [
    (np.array([1.0, 0.0]), r"$\boldsymbol{\mu} = (1,\ 0)$"),
    (np.array([-0.5, 0.0]), r"$\boldsymbol{\mu} = (-0.5,\ 0)$"),
    (np.array([-1.0, -1.5]), r"$\boldsymbol{\mu} = (-1,\ -1.5)$"),
]

lim = 3.0
axis = np.linspace(-lim, lim, 400)
gx, gy = np.meshgrid(axis, axis)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), dpi=300)

accent = "#f1c40f"
mark = "#e74c3c"
for ax, (mean, title) in zip(axes, means):
    density = gaussian_density(gx, gy, mean)
    levels = np.linspace(density.max() * 0.03, density.max() * 0.95, 6)
    ax.contour(gx, gy, density, levels=levels, colors=accent, linewidths=1.4)
    ax.plot(mean[0], mean[1], "x", color=mark, markersize=11, markeredgewidth=2.6, zorder=5)
    ax.axhline(0, color="#555555", lw=0.6, alpha=0.6)
    ax.axvline(0, color="#555555", lw=0.6, alpha=0.6)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])

plt.tight_layout()
_out = Path(__file__).resolve().parent.parent / "images" / "gaussian_mean_gallery.png"
plt.savefig(_out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print("saved", _out)

# Embed in Quarto:
#   ![Caption](images/gaussian_mean_gallery.png){#fig-gaussian_mean_gallery fig-align="center" width=100%}

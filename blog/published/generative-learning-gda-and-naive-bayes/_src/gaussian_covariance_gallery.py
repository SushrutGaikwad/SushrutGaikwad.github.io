# Runs via the root uv environment: uv run python blog/published/generative-learning-gda-and-naive-bayes/_src/gaussian_covariance_gallery.py
#
# STATIC PNG: a row of 2D contour plots of zero-mean bivariate Gaussians,
# showing how the covariance matrix controls the SHAPE and ORIENTATION of
# the density. Left to right: identity (circular), positive off-diagonal
# (tilted along the 45-degree line, increasingly compressed), and negative
# off-diagonal (tilted along the -45-degree line).

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---- Dark theme matching the darkly Quarto theme ----
# usetex gives genuine LaTeX bold for the covariance symbol in the titles
# (mathtext cannot bold uppercase Greek); requires a LaTeX install on PATH.
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


def gaussian_density(grid_x, grid_y, cov):
    """Zero-mean bivariate Gaussian density evaluated on a meshgrid."""
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(det))
    quad = (inv[0, 0] * grid_x**2
            + (inv[0, 1] + inv[1, 0]) * grid_x * grid_y
            + inv[1, 1] * grid_y**2)
    return norm * np.exp(-0.5 * quad)


# Unit-variance Gaussians, so the off-diagonal entry equals the correlation.
covariances = [
    (np.array([[1.0, 0.0], [0.0, 1.0]]), r"$\boldsymbol{\Sigma} = \mathbf{I}$  (off-diag $0$)"),
    (np.array([[1.0, 0.5], [0.5, 1.0]]), r"off-diag $= +0.5$"),
    (np.array([[1.0, 0.8], [0.8, 1.0]]), r"off-diag $= +0.8$"),
    (np.array([[1.0, -0.8], [-0.8, 1.0]]), r"off-diag $= -0.8$"),
]

lim = 3.0
axis = np.linspace(-lim, lim, 400)
gx, gy = np.meshgrid(axis, axis)

fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=300)

accent = "#f1c40f"
for ax, (cov, title) in zip(axes, covariances):
    density = gaussian_density(gx, gy, cov)
    levels = np.linspace(density.max() * 0.03, density.max() * 0.95, 6)
    ax.contour(gx, gy, density, levels=levels, colors=accent, linewidths=1.4)
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
_out = Path(__file__).resolve().parent.parent / "images" / "gaussian_covariance_gallery.png"
plt.savefig(_out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print("saved", _out)

# Embed in Quarto:
#   ![Caption](images/gaussian_covariance_gallery.png){#fig-gaussian_covariance_gallery fig-align="center" width=100%}

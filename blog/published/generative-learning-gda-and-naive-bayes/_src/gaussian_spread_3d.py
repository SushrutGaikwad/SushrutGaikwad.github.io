# Runs via the root uv environment: uv run python blog/published/generative-learning-gda-and-naive-bayes/_src/gaussian_spread_3d.py
#
# STATIC PNG: a row of three 3D surface plots of zero-mean bivariate Gaussian
# densities that differ only in the scale of the covariance matrix. As Sigma
# shrinks the bump gets taller and narrower; as it grows the bump flattens and
# spreads out. This builds intuition for "covariance = spread".

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ---- Dark theme matching the darkly Quarto theme ----
# usetex gives genuine LaTeX bold for the covariance symbol in the titles
# (mathtext cannot bold uppercase Greek); requires a LaTeX install on PATH.
plt.style.use("dark_background")
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "figure.facecolor": "#222222",
    "savefig.facecolor": "#222222",
    "axes.labelcolor": "#e6e6e6",
    "xtick.color": "#e6e6e6",
    "ytick.color": "#e6e6e6",
    "axes.titlecolor": "#e6e6e6",
    "text.color": "#e6e6e6",
})


def gaussian_density(grid_x, grid_y, scale):
    """Zero-mean bivariate Gaussian with covariance scale * I."""
    norm = 1.0 / (2.0 * np.pi * scale)
    quad = (grid_x**2 + grid_y**2) / scale
    return norm * np.exp(-0.5 * quad)


panels = [
    (1.0, r"$\boldsymbol{\Sigma} = \mathbf{I}$"),
    (0.6, r"$\boldsymbol{\Sigma} = 0.6\, \mathbf{I}$"),
    (2.0, r"$\boldsymbol{\Sigma} = 2\, \mathbf{I}$"),
]

lim = 3.0
axis = np.linspace(-lim, lim, 120)
gx, gy = np.meshgrid(axis, axis)

fig = plt.figure(figsize=(15, 4.8), dpi=300)
zmax = gaussian_density(np.array([[0.0]]), np.array([[0.0]]), 0.6)[0, 0]

for idx, (scale, title) in enumerate(panels, start=1):
    ax = fig.add_subplot(1, 3, idx, projection="3d")
    ax.set_facecolor("#222222")
    density = gaussian_density(gx, gy, scale)
    ax.plot_surface(gx, gy, density, cmap=cm.viridis, linewidth=0,
                    antialiased=True, rstride=2, cstride=2)
    ax.set_title(title, fontsize=15, pad=2)
    ax.set_xlabel(r"$x_1$", fontsize=11, labelpad=-4)
    ax.set_ylabel(r"$x_2$", fontsize=11, labelpad=-4)
    ax.set_zlim(0, zmax * 1.02)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.set_zticks([])
    ax.tick_params(axis="both", labelsize=8, pad=-2)
    ax.view_init(elev=32, azim=-60)
    # Match the pane backgrounds to the dark theme.
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0.13, 0.13, 0.13, 1.0))
        pane.pane.set_edgecolor("#444444")

plt.tight_layout()
_out = Path(__file__).resolve().parent.parent / "images" / "gaussian_spread_3d.png"
plt.savefig(_out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print("saved", _out)

# Embed in Quarto:
#   ![Caption](images/gaussian_spread_3d.png){#fig-gaussian_spread_3d fig-align="center" width=100%}

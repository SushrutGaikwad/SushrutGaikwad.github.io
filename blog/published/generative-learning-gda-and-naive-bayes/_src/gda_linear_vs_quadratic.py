# Runs via the root uv environment: uv run python blog/published/generative-learning-gda-and-naive-bayes/_src/gda_linear_vs_quadratic.py
#
# STATIC PNG: two panels, side by side, showing WHY the GDA boundary is
# straight. Left: both classes share one covariance matrix, so the boundary
# (the set of points with equal density under the two Gaussians) is a straight
# line. Right: the classes have DIFFERENT covariance matrices, and the same
# equal-density boundary curves into a quadratic (this is QDA).

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---- Dark theme matching the darkly Quarto theme ----
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


def log_density(grid_x, grid_y, mean, cov):
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    dx = grid_x - mean[0]
    dy = grid_y - mean[1]
    quad = inv[0, 0] * dx**2 + (inv[0, 1] + inv[1, 0]) * dx * dy + inv[1, 1] * dy**2
    return -0.5 * quad - 0.5 * np.log(det) - np.log(2.0 * np.pi)


def density(grid_x, grid_y, mean, cov):
    return np.exp(log_density(grid_x, grid_y, mean, cov))


dog_color = "#3498db"
eleph_color = "#e67e22"

mu0 = np.array([2.6, 3.0])
mu1 = np.array([5.4, 5.0])

lo, hi = -0.5, 8.5
axis = np.linspace(lo, hi, 500)
gx, gy = np.meshgrid(axis, axis)

fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), dpi=300)

panels = [
    {
        "title": "Shared covariance  $\\Rightarrow$  straight boundary",
        "cov0": np.array([[1.0, 0.35], [0.35, 1.0]]),
        "cov1": np.array([[1.0, 0.35], [0.35, 1.0]]),
    },
    {
        # cov1 is larger than cov0 in every direction (cov1 - cov0 is positive
        # definite), so the equal-density boundary is an ellipse that closes
        # around the smaller-covariance class 0, rather than a hyperbola.
        "title": "Different covariances  $\\Rightarrow$  curved boundary",
        "cov0": np.array([[0.5, 0.0], [0.0, 0.5]]),
        "cov1": np.array([[2.6, 0.9], [0.9, 2.2]]),
    },
]

def classify_boundary(cov0, cov1):
    """The equal-density boundary is x^T M x + b^T x + c = 0 with
    M = cov0^-1 - cov1^-1. Its conic type is set by det(M):
    zero matrix -> straight line, det>0 -> ellipse, det<0 -> hyperbola,
    det==0 (but M nonzero) -> parabola."""
    M = np.linalg.inv(cov0) - np.linalg.inv(cov1)
    if np.allclose(M, 0.0, atol=1e-9):
        return "line (linear boundary)"
    det_m = np.linalg.det(M)
    if det_m > 1e-9:
        return "ellipse"
    if det_m < -1e-9:
        return "hyperbola"
    return "parabola"


for ax, cfg in zip(axes, panels):
    cov0, cov1 = cfg["cov0"], cfg["cov1"]
    # Verify the boundary shape programmatically rather than assume it.
    print(f"{cfg['title']!r} -> {classify_boundary(cov0, cov1)}")
    peak0 = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(cov0)))
    peak1 = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(cov1)))
    d0 = density(gx, gy, mu0, cov0)
    d1 = density(gx, gy, mu1, cov1)
    ax.contour(gx, gy, d0, levels=peak0 * np.array([0.15, 0.4, 0.7]),
               colors=dog_color, linewidths=1.2, alpha=0.9)
    ax.contour(gx, gy, d1, levels=peak1 * np.array([0.15, 0.4, 0.7]),
               colors=eleph_color, linewidths=1.2, alpha=0.9)
    # Equal-density boundary: where log d1 - log d0 = 0.
    disc = log_density(gx, gy, mu1, cov1) - log_density(gx, gy, mu0, cov0)
    ax.contour(gx, gy, disc, levels=[0.0], colors="#ecf0f1", linewidths=2.4,
               linestyles="dashed")
    ax.plot(*mu0, "P", color="#ffffff", markersize=11, markeredgecolor=dog_color,
            markeredgewidth=2, zorder=6)
    ax.plot(*mu1, "P", color="#ffffff", markersize=11, markeredgecolor=eleph_color,
            markeredgewidth=2, zorder=6)
    ax.set_title(cfg["title"], fontsize=13.5, pad=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.grid(True, color="#3a3a3a", linestyle="--", linewidth=0.4, alpha=0.5)

plt.tight_layout()
_out = Path(__file__).resolve().parent.parent / "images" / "gda_linear_vs_quadratic.png"
plt.savefig(_out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print("saved", _out)

# Embed in Quarto:
#   ![Caption](images/gda_linear_vs_quadratic.png){#fig-gda_linear_vs_quadratic fig-align="center" width=100%}

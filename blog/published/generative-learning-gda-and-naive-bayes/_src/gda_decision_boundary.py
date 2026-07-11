# Runs via the root uv environment: uv run python blog/published/generative-learning-gda-and-naive-bayes/_src/gda_decision_boundary.py
#
# STATIC PNG: the "money shot" for GDA. A 2D training set with two classes
# (dogs vs elephants, described by two continuous features). We fit the GDA
# model with the maximum likelihood formulas, then overlay:
#   - the two fitted Gaussian contour families (SAME shape and orientation,
#     because they share one covariance matrix, but different centers), and
#   - the straight decision boundary where p(y = 1 | x) = 0.5.

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

# ---- Generate a 2-class training set from two shared-covariance Gaussians ----
rng = np.random.default_rng(7)
true_cov = np.array([[1.0, 0.45], [0.45, 1.0]])
mean_dog = np.array([2.3, 2.3])       # class 0: light and short
mean_eleph = np.array([5.6, 5.4])     # class 1: heavy and tall
n0, n1 = 45, 40
X0 = rng.multivariate_normal(mean_dog, true_cov, size=n0)
X1 = rng.multivariate_normal(mean_eleph, true_cov, size=n1)

# ---- Fit GDA with the maximum likelihood estimates ----
X = np.vstack([X0, X1])
y = np.concatenate([np.zeros(n0), np.ones(n1)])
phi = y.mean()
mu0 = X0.mean(axis=0)
mu1 = X1.mean(axis=0)
centered = np.vstack([X0 - mu0, X1 - mu1])
Sigma = (centered.T @ centered) / len(y)
Sigma_inv = np.linalg.inv(Sigma)


def gaussian_density(grid_x, grid_y, mean, cov):
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(det))
    dx = grid_x - mean[0]
    dy = grid_y - mean[1]
    quad = inv[0, 0] * dx**2 + (inv[0, 1] + inv[1, 0]) * dx * dy + inv[1, 1] * dy**2
    return norm * np.exp(-0.5 * quad)


# ---- Plot ----
fig, ax = plt.subplots(figsize=(8.4, 7.0), dpi=300)

lo, hi = -0.5, 8.5
axis = np.linspace(lo, hi, 500)
gx, gy = np.meshgrid(axis, axis)

dog_color = "#3498db"
eleph_color = "#e67e22"

d0 = gaussian_density(gx, gy, mu0, Sigma)
d1 = gaussian_density(gx, gy, mu1, Sigma)
peak = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(Sigma)))
levels = peak * np.array([0.1, 0.3, 0.55, 0.8])
ax.contour(gx, gy, d0, levels=levels, colors=dog_color, linewidths=1.3, alpha=0.9)
ax.contour(gx, gy, d1, levels=levels, colors=eleph_color, linewidths=1.3, alpha=0.9)

ax.scatter(X0[:, 0], X0[:, 1], c=dog_color, s=42, edgecolors="#111111",
           linewidths=0.6, zorder=4, label=r"Dogs ($y = 0$)")
ax.scatter(X1[:, 0], X1[:, 1], c=eleph_color, s=42, marker="^",
           edgecolors="#111111", linewidths=0.6, zorder=4, label=r"Elephants ($y = 1$)")

# Class means.
ax.plot(*mu0, "P", color="#ffffff", markersize=12, markeredgecolor=dog_color,
        markeredgewidth=2, zorder=6)
ax.plot(*mu1, "P", color="#ffffff", markersize=12, markeredgecolor=eleph_color,
        markeredgewidth=2, zorder=6)

# ---- Straight decision boundary: w^T x = c where p(y=1|x) = 0.5 ----
w = Sigma_inv @ (mu1 - mu0)
c = 0.5 * (mu1 @ Sigma_inv @ mu1 - mu0 @ Sigma_inv @ mu0) + np.log((1 - phi) / phi)
xs = np.array([lo, hi])
ys = (c - w[0] * xs) / w[1]
ax.plot(xs, ys, color="#ecf0f1", lw=2.4, linestyle="--", zorder=5,
        label=r"Boundary: $p(y=1 \mid \mathbf{x}) = 0.5$")

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal")
ax.set_xlabel(r"$x_1$ (weight, standardized)", fontsize=12)
ax.set_ylabel(r"$x_2$ (height, standardized)", fontsize=12)
ax.grid(True, color="#3a3a3a", linestyle="--", linewidth=0.4, alpha=0.5)
ax.legend(loc="upper left", framealpha=0.35, fontsize=10.5)

plt.tight_layout()
_out = Path(__file__).resolve().parent.parent / "images" / "gda_decision_boundary.png"
plt.savefig(_out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print("saved", _out)

# Embed in Quarto:
#   ![Caption](images/gda_decision_boundary.png){#fig-gda_decision_boundary fig-align="center" width=75%}

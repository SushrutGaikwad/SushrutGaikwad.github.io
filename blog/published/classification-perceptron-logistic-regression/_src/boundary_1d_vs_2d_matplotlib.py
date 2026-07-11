# Runs via the root uv environment: uv run python blog/published/classification-perceptron-logistic-regression/_src/boundary_1d_vs_2d_matplotlib.py
#
# Static 2-panel PNG that bridges the running tumor example from one feature to
# two. The single idea on display: a decision boundary is whatever object
# separates the two classes, and its shape depends on the dimension.
#
#   Left panel  (1D): one feature (tumor size). The boundary is a POINT on the
#                     size axis. We overlay the perceptron's hard step and the
#                     logistic sigmoid so the reader sees both squashing
#                     functions crossing at the same boundary point.
#   Right panel (2D): two features (size and cell-shape irregularity). The
#                     boundary is now a LINE. We shade the two half-spaces so
#                     the line reads as a separator, not just a stray line.
#
# Both panels use the same fitted logistic regression so the boundary point in
# 1D and the boundary line in 2D are honest (not hand-placed).

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

# ---- Shared colors (consistent with the post's other figures) ----
BENIGN = "#3498db"      # blue, y = 0
MALIGNANT = "#e67e22"   # orange, y = 1
SIGMOID = "#1abc9c"     # teal
STEP = "#f1c40f"        # yellow
BOUNDARY = "#ecf0f1"    # near-white


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# =====================================================================
# 1D DATA (same tumor sizes used elsewhere in the post)
# =====================================================================
benign_x = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.3])
malignant_x = np.array([3.5, 4.0, 4.2, 4.5, 5.0, 5.5])
x1d = np.concatenate([benign_x, malignant_x])
y1d = np.concatenate([np.zeros_like(benign_x), np.ones_like(malignant_x)])


def nll_1d(theta, x, y):
    z = theta[0] + theta[1] * x
    log_h = -np.logaddexp(0, -z)
    log_1mh = -np.logaddexp(0, z)
    return -np.sum(y * log_h + (1 - y) * log_1mh)


fit1d = minimize(nll_1d, np.array([0.0, 0.0]), args=(x1d, y1d), method="BFGS").x
boundary_1d = -fit1d[0] / fit1d[1]   # where theta0 + theta1 x = 0


# =====================================================================
# 2D DATA (size and cell-shape irregularity), clearly separable
# =====================================================================
rng = np.random.default_rng(seed=3)
n_per = 18
benign_center = np.array([1.4, 3.0])
malignant_center = np.array([4.3, 7.0])
spread = np.array([0.55, 1.05])

Xb = benign_center + spread * rng.standard_normal((n_per, 2))
Xm = malignant_center + spread * rng.standard_normal((n_per, 2))
X2d = np.vstack([Xb, Xm])
y2d = np.concatenate([np.zeros(n_per), np.ones(n_per)])


def nll_2d(theta, X, y, l2=1e-3):
    z = theta[0] + X @ theta[1:]
    log_h = -np.logaddexp(0, -z)
    log_1mh = -np.logaddexp(0, z)
    return -np.sum(y * log_h + (1 - y) * log_1mh) + l2 * np.sum(theta[1:] ** 2)


fit2d = minimize(nll_2d, np.zeros(3), args=(X2d, y2d), method="BFGS").x
b0, b1, b2 = fit2d


# =====================================================================
# FIGURE
# =====================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=300)
for ax in (axL, axR):
    ax.set_facecolor("#222222")

# ---------------------------------------------------------------------
# LEFT: 1D, boundary is a point
# ---------------------------------------------------------------------
axL.scatter(benign_x, np.zeros_like(benign_x), c=BENIGN, s=90,
            edgecolors="#222222", linewidths=1.0, zorder=4,
            label=r"Benign ($y = 0$)")
axL.scatter(malignant_x, np.ones_like(malignant_x), c=MALIGNANT, s=90,
            edgecolors="#222222", linewidths=1.0, zorder=4,
            label=r"Malignant ($y = 1$)")

xg = np.linspace(0, 6.5, 500)
# Logistic sigmoid (soft)
axL.plot(xg, sigmoid(fit1d[0] + fit1d[1] * xg), color=SIGMOID, lw=2.5,
         zorder=3, label="Logistic (soft)")
# Perceptron hard step, drawn crossing at the same boundary point
step = np.where(xg >= boundary_1d, 1.0, 0.0)
axL.plot(xg, step, color=STEP, lw=2.2, zorder=2, label="Perceptron (hard)")

axL.axhline(0.5, color="#95a5a6", ls="--", lw=0.9, alpha=0.5, zorder=1)
axL.axvline(boundary_1d, color=BOUNDARY, ls=":", lw=1.6, alpha=0.85, zorder=1)
axL.scatter([boundary_1d], [0.5], facecolors=BOUNDARY, edgecolors="#222222",
            s=120, zorder=6)
axL.annotate(r"boundary: a point" + f"\n$x \\approx {boundary_1d:.2f}$ cm",
             xy=(boundary_1d, 0.5), xytext=(boundary_1d + 0.25, 0.30),
             color=BOUNDARY, fontsize=10, ha="left", va="center")

axL.set_xlim(0, 6.5)
axL.set_ylim(-0.18, 1.25)
axL.set_yticks([0, 0.5, 1])
axL.set_xlabel(r"$x$ (tumor size, cm)", fontsize=12)
axL.set_ylabel(r"$y$  or  $h_{\boldsymbol{\theta}}(x)$", fontsize=12)
axL.set_title("One feature: the boundary is a point", fontsize=12)
axL.grid(True, color="#444444", ls="--", lw=0.4, alpha=0.5)
axL.legend(loc="center right", framealpha=0.4, fontsize=8.5)

# ---------------------------------------------------------------------
# RIGHT: 2D, boundary is a line
# ---------------------------------------------------------------------
# Faint half-space tints, so the line reads as a separator.
xx = np.linspace(-0.5, 6.6, 300)
yy = np.linspace(0.0, 10.5, 300)
XX, YY = np.meshgrid(xx, yy)
ZZ = b0 + b1 * XX + b2 * YY
half_cmap = LinearSegmentedColormap.from_list("half", [BENIGN, MALIGNANT])
axR.contourf(XX, YY, (ZZ >= 0).astype(float), levels=[-0.5, 0.5, 1.5],
             colors=[BENIGN, MALIGNANT], alpha=0.10, zorder=0)

axR.scatter(Xb[:, 0], Xb[:, 1], c=BENIGN, s=80, edgecolors="#222222",
            linewidths=1.0, zorder=4, label=r"Benign ($y = 0$)")
axR.scatter(Xm[:, 0], Xm[:, 1], c=MALIGNANT, s=80, edgecolors="#222222",
            linewidths=1.0, zorder=4, label=r"Malignant ($y = 1$)")

# Decision boundary line: b0 + b1 x1 + b2 x2 = 0  ->  x2 = -(b0 + b1 x1) / b2
x1_line = np.array([-0.5, 6.6])
x2_line = -(b0 + b1 * x1_line) / b2
axR.plot(x1_line, x2_line, color=BOUNDARY, lw=2.5, zorder=3)
axR.annotate("boundary: a line", xy=(3.0, -(b0 + b1 * 3.0) / b2),
             xytext=(0.2, 8.7), color=BOUNDARY, fontsize=10, ha="left",
             arrowprops=dict(arrowstyle="->", color=BOUNDARY, lw=1.2))

axR.set_xlim(-0.5, 6.6)
axR.set_ylim(0.0, 10.5)
axR.set_xlabel(r"$x_1$ (tumor size, cm)", fontsize=12)
axR.set_ylabel(r"$x_2$ (cell-shape irregularity)", fontsize=12)
axR.set_title("Two features: the boundary is a line", fontsize=12)
axR.grid(True, color="#444444", ls="--", lw=0.4, alpha=0.5)
axR.legend(loc="lower right", framealpha=0.4, fontsize=9)

plt.tight_layout()

from pathlib import Path
out = Path(__file__).resolve().parent.parent / "images" / "boundary_1d_vs_2d.png"
plt.savefig(out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# Embed in Quarto with:
#   ![Caption](images/boundary_1d_vs_2d.png){#fig-boundary_1d_vs_2d fig-align="center" width=100%}

# Runs via the root uv environment:
#   uv run python blog/published/linear-regression/_src/lwr_bandwidth_matplotlib.py
#
# STATIC PNG (3 panels) showing how the bandwidth tau controls the locality
# of locally weighted linear regression on the SAME curved dataset.
#   LEFT   (small tau): very local -> wiggly curve that chases noise.
#   MIDDLE (medium tau): a smooth, sensible fit.
#   RIGHT  (large tau): nearly a straight line -> approaches ordinary linear
#          regression and underfits the curvature.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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


def make_dataset():
    rng = np.random.default_rng(7)
    x = np.linspace(0.5, 9.5, 11)
    f = lambda t: 2.2 * np.sin(0.55 * t) + 0.35 * t + 1.0
    y = f(x) + rng.normal(0.0, 0.40, size=x.shape)
    return x, y


def lwr_predict(x_train, y_train, xq, tau):
    X = np.column_stack([np.ones_like(x_train), x_train])
    w = np.exp(-((x_train - xq) ** 2) / (2.0 * tau ** 2))
    sw = np.sqrt(w)
    theta, *_ = np.linalg.lstsq(X * sw[:, None], y_train * sw, rcond=None)
    return theta[0] + theta[1] * xq


x, y = make_dataset()
x_grid = np.linspace(x.min(), x.max(), 300)

point_color = "#e74c3c"
curve_color = "#f1c40f"

taus = [
    (0.35, r"Small $\tau = 0.35$", "too local: wiggly"),
    (1.10, r"Medium $\tau = 1.1$", "a sensible fit"),
    (6.00, r"Large $\tau = 6.0$", "too global: nearly a line"),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=300, sharey=True)

for ax, (tau, title, subtitle) in zip(axes, taus):
    ax.set_facecolor("#222222")
    curve = np.array([lwr_predict(x, y, xg, tau) for xg in x_grid])
    ax.plot(x_grid, curve, color=curve_color, lw=2.4, zorder=2)
    ax.scatter(x, y, marker="x", c=point_color, s=75, linewidths=2.2, zorder=3)
    ax.set_title(f"{title}\n({subtitle})", fontsize=13, pad=8)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.grid(True, color="#444444", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_xlim(x_grid.min(), x_grid.max())

axes[0].set_ylabel(r"$y$", fontsize=12)
ymin, ymax = y.min() - 1.6, y.max() + 1.6
for ax in axes:
    ax.set_ylim(ymin, ymax)

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "images" / "lwr_bandwidth.png"
plt.savefig(out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# Output: images/lwr_bandwidth.png
# Embed in Quarto (dark figure, so no .invert):
#   ![Caption](images/lwr_bandwidth.png){#fig-lwr_bandwidth fig-align="center" width=100%}

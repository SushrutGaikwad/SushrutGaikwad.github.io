# Runs via the root uv environment:
#   uv run python blog/published/kernel-methods/_src/linear_vs_cubic_fit_matplotlib.py
#
# STATIC PNG (2 panels) motivating feature maps on the SAME curved 1D dataset.
#   LEFT:   the best straight line y = theta_0 + theta_1 x cannot bend -> UNDERFITTING.
#   RIGHT:  a cubic y = theta_0 + theta_1 x + theta_2 x^2 + theta_3 x^3 follows the
#           curve, yet it is still a LINEAR model over the features [1, x, x^2, x^3].
# This grounds the post's opening claim: a nonlinear fit in x is a linear fit in phi(x).

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


# ---- A curved 1D dataset: think "price vs living area", but with a genuine
#      nonlinear (cubic-ish) trend that a straight line simply cannot capture. ----
def make_dataset():
    rng = np.random.default_rng(3)
    x = np.linspace(0.6, 6.4, 14)
    f = lambda t: 0.10 * (t - 3.5) ** 3 - 0.95 * (t - 3.5) + 4.0
    y = f(x) + rng.normal(0.0, 0.30, size=x.shape)
    return x, y


x, y = make_dataset()
x_grid = np.linspace(x.min() - 0.2, x.max() + 0.2, 500)

point_color = "#e74c3c"   # red crosses (matches the classic CS229 figure)
line_color = "#5dade2"    # blue straight line (underfit)
curve_color = "#f1c40f"   # bright yellow cubic fit


def fit_poly(deg):
    coeffs = np.polyfit(x, y, deg)
    return np.polyval(coeffs, x_grid)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=300, sharey=True)

# LEFT: straight line
axes[0].plot(x_grid, fit_poly(1), color=line_color, lw=2.4, zorder=2)
axes[0].scatter(x, y, marker="x", c=point_color, s=80, linewidths=2.3, zorder=3)
axes[0].set_title("A straight line underfits\n" r"$y = \theta_0 + \theta_1 x$",
                  fontsize=13, pad=8)

# RIGHT: cubic
axes[1].plot(x_grid, fit_poly(3), color=curve_color, lw=2.4, zorder=2)
axes[1].scatter(x, y, marker="x", c=point_color, s=80, linewidths=2.3, zorder=3)
axes[1].set_title("A cubic follows the curve\n"
                  r"$y = \theta_0 + \theta_1 x + \theta_2 x^2 + \theta_3 x^3$",
                  fontsize=13, pad=8)

for ax in axes:
    ax.set_facecolor("#222222")
    ax.set_xlabel(r"$x$  (living area)", fontsize=12)
    ax.grid(True, color="#444444", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_xlim(x_grid.min(), x_grid.max())

axes[0].set_ylabel(r"$y$  (price)", fontsize=12)

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "images" / "linear_vs_cubic_fit.png"
plt.savefig(out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# Output: images/linear_vs_cubic_fit.png
# Embed in Quarto (dark figure, so no .invert):
#   ![Caption](images/linear_vs_cubic_fit.png){#fig-linear_vs_cubic_fit fig-align="center" width=100%}

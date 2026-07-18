# Runs via the root uv environment:
#   uv run python blog/published/linear-regression/_src/underfitting_overfitting_matplotlib.py
#
# STATIC PNG (3 panels) illustrating underfitting, a reasonable fit, and
# overfitting on the SAME curved 1D dataset (shared across all LWR figures).
#   LEFT:   a straight line y = theta_0 + theta_1 x misses the curvature -> UNDERFITTING.
#   MIDDLE: a quadratic follows the shape much better -> a reasonable fit.
#   RIGHT:  a high-order polynomial threads every point but wiggles wildly -> OVERFITTING.

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


# ---- Shared curved dataset (identical across underfit/overfit, weighting,
#      in-action, and bandwidth figures so the reader sees the same points) ----
def make_dataset():
    rng = np.random.default_rng(7)
    x = np.linspace(0.5, 9.5, 11)
    f = lambda t: 2.2 * np.sin(0.55 * t) + 0.35 * t + 1.0
    y = f(x) + rng.normal(0.0, 0.40, size=x.shape)
    return x, y


x, y = make_dataset()
x_grid = np.linspace(x.min() - 0.2, x.max() + 0.2, 500)

point_color = "#e74c3c"   # red crosses (matches the classic CS229 figure)
curve_color = "#f1c40f"   # bright yellow fitted curve

OVERFIT_DEG = 9           # high-order polynomial for the overfitting panel


def fit_poly(deg):
    coeffs = np.polyfit(x, y, deg)
    return np.polyval(coeffs, x_grid)


fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=300, sharey=True)

panels = [
    (1, "Underfitting", r"$y = \theta_0 + \theta_1 x$"),
    (2, "A reasonable fit", r"$y = \theta_0 + \theta_1 x + \theta_2 x^2$"),
    (OVERFIT_DEG, "Overfitting", rf"$y = \sum_{{j=0}}^{{{OVERFIT_DEG}}} \theta_j x^j$"),
]

for ax, (deg, title, formula) in zip(axes, panels):
    ax.set_facecolor("#222222")
    ax.plot(x_grid, fit_poly(deg), color=curve_color, lw=2.2, zorder=2)
    ax.scatter(x, y, marker="x", c=point_color, s=75, linewidths=2.2, zorder=3)
    ax.set_title(f"{title}\n{formula}", fontsize=13, pad=8)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.grid(True, color="#444444", linestyle="--", linewidth=0.4, alpha=0.5)

axes[0].set_ylabel(r"$y$", fontsize=12)

# Clamp the shared y-range so the overfit panel's excursions stay readable.
ymin, ymax = y.min() - 2.2, y.max() + 2.2
for ax in axes:
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(x_grid.min(), x_grid.max())

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "images" / "underfitting_overfitting.png"
plt.savefig(out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# Output: images/underfitting_overfitting.png
# Embed in Quarto (dark figure, so no .invert):
#   ![Caption](images/underfitting_overfitting.png){#fig-underfitting_overfitting fig-align="center" width=100%}

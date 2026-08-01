# Runs via the root uv environment:
#   uv run python blog/published/kernel-methods/_src/linear_vs_cubic_fit_matplotlib.py
#
# STATIC PNG (2 panels) motivating feature maps on the running 14-house dataset.
#   LEFT:   the best straight line y = theta_0 + theta_1 x cannot bend. Dashed
#           vertical segments show the residuals, so the underfit is visible and
#           not merely asserted: the line is systematically wrong in a pattern.
#   RIGHT:  a cubic follows the S-shaped trend, yet it is still a LINEAR model
#           over the features [1, x, x^2, x^3].
# This grounds the post's opening claim: a nonlinear fit in x is a linear fit in phi(x).

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import (BG, CURVE, LINE, MUTED, POINT, X_LABEL, Y_LABEL,
                     house_data, style_axes, use_dark_theme)

use_dark_theme()

x, y = house_data()
x_grid = np.linspace(x.min() - 0.15, x.max() + 0.15, 500)


def fit_poly(deg):
    coeffs = np.polyfit(x, y, deg)
    return np.polyval(coeffs, x_grid), np.polyval(coeffs, x)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=300, sharey=True)

line_curve, line_at_x = fit_poly(1)
cubic_curve, cubic_at_x = fit_poly(3)

# LEFT: straight line, with residual segments drawn in.
for xi, yi, fi in zip(x, y, line_at_x):
    axes[0].plot([xi, xi], [yi, fi], color=MUTED, lw=1.1, ls="--", zorder=1)
axes[0].plot(x_grid, line_curve, color=LINE, lw=2.4, zorder=2)
axes[0].scatter(x, y, marker="x", c=POINT, s=80, linewidths=2.3, zorder=3)
axes[0].set_title("A straight line underfits\n" r"$y = \theta_0 + \theta_1 x$",
                  fontsize=13, pad=8)
axes[0].annotate("the errors swing in a pattern,\nthe fingerprint of a shape\nthe line cannot follow",
                 xy=(3.15, 6.55), xytext=(1.55, 8.4), fontsize=10.5, color=MUTED,
                 ha="left",
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))

# RIGHT: cubic.
for xi, yi, fi in zip(x, y, cubic_at_x):
    axes[1].plot([xi, xi], [yi, fi], color=MUTED, lw=1.1, ls="--", zorder=1)
axes[1].plot(x_grid, cubic_curve, color=CURVE, lw=2.4, zorder=2)
axes[1].scatter(x, y, marker="x", c=POINT, s=80, linewidths=2.3, zorder=3)
axes[1].set_title("A cubic follows the curve\n"
                  r"$y = \theta_0 + \theta_1 x + \theta_2 x^2 + \theta_3 x^3$",
                  fontsize=13, pad=8)

for ax in axes:
    style_axes(ax)
    ax.set_xlim(x_grid.min(), x_grid.max())

axes[0].set_ylabel(Y_LABEL, fontsize=12)

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "images" / "linear_vs_cubic_fit.png"
plt.savefig(out, dpi=300, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# Output: images/linear_vs_cubic_fit.png
# Embed in Quarto (dark figure, so no .invert):
#   ![Caption](images/linear_vs_cubic_fit.png){#fig-linear_vs_cubic_fit fig-align="center" width=100%}

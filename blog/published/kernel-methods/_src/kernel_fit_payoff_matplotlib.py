# Runs via the root uv environment:
#   uv run python blog/published/kernel-methods/_src/kernel_fit_payoff_matplotlib.py
#
# STATIC PNG (3 panels): THE PAYOFF SHOT of the post. The very same 14 houses that
# opened the post, now fit by the kernelized LMS algorithm (beta := beta + alpha
# (y - K beta), then h(x) = sum_i beta_i K(x_i, x)) with the Gaussian kernel at
# three bandwidths. The straight line from the opening figure is kept as a faint
# reference in every panel, so the reader sees exactly what the kernel bought.
#   sigma too small -> the fit chases every point (too local),
#   sigma just right -> the S-shaped trend appears,
#   sigma too large  -> everything looks similar to everything, back to a line.
# No feature vector is ever built: the whole fit runs on kernel values alone.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import (BG, CURVE, LINE, MUTED, POINT, X_LABEL, Y_LABEL,
                     house_data, kernel_predict, kernelized_lms, style_axes,
                     use_dark_theme)

use_dark_theme()

x, y = house_data()
x_grid = np.linspace(x.min() - 0.15, x.max() + 0.15, 600)
line_fit = np.polyval(np.polyfit(x, y, 1), x_grid)

PANELS = [
    (0.15, "too small", r"$\sigma = 0.15$", "every point pulls\nits own bump"),
    (0.55, "about right", r"$\sigma = 0.55$", "the S-shaped trend,\nrecovered"),
    (3.00, "too large", r"$\sigma = 3$", "everything looks similar\nto everything: back to a line"),
]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=300, sharey=True)

for ax, (sigma, verdict, label, note) in zip(axes, PANELS):
    beta = kernelized_lms(x, y, sigma)
    fit = kernel_predict(beta, x, x_grid, sigma)

    ax.plot(x_grid, line_fit, color=LINE, lw=1.6, ls=":", alpha=0.75,
            zorder=2, label="straight line")
    ax.plot(x_grid, fit, color=CURVE, lw=2.6, zorder=3, label="kernelized LMS")
    ax.scatter(x, y, marker="x", c=POINT, s=70, linewidths=2.1, zorder=4)

    ax.set_title(f"{label}  ({verdict})", fontsize=13, pad=8)
    ax.text(0.97, 0.05, note, transform=ax.transAxes, fontsize=10.5,
            color=MUTED, ha="right", va="bottom")
    style_axes(ax)
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_ylim(0.8, 11.2)

axes[0].set_ylabel(Y_LABEL, fontsize=12)
leg = axes[1].legend(frameon=False, fontsize=10.5, loc="upper left")
for txt in leg.get_texts():
    txt.set_color(MUTED)

fig.suptitle("The same 14 houses, fit with a Gaussian kernel and no feature vectors",
             fontsize=14, y=1.02)

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "images" / "kernel_fit_payoff.png"
plt.savefig(out, dpi=300, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# Output: images/kernel_fit_payoff.png
# Embed in Quarto (dark figure, so no .invert):
#   ![Caption](images/kernel_fit_payoff.png){#fig-kernel_fit_payoff fig-align="center" width=100%}

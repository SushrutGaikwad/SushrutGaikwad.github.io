# Runs via the root uv environment:
#   uv run python blog/published/kernel-methods/_src/prediction_weighted_vote_matplotlib.py
#
# STATIC PNG (2 stacked panels) unpacking the prediction rule
#       h(x_test) = sum_i beta_i K(x_i, x_test).
#   TOP:    the 14 houses and the Gaussian-kernel fit. Each training cross is drawn
#           with an opacity proportional to K(x_i, x_test): near houses shout, far
#           houses whisper. A dashed line marks the new house being priced.
#   BOTTOM: the 14 signed terms beta_i K(x_i, x_test) as bars. They literally add
#           up to the prediction drawn above.
# The punchline the figure is built for: NOT ONE of the 14 terms is zero, so every
# training house has to be kept around at test time. (In the next post the SVM
# drives most of them to exactly zero, leaving only the support vectors.)

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import (BG, CURVE, FG, GOOD, LINE, MUTED, POINT, X_LABEL, Y_LABEL,
                     gaussian_kernel_matrix, house_data, style_axes,
                     use_dark_theme)

use_dark_theme()

SIGMA = 0.35
X_TEST = 3.30

x, y = house_data()

# Kernelized LMS with the plain Gaussian kernel of the post.
K = gaussian_kernel_matrix(x, x, SIGMA)
beta = np.zeros_like(y)
for _ in range(4000):
    beta = beta + 0.02 * (y - K @ beta)

x_grid = np.linspace(x.min() - 0.15, x.max() + 0.15, 600)
fit = gaussian_kernel_matrix(x_grid, x, SIGMA) @ beta

k_test = gaussian_kernel_matrix(np.array([X_TEST]), x, SIGMA)[0]
contrib = beta * k_test
prediction = contrib.sum()

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(10.5, 7.4), dpi=300, sharex=True,
    gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.12})

# ---------------- TOP: the fit, with each house's "loudness" ----------------
ax_top.plot(x_grid, fit, color=CURVE, lw=2.4, zorder=2)
for xi, yi, ki in zip(x, y, k_test):
    ax_top.scatter([xi], [yi], marker="x", c=POINT, s=60 + 90 * ki,
                   linewidths=2.0, alpha=0.25 + 0.75 * ki, zorder=3)

ax_top.axvline(X_TEST, color=GOOD, lw=1.6, ls="--", zorder=1)
ax_top.scatter([X_TEST], [prediction], marker="o", s=110, facecolor=GOOD,
               edgecolor=FG, linewidths=1.2, zorder=5)
ax_top.annotate(rf"$h(x_{{\mathrm{{test}}}}) = {prediction:.2f}$",
                xy=(X_TEST, prediction), xytext=(X_TEST - 1.55, prediction + 2.2),
                fontsize=12, color=GOOD,
                arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.2))
ax_top.text(X_TEST + 0.06, 1.5, "a new house\nto price", color=GOOD, fontsize=10.5,
            ha="left", va="bottom")
ax_top.text(0.02, 0.95,
            "crosses are drawn brighter and bigger\n"
            r"the larger $K(x_i, x_{\mathrm{test}})$ is",
            transform=ax_top.transAxes, fontsize=10.5, color=MUTED,
            ha="left", va="top")

style_axes(ax_top, xlabel=None, ylabel=Y_LABEL)
ax_top.set_xlim(x_grid.min(), x_grid.max())
ax_top.set_ylim(0.8, 11.6)
ax_top.set_title(r"Prediction is a weighted vote over every training house",
                 fontsize=14, pad=10)

# ---------------- BOTTOM: the 14 terms that add up to the answer ----------------
colors = [GOOD if c >= 0 else LINE for c in contrib]
ax_bot.bar(x, contrib, width=0.16, color=colors, alpha=0.9, zorder=3)
ax_bot.axhline(0.0, color=FG, lw=1.0, zorder=2)

ax_bot.text(0.02, 0.95,
            r"each bar is one term $\beta_i \, K(x_i, x_{\mathrm{test}})$"
            "\n"
            rf"all 14 add up to ${prediction:.2f}$",
            transform=ax_bot.transAxes, fontsize=11, color=MUTED,
            ha="left", va="top")
ax_bot.annotate("far houses barely matter, but we still\n"
                "store them and evaluate $K$ against each one",
                xy=(1.06, 0.0), xytext=(1.32, -1.6),
                fontsize=10.5, color=MUTED, ha="left",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))

style_axes(ax_bot, xlabel=X_LABEL,
           ylabel=r"contribution to $h(x_{\mathrm{test}})$")
ax_bot.set_xlim(x_grid.min(), x_grid.max())

out = Path(__file__).resolve().parent.parent / "images" / "prediction_weighted_vote.png"
plt.savefig(out, dpi=300, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}  (prediction = {prediction:.3f})")

# Output: images/prediction_weighted_vote.png
# Embed in Quarto (dark figure, so no .invert):
#   ![Caption](images/prediction_weighted_vote.png){#fig-prediction_weighted_vote fig-align="center" width=95%}

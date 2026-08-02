# Runs via the root uv environment with uv run (see command at the bottom).
"""The threshold and three of its softer relatives.

Everything in this post is built out of the hard threshold on the left, because
it is the easiest thing to reason about. Nothing in the perceptron picture
requires it: once the unit is written as an activation applied to an affine
value, the activation becomes a free choice, and these three are the ones that
took over in practice.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import ACCENT, BG, FG, GOOD, MUTED, QUIET, use_dark_theme

use_dark_theme()

z = np.linspace(-6, 6, 1200)

PANELS = [
    ("threshold", r"$f(z) = \mathbf{1}[z \geq 0]$", None, ACCENT),
    ("sigmoid", r"$f(z) = \dfrac{1}{1 + e^{-z}}$", 1 / (1 + np.exp(-z)), QUIET),
    ("softplus", r"$f(z) = \log\left(1 + e^{z}\right)$", np.log1p(np.exp(z)), GOOD),
    ("ReLU", r"$f(z) = \max(0, z)$", np.maximum(0, z), "#c39bd3"),
]

# 2x2 rather than 1x4: at the width the post embeds this, four panels in a row
# leaves each one about a quarter of the column, and the titles render at half
# the size of body text.
fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.8))

for ax, (name, formula, values, colour) in zip(axes.ravel(), PANELS):
    if values is None:
        # Drawn as two separate segments plus an open/closed pair, because the
        # step genuinely is discontinuous and joining it would be a lie.
        ax.plot(z[z < 0], np.zeros_like(z[z < 0]), color=colour, lw=2.6)
        ax.plot(z[z >= 0], np.ones_like(z[z >= 0]), color=colour, lw=2.6)
        ax.plot([0], [1], marker="o", color=colour, markersize=6)
        ax.plot([0], [0], marker="o", color=BG, markeredgecolor=colour,
                markersize=6, markeredgewidth=1.8)
        ax.set_ylim(-0.45, 2.6)
    else:
        ax.plot(z, values, color=colour, lw=2.6)
        ax.set_ylim(-0.45, 2.6)

    ax.axhline(0, color=MUTED, lw=0.8, alpha=0.6)
    ax.axvline(0, color=MUTED, lw=0.8, alpha=0.6)
    ax.set_xlim(-6, 6)
    ax.set_facecolor(BG)
    ax.grid(True, color="#444444", linestyle="--", linewidth=0.4, alpha=0.45)
    ax.set_title(f"{name}\n{formula}", fontsize=14, pad=12)

# Axis labels only on the outer edges, so the inner panels stay uncluttered.
for ax in axes[1, :]:
    ax.set_xlabel(r"$z$", fontsize=13)
for ax in axes[:, 0]:
    ax.set_ylabel(r"$f(z)$", fontsize=13)

fig.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.09,
                    wspace=0.20, hspace=0.34)

out = Path(__file__).resolve().parent.parent / "images" / "activation_functions.png"
fig.savefig(out, dpi=200, facecolor=BG)
print(f"wrote {out}")

# Run (from the repo root):
#   uv run python blog/published/neural-networks-history-of-an-idea/_src/activation_functions.py
# Output: blog/published/neural-networks-history-of-an-idea/images/activation_functions.png
# Embed with:
#   ![...](images/activation_functions.png){#fig-activation_functions fig-align="center" width=100%}

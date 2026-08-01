# Runs via the root uv environment:
#   uv run python blog/published/kernel-methods/_src/feature_explosion_matplotlib.py
#
# STATIC PNG (log-log): the WALL, drawn. How long is the feature vector phi(x) if we
# take all monomials of degree <= k in d attributes (counting repetitions, so
# p = 1 + d + d^2 + ... + d^k), compared with the cost of ONE kernel evaluation, O(d).
# The d = 1000 marker makes the gap concrete: a billion-entry vector versus a
# thousand multiplications. That vertical gap IS the kernel trick.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import BG, CURVE, FG, GOOD, GRID, LINE, MUTED, POINT, use_dark_theme

use_dark_theme()

d = np.logspace(0.3, 3.3, 300)   # about 2 to 2000 attributes


def p_of_d(dd, k):
    """1 + d + d^2 + ... + d^k, the length of the degree-<=k monomial feature vector."""
    return sum(dd ** j for j in range(k + 1))


fig, ax = plt.subplots(figsize=(9.6, 5.6), dpi=300)

DEGREES = [(2, CURVE, r"length of $\boldsymbol{\phi}(\mathbf{x})$, degree $\leq 2$:  $p \sim d^{2}$"),
           (3, POINT, r"length of $\boldsymbol{\phi}(\mathbf{x})$, degree $\leq 3$:  $p \sim d^{3}$")]

for k, color, label in DEGREES:
    ax.plot(d, p_of_d(d, k), color=color, lw=2.4, label=label, zorder=3)

ax.plot(d, d, color=GOOD, lw=2.8, ls="--", zorder=4,
        label=r"cost of one kernel evaluation:  $O(d)$")

# ---- The d = 1000 story ----
D0 = 1000.0
p0 = p_of_d(D0, 3)
ax.axvline(D0, color=MUTED, lw=1.1, ls=":", zorder=2)
ax.scatter([D0, D0], [p0, D0], s=70, facecolor=BG,
           edgecolor=[POINT, GOOD], linewidths=2.0, zorder=6)

ax.annotate(r"$d = 1000 \;\Rightarrow\; p \approx 10^{9}$" "\n"
            "a billion numbers to build,\nstore, and update every step",
            xy=(D0, p0), xytext=(60, 3.0e8), fontsize=11, color=POINT,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=POINT, lw=1.2))
ax.annotate("the kernel route stays down here:\n"
            r"$1000$ multiplications",
            xy=(D0, D0 * 0.85), xytext=(90, 5.0), fontsize=11, color=GOOD,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.2))

# The gap itself.
ax.annotate("", xy=(1300, D0), xytext=(1300, p0),
            arrowprops=dict(arrowstyle="<->", color=FG, lw=1.4))
ax.text(1450, 1.2e5, r"$10^{6} \times$", color=FG, fontsize=13,
        ha="left", va="center", rotation=90)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(2, 3600)
ax.set_ylim(1, 3e10)
ax.set_xlabel(r"$d$  (number of attributes)", fontsize=12)
ax.set_ylabel(r"numbers we must handle", fontsize=12)
ax.set_title("The feature vector explodes. The kernel evaluation does not.",
             fontsize=13.5, pad=10)
ax.set_facecolor(BG)
ax.grid(True, which="major", color=GRID, linestyle="--", linewidth=0.4, alpha=0.5)

leg = ax.legend(frameon=False, fontsize=11, loc="upper left")
for txt in leg.get_texts():
    txt.set_color(FG)

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "images" / "feature_explosion.png"
plt.savefig(out, dpi=300, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# Output: images/feature_explosion.png
# Embed in Quarto (dark figure, so no .invert):
#   ![Caption](images/feature_explosion.png){#fig-feature_explosion fig-align="center" width=90%}

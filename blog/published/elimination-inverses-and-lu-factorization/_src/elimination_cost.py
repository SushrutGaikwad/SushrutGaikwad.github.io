"""Cost-of-elimination curve for Post 2: operations grow like n^3 / 3.

Runs via the root uv environment:
  uv run python blog/published/elimination-inverses-and-lu-factorization/_src/elimination_cost.py
Output: blog/published/elimination-inverses-and-lu-factorization/images/elimination_cost.svg
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")
plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})

LIGHT = "#e6e6e6"
DARK = "#222222"

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)

n = np.linspace(1, 50, 200)
ops = n**3 / 3.0

ax.plot(n, ops, color="#5dade2", lw=2.5, label=r"$\frac{1}{3}n^3$ operations")

# Mark the doubling 25 -> 50 to make the "8x work" point visible.
for nv in (25, 50):
    ax.plot(nv, nv**3 / 3.0, "o", color="#ec7063", ms=8, zorder=5)
ax.annotate(
    r"double $n$, about $8\times$ the work",
    xy=(50, 50**3 / 3.0),
    xytext=(18, 50**3 / 3.0 * 0.78),
    color=LIGHT,
    fontsize=11,
    arrowprops=dict(arrowstyle="->", color=LIGHT),
)

ax.set_xlim(0, 52)
ax.set_ylim(0, 50**3 / 3.0 * 1.08)
ax.set_xlabel(r"$n$  (number of equations)", color=LIGHT, fontsize=13)
ax.set_ylabel("operations", color=LIGHT, fontsize=13)
ax.tick_params(colors=LIGHT)
for spine in ax.spines.values():
    spine.set_color(LIGHT)
ax.grid(True, color="#3a3a3a", lw=0.6)
ax.legend(facecolor="#333333", edgecolor=LIGHT, labelcolor=LIGHT, fontsize=12, loc="upper left")

out = Path(__file__).resolve().parent.parent / "images" / "elimination_cost.svg"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=DARK, bbox_inches="tight")
print(f"saved {out}")

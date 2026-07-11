"""Row picture for Post 1: two lines crossing at the solution (2, 3).

Runs via the root uv environment: uv run python blog/published/the-geometry-of-linear-equations/_src/row_picture.py
Output: blog/published/the-geometry-of-linear-equations/images/row_picture.svg
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")
plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})

LIGHT = "#e6e6e6"
DARK = "#222222"

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)

x = np.linspace(-0.5, 5.0, 200)
y1 = 4.0 - 0.5 * x  # x + 2y = 8
y2 = 9.0 - 3.0 * x  # 3x + y = 9

ax.plot(x, y1, color="#5dade2", lw=2.5, label=r"$x + 2y = 8$  (morning 1)")
ax.plot(x, y2, color="#f5b041", lw=2.5, label=r"$3x + y = 9$  (morning 2)")

# Solution point where the two lines cross.
ax.plot(2, 3, "o", color="#ec7063", ms=11, zorder=5)
ax.annotate(
    r"$(2,\ 3)$",
    (2, 3),
    textcoords="offset points",
    xytext=(14, 8),
    color=LIGHT,
    fontsize=16,
)

ax.axhline(0, color="#888888", lw=1)
ax.axvline(0, color="#888888", lw=1)
ax.set_xlim(-0.5, 5.0)
ax.set_ylim(-0.5, 9.5)
ax.set_xlabel(r"$x$  (price of a coffee)", color=LIGHT, fontsize=13)
ax.set_ylabel(r"$y$  (price of a muffin)", color=LIGHT, fontsize=13)
ax.tick_params(colors=LIGHT)
for spine in ax.spines.values():
    spine.set_color(LIGHT)
ax.grid(True, color="#3a3a3a", lw=0.6)
ax.legend(
    facecolor="#333333",
    edgecolor=LIGHT,
    labelcolor=LIGHT,
    fontsize=11,
    loc="upper right",
)

out = Path(__file__).resolve().parent.parent / "images" / "row_picture.svg"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=DARK, bbox_inches="tight")
print(f"saved {out}")

"""Bowl-versus-saddle quadratic-form surfaces for Post 13.

Left: a positive definite form (a bowl). Right: an indefinite form (a saddle).
Runs via the root uv environment:
  uv run python blog/published/symmetric-and-positive-definite-matrices/_src/bowl_saddle.py
Output: blog/published/symmetric-and-positive-definite-matrices/images/bowl_saddle.png
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")
plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})

DARK = "#222222"
LIGHT = "#e6e6e6"
PANE = (0.14, 0.14, 0.14, 1.0)

x = np.linspace(-2, 2, 90)
y = np.linspace(-2, 2, 90)
X, Y = np.meshgrid(x, y)
Z_bowl = 2 * X**2 + 2 * X * Y + 2 * Y**2  # A = [[2,1],[1,2]], positive definite
Z_saddle = X**2 - Y**2                    # indefinite

fig = plt.figure(figsize=(12, 5))
fig.patch.set_facecolor(DARK)


def style3d(ax, title):
    ax.set_facecolor(DARK)
    ax.xaxis.set_pane_color(PANE)
    ax.yaxis.set_pane_color(PANE)
    ax.zaxis.set_pane_color(PANE)
    ax.tick_params(colors=LIGHT, labelsize=8)
    ax.set_xlabel("x", color=LIGHT)
    ax.set_ylabel("y", color=LIGHT)
    ax.set_zlabel("z", color=LIGHT)
    ax.set_title(title, color=LIGHT, fontsize=14, pad=12)
    ax.grid(False)


ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(X, Y, Z_bowl, cmap="viridis", edgecolor="none", alpha=0.95)
style3d(ax1, "Positive definite: a bowl")
ax1.view_init(elev=28, azim=-60)

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(X, Y, Z_saddle, cmap="plasma", edgecolor="none", alpha=0.95)
style3d(ax2, "Indefinite: a saddle")
ax2.view_init(elev=28, azim=-60)

out = Path(__file__).resolve().parent.parent / "images" / "bowl_saddle.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=DARK, bbox_inches="tight", dpi=150)
print(f"saved {out}")

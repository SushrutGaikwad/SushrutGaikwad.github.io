"""Eigenvector action animation for Post 10.

As x sweeps the unit circle, Ax traces an ellipse. Along the eigenvector
directions (dashed) the image stays on the same line. A = [[2,1],[1,2]],
eigenvalues 3 (eigenvector (1,1)) and 1 (eigenvector (1,-1)).

Runs via the root uv environment:
  uv run python blog/published/eigenvalues-and-eigenvectors/_src/eigen_action.py
Output: blog/published/eigenvalues-and-eigenvectors/images/eigen_action.mp4
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use("dark_background")
plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})

LIGHT = "#e6e6e6"
DARK = "#222222"
BLUE = "#5dade2"
ORANGE = "#f5b041"
GRAY = "#9aa0a6"

A = np.array([[2.0, 1.0], [1.0, 2.0]])

fig, ax = plt.subplots(figsize=(6.4, 6.4))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)

theta = np.linspace(0, 2 * np.pi, 240)
circle = np.array([np.cos(theta), np.sin(theta)])
ellipse = A @ circle

# Static background.
ax.plot(circle[0], circle[1], color=GRAY, lw=1.2, alpha=0.7, label="unit circle of $\\mathbf{x}$")
ax.plot(ellipse[0], ellipse[1], color=ORANGE, lw=1.2, alpha=0.5, label="image $\\mathbf{Ax}$")
for d, lab in [(np.array([1, 1]), r"eigvec $\lambda=3$"), (np.array([1, -1]), r"eigvec $\lambda=1$")]:
    d = d / np.linalg.norm(d)
    ax.plot([-3.4 * d[0], 3.4 * d[0]], [-3.4 * d[1], 3.4 * d[1]],
            color=LIGHT, lw=1.0, ls="--", alpha=0.6)

x_arrow, = ax.plot([], [], color=BLUE, lw=3, solid_capstyle="round")
Ax_arrow, = ax.plot([], [], color=ORANGE, lw=3, solid_capstyle="round")
x_head = ax.scatter([], [], color=BLUE, s=40, zorder=5)
Ax_head = ax.scatter([], [], color=ORANGE, s=40, zorder=5)

ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect("equal")
ax.axhline(0, color="#555555", lw=0.8)
ax.axvline(0, color="#555555", lw=0.8)
ax.tick_params(colors=LIGHT)
for s in ax.spines.values():
    s.set_color(LIGHT)
ax.set_title(r"$\mathbf{x}$ (blue) and $\mathbf{Ax}$ (orange)", color=LIGHT, fontsize=14)
ax.legend(facecolor="#333333", edgecolor=LIGHT, labelcolor=LIGHT, fontsize=9, loc="upper left")


def update(i):
    t = theta[i]
    x = np.array([np.cos(t), np.sin(t)])
    Ax = A @ x
    x_arrow.set_data([0, x[0]], [0, x[1]])
    Ax_arrow.set_data([0, Ax[0]], [0, Ax[1]])
    x_head.set_offsets([x])
    Ax_head.set_offsets([Ax])
    return x_arrow, Ax_arrow, x_head, Ax_head


ani = FuncAnimation(fig, update, frames=len(theta), interval=40, blit=True)

out = Path(__file__).resolve().parent.parent / "images" / "eigen_action.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
ani.save(out, writer="ffmpeg", fps=30, dpi=140, savefig_kwargs={"facecolor": DARK})
print(f"saved {out}")

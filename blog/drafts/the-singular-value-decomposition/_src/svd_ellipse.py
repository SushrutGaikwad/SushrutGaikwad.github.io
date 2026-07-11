"""SVD geometry animation for Post 16.

Morphs the unit circle into the ellipse A(circle) for A = [[4,4],[-3,3]], showing
the orthonormal input directions v1, v2 carried to the perpendicular output
directions sigma1*u1, sigma2*u2. Saves an MP4.

Runs via the root uv environment:
  uv run python blog/published/the-singular-value-decomposition/_src/svd_ellipse.py
Output: blog/published/the-singular-value-decomposition/images/svd_ellipse.mp4
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.style.use("dark_background")
plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})

LIGHT = "#e6e6e6"
DARK = "#222222"
BLUE_A = "#5dade2"
ORANGE_A = "#f5b041"
GRAY_L = "#9aa0a6"

A = np.array([[4.0, 4.0], [-3.0, 3.0]])
I2 = np.eye(2)

th = np.linspace(0, 2 * np.pi, 240)
circle = np.vstack([np.cos(th), np.sin(th)])
v1 = np.array([1.0, 1.0]) / np.sqrt(2)
v2 = np.array([1.0, -1.0]) / np.sqrt(2)

# Frame schedule: hold, ramp 0->1, hold.
ts = np.concatenate([np.zeros(12), np.linspace(0, 1, 60), np.ones(22)])

fig, ax = plt.subplots(figsize=(6.5, 6))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)
ax.set_xlim(-7.5, 7.5)
ax.set_ylim(-6.5, 6.5)
ax.set_aspect("equal")
ax.axhline(0, color="#555555", lw=1)
ax.axvline(0, color="#555555", lw=1)
ax.tick_params(colors=LIGHT)
for s in ax.spines.values():
    s.set_color("#555555")
ax.set_title(r"$\mathbf{A} = \mathbf{U}\,\Sigma\,\mathbf{V}^{\top}$:  the unit circle becomes an ellipse",
             color=LIGHT, fontsize=13, pad=12)

# Static reference unit circle.
ax.plot(circle[0], circle[1], color=GRAY_L, lw=1.2, ls="--", alpha=0.6)

(shape_line,) = ax.plot([], [], color=LIGHT, lw=2.5)
(v1_line,) = ax.plot([], [], color=BLUE_A, lw=3)
(v2_line,) = ax.plot([], [], color=ORANGE_A, lw=3)
(v1_tip,) = ax.plot([], [], "o", color=BLUE_A, ms=8)
(v2_tip,) = ax.plot([], [], "o", color=ORANGE_A, ms=8)
v1_lab = ax.text(0, 0, "", color=BLUE_A, fontsize=14)
v2_lab = ax.text(0, 0, "", color=ORANGE_A, fontsize=14)


def init():
    shape_line.set_data([], [])
    for ln in (v1_line, v2_line, v1_tip, v2_tip):
        ln.set_data([], [])
    return shape_line, v1_line, v2_line, v1_tip, v2_tip, v1_lab, v2_lab


def update(frame):
    t = ts[frame]
    M = (1 - t) * I2 + t * A
    pts = M @ circle
    shape_line.set_data(pts[0], pts[1])

    w1 = M @ v1
    w2 = M @ v2
    v1_line.set_data([0, w1[0]], [0, w1[1]])
    v2_line.set_data([0, w2[0]], [0, w2[1]])
    v1_tip.set_data([w1[0]], [w1[1]])
    v2_tip.set_data([w2[0]], [w2[1]])

    lab1 = r"$\mathbf{v}_1$" if t < 0.5 else r"$\sigma_1 \mathbf{u}_1$"
    lab2 = r"$\mathbf{v}_2$" if t < 0.5 else r"$\sigma_2 \mathbf{u}_2$"
    v1_lab.set_position((w1[0] + 0.25, w1[1] + 0.25))
    v1_lab.set_text(lab1)
    v2_lab.set_position((w2[0] + 0.25, w2[1] - 0.45))
    v2_lab.set_text(lab2)
    return shape_line, v1_line, v2_line, v1_tip, v2_tip, v1_lab, v2_lab


ani = animation.FuncAnimation(fig, update, frames=len(ts), init_func=init, blit=False, interval=40)

out = Path(__file__).resolve().parent.parent / "images" / "svd_ellipse.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
ani.save(out, writer="ffmpeg", fps=25, dpi=120, savefig_kwargs={"facecolor": DARK})
print(f"saved {out}")

# Embed in the post with:
#   {{< video images/svd_ellipse.mp4 >}}

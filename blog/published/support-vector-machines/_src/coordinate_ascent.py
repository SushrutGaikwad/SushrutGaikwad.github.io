# Runs via the root uv environment:
#   uv run python blog/published/support-vector-machines/_src/coordinate_ascent.py
#
# COORDINATE ASCENT (CS229): maximize a concave quadratic W(a1, a2) one variable
# at a time. Each step re-optimizes a single coordinate, so every segment of the
# path is parallel to an axis. The contours are ellipses; the path zig-zags to
# the peak. This mirrors the CS229 figure that motivates SMO.
#
# Two things are deliberately tuned so the picture matches the caption:
#   * The correlation in Q is 0.75, not 0.85. At 0.85 the ellipses are so
#     elongated that none of them CLOSES inside a sensible window, so the figure
#     showed open arcs while the caption promised ellipses with a centre.
#   * The contour levels are chosen explicitly (rather than letting matplotlib
#     pick 18 of them over the whole grid) so every drawn contour is a closed
#     ellipse around the maximum.
# The zig-zag is still pronounced: one full sweep shrinks the error by r^2 = 0.56.
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from _common import ACCENT

# Concave quadratic  W(a) = -1/2 (a - m)^T Q (a - m) , maximized at a = m.
r = 0.75
Q = np.array([[1.0, r], [r, 1.0]])
m = np.array([0.4, 0.5])


def W(a1, a2):
    d1, d2 = a1 - m[0], a2 - m[1]
    return -0.5 * (Q[0, 0] * d1 * d1 + 2 * Q[0, 1] * d1 * d2 + Q[1, 1] * d2 * d2)


# Start on the flat (small-eigenvalue) axis of the ellipse, which is the
# direction (1,-1)/sqrt(2) and the one that makes coordinate ascent zig-zag most.
flat = np.array([1.0, -1.0]) / np.sqrt(2.0)
start = m + 2.45 * flat

# Coordinate ascent: fix one coordinate, jump to the exact max in the other.
# d/da1 = 0  ->  a1 = m1 - (Q12/Q11)(a2 - m2) , and symmetrically for a2.
path = [start.copy()]
a1, a2 = start
for _ in range(12):
    a1 = m[0] - (Q[0, 1] / Q[0, 0]) * (a2 - m[1])
    path.append(np.array([a1, a2]))
    a2 = m[1] - (Q[0, 1] / Q[1, 1]) * (a1 - m[0])
    path.append(np.array([a1, a2]))
path = np.array(path)

# Window centred on the maximum so the ellipses sit in the middle of the frame.
half = 3.1
gx = np.linspace(m[0] - half, m[0] + half, 500)
gy = np.linspace(m[1] - half, m[1] + half, 500)
A1, A2 = np.meshgrid(gx, gy)
Z = W(A1, A2)

# Levels chosen so each contour closes inside the window. The flat semi-axis of
# the level set W = -L is sqrt(2L / (1 - r)), which must stay under ~2.6.
levels = -np.linspace(0.05, 0.80, 9)[::-1]

fig, ax = plt.subplots(figsize=(6.2, 6.0), dpi=300)

# A very subtle grey ramp carries the "height" information, and light steel-blue
# contour lines carry the shape. Neither competes with the yellow ascent path.
# (viridis was putting near-black purple contours on the near-black background.)
ax.contourf(A1, A2, Z, levels=np.concatenate([[Z.min()], levels, [0.0]]),
            colors=["#242424", "#272727", "#2a2a2a", "#2d2d2d", "#303030",
                    "#333333", "#363636", "#393939", "#3c3c3c", "#404040"],
            zorder=0)
cs = ax.contour(A1, A2, Z, levels=levels, colors="#9bb8d3",
                linewidths=0.9, alpha=0.85, zorder=1)

ax.plot(path[:, 0], path[:, 1], color=ACCENT, lw=1.8, marker="o",
        markersize=4, zorder=4)
ax.scatter([m[0]], [m[1]], marker="*", s=280, c="#ffffff",
           edgecolors="#000000", linewidths=0.6, zorder=5, label="maximum")
ax.scatter([path[0, 0]], [path[0, 1]], marker="o", s=75, c="#e74c3c",
           zorder=5, label="start")

ax.set_facecolor("#222222")
ax.set_aspect("equal")
ax.set_xlim(m[0] - half, m[0] + half)
ax.set_ylim(m[1] - half, m[1] + half)
ax.set_xlabel(r"$\alpha_1$", fontsize=13)
ax.set_ylabel(r"$\alpha_2$", fontsize=13)
ax.set_title("Coordinate ascent moves one axis at a time", fontsize=12, pad=8)
ax.grid(True, color="#444444", linestyle="--", linewidth=0.4, alpha=0.35)
ax.legend(loc="lower right", framealpha=0.15, fontsize=10)

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "images" / "coordinate_ascent.png"
plt.savefig(out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# Output: images/coordinate_ascent.png  (dark figure -> embed WITHOUT .invert)

# Runs via the root uv environment:
#   uv run python blog/published/support-vector-machines/_src/smo_box.py
#
# SMO FEASIBLE REGION (CS229): when SMO updates two multipliers (alpha_1, alpha_2)
# it must respect BOTH the box 0 <= alpha_i <= C AND the linear equality
#   alpha_1 y_1 + alpha_2 y_2 = zeta.
# The feasible set is therefore the segment where the diagonal line crosses the
# box, and alpha_2 is confined to [L, H]. Shown here for y_1 != y_2 (slope +1).
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from _common import ACCENT, LINE_COLOR

C = 3.0
# Case y1 != y2:  alpha_1 - alpha_2 = k  ->  alpha_2 = alpha_1 - k.
k = 1.1

fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=300)

# The box [0, C] x [0, C].
ax.add_patch(plt.Rectangle((0, 0), C, C, fill=False, edgecolor="#e6e6e6",
                           lw=1.8, zorder=2))

# The equality line across the box.
a1 = np.linspace(-0.5, C + 0.5, 200)
ax.plot(a1, a1 - k, color=LINE_COLOR, lw=2.6, zorder=3,
        label=r"$\alpha_1 y_1 + \alpha_2 y_2 = \zeta$")

# Feasible segment: alpha_2 in [L, H] with alpha_1 = alpha_2 + k in [0, C].
L = max(0.0, -k)
H = min(C, C - k)
seg_a2 = np.array([L, H])
seg_a1 = seg_a2 + k
ax.plot(seg_a1, seg_a2, color=ACCENT, lw=5.0, solid_capstyle="round",
        zorder=4, label="feasible segment")

# L and H markers on the alpha_2 axis.
for val, name in [(L, "L"), (H, "H")]:
    ax.axhline(val, color="#888888", lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.text(-0.35, val, name, color=ACCENT, fontsize=14, fontweight="bold",
            va="center", ha="center")

# Corner labels.
corners = {(0, 0): "(0,0)", (C, 0): "(C,0)", (0, C): "(0,C)", (C, C): "(C,C)"}
for (cx, cy), lab in corners.items():
    dx = 0.12 if cx == 0 else -0.12
    dy = 0.16 if cy == 0 else -0.16
    ha = "left" if cx == 0 else "right"
    va = "bottom" if cy == 0 else "top"
    ax.text(cx + dx, cy + dy, lab, color="#cfcfcf", fontsize=10, ha=ha, va=va)

ax.set_facecolor("#222222")
ax.set_aspect("equal")
ax.set_xlim(-0.8, C + 0.8)
ax.set_ylim(-0.8, C + 0.8)
ax.set_xlabel(r"$\alpha_1$", fontsize=13)
ax.set_ylabel(r"$\alpha_2$", fontsize=13)
ax.set_title(r"Two constraints pin $(\alpha_1,\alpha_2)$ to a line segment",
             fontsize=12, pad=8)
ax.grid(True, color="#3a3a3a", linestyle="--", linewidth=0.4, alpha=0.4)
ax.legend(loc="upper left", framealpha=0.15, fontsize=10)

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "images" / "smo_box.png"
plt.savefig(out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# Output: images/smo_box.png  (dark figure -> embed WITHOUT .invert)

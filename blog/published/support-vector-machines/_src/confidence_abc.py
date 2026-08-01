# Runs via the root uv environment:
#   uv run python blog/published/support-vector-machines/_src/confidence_abc.py
#
# INTUITION FIGURE (CS229): a single separating boundary with three admitted
# points A, B, C at DIFFERENT (perpendicular) distances from it. A is far (very
# confident), C is right next to the boundary (a small nudge could flip it).
# Each point gets its own perpendicular dropline to its own foot on the boundary,
# so the three distances are visibly distinct. The context clusters are dimmed
# so the three highlighted points stand out.
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from _common import make_data, style_axes, LINE_COLOR, POS_COLOR, NEG_COLOR

pos, neg = make_data(n=8)  # sparser context so A, B, C stand out

fig, ax = plt.subplots(figsize=(6.8, 6.3), dpi=300)

# Dimmed context clusters (they only say "here are the two classes"). alpha=0.35
# over the #222222 background rendered these almost black, so they are dimmed to
# 0.6 instead: still clearly secondary to A/B/C, but actually visible.
ax.scatter(pos[:, 0], pos[:, 1], marker="^", c=POS_COLOR, s=64, alpha=0.6,
           edgecolors="none", zorder=2)
ax.scatter(neg[:, 0], neg[:, 1], marker="x", c=NEG_COLOR, s=64, alpha=0.6,
           linewidths=2.0, zorder=2)

# Boundary x1 + x2 = 0.
xs = np.linspace(-3.4, 3.4, 200)
ax.plot(xs, -xs, color=LINE_COLOR, lw=2.6, zorder=3,
        label=r"boundary $\mathbf{w}^{\intercal}\mathbf{x}+b=0$")

nhat = np.array([1.0, 1.0]) / np.sqrt(2.0)

# Each admitted point = foot-on-boundary (tangential offset t) + gamma * normal.
# Well-separated t values give three distinct feet, hence three distinct droplines.
specs = {
    "A": dict(t=0.9, gamma=2.3, desc="far:\nconfident", dcolor="#7ee787",
              letter_off=(-0.40, 0.0), desc_pos=(2.45, 1.22)),
    "B": dict(t=-0.1, gamma=1.45, desc="in between", dcolor="#e6e6e6",
              letter_off=(-0.42, -0.02), desc_pos=(1.35, 0.35)),
    "C": dict(t=-0.9, gamma=0.2, desc="close:\nunsure", dcolor="#ffab91",
              letter_off=(0.16, 0.10), desc_pos=(-0.78, 1.65)),
}

for name, s in specs.items():
    foot = np.array([s["t"], -s["t"]])
    p = foot + s["gamma"] * nhat
    ax.plot([p[0], foot[0]], [p[1], foot[1]], color="#e6e6e6",
            ls=(0, (3, 3)), lw=1.4, zorder=3)
    ax.scatter([foot[0]], [foot[1]], c=LINE_COLOR, s=28, zorder=4)
    ax.scatter([p[0]], [p[1]], marker="^", c=POS_COLOR, s=175,
               edgecolors="#ffffff", linewidths=1.8, zorder=5)
    ax.text(p[0] + s["letter_off"][0], p[1] + s["letter_off"][1], name,
            color="#ffffff", fontsize=16, fontweight="bold", zorder=6)
    ax.text(s["desc_pos"][0], s["desc_pos"][1], s["desc"], color=s["dcolor"],
            fontsize=10.5, ha="center", va="center", zorder=6)

style_axes(ax)
ax.set_xlim(-3.4, 3.4)
ax.set_ylim(-3.4, 3.4)
ax.set_title("Distance from the boundary = confidence", fontsize=13, pad=8)
ax.legend(loc="lower left", framealpha=0.15, fontsize=10)

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "images" / "confidence_abc.png"
plt.savefig(out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# Output: images/confidence_abc.png  (dark figure -> embed WITHOUT .invert)

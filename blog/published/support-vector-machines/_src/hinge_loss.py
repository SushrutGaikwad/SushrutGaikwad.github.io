# Runs via the root uv environment:
#   uv run python blog/published/support-vector-machines/_src/hinge_loss.py
#
# HINGE LOSS: plot  max(0, 1 - m)  against the functional margin
#   m = y * (w.x + b)  ("score times the true label").
# For m >= 1 (correct AND outside the margin) the loss is exactly 0; as m drops
# below 1 the loss rises linearly. The 0/1 misclassification loss is overlaid
# for contrast: hinge is its convex, differentiable-friendly upper bound.
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from _common import ACCENT

m = np.linspace(-2.0, 3.0, 500)
hinge = np.maximum(0.0, 1.0 - m)
zero_one = (m < 0).astype(float)

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)

ax.plot(m, hinge, color=ACCENT, lw=2.8,
        label=r"hinge loss  $\max(0,\,1-m)$", zorder=4)
# Brighter than the old #888888, which was barely separable from the shaded
# bands behind it on the dark background.
ax.step(m, zero_one, where="post", color="#c5d0db", lw=2.0, ls=(0, (4, 3)),
        label=r"0/1 loss", zorder=3)

# Shade the three regimes the post walks through. Kept faint so the two curves
# stay dominant, but lifted from 0.10 to 0.13 so the band edges are legible.
ax.axvspan(1.0, 3.0, color="#2ecc71", alpha=0.13)
ax.axvspan(0.0, 1.0, color="#f1c40f", alpha=0.13)
ax.axvspan(-2.0, 0.0, color="#e74c3c", alpha=0.13)

ax.axvline(1.0, color="#e6e6e6", lw=0.9, ls=":", alpha=0.7)
ax.axvline(0.0, color="#e6e6e6", lw=0.9, ls=":", alpha=0.7)

ax.text(2.15, 1.55, "correct &\noutside margin\n(loss $=0$)",
        color="#5ee89a", fontsize=10.5, ha="center", va="center")
ax.text(0.5, 2.55, "correct but\ninside margin\n(loss in $(0,1)$)",
        color="#f7d64a", fontsize=10.5, ha="center", va="top")
ax.text(-1.0, 2.55, "wrong side\n(loss $>1$)",
        color="#ff7a6b", fontsize=10.5, ha="center", va="top")

ax.set_facecolor("#222222")
ax.set_xlabel(r"functional margin  $m = y\,(\mathbf{w}^{\intercal}\mathbf{x}+b)$",
              fontsize=12)
ax.set_ylabel("loss", fontsize=12)
ax.set_title("The hinge loss penalizes points that are not confidently correct",
             fontsize=12, pad=8)
ax.grid(True, color="#444444", linestyle="--", linewidth=0.4, alpha=0.5)
ax.set_xlim(-2.0, 3.0)
ax.set_ylim(-0.15, 3.0)
ax.legend(loc="upper right", framealpha=0.15, fontsize=10)

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "images" / "hinge_loss.png"
plt.savefig(out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# Output: images/hinge_loss.png  (dark figure -> embed WITHOUT .invert)

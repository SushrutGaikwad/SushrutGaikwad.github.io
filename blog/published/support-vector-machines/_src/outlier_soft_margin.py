# Runs via the root uv environment:
#   uv run python blog/published/support-vector-machines/_src/outlier_soft_margin.py
#
# SOFT-MARGIN MOTIVATION (CS229): a single outlier drags the hard-margin boundary.
# LEFT  : clean data, the max-margin boundary sits comfortably in the gap.
# RIGHT : add ONE positive point deep in negative territory; the hard-margin
#         boundary must swing to keep everything correctly classified, and the
#         margin collapses. Both boundaries are the genuine hard-margin SVM
#         solutions (solved with SLSQP), not hand-drawn.
#
# The collapsed margin is only 0.05 wide, so at the main scale the boundary and
# BOTH margin lines land on the same pixels and the panel looks like it is
# missing its margin lines. A zoom inset on the outlier makes the collapse
# legible: the three lines really are there, just crushed together.
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from _common import (make_data, solve_hard_svm, scatter_classes, style_axes,
                     draw_boundary_and_margins, circle_support_vectors,
                     POS_COLOR)

pos, neg = make_data()

LIM = 4.0


def fit(pos, neg):
    X = np.vstack([pos, neg])
    y = np.hstack([np.ones(len(pos)), -np.ones(len(neg))])
    w, b = solve_hard_svm(X, y)
    return X, y, w, b


def draw_svm(ax, pos, neg, title):
    X, y, w, b = fit(pos, neg)
    scatter_classes(ax, pos, neg)
    draw_boundary_and_margins(ax, w, b, (-LIM, LIM))
    circle_support_vectors(ax, X[y * (X @ w + b) < 1.05])

    margin_width = 2.0 / np.linalg.norm(w)
    ax.set_title(f"{title}\nmargin $=2/\\|\\mathbf{{w}}\\|={margin_width:.2f}$",
                 fontsize=12, pad=8)
    style_axes(ax)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    return w, b, margin_width


fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.8), dpi=300)
w_clean, b_clean, mw_clean = draw_svm(
    axes[0], pos, neg, "No outlier: a wide, comfortable margin")

# Add one positive outlier deep in the negative cluster's territory.
outlier = np.array([[-1.7, -0.9]])
pos_out = np.vstack([pos, outlier])
w_out, b_out, mw_out = draw_svm(
    axes[1], pos_out, neg, "One outlier: the margin collapses")

axes[1].scatter(outlier[:, 0], outlier[:, 1], marker="^", c=POS_COLOR, s=150,
                edgecolors="#ffffff", linewidths=1.8, zorder=6)
axes[1].annotate("outlier", xy=(outlier[0, 0], outlier[0, 1]),
                 xytext=(-3.6, -2.9), color="#ffffff", fontsize=11,
                 arrowprops=dict(arrowstyle="->", color="#ffffff", lw=1.4))

# --- zoom inset: the three lines really are distinct, just 0.05 apart ---------
half = 0.30
cx, cy = outlier[0]
axins = axes[1].inset_axes([0.56, 0.06, 0.40, 0.40])
scatter_classes(axins, pos_out, neg, s=90)
axins.scatter(outlier[:, 0], outlier[:, 1], marker="^", c=POS_COLOR, s=150,
              edgecolors="#ffffff", linewidths=1.8, zorder=6)
draw_boundary_and_margins(axins, w_out, b_out, (cx - half, cx + half), lw=2.0)
axins.set_xlim(cx - half, cx + half)
axins.set_ylim(cy - half, cy + half)
axins.set_aspect("equal")
axins.set_xticks([])
axins.set_yticks([])
axins.set_facecolor("#1a1a1a")
for spine in axins.spines.values():
    spine.set_edgecolor("#f1c40f")
    spine.set_linewidth(1.4)
axins.set_title(rf"zoom $\times{LIM / half / 2:.0f}$", fontsize=9,
                color="#f1c40f", pad=3)
axes[1].indicate_inset_zoom(axins, edgecolor="#f1c40f", alpha=0.9, lw=1.2)

# How far did the boundary swing? Report it so the caption can quote a number.
ang = np.degrees(np.arccos(abs(
    np.dot(w_clean, w_out) / (np.linalg.norm(w_clean) * np.linalg.norm(w_out)))))

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "images" / "outlier_soft_margin.png"
plt.savefig(out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")
print(f"  clean margin   = {mw_clean:.2f}")
print(f"  outlier margin = {mw_out:.2f}")
print(f"  boundary rotated by {ang:.0f} degrees")

# Output: images/outlier_soft_margin.png  (dark figure -> embed WITHOUT .invert)

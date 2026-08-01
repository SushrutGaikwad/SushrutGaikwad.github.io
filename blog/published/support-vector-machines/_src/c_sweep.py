# Runs via the root uv environment:
#   uv run python blog/published/support-vector-machines/_src/c_sweep.py
#
# WHAT C ACTUALLY DOES. The post explains the C trade-off in words ("a large C
# punishes violations harshly... a small C tolerates them for a wider margin").
# This figure shows it: same data, three values of C, everything else identical.
#
#   C = 0.01  soft: a very wide margin, most points inside it, many SVs
#   C = 1     balanced
#   C = 100   nearly hard: a narrow margin squeezed into the overlap, few SVs
#
# The data OVERLAPS rather than containing one outlier. An earlier draft reused
# the single-outlier set from the previous figure, but there the outlier's slack
# moved non-monotonically in C (1.32 -> 1.97 -> 1.61), which reads as noise
# rather than as a trade-off. With genuinely overlapping classes the margin width
# and the support-vector count both move monotonically, so the knob does the one
# thing the text says it does.
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import SVC

from _common import (scatter_classes, style_axes, draw_boundary_and_margins,
                     circle_support_vectors)


def make_overlapping(seed=11, n=45, sep=0.75, scale=0.85):
    """Two Gaussian clouds close enough to genuinely overlap, so no straight
    line separates them and the soft margin has real work to do."""
    rng = np.random.default_rng(seed)
    pos = rng.normal([sep, sep], scale, size=(n, 2))
    neg = rng.normal([-sep, -sep], scale, size=(n, 2))
    return pos, neg


pos, neg = make_overlapping()
X = np.vstack([pos, neg])
y = np.hstack([np.ones(len(pos)), -np.ones(len(neg))])

LIM = 3.6
Cs = [0.01, 1.0, 100.0]
labels = ["soft: wide margin, many violations",
          "balanced",
          "nearly hard: narrow margin"]

fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.2), dpi=300)

for ax, C, sub in zip(axes, Cs, labels):
    clf = SVC(kernel="linear", C=C).fit(X, y)
    w = clf.coef_[0]
    b = clf.intercept_[0]
    margin = 2.0 / np.linalg.norm(w)

    scatter_classes(ax, pos, neg)
    draw_boundary_and_margins(ax, w, b, (-LIM, LIM))
    circle_support_vectors(ax, clf.support_vectors_, s=170, lw=1.4)

    # Total slack actually paid: sum of max(0, 1 - y f(x)) over the training set.
    xi = np.maximum(0.0, 1.0 - y * (X @ w + b))
    n_sv = len(clf.support_vectors_)

    ax.set_title(f"$C = {C:g}$\n{sub}", fontsize=12, pad=8)
    ax.text(0.5, -0.16,
            f"margin $2/\\|\\mathbf{{w}}\\| = {margin:.2f}$   |   "
            f"total slack $\\sum_i \\xi_i = {xi.sum():.1f}$   |   "
            f"{n_sv} SVs",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color="#e6e6e6")

    style_axes(ax)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    print(f"  C={C:<7g} margin={margin:.3f}  sum_xi={xi.sum():.2f}  "
          f"n_sv={n_sv}  train_acc={clf.score(X, y):.3f}")

axes[0].legend(loc="upper left", framealpha=0.2, fontsize=9)

fig.suptitle(r"The soft-margin parameter $C$ trades margin width against "
             r"margin violations", fontsize=14, y=0.99)
plt.tight_layout(rect=[0, 0.04, 1, 0.95])
out = Path(__file__).resolve().parent.parent / "images" / "c_sweep.png"
plt.savefig(out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# Output: images/c_sweep.png  (dark figure -> embed WITHOUT .invert)

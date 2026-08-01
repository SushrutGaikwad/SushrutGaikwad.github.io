# Runs via the root uv environment:
#   uv run python blog/published/support-vector-machines/_src/candidate_lines.py
#
# MOTIVATION FIGURE: three candidate separating lines on the same admissions data.
# All three correctly separate the two classes, but only the middle one leaves
# generous "breathing room" on both sides. This sets up the whole SVM idea.
#
# Each line is drawn with its own translucent CORRIDOR: the widest band centred
# on that line that still contains no training point. The corridor is what the
# post calls "breathing room", so the reader can SEE that L2 wins instead of
# having to take the caption's word for it. The corridor half-width is exactly
# the geometric margin of that line, which is the quantity the next few sections
# make precise.
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from _common import make_data, scatter_classes, style_axes, solve_hard_svm

pos, neg = make_data()
X = np.vstack([pos, neg])
y = np.hstack([np.ones(len(pos)), -np.ones(len(neg))])


def corridor_halfwidth(a, b, c):
    """Perpendicular distance from the line to the closest training point.
    This is the geometric margin of the line, i.e. its breathing room."""
    return np.min(np.abs(a * X[:, 0] + b * X[:, 1] + c)) / np.hypot(a, b)


# L2 is the GENUINE maximum-margin line, straight from the hard-margin solver,
# not a hand-picked line that merely looks central.
#
# This matters. An earlier version of this figure hand-picked three lines and
# asserted in the caption that the middle one had the most room. It did not:
# the hand-picked "hugging" line scored 0.85 against the middle line's 0.46, so
# the figure was quietly contradicting the sentence next to it. Deriving L2 from
# the solver and asserting the ordering below makes that failure impossible.
w_star, b_star = solve_hard_svm(X, y)


def hugging_line(deg, side):
    """A line that still separates the data but is pushed up against one cluster.

    Rotate the optimal normal by `deg` degrees, then slide the intercept until
    the line almost grazes the chosen cluster. Any intercept strictly between
    -min_pos and -max_neg separates; sitting near one end of that interval is
    exactly what "hugging" means.
    """
    th = np.radians(deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    u = rot @ (w_star / np.linalg.norm(w_star))
    p = X @ u
    lo, hi = -p[y > 0].min(), -p[y < 0].max()   # feasible intercept interval
    assert lo < hi, f"direction rotated by {deg} deg does not separate the data"
    eps = 0.10
    c = hi - eps if side == "neg" else lo + eps
    return float(u[0]), float(u[1]), float(c)


# Only a small rotation. At 13 degrees the three lines fanned out and crossed
# inside the plot, which made the corridors overlap into mud; at 5 degrees they
# stay near-parallel across the data and each corridor reads as its own band.
lines = {
    "L1": hugging_line(+5, "neg"),    # hugs the rejected cluster
    "L2": (float(w_star[0]), float(w_star[1]), float(b_star)),   # the best line
    "L3": hugging_line(-5, "pos"),    # hugs the admitted cluster
}

# Every line must actually separate the data, and L2 must actually be the widest.
for name, (a, b, c) in lines.items():
    vals = a * X[:, 0] + b * X[:, 1] + c
    ok = np.all(vals[y > 0] > 0) and np.all(vals[y < 0] < 0)
    assert ok, f"line {name} does not separate the data"

_room = {k: corridor_halfwidth(*v) for k, v in lines.items()}
assert _room["L2"] > _room["L1"] and _room["L2"] > _room["L3"], (
    f"L2 must have the most breathing room, got {_room}")


# Three clearly separated hues. L2 (the winner) gets the bright accent; the two
# losers get colours far from it in hue AND far from the green/red data markers.
styles = {
    "L1": dict(color="#c56cf0", ls=(0, (6, 3)), lw=2.2),   # violet
    "L2": dict(color="#f1c40f", ls="-", lw=3.0),           # amber, the hero line
    "L3": dict(color="#48dbfb", ls=(0, (6, 3)), lw=2.2),   # light cyan
}

xs = np.linspace(-3.0, 3.0, 400)
fig, ax = plt.subplots(figsize=(6.6, 6.2), dpi=300)

# Corridors first, so the data and the lines sit on top of them. Each band also
# gets crisp edge lines: filled alpha alone turned into an indistinct wash where
# the three bands overlapped.
halfwidths = {}
for name, (a, b, c) in lines.items():
    h = corridor_halfwidth(a, b, c)
    halfwidths[name] = h
    norm = np.hypot(a, b)
    lo = -(a * xs + c - h * norm) / b
    hi = -(a * xs + c + h * norm) / b
    col = styles[name]["color"]
    ax.fill_between(xs, lo, hi, color=col, alpha=0.11, lw=0, zorder=1)
    for edge in (lo, hi):
        ax.plot(xs, edge, color=col, lw=0.7, alpha=0.35, zorder=1)

scatter_classes(ax, pos, neg)

for name, (a, b, c) in lines.items():
    ax.plot(xs, -(a * xs + c) / b, **styles[name], zorder=3)

# Label each line ON the line itself, at a chosen fraction along the part of the
# line that is actually INSIDE the axes. Anchoring by a fixed x sent labels off
# the bottom of the plot as soon as the line geometry changed; this cannot.
label_frac = {"L1": 0.30, "L2": 0.50, "L3": 0.70}
for name, (a, b, c) in lines.items():
    ys_line = -(a * xs + c) / b
    inside = (ys_line >= -3.0) & (ys_line <= 3.0)
    xv, yv = xs[inside], ys_line[inside]
    k = int(label_frac[name] * (len(xv) - 1))
    ax.text(xv[k], yv[k], f"$L_{name[1]}$",
            color=styles[name]["color"], fontsize=19,
            ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc="#222222", ec="none",
                      alpha=0.88))

# Spell out the punchline with the actual numbers, so the shading is quantified.
summary = "     ".join(
    f"$L_{name[1]}$: {halfwidths[name]:.2f}" for name in ["L1", "L2", "L3"])
ax.text(0.5, 0.025,
        "breathing room (distance to the closest point)\n" + summary,
        transform=ax.transAxes, fontsize=10.5, color="#e6e6e6",
        ha="center", va="bottom", linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.35", fc="#2b2b2b", ec="#555555",
                  alpha=0.92))

style_axes(ax)
ax.set_xlim(-3.0, 3.0)
ax.set_ylim(-3.0, 3.0)
ax.set_title("Which line separates the two classes best?", fontsize=13, pad=8)
ax.legend(loc="upper right", framealpha=0.15, fontsize=10)

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "images" / "candidate_lines.png"
plt.savefig(out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")
for name in ["L1", "L2", "L3"]:
    print(f"  {name} breathing room = {halfwidths[name]:.3f}")

# Output: images/candidate_lines.png  (dark figure -> embed WITHOUT .invert)

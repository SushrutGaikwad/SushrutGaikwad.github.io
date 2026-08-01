# Shared helpers for the SVM post figures (dark "darkly" theme + a tiny hard-margin
# SVM solver). Imported by the other matplotlib scripts in this folder.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Dark theme matching the darkly Quarto theme.
plt.style.use("dark_background")
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.facecolor": "#222222",
    "figure.facecolor": "#222222",
    "savefig.facecolor": "#222222",
    "axes.edgecolor": "#e6e6e6",
    "axes.labelcolor": "#e6e6e6",
    "xtick.color": "#e6e6e6",
    "ytick.color": "#e6e6e6",
    "axes.titlecolor": "#e6e6e6",
    "text.color": "#e6e6e6",
})

# ---------------------------------------------------------------------------
# VISUAL CONVENTION, held across every figure in this post (matplotlib AND
# manim). If you change a colour here, change the matching hex in the manim
# scripts too, since they cannot share matplotlib rcParams.
#
#   class +1 ............ green triangle          POS_COLOR
#   class -1 ............ red cross               NEG_COLOR
#   decision boundary ... SOLID blue              LINE_COLOR
#   margin lines +/-1 ... DASHED muted grey       MARGIN_COLOR
#   support vectors ..... yellow ring             SV_COLOR
#   derived quantities .. yellow (arrows, braces) ACCENT
#
# The one rule that matters: the boundary is the only solid blue line, and the
# margin lines are never blue. Earlier drafts drew margins in blue in one
# figure and grey in another, which made the same object look like two things.
# ---------------------------------------------------------------------------
POS_COLOR = "#2ecc71"   # green triangles, class +1
NEG_COLOR = "#e74c3c"   # red crosses,     class -1
LINE_COLOR = "#5dade2"  # blue decision boundary (SOLID, and only the boundary)
MARGIN_COLOR = "#aab7c4"  # muted DASHED margin lines
SV_COLOR = "#f1c40f"    # yellow support-vector rings
ACCENT = "#f1c40f"

# Dash pattern used for every margin line, so they read as one object.
MARGIN_DASH = (0, (5, 3))


def make_data(seed=7, n=11, gap=0.6, scale=0.55):
    """Two linearly separable blobs with a clean gap around the anti-diagonal
    x1 + x2 = 0. Positives sit upper-right (class +1), negatives lower-left (-1).
    Rejection sampling keeps a visible margin so every drawn line stays honest."""
    rng = np.random.default_rng(seed)
    pos, neg = [], []
    while len(pos) < n:
        p = rng.normal([1.2, 1.2], scale, size=2)
        if p[0] + p[1] > gap:
            pos.append(p)
    while len(neg) < n:
        p = rng.normal([-1.2, -1.2], scale, size=2)
        if p[0] + p[1] < -gap:
            neg.append(p)
    return np.array(pos), np.array(neg)


def solve_hard_svm(X, y):
    """Solve  min 1/2||w||^2  s.t.  y_i (w . x_i + b) >= 1  with SLSQP.
    Returns (w, b). X is (n,2), y is (n,) in {-1,+1}."""
    def obj(v):
        w = v[:2]
        return 0.5 * np.dot(w, w)

    def obj_grad(v):
        return np.array([v[0], v[1], 0.0])

    cons = [{
        "type": "ineq",
        "fun": (lambda v, xi=xi, yi=yi: yi * (v[:2] @ xi + v[2]) - 1.0),
        "jac": (lambda _v, xi=xi, yi=yi: np.array([yi * xi[0], yi * xi[1], yi])),
    } for xi, yi in zip(X, y)]

    v0 = np.array([1.0, 1.0, 0.0])
    res = minimize(obj, v0, jac=obj_grad, constraints=cons, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-9})
    return res.x[:2], res.x[2]


def scatter_classes(ax, pos, neg, s=90):
    ax.scatter(pos[:, 0], pos[:, 1], marker="^", c=POS_COLOR, s=s,
               linewidths=1.4, edgecolors="#145a32", zorder=3, label="admitted ($y=+1$)")
    ax.scatter(neg[:, 0], neg[:, 1], marker="x", c=NEG_COLOR, s=s,
               linewidths=2.4, zorder=3, label="rejected ($y=-1$)")


def style_axes(ax):
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$  (standardized GPA)", fontsize=12)
    ax.set_ylabel(r"$x_2$  (standardized MCAT)", fontsize=12)
    ax.grid(True, color="#444444", linestyle="--", linewidth=0.4, alpha=0.5)


def draw_boundary_and_margins(ax, w, b, xlim, margins=True, lw=2.6, alpha=1.0):
    """Draw w.x + b = 0 solid blue, and w.x + b = +/-1 dashed grey.
    Follows the visual convention documented at the top of this module."""
    xs = np.linspace(xlim[0], xlim[1], 200)
    ax.plot(xs, -(w[0] * xs + b) / w[1], color=LINE_COLOR, lw=lw, ls="-",
            alpha=alpha, zorder=2)
    if margins:
        for level in (+1, -1):
            ax.plot(xs, -(w[0] * xs + b - level) / w[1], color=MARGIN_COLOR,
                    lw=1.6, ls=MARGIN_DASH, alpha=alpha, zorder=2)


def circle_support_vectors(ax, pts, s=240, lw=2.0):
    """Yellow rings marking support vectors."""
    pts = np.atleast_2d(pts)
    if len(pts) == 0:
        return
    ax.scatter(pts[:, 0], pts[:, 1], s=s, facecolors="none",
               edgecolors=SV_COLOR, linewidths=lw, zorder=4)

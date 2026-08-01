# Runs via the root uv environment:
#   uv run python blog/published/support-vector-machines/_src/kernel_comparison.py
#
# KERNELS COME HOME: the payoff figure for the kernel section. Same data, three
# kernels. The data is deliberately NOT linearly separable: admitted students sit
# near the school's target profile and rejected students surround them on all
# sides, so no straight line can carve them apart.
#
#   LEFT   linear kernel  -> a straight boundary, and it fails badly
#   MIDDLE polynomial (d=3) -> a curved boundary, better but still awkward
#   RIGHT  Gaussian (RBF) -> a closed curve that wraps the admitted blob
#
# The point the post makes: NOTHING about the SVM changed between panels except
# which K(x_i, x_j) is plugged into the dual. The boundary is still linear in the
# kernel's feature space; it only looks curved back in the original coordinates.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from sklearn.svm import SVC

from _common import POS_COLOR, NEG_COLOR, LINE_COLOR, SV_COLOR

rng = np.random.default_rng(3)

# Admitted: near the target profile (small radius). Rejected: an annulus around
# it, i.e. lopsided or under-qualified profiles in any direction.
n_in, n_out = 45, 70
r_in = rng.uniform(0.0, 1.05, n_in)
th_in = rng.uniform(0, 2 * np.pi, n_in)
pos = np.column_stack([r_in * np.cos(th_in), r_in * np.sin(th_in)])

r_out = rng.uniform(1.85, 2.75, n_out)
th_out = rng.uniform(0, 2 * np.pi, n_out)
neg = np.column_stack([r_out * np.cos(th_out), r_out * np.sin(th_out)])

X = np.vstack([pos, neg])
y = np.hstack([np.ones(n_in), -np.ones(n_out)])

kernels = [
    dict(name="linear", kw=dict(kernel="linear", C=1.0),
         title="Linear kernel", sub=r"$K(\mathbf{x}_i,\mathbf{x}_j)="
                                   r"\langle \mathbf{x}_i,\mathbf{x}_j\rangle$"),
    dict(name="poly", kw=dict(kernel="poly", degree=3, gamma="scale", coef0=1.0, C=1.0),
         title="Polynomial kernel ($d=3$)",
         sub=r"$K=(\langle \mathbf{x}_i,\mathbf{x}_j\rangle+1)^3$"),
    dict(name="rbf", kw=dict(kernel="rbf", gamma=0.8, C=1.0),
         title="Gaussian (RBF) kernel",
         sub=r"$K=\exp(-\gamma\|\mathbf{x}_i-\mathbf{x}_j\|^2)$"),
]

LIM = 3.2
gg = np.linspace(-LIM, LIM, 400)
XX, YY = np.meshgrid(gg, gg)
grid = np.column_stack([XX.ravel(), YY.ravel()])

# Faint region tint: red half-plane / green half-plane, kept very low contrast so
# the data markers and the boundary stay dominant.
region_cmap = ListedColormap(["#3a2320", "#1f3328"])

fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.0), dpi=300)

for ax, spec in zip(axes, kernels):
    clf = SVC(**spec["kw"]).fit(X, y)
    Z = clf.decision_function(grid).reshape(XX.shape)

    ax.contourf(XX, YY, Z, levels=[-1e9, 0, 1e9], cmap=region_cmap, zorder=0)
    # Decision boundary (solid blue) and the +/-1 margin contours (dashed grey),
    # exactly the convention used everywhere else in the post.
    ax.contour(XX, YY, Z, levels=[0], colors=[LINE_COLOR], linewidths=2.6,
               zorder=3)
    ax.contour(XX, YY, Z, levels=[-1, 1], colors=["#aab7c4"], linewidths=1.3,
               linestyles="dashed", zorder=3)

    ax.scatter(pos[:, 0], pos[:, 1], marker="^", c=POS_COLOR, s=62,
               edgecolors="#145a32", linewidths=1.0, zorder=4,
               label="admitted ($y=+1$)")
    ax.scatter(neg[:, 0], neg[:, 1], marker="x", c=NEG_COLOR, s=62,
               linewidths=1.8, zorder=4, label="rejected ($y=-1$)")
    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=190, facecolors="none",
               edgecolors=SV_COLOR, linewidths=1.5, zorder=5)

    acc = clf.score(X, y)
    ax.set_title(f"{spec['title']}\n{spec['sub']}", fontsize=11.5, pad=8)
    ax.text(0.5, -0.13,
            f"training accuracy {acc:.0%}   |   {len(sv)} support vectors",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color="#e6e6e6")

    # With the linear kernel the SVM cannot do better than calling everything
    # "rejected", so the decision boundary sits entirely outside the frame and
    # the panel would otherwise look like a rendering bug. The note goes BELOW
    # the panel rather than inside it: an in-panel box sat on top of roughly
    # fifteen data points, which is a poor trade for one sentence.
    if Z.min() > 0 or Z.max() < 0:
        ax.text(0.5, -0.235,
                "no straight line can split these,\n"
                "so the SVM labels everything rejected\n"
                "and its boundary lies off the frame",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color="#ff8a7a", linespacing=1.5)

    ax.set_aspect("equal")
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_facecolor("#222222")
    print(f"  {spec['name']:6s} acc={acc:.3f}  n_sv={len(sv)}")

axes[0].legend(loc="upper left", framealpha=0.2, fontsize=9)

fig.suptitle("Same SVM, same data, three kernels: only $K$ changed",
             fontsize=14, y=0.99)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
out = Path(__file__).resolve().parent.parent / "images" / "kernel_comparison.png"
plt.savefig(out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# Output: images/kernel_comparison.png  (dark figure -> embed WITHOUT .invert)

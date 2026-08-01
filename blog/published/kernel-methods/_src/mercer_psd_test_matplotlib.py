# Runs via the root uv environment:
#   uv run python blog/published/kernel-methods/_src/mercer_psd_test_matplotlib.py
#
# STATIC PNG (2 x 2): Mercer's condition, applied. Two symmetric functions that both
# LOOK like reasonable similarity scores are evaluated on the same 7 points, and the
# eigenvalues of the resulting kernel matrices are plotted next to each one.
#   TOP    the Gaussian kernel: every eigenvalue >= 0, so the matrix is positive
#          semidefinite. Mercer says a feature map exists, even though we never find it.
#   BOTTOM the sigmoid ("tanh") kernel, which people really do use: it has a clearly
#          negative eigenvalue on these points, so no feature map can exist and the
#          derivation behind the kernelized algorithm does not apply.
# The point: PSD is a property we can CHECK, numerically, without ever seeing phi.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import BG, FG, GOOD, GRID, MUTED, POINT, use_dark_theme

use_dark_theme()

pts = np.linspace(0.0, 3.0, 7)


def gaussian(a, b, sigma=1.0):
    return np.exp(-((a[:, None] - b[None, :]) ** 2) / (2.0 * sigma ** 2))


def sigmoid_kernel(a, b, kappa=0.6, c=-1.0):
    return np.tanh(kappa * (a[:, None] * b[None, :]) + c)


CASES = [
    (gaussian(pts, pts),
     r"$K(x, z) = \exp\!\left(-\|x-z\|^2 / 2\right)$",
     "valid: every eigenvalue is $\\geq 0$"),
    (sigmoid_kernel(pts, pts),
     r"$K(x, z) = \tanh\!\left(0.6\, x z - 1\right)$",
     "not a kernel: an eigenvalue is $< 0$"),
]

fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), dpi=300,
                         gridspec_kw={"width_ratios": [1.0, 1.25],
                                      "hspace": 0.42, "wspace": 0.28})

for row, (K, formula, verdict) in enumerate(CASES):
    ax_mat, ax_eig = axes[row]
    lim = np.abs(K).max()

    im = ax_mat.imshow(K, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax_mat.set_title(formula, fontsize=12.5, pad=8)
    ax_mat.set_xlabel(r"$j$", fontsize=11)
    ax_mat.set_ylabel(r"$i$", fontsize=11)
    ax_mat.set_xticks(range(len(pts)))
    ax_mat.set_yticks(range(len(pts)))
    cb = fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=FG)

    eigs = np.sort(np.linalg.eigvalsh(K))[::-1]
    colors = [GOOD if e >= -1e-12 else POINT for e in eigs]
    ax_eig.bar(range(len(eigs)), eigs, color=colors, width=0.62, zorder=3)
    ax_eig.axhline(0.0, color=FG, lw=1.0, zorder=2)
    ax_eig.set_title(verdict, fontsize=12.5, pad=8,
                     color=GOOD if row == 0 else POINT)
    ax_eig.set_xlabel("eigenvalue index", fontsize=11)
    ax_eig.set_ylabel(r"eigenvalue of $\mathbf{K}$", fontsize=11)
    ax_eig.set_facecolor(BG)
    ax_eig.grid(True, axis="y", color=GRID, linestyle="--", linewidth=0.4, alpha=0.5)
    ax_eig.set_xticks(range(len(eigs)))

    if row == 1:
        worst = int(np.argmin(eigs))
        ax_eig.set_ylim(eigs.min() - 0.9, eigs.max() * 1.15)
        ax_eig.annotate(f"{eigs[worst]:.2f}", xy=(worst, eigs[worst] * 0.98),
                        xytext=(worst - 2.4, eigs[worst] * 0.72),
                        fontsize=11.5, color=POINT, ha="center",
                        arrowprops=dict(arrowstyle="->", color=POINT, lw=1.1))
        ax_eig.text(0.98, 0.95,
                    "one negative eigenvalue is enough\n"
                    r"to make $\mathbf{z}^{\intercal}\mathbf{K}\mathbf{z} < 0$",
                    transform=ax_eig.transAxes, fontsize=10.5, color=MUTED,
                    ha="right", va="top")

fig.suptitle("Checking Mercer's condition on 7 points, without ever building a feature map",
             fontsize=14, y=0.99)

out = Path(__file__).resolve().parent.parent / "images" / "mercer_psd_test.png"
plt.savefig(out, dpi=300, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# Output: images/mercer_psd_test.png
# Embed in Quarto (dark figure, so no .invert):
#   ![Caption](images/mercer_psd_test.png){#fig-mercer_psd_test fig-align="center" width=100%}

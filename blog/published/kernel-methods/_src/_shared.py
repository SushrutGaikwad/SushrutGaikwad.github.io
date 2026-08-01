# Shared dataset + dark-theme styling for the kernel-methods figures.
# Not a figure script itself: imported by the others so that every figure in the
# post talks about the SAME 14 houses.

import matplotlib.pyplot as plt
import numpy as np

# ---- Palette (kept consistent across every figure in the post) ----
BG = "#222222"
FG = "#e6e6e6"
MUTED = "#9a9a9a"
GRID = "#444444"

POINT = "#e74c3c"   # training points (red crosses)
LINE = "#5dade2"    # the straight-line / linear-model colour (blue)
CURVE = "#f1c40f"   # the flexible/nonlinear fit colour (yellow)
GOOD = "#2ecc71"    # the "cheap kernel route" colour (green)


def use_dark_theme():
    """Match the darkly Quarto theme, with Computer Modern math."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": FG,
        "axes.labelcolor": FG,
        "xtick.color": FG,
        "ytick.color": FG,
        "axes.titlecolor": FG,
        "text.color": FG,
    })


# ---- The running example: 14 houses ----
# x = living area in thousands of square feet, y = price in hundreds of thousands
# of dollars. The true trend rises steeply for small homes, plateaus through the
# mid market, then climbs again at the luxury end: monotone, but with a genuine
# S-shape that no straight line can follow.
X_LABEL = r"$x$  (living area, 1000s sq ft)"
Y_LABEL = r"$y$  (price, \$100k)"


def true_trend(t):
    return 0.75 * (t - 2.5) ** 3 + 0.12 * (t - 2.5) + 6.0


def house_data():
    rng = np.random.default_rng(7)
    x = np.linspace(0.8, 4.2, 14)
    y = true_trend(x) + rng.normal(0.0, 0.22, size=x.shape)
    return x, y


def style_axes(ax, xlabel=X_LABEL, ylabel=None):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linestyle="--", linewidth=0.4, alpha=0.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)


# ---- The kernelized LMS of the post, verbatim ----
def gaussian_kernel_matrix(a, b, sigma):
    diff = np.asarray(a)[:, None] - np.asarray(b)[None, :]
    return np.exp(-(diff ** 2) / (2.0 * sigma ** 2))


def kernel_matrix(a, b, sigma):
    """K(x, z) = 1 + exp(-|x - z|^2 / 2 sigma^2).

    The constant 1 is itself a valid kernel (phi = 1) and sums of kernels are
    kernels, so this is legal; it simply gives the model an intercept.
    """
    return 1.0 + gaussian_kernel_matrix(a, b, sigma)


def kernelized_lms(x, y, sigma, alpha=0.02, iters=4000):
    """beta := beta + alpha (y - K beta), exactly as in the post's algorithm."""
    K = kernel_matrix(x, x, sigma)
    beta = np.zeros_like(y)
    for _ in range(iters):
        beta = beta + alpha * (y - K @ beta)
    return beta


def kernel_predict(beta, x_train, x_query, sigma):
    """h(x) = sum_i beta_i K(x_i, x)."""
    return kernel_matrix(np.atleast_1d(x_query), x_train, sigma) @ beta

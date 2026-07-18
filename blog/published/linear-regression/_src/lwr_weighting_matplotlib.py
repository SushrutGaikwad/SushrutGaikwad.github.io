# Runs via the root uv environment:
#   uv run python blog/published/linear-regression/_src/lwr_weighting_matplotlib.py
#
# STATIC PNG showing how locally weighted linear regression weights the
# training examples around a single query point x. A bell-shaped weight
# curve w(x_i) = exp(-(x_i - x)^2 / (2 tau^2)) sits over the data; each point
# is sized and colored by its weight (near the query -> heavy/bright, far ->
# light/dim). The weighted local line is drawn against the ordinary global
# line so the reader sees the query neighborhood pull the fit toward itself.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path

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


def make_dataset():
    rng = np.random.default_rng(7)
    x = np.linspace(0.5, 9.5, 11)
    f = lambda t: 2.2 * np.sin(0.55 * t) + 0.35 * t + 1.0
    y = f(x) + rng.normal(0.0, 0.40, size=x.shape)
    return x, y


def weights(x_train, xq, tau):
    return np.exp(-((x_train - xq) ** 2) / (2.0 * tau ** 2))


def weighted_line(x_train, y_train, xq, tau):
    """Weighted least-squares fit; returns intercept, slope."""
    X = np.column_stack([np.ones_like(x_train), x_train])
    w = weights(x_train, xq, tau)
    sw = np.sqrt(w)
    theta, *_ = np.linalg.lstsq(X * sw[:, None], y_train * sw, rcond=None)
    return theta


def ordinary_line(x_train, y_train):
    X = np.column_stack([np.ones_like(x_train), x_train])
    theta, *_ = np.linalg.lstsq(X, y_train, rcond=None)
    return theta


x, y = make_dataset()
xq = 6.5           # query point
tau = 1.1          # bandwidth
w = weights(x, xq, tau)

x_grid = np.linspace(x.min() - 0.2, x.max() + 0.2, 400)
bell = weights(x_grid, xq, tau)

theta_local = weighted_line(x, y, xq, tau)
theta_global = ordinary_line(x, y)

fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=300)
ax.set_facecolor("#222222")

# Colour points by their weight.
cmap = plt.get_cmap("plasma")
norm = Normalize(vmin=0.0, vmax=1.0)
sizes = 60 + 320 * w
ax.scatter(x, y, c=w, cmap=cmap, norm=norm, s=sizes, zorder=4,
           edgecolors="#e6e6e6", linewidths=0.8)

# Ordinary global line vs. weighted local line, over the query neighbourhood.
ax.plot(x_grid, theta_global[0] + theta_global[1] * x_grid,
        color="#7f8c8d", lw=1.8, ls="--", zorder=2,
        label="ordinary global line")
win = (x_grid > xq - 2.4 * tau) & (x_grid < xq + 2.4 * tau)
ax.plot(x_grid[win], theta_local[0] + theta_local[1] * x_grid[win],
        color="#2ecc71", lw=3.0, zorder=3, label="weighted local line")

# Query point marker on the local line.
yq = theta_local[0] + theta_local[1] * xq
ax.axvline(xq, color="#e6e6e6", ls=":", lw=1.2, alpha=0.7, zorder=1)
ax.scatter([xq], [yq], marker="*", s=320, c="#f1c40f", edgecolors="#222222",
           linewidths=0.8, zorder=6)
ax.annotate(r"query point $x$", xy=(xq, ax.get_ylim()[0]),
            xytext=(xq + 0.15, y.min() - 1.3), color="#e6e6e6", fontsize=12)

# Bell-shaped weight curve on a twin axis (0..1). Point brightness/size and
# the bell both encode the same weight, so no separate colour bar is needed.
ax2 = ax.twinx()
ax2.plot(x_grid, bell, color="#e67e22", lw=2.2, alpha=0.9, zorder=2)
ax2.fill_between(x_grid, 0, bell, color="#e67e22", alpha=0.12, zorder=0)
ax2.set_ylim(0, 1.5)
ax2.set_ylabel(r"weight $w_i$", color="#e67e22", fontsize=12)
ax2.tick_params(axis="y", colors="#e67e22")

# Bandwidth annotation (width of the bell at a reference height).
ax2.annotate("", xy=(xq - tau, 0.61), xytext=(xq + tau, 0.61),
             arrowprops=dict(arrowstyle="<->", color="#e6e6e6", lw=1.3))
ax2.text(xq, 0.66, r"$\sim \tau$", color="#e6e6e6", fontsize=12, ha="center")

# Weight formula, placed in the clear top-right corner inside the plot.
ax.text(0.975, 0.955,
        r"$w_i = \exp\!\left(-\dfrac{(x_i - x)^2}{2\tau^2}\right)$",
        transform=ax.transAxes, ha="right", va="top", color="#e67e22",
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#222222",
                  edgecolor="#e67e22", alpha=0.85))

ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel(r"$y$", fontsize=12)
ax.set_xlim(x_grid.min(), x_grid.max())
ax.set_ylim(y.min() - 1.6, y.max() + 1.2)
ax.grid(True, color="#444444", linestyle="--", linewidth=0.4, alpha=0.5)
ax.legend(loc="upper left", framealpha=0.3, fontsize=10)

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "images" / "lwr_weighting.png"
plt.savefig(out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")

# Output: images/lwr_weighting.png
# Embed in Quarto (dark figure, so no .invert):
#   ![Caption](images/lwr_weighting.png){#fig-lwr_weighting fig-align="center" width=85%}

# Runs via the root uv environment:
#   uv run python blog/published/linear-regression/_src/lwr_in_action_matplotlib.py
#
# ANIMATION (GIF) that is the centerpiece of the LWR section. A query point
# sweeps left to right across the data. At each position we:
#   - recompute the weights (shown as a sliding bell-shaped window),
#   - fit a weighted straight line to the neighbourhood,
#   - read off its prediction at the query point,
#   - and reveal that prediction on a growing curve.
# The punchline: a smooth curve emerges, assembled entirely from local
# straight-line fits, without ever choosing polynomial features.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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


TAU = 1.1


def weights(x_train, xq, tau=TAU):
    return np.exp(-((x_train - xq) ** 2) / (2.0 * tau ** 2))


def weighted_theta(x_train, y_train, xq, tau=TAU):
    X = np.column_stack([np.ones_like(x_train), x_train])
    w = weights(x_train, xq, tau)
    sw = np.sqrt(w)
    theta, *_ = np.linalg.lstsq(X * sw[:, None], y_train * sw, rcond=None)
    return theta


x, y = make_dataset()

# Query grid the point sweeps over, and the precomputed LWR curve.
q = np.linspace(x.min(), x.max(), 90)
yhat = np.array([weighted_theta(x, y, xq) @ np.array([1.0, xq]) for xq in q])

fig, ax = plt.subplots(figsize=(9.0, 5.6), dpi=150)
ax.set_facecolor("#222222")

ymin, ymax = y.min() - 1.8, y.max() + 1.4
ax.set_xlim(x.min() - 0.3, x.max() + 0.3)
ax.set_ylim(ymin, ymax)
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel(r"$y$", fontsize=12)
ax.grid(True, color="#444444", linestyle="--", linewidth=0.4, alpha=0.5)

# Ghost of the final LWR curve, always faintly present.
ax.plot(q, yhat, color="#f1c40f", lw=1.4, alpha=0.22, zorder=1)

# The data points (recoloured/resized by weight each frame).
base_w = weights(x, q[0])
scat = ax.scatter(x, y, c=base_w, cmap="plasma", vmin=0, vmax=1,
                  s=60 + 300 * base_w, edgecolors="#e6e6e6",
                  linewidths=0.7, zorder=4)

# Sliding weight window on a twin axis.
ax2 = ax.twinx()
ax2.set_ylim(0, 1.7)
ax2.set_yticks([])
xg = np.linspace(x.min() - 0.3, x.max() + 0.3, 400)
(bell_line,) = ax2.plot(xg, weights(xg, q[0]), color="#e67e22", lw=1.8, alpha=0.8)
bell_fill = ax2.fill_between(xg, 0, weights(xg, q[0]), color="#e67e22", alpha=0.10)

# The weighted local line (short segment around the query point).
(local_line,) = ax.plot([], [], color="#2ecc71", lw=3.0, zorder=3,
                        label="weighted local line")
# The revealed LWR curve, growing as the query point advances.
(revealed,) = ax.plot([], [], color="#f1c40f", lw=2.8, zorder=2,
                      label="LWR prediction")
# Query marker and vertical guide.
qmark = ax.scatter([q[0]], [yhat[0]], marker="*", s=340, c="#f1c40f",
                   edgecolors="#222222", linewidths=0.8, zorder=6)
qvline = ax.axvline(q[0], color="#e6e6e6", ls=":", lw=1.1, alpha=0.6, zorder=1)

ax.legend(loc="upper right", framealpha=0.3, fontsize=10)
ax.set_title(r"Locally weighted linear regression: a curve from local straight lines",
             fontsize=12, pad=10)


def update(i):
    global bell_fill
    xq = q[i]
    w = weights(x, xq)
    scat.set_array(w)
    scat.set_sizes(60 + 300 * w)

    theta = weighted_theta(x, y, xq)
    win = np.linspace(xq - 2.3 * TAU, xq + 2.3 * TAU, 40)
    win = win[(win >= x.min() - 0.3) & (win <= x.max() + 0.3)]
    local_line.set_data(win, theta[0] + theta[1] * win)

    revealed.set_data(q[: i + 1], yhat[: i + 1])
    qmark.set_offsets([[xq, yhat[i]]])
    qvline.set_xdata([xq, xq])

    bell_line.set_ydata(weights(xg, xq))
    bell_fill.remove()
    bell_fill = ax2.fill_between(xg, 0, weights(xg, xq), color="#e67e22", alpha=0.10)
    return scat, local_line, revealed, qmark, qvline, bell_line


ani = FuncAnimation(fig, update, frames=len(q), interval=55, blit=False)

out = Path(__file__).resolve().parent.parent / "images" / "lwr_in_action.gif"
ani.save(out, writer="pillow", fps=20, dpi=110,
         savefig_kwargs={"facecolor": "#222222"})
plt.close(fig)
print(f"saved {out}")

# Output: images/lwr_in_action.gif  (short loop, ~4.5 s)
# Embed in Quarto:
#   ![Caption](images/lwr_in_action.gif){#fig-lwr_in_action fig-align="center" width=85%}

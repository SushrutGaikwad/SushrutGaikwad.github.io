# Setup (one-time, on Windows):
#   1. Install Python 3.9+
#   2. pip install numpy matplotlib
#   3. For MP4: install FFmpeg and ensure it is on PATH
#   4. For GIF: pip install pillow (ships with matplotlib)
#
# This script generates a side-by-side MATPLOTLIB ANIMATION comparing
# Newton's method (left) and gradient descent (right) for minimizing the
# same convex 1D function. We use a function with non-uniform curvature
# so Newton's method's curvature-awareness is visually obvious. Both
# methods start at the same theta_0.
#
# Function: l(theta) = 0.5 * (theta - 1.0)^2 + 0.05 * theta^4 + 1
# This is a "narrow valley near the minimum" type function: the second
# derivative is 1 + 0.6 * theta^2, so curvature is bigger far from the
# minimum, smaller near it. Newton handles this gracefully; gradient
# descent with a fixed learning rate has to be cautious and takes many
# small steps.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ---- Style: dark theme matching darkly Quarto ----
plt.style.use('dark_background')
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


# Function l(theta), its derivative, and its second derivative
def l(theta):
    return 0.5 * (theta - 1.0) ** 2 + 0.05 * theta ** 4 + 1.0

def dl(theta):
    return (theta - 1.0) + 0.2 * theta ** 3

def d2l(theta):
    return 1.0 + 0.6 * theta ** 2


# Both methods start here
theta_init = 4.0


# Newton iterates
newton_iters = [theta_init]
for _ in range(7):
    t = newton_iters[-1]
    next_t = t - dl(t) / d2l(t)
    newton_iters.append(next_t)


# Gradient descent iterates with a fixed learning rate
# We pick alpha such that GD doesn't diverge and converges visibly slower than Newton
alpha = 0.05
gd_iters = [theta_init]
for _ in range(40):
    t = gd_iters[-1]
    next_t = t - alpha * dl(t)
    gd_iters.append(next_t)


# Determine total animation length
N_FRAMES = 80


# ---- Set up two-panel figure ----
fig, (ax_newton, ax_gd) = plt.subplots(1, 2, figsize=(8, 5), dpi=500)
fig.patch.set_facecolor('#222222')
for ax in (ax_newton, ax_gd):
    ax.set_facecolor('#222222')

theta_grid = np.linspace(-1.5, 5.0, 400)
y_grid = l(theta_grid)

# Plot l(theta) on both axes
for ax in (ax_newton, ax_gd):
    ax.plot(theta_grid, y_grid, color='#1abc9c', linewidth=2.0, label=r'$\ell(\theta)$')
    ax.axhline(l(0.83), color='#444444', linewidth=0.5, linestyle='--', alpha=0.4)
    # The actual minimum: solve dl(theta) = 0 numerically
    # (theta - 1) + 0.2*theta^3 = 0; minimum is around theta ~ 0.83
    ax.set_xlabel(r'$\theta$', fontsize=11)
    ax.set_ylabel(r'$\ell(\theta)$', fontsize=11)
    ax.set_xlim(-1.0, 4.5)
    ax.set_ylim(0.5, 17)
    ax.grid(True, color='#444444', linestyle='--', linewidth=0.4)

ax_newton.set_title("Newton's method", fontsize=12)
ax_gd.set_title('Gradient descent', fontsize=12)


# Trail of past iterates (line + dots)
newton_trail, = ax_newton.plot([], [], 'o-', color='#f1c40f', markersize=7,
                                linewidth=1.0, markeredgecolor='#222222')
newton_current, = ax_newton.plot([], [], 'o', color='#e74c3c', markersize=10)

gd_trail, = ax_gd.plot([], [], 'o-', color='#f1c40f', markersize=4,
                        linewidth=0.8, markeredgecolor='#222222')
gd_current, = ax_gd.plot([], [], 'o', color='#e74c3c', markersize=10)

newton_text = ax_newton.text(0.03, 0.97, '', transform=ax_newton.transAxes, fontsize=10,
                             verticalalignment='top',
                             bbox=dict(facecolor='#222222', edgecolor='#e6e6e6', boxstyle='round,pad=0.3'))
gd_text = ax_gd.text(0.03, 0.97, '', transform=ax_gd.transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(facecolor='#222222', edgecolor='#e6e6e6', boxstyle='round,pad=0.3'))


def init():
    newton_trail.set_data([], [])
    newton_current.set_data([], [])
    gd_trail.set_data([], [])
    gd_current.set_data([], [])
    newton_text.set_text('')
    gd_text.set_text('')
    return newton_trail, newton_current, gd_trail, gd_current, newton_text, gd_text


def update(frame):
    # Newton: 1 frame per iteration, slowed down so we hold each step
    # Show iterate index newton_idx = min(frame // 8, len(newton_iters) - 1)
    newton_idx = min(frame // 10, len(newton_iters) - 1)
    nx = newton_iters[: newton_idx + 1]
    ny = [l(t) for t in nx]
    newton_trail.set_data(nx, ny)
    newton_current.set_data([nx[-1]], [ny[-1]])
    newton_text.set_text(rf'Iteration {newton_idx}: $\theta = {nx[-1]:.3f}$')

    # Gradient descent: 1 frame per iteration (faster)
    gd_idx = min(frame // 2, len(gd_iters) - 1)
    gx = gd_iters[: gd_idx + 1]
    gy = [l(t) for t in gx]
    gd_trail.set_data(gx, gy)
    gd_current.set_data([gx[-1]], [gy[-1]])
    gd_text.set_text(rf'Iteration {gd_idx}: $\theta = {gx[-1]:.3f}$')

    return newton_trail, newton_current, gd_trail, gd_current, newton_text, gd_text


anim = FuncAnimation(
    fig, update, init_func=init,
    frames=N_FRAMES, interval=80, blit=True,
)


# anim.save("newton_vs_gd.gif", writer="pillow", fps=15, dpi=500,
#           savefig_kwargs={"facecolor": "#222222"})
# Or for MP4:
from pathlib import Path
_out = Path(__file__).resolve().parent.parent / "images" / "newton_vs_gd.mp4"
anim.save(_out, writer="ffmpeg", fps=20, dpi=500,
          savefig_kwargs={"facecolor": "#222222"})


plt.close(fig)


# How to run (Windows):
#   python newton_vs_gd_matplotlib.py
#
# Output:
#   newton_vs_gd.gif (in the same directory)
#
# Recommended embed: this is a moderate-length animation (~6-10 sec).
# GIF works fine; MP4 also works via the {{< video >}} shortcode.
#
# Embed in Quarto using (after copying to images/):
#   ::: {#fig-newton_vs_gd}
#   {{< video images/newton_vs_gd.mp4 >}}
#
#   Caption text.
#   :::
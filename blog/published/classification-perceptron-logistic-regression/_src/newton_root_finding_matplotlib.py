# Setup (one-time, on Windows):
#   1. Install Python 3.9+
#   2. pip install numpy matplotlib
#   3. For saving as MP4: install FFmpeg and ensure it is on PATH (https://www.gyan.dev/ffmpeg/builds/ or `winget install ffmpeg`).
#   4. For saving as GIF: pip install pillow  (Pillow writer ships with matplotlib)
#
# This script generates a MATPLOTLIB ANIMATION showing Newton's method
# for root finding. Starting from theta^(0) = 4.5, the algorithm draws the
# tangent line to f at the current iterate, finds where the tangent crosses
# zero, and uses that as the next iterate. The true root is at about
# theta = 1.3.
#
# The function used is f(theta) = theta^3 - 6*theta^2 + 11*theta - 6.6
# which has roots at approximately 1.3, 2.0, 2.7 (chosen to give a clean
# Newton's method demo from theta^(0) = 4.5).
# Actually, we will use a simpler function with a single visible root:
#   f(theta) = (theta - 1.3) * (theta - 4) + 0.5  
# but we prefer a function that decreases monotonically near the right root
# starting from 4.5, so that Newton converges cleanly. We'll use:
#   f(theta) = 0.6 * (theta - 1.3) ** 2 - 0.5,   no wait that's a parabola.
# Cleanest demo: f(theta) = exp(0.5 * (theta - 1.3)) - 1
# whose unique real root is theta = 1.3, derivative f'(theta) = 0.5 * exp(0.5*(theta-1.3)),
# always positive. Newton converges from any starting point.
#
# The iteration index is denoted by a parenthesized superscript, e.g.
# theta^(0), theta^(1), theta^(t), theta^(t+1). This matches the convention
# used in the post text and avoids a notational clash with the plain
# subscript theta_j, which is reserved for the j-th component of the
# parameter vector when we generalize to the vector case.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ---- Style: dark theme matching the darkly Quarto theme ----
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


# ---- Function and Newton iteration ----
def f(theta):
    return np.exp(0.5 * (theta - 1.3)) - 1.0


def df(theta):
    return 0.5 * np.exp(0.5 * (theta - 1.3))


# Compute a sequence of Newton iterates
theta0 = 4.5
iterates = [theta0]
for _ in range(8):
    t = iterates[-1]
    next_t = t - f(t) / df(t)
    iterates.append(next_t)
# Keep only the first ~5 iterations for visibility
iterates = iterates[:6]


# ---- Set up figure ----
fig, ax = plt.subplots(figsize=(8, 5), dpi=500)
fig.patch.set_facecolor('#222222')
ax.set_facecolor('#222222')

theta_min, theta_max = -0.5, 5.5
theta_grid = np.linspace(theta_min, theta_max, 400)

ax.plot(theta_grid, f(theta_grid), color='#1abc9c', linewidth=2.5, label=r'$f(\theta)$')
ax.axhline(0, color='#e6e6e6', linewidth=1, linestyle='-', alpha=0.6)
ax.set_xlim(theta_min, theta_max)
ax.set_ylim(-2.0, 6.0)
ax.set_xlabel(r'$\theta$', fontsize=12)
ax.set_ylabel(r'$f(\theta)$', fontsize=12)
ax.set_title(r"Newton's method: tangent line at $\theta^{(t)}$ crosses zero at $\theta^{(t+1)}$", fontsize=11)
ax.grid(True, color='#444444', linestyle='--', linewidth=0.5)
ax.legend(loc='lower right', framealpha=0.4)


# Pre-create artists that will be updated each frame
tangent_line, = ax.plot([], [], color='#f1c40f', linewidth=1.8, linestyle='-')
current_pt, = ax.plot([], [], 'o', color='#e74c3c', markersize=9)
next_pt_marker, = ax.plot([], [], 'v', color='#e67e22', markersize=10)
vline = ax.plot([], [], color='#e74c3c', linewidth=1, linestyle=':')[0]
iter_text = ax.text(0.03, 0.97, '', transform=ax.transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(facecolor='#222222', edgecolor='#e6e6e6', boxstyle='round,pad=0.3'))


# ---- Animation logic ----
# Per Newton iteration we have 3 phases:
# 1) Show current point on f
# 2) Draw tangent line at current point
# 3) Drop a vertical line from the tangent's x-intercept to the curve and
#    advance to the next iterate (which becomes the new current point)
FRAMES_PER_PHASE = 25
PHASES = 3  # show point, draw tangent, jump to next iterate
N_ITERS = len(iterates) - 1  # number of Newton steps
TOTAL_FRAMES = N_ITERS * PHASES * FRAMES_PER_PHASE


def init():
    tangent_line.set_data([], [])
    current_pt.set_data([], [])
    next_pt_marker.set_data([], [])
    vline.set_data([], [])
    iter_text.set_text('')
    return tangent_line, current_pt, next_pt_marker, vline, iter_text


def update(frame):
    # Determine which iteration and phase
    iter_idx = frame // (PHASES * FRAMES_PER_PHASE)
    if iter_idx >= N_ITERS:
        iter_idx = N_ITERS - 1
    phase_frame = frame - iter_idx * PHASES * FRAMES_PER_PHASE
    phase = phase_frame // FRAMES_PER_PHASE
    progress = (phase_frame % FRAMES_PER_PHASE) / FRAMES_PER_PHASE

    t_current = iterates[iter_idx]
    t_next = iterates[iter_idx + 1]
    f_curr = f(t_current)
    df_curr = df(t_current)

    # The tangent line spans theta values around the current iterate
    tx = np.array([min(t_current, t_next) - 0.2, max(t_current, t_next) + 0.2])
    ty = df_curr * (tx - t_current) + f_curr

    # Always show the current point
    current_pt.set_data([t_current], [f_curr])

    if phase == 0:
        # Phase 0: just show the current point with a vertical dotted line down to x-axis
        tangent_line.set_data([], [])
        next_pt_marker.set_data([], [])
        vline.set_data([t_current, t_current], [0, f_curr])
        iter_text.set_text(rf'Iteration {iter_idx}: $\theta^{{({iter_idx})}} = {t_current:.3f}$')
    elif phase == 1:
        # Phase 1: draw tangent line, growing
        # Show partial tangent based on progress
        tx_start = tx[0]
        tx_end = tx[0] + progress * (tx[1] - tx[0])
        tangent_grow_x = np.linspace(tx_start, tx_end, 50)
        tangent_grow_y = df_curr * (tangent_grow_x - t_current) + f_curr
        tangent_line.set_data(tangent_grow_x, tangent_grow_y)
        next_pt_marker.set_data([], [])
        vline.set_data([t_current, t_current], [0, f_curr])
        iter_text.set_text(rf'Tangent at $\theta^{{({iter_idx})}}$, slope $= f^{{\prime}}(\theta^{{({iter_idx})}}) = {df_curr:.3f}$')
    else:
        # Phase 2: tangent fully drawn; show its x-intercept and label as theta^(t+1)
        tangent_line.set_data(tx, ty)
        # show next iterate marker on x-axis
        next_pt_marker.set_data([t_next], [0])
        # vertical guide from current f value down through 0 + extension up from x-axis to function value at t_next
        # Show two vlines: one from current pt down (always), and one from t_next on x-axis up to f(t_next), only at end
        if progress > 0.5:
            vline.set_data([t_next, t_next], [0, f(t_next)])
        else:
            vline.set_data([t_current, t_current], [0, f_curr])
        iter_text.set_text(rf'Next iterate: $\theta^{{({iter_idx + 1})}} = {t_next:.3f}$')

    return tangent_line, current_pt, next_pt_marker, vline, iter_text


anim = FuncAnimation(
    fig, update, init_func=init,
    frames=TOTAL_FRAMES, interval=40, blit=True,
)


# ---- Save ----
# anim.save("newton_root_finding.gif", writer="pillow", fps=25, dpi=500,
#           savefig_kwargs={"facecolor": "#222222"})
# To save as MP4 instead, comment out the line above and uncomment the next:
from pathlib import Path
_out = Path(__file__).resolve().parent.parent / "images" / "newton_root_finding.mp4"
anim.save(_out, writer="ffmpeg", fps=25, dpi=500,
          savefig_kwargs={"facecolor": "#222222"})


plt.close(fig)


# How to run (Windows):
#   python newton_root_finding_matplotlib.py
# (run from the directory containing this script in PowerShell or cmd)
#
# Output:
#   newton_root_finding.gif (in the same directory)
#
# Recommended embed: the animation is short (~10 sec) so a GIF is fine.
#
# Embed in Quarto using (after copying the gif to images/):
#   ::: {#fig-newton_root_finding}
#   {{< video images/newton_root_finding.mp4 >}}
#
#   Caption text.
#   :::
# Note: if you save as GIF, embed with ![Caption](images/newton_root_finding.gif){...} instead.
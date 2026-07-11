# Setup (one-time, on Windows):
#   1. Install Python 3.9+
#   2. pip install numpy matplotlib
#   3. For saving as MP4: install FFmpeg and ensure it is on PATH
#      (https://www.gyan.dev/ffmpeg/builds/ or `winget install ffmpeg`).
#   4. For saving as GIF: pip install pillow (Pillow writer ships with matplotlib).
#
# This script generates a MATPLOTLIB ANIMATION showing the full perceptron
# training algorithm on a synthetic 2D linearly-separable dataset, starting
# from a parameter vector pointed in the EXACTLY OPPOSITE direction to the
# correct one. With this initialization every example is initially
# misclassified, so every visit triggers an update. We watch the boundary
# swing through the data and converge to a separating hyperplane.
#
# Pedagogical design choices:
#
# (1) NO INTERCEPT. We use theta = (theta_1, theta_2) and x = (x_1, x_2),
#     no theta_0 term. This means the boundary theta^T x = 0 is always a line
#     through the origin perpendicular to theta, which matches the prose in
#     the post text exactly ("theta is perpendicular to the decision boundary
#     and the boundary passes through the origin"). With an intercept the
#     boundary would be offset from the origin, which would muddle the
#     visual story. The dataset is designed to be separable through the
#     origin, so the no-intercept perceptron can solve it.
#
# (2) INITIALIZATION. Compute the natural good direction theta_star from
#     the cluster centers, then initialize theta to -theta_star. This makes
#     EVERY example initially misclassified. No "no-op" frames at the start.
#
# (3) HYBRID PACING. The first three updates get full pacing with all
#     phases visible. Subsequent updates are condensed. The final converged
#     state is held for two seconds.
#
# (4) FRAME PHASES. Each update has three phases:
#       Phase A (highlight current example): a yellow ring appears around
#         x_i, and a status text shows iteration number, true label, and
#         prediction.
#       Phase B (update arrow): if y_i != prediction, an arrow showing the
#         update vector alpha * (y_i - h) * x_i is drawn from the head of
#         the current theta to the head of the new theta.
#       Phase C (rotate boundary): theta and the boundary smoothly rotate
#         to the new orientation. Half-space tints update accordingly.
#
# (5) HALF-SPACE TINTS. The side theta points into is faintly orange
#     (predicted positive); the other side is faintly blue (predicted
#     negative). Updates re-tint as theta rotates.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MplPolygon


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


# =====================================================================
# DATASET
# =====================================================================
# Two clusters along the +/- (1, 1) direction, slightly elongated
# perpendicular to that direction so the cluster spread is visible.
# This makes the boundary's rotation more dramatic.

rng = np.random.default_rng(seed=7)
n_per_class = 9

# Direction of true separator (perpendicular to the line between the cluster
# centers). The good theta_star points along (1, 1), so the boundary is the
# line x_1 + x_2 = 0 (the "anti-diagonal").
center_pos = np.array([2.5, 2.5])
center_neg = np.array([-2.5, -2.5])

# Elongation: cluster spread is wider perpendicular to the separating
# direction, so the dataset visually clusters but is still cleanly separable.
along = np.array([1.0, 1.0]) / np.sqrt(2)        # parallel to theta_star
across = np.array([1.0, -1.0]) / np.sqrt(2)      # perpendicular to theta_star

spread_along = 0.55
spread_across = 1.4

X_pos = (center_pos
         + spread_along * rng.standard_normal((n_per_class, 1)) * along
         + spread_across * rng.standard_normal((n_per_class, 1)) * across)
X_neg = (center_neg
         + spread_along * rng.standard_normal((n_per_class, 1)) * along
         + spread_across * rng.standard_normal((n_per_class, 1)) * across)
X = np.vstack([X_pos, X_neg])
y = np.concatenate([np.ones(n_per_class), np.zeros(n_per_class)])

# Sanity check: the dataset is linearly separable through the origin, since
# the clusters lie on opposite sides of the line x_1 + x_2 = 0 (with margin).
assert np.all(X_pos.sum(axis=1) > 0)
assert np.all(X_neg.sum(axis=1) < 0)


# =====================================================================
# PERCEPTRON: run the algorithm and record the iterates
# =====================================================================

def step_predict(theta, x):
    """Perceptron prediction: 1 if theta . x >= 0, else 0."""
    return 1 if np.dot(theta, x) >= 0 else 0


theta_star = (center_pos - center_neg)
theta_star = theta_star / np.linalg.norm(theta_star)

# Initialize theta to point in the EXACTLY OPPOSITE direction.
# We use a relatively large initial magnitude (init_scale = 2.0) and a small
# learning rate (alpha = 0.05) so that individual updates are gentle nudges
# rather than dramatic swings. With these settings, the boundary moves
# slightly with each of the first ~10 updates, then the cumulative effect
# of those nudges reaches a tipping point and the boundary swings around
# decisively in the final 1-2 updates, snapping into a separating
# configuration. This makes the visual story clearer than a setting where
# a single large update would solve the problem outright.
theta_init = -theta_star * 2.0
alpha = 0.05  # learning rate

# Visit order: shuffle once and cycle.
visit_order = rng.permutation(len(X))

history = []  # list of dicts capturing each step
theta_curr = theta_init.copy()
max_iters = 80
ix = 0
n = len(X)

while ix < max_iters:
    i = visit_order[ix % n]
    x_i = X[i]
    y_i = y[i]
    pred = step_predict(theta_curr, x_i)
    miss = (pred != y_i)
    delta = alpha * (y_i - pred) * x_i if miss else np.zeros(2)
    theta_next = theta_curr + delta
    history.append({
        "ix": ix,
        "i": i,
        "x_i": x_i.copy(),
        "y_i": int(y_i),
        "pred": pred,
        "miss": miss,
        "theta_before": theta_curr.copy(),
        "theta_after": theta_next.copy(),
        "delta": delta.copy(),
    })
    theta_curr = theta_next

    # Check global convergence: every example correctly classified.
    preds_all = np.array([step_predict(theta_curr, X[j]) for j in range(n)])
    if np.all(preds_all == y) and ix >= n:
        break
    ix += 1

print(f"Perceptron converged after {len(history)} visits.")

# Trim trailing no-op frames (correctly-classified examples that did not
# trigger updates) at the very end except keep one to show "we visited the
# remaining examples and they were all correct."
# Actually, keep the full sequence; "no-op" frames after convergence are
# pedagogically valuable: they show the algorithm checking each example
# and finding nothing to fix.

# Identify which steps caused updates (for pacing decisions later).
update_steps = [k for k, h in enumerate(history) if h["miss"]]


# =====================================================================
# FIGURE SETUP
# =====================================================================

fig, ax = plt.subplots(figsize=(9, 6.5), dpi=500)
fig.patch.set_facecolor('#222222')
ax.set_facecolor('#222222')

# Plot extents
LIM = 5.5
ax.set_xlim(-LIM, LIM)
ax.set_ylim(-LIM, LIM)
ax.set_aspect('equal')
ax.set_xlabel(r'$x_1$', fontsize=12)
ax.set_ylabel(r'$x_2$', fontsize=12)
ax.grid(True, color='#444444', linestyle='--', linewidth=0.4, alpha=0.4)
ax.axhline(0, color='#555555', linewidth=0.6, alpha=0.5)
ax.axvline(0, color='#555555', linewidth=0.6, alpha=0.5)

# Colors
COL_POS = '#e67e22'
COL_NEG = '#3498db'
COL_THETA = '#2ecc71'
COL_BOUNDARY = '#ecf0f1'
COL_HIGHLIGHT = '#f1c40f'
COL_DELTA = '#e74c3c'

# Static scatter (always visible)
sc_pos = ax.scatter(X_pos[:, 0], X_pos[:, 1], c=COL_POS, s=80,
                     edgecolors='#222222', linewidths=1.0, zorder=4,
                     label=r'$y_i = 1$')
sc_neg = ax.scatter(X_neg[:, 0], X_neg[:, 1], c=COL_NEG, s=80,
                     edgecolors='#222222', linewidths=1.0, zorder=4,
                     label=r'$y_i = 0$')

# Pre-create animated artists
# Half-space tints (filled polygons) - we'll update their vertices each frame
pos_polygon = MplPolygon([[0, 0]], closed=True, facecolor=COL_POS,
                          alpha=0.10, edgecolor='none', zorder=1)
neg_polygon = MplPolygon([[0, 0]], closed=True, facecolor=COL_NEG,
                          alpha=0.10, edgecolor='none', zorder=1)
ax.add_patch(pos_polygon)
ax.add_patch(neg_polygon)

# Boundary line
boundary_line, = ax.plot([], [], color=COL_BOUNDARY, linewidth=2.5, zorder=2)

# theta arrow (we will recreate this each frame because Arrow patches don't
# update cleanly; use ax.annotate which can be recreated)
theta_arrow_artist = [None]  # mutable container

# Highlight ring around current example
highlight_ring, = ax.plot([], [], 'o', markerfacecolor='none',
                           markeredgecolor=COL_HIGHLIGHT, markersize=22,
                           markeredgewidth=2.5, zorder=5)

# Misclassification rings (drawn on all currently-misclassified points)
miss_rings, = ax.plot([], [], 'o', markerfacecolor='none',
                       markeredgecolor=COL_DELTA, markersize=14,
                       markeredgewidth=1.5, zorder=3, alpha=0.7)

# Update vector arrow (delta), shown briefly during phase B
delta_arrow_artist = [None]

# Status text box
status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                      verticalalignment='top',
                      bbox=dict(facecolor='#222222', edgecolor='#e6e6e6',
                                boxstyle='round,pad=0.4'),
                      family='serif')

# Legend
ax.legend(loc='lower right', framealpha=0.4, fontsize=10)

# Title
ax.set_title("Perceptron training: from anti-aligned init to convergence",
              fontsize=12)


def draw_theta_arrow(theta):
    """Recreate the theta arrow (annotate-based for clean redraw)."""
    if theta_arrow_artist[0] is not None:
        theta_arrow_artist[0].remove()
    # Scale theta for visibility but keep direction faithful.
    norm = np.linalg.norm(theta)
    if norm < 1e-9:
        theta_arrow_artist[0] = None
        return
    # Draw arrow from origin to theta, with length capped to keep on-figure.
    visible_theta = theta * (min(3.0, norm) / norm)
    theta_arrow_artist[0] = ax.annotate(
        '', xy=visible_theta, xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color=COL_THETA,
                         lw=3.0, mutation_scale=22),
        zorder=6,
    )


def draw_delta_arrow(theta_before, theta_after):
    """Show the update vector as an arrow from head of theta_before to head of theta_after."""
    if delta_arrow_artist[0] is not None:
        delta_arrow_artist[0].remove()
        delta_arrow_artist[0] = None
    # Apply same on-screen scaling as draw_theta_arrow used.
    def visible(v):
        n = np.linalg.norm(v)
        if n < 1e-9:
            return v
        return v * (min(3.0, n) / n)
    vb = visible(theta_before)
    va = visible(theta_after)
    if np.linalg.norm(va - vb) < 0.05:
        return
    delta_arrow_artist[0] = ax.annotate(
        '', xy=va, xytext=vb,
        arrowprops=dict(arrowstyle='->', color=COL_DELTA,
                         lw=2.0, mutation_scale=18, alpha=0.9),
        zorder=6,
    )


def clear_delta_arrow():
    if delta_arrow_artist[0] is not None:
        delta_arrow_artist[0].remove()
        delta_arrow_artist[0] = None


def draw_boundary_and_halfspaces(theta):
    """Update the boundary line and half-space polygon vertices for a given theta."""
    norm = np.linalg.norm(theta)
    if norm < 1e-9:
        boundary_line.set_data([], [])
        pos_polygon.set_xy([[0, 0]])
        neg_polygon.set_xy([[0, 0]])
        return
    # Perpendicular direction to theta
    th = theta / norm
    perp = np.array([-th[1], th[0]])
    # Boundary line spans from -BIG*perp to +BIG*perp
    BIG = 10.0
    p1 = BIG * perp
    p2 = -BIG * perp
    boundary_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])

    # Half-space polygons: each is a quadrilateral.
    # Positive side is the side theta points into.
    BIG2 = 10.0
    pos_quad = np.array([
        p1,
        p1 + BIG2 * th,
        p2 + BIG2 * th,
        p2,
    ])
    neg_quad = np.array([
        p1,
        p1 - BIG2 * th,
        p2 - BIG2 * th,
        p2,
    ])
    pos_polygon.set_xy(pos_quad)
    neg_polygon.set_xy(neg_quad)


def update_miss_rings(theta):
    """Draw red rings on all currently-misclassified points."""
    preds = np.array([step_predict(theta, X[j]) for j in range(n)])
    miss_mask = preds != y
    if np.any(miss_mask):
        miss_rings.set_data(X[miss_mask, 0], X[miss_mask, 1])
    else:
        miss_rings.set_data([], [])


# =====================================================================
# ANIMATION TIMELINE
# =====================================================================
# Each step has up to three phases. We allocate a different number of
# frames to each phase depending on whether we're in the "slow first
# updates" portion or the "fast tail" portion.

SLOW_UPDATES = 3  # number of update steps that get slow pacing
SLOW_FRAMES = {"A": 18, "B": 14, "C": 22}  # phase frame counts (slow)
FAST_FRAMES = {"A": 6, "B": 4, "C": 8}     # phase frame counts (fast)
HOLD_FRAMES = 50  # final hold

# Build the frame schedule: a list of (step_index, phase_name, phase_progress)
# where progress is a float in [0, 1].
schedule = []
n_slow_used = 0
for k, h in enumerate(history):
    is_update = h["miss"]
    if is_update and n_slow_used < SLOW_UPDATES:
        F = SLOW_FRAMES
        n_slow_used += 1
    else:
        F = FAST_FRAMES

    # Phase A: highlight current example
    for f in range(F["A"]):
        schedule.append((k, "A", f / max(1, F["A"] - 1)))
    # Phase B: show update vector (only if it's an update)
    if is_update:
        for f in range(F["B"]):
            schedule.append((k, "B", f / max(1, F["B"] - 1)))
    # Phase C: rotate theta and boundary
    for f in range(F["C"]):
        schedule.append((k, "C", f / max(1, F["C"] - 1)))

# Final hold
last_step = len(history) - 1
for f in range(HOLD_FRAMES):
    schedule.append((last_step, "HOLD", 1.0))

TOTAL_FRAMES = len(schedule)
print(f"Total animation frames: {TOTAL_FRAMES}")


# =====================================================================
# FRAME RENDERING
# =====================================================================

def init():
    boundary_line.set_data([], [])
    highlight_ring.set_data([], [])
    miss_rings.set_data([], [])
    status_text.set_text('')
    return (boundary_line, highlight_ring, miss_rings, status_text,
            pos_polygon, neg_polygon)


def render_frame(frame_idx):
    step_idx, phase, progress = schedule[frame_idx]
    h = history[step_idx]

    # During phase C, theta interpolates from theta_before to theta_after.
    # In other phases, theta is "before" (A) or "before" (B).
    if phase == "C":
        # Interpolate angle (slerp-ish; for 2D we can just linearly interp
        # the vector and renormalize, or interp the angle directly).
        tb = h["theta_before"]
        ta = h["theta_after"]
        theta_now = (1 - progress) * tb + progress * ta
    elif phase == "HOLD":
        theta_now = history[-1]["theta_after"]
    else:
        theta_now = h["theta_before"]

    # Boundary and half-space tints.
    draw_boundary_and_halfspaces(theta_now)

    # theta arrow.
    draw_theta_arrow(theta_now)

    # Misclassification rings (against the current theta_now).
    update_miss_rings(theta_now)

    # Highlight ring on current example, only during phases A and B.
    if phase in ("A", "B"):
        highlight_ring.set_data([h["x_i"][0]], [h["x_i"][1]])
    else:
        highlight_ring.set_data([], [])

    # Delta arrow only during phase B.
    if phase == "B" and h["miss"]:
        # Fade in over the phase
        draw_delta_arrow(h["theta_before"], h["theta_after"])
    elif phase == "C":
        # Keep showing delta arrow (faintly) during the rotation phase
        # to reinforce the connection. Actually clearer to clear it.
        clear_delta_arrow()
    elif phase == "HOLD":
        clear_delta_arrow()
    else:
        clear_delta_arrow()

    # Status text.
    if phase == "HOLD":
        status_text.set_text(
            f"Converged after {len(history)} visits "
            f"({len(update_steps)} updates).\n"
            f"All training examples correctly classified."
        )
    else:
        # Use ASCII labels in the status box; mathtext for theta.
        status_text.set_text(
            f"Visit #{h['ix'] + 1}\n"
            f"Example $i = {h['i']}$\n"
            f"True label $y_i = {h['y_i']}$\n"
            f"Prediction $h(x_i) = {h['pred']}$\n"
            + ("UPDATE: $\\theta \\leftarrow \\theta + \\alpha\\, x_i$"
               if (h['miss'] and h['y_i'] == 1)
               else "UPDATE: $\\theta \\leftarrow \\theta - \\alpha\\, x_i$"
               if (h['miss'] and h['y_i'] == 0)
               else "No update (correct).")
        )

    return (boundary_line, highlight_ring, miss_rings, status_text,
            pos_polygon, neg_polygon)


anim = FuncAnimation(
    fig, render_frame, init_func=init,
    frames=TOTAL_FRAMES, interval=40, blit=False,  # blit off because annotate
)

# Pacing: at 25 fps, the schedule above gives roughly:
#   slow updates: ~54 frames each at slow pacing
#   fast updates: ~18 frames each
#   no-ops:        ~6 to 14 frames
# Tune `interval` and the schedule constants above if the result feels off.

# ---- Save ----
# MP4 is recommended (animation is too long for GIF). Requires FFmpeg on PATH.
from pathlib import Path
_out = Path(__file__).resolve().parent.parent / "images" / "perceptron_full_training.mp4"
anim.save(_out, writer="ffmpeg", fps=25, dpi=500,
           savefig_kwargs={"facecolor": "#222222"})

# Alternative GIF output (larger file, lower quality):
# anim.save("perceptron_full_training.gif", writer="pillow", fps=20, dpi=500,
#           savefig_kwargs={"facecolor": "#222222"})

plt.close(fig)


# How to run (Windows):
#   python perceptron_full_training_matplotlib.py
# (run from the directory containing this script in PowerShell or cmd)
#
# Output:
#   perceptron_full_training.mp4 (in the same directory)
#
# Recommended embed: MP4 (the animation is around 18 to 22 seconds, too long
# for a GIF). Move the MP4 into your post's images/ directory.
#
# Embed in Quarto using:
#   ::: {#fig-perceptron_full_training}
#   {{< video images/perceptron_full_training.mp4 >}}
#
#   The perceptron in action on a synthetic 2D dataset, initialized so that
#   theta points in exactly the wrong direction (every example initially
#   misclassified). At each step we highlight the current example (yellow
#   ring), show the update vector if needed (red arrow), and rotate theta
#   and the boundary accordingly. The boundary swings around through the
#   data and converges to a separating hyperplane.
#   :::
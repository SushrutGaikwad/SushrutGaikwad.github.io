# Runs via the root uv environment:
#   uv run python blog/published/kernel-methods/_src/feature_map_lift_animation.py
#
# MATPLOTLIB ANIMATION (MP4): the classic "lift" that makes a feature map's power
# visceral, told in five beats.
#   1. A genuine 2D panel: an inner disk and an outer ring, with a straight line
#      spinning through every angle and failing at every one of them.
#   2. The feature map phi(x1, x2) = (x1, x2, x1^2 + x2^2) lifts each point to a
#      height equal to its squared distance from the origin, while the camera tilts.
#   3. A flat separating plane fades in at height 2.5.
#   4. A slow orbit, so the reader sees the separation is real and not a trick of
#      the viewing angle.
#   5. THE PAYOFF: the points sink back down and the plane's shadow is drawn on the
#      floor. A flat plane upstairs is a CIRCLE downstairs. That is what the feature
#      map bought us.
# The first beat uses a real 2D axes (not a top-down 3D one), so there are no
# collapsed panes or edge-on z-axis artifacts.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Dark theme matching the darkly Quarto theme. Titles use genuine LaTeX (bmatrix,
# \boldsymbol, \upphi, \intercal), rendered through the TeX Live install on PATH.
plt.style.use("dark_background")
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{upgreek}",
    "font.family": "serif",
    "text.color": "#e6e6e6",
    "axes.labelcolor": "#e6e6e6",
})

BG = "#222222"
FG = "#e6e6e6"
DIM_GRID = "#3a3a3a"
PANE = (0.145, 0.145, 0.145, 1.0)

INNER_COLOR = "#5dade2"   # inner disk  (class A)
OUTER_COLOR = "#e74c3c"   # outer ring  (class B)
PLANE_COLOR = "#2ecc71"   # translucent separating plane, and its shadow

# ---- Two concentric classes in 2D ----
rng = np.random.default_rng(11)
n_per = 120

r_in = np.sqrt(rng.uniform(0.0, 1.15 ** 2, n_per))
th_in = rng.uniform(0, 2 * np.pi, n_per)
inner = np.column_stack([r_in * np.cos(th_in), r_in * np.sin(th_in)])

r_out = np.sqrt(rng.uniform(1.95 ** 2, 3.0 ** 2, n_per))
th_out = rng.uniform(0, 2 * np.pi, n_per)
outer = np.column_stack([r_out * np.cos(th_out), r_out * np.sin(th_out)])

pts = np.vstack([inner, outer])
z_full = pts[:, 0] ** 2 + pts[:, 1] ** 2          # the lifted height, x1^2 + x2^2
colors = np.array([INNER_COLOR] * n_per + [OUTER_COLOR] * n_per)

Z_SEP = 2.5      # plane height: above the inner max (~1.32), below the outer min (~3.80)
Z_MAX = 9.5
R_SEP = np.sqrt(Z_SEP)   # the circle the plane casts on the floor
LIM = 3.35

# ---- Timeline (30 fps) ----
# Paced for READING, not for motion. Every beat that puts new words on screen gets
# its own hold, and the two beats carrying the most text (the opening claim and the
# closing payoff) get the longest ones. Total is a little over 18 seconds.
F_FLAT = 78        # 2D panel, line spinning and failing
F_FLAT_PHI = 54    # still 2D, but the feature map appears so it can be read first
F_LIFT = 78        # lift the points, tilt the camera
F_LIFT_HOLD = 42   # hold on the lifted cloud before anything else changes
F_PLANE = 30       # fade the plane in
F_PLANE_HOLD = 42  # hold, so "a flat plane now separates them" can be read
F_ORBIT = 78       # slow orbit
F_DROP = 66        # sink back down, plane fades out, shadow circle appears
F_HOLD = 96        # hold on the circle: the payoff line, and the longest beat
TOTAL = (F_FLAT + F_FLAT_PHI + F_LIFT + F_LIFT_HOLD + F_PLANE + F_PLANE_HOLD
         + F_ORBIT + F_DROP + F_HOLD)

LIFT_TITLE = (
    r"Lift with "
    r"$\boldsymbol{\upphi}\!\left(\begin{bmatrix} x_1 & x_2 \end{bmatrix}^{\intercal}\right)"
    r" = \begin{bmatrix} x_1 & x_2 & x_1^2 + x_2^2 \end{bmatrix}^{\intercal}$"
)

# 7.2 x 6.0 inches at 180 dpi = 1296 x 1080 pixels, i.e. a true 1080p-tall frame
# (both dimensions even, which h264 requires). Font sizes are in points, so they
# scale with the dpi and the text stays the same relative size, just sharper.
fig = plt.figure(figsize=(7.2, 6.0), dpi=180)
fig.patch.set_facecolor(BG)

# A real 2D axes for beat 1, and a 3D axes for the rest. Both occupy the same box,
# and only one is ever visible, so the cut between them lands on matching geometry.
BOX = [0.09, 0.07, 0.82, 0.80]
ax2 = fig.add_axes(BOX)
ax3 = fig.add_axes(BOX, projection="3d")
ax3.set_proj_type("ortho")


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def style_2d():
    ax2.clear()
    ax2.set_facecolor(BG)
    ax2.set_xlim(-LIM, LIM)
    ax2.set_ylim(-LIM, LIM)
    ax2.set_aspect("equal")
    ax2.grid(True, color=DIM_GRID, linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.set_xlabel(r"$x_1$", fontsize=13)
    ax2.set_ylabel(r"$x_2$", fontsize=13)
    ax2.tick_params(colors="#b0b0b0", labelsize=9)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax2.spines[side].set_color("#666666")


def style_3d():
    ax3.clear()
    ax3.set_facecolor(BG)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.set_pane_color(PANE)
        axis.line.set_color("#666666")
        axis._axinfo["grid"]["color"] = DIM_GRID
        axis._axinfo["grid"]["linewidth"] = 0.5
    ax3.tick_params(colors="#b0b0b0", labelsize=8)
    # mplot3d leaves a wide internal margin; zoom in so the scene fills the frame.
    ax3.set_box_aspect((1.0, 1.0, 0.9), zoom=1.15)
    ax3.set_xlim(-LIM, LIM)
    ax3.set_ylim(-LIM, LIM)
    ax3.set_zlim(0.0, Z_MAX)
    ax3.set_xlabel(r"$x_1$", fontsize=12, labelpad=2)
    ax3.set_ylabel(r"$x_2$", fontsize=12, labelpad=2)
    ax3.set_zlabel(r"$x_1^2 + x_2^2$", fontsize=12, labelpad=6)


# Plane mesh and the circle it casts on the floor.
gx = np.linspace(-LIM, LIM, 2)
GX, GY = np.meshgrid(gx, gx)
GZ = np.full_like(GX, Z_SEP)
circle_t = np.linspace(0, 2 * np.pi, 200)
CIRCLE_X = R_SEP * np.cos(circle_t)
CIRCLE_Y = R_SEP * np.sin(circle_t)


def set_title(text, subtitle=None):
    """Always reserve two lines, so the frame never jumps when the caption changes."""
    fig.suptitle(text + "\n" + (subtitle if subtitle else r"\ "),
                 color=FG, fontsize=12.5, y=0.975, linespacing=1.5)


def update(frame):
    fig.suptitle("")

    # ---------------- Beats 1 and 2: the honest 2D picture ----------------
    if frame < F_FLAT + F_FLAT_PHI:
        ax3.set_visible(False)
        ax2.set_visible(True)
        style_2d()

        # A line through the origin, spinning through a full half-turn during the
        # first beat and then parked, so it stops competing with the formula.
        angle = np.pi * min(frame, F_FLAT) / F_FLAT
        t = np.linspace(-LIM * 1.5, LIM * 1.5, 2)
        ax2.plot(t * np.cos(angle), t * np.sin(angle),
                 color="#f1c40f", lw=2.2, zorder=2,
                 alpha=1.0 if frame < F_FLAT else 0.35)
        ax2.scatter(pts[:, 0], pts[:, 1], c=colors, s=20, zorder=3,
                    edgecolors="none")

        if frame < F_FLAT:
            set_title("In 2D, no straight line separates the two classes",
                      r"spin it however you like: red always lands on "
                      r"\emph{both} sides")
        else:
            # The feature map goes up while we are still in 2D, so the reader can
            # read it before anything starts moving.
            set_title(LIFT_TITLE,
                      r"so give every point a third coordinate, and watch")
        return []

    ax2.set_visible(False)
    ax3.set_visible(True)
    style_3d()

    # ---------------- Beats 3 to 6 ----------------
    f = frame - F_FLAT - F_FLAT_PHI
    plane_alpha, shadow_alpha = 0.0, 0.0

    t_lift = F_LIFT
    t_lift_hold = t_lift + F_LIFT_HOLD
    t_plane = t_lift_hold + F_PLANE
    t_plane_hold = t_plane + F_PLANE_HOLD
    t_orbit = t_plane_hold + F_ORBIT
    t_drop = t_orbit + F_DROP

    if f < t_lift:                                     # lift and tilt
        p = smoothstep(f / F_LIFT)
        lift = p
        elev, azim = 89.0 + p * (22.0 - 89.0), -90.0 + p * 30.0
        title = LIFT_TITLE
        subtitle = r"every point rises to its own squared distance from the origin"
    elif f < t_lift_hold:                              # hold on the lifted cloud
        lift, elev, azim = 1.0, 22.0, -60.0
        title = LIFT_TITLE
        subtitle = r"the inner class stays low, the outer ring shoots up"
    elif f < t_plane:                                  # plane fades in
        p = smoothstep((f - t_lift_hold) / F_PLANE)
        lift, elev, azim = 1.0, 22.0, -60.0
        plane_alpha = 0.16 * p
        title = r"A \emph{flat} plane now separates them"
        subtitle = r"the plane $x_1^2 + x_2^2 = 2.5$, drawn upstairs"
    elif f < t_plane_hold:                             # hold on the plane
        lift, elev, azim = 1.0, 22.0, -60.0
        plane_alpha = 0.16
        title = r"A \emph{flat} plane now separates them"
        subtitle = r"the plane $x_1^2 + x_2^2 = 2.5$, drawn upstairs"
    elif f < t_orbit:                                  # orbit
        p = (f - t_plane_hold) / max(F_ORBIT - 1, 1)
        lift, elev, azim = 1.0, 22.0, -60.0 + p * 45.0
        plane_alpha = 0.16
        title = r"A \emph{flat} plane now separates them"
        subtitle = r"from any angle: blue below, red above"
    elif f < t_drop:                                   # sink back, shadow appears
        p = smoothstep((f - t_orbit) / F_DROP)
        lift = 1.0 - p
        elev, azim = 22.0 + p * 26.0, -15.0
        plane_alpha = 0.16 * (1.0 - p)
        shadow_alpha = p
        title = r"Now come back down"
        subtitle = r"what does that flat plane look like in the original plane?"
    else:                                              # hold on the circle
        lift, elev, azim = 0.0, 48.0, -15.0
        shadow_alpha = 1.0
        title = r"A flat plane upstairs is a \emph{circle} downstairs"
        subtitle = (r"the feature map did not make the boundary straight; "
                    r"it made it \emph{reachable}")

    ax3.view_init(elev=elev, azim=azim)

    if plane_alpha > 0.0:
        ax3.plot_surface(GX, GY, GZ, color=PLANE_COLOR, alpha=plane_alpha,
                         shade=False, zorder=1)
    if shadow_alpha > 0.0:
        ax3.plot(CIRCLE_X, CIRCLE_Y, np.zeros_like(CIRCLE_X), color=PLANE_COLOR,
                 lw=3.0, alpha=shadow_alpha, zorder=6)

    ax3.scatter(pts[:, 0], pts[:, 1], lift * z_full, c=colors, s=20,
                depthshade=False, edgecolors="none", zorder=3)

    set_title(title, subtitle)
    return []


ani = FuncAnimation(fig, update, frames=TOTAL, interval=1000 / 30, blit=False)

out = Path(__file__).resolve().parent.parent / "images" / "feature_map_lift.mp4"
# dpi must match the figure's own dpi or the 1296 x 1080 frame is silently resampled.
# yuv420p is what browsers actually decode; crf 20 keeps the LaTeX text crisp.
ani.save(out, writer="ffmpeg", fps=30, dpi=180,
         savefig_kwargs={"facecolor": BG},
         extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                     "-crf", "20", "-preset", "slow"])
plt.close(fig)
print(f"saved {out}")

# Output: images/feature_map_lift.mp4  (1296 x 1080, about 18.8 s)
# Embed in Quarto with the video shortcode wrapped in a div, which is how every
# other post on the site does it: the clip becomes a numbered, cross-referenceable
# figure and Quarto copies the .mp4 into _site automatically.
#   ::: {#fig-feature_map_lift}
#   {{< video images/feature_map_lift.mp4 >}}
#
#   Caption text.
#   :::

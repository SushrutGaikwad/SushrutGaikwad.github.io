# Runs via the root uv environment:
#   uv run python blog/published/kernel-methods/_src/feature_map_lift_animation.py
#
# MATPLOTLIB ANIMATION (GIF): the classic "lift" that makes a feature map's power
# visceral. In 2D, an inner class (a disk) and an outer class (a ring) are NOT
# separable by any straight line. Apply the feature map
#       phi(x1, x2) = (x1, x2, x1^2 + x2^2),
# i.e. give every point a height equal to its squared distance from the origin.
# The inner disk stays low, the ring rises high, and a flat horizontal PLANE now
# separates the two classes cleanly. The animation starts top-down (looks 2D),
# then lifts the points into the third dimension while the camera tilts, and finally
# fades in the separating plane.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

# Dark theme matching the darkly Quarto theme.
# The 3D-phase title uses genuine LaTeX (bmatrix, \boldsymbol, \upphi, \intercal),
# so text is rendered through a real LaTeX install (text.usetex=True). TeX Live is
# installed and on PATH on this machine, so amsmath/amssymb/upgreek are available.
plt.style.use("dark_background")
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{upgreek}",
    "font.family": "serif",
    "text.color": "#e6e6e6",
    "axes.labelcolor": "#e6e6e6",
})

INNER_COLOR = "#5dade2"   # inner disk  (class A)
OUTER_COLOR = "#e74c3c"   # outer ring  (class B)
PLANE_COLOR = "#2ecc71"   # translucent separating plane

# ---- Two concentric classes in 2D ----
rng = np.random.default_rng(11)
n_per = 130

r_in = np.sqrt(rng.uniform(0.0, 1.15 ** 2, n_per))
th_in = rng.uniform(0, 2 * np.pi, n_per)
inner = np.column_stack([r_in * np.cos(th_in), r_in * np.sin(th_in)])

r_out = np.sqrt(rng.uniform(1.95 ** 2, 3.0 ** 2, n_per))
th_out = rng.uniform(0, 2 * np.pi, n_per)
outer = np.column_stack([r_out * np.cos(th_out), r_out * np.sin(th_out)])

pts = np.vstack([inner, outer])
z_full = (pts[:, 0] ** 2 + pts[:, 1] ** 2)          # the lifted height, x1^2 + x2^2
colors = np.array([INNER_COLOR] * n_per + [OUTER_COLOR] * n_per)

Z_SEP = 2.5     # separating plane: between inner max (~1.32) and outer min (~3.80)
Z_MAX = 9.5

# ---- Animation timeline (frames) ----
F_HOLD = 18                          # top-down flat hold (reads as pure 2D)
F_LIFT = 60                          # lift points + tilt camera
F_PLANE = 20                         # fade the plane in
F_SPIN = 42                          # gentle rotation while holding the 3D view
TOTAL = F_HOLD + F_LIFT + F_PLANE + F_SPIN


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


fig = plt.figure(figsize=(7.2, 6.2), dpi=150)
fig.patch.set_facecolor("#222222")
ax = fig.add_subplot(111, projection="3d")

# Precompute a mesh for the separating plane.
gx = np.linspace(-3.2, 3.2, 2)
gy = np.linspace(-3.2, 3.2, 2)
GX, GY = np.meshgrid(gx, gy)
GZ = np.full_like(GX, Z_SEP)


def style_axes():
    ax.set_facecolor("#222222")
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0.13, 0.13, 0.13, 1.0))
        pane.line.set_color("#666666")
    ax.tick_params(colors="#b0b0b0", labelsize=8)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_zlim(0.0, Z_MAX)
    ax.set_xlabel(r"$x_1$", fontsize=12, labelpad=2)
    ax.set_ylabel(r"$x_2$", fontsize=12, labelpad=2)
    ax.set_zlabel(r"$x_1^2 + x_2^2$", fontsize=12, labelpad=4)


def update(frame):
    ax.cla()
    style_axes()

    # Phase progress -------------------------------------------------------
    if frame < F_HOLD:
        lift = 0.0
        elev, azim = 89.0, -90.0
        plane_alpha = 0.0
    elif frame < F_HOLD + F_LIFT:
        p = smoothstep((frame - F_HOLD) / F_LIFT)
        lift = p
        elev = 89.0 + p * (22.0 - 89.0)
        azim = -90.0 + p * (-60.0 + 90.0)
        plane_alpha = 0.0
    elif frame < F_HOLD + F_LIFT + F_PLANE:
        p = (frame - F_HOLD - F_LIFT) / F_PLANE
        lift = 1.0
        elev, azim = 22.0, -60.0
        plane_alpha = 0.32 * smoothstep(p)
    else:
        p = (frame - F_HOLD - F_LIFT - F_PLANE) / max(F_SPIN - 1, 1)
        lift = 1.0
        elev = 22.0
        azim = -60.0 + p * 40.0
        plane_alpha = 0.32

    ax.view_init(elev=elev, azim=azim)

    z = lift * z_full
    # Draw the plane first (underneath) once it starts appearing.
    if plane_alpha > 0.0:
        ax.plot_surface(GX, GY, GZ, color=PLANE_COLOR, alpha=plane_alpha,
                        shade=False, zorder=1)

    ax.scatter(pts[:, 0], pts[:, 1], z, c=colors, s=22,
               depthshade=True, edgecolors="none", zorder=3)

    if lift < 0.15:
        ax.set_title("In 2D: no straight line separates the two classes",
                     color="#e6e6e6", fontsize=13, pad=6)
    else:
        ax.set_title(
            r"Lift with "
            r"$\boldsymbol{\upphi}\!\left(\begin{bmatrix} x_1 & x_2 \end{bmatrix}^{\intercal}\right)"
            r" = \begin{bmatrix} x_1 & x_2 & x_1^2 + x_2^2 \end{bmatrix}^{\intercal}$"
            "\n" r"a flat plane now separates them",
            color="#e6e6e6", fontsize=12.5, pad=6)
    return []


ani = FuncAnimation(fig, update, frames=TOTAL, interval=1000 / 30, blit=False)

out = Path(__file__).resolve().parent.parent / "images" / "feature_map_lift.gif"
ani.save(out, writer="pillow", fps=30, dpi=150,
         savefig_kwargs={"facecolor": "#222222"})
plt.close(fig)
print(f"saved {out}")

# Output: images/feature_map_lift.gif  (~4.7 s loop)
# Embed in Quarto (dark GIF, so no .invert):
#   ![Caption](images/feature_map_lift.gif){fig-align="center" width=85%}

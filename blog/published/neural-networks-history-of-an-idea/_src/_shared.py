# Shared dark-theme styling and palette for the "history of an idea" figures.
# Not a figure script itself: imported by the others so every figure in the post
# uses the same colours, the same Computer Modern math, and the same darkly
# background (#222222).

import matplotlib.pyplot as plt

# ---- Palette (kept consistent across every figure in the post) ----
BG = "#222222"
FG = "#e6e6e6"
MUTED = "#9a9a9a"
GRID = "#444444"

FIRE = "#e74c3c"     # a unit that is firing / the "1" class / excitatory
QUIET = "#5dade2"    # a unit that is silent / the "0" class / the calm colour
ACCENT = "#f1c40f"   # the object of interest: the boundary, the pulse, the region
GOOD = "#2ecc71"     # a construction that works
BAD = "#e67e22"      # a construction that fails, or an unstable quantity


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
        "legend.facecolor": BG,
        "legend.edgecolor": MUTED,
    })


def style_axes(ax, xlabel=None, ylabel=None, grid=True):
    """Apply the post's axis styling in one call."""
    ax.set_facecolor(BG)
    if grid:
        ax.grid(True, color=GRID, linestyle="--", linewidth=0.4, alpha=0.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)


def unit_square_axes(ax):
    """The (0,0)-(1,1) Boolean input square that AND, OR, NOT and XOR live on."""
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(-0.45, 1.45)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_facecolor(BG)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.grid(False)


# ---- The running example ----
# A two-sensor storm detector. x1 = "flash seen", x2 = "rumble heard", either as
# Booleans (sections on gates) or as real-valued brightness/loudness readings
# (sections on hyperplanes and decision regions).
X1_LABEL = r"$x_1$  (flash)"
X2_LABEL = r"$x_2$  (rumble)"

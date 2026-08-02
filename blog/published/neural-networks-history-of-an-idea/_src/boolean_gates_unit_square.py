# Runs via the root uv environment with uv run (see command at the bottom).
"""Four Boolean functions of two sensors, drawn on the unit square.

The first three (AND, OR, NOT) each have a straight line that puts every firing
corner on one side and every silent corner on the other, so one threshold unit
computes them. XOR does not: its two firing corners are diagonally opposite, so
no line can separate them from the other diagonal. That is the whole argument
for why a single perceptron cannot compute XOR, made visually.

Laid out 2x2 rather than 1x4 so that at the width the post embeds it, each panel
gets about half the column instead of a quarter. XOR sits bottom right, last in
reading order, because it is the punchline.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import ACCENT, BG, FG, FIRE, MUTED, QUIET, use_dark_theme, unit_square_axes

use_dark_theme()

CORNERS = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])


def shade_halfplane(ax, w1, w2, b, color=ACCENT, alpha=0.20):
    """Shade the region where w1*x1 + w2*x2 + b >= 0 and draw its boundary."""
    xs = np.linspace(-0.45, 1.45, 400)
    ys = np.linspace(-0.45, 1.45, 400)
    gx, gy = np.meshgrid(xs, ys)
    ax.contourf(gx, gy, w1 * gx + w2 * gy + b, levels=[0, 1e6],
                colors=[color], alpha=alpha)
    # The boundary line w1*x1 + w2*x2 + b = 0, clipped to the visible box.
    if abs(w2) > 1e-9:
        ax.plot(xs, -(w1 * xs + b) / w2, color=color, lw=2.4)
    else:
        ax.axvline(-b / w1, color=color, lw=2.4)


def draw_corners(ax, labels):
    """Plot the four Boolean inputs, coloured by the value the gate outputs."""
    for (x1, x2), out in zip(CORNERS, labels):
        ax.scatter([x1], [x2], s=210, zorder=5,
                   color=FIRE if out else QUIET,
                   edgecolors=FG, linewidths=1.2)
        ax.annotate(f"({x1},{x2})", (x1, x2), textcoords="offset points",
                    xytext=(0, 18), ha="center", fontsize=11, color=MUTED)


# (title, halfplane or None, outputs at (0,0) (1,0) (0,1) (1,1))
PANELS = [
    (r"AND $\;=\;$ storm confirmed" "\n" r"$x_1 + x_2 - 1.5 \geq 0$",
     (1, 1, -1.5), [0, 0, 0, 1]),
    (r"OR $\;=\;$ something happened" "\n" r"$x_1 + x_2 - 0.5 \geq 0$",
     (1, 1, -0.5), [0, 1, 1, 1]),
    (r"NOT $x_1$ $\;=\;$ no flash" "\n" r"$-x_1 + 0.5 \geq 0$",
     (-1, 0, 0.5), [1, 0, 1, 0]),
    (r"XOR $\;=\;$ exactly one sensor" "\n" r"no line works",
     None, [0, 1, 1, 0]),
]

fig, axes = plt.subplots(2, 2, figsize=(9.6, 10.2))
flat = axes.ravel()

for ax, (title, halfplane, outputs) in zip(flat, PANELS):
    unit_square_axes(ax)
    if halfplane is not None:
        shade_halfplane(ax, *halfplane)
    else:
        # XOR: three honest attempts, each one corner short.
        xs = np.linspace(-0.45, 1.45, 200)
        for (w1, w2, b), style in [((1, 1, -0.5), (4, 3)), ((1, 1, -1.5), (1, 2)),
                                   ((1, -1, 0.0), (6, 2))]:
            ax.plot(xs, -(w1 * xs + b) / w2, color=MUTED, lw=1.5,
                    dashes=style, alpha=0.75, zorder=1)
        ax.annotate("every line leaves one\ncorner on the wrong side",
                    xy=(0.5, -0.30), ha="center", va="center",
                    fontsize=11.5, color=MUTED,
                    bbox=dict(facecolor=BG, edgecolor="none", alpha=0.9, pad=3))
    draw_corners(ax, outputs)
    ax.set_title(title, fontsize=14, pad=14)

# Axis labels only on the outer edges, so the inner panels stay uncluttered.
for ax in axes[1, :]:
    ax.set_xlabel(r"$x_1$ (flash)", fontsize=12.5)
for ax in axes[:, 0]:
    ax.set_ylabel(r"$x_2$ (rumble)", fontsize=12.5)

handles = [
    plt.Line2D([], [], marker="o", ls="", color=FIRE, markersize=12,
               markeredgecolor=FG, label="unit fires (output 1)"),
    plt.Line2D([], [], marker="o", ls="", color=QUIET, markersize=12,
               markeredgecolor=FG, label="unit stays silent (output 0)"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           fontsize=13, bbox_to_anchor=(0.5, 0.006))

# bottom=0.125 reserves a strip for the legend that the bottom row's x-labels
# cannot reach down into.
fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.125,
                    wspace=0.16, hspace=0.28)

out = Path(__file__).resolve().parent.parent / "images" / "boolean_gates_unit_square.png"
fig.savefig(out, dpi=200, facecolor=BG)
print(f"wrote {out}  ({fig.get_size_inches()[0]:.1f} x "
      f"{fig.get_size_inches()[1]:.1f} in)")

# Run (from the repo root):
#   uv run python blog/published/neural-networks-history-of-an-idea/_src/boolean_gates_unit_square.py
# Output: blog/published/neural-networks-history-of-an-idea/images/boolean_gates_unit_square.png
# Embed with:
#   ![...](images/boolean_gates_unit_square.png){#fig-boolean_gates_unit_square fig-align="center" width=100%}

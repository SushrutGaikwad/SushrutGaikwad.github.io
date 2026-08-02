# Runs via the root uv environment with uv run (see command at the bottom).
"""From one square pulse to any function you like.

A pair of threshold units, summed with weights +1 and -1, gives a pulse that is
1 between two thresholds and 0 everywhere else. Scale that pulse by a number and
you own one narrow slab of the output. Lay slabs side by side across the domain
and you own the whole function, to whatever accuracy you are willing to pay for
in units: halve the width, halve the worst-case error.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import ACCENT, BG, FG, GOOD, MUTED, QUIET, use_dark_theme

use_dark_theme()

X_LO, X_HI = 0.0, 6.0
xs = np.linspace(X_LO, X_HI, 2000)


def target(x):
    """A deliberately awkward function: humps, a dip, and no symmetry to exploit."""
    return (1.05 * np.sin(1.5 * x)
            + 0.55 * np.sin(3.1 * x + 0.8)
            + 0.22 * (x - 3.0)
            + 0.35)


f_true = target(xs)


def staircase(width):
    """Pulse edges and heights for a given pulse width, height = midpoint value."""
    edges = np.arange(X_LO, X_HI + 1e-9, width)
    mids = (edges[:-1] + edges[1:]) / 2
    return edges, target(mids)


def approx_curve(edges, heights, n_shown):
    """The approximation using only the first n_shown pulses; NaN elsewhere."""
    out = np.full_like(xs, np.nan)
    for k in range(n_shown):
        band = (xs >= edges[k]) & (xs <= edges[k + 1])
        out[band] = heights[k]
    return out


STAGES = [
    (0.75, 12, "one pulse at a time"),      # width, frames per pulse, note
    (0.375, 3, "halve the width"),
    (0.1875, 1, "halve it again"),
]

# ---- Frame schedule -------------------------------------------------------
frames = []
for s, (width, per_pulse, _) in enumerate(STAGES):
    edges, heights = staircase(width)
    n = len(heights)
    for k in range(1, n + 1):
        frames += [(s, k)] * per_pulse
    frames += [(s, n)] * (75 if s == 0 else 60)      # hold and read the error
frames += [(len(STAGES) - 1, len(staircase(STAGES[-1][0])[1]))] * 60

fig, ax = plt.subplots(figsize=(10.0, 5.6), dpi=150)
fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.13)


def draw(frame):
    s, n_shown = frame
    width, _, note = STAGES[s]
    edges, heights = staircase(width)

    ax.clear()
    ax.set_facecolor(BG)
    ax.grid(True, color="#444444", linestyle="--", linewidth=0.4, alpha=0.4)

    # The function we are chasing.
    ax.plot(xs, f_true, color=QUIET, lw=3.0, label=r"target $f(x)$", zorder=4)

    # The pulses placed so far, drawn as filled slabs.
    for k in range(n_shown):
        ax.fill_between([edges[k], edges[k + 1]], 0, heights[k],
                        color=ACCENT, alpha=0.30, zorder=2)
        ax.plot([edges[k], edges[k + 1]], [heights[k], heights[k]],
                color=ACCENT, lw=2.2, zorder=3)
        ax.plot([edges[k], edges[k]], [0, heights[k]], color=ACCENT, lw=0.9,
                alpha=0.55, zorder=2)
        ax.plot([edges[k + 1], edges[k + 1]], [0, heights[k]], color=ACCENT,
                lw=0.9, alpha=0.55, zorder=2)

    # The newest pulse, highlighted while the stage is still filling in.
    if n_shown < len(heights):
        k = n_shown - 1
        ax.fill_between([edges[k], edges[k + 1]], 0, heights[k],
                        color=GOOD, alpha=0.45, zorder=3)

    ax.axhline(0, color=MUTED, lw=0.9, alpha=0.7)
    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(-2.3, 2.9)
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$y$", fontsize=13)
    ax.legend(loc="upper left", fontsize=11, frameon=False)

    done = n_shown == len(heights)
    if done:
        approx = approx_curve(edges, heights, n_shown)
        worst = float(np.nanmax(np.abs(f_true - approx)))
        title = f"{len(heights)} pulses, width {width:g}"
        # Adjacent pulses share an edge, so n pulses need n+1 threshold units,
        # not 2n: pulse k is (step at T_k) minus (step at T_{k+1}).
        sub = (f"worst-case error {worst:.3f}      "
               f"({len(heights) + 1} threshold units and one summing unit)")
        colour = GOOD
    else:
        title = f"{note}: pulse {n_shown} of {len(heights)}"
        sub = ("each pulse is two threshold units summed with weights "
               r"$+1$ and $-1$, scaled by the height of $f$ there")
        colour = FG

    ax.set_title(title, fontsize=15, pad=34, color=colour)
    ax.text(0.5, 1.04, sub, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11, color=MUTED)
    return ()


ani = FuncAnimation(fig, draw, frames=frames, interval=1000 / 30, blit=False)

out = Path(__file__).resolve().parent.parent / "images" / "pulse_stacking.mp4"
ani.save(out, writer="ffmpeg", fps=30, dpi=150,
         savefig_kwargs={"facecolor": BG},
         extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                     "-crf", "20", "-preset", "slow"])
print(f"wrote {out}")

# Run (from the repo root):
#   uv run python blog/published/neural-networks-history-of-an-idea/_src/pulse_stacking.py
# Output: blog/published/neural-networks-history-of-an-idea/images/pulse_stacking.mp4
# Embed as a numbered figure with:
#   ::: {#fig-pulse_stacking}
#   {{< video images/pulse_stacking.mp4 >}}
#
#   Caption text.
#   :::

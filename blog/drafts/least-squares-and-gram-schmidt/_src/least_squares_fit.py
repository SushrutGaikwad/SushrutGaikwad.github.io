"""Least-squares line-fitting animation for Post 8.

The line y = C + D t swings from a poor guess to the best fit y = 2/3 + t/2,
with vertical residual segments shrinking and the sum of squared errors shown.

Runs via the root uv environment:
  uv run python blog/published/least-squares-and-gram-schmidt/_src/least_squares_fit.py
Output (MP4): blog/published/least-squares-and-gram-schmidt/images/least_squares_fit.mp4
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use("dark_background")
plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})

LIGHT = "#e6e6e6"
DARK = "#222222"
BLUE = "#5dade2"
ORANGE = "#f5b041"
RED = "#ec7063"

t = np.array([1.0, 2.0, 3.0])
y = np.array([1.0, 2.0, 2.0])

C_star, D_star = 2.0 / 3.0, 0.5      # best fit
C0, D0 = 2.6, -0.35                  # deliberately poor starting line

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)

tt = np.linspace(0.4, 3.6, 100)
(line_artist,) = ax.plot([], [], color=BLUE, lw=2.5, label="current fit")
ax.scatter(t, y, color=ORANGE, s=90, zorder=5, label="data")
resid_lines = [ax.plot([], [], color=RED, lw=2, ls="--")[0] for _ in t]
sse_text = ax.text(0.05, 0.92, "", transform=ax.transAxes, color=LIGHT, fontsize=13)

ax.set_xlim(0.4, 3.6)
ax.set_ylim(0, 3.2)
ax.set_xlabel(r"$t$", color=LIGHT, fontsize=13)
ax.set_ylabel(r"$y$", color=LIGHT, fontsize=13)
ax.tick_params(colors=LIGHT)
for spine in ax.spines.values():
    spine.set_color(LIGHT)
ax.grid(True, color="#3a3a3a", lw=0.6)
ax.legend(facecolor="#333333", edgecolor=LIGHT, labelcolor=LIGHT, fontsize=11, loc="lower right")

N = 90


def ease(s):
    return 0.5 - 0.5 * np.cos(np.pi * s)


def update(frame):
    s = ease(min(frame, N) / N)
    C = C0 + (C_star - C0) * s
    D = D0 + (D_star - D0) * s
    line_artist.set_data(tt, C + D * tt)
    pred = C + D * t
    for k in range(len(t)):
        resid_lines[k].set_data([t[k], t[k]], [y[k], pred[k]])
    sse = float(np.sum((y - pred) ** 2))
    sse_text.set_text(rf"sum of squared errors $= {sse:.3f}$")
    return [line_artist, sse_text, *resid_lines]


# Hold on the final fit for a beat at the end.
frames = list(range(N + 1)) + [N] * 20
ani = FuncAnimation(fig, update, frames=frames, interval=40, blit=True)

out = Path(__file__).resolve().parent.parent / "images" / "least_squares_fit.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
ani.save(out, writer="ffmpeg", fps=30, dpi=150, savefig_kwargs={"facecolor": DARK})
print(f"saved {out}")

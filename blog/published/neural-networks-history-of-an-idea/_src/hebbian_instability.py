# Runs via the root uv environment with uv run (see command at the bottom).
"""Why Hebb's rule, on its own, erases what it learns.

Three connections into the same neuron, co-firing at three different rates. Hebb
only ever adds, so every one of them climbs. The frequent connection gets to the
ceiling first and the rare one gets there last, but they all get there, and once
they do the neuron cannot tell them apart any more. The bars on the right start
out informative and end up flat, which is the failure the lecture calls
"fundamentally unstable".
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import ACCENT, BAD, BG, FG, GOOD, MUTED, QUIET, use_dark_theme

use_dark_theme()

# ---- The simulation -------------------------------------------------------
ETA = 0.02          # learning rate
W_MAX = 1.0         # the synapse cannot grow past this
T = 300             # co-firing opportunities
RATES = [0.90, 0.55, 0.22]
NAMES = [r"frequent co-firing  ($p = 0.90$)",
         r"occasional  ($p = 0.55$)",
         r"rare  ($p = 0.22$)"]
COLOURS = [BAD, ACCENT, QUIET]

rng = np.random.default_rng(11)
fires = np.array([rng.random(T) < p for p in RATES])       # 3 x T
weights = np.zeros((3, T + 1))
for t in range(T):
    weights[:, t + 1] = np.minimum(weights[:, t] + ETA * fires[:, t], W_MAX)

saturated_at = [int(np.argmax(weights[i] >= W_MAX - 1e-9)) for i in range(3)]

# ---- The figure -----------------------------------------------------------
fig = plt.figure(figsize=(10.8, 4.8), dpi=140)
gs = fig.add_gridspec(1, 2, width_ratios=[2.05, 1.0], wspace=0.28,
                      left=0.075, right=0.975, top=0.80, bottom=0.16)
ax = fig.add_subplot(gs[0, 0])
bar_ax = fig.add_subplot(gs[0, 1])

ax.set_xlim(0, T)
ax.set_ylim(0, 1.22)
ax.set_xlabel("co-firing opportunities", fontsize=11.5)
ax.set_ylabel(r"connection strength $w$", fontsize=11.5)
ax.set_facecolor(BG)
ax.grid(True, color="#444444", linestyle="--", linewidth=0.4, alpha=0.45)
ax.axhline(W_MAX, color=MUTED, lw=1.3, dashes=(6, 4))
ax.annotate("saturation ceiling", xy=(6, W_MAX + 0.035), fontsize=10.5,
            color=MUTED)

lines, heads = [], []
for i in range(3):
    (ln,) = ax.plot([], [], color=COLOURS[i], lw=2.6, label=NAMES[i])
    (hd,) = ax.plot([], [], marker="o", color=COLOURS[i], markersize=7, ls="")
    lines.append(ln)
    heads.append(hd)
ax.legend(loc="lower right", fontsize=10, frameon=False)

bars = bar_ax.bar([0, 1, 2], [0, 0, 0], color=COLOURS, width=0.62,
                  edgecolor=FG, linewidth=0.8)
bar_ax.set_ylim(0, 1.22)
bar_ax.set_xlim(-0.65, 2.65)
bar_ax.set_xticks([0, 1, 2])
bar_ax.set_xticklabels(["frequent", "occasional", "rare"], fontsize=10)
bar_ax.set_facecolor(BG)
bar_ax.axhline(W_MAX, color=MUTED, lw=1.3, dashes=(6, 4))
bar_ax.grid(True, axis="y", color="#444444", linestyle="--", linewidth=0.4,
            alpha=0.45)
bar_ax.set_title("the three weights, right now", fontsize=11.5, pad=10)

caption = fig.text(0.5, 0.90, "", ha="center", va="center", fontsize=13.5,
                   color=FG)
rule = fig.text(0.5, 0.035, r"Hebb's rule:  $w \leftarrow w + \eta\,x\,y$"
                            r"   (it only ever adds)",
                ha="center", va="center", fontsize=11.5, color=MUTED)


def caption_for(t):
    if t < saturated_at[0]:
        return "Every co-firing makes a connection stronger. Nothing weakens it."
    if t < saturated_at[1]:
        return "The busiest connection hits the ceiling and stops carrying information."
    if t < saturated_at[2]:
        return "The next one follows. The gap between them is closing."
    return "All three are maxed out and identical. The learning has erased itself."


# ---- Frame schedule ------------------------------------------------------
# The curves are sampled every other step, but the caption changes at the three
# saturation points, and two of those spans are under a second long at that
# sampling rate. So freeze on the frame where each new line appears: text the
# reader is expected to read gets at least CAPTION_HOLD frames to itself.
STRIDE = 2
CAPTION_HOLD = 54           # 1.8 s

frames = []
previous_line = None
for t in range(0, T + 1, STRIDE):
    line = caption_for(t)          # not `caption`: that name is the Text artist
    if line != previous_line:
        frames += [t] * CAPTION_HOLD
        previous_line = line
    frames.append(t)
frames += [T] * 90          # ~3 s hold on the flat, useless end state


def init():
    for ln, hd in zip(lines, heads):
        ln.set_data([], [])
        hd.set_data([], [])
    return (*lines, *heads, *bars, caption)


def update(t):
    xs = np.arange(t + 1)
    for i in range(3):
        lines[i].set_data(xs, weights[i, : t + 1])
        heads[i].set_data([t], [weights[i, t]])
        bars[i].set_height(weights[i, t])
        # Once a connection saturates it stops being informative: grey it out.
        faded = weights[i, t] >= W_MAX - 1e-9
        bars[i].set_alpha(0.45 if faded else 1.0)
    caption.set_text(caption_for(t))
    return (*lines, *heads, *bars, caption)


ani = FuncAnimation(fig, update, frames=frames, init_func=init,
                    interval=1000 / 30, blit=False)

out = Path(__file__).resolve().parent.parent / "images" / "hebbian_instability.mp4"
ani.save(out, writer="ffmpeg", fps=30, dpi=140,
         savefig_kwargs={"facecolor": BG},
         extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                     "-crf", "20", "-preset", "slow"])
print(f"wrote {out}")

# Run (from the repo root):
#   uv run python blog/published/neural-networks-history-of-an-idea/_src/hebbian_instability.py
# Output: blog/published/neural-networks-history-of-an-idea/images/hebbian_instability.mp4
# Embed as a numbered figure with:
#   ::: {#fig-hebbian_instability}
#   {{< video images/hebbian_instability.mp4 >}}
#
#   Caption text.
#   :::

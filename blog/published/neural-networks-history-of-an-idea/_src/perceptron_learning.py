# Runs via the root uv environment with uv run (see command at the bottom).
"""Rosenblatt's rule, running.

Paced in three acts, because a full run is far too many example-checks to watch
one at a time. Act one steps slowly through the first few checks, long enough to
read what the rule is doing on each: silent when it is already right, one step
along the input when it is wrong. Act two then lets the remaining checks run
quickly under a caption that does NOT change, so nothing has to be read while
the boundary settles. Act three holds on the converged state.

The rule for the pacing: any text the reader has to read gets at least 1.8
seconds on screen, and text is never allowed to change during the fast act.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import ACCENT, BG, FG, FIRE, GOOD, MUTED, QUIET, use_dark_theme

use_dark_theme()

ETA = 0.35
MAX_EPOCHS = 12
FPS = 30

# Act one runs until the reader has seen this many of each outcome slowly.
TEACH_CORRECT = 2
TEACH_WRONG = 3
FRAMES_WRONG = 78        # 2.6 s: there is an equation to read
FRAMES_CORRECT = 56      # 1.9 s: shorter line, less to read
FRAMES_FAST = 2          # act two, where nothing needs reading
FRAMES_HOLD = 165        # 5.5 s on the final state

# ---- The data: two sensors, two verdicts ----------------------------------
rng = np.random.default_rng(3)
storm = rng.normal([2.35, 2.15], 0.42, size=(12, 2))     # label 1
calm = rng.normal([0.85, 0.95], 0.42, size=(12, 2))      # label 0
X = np.vstack([storm, calm])
d = np.hstack([np.ones(len(storm)), np.zeros(len(calm))])

order = rng.permutation(len(X))
X, d = X[order], d[order]


def predict(w, b, x):
    return 1.0 if float(w @ x) + b >= 0.0 else 0.0


# ---- Replay the algorithm once, recording every visit ---------------------
w = np.array([-1.10, 0.85])
b = 0.15
visits = []          # (index, w_before, b_before, wrong, mistakes_so_far, epoch)
mistakes_total = 0
converged = False
for epoch in range(1, MAX_EPOCHS + 1):
    epoch_mistakes = 0
    for i in range(len(X)):
        y = predict(w, b, X[i])
        wrong = y != d[i]
        w_before, b_before = w.copy(), b
        if wrong:
            err = d[i] - y
            w = w + ETA * err * X[i]
            b = b + ETA * err
            epoch_mistakes += 1
            mistakes_total += 1
        visits.append((i, w_before, b_before, wrong, mistakes_total, epoch))
    if epoch_mistakes == 0:
        converged = True
        break
w_final, b_final = w.copy(), b

# ---- Work out where act one ends ------------------------------------------
seen_correct = seen_wrong = 0
teach_end = 0
for k, v in enumerate(visits):
    if v[3]:
        seen_wrong += 1
    else:
        seen_correct += 1
    if seen_correct >= TEACH_CORRECT and seen_wrong >= TEACH_WRONG:
        teach_end = k + 1
        break

# ---- Frame schedule: (visit index, act) -----------------------------------
frames = []
for k in range(teach_end):
    hold = FRAMES_WRONG if visits[k][3] else FRAMES_CORRECT
    frames += [(k, "teach")] * hold
for k in range(teach_end, len(visits)):
    frames += [(k, "fast")] * FRAMES_FAST
frames += [(len(visits) - 1, "done")] * FRAMES_HOLD

print(f"converged={converged}: {len(visits)} visits, {mistakes_total} updates, "
      f"{epoch} passes | act one = {teach_end} visits, "
      f"total {len(frames) / FPS:.1f} s")

# ---- The figure -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 6.4), dpi=180)
fig.subplots_adjust(left=0.10, right=0.97, top=0.76, bottom=0.10)

# The rule itself never moves and never changes, so it never needs re-reading.
fig.text(0.5, 0.955, "Rosenblatt's rule", ha="center", va="center",
         fontsize=15, color=FG)
fig.text(0.5, 0.895,
         r"$\mathbf{w} \leftarrow \mathbf{w} + \eta\,"
         r"\left(d(\mathbf{x}) - y(\mathbf{x})\right)\,\mathbf{x}$",
         ha="center", va="center", fontsize=15, color=ACCENT)

PAD = 1.1
xlim = (X[:, 0].min() - PAD, X[:, 0].max() + PAD)
ylim = (X[:, 1].min() - PAD, X[:, 1].max() + PAD)
gx, gy = np.meshgrid(np.linspace(*xlim, 300), np.linspace(*ylim, 300))


def ring_colour(wrong):
    return FIRE if wrong else GOOD


def draw(frame):
    k, act = frame
    idx, w_b, b_b, wrong, mistakes, epoch = visits[k]
    w_now, b_now = (w_final, b_final) if act == "done" else (w_b, b_b)

    ax.clear()
    ax.set_facecolor(BG)

    z = w_now[0] * gx + w_now[1] * gy + b_now
    ax.contourf(gx, gy, z, levels=[0, 1e9], colors=[ACCENT], alpha=0.16)
    ax.contour(gx, gy, z, levels=[0], colors=[ACCENT], linewidths=2.4)

    fires = d == 1
    ax.scatter(X[fires, 0], X[fires, 1], s=95, color=FIRE, edgecolors=FG,
               linewidths=0.9, zorder=4, label="storm  ($d = 1$)")
    ax.scatter(X[~fires, 0], X[~fires, 1], s=95, color=QUIET, edgecolors=FG,
               linewidths=0.9, zorder=4, label="calm  ($d = 0$)")

    if act != "done":
        ax.scatter([X[idx, 0]], [X[idx, 1]], s=340, facecolors="none",
                   edgecolors=ring_colour(wrong), linewidths=2.6, zorder=5)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, color="#444444", linestyle="--", linewidth=0.4, alpha=0.4)
    ax.set_xlabel(r"$x_1$  (flash brightness)", fontsize=12)
    ax.set_ylabel(r"$x_2$  (rumble loudness)", fontsize=12)
    ax.legend(loc="lower right", fontsize=10.5, frameon=False)

    # The verdict line changes only during act one, where there is time to read
    # it, and during the final hold. Act two keeps one fixed sentence.
    if act == "teach":
        if wrong:
            verdict = "Wrong. Step along this input, and the boundary swings."
            colour = FIRE
        else:
            verdict = "Correct. The update term is zero, so nothing moves."
            colour = GOOD
        status = f"pass {epoch}   |   updates so far: {mistakes}"
    elif act == "fast":
        verdict = "Now the rest of the run, at speed."
        colour = MUTED
        status = f"pass {epoch}   |   updates so far: {mistakes}"
    else:
        verdict = "A full pass with no mistakes. The algorithm stops."
        colour = GOOD
        status = (f"{mistakes} updates in total   |   any separating line "
                  "would have done")

    ax.set_title(verdict, fontsize=13.5, pad=26, color=colour)
    ax.text(0.5, 1.028, status, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=10.5, color=MUTED)
    return ()


ani = FuncAnimation(fig, draw, frames=frames, interval=1000 / FPS, blit=False)

out = Path(__file__).resolve().parent.parent / "images" / "perceptron_learning.mp4"
ani.save(out, writer="ffmpeg", fps=FPS, dpi=180,
         savefig_kwargs={"facecolor": BG},
         extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                     "-crf", "20", "-preset", "slow"])
print(f"wrote {out}")

# Run (from the repo root):
#   uv run python blog/published/neural-networks-history-of-an-idea/_src/perceptron_learning.py
# Output: blog/published/neural-networks-history-of-an-idea/images/perceptron_learning.mp4
# Embed as a numbered figure with:
#   ::: {#fig-perceptron_learning}
#   {{< video images/perceptron_learning.mp4 >}}
#
#   Caption text.
#   :::

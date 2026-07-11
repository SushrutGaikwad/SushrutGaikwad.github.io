# Runs in the root uv environment: uv run python blog/published/exponential-family-and-generalized-linear-models/_src/softmax_logits_matplotlib.py
#
# Three-panel STATIC PNG showing the softmax function turning raw class scores
# (logits) into a probability distribution, in two moves:
#   1. exponentiate  -> everything becomes positive, order preserved
#   2. normalize     -> the positive numbers are scaled to sum to 1
# The largest logit ends up with the largest probability. This makes the
# "exp then normalize" recipe concrete.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("dark_background")
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

classes = ["1", "2", "3", "4"]
logits = np.array([2.0, -0.5, 1.0, -1.5])          # t_c = theta_c^T x
exps = np.exp(logits)
probs = exps / exps.sum()

bar_colors = ["#e67e22", "#5dade2", "#1abc9c", "#af7ac5"]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), dpi=400)

# panel 1: raw logits
ax = axes[0]
ax.bar(classes, logits, color=bar_colors, edgecolor="#222222", linewidth=1.0)
ax.axhline(0, color="#888888", lw=0.8)
ax.set_title(r"Raw logits  $t_c = \boldsymbol{\theta}_c^\top \mathbf{x}$", fontsize=12.5, pad=10)
ax.set_ylabel("value", fontsize=11)
ax.set_ylim(-2.2, 2.6)
for i, v in enumerate(logits):
    ax.text(i, v + (0.12 if v >= 0 else -0.22), f"{v:+.1f}", ha="center",
            fontsize=10, color="#e6e6e6")
ax.text(0.5, -1.9, "can be negative", fontsize=9.5, color="#bbbbbb",
        style="italic", ha="center")

# panel 2: exponentiated
ax = axes[1]
ax.bar(classes, exps, color=bar_colors, edgecolor="#222222", linewidth=1.0)
ax.set_title(r"Exponentiate  $e^{t_c}$", fontsize=12.5, pad=10)
ax.set_ylabel("value", fontsize=11)
ax.set_ylim(0, exps.max() * 1.25)
for i, v in enumerate(exps):
    ax.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=10, color="#e6e6e6")
ax.text(1.5, exps.max() * 1.12, "all positive, order kept", fontsize=9.5,
        color="#bbbbbb", style="italic", ha="center")

# panel 3: normalized probabilities
ax = axes[2]
ax.bar(classes, probs, color=bar_colors, edgecolor="#222222", linewidth=1.0)
ax.set_title(r"Normalize  $\phi_c = \dfrac{e^{t_c}}{\sum_{c'} e^{t_{c'}}}$",
             fontsize=12.5, pad=10)
ax.set_ylabel("probability", fontsize=11)
ax.set_ylim(0, 0.75)
for i, v in enumerate(probs):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10, color="#e6e6e6")
ax.text(1.5, 0.68, r"sums to $1$", fontsize=9.5, color="#bbbbbb",
        style="italic", ha="center")

for ax in axes:
    ax.set_xlabel("class $c$", fontsize=11)
    ax.grid(True, axis="y", color="#3a3a3a", linestyle="--", linewidth=0.4, alpha=0.5)

plt.tight_layout()
_out = Path(__file__).resolve().parent.parent / "images" / "softmax_logits.png"
plt.savefig(_out, dpi=400, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print(f"wrote {_out}")

# Output: images/softmax_logits.png
# Embed:
#   ![Caption](images/softmax_logits.png){#fig-softmax_logits fig-align="center" width=100%}

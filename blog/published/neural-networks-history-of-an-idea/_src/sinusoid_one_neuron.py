# Runs via the root uv environment with uv run (see command at the bottom).
"""One sinusoidal unit is enough for cos(2x), because cos(2x) IS a sine.

The universal-approximation construction needs an unbounded pile of threshold
units to build cos(2x) out of little pulses. Swap the activation for a sine and
the same function needs exactly one unit, with weight 2 and bias pi/2. The
number of units you need is a fact about the pairing of activation and target,
not about the target alone.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import ACCENT, BG, FG, MUTED, QUIET, use_dark_theme

use_dark_theme()

x = np.linspace(-np.pi, np.pi, 1000)
target = np.cos(2 * x)
one_unit = np.sin(2 * x + np.pi / 2)

fig, ax = plt.subplots(figsize=(9.0, 4.4))

ax.plot(x, target, color=ACCENT, lw=6.0, alpha=0.45,
        label=r"target: $y = \cos(2x)$")
ax.plot(x, one_unit, color=QUIET, lw=2.0, dashes=(6, 4),
        label=r"one unit: $y = \sin\left(w x + b\right),\ w = 2,\ b = \pi/2$")

ax.axhline(0, color=MUTED, lw=0.8, alpha=0.6)
ax.axvline(0, color=MUTED, lw=0.8, alpha=0.6)
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-1.45, 1.75)
ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
ax.set_facecolor(BG)
ax.grid(True, color="#444444", linestyle="--", linewidth=0.4, alpha=0.45)
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel(r"$y$", fontsize=12)
ax.set_title(r"The two curves are the same curve", fontsize=13, pad=12)
ax.legend(loc="upper center", fontsize=10.5, frameon=False, ncol=1)

fig.tight_layout()

out = Path(__file__).resolve().parent.parent / "images" / "sinusoid_one_neuron.png"
fig.savefig(out, dpi=200, facecolor=BG)
print(f"wrote {out}")

# Run (from the repo root):
#   uv run python blog/published/neural-networks-history-of-an-idea/_src/sinusoid_one_neuron.py
# Output: blog/published/neural-networks-history-of-an-idea/images/sinusoid_one_neuron.png
# Embed with:
#   ![...](images/sinusoid_one_neuron.png){#fig-sinusoid_one_neuron fig-align="center" width=80%}

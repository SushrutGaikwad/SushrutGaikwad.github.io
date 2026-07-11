"""Stability regions for Post 11.

Left: difference equations are stable when all eigenvalues lie inside the unit
circle. Right: differential equations are stable when all eigenvalues lie in
the left half of the complex plane.

Runs via the root uv environment:
  uv run python blog/published/diagonalization-powers-and-differential-equations/_src/stability_regions.py
Output: blog/published/diagonalization-powers-and-differential-equations/images/stability_regions.svg
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")
plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})

LIGHT = "#e6e6e6"
DARK = "#222222"
BLUE = "#5dade2"
RED = "#ec7063"
GREEN = "#10a37f"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.2))
for ax in (axL, axR):
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    ax.set_aspect("equal")
    ax.axhline(0, color="#555555", lw=0.8)
    ax.axvline(0, color="#555555", lw=0.8)
    ax.tick_params(colors=LIGHT)
    for s in ax.spines.values():
        s.set_color(LIGHT)
    ax.set_xlabel(r"$\mathrm{Re}(\lambda)$", color=LIGHT, fontsize=12)
    ax.set_ylabel(r"$\mathrm{Im}(\lambda)$", color=LIGHT, fontsize=12)

# Left: unit disk.
th = np.linspace(0, 2 * np.pi, 300)
axL.fill(np.cos(th), np.sin(th), color=GREEN, alpha=0.22)
axL.plot(np.cos(th), np.sin(th), color=GREEN, lw=2)
axL.scatter([0.4, -0.6], [0.3, -0.4], color=BLUE, s=55, zorder=5, label="stable")
axL.scatter([1.4], [0.6], color=RED, s=55, zorder=5, label="unstable")
axL.set_xlim(-2, 2)
axL.set_ylim(-2, 2)
axL.set_title(r"Difference eq.: stable if $|\lambda|<1$", color=LIGHT, fontsize=13)
axL.legend(facecolor="#333333", edgecolor=LIGHT, labelcolor=LIGHT, fontsize=10, loc="upper left")

# Right: left half-plane.
axR.axvspan(-2, 0, color=GREEN, alpha=0.22)
axR.axvline(0, color=GREEN, lw=2)
axR.scatter([-1.2, -0.5], [0.6, -0.9], color=BLUE, s=55, zorder=5, label="stable")
axR.scatter([0.8], [0.5], color=RED, s=55, zorder=5, label="unstable")
axR.set_xlim(-2, 2)
axR.set_ylim(-2, 2)
axR.set_title(r"Differential eq.: stable if $\mathrm{Re}(\lambda)<0$", color=LIGHT, fontsize=13)
axR.legend(facecolor="#333333", edgecolor=LIGHT, labelcolor=LIGHT, fontsize=10, loc="upper left")

out = Path(__file__).resolve().parent.parent / "images" / "stability_regions.svg"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=DARK, bbox_inches="tight")
print(f"saved {out}")

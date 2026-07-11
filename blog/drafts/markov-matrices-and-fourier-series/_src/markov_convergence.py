"""Markov convergence for Post 12.

Iterates u_{k+1} = A u_k with the migration matrix A = [[0.9,0.2],[0.1,0.8]]
starting from everyone in Massachusetts, and shows convergence to the
steady-state split of about 667 / 333 (the 2:1 eigenvector of lambda = 1).

Runs via the root uv environment:
  uv run python blog/published/markov-matrices-and-fourier-series/_src/markov_convergence.py
Output: blog/published/markov-matrices-and-fourier-series/images/markov_convergence.svg
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")
plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})

LIGHT = "#e6e6e6"
DARK = "#222222"
BLUE = "#5dade2"
ORANGE = "#f5b041"

A = np.array([[0.9, 0.2], [0.1, 0.8]])
steps = 26
u = np.array([0.0, 1000.0])
hist = [u.copy()]
for _ in range(steps):
    u = A @ u
    hist.append(u.copy())
hist = np.array(hist)
k = np.arange(steps + 1)

fig, ax = plt.subplots(figsize=(7.5, 5))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)

ax.plot(k, hist[:, 0], "-o", color=BLUE, ms=4, lw=2, label="California")
ax.plot(k, hist[:, 1], "-o", color=ORANGE, ms=4, lw=2, label="Massachusetts")
ax.axhline(2000.0 / 3.0, color=BLUE, ls="--", lw=1.2, alpha=0.7)
ax.axhline(1000.0 / 3.0, color=ORANGE, ls="--", lw=1.2, alpha=0.7)
ax.annotate(r"steady state $\approx 667$", (steps, 2000 / 3), color=BLUE,
            fontsize=10, va="bottom", ha="right")
ax.annotate(r"steady state $\approx 333$", (steps, 1000 / 3), color=ORANGE,
            fontsize=10, va="bottom", ha="right")

ax.set_xlabel(r"year $k$", color=LIGHT, fontsize=13)
ax.set_ylabel(r"population", color=LIGHT, fontsize=13)
ax.set_title(r"Migration settles to the $2:1$ steady state", color=LIGHT, fontsize=13)
ax.tick_params(colors=LIGHT)
for s in ax.spines.values():
    s.set_color(LIGHT)
ax.grid(True, color="#3a3a3a", lw=0.6)
ax.legend(facecolor="#333333", edgecolor=LIGHT, labelcolor=LIGHT, fontsize=12, loc="center right")

out = Path(__file__).resolve().parent.parent / "images" / "markov_convergence.svg"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=DARK, bbox_inches="tight")
print(f"saved {out}")

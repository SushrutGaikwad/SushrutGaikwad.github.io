"""Fibonacci growth vs the dominant eigenvalue for Post 11.

Plots the Fibonacci numbers against the curve phi^k / sqrt(5), where
phi = (1 + sqrt 5)/2 is the dominant eigenvalue of [[1,1],[1,0]].

Runs via the root uv environment:
  uv run python blog/published/diagonalization-powers-and-differential-equations/_src/fibonacci.py
Output: blog/published/diagonalization-powers-and-differential-equations/images/fibonacci.svg
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

phi = (1 + np.sqrt(5)) / 2
fib = [0, 1]
for _ in range(13):
    fib.append(fib[-1] + fib[-2])
fib = np.array(fib)
k = np.arange(len(fib))

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)

kk = np.linspace(0, len(fib) - 1, 300)
ax.plot(kk, phi**kk / np.sqrt(5), color=ORANGE, lw=2.5,
        label=r"$\lambda_1^{\,k}/\sqrt{5}$,  $\lambda_1=\frac{1+\sqrt{5}}{2}$")
ax.scatter(k, fib, color=BLUE, s=45, zorder=5, label=r"Fibonacci $F_k$")

ax.set_xlabel(r"$k$", color=LIGHT, fontsize=13)
ax.set_ylabel(r"value", color=LIGHT, fontsize=13)
ax.set_title(r"Fibonacci numbers grow like the dominant eigenvalue", color=LIGHT, fontsize=13)
ax.tick_params(colors=LIGHT)
for s in ax.spines.values():
    s.set_color(LIGHT)
ax.grid(True, color="#3a3a3a", lw=0.6)
ax.legend(facecolor="#333333", edgecolor=LIGHT, labelcolor=LIGHT, fontsize=12, loc="upper left")

out = Path(__file__).resolve().parent.parent / "images" / "fibonacci.svg"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=DARK, bbox_inches="tight")
print(f"saved {out}")

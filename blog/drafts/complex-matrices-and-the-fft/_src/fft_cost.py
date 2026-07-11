"""FFT cost comparison for Post 14: naive n^2 versus FFT (n/2) log2 n.

Runs via the root uv environment:
  uv run python blog/published/complex-matrices-and-the-fft/_src/fft_cost.py
Output: blog/published/complex-matrices-and-the-fft/images/fft_cost.svg
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")
plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})

DARK = "#222222"
LIGHT = "#e6e6e6"

n = 2.0 ** np.arange(1, 13)  # 2, 4, ..., 4096
naive = n**2
fft = (n / 2.0) * np.log2(n)

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)

ax.loglog(n, naive, "o-", color="#ec7063", lw=2.3, ms=5, label=r"naive transform: $n^2$")
ax.loglog(n, fft, "o-", color="#5dade2", lw=2.3, ms=5, label=r"FFT: $\frac{n}{2}\log_2 n$")

ax.set_xlabel(r"$n$  (transform size)", color=LIGHT, fontsize=13)
ax.set_ylabel("multiplications", color=LIGHT, fontsize=13)
ax.tick_params(colors=LIGHT)
for spine in ax.spines.values():
    spine.set_color(LIGHT)
ax.grid(True, which="both", color="#3a3a3a", lw=0.5)
ax.legend(facecolor="#333333", edgecolor=LIGHT, labelcolor=LIGHT, fontsize=12, loc="upper left")

out = Path(__file__).resolve().parent.parent / "images" / "fft_cost.svg"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=DARK, bbox_inches="tight")
print(f"saved {out}")

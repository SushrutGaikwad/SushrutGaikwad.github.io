# Runs via the root uv environment: uv run python blog/published/generative-learning-gda-and-naive-bayes/_src/laplace_smoothing.py
#
# STATIC PNG: a coin lands heads n times in a row. The maximum likelihood
# estimate of P(heads) is stuck at exactly 1 for every n >= 1 (and P(tails) is
# exactly 0), which is an over-confident claim from finite data. Laplace
# smoothing starts each count at 1, so P(heads) = (n+1)/(n+2) climbs toward 1
# but never reaches it, and P(tails) = 1/(n+2) shrinks toward 0 but never
# reaches it.

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---- Dark theme matching the darkly Quarto theme ----
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

n = np.arange(1, 21)
mle_heads = np.ones_like(n, dtype=float)          # n / n = 1
laplace_heads = (n + 1) / (n + 2)
laplace_tails = 1.0 / (n + 2)

fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=300)

ax.plot(n, mle_heads, "o-", color="#e74c3c", lw=2.0, markersize=5,
        label=r"MLE  $P(\mathrm{heads}) = n/n = 1$")
ax.plot(n, laplace_heads, "o-", color="#2ecc71", lw=2.0, markersize=5,
        label=r"Laplace  $P(\mathrm{heads}) = \frac{n+1}{n+2}$")
ax.plot(n, laplace_tails, "o-", color="#f1c40f", lw=2.0, markersize=5,
        label=r"Laplace  $P(\mathrm{tails}) = \frac{1}{n+2}$")

ax.axhline(1.0, color="#666666", lw=0.8, ls=":", alpha=0.7)
ax.axhline(0.0, color="#666666", lw=0.8, ls=":", alpha=0.7)

ax.set_xlabel(r"$n$ = number of consecutive heads observed", fontsize=12)
ax.set_ylabel(r"estimated probability", fontsize=12)
ax.set_xlim(0.5, 20.5)
ax.set_ylim(-0.05, 1.08)
ax.set_xticks(np.arange(2, 21, 2))
ax.grid(True, color="#3a3a3a", linestyle="--", linewidth=0.4, alpha=0.5)
ax.legend(loc="center right", framealpha=0.4, fontsize=11)

plt.tight_layout()
_out = Path(__file__).resolve().parent.parent / "images" / "laplace_smoothing.png"
plt.savefig(_out, dpi=300, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print("saved", _out)

# Embed in Quarto:
#   ![Caption](images/laplace_smoothing.png){#fig-laplace_smoothing fig-align="center" width=80%}

# Runs in the root uv environment: uv run python blog/published/exponential-family-and-generalized-linear-models/_src/glm_data_generating_matplotlib.py
#
# Two-panel STATIC PNG showing the GLM "data-generating story" for the two
# GLMs the reader already knows:
#   Left  (regression):     y is sampled from a Gaussian centered ON the line
#                           eta = theta^T x = mu. Little sideways bells stand
#                           up at a few x positions; one sampled y per bell.
#   Right (classification): y is sampled from a Bernoulli whose mean phi = g(eta)
#                           is read off the sigmoid curve. Points land at 0 or 1.
#
# This is the lecture's signature picture: a GLM is a recipe for GENERATING y
# from x, one exponential-family distribution per example.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

rng = np.random.default_rng(7)

line_color = "#e6e6e6"
bell_color = "#1abc9c"
sample_color = "#f1c40f"
data_color = "#5dade2"
curve_color = "#1abc9c"
pos_color = "#e67e22"
neg_color = "#5dade2"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=400)
for ax in (axL, axR):
    ax.set_facecolor("#222222")

# =====================================================================
# LEFT PANEL: regression as sampling from a Gaussian centered on the line
# =====================================================================
a, b, sigma = 0.5, 0.55, 0.68            # true line mu(x) = a + b x, noise sigma
x_line = np.linspace(0, 10, 200)
mu_line = a + b * x_line

# faint background cloud of observed data
x_bg = rng.uniform(0.3, 9.7, 60)
y_bg = a + b * x_bg + rng.normal(0, sigma, x_bg.size)
axL.scatter(x_bg, y_bg, s=16, color=data_color, alpha=0.35, zorder=1)

# the true (hidden) line eta = theta^T x = mu
axL.plot(x_line, mu_line, color=line_color, lw=2.2, zorder=3,
         label=r"$\eta = \boldsymbol{\theta}^\top x = \mu$")

# highlighted Gaussians with one sampled point each
bell_scale = 1.15
for x0 in [2.0, 4.5, 7.0, 9.0]:
    mu0 = a + b * x0
    ys = np.linspace(mu0 - 3 * sigma, mu0 + 3 * sigma, 200)
    dens = np.exp(-0.5 * ((ys - mu0) / sigma) ** 2)
    # vertical baseline at x0
    axL.plot([x0, x0], [ys[0], ys[-1]], color="#888888", lw=0.8, zorder=2)
    # sideways bell bulging to the right
    axL.plot(x0 + bell_scale * dens, ys, color=bell_color, lw=1.8, zorder=4)
    axL.fill_betweenx(ys, x0, x0 + bell_scale * dens, color=bell_color,
                      alpha=0.15, zorder=2)
    # one sampled y from this Gaussian
    y0 = mu0 + rng.normal(0, sigma)
    axL.scatter([x0], [y0], s=70, color=sample_color, edgecolors="#222222",
                linewidths=0.8, zorder=5)

axL.set_xlim(0, 11.2)
axL.set_ylim(-1.0, 7.6)
axL.set_xlabel(r"$x$  (feature)", fontsize=12)
axL.set_ylabel(r"$y$  (continuous output)", fontsize=12)
axL.set_title(r"Regression: sample $y$ from a Gaussian on the line",
              fontsize=12.5, pad=10)
axL.grid(True, color="#3a3a3a", linestyle="--", linewidth=0.4, alpha=0.5)
axL.legend(loc="upper left", framealpha=0.35, fontsize=10)

# =====================================================================
# RIGHT PANEL: classification as sampling from a Bernoulli under the curve
# =====================================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

c, d = -4.4, 1.0                         # eta(x) = c + d x
phi_line = sigmoid(c + d * x_line)

axR.plot(x_line, phi_line, color=curve_color, lw=2.4, zorder=3,
         label=r"$\phi = g(\eta) = \dfrac{1}{1 + e^{-\eta}}$")
axR.axhline(0.5, color="#95a5a6", ls="--", lw=0.8, alpha=0.6, zorder=1)

# sample data: y ~ Bernoulli(phi(x))
x_pts = rng.uniform(0.3, 9.7, 44)
phi_pts = sigmoid(c + d * x_pts)
y_pts = (rng.uniform(size=x_pts.size) < phi_pts).astype(float)
jit = rng.normal(0, 0.02, x_pts.size)
for xi, yi, ji in zip(x_pts, y_pts, jit):
    col = pos_color if yi == 1 else neg_color
    axR.scatter([xi], [yi + ji], s=42, color=col, alpha=0.85,
                edgecolors="#222222", linewidths=0.5, zorder=4)

# annotate phi at a couple of x positions (offsets chosen to avoid the data clusters)
phi_marks = [(2.5, 0.15, 0.09, "left"), (4.4, 0.18, 0.07, "left"), (7.0, -0.22, -0.14, "right")]
for x0, dx, dy, ha in phi_marks:
    p0 = sigmoid(c + d * x0)
    axR.plot([x0, x0], [0, p0], color=sample_color, ls=":", lw=1.1,
             alpha=0.8, zorder=2)
    axR.scatter([x0], [p0], s=70, facecolors="none", edgecolors=sample_color,
                linewidths=2.0, zorder=5)
    axR.annotate(rf"$\phi = {p0:.2f}$", xy=(x0, p0), xytext=(x0 + dx, p0 + dy),
                 fontsize=9, color=sample_color, ha=ha)

axR.set_xlim(0, 10.2)
axR.set_ylim(-0.15, 1.25)
axR.set_yticks([0, 0.5, 1])
axR.set_xlabel(r"$x$  (feature)", fontsize=12)
axR.set_ylabel(r"$y \in \{0, 1\}$   or   $\phi$", fontsize=12)
axR.set_title(r"Classification: sample $y$ from a Bernoulli under the curve",
              fontsize=12.5, pad=10)
axR.grid(True, color="#3a3a3a", linestyle="--", linewidth=0.4, alpha=0.5)
axR.legend(loc="upper left", framealpha=0.35, fontsize=10)

plt.tight_layout()
_out = Path(__file__).resolve().parent.parent / "images" / "glm_data_generating.png"
plt.savefig(_out, dpi=400, facecolor="#222222", bbox_inches="tight")
plt.close(fig)
print(f"wrote {_out}")

# Output: images/glm_data_generating.png
# Embed:
#   ![Caption](images/glm_data_generating.png){#fig-glm_data_generating fig-align="center" width=100%}

# Runs via the root uv environment: uv run python blog/published/classification-perceptron-logistic-regression/_src/boundary_comparison_matplotlib.py
#
# Side-by-side MATPLOTLIB ANIMATION contrasting the two boundaries the post
# builds, on the SAME 2D tumor dataset (size and cell-shape irregularity):
#
#   Left  (perceptron):          the boundary as a HARD cut. Two flat
#                                half-spaces (predict benign / predict
#                                malignant). Misclassified points get red
#                                rings. The boundary settles into a separating
#                                orientation.
#   Right (logistic regression): the SAME kind of straight boundary, but
#                                wrapped in a smooth PROBABILITY FIELD. The
#                                field starts flat (everything near P = 0.5)
#                                and sharpens, with parallel level sets at
#                                P = 0.25, 0.5, 0.75.
#
# Both destinations are the real fitted boundaries: the left target comes from
# running the perceptron update rule to convergence, the right from batch
# gradient ascent on the log-likelihood. The two panels then "form" toward
# those destinations over a shared window so they finish together:
#   - A hard boundary is scale-invariant (multiplying theta by a positive
#     number does not move the z = 0 line), so the perceptron boundary cannot
#     animate by sharpening; it has to MOVE. We sweep it geometrically from a
#     sensible 'classify by irregularity alone' start (a horizontal line) into
#     its final separating orientation. The detailed per-step update mechanics
#     are shown separately in the perceptron_full_training animation; here we
#     only need the boundary to settle.
#   - The logistic field DOES sharpen: scaling theta from 0 to its fitted value
#     keeps the final boundary fixed while confidence grows around it. A small
#     L2 penalty during the fit keeps the weights finite, so the field settles
#     into a stable gradient instead of collapsing to a hard step (logistic
#     regression's weights would otherwise diverge on perfectly separable data).
#
# Optimization is done in standardized feature space for numerical stability;
# everything is drawn back in original (cm / irregularity) coordinates.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

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

BENIGN = "#3498db"
MALIGNANT = "#e67e22"
BOUNDARY = "#ecf0f1"
MISS = "#e74c3c"
MIDFIELD = "#2b2b2b"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# =====================================================================
# DATASET (identical generation to boundary_1d_vs_2d_matplotlib.py)
# =====================================================================
rng = np.random.default_rng(seed=3)
n_per = 18
benign_center = np.array([1.4, 3.0])
malignant_center = np.array([4.3, 7.0])
spread = np.array([0.55, 1.05])

Xb = benign_center + spread * rng.standard_normal((n_per, 2))
Xm = malignant_center + spread * rng.standard_normal((n_per, 2))
X = np.vstack([Xb, Xm])
y = np.concatenate([np.zeros(n_per), np.ones(n_per)])

mu = X.mean(axis=0)
sd = X.std(axis=0)
Xs = (X - mu) / sd
Xaug = np.hstack([np.ones((len(Xs), 1)), Xs])    # [1, x1_std, x2_std]


def z_on_grid(theta, XX, YY):
    XXs = (XX - mu[0]) / sd[0]
    YYs = (YY - mu[1]) / sd[1]
    return theta[0] + theta[1] * XXs + theta[2] * YYs


def z_on_data(theta):
    return Xaug @ theta


# =====================================================================
# FINAL FITS (run the real algorithms to get the destination boundaries)
# =====================================================================
def perceptron_final():
    theta = np.zeros(3)
    order = rng.permutation(len(Xaug))
    alpha = 0.5
    for _ in range(200):
        updated = False
        for i in order:
            pred = 1.0 if Xaug[i] @ theta >= 0 else 0.0
            if pred != y[i]:
                theta = theta + alpha * (y[i] - pred) * Xaug[i]
                updated = True
        if not updated:
            break
    if np.mean(z_on_data(theta)[y == 1]) < 0:    # orient: + side = malignant
        theta = -theta
    return theta


def logistic_final():
    theta = np.zeros(3)
    alpha, lam, iters, n = 0.6, 0.01, 600, len(Xaug)
    reg = np.array([0.0, 1.0, 1.0])
    for _ in range(iters):
        p = sigmoid(Xaug @ theta)
        theta = theta + alpha * (Xaug.T @ (y - p) / n - lam * reg * theta)
    return theta


t_perc = perceptron_final()
t_logi = logistic_final()
print(f"perceptron final misclassified: "
      f"{int(((z_on_data(t_perc) >= 0).astype(float) != y).sum())}")
print(f"logistic final theta (std): {t_logi}")


def smoothstep(f):
    f = min(max(f, 0.0), 1.0)
    return f * f * (3 - 2 * f)


# Perceptron boundary: sweep from a horizontal 'classify by irregularity'
# start into the real separating orientation.
nf = t_perc[1:]
norm_nf = np.linalg.norm(nf)
a_f = np.arctan2(nf[1], nf[0])
rho_f = t_perc[0] / norm_nf
a_s = np.pi / 2.0          # horizontal boundary (normal points up in x2)
rho_s = 0.0


def perc_theta(frac):
    s = smoothstep(frac)
    a = a_s + (a_f - a_s) * s
    rho = rho_s + (rho_f - rho_s) * s
    return np.array([rho, np.cos(a), np.sin(a)])


def logi_theta(frac):
    return smoothstep(frac) * t_logi


# =====================================================================
# FIGURE
# =====================================================================
fig, (axP, axL) = plt.subplots(1, 2, figsize=(12.5, 6.0), dpi=170)
for ax in (axP, axL):
    ax.set_facecolor("#222222")
    ax.set_xlim(-0.5, 6.6)
    ax.set_ylim(0.0, 10.6)
    ax.set_xlabel(r"$x_1$ (tumor size, cm)", fontsize=12)
    ax.grid(True, color="#444444", ls="--", lw=0.4, alpha=0.4)
axP.set_ylabel(r"$x_2$ (cell-shape irregularity)", fontsize=12)
axP.set_title("Perceptron: a hard cut", fontsize=13)
axL.set_title("Logistic regression: a probability field", fontsize=13)
fig.subplots_adjust(left=0.06, right=0.92, top=0.93, bottom=0.155, wspace=0.18)

gx = np.linspace(-0.5, 6.6, 240)
gy = np.linspace(0.0, 10.6, 240)
GX, GY = np.meshgrid(gx, gy)

prob_cmap = LinearSegmentedColormap.from_list(
    "prob", [BENIGN, "#2b3a4a", MIDFIELD, "#4a3320", MALIGNANT])

for ax in (axP, axL):
    ax.scatter(Xb[:, 0], Xb[:, 1], c=BENIGN, s=70, edgecolors="#111111",
               linewidths=1.0, zorder=5, label=r"Benign ($y = 0$)")
    ax.scatter(Xm[:, 0], Xm[:, 1], c=MALIGNANT, s=70, edgecolors="#111111",
               linewidths=1.0, zorder=5, label=r"Malignant ($y = 1$)")
    ax.legend(loc="lower right", framealpha=0.5, fontsize=9)

sm = ScalarMappable(norm=Normalize(0, 1), cmap=prob_cmap)
cb = fig.colorbar(sm, ax=axL, fraction=0.046, pad=0.03)
cb.set_label(r"$P(\mathrm{malignant} \mid \mathbf{x})$", fontsize=10)
cb.ax.tick_params(labelsize=8)

art = {"pfill": None, "lfill": None, "pbound": None, "lbound": None,
       "llevels": None}
miss_ring, = axP.plot([], [], "o", mfc="none", mec=MISS, ms=15, mew=2.0,
                      zorder=6)
status = fig.text(0.49, 0.035, "", ha="center", va="bottom", fontsize=10.5,
                  color="#cfcfcf")

N_ACTIVE = 80
N_HOLD = 26
TOTAL = N_ACTIVE + N_HOLD


def clear(key):
    obj = art[key]
    if obj is None:
        return
    if hasattr(obj, "collections"):
        for c in obj.collections:
            c.remove()
    else:
        obj.remove()
    art[key] = None


def render(frame):
    frac = 1.0 if frame >= N_ACTIVE else frame / (N_ACTIVE - 1)
    tp = perc_theta(frac)
    tl = logi_theta(frac)
    zp = z_on_grid(tp, GX, GY)
    zl = z_on_grid(tl, GX, GY)

    for k in ("pfill", "lfill", "pbound", "lbound", "llevels"):
        clear(k)

    art["pfill"] = axP.contourf(GX, GY, (zp >= 0).astype(float),
                                levels=[-0.5, 0.5, 1.5],
                                colors=[BENIGN, MALIGNANT], alpha=0.16,
                                zorder=0)
    art["pbound"] = axP.contour(GX, GY, zp, levels=[0], colors=[BOUNDARY],
                                linewidths=2.5, zorder=3)

    art["lfill"] = axL.contourf(GX, GY, sigmoid(zl),
                                levels=np.linspace(0, 1, 41),
                                cmap=prob_cmap, alpha=0.85, zorder=0)
    art["llevels"] = axL.contour(GX, GY, sigmoid(zl), levels=[0.25, 0.75],
                                 colors=["#bdc3c7"], linewidths=1.0,
                                 linestyles="dashed", zorder=2)
    art["lbound"] = axL.contour(GX, GY, sigmoid(zl), levels=[0.5],
                                colors=[BOUNDARY], linewidths=2.5, zorder=3)

    preds = (z_on_data(tp) >= 0).astype(float)
    miss = preds != y
    if miss.any():
        miss_ring.set_data(X[miss, 0], X[miss, 1])
    else:
        miss_ring.set_data([], [])

    if frame >= N_ACTIVE:
        status.set_text("Boundary settled: 0 errors. Hard side on the left, "
                        "probability on the right.")
    else:
        status.set_text(f"forming...    perceptron misclassified: "
                        f"{int(miss.sum())}")
    return ()


anim = FuncAnimation(fig, render, frames=TOTAL, interval=70, blit=False)

from pathlib import Path
out = Path(__file__).resolve().parent.parent / "images" / "boundary_comparison.mp4"
anim.save(out, writer="ffmpeg", fps=18, dpi=170,
          savefig_kwargs={"facecolor": "#222222"})
plt.close(fig)
print(f"saved {out}")

# Recommended embed (MP4; the clip is ~6 seconds):
#   ::: {#fig-boundary_comparison}
#   {{< video images/boundary_comparison.mp4 >}}
#   Caption...
#   :::

# Setup (one-time, on Windows):
#   1. Install Python 3.9+
#   2. pip install numpy scipy matplotlib
#   (no LaTeX or FFmpeg needed for a static PNG)
#
# This script generates a STATIC PNG showing the probabilistic interpretation
# of logistic regression on the 1D tumor dataset. We:
#   1. Fit logistic regression on the tumor data (using scipy.optimize to
#      maximize the log-likelihood) to get learned theta = (theta_0, theta_1).
#   2. Plot the data points (benign at y=0, malignant at y=1) along the
#      tumor-size axis.
#   3. Overlay the fitted sigmoid curve h_theta(x) = g(theta_0 + theta_1 * x)
#      from y = 0 to y = 1.
#   4. Highlight three sample tumors (one benign, one near the boundary,
#      one malignant) showing their probabilities.
#
# This is much more elegant than the original 2D version because the
# sigmoid curve and the data live in the same coordinate system: the
# x-axis is tumor size (the input feature) and the y-axis spans the
# label/probability range [0, 1]. Each highlighted tumor's probability
# can be read directly off the curve, no inset needed.

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# ---- Dark theme matching darkly Quarto ----
plt.style.use('dark_background')
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


# ---- Tumor dataset ----
benign_x = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.3])
malignant_x = np.array([3.5, 4.0, 4.2, 4.5, 5.0, 5.5])
x_data = np.concatenate([benign_x, malignant_x])
y_data = np.concatenate([np.zeros_like(benign_x), np.ones_like(malignant_x)])


# ---- Fit logistic regression by maximizing the log-likelihood ----
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def neg_log_likelihood(theta, x, y):
    """Negative log-likelihood for logistic regression with intercept.

    theta: (theta_0, theta_1)
    """
    z = theta[0] + theta[1] * x
    # Numerically stable form: log sigmoid(z) = -log(1+exp(-z)) = z - log(1+exp(z))
    log_h = -np.logaddexp(0, -z)
    log_one_minus_h = -np.logaddexp(0, z)
    return -np.sum(y * log_h + (1 - y) * log_one_minus_h)


# Initial guess
theta_init = np.array([0.0, 0.0])
result = minimize(neg_log_likelihood, theta_init, args=(x_data, y_data),
                   method='BFGS')
theta_fit = result.x
print(f"Fitted parameters: theta_0 = {theta_fit[0]:.4f}, theta_1 = {theta_fit[1]:.4f}")
# Decision boundary: sigmoid(theta_0 + theta_1 * x) = 0.5  =>  theta_0 + theta_1 * x = 0
x_boundary = -theta_fit[0] / theta_fit[1]
print(f"Decision boundary at x = {x_boundary:.3f} cm")


# ---- Set up figure ----
fig, ax = plt.subplots(figsize=(11, 5), dpi=500)
fig.patch.set_facecolor('#222222')
ax.set_facecolor('#222222')

benign_color = '#3498db'
malignant_color = '#e67e22'
sigmoid_color = '#1abc9c'
boundary_color = '#ecf0f1'
highlight_color = '#f1c40f'


# Plot the data
ax.scatter(benign_x, np.zeros_like(benign_x), c=benign_color, s=100,
            edgecolors='#222222', linewidths=1.0, zorder=3,
            label=r'Benign ($y = 0$)')
ax.scatter(malignant_x, np.ones_like(malignant_x), c=malignant_color, s=100,
            edgecolors='#222222', linewidths=1.0, zorder=3,
            label=r'Malignant ($y = 1$)')

# Plot the fitted sigmoid
x_grid = np.linspace(0, 6.5, 400)
h_grid = sigmoid(theta_fit[0] + theta_fit[1] * x_grid)
ax.plot(x_grid, h_grid, color=sigmoid_color, linewidth=2.5, zorder=2,
         label=r'$h_{\boldsymbol{\theta}}(x) = g(\theta_0 + \theta_1 x)$')

# 0.5 reference line
ax.axhline(0.5, color='#95a5a6', linestyle='--', linewidth=0.9, alpha=0.6, zorder=1)
ax.text(6.45, 0.52, '0.5', color='#95a5a6', fontsize=9, va='bottom', ha='right')

# Decision boundary vertical line
ax.axvline(x_boundary, color=boundary_color, linestyle=':', linewidth=1.5,
            alpha=0.8, zorder=1)
ax.text(x_boundary + 0.06, 1.18, f'boundary\n$x \\approx {x_boundary:.2f}$ cm',
        color=boundary_color, fontsize=9, va='top', ha='left')


# Highlight three sample tumors:
#   - small benign tumor
#   - tumor near the decision boundary (use a slightly smaller new size)
#   - large malignant tumor
sample_xs = [1.0, x_boundary, 4.5]
sample_labels = ['small tumor', 'borderline tumor', 'large tumor']

for x_s, name in zip(sample_xs, sample_labels):
    p_s = sigmoid(theta_fit[0] + theta_fit[1] * x_s)
    # Ring on the sigmoid curve at (x_s, p_s)
    ax.scatter([x_s], [p_s], facecolors='none',
                edgecolors=highlight_color, s=320, linewidths=2.5, zorder=4)
    # Vertical and horizontal dotted guides
    ax.plot([x_s, x_s], [0, p_s], color=highlight_color,
             linestyle=':', linewidth=1.0, alpha=0.7, zorder=1)
    ax.plot([0, x_s], [p_s, p_s], color=highlight_color,
             linestyle=':', linewidth=1.0, alpha=0.7, zorder=1)
    # Label
    label_text = f'{name}\n$x = {x_s:.2f}$ cm\n$P = {p_s:.2f}$'
    # Position label: small tumor below to the right; boundary above to the right;
    # large tumor below and to the right
    if x_s < 2:
        xy_text = (x_s + 0.25, p_s + 0.15)
        ha = 'left'
    elif x_s < 3.5:
        xy_text = (x_s + 0.25, p_s - 0.07)
        ha = 'left'
    else:
        xy_text = (x_s - 0.25, p_s - 0.20)
        ha = 'right'
    ax.annotate(label_text, xy=(x_s, p_s), xytext=xy_text,
                 fontsize=9, color=highlight_color, ha=ha, va='center')

ax.set_xlim(0, 6.5)
ax.set_ylim(-0.20, 1.30)
ax.set_xlabel(r'$x$ (tumor size, cm)', fontsize=12)
ax.set_ylabel(r'$y$  or  $h_{\boldsymbol{\theta}}(x)$', fontsize=12)
ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels(['0', '0.5', '1'])
ax.grid(True, color='#444444', linestyle='--', linewidth=0.4, alpha=0.5)
ax.legend(loc='lower right', framealpha=0.4, fontsize=9)


plt.tight_layout()
from pathlib import Path
_out = Path(__file__).resolve().parent.parent / "images" / "probability_interpretation.png"
plt.savefig(_out, dpi=500,
             facecolor='#222222', bbox_inches='tight')
plt.close(fig)


# How to run (Windows):
#   python probability_interpretation_matplotlib.py
# (run from the directory containing this script in PowerShell or cmd)
#
# Output:
#   probability_interpretation.png  (in the same directory)
#
# Embed in Quarto using:
#   ![Caption](images/probability_interpretation.png){#fig-probability_interpretation fig-align="center" width=85% .invert}
#
# Note: this script requires scipy in addition to numpy and matplotlib because
# we use scipy.optimize.minimize to fit logistic regression. If you don't have
# scipy installed, run: pip install scipy
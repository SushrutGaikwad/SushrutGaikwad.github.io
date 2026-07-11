# Setup (one-time, on Windows):
#   1. Install Python 3.9+
#   2. pip install numpy matplotlib
#   3. (no FFmpeg or LaTeX needed for a static PNG)
#
# This script generates a STATIC PNG showing the canonical 1D failure of
# linear regression on a binary classification problem. We use the classic
# Andrew Ng "tumor size" pedagogical setup: a single feature (tumor size)
# and a binary label (malignant 1 / benign 0). This is the cleanest setup
# because the failure is geometrically obvious in 1D.
#
# Left panel: a clean dataset where most data fits a single 1D pattern
# (small tumors are benign, large tumors malignant). Linear regression
# fitted to this data gives a line whose 0.5 crossing nicely separates
# the classes.
#
# Right panel: the SAME dataset plus one far-right "very obvious malignant"
# point (a tumor much larger than any seen in training). Linear regression
# refit on this data has its line tilted (smaller slope, since the new
# point is far above where the original line predicts), and the 0.5
# threshold now crosses to the right of some malignant points -> they get
# misclassified as benign.

import numpy as np
import matplotlib.pyplot as plt


# Dark theme matching darkly Quarto theme
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


# ---- Build the dataset ----
# Feature x = tumor size (in cm). Benign tumors tend to be small, malignant ones larger.
benign_x = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.3])
malignant_x = np.array([3.5, 4.0, 4.2, 4.5, 5.0, 5.5])

x_orig = np.concatenate([benign_x, malignant_x])
y_orig = np.concatenate([np.zeros_like(benign_x), np.ones_like(malignant_x)])

# Outlier: a much larger malignant tumor (way out at x = 12)
outlier_x = np.array([12.0])
outlier_y = np.array([1.0])

x_with = np.concatenate([x_orig, outlier_x])
y_with = np.concatenate([y_orig, outlier_y])


# ---- Linear regression: theta = (X^T X)^-1 X^T y, with intercept ----
def fit_linear_regression(x, y):
    """Fit y = theta_0 + theta_1 * x by ordinary least squares."""
    X = np.column_stack([np.ones_like(x), x])
    theta = np.linalg.solve(X.T @ X, X.T @ y)
    return theta  # theta[0] = intercept, theta[1] = slope


theta_orig = fit_linear_regression(x_orig, y_orig)
theta_with = fit_linear_regression(x_with, y_with)

# Decision boundary: the x at which h(x) = 0.5
# 0.5 = theta_0 + theta_1 * x  =>  x = (0.5 - theta_0) / theta_1
x_boundary_orig = (0.5 - theta_orig[0]) / theta_orig[1]
x_boundary_with = (0.5 - theta_with[0]) / theta_with[1]

print(f"Without outlier: theta = {theta_orig}, decision boundary at x = {x_boundary_orig:.3f}")
print(f"With outlier:    theta = {theta_with}, decision boundary at x = {x_boundary_with:.3f}")


# ---- Plot ----
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5), dpi=500)
fig.patch.set_facecolor('#222222')

malignant_color = '#e67e22'
benign_color = '#3498db'
line_color = '#ecf0f1'
bad_line_color = '#e74c3c'
outlier_color = '#e74c3c'
threshold_color = '#95a5a6'


def plot_panel(ax, x, y, theta, x_boundary, line_color_used,
               title, x_lim, show_outlier=False, x_outlier=None, y_outlier=None,
               highlight_misclass=False, x_misclass_threshold=None):
    ax.set_facecolor('#222222')

    # Plot the data points
    benign_mask = (y == 0)
    malignant_mask = (y == 1)
    ax.scatter(x[benign_mask], y[benign_mask], c=benign_color, s=80,
               edgecolors='#222222', linewidths=1.0, zorder=3,
               label='Benign ($y=0$)')
    ax.scatter(x[malignant_mask], y[malignant_mask], c=malignant_color, s=80,
               edgecolors='#222222', linewidths=1.0, zorder=3,
               label='Malignant ($y=1$)')

    if show_outlier and x_outlier is not None:
        ax.scatter(x_outlier, y_outlier, c=outlier_color, s=130,
                   edgecolors='#222222', linewidths=1.5, zorder=4)
        ax.annotate('outlier', xy=(x_outlier[0], y_outlier[0]),
                    xytext=(x_outlier[0] - 1.2, 0.78),
                    fontsize=11, color=outlier_color,
                    arrowprops=dict(arrowstyle='->', color=outlier_color, lw=1.2))

    # Plot the fitted line
    x_line = np.linspace(x_lim[0], x_lim[1], 200)
    y_line = theta[0] + theta[1] * x_line
    ax.plot(x_line, y_line, color=line_color_used, linewidth=2.2,
            label=r'$h_{\boldsymbol{\theta}}(x) = \theta_0 + \theta_1 x$')

    # Reference horizontal line at y=0.5 (the threshold)
    ax.axhline(0.5, color=threshold_color, linestyle='--', linewidth=1.0, alpha=0.7,
               label=r'threshold $= 0.5$')

    # Vertical line at the decision boundary
    ax.axvline(x_boundary, color=line_color_used, linestyle=':', linewidth=1.5, alpha=0.8)
    ax.text(x_boundary + 0.15, -0.15,
            f'boundary $x = {x_boundary:.2f}$',
            color=line_color_used, fontsize=9)

    # Highlight points that get misclassified
    if highlight_misclass and x_misclass_threshold is not None:
        # Malignant points that fall to the LEFT of the new (worse) boundary are misclassified
        misclass_mask = malignant_mask & (x < x_misclass_threshold)
        if np.any(misclass_mask):
            for xi, yi in zip(x[misclass_mask], y[misclass_mask]):
                ax.scatter([xi], [yi], facecolors='none',
                           edgecolors=bad_line_color, s=300, linewidths=2.5, zorder=5)

    ax.set_xlim(x_lim)
    ax.set_ylim(-0.4, 1.4)
    ax.set_xlabel(r'$x$ (tumor size, cm)', fontsize=11)
    ax.set_ylabel(r'$y$', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, color='#444444', linestyle='--', linewidth=0.4, alpha=0.5)
    ax.legend(loc='lower right', framealpha=0.4, fontsize=9)


# Left: without outlier
plot_panel(
    ax_left, x_orig, y_orig, theta_orig, x_boundary_orig,
    line_color_used=line_color,
    title='Without outlier: linear regression separates the classes',
    x_lim=(0, 7),
)

# Right: with outlier. The boundary shifts right, misclassifying some malignant points.
plot_panel(
    ax_right, x_with, y_with, theta_with, x_boundary_with,
    line_color_used=bad_line_color,
    title='With far-right outlier: boundary shifts, misclassifying malignant points',
    x_lim=(0, 13),
    show_outlier=True, x_outlier=outlier_x, y_outlier=outlier_y,
    highlight_misclass=True, x_misclass_threshold=x_boundary_with,
)


plt.tight_layout()
from pathlib import Path
_out = Path(__file__).resolve().parent.parent / "images" / "linear_regression_fails.png"
plt.savefig(_out, dpi=500, facecolor='#222222', bbox_inches='tight')
plt.close(fig)


# How to run (Windows):
#   python linear_regression_fails_matplotlib.py
# (run from the directory containing this script in PowerShell or cmd)
#
# Output:
#   linear_regression_fails.png  (in the same directory)
#
# Embed in Quarto using:
#   ![Caption](images/linear_regression_fails.png){#fig-linear_regression_fails fig-align="center" width=95% .invert}
#
# Note: this script switches the example from "spam vs. ham (2D)" to
# "tumor size vs. malignancy (1D)". The 1D version is the textbook-canonical
# illustration of linear regression's failure on classification (it's the
# example used in the lecture this post is based on), and unlike the 2D
# version, it makes the failure mode geometrically transparent.
#
# IMPORTANT: if you switch this figure to the tumor-size example, the
# corresponding paragraphs in the blog post (which currently reference the
# spam example) need to be updated to match. Two options:
#   A) Use the tumor-size example only for this one figure and rewrite the
#      "naive linear regression" section to talk about tumor size, then go
#      back to spam afterward. (The lecture itself essentially does this.)
#   B) Rebuild the figure in 2D with the spam example, doing real OLS in
#      the script. This is harder to make visually convincing because the
#      2D failure mode is more subtle, but it preserves the running example.
#
# I recommend option A: it matches the lecture, the failure is more
# obvious, and it's only a brief detour before returning to the main spam
# narrative for the perceptron and logistic regression.
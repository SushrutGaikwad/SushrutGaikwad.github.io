# Setup (one-time, on Windows):
#   1. Install Python 3.9+
#   2. pip install numpy matplotlib
#   (no LaTeX or FFmpeg needed for a static PNG)
#
# This script generates a STATIC PNG showing the toy tumor dataset on a
# 1D number line: tumor size (cm) on the horizontal axis, with each
# point colored blue (benign, y=0) or orange (malignant, y=1). Because
# the dataset is 1D, this is naturally a 1D scatter rather than a 2D
# scatter; we vertically separate the two classes purely for visual
# legibility (so points don't overlap) and label the y axis as the class
# label y.

import numpy as np
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

benign_color = '#3498db'
malignant_color = '#e67e22'


# ---- Plot ----
fig, ax = plt.subplots(figsize=(10, 3.5), dpi=500)
fig.patch.set_facecolor('#222222')
ax.set_facecolor('#222222')

# Plot benign at y=0, malignant at y=1
ax.scatter(benign_x, np.zeros_like(benign_x), c=benign_color, s=110,
            edgecolors='#222222', linewidths=1.0, zorder=3,
            label=r'Benign ($y = 0$)')
ax.scatter(malignant_x, np.ones_like(malignant_x), c=malignant_color, s=110,
            edgecolors='#222222', linewidths=1.0, zorder=3,
            label=r'Malignant ($y = 1$)')

# Light reference horizontal lines at y=0 and y=1
ax.axhline(0, color='#555555', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)
ax.axhline(1, color='#555555', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)

ax.set_xlim(0, 6.5)
ax.set_ylim(-0.6, 1.6)
ax.set_xlabel(r'$x$ (tumor size, cm)', fontsize=12)
ax.set_ylabel(r'$y$', fontsize=12)
ax.set_yticks([0, 1])
ax.set_yticklabels(['0\n(benign)', '1\n(malignant)'])
ax.grid(True, color='#444444', linestyle='--', linewidth=0.4, alpha=0.5, axis='x')
ax.legend(loc='center right', framealpha=0.4, fontsize=10)


plt.tight_layout()
from pathlib import Path
_out = Path(__file__).resolve().parent.parent / "images" / "tumor_scatter.png"
plt.savefig(_out, dpi=500, facecolor='#222222', bbox_inches='tight')
plt.close(fig)


# How to run (Windows):
#   python tumor_scatter_matplotlib.py
# (run from the directory containing this script in PowerShell or cmd)
#
# Output:
#   tumor_scatter.png  (in the same directory)
#
# Embed in Quarto using:
#   ![Caption](images/tumor_scatter.png){#fig-tumor_scatter fig-align="center" width=80% .invert}
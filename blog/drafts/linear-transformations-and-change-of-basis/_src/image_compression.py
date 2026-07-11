"""Image-compression demo for Post 17.

Re-expresses a synthetic grayscale image in the 2-D cosine (JPEG) basis, keeps
only the largest few percent of coefficients, and transforms back. Shows the
original and three reconstructions. Saves an SVG.

Runs via the root uv environment:
  uv run python blog/published/linear-transformations-and-change-of-basis/_src/image_compression.py
Output: blog/published/linear-transformations-and-change-of-basis/images/image_compression.svg
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

plt.style.use("dark_background")
plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})

LIGHT = "#e6e6e6"
DARK = "#222222"

# Synthesize a smooth grayscale image with a few structured features.
N = 128
x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x)
img = 0.5 + 0.25 * np.sin(3 * np.pi * X) * np.cos(2 * np.pi * Y)  # smooth waves
img += 0.6 * np.exp(-((X + 0.3) ** 2 + (Y - 0.2) ** 2) / 0.08)    # bright blob
img -= 0.4 * np.exp(-((X - 0.4) ** 2 + (Y + 0.4) ** 2) / 0.03)    # dark spot
disk = (X ** 2 + Y ** 2) < 0.55 ** 2
img[disk] += 0.15
img = np.clip(img, 0, 1)

coeffs = dctn(img, norm="ortho")
flat = np.abs(coeffs).ravel()
total = flat.size


def keep_fraction(frac):
    """Keep the largest `frac` of DCT coefficients, zero the rest, transform back."""
    k = max(1, int(frac * total))
    thresh = np.sort(flat)[-k]
    mask = np.abs(coeffs) >= thresh
    return idctn(coeffs * mask, norm="ortho")


fractions = [0.01, 0.05, 0.20]
panels = [("Original", img)] + [(f"{int(f * 100)}% of coefficients", keep_fraction(f)) for f in fractions]

fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
fig.patch.set_facecolor(DARK)
for ax, (label, im) in zip(axes, panels):
    ax.imshow(im, cmap="gray", vmin=0, vmax=1)
    ax.set_title(label, color=LIGHT, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#555555")

fig.suptitle("Image compression by changing to the cosine basis and keeping the largest coefficients",
             color=LIGHT, fontsize=13, y=1.02)

out = Path(__file__).resolve().parent.parent / "images" / "image_compression.svg"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=DARK, bbox_inches="tight")
print(f"saved {out}")

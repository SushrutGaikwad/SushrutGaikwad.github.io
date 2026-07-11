"""House-transformation animation for Post 17.

A "house" polygon is carried through a rotation, a shear, and a stretch, each a
linear map applied to the house's coordinates. Saves an MP4.

Runs via the root uv environment:
  uv run python blog/published/linear-transformations-and-change-of-basis/_src/house_transform.py
Output: blog/published/linear-transformations-and-change-of-basis/images/house_transform.mp4
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.style.use("dark_background")
plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})

LIGHT = "#e6e6e6"
DARK = "#222222"
BLUE_A = "#5dade2"
ORANGE_A = "#f5b041"

# House outline and door (columns are points).
outline = np.array([
    [-2, -2], [2, -2], [2, 1], [0, 2.6], [-2, 1], [-2, -2],
]).T
door = np.array([
    [-0.6, -2], [-0.6, -0.4], [0.6, -0.4], [0.6, -2],
]).T


def rot(s):
    a = s * np.deg2rad(45)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def shear(s):
    return np.array([[1.0, 0.6 * s], [0.0, 1.0]])


def stretch(s):
    return np.array([[1.0 + 0.4 * s, 0.0], [0.0, 1.0 - 0.4 * s]])


SEGMENTS = [("Rotation", rot), ("Shear", shear), ("Stretch", stretch)]

# Build the frame schedule: per segment, ramp s 0->1 then hold.
schedule = []  # list of (segment_index, s)
for k in range(len(SEGMENTS)):
    schedule += [(k, 0.0)] * 6
    schedule += [(k, s) for s in np.linspace(0, 1, 30)]
    schedule += [(k, 1.0)] * 14

fig, ax = plt.subplots(figsize=(6.5, 6))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.0, 4.0)
ax.set_aspect("equal")
ax.axhline(0, color="#555555", lw=1)
ax.axvline(0, color="#555555", lw=1)
ax.tick_params(colors=LIGHT)
for s in ax.spines.values():
    s.set_color("#555555")

(outline_line,) = ax.plot([], [], color=BLUE_A, lw=3)
(door_line,) = ax.plot([], [], color=ORANGE_A, lw=3)
title = ax.set_title("", color=LIGHT, fontsize=14, pad=12)
# Faint original house for reference.
ax.plot(outline[0], outline[1], color="#555555", lw=1.2, ls="--", alpha=0.7)
ax.plot(door[0], door[1], color="#555555", lw=1.2, ls="--", alpha=0.7)


def init():
    outline_line.set_data([], [])
    door_line.set_data([], [])
    return outline_line, door_line, title


def update(frame):
    k, s = schedule[frame]
    name, fn = SEGMENTS[k]
    M = fn(s)
    o = M @ outline
    d = M @ door
    outline_line.set_data(o[0], o[1])
    door_line.set_data(d[0], d[1])
    title.set_text(f"Linear transformation: {name}")
    return outline_line, door_line, title


ani = animation.FuncAnimation(fig, update, frames=len(schedule), init_func=init, blit=False, interval=40)

out = Path(__file__).resolve().parent.parent / "images" / "house_transform.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
ani.save(out, writer="ffmpeg", fps=25, dpi=120, savefig_kwargs={"facecolor": DARK})
print(f"saved {out}")

# Embed with: {{< video images/house_transform.mp4 >}}

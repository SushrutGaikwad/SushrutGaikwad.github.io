# Runs via the root uv environment with `uv run`.
"""The n-th roots of unity on the unit circle for Post 14 (shown for n = 8).

All text uses MathTex (LaTeX / Computer Modern); bold uses \\mathbf.
"""

import numpy as np
from manim import *

LIGHT = "#e6e6e6"
BLUE_A = "#5dade2"
ORANGE_A = "#f5b041"
GRAY_L = "#9aa0a6"


class RootsOfUnity(Scene):
    def construct(self):
        self.camera.background_color = "#222222"
        R = 2.4
        n = 8

        # Axes (real and imaginary).
        re_axis = Line([-R - 0.6, 0, 0], [R + 0.6, 0, 0], color=GRAY_L, stroke_width=2)
        im_axis = Line([0, -R - 0.6, 0], [0, R + 0.6, 0], color=GRAY_L, stroke_width=2)
        re_lab = MathTex(r"\mathrm{Re}", color=GRAY_L).scale(0.6).next_to(re_axis, RIGHT, buff=0.15).shift(DOWN * 0.4)
        im_lab = MathTex(r"\mathrm{Im}", color=GRAY_L).scale(0.6).next_to(im_axis, UP, buff=0.15).shift(RIGHT * 0.45)
        circle = Circle(radius=R, color=GRAY_L, stroke_width=2.5)
        self.add(re_axis, im_axis, re_lab, im_lab, circle)

        # The roots and their labels.
        for k in range(n):
            ang = 2 * np.pi * k / n
            p = np.array([R * np.cos(ang), R * np.sin(ang), 0.0])
            is_w = (k == 1)
            color = ORANGE_A if is_w else BLUE_A
            self.add(Dot(p, color=color, radius=0.08))
            if k == 0:
                txt = r"1"
            elif k == 1:
                txt = r"w"
            else:
                txt = rf"w^{{{k}}}"
            lab = MathTex(txt, color=color).scale(0.75)
            lab.move_to(1.22 * p)
            self.add(lab)

        # Definition of w in a corner.
        wdef = MathTex(r"w = e^{2\pi i / 8}", color=LIGHT).scale(0.9)
        wdef.to_corner(UL, buff=0.6)
        self.add(wdef)


# Render the last frame only as a PNG, from the repo root (unique media dir):
#   uv run manim -qh -s --media_dir media_roots blog/published/complex-matrices-and-the-fft/_src/roots_of_unity.py RootsOfUnity
# Move media_roots/images/roots_of_unity/RootsOfUnity*.png to
#   blog/published/complex-matrices-and-the-fft/images/roots_of_unity.png

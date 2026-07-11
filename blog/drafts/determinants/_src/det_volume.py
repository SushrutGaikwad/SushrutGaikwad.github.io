# Runs via the root uv environment with `uv run`.
"""Determinant-as-area figure for Post 9.

The rows (3,0) and (1,2) span a parallelogram of area |det| = 6. All text uses
MathTex (LaTeX / Computer Modern); bold uses \\mathbf.
"""

import numpy as np
from manim import *

LIGHT = "#e6e6e6"
BLUE = "#5dade2"
ORANGE = "#f5b041"
GREEN = "#10a37f"


class DetVolume(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        O = LEFT * 2.8 + DOWN * 1.6
        s = 1.25

        def pt(x, y):
            return O + np.array([x, y, 0.0]) * s

        # Light axes.
        self.add(Line(O + LEFT * 0.4, pt(5, 0), color="#555555", stroke_width=2))
        self.add(Line(O + DOWN * 0.4, pt(0, 3), color="#555555", stroke_width=2))

        v1 = pt(3, 0)
        v2 = pt(1, 2)
        vsum = pt(4, 2)

        para = Polygon(O, v1, vsum, v2, color=GREEN, stroke_width=2)
        para.set_fill(GREEN, opacity=0.25)
        self.add(para)

        a1 = Arrow(O, v1, buff=0, color=BLUE, stroke_width=6)
        a2 = Arrow(O, v2, buff=0, color=ORANGE, stroke_width=6)
        self.add(a1, a2)

        lab1 = MathTex(r"(3,0)", color=BLUE).scale(0.8).next_to(a1.get_end(), DOWN, buff=0.2)
        lab2 = MathTex(r"(1,2)", color=ORANGE).scale(0.8).next_to(a2.get_end(), LEFT, buff=0.2)
        self.add(lab1, lab2)

        area = MathTex(r"\text{Area} = |\det| = 6", color=LIGHT).scale(0.9)
        area.move_to(pt(2.0, 1.0))
        self.add(area)

        formula = MathTex(r"\det\begin{pmatrix}3&0\\1&2\end{pmatrix} = 3\cdot 2 - 0\cdot 1 = 6",
                          color=LIGHT).scale(0.85)
        formula.to_edge(UP, buff=0.7)
        self.add(formula)


# Render the last frame only as a PNG (use a unique media dir), from the repo root:
#   uv run manim -qh -s --media_dir media_determinants blog/published/determinants/_src/det_volume.py DetVolume
# Move media_determinants/images/det_volume/DetVolume*.png to
#   blog/published/determinants/images/det_volume.png and embed with:
#   ![Caption.](images/det_volume.png){#fig-det_volume fig-align="center" width=80%}

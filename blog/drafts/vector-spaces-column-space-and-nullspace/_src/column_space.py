# Runs via the root uv environment with `uv run`.
"""Column-space figure for Post 3.

A schematic of C(A) as a plane through the origin in R^3, spanned by the
columns a1, a2. One target b lies in the plane (reachable); another pokes out
of the plane (not reachable). Drawn in 2D with a tilted parallelogram to suggest
the plane. All text uses MathTex (LaTeX / Computer Modern); bold uses \\mathbf.
"""

import numpy as np
from manim import *

LIGHT = "#e6e6e6"
BLUE_A = "#5dade2"
ORANGE_A = "#f5b041"
RED_B = "#ec7063"
PLANE_C = "#5dade2"


class ColumnSpace(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        O = LEFT * 1.6 + DOWN * 1.3  # the origin
        e1 = np.array([3.7, 0.5, 0.0])   # one in-plane screen direction
        e2 = np.array([1.4, 2.4, 0.0])   # the other (suggests depth)

        # The plane patch = column space.
        plane = Polygon(O, O + e1, O + e1 + e2, O + e2,
                        color=PLANE_C, stroke_width=2)
        plane.set_fill(PLANE_C, opacity=0.18)
        self.add(plane)
        ca_lab = MathTex(r"C(\mathbf{A})", color=LIGHT).scale(0.9)
        ca_lab.move_to(O + 0.78 * e1 + 0.82 * e2)
        self.add(ca_lab)

        # Spanning columns a1, a2 lying in the plane.
        arr_a1 = Arrow(O, O + 0.52 * e1, buff=0, color=BLUE_A, stroke_width=5)
        arr_a2 = Arrow(O, O + 0.5 * e2, buff=0, color=ORANGE_A, stroke_width=5)
        lab_a1 = MathTex(r"\mathbf{a}_1", color=BLUE_A).scale(0.8).next_to(arr_a1.get_end(), DOWN, buff=0.12)
        lab_a2 = MathTex(r"\mathbf{a}_2", color=ORANGE_A).scale(0.8).next_to(arr_a2.get_end(), LEFT, buff=0.12)
        self.add(arr_a1, arr_a2, lab_a1, lab_a2)

        # A reachable target inside the plane.
        pr = O + 0.62 * e1 + 0.52 * e2
        arr_br = Arrow(O, pr, buff=0, color=RED_B, stroke_width=5)
        dot_br = Dot(pr, color=RED_B, radius=0.06)
        lab_br = MathTex(r"\mathbf{b}\ \text{reachable}", color=RED_B).scale(0.7).next_to(dot_br, RIGHT, buff=0.12)
        self.add(arr_br, dot_br, lab_br)

        # An unreachable target poking out of the plane.
        pb = O + 0.42 * e1 + 0.26 * e2          # foot in the plane
        b_out = pb + np.array([0.0, 1.8, 0.0])  # lift straight up, off the plane
        drop = DashedLine(pb, b_out, color=LIGHT, stroke_width=2)
        arr_bo = Arrow(O, b_out, buff=0, color=RED_B, stroke_width=5)
        dot_bo = Dot(b_out, color=RED_B, radius=0.06)
        foot = Dot(pb, color=LIGHT, radius=0.04)
        lab_bo = MathTex(r"\mathbf{b}\ \text{not reachable}", color=RED_B).scale(0.7).next_to(dot_bo, UP, buff=0.12)
        self.add(drop, foot, arr_bo, dot_bo, lab_bo)


# Render the last frame only as a PNG, from the repo root:
#   uv run manim -qh -s blog/published/vector-spaces-column-space-and-nullspace/_src/column_space.py ColumnSpace
# Manim saves it at media/images/column_space/ColumnSpace.png.
# Move it to blog/published/vector-spaces-column-space-and-nullspace/images/column_space.png and embed with:
#   ![Caption.](images/column_space.png){#fig-column_space fig-align="center" width=80%}

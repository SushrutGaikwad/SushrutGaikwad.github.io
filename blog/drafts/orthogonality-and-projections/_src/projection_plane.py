# Runs via the root uv environment with `uv run`.
"""Projection-onto-a-subspace figure for Post 7.

The column space C(A) is a plane; b sticks out of it; p is the foot of the
perpendicular and e = b - p is orthogonal to the plane (so b = p + e). Drawn in
2D with a tilted parallelogram for the plane. All text uses MathTex
(LaTeX / Computer Modern); bold uses \\mathbf.
"""

import numpy as np
from manim import *

LIGHT = "#e6e6e6"
BLUE_A = "#5dade2"     # b
ORANGE_A = "#f5b041"   # p
RED_B = "#ec7063"      # e
PLANE_C = "#5dade2"


class ProjectionPlane(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        O = LEFT * 1.4 + DOWN * 1.4
        e1 = np.array([3.8, 0.5, 0.0])
        e2 = np.array([1.5, 2.4, 0.0])

        plane = Polygon(O, O + e1, O + e1 + e2, O + e2, color=PLANE_C, stroke_width=2)
        plane.set_fill(PLANE_C, opacity=0.16)
        self.add(plane)
        ca = MathTex(r"C(\mathbf{A})", color=LIGHT).scale(0.9).move_to(O + 0.92 * e1 + 0.2 * e2)
        self.add(ca)

        # Projection p (foot, in the plane) and target b (above the plane).
        p = O + 0.5 * e1 + 0.42 * e2
        b = p + np.array([0.0, 1.9, 0.0])

        p_arr = Arrow(O, p, buff=0, color=ORANGE_A, stroke_width=5)
        b_arr = Arrow(O, b, buff=0, color=BLUE_A, stroke_width=5)
        e_line = Line(p, b, color=RED_B, stroke_width=5)

        p_lab = MathTex(r"\mathbf{p}", color=ORANGE_A).scale(0.85).next_to(p, DR, buff=0.1)
        b_lab = MathTex(r"\mathbf{b}", color=BLUE_A).scale(0.85).next_to(b, UP, buff=0.12)
        e_lab = MathTex(r"\mathbf{e}", color=RED_B).scale(0.85).next_to(e_line.get_center(), RIGHT, buff=0.16)

        # Small right-angle mark where e meets the plane.
        ra = Square(side_length=0.28, color=RED_B, stroke_width=2)
        ra.move_to(p + np.array([0.0, 0.14, 0.0]) + 0.14 * (e1 / np.linalg.norm(e1)))

        foot = Dot(p, color=LIGHT, radius=0.04)
        self.add(p_arr, b_arr, e_line, foot, ra, p_lab, b_lab, e_lab)

        eq = MathTex(r"\mathbf{b} = \mathbf{p} + \mathbf{e}", color=LIGHT).scale(0.9)
        eq.to_corner(UR, buff=0.8)
        self.add(eq)


# Render the last frame only as a PNG, from the repo root:
#   uv run manim -qh -s --media_dir media_p7b blog/published/orthogonality-and-projections/_src/projection_plane.py ProjectionPlane
# Output at media_p7b/images/projection_plane/ProjectionPlane.png. Move to
#   blog/published/orthogonality-and-projections/images/projection_plane.png

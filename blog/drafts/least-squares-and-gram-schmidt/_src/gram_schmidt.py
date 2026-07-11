# Runs via the root uv environment with `uv run`.
"""Gram-Schmidt figure for Post 8.

The key step: subtract from a2 its projection onto a1, leaving a perpendicular
remainder. Drawn in 2D for clarity. All text uses MathTex (LaTeX / Computer
Modern); bold uses \\mathbf.
"""

import numpy as np
from manim import *

LIGHT = "#e6e6e6"
BLUE_A = "#5dade2"     # a1
ORANGE_A = "#f5b041"   # a2
RED_B = "#ec7063"      # perpendicular remainder
GRAY_L = "#9aa0a6"


class GramSchmidt(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        O = LEFT * 3.0 + DOWN * 1.8
        S = 1.5

        def Pt(x, y):
            return O + S * np.array([x, y, 0.0])

        a1_tip = Pt(3, 0)
        a2_tip = Pt(2, 2)
        proj_tip = Pt(2, 0)  # projection of a2 onto a1 (direction (1,0))

        a1 = Arrow(O, a1_tip, buff=0, color=BLUE_A, stroke_width=5)
        a2 = Arrow(O, a2_tip, buff=0, color=ORANGE_A, stroke_width=5)
        a1_lab = MathTex(r"\mathbf{a}_1", color=BLUE_A).scale(0.85).next_to(a1_tip, RIGHT, buff=0.12)
        a2_lab = MathTex(r"\mathbf{a}_2", color=ORANGE_A).scale(0.85).next_to(a2_tip, UP, buff=0.12)

        # Projection of a2 onto a1 (a faint segment along a1).
        proj = Line(O, proj_tip, color=GRAY_L, stroke_width=4)
        proj_lab = MathTex(r"\text{proj of } \mathbf{a}_2 \text{ on } \mathbf{a}_1", color=GRAY_L).scale(0.55)
        proj_lab.next_to(proj.get_center(), DOWN, buff=0.18)

        # Dashed drop from a2 to its projection, and the perpendicular remainder.
        drop = DashedLine(a2_tip, proj_tip, color=GRAY_L, stroke_width=2)
        perp = Arrow(proj_tip, a2_tip, buff=0, color=RED_B, stroke_width=5)
        perp_lab = MathTex(r"\mathbf{a}_2 - \text{proj}\ \perp\ \mathbf{a}_1", color=RED_B).scale(0.6)
        perp_lab.next_to(perp.get_center(), RIGHT, buff=0.18)
        ra = Square(side_length=0.26, color=RED_B, stroke_width=2).move_to(proj_tip + np.array([-0.16, 0.16, 0.0]))

        self.add(proj, drop, a1, a2, perp, ra)
        self.add(a1_lab, a2_lab, proj_lab, perp_lab)

        title = MathTex(r"\text{subtract the projection to get a perpendicular direction}", color=LIGHT).scale(0.62)
        title.to_edge(UP, buff=0.6)
        self.add(title)

        note = MathTex(r"\mathbf{q}_1 = \frac{\mathbf{a}_1}{\|\mathbf{a}_1\|}, \quad \mathbf{q}_2 = \frac{\mathbf{a}_2-\text{proj}}{\|\mathbf{a}_2-\text{proj}\|}", color=LIGHT).scale(0.7)
        note.to_edge(DOWN, buff=0.5)
        self.add(note)


# Render the last frame only as a PNG, from the repo root:
#   uv run manim -qh -s --media_dir media_p8 blog/published/least-squares-and-gram-schmidt/_src/gram_schmidt.py GramSchmidt
# Output at media_p8/images/gram_schmidt/GramSchmidt.png. Move to
#   blog/published/least-squares-and-gram-schmidt/images/gram_schmidt.png

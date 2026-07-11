# Runs via the root uv environment with `uv run`.
"""Projection-onto-a-line animation for Post 7.

b is projected onto the line through a; the foot p is the closest point and the
error e = b - p meets the line at a right angle. All text uses MathTex
(LaTeX / Computer Modern); bold uses \\mathbf.
"""

import numpy as np
from manim import *

LIGHT = "#e6e6e6"
BLUE_A = "#5dade2"     # b
ORANGE_A = "#f5b041"   # projection p
RED_B = "#ec7063"      # error e
GRAY_L = "#9aa0a6"


class ProjectionLine(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        O = LEFT * 2.2 + DOWN * 1.8
        S = 1.25

        def Pt(x, y):
            return O + S * np.array([x, y, 0.0])

        # The line through a (direction (3,1)).
        line = Line(Pt(-1.2, -0.4), Pt(4.6, 1.533), color=GRAY_L, stroke_width=3)
        line_lab = MathTex(r"\text{line through } \mathbf{a}", color=GRAY_L).scale(0.6)
        line_lab.next_to(Pt(4.6, 1.533), RIGHT, buff=0.1)

        b = Pt(1, 3)
        p = Pt(1.8, 0.6)

        b_arr = Arrow(O, b, buff=0, color=BLUE_A, stroke_width=5)
        b_lab = MathTex(r"\mathbf{b}", color=BLUE_A).scale(0.85).next_to(b, UP, buff=0.12)

        drop = DashedLine(b, p, color=GRAY_L, stroke_width=2)
        p_arr = Arrow(O, p, buff=0, color=ORANGE_A, stroke_width=5)
        p_lab = MathTex(r"\mathbf{p}", color=ORANGE_A).scale(0.85).next_to(p, DR, buff=0.12)

        e_line = Line(p, b, color=RED_B, stroke_width=5)
        e_lab = MathTex(r"\mathbf{e} = \mathbf{b} - \mathbf{p}", color=RED_B).scale(0.72)
        e_lab.next_to(e_line.get_center(), LEFT, buff=0.18)
        ra = RightAngle(line, e_line, length=0.28, quadrant=(-1, 1), color=RED_B, stroke_width=2)

        # Animate.
        self.play(Create(line), FadeIn(line_lab))
        self.play(GrowArrow(b_arr), FadeIn(b_lab))
        self.wait(0.3)
        self.play(Create(drop))
        self.play(GrowArrow(p_arr), FadeIn(p_lab))
        self.wait(0.2)
        self.play(Create(e_line), FadeIn(e_lab), Create(ra))
        self.wait(1.2)


# Render an MP4 (NOT a -qh GIF), from the repo root:
#   uv run manim -qh --media_dir media_p7a blog/published/orthogonality-and-projections/_src/projection_line.py ProjectionLine
# Output at media_p7a/videos/projection_line/1080p60/ProjectionLine.mp4. Move to
#   blog/published/orthogonality-and-projections/images/projection_line.mp4

# Runs via the root uv environment with `uv run`.
"""Static column picture for Post 1.

Shows 2*a1 + 3*a2 = b drawn tip to tail, with a1 = (1, 3), a2 = (2, 1), b = (8, 9).
All text is rendered with MathTex (LaTeX / Computer Modern). Note: Manim's LaTeX
template does not load the physics package, so bold vectors use \\mathbf, not \\vb.
"""

from manim import *

BLUE_A = "#5dade2"   # a1 and its multiples
ORANGE_A = "#f5b041"  # a2 and its multiples
RED_B = "#ec7063"    # the target b
LIGHT = "#e6e6e6"


class ColumnPicture(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        plane = NumberPlane(
            x_range=[-1, 9, 1],
            y_range=[-1, 10, 1],
            x_length=7.0,
            y_length=7.7,
            background_line_style={"stroke_color": "#3a3a3a", "stroke_width": 1},
            axis_config={"stroke_color": "#888888", "include_ticks": True},
        )
        plane.to_edge(LEFT, buff=0.6)
        plane.add_coordinates(font_size=20)
        plane.coordinate_labels.set_color("#9aa0a6")
        self.add(plane)

        o = plane.c2p(0, 0)

        # The combination, tip to tail: 2*a1 then 3*a2, landing on b = (8, 9).
        # The scaled arrows already show each column's direction, so the unit
        # building blocks a1, a2 are left out to keep the figure uncluttered.
        step1 = Arrow(o, plane.c2p(2, 6), buff=0, color=BLUE_A, stroke_width=6)
        step2 = Arrow(plane.c2p(2, 6), plane.c2p(8, 9), buff=0, color=ORANGE_A, stroke_width=6)
        bvec = Arrow(o, plane.c2p(8, 9), buff=0, color=RED_B, stroke_width=6)

        step1_lab = MathTex(r"2\mathbf{a}_1", color=BLUE_A).scale(0.9).next_to(step1.get_center(), RIGHT, buff=0.2)
        step2_lab = MathTex(r"3\mathbf{a}_2", color=ORANGE_A).scale(0.9).next_to(step2.get_center(), UP, buff=0.25)
        b_lab = MathTex(r"\mathbf{b} = (8, 9)", color=RED_B).scale(0.9).next_to(bvec.get_end(), UP, buff=0.2)

        dot_b = Dot(plane.c2p(8, 9), color=RED_B, radius=0.06)

        self.add(bvec, step1, step2, dot_b)
        self.add(step1_lab, step2_lab, b_lab)

        # Equations on the right.
        eq1 = MathTex(r"x\,\mathbf{a}_1 + y\,\mathbf{a}_2 = \mathbf{b}", color=LIGHT).scale(0.95)
        eq2 = MathTex(r"2\,\mathbf{a}_1 + 3\,\mathbf{a}_2 = \mathbf{b}", color=LIGHT).scale(0.95)
        eqs = VGroup(eq1, eq2).arrange(DOWN, buff=0.5).to_edge(RIGHT, buff=0.7)
        self.add(eqs)


# Render the last frame only as a PNG, from the repo root:
#   uv run manim -qh -s blog/published/the-geometry-of-linear-equations/_src/column_picture.py ColumnPicture
# Manim saves it at media/images/column_picture/ColumnPicture.png.
# Move it to blog/published/the-geometry-of-linear-equations/images/column_picture.png and embed with:
#   ![Caption.](images/column_picture.png){#fig-column_picture fig-align="center" width=80%}

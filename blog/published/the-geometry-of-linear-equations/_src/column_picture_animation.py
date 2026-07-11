# Runs via the root uv environment with `uv run`.
"""Column-picture animation for Post 1.

Builds the linear combination 2*a1 + 3*a2 tip to tail until it reaches b = (8, 9),
with a1 = (1, 3), a2 = (2, 1). All text uses MathTex (LaTeX / Computer Modern);
Manim's LaTeX template lacks the physics package, so bold vectors use \\mathbf.
"""

from manim import *

BLUE_A = "#5dade2"
ORANGE_A = "#f5b041"
RED_B = "#ec7063"
LIGHT = "#e6e6e6"


class ColumnPictureSweep(Scene):
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

        title = MathTex(r"2\,\mathbf{a}_1 + 3\,\mathbf{a}_2 = \mathbf{b}", color=LIGHT)
        title.scale(0.95).to_edge(RIGHT, buff=0.7).shift(UP * 2.5)
        self.add(title)

        # The target b, shown faintly from the start as the goal.
        bvec = Arrow(o, plane.c2p(8, 9), buff=0, color=RED_B, stroke_width=5)
        b_lab = MathTex(r"\mathbf{b}", color=RED_B).scale(0.9).next_to(bvec.get_end(), UP, buff=0.15)
        dot_b = Dot(plane.c2p(8, 9), color=RED_B, radius=0.07)
        self.play(GrowArrow(bvec), FadeIn(b_lab), FadeIn(dot_b))
        self.wait(0.4)

        # Step 1: lay down 2*a1 from the origin.
        step1 = Arrow(o, plane.c2p(2, 6), buff=0, color=BLUE_A, stroke_width=6)
        step1_lab = MathTex(r"2\mathbf{a}_1", color=BLUE_A).scale(0.85).next_to(step1.get_center(), RIGHT, buff=0.2)
        self.play(GrowArrow(step1))
        self.play(FadeIn(step1_lab))
        self.wait(0.3)

        # Step 2: from the tip of 2*a1, lay down 3*a2 to reach b.
        step2 = Arrow(plane.c2p(2, 6), plane.c2p(8, 9), buff=0, color=ORANGE_A, stroke_width=6)
        step2_lab = MathTex(r"3\mathbf{a}_2", color=ORANGE_A).scale(0.85).next_to(step2.get_center(), UP, buff=0.2)
        self.play(GrowArrow(step2))
        self.play(FadeIn(step2_lab))
        self.wait(0.5)

        # Flash to confirm we landed on b.
        self.play(Flash(plane.c2p(8, 9), color=RED_B, flash_radius=0.4))
        self.wait(1.0)


# Render an MP4 (a -qh GIF is enormous; MP4 is tiny and matches the site's other
# Manim clips), from the repo root:
#   uv run manim -qh blog/published/the-geometry-of-linear-equations/_src/column_picture_animation.py ColumnPictureSweep
# Use -ql instead of -qh while iterating. Manim writes to
#   media/videos/column_picture_animation/<quality>/ColumnPictureSweep.mp4
# Move it to blog/published/the-geometry-of-linear-equations/images/column_picture_sweep.mp4 and embed with:
#   {{< video images/column_picture_sweep.mp4 >}}

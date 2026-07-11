# Setup (one-time, on Windows):
#   1. Install Python 3.9+
#   2. Install MiKTeX
#   3. Install FFmpeg and add it to PATH
#   4. pip install manim
#
# This script generates a STATIC PNG illustrating the geometric intuition:
# adding alpha*b to a rotates a toward b, decreasing the angle between
# them and increasing the dot product a^T b.

from manim import *
import numpy as np

class DotProductIntuition(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        light = "#e6e6e6"
        a_color = "#3498db"        # blue
        b_color = "#f1c40f"        # yellow
        new_a_color = "#2ecc71"    # green (a + alpha*b)

        # Origin
        origin = ORIGIN + DOWN * 0.5

        # Vectors a and b at obtuse angle
        # a points up-left at about 130 degrees from positive x-axis
        a_angle = 135 * DEGREES
        a_length = 3.0
        a_end = origin + a_length * np.array([np.cos(a_angle), np.sin(a_angle), 0])

        # b points right at about 10 degrees
        b_angle = 10 * DEGREES
        b_length = 2.5
        b_end = origin + b_length * np.array([np.cos(b_angle), np.sin(b_angle), 0])

        a_arrow = Arrow(origin, a_end, color=a_color, buff=0, stroke_width=4,
                        max_tip_length_to_length_ratio=0.12)
        b_arrow = Arrow(origin, b_end, color=b_color, buff=0, stroke_width=4,
                        max_tip_length_to_length_ratio=0.12)

        a_label = MathTex(r"\mathbf{a}").set_color(a_color).scale(0.9)
        a_label.next_to(a_end, UP + LEFT * 0.5, buff=0.1)
        b_label = MathTex(r"\mathbf{b}").set_color(b_color).scale(0.9)
        b_label.next_to(b_end, UP, buff=0.1)

        # alpha * b (a small fraction of b, drawn from the tip of a)
        alpha = 0.5
        ab_end = a_end + alpha * (b_end - origin)
        ab_arrow = Arrow(a_end, ab_end, color=b_color, buff=0,
                         stroke_width=2.5,
                         max_tip_length_to_length_ratio=0.18)
        ab_label = MathTex(r"\alpha \mathbf{b}").set_color(b_color).scale(0.7)
        ab_label.next_to(ab_arrow, UP, buff=0.05)

        # New vector a + alpha*b (from origin)
        new_a_arrow = Arrow(origin, ab_end, color=new_a_color, buff=0, stroke_width=4,
                            max_tip_length_to_length_ratio=0.12)
        new_a_label = MathTex(r"\mathbf{a} + \alpha \mathbf{b}").set_color(new_a_color).scale(0.85)
        new_a_label.next_to(ab_end, UP + RIGHT * 0.3, buff=0.15)

        # Origin dot
        origin_dot = Dot(origin, color=light, radius=0.05)

        # Title and subtitle
        title = Tex(r"Adding $\alpha \mathbf{b}$ to $\mathbf{a}$ rotates $\mathbf{a}$ toward $\mathbf{b}$").set_color(light).scale(0.7)
        title.to_edge(UP, buff=0.5)

        # Annotation about the dot product
        annotation = MathTex(
            r"\Rightarrow \quad \left(\mathbf{a} + \alpha \mathbf{b}\right)^{\intercal} \mathbf{b}",
            r"=",
            r"\mathbf{a}^{\intercal}\mathbf{b} + \alpha \left\Vert \mathbf{b} \right\Vert^2",
            r"\geq",
            r"\mathbf{a}^{\intercal}\mathbf{b}",
        ).set_color(light).scale(0.65)
        annotation.to_edge(DOWN, buff=2)

        self.add(
            title, origin_dot,
            a_arrow, a_label,
            b_arrow, b_label,
            ab_arrow, ab_label,
            new_a_arrow, new_a_label,
            annotation,
        )

# Render command (Windows):
#   manim -qh -s dot_product_intuition_manim_png.py DotProductIntuition
#
# Output PNG location:
#   media\images\dot_product_intuition_manim_png\DotProductIntuition.png
#
# Embed in Quarto using:
#   ![Caption](images/dot_product_intuition.png){#fig-dot_product_intuition fig-align="center" width=60% .invert}
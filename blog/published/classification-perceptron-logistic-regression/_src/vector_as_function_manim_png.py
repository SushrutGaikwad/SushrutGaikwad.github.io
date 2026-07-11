# Setup (one-time, on Windows):
#   1. Install Python 3.9+
#   2. Install MiKTeX
#   3. Install FFmpeg and add it to PATH
#   4. pip install manim
#
# This script generates a STATIC PNG showing the analogy between a vector
# in R^3 (left, as three components) and a function defined on {1, 2, 3}
# (right, as three bars). The two are literally the same object presented
# differently.

from manim import *
import numpy as np


class VectorAsFunction(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        light = "#e6e6e6"
        bar_color = "#1abc9c"
        vec_color = "#3498db"

        # Left panel: vector as a column of numbers
        components = [2.0, 3.5, 1.5]

        # The matrix display
        v_matrix = MathTex(
            r"\mathbf{v} = ",
            r"\begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}",
            r"=",
            r"\begin{bmatrix} 2.0 \\ 3.5 \\ 1.5 \end{bmatrix}",
        ).set_color(light).scale(0.85)
        v_matrix.shift(LEFT * 4.2 + UP * 0.3)

        left_title = Tex(r"Vector $\mathbf{v} \in \mathbb{R}^3$").set_color(light).scale(0.7)
        left_title.next_to(v_matrix, UP, buff=1.5)

        # Right panel: bar plot of the same numbers, viewed as a function
        # f: {1, 2, 3} -> R with f(1)=2.0, f(2)=3.5, f(3)=1.5
        bar_axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4.5, 1],
            x_length=4.5,
            y_length=3.5,
            axis_config={"color": light, "include_tip": True, "stroke_width": 1.5},
            x_axis_config={"include_numbers": True, "decimal_number_config": {"num_decimal_places": 0}},
            y_axis_config={"include_numbers": True, "decimal_number_config": {"num_decimal_places": 0}},
        ).shift(RIGHT * 3.0 + UP * 0.0)
        bar_x_label = Tex(r"input $i$").set_color(light).scale(0.55)
        bar_x_label.next_to(bar_axes.x_axis, DOWN, buff=0.3)
        bar_y_label = Tex(r"$\mathbf{v}(i)$").set_color(light).scale(0.55)
        bar_y_label.next_to(bar_axes.y_axis, UP, buff=0.2)

        # Bars at i = 1, 2, 3
        bars = VGroup()
        for i, val in enumerate(components, start=1):
            bottom = bar_axes.coords_to_point(i, 0)
            top = bar_axes.coords_to_point(i, val)
            # Make a thin rectangle
            bar_height = top[1] - bottom[1]
            bar_width = 0.4
            bar_rect = Rectangle(
                width=bar_width,
                height=bar_height,
                color=bar_color,
                fill_color=bar_color,
                fill_opacity=0.7,
                stroke_width=2,
            )
            bar_rect.move_to((bottom + top) / 2)
            bars.add(bar_rect)
            # Label at top of bar
            value_label = MathTex(f"{val:.1f}").set_color(light).scale(0.45)
            value_label.next_to(top, UP, buff=0.1)
            bars.add(value_label)

        right_title = Tex(r"Function $\mathbf{v} : \{1,2,3\} \to \mathbb{R}$").set_color(light).scale(0.7)
        right_title.next_to(bar_axes, UP, buff=0.7)

        # Big-picture caption
        caption = Tex(
            r"The same three numbers, viewed as a vector (left) or as a function on three inputs (right)."
        ).set_color(light).scale(0.6)
        caption.to_edge(DOWN, buff=0.3)

        # Connector arrow showing they are the same
        connector = MathTex(r"\Longleftrightarrow").set_color(light).scale(1.0)
        connector.move_to(ORIGIN + UP * 0.3 + LEFT * 0.9)

        self.add(
            v_matrix, left_title,
            bar_axes, bar_x_label, bar_y_label, bars,
            right_title,
            connector,
            caption,
        )


# Render command (Windows):
#   manim -qh -s vector_as_function_manim_png.py VectorAsFunction
#
# Output PNG location:
#   media\images\vector_as_function_manim_png\VectorAsFunction.png
#
# Embed in Quarto using:
#   ![Caption](images/vector_as_function.png){#fig-vector_as_function fig-align="center" width=85% .invert}
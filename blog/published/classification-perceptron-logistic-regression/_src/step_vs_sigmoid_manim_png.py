# Setup (one-time, on Windows):
#   1. Install Python 3.9+
#   2. Install MiKTeX
#   3. Install FFmpeg and add it to PATH
#   4. pip install manim
#
# This script generates a STATIC PNG with two side-by-side panels showing
# the threshold function (left, perceptron) and the sigmoid function
# (right, logistic regression).

from manim import *
import numpy as np

class StepVsSigmoid(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        light = "#e6e6e6"
        curve_color = "#1abc9c"

        # Common axis ranges
        x_range = [-6, 6, 2]
        y_range = [-0.2, 1.3, 0.5]

        # LEFT: hard threshold
        left_axes = Axes(
            x_range=x_range,
            y_range=y_range,
            x_length=5.0,
            y_length=3.5,
            axis_config={"color": light, "include_tip": True, "stroke_width": 1.5},
            y_axis_config={"include_numbers": True, "decimal_number_config": {"num_decimal_places": 0}},
        ).shift(LEFT * 3.4)
        left_title = Tex(r"Hard threshold (perceptron)").set_color(light).scale(0.65)
        left_title.next_to(left_axes, UP, buff=1)
        left_xlabel = Tex(r"$z$").set_color(light).scale(0.6).next_to(left_axes.x_axis, RIGHT, buff=0.15)
        left_ylabel = Tex(r"$g(z)$").set_color(light).scale(0.6).next_to(left_axes.y_axis, UP, buff=0.15)

        # Hard threshold: g(z) = 0 for z < 0, g(z) = 1 for z >= 0
        # Draw two segments + a vertical jump
        left_part = Line(
            left_axes.coords_to_point(-6, 0),
            left_axes.coords_to_point(0, 0),
            color=curve_color, stroke_width=4,
        )
        right_part = Line(
            left_axes.coords_to_point(0, 1),
            left_axes.coords_to_point(6, 1),
            color=curve_color, stroke_width=4,
        )
        # Open / closed circles at the discontinuity
        open_circle = Circle(radius=0.07, color=curve_color, stroke_width=3).move_to(left_axes.coords_to_point(0, 0))
        closed_circle = Dot(left_axes.coords_to_point(0, 1), color=curve_color, radius=0.08)

        left_formula = MathTex(
            r"g(z) = \begin{cases} 1 & z \geq 0 \\ 0 & z < 0 \end{cases}"
        ).set_color(light).scale(0.6)
        left_formula.next_to(left_axes, DOWN, buff=0.4)

        # RIGHT: sigmoid
        right_axes = Axes(
            x_range=x_range,
            y_range=y_range,
            x_length=5.0,
            y_length=3.5,
            axis_config={"color": light, "include_tip": True, "stroke_width": 1.5},
            y_axis_config={"include_numbers": True, "decimal_number_config": {"num_decimal_places": 0}},
        ).shift(RIGHT * 3.4)
        right_title = Tex(r"Sigmoid (logistic regression)").set_color(light).scale(0.65)
        right_title.next_to(right_axes, UP, buff=1)
        right_xlabel = Tex(r"$z$").set_color(light).scale(0.6).next_to(right_axes.x_axis, RIGHT, buff=0.15)
        right_ylabel = Tex(r"$g(z)$").set_color(light).scale(0.6).next_to(right_axes.y_axis, UP, buff=0.15)

        # Sigmoid curve
        sigmoid_curve = right_axes.plot(
            lambda z: 1.0 / (1.0 + np.exp(-z)),
            x_range=[-6, 6],
            color=curve_color,
            stroke_width=4,
        )
        # Reference dashed line at y=0.5
        half_line = DashedLine(
            right_axes.coords_to_point(-6, 0.5),
            right_axes.coords_to_point(6, 0.5),
            color=light,
            stroke_width=1,
            dash_length=0.08,
        )
        half_label = Tex(r"$0.5$").set_color(light).scale(0.5)
        half_label.next_to(right_axes.coords_to_point(6, 0.5), RIGHT, buff=0.1)

        right_formula = MathTex(
            r"g(z) = \frac{1}{1 + e^{-z}}"
        ).set_color(light).scale(0.6)
        right_formula.next_to(right_axes, DOWN, buff=0.4)

        self.add(
            left_axes, left_title, left_xlabel, left_ylabel,
            left_part, right_part, open_circle, closed_circle, left_formula,
            right_axes, right_title, right_xlabel, right_ylabel,
            sigmoid_curve, half_line, half_label, right_formula,
        )

# Render command (Windows):
#   manim -qh -s step_vs_sigmoid_manim_png.py StepVsSigmoid
#
# Output PNG location:
#   media\images\step_vs_sigmoid_manim_png\StepVsSigmoid.png
#
# Embed in Quarto using:
#   ![Caption](images/step_vs_sigmoid.png){#fig-step_vs_sigmoid fig-align="center" width=85% .invert}
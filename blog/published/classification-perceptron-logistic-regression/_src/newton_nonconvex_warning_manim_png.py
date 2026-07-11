# Setup (one-time, on Windows):
#   1. Install Python 3.9+
#   2. Install MiKTeX
#   3. Install FFmpeg and add it to PATH
#   4. pip install manim
#
# This script generates a STATIC PNG with two side-by-side panels
# illustrating the danger of applying Newton's method to a non-convex
# function. Left panel: a non-convex loss l(theta) with multiple
# stationary points (a local max and two local mins). Right panel: its
# derivative l'(theta), which has multiple roots; Newton's method finds
# whichever root is nearest to the initialization, regardless of whether
# it corresponds to a max or a min.

from manim import *
import numpy as np


class NewtonNonconvexWarning(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        light = "#e6e6e6"
        loss_color = "#1abc9c"
        deriv_color = "#f1c40f"
        max_color = "#e74c3c"
        min_color = "#3498db"

        # Define a function with two minima and a maximum between them
        # f(theta) = (theta - 2)^2 * (theta + 2)^2 / 16   (two minima at +-2, max at 0)
        # f'(theta) = ?   This is symmetric and yields a clean visual.
        # Use sympy mentally:
        # f(theta) = ((theta^2 - 4)^2) / 16
        # f'(theta) = (2*(theta^2 - 4)*2*theta)/16 = (theta^2 - 4)*theta / 4
        # roots of f': theta = 0, theta = +-2.

        def f(t):
            return (t**2 - 4)**2 / 16

        def fp(t):
            return (t**2 - 4) * t / 4

        # LEFT panel: l(theta)
        left_axes = Axes(
            x_range=[-3.5, 3.5, 1],
            y_range=[-0.5, 2.5, 1],
            x_length=5.0,
            y_length=3.5,
            axis_config={"color": light, "include_tip": True, "stroke_width": 1.5},
        ).shift(LEFT * 3.4 + DOWN * 0.3)
        left_title = Tex(r"Non-convex loss $\ell(\theta)$").set_color(light).scale(0.65)
        left_title.next_to(left_axes, UP, buff=0.45)
        left_xlabel = Tex(r"$\theta$").set_color(light).scale(0.55).next_to(left_axes.x_axis, RIGHT, buff=0.1)
        left_ylabel = Tex(r"$\ell(\theta)$").set_color(light).scale(0.55).next_to(left_axes.y_axis, UP, buff=0.1)

        left_curve = left_axes.plot(f, x_range=[-3.3, 3.3], color=loss_color, stroke_width=3)

        # Mark the three stationary points on the loss curve
        left_max_pt = Dot(left_axes.coords_to_point(0, f(0)), color=max_color, radius=0.08)
        left_max_lbl = Tex(r"local max").set_color(max_color).scale(0.45)
        left_max_lbl.next_to(left_max_pt, UP, buff=0.1)

        left_min1_pt = Dot(left_axes.coords_to_point(-2, f(-2)), color=min_color, radius=0.08)
        left_min2_pt = Dot(left_axes.coords_to_point(2, f(2)), color=min_color, radius=0.08)
        left_min_lbl = Tex(r"local minima").set_color(min_color).scale(0.45)
        left_min_lbl.next_to(left_axes.coords_to_point(0, -0.3), DOWN, buff=0.3)

        # RIGHT panel: l'(theta)
        right_axes = Axes(
            x_range=[-3.5, 3.5, 1],
            y_range=[-3, 3, 1],
            x_length=5.0,
            y_length=3.5,
            axis_config={"color": light, "include_tip": True, "stroke_width": 1.5},
        ).shift(RIGHT * 3.4 + DOWN * 0.3)
        right_title = Tex(r"Derivative $\ell^{\prime}(\theta)$").set_color(light).scale(0.65)
        right_title.next_to(right_axes, UP, buff=0.45)
        right_xlabel = Tex(r"$\theta$").set_color(light).scale(0.55).next_to(right_axes.x_axis, RIGHT, buff=0.1)
        right_ylabel = Tex(r"$\ell^{\prime}(\theta)$").set_color(light).scale(0.55).next_to(right_axes.y_axis, UP, buff=0.1)

        right_curve = right_axes.plot(fp, x_range=[-3.3, 3.3], color=deriv_color, stroke_width=3)

        # Mark the three roots of fp on the right axis
        for tt, color, label_text, label_dir in [
            (-2, min_color, r"min root", DR),
            (0, max_color, r"max root", UP),
            (2, min_color, r"min root", DL),
        ]:
            dot = Dot(right_axes.coords_to_point(tt, 0), color=color, radius=0.08)
            self.add(dot)
            label = Tex(label_text).set_color(color).scale(0.45)
            label.next_to(dot, label_dir, buff=0.1)
            self.add(label)

        # Caption explaining the warning
        caption = Tex(
            r"Newton's method applied to $\ell^{\prime}$ finds the \emph{nearest} root to the initialization, ",
            r"which could correspond to a maximum, minimum, or saddle point of $\ell$."
        ).set_color(light).scale(0.50)
        caption.to_edge(DOWN, buff=0.2)

        self.add(
            left_axes, left_title, left_xlabel, left_ylabel, left_curve,
            left_max_pt, left_max_lbl, left_min1_pt, left_min2_pt, left_min_lbl,
            right_axes, right_title, right_xlabel, right_ylabel, right_curve,
            caption,
        )

# Render command (Windows):
#   manim -qh -s newton_nonconvex_warning_manim_png.py NewtonNonconvexWarning
#
# Output PNG location:
#   media\images\newton_nonconvex_warning_manim_png\NewtonNonconvexWarning.png
#
# Embed in Quarto using:
#   ![Caption](images/newton_nonconvex_warning.png){#fig-newton_nonconvex_warning fig-align="center" width=85% .invert}
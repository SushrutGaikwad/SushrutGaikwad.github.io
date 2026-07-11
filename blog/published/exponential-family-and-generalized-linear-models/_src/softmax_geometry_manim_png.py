# Runs in the root uv environment (Manim CE + TeX Live).
#
# STATIC PNG: the geometric intuition behind softmax / multi-class classification.
# Each class c gets its own parameter vector theta_c. An example x is scored by
# every theta_c^T x, and we predict the class whose vector x aligns with best
# (largest dot product = smallest angle). The theta_c arrows point toward their
# own class clusters.

from manim import *
import numpy as np


class SoftmaxGeometry(Scene):
    def construct(self):
        self.camera.background_color = "#222222"
        light = "#e6e6e6"

        rng = np.random.default_rng(3)

        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=6.2,
            y_length=6.0,
            axis_config={"color": "#777777", "include_tip": True, "stroke_width": 1.6},
        ).shift(DOWN * 0.45)
        x_lab = MathTex(r"x_1", color=light).scale(0.6).next_to(axes.x_axis, RIGHT, buff=0.15)
        y_lab = MathTex(r"x_2", color=light).scale(0.6).next_to(axes.y_axis.get_top(), LEFT, buff=0.2)

        # three class clusters, evenly spaced around the origin.
        # "lpos" places each theta label BESIDE its arrow shaft, in the empty
        # space between the origin and the cluster, so labels never sit on points.
        class_cfg = [
            {"angle": 90, "color": "#e67e22", "name": r"\boldsymbol{\theta}_1", "lpos": (-0.6, 1.35)},
            {"angle": 210, "color": "#5dade2", "name": r"\boldsymbol{\theta}_2", "lpos": (-0.8, -1.25)},
            {"angle": 330, "color": "#af7ac5", "name": r"\boldsymbol{\theta}_3", "lpos": (0.8, -1.25)},
        ]

        dots = VGroup()
        arrows = VGroup()
        labels = VGroup()
        for cfg in class_cfg:
            ang = np.deg2rad(cfg["angle"])
            center = 2.5 * np.array([np.cos(ang), np.sin(ang)])
            # cluster of points
            pts = center + rng.normal(0, 0.38, size=(7, 2))
            for p in pts:
                dots.add(Dot(axes.c2p(p[0], p[1]), radius=0.06, color=cfg["color"]))
            # theta_c arrow from origin toward the cluster direction
            direction = center / np.linalg.norm(center)
            tip = 2.15 * direction
            arr = Arrow(
                axes.c2p(0, 0), axes.c2p(tip[0], tip[1]), buff=0,
                color=cfg["color"], stroke_width=5, max_tip_length_to_length_ratio=0.16,
            )
            arrows.add(arr)
            lab = MathTex(cfg["name"], color=cfg["color"]).scale(0.75)
            lab.move_to(axes.c2p(*cfg["lpos"]))
            labels.add(lab)

        title = Tex(
            r"One $\boldsymbol{\theta}_c$ per class; predict $\arg\max_c\ \boldsymbol{\theta}_c^\top \mathbf{x}$",
            color=light,
        ).scale(0.62)
        title.to_edge(UP, buff=0.2)

        self.add(axes, x_lab, y_lab, dots, arrows, labels, title)


# Render (from repo root):
#   uv run manim -qh -s blog/published/exponential-family-and-generalized-linear-models/_src/softmax_geometry_manim_png.py SoftmaxGeometry
# Output PNG: media/images/softmax_geometry_manim_png/SoftmaxGeometry.png
#   -> move to images/softmax_geometry.png
# Embed:
#   ![Caption](images/softmax_geometry.png){#fig-softmax_geometry fig-align="center" width=75%}

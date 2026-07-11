# Runs in the root uv environment (Manim CE + TeX Live).
#
# STATIC PNG: the GLM "pipeline" / data-generating process, laid out as a chain
#   x_i  --(theta^T x_i)-->  eta_i  --(g)-->  mean param  --(sample)-->  y_i
# annotated with the three kinds of parameters (global theta, per-example eta,
# mean parameter = the prediction). This one diagram anchors the sections on
# constructing GLMs, the three-parameter picture, and the zoo of GLMs.

from manim import *


class GLMPipeline(Scene):
    def construct(self):
        self.camera.background_color = "#222222"
        light = "#e6e6e6"
        theta_col = "#2ecc71"
        eta_col = "#1abc9c"
        mean_col = "#f1c40f"

        def boxed(mob, stroke=light, fill="#2d2d2d"):
            box = RoundedRectangle(
                corner_radius=0.15,
                width=mob.width + 0.55,
                height=mob.height + 0.45,
            ).set_stroke(stroke, width=2.2).set_fill(fill, opacity=1.0)
            mob.move_to(box)
            return VGroup(box, mob)

        # --- the four nodes ---
        n_x = boxed(MathTex(r"\mathbf{x}_i \in \mathbb{R}^d", color=light).scale(0.7))
        n_eta = boxed(MathTex(r"\eta_i", color=eta_col).scale(0.8), stroke=eta_col)
        n_mean = boxed(
            MathTex(r"\mu_i \;/\; \phi_i \;/\; \lambda_i", color=mean_col).scale(0.7),
            stroke=mean_col,
        )
        n_y = boxed(MathTex(r"y_i", color=light).scale(0.8))

        nodes = VGroup(n_x, n_eta, n_mean, n_y).arrange(RIGHT, buff=1.55)
        nodes.scale_to_fit_width(12.2).move_to(ORIGIN + UP * 0.5)

        # --- arrows between nodes ---
        def connect(a, b):
            return Arrow(
                a.get_right(), b.get_left(), buff=0.15,
                stroke_width=3.5, color=light, max_tip_length_to_length_ratio=0.18,
            )

        ar1 = connect(n_x, n_eta)
        ar2 = connect(n_eta, n_mean)
        ar3 = connect(n_mean, n_y)

        # --- arrow labels (above) ---
        lab1 = MathTex(r"\eta_i = \boldsymbol{\theta}^\top \mathbf{x}_i", color=theta_col).scale(0.62)
        lab1.next_to(ar1, UP, buff=0.18)
        lab2 = MathTex(r"g", color=light).scale(0.7).next_to(ar2, UP, buff=0.12)
        lab3 = Tex(r"sample", color=light).scale(0.55).next_to(ar3, UP, buff=0.18)

        # --- annotations (below) ---
        an_x = Tex(r"input", color="#bbbbbb").scale(0.5).next_to(n_x, DOWN, buff=0.3)
        an_theta = Tex(r"$\boldsymbol{\theta}$ is \emph{global}", color=theta_col).scale(0.5)
        an_theta.next_to(lab1, UP, buff=0.22)
        an_eta = Tex(r"natural parameter\\(one per example)", color=eta_col).scale(0.48)
        an_eta.next_to(n_eta, DOWN, buff=0.3)
        an_mean = Tex(r"mean parameter\\$= h_{\boldsymbol{\theta}}(\mathbf{x}_i) = \mathbb{E}[y_i \mid \mathbf{x}_i]$",
                      color=mean_col).scale(0.48)
        an_mean.next_to(n_mean, DOWN, buff=0.3)
        an_y = Tex(r"output", color="#bbbbbb").scale(0.5).next_to(n_y, DOWN, buff=0.3)

        # --- the distribution note spanning eta -> mean -> y ---
        dist_note = Tex(
            r"fixed choice of $\left(T, a, b\right)$ = the exponential-family distribution",
            color=light,
        ).scale(0.52)
        brace_group = VGroup(n_eta, n_mean, n_y)
        brace = Brace(brace_group, UP, color="#888888")
        dist_note.next_to(brace, UP, buff=0.15)

        everything = VGroup(
            nodes, ar1, ar2, ar3,
            lab1, lab2, lab3,
            an_x, an_theta, an_eta, an_mean, an_y,
            brace, dist_note,
        )
        everything.move_to(ORIGIN).scale_to_fit_width(13.2)
        self.add(everything)


# Render (from repo root):
#   uv run manim -qh -s blog/published/exponential-family-and-generalized-linear-models/_src/glm_pipeline_manim_png.py GLMPipeline
# Output PNG: media/images/glm_pipeline_manim_png/GLMPipeline.png
#   -> move to images/glm_pipeline.png
# Embed:
#   ![Caption](images/glm_pipeline.png){#fig-glm_pipeline fig-align="center" width=100%}

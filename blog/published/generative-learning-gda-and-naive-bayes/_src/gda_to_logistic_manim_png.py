# Runs via the root uv environment.
#
# STATIC PNG (Manim): the one-directional relationship between GDA and logistic
# regression. A GDA model with shared covariance ALWAYS implies a logistic
# posterior, but a logistic posterior does NOT imply the data came from a GDA
# model. The forward arrow is solid; the reverse arrow is dashed and struck out.

from manim import (
    Arrow,
    Cross,
    MathTex,
    Scene,
    SurroundingRectangle,
    Tex,
    VGroup,
    UP,
    DOWN,
    LEFT,
    RIGHT,
)

BG = "#222222"
LIGHT = "#e6e6e6"
GREEN = "#2ecc71"
RED = "#e74c3c"


class GdaToLogistic(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ---- Left box: the GDA model (shared covariance) ----
        gda_title = Tex(r"GDA model (shared $\boldsymbol{\Sigma}$)", color=LIGHT).scale(0.72)
        gda_body = MathTex(
            r"y &\sim \text{Bernoulli}(\phi)\\"
            r"\mathbf{x} \mid y=0 &\sim \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma})\\"
            r"\mathbf{x} \mid y=1 &\sim \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma})",
            color=LIGHT,
        ).scale(0.72)
        gda_body.next_to(gda_title, DOWN, buff=0.35)
        gda_group = VGroup(gda_title, gda_body)
        gda_box = SurroundingRectangle(gda_group, color=GREEN, buff=0.35, corner_radius=0.12)
        left = VGroup(gda_box, gda_group).to_edge(LEFT, buff=0.7)

        # ---- Right box: the logistic posterior ----
        log_title = Tex(r"Logistic posterior", color=LIGHT).scale(0.72)
        log_body = MathTex(
            r"p(y=1 \mid \mathbf{x}) = \frac{1}{1 + e^{-\boldsymbol{\theta}^{\top}\mathbf{x}}}",
            color=LIGHT,
        ).scale(0.78)
        log_body.next_to(log_title, DOWN, buff=0.35)
        log_note = Tex(r"($\boldsymbol{\theta}$ a function of $\phi, \boldsymbol{\Sigma}, \boldsymbol{\mu}_0, \boldsymbol{\mu}_1$)", color=LIGHT).scale(0.5)
        log_note.next_to(log_body, DOWN, buff=0.3)
        log_group = VGroup(log_title, log_body, log_note)
        log_box = SurroundingRectangle(log_group, color=LIGHT, buff=0.35, corner_radius=0.12)
        right = VGroup(log_box, log_group).to_edge(RIGHT, buff=0.7)

        # ---- Forward arrow: always holds ----
        fwd = Arrow(left.get_right(), right.get_left(), color=GREEN, buff=0.25)
        fwd.shift(UP * 0.9)
        fwd_label = Tex(r"always", color=GREEN).scale(0.6).next_to(fwd, UP, buff=0.12)

        # ---- Reverse arrow: does not hold ----
        rev = Arrow(right.get_left(), left.get_right(), color=RED, buff=0.25)
        rev.shift(DOWN * 0.9)
        rev.set_stroke(width=4)
        cross = Cross(rev, stroke_color=RED, stroke_width=6).scale(0.28)
        rev_label = Tex(r"not always", color=RED).scale(0.6).next_to(rev, DOWN, buff=0.12)

        self.add(left, right, fwd, fwd_label, rev, cross, rev_label)


# Render (from repo root):
#   uv run manim -qh -s blog/published/generative-learning-gda-and-naive-bayes/_src/gda_to_logistic_manim_png.py GdaToLogistic
# Output: media/images/gda_to_logistic_manim_png/GdaToLogistic.png
# Move it to blog/published/generative-learning-gda-and-naive-bayes/images/gda_to_logistic.png
# Embed:
#   ![Caption](images/gda_to_logistic.png){#fig-gda_to_logistic fig-align="center" width=95%}

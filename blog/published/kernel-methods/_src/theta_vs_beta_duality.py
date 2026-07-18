# Runs via the root uv environment with uv run (see render command at the bottom).
from manim import *

# Load upgreek so upright-bold Greek vectors (\boldsymbol{\uptheta}, \boldsymbol{\upbeta},
# \boldsymbol{\upphi}) match the blog's notation convention.
_TEMPLATE = TexTemplate()
_TEMPLATE.add_to_preamble(r"\usepackage{upgreek}")
config.tex_template = _TEMPLATE


class ThetaVsBeta(Scene):
    """The dual representation at the heart of the kernel trick, shown as two
    column vectors side by side:
      LEFT  theta in R^p : one weight per FEATURE (p can be huge, even infinite),
      RIGHT beta  in R^n : one weight per training EXAMPLE (n is always finite),
    tied together by  theta = sum_i beta_i phi(x_i)  underneath."""

    def construct(self):
        self.camera.background_color = "#222222"

        light = "#e6e6e6"
        theta_color = "#f1c40f"   # yellow
        beta_color = "#5dade2"    # blue

        title = Tex(r"Two ways to hold the \emph{same} model",
                    font_size=40, color=light)

        left = self.make_panel(
            vec=r"\boldsymbol{\uptheta} = \begin{bmatrix} \theta_1 \\ \theta_2 \\ \vdots \\ \theta_p \end{bmatrix}",
            heading=r"one weight per \textbf{feature}",
            note=r"$p = \dim$ of feature space\\(can be enormous, even $\infty$)",
            accent=theta_color,
        )
        right = self.make_panel(
            vec=r"\boldsymbol{\upbeta} = \begin{bmatrix} \beta_1 \\ \beta_2 \\ \vdots \\ \beta_n \end{bmatrix}",
            heading=r"one weight per \textbf{example}",
            note=r"$n = $ number of training\\examples (always finite)",
            accent=beta_color,
        )

        panels = VGroup(left, right).arrange(RIGHT, buff=2.0)

        # The relation that ties the two representations together. Color the
        # theta and the beta_i to match their panels (yellow and blue), so the
        # boxed equation visibly links both sides; the rest stays light.
        relation = MathTex(
            r"\boldsymbol{\uptheta}",
            r"\;=\; \sum_{i=1}^{n}",
            r"\beta_i",
            r"\, \boldsymbol{\upphi}(\mathbf{x}_i)",
            font_size=44, color=light,
        )
        relation[0].set_color(theta_color)
        relation[2].set_color(beta_color)
        box = SurroundingRectangle(relation, color="#777777", buff=0.25,
                                   corner_radius=0.12)
        relation_group = VGroup(box, relation)

        # Stack everything and scale to fit the frame with margins.
        content = VGroup(title, panels, relation_group).arrange(DOWN, buff=0.55)
        content.scale_to_fit_height(7.2).move_to(ORIGIN)

        self.add(content)

    def make_panel(self, vec, heading, note, accent):
        head = Tex(heading, font_size=30, color=accent)
        vector = MathTex(vec, font_size=52, color=accent)
        vector.next_to(head, DOWN, buff=0.3)
        note_tex = Tex(note, font_size=25, color="#b0b0b0")
        note_tex.next_to(vector, DOWN, buff=0.35)
        return VGroup(head, vector, note_tex)


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/kernel-methods/_src/theta_vs_beta_duality.py ThetaVsBeta
# Manim writes the PNG to media\images\theta_vs_beta_duality\ThetaVsBeta.png.
# Move it to blog/published/kernel-methods/images/theta_vs_beta_duality.png
# Embed in the .qmd with:
#   ![...](images/theta_vs_beta_duality.png){#fig-theta_vs_beta_duality fig-align="center" width=85%}

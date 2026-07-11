# Runs via the root uv environment with `uv run`.
"""Orthonormal-expansion analogy figure for Post 12.

Left: a vector in an orthonormal basis, coordinates are dot products.
Right: a function in the Fourier basis, coefficients are integrals.
All text uses MathTex/Tex (LaTeX / Computer Modern); bold uses \\mathbf.
"""

from manim import *

LIGHT = "#e6e6e6"
BLUE = "#5dade2"
ORANGE = "#f5b041"
GRAY = "#9aa0a6"


class FourierBasis(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        title = Tex(r"Same idea: expand in an orthonormal basis", color=LIGHT).scale(0.9)
        title.to_edge(UP, buff=0.5)
        self.add(title)

        # Left column: vectors.
        left_head = Tex(r"Vectors in $\mathbb{R}^n$", color=BLUE).scale(0.8)
        left_body = VGroup(
            MathTex(r"\mathbf{v} = c_1\mathbf{q}_1 + \cdots + c_n\mathbf{q}_n", color=LIGHT),
            MathTex(r"c_i = \mathbf{q}_i^{\intercal}\mathbf{v}", color=LIGHT),
        ).arrange(DOWN, buff=0.5).scale(0.85)
        left = VGroup(left_head, left_body).arrange(DOWN, buff=0.5)
        left.move_to(LEFT * 3.4 + DOWN * 0.4)
        self.add(left)

        # Right column: functions.
        right_head = Tex(r"Functions", color=ORANGE).scale(0.8)
        right_body = VGroup(
            MathTex(r"f(x) = a_0 + \sum_n a_n\cos nx + b_n\sin nx", color=LIGHT).scale(0.8),
            MathTex(r"a_n = \frac{1}{\pi}\int_0^{2\pi} f(x)\cos nx \, dx", color=LIGHT).scale(0.9),
        ).arrange(DOWN, buff=0.5)
        right = VGroup(right_head, right_body).arrange(DOWN, buff=0.5)
        right.move_to(RIGHT * 3.4 + DOWN * 0.4)
        self.add(right)

        # Divider.
        self.add(Line(UP * 1.8, DOWN * 2.6, color=GRAY, stroke_width=2))


# Render the last frame only as a PNG (use a unique media dir), from the repo root:
#   uv run manim -qh -s --media_dir media_markov blog/published/markov-matrices-and-fourier-series/_src/fourier_basis.py FourierBasis
# Move media_markov/images/fourier_basis/FourierBasis*.png to
#   blog/published/markov-matrices-and-fourier-series/images/fourier_basis.png and embed with:
#   ![Caption.](images/fourier_basis.png){#fig-fourier_basis fig-align="center" width=95%}

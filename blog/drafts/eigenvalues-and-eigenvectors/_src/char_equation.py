# Runs via the root uv environment with `uv run`.
"""Characteristic-equation pipeline for Post 10.

Walks A - lambda I -> det = 0 -> eigenvalues -> eigenvectors for the swap matrix
[[0,1],[1,0]]. All text uses MathTex/Tex (LaTeX / Computer Modern); bold uses \\mathbf.
"""

from manim import *

LIGHT = "#e6e6e6"
ORANGE = "#f5b041"
BLUE = "#5dade2"


class CharEquation(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        title = Tex(r"The characteristic equation for $\mathbf{A}=\left(\begin{smallmatrix}0&1\\1&0\end{smallmatrix}\right)$",
                    color=LIGHT).scale(0.8)
        title.to_edge(UP, buff=0.5)
        self.add(title)

        lines = VGroup(
            MathTex(r"\mathbf{A}-\lambda\mathbf{I} = \begin{pmatrix} -\lambda & 1 \\ 1 & -\lambda \end{pmatrix}", color=LIGHT),
            MathTex(r"\det(\mathbf{A}-\lambda\mathbf{I}) = \lambda^2 - 1", color=LIGHT),
            MathTex(r"\lambda^2 - 1 = 0 \quad\Longrightarrow\quad \lambda = 1,\ -1", color=ORANGE),
            MathTex(r"\lambda=1:\ \mathbf{x}=(1,1) \qquad \lambda=-1:\ \mathbf{x}=(1,-1)", color=BLUE),
        ).scale(0.9)
        lines.arrange(DOWN, buff=0.55).next_to(title, DOWN, buff=0.6)
        self.add(lines)

        for a, b in zip(lines[:-1], lines[1:]):
            self.add(Arrow(a.get_bottom(), b.get_top(), buff=0.12, color="#9aa0a6", stroke_width=3))


# Render the last frame only as a PNG (use a unique media dir), from the repo root:
#   uv run manim -qh -s --media_dir media_eig blog/published/eigenvalues-and-eigenvectors/_src/char_equation.py CharEquation
# Move media_eig/images/char_equation/CharEquation*.png to
#   blog/published/eigenvalues-and-eigenvectors/images/char_equation.png and embed with:
#   ![Caption.](images/char_equation.png){#fig-char_equation fig-align="center" width=95%}

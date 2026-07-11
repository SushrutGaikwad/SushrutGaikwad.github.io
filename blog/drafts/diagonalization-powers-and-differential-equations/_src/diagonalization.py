# Runs via the root uv environment with `uv run`.
"""Diagonalization / powers figure for Post 11.

Shows A = S Lambda S^{-1} and A^k = S Lambda^k S^{-1}, with Lambda^k just
powering the diagonal. All text uses MathTex/Tex (LaTeX / Computer Modern);
bold uses \\mathbf.
"""

from manim import *

LIGHT = "#e6e6e6"
ORANGE = "#f5b041"
BLUE = "#5dade2"


class Diagonalization(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        eq1 = MathTex(r"\mathbf{A} = \mathbf{S}\,\boldsymbol{\Lambda}\,\mathbf{S}^{-1}", color=LIGHT).scale(1.1)
        eq2 = MathTex(r"\mathbf{A}^k = \mathbf{S}\,\boldsymbol{\Lambda}^k\,\mathbf{S}^{-1}", color=LIGHT).scale(1.1)
        eqs = VGroup(eq1, eq2).arrange(DOWN, buff=0.7).to_edge(UP, buff=1.0)
        self.add(eqs)

        note = Tex(r"the eigenvectors $\mathbf{S}$ stay fixed; only the eigenvalues get powered",
                   color=LIGHT).scale(0.7)
        note.next_to(eqs, DOWN, buff=0.6)
        self.add(note)

        Lam = MathTex(r"\boldsymbol{\Lambda} = \begin{pmatrix}\lambda_1 & 0\\ 0 & \lambda_2\end{pmatrix}",
                      color=ORANGE).scale(0.95)
        Lamk = MathTex(r"\boldsymbol{\Lambda}^k = \begin{pmatrix}\lambda_1^{\,k} & 0\\ 0 & \lambda_2^{\,k}\end{pmatrix}",
                       color=ORANGE).scale(0.95)
        mats = VGroup(Lam, Lamk).arrange(RIGHT, buff=1.4)
        mats.next_to(note, DOWN, buff=0.7)
        self.add(mats)
        self.add(Arrow(Lam.get_right(), Lamk.get_left(), buff=0.3, color="#9aa0a6"))


# Render the last frame only as a PNG (use a unique media dir), from the repo root:
#   uv run manim -qh -s --media_dir media_diag blog/published/diagonalization-powers-and-differential-equations/_src/diagonalization.py Diagonalization
# Move media_diag/images/diagonalization/Diagonalization*.png to
#   blog/published/diagonalization-powers-and-differential-equations/images/diagonalization.png and embed with:
#   ![Caption.](images/diagonalization.png){#fig-diagonalization fig-align="center" width=90%}

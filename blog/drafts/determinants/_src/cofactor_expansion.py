# Runs via the root uv environment with `uv run`.
"""Cofactor-expansion structure figure for Post 9.

Shows deleting row 1 and column 2 of a 3x3 matrix to form the minor M_12,
the signed cofactor C_12, and the first-row cofactor expansion. All text uses
MathTex/Tex (LaTeX / Computer Modern); bold uses \\mathbf.
"""

from manim import *

LIGHT = "#e6e6e6"
ORANGE = "#f5b041"
RED = "#ec7063"
BLUE = "#5dade2"


class CofactorExpansion(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        title = Tex(r"Cofactor: delete a row and a column", color=LIGHT).scale(0.85)
        title.to_edge(UP, buff=0.5)
        self.add(title)

        M = Matrix([[2, 1, 1], [1, 2, 1], [1, 1, 2]], h_buff=1.0).set_color(LIGHT)
        M.shift(LEFT * 3.2 + UP * 0.3)
        self.add(M)
        ents = M.get_entries()

        # Strike row 1 (entries 0,1,2) and column 2 (entries 1,4,7).
        row_line = Line(ents[0].get_left() + LEFT * 0.15, ents[2].get_right() + RIGHT * 0.15,
                        color=RED, stroke_width=5)
        col_line = Line(ents[1].get_top() + UP * 0.15, ents[7].get_bottom() + DOWN * 0.15,
                        color=RED, stroke_width=5)
        self.add(row_line, col_line)

        # Highlight the surviving minor entries: (2,1)=idx3, (2,3)=idx5, (3,1)=idx6, (3,3)=idx8.
        for idx in (3, 5, 6, 8):
            ents[idx].set_color(ORANGE)

        # The minor and cofactor.
        minor = Matrix([[1, 1], [1, 2]], h_buff=0.9).set_color(ORANGE).scale(0.9)
        minor.shift(RIGHT * 2.4 + UP * 0.6)
        arrow = Arrow(M.get_right(), minor.get_left(), buff=0.3, color=LIGHT)
        self.add(arrow, minor)

        cof = MathTex(r"C_{12} = (-1)^{1+2}\,\det\!\begin{pmatrix}1&1\\1&2\end{pmatrix} = -1",
                      color=LIGHT).scale(0.8)
        cof.next_to(minor, DOWN, buff=0.5).shift(LEFT * 0.3)
        self.add(cof)

        # First-row expansion at the bottom.
        expansion = MathTex(
            r"\det \mathbf{A} = a_{11}C_{11} + a_{12}C_{12} + a_{13}C_{13}",
            color=LIGHT).scale(0.9)
        expansion.to_edge(DOWN, buff=0.7)
        self.add(expansion)


# Render the last frame only as a PNG (use a unique media dir), from the repo root:
#   uv run manim -qh -s --media_dir media_determinants blog/published/determinants/_src/cofactor_expansion.py CofactorExpansion
# Move media_determinants/images/cofactor_expansion/CofactorExpansion*.png to
#   blog/published/determinants/images/cofactor_expansion.png and embed with:
#   ![Caption.](images/cofactor_expansion.png){#fig-cofactor fig-align="center" width=92%}

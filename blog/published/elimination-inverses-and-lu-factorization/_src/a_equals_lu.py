# Runs via the root uv environment with `uv run`.
"""Static A = LU figure for Post 2.

Shows A = L U for the worked 3x3 example, with the elimination multipliers
highlighted in L (below the diagonal) and the pivots highlighted on U's diagonal.
All text uses MathTex (LaTeX / Computer Modern); Manim's template lacks the
physics package, so bold symbols use \\mathbf, not \\vb.
"""

from manim import *

LIGHT = "#e6e6e6"
BLUE_A = "#5dade2"   # multipliers
ORANGE_A = "#f5b041"  # pivots


class AEqualsLU(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        A = Matrix([[1, 2, 1], [3, 8, 1], [0, 4, 1]], h_buff=1.1).set_color(LIGHT)
        L = Matrix([[1, 0, 0], [3, 1, 0], [0, 2, 1]], h_buff=1.1).set_color(LIGHT)
        U = Matrix([[1, 2, 1], [0, 2, -2], [0, 0, 5]], h_buff=1.1).set_color(LIGHT)
        eq = MathTex("=", color=LIGHT)

        group = VGroup(A, eq, L, U).arrange(RIGHT, buff=0.45).scale(0.9)
        group.move_to(ORIGIN).shift(UP * 0.4)
        self.add(group)

        # Highlight the multipliers in L: entry (2,1)=3 -> index 3, (3,2)=2 -> index 7.
        for idx in (3, 7):
            L.get_entries()[idx].set_color(BLUE_A)
        # Highlight the pivots on U's diagonal: indices 0, 4, 8 (values 1, 2, 5).
        for idx in (0, 4, 8):
            U.get_entries()[idx].set_color(ORANGE_A)

        # Matrix name labels below each matrix.
        lab_A = MathTex(r"\mathbf{A}", color=LIGHT).next_to(A, DOWN, buff=0.3)
        lab_L = MathTex(r"\mathbf{L}", color=LIGHT).next_to(L, DOWN, buff=0.3)
        lab_U = MathTex(r"\mathbf{U}", color=LIGHT).next_to(U, DOWN, buff=0.3)
        self.add(lab_A, lab_L, lab_U)

        # Legend at the bottom.
        legend = VGroup(
            MathTex(r"\text{multipliers}", color=BLUE_A).scale(0.8),
            MathTex(r"\text{pivots}", color=ORANGE_A).scale(0.8),
        ).arrange(RIGHT, buff=1.0)
        legend.next_to(group, DOWN, buff=1.1)
        self.add(legend)


# Render the last frame only as a PNG, from the repo root:
#   uv run manim -qh -s blog/published/elimination-inverses-and-lu-factorization/_src/a_equals_lu.py AEqualsLU
# Manim saves it at media/images/a_equals_lu/AEqualsLU.png.
# Move it to blog/published/elimination-inverses-and-lu-factorization/images/a_equals_lu.png and embed with:
#   ![Caption.](images/a_equals_lu.png){#fig-a_equals_lu fig-align="center" width=90%}

# Runs via the root uv environment with `uv run`.
"""A 4x4 Jordan block for Post 15.

Eigenvalue lambda on the diagonal (highlighted), 1's on the superdiagonal
(highlighted), zeros elsewhere. All text uses MathTex (LaTeX / Computer Modern).
"""

from manim import *

LIGHT = "#e6e6e6"
BLUE_A = "#5dade2"
ORANGE_A = "#f5b041"


class JordanBlock(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        J = Matrix(
            [
                [r"\lambda", "1", "0", "0"],
                ["0", r"\lambda", "1", "0"],
                ["0", "0", r"\lambda", "1"],
                ["0", "0", "0", r"\lambda"],
            ],
            h_buff=1.0,
            v_buff=0.9,
        ).set_color(LIGHT)
        J.scale(1.0).shift(LEFT * 1.0)

        entries = J.get_entries()
        # Diagonal lambdas at indices 0, 5, 10, 15.
        for idx in (0, 5, 10, 15):
            entries[idx].set_color(ORANGE_A)
        # Superdiagonal 1's at indices 1, 6, 11.
        for idx in (1, 6, 11):
            entries[idx].set_color(BLUE_A)

        name = MathTex(r"\mathbf{J}", color=LIGHT).scale(1.1).next_to(J, LEFT, buff=0.4)
        self.add(J, name)

        # Annotations on the right.
        ann1 = MathTex(r"\lambda \text{ repeated on the diagonal}", color=ORANGE_A).scale(0.7)
        ann2 = MathTex(r"1\text{'s on the superdiagonal}", color=BLUE_A).scale(0.7)
        ann3 = Tex(r"only \emph{one} eigenvector", color=LIGHT).scale(0.75)
        anns = VGroup(ann1, ann2, ann3).arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        anns.next_to(J, RIGHT, buff=1.0)
        self.add(anns)


# Render the last frame only as a PNG, from the repo root (unique media dir):
#   uv run manim -qh -s --media_dir media_jordan blog/published/similar-matrices-and-jordan-form/_src/jordan_block.py JordanBlock
# Move media_jordan/images/jordan_block/JordanBlock*.png to
#   blog/published/similar-matrices-and-jordan-form/images/jordan_block.png

# Runs via the root uv environment with `uv run`.
"""Rank-one matrix figure for Post 6: A = u v^T (column times row).

All text uses MathTex (LaTeX / Computer Modern); bold uses \\mathbf and
transpose uses \\top (no physics package needed in Manim's template).
"""

from manim import *

LIGHT = "#e6e6e6"
BLUE_A = "#5dade2"
ORANGE_A = "#f5b041"


class RankOne(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        u = Matrix([[1], [2]]).set_color(BLUE_A)
        v = Matrix([[1, 3, 4]]).set_color(ORANGE_A)
        eq = MathTex("=", color=LIGHT)
        A = Matrix([[1, 3, 4], [2, 6, 8]]).set_color(LIGHT)

        group = VGroup(u, v, eq, A).arrange(RIGHT, buff=0.45).scale(0.95)
        group.move_to(ORIGIN).shift(UP * 0.2)
        self.add(group)

        # Labels underneath each factor.
        lu = MathTex(r"\mathbf{u}", color=BLUE_A).scale(0.8).next_to(u, DOWN, buff=0.3)
        lv = MathTex(r"\mathbf{v}^{\top}", color=ORANGE_A).scale(0.8).next_to(v, DOWN, buff=0.3)
        lA = MathTex(r"\mathbf{A}\ (\text{rank } 1)", color=LIGHT).scale(0.8).next_to(A, DOWN, buff=0.3)
        self.add(lu, lv, lA)

        title = MathTex(r"\mathbf{A} = \mathbf{u}\,\mathbf{v}^{\top}", color=LIGHT).scale(1.05)
        title.next_to(group, UP, buff=0.6)
        self.add(title)


# Render the last frame only as a PNG, from the repo root:
#   uv run manim -qh -s --media_dir media_p6 blog/published/matrix-spaces-rank-one-and-graphs/_src/rank_one.py RankOne
# Output at media_p6/images/rank_one/RankOne.png. Move to
#   blog/published/matrix-spaces-rank-one-and-graphs/images/rank_one.png

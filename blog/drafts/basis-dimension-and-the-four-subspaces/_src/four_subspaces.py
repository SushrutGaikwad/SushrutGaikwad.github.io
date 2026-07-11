# Runs via the root uv environment with `uv run`.
"""The four fundamental subspaces "big picture" for Post 5.

Input space R^n (row space + nullspace) on the left, output space R^m
(column space + left nullspace) on the right, with the map A between them.
All text uses Tex/MathTex (LaTeX / Computer Modern); bold uses \\mathbf, and
transpose uses \\top (Manim's template need not have the physics package).
"""

from manim import *

LIGHT = "#e6e6e6"
BLUE_A = "#5dade2"
RED_B = "#ec7063"
GRAY_L = "#9aa0a6"


def labelled_box(title, dim, color, center):
    box = RoundedRectangle(corner_radius=0.18, width=3.6, height=2.25,
                           color=color, stroke_width=2.5)
    box.set_fill(color, opacity=0.16)
    box.move_to(center)
    name = Tex(title, color=LIGHT).scale(0.62)
    d = MathTex(dim, color=color).scale(0.62)
    txt = VGroup(name, d).arrange(DOWN, buff=0.18).move_to(center)
    return VGroup(box, txt)


class FourSubspaces(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        big_n = RoundedRectangle(corner_radius=0.3, width=4.4, height=6.4, color=GRAY_L, stroke_width=2)
        big_n.move_to(LEFT * 4.1)
        big_m = RoundedRectangle(corner_radius=0.3, width=4.4, height=6.4, color=GRAY_L, stroke_width=2)
        big_m.move_to(RIGHT * 4.1)
        self.add(big_n, big_m)

        rn = MathTex(r"\mathbb{R}^n", color=LIGHT).scale(0.9).next_to(big_n, UP, buff=0.18)
        rm = MathTex(r"\mathbb{R}^m", color=LIGHT).scale(0.9).next_to(big_m, UP, buff=0.18)
        self.add(rn, rm)

        row = labelled_box(r"row space $C(\mathbf{A}^{\top})$", r"\dim = r", BLUE_A, LEFT * 4.1 + UP * 1.5)
        nul = labelled_box(r"nullspace $N(\mathbf{A})$", r"\dim = n - r", RED_B, LEFT * 4.1 + DOWN * 1.5)
        col = labelled_box(r"column space $C(\mathbf{A})$", r"\dim = r", BLUE_A, RIGHT * 4.1 + UP * 1.5)
        lnu = labelled_box(r"left nullspace $N(\mathbf{A}^{\top})$", r"\dim = m - r", RED_B, RIGHT * 4.1 + DOWN * 1.5)
        self.add(row, nul, col, lnu)

        # Perpendicular hints between the stacked subspaces.
        for cx in (-4.1, 4.1):
            perp = MathTex(r"\perp", color=GRAY_L).scale(0.7).move_to(np.array([cx, 0.0, 0.0]))
            self.add(perp)

        # The map A from the input space to the output space.
        arr = Arrow(big_n.get_right(), big_m.get_left(), buff=0.15, color=LIGHT, stroke_width=5)
        a_lab = MathTex(r"\mathbf{A}", color=LIGHT).scale(1.0).next_to(arr, UP, buff=0.12)
        self.add(arr, a_lab)
        note = Tex(r"row space $\to$ column space;\\ nullspace $\to \mathbf{0}$", color=GRAY_L).scale(0.5)
        note.next_to(arr, DOWN, buff=0.2)
        self.add(note)


# Render the last frame only as a PNG, from the repo root:
#   uv run manim -qh -s --media_dir media_p5 blog/published/basis-dimension-and-the-four-subspaces/_src/four_subspaces.py FourSubspaces
# Output at media_p5/images/four_subspaces/FourSubspaces.png. Move to
#   blog/published/basis-dimension-and-the-four-subspaces/images/four_subspaces.png

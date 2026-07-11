# Runs via the root uv environment with `uv run`.
"""SVD four-subspaces figure for Post 16.

Schematic: the input space splits into row space (dim r) and nullspace (dim n-r);
the output space into column space (dim r) and left nullspace (dim m-r). A sends
the row space onto the column space (scaling v_i by sigma_i to u_i) and the
nullspace to 0. All text uses MathTex/Tex; bold uses \\mathbf.
"""

from manim import *

LIGHT = "#e6e6e6"
BLUE_A = "#5dade2"
ORANGE_A = "#f5b041"
GRAY_L = "#9aa0a6"


def labeled_box(title, dim, color, width=3.4, height=1.5):
    box = RoundedRectangle(width=width, height=height, corner_radius=0.16,
                           stroke_color=color, fill_color=color, fill_opacity=0.12)
    t = Tex(title, color=LIGHT).scale(0.62)
    d = MathTex(dim, color=color).scale(0.62)
    txt = VGroup(t, d).arrange(DOWN, buff=0.14).move_to(box.get_center())
    return VGroup(box, txt)


class SVDFourSubspaces(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        row = labeled_box(r"row space", r"\dim = r", ORANGE_A).move_to(LEFT * 3.7 + UP * 1.4)
        null = labeled_box(r"nullspace", r"\dim = n - r", GRAY_L).move_to(LEFT * 3.7 + DOWN * 1.5)
        col = labeled_box(r"column space", r"\dim = r", BLUE_A).move_to(RIGHT * 3.7 + UP * 1.4)
        lnull = labeled_box(r"left nullspace", r"\dim = m - r", GRAY_L).move_to(RIGHT * 3.7 + DOWN * 1.5)
        self.add(row, null, col, lnull)

        # Column headers.
        hin = MathTex(r"\mathbb{R}^n\ \text{(inputs)}", color=LIGHT).scale(0.7).next_to(row, UP, buff=0.35)
        hout = MathTex(r"\mathbb{R}^m\ \text{(outputs)}", color=LIGHT).scale(0.7).next_to(col, UP, buff=0.35)
        self.add(hin, hout)

        # Row space -> column space.
        a1 = Arrow(row[0].get_right(), col[0].get_left(), buff=0.15, color=LIGHT, stroke_width=4)
        a1_lab = MathTex(r"\mathbf{v}_i \mapsto \sigma_i \mathbf{u}_i", color=LIGHT).scale(0.62)
        a1_lab.next_to(a1, UP, buff=0.12)
        self.add(a1, a1_lab)

        # Nullspace -> 0.
        zero = MathTex(r"\mathbf{0}", color=LIGHT).scale(0.8).move_to((null[0].get_right() + lnull[0].get_left()) / 2)
        a2 = Arrow(null[0].get_right(), zero.get_left(), buff=0.15, color=GRAY_L, stroke_width=4)
        a2_lab = MathTex(r"\mapsto \mathbf{0}", color=GRAY_L).scale(0.58).next_to(a2, DOWN, buff=0.1)
        self.add(zero, a2, a2_lab)


# Render: uv run manim -qh -s --media_dir media_svd blog/published/the-singular-value-decomposition/_src/svd_subspaces.py SVDFourSubspaces
# Move media_svd/images/svd_subspaces/SVDFourSubspaces.png -> images/svd_subspaces.png

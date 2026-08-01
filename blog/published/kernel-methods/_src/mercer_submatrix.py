# Runs via the root uv environment with uv run (see render command at the bottom).
from manim import *

_TEMPLATE = TexTemplate()
_TEMPLATE.add_to_preamble(r"\usepackage{upgreek}")
config.tex_template = _TEMPLATE


class MercerSubmatrix(Scene):
    """The picture behind Mercer's theorem.

    Think of the kernel function K(x, z) as one enormous matrix with a row for every
    possible input x and a column for every possible input z. Choosing n points is
    choosing n rows and the matching n columns; the kernel matrix K is exactly the
    little submatrix sitting at their intersections. Mercer's condition then reads:
    K is a valid kernel precisely when EVERY submatrix you can carve out this way is
    symmetric and positive semidefinite."""

    def construct(self):
        self.camera.background_color = "#222222"

        light = "#e6e6e6"
        muted = "#9a9a9a"
        blue = "#5dade2"
        green = "#2ecc71"

        title = Tex(r"Mercer's condition, pictured", font_size=42,
                    color=light).to_edge(UP, buff=0.32)

        # ---------- LEFT: the kernel as one enormous matrix ----------
        S = 3.9
        centre = np.array([-3.95, 0.55, 0.0])
        big = Rectangle(width=S, height=S, color="#777777", stroke_width=2.5)
        big.move_to(centre)

        # Four chosen points: their rows and their columns.
        frac = [0.16, 0.40, 0.62, 0.86]
        rows = [centre[1] + S / 2 - f * S for f in frac]
        cols = [centre[0] - S / 2 + f * S for f in frac]

        lines = VGroup()
        for y in rows:
            lines.add(Line([centre[0] - S / 2, y, 0], [centre[0] + S / 2, y, 0],
                           color=blue, stroke_width=1.6, stroke_opacity=0.55))
        for x in cols:
            lines.add(Line([x, centre[1] - S / 2, 0], [x, centre[1] + S / 2, 0],
                           color=blue, stroke_width=1.6, stroke_opacity=0.55))

        cells = VGroup()
        for y in rows:
            for x in cols:
                cells.add(Square(side_length=0.30, color=blue, stroke_width=2.0)
                          .set_fill(blue, opacity=0.55).move_to([x, y, 0]))

        # Labels for the chosen rows and columns.
        row_labels = VGroup()
        for y, name in zip(rows, [r"\mathbf{x}_1", r"\mathbf{x}_2",
                                  r"\mathbf{x}_3", r"\mathbf{x}_n"]):
            lab = MathTex(name, font_size=26, color=blue)
            lab.next_to([centre[0] - S / 2, y, 0], LEFT, buff=0.16)
            row_labels.add(lab)
        col_labels = VGroup()
        for x, name in zip(cols, [r"\mathbf{x}_1", r"\mathbf{x}_2",
                                  r"\mathbf{x}_3", r"\mathbf{x}_n"]):
            lab = MathTex(name, font_size=26, color=blue)
            lab.next_to([x, centre[1] + S / 2, 0], UP, buff=0.16)
            col_labels.add(lab)

        forever_r = Tex(r"$\cdots$", font_size=40, color=muted)
        forever_r.next_to(big, RIGHT, buff=0.14)
        forever_d = Tex(r"$\vdots$", font_size=40, color=muted)
        forever_d.next_to(big, DOWN, buff=0.10)
        big_caption = Tex(r"the kernel $K(\mathbf{x},\mathbf{z})$ as one enormous matrix:\\"
                          r"a row and a column for \emph{every} possible input",
                          font_size=25, color=muted)
        big_caption.next_to(forever_d, DOWN, buff=0.20)

        # ---------- ARROW ----------
        arrow = Arrow([-1.30, 0.55, 0], [0.30, 0.55, 0], color=light,
                      stroke_width=5, buff=0.05,
                      max_tip_length_to_length_ratio=0.22)
        arrow_lbl = Tex(r"pick $n$ points", font_size=26, color=light)
        arrow_lbl.next_to(arrow, UP, buff=0.14)

        # ---------- RIGHT: the extracted finite matrix ----------
        mat = MathTex(
            r"\mathbf{K} = \begin{bmatrix}"
            r"K(\mathbf{x}_1,\mathbf{x}_1) & \cdots & K(\mathbf{x}_1,\mathbf{x}_n)\\"
            r"\vdots & \ddots & \vdots\\"
            r"K(\mathbf{x}_n,\mathbf{x}_1) & \cdots & K(\mathbf{x}_n,\mathbf{x}_n)"
            r"\end{bmatrix}",
            font_size=29, color=blue)
        mat.move_to([3.85, 1.35, 0])

        verdict = Tex(r"symmetric?\\ positive semidefinite?",
                      font_size=30, color=green)
        verdict.next_to(mat, DOWN, buff=0.75)
        vbox = SurroundingRectangle(verdict, color=green, buff=0.24,
                                    corner_radius=0.1)

        # ---------- The statement ----------
        moral = Tex(r"$K$ is a valid kernel \emph{exactly} when every submatrix "
                    r"carved out this way passes both tests.",
                    font_size=31, color=light)
        moral.to_edge(DOWN, buff=0.3)

        self.add(title, big, lines, cells, row_labels, col_labels,
                 forever_r, forever_d, big_caption,
                 arrow, arrow_lbl, mat, verdict, vbox, moral)


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/kernel-methods/_src/mercer_submatrix.py MercerSubmatrix
# Manim writes the PNG to media\images\mercer_submatrix\MercerSubmatrix_ManimCE_v0.20.1.png.
# Move it to blog/published/kernel-methods/images/mercer_submatrix.png
# Embed in the .qmd with:
#   ![...](images/mercer_submatrix.png){#fig-mercer_submatrix fig-align="center" width=100%}

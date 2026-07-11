# Runs via the root uv environment with uv run (see render command at the bottom).
from manim import *


class SingularVsInvertible(Scene):
    """Two column-pictures side by side. Left: a singular matrix whose two columns
    lie on one line. Right: an invertible matrix whose columns span the plane."""

    def construct(self):
        self.camera.background_color = "#222222"

        left = self.make_panel(
            second_col=(3, 6),
            title=r"Singular: $\begin{bmatrix} 1 & 3 \\ 2 & 6 \end{bmatrix}$",
            subtitle=r"columns lie on one line",
            collinear=True,
        )
        right = self.make_panel(
            second_col=(3, 1),
            title=r"Invertible: $\begin{bmatrix} 1 & 3 \\ 2 & 1 \end{bmatrix}$",
            subtitle=r"columns span the plane",
            collinear=False,
        )

        panels = VGroup(left, right).arrange(RIGHT, buff=1.4)
        panels.move_to(ORIGIN)
        self.add(panels)

    def make_panel(self, second_col, title, subtitle, collinear):
        # Faint grid so the directions of the two column vectors are readable.
        plane = NumberPlane(
            x_range=[-1, 4, 1],
            y_range=[-1, 7, 1],
            x_length=3.0,
            y_length=4.8,
            background_line_style={"stroke_color": "#3d3d3d", "stroke_width": 1},
            axis_config={"color": "#888888", "stroke_width": 2},
        )

        origin = plane.c2p(0, 0)
        p1 = plane.c2p(1, 2)
        p2 = plane.c2p(*second_col)

        elements = VGroup(plane)

        # When the columns are dependent, draw the common line they both sit on.
        if collinear:
            line = DashedLine(
                plane.c2p(-0.5, -1.0),
                plane.c2p(3.5, 7.0),
                color="#cc6666",
                stroke_width=2,
            )
            elements.add(line)

        color1 = ORANGE
        color2 = ORANGE if collinear else TEAL

        # Draw the longer arrow first so the shorter one stays visible on top.
        arr2 = Arrow(origin, p2, buff=0, color=color2,
                     stroke_width=6, max_tip_length_to_length_ratio=0.10)
        arr1 = Arrow(origin, p1, buff=0, color=color1,
                     stroke_width=6, max_tip_length_to_length_ratio=0.14)

        lab1 = MathTex(r"(1,2)", color=color1, font_size=26).next_to(p1, LEFT, buff=0.12)
        lab2 = MathTex(rf"({second_col[0]},{second_col[1]})", color=color2,
                       font_size=26).next_to(p2, RIGHT, buff=0.12)

        elements.add(arr2, arr1, lab2, lab1)

        title_tex = Tex(title, font_size=32, color="#e6e6e6").next_to(plane, UP, buff=0.25)
        sub_tex = Tex(subtitle, font_size=26, color="#b0b0b0").next_to(plane, DOWN, buff=0.25)

        return VGroup(title_tex, elements, sub_tex)


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/elimination-inverses-and-lu-factorization/_src/singular_vs_invertible.py SingularVsInvertible
# Manim writes the PNG to media\images\singular_vs_invertible\SingularVsInvertible.png.
# Move it to blog/published/elimination-inverses-and-lu-factorization/images/singular_vs_invertible.png
# Embed in the .qmd with:
#   ![...](images/singular_vs_invertible.png){#fig-singular_vs_invertible fig-align="center" width=95%}

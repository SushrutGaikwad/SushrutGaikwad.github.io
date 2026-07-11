# Runs via the root uv environment with `uv run`.
"""The four equivalent positive-definiteness tests for Post 13.

Four boxed conditions in a 2x2 grid; any one implies the others.
All text uses Tex/MathTex (LaTeX / Computer Modern); bold uses \\mathbf.
"""

from manim import *

LIGHT = "#e6e6e6"
BLUE_A = "#5dade2"
ORANGE_A = "#f5b041"


class PDTests(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        title = Tex(
            r"A symmetric $\mathbf{A}$ is \emph{positive definite} when any one of these holds:",
            color=LIGHT,
        ).scale(0.72)

        tests = [
            r"\text{all eigenvalues } \lambda_i > 0",
            r"\text{all pivots } d_i > 0",
            r"\text{all leading minors } > 0",
            r"\mathbf{x}^{\top}\mathbf{A}\,\mathbf{x} > 0 \ \text{ for all } \mathbf{x} \neq \mathbf{0}",
        ]

        boxes = VGroup()
        for i, t in enumerate(tests, start=1):
            num = MathTex(rf"{i}.", color=ORANGE_A).scale(0.95)
            body = MathTex(t, color=LIGHT).scale(0.95)
            row = VGroup(num, body).arrange(RIGHT, buff=0.28)
            rect = SurroundingRectangle(row, color=BLUE_A, buff=0.32, corner_radius=0.18)
            boxes.add(VGroup(rect, row))

        boxes.arrange_in_grid(rows=2, cols=2, buff=(1.0, 0.8))
        boxes.scale_to_fit_width(11.5)

        group = VGroup(title, boxes).arrange(DOWN, buff=0.7)
        group.move_to(ORIGIN)
        self.add(group)


# Render the last frame only as a PNG, from the repo root (unique media dir):
#   uv run manim -qh -s --media_dir media_pdtests blog/published/symmetric-and-positive-definite-matrices/_src/pd_tests.py PDTests
# Move media_pdtests/images/pd_tests/PDTests*.png to
#   blog/published/symmetric-and-positive-definite-matrices/images/pd_tests.png

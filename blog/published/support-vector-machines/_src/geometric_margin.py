# Runs via the root uv environment with uv run (see render command at the bottom).
#
# STATIC PNG: the geometric-margin derivation. The boundary w^T x + b = 0, the
# weight vector w orthogonal to it, a positive point A = x_i, its foot B on the
# boundary, and the perpendicular distance gamma_i. The key construction
# B = x_i - gamma_i * w/||w|| is annotated so the algebra in the post has a picture.
#
# Layout fixes over the first draft: a squarer frame and a bigger diagram, since
# the 16:9 version left roughly 40% of the image empty; and A is now drawn as a
# green TRIANGLE like every other admitted student in the post, with plain dots
# reserved for construction points (B, the foot of the perpendicular).
#
# Note on correctness: B is computed as A - gamma * w_hat, so the segment AB is
# perpendicular to the boundary by construction, and B lands on x1 + x2 = 0 to
# within rounding. The right-angle tick is therefore honest, not decorative.
from manim import *
import numpy as np

_TEMPLATE = TexTemplate()
_TEMPLATE.add_to_preamble(r"\usepackage{upgreek}")
config.tex_template = _TEMPLATE

config.pixel_width = 1500
config.pixel_height = 1080


class GeometricMargin(Scene):
    def construct(self):
        self.camera.background_color = "#222222"
        light = "#e6e6e6"
        blue = "#5dade2"
        green = "#2ecc71"
        red = "#e74c3c"
        yellow = "#f1c40f"
        orange = "#e67e22"

        scale = 1.55
        shift = LEFT * 1.9 + DOWN * 0.15

        def P(x1, x2):
            return np.array([x1, x2, 0.0]) * scale + shift

        # Decision boundary: the line x1 + x2 = 0 (unit normal w_hat = (1,1)/sqrt2).
        w_hat = np.array([1.0, 1.0]) / np.sqrt(2.0)
        boundary = Line(P(-2.1, 2.1), P(1.9, -1.9), color=blue, stroke_width=5)
        boundary_lbl = MathTex(r"\mathbf{w}^{\intercal}\mathbf{x}+b=0",
                               color=blue, font_size=32)
        boundary_lbl.move_to(np.array([-0.9, -3.35, 0.0]))

        # Point A = x_i on the positive side; foot B on the boundary.
        gamma = 1.8
        A = np.array([0.849, 1.697])          # = gamma*w_hat - 0.6*tangent
        B = A - gamma * w_hat                   # foot of the perpendicular
        assert abs(B[0] + B[1]) < 1e-3, "B must land on the boundary x1 + x2 = 0"

        markA = Triangle(color=green, fill_opacity=1.0).scale(0.16).move_to(P(*A))
        dotB = Dot(P(*B), color=light, radius=0.075)
        lblA = MathTex(r"A \;=\; \mathbf{x}_i", color=green, font_size=32)
        lblA.next_to(markA, RIGHT, buff=0.18)
        lblB = MathTex(r"B", color=light, font_size=32)
        lblB.next_to(dotB, LEFT, buff=0.14).shift(DOWN * 0.05)

        # The margin segment A->B (dashed) with a gamma_i label on its right.
        seg = DashedLine(P(*A), P(*B), color=yellow, stroke_width=4, dash_length=0.12)
        seg_lbl = MathTex(r"\gamma_i", color=yellow, font_size=38)
        seg_lbl.move_to((P(*A) + P(*B)) / 2.0).shift(RIGHT * 0.52)

        # Right-angle tick at B.
        ra = RightAngle(seg, boundary, length=0.22, quadrant=(1, 1), color=light,
                        stroke_width=2.5)

        # The weight vector w, orthogonal to the boundary, drawn to the upper-left
        # so it does not crowd the A->B segment.
        O = np.array([-1.05, 1.05])            # on boundary (x1+x2=0)
        w_arrow = Arrow(P(*O), P(*(O + 1.1 * w_hat)), color=orange,
                        stroke_width=6, buff=0.0, max_tip_length_to_length_ratio=0.18)
        w_lbl = MathTex(r"\mathbf{w}", color=orange, font_size=38)
        w_lbl.next_to(w_arrow.get_center(), LEFT, buff=0.14)

        # A few class markers for context.
        markers = VGroup()
        for x1, x2 in [(1.55, 1.35), (1.15, 2.0), (1.35, 0.95)]:
            markers.add(Triangle(color=green, fill_opacity=1.0).scale(0.12)
                        .move_to(P(x1, x2)))
        for x1, x2 in [(-1.55, -1.15), (-1.15, -1.75), (-1.95, -0.85)]:
            markers.add(MathTex(r"\times", color=red, font_size=34)
                        .move_to(P(x1, x2)))

        # Construction annotation (right panel).
        note = VGroup(
            MathTex(r"B = \mathbf{x}_i - \gamma_i\,\frac{\mathbf{w}}{\|\mathbf{w}\|}",
                    color=light, font_size=34),
            MathTex(r"\Downarrow", color=light, font_size=34),
            MathTex(r"\gamma_i = \frac{\mathbf{w}^{\intercal}\mathbf{x}_i + b}{\|\mathbf{w}\|}",
                    color=yellow, font_size=38),
        ).arrange(DOWN, buff=0.32)
        note.move_to(np.array([3.35, 0.45, 0.0]))
        note_box = SurroundingRectangle(note, color="#777777", buff=0.3,
                                        corner_radius=0.1)

        self.add(boundary, boundary_lbl, markers, w_arrow, w_lbl,
                 seg, ra, seg_lbl, markA, dotB, lblA, lblB, note, note_box)


# Render (from the repo root), saving only the final frame as a PNG. The script
# pins its own pixel dimensions, so -qh and -ql give the same size here:
#   uv run manim -qh -s blog/published/support-vector-machines/_src/geometric_margin.py GeometricMargin
# Manim writes the PNG to media\images\geometric_margin\GeometricMargin.png
# Move it to blog/published/support-vector-machines/images/geometric_margin.png
# Embed (dark background -> no .invert):
#   ![...](images/geometric_margin.png){#fig-geometric_margin fig-align="center" width=90%}

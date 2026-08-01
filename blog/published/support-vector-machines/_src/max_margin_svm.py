# Runs via the root uv environment with uv run (see render command at the bottom).
#
# STATIC PNG: the optimal-margin classifier. Solid boundary w^T x + b = 0, two
# dashed margin lines w^T x + b = +/-1, the margin width 2/||w|| as a double
# arrow, and three circled SUPPORT VECTORS lying exactly on the margin lines.
#
# Two layout fixes over the first draft:
#   * Everything is parametrized in (s, t) coordinates and the extents are
#     chosen so no element leaves the 14.22 x 8 frame. Previously the upper
#     margin line ran off the top edge and the boundary label was clipped by the
#     bottom-right corner.
#   * The "+1", "0" and "-1" labels now sit at the SAME (lower-right) ends of
#     their lines, so they read top-to-bottom in the order +1, 0, -1. Putting
#     "+1" and "-1" at opposite ends of the frame made the eye pair each label
#     with the wrong line.
from manim import *
import numpy as np

_TEMPLATE = TexTemplate()
_TEMPLATE.add_to_preamble(r"\usepackage{upgreek}")
config.tex_template = _TEMPLATE


class MaxMarginSVM(Scene):
    def construct(self):
        self.camera.background_color = "#222222"
        light = "#e6e6e6"
        blue = "#5dade2"        # boundary: the ONLY solid blue line
        green = "#2ecc71"
        red = "#e74c3c"
        yellow = "#f1c40f"
        muted = "#aab7c4"       # margin lines: dashed, never blue

        scale = 1.5
        shift = LEFT * 0.8

        # s = x1 + x2 (0 on the boundary, +/-m on the margin lines);
        # t = a tangential coordinate running along the boundary direction.
        def pt(s, t):
            return np.array([s / 2 + t, s / 2 - t, 0.0]) * scale + shift

        m = 1.4
        T = 1.7          # keeps every line inside the frame

        def line_pts(c):
            return pt(c, -T), pt(c, T)

        boundary = Line(*line_pts(0), color=blue, stroke_width=5)
        up = DashedLine(*line_pts(m), color=muted, stroke_width=3.5, dash_length=0.14)
        lo = DashedLine(*line_pts(-m), color=muted, stroke_width=3.5, dash_length=0.14)

        # All three labels at the lower-right ends, stacked +1 / 0 / -1.
        lblp = MathTex(r"+1", color=muted, font_size=32)
        lblp.next_to(up.get_end(), RIGHT, buff=0.15)
        lbl0 = MathTex(r"\mathbf{w}^{\intercal}\mathbf{x}+b=0", color=blue, font_size=30)
        lbl0.next_to(boundary.get_end(), RIGHT, buff=0.15)
        lbln = MathTex(r"-1", color=muted, font_size=32)
        lbln.next_to(lo.get_end(), RIGHT, buff=0.15)

        # One clarifier in the otherwise empty upper-left, so the compact "+1"
        # and "-1" ticks are unambiguous.
        note = MathTex(r"\text{dashed: } \mathbf{w}^{\intercal}\mathbf{x}+b = \pm 1",
                       color=muted, font_size=30)
        note.to_corner(UL, buff=0.55)

        # Support vectors sit exactly ON the margin lines; everything else is
        # strictly outside.
        pos_sv = [(m, -0.30), (m, 0.75)]
        neg_sv = [(-m, 0.10)]
        pos_free = [(2.5, -0.15), (2.9, 0.55), (2.2, 1.15)]
        # Kept well out to the left so the "support vectors" caption below has a
        # clear pocket; an earlier layout ran the caption straight over a cross.
        neg_free = [(-2.6, -1.30), (-3.0, -0.80), (-2.4, -1.80)]

        marks = VGroup()
        sv_rings = VGroup()
        for s, t in pos_sv:
            marks.add(Triangle(color=green, fill_opacity=1.0).scale(0.15).move_to(pt(s, t)))
            sv_rings.add(Circle(radius=0.28, color=yellow, stroke_width=4).move_to(pt(s, t)))
        for s, t in neg_sv:
            marks.add(MathTex(r"\times", color=red, font_size=48).move_to(pt(s, t)))
            sv_rings.add(Circle(radius=0.28, color=yellow, stroke_width=4).move_to(pt(s, t)))
        for s, t in pos_free:
            marks.add(Triangle(color=green, fill_opacity=1.0).scale(0.15).move_to(pt(s, t)))
        for s, t in neg_free:
            marks.add(MathTex(r"\times", color=red, font_size=48).move_to(pt(s, t)))

        # Margin-width double arrow, drawn along the normal.
        t_arrow = -0.75
        darr = DoubleArrow(pt(-m, t_arrow), pt(m, t_arrow), color=yellow,
                           stroke_width=5, buff=0.0,
                           max_tip_length_to_length_ratio=0.14)
        # The arrow's midpoint lies ON the boundary by construction, and any
        # perpendicular offset just slides the label ALONG the boundary, so the
        # label cannot be centred on the arrow without touching the blue line.
        # An opaque backing plate was worse still: it punched a visible hole in
        # the boundary, making the line look broken. So the label goes just past
        # the arrow's +1 tip, outside the margin band, where nothing else sits.
        darr_lbl = MathTex(r"\frac{2}{\|\mathbf{w}\|}", color=yellow, font_size=40)
        darr_lbl.next_to(darr.get_end(), UP, buff=0.22)

        # Caption the support vectors, with a short leader to the circled cross.
        sv_note = VGroup(
            Tex(r"support", color=yellow, font_size=32),
            Tex(r"vectors", color=yellow, font_size=32),
        ).arrange(DOWN, buff=0.08)
        sv_note.move_to(np.array([-3.05, -2.55, 0.0]))
        leader = Arrow(sv_note.get_top(), pt(-m, 0.10), color=yellow,
                       stroke_width=3, buff=0.22,
                       max_tip_length_to_length_ratio=0.09)

        self.add(boundary, up, lo, lblp, lbl0, lbln, note,
                 darr, darr_lbl, marks, sv_rings, leader, sv_note)


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/support-vector-machines/_src/max_margin_svm.py MaxMarginSVM
# Manim writes the PNG to media\images\max_margin_svm\MaxMarginSVM.png
# Move it to blog/published/support-vector-machines/images/max_margin_svm.png
# Embed (dark background -> no .invert):
#   ![...](images/max_margin_svm.png){#fig-max_margin_svm fig-align="center" width=100%}

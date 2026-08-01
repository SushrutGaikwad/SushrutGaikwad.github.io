# Runs via the root uv environment with uv run (see render command at the bottom).
#
# STATIC PNG: soft-margin slack variables. Each point that violates its margin
# pays a slack xi_i equal to how far (in functional-margin units) it falls short
# of 1. A correctly-classified point outside the margin has xi = 0; a point
# inside the margin has 0 < xi < 1; a misclassified point has xi > 1.
#
# Three fixes over the first draft:
#   * No rotated text. The "0 < xi_i < 1" label used to be rotated 45 degrees and
#     laid across the margin band, which was the hardest thing in the post to
#     read. Every label is horizontal now, with an opaque backing plate.
#   * A REJECTED student inside its own margin is annotated too. Previously all
#     three annotated points were admitted (green), which quietly suggested that
#     slack is something only the positive class pays.
#   * Squarer frame, so the diagram fills the image instead of floating in a
#     16:9 letterbox.
from manim import *
import numpy as np

_TEMPLATE = TexTemplate()
_TEMPLATE.add_to_preamble(r"\usepackage{upgreek}")
config.tex_template = _TEMPLATE

config.pixel_width = 1600
config.pixel_height = 1080


def leader(start, end, color):
    """A thin line tying an off-to-the-side label to the thing it describes.

    Opaque backing plates were tried first and rejected: parking a plate over a
    line punches a visible hole in it, so the boundary looked broken. Leaders
    let every label sit in genuinely empty space while staying unambiguous, and
    a thin line crossing a dashed line costs the reader nothing.
    """
    return Line(start, end, color=color, stroke_width=2, stroke_opacity=0.75)


class SlackVariables(Scene):
    def construct(self):
        self.camera.background_color = "#222222"
        blue = "#5dade2"        # boundary only
        green = "#2ecc71"
        red = "#e74c3c"
        yellow = "#f1c40f"
        orange = "#e67e22"
        muted = "#aab7c4"       # margin lines

        scale = 1.4
        shift = LEFT * 0.9

        # A point is parametrized by (s, t): s = x1 + x2 (its signed "level",
        # so s = 0 on the boundary and s = +/-m on the margin lines) and t = a
        # tangential coordinate along the boundary direction.
        def pt(s, t):
            return np.array([s / 2 + t, s / 2 - t, 0.0]) * scale + shift

        m = 1.5
        T = 1.85

        def line_pts(c):
            return pt(c, -T), pt(c, T)

        boundary = Line(*line_pts(0), color=blue, stroke_width=5)
        up = DashedLine(*line_pts(m), color=muted, stroke_width=3.5, dash_length=0.14)
        lo = DashedLine(*line_pts(-m), color=muted, stroke_width=3.5, dash_length=0.14)

        # Line labels stacked +1 / 0 / -1 at the lower-right ends, matching the
        # ordering used in the other margin figures.
        lblp = MathTex(r"+1", color=muted, font_size=32).next_to(
            up.get_end(), RIGHT, buff=0.14)
        lbl0 = MathTex(r"0", color=blue, font_size=32).next_to(
            boundary.get_end(), RIGHT, buff=0.14)
        lbln = MathTex(r"-1", color=muted, font_size=32).next_to(
            lo.get_end(), RIGHT, buff=0.14)

        marks = VGroup()
        notes = VGroup()

        # (1) Well-classified admitted student, OUTSIDE the margin: xi = 0.
        # Sits in open space, so its label needs no leader.
        p_ok = pt(2.6, 0.5)
        marks.add(Triangle(color=green, fill_opacity=1.0).scale(0.15).move_to(p_ok))
        notes.add(MathTex(r"\xi_i = 0", color=green, font_size=32)
                  .next_to(p_ok, RIGHT, buff=0.22))

        # (2) Admitted student INSIDE the margin (still on the correct side):
        # 0 < xi < 1. Its slack is measured to its OWN margin line, the +1 line.
        p_in, foot_in = pt(0.6, -1.2), pt(m, -1.2)
        marks.add(Triangle(color=green, fill_opacity=1.0).scale(0.15).move_to(p_in))
        slack1 = Arrow(p_in, foot_in, color=yellow, stroke_width=5, buff=0.05,
                       max_tip_length_to_length_ratio=0.16)
        lbl2 = MathTex(r"0 < \xi_i < 1", color=yellow, font_size=30)
        lbl2.move_to(np.array([-3.55, 3.25, 0.0]))
        notes.add(lbl2, leader(lbl2.get_right(), (p_in + foot_in) / 2, yellow))

        # (3) Admitted student MISCLASSIFIED (across on the rejected side):
        # xi > 1. Its slack arrow still measures to the +1 line, so it must cross
        # the boundary to get there. That crossing is the signature of xi > 1.
        p_wrong, foot_wrong = pt(-0.7, 0.3), pt(m, 0.3)
        marks.add(Triangle(color=green, fill_opacity=1.0).scale(0.15).move_to(p_wrong))
        slack2 = Arrow(p_wrong, foot_wrong, color=orange, stroke_width=5, buff=0.08,
                       max_tip_length_to_length_ratio=0.10)
        lbl3 = MathTex(r"\xi_i > 1", color=orange, font_size=32)
        lbl3.move_to(np.array([-3.95, -1.95, 0.0]))
        notes.add(lbl3, leader(lbl3.get_right(), p_wrong, orange))

        # (4) REJECTED student inside ITS own margin: slack is symmetric across
        # the classes, and its foot is on the -1 line.
        p_neg, foot_neg = pt(-0.7, 0.6), pt(-m, 0.6)
        marks.add(MathTex(r"\times", color=red, font_size=46).move_to(p_neg))
        slack3 = Arrow(p_neg, foot_neg, color=yellow, stroke_width=5, buff=0.05,
                       max_tip_length_to_length_ratio=0.16)
        lbl4 = MathTex(r"0 < \xi_i < 1", color=yellow, font_size=30)
        lbl4.move_to(np.array([-1.25, -3.05, 0.0]))
        notes.add(lbl4, leader(lbl4.get_top(), (p_neg + foot_neg) / 2, yellow))

        # A few quiet background points on their correct sides.
        for s, t in [(2.9, -0.55), (2.4, 1.25)]:
            marks.add(Triangle(color=green, fill_opacity=1.0).scale(0.13)
                      .move_to(pt(s, t)))
        for s, t in [(-2.7, -0.75), (-2.4, -1.35), (-2.9, 0.05)]:
            marks.add(MathTex(r"\times", color=red, font_size=44).move_to(pt(s, t)))

        self.add(boundary, up, lo, lblp, lbl0, lbln,
                 slack1, slack2, slack3, marks, notes)


# Render (from the repo root), saving only the final frame as a PNG. The script
# pins its own pixel dimensions, so -qh and -ql give the same size here:
#   uv run manim -qh -s blog/published/support-vector-machines/_src/slack_variables.py SlackVariables
# Manim writes the PNG to media\images\slack_variables\SlackVariables.png
# Move it to blog/published/support-vector-machines/images/slack_variables.png
# Embed (dark background -> no .invert):
#   ![...](images/slack_variables.png){#fig-slack_variables fig-align="center" width=90%}

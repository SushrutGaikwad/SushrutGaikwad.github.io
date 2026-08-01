# Runs via the root uv environment with uv run (see render command at the bottom).
#
# STATIC PNG for the "geometric sanity check": why the margin width is 2/||w||.
# Pick x_+ on the +1 line, walk perpendicular to the lines (direction -w/||w||,
# i.e. opposite to w) until reaching x_- on the -1 line. That walk has length
# t = 2/||w||, the gap between the two margin lines.
#
# Three fixes over the first draft:
#   * The frame is squarer (1400x1080 instead of 16:9). The diagram is a compact
#     diagonal band, so a 16:9 frame left roughly half the image empty.
#   * The DECISION BOUNDARY is now drawn (faint, solid blue) instead of being
#     described as "not drawn" in the caption. Readers should not have to
#     imagine the very line the margin is measured around.
#   * The margin lines are dashed grey, not blue. Blue is reserved for the
#     boundary in every other figure in this post; drawing the margins in blue
#     here made the same object look like two different things across figures.
from manim import *
import numpy as np

_TEMPLATE = TexTemplate()
_TEMPLATE.add_to_preamble(r"\usepackage{upgreek}")
config.tex_template = _TEMPLATE

# Squarer frame than the 16:9 default, so the diagram fills the image.
config.pixel_width = 1400
config.pixel_height = 1080


class MarginWidthDerivation(Scene):
    def construct(self):
        self.camera.background_color = "#222222"
        light = "#e6e6e6"
        blue = "#5dade2"        # boundary only
        green = "#2ecc71"
        red = "#e74c3c"
        yellow = "#f1c40f"
        orange = "#e67e22"
        muted = "#aab7c4"       # margin lines

        scale = 1.75
        shift = LEFT * 1.6

        # s = x1 + x2 (0 on the boundary, +/-m on the margin lines); t = tangential.
        def pt(s, t):
            return np.array([s / 2 + t, s / 2 - t, 0.0]) * scale + shift

        m = 1.35
        T = 1.25
        w_hat = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)

        def line_pts(c):
            return pt(c, -T), pt(c, T)

        # The boundary sits exactly halfway between the margin lines. Drawn
        # faint so it orients the reader without competing with the two lines
        # whose separation is the point of the figure.
        boundary = Line(*line_pts(0), color=blue, stroke_width=3.5,
                        stroke_opacity=0.55)
        up = DashedLine(*line_pts(m), color=muted, stroke_width=4, dash_length=0.16)
        lo = DashedLine(*line_pts(-m), color=muted, stroke_width=4, dash_length=0.16)

        lblp = MathTex(r"\mathbf{w}^{\intercal}\mathbf{x}+b=+1", color=muted,
                       font_size=28).next_to(up.get_end(), RIGHT, buff=0.15)
        lbl0 = MathTex(r"\mathbf{w}^{\intercal}\mathbf{x}+b=0", color=blue,
                       font_size=28).next_to(boundary.get_end(), RIGHT, buff=0.15)
        lbln = MathTex(r"\mathbf{w}^{\intercal}\mathbf{x}+b=-1", color=muted,
                       font_size=28).next_to(lo.get_end(), RIGHT, buff=0.15)

        # A little context so the figure is anchored to data like its neighbours:
        # admitted students beyond the +1 line, rejected beyond the -1 line.
        # Placed well clear of the w arrow's tip: an earlier layout parked a
        # triangle directly on top of the "w" label.
        ctx = VGroup()
        for s, t in [(2.3, 0.6), (2.3, 1.0), (2.9, 0.6)]:
            ctx.add(Triangle(color=green, fill_opacity=1.0).scale(0.12)
                    .move_to(pt(s, t)))
        for s, t in [(-2.3, -0.6), (-2.3, -0.2), (-2.6, -0.5)]:
            ctx.add(MathTex(r"\times", color=red, font_size=36).move_to(pt(s, t)))

        # x_+ on the +1 line; x_- straight along the normal on the -1 line.
        tang = 0.05
        xp = pt(m, tang)
        xm = pt(-m, tang)
        dot_p = Dot(xp, color=green, radius=0.09)
        dot_m = Dot(xm, color=light, radius=0.085)
        # Straight up from the dot: up-right is the w arrow, down-right is the
        # brace, and down-left is the right-angle tick, so UP is the only clear
        # side. Backing plates keep both labels legible over the dashed lines.
        lbl_xp = MathTex(r"\mathbf{x}_{+}", color=green, font_size=36).next_to(
            dot_p, UP, buff=0.18)
        lbl_xm = MathTex(r"\mathbf{x}_{-}", color=light, font_size=36).next_to(
            dot_m, LEFT, buff=0.22)
        lbl_xp_bg = BackgroundRectangle(lbl_xp, color="#222222",
                                        fill_opacity=1.0, buff=0.08)
        lbl_xm_bg = BackgroundRectangle(lbl_xm, color="#222222",
                                        fill_opacity=1.0, buff=0.08)

        # The perpendicular walk from x_+ to x_- (direction -w/||w||).
        walk = Arrow(xp, xm, color=yellow, stroke_width=6, buff=0.0,
                     max_tip_length_to_length_ratio=0.09)
        walk_line = Line(xp, xm)
        perp = np.array([w_hat[1], -w_hat[0], 0.0])   # tangential, toward lower-right
        brace = Brace(walk_line, direction=perp, color=light, buff=0.12)
        brace_lbl = MathTex(r"t = \frac{2}{\|\mathbf{w}\|}", color=yellow, font_size=40)
        brace.put_at_tip(brace_lbl, buff=0.2)

        # The weight vector w (normal), drawn from x_+ opposite to the walk.
        w_arrow = Arrow(xp, xp + 1.25 * w_hat, color=orange, stroke_width=6, buff=0.0,
                        max_tip_length_to_length_ratio=0.2)
        w_lbl = MathTex(r"\mathbf{w}", color=orange, font_size=36)
        w_lbl.next_to(w_arrow.get_end(), UR, buff=0.05)

        # Right-angle ticks: the walk meets each line perpendicularly.
        ra_p = RightAngle(up, walk_line, length=0.24, quadrant=(-1, -1),
                          color=light, stroke_width=2.5)
        ra_m = RightAngle(lo, walk_line, length=0.24, quadrant=(1, 1),
                          color=light, stroke_width=2.5)

        self.add(boundary, up, lo, lblp, lbl0, lbln, ctx, walk, brace, brace_lbl,
                 w_arrow, w_lbl, ra_p, ra_m, dot_p, dot_m,
                 lbl_xp_bg, lbl_xp, lbl_xm_bg, lbl_xm)


# Render (from the repo root), saving only the final frame as a PNG. The script
# pins its own pixel dimensions, so -qh and -ql give the same size here:
#   uv run manim -qh -s blog/published/support-vector-machines/_src/margin_width_derivation.py MarginWidthDerivation
# Manim writes the PNG to media\images\margin_width_derivation\MarginWidthDerivation.png
# Move it to blog/published/support-vector-machines/images/margin_width_derivation.png
# Embed (dark background -> no .invert):
#   ![...](images/margin_width_derivation.png){#fig-margin_width_derivation fig-align="center" width=80%}

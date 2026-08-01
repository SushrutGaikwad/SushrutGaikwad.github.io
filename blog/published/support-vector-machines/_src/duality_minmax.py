# Runs via the root uv environment with uv run (see render command at the bottom).
#
# STATIC PNG: weak vs strong duality. The primal is a min-of-max, the dual a
# max-of-min; in general d* <= p* (weak duality). Under convexity + Slater's
# condition the gap closes, d* = p*, so we may solve the easier (dual) problem.
#
# The first draft of this figure just re-typeset two equations the reader had
# already met three paragraphs earlier, which is not worth a figure. The version
# below adds the thing prose handles badly: an OBJECTIVE-VALUE AXIS showing d*
# and p* as two points with the duality gap braced between them, and a second
# axis where the gap has closed. That is the geometric content of "weak" vs
# "strong" duality, and it is what the reader needs to carry into the SVM dual.
from manim import *

_TEMPLATE = TexTemplate()
_TEMPLATE.add_to_preamble(r"\usepackage{upgreek}")
config.tex_template = _TEMPLATE


class DualityMinMax(Scene):
    def construct(self):
        self.camera.background_color = "#222222"
        light = "#e6e6e6"
        blue = "#5dade2"
        green = "#2ecc71"
        yellow = "#f1c40f"
        muted = "#aab7c4"

        title = Tex(r"Primal vs.\ dual: two numbers on one axis",
                    font_size=40, color=light).to_edge(UP, buff=0.35)

        # --- the two problems, stated compactly ------------------------------
        dual_body = VGroup(
            Tex(r"\textbf{Dual}", font_size=32, color=green),
            MathTex(r"d^{*} = \max_{\boldsymbol{\upalpha},\boldsymbol{\upbeta}:\,\alpha_i \geq 0}"
                    r"\ \min_{w}\ \mathcal{L}", font_size=32, color=light),
        ).arrange(DOWN, buff=0.22)
        dual = VGroup(SurroundingRectangle(dual_body, color=green, buff=0.25,
                                           corner_radius=0.12), dual_body)
        dual.move_to(LEFT * 3.6 + UP * 1.95)

        primal_body = VGroup(
            Tex(r"\textbf{Primal}", font_size=32, color=blue),
            MathTex(r"p^{*} = \min_{w}\ \max_{\boldsymbol{\upalpha},\boldsymbol{\upbeta}:\,\alpha_i \geq 0}"
                    r"\ \mathcal{L}", font_size=32, color=light),
        ).arrange(DOWN, buff=0.22)
        primal = VGroup(SurroundingRectangle(primal_body, color=blue, buff=0.25,
                                             corner_radius=0.12), primal_body)
        primal.move_to(RIGHT * 3.6 + UP * 1.95)

        # --- axis 1: the general case, a gap ---------------------------------
        y1 = -0.35
        ax1 = Arrow(LEFT * 5.2 + UP * y1, RIGHT * 5.2 + UP * y1, color=muted,
                    stroke_width=3, buff=0.0, max_tip_length_to_length_ratio=0.025)
        ax1_lbl = Tex(r"objective value", font_size=26, color=muted)
        ax1_lbl.next_to(ax1.get_end(), DOWN, buff=0.18).shift(LEFT * 0.35)

        d_pos = LEFT * 2.3 + UP * y1
        p_pos = RIGHT * 1.5 + UP * y1
        d_dot = Dot(d_pos, color=green, radius=0.11)
        p_dot = Dot(p_pos, color=blue, radius=0.11)
        d_lbl = MathTex(r"d^{*}", color=green, font_size=40).next_to(d_dot, UP, buff=0.18)
        p_lbl = MathTex(r"p^{*}", color=blue, font_size=40).next_to(p_dot, UP, buff=0.18)

        gap_line = Line(d_pos, p_pos)
        gap_brace = Brace(gap_line, direction=DOWN, color=yellow, buff=0.16)
        gap_lbl = Tex(r"duality gap", font_size=30, color=yellow)
        gap_brace.put_at_tip(gap_lbl, buff=0.14)

        weak_note = MathTex(r"d^{*} \leq p^{*} \quad\text{always (weak duality)}",
                            font_size=32, color=light)
        weak_note.next_to(ax1, UP, buff=0.55).shift(LEFT * 0.2)

        # --- axis 2: under Slater + convexity, the gap closes ----------------
        # Lifted from -3.0: the two lines of text hanging below this axis were
        # running off the bottom of the frame.
        y2 = -2.45
        ax2 = Arrow(LEFT * 5.2 + UP * y2, RIGHT * 5.2 + UP * y2, color=muted,
                    stroke_width=3, buff=0.0, max_tip_length_to_length_ratio=0.025)
        both_pos = LEFT * 0.4 + UP * y2
        both_dot = Dot(both_pos, color=yellow, radius=0.13)
        both_lbl = MathTex(r"d^{*} = p^{*}", color=yellow, font_size=40)
        both_lbl.next_to(both_dot, UP, buff=0.18)

        strong_note = MathTex(
            r"\text{convex } f, g_i \;+\; \text{affine } h_i \;+\; \text{Slater}"
            r"\quad\Longrightarrow\quad \text{the gap closes}",
            font_size=30, color=light)
        strong_note.next_to(ax2, DOWN, buff=0.26)

        payoff = Tex(r"so we may solve whichever problem is easier",
                     font_size=28, color=yellow)
        payoff.next_to(strong_note, DOWN, buff=0.2)

        self.add(title, dual, primal,
                 ax1, ax1_lbl, weak_note, d_dot, p_dot, d_lbl, p_lbl,
                 gap_brace, gap_lbl,
                 ax2, both_dot, both_lbl, strong_note, payoff)


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/support-vector-machines/_src/duality_minmax.py DualityMinMax
# Manim writes the PNG to media\images\duality_minmax\DualityMinMax.png
# Move it to blog/published/support-vector-machines/images/duality_minmax.png
# Embed (dark background -> no .invert):
#   ![...](images/duality_minmax.png){#fig-duality_minmax fig-align="center" width=100%}

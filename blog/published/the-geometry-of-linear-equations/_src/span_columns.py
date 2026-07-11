# Runs via the root uv environment with `uv run`.
"""Span animations for Post 1: which targets b can the columns reach?

Each scene keeps dimmed reference copies of the columns a1, a2 in the
background, then smoothly varies the weights x and y. Every frame shows the
live scaled vectors x*a1 (blue) and y*a2 (orange, tip to tail), their resultant
b = x*a1 + y*a2 (red), a numeric readout of x and y, and the faint traced path
of b's tip (the set of reachable targets).
  - SpanInvertible: a1 = (1,3), a2 = (2,1) point in different directions, so b
    roams over a 2-D region (the whole plane is reachable).
  - SpanSingular: a2 = 2*a1, so b is pinned to a single line; a target off the
    line is unreachable.
All text uses MathTex/Tex (LaTeX / Computer Modern); Manim's template lacks the
physics package, so bold uses \\mathbf, not \\vb.
"""

import numpy as np
from manim import *

BLUE_A = "#5dade2"
ORANGE_A = "#f5b041"
RED_B = "#ec7063"
LIGHT = "#e6e6e6"
GRAY_L = "#9aa0a6"


def make_plane():
    plane = NumberPlane(
        x_range=[-7, 7, 1],
        y_range=[-7, 7, 1],
        x_length=7.7,
        y_length=7.7,
        background_line_style={"stroke_color": "#3a3a3a", "stroke_width": 1},
        axis_config={"stroke_color": "#777777", "include_ticks": False},
    )
    plane.to_edge(LEFT, buff=0.4)
    return plane


def safe_arrow(start, end, color, sw=5, tip_ratio=0.28):
    """An Arrow, or nothing if it would be degenerately short."""
    if np.linalg.norm(np.array(end) - np.array(start)) < 0.12:
        return VGroup()
    return Arrow(start, end, buff=0, color=color, stroke_width=sw,
                 max_tip_length_to_length_ratio=tip_ratio)


def readout(plane, xy_func):
    """A right-side panel: the formula and live x, y values."""
    formula = MathTex(r"x\,\mathbf{a}_1", r"+", r"y\,\mathbf{a}_2", r"=", r"\mathbf{b}")
    formula[0].set_color(BLUE_A)
    formula[2].set_color(ORANGE_A)
    formula[4].set_color(RED_B)
    formula.scale(0.85)

    x_num = DecimalNumber(0.0, num_decimal_places=2, include_sign=True, color=BLUE_A, font_size=36)
    y_num = DecimalNumber(0.0, num_decimal_places=2, include_sign=True, color=ORANGE_A, font_size=36)
    x_num.add_updater(lambda m: m.set_value(xy_func()[0]))
    y_num.add_updater(lambda m: m.set_value(xy_func()[1]))
    x_row = VGroup(MathTex(r"x =", color=LIGHT).scale(0.85), x_num).arrange(RIGHT, buff=0.18)
    y_row = VGroup(MathTex(r"y =", color=LIGHT).scale(0.85), y_num).arrange(RIGHT, buff=0.18)
    rows = VGroup(x_row, y_row).arrange(DOWN, aligned_edge=LEFT, buff=0.3)

    panel = VGroup(formula, rows).arrange(DOWN, buff=0.55)
    panel.to_edge(RIGHT, buff=0.7)
    return panel


def caption(text):
    cap = Tex(text, color=LIGHT).scale(0.65).to_edge(DOWN, buff=0.3)
    cap.add_background_rectangle(color="#222222", opacity=0.9, buff=0.12)
    return cap


class _SpanBase(Scene):
    A1 = np.array([1.0, 3.0])
    A2 = np.array([2.0, 1.0])
    AMP = 1.4
    CAP = ""

    def construct(self):
        self.camera.background_color = "#222222"
        plane = make_plane()
        self.add(plane)
        A1, A2 = self.A1, self.A2
        o = plane.c2p(0, 0)

        # Dimmed reference columns a1, a2 (the building blocks).
        ref_a1 = Arrow(o, plane.c2p(*A1), buff=0, color=BLUE_A, stroke_width=4).set_opacity(0.35)
        ref_a2 = Arrow(o, plane.c2p(*A2), buff=0, color=ORANGE_A, stroke_width=4).set_opacity(0.35)
        ref_a1_l = MathTex(r"\mathbf{a}_1", color=BLUE_A).scale(0.7).set_opacity(0.55).next_to(ref_a1.get_end(), LEFT, buff=0.1)
        ref_a2_l = MathTex(r"\mathbf{a}_2", color=ORANGE_A).scale(0.7).set_opacity(0.55).next_to(ref_a2.get_end(), RIGHT, buff=0.1)
        self.add(ref_a1, ref_a2, ref_a1_l, ref_a2_l)

        s = ValueTracker(0.0)

        def xy():
            sv = s.get_value()
            x = self.AMP * np.sin(2 * np.pi * 2 * sv)
            y = self.AMP * np.sin(2 * np.pi * 3 * sv + np.pi / 2)
            return x, y

        def res_point():
            x, y = xy()
            p = x * A1 + y * A2
            return plane.c2p(p[0], p[1])

        # Faint trail of the resultant's tip = the reachable set.
        tip = always_redraw(lambda: Dot(res_point(), radius=0.05, color=RED_B))
        path = TracedPath(tip.get_center, stroke_color=RED_B, stroke_width=2, stroke_opacity=0.45)
        self.add(path)

        # Live tip-to-tail construction: x*a1, then y*a2, then the resultant b.
        def construction():
            x, y = xy()
            p1 = plane.c2p(*(x * A1))
            p2 = plane.c2p(*(x * A1 + y * A2))
            g = VGroup()
            g.add(safe_arrow(o, p1, BLUE_A, sw=5))
            g.add(safe_arrow(p1, p2, ORANGE_A, sw=5))
            g.add(safe_arrow(o, p2, RED_B, sw=6, tip_ratio=0.24))
            return g

        self.add(always_redraw(construction), tip)
        self.add(readout(plane, xy))

        self.play(s.animate.set_value(1.0), run_time=10, rate_func=linear)
        self.wait(0.2)
        self.extra(plane)
        self.play(FadeIn(caption(self.CAP)))
        self.wait(1.4)

    def extra(self, plane):
        """Hook for scene-specific final annotations."""
        pass


class SpanInvertible(_SpanBase):
    A1 = np.array([1.0, 3.0])
    A2 = np.array([2.0, 1.0])
    AMP = 1.4
    CAP = r"The tip of $\mathbf{b}$ roams over a 2-D region: every $\mathbf{b}$ is reachable."


class SpanSingular(_SpanBase):
    A1 = np.array([1.0, 3.0])
    A2 = np.array([2.0, 6.0])  # a2 = 2 a1: collinear columns
    AMP = 0.6
    CAP = r"However $x$ and $y$ vary, $\mathbf{b}$ never leaves this line."

    def extra(self, plane):
        # A target off the line is unreachable.
        b_off = plane.c2p(4.0, 1.0)
        dot_b = Dot(b_off, color=RED_B, radius=0.08)
        cross = Cross(scale_factor=0.18).move_to(b_off).set_color(RED_B)
        b_lab = MathTex(r"\mathbf{b}", color=RED_B).scale(0.85).next_to(dot_b, RIGHT, buff=0.14)
        self.play(FadeIn(dot_b), FadeIn(b_lab))
        self.play(Create(cross))
        self.wait(0.2)


# Render MP4s (a -qh GIF would be huge; MP4 is tiny), from the repo root, using a
# unique --media_dir so the shared top-level media/ folder is never touched:
#   uv run manim -qh --media_dir media_geo blog/published/the-geometry-of-linear-equations/_src/span_columns.py SpanInvertible
#   uv run manim -qh --media_dir media_geo blog/published/the-geometry-of-linear-equations/_src/span_columns.py SpanSingular
# Use -ql while iterating. Move
#   media_geo/videos/span_columns/<quality>/SpanInvertible.mp4 -> images/span_invertible.mp4
#   media_geo/videos/span_columns/<quality>/SpanSingular.mp4   -> images/span_singular.mp4
# then delete media_geo. Embed with {{< video images/span_invertible.mp4 >}} and {{< video images/span_singular.mp4 >}}.

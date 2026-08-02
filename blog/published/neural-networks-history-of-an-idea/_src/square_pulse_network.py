# Runs via the root uv environment with uv run (see render command at the bottom).
"""Three units make a bump.

Two step units, one turning on at T1 and the other at T2, are subtracted from
one another. Below T1 neither is on, above T2 both are and they cancel, and in
between only the first is on. That leaves a pulse of height 1 sitting exactly
where we chose to put it, which is the only building block universal
approximation needs.
"""

from manim import *

_TEMPLATE = TexTemplate()
_TEMPLATE.add_to_preamble(r"\usepackage{upgreek}")
config.tex_template = _TEMPLATE

LIGHT = "#e6e6e6"
MUTED = "#9a9a9a"
QUIET = "#5dade2"
ACCENT = "#f1c40f"
GOOD = "#2ecc71"


class SquarePulseNetwork(Scene):
    """x into two thresholds, summed with +1 and -1, out comes a square pulse."""

    def construct(self):
        self.camera.background_color = "#222222"

        title = Tex(r"A three-unit network that outputs 1 between "
                    r"$T_1$ and $T_2$, and 0 everywhere else",
                    font_size=34, color=LIGHT).to_edge(UP, buff=0.28)

        r = 0.46
        x_pt = np.array([-4.75, 0.05, 0.0])
        u1_pt = np.array([-2.05, 1.55, 0.0])
        u2_pt = np.array([-2.05, -1.45, 0.0])
        sum_pt = np.array([0.95, 0.05, 0.0])

        x_lbl = MathTex("x", font_size=44, color=QUIET).move_to(x_pt)

        u1 = self.unit(u1_pt, r, "T_1")
        u2 = self.unit(u2_pt, r, "T_2")
        sum_body = Circle(radius=0.55, color=GOOD, stroke_width=4.5,
                          fill_color="#333333", fill_opacity=1.0).move_to(sum_pt)
        sum_sym = MathTex("+", font_size=44, color=GOOD).move_to(sum_pt)

        edges = VGroup(
            self.edge(x_pt, u1_pt, r, "1", 0.32),
            self.edge(x_pt, u2_pt, r, "1", -0.32),
            self.edge(u1_pt, sum_pt, 0.55, "1", 0.32, colour=GOOD,
                      start_radius=r),
            self.edge(u2_pt, sum_pt, 0.55, "-1", -0.32, colour=GOOD,
                      start_radius=r),
        )

        out_arrow = Arrow(sum_pt + np.array([0.55, 0, 0]),
                          sum_pt + np.array([1.75, 0, 0]), color=GOOD,
                          stroke_width=4.5, buff=0.04,
                          max_tip_length_to_length_ratio=0.14)

        # ---- The three little pictures: two steps and their difference ----
        step1 = self.step_glyph(r"T_1", QUIET)
        step1.next_to(u1, UP, buff=0.36)
        step2 = self.step_glyph(r"T_2", QUIET)
        step2.next_to(u2, DOWN, buff=0.36)
        pulse = self.pulse_glyph()
        pulse.next_to(out_arrow, RIGHT, buff=0.32)

        rule = MathTex(
            r"f(x) \;=\; \mathbf{1}\left[x \geq T_1\right] "
            r"\;-\; \mathbf{1}\left[x \geq T_2\right]",
            font_size=36, color=LIGHT)
        note = Tex(r"$T_1$ and $T_2$ are ours to choose, so the pulse "
                   r"can sit anywhere and be as narrow as we like",
                   font_size=26, color=MUTED)
        bottom = VGroup(rule, note).arrange(DOWN, buff=0.22)
        bottom.to_edge(DOWN, buff=0.22)

        # Fit the network into whatever room the title and the closing lines
        # leave, rather than trusting hand-picked coordinates not to collide.
        diagram = VGroup(edges, x_lbl, u1, u2, sum_body, sum_sym, out_arrow,
                         step1, step2, pulse)
        top = title.get_bottom()[1] - 0.20
        floor = bottom.get_top()[1] + 0.20
        if diagram.height > top - floor:
            diagram.scale_to_fit_height(top - floor)
        diagram.move_to([diagram.get_center()[0], (top + floor) / 2, 0.0])

        self.add(title, diagram, bottom)

    def unit(self, centre, radius, label):
        body = Circle(radius=radius, color=LIGHT, stroke_width=4,
                      fill_color="#333333", fill_opacity=1.0).move_to(centre)
        thr = MathTex(label, font_size=30, color=ACCENT).move_to(centre)
        return VGroup(body, thr)

    def edge(self, start, end, radius, weight, label_offset, colour=QUIET,
             start_radius=0.34):
        direction = normalize(end - start)
        a = start + start_radius * direction
        b = end - radius * direction
        arrow = Arrow(a, b, color=colour, stroke_width=4, buff=0.02,
                      max_tip_length_to_length_ratio=0.07)
        anchor = a + 0.45 * (b - a)
        lbl = MathTex(weight, font_size=30, color=ACCENT
                      ).move_to(anchor + np.array([0, label_offset, 0]))
        return VGroup(arrow, lbl)

    def step_glyph(self, tick_label, colour):
        """A tiny plot of a step that switches on at one threshold."""
        axis = Line([-0.85, 0, 0], [0.95, 0, 0], color=MUTED, stroke_width=2.5)
        curve = VGroup(
            Line([-0.85, 0, 0], [0.0, 0, 0], color=colour, stroke_width=4),
            Line([0.0, 0, 0], [0.0, 0.52, 0], color=colour, stroke_width=4),
            Line([0.0, 0.52, 0], [0.95, 0.52, 0], color=colour, stroke_width=4),
        )
        tick = MathTex(tick_label, font_size=26, color=ACCENT
                       ).move_to([0.0, -0.30, 0])
        return VGroup(axis, curve, tick)

    def pulse_glyph(self):
        axis = Line([-1.05, 0, 0], [1.15, 0, 0], color=MUTED, stroke_width=2.5)
        curve = VGroup(
            Line([-1.05, 0, 0], [-0.35, 0, 0], color=ACCENT, stroke_width=4),
            Line([-0.35, 0, 0], [-0.35, 0.62, 0], color=ACCENT, stroke_width=4),
            Line([-0.35, 0.62, 0], [0.35, 0.62, 0], color=ACCENT, stroke_width=4),
            Line([0.35, 0.62, 0], [0.35, 0, 0], color=ACCENT, stroke_width=4),
            Line([0.35, 0, 0], [1.15, 0, 0], color=ACCENT, stroke_width=4),
        )
        t1 = MathTex("T_1", font_size=24, color=MUTED).move_to([-0.35, -0.30, 0])
        t2 = MathTex("T_2", font_size=24, color=MUTED).move_to([0.40, -0.30, 0])
        lbl = MathTex("f(x)", font_size=30, color=ACCENT).move_to([0.0, 1.02, 0])
        return VGroup(axis, curve, t1, t2, lbl)


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/neural-networks-history-of-an-idea/_src/square_pulse_network.py SquarePulseNetwork
# Manim writes the PNG to media\images\square_pulse_network\SquarePulseNetwork_ManimCE_v<version>.png
# Move it to blog/published/neural-networks-history-of-an-idea/images/square_pulse_network.png
# Embed with:
#   ![...](images/square_pulse_network.png){#fig-square_pulse_network fig-align="center" width=100%}

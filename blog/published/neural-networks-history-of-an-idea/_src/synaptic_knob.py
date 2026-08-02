# Runs via the root uv environment with uv run (see render command at the bottom).
"""What Hebb's rule is a model of.

The two cells never touch. A bulb at the end of the first cell's axon releases
chemicals across a gap onto the second cell's dendrite, and Hebb's proposal is
that every time the first cell successfully helps fire the second, that bulb
grows a little. A bigger bulb is a stronger connection, which in the mathematics
is a larger weight.
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


class SynapticKnob(Scene):
    """The same synapse before and after repeated successful co-firing."""

    def construct(self):
        self.camera.background_color = "#222222"

        title = Tex(r"``Neurons that fire together wire together''",
                    font_size=40, color=LIGHT).to_edge(UP, buff=0.30)

        before = self.synapse_panel(0.42, r"\text{before}", QUIET,
                                    r"a small knob, a weak connection")
        after = self.synapse_panel(0.78, r"\text{after many co-firings}", GOOD,
                                   r"a larger knob, a stronger connection")
        panels = VGroup(before, after).arrange(RIGHT, buff=0.85)
        panels.scale_to_fit_width(12.4)
        panels.next_to(title, DOWN, buff=0.45)

        rule = MathTex(r"w_{xy} \;\leftarrow\; w_{xy} \;+\; \eta\, x\, y",
                       font_size=42, color=ACCENT)
        note = Tex(r"the weight rises whenever both cells fire, and there is "
                   r"no term anywhere that can bring it back down",
                   font_size=28, color=MUTED)
        bottom = VGroup(rule, note).arrange(DOWN, buff=0.24)
        bottom.to_edge(DOWN, buff=0.28)

        self.add(title, panels, bottom)

    def synapse_panel(self, knob_radius, heading, colour, caption):
        # Cell X's axon, coming down from the top left.
        axon = Line([-1.55, 1.15, 0], [-0.15, 0.42, 0], color=QUIET,
                    stroke_width=6)
        knob = Circle(radius=knob_radius, color=colour, stroke_width=4,
                      fill_color=colour, fill_opacity=0.35)
        knob.move_to([0.10, 0.28, 0])

        # The gap, and cell Y's dendrite below it.
        dendrite = Line([-1.75, -0.72, 0], [1.75, -0.72, 0], color=ACCENT,
                        stroke_width=6)
        gap = DashedLine(knob.get_bottom(), [knob.get_center()[0], -0.72, 0],
                         color=MUTED, stroke_width=2.5, dash_length=0.08)

        x_lbl = MathTex("x", font_size=34, color=QUIET).move_to([-1.85, 1.28, 0])
        y_lbl = MathTex("y", font_size=34, color=ACCENT).move_to([2.05, -0.72, 0])

        drawing = VGroup(axon, gap, dendrite, knob, x_lbl, y_lbl)
        label = VGroup(
            MathTex(heading, font_size=30, color=colour),
            Tex(caption, font_size=25, color=MUTED),
        ).arrange(DOWN, buff=0.14)
        body = VGroup(drawing, label).arrange(DOWN, buff=0.34)
        frame = SurroundingRectangle(body, color="#555555", buff=0.30,
                                     corner_radius=0.14)
        return VGroup(frame, body)


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/neural-networks-history-of-an-idea/_src/synaptic_knob.py SynapticKnob
# Manim writes the PNG to media\images\synaptic_knob\SynapticKnob_ManimCE_v<version>.png
# Move it to blog/published/neural-networks-history-of-an-idea/images/synaptic_knob.png
# Embed with:
#   ![...](images/synaptic_knob.png){#fig-synaptic_knob fig-align="center" width=90%}

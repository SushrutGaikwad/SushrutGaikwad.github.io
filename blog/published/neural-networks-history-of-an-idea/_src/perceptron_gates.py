# Runs via the root uv environment with uv run (see render command at the bottom).
"""Three gates, one unit each, differing only in the numbers.

Same shape, same rule, three different Boolean functions: move the threshold and
AND becomes OR, flip a weight's sign and you get NOT. The fourth panel is the
one that has no numbers to fill in, and finding out why is what breaks the
single-unit story open.
"""

from manim import *

_TEMPLATE = TexTemplate()
_TEMPLATE.add_to_preamble(r"\usepackage{upgreek}")
config.tex_template = _TEMPLATE

LIGHT = "#e6e6e6"
MUTED = "#9a9a9a"
FIRE = "#e74c3c"
QUIET = "#5dade2"
ACCENT = "#f1c40f"


class PerceptronGates(Scene):
    """AND, OR, NOT as single threshold units, and XOR as the one that is not."""

    def construct(self):
        self.camera.background_color = "#222222"

        title = Tex(r"Numbers on the edges are weights; the number inside "
                    r"the circle is the threshold",
                    font_size=30, color=MUTED).to_edge(UP, buff=0.25)

        panels = VGroup(
            self.gate([("X", "1"), ("Y", "1")], "2", r"X \wedge Y", LIGHT,
                      r"needs both, since only $1+1$ reaches 2"),
            self.gate([("X", "1"), ("Y", "1")], "1", r"X \vee Y", LIGHT,
                      r"either one is enough to reach 1"),
            self.gate([("X", "-1")], "0", r"\overline{X}", LIGHT,
                      r"one input, negated: fires when $X = 0$"),
            self.gate([("X", "?"), ("Y", "?")], "?", r"X \oplus Y", FIRE,
                      r"no numbers exist that do this"),
        ).arrange_in_grid(rows=2, cols=2, buff=(0.9, 0.55))

        panels.scale_to_fit_width(12.6)
        if panels.height > 6.4:
            panels.scale_to_fit_height(6.4)
        panels.next_to(title, DOWN, buff=0.30)

        self.add(title, panels)

    def gate(self, inputs, threshold, output, colour, caption):
        """One threshold unit with its inputs, weights, threshold and output."""
        body = Circle(radius=0.52, color=colour, stroke_width=4.5,
                      fill_color="#333333", fill_opacity=1.0)
        thr = MathTex(threshold, font_size=34, color=ACCENT).move_to(body.get_center())

        wires, labels, weights = VGroup(), VGroup(), VGroup()
        offsets = [0.85, -0.85] if len(inputs) == 2 else [0.0]
        for (name, w), dy in zip(inputs, offsets):
            start = np.array([-2.25, dy, 0.0])
            end = 0.52 * normalize(ORIGIN - start) + ORIGIN
            wires.add(Arrow(start, end, color=QUIET, stroke_width=4, buff=0.03,
                            max_tip_length_to_length_ratio=0.10))
            labels.add(MathTex(name, font_size=34, color=QUIET
                               ).next_to(start, LEFT, buff=0.16))
            mid = start + 0.5 * (end - start)
            weights.add(MathTex(w, font_size=30, color=ACCENT
                                ).move_to(mid + np.array([0, 0.32, 0])))

        out = Arrow([0.52, 0, 0], [1.55, 0, 0], color=colour, stroke_width=4,
                    buff=0.03, max_tip_length_to_length_ratio=0.14)
        out_lbl = MathTex(output, font_size=34, color=colour
                          ).next_to(out, RIGHT, buff=0.18)

        drawing = VGroup(wires, labels, weights, body, thr, out, out_lbl)
        cap = Tex(caption, font_size=25, color=MUTED)
        stacked = VGroup(drawing, cap).arrange(DOWN, buff=0.34)
        frame = SurroundingRectangle(stacked, color="#555555", buff=0.28,
                                     corner_radius=0.12)
        return VGroup(frame, stacked)


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/neural-networks-history-of-an-idea/_src/perceptron_gates.py PerceptronGates
# Manim writes the PNG to media\images\perceptron_gates\PerceptronGates_ManimCE_v<version>.png
# Move it to blog/published/neural-networks-history-of-an-idea/images/perceptron_gates.png
# Embed with:
#   ![...](images/perceptron_gates.png){#fig-perceptron_gates fig-align="center" width=100%}

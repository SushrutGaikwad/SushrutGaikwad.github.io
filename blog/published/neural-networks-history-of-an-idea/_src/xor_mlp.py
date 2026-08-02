# Runs via the root uv environment with uv run (see render command at the bottom).
"""XOR, in two layers.

XOR is not a single half-plane, but it is the AND of two things that are:
"at least one fired" and "not both fired". Give each of those its own unit, then
AND the pair. The two units in the middle are hidden only in the sense that
nobody asks them for an answer; the wiring is right there in the open.
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


class XorMLP(Scene):
    """Two hidden threshold units and one output unit, with every number shown."""

    def construct(self):
        self.camera.background_color = "#222222"

        title = Tex(r"$X \oplus Y \;=\; (X \vee Y) \;\wedge\; "
                    r"(\overline{X} \vee \overline{Y})$",
                    font_size=40, color=LIGHT).to_edge(UP, buff=0.30)

        x_in = np.array([-4.85, 1.30, 0.0])
        y_in = np.array([-4.85, -1.70, 0.0])
        h1_c = np.array([-1.30, 1.30, 0.0])
        h2_c = np.array([-1.30, -1.70, 0.0])
        out_c = np.array([2.30, -0.20, 0.0])
        r = 0.58

        h1 = self.unit(h1_c, r, "1")
        h2 = self.unit(h2_c, r, "-1")
        out = self.unit(out_c, r, "2", colour=GOOD)

        x_lbl = MathTex("X", font_size=42, color=QUIET).move_to(x_in)
        y_lbl = MathTex("Y", font_size=42, color=QUIET).move_to(y_in)

        # The two diagonal edges cross, so their labels sit at different points
        # along the edge (t) to keep them apart.
        edges = VGroup(
            self.edge(x_in, h1_c, r, "1", 0.30, t=0.45),
            self.edge(y_in, h1_c, r, "1", 0.30, t=0.75),
            self.edge(x_in, h2_c, r, "-1", -0.34, t=0.75),
            self.edge(y_in, h2_c, r, "-1", -0.34, t=0.45),
            self.edge(h1_c, out_c, r, "1", 0.32, t=0.45, colour=GOOD),
            self.edge(h2_c, out_c, r, "1", -0.32, t=0.45, colour=GOOD),
        )

        h1_lbl = MathTex(r"X \vee Y", font_size=34, color=ACCENT)
        h1_lbl.next_to(h1[0], UP, buff=0.22)
        h2_lbl = MathTex(r"\overline{X} \vee \overline{Y}", font_size=34,
                         color=ACCENT)
        h2_lbl.next_to(h2[0], DOWN, buff=0.22)

        out_arrow = Arrow(out_c + np.array([r, 0, 0]),
                          out_c + np.array([r + 1.25, 0, 0]), color=GOOD,
                          stroke_width=4.5, buff=0.04,
                          max_tip_length_to_length_ratio=0.14)
        out_lbl = MathTex(r"X \oplus Y", font_size=38, color=GOOD
                          ).next_to(out_arrow, RIGHT, buff=0.20)

        hidden_box = DashedVMobject(
            SurroundingRectangle(VGroup(h1, h2), color=MUTED, buff=0.62,
                                 corner_radius=0.16), num_dashes=48)
        hidden_lbl = Tex(r"hidden layer", font_size=28, color=MUTED)
        hidden_lbl.next_to(hidden_box, UP, buff=0.14)

        note = Tex(r"Numbers on the edges are weights; the number inside "
                   r"each circle is that unit's threshold.",
                   font_size=28, color=MUTED).to_edge(DOWN, buff=0.30)

        self.add(title, hidden_box, hidden_lbl, edges, h1, h2, out,
                 x_lbl, y_lbl, h1_lbl, h2_lbl, out_arrow, out_lbl, note)

    def unit(self, centre, radius, threshold, colour=LIGHT):
        body = Circle(radius=radius, color=colour, stroke_width=4.5,
                      fill_color="#333333", fill_opacity=1.0).move_to(centre)
        thr = MathTex(threshold, font_size=34, color=ACCENT).move_to(centre)
        return VGroup(body, thr)

    def edge(self, start, end, radius, weight, label_offset, t=0.45,
             colour=QUIET):
        direction = normalize(end - start)
        a = start + 0.42 * direction
        b = end - radius * direction
        arrow = Arrow(a, b, color=colour, stroke_width=4, buff=0.02,
                      max_tip_length_to_length_ratio=0.07)
        anchor = a + t * (b - a)
        lbl = MathTex(weight, font_size=30, color=ACCENT
                      ).move_to(anchor + np.array([0, label_offset, 0]))
        return VGroup(arrow, lbl)


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/neural-networks-history-of-an-idea/_src/xor_mlp.py XorMLP
# Manim writes the PNG to media\images\xor_mlp\XorMLP_ManimCE_v<version>.png
# Move it to blog/published/neural-networks-history-of-an-idea/images/xor_mlp.png
# Embed with:
#   ![...](images/xor_mlp.png){#fig-xor_mlp fig-align="center" width=100%}

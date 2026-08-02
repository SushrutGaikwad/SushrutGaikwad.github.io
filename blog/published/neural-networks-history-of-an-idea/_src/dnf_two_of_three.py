# Runs via the root uv environment with uv run (see render command at the bottom).
"""The general recipe, on a function that genuinely needs two layers.

"Exactly two of the three sensors fired" cannot be done by one unit: the input
(1,1,1) sits inside the triangle formed by the three inputs that should fire, so
no plane separates them. Written as a disjunction of its three true rows,
though, it is three ANDs and an OR, which is one hidden layer and one output
unit. Every Boolean function has such a form, which is why every Boolean
function has a two-layer network.
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
GOOD = "#2ecc71"

# Each hidden unit detects one true row. +1 on a plain letter, -1 on a barred
# one, threshold = however many plain letters the row has.
MINTERMS = [
    ((+1, +1, -1), 2, r"X \wedge Y \wedge \overline{Z}"),
    ((+1, -1, +1), 2, r"X \wedge \overline{Y} \wedge Z"),
    ((-1, +1, +1), 2, r"\overline{X} \wedge Y \wedge Z"),
]


class DnfTwoOfThree(Scene):
    """Three AND units feeding one OR unit: the disjunctive normal form, wired."""

    def construct(self):
        self.camera.background_color = "#222222"

        title = MathTex(r"f(X,Y,Z) \;=\; (X \wedge Y \wedge \overline{Z}) "
                        r"\;\vee\; (X \wedge \overline{Y} \wedge Z) "
                        r"\;\vee\; (\overline{X} \wedge Y \wedge Z)",
                        font_size=38, color=LIGHT).to_edge(UP, buff=0.26)
        subtitle = Tex(r"``exactly two of the three sensors fired''",
                       font_size=28, color=MUTED).next_to(title, DOWN, buff=0.16)

        r = 0.46
        in_x, hid_x, out_x = -5.35, -1.55, 2.35
        in_ys = [1.00, -0.45, -1.90]
        hid_ys = [1.55, -0.45, -2.45]

        in_pts = [np.array([in_x, y, 0.0]) for y in in_ys]
        hid_pts = [np.array([hid_x, y, 0.0]) for y in hid_ys]
        out_pt = np.array([out_x, -0.45, 0.0])

        in_labels = VGroup(*[
            MathTex(name, font_size=40, color=QUIET).move_to(p)
            for name, p in zip(["X", "Y", "Z"], in_pts)])

        hidden, hid_labels, edges = VGroup(), VGroup(), VGroup()
        for (signs, thr, formula), hp in zip(MINTERMS, hid_pts):
            hidden.add(self.unit(hp, r, str(thr)))
            hid_labels.add(MathTex(formula, font_size=27, color=ACCENT
                                   ).next_to(hp, UP, buff=0.62))
            for sign, ip in zip(signs, in_pts):
                edges.add(self.link(ip, hp, r,
                                    GOOD if sign > 0 else FIRE))

        out = self.unit(out_pt, r, "1", colour=GOOD)
        for hp in hid_pts:
            edges.add(self.link(hp, out_pt, r, GOOD, start_radius=r))

        out_arrow = Arrow(out_pt + np.array([r, 0, 0]),
                          out_pt + np.array([r + 1.25, 0, 0]), color=GOOD,
                          stroke_width=4, buff=0.04,
                          max_tip_length_to_length_ratio=0.14)
        out_lbl = MathTex(r"f(X,Y,Z)", font_size=34, color=GOOD
                          ).next_to(out_arrow, RIGHT, buff=0.18)

        hidden_box = DashedVMobject(
            SurroundingRectangle(VGroup(hidden, hid_labels), color=MUTED,
                                 buff=0.28, corner_radius=0.14), num_dashes=64)
        hidden_lbl = Tex(r"one AND unit per true row", font_size=26,
                         color=MUTED).next_to(hidden_box, DOWN, buff=0.12)
        or_lbl = Tex(r"one OR unit", font_size=26, color=GOOD
                     ).next_to(out, DOWN, buff=0.42)

        key = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.5, color=GOOD, stroke_width=4),
                   MathTex(r"\text{weight } +1", font_size=26, color=GOOD)
                   ).arrange(RIGHT, buff=0.18),
            VGroup(Line(ORIGIN, RIGHT * 0.5, color=FIRE, stroke_width=4),
                   MathTex(r"\text{weight } -1", font_size=26, color=FIRE)
                   ).arrange(RIGHT, buff=0.18),
            Tex(r"numbers inside the circles are thresholds", font_size=26,
                color=MUTED),
        ).arrange(RIGHT, buff=0.80).to_edge(DOWN, buff=0.20)

        self.add(title, subtitle, hidden_box, hidden_lbl, edges, in_labels,
                 hidden, hid_labels, out, or_lbl, out_arrow, out_lbl, key)

    def unit(self, centre, radius, threshold, colour=LIGHT):
        body = Circle(radius=radius, color=colour, stroke_width=4,
                      fill_color="#333333", fill_opacity=1.0).move_to(centre)
        thr = MathTex(threshold, font_size=30, color=ACCENT).move_to(centre)
        return VGroup(body, thr)

    def link(self, start, end, radius, colour, start_radius=0.34):
        direction = normalize(end - start)
        return Arrow(start + start_radius * direction,
                     end - radius * direction, color=colour, stroke_width=3.4,
                     buff=0.02, max_tip_length_to_length_ratio=0.06)


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/neural-networks-history-of-an-idea/_src/dnf_two_of_three.py DnfTwoOfThree
# Manim writes the PNG to media\images\dnf_two_of_three\DnfTwoOfThree_ManimCE_v<version>.png
# Move it to blog/published/neural-networks-history-of-an-idea/images/dnf_two_of_three.png
# Embed with:
#   ![...](images/dnf_two_of_three.png){#fig-dnf_two_of_three fig-align="center" width=100%}

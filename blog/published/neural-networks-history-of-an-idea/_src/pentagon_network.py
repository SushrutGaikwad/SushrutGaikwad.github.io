# Runs via the root uv environment with uv run (see render command at the bottom).
"""The network behind the pentagon.

Five units, each holding one side of the polygon, and one more that fires only
when all five of them do. The output unit's threshold is doing the counting: set
it to 5 and you have an AND over five half-planes, which is the pentagon.
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


class PentagonNetwork(Scene):
    """Two real inputs, five half-plane units, one counting unit."""

    def construct(self):
        self.camera.background_color = "#222222"

        title = Tex(r"Five half-planes, and a unit that counts them",
                    font_size=36, color=LIGHT).to_edge(UP, buff=0.28)

        r = 0.42
        in_pts = [np.array([-5.15, 0.75, 0.0]), np.array([-5.15, -1.05, 0.0])]
        hid_ys = [2.05, 1.05, 0.05, -0.95, -1.95]
        hid_pts = [np.array([-1.85, y, 0.0]) for y in hid_ys]
        out_pt = np.array([1.55, 0.05, 0.0])

        in_labels = VGroup(
            MathTex("x_1", font_size=40, color=QUIET).move_to(in_pts[0]),
            MathTex("x_2", font_size=40, color=QUIET).move_to(in_pts[1]),
        )

        hidden, hid_labels, edges = VGroup(), VGroup(), VGroup()
        for hp in hid_pts:
            hidden.add(Circle(radius=r, color=LIGHT, stroke_width=4,
                              fill_color="#333333", fill_opacity=1.0
                              ).move_to(hp))
            for ip in in_pts:
                edges.add(self.link(ip, hp, r, QUIET))

        out_body = Circle(radius=0.60, color=GOOD, stroke_width=4.5,
                          fill_color="#333333", fill_opacity=1.0).move_to(out_pt)
        out_thr = MathTex("5", font_size=38, color=ACCENT).move_to(out_pt)
        # Name each hidden output on its own outgoing edge; the edges fan in, so
        # labels placed a third of the way along them stay separated.
        for k, hp in enumerate(hid_pts, start=1):
            edges.add(self.link(hp, out_pt, 0.60, GOOD, start_radius=r))
            direction = normalize(out_pt - hp)
            anchor = hp + r * direction + 0.30 * (out_pt - hp)
            hid_labels.add(MathTex(f"y_{k}", font_size=30, color=ACCENT
                                   ).move_to(anchor + np.array([0, 0.30, 0])))

        out_arrow = Arrow(out_pt + np.array([0.60, 0, 0]),
                          out_pt + np.array([1.85, 0, 0]), color=GOOD,
                          stroke_width=4.5, buff=0.04,
                          max_tip_length_to_length_ratio=0.14)

        # The shape the whole thing computes, drawn at the output.
        angles = np.pi / 2 + np.arange(5) * 2 * np.pi / 5
        pent_pts = [np.array([0.52 * np.cos(a), 0.52 * np.sin(a), 0.0])
                    for a in angles]
        pentagon = Polygon(*pent_pts, color=ACCENT, stroke_width=4,
                           fill_color=ACCENT, fill_opacity=0.35)
        pentagon.next_to(out_arrow, RIGHT, buff=0.28)

        rule = MathTex(r"\text{output} \;=\; 1 \iff \sum_{i=1}^{5} y_i \geq 5",
                       font_size=38, color=GOOD)
        rule.move_to([1.90, -2.35, 0])

        note = Tex(r"each $y_i$ fires on one side of one line; the output "
                   r"fires only where all five do at once",
                   font_size=28, color=MUTED).to_edge(DOWN, buff=0.26)

        self.add(title, edges, in_labels, hidden, hid_labels, out_body,
                 out_thr, out_arrow, pentagon, rule, note)

    def link(self, start, end, radius, colour, start_radius=0.34):
        direction = normalize(end - start)
        return Arrow(start + start_radius * direction,
                     end - radius * direction, color=colour, stroke_width=3.2,
                     buff=0.02, max_tip_length_to_length_ratio=0.06)


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/neural-networks-history-of-an-idea/_src/pentagon_network.py PentagonNetwork
# Manim writes the PNG to media\images\pentagon_network\PentagonNetwork_ManimCE_v<version>.png
# Move it to blog/published/neural-networks-history-of-an-idea/images/pentagon_network.png
# Embed with:
#   ![...](images/pentagon_network.png){#fig-pentagon_network fig-align="center" width=100%}

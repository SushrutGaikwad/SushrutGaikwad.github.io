# Runs via the root uv environment with uv run (see render command at the bottom).
"""The McCulloch-Pitts unit, with both kinds of synapse.

Excitatory endings (filled dots, McCulloch and Pitts' own convention) add into
the total. An inhibitory ending (an open hoop) does not subtract: if it carries
any signal at all, the unit is silenced outright, whatever the excitation. That
absolute veto is the part of the 1943 model that later units dropped.
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


class McCullochPittsUnit(Scene):
    """One unit: three excitatory inputs, one inhibitory veto, one threshold."""

    def construct(self):
        self.camera.background_color = "#222222"

        body_c = np.array([0.15, 0.55, 0.0])
        body_r = 0.92
        body = Circle(radius=body_r, color=LIGHT, stroke_width=5,
                      fill_color="#333333", fill_opacity=1.0).move_to(body_c)
        theta = MathTex(r"\theta", font_size=52, color=ACCENT).move_to(body_c)

        # ---- Three excitatory inputs, each ending in a filled dot ----
        excit = VGroup()
        labels = VGroup()
        for k, (dy, name) in enumerate([(1.55, "x_1"), (0.55, "x_2"),
                                        (-0.45, "x_3")]):
            start = np.array([-4.6, body_c[1] + dy - 0.55, 0.0])
            end = body_c + body_r * normalize(
                np.array([-4.0, dy - 0.55, 0.0]))
            excit.add(Line(start, end, color=QUIET, stroke_width=5))
            excit.add(Dot(end, radius=0.11, color=QUIET))
            lab = MathTex(name, font_size=40, color=QUIET)
            lab.next_to(start, LEFT, buff=0.18)
            labels.add(lab)

        # ---- One inhibitory input, ending in an open hoop ----
        inh_start = np.array([-4.6, body_c[1] - 2.05, 0.0])
        inh_end = body_c + body_r * normalize(np.array([-1.5, -2.6, 0.0]))
        inhibitory = VGroup(
            Line(inh_start, inh_end, color=FIRE, stroke_width=5),
            Circle(radius=0.13, color=FIRE, stroke_width=4,
                   fill_opacity=0.0).move_to(inh_end),
        )
        inh_label = MathTex(r"x_{\text{inh}}", font_size=40, color=FIRE)
        inh_label.next_to(inh_start, LEFT, buff=0.18)

        # ---- Output ----
        out = Arrow(body_c + np.array([body_r, 0, 0]),
                    body_c + np.array([body_r + 1.5, 0, 0]),
                    color=LIGHT, stroke_width=5, buff=0.05,
                    max_tip_length_to_length_ratio=0.18)
        out_label = MathTex("y", font_size=44, color=LIGHT)
        out_label.next_to(out, RIGHT, buff=0.20)

        # ---- The key to the two endings ----
        key = VGroup(
            VGroup(Dot(ORIGIN, radius=0.10, color=QUIET),
                   Tex(r"excitatory synapse: adds to the total",
                       font_size=28, color=QUIET)).arrange(RIGHT, buff=0.28),
            VGroup(Circle(radius=0.12, color=FIRE, stroke_width=4,
                          fill_opacity=0.0),
                   Tex(r"inhibitory synapse: an absolute veto",
                       font_size=28, color=FIRE)).arrange(RIGHT, buff=0.24),
        ).arrange(DOWN, buff=0.26, aligned_edge=LEFT)
        key.move_to([3.85, 2.35, 0])

        # ---- The firing rule ----
        rule = MathTex(
            r"y \;=\; \begin{cases}"
            r"0 & \text{if } x_{\text{inh}} = 1 \quad (\text{whatever the rest do}),\\[4pt]"
            r"1 & \text{if } x_{\text{inh}} = 0 \ \text{ and } \ "
            r"\displaystyle\sum_{i} x_i \geq \theta,\\[4pt]"
            r"0 & \text{otherwise},"
            r"\end{cases}",
            font_size=38, color=LIGHT)
        rule.to_edge(DOWN, buff=0.45)

        self.add(excit, labels, inhibitory, inh_label, body, theta, out,
                 out_label, key, rule)


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/neural-networks-history-of-an-idea/_src/mcculloch_pitts_unit.py McCullochPittsUnit
# Manim writes the PNG to media\images\mcculloch_pitts_unit\McCullochPittsUnit_ManimCE_v<version>.png
# Move it to blog/published/neural-networks-history-of-an-idea/images/mcculloch_pitts_unit.png
# Embed with:
#   ![...](images/mcculloch_pitts_unit.png){#fig-mcculloch_pitts_unit fig-align="center" width=100%}

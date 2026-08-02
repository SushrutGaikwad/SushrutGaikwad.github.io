# Runs via the root uv environment with uv run (see render command at the bottom).
"""The perceptron, and the change of view that makes everything else possible.

Read left to right it is a weighted sum compared against a threshold. Move the
threshold across the inequality, call it a bias, and the same unit becomes an
affine function of the inputs followed by an activation. Nothing about the unit
changed, but the activation is now a separate, swappable component, which is
where sigmoids, softplus and ReLU come from.
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


class PerceptronUnit(Scene):
    """Inputs, weights, the affine value z, the activation f, the output y."""

    def construct(self):
        self.camera.background_color = "#222222"

        # ---- Inputs and their weights ----
        rows = [(1.85, "x_1", "w_1"), (0.85, "x_2", "w_2"),
                (-0.15, "x_3", "w_3"), (-1.75, "x_N", "w_N")]
        sum_c = np.array([-0.55, 0.30, 0.0])
        sum_node = Circle(radius=0.62, color=LIGHT, stroke_width=4.5,
                          fill_color="#333333", fill_opacity=1.0).move_to(sum_c)
        sum_sym = MathTex(r"\Sigma", font_size=44, color=LIGHT).move_to(sum_c)

        wires, in_labels, w_labels = VGroup(), VGroup(), VGroup()
        for y, xname, wname in rows:
            start = np.array([-5.35, y, 0.0])
            end = sum_c + 0.62 * normalize(np.array([-5.35, y, 0.0]) - sum_c)
            wires.add(Arrow(start, end, color=QUIET, stroke_width=4, buff=0.02,
                            max_tip_length_to_length_ratio=0.055))
            in_labels.add(MathTex(xname, font_size=38, color=QUIET
                                  ).next_to(start, LEFT, buff=0.18))
            mid = start + 0.42 * (end - start)
            w_labels.add(MathTex(wname, font_size=32, color=ACCENT
                                 ).move_to(mid + np.array([0, 0.34, 0])))
        dots = MathTex(r"\vdots", font_size=38, color=QUIET).move_to([-5.35, -0.95, 0])

        # ---- The bias, entering as one more weight on a constant 1 ----
        bias_start = np.array([-0.55, -2.65, 0.0])
        bias_wire = Arrow(bias_start, sum_c + 0.62 * DOWN, color=GOOD,
                          stroke_width=4, buff=0.02,
                          max_tip_length_to_length_ratio=0.10)
        bias_node = MathTex("1", font_size=36, color=GOOD
                            ).next_to(bias_start, DOWN, buff=0.14)
        bias_lbl = MathTex("b", font_size=32, color=GOOD
                           ).next_to(bias_wire, RIGHT, buff=0.16)

        # ---- z, the activation, and the output ----
        z_arrow = Arrow(sum_c + np.array([0.62, 0, 0]), [2.05, 0.30, 0],
                        color=LIGHT, stroke_width=4.5, buff=0.04,
                        max_tip_length_to_length_ratio=0.12)
        z_lbl = MathTex("z", font_size=38, color=ACCENT).move_to([1.35, 0.78, 0])

        act_box = RoundedRectangle(width=1.55, height=1.35, corner_radius=0.14,
                                   color=LIGHT, stroke_width=4.5,
                                   fill_color="#333333", fill_opacity=1.0)
        act_box.move_to([2.85, 0.30, 0])
        step = VGroup(
            Line([-0.42, -0.28, 0], [0.0, -0.28, 0], color=ACCENT, stroke_width=4),
            Line([0.0, -0.28, 0], [0.0, 0.30, 0], color=ACCENT, stroke_width=4),
            Line([0.0, 0.30, 0], [0.42, 0.30, 0], color=ACCENT, stroke_width=4),
        ).move_to(act_box.get_center() + np.array([0, 0.04, 0]))
        act_lbl = MathTex("f", font_size=32, color=LIGHT
                          ).next_to(act_box, UP, buff=0.14)

        out_arrow = Arrow(act_box.get_right(), [5.05, 0.30, 0], color=LIGHT,
                          stroke_width=4.5, buff=0.04,
                          max_tip_length_to_length_ratio=0.14)
        out_lbl = MathTex("y", font_size=40, color=LIGHT
                          ).next_to(out_arrow, RIGHT, buff=0.18)

        diagram = VGroup(wires, in_labels, w_labels, dots, bias_wire, bias_node,
                         bias_lbl, sum_node, sum_sym, z_arrow, z_lbl, act_box,
                         step, act_lbl, out_arrow, out_lbl)

        # ---- The same unit, said two ways ----
        left = VGroup(
            Tex(r"\textbf{threshold form}", font_size=30, color=MUTED),
            MathTex(r"y = \begin{cases} 1 & \text{if } "
                    r"\sum_i w_i x_i \geq T,\\ 0 & \text{otherwise}\end{cases}",
                    font_size=34, color=LIGHT),
        ).arrange(DOWN, buff=0.22)

        right = VGroup(
            Tex(r"\textbf{affine form}", font_size=30, color=MUTED),
            MathTex(r"z = \mathbf{w}^{\intercal}\mathbf{x} + b, \qquad y = f(z)",
                    font_size=34, color=LIGHT),
        ).arrange(DOWN, buff=0.22)

        equiv = MathTex(r"\Longleftrightarrow", font_size=44, color=GOOD)
        pair = VGroup(left, equiv, right).arrange(RIGHT, buff=0.75)
        note = MathTex(r"b = -T", font_size=32, color=GOOD)
        note.next_to(equiv, DOWN, buff=0.30)

        bottom = VGroup(pair, note)
        frame = SurroundingRectangle(bottom, color="#555555", buff=0.30,
                                     corner_radius=0.12)
        boxed = VGroup(frame, bottom).to_edge(DOWN, buff=0.25)

        # Stack the diagram on top of the box rather than guessing coordinates,
        # so the bias arrow can never collide with the frame.
        diagram.next_to(boxed, UP, buff=0.30)
        if diagram.get_top()[1] > 3.85:
            diagram.shift(DOWN * (diagram.get_top()[1] - 3.85))

        self.add(diagram, boxed)


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/neural-networks-history-of-an-idea/_src/perceptron_unit.py PerceptronUnit
# Manim writes the PNG to media\images\perceptron_unit\PerceptronUnit_ManimCE_v<version>.png
# Move it to blog/published/neural-networks-history-of-an-idea/images/perceptron_unit.png
# Embed with:
#   ![...](images/perceptron_unit.png){#fig-perceptron_unit fig-align="center" width=100%}

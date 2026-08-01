# Runs via the root uv environment with uv run (see render command at the bottom).
from manim import *

# Load upgreek so upright-bold Greek (\boldsymbol{\upphi}) matches the blog's notation.
_TEMPLATE = TexTemplate()
_TEMPLATE.add_to_preamble(r"\usepackage{upgreek}")
config.tex_template = _TEMPLATE


class KernelTrickSchematic(Scene):
    """Two routes from the inputs x, z to the SAME scalar K(x, z):
      TOP    the explicit route: build the huge feature vectors phi(x), phi(z),
             then take their inner product  ->  O(d^3) for the degree-3 map,
      BOTTOM the kernel route: evaluate one cheap closed form directly  ->  O(d).
    The strip along the bottom runs both routes on actual numbers with d = 3, so the
    claim "same answer" is something the reader can check rather than take on trust:
    the 40-entry feature vectors dot to 85, and 1 + 4 + 4^2 + 4^3 is also 85."""

    def construct(self):
        self.camera.background_color = "#222222"

        light = "#e6e6e6"
        muted = "#9a9a9a"
        red = "#e74c3c"
        green = "#2ecc71"

        title = Tex(r"Same number, two costs", font_size=42,
                    color=light).to_edge(UP, buff=0.32)

        # ---- Inputs (left) ----
        inputs = self.boxed(r"\mathbf{x},\ \mathbf{z} \in \mathbb{R}^{d}",
                            light, "#888888", fs=34)
        inputs.move_to([-5.35, 0.75, 0])

        # ---- Explicit route (top) ----
        explicit = self.boxed(
            r"\boldsymbol{\upphi}(\mathbf{x})^{\!\intercal}\boldsymbol{\upphi}(\mathbf{z})",
            red, red, fs=38)
        explicit.move_to([0.25, 2.45, 0])
        explicit_note = Tex(
            r"first build $\boldsymbol{\upphi}(\mathbf{x}),\ \boldsymbol{\upphi}(\mathbf{z})"
            r" \in \mathbb{R}^{p}$ with $p \approx d^{3}$",
            font_size=28, color=red)
        explicit_note.next_to(explicit, DOWN, buff=0.18)

        # ---- Kernel route (bottom) ----
        kernel = self.boxed(
            r"K(\mathbf{x},\mathbf{z}) = 1 + \langle \mathbf{x},\mathbf{z}\rangle "
            r"+ \langle \mathbf{x},\mathbf{z}\rangle^{2} + \langle \mathbf{x},\mathbf{z}\rangle^{3}",
            green, green, fs=34)
        kernel.move_to([0.25, -0.85, 0])

        # ---- Result (right) ----
        result = self.boxed(r"\text{the same}\\ \text{scalar}", light, "#888888",
                            fs=32, tex=True)
        result.move_to([5.5, 0.75, 0])

        # ---- Arrows with cost tags ----
        a1 = Arrow(inputs.get_top(), explicit.get_left(), color=red,
                   stroke_width=5, buff=0.15, max_tip_length_to_length_ratio=0.07)
        a2 = Arrow(inputs.get_bottom(), kernel.get_left(), color=green,
                   stroke_width=5, buff=0.15, max_tip_length_to_length_ratio=0.07)
        a3 = Arrow(explicit.get_right(), result.get_top(), color=red,
                   stroke_width=5, buff=0.15, max_tip_length_to_length_ratio=0.07)
        a4 = Arrow(kernel.get_right(), result.get_bottom(), color=green,
                   stroke_width=5, buff=0.15, max_tip_length_to_length_ratio=0.07)

        cost_top = MathTex(r"O(d^{3})", font_size=36, color=red)
        cost_top.next_to(a1.get_center(), UP, buff=0.14).shift(LEFT * 0.28)
        cost_bot = MathTex(r"O(d)", font_size=36, color=green)
        cost_bot.next_to(a2.get_center(), DOWN, buff=0.14).shift(LEFT * 0.28)

        # ---- The same two routes, run on actual numbers ----
        example = self.worked_example(light, muted, red, green)
        example.move_to([0.0, -2.72, 0])

        self.add(title, inputs, explicit, explicit_note, kernel, result,
                 a1, a2, a3, a4, cost_top, cost_bot, example)

    def worked_example(self, light, muted, red, green):
        header = Tex(r"Check it, with $d = 3$, \ "
                     r"$\mathbf{x} = \begin{bmatrix} 1 & 2 & 1 \end{bmatrix}^{\intercal}$, \ "
                     r"$\mathbf{z} = \begin{bmatrix} 0 & 1 & 2 \end{bmatrix}^{\intercal}$",
                     font_size=30, color=light)

        left = Tex(r"dot two $40$-entry vectors: \ $\mathbf{85}$",
                   font_size=29, color=red)
        right = Tex(r"$\langle \mathbf{x},\mathbf{z}\rangle = 4$, then "
                    r"$1 + 4 + 16 + 64 = \mathbf{85}$",
                    font_size=29, color=green)

        row = VGroup(left, right).arrange(RIGHT, buff=1.15)
        body = VGroup(header, row).arrange(DOWN, buff=0.26)
        frame = SurroundingRectangle(body, color="#666666", buff=0.28,
                                     corner_radius=0.12)
        return VGroup(frame, body)

    def boxed(self, s, text_color, border_color, fs=34, tex=False):
        body = Tex(s, font_size=fs, color=text_color) if tex \
            else MathTex(s, font_size=fs, color=text_color)
        rect = SurroundingRectangle(body, color=border_color, buff=0.25,
                                    corner_radius=0.1)
        return VGroup(rect, body)


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/kernel-methods/_src/kernel_trick_schematic.py KernelTrickSchematic
# Manim writes the PNG to media\images\kernel_trick_schematic\KernelTrickSchematic_ManimCE_v0.20.1.png.
# Move it to blog/published/kernel-methods/images/kernel_trick_schematic.png
# Embed in the .qmd with:
#   ![...](images/kernel_trick_schematic.png){#fig-kernel_trick_schematic fig-align="center" width=100%}

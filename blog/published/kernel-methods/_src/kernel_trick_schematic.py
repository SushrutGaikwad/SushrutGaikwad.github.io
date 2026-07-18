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
    The point of the picture: identical output, wildly different cost."""

    def construct(self):
        self.camera.background_color = "#222222"

        light = "#e6e6e6"
        red = "#e74c3c"
        green = "#2ecc71"

        title = Tex(r"Same number, two costs", font_size=42,
                    color=light).to_edge(UP, buff=0.5)

        # ---- Inputs (left) ----
        inputs = self.boxed(
            r"\mathbf{x},\ \mathbf{z} \in \mathbb{R}^{d}",
            light, "#888888", fs=36)
        inputs.move_to(LEFT * 5.2)

        # ---- Explicit route (top) ----
        explicit = self.boxed(
            r"\underbrace{\boldsymbol{\upphi}(\mathbf{x})^{\!\intercal}\boldsymbol{\upphi}(\mathbf{z})}"
            r"_{\text{build } \boldsymbol{\upphi}(\mathbf{x}),\,\boldsymbol{\upphi}(\mathbf{z})\ \in\ \mathbb{R}^{p},\ "
            r"p\,\approx\, d^{3}}",
            red, red, fs=34)
        explicit.move_to(UP * 1.9 + RIGHT * 0.2)

        # ---- Kernel route (bottom) ----
        kernel = self.boxed(
            r"K(\mathbf{x},\mathbf{z}) = 1 + \langle \mathbf{x},\mathbf{z}\rangle "
            r"+ \langle \mathbf{x},\mathbf{z}\rangle^{2} + \langle \mathbf{x},\mathbf{z}\rangle^{3}",
            green, green, fs=34)
        kernel.move_to(DOWN * 1.9 + RIGHT * 0.2)

        # ---- Result (right) ----
        result = self.boxed(
            r"\text{the same}\\ \text{scalar}",
            light, "#888888", fs=34, tex=True)
        result.move_to(RIGHT * 5.4)

        # ---- Arrows with cost tags ----
        a1 = Arrow(inputs.get_top() + UP * 0.02, explicit.get_left(),
                   color=red, stroke_width=5, buff=0.15,
                   max_tip_length_to_length_ratio=0.06)
        a2 = Arrow(inputs.get_bottom() + DOWN * 0.02, kernel.get_left(),
                   color=green, stroke_width=5, buff=0.15,
                   max_tip_length_to_length_ratio=0.06)
        a3 = Arrow(explicit.get_right(), result.get_top() + UP * 0.02,
                   color=red, stroke_width=5, buff=0.15,
                   max_tip_length_to_length_ratio=0.06)
        a4 = Arrow(kernel.get_right(), result.get_bottom() + DOWN * 0.02,
                   color=green, stroke_width=5, buff=0.15,
                   max_tip_length_to_length_ratio=0.06)

        cost_top = Tex(r"$O(d^{3})$", font_size=34, color=red)
        cost_top.next_to(a1.get_center(), UP, buff=0.12).shift(LEFT * 0.2)
        cost_bot = Tex(r"$O(d)$", font_size=34, color=green)
        cost_bot.next_to(a2.get_center(), DOWN, buff=0.12).shift(LEFT * 0.2)

        self.add(title, inputs, explicit, kernel, result,
                 a1, a2, a3, a4, cost_top, cost_bot)

    def boxed(self, s, text_color, border_color, fs=34, tex=False):
        body = Tex(s, font_size=fs, color=text_color) if tex \
            else MathTex(s, font_size=fs, color=text_color)
        rect = SurroundingRectangle(body, color=border_color, buff=0.25,
                                    corner_radius=0.1)
        return VGroup(rect, body)


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/kernel-methods/_src/kernel_trick_schematic.py KernelTrickSchematic
# Manim writes the PNG to media\images\kernel_trick_schematic\KernelTrickSchematic.png.
# Move it to blog/published/kernel-methods/images/kernel_trick_schematic.png
# Embed in the .qmd with:
#   ![...](images/kernel_trick_schematic.png){#fig-kernel_trick_schematic fig-align="center" width=100%}

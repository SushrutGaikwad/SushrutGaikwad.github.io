# Runs via the root uv environment with uv run (see render command at the bottom).
"""Turing's unorganised machines, 1948.

An A-type machine is a scramble of two-input NAND units wired at random, and it
cannot be taught anything. A B-type is the same scramble with a modifier spliced
into every connection: a tiny switch with two settings, one that lets the signal
through and one that pins the line high. Training does not touch the units at
all, only the switches, which is a connectionist learning rule a decade before
Rosenblatt's.
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


class TuringBType(Scene):
    """Left: a random A-type net. Right: the modifier that makes it learnable."""

    def construct(self):
        self.camera.background_color = "#222222"

        title = Tex(r"Turing's \emph{unorganised machines} (1948)",
                    font_size=40, color=LIGHT).to_edge(UP, buff=0.28)

        left = self.a_type_panel()
        right = self.b_type_panel()
        panels = VGroup(left, right).arrange(RIGHT, buff=0.75)
        panels.scale_to_fit_width(13.2)
        panels.next_to(title, DOWN, buff=0.40)

        closing = Tex(r"Learning never rewires the units. It only decides which "
                      r"way each modifier is set.",
                      font_size=30, color=LIGHT).to_edge(DOWN, buff=0.28)

        self.add(title, panels, closing)

    # ---- Left panel: the random NAND scramble ------------------------------
    def a_type_panel(self):
        rng = np.random.default_rng(5)
        n = 11
        pts = []
        while len(pts) < n:
            candidate = np.array([rng.uniform(-2.2, 2.2),
                                  rng.uniform(-1.5, 1.5), 0.0])
            if all(np.linalg.norm(candidate - p) > 0.85 for p in pts):
                pts.append(candidate)

        nodes = VGroup(*[Dot(p, radius=0.15, color=QUIET) for p in pts])
        wires = VGroup()
        for i, p in enumerate(pts):
            # Each unit takes two inputs, chosen at random, as Turing specified.
            for j in rng.choice([k for k in range(n) if k != i], size=2,
                                replace=False):
                wires.add(Line(pts[j], p, color=MUTED, stroke_width=2.0))

        drawing = VGroup(wires, nodes)
        caption = VGroup(
            Tex(r"\textbf{A-type}", font_size=32, color=QUIET),
            Tex(r"two-input NAND units, wired at random,\\ "
                r"with no way to learn anything",
                font_size=26, color=MUTED),
        ).arrange(DOWN, buff=0.16)
        body = VGroup(drawing, caption).arrange(DOWN, buff=0.36)
        frame = SurroundingRectangle(body, color="#555555", buff=0.32,
                                     corner_radius=0.14)
        return VGroup(frame, body)

    # ---- Right panel: one connection, with its modifier in both states -----
    def b_type_panel(self):
        pass_through = self.modifier_row(
            r"green wire on", GOOD,
            r"the signal sails through unchanged", passes=True)
        pinned = self.modifier_row(
            r"red wire on", FIRE,
            r"the line is held at 1, whatever arrives", passes=False)
        rows = VGroup(pass_through, pinned).arrange(DOWN, buff=0.52)

        caption = VGroup(
            Tex(r"\textbf{B-type}", font_size=32, color=ACCENT),
            Tex(r"every connection carries a two-state modifier",
                font_size=26, color=MUTED),
        ).arrange(DOWN, buff=0.16)
        body = VGroup(caption, rows).arrange(DOWN, buff=0.36)
        frame = SurroundingRectangle(body, color="#555555", buff=0.32,
                                     corner_radius=0.14)
        return VGroup(frame, body)

    def modifier_row(self, state, colour, effect, passes):
        a = Dot([-1.85, 0, 0], radius=0.16, color=QUIET)
        b = Dot([1.85, 0, 0], radius=0.16, color=QUIET)
        wire_in = Line([-1.85, 0, 0], [-0.32, 0, 0], color=QUIET,
                       stroke_width=3.5)
        wire_out = Arrow([0.32, 0, 0], [1.85, 0, 0], color=QUIET,
                         stroke_width=3.5, buff=0.10,
                         max_tip_length_to_length_ratio=0.14)
        box = Square(side_length=0.52, color=colour, stroke_width=4,
                     fill_color="#333333", fill_opacity=1.0)
        toggle = Line([0, -0.16, 0], [0, 0.16, 0], color=colour,
                      stroke_width=5)
        if not passes:
            toggle.rotate(PI / 4)
        pinned_lbl = MathTex("1" if not passes else "", font_size=26,
                             color=colour).next_to(box, UP, buff=0.10)

        drawing = VGroup(wire_in, wire_out, a, b, box, toggle, pinned_lbl)
        label = VGroup(
            Tex(state, font_size=26, color=colour),
            Tex(effect, font_size=24, color=MUTED),
        ).arrange(DOWN, buff=0.10)
        return VGroup(drawing, label).arrange(DOWN, buff=0.20)


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/neural-networks-history-of-an-idea/_src/turing_b_type.py TuringBType
# Manim writes the PNG to media\images\turing_b_type\TuringBType_ManimCE_v<version>.png
# Move it to blog/published/neural-networks-history-of-an-idea/images/turing_b_type.png
# Embed with:
#   ![...](images/turing_b_type.png){#fig-turing_b_type fig-align="center" width=100%}

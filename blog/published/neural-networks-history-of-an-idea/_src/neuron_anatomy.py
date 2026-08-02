# Runs via the root uv environment with uv run (see render command at the bottom).
"""The one piece of biology the rest of the post rests on.

Many inputs arrive through the dendrites, the soma accumulates them, and if the
accumulation crosses a threshold a pulse leaves along the axon. There is exactly
one axon, so the neuron has many inputs but a single output, which is precisely
the shape of every artificial unit that follows.
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
FAT = "#c39bd3"


class NeuronAnatomy(Scene):
    """A labelled schematic neuron: dendrites, soma, axon, myelin, terminals."""

    def construct(self):
        self.camera.background_color = "#222222"

        soma_centre = np.array([-3.15, 0.35, 0.0])
        soma_r = 0.78
        axon_y = soma_centre[1]
        axon_start = soma_centre + np.array([soma_r, 0.0, 0.0])
        axon_end = np.array([3.55, axon_y, 0.0])

        # ---- Dendrites: several forked branches feeding into the soma ----
        dendrites = VGroup()
        for angle_deg, length in [(150, 1.35), (172, 1.70), (196, 1.55),
                                  (124, 0.95), (218, 1.25)]:
            angle = angle_deg * DEGREES
            direction = np.array([np.cos(angle), np.sin(angle), 0.0])
            base = soma_centre + soma_r * direction
            tip = soma_centre + (soma_r + length) * direction
            trunk = Line(base, tip, color=QUIET, stroke_width=5)
            dendrites.add(trunk)
            # Two small forks at the far end, so it reads as a dendrite.
            for fork in (28 * DEGREES, -28 * DEGREES):
                rot = np.array([np.cos(angle + fork), np.sin(angle + fork), 0.0])
                dendrites.add(Line(tip, tip + 0.42 * rot, color=QUIET,
                                   stroke_width=3.5))

        # ---- Soma with its nucleus ----
        soma = Circle(radius=soma_r, color=LIGHT, stroke_width=5,
                      fill_color="#3a3a3a", fill_opacity=1.0)
        soma.move_to(soma_centre)
        nucleus = Circle(radius=0.24, color=MUTED, stroke_width=3,
                         fill_color=MUTED, fill_opacity=0.85)
        nucleus.move_to(soma_centre)

        # ---- Axon: one line out, wrapped in myelin ----
        axon = Line(axon_start, axon_end, color=ACCENT, stroke_width=6)
        myelin = VGroup()
        for x in np.arange(-1.75, 3.1, 1.02):
            seg = RoundedRectangle(width=0.78, height=0.42, corner_radius=0.19,
                                   color=FAT, stroke_width=3,
                                   fill_color=FAT, fill_opacity=0.30)
            seg.move_to([x, axon_y, 0])
            myelin.add(seg)

        # ---- Axon terminals ----
        terminals = VGroup()
        for dy in (0.62, 0.0, -0.62):
            tip = axon_end + np.array([0.95, dy, 0.0])
            terminals.add(Line(axon_end, tip, color=ACCENT, stroke_width=4))
            terminals.add(Dot(tip, radius=0.13, color=ACCENT))

        # ---- The direction of travel along the axon ----
        flow_out = Arrow([0.55, axon_y - 0.85, 0], [1.85, axon_y - 0.85, 0],
                         color=ACCENT, stroke_width=4, buff=0,
                         max_tip_length_to_length_ratio=0.22)

        # ---- Labels ----
        lbl_dend = Tex(r"\textbf{Dendrites}\\ inputs from\\ many other neurons",
                       font_size=30, color=QUIET)
        lbl_dend.move_to([-5.35, 2.75, 0])

        lbl_soma = Tex(r"\textbf{Soma}\\ accumulates the\\ incoming signal",
                       font_size=30, color=LIGHT)
        lbl_soma.move_to([-3.15, -2.15, 0])

        lbl_axon = Tex(r"\textbf{Axon}\\ exactly one per neuron",
                       font_size=30, color=ACCENT)
        lbl_axon.move_to([0.95, 2.30, 0])

        lbl_myelin = Tex(r"\textbf{Myelin sheath}\\ fatty insulation\\ from glial cells",
                         font_size=28, color=FAT)
        lbl_myelin.move_to([3.05, -2.20, 0])

        lbl_term = Tex(r"\textbf{Terminals}\\ onto the next\\ neuron's dendrites",
                       font_size=28, color=ACCENT)
        lbl_term.move_to([5.75, 1.95, 0])

        arrows = VGroup(
            Arrow(lbl_dend.get_bottom(), [-5.15, 0.95, 0], color=QUIET,
                  stroke_width=3, buff=0.12,
                  max_tip_length_to_length_ratio=0.14),
            Arrow(lbl_soma.get_top(), soma_centre + np.array([0, -soma_r, 0]),
                  color=LIGHT, stroke_width=3, buff=0.12,
                  max_tip_length_to_length_ratio=0.14),
            Arrow(lbl_axon.get_bottom(), [0.95, axon_y + 0.30, 0], color=ACCENT,
                  stroke_width=3, buff=0.12,
                  max_tip_length_to_length_ratio=0.14),
            Arrow(lbl_myelin.get_top(), [2.30, axon_y - 0.28, 0], color=FAT,
                  stroke_width=3, buff=0.12,
                  max_tip_length_to_length_ratio=0.14),
            Arrow(lbl_term.get_bottom(), [4.55, axon_y + 0.68, 0], color=ACCENT,
                  stroke_width=3, buff=0.12,
                  max_tip_length_to_length_ratio=0.14),
        )

        rule = Tex(r"Many inputs in, one output out: the neuron fires only if "
                   r"the accumulated input crosses a threshold.",
                   font_size=32, color=LIGHT)
        rule.to_edge(DOWN, buff=0.30)

        self.add(dendrites, myelin, axon, terminals, soma, nucleus,
                 flow_out, lbl_dend, lbl_soma, lbl_axon, lbl_myelin,
                 lbl_term, arrows, rule)


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/neural-networks-history-of-an-idea/_src/neuron_anatomy.py NeuronAnatomy
# Manim writes the PNG to media\images\neuron_anatomy\NeuronAnatomy_ManimCE_v<version>.png
# Move it to blog/published/neural-networks-history-of-an-idea/images/neuron_anatomy.png
# Embed with:
#   ![...](images/neuron_anatomy.png){#fig-neuron_anatomy fig-align="center" width=100%}

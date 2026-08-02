# Runs via the root uv environment with uv run (see render command at the bottom).
"""Figure 1 of McCulloch and Pitts (1943), redrawn.

In their convention every neuron needs exactly two excitatory inputs and no
inhibitory input in order to fire, so a gate is built by choosing how many
endings each source lands on the target. Two endings from one source is enough
on its own (OR); one ending each from two sources needs both (AND); a hoop
vetoes (AND-NOT). Every unit also costs one tick of delay, which is why the
formulas relate the output at t to the inputs at t-1.
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


class McCullochPittsGates(Scene):
    """Four circuits: delay, OR, AND, and AND-NOT."""

    def construct(self):
        self.camera.background_color = "#222222"

        title = Tex(r"A unit fires exactly when it gets \textbf{two} "
                    r"excitatory endings and \textbf{no} inhibitory one",
                    font_size=30, color=LIGHT).to_edge(UP, buff=0.22)

        key = VGroup(
            VGroup(Dot(ORIGIN, radius=0.09, color=QUIET),
                   Tex(r"excitatory ending", font_size=24, color=QUIET)
                   ).arrange(RIGHT, buff=0.22),
            VGroup(Circle(radius=0.10, color=FIRE, stroke_width=3.5,
                          fill_opacity=0.0),
                   Tex(r"inhibitory ending", font_size=24, color=FIRE)
                   ).arrange(RIGHT, buff=0.20),
        ).arrange(RIGHT, buff=1.1)
        key.to_edge(DOWN, buff=0.16)

        panels = VGroup(
            self.panel_delay(),
            self.panel_or(),
            self.panel_and(),
            self.panel_and_not(),
        ).arrange_in_grid(rows=2, cols=2, buff=(0.55, 0.50))

        # Fit the grid into whatever vertical room the title and key leave.
        top = title.get_bottom()[1] - 0.22
        bottom = key.get_top()[1] + 0.22
        panels.scale_to_fit_width(13.4)
        if panels.height > top - bottom:
            panels.scale_to_fit_height(top - bottom)
        panels.move_to([0.0, (top + bottom) / 2, 0.0])

        self.add(title, panels, key)

    # ---- Building blocks ---------------------------------------------------
    def cell(self, label, colour=LIGHT):
        """A neuron, drawn as a right-pointing triangle with its number inside."""
        tri = Triangle(color=colour, stroke_width=4,
                       fill_color="#333333", fill_opacity=1.0)
        tri.rotate(-PI / 2).scale(0.46)
        num = MathTex(label, font_size=30, color=colour).move_to(
            tri.get_center() + LEFT * 0.10)
        return VGroup(tri, num)

    def ending(self, point, inhibitory=False):
        if inhibitory:
            return Circle(radius=0.10, color=FIRE, stroke_width=3.5,
                          fill_opacity=0.0).move_to(point)
        return Dot(point, radius=0.09, color=QUIET)

    def wire(self, start, end, inhibitory=False):
        colour = FIRE if inhibitory else QUIET
        return VGroup(Line(start, end, color=colour, stroke_width=3.5),
                      self.ending(end, inhibitory))

    def framed(self, drawing, formula, caption):
        text = VGroup(MathTex(formula, font_size=32, color=ACCENT),
                      Tex(caption, font_size=26, color=MUTED)
                      ).arrange(DOWN, buff=0.16)
        body = VGroup(drawing, text).arrange(DOWN, buff=0.30)
        frame = SurroundingRectangle(body, color="#555555", buff=0.28,
                                     corner_radius=0.12)
        return VGroup(frame, body)

    # ---- The four circuits -------------------------------------------------
    def panel_delay(self):
        n1 = self.cell("1").move_to([-1.15, 0, 0])
        n2 = self.cell("2").move_to([1.15, 0, 0])
        # Two endings from the same source: one source alone is enough.
        wires = VGroup(
            self.wire(n1.get_right(), n2.get_left() + np.array([0, 0.13, 0])),
            self.wire(n1.get_right(), n2.get_left() + np.array([0, -0.13, 0])),
        )
        out = Line(n2.get_right(), n2.get_right() + RIGHT * 0.55,
                   color=LIGHT, stroke_width=3.5)
        return self.framed(VGroup(n1, wires, n2, out),
                           r"N_2(t) \iff N_1(t-1)",
                           r"a delay: unit 2 simply repeats unit 1, one tick later")

    def panel_or(self):
        n1 = self.cell("1").move_to([-1.15, 0.62, 0])
        n2 = self.cell("2").move_to([-1.15, -0.62, 0])
        n3 = self.cell("3").move_to([1.30, 0, 0])
        wires = VGroup(
            self.wire(n1.get_right(), n3.get_left() + np.array([0, 0.20, 0])),
            self.wire(n1.get_right(), n3.get_left() + np.array([0.05, 0.06, 0])),
            self.wire(n2.get_right(), n3.get_left() + np.array([0.05, -0.06, 0])),
            self.wire(n2.get_right(), n3.get_left() + np.array([0, -0.20, 0])),
        )
        out = Line(n3.get_right(), n3.get_right() + RIGHT * 0.55,
                   color=LIGHT, stroke_width=3.5)
        return self.framed(VGroup(n1, n2, wires, n3, out),
                           r"N_3(t) \iff N_1(t-1) \vee N_2(t-1)",
                           r"OR: two endings each, so either source alone fires it")

    def panel_and(self):
        n1 = self.cell("1").move_to([-1.15, 0.62, 0])
        n2 = self.cell("2").move_to([-1.15, -0.62, 0])
        n3 = self.cell("3").move_to([1.30, 0, 0])
        wires = VGroup(
            self.wire(n1.get_right(), n3.get_left() + np.array([0, 0.13, 0])),
            self.wire(n2.get_right(), n3.get_left() + np.array([0, -0.13, 0])),
        )
        out = Line(n3.get_right(), n3.get_right() + RIGHT * 0.55,
                   color=LIGHT, stroke_width=3.5)
        return self.framed(VGroup(n1, n2, wires, n3, out),
                           r"N_3(t) \iff N_1(t-1) \wedge N_2(t-1)",
                           r"AND: one ending each, so it takes both to reach two")

    def panel_and_not(self):
        n1 = self.cell("1").move_to([-1.15, 0.62, 0])
        n2 = self.cell("2", colour=FIRE).move_to([-1.15, -0.62, 0])
        n3 = self.cell("3").move_to([1.30, 0, 0])
        wires = VGroup(
            self.wire(n1.get_right(), n3.get_left() + np.array([0, 0.20, 0])),
            self.wire(n1.get_right(), n3.get_left() + np.array([0.05, 0.06, 0])),
            self.wire(n2.get_right(), n3.get_left() + np.array([0, -0.18, 0]),
                      inhibitory=True),
        )
        out = Line(n3.get_right(), n3.get_right() + RIGHT * 0.55,
                   color=LIGHT, stroke_width=3.5)
        return self.framed(VGroup(n1, n2, wires, n3, out),
                           r"N_3(t) \iff N_1(t-1) \wedge \neg N_2(t-1)",
                           r"AND-NOT: unit 2's hoop silences it outright")


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/neural-networks-history-of-an-idea/_src/mcculloch_pitts_gates.py McCullochPittsGates
# Manim writes the PNG to media\images\mcculloch_pitts_gates\McCullochPittsGates_ManimCE_v<version>.png
# Move it to blog/published/neural-networks-history-of-an-idea/images/mcculloch_pitts_gates.png
# Embed with:
#   ![...](images/mcculloch_pitts_gates.png){#fig-mcculloch_pitts_gates fig-align="center" width=100%}

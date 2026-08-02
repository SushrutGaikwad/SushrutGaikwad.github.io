# Runs via the root uv environment with uv run (see render command at the bottom).
"""Figure 2 of McCulloch and Pitts (1943): why a brief touch of cold feels hot.

Six units. The cold receptor feeds a delay unit; the delay unit and the cold
receptor together fire the cold sensation, so cold has to persist for two ticks
to register as cold. The delay unit also feeds a unit that the cold receptor
inhibits, so that path survives only when the cold has already stopped, and it
is that path which fires the heat sensation. Brief cold takes the second route
and feels hot. Sustained cold vetoes it and feels cold.

The firing schedule below is not hand-authored: it is simulated from the wiring,
so what the animation shows is what the circuit actually does.
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
IDLE_FILL = "#333333"

T_MAX = 5
KEYS = ["heat_rec", "cold_rec", "delay", "red", "heat_sens", "cold_sens"]


def simulate(cold, heat):
    """Run the net forward. Every unit's output at t depends only on t-1."""
    s = {k: [False] * (T_MAX + 1) for k in KEYS}
    for t in range(T_MAX + 1):
        s["heat_rec"][t] = heat[t]
        s["cold_rec"][t] = cold[t]
        if t == 0:
            continue
        s["delay"][t] = s["cold_rec"][t - 1]
        s["red"][t] = s["delay"][t - 1] and not s["cold_rec"][t - 1]
        s["cold_sens"][t] = s["cold_rec"][t - 1] and s["delay"][t - 1]
        s["heat_sens"][t] = s["heat_rec"][t - 1] or s["red"][t - 1]
    return s


BRIEF = simulate(cold=[False, True, False, False, False, False],
                 heat=[False] * (T_MAX + 1))
SUSTAINED = simulate(cold=[False, True, True, True, True, True],
                     heat=[False] * (T_MAX + 1))

BRIEF_CAPTIONS = [
    r"The finger touches something cold, for one instant only.",
    r"The delay unit picks it up. Cold sensation needs cold \emph{now} and "
    r"cold \emph{a moment ago}, and it only has one of those.",
    r"The cold is gone, so nothing vetoes the middle unit and it fires.",
    r"That unit fires the heat sensation. A brief touch of cold feels hot.",
    r"Nothing further arrives, and the net falls silent.",
]

SUSTAINED_CAPTIONS = [
    r"The same finger, but this time it stays on the cold surface.",
    r"The delay unit picks it up, exactly as before.",
    r"Now cold is present \emph{and} was present a moment ago: the cold "
    r"sensation fires, and the ongoing cold vetoes the middle unit.",
    r"With that unit silenced, the heat sensation never fires at all.",
    r"Sustained cold feels cold. The only difference was how long it lasted.",
]


class HeatIllusion(Scene):
    """The circuit, stepped through both scenarios one tick at a time."""

    def construct(self):
        self.camera.background_color = "#222222"

        self.positions = {
            "heat_rec": np.array([-5.0, 2.15, 0.0]),
            "cold_rec": np.array([-5.0, -2.15, 0.0]),
            "delay": np.array([-1.45, -1.05, 0.0]),
            "red": np.array([-1.45, 1.05, 0.0]),
            "heat_sens": np.array([2.60, 2.15, 0.0]),
            "cold_sens": np.array([2.60, -2.15, 0.0]),
        }
        self.radius = 0.52

        self.nodes = {k: self.make_node(k) for k in KEYS}
        self.build_labels()
        wires = self.build_wires()

        self.time_lbl = MathTex("t = 0", font_size=40, color=ACCENT)
        self.time_lbl.to_corner(UR, buff=0.45)

        self.headline = Tex("", font_size=34, color=LIGHT).to_edge(UP, buff=0.28)
        self.caption = Tex("", font_size=28, color=MUTED).to_edge(DOWN, buff=0.30)

        self.add(wires, *[n for n in self.nodes.values()], self.labels,
                 self.time_lbl, self.headline, self.caption)

        self.run_scenario("A brief touch of cold", BRIEF, BRIEF_CAPTIONS, GOOD)
        self.run_scenario("Cold that stays", SUSTAINED, SUSTAINED_CAPTIONS,
                          QUIET)

    # ---- Scene furniture ---------------------------------------------------
    def make_node(self, key):
        colour = ACCENT if key.endswith("_sens") else LIGHT
        body = Circle(radius=self.radius, color=colour, stroke_width=4.5,
                      fill_color=IDLE_FILL, fill_opacity=1.0)
        return body.move_to(self.positions[key])

    def build_labels(self):
        texts = {
            "heat_rec": r"heat\\ receptor",
            "cold_rec": r"cold\\ receptor",
            "delay": r"delay",
            "red": r"vetoable\\ unit",
            "heat_sens": r"\textbf{heat}\\ \textbf{sensation}",
            "cold_sens": r"\textbf{cold}\\ \textbf{sensation}",
        }
        placement = {
            "heat_rec": LEFT, "cold_rec": LEFT, "delay": DOWN,
            "heat_sens": RIGHT, "cold_sens": RIGHT,
        }
        self.labels = VGroup()
        for key, text in texts.items():
            colour = ACCENT if key.endswith("_sens") else MUTED
            lbl = Tex(text, font_size=26, color=colour)
            if key == "red":
                # The only label with nowhere clear to sit next to its node:
                # park it in the empty pocket between the two left-hand units.
                lbl.move_to([-3.15, 1.55, 0])
            else:
                lbl.next_to(self.nodes[key], placement[key], buff=0.24)
            self.labels.add(lbl)

    def build_wires(self):
        """The wiring of Figure 2, ending by ending."""
        return VGroup(
            self.connect("heat_rec", "heat_sens", endings=2),
            self.connect("cold_rec", "delay", endings=2),
            self.connect("delay", "red", endings=2),
            self.connect("cold_rec", "red", endings=1, inhibitory=True),
            self.connect("cold_rec", "cold_sens", endings=1),
            self.connect("delay", "cold_sens", endings=1),
            self.connect("red", "heat_sens", endings=2),
        )

    def connect(self, src, dst, endings=1, inhibitory=False):
        a, b = self.positions[src], self.positions[dst]
        direction = normalize(b - a)
        perp = np.array([-direction[1], direction[0], 0.0])
        colour = FIRE if inhibitory else QUIET

        group = VGroup()
        offsets = [0.13, -0.13] if endings == 2 else [0.0]
        for off in offsets:
            start = a + self.radius * direction + off * perp
            end = b - self.radius * direction + off * perp
            group.add(Line(start, end, color=colour, stroke_width=3.2))
            if inhibitory:
                group.add(Circle(radius=0.11, color=FIRE, stroke_width=3.5,
                                 fill_opacity=0.0).move_to(end))
            else:
                group.add(Dot(end, radius=0.085, color=colour))
        return group

    # ---- Playing a scenario ------------------------------------------------
    def run_scenario(self, headline, states, captions, colour):
        self.play(
            self.headline.animate.become(
                Tex(headline, font_size=36, color=colour
                    ).to_edge(UP, buff=0.28)),
            run_time=0.5)

        for t in range(1, T_MAX + 1):
            anims = [self.time_lbl.animate.become(
                MathTex(f"t = {t}", font_size=40, color=ACCENT
                        ).to_corner(UR, buff=0.45))]
            for key in KEYS:
                target = FIRE if states[key][t] else IDLE_FILL
                anims.append(self.nodes[key].animate.set_fill(target,
                                                              opacity=1.0))
            anims.append(self.caption.animate.become(
                Tex(captions[t - 1], font_size=28, color=MUTED
                    ).to_edge(DOWN, buff=0.30)))
            self.play(*anims, run_time=0.6)
            self.wait(2.1)          # long enough to read the caption

        # Reset before the next scenario so the two runs never bleed together.
        self.play(*[self.nodes[k].animate.set_fill(IDLE_FILL, opacity=1.0)
                    for k in KEYS], run_time=0.4)
        self.wait(0.5)


# Render (from the repo root):
#   uv run manim -qh blog/published/neural-networks-history-of-an-idea/_src/heat_illusion.py HeatIllusion
# Manim writes the video to media\videos\heat_illusion\1080p60\HeatIllusion.mp4
# Move it to blog/published/neural-networks-history-of-an-idea/images/heat_illusion.mp4
# Embed as a numbered figure with:
#   ::: {#fig-heat_illusion}
#   {{< video images/heat_illusion.mp4 >}}
#
#   Caption text.
#   :::

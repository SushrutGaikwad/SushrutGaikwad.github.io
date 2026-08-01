# Runs via the root uv environment with uv run (see render command at the bottom).
from manim import *

# Load upgreek so upright-bold Greek (\boldsymbol{\upphi}) matches the blog's notation.
_TEMPLATE = TexTemplate()
_TEMPLATE.add_to_preamble(r"\usepackage{upgreek}")
config.tex_template = _TEMPLATE


class ThetaLivesInSpan(Scene):
    """The geometric content of kernel-trick step 1.

    The ambient feature space R^p is drawn as a huge region; the span of the n
    feature vectors phi(x_1), ..., phi(x_n) is the small tilted slice inside it.
    Gradient descent starts at the origin and every step it ever takes stays inside
    that slice, because each step adds another multiple of some phi(x_i). So the
    parameter we are really searching for has n degrees of freedom, not p, no matter
    how astronomical (or infinite) p happens to be."""

    def construct(self):
        self.camera.background_color = "#222222"

        light = "#e6e6e6"
        muted = "#9a9a9a"
        blue = "#5dade2"
        yellow = "#f1c40f"
        green = "#2ecc71"

        title = Tex(r"Gradient descent never leaves the span of the data",
                    font_size=40, color=light).to_edge(UP, buff=0.35)

        # ---- The ambient feature space ----
        ambient = Ellipse(width=12.6, height=5.9, color="#777777",
                          stroke_width=2.5).move_to([-0.3, -0.3, 0])
        ambient.set_stroke(opacity=0.8)
        ambient_label = MathTex(r"\mathbb{R}^{p}", font_size=44, color=muted)
        ambient_label.move_to([-4.25, 1.30, 0])
        ambient_note = Tex(r"$p \approx d^{3}$, or $\infty$",
                           font_size=26, color=muted)
        ambient_note.next_to(ambient_label, DOWN, buff=0.14)

        # ---- The slice the parameter can actually reach ----
        C = np.array([-0.6, -0.6, 0.0])
        u = np.array([3.4, -0.7, 0.0])
        v = np.array([1.6, 1.5, 0.0])
        slice_plane = Polygon(C - u - v, C + u - v, C + u + v, C - u + v,
                              color=blue, stroke_width=3)
        slice_plane.set_fill(blue, opacity=0.16)

        slice_label = Tex(r"$\mathrm{span}\left\{ \boldsymbol{\upphi}(\mathbf{x}_1), "
                          r"\ldots, \boldsymbol{\upphi}(\mathbf{x}_n) \right\}$",
                          font_size=32, color=blue)
        slice_label.move_to([2.05, -1.75, 0])
        slice_dim = Tex(r"at most $n$ dimensions", font_size=26, color=blue)
        slice_dim.next_to(slice_label, DOWN, buff=0.16)

        # ---- Two of the n feature vectors, drawn inside the slice ----
        O = np.array([-3.6, -1.2, 0.0])
        phi1 = O + np.array([2.9, -0.55, 0.0])
        phi2 = O + np.array([1.45, 1.40, 0.0])

        arrow1 = Arrow(O, phi1, color=yellow, buff=0, stroke_width=4,
                       max_tip_length_to_length_ratio=0.09)
        arrow2 = Arrow(O, phi2, color=yellow, buff=0, stroke_width=4,
                       max_tip_length_to_length_ratio=0.12)
        lbl1 = MathTex(r"\boldsymbol{\upphi}(\mathbf{x}_1)", font_size=30, color=yellow)
        lbl1.next_to(phi1, DOWN, buff=0.14)
        lbl2 = MathTex(r"\boldsymbol{\upphi}(\mathbf{x}_2)", font_size=30, color=yellow)
        lbl2.next_to(phi2, UP, buff=0.14)

        # ---- The trajectory, every step of it inside the slice ----
        steps = [O,
                 O + np.array([0.95, -0.05, 0.0]),
                 O + np.array([1.95, 0.25, 0.0]),
                 O + np.array([2.75, 0.72, 0.0]),
                 O + np.array([3.35, 0.95, 0.0])]
        path = VGroup()
        for a, b in zip(steps[:-1], steps[1:]):
            path.add(Arrow(a, b, color=green, buff=0.06, stroke_width=5,
                           max_tip_length_to_length_ratio=0.22))
        dots = VGroup(*[Dot(s, radius=0.055, color=green) for s in steps])

        start_lbl = MathTex(r"\boldsymbol{\uptheta} = \mathbf{0}",
                            font_size=30, color=green).next_to(O, LEFT, buff=0.18)
        end_lbl = MathTex(r"\boldsymbol{\uptheta} = \sum_{i=1}^{n} \beta_i \,"
                          r"\boldsymbol{\upphi}(\mathbf{x}_i)",
                          font_size=32, color=green)
        end_lbl.next_to(steps[-1], UP, buff=0.22).shift(RIGHT * 0.55)

        # ---- The moral ----
        moral = Tex(r"Whatever $p$ is, the search has only $n$ knobs. "
                    r"So track $\boldsymbol{\upbeta} \in \mathbb{R}^{n}$, "
                    r"never $\boldsymbol{\uptheta} \in \mathbb{R}^{p}$.",
                    font_size=32, color=light)
        moral.to_edge(DOWN, buff=0.35)

        self.add(title, ambient, ambient_label, ambient_note,
                 slice_plane, slice_label, slice_dim,
                 arrow1, arrow2, lbl1, lbl2,
                 path, dots, start_lbl, end_lbl, moral)


# Render (from the repo root), saving only the final frame as a PNG:
#   uv run manim -qh -s blog/published/kernel-methods/_src/theta_lives_in_span.py ThetaLivesInSpan
# Manim writes the PNG to media\images\theta_lives_in_span\ThetaLivesInSpan.png.
# Move it to blog/published/kernel-methods/images/theta_lives_in_span.png
# Embed in the .qmd with:
#   ![...](images/theta_lives_in_span.png){#fig-theta_lives_in_span fig-align="center" width=100%}

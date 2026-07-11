# Runs via the root uv environment with `uv run`.
"""Subspace-closure contrast for Post 3.

Left: a line through the origin is closed under addition (a subspace).
Right: a line that misses the origin is not (the sum of two of its vectors
leaves the line, and it does not contain 0).
All text uses MathTex/Tex (LaTeX / Computer Modern); bold uses \\mathbf.
"""

import numpy as np
from manim import *

LIGHT = "#e6e6e6"
BLUE_A = "#5dade2"
ORANGE_A = "#f5b041"
RED_B = "#ec7063"
GRAY_L = "#9aa0a6"

DIR = np.array([0.9, 0.38, 0.0])  # common line direction (screen units)


class SubspaceClosure(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        left = self._panel(
            origin=LEFT * 3.7 + DOWN * 0.4,
            offset=np.array([0.0, 0.0, 0.0]),
            u_t=1.7,
            v_t=-0.9,
            title=r"Through the origin: a subspace",
            closed=True,
        )
        right = self._panel(
            origin=RIGHT * 3.7 + DOWN * 0.4,
            offset=np.array([0.0, 1.25, 0.0]),
            u_t=1.2,
            v_t=-0.6,
            title=r"Not through the origin: not a subspace",
            closed=False,
        )
        self.add(left, right)

    def _panel(self, origin, offset, u_t, v_t, title, closed):
        g = VGroup()

        p0 = origin + offset  # a point on the line
        line = Line(p0 - 2.1 * DIR, p0 + 2.6 * DIR, color=GRAY_L, stroke_width=3)
        g.add(line)

        # Origin marker.
        o_dot = Dot(origin, color=LIGHT, radius=0.05)
        o_lab = MathTex(r"\mathbf{0}", color=LIGHT).scale(0.7).next_to(o_dot, DOWN, buff=0.12)
        g.add(o_dot, o_lab)

        # Two vectors with tips on the line, drawn from the origin.
        u_tip = p0 + u_t * DIR
        v_tip = p0 + v_t * DIR
        u_vec = u_tip - origin
        v_vec = v_tip - origin

        arr_u = Arrow(origin, u_tip, buff=0, color=BLUE_A, stroke_width=5)
        arr_v = Arrow(origin, v_tip, buff=0, color=ORANGE_A, stroke_width=5)
        lab_u = MathTex(r"\mathbf{u}", color=BLUE_A).scale(0.75).next_to(arr_u.get_end(), UR, buff=0.08)
        lab_v = MathTex(r"\mathbf{v}", color=ORANGE_A).scale(0.75).next_to(arr_v.get_end(), DL, buff=0.08)
        g.add(arr_u, arr_v, lab_u, lab_v)

        # Tip-to-tail sum: from u's tip, add v.
        sum_pt = origin + u_vec + v_vec
        tail = DashedLine(u_tip, sum_pt, color=GRAY_L, stroke_width=2)
        sum_dot = Dot(sum_pt, color=RED_B, radius=0.07)
        lab_sum = MathTex(r"\mathbf{u}+\mathbf{v}", color=RED_B).scale(0.72)
        lab_sum.next_to(sum_dot, RIGHT if closed else UR, buff=0.12)
        g.add(tail, sum_dot, lab_sum)

        if not closed:
            note = Tex(r"sum leaves the line", color=RED_B).scale(0.55)
            note.next_to(sum_dot, DOWN, buff=0.15)
            g.add(note)

        ttl = Tex(title, color=LIGHT).scale(0.62)
        ttl.next_to(g, UP, buff=0.35)
        g.add(ttl)
        return g


# Render the last frame only as a PNG, from the repo root:
#   uv run manim -qh -s blog/published/vector-spaces-column-space-and-nullspace/_src/subspace_closure.py SubspaceClosure
# Manim saves it at media/images/subspace_closure/SubspaceClosure.png.
# Move it to blog/published/vector-spaces-column-space-and-nullspace/images/subspace_closure.png and embed with:
#   ![Caption.](images/subspace_closure.png){#fig-subspace_closure fig-align="center" width=95%}

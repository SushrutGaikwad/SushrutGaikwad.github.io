# Runs via the root uv environment with `uv run`.
"""Complete-solution figure for Post 4.

Shows the solution set of Ax = b as the nullspace line (through the origin)
shifted to pass through a particular solution xp. Drawn with a 1-D nullspace
so it fits on the page. All text uses MathTex (LaTeX / Computer Modern);
bold uses \\mathbf.
"""

import numpy as np
from manim import *

LIGHT = "#e6e6e6"
BLUE_A = "#5dade2"     # nullspace
ORANGE_A = "#f5b041"   # solution set
RED_B = "#ec7063"      # particular solution / points
GRAY_L = "#9aa0a6"


class CompleteSolution(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        O = LEFT * 2.6 + DOWN * 1.6
        d = np.array([1.0, 0.42, 0.0])  # common direction of both parallel lines

        # Nullspace: a line through the origin.
        null_line = Line(O - 3.0 * d, O + 4.2 * d, color=BLUE_A, stroke_width=4)
        self.add(null_line)
        null_lab = MathTex(r"N(\mathbf{A})", color=BLUE_A).scale(0.85)
        null_lab.next_to(O + 4.2 * d, RIGHT, buff=0.15)
        self.add(null_lab)

        # Origin.
        o_dot = Dot(O, color=LIGHT, radius=0.05)
        o_lab = MathTex(r"\mathbf{0}", color=LIGHT).scale(0.7).next_to(o_dot, DOWN, buff=0.12)
        self.add(o_dot, o_lab)

        # Particular solution xp (off the nullspace line).
        P = O + np.array([1.1, 2.7, 0.0])
        xp_arr = Arrow(O, P, buff=0, color=RED_B, stroke_width=5)
        xp_lab = MathTex(r"\mathbf{x}_p", color=RED_B).scale(0.85)
        xp_lab.next_to(xp_arr.get_center(), LEFT, buff=0.18)
        self.add(xp_arr, xp_lab)

        # Solution set: the parallel line through xp.
        sol_line = Line(P - 3.0 * d, P + 4.2 * d, color=ORANGE_A, stroke_width=4)
        self.add(sol_line)
        sol_lab = MathTex(r"\mathbf{x}_p + N(\mathbf{A})", color=ORANGE_A).scale(0.85)
        sol_lab.next_to(P + 4.2 * d, RIGHT, buff=0.15)
        self.add(sol_lab)

        # The particular solution as a point, plus another solution xp + xn.
        P_dot = Dot(P, color=RED_B, radius=0.07)
        Q = P + 2.0 * d
        Q_dot = Dot(Q, color=RED_B, radius=0.07)
        add_seg = DashedLine(P, Q, color=GRAY_L, stroke_width=2)
        Q_lab = MathTex(r"\mathbf{x}_p + \mathbf{x}_n", color=RED_B).scale(0.75)
        Q_lab.next_to(Q_dot, UP, buff=0.12)
        self.add(add_seg, P_dot, Q_dot, Q_lab)


# Render the last frame only as a PNG, from the repo root:
#   uv run manim -qh -s blog/published/rank-and-the-complete-solution/_src/complete_solution.py CompleteSolution
# Manim saves it at media/images/complete_solution/CompleteSolution.png.
# Move it to blog/published/rank-and-the-complete-solution/images/complete_solution.png and embed with:
#   ![Caption.](images/complete_solution.png){#fig-complete_solution fig-align="center" width=80%}

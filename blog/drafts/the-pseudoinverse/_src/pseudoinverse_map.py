# Runs via the root uv environment with `uv run`.
"""Pseudoinverse mapping figure for Post 18.

A is an invertible bijection between the row space and the column space; A^+
reverses it. Each map sends the null space on its own side to zero. All text
uses MathTex/Tex; bold uses \\mathbf.
"""

from manim import *

LIGHT = "#e6e6e6"
BLUE_A = "#5dade2"
ORANGE_A = "#f5b041"
GRAY_L = "#9aa0a6"


def labeled_box(title, color, width=3.4, height=1.4):
    box = RoundedRectangle(width=width, height=height, corner_radius=0.16,
                           stroke_color=color, fill_color=color, fill_opacity=0.12)
    t = Tex(title, color=LIGHT).scale(0.6).move_to(box.get_center())
    return VGroup(box, t)


class PseudoinverseMap(Scene):
    def construct(self):
        self.camera.background_color = "#222222"

        row = labeled_box(r"row space", ORANGE_A).move_to(LEFT * 3.7 + UP * 1.5)
        null = labeled_box(r"nullspace", GRAY_L).move_to(LEFT * 3.7 + DOWN * 1.6)
        col = labeled_box(r"column space", BLUE_A).move_to(RIGHT * 3.7 + UP * 1.5)
        lnull = labeled_box(r"left nullspace", GRAY_L).move_to(RIGHT * 3.7 + DOWN * 1.6)
        self.add(row, null, col, lnull)

        hin = MathTex(r"\mathbb{R}^n", color=LIGHT).scale(0.75).next_to(row, UP, buff=0.3)
        hout = MathTex(r"\mathbb{R}^m", color=LIGHT).scale(0.75).next_to(col, UP, buff=0.3)
        self.add(hin, hout)

        # A: row space -> column space (top curve).
        aA = CurvedArrow(row[0].get_right() + UP * 0.25, col[0].get_left() + UP * 0.25,
                         angle=-0.7, color=BLUE_A, stroke_width=4)
        aA_lab = MathTex(r"\mathbf{A}", color=BLUE_A).scale(0.8).next_to(aA, UP, buff=0.08)
        # A^+: column space -> row space (bottom curve).
        aP = CurvedArrow(col[0].get_left() + DOWN * 0.25, row[0].get_right() + DOWN * 0.25,
                         angle=-0.7, color=ORANGE_A, stroke_width=4)
        aP_lab = MathTex(r"\mathbf{A}^{+}", color=ORANGE_A).scale(0.8).next_to(aP, DOWN, buff=0.08)
        self.add(aA, aA_lab, aP, aP_lab)

        bij = Tex(r"invertible bijection", color=LIGHT).scale(0.5).move_to((row[0].get_right() + col[0].get_left()) / 2 + UP * 1.05)
        self.add(bij)

        # Both null spaces collapse to 0.
        zero = MathTex(r"\mathbf{0}", color=LIGHT).scale(0.8).move_to((null[0].get_right() + lnull[0].get_left()) / 2)
        z1 = Arrow(null[0].get_right(), zero.get_left(), buff=0.15, color=GRAY_L, stroke_width=3)
        z2 = Arrow(lnull[0].get_left(), zero.get_right(), buff=0.15, color=GRAY_L, stroke_width=3)
        z1_lab = MathTex(r"\mathbf{A}", color=GRAY_L).scale(0.55).next_to(z1, UP, buff=0.06)
        z2_lab = MathTex(r"\mathbf{A}^{+}", color=GRAY_L).scale(0.55).next_to(z2, UP, buff=0.06)
        self.add(zero, z1, z2, z1_lab, z2_lab)


# Render: uv run manim -qh -s --media_dir media_pinv blog/published/the-pseudoinverse/_src/pseudoinverse_map.py PseudoinverseMap
# Move media_pinv/images/pseudoinverse_map/PseudoinverseMap.png -> images/pseudoinverse_map.png

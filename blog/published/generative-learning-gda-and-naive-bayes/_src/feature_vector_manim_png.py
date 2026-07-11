# Runs via the root uv environment.
#
# STATIC PNG (Manim): the multi-hot feature vector for Naive Bayes. The short
# email "buy our lottery" is encoded into a fixed-length 0/1 vector indexed by
# the vocabulary: a 1 wherever the corresponding word appears, a 0 otherwise.

from manim import (
    Arrow,
    Matrix,
    RoundedRectangle,
    Scene,
    Tex,
    VGroup,
    RIGHT,
    LEFT,
    UP,
    DOWN,
    ORIGIN,
)

BG = "#222222"
LIGHT = "#e6e6e6"
ACCENT = "#2ecc71"


class FeatureVector(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ---- Left: the raw email message ----
        msg_text = Tex(r"``buy our lottery''", color=LIGHT).scale(0.9)
        msg_label = Tex(r"Email", color=LIGHT).scale(0.7)
        msg_label.next_to(msg_text, UP, buff=0.25)
        msg_group = VGroup(msg_label, msg_text)
        box = RoundedRectangle(
            corner_radius=0.15,
            width=msg_group.width + 0.8,
            height=msg_group.height + 0.6,
            color=LIGHT,
        )
        box.move_to(msg_group)
        left = VGroup(box, msg_group).to_edge(LEFT, buff=0.9)

        # ---- Right: the multi-hot vector with word annotations ----
        rows = ["0", "0", r"\vdots", "1", r"\vdots", "1", r"\vdots", "1", r"\vdots", "0"]
        words = ["a", "aardvark", "", "buy", "", "lottery", "", "our", "", "zymurgy"]
        present = {3, 5, 7}  # rows whose value is 1

        # element_alignment_corner=ORIGIN centers each entry (the default DR
        # aligns by the right edge, which pushes the narrower \vdots glyph right).
        matrix = Matrix(
            [[r] for r in rows],
            v_buff=0.62,
            bracket_h_buff=0.18,
            element_alignment_corner=ORIGIN,
        )
        matrix.set_color(LIGHT)
        entries = matrix.get_entries()
        for idx in present:
            entries[idx].set_color(ACCENT)

        # Word labels aligned to each matrix row.
        row_mobjects = matrix.get_rows()
        annotations = VGroup()
        word_x = matrix.get_right()[0] + 0.55
        for i, word in enumerate(words):
            if not word:
                continue
            color = ACCENT if i in present else LIGHT
            label = Tex(word, color=color).scale(0.6)
            label.move_to([word_x + label.width / 2, row_mobjects[i].get_center()[1], 0])
            annotations.add(label)

        x_symbol = Tex(r"$\mathbf{x} = $", color=LIGHT).scale(0.9)
        x_symbol.next_to(matrix, LEFT, buff=0.2)

        right = VGroup(x_symbol, matrix, annotations).to_edge(RIGHT, buff=1.1)

        # ---- Connecting arrow ----
        arrow = Arrow(left.get_right(), right.get_left(), color=LIGHT, buff=0.35)
        encode = Tex(r"encode", color=LIGHT).scale(0.6)
        encode.next_to(arrow, UP, buff=0.15)

        self.add(left, arrow, encode, right)


# Render (from repo root):
#   uv run manim -qh -s blog/published/generative-learning-gda-and-naive-bayes/_src/feature_vector_manim_png.py FeatureVector
# Output: media/images/feature_vector_manim_png/FeatureVector.png
# Move it to blog/published/generative-learning-gda-and-naive-bayes/images/feature_vector.png
# Embed:
#   ![Caption](images/feature_vector.png){#fig-feature_vector fig-align="center" width=75%}

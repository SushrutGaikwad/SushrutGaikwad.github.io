# Runs via the root uv environment.
#
# STATIC PNG (Manim): the same spam email encoded two ways. The Bernoulli event
# model records only whether each word is PRESENT (0/1, counted once per email),
# while the Multinomial event model records how many TIMES each word occurs.
# The repeated words ("this", "watch") are where the two models disagree.

from manim import (
    Line,
    MathTex,
    Scene,
    Tex,
    VGroup,
    UP,
    DOWN,
)

BG = "#222222"
LIGHT = "#e6e6e6"
GREEN = "#2ecc71"
MUTED = "#9aa0a6"


class BernoulliVsMultinomial(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ---- The email ----
        message = Tex(r"Spam email:\ \ ``buy this watch this watch''", color=LIGHT).scale(0.8)

        # ---- Table data: (word, bernoulli, multinomial, differ?) ----
        header = ["word", r"Bernoulli\\(present?)", r"Multinomial\\(count)"]
        rows = [
            ("buy", "1", "1", False),
            ("this", "1", "2", True),
            ("watch", "1", "2", True),
            ("exam", "0", "0", False),
        ]

        col_x = [-3.4, 0.1, 3.3]
        top_y = 1.6
        row_step = 0.85

        table = VGroup()

        # Header row.
        for x, text in zip(col_x, header):
            head = Tex(text, color=LIGHT).scale(0.6)
            head.move_to([x, top_y, 0])
            table.add(head)

        # Data rows.
        for r, (word, bern, multi, differ) in enumerate(rows):
            y = top_y - (r + 1) * row_step
            word_color = GREEN if differ else LIGHT
            table.add(Tex(word, color=word_color).scale(0.62).move_to([col_x[0], y, 0]))
            table.add(MathTex(bern, color=LIGHT).scale(0.75).move_to([col_x[1], y, 0]))
            multi_color = GREEN if differ else LIGHT
            table.add(MathTex(multi, color=multi_color).scale(0.75).move_to([col_x[2], y, 0]))

        # Separator lines (same coordinate frame as the cells).
        left_x = col_x[0] - 1.15
        right_x = col_x[2] + 1.15
        bottom_y = top_y - len(rows) * row_step - 0.1
        header_sep_y = top_y - row_step / 2
        table.add(Line([left_x, header_sep_y, 0], [right_x, header_sep_y, 0], color=MUTED, stroke_width=1.5))
        for mid_x in [(col_x[0] + col_x[1]) / 2, (col_x[1] + col_x[2]) / 2]:
            table.add(Line([mid_x, top_y + 0.5, 0], [mid_x, bottom_y, 0], color=MUTED, stroke_width=1.2))

        # ---- Assemble and position ----
        message.to_edge(UP, buff=0.7)
        table.next_to(message, DOWN, buff=0.65)

        caption = Tex(
            r"Repeated words count \textbf{once} (Bernoulli) vs \textbf{every time} (Multinomial).",
            color=MUTED,
        ).scale(0.55)
        caption.next_to(table, DOWN, buff=0.6)

        self.add(message, table, caption)


# Render (from repo root):
#   uv run manim -qh -s blog/published/generative-learning-gda-and-naive-bayes/_src/bernoulli_vs_multinomial_manim_png.py BernoulliVsMultinomial
# Output: media/images/bernoulli_vs_multinomial_manim_png/BernoulliVsMultinomial.png
# Move it to blog/published/generative-learning-gda-and-naive-bayes/images/bernoulli_vs_multinomial.png
# Embed:
#   ![Caption](images/bernoulli_vs_multinomial.png){#fig-bernoulli_vs_multinomial fig-align="center" width=80%}

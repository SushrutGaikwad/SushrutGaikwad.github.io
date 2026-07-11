# Setup (one-time, on Windows):
#   1. Install Python 3.9+ (https://www.python.org/downloads/windows/)
#   2. Install MiKTeX (https://miktex.org/download). Allow MiKTeX to install
#      missing packages on the fly when prompted the first time it runs.
#   3. Install FFmpeg (https://www.gyan.dev/ffmpeg/builds/) or via `winget install ffmpeg`.
#   4. Install Manim Community: pip install manim
#
# This script generates a STATIC PNG illustrating Observation 1 in the
# perceptron section: the parameter vector theta is perpendicular to the
# decision boundary, and points into the positive half-space. The figure
# shows a 2D coordinate system, theta as a green arrow, the boundary as
# a white line through the origin perpendicular to theta, the two
# half-spaces faintly tinted (orange for positive, blue for negative),
# a small right-angle marker at the origin where theta meets the
# boundary, and two sample points (one on each side) with their signs
# labeled.

from manim import (
    Scene, NumberPlane, Arrow, Line, Polygon, Dot, Tex, MathTex, RightAngle,
    VGroup, Create, ORIGIN, UP, DOWN, LEFT, RIGHT, UR, UL, DR, DL,
    config, np
)

# Configure for a static PNG output
config.background_color = "#222222"
config.frame_height = 8.0
config.frame_width = 14.0
config.pixel_height = 1080
config.pixel_width = 1920


class DecisionBoundaryGeometry(Scene):
    def construct(self):
        # ---- Coordinate plane ----
        plane = NumberPlane(
            x_range=[-5, 5, 1], y_range=[-3.5, 3.5, 1],
            background_line_style={
                "stroke_color": "#444444",
                "stroke_width": 1,
                "stroke_opacity": 0.6,
            },
            axis_config={
                "stroke_color": "#888888",
                "stroke_width": 1.5,
                "include_numbers": False,
            },
        )

        # Axis labels
        x1_label = MathTex(r"x_1", color="#cccccc").scale(0.7).next_to(
            plane.x_axis.get_end(), DR, buff=0.1
        )
        x2_label = MathTex(r"x_2", color="#cccccc").scale(0.7).next_to(
            plane.y_axis.get_end(), UR, buff=0.1
        )

        # ---- theta vector ----
        # Direction: roughly upper-right at about 30 degrees above horizontal.
        theta_dir = np.array([np.cos(np.deg2rad(30)),
                              np.sin(np.deg2rad(30)), 0.0])
        theta_length = 2.6
        theta_end = theta_dir * theta_length

        theta_arrow = Arrow(
            start=ORIGIN, end=theta_end,
            color="#2ecc71", buff=0,
            stroke_width=6, max_tip_length_to_length_ratio=0.12,
        )
        theta_label = MathTex(r"\boldsymbol{\theta}", color="#2ecc71").scale(0.95)
        theta_label.next_to(theta_end, UR, buff=0.05)
        theta_label.shift(0.05 * UP)

        # ---- Decision boundary: perpendicular to theta, through origin ----
        # Perpendicular direction
        perp = np.array([-theta_dir[1], theta_dir[0], 0.0])
        boundary_extent = 4.5
        boundary_p1 = perp * boundary_extent
        boundary_p2 = -perp * boundary_extent

        boundary = Line(
            start=boundary_p1, end=boundary_p2,
            color="#ecf0f1", stroke_width=4,
        )
        boundary_label = MathTex(
            r"\boldsymbol{\theta}^{\intercal}\mathbf{x} = 0",
            color="#ecf0f1",
        ).scale(0.75)
        # Place the label near the lower end of the boundary
        boundary_label.move_to(boundary_p2 * 0.85 + 0.8 * theta_dir)

        # ---- Right-angle marker at the origin ----
        # Small square indicating perpendicularity, drawn along theta and perp
        small_size = 0.32
        ra = Polygon(
            ORIGIN,
            small_size * theta_dir,
            small_size * theta_dir + small_size * perp,
            small_size * perp,
            color="#ecf0f1", stroke_width=2,
        )

        # ---- Half-space tints ----
        # Positive half-space (the side theta points into) tinted faint orange.
        # Negative half-space tinted faint blue.
        # Construct big polygons covering each half of the visible plane.
        big = 8.0
        # Positive side: spans from boundary across in the direction of theta.
        pos_polygon = Polygon(
            boundary_p1,
            boundary_p1 + big * theta_dir,
            boundary_p2 + big * theta_dir,
            boundary_p2,
            color="#e67e22", fill_color="#e67e22",
            fill_opacity=0.10, stroke_opacity=0,
        )
        neg_polygon = Polygon(
            boundary_p1,
            boundary_p1 - big * theta_dir,
            boundary_p2 - big * theta_dir,
            boundary_p2,
            color="#3498db", fill_color="#3498db",
            fill_opacity=0.10, stroke_opacity=0,
        )

        # ---- Sample points and their score labels ----
        # Positive side point
        x_pos = theta_dir * 1.7 + perp * 1.2
        x_pos_dot = Dot(point=x_pos, color="#e67e22", radius=0.10)
        x_pos_label = MathTex(r"\mathbf{x}", color="#e67e22").scale(0.7)
        x_pos_label.next_to(x_pos_dot, UR, buff=0.05)
        x_pos_score = MathTex(
            r"\boldsymbol{\theta}^{\intercal}\mathbf{x} > 0",
            color="#e67e22",
        ).scale(0.6)
        x_pos_score.next_to(x_pos_dot, DR, buff=-0.8)

        # Negative side point
        x_neg = -theta_dir * 1.5 + perp * (-1.0)
        x_neg_dot = Dot(point=x_neg, color="#3498db", radius=0.10)
        x_neg_label = MathTex(r"\mathbf{x}'", color="#3498db").scale(0.7)
        x_neg_label.next_to(x_neg_dot, DL, buff=0.05)
        x_neg_score = MathTex(
            r"\boldsymbol{\theta}^{\intercal}\mathbf{x}' < 0",
            color="#3498db",
        ).scale(0.6)
        x_neg_score.next_to(x_neg_dot, UL, buff=0.1)

        # ---- Half-space text labels ----
        pos_text = Tex(r"Predicted positive", color="#e67e22").scale(0.65)
        pos_text.move_to(theta_dir * 3.5 + perp * 2)
        neg_text = Tex(r"Predicted negative", color="#3498db").scale(0.65)
        neg_text.move_to(-theta_dir * 3.0 + perp * 2.3)

        # ---- Add everything to the scene ----
        self.add(
            plane,
            pos_polygon, neg_polygon,
            boundary, boundary_label,
            ra,
            theta_arrow, theta_label,
            x_pos_dot, x_pos_label, x_pos_score,
            x_neg_dot, x_neg_label, x_neg_score,
            pos_text, neg_text,
            x1_label, x2_label,
        )


# How to run (Windows):
#   manim -qh -s decision_boundary_geometry_manim_png.py DecisionBoundaryGeometry
# The `-s` flag tells Manim to save only the final frame as a PNG (no video).
# `-qh` renders at high resolution.
#
# Output:
#   media\images\decision_boundary_geometry_manim_png\DecisionBoundaryGeometry.png
#
# Embed in Quarto using:
#   ![The decision boundary is the line where theta^T x = 0, perpendicular to theta. The vector theta itself points into the positive half-space (where theta^T x > 0). Sample points on each side are labeled with the sign of their score.](images/decision_boundary_geometry.png){#fig-decision_boundary_geometry fig-align="center" width=80% .invert}
#
# (You will want to rename or copy the output PNG to images/decision_boundary_geometry.png in your post directory.)
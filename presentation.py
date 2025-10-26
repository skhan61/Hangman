"""
Hangman AI Solver - Manim Presentation
Clean, professional mathematical animation

Install: pip install manim
Run: manim -pql presentation.py HangmanPresentation
"""

from manim import *
import numpy as np

# Professional color palette
BG = "#0d1117"           # Dark background
PRIMARY = "#58a6ff"      # Blue
SUCCESS = "#3fb950"      # Green
WARNING = "#d29922"      # Gold
DANGER = "#f85149"       # Red
TEXT = "#c9d1d9"         # Light gray
TEXT_DIM = "#8b949e"     # Dimmed gray


class HangmanPresentation(Scene):
    """Main presentation combining all scenes"""

    def construct(self):
        self.camera.background_color = BG

        # Run all scenes
        self.title_scene()
        self.wait(1.5)
        self.clear()

        self.problem_scene()
        self.wait(1.5)
        self.clear()

        self.traditional_scene()
        self.wait(1.5)
        self.clear()

        self.solution_scene()
        self.wait(1.5)
        self.clear()

        self.architecture_scene()
        self.wait(1.5)
        self.clear()

        self.data_scene()
        self.wait(1.5)
        self.clear()

        self.results_scene()
        self.wait(1.5)
        self.clear()

        self.personal_note_scene()
        self.wait(1.5)
        self.clear()

        self.takeaway_scene()
        self.wait(2)

    def title_scene(self):
        """Title with clean centered metric"""
        # Title and subtitle
        title = Text("Hangman AI Solver", font_size=48, weight=BOLD, color=PRIMARY)
        subtitle = Text("Position-wise Neural Approach", font_size=24, color=TEXT_DIM)

        title_group = VGroup(title, subtitle).arrange(DOWN, buff=0.2)
        title_group.to_edge(UP, buff=0.5)

        self.play(Write(title), run_time=1)
        self.play(FadeIn(subtitle))
        self.wait(0.3)

        # Center: Build metric display
        # First create the text elements centered
        percentage = Text("67.2%", font_size=72, weight=BOLD, color=SUCCESS)
        label = Text("Win Rate", font_size=28, color=TEXT)

        # Stack them vertically
        metric_text = VGroup(percentage, label).arrange(DOWN, buff=0.3)
        metric_text.move_to(ORIGIN)

        # Create circle around the text group
        circle = Circle(radius=1.8, color=SUCCESS, stroke_width=8)
        circle.move_to(metric_text.get_center())

        # Animate: circle first, then text inside
        self.play(Create(circle), run_time=1.2)
        self.play(Write(percentage))
        self.play(Write(label))
        self.wait(0.5)

        # Bottom section
        improvement = Text("3.7× Better than Baseline",
                          font_size=30, color=WARNING, weight=BOLD)
        improvement.to_edge(DOWN, buff=0.9)

        author = Text("by Sayem Khan", font_size=20, color=TEXT_DIM)
        author.to_edge(DOWN, buff=0.3)

        self.play(FadeIn(improvement))
        self.play(Write(author))

    def problem_scene(self):
        """Show the problem clearly"""
        title = Text("The Problem", font_size=48, weight=BOLD, color=PRIMARY)
        title.to_edge(UP, buff=0.6)
        self.play(Write(title))

        # Center: Hangman word (larger)
        word_text = Text("_____", font_size=96, font="Monospace", color=TEXT, weight=BOLD)
        word_text.move_to(ORIGIN + UP * 0.8)

        self.play(Write(word_text))
        self.wait(0.5)

        # Animate guessing
        guesses = [("E", "____E"), ("A", "A___E"), ("P", "APP_E"), ("L", "APPLE")]

        for letter, new_word in guesses:
            guess_label = Text(f"Guess: {letter}", font_size=40, color=WARNING)
            guess_label.next_to(word_text, UP, buff=0.8)

            self.play(Write(guess_label), run_time=0.3)
            self.wait(0.2)

            new_text = Text(new_word, font_size=96, font="Monospace",
                          color=SUCCESS, weight=BOLD)
            new_text.move_to(word_text)

            self.play(Transform(word_text, new_text), run_time=0.4)
            self.play(FadeOut(guess_label), run_time=0.2)

        # Success checkmark
        check = Text("✓", font_size=120, color=SUCCESS, weight=BOLD)
        check.next_to(word_text, RIGHT, buff=0.8)
        self.play(Write(check), run_time=0.5)
        self.wait(0.5)

        self.play(FadeOut(word_text), FadeOut(check))

        # Show constraints
        constraints = VGroup(
            Text("• 6 wrong guesses allowed", font_size=28, color=TEXT),
            Text("• 250K training words", font_size=28, color=TEXT),
            Text("• 250K test words (unseen)", font_size=28, color=TEXT),
            Text("• Must beat 18% baseline", font_size=28, color=DANGER, weight=BOLD)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        constraints.move_to(ORIGIN)

        self.play(FadeIn(constraints, shift=UP * 0.3))

    def traditional_scene(self):
        """Traditional frequency approach failure"""
        title = Text("Traditional Approach Fails", font_size=48, weight=BOLD, color=DANGER)
        title.to_edge(UP, buff=0.6)
        self.play(Write(title))

        # Pattern
        pattern = Text("_pp_e", font_size=72, font="Monospace", color=WARNING, weight=BOLD)
        pattern.move_to(UP * 2)
        self.play(Write(pattern))

        # Approach explanation
        approach_label = Text("Frequency Counting:", font_size=32, color=TEXT)
        approach_label.next_to(pattern, DOWN, buff=0.8)
        self.play(Write(approach_label))

        # Frequency counts in a nice box
        freq_box = VGroup(
            Text("a: 45", font_size=32, color=SUCCESS),
            Text("l: 38", font_size=32, color=WARNING),
            Text("m: 15", font_size=32, color="#e3b341"),
            Text("o: 12", font_size=32, color=DANGER)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        freq_box.next_to(approach_label, DOWN, buff=0.5)

        rect = SurroundingRectangle(freq_box, color=TEXT_DIM, buff=0.3, stroke_width=3)

        self.play(Create(rect), FadeIn(freq_box))
        self.wait(0.5)

        # Problem
        problem = Text("Ignores Position Context!", font_size=36,
                      color=DANGER, weight=BOLD)
        problem.to_edge(DOWN, buff=1.5)
        self.play(Write(problem))

        # Win rate
        bad_rate = Text("Win Rate: 18%", font_size=48, color=DANGER, weight=BOLD)
        bad_rate.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(bad_rate))

    def solution_scene(self):
        """Our position-wise solution"""
        title = Text("Our Solution: Position-wise Prediction",
                    font_size=44, weight=BOLD, color=SUCCESS)
        title.to_edge(UP, buff=0.6)
        self.play(Write(title))

        # Pattern
        pattern = Text("_pp_e", font_size=64, font="Monospace",
                      color=WARNING, weight=BOLD)
        pattern.to_edge(UP, buff=1.8)
        self.play(Write(pattern))

        # Position boxes - larger and properly spaced
        squares = VGroup(*[
            Square(side_length=1.0, color=TEXT, stroke_width=4) for _ in range(5)
        ]).arrange(RIGHT, buff=0.2)
        squares.move_to(UP * 0.8)

        # Labels inside squares
        labels = VGroup(*[
            Text(c, font_size=48, font="Monospace", color=TEXT, weight=BOLD)
            for c in "_pp_e"
        ])
        for label, square in zip(labels, squares):
            label.move_to(square.get_center())

        self.play(Create(squares), Write(labels))

        # Highlight masked positions
        highlight_0 = squares[0].copy().set_stroke(WARNING, width=10)
        highlight_3 = squares[3].copy().set_stroke(WARNING, width=10)
        self.play(Create(highlight_0), Create(highlight_3))

        # Predictions side by side, properly aligned
        pred_left = VGroup(
            Text("Position 0:", font_size=28, color=TEXT, weight=BOLD),
            Text("P(a) = 0.95", font_size=26, color=SUCCESS),
            Text("P(o) = 0.02", font_size=24, color=TEXT_DIM),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)

        pred_right = VGroup(
            Text("Position 3:", font_size=28, color=TEXT, weight=BOLD),
            Text("P(l) = 0.92", font_size=26, color=SUCCESS),
            Text("P(k) = 0.05", font_size=24, color=TEXT_DIM),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)

        # Position them below squares
        pred_left.next_to(squares, DOWN, buff=1.0).shift(LEFT * 2.5)
        pred_right.next_to(squares, DOWN, buff=1.0).shift(RIGHT * 2.5)

        self.play(FadeIn(pred_left), FadeIn(pred_right))
        self.wait(0.8)

        # Result at bottom
        result = Text("Best Guess: 'a'", font_size=40, color=SUCCESS, weight=BOLD)
        result.to_edge(DOWN, buff=1.2)

        win_rate = Text("Win Rate: 67.2%!", font_size=52, color=SUCCESS, weight=BOLD)
        win_rate.to_edge(DOWN, buff=0.4)

        self.play(Write(result))
        self.play(Write(win_rate))

    def architecture_scene(self):
        """BiLSTM architecture diagram"""
        title = Text("Neural Architecture: BiLSTM",
                    font_size=48, weight=BOLD, color=PRIMARY)
        title.to_edge(UP, buff=0.6)
        self.play(Write(title))

        # Larger boxes with better proportions
        box_width = 7
        boxes = []

        # Input
        input_box = Rectangle(width=box_width, height=1.0,
                             fill_color=PRIMARY, fill_opacity=0.2,
                             stroke_color=PRIMARY, stroke_width=4)
        input_label = Text("Input: [MASK, p, p, MASK, e]",
                          font_size=26, color=TEXT, weight=BOLD)
        input_label.move_to(input_box)
        boxes.append(VGroup(input_box, input_label))

        # Embedding
        emb_box = Rectangle(width=box_width, height=1.0,
                           fill_color=SUCCESS, fill_opacity=0.2,
                           stroke_color=SUCCESS, stroke_width=4)
        emb_label = Text("Embedding (256d)", font_size=26, color=TEXT, weight=BOLD)
        emb_label.move_to(emb_box)
        boxes.append(VGroup(emb_box, emb_label))

        # BiLSTM (larger)
        lstm_box = Rectangle(width=box_width, height=1.4,
                            fill_color=WARNING, fill_opacity=0.2,
                            stroke_color=WARNING, stroke_width=4)
        lstm_label = Text("BiLSTM (4 layers, 512d)",
                         font_size=28, color=TEXT, weight=BOLD)
        lstm_label.move_to(lstm_box)
        boxes.append(VGroup(lstm_box, lstm_label))

        # Dropout
        drop_box = Rectangle(width=box_width, height=0.8,
                            fill_color="#e3b341", fill_opacity=0.2,
                            stroke_color="#e3b341", stroke_width=4)
        drop_label = Text("Dropout (0.3)", font_size=24, color=TEXT)
        drop_label.move_to(drop_box)
        boxes.append(VGroup(drop_box, drop_label))

        # Output
        out_box = Rectangle(width=box_width, height=1.0,
                           fill_color=DANGER, fill_opacity=0.2,
                           stroke_color=DANGER, stroke_width=4)
        out_label = Text("Output: [batch, len, 26]",
                        font_size=26, color=TEXT, weight=BOLD)
        out_label.move_to(out_box)
        boxes.append(VGroup(out_box, out_label))

        # Arrange all boxes
        all_boxes = VGroup(*boxes).arrange(DOWN, buff=0.35)
        all_boxes.move_to(ORIGIN + UP * 0.2)

        # Arrows
        arrows = []
        for i in range(len(boxes) - 1):
            arrow = Arrow(boxes[i].get_bottom(), boxes[i + 1].get_top(),
                         buff=0.1, color=PRIMARY, stroke_width=8,
                         max_tip_length_to_length_ratio=0.15)
            arrows.append(arrow)

        # Animate
        for box, arrow in zip(boxes[:-1], arrows):
            self.play(FadeIn(box), run_time=0.5)
            self.play(Create(arrow), run_time=0.3)
        self.play(FadeIn(boxes[-1]))

        # Stats at bottom
        stats = VGroup(
            Text("• 5.2M parameters", font_size=26, color=SUCCESS),
            Text("• 20 epochs", font_size=26, color=WARNING),
            Text("• ~30 min training", font_size=26, color=PRIMARY)
        ).arrange(RIGHT, buff=0.8)
        stats.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(stats))

    def data_scene(self):
        """Training data generation"""
        title = Text("Training Data: 13 Masking Strategies",
                    font_size=44, weight=BOLD, color=PRIMARY)
        title.to_edge(UP, buff=0.6)
        self.play(Write(title))

        # Word example
        word = Text("APPLE", font_size=64, font="Monospace",
                   color=SUCCESS, weight=BOLD)
        word.move_to(UP * 2.2)
        self.play(Write(word))

        # Masking strategies in a grid
        strategies = [
            ("Random", "_P_LE"),
            ("Left→Right", "A____"),
            ("Vowels First", "_PP__"),
            ("Center Out", "__P__"),
            ("Edges First", "_PPL_")
        ]

        strategy_items = VGroup()
        for name, masked in strategies:
            line = VGroup(
                Text(name + ":", font_size=26, color=WARNING, weight=BOLD),
                Text(masked, font_size=36, font="Monospace", color=TEXT, weight=BOLD)
            ).arrange(RIGHT, buff=0.5)
            strategy_items.add(line)

        strategy_items.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        strategy_items.move_to(ORIGIN + DOWN * 0.2)

        for item in strategy_items:
            self.play(FadeIn(item, shift=RIGHT * 0.3), run_time=0.4)

        # Stats at bottom
        stats = VGroup(
            Text("227K words", font_size=28, color=SUCCESS, weight=BOLD),
            Text("→", font_size=28, color=TEXT_DIM),
            Text("21M samples", font_size=28, color=PRIMARY, weight=BOLD),
            Text("→", font_size=28, color=TEXT_DIM),
            Text("13 strategies", font_size=28, color=WARNING, weight=BOLD)
        ).arrange(RIGHT, buff=0.3)
        stats.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(stats))

    def results_scene(self):
        """Results bar chart"""
        title = Text("Results: 3.7× Improvement",
                    font_size=48, weight=BOLD, color=SUCCESS)
        title.to_edge(UP, buff=0.6)
        self.play(Write(title))

        # Strategies and rates
        names = ["Baseline\n18%", "Frequency\n15.1%", "Positional\n17%", "Neural\n67.2%"]
        rates = [0.18, 0.151, 0.17, 0.672]
        colors = [DANGER, "#e3b341", WARNING, SUCCESS]

        # Create bars manually with better layout
        bars = VGroup()
        labels_group = VGroup()

        bar_width = 1.2
        max_height = 4.5
        spacing = 2.0

        for i, (name, rate, color) in enumerate(zip(names, rates, colors)):
            # Bar
            bar_height = rate * (max_height / 0.75)
            bar = Rectangle(
                width=bar_width,
                height=bar_height,
                fill_color=color,
                fill_opacity=0.8,
                stroke_color=color,
                stroke_width=4
            )
            # Position bars evenly
            x_pos = (i - 1.5) * spacing
            bar.move_to([x_pos, bar_height / 2 - 1.5, 0])
            bars.add(bar)

            # Label
            label = Text(name, font_size=22, color=TEXT, weight=BOLD)
            label.next_to(bar, DOWN, buff=0.3)
            labels_group.add(label)

            # Animate
            self.play(GrowFromEdge(bar, DOWN), Write(label), run_time=0.6)

        # Highlight neural
        neural_bar = bars[3]
        highlight_rect = SurroundingRectangle(
            neural_bar, color=SUCCESS, stroke_width=10, buff=0.15
        )
        self.play(Create(highlight_rect))

        # Bottom text
        final_text = Text("67.2% on official test | 2.1 tries remaining",
                         font_size=28, color=SUCCESS, weight=BOLD)
        final_text.to_edge(DOWN, buff=0.4)
        self.play(Write(final_text))

    def personal_note_scene(self):
        """Personal thoughts and reflections"""
        title = Text("Personal Note", font_size=48, weight=BOLD, color=PRIMARY)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # Main message about Trexquant
        note1 = Text("Trexquant never replied.", font_size=28, color=TEXT)
        note1.move_to(UP * 2)
        self.play(Write(note1))
        self.wait(0.5)

        # But I don't care
        note2 = Text("But I don't care.", font_size=32, color=WARNING, weight=BOLD)
        note2.next_to(note1, DOWN, buff=0.4)
        self.play(Write(note2))
        self.wait(0.8)

        # What matters
        points = VGroup(
            Text("• This was a REAL problem-solving challenge", font_size=24, color=SUCCESS),
            Text("• End-to-end: design → build → train → deploy", font_size=24, color=TEXT),
            Text("• Not LeetCode memorization", font_size=24, color=TEXT_DIM),
            Text("• Not high school probability questions", font_size=24, color=TEXT_DIM),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        points.move_to(ORIGIN)

        for point in points:
            self.play(FadeIn(point, shift=RIGHT * 0.2), run_time=0.5)

        self.wait(0.8)

        # Final thought
        final = VGroup(
            Text("The journey of building, debugging, optimizing,", font_size=22, color=TEXT),
            Text("and seeing it work is MORE valuable than", font_size=22, color=TEXT),
            Text("any company's hiring decision.", font_size=22, color=SUCCESS, weight=BOLD)
        ).arrange(DOWN, buff=0.2)
        final.to_edge(DOWN, buff=0.8)

        self.play(FadeIn(final))

    def takeaway_scene(self):
        """Final takeaway"""
        title = Text("Key Takeaway", font_size=52, weight=BOLD, color=PRIMARY)
        title.to_edge(UP, buff=0.8)
        self.play(Write(title))

        # Main messages
        messages = VGroup(
            Text("Build things.", font_size=48, color=SUCCESS, weight=BOLD),
            Text("Ship them.", font_size=48, color=WARNING, weight=BOLD),
            Text("Learn from them.", font_size=48, color=PRIMARY, weight=BOLD),
            Text("Repeat.", font_size=48, color=DANGER, weight=BOLD)
        ).arrange(DOWN, buff=0.5)
        messages.move_to(ORIGIN + UP * 0.5)

        for msg in messages:
            self.play(Write(msg), run_time=0.7)
            self.wait(0.2)

        self.wait(0.8)

        # Humor
        humor = Text("(And beg money from friends to live)",
                    font_size=24, color=TEXT_DIM, slant=ITALIC)
        humor.next_to(messages, DOWN, buff=0.8)
        self.play(FadeIn(humor))

        self.wait(1)

        # Contact
        contact = VGroup(
            Text("Sayem Khan", font_size=32, weight=BOLD, color=PRIMARY),
            Text("sayem.eee.kuet@gmail.com", font_size=24, color=TEXT),
            Text("github.com/skhan61/Hangman", font_size=24, color=WARNING)
        ).arrange(DOWN, buff=0.3)
        contact.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(contact))


# Individual scenes for testing
class TitleScene(Scene):
    def construct(self):
        self.camera.background_color = BG
        HangmanPresentation.title_scene(self)


class ProblemScene(Scene):
    def construct(self):
        self.camera.background_color = BG
        HangmanPresentation.problem_scene(self)


class TraditionalScene(Scene):
    def construct(self):
        self.camera.background_color = BG
        HangmanPresentation.traditional_scene(self)


class SolutionScene(Scene):
    def construct(self):
        self.camera.background_color = BG
        HangmanPresentation.solution_scene(self)


class ArchitectureScene(Scene):
    def construct(self):
        self.camera.background_color = BG
        HangmanPresentation.architecture_scene(self)


class DataScene(Scene):
    def construct(self):
        self.camera.background_color = BG
        HangmanPresentation.data_scene(self)


class ResultsScene(Scene):
    def construct(self):
        self.camera.background_color = BG
        HangmanPresentation.results_scene(self)


class PersonalNoteScene(Scene):
    def construct(self):
        self.camera.background_color = BG
        HangmanPresentation.personal_note_scene(self)


class TakeawayScene(Scene):
    def construct(self):
        self.camera.background_color = BG
        HangmanPresentation.takeaway_scene(self)

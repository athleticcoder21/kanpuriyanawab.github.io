from manim import *


class WeightSync(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        def txt(content, **kwargs):
            return Text(content, font="Arial", **kwargs)

        ink = "#314766"
        muted = "#68758A"
        merged = "#E96535"
        base = "#8A8984"
        pale = "#F4F5F7"
        green = "#23836A"

        title = txt("Why the last chunk went stale", font_size=36, color=ink)
        title.to_edge(UP, buff=0.35)
        subtitle = txt(
            "TRL yields live weights · vLLM copies them in chunks",
            font_size=21,
            color=muted,
        ).next_to(title, DOWN, buff=0.12)
        self.add(title, subtitle)

        def card(center, width, height, label, detail, color):
            box = RoundedRectangle(
                width=width,
                height=height,
                corner_radius=0.12,
                stroke_color=color,
                stroke_width=2,
                fill_color=pale,
                fill_opacity=1,
            ).move_to(center)
            main = txt(label, font_size=29, color=color, weight=BOLD)
            main.move_to(box.get_center() + UP * 0.13)
            small = txt(detail, font_size=16, color=muted)
            small.move_to(box.get_center() + DOWN * 0.30)
            return VGroup(box, main, small)

        source_label = txt("TRL", font_size=20, color=ink, weight=BOLD)
        source_label.move_to(LEFT * 5.35 + UP * 0.75)
        queue_label = txt("vLLM buffer", font_size=20, color=ink, weight=BOLD)
        queue_label.move_to(UP * 0.75)
        sent_label = txt("vLLM weights", font_size=20, color=ink, weight=BOLD)
        sent_label.move_to(RIGHT * 4.95 + UP * 0.75)

        source = card(LEFT * 3.05 + DOWN * 0.10, 2.35, 1.15, "W + Δ", "adapter merged", merged)
        queue = RoundedRectangle(
            width=2.55,
            height=1.15,
            corner_radius=0.12,
            stroke_color=muted,
            stroke_width=2,
            fill_color=pale,
            fill_opacity=1,
        ).move_to(DOWN * 0.10)
        waiting = txt("waiting", font_size=19, color=muted).move_to(queue.get_center())
        sent = RoundedRectangle(
            width=2.35,
            height=1.15,
            corner_radius=0.12,
            stroke_color=muted,
            stroke_width=2,
            fill_color=pale,
            fill_opacity=1,
        ).move_to(RIGHT * 4.00 + DOWN * 0.10)
        sent_waiting = txt("—", font_size=24, color=muted).move_to(sent.get_center())

        arrow_a = Arrow(LEFT * 1.78 + DOWN * 0.10, LEFT * 1.36 + DOWN * 0.10, color=muted, buff=0.05)
        arrow_b = Arrow(RIGHT * 1.38 + DOWN * 0.10, RIGHT * 2.80 + DOWN * 0.10, color=muted, buff=0.05)
        self.play(FadeIn(source_label), FadeIn(queue_label), FadeIn(sent_label))
        self.play(FadeIn(source), FadeIn(queue), FadeIn(waiting), FadeIn(sent), FadeIn(sent_waiting))
        self.play(Create(arrow_a), Create(arrow_b))

        note = txt("TRL merges the adapter, then yields each live tensor", font_size=21, color=ink)
        note.move_to(DOWN * 1.35)
        self.play(Write(note))
        self.wait(0.8)

        # Earlier chunks are copied while TRL is paused at a yield.
        ref_one = RoundedRectangle(width=0.54, height=0.42, corner_radius=0.06, color=merged, fill_color=merged, fill_opacity=1)
        ref_two = RoundedRectangle(width=0.54, height=0.42, corner_radius=0.06, color=merged, fill_color=merged, fill_opacity=1)
        refs = VGroup(ref_one, ref_two).arrange(RIGHT, buff=0.16).move_to(queue.get_center())
        next_note = txt("Earlier chunk: copied while the adapter is still merged", font_size=21, color=ink)
        next_note.move_to(DOWN * 1.35)
        self.play(FadeOut(waiting), FadeIn(refs), FadeOut(note), FadeIn(next_note))
        note = next_note

        early_chunk = RoundedRectangle(width=1.55, height=0.56, corner_radius=0.08, color=merged, fill_color=merged, fill_opacity=1)
        early_text = txt("chunk 1", font_size=17, color=WHITE, weight=BOLD).move_to(early_chunk.get_center())
        early = VGroup(early_chunk, early_text).move_to(sent.get_center() + UP * 0.22)
        self.play(TransformFromCopy(refs, early), run_time=0.8)
        self.play(FadeOut(refs), FadeIn(waiting), run_time=0.35)
        self.wait(0.5)

        # The final chunk still holds references to the live model parameters.
        final_refs = VGroup(
            RoundedRectangle(width=0.54, height=0.42, corner_radius=0.06, color=merged, fill_color=merged, fill_opacity=1),
            RoundedRectangle(width=0.54, height=0.42, corner_radius=0.06, color=merged, fill_color=merged, fill_opacity=1),
        ).arrange(RIGHT, buff=0.16).move_to(queue.get_center())
        next_note = txt("Last chunk: vLLM is still holding references", font_size=21, color=ink)
        next_note.move_to(DOWN * 1.35)
        self.play(FadeOut(waiting), FadeIn(final_refs), FadeOut(note), FadeIn(next_note))
        note = next_note
        self.wait(0.6)

        # Exhausting the generator resumes TRL after its final yield and unmerges W.
        base_source = card(LEFT * 3.05 + DOWN * 0.10, 2.35, 1.15, "W", "adapter unmerged", base)
        unmerge_note = txt("generator ends → TRL unmerges", font_size=21, color=base)
        unmerge_note.move_to(DOWN * 1.35)
        self.play(
            Transform(source, base_source),
            Transform(note, unmerge_note),
            *[ref.animate.set_fill(base).set_stroke(base) for ref in final_refs],
            run_time=0.8,
        )
        self.wait(0.5)

        # vLLM now copies the last chunk, after the referenced weights changed back to W.
        final_chunk = RoundedRectangle(width=1.55, height=0.56, corner_radius=0.08, color=base, fill_color=base, fill_opacity=1)
        final_text = txt("last chunk", font_size=16, color=WHITE, weight=BOLD).move_to(final_chunk.get_center())
        final = VGroup(final_chunk, final_text).move_to(sent.get_center() + DOWN * 0.27)
        self.play(TransformFromCopy(final_refs, final), run_time=0.8)
        self.play(FadeOut(final_refs), FadeOut(sent_waiting))

        summary = txt("Sent: merged weights first · base weights last", font_size=23, color=ink, weight=BOLD)
        summary.move_to(DOWN * 2.15)
        legend_merged = Dot(color=merged).scale(0.8)
        legend_base = Dot(color=base).scale(0.8)
        legend = VGroup(
            legend_merged,
            txt("merged", font_size=16, color=muted),
            legend_base,
            txt("base", font_size=16, color=muted),
        ).arrange(RIGHT, buff=0.12).move_to(DOWN * 3.15)
        self.play(Write(summary), FadeIn(legend))
        self.wait(2)

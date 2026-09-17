"""Render the reachAQ overview deck from content.yaml.

All copy lives in content.yaml; this file only lays it out. Edit the YAML and
rerun to change wording, reorder slides, or add and drop them.

Slides are built on the template's own layouts and their placeholders rather
than free-floating text boxes, so the result behaves like a normal deck: the
Outline view shows the text, tab and shift-tab change bullet level, and
restyling from the Slide Master reaches every slide at once.

    python docs/presentation/make_assets.py --media-root <checkout>/temp
    python docs/presentation/build_deck.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
CONTENT = HERE / "content.yaml"
OUT = HERE / "reachAQ-overview.pptx"

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)

# One palette, used everywhere. Change it here and every slide follows.
INK = RGBColor(0x1E, 0x24, 0x2B)
MUTED = RGBColor(0x6B, 0x72, 0x7B)
ACCENT = RGBColor(0xB3, 0x59, 0x3F)
OLD = RGBColor(0x6B, 0x7F, 0x9E)

# Layout indices in the default template.
L_TITLE, L_CONTENT, L_TWO, L_TITLE_ONLY = 0, 1, 3, 5

MARGIN = Inches(0.72)
BODY_TOP = Inches(1.95)
CONTENT_W = SLIDE_W - 2 * MARGIN


def _style(run, size, color=INK, bold=False, italic=False):
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = "Segoe UI"


def _set_title(slide, text):
    title = slide.shapes.title
    title.left, title.top = MARGIN, Inches(0.5)
    title.width, title.height = CONTENT_W, Inches(0.95)
    frame = title.text_frame
    frame.word_wrap = True
    frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    run = frame.paragraphs[0].add_run()
    run.text = text
    _style(run, 30, INK, bold=True)
    return title


def _lead(slide, text, top=Inches(1.42)):
    """One line under the title that frames the slide."""
    box = slide.shapes.add_textbox(MARGIN, top, CONTENT_W, Inches(0.5))
    frame = box.text_frame
    frame.word_wrap = True
    run = frame.paragraphs[0].add_run()
    run.text = text
    _style(run, 14, MUTED, italic=True)
    return box


def _fill_bullets(placeholder, items, size=15, color=INK, space_after=10):
    frame = placeholder.text_frame
    frame.word_wrap = True
    frame.clear()
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.level = 0
        paragraph.space_after = Pt(space_after)
        run = paragraph.add_run()
        run.text = str(item)
        _style(run, size, color)


def _column_heading(placeholder, text, color):
    frame = placeholder.text_frame
    frame.word_wrap = True
    frame.clear()
    run = frame.paragraphs[0].add_run()
    run.text = text
    _style(run, 15, color, bold=True)


def _picture(slide, name, left, top, max_w, max_h):
    """Fit an asset inside a box, preserving aspect and centring it.

    Sizing by width alone silently overruns whatever sits below, and a bounds
    check does not catch it because the shape is still on the slide.
    """
    path = ASSETS / name
    if not path.is_file():
        box = slide.shapes.add_textbox(left, top, max_w, Inches(0.5))
        run = box.text_frame.paragraphs[0].add_run()
        run.text = f"[missing asset: {name} — run make_assets.py]"
        _style(run, 11, MUTED, italic=True)
        return
    with Image.open(path) as image:
        native_w, native_h = image.size
    scale = min(max_w / native_w, max_h / native_h)
    width, height = int(native_w * scale), int(native_h * scale)
    slide.shapes.add_picture(
        str(path), int(left) + int((max_w - width) / 2), int(top), width=width, height=height
    )
    return height


def _caption(slide, text, top):
    box = slide.shapes.add_textbox(MARGIN, top, CONTENT_W, Inches(0.55))
    frame = box.text_frame
    frame.word_wrap = True
    frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    run = frame.paragraphs[0].add_run()
    run.text = " ".join(text.split())
    _style(run, 11.5, MUTED, italic=True)


def _rule(slide, top):
    """A thin accent rule under the title, for a bit of structure."""
    from pptx.enum.shapes import MSO_SHAPE

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, top, Inches(1.1), Pt(3))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    bar.shadow.inherit = False


def _footer(slide, text, number):
    box = slide.shapes.add_textbox(MARGIN, SLIDE_H - Inches(0.52),
                                   CONTENT_W, Inches(0.32))
    frame = box.text_frame
    paragraph = frame.paragraphs[0]
    run = paragraph.add_run()
    run.text = text
    _style(run, 9, MUTED)

    box = slide.shapes.add_textbox(SLIDE_W - MARGIN - Inches(0.7),
                                   SLIDE_H - Inches(0.52), Inches(0.7), Inches(0.32))
    frame = box.text_frame
    frame.paragraphs[0].alignment = PP_ALIGN.RIGHT
    run = frame.paragraphs[0].add_run()
    run.text = str(number)
    _style(run, 9, MUTED)


def _notes(slide, text):
    if text:
        slide.notes_slide.notes_text_frame.text = " ".join(str(text).split())


# ----------------------------------------------------------------- layouts


def build_cover(prs, spec, meta):
    slide = prs.slides.add_slide(prs.slide_layouts[L_TITLE])
    title, subtitle = slide.placeholders[0], slide.placeholders[1]

    title.left, title.top = MARGIN, Inches(2.45)
    title.width, title.height = CONTENT_W, Inches(1.35)
    frame = title.text_frame
    frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    run = frame.paragraphs[0].add_run()
    run.text = spec["title"]
    _style(run, 54, INK, bold=True)

    subtitle.left, subtitle.top = MARGIN, Inches(3.85)
    subtitle.width, subtitle.height = Inches(9.5), Inches(1.0)
    frame = subtitle.text_frame
    frame.word_wrap = True
    frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    run = frame.paragraphs[0].add_run()
    run.text = spec.get("subtitle", "")
    _style(run, 17, MUTED)

    _rule(slide, Inches(2.15))
    return slide


def build_bullets(prs, spec, meta):
    slide = prs.slides.add_slide(prs.slide_layouts[L_CONTENT])
    _set_title(slide, spec["title"])
    _rule(slide, Inches(1.28))

    top = BODY_TOP
    if spec.get("lead"):
        _lead(slide, spec["lead"], top=Inches(1.5))
        top = Inches(2.25)

    body = slide.placeholders[1]
    body.left, body.top = MARGIN, top
    body.width, body.height = CONTENT_W, SLIDE_H - top - Inches(0.85)
    _fill_bullets(body, spec["bullets"], size=15, space_after=13)
    return slide


def build_split(prs, spec, meta):
    slide = prs.slides.add_slide(prs.slide_layouts[L_TWO])
    _set_title(slide, spec["title"])
    _rule(slide, Inches(1.28))

    top = BODY_TOP
    if spec.get("lead"):
        _lead(slide, spec["lead"], top=Inches(1.5))
        top = Inches(2.3)

    gutter = Inches(0.55)
    column_w = int((CONTENT_W - gutter) / 2)
    height = SLIDE_H - top - Inches(0.85)

    left, right = slide.placeholders[1], slide.placeholders[2]
    for placeholder, x in ((left, MARGIN), (right, MARGIN + column_w + gutter)):
        placeholder.left, placeholder.top = x, top + Inches(0.42)
        placeholder.width, placeholder.height = column_w, height - Inches(0.42)

    for heading, color, x in (
        (spec.get("left_heading", ""), OLD, MARGIN),
        (spec.get("right_heading", ""), ACCENT, MARGIN + column_w + gutter),
    ):
        box = slide.shapes.add_textbox(x, top, column_w, Inches(0.38))
        run = box.text_frame.paragraphs[0].add_run()
        run.text = heading
        _style(run, 15, color, bold=True)

    _fill_bullets(left, spec.get("left", []), size=13.5, space_after=11)
    _fill_bullets(right, spec.get("right", []), size=13.5, space_after=11)
    return slide


def build_image(prs, spec, meta):
    slide = prs.slides.add_slide(prs.slide_layouts[L_TITLE_ONLY])
    _set_title(slide, spec["title"])
    _rule(slide, Inches(1.28))

    top = Inches(1.85)
    if spec.get("lead"):
        _lead(slide, spec["lead"], top=Inches(1.5))
        top = Inches(2.02)

    # Charts are wide, so they end up height-limited; give them the room rather
    # than leaving a third of the slide empty beside a small figure.
    caption = spec.get("caption")
    bottom = SLIDE_H - Inches(0.7) - (Inches(0.6) if caption else Inches(0))
    height = _picture(slide, spec["image"], MARGIN, top, CONTENT_W, bottom - top)

    if caption:
        _caption(slide, caption, top + (height or 0) + Inches(0.14))
    return slide


BUILDERS = {
    "cover": build_cover,
    "bullets": build_bullets,
    "split": build_split,
    "image": build_image,
}


def main() -> int:
    content = yaml.safe_load(CONTENT.read_text(encoding="utf-8"))
    meta = content.get("meta", {})

    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H

    for index, spec in enumerate(content["slides"], start=1):
        layout = spec.get("layout", "bullets")
        builder = BUILDERS.get(layout)
        if builder is None:
            raise SystemExit(f"slide {index}: unknown layout {layout!r}; "
                             f"expected one of {', '.join(sorted(BUILDERS))}")
        slide = builder(prs, spec, meta)
        if layout != "cover":
            _footer(slide, meta.get("footer", ""), index)
        _notes(slide, spec.get("notes"))

    prs.save(OUT)
    print(f"wrote {OUT} with {len(prs.slides._sldIdLst)} slides")
    return 0


if __name__ == "__main__":
    sys.exit(main())

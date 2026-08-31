"""Synthetic playing-card renderer — the source of training data for the CNN.

There is no public dataset of every card in every skin, so the vision model is
trained on cards this module draws.  The important part is *variety*: several
"deck styles" (different fonts, palettes, corner layouts, pip drawing methods),
so the network learns what a queen of hearts looks like rather than what one
particular renderer's queen of hearts looks like.

Styles are split into a training set and a held-out set, and the reported
accuracy is measured on decks the network never saw.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from ..cards import RANK_CHARS, SUIT_CHARS, rank_of, suit_of

FONT_DIR_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/freefont",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts/truetype/noto",
]

SUIT_GLYPHS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
RED_SUITS = ("d", "h")


def _find_fonts() -> List[str]:
    out = []
    for directory in FONT_DIR_CANDIDATES:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if name.lower().endswith(".ttf"):
                out.append(os.path.join(directory, name))
    return out


AVAILABLE_FONTS = _find_fonts()


def _font(path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    if path and os.path.exists(path):
        try:
            return ImageFont.truetype(path, size)
        except OSError:  # pragma: no cover - corrupt font
            pass
    return ImageFont.load_default(size=size)


@dataclass
class CardStyle:
    """One deck skin."""

    name: str
    font: Optional[str] = None
    face: Tuple[int, int, int] = (250, 250, 246)
    black: Tuple[int, int, int] = (25, 25, 30)
    red: Tuple[int, int, int] = (190, 30, 40)
    border: Tuple[int, int, int] = (160, 160, 160)
    radius: float = 0.10
    corner_only: bool = False     # four-colour "index only" decks
    centre_pip: bool = True
    pip_mode: str = "glyph"       # "glyph" or "vector"
    rank_scale: float = 0.34
    suit_scale: float = 0.24
    four_colour: bool = False


FOUR_COLOUR = {"c": (20, 130, 60), "d": (30, 90, 200), "h": (190, 30, 40), "s": (25, 25, 30)}


def _pick(fonts: Sequence[str], *keywords: str) -> Optional[str]:
    for keyword in keywords:
        for path in fonts:
            if keyword.lower() in os.path.basename(path).lower():
                return path
    return fonts[0] if fonts else None


def build_styles() -> Tuple[List[CardStyle], List[CardStyle]]:
    """Return ``(train_styles, holdout_styles)`` — disjoint deck skins."""
    f = AVAILABLE_FONTS
    train = [
        CardStyle("classic", _pick(f, "DejaVuSans.ttf", "FreeSans.ttf")),
        CardStyle("serif", _pick(f, "DejaVuSerif.ttf", "FreeSerif.ttf"),
                  face=(255, 253, 240), radius=0.06),
        CardStyle("bold", _pick(f, "DejaVuSans-Bold", "FreeSansBold"),
                  face=(244, 246, 250), red=(210, 20, 25), rank_scale=0.40),
        CardStyle("mono", _pick(f, "DejaVuSansMono", "FreeMono"),
                  face=(248, 248, 248), border=(120, 120, 120)),
        CardStyle("vector", _pick(f, "DejaVuSans.ttf"), pip_mode="vector",
                  face=(252, 250, 245)),
        CardStyle("fourcolour", _pick(f, "FreeSans.ttf", "DejaVuSans.ttf"),
                  four_colour=True, face=(255, 255, 255)),
        CardStyle("dim", _pick(f, "DejaVuSans.ttf"), face=(225, 222, 214),
                  black=(45, 45, 50), red=(160, 40, 45)),
        CardStyle("tight", _pick(f, "FreeSansBold", "DejaVuSans-Bold"),
                  rank_scale=0.30, suit_scale=0.20, radius=0.14),
        # A third font family widens the range of letterforms the network sees,
        # which is what it needs to read a deck drawn in a font it has never met.
        CardStyle("liberation", _pick(f, "LiberationSans-Regular", "FreeSans.ttf"),
                  face=(251, 249, 244)),
        CardStyle("liberation-serif", _pick(f, "LiberationSerif-Bold", "DejaVuSerif-Bold"),
                  face=(255, 252, 246), rank_scale=0.36, radius=0.05),
        CardStyle("liberation-mono-4c", _pick(f, "LiberationMono-Regular", "DejaVuSansMono"),
                  face=(252, 252, 255), four_colour=True, rank_scale=0.31),
        CardStyle("italic", _pick(f, "LiberationSans-Italic", "FreeSansOblique"),
                  face=(247, 245, 238), rank_scale=0.35),
    ]
    holdout = [
        CardStyle("holdout-serif-bold", _pick(f, "DejaVuSerif-Bold", "FreeSerifBold"),
                  face=(253, 251, 244), red=(175, 25, 35), rank_scale=0.37),
        CardStyle("holdout-vector-dark", _pick(f, "FreeSerif.ttf", "DejaVuSerif.ttf"),
                  pip_mode="vector", face=(238, 236, 228), border=(90, 90, 90),
                  radius=0.04),
        CardStyle("holdout-4c-mono", _pick(f, "FreeMonoBold", "DejaVuSansMono-Bold"),
                  four_colour=True, face=(250, 250, 255), rank_scale=0.32),
    ]
    return train, holdout


TRAIN_STYLES, HOLDOUT_STYLES = build_styles()


def suit_colour(style: CardStyle, suit_char: str) -> Tuple[int, int, int]:
    if style.four_colour:
        return FOUR_COLOUR[suit_char]
    return style.red if suit_char in RED_SUITS else style.black


def _draw_vector_pip(draw: ImageDraw.ImageDraw, suit: str, box: Tuple[float, float, float, float],
                     colour) -> None:
    """Draw a suit symbol from primitives instead of a font glyph."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx = x0 + w / 2
    if suit == "d":
        draw.polygon([(cx, y0), (x1, y0 + h / 2), (cx, y1), (x0, y0 + h / 2)], fill=colour)
    elif suit == "h":
        r = w / 4
        draw.ellipse([x0, y0, x0 + w / 2, y0 + h * 0.6], fill=colour)
        draw.ellipse([x0 + w / 2, y0, x1, y0 + h * 0.6], fill=colour)
        draw.polygon([(x0, y0 + h * 0.32), (x1, y0 + h * 0.32), (cx, y1)], fill=colour)
    elif suit == "s":
        draw.polygon([(cx, y0), (x1, y0 + h * 0.55), (x0, y0 + h * 0.55)], fill=colour)
        draw.ellipse([x0, y0 + h * 0.30, x0 + w / 2, y0 + h * 0.78], fill=colour)
        draw.ellipse([x0 + w / 2, y0 + h * 0.30, x1, y0 + h * 0.78], fill=colour)
        draw.rectangle([cx - w * 0.08, y0 + h * 0.6, cx + w * 0.08, y1], fill=colour)
    else:  # clubs
        r = w * 0.30
        draw.ellipse([cx - r, y0, cx + r, y0 + 2 * r], fill=colour)
        draw.ellipse([x0, y0 + h * 0.28, x0 + 2 * r, y0 + h * 0.28 + 2 * r], fill=colour)
        draw.ellipse([x1 - 2 * r, y0 + h * 0.28, x1, y0 + h * 0.28 + 2 * r], fill=colour)
        draw.rectangle([cx - w * 0.09, y0 + h * 0.55, cx + w * 0.09, y1], fill=colour)


def render_card(card: int, style: Optional[CardStyle] = None, size: Tuple[int, int] = (96, 136),
                rng: Optional[random.Random] = None) -> Image.Image:
    """Render one card as an RGB image."""
    style = style or TRAIN_STYLES[0]
    rng = rng or random.Random()
    w, h = size
    scale = 3  # supersample, then downscale for clean edges
    img = Image.new("RGB", (w * scale, h * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    W, H = w * scale, h * scale
    radius = max(2, int(style.radius * min(W, H)))

    draw.rounded_rectangle([0, 0, W - 1, H - 1], radius=radius, fill=style.face,
                           outline=style.border, width=max(1, scale))

    rank_char = RANK_CHARS[rank_of(card)]
    suit_char = SUIT_CHARS[suit_of(card)]
    colour = suit_colour(style, suit_char)

    rank_font = _font(style.font, int(H * style.rank_scale))
    suit_font = _font(style.font, int(H * style.suit_scale))
    margin = int(W * 0.09)

    draw.text((margin, margin), rank_char, font=rank_font, fill=colour)
    rank_box = draw.textbbox((margin, margin), rank_char, font=rank_font)
    pip_y = rank_box[3] + int(H * 0.01)
    pip_w = int(W * 0.20)
    if style.pip_mode == "vector":
        _draw_vector_pip(draw, suit_char, (margin, pip_y, margin + pip_w, pip_y + pip_w), colour)
    else:
        draw.text((margin, pip_y), SUIT_GLYPHS[suit_char], font=suit_font, fill=colour)

    if style.centre_pip and not style.corner_only:
        big = int(W * 0.42)
        cx0 = (W - big) / 2
        cy0 = (H - big) / 2 + H * 0.05
        if style.pip_mode == "vector":
            _draw_vector_pip(draw, suit_char, (cx0, cy0, cx0 + big, cy0 + big), colour)
        else:
            big_font = _font(style.font, int(H * 0.42))
            draw.text((W / 2, H * 0.58), SUIT_GLYPHS[suit_char], font=big_font,
                      fill=colour, anchor="mm")

    # Mirrored index in the bottom-right corner, as real cards have.
    corner = img.crop((0, 0, int(W * 0.34), int(H * 0.34))).rotate(180)
    img.paste(corner, (W - corner.width, H - corner.height))

    return img.resize((w, h), Image.LANCZOS)


def render_board(cards: Sequence[int], style: Optional[CardStyle] = None,
                 card_size: Tuple[int, int] = (96, 136), gap: int = 14,
                 felt: Tuple[int, int, int] = (16, 92, 60),
                 pad: int = 34, rng: Optional[random.Random] = None,
                 jitter: int = 0) -> Image.Image:
    """Lay several cards out on a felt background, like a table screenshot."""
    rng = rng or random.Random()
    w, h = card_size
    width = pad * 2 + len(cards) * w + (len(cards) - 1) * gap
    height = pad * 2 + h + jitter * 2
    table = Image.new("RGB", (width, height), felt)
    x = pad
    for card in cards:
        img = render_card(card, style, card_size, rng)
        dy = rng.randint(-jitter, jitter) if jitter else 0
        table.paste(img, (x, pad + jitter + dy))
        x += w + gap
    return table

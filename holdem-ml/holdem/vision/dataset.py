"""Build augmented training data from the synthetic renderer.

Augmentation is what makes a model trained on drawn cards work on photographed
or screenshotted ones: rotation, perspective-ish scaling, lighting and gamma
shifts, blur, sensor noise, JPEG-like blocking and partial occlusion.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ..cards import NUM_CARDS, rank_of, suit_of
from .render import CardStyle, TRAIN_STYLES, render_card

IMG_H, IMG_W = 48, 32
NUM_RANKS, NUM_SUITS = 13, 4

FELT_COLOURS = [
    (16, 92, 60), (10, 70, 50), (26, 60, 100), (90, 30, 40),
    (40, 40, 46), (18, 110, 88), (60, 55, 30),
]


def augment(img: Image.Image, rng: random.Random, strength: float = 1.0) -> Image.Image:
    """Apply a random photometric + geometric perturbation to a card image."""
    w, h = img.size
    pad = int(max(w, h) * 0.35)
    felt = FELT_COLOURS[rng.randrange(len(FELT_COLOURS))]
    felt = tuple(int(np.clip(c + rng.gauss(0, 18), 0, 255)) for c in felt)
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), felt)
    canvas.paste(img, (pad, pad))

    angle = rng.gauss(0, 6.0 * strength)
    canvas = canvas.rotate(angle, resample=Image.BILINEAR, fillcolor=felt)

    # Non-uniform scale stands in for camera perspective.
    sx = 1.0 + rng.gauss(0, 0.10 * strength)
    sy = 1.0 + rng.gauss(0, 0.10 * strength)
    nw, nh = max(8, int(canvas.width * sx)), max(8, int(canvas.height * sy))
    canvas = canvas.resize((nw, nh), Image.BILINEAR)

    # Crop back to the card with a random offset, so the card is not centred.
    cx, cy = nw / 2, nh / 2
    half_w = w * sx / 2 * (1.0 + rng.uniform(-0.06, 0.16))
    half_h = h * sy / 2 * (1.0 + rng.uniform(-0.06, 0.16))
    dx = rng.gauss(0, w * 0.05 * strength)
    dy = rng.gauss(0, h * 0.05 * strength)
    box = (cx - half_w + dx, cy - half_h + dy, cx + half_w + dx, cy + half_h + dy)
    canvas = canvas.crop(tuple(int(v) for v in box))
    if canvas.width < 4 or canvas.height < 4:  # pragma: no cover - degenerate crop
        canvas = img.copy()

    if rng.random() < 0.5 * strength:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.3)))
    canvas = ImageEnhance.Brightness(canvas).enhance(1.0 + rng.gauss(0, 0.16 * strength))
    canvas = ImageEnhance.Contrast(canvas).enhance(1.0 + rng.gauss(0, 0.16 * strength))
    canvas = ImageEnhance.Color(canvas).enhance(1.0 + rng.gauss(0, 0.18 * strength))

    if rng.random() < 0.25 * strength:
        # Occlusion: a chip, a finger, another card overlapping the corner.
        d = canvas.copy()
        from PIL import ImageDraw

        draw = ImageDraw.Draw(d)
        ow = rng.uniform(0.12, 0.32) * canvas.width
        oh = rng.uniform(0.12, 0.32) * canvas.height
        ox = rng.uniform(0, canvas.width - ow)
        oy = rng.uniform(0, canvas.height - oh)
        draw.rectangle([ox, oy, ox + ow, oy + oh],
                       fill=tuple(rng.randrange(256) for _ in range(3)))
        canvas = d

    canvas = canvas.resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    if rng.random() < 0.6 * strength:
        arr = arr + np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, rng.uniform(0.01, 0.06), arr.shape
        ).astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return arr.transpose(2, 0, 1)  # (C, H, W)


def make_batch(n: int, styles: Sequence[CardStyle], rng: random.Random,
               strength: float = 1.0, cache: Optional[dict] = None
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(images, rank labels, suit labels)`` for ``n`` random cards."""
    cache = {} if cache is None else cache
    X = np.zeros((n, 3, IMG_H, IMG_W), dtype=np.float32)
    yr = np.zeros(n, dtype=np.int64)
    ys = np.zeros(n, dtype=np.int64)
    for i in range(n):
        card = rng.randrange(NUM_CARDS)
        style = styles[rng.randrange(len(styles))]
        key = (card, style.name)
        base = cache.get(key)
        if base is None:
            base = render_card(card, style, (96, 136), rng)
            cache[key] = base
        X[i] = augment(base, rng, strength)
        yr[i] = rank_of(card)
        ys[i] = suit_of(card)
    return X, yr, ys


def clean_batch(cards: Sequence[int], style: CardStyle,
                rng: Optional[random.Random] = None) -> np.ndarray:
    """Un-augmented images, for checking the pipeline end to end."""
    rng = rng or random.Random(0)
    X = np.zeros((len(cards), 3, IMG_H, IMG_W), dtype=np.float32)
    for i, card in enumerate(cards):
        img = render_card(card, style, (96, 136), rng).resize((IMG_W, IMG_H), Image.BILINEAR)
        X[i] = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return X


def image_to_input(img: Image.Image) -> np.ndarray:
    """Prepare an arbitrary PIL image (a real crop) for the network."""
    arr = np.asarray(img.convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR),
                     dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None, :, :, :]

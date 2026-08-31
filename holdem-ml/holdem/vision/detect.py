"""Find the cards in a table image and read them.

Detection is deliberately classical rather than a second neural net: card faces
are bright, low-saturation rectangles, which a threshold plus connected
components finds reliably and in milliseconds.  The CNN then does the part that
actually needs learning — deciding *which* card each crop is.

Connected components are computed run-length-wise (a handful of runs per row
joined by a union-find) rather than pixel-by-pixel, so it stays fast in pure
Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from ..cards import card_str
from .cardnet import CardNet, load_cardnet
from .dataset import IMG_H, IMG_W


@dataclass
class Detection:
    card: int
    confidence: float
    box: Tuple[int, int, int, int]  # left, top, right, bottom

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"{card_str(self.card)}@{self.confidence:.2f}"


def card_mask(img: Image.Image, brightness: float = 0.62,
              saturation: float = 0.38) -> np.ndarray:
    """Boolean mask of pixels that look like a printed card face."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return (mx > brightness) & (sat < saturation)


def connected_components(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Bounding boxes of 4-connected True regions, as ``(l, t, r, b)``."""
    h, w = mask.shape
    parent: List[int] = [0]

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    prev_runs: List[Tuple[int, int, int]] = []
    all_runs: List[Tuple[int, int, int, int]] = []  # row, start, end, label

    for y in range(h):
        row = mask[y]
        if not row.any():
            prev_runs = []
            continue
        padded = np.concatenate(([False], row, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        starts, ends = edges[0::2], edges[1::2]
        runs: List[Tuple[int, int, int]] = []
        for s, e in zip(starts, ends):
            label = 0
            for ps, pe, plabel in prev_runs:
                if ps < e and s < pe:      # overlapping run on the row above
                    if label == 0:
                        label = plabel
                    else:
                        union(label, plabel)
            if label == 0:
                label = len(parent)
                parent.append(label)
            runs.append((int(s), int(e), label))
            all_runs.append((y, int(s), int(e), label))
        prev_runs = runs

    boxes: dict = {}
    for y, s, e, label in all_runs:
        root = find(label)
        box = boxes.get(root)
        if box is None:
            boxes[root] = [s, y, e, y + 1]
        else:
            box[0] = min(box[0], s)
            box[1] = min(box[1], y)
            box[2] = max(box[2], e)
            box[3] = max(box[3], y + 1)
    return [tuple(b) for b in boxes.values()]


def find_cards(img: Image.Image, min_area_frac: float = 0.002,
               aspect_range: Tuple[float, float] = (1.05, 2.0),
               brightness: float = 0.62) -> List[Tuple[int, int, int, int]]:
    """Bounding boxes of card-shaped bright regions, left to right."""
    mask = card_mask(img, brightness=brightness)
    total = mask.size
    boxes = []
    for (l, t, r, b) in connected_components(mask):
        w, h = r - l, b - t
        if w < 6 or h < 8:
            continue
        if w * h < min_area_frac * total:
            continue
        aspect = h / w
        if not aspect_range[0] <= aspect <= aspect_range[1]:
            continue
        boxes.append((l, t, r, b))

    # Enclosed glyph counters (the hole in a Q, the loop of a 9) form their own
    # bright regions inside a card face — drop anything nested in a bigger box.
    boxes.sort(key=lambda box: -(box[2] - box[0]) * (box[3] - box[1]))
    kept: List[Tuple[int, int, int, int]] = []
    for box in boxes:
        if any(_overlap_fraction(box, big) > 0.6 for big in kept):
            continue
        kept.append(box)
    kept.sort(key=lambda box: box[0])
    return kept


def _overlap_fraction(inner, outer) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``."""
    l = max(inner[0], outer[0])
    t = max(inner[1], outer[1])
    r = min(inner[2], outer[2])
    b = min(inner[3], outer[3])
    if r <= l or b <= t:
        return 0.0
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (r - l) * (b - t) / max(1, area)


class CardReader:
    """Read cards out of images with a trained :class:`CardNet`."""

    def __init__(self, net: Optional[CardNet] = None, path: Optional[str] = None):
        self.net = net or load_cardnet(path) if (net or path) else load_cardnet()

    def read_crop(self, img: Image.Image) -> Tuple[int, float]:
        arr = np.asarray(img.convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR),
                         dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        cards, conf = self.net.predict(arr)
        return int(cards[0]), float(conf[0])

    def read_table(self, img: Image.Image, min_confidence: float = 0.0,
                   **kwargs) -> List[Detection]:
        boxes = find_cards(img, **kwargs)
        if not boxes:
            return []
        batch = np.zeros((len(boxes), 3, IMG_H, IMG_W), dtype=np.float32)
        for i, box in enumerate(boxes):
            crop = img.convert("RGB").crop(box).resize((IMG_W, IMG_H), Image.BILINEAR)
            batch[i] = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0
        cards, conf = self.net.predict(batch)
        return [Detection(int(c), float(p), box)
                for c, p, box in zip(cards, conf, boxes) if p >= min_confidence]

    def read_file(self, path: str, **kwargs) -> List[Detection]:
        return self.read_table(Image.open(path), **kwargs)

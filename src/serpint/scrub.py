"""Blur the parts of a results-page screenshot that give the location away.

Works on the image alone, so it applies to any screenshot someone wants to share,
not only the ones rendered here. OCR finds the text lines, rules pick the lines and
regions to blur, and those regions are pixelated and blurred until unreadable.

Levels (each includes the ones before it):

- ``names``: lines that name a place (fuzzy, to survive OCR misreads), Google's
  location lines ("From your IP address", "Results for…", "Choose area"), PIN codes
  and landline numbers (their STD codes are regional).
- ``sections``: also the whole blocks the leak map found leaking beyond names:
  related searches, the local-results pack with its map, and top stories / local news.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from serpint.client import atomic_write_bytes
from serpint.config import Settings

LEVELS = ("none", "names", "sections")

# Fuzzy matching: an OCR'd word sequence matches a place name of at least
# FUZZY_MIN_LEN letters when difflib's ratio reaches FUZZY_RATIO. Shorter names
# ("Goa", "Pune", "Kota") must match exactly, as whole words.
FUZZY_MIN_LEN = 6
FUZZY_RATIO = 0.86
# Names this long are also found glued inside a word ("Annasalai-Chennai" OCR'd as one).
GLUED_MIN_LEN = 7

LOCATION_LINE = re.compile(
    r"from your ip address|update location|results for\b|choose area|"
    r"based on your (?:past )?activity|your places",
    re.IGNORECASE,
)
PIN_CODE = re.compile(r"(?<![\d,.])[1-9]\d{2} ?\d{3}(?![\d,.])")
LANDLINE = re.compile(r"(?<!\d)0\d{2,4}[ -]?\d{6,8}(?!\d)")


@dataclass(frozen=True)
class Section:
    """A page block blurred whole at the ``sections`` level.

    Patterns match a whole line with spacing removed and lower-cased, because OCR
    often glues words ("Peoplealsosearchfor").
    """

    name: str
    heading: re.Pattern[str]
    end: re.Pattern[str] | None  # a line that closes the block (blurred with it)
    max_gap: int  # px of vertical gap after which the block is over


SECTIONS = (
    Section(
        "related_searches",
        re.compile(r"peoplealsosearch(?:for)?|relatedsearches"),
        re.compile(r"next|\d{4,}"),  # the pagination row closes it
        110,
    ),
    Section(
        "local_results",
        re.compile(
            r"places|businesses|restaurants|hotels|(?:in)?(?:shops|stores)nearby|nearbystores"
        ),
        re.compile(r"more(?:places|businesses|restaurants|hotels|shops|stores)"),
        260,  # the map has stretches with no text in them
    ),
    Section(
        "news",
        re.compile(r"topstories|localnews|newsnearyou"),
        re.compile(r"morenews|viewall"),
        110,
    ),
)


def squash(text: str) -> str:
    """Lower-case and drop whitespace, for matching headings OCR may have glued."""
    return re.sub(r"\s+", "", text.lower())


@dataclass(frozen=True)
class Line:
    """An OCR'd text line: its box (x0, y0, x1, y1) in pixels, text and confidence."""

    box: tuple[int, int, int, int]
    text: str
    conf: float


@dataclass(frozen=True)
class Redaction:
    """A region to blur and why."""

    box: tuple[int, int, int, int]
    reason: str  # place_name | location_line | pin_code | landline | section:<name>
    detail: str  # what matched (the place, or the section)


# --------------------------------------------------------------------- OCR


@cache
def _engine() -> Any:
    from rapidocr_onnxruntime import RapidOCR

    # Full resolution (small text matters), no angle classifier (screenshots are
    # upright), one thread so several processes can OCR side by side.
    return RapidOCR(
        max_side_len=4000, use_cls=False, intra_op_num_threads=1, inter_op_num_threads=1
    )


def ocr(image: Any) -> list[Line]:
    """OCR an image (path, PIL image or array) into lines, top to bottom."""
    import numpy as np

    if not isinstance(image, (str, Path)):
        image = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else image
    result, _ = _engine()(str(image) if isinstance(image, Path) else image)
    lines = []
    for quad, text, conf in result or []:
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        lines.append(
            Line((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))), text, float(conf))
        )
    return sorted(lines, key=lambda ln: (ln.box[1], ln.box[0]))


def ocr_cached(png: Path, cache_path: Path) -> list[Line]:
    """OCR ``png`` once and keep the result next to it as JSON."""
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return [Line(tuple(d["box"]), d["text"], d["conf"]) for d in data]
    lines = ocr(png)
    payload = [{"box": list(ln.box), "text": ln.text, "conf": ln.conf} for ln in lines]
    atomic_write_bytes(cache_path, (json.dumps(payload, ensure_ascii=False) + "\n").encode())
    return lines


# --------------------------------------------------------------- gazetteer


def _letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z]+", s.lower())


@dataclass(frozen=True)
class Gazetteer:
    """Place names to blur, matched exactly, fuzzily or glued inside longer words."""

    names: tuple[str, ...]

    @classmethod
    def load(cls, settings: Settings, places_yaml: Path | None = None) -> Gazetteer:
        """Combine ``config/places.yaml`` with every configured city, alias and state."""
        path = places_yaml or settings.root / "config" / "places.yaml"
        with path.open(encoding="utf-8") as f:
            extra = yaml.safe_load(f)
        names = {n for c in settings.cities for n in (c.name, c.state, *c.aliases)}
        names.update(extra.get("states", []), extra.get("cities", []))
        return cls(tuple(sorted(names)))

    def find(self, text: str) -> str | None:
        """Return the place named in ``text``, if any."""
        words = _words(text)
        glued = _letters(text)
        for name in self.names:
            target = _words(name)
            n = len(target)
            joined = " ".join(target)
            letters = "".join(target)
            for i in range(len(words) - n + 1):
                window = " ".join(words[i : i + n])
                if window == joined:
                    return name
                if (
                    len(letters) >= FUZZY_MIN_LEN
                    and difflib.SequenceMatcher(None, window, joined).ratio() >= FUZZY_RATIO
                ):
                    return name
            if len(letters) >= GLUED_MIN_LEN and letters in glued:
                return name
        return None


# ------------------------------------------------------------------- rules


def _union(boxes: Iterable[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    bs = list(boxes)
    return (
        min(b[0] for b in bs),
        min(b[1] for b in bs),
        max(b[2] for b in bs),
        max(b[3] for b in bs),
    )


def _column_right(lines: list[Line], left: int) -> int:
    """Right edge of the text column that starts at ``left`` (for blurring images too)."""
    rights = [ln.box[2] for ln in lines if abs(ln.box[0] - left) <= 30]
    return max(rights) if rights else left


# The URL breadcrumb under a result's site name ("https://www.croma.com › unboxed").
BREADCRUMB = re.compile(r"https?://|www\.|\s›\s")
OTHER_HEADINGS = re.compile(r"peoplealsoask|thingstoknow|discussionsandforums|videos")


def _starts_block(line: Line) -> bool:
    """Whether a line is the heading of another block, which ends the current one."""
    s = squash(line.text)
    return any(sec.heading.fullmatch(s) for sec in SECTIONS) or bool(OTHER_HEADINGS.fullmatch(s))


def _alone_on_row(line: Line, lines: list[Line]) -> bool:
    """A section heading has its row to itself; a navigation tab ("News") doesn't."""
    mid = (line.box[1] + line.box[3]) / 2
    return not any(other is not line and other.box[1] <= mid <= other.box[3] for other in lines)


def find_sections(lines: list[Line]) -> list[Redaction]:
    """Find whole blocks (heading to end marker or gap) to blur."""
    out = []
    for i, ln in enumerate(lines):
        for sec in SECTIONS:
            if not sec.heading.fullmatch(squash(ln.text)) or not _alone_on_row(ln, lines):
                continue
            left = ln.box[0]
            members = [ln]
            bottom = ln.box[3]
            for nxt in lines[i + 1 :]:
                if nxt.box[1] - bottom > sec.max_gap or _starts_block(nxt):
                    break
                if nxt.box[0] < left - 40:  # left the column
                    continue
                if BREADCRUMB.search(nxt.text):
                    # An organic result starts: its site name sits just above the URL.
                    if len(members) > 1 and nxt.box[1] - members[-1].box[3] <= 30:
                        members.pop()
                    break
                members.append(nxt)
                bottom = max(bottom, nxt.box[3])
                if sec.end is not None and sec.end.fullmatch(squash(nxt.text)):
                    break
            x0, y0, x1, y1 = _union(m.box for m in members)
            x1 = max(x1, _column_right(lines, left))
            out.append(Redaction((x0, y0, x1, y1), f"section:{sec.name}", ln.text.strip()))
            break
    return out


def plan_redactions(lines: list[Line], gazetteer: Gazetteer, level: str) -> list[Redaction]:
    """Return what to blur at ``level`` (see the module docstring)."""
    if level not in LEVELS:
        raise ValueError(f"Unknown scrub level {level!r}; expected one of {LEVELS}")
    if level == "none":
        return []
    out = []
    for ln in lines:
        if LOCATION_LINE.search(ln.text):
            out.append(Redaction(ln.box, "location_line", ln.text))
        elif (place := gazetteer.find(ln.text)) is not None:
            out.append(Redaction(ln.box, "place_name", place))
        elif m := PIN_CODE.search(ln.text):
            out.append(Redaction(ln.box, "pin_code", m.group(0)))
        elif m := LANDLINE.search(ln.text):
            out.append(Redaction(ln.box, "landline", m.group(0)))
    if level == "sections":
        out += find_sections(lines)
    return out


def overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Return whether two (x0, y0, x1, y1) boxes overlap."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def kept_lines(lines: list[Line], redactions: list[Redaction]) -> list[Line]:
    """Return the lines that stay readable after blurring."""
    return [ln for ln in lines if not any(overlaps(ln.box, r.box) for r in redactions)]


# ------------------------------------------------------------------- image


PAD = 4


SECTION_FILL = (232, 234, 237)  # flat light grey


def apply(image: Any, redactions: list[Redaction], block: int = 14) -> Any:
    """Return a copy of a PIL image with every redaction made unreadable.

    Text lines are pixelated and blurred, which keeps the page's look. Whole sections
    are filled flat: pixelating a map still shows a coastline or a river.
    """
    from PIL import ImageDraw, ImageFilter

    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for r in redactions:
        x0, y0, x1, y1 = r.box
        box = (
            max(0, x0 - PAD),
            max(0, y0 - PAD),
            min(out.width, x1 + PAD),
            min(out.height, y1 + PAD),
        )
        w, h = box[2] - box[0], box[3] - box[1]
        if w <= 0 or h <= 0:
            continue
        if r.reason.startswith("section:"):
            draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=SECTION_FILL)
            continue
        region = out.crop(box)
        small = region.resize((max(1, w // block), max(1, h // block)))
        region = small.resize((w, h)).filter(ImageFilter.GaussianBlur(block / 2))
        out.paste(region, box)
    return out


def blurred_fraction(size: tuple[int, int], redactions: list[Redaction]) -> float:
    """Share of the image area covered by redactions (overlaps counted once)."""
    import numpy as np

    w, h = size
    mask = np.zeros((h, w), dtype=bool)
    for r in redactions:
        x0, y0, x1, y1 = r.box
        mask[max(0, y0 - PAD) : min(h, y1 + PAD), max(0, x0 - PAD) : min(w, x1 + PAD)] = True
    return float(mask.mean())


def scrub_file(
    src: Path, dst: Path, settings: Settings, level: str = "sections"
) -> list[Redaction]:
    """Scrub one screenshot file and write the result; return what was blurred."""
    from io import BytesIO

    from PIL import Image

    with Image.open(src) as im:
        lines = ocr(im)
        redactions = plan_redactions(lines, Gazetteer.load(settings), level)
        buf = BytesIO()
        apply(im, redactions).save(buf, format="PNG")
    atomic_write_bytes(dst, buf.getvalue())
    return redactions

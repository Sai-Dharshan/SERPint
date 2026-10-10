"""Measure how much the scrubber stops a city being read off a screenshot, and at what cost.

Threat model: someone trains a model on ordinary (unscrubbed) screenshots and points it
at a shared one. So for every query, the attackers below are trained on the OCR text of
the *other* queries' unscrubbed screenshots and tested on the held-out query's
screenshots at each scrub level:

- ``mentions``: the city the readable text names most.
- ``text``: TF-IDF + logistic regression over the readable text.

Cost is the share of the image blurred and the share of text lines lost. Blurring is
checked, too: a sample of scrubbed images is OCR'd again, and any text read back inside
a blurred region is counted.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from joblib import Parallel, delayed

from serpint import scrub
from serpint.config import TIERS, Settings
from serpint.models import (
    MentionModel,
    Prediction,
    Summary,
    TextModel,
    leave_one_query_out,
    summarize,
)
from serpint.parse import Item, Page
from serpint.screenshots import Shot, screenshot_dir


def ocr_path(settings: Settings, shot: Shot) -> Path:
    """Return where the OCR result of a screenshot is cached."""
    return (
        settings.data_dir
        / "processed"
        / "ocr"
        / shot.query_id
        / shot.city_id
        / f"{shot.timestamp}.json"
    )


def ocr_all(shots: list[Shot], settings: Settings, n_jobs: int = -1) -> list[list[scrub.Line]]:
    """OCR every screenshot (cached; parallel across processes)."""
    return Parallel(n_jobs=n_jobs)(
        delayed(scrub.ocr_cached)(s.path, ocr_path(settings, s)) for s in shots
    )


def _page(shot: Shot, tier: str, lines: list[scrub.Line]) -> Page:
    return Page(
        shot.query_id, shot.city_id, tier, shot.timestamp, {"ocr": [Item(ln.text) for ln in lines]}
    )


@dataclass(frozen=True)
class LevelResult:
    """Attack results and cost at one scrub level."""

    level: str
    preds: dict[str, list[Prediction]]  # by attacker
    blurred: list[float]  # per page, share of image area
    lines_lost: list[float]  # per page, share of OCR lines blurred
    reasons: dict[str, int]  # redactions by reason


# A re-OCR'd line counts as read back from under the blur when at least this much of it
# lies inside a blurred region; less is neighbouring text touching a region's edge.
READBACK_INSIDE = 0.5


@dataclass(frozen=True)
class Readback:
    """Text OCR could still read inside blurred regions of re-OCR'd scrubbed images."""

    images: int
    blurred_regions: int
    readable: list[tuple[str, str]]  # (cell, text) mostly inside a blurred region
    edge: int  # lines that only touch a blurred region's edge


def evaluate(
    shots: list[Shot], lines: list[list[scrub.Line]], settings: Settings
) -> list[LevelResult]:
    """Attack every scrub level; attackers always train on unscrubbed pages."""
    tier = {q.id: q.tier for q in settings.queries}
    gaz = scrub.Gazetteer.load(settings)
    clean = [_page(s, tier[s.query_id], ls) for s, ls in zip(shots, lines, strict=True)]
    candidates = sorted({s.city_id for s in shots})
    out = []
    for level in scrub.LEVELS:
        test, blurred, lost = [], [], []
        reasons: dict[str, int] = {}
        for s, ls in zip(shots, lines, strict=True):
            red = scrub.plan_redactions(ls, gaz, level)
            kept = scrub.kept_lines(ls, red)
            test.append(_page(s, tier[s.query_id], kept))
            blurred.append(scrub.blurred_fraction((s.width, s.height), red))
            lost.append(1 - len(kept) / len(ls) if ls else 0.0)
            for r in red:
                reasons[r.reason] = reasons.get(r.reason, 0) + 1
        attackers = {"mentions": MentionModel, "text": lambda: TextModel(seed=settings.seed)}
        preds = {
            name: leave_one_query_out(clean, settings, make, candidates, test=test)
            for name, make in attackers.items()
        }
        out.append(LevelResult(level, preds, blurred, lost, reasons))
    return out


def _readback_one(
    shot: Shot, lines: list[scrub.Line], settings: Settings, level: str, out_dir: Path
) -> tuple[int, list[tuple[str, str]], int]:
    from PIL import Image

    gaz = scrub.Gazetteer.load(settings)
    red = scrub.plan_redactions(lines, gaz, level)
    with Image.open(shot.path) as im:
        scrubbed = scrub.apply(im, red)
    dst = out_dir / shot.query_id / f"{shot.city_id}.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    scrubbed.save(dst)
    again = scrub.ocr(scrubbed)
    found, edge = [], 0
    for ln in again:
        if sum(ch.isalpha() for ch in ln.text) < 3:
            continue
        inside = scrub.inside_fraction(ln.box, red)
        if inside >= READBACK_INSIDE:
            found.append((f"{shot.query_id}@{shot.city_id}", ln.text))
        elif inside > 0:
            edge += 1
    return len(red), found, edge


def readback(
    shots: list[Shot],
    lines: list[list[scrub.Line]],
    settings: Settings,
    level: str = "sections",
    n_jobs: int = -1,
) -> Readback:
    """Scrub one screenshot per query, OCR it again, and report any text read back.

    The scrubbed images are kept in ``data/screenshots_scrubbed/<level>/`` to look at.
    """
    first: dict[str, int] = {}
    for i, s in enumerate(shots):
        first.setdefault(s.query_id, i)
    out_dir = settings.data_dir / "screenshots_scrubbed" / level
    results = Parallel(n_jobs=n_jobs)(
        delayed(_readback_one)(shots[i], lines[i], settings, level, out_dir) for i in first.values()
    )
    return Readback(
        images=len(first),
        blurred_regions=sum(n for n, _, _ in results),
        readable=[x for _, found, _ in results for x in found],
        edge=sum(e for _, _, e in results),
    )


# ------------------------------------------------------------------ report


def _cell(s: Summary) -> str:
    return "-" if not s.n else f"{s.accuracy:.0%} / {s.mean_score:,.0f}"


def scrub_report(
    results: list[LevelResult], rb: Readback | None, settings: Settings, n: int
) -> str:
    """Render the evaluation as Markdown."""
    tiers = [t for t in TIERS if any(p.tier == t for p in results[0].preds["text"])]
    n_cities = len({p.city_id for p in results[0].preds["text"]})
    out = [
        f"# Scrubber evaluation ({n} screenshots)",
        "",
        "Generated by `serpint scrub-eval`. Screenshots are full pages rendered offline",
        "from the saved HTML, with Google's location footer reconstructed (see",
        "`serpint screenshots`). Attackers are trained leave-one-query-out on the OCR'd",
        "text of *unscrubbed* screenshots and tested on the held-out query's screenshots",
        "at each level. Cells are city accuracy / mean GeoGuessr-style score; chance is "
        f"1/{n_cities} = {1 / n_cities:.0%}.",
        "",
        "- `none`: the screenshot as rendered.",
        "- `names`: place names, location lines, PIN codes and landlines blurred.",
        "- `sections`: also related searches, local results (with the map) and news.",
        "",
        "## Leak and cost by level",
        "",
        "| level | mentions | text | image blurred (mean) | lines lost (mean) |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        out.append(
            f"| `{r.level}` | {_cell(summarize(r.preds['mentions'], settings))} | "
            f"{_cell(summarize(r.preds['text'], settings))} | "
            f"{statistics.fmean(r.blurred):.1%} | {statistics.fmean(r.lines_lost):.1%} |"
        )

    out += ["", "## By tier (text attacker)", ""]
    out += ["| tier | " + " | ".join(f"`{r.level}`" for r in results) + " |"]
    out += ["|---|" + "---|" * len(results)]
    for t in tiers:
        cells = [
            _cell(summarize([p for p in r.preds["text"] if p.tier == t], settings)) for r in results
        ]
        out.append(f"| {t} | " + " | ".join(cells) + " |")

    out += ["", "## By query (text attacker)", ""]
    out += ["| query | tier | " + " | ".join(f"`{r.level}`" for r in results) + " |"]
    out += ["|---|---|" + "---|" * len(results)]
    for q in settings.queries:
        cells = [
            _cell(summarize([p for p in r.preds["text"] if p.query_id == q.id], settings))
            for r in results
        ]
        if any(c != "-" for c in cells):
            out.append(f"| {q.id} | {q.tier} | " + " | ".join(cells) + " |")

    out += ["", "## What was blurred", "", "Redactions by rule, over all screenshots.", ""]
    out += ["| rule | " + " | ".join(f"`{r.level}`" for r in results[1:]) + " |"]
    out += ["|---|" + "---|" * (len(results) - 1)]
    rules = sorted({k for r in results for k in r.reasons})
    for k in rules:
        out.append(f"| {k} | " + " | ".join(str(r.reasons.get(k, 0)) for r in results[1:]) + " |")

    if rb is not None:
        out += [
            "",
            "## Is the blur unreadable?",
            "",
            f"{rb.images} scrubbed screenshots (one per query, `sections` level) were OCR'd",
            f"again. Across {rb.blurred_regions} blurred regions, OCR read "
            f"{len(rb.readable)} line(s) of three or more letters lying at least "
            f"{READBACK_INSIDE:.0%} inside one" + (":" if rb.readable else "."),
        ]
        if rb.readable:
            out += [""] + [f"- {cell}: `{text}`" for cell, text in rb.readable]
        out += [
            "",
            f"Another {rb.edge} readable line(s) only touch a blurred region's edge: "
            "neighbouring text, such as a site name above a blurred line.",
        ]
    return "\n".join(out) + "\n"


def load_shots(settings: Settings, plan: str) -> list[Shot]:
    """Return the rendered screenshots of a plan, in plan order."""
    from serpint.collect import plan_cells

    shots = []
    for c in plan_cells(settings, plan):
        d = screenshot_dir(settings) / c.query.id / c.city.id
        metas = sorted(d.glob("*.json"))
        if metas:
            meta = json.loads(metas[-1].read_text())
            shots.append(Shot(**{**meta, "path": metas[-1].with_suffix(".png")}))
    return shots

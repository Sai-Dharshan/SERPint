"""Leak map: how much each page element reveals about the city.

For every element (``organic_results``, ``related_searches``, ``local_results``…) the
text model from ``models`` is evaluated leave-one-query-out in two ablations:

- **alone**: it sees only that element, on the pages that have it. How much does the
  element give away by itself?
- **without**: it sees every element but that one, on all pages. How much accuracy does
  removing the element take away, on the pages that had it? That is what blurring it
  in a screenshot would buy, given everything else stays visible.

Both run with and without city and state names masked, which separates elements that
leak by naming the place (a text scrubber can redact the name) from elements that leak
implicitly (only blurring the whole element helps).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace

import numpy as np
from joblib import Parallel, delayed

from serpint.config import Settings
from serpint.models import (
    Geo,
    Prediction,
    Summary,
    TextModel,
    document,
    leave_one_query_out,
    mask_pattern,
    summarize,
)
from serpint.parse import Page, mention_patterns

VARIANTS = ("text", "masked")


def keep_only(page: Page, element: str) -> Page:
    """Return the page with every element but ``element`` removed."""
    return replace(page, elements={element: page.elements[element]})


def drop(page: Page, element: str) -> Page:
    """Return the page with ``element`` removed."""
    return replace(page, elements={k: v for k, v in page.elements.items() if k != element})


def _loqo(
    pages: list[Page], settings: Settings, masked: bool, candidates: list[str]
) -> list[Prediction] | None:
    """Run the text model leave-one-query-out; ``None`` if it can't be trained.

    Module-level (not a closure) so joblib can ship it to worker processes.
    """
    mask = mask_pattern(settings.cities) if masked else None
    try:
        return leave_one_query_out(
            pages, settings, lambda: TextModel(mask, seed=settings.seed), candidates
        )
    except ValueError:  # empty vocabulary, or a single city in some training fold
        return None


@dataclass(frozen=True)
class Ablation:
    """One variant's numbers for one element, all over the pages that have it."""

    alone: Summary | None  # None: the element occurs in a single query, so no LOQO
    full: Summary  # the full model, for reference
    without: Summary


@dataclass(frozen=True)
class ElementLeak:
    """What one element reveals."""

    element: str
    pages: int
    queries: int
    names_own_city: float  # share of its pages where its text names the page's city
    ablations: dict[str, Ablation]  # by variant


def _on(preds: list[Prediction], cells: set[tuple[str, str]]) -> list[Prediction]:
    return [p for p in preds if (p.query_id, p.city_id) in cells]


def leak_map(
    pages: list[Page], settings: Settings, n_jobs: int = -1
) -> tuple[list[ElementLeak], dict[str, Summary]]:
    """Ablate every element and return the leaks, plus the full models' summaries.

    Args:
        pages: Parsed pages.
        settings: Cities and score scale.
        n_jobs: Worker processes for joblib (-1: all cores).
    """
    candidates = sorted({p.city_id for p in pages})
    present: dict[str, list[Page]] = defaultdict(list)
    for p in pages:
        for name in p.elements:
            present[name].append(p)
    elements = sorted(present, key=lambda e: (-len(present[e]), e))

    jobs: list[tuple[str, str, bool, list[Page]]] = []
    for v in VARIANTS:
        masked = v == "masked"
        jobs.append(("full", "", masked, pages))
        for e in elements:
            subset = [keep_only(p, e) for p in present[e]]
            if len({p.query_id for p in subset}) >= 2:
                jobs.append(("alone", e, masked, subset))
            jobs.append(("without", e, masked, [drop(p, e) for p in pages]))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_loqo)(ps, settings, masked, candidates) for _, _, masked, ps in jobs
    )
    runs = {
        (kind, e, "masked" if masked else "text"): preds
        for (kind, e, masked, _), preds in zip(jobs, results, strict=True)
    }

    patterns = mention_patterns(settings.cities)
    leaks = []
    for e in elements:
        cells = {(p.query_id, p.city_id) for p in present[e]}
        named = sum(
            bool(patterns[p.city_id].city.search("\n".join(i.text for i in p.elements[e])))
            for p in present[e]
        )
        ablations = {}
        for v in VARIANTS:
            alone = runs.get(("alone", e, v))
            full = runs[("full", "", v)]
            without = runs[("without", e, v)]
            ablations[v] = Ablation(
                alone=None if alone is None else summarize(alone, settings),
                full=summarize(_on(full, cells), settings),
                without=summarize(_on(without, cells), settings)
                if without is not None
                else summarize([], settings),
            )
        leaks.append(
            ElementLeak(
                element=e,
                pages=len(present[e]),
                queries=len({p.query_id for p in present[e]}),
                names_own_city=named / len(present[e]),
                ablations=ablations,
            )
        )
    full = {v: summarize(runs[("full", "", v)], settings) for v in VARIANTS}
    return leaks, full


# ----------------------------------------------------------- top features


@dataclass(frozen=True)
class Feature:
    """A token the masked model weighs heavily for a city, and where it shows up."""

    token: str
    weight: float
    element: str  # the element it occurs in most often on that city's pages
    pages: int  # how many of that city's pages contain it


def top_features(pages: list[Page], settings: Settings, k: int = 6) -> dict[str, list[Feature]]:
    """Fit the masked model on every page and list each city's strongest tokens.

    Descriptive only: trained on everything, so it shows what the model keys on, not
    how well that generalizes (the leak map does that).
    """
    mask = mask_pattern(settings.cities)
    geo = Geo.of(
        (c for c in settings.cities if c.id in {p.city_id for p in pages}),
        settings.geoguessr_scale_km,
    )
    model = TextModel(mask, seed=settings.seed)
    model.fit(pages, geo)
    vocab = model.vectorizer.get_feature_names_out()
    analyze = model.vectorizer.build_analyzer()
    out: dict[str, list[Feature]] = {}
    for row, city in zip(model.clf.coef_, model.clf.classes_, strict=True):
        city_pages = [p for p in pages if p.city_id == city]
        where: dict[str, Counter[str]] = defaultdict(Counter)
        page_hits: Counter[str] = Counter()
        for p in city_pages:
            seen: set[str] = set()
            for e in p.elements:
                for tok in set(analyze(document(p, mask, [e]))):
                    where[tok][e] += 1
                    seen.add(tok)
            page_hits.update(seen)
        feats = []
        for j in np.argsort(row)[::-1]:
            tok = str(vocab[j])
            if row[j] <= 0 or len(feats) == k:
                break
            if tok in where:
                feats.append(
                    Feature(tok, float(row[j]), where[tok].most_common(1)[0][0], page_hits[tok])
                )
        out[str(city)] = feats
    return out


# ------------------------------------------------------------------ report


def _acc(s: Summary | None) -> str:
    if s is None or not s.n:
        return "n/a"
    return f"{s.accuracy:.0%} / {s.mean_score:,.0f}"


def _delta(a: Ablation) -> str:
    if not a.without.n:
        return "n/a"
    d = a.without.accuracy - a.full.accuracy
    return f"{a.without.accuracy:.0%} ({d:+.0%})"


def leak_report(
    leaks: list[ElementLeak],
    full: dict[str, Summary],
    features: dict[str, list[Feature]],
    settings: Settings,
    n_pages: int,
) -> str:
    """Render the leak map as Markdown."""
    n_cities = len(features) or len({c.id for c in settings.cities})

    def alone_key(x: ElementLeak) -> tuple[int, float]:
        a = x.ablations["text"].alone
        return (0, -a.accuracy) if a is not None and a.n else (1, -x.pages)

    out = [
        f"# Leak map ({n_pages} pages)",
        "",
        "Generated by `serpint leaks`. Each element is ablated in the leave-one-query-out",
        "text model from `serpint model` (accuracy / mean GeoGuessr-style score). Every",
        "number is over the pages that have the element. Chance accuracy is "
        f"1/{n_cities} = {1 / n_cities:.0%}.",
        "",
        "- **alone**: the model sees only this element. What the element gives away by",
        "  itself. `n/a`: the element occurs in a single query, so there is no other query",
        "  to learn it from.",
        "- **without**: the model sees everything but this element; in brackets, the change",
        "  from the full model on the same pages. What blurring the element would buy.",
        "- **masked**: every configured city and state name blanked first. Leaks that",
        "  survive masking can't be fixed by redacting names; the element has to go.",
        "- **names city**: share of its pages where the element's text names the page's",
        "  own city outright.",
        "",
        "Full model on all pages: " + "; ".join(f"{v} {_acc(full[v])}" for v in VARIANTS) + ".",
        "",
        "| element | pages | queries | names city | alone | alone, masked | without | "
        "without, masked |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for x in sorted(leaks, key=alone_key):
        t, m = x.ablations["text"], x.ablations["masked"]
        out.append(
            f"| `{x.element}` | {x.pages} | {x.queries} | {x.names_own_city:.0%} | "
            f"{_acc(t.alone)} | {_acc(m.alone)} | {_delta(t)} | {_delta(m)} |"
        )

    out += [
        "",
        "## What the masked model keys on",
        "",
        "The strongest tokens per city for the masked model fitted on every page (names",
        "blanked as `place`), with the element each occurs in most and on how many of",
        "that city's pages. Descriptive: fitted on all pages, so it shows what the model",
        "uses, not how well it carries over.",
        "",
        "| city | tokens (element, pages) |",
        "|---|---|",
    ]
    for c in settings.cities:
        feats = features.get(c.id)
        if feats:
            cells = ", ".join(f"`{f.token}` ({f.element}, {f.pages})" for f in feats)
            out.append(f"| {c.id} | {cells} |")
    return "\n".join(out) + "\n"

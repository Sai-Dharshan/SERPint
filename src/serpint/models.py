"""Predict the city from a page's visible text, and measure how far the signal carries.

There is one page per (query, city), so a model can't be trained and tested on the same
query. Every model is evaluated leave-one-query-out: train on the pages of all other
queries, predict the city of each page of the held-out query. That asks the research
question directly: does what a page reveals about location carry over to a search the
model has never seen?

Models:

- ``constant``: ignores the page; always guesses the most central city (chance floor).
- ``mentions``: counts how often the page names each city (and its state). No training.
- ``text``: TF-IDF over visible text and result domains, logistic regression.
- ``text_masked``: the same, with every configured city and state name blanked out, so
  it can only use implicit signal (localities, regional outlets, local-language words).

Each model outputs a probability per city; its guess is the most likely one, and the
distance and score are of that guess. (Picking the city with the best *expected* score
would be optimal only for calibrated probabilities. Logistic regression over 16 classes
and ~200 pages gives flat ones, and that rule then walks away from a correct top pick.)
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances

from serpint.config import TIERS, City, Settings
from serpint.geo import geoguessr_score, haversine_km
from serpint.parse import Page, mention_patterns

MASK = "PLACE"
# Inverse regularization strength for the text models. Not tuned: with one page per
# (query, city) there's no held-out data to tune it on without leaking the test query.
LOGREG_C = 10.0


# --------------------------------------------------------------- documents


def mask_pattern(cities: Iterable[City]) -> re.Pattern[str]:
    """Match any configured city name, alias or state, through the end of the word.

    The whole word goes, so "Keralam", "Mumbaikar" and "flats-in-chennai" lose the name
    without leaving a recognizable suffix behind.
    """
    names = {n for c in cities for n in (c.name, c.state, *c.aliases)}
    alts = sorted((re.escape(n) for n in names), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(alts) + r")\w*", re.IGNORECASE)


def document(
    page: Page, mask: re.Pattern[str] | None = None, elements: Iterable[str] | None = None
) -> str:
    """Return the text a model reads for a page: visible text plus one token per domain.

    Domains become single tokens (``dom_magicbricks_com``) so a regional outlet counts
    as one feature rather than as the words in its name.
    """
    keep = None if elements is None else set(elements)
    lines = []
    for name, items in page.elements.items():
        if keep is not None and name not in keep:
            continue
        for item in items:
            text, domain = item.text, item.domain
            if mask is not None:
                text = mask.sub(MASK, text)
                domain = None if domain is None else mask.sub(MASK, domain)
            lines.append(text)
            if domain:
                lines.append("dom_" + re.sub(r"[^\w]", "_", domain))
    return "\n".join(lines)


# --------------------------------------------------------------------- geo


@dataclass(frozen=True)
class Geo:
    """The candidate cities and the distances between them."""

    cities: tuple[City, ...]
    km: np.ndarray  # km[i, j]: distance between cities[i] and cities[j]
    scale_km: float  # GeoGuessr score scale

    @classmethod
    def of(cls, cities: Iterable[City], scale_km: float) -> Geo:
        """Build the distance matrix for ``cities``."""
        cs = tuple(cities)
        km = np.array([[haversine_km(a.lat, a.lon, b.lat, b.lon) for b in cs] for a in cs])
        return cls(cs, km, scale_km)

    @property
    def ids(self) -> list[str]:
        """Return the city ids, in matrix order."""
        return [c.id for c in self.cities]

    def central(self) -> int:
        """Return the index of the city with the best mean score against all the others."""
        return int(np.argmax(np.exp(-self.km / self.scale_km).sum(axis=1)))


# ------------------------------------------------------------------ models


class Model:
    """A city classifier over pages; ``predict_proba`` columns follow ``Geo.ids``."""

    name = "model"

    def fit(self, pages: list[Page], geo: Geo) -> Model:
        """Train on ``pages`` (default: nothing to learn)."""
        self.geo = geo
        return self

    def predict_proba(self, pages: list[Page]) -> np.ndarray:
        """Return one row of city probabilities per page."""
        raise NotImplementedError


class ConstantModel(Model):
    """Always the most central city: the best guess knowing nothing about the page."""

    name = "constant"

    def predict_proba(self, pages: list[Page]) -> np.ndarray:
        """Return all probability on the city with the best mean score to all others."""
        n = len(self.geo.cities)
        out = np.zeros((len(pages), n))
        out[:, self.geo.central()] = 1.0
        return out


class MentionModel(Model):
    """Weight each city by how often the page names it, or (half as much) its state."""

    name = "mentions"
    STATE_WEIGHT = 0.5

    def fit(self, pages: list[Page], geo: Geo) -> Model:
        """Build name patterns for the candidate cities."""
        super().fit(pages, geo)
        self.patterns = mention_patterns(geo.cities)
        return self

    def predict_proba(self, pages: list[Page]) -> np.ndarray:
        """Return mention counts normalized per page; the central city if nothing is named."""
        out = np.zeros((len(pages), len(self.geo.cities)))
        for i, page in enumerate(pages):
            text = page.text()
            for j, c in enumerate(self.geo.cities):
                p = self.patterns[c.id]
                out[i, j] = len(p.city.findall(text)) + self.STATE_WEIGHT * len(
                    p.state.findall(text)
                )
            total = out[i].sum()
            if total:
                out[i] /= total
            else:
                out[i, self.geo.central()] = 1.0
        return out


class TextModel(Model):
    """TF-IDF over visible text and domains, multinomial logistic regression."""

    def __init__(self, mask: re.Pattern[str] | None = None, seed: int = 0) -> None:
        self.mask = mask
        self.seed = seed
        self.name = "text_masked" if mask is not None else "text"

    def fit(self, pages: list[Page], geo: Geo) -> Model:
        """Fit the vectorizer and classifier on the training pages only."""
        super().fit(pages, geo)
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), min_df=2, lowercase=True
        )
        x = self.vectorizer.fit_transform([document(p, self.mask) for p in pages])
        y = [p.city_id for p in pages]
        self.clf = LogisticRegression(C=LOGREG_C, max_iter=5000, random_state=self.seed)
        self.clf.fit(x, y)
        return self

    def predict_proba(self, pages: list[Page]) -> np.ndarray:
        """Return probabilities aligned to ``Geo.ids`` (0 for cities unseen in training)."""
        x = self.vectorizer.transform([document(p, self.mask) for p in pages])
        proba = self.clf.predict_proba(x)
        out = np.zeros((len(pages), len(self.geo.cities)))
        col = {cid: j for j, cid in enumerate(self.geo.ids)}
        for k, cid in enumerate(self.clf.classes_):
            out[:, col[cid]] = proba[:, k]
        return out


# -------------------------------------------------------------- evaluation


@dataclass(frozen=True)
class Prediction:
    """A model's answer for one held-out page."""

    model: str
    query_id: str
    tier: str
    city_id: str
    guess: str
    p_true: float
    distance_km: float
    score: float


def leave_one_query_out(
    pages: list[Page],
    settings: Settings,
    make_model: Callable[[], Model],
    candidates: Iterable[str] | None = None,
    test: list[Page] | None = None,
) -> list[Prediction]:
    """Evaluate a model by holding out each query in turn.

    Args:
        pages: Pages to train on; every query among them is held out once.
        settings: Cities, coordinates and score scale.
        make_model: Returns a fresh, unfitted model.
        candidates: City ids a guess may name. Defaults to the cities among ``pages``;
            pass the full set when ``pages`` is a subset, so chance stays the same.
        test: Pages to predict for each held-out query, if not ``pages`` themselves
            (e.g. scrubbed versions, while training stays on ordinary pages).
    """
    wanted = set(candidates) if candidates is not None else {p.city_id for p in pages}
    geo = Geo.of((c for c in settings.cities if c.id in wanted), settings.geoguessr_scale_km)
    index = {cid: i for i, cid in enumerate(geo.ids)}
    out = []
    for qid in dict.fromkeys(p.query_id for p in pages):
        train = [p for p in pages if p.query_id != qid]
        held_out = [p for p in (pages if test is None else test) if p.query_id == qid]
        model = make_model().fit(train, geo)
        for page, probs in zip(held_out, model.predict_proba(held_out), strict=True):
            t = index[page.city_id]
            g = int(np.argmax(probs))
            km = float(geo.km[t, g])
            out.append(
                Prediction(
                    model=model.name,
                    query_id=qid,
                    tier=page.tier,
                    city_id=page.city_id,
                    guess=geo.ids[g],
                    p_true=float(probs[t]),
                    distance_km=km,
                    score=geoguessr_score(km, settings.geoguessr_scale_km),
                )
            )
    return out


@dataclass(frozen=True)
class Summary:
    """Aggregate metrics over a set of predictions."""

    n: int
    accuracy: float  # the guess is the right city
    state_accuracy: float  # the guess is in the right state
    median_km: float
    mean_score: float


def summarize(preds: list[Prediction], settings: Settings) -> Summary:
    """Summarize predictions (``n == 0`` gives NaNs)."""
    if not preds:
        nan = float("nan")
        return Summary(0, nan, nan, nan, nan)
    state = {c.id: c.state for c in settings.cities}
    return Summary(
        n=len(preds),
        accuracy=sum(p.guess == p.city_id for p in preds) / len(preds),
        state_accuracy=sum(state[p.guess] == state[p.city_id] for p in preds) / len(preds),
        median_km=statistics.median(p.distance_km for p in preds),
        mean_score=statistics.fmean(p.score for p in preds),
    )


# ----------------------------------------------------- similarity vs distance


@dataclass(frozen=True)
class DistanceSignal:
    """Whether pages for one query look more alike for cities closer together."""

    query_id: str
    tier: str
    mean_text_distance: float  # mean pairwise cosine distance; 0 = identical pages
    rho: float  # Spearman correlation of text distance with km; NaN if pages identical
    p_value: float  # one-sided Mantel permutation test (rho > 0)


def _ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    r[order] = np.arange(len(v))
    # Average ranks over ties.
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return (sums / counts)[inv]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den else float("nan")


def distance_signal(
    pages: list[Page],
    settings: Settings,
    mask: re.Pattern[str] | None = None,
    n_perm: int = 999,
) -> list[DistanceSignal]:
    """For each query, correlate page dissimilarity with geographic distance.

    Needs no training and no labels at prediction time: if pages from nearby cities are
    more alike than pages from distant ones, the page carries location. Significance is
    a Mantel test: cities are shuffled ``n_perm`` times to get the null distribution.
    """
    rng = np.random.default_rng(settings.seed)
    by_city = {c.id: c for c in settings.cities}
    out = []
    for qid in dict.fromkeys(p.query_id for p in pages):
        qpages = [p for p in pages if p.query_id == qid]
        geo = Geo.of((by_city[p.city_id] for p in qpages), settings.geoguessr_scale_km)
        docs = [document(p, mask) for p in qpages]
        iu = np.triu_indices(len(qpages), k=1)
        try:
            x = TfidfVectorizer(sublinear_tf=True).fit_transform(docs)
            text_d = cosine_distances(x)
        except ValueError:  # every document empty after masking
            text_d = np.zeros((len(qpages), len(qpages)))
        t = text_d[iu]
        rho = p = float("nan")
        if np.ptp(t) > 1e-12:
            rt = _ranks(t)
            geo_rank = np.zeros_like(geo.km)
            geo_rank[iu] = _ranks(geo.km[iu])
            geo_rank = geo_rank + geo_rank.T
            rho = _corr(rt, geo_rank[iu])
            hits = 0
            for _ in range(n_perm):
                perm = rng.permutation(len(qpages))
                hits += _corr(rt, geo_rank[np.ix_(perm, perm)][iu]) >= rho
            p = (hits + 1) / (n_perm + 1)
        out.append(DistanceSignal(qid, qpages[0].tier, float(t.mean()), rho, p))
    return out


# ------------------------------------------------------------------ report


def _f(x: float, fmt: str) -> str:
    return "-" if x != x else format(x, fmt)  # NaN check


def _row(label: str, s: Summary) -> str:
    return (
        f"| {label} | {s.n} | {_f(s.accuracy, '.0%')} | {_f(s.state_accuracy, '.0%')} | "
        f"{_f(s.median_km, ',.0f')} | {_f(s.mean_score, ',.0f')} |"
    )


def model_report(
    preds: dict[str, list[Prediction]],
    signals: dict[str, list[DistanceSignal]],
    settings: Settings,
    n_pages: int,
) -> str:
    """Render the evaluation as Markdown."""
    tiers = [t for t in TIERS if any(p.tier == t for ps in preds.values() for p in ps)]
    n_cities = len({p.city_id for ps in preds.values() for p in ps})
    out = [
        f"# City prediction ({n_pages} pages, {n_cities} cities)",
        "",
        "Generated by `serpint model`. Leave-one-query-out: each query's pages are",
        "predicted by a model trained on every other query, so nothing about the held-out",
        "query is seen in training. Chance accuracy is "
        f"1/{n_cities} = {1 / n_cities:.0%}. Score is GeoGuessr-style, "
        f"`5000 * exp(-km / {settings.geoguessr_scale_km:g})`, of each model's most",
        "likely city.",
        "",
        "- `constant`: no page at all; always the most central city.",
        "- `mentions`: the city the page names most (state names count half).",
        "- `text`: TF-IDF of visible text and domains, logistic regression.",
        "- `text_masked`: the same with every configured city and state name blanked.",
        "",
        "## By tier",
        "",
    ]
    head = [
        "| model | pages | city acc | state acc | median km | mean score |",
        "|---|---|---|---|---|---|",
    ]
    for tier in [*tiers, "all"]:
        out += [f"### {tier}", "", *head]
        for name, ps in preds.items():
            sel = ps if tier == "all" else [p for p in ps if p.tier == tier]
            out.append(_row(f"`{name}`", summarize(sel, settings)))
        out.append("")

    names = list(preds)
    out += [
        "## By query",
        "",
        "City accuracy / mean score per model.",
        "",
        "| query | tier | " + " | ".join(f"`{n}`" for n in names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for q in settings.queries:
        cells = []
        for n in names:
            s = summarize([p for p in preds[n] if p.query_id == q.id], settings)
            if s.n:
                cells.append(f"{_f(s.accuracy, '.0%')} / {_f(s.mean_score, ',.0f')}")
        if cells:
            out.append(f"| {q.id} | {q.tier} | " + " | ".join(cells) + " |")

    out += [
        "",
        "## Do nearby cities get more similar pages?",
        "",
        "Per query, the Spearman correlation between how different two cities' pages are",
        "(TF-IDF cosine distance) and how far apart the cities are, with a one-sided",
        "Mantel permutation p-value. No model is trained, so this also covers queries",
        "where no signal carries over from other queries. Mean distance 0 means every",
        "city got the same page. With one test per query, about one in twenty p-values",
        f"falls under 0.05 by chance; Bonferroni over these {len(signals[next(iter(signals))])}",
        "queries puts the bar at 0.05 / that count.",
        "",
    ]
    variants = list(signals)
    out += [
        "| query | tier | " + " | ".join(f"{v}: mean dist | {v}: rho (p)" for v in variants) + " |",
        "|---|---|" + "---|---|" * len(variants),
    ]
    by_variant = {v: {s.query_id: s for s in signals[v]} for v in variants}
    for q in settings.queries:
        if q.id not in by_variant[variants[0]]:
            continue
        cells = []
        for v in variants:
            s = by_variant[v][q.id]
            rho = "-" if s.rho != s.rho else f"{s.rho:+.2f} ({s.p_value:.3f})"
            cells += [f"{s.mean_text_distance:.3f}", rho]
        out.append(f"| {q.id} | {q.tier} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def evaluate(
    pages: list[Page], settings: Settings, n_perm: int = 999
) -> tuple[dict[str, list[Prediction]], dict[str, list[DistanceSignal]]]:
    """Run every model and both distance-signal variants."""
    mask = mask_pattern(settings.cities)
    factories: list[Callable[[], Model]] = [
        ConstantModel,
        MentionModel,
        lambda: TextModel(seed=settings.seed),
        lambda: TextModel(mask, seed=settings.seed),
    ]
    preds: dict[str, list[Prediction]] = {}
    for make in factories:
        ps = leave_one_query_out(pages, settings, make)
        preds[ps[0].model] = ps
    signals = {
        "text": distance_signal(pages, settings, None, n_perm),
        "masked": distance_signal(pages, settings, mask, n_perm),
    }
    return preds, signals

"""Demo app: upload a results-page screenshot, see what it gives away, get it scrubbed.

Uploads are processed in memory and never stored: the image is read from the request,
OCR'd, scrubbed and sent back inline in the response page. Nothing is written to disk
and no upload content is logged. The app makes no network requests of its own (no
SerpAPI calls; the page loads no external assets).

The attacker it shows is the ``text`` model from ``serpint model``, trained at startup
on the visible text of the collected pages, plus the name counter. Its guesses are
limited to the cities in that data.
"""

from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from flask import Flask, Response, render_template_string, request

from serpint import parse, scrub
from serpint.client import SerpApiClient
from serpint.config import Settings, load_settings
from serpint.models import Geo, TextModel
from serpint.parse import Item, Page

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 40_000_000  # e.g. 1920 x 20000; full-page screenshots are tall
TOP_K = 3


@dataclass(frozen=True)
class Guess:
    """A city and how sure the attacker is."""

    city: str
    prob: float


class Attacker:
    """What someone with this project's data would guess from a screenshot's text."""

    def __init__(self, pages: list[Page], settings: Settings) -> None:
        self.settings = settings
        names = {c.id: c.name for c in settings.cities}
        self.names = names
        self.geo = Geo.of(
            (c for c in settings.cities if c.id in {p.city_id for p in pages}),
            settings.geoguessr_scale_km,
        )
        self.text = TextModel(seed=settings.seed)
        self.text.fit(pages, self.geo)
        self.patterns = parse.mention_patterns(settings.cities)

    @classmethod
    def from_plan(cls, settings: Settings, plan: str = "core") -> Attacker:
        """Train on the parsed pages of a plan (read from the committed cache)."""
        result = parse.parse_plan(SerpApiClient(settings, None), settings, plan)
        if not result.pages:
            raise RuntimeError(f"No collected pages for plan {plan!r}; nothing to train on.")
        return cls(result.pages, settings)

    def guess(self, lines: list[str]) -> list[Guess]:
        """Return the top guesses for a screenshot's readable text."""
        if not any(s.strip() for s in lines):
            return []
        page = Page("upload", "", "", "", {"ocr": [Item(s) for s in lines]})
        probs = self.text.predict_proba([page])[0]
        order = probs.argsort()[::-1][:TOP_K]
        return [Guess(self.names[self.geo.ids[i]], float(probs[i])) for i in order]

    def named(self, lines: list[str]) -> list[tuple[str, int]]:
        """Return the configured cities the text names outright, most often first."""
        text = "\n".join(lines)
        counts = Counter(
            {
                self.names[cid]: len(p.city.findall(text))
                for cid, p in self.patterns.items()
                if p.city.search(text)
            }
        )
        return counts.most_common()


REASONS = {
    "location_line": "Google's location line",
    "place_name": "names a place",
    "pin_code": "PIN code",
    "landline": "landline number",
    "section:related_searches": "related searches",
    "section:local_results": "local results and map",
    "section:news": "news block",
}

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SERPint scrubber</title>
<style>
  :root { --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6b66; --line:#e3e2dd; --card:#fff;
          --accent:#2457c5; --warn:#b3261e; --ok:#1e7a46; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161615; --fg:#ecebe6; --muted:#a3a29b; --line:#33322f; --card:#1f1f1d;
            --accent:#8fb0ff; --warn:#ff8a80; --ok:#7fd3a2; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 980px; margin: 0 auto; padding: 32px 16px 64px; }
  h1 { font-size: 1.6rem; margin: 0 0 4px; }
  h2 { font-size: 1.1rem; margin: 28px 0 8px; }
  p.lede { color: var(--muted); margin: 0 0 24px; }
  form, .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
                padding: 20px; }
  label { display: block; font-weight: 600; margin-bottom: 6px; }
  .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: end; }
  input[type=file] { max-width: 100%; }
  select, button { font: inherit; padding: 8px 12px; border-radius: 8px;
                   border: 1px solid var(--line); background: var(--bg); color: var(--fg); }
  button { background: var(--accent); border-color: var(--accent); color: #fff;
           font-weight: 600; cursor: pointer; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 720px) { .grid { grid-template-columns: 1fr; } }
  .guess { display:flex; justify-content:space-between; padding:6px 0;
           border-bottom:1px solid var(--line); }
  .guess:last-child { border-bottom: 0; }
  .big { font-size: 1.25rem; font-weight: 700; }
  .warn { color: var(--warn); } .ok { color: var(--ok); } .muted { color: var(--muted); }
  ul { padding-left: 20px; margin: 6px 0; }
  img { width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; }
  .error { border-color: var(--warn); color: var(--warn); }
  a.dl { display:inline-block; margin-top:12px; font-weight:600; color: var(--accent); }
  footer { margin-top: 40px; color: var(--muted); font-size: .9rem; }
</style>
</head>
<body>
<main>
  <h1>Does your search screenshot give away your city?</h1>
  <p class="lede">Upload a screenshot of a Google results page. It's checked for what
  reveals where you are, and you get back a copy with those parts blurred. The image is
  processed in memory and not stored.</p>

  <form method="post" action="{{ url_for('scrub_upload') }}" enctype="multipart/form-data">
    <div class="row">
      <div>
        <label for="image">Screenshot (PNG or JPEG, up to {{ max_mb }} MB)</label>
        <input id="image" name="image" type="file" accept="image/png,image/jpeg" required>
      </div>
      <div>
        <label for="level">Blur</label>
        <select id="level" name="level">
          <option value="sections" {% if level == 'sections' %}selected{% endif %}>
            Place names and local sections</option>
          <option value="names" {% if level == 'names' %}selected{% endif %}>
            Place names only</option>
        </select>
      </div>
      <button type="submit">Check and scrub</button>
    </div>
  </form>

  {% if error %}<h2>Couldn't process that</h2><div class="card error">{{ error }}</div>{% endif %}

  {% if result %}
  <h2>What it gives away</h2>
  <div class="grid">
    <div class="card">
      <div class="muted">Before scrubbing, a model would guess</div>
      {% for g in result.before %}
        <div class="guess"><span {% if loop.first %}class="big"{% endif %}>{{ g.city }}</span>
          <span>{{ '%.0f' % (g.prob * 100) }}%</span></div>
      {% else %}<p class="muted">No readable text.</p>{% endfor %}
      {% if result.named %}
        <p class="warn">Names on the page: {{ result.named }}</p>
      {% endif %}
    </div>
    <div class="card">
      <div class="muted">After scrubbing</div>
      {% for g in result.after %}
        <div class="guess"><span {% if loop.first %}class="big"{% endif %}>{{ g.city }}</span>
          <span>{{ '%.0f' % (g.prob * 100) }}%</span></div>
      {% else %}<p class="muted">No readable text left.</p>{% endfor %}
      {% if result.named_after %}
        <p class="warn">Still named: {{ result.named_after }}</p>
      {% else %}<p class="ok">No configured city is named in what's left.</p>{% endif %}
    </div>
  </div>
  <p class="muted">Guesses are limited to the {{ n_cities }} cities this project
  collected, and with that many to choose from, a guess near
  {{ '%.0f' % (100 / n_cities) }}% is no better than chance.</p>

  <h2>What was blurred ({{ result.blurred_pct }} of the image)</h2>
  <div class="card">
    {% if result.reasons %}
    <ul>{% for why, n in result.reasons %}<li>{{ why }}: {{ n }}</li>{% endfor %}</ul>
    {% else %}<p class="muted">Nothing needed blurring.</p>{% endif %}
  </div>

  <h2>Scrubbed screenshot</h2>
  <div class="card">
    <img src="data:image/png;base64,{{ result.png }}" alt="The scrubbed screenshot">
    <a class="dl" href="data:image/png;base64,{{ result.png }}"
       download="scrubbed.png">Download scrubbed.png</a>
  </div>
  {% endif %}

  <footer>SERPint research prototype. Text is read with OCR, so check the scrubbed
  image before sharing it: blurring catches what the rules know about, not everything
  a person could recognise.</footer>
</main>
</body>
</html>
"""


def _read_image(data: bytes) -> Any:
    from PIL import Image, UnidentifiedImageError

    try:
        im = Image.open(BytesIO(data))
        if im.width * im.height > MAX_PIXELS:
            raise ValueError("That image is too large; try a smaller screenshot.")
        if im.format not in ("PNG", "JPEG"):
            raise ValueError("Please upload a PNG or JPEG screenshot.")
        im.load()
        return im.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as e:
        raise ValueError("That file isn't a readable PNG or JPEG image.") from e


def _counts(named: list[tuple[str, int]]) -> str:
    return ", ".join(f"{name} ({n})" for name, n in named)


def analyse(image: Any, attacker: Attacker, gazetteer: scrub.Gazetteer, level: str) -> dict:
    """OCR, guess, scrub and guess again; everything stays in memory."""
    lines = scrub.ocr(image)
    redactions = scrub.plan_redactions(lines, gazetteer, level)
    kept = [ln.text for ln in scrub.kept_lines(lines, redactions)]
    all_text = [ln.text for ln in lines]
    buf = BytesIO()
    scrub.apply(image, redactions).save(buf, format="PNG")
    reasons = Counter(REASONS.get(r.reason, r.reason) for r in redactions)
    return {
        "before": attacker.guess(all_text),
        "after": attacker.guess(kept),
        "named": _counts(attacker.named(all_text)),
        "named_after": _counts(attacker.named(kept)),
        "reasons": reasons.most_common(),
        "blurred_pct": f"{scrub.blurred_fraction(image.size, redactions):.0%}",
        "png": base64.b64encode(buf.getvalue()).decode("ascii"),
    }


def create_app(settings: Settings | None = None, attacker: Attacker | None = None) -> Flask:
    """Build the app; trains the attacker unless one is given."""
    settings = settings or load_settings()
    attacker = attacker or Attacker.from_plan(settings)
    gazetteer = scrub.Gazetteer.load(settings)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    def page(status: int = 200, **ctx: Any) -> Response:
        ctx.setdefault("level", "sections")
        html = render_template_string(
            PAGE,
            max_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
            n_cities=len(attacker.geo.ids),
            **ctx,
        )
        return Response(html, status=status, headers={"Cache-Control": "no-store"})

    @app.get("/")
    def index() -> Response:
        return page()

    @app.post("/scrub")
    def scrub_upload() -> Response:
        level = request.form.get("level", "sections")
        if level not in ("names", "sections"):
            return page(400, error="Unknown blur level.")
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return page(400, error="Choose a screenshot to upload.", level=level)
        try:
            image = _read_image(upload.read())
        except ValueError as e:
            return page(400, error=str(e), level=level)
        return page(result=analyse(image, attacker, gazetteer, level), level=level)

    @app.errorhandler(413)
    def too_large(_: Exception) -> Response:
        mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        return page(413, error=f"That file is over {mb} MB.")

    return app

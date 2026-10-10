# SERPint

How precisely can someone's city be inferred from their Google results page, even
for searches that have nothing to do with location? And can we blur the parts of a
screenshot that give it away before it's shared?

SERPint is a research prototype. It collects Google results pages through
[SerpAPI](https://serpapi.com) at **simulated** locations across 20 Indian cities,
trains models to predict the city from the page, maps which page elements leak, and
builds a scrubber that blurs them.

> **Status:** Phase 1 (config and data collection) is implemented. Parsing, models, the
> leak map, the scrubber and the demo app follow phase by phase.

## Setup

```sh
make setup                 # Python 3.11 venv + pinned requirements + package
cp .env.example .env       # then put your key in .env: SERPAPI_API_KEY=...
make test lint
```

The key is read from `SERPAPI_API_KEY` (environment or `.env`). It is never printed,
logged or written to disk; `.env` is git-ignored.

## Budget and caching

SerpAPI searches cost money or quota, so **every** search goes through one client,
`src/serpint/client.py`:

| Guarantee | How |
|---|---|
| Never pay twice | Responses are cached at `data/raw/{query_id}/{city_id}/{timestamp}.json`. A cell with a cached file is never re-fetched. |
| Spend is always known | `data/ledger.jsonl` gets a `started` line **before** each paid request and a `finished` line after. Spend = number of `started` lines, so a crash mid-request over-counts rather than under-counts. |
| Hard ceiling | A run that would take the ledger past `budget.max_searches` (`config/settings.yaml`, 420) is refused unless you pass `--allow-over-budget N`. It's also refused if it needs more searches than the SerpAPI account has left this month (Account API, free). |
| See before you spend | `--dry-run` prints every planned call, marks cached ones as free, and totals the new searches. It makes no requests. |
| Resumable | Files are written atomically (temp file + rename); re-running a plan skips cached cells and retries only the missing ones. A lock file stops two runs overlapping. |
| No leaked keys | The key is only a request parameter; it is redacted from every error message, and a response containing it is refused rather than written. |

Free endpoints don't touch the ledger: the Locations API (cached in
`data/locations_raw/`), the Account API, and each search's `raw_html_file`
(gzipped to `data/html/…`, used later for screenshots).

Every search uses `gl=in`, `hl=en`, `google_domain=google.co.in`, `device=desktop`,
and a `location` resolved to SerpAPI's canonical name (`data/locations.json`).
Cells are fetched query by query, so time-varying content (scores, news) stays
roughly constant across cities and can't masquerade as a location signal.

Raw responses, HTML, the ledger and resolved locations are committed to git, so the
spent quota isn't lost if a machine goes away.

## Reproducing

Collection is split into nested plans so it fits SerpAPI's free tier. Every plan
reuses what earlier plans cached, so moving to a bigger plan only pays for new cells.

| Plan | Cities × queries | Searches | Purpose |
|---|---|---|---|
| `pilot` | 5 × 5 | 25 | Inspect which fields come back before spending more |
| `core` | 16 × 15 | 240 (215 after the pilot) | Main dataset; fits one free-tier month |
| `grid` | 20 × 20 | 400 (160 after core) | Full design; top up in a later month |

`core` keeps all four same-state pairs and every neutral and implicit query except
"best schools" and "how to make masala chai", and trims the explicit_local tier (the
easy upper bound) to "weather" and "restaurants near me". It drops Indore, Patna,
Bhubaneswar and Visakhapatnam.

```sh
make locations      # resolve the 20 cities (free); check the table
make pilot-dry      # 25 planned searches
make pilot          # run them
make inspect-pilot  # reports/pilot_fields.md: which fields SerpAPI returned
make core-dry       # 215 more; pilot cells show as cached
make core
make collect-dry    # later month: the 160 grid cells core doesn't have
make collect
```

A run that needs more searches than the account has left this month is refused; use
`serpint collect <plan> --limit N` to fetch what fits and re-run after the reset.

## Configuration

- `config/cities.yaml`: 20 cities with state and coordinates, including the same-state
  pairs Chennai/Coimbatore, Mumbai/Pune, Bengaluru/Mysuru and Kochi/Thiruvananthapuram.
- `config/queries.yaml`: 20 queries tagged `explicit_local`, `implicit_local` or
  `neutral`.
- `config/settings.yaml`: search parameters, budget, score scale, and the `pilot`
  and `core` plans.

## Ethics

- **Synthetic searches only.** Every page in this project comes from searches we ran
  ourselves through SerpAPI at simulated locations. No real user's searches,
  screenshots or data are collected.
- **Purpose.** The measurement exists to show people how much a casually shared
  screenshot of a search page can reveal, and the scrubber exists to remove it.
  The demo app processes uploaded screenshots in memory and does not store them.

"""Command-line entry point: ``serpint <command>`` (also driven by the Makefile)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from serpint import collect, parse
from serpint.client import SerpApiClient, SerpApiError
from serpint.config import PROJECT_ROOT, Settings, load_settings


def _client(settings: Settings, allow_over_budget: int = 0) -> SerpApiClient:
    load_dotenv(PROJECT_ROOT / ".env")
    return SerpApiClient(
        settings, os.environ.get("SERPAPI_API_KEY"), allow_over_budget=allow_over_budget
    )


def cmd_locations(args: argparse.Namespace) -> int:
    """Resolve every city to a SerpAPI canonical location (free) and print the table."""
    settings = load_settings()
    resolved = collect.resolve_locations(_client(settings), settings, refresh=args.refresh)
    print(collect.format_locations(settings, resolved))
    print(f"\nSaved to {settings.data_dir / 'locations.json'}")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    """Plan (and unless --dry-run, run) a collection."""
    settings = load_settings()
    client = _client(settings, args.allow_over_budget)
    cells = collect.plan_cells(settings, args.plan)
    locations = collect.load_locations(settings)

    if args.dry_run:
        text, _, _ = collect.describe_plan(client, cells, locations, None, args.limit)
        print(text)
        print("\nDry run: no requests were made. The account balance is checked on a real run.")
        return 0

    if any(c.city.id not in locations for c in cells):
        print("Resolving locations (Locations API, free)...")
        locations = collect.resolve_locations(client, settings)
        print(collect.format_locations(settings, locations) + "\n")

    account_left = None
    if settings.check_account and not args.skip_account_check:
        account_left = collect.account_searches_left(client)

    text, budget, to_fetch = collect.describe_plan(
        client, cells, locations, account_left, args.limit
    )
    print(text + "\n")
    if not budget.fits:
        local_left = max(0, budget.limit - budget.spent)
        print(f"REFUSED: {budget.planned_new} new searches planned, {budget.allowed} allowed.")
        if account_left is not None and account_left < local_left:
            print("The SerpAPI account balance is the limit; wait for the monthly reset.")
        else:
            print("Approve going past max_searches with --allow-over-budget N,")
        print(f"or fetch a subset with --limit {budget.allowed}.")
        return 2
    if not to_fetch:
        print("Nothing new to fetch; checking raw HTML for cached cells.")

    report = collect.run(client, cells, locations, to_fetch)
    print(
        f"\nfetched {len(report.fetched)}, cached {len(report.cached)}, "
        f"failed {len(report.failed)}, raw HTML failures {len(report.html_failed)}; "
        f"ledger spend {client.ledger.spent()} / {client.budget_limit}"
    )
    return 1 if report.failed else 0


def cmd_budget(args: argparse.Namespace) -> int:
    """Print ledger spend and, unless --offline, the SerpAPI account balance."""
    settings = load_settings()
    client = _client(settings)
    entries = client.ledger.entries()
    finished = [e for e in entries if e.get("event") == "finished"]
    statuses: dict[str, int] = {}
    for e in finished:
        statuses[e["status"]] = statuses.get(e["status"], 0) + 1
    print(f"ledger spend:     {client.ledger.spent()} / {settings.max_searches}")
    print(f"finished calls:   {statuses or 'none'}")
    if not args.offline:
        print(f"account left:     {collect.account_searches_left(client)}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Write a Markdown inventory of the fields in cached responses for a plan."""
    settings = load_settings()
    client = _client(settings)
    rows = []
    for c in collect.plan_cells(settings, args.plan):
        hit = client.cached(c.query.id, c.city.id)
        if hit is not None:
            rows.append((c.query.id, c.city.id, hit.data))
    if not rows:
        print(f"No cached responses for plan {args.plan!r}. Run `serpint collect {args.plan}`.")
        return 1
    out = args.out or settings.root / "reports" / f"{args.plan}_fields.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(collect.field_inventory(rows, settings), encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} responses)")
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    """Parse cached responses for a plan into visible page elements, plus a report."""
    settings = load_settings()
    result = parse.parse_plan(_client(settings), settings, args.plan)
    if not result.pages:
        print(f"No cached responses for plan {args.plan!r}. Run `serpint collect {args.plan}`.")
        return 1
    pages_path = settings.data_dir / "processed" / f"{args.plan}_pages.jsonl"
    parse.write_pages(result.pages, pages_path)
    report = settings.root / "reports" / f"{args.plan}_parse.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(parse.parse_report(result.pages, settings), encoding="utf-8")
    print(f"Parsed {len(result.pages)} pages -> {pages_path}")
    print(f"Report -> {report}")
    if result.missing:
        print(f"{len(result.missing)} cells not collected yet: {', '.join(result.missing[:5])}...")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(prog="serpint", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("locations", help="resolve cities to SerpAPI canonical locations (free)")
    s.add_argument("--refresh", action="store_true", help="re-resolve instead of reusing")
    s.set_defaults(func=cmd_locations)

    s = sub.add_parser("collect", help="fetch a plan of searches (budget-guarded, resumable)")
    s.add_argument("plan", help="a plan from config/settings.yaml (pilot, core) or grid")
    s.add_argument("--dry-run", action="store_true", help="print planned calls; no requests")
    s.add_argument("--limit", type=int, help="fetch at most this many new cells")
    s.add_argument(
        "--allow-over-budget",
        type=int,
        default=0,
        metavar="N",
        help="explicitly approve up to N searches beyond max_searches",
    )
    s.add_argument(
        "--skip-account-check", action="store_true", help="don't query the account balance"
    )
    s.set_defaults(func=cmd_collect)

    s = sub.add_parser("budget", help="show ledger spend and account balance")
    s.add_argument("--offline", action="store_true", help="ledger only; no request")
    s.set_defaults(func=cmd_budget)

    s = sub.add_parser("inspect", help="inventory the fields in cached responses")
    s.add_argument("plan", help="a plan from config/settings.yaml (pilot, core) or grid")
    s.add_argument("--out", type=Path, help="output path (default reports/<plan>_fields.md)")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("parse", help="extract the visible text of cached pages (no requests)")
    s.add_argument("plan", help="a plan from config/settings.yaml (pilot, core) or grid")
    s.set_defaults(func=cmd_parse)
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (SerpApiError, RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

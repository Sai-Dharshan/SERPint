from pathlib import Path

from serpint import scrub, scrub_eval
from serpint.screenshots import Shot

CITIES = ["chennai", "mumbai", "delhi", "kolkata"]
NAMES = {"chennai": "Chennai", "mumbai": "Mumbai", "delhi": "Delhi", "kolkata": "Kolkata"}


def test_scrubbing_names_defeats_a_name_reading_attacker(settings):
    shots, lines = [], []
    for q in ("weather", "ipl_score", "what_is_gst"):
        for c in CITIES:
            shots.append(Shot(q, c, "t", Path("unused.png"), 1000, 1000, None))
            lines.append(
                [
                    scrub.Line((0, 0, 500, 20), f"result about {q}", 1.0),
                    scrub.Line((0, 900, 500, 920), f"{NAMES[c]} - From your IP address", 1.0),
                ]
            )
    results = {r.level: r for r in scrub_eval.evaluate(shots, lines, settings)}

    def acc(level, attacker):
        preds = results[level].preds[attacker]
        return sum(p.guess == p.city_id for p in preds) / len(preds)

    assert acc("none", "mentions") == 1.0
    assert acc("none", "text") == 1.0
    assert acc("names", "mentions") == 0.25  # the central-city fallback, right 1 time in 4
    assert results["names"].reasons == {"location_line": 12}
    assert all(f > 0 for f in results["names"].blurred)
    assert results["names"].lines_lost == [0.5] * 12
    report = scrub_eval.scrub_report(list(results.values()), None, settings, 12)
    assert "| `none` | 100% / 5,000 | 100% / 5,000 |" in report

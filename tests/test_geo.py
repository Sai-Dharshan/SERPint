import math

import pytest

from serpint.geo import geoguessr_score, haversine_km


def test_haversine_zero_and_known_distance():
    assert haversine_km(13.08, 80.27, 13.08, 80.27) == 0
    # Delhi to Mumbai is ~1,150 km great-circle.
    assert haversine_km(28.6139, 77.2090, 19.0760, 72.8777) == pytest.approx(1150, abs=15)


def test_score_shape():
    assert geoguessr_score(0, 400) == 5000
    assert geoguessr_score(400, 400) == pytest.approx(5000 / math.e)
    assert geoguessr_score(100, 400) > geoguessr_score(200, 400)

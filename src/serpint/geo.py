"""Great-circle distance and the GeoGuessr-style score."""

from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in km between two (lat, lon) points in degrees."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def geoguessr_score(distance_km: float, scale_km: float) -> float:
    """Return ``5000 * exp(-distance / scale)``: 5000 for a perfect guess, decaying with error."""
    return 5000.0 * math.exp(-distance_km / scale_km)

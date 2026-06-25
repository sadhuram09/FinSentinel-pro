"""Unit tests for the forecaster's pure mapping logic (no model loading)."""

from __future__ import annotations

import math

import pytest

from backend.ml.models.forecaster import NEUTRAL_BAND, _confidence_interval, _direction_from_prob
from backend.schemas import Direction


# --- neutral-band direction mapping ------------------------------------------
@pytest.mark.parametrize(
    "prob, expected",
    [
        (0.90, Direction.UP),
        (0.56, Direction.UP),
        (0.55, Direction.UP),     # exactly 0.5 + band
        (0.54, Direction.FLAT),
        (0.50, Direction.FLAT),
        (0.46, Direction.FLAT),
        (0.45, Direction.DOWN),   # exactly 0.5 - band
        (0.44, Direction.DOWN),
        (0.10, Direction.DOWN),
    ],
)
def test_direction_from_prob(prob, expected):
    assert _direction_from_prob(prob) is expected


def test_neutral_band_value():
    # Sanity: the band the mapping uses is the documented 0.05.
    assert NEUTRAL_BAND == 0.05
    assert _direction_from_prob(0.5 + NEUTRAL_BAND) is Direction.UP
    assert _direction_from_prob(0.5 - NEUTRAL_BAND) is Direction.DOWN


# --- confidence interval width -----------------------------------------------
@pytest.mark.parametrize(
    "p_a, p_b, lower, upper",
    [
        (0.52, 0.58, 0.52, 0.58),   # ordered
        (0.58, 0.52, 0.52, 0.58),   # unordered -> sorted
        (0.50, 0.50, 0.50, 0.50),   # zero width
    ],
)
def test_confidence_interval_bounds(p_a, p_b, lower, upper):
    lo, hi = _confidence_interval(p_a, p_b)
    assert math.isclose(lo, lower, abs_tol=1e-9)
    assert math.isclose(hi, upper, abs_tol=1e-9)
    assert math.isclose(hi - lo, abs(p_a - p_b), abs_tol=1e-9)  # width = |spread|


def test_confidence_interval_clamps_to_unit_range():
    lo, hi = _confidence_interval(-0.1, 1.2)
    assert lo == 0.0
    assert hi == 1.0

"""Tests for alpha-shape geometry functions."""

import pytest

from spices.pourbaix.geometry import (
    get_alpha_shape,
    find_ph_range_at_voltage,
    find_stability_area_around_voltage,
)


class TestGetAlphaShape:
    def test_square_points(self):
        points = [(7, 0), (7, 1), (8, 1), (8, 0)]
        shape = get_alpha_shape(points)
        assert shape.area > 0
        assert abs(shape.area - 1.0) < 0.1  # roughly a unit square

    def test_collinear_points(self):
        points = [(7, 0), (7.5, 0), (8, 0)]
        shape = get_alpha_shape(points)
        # Collinear points produce a LineString or degenerate polygon
        assert shape.length > 0


class TestFindPhRangeAtVoltage:
    def test_overlapping_regions(self):
        """Two overlapping squares should have a pH range at their shared voltage."""
        # Phase A: square at pH 7-8, V 0-1
        points_A = [(7, 0), (7, 1), (8, 1), (8, 0)]
        # Phase B: square at pH 7.5-8.5, V 0.5-1.5
        points_B = [(7.5, 0.5), (7.5, 1.5), (8.5, 1.5), (8.5, 0.5)]

        # At V=0.75, both regions overlap between pH 7.5 and 8
        result = find_ph_range_at_voltage(points_A, points_B, 0.75)
        assert len(result) == 2
        assert result[0] < result[1]
        assert abs(result[0] - 7.5) < 0.1
        assert abs(result[1] - 8.0) < 0.1

    def test_no_overlap(self):
        """Non-overlapping regions return empty tuple."""
        points_A = [(7, 0), (7, 1), (7.5, 1), (7.5, 0)]
        points_B = [(8, 2), (8, 3), (8.5, 3), (8.5, 2)]

        result = find_ph_range_at_voltage(points_A, points_B, 0.5)
        assert result == ()

    def test_voltage_outside_both_regions(self):
        """Voltage above both regions returns empty tuple."""
        points_A = [(7, 0), (7, 1), (8, 1), (8, 0)]
        points_B = [(7, 0), (7, 1), (8, 1), (8, 0)]

        result = find_ph_range_at_voltage(points_A, points_B, 5.0)
        assert result == ()


class TestFindStabilityAreaAroundVoltage:
    def test_full_overlap_in_water_region(self):
        """Both phases cover the entire water region."""
        # Large squares covering pH 6-9, V -2 to 3
        points_A = [(6, -2), (6, 3), (9, 3), (9, -2)]
        points_B = [(6, -2), (6, 3), (9, 3), (9, -2)]

        total, below, above = find_stability_area_around_voltage(
            points_A, points_B, voltage=0.5,
        )
        assert total > 0.9  # should be ~1.0 (full coverage)
        assert below > 0
        assert above > 0

    def test_no_overlap(self):
        """Non-overlapping regions return zeros."""
        points_A = [(7, 0), (7, 0.5), (7.5, 0.5), (7.5, 0)]
        points_B = [(8, 2), (8, 3), (8.5, 3), (8.5, 2)]

        total, below, above = find_stability_area_around_voltage(
            points_A, points_B, voltage=1.0,
        )
        assert total == 0.0
        assert below == 0.0
        assert above == 0.0

    def test_returns_three_floats(self):
        """Result is always a 3-tuple of floats."""
        points_A = [(7, 0), (7, 1), (8, 1), (8, 0)]
        points_B = [(7, 0), (7, 1), (8, 1), (8, 0)]

        result = find_stability_area_around_voltage(
            points_A, points_B, voltage=0.5,
        )
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

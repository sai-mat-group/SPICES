"""Tests for spices.pourbaix.geometry."""

from spices.pourbaix.geometry import (
    check_overlap,
    check_overlap_with_water_stability_lines,
    check_overlap_with_extended_water_stability_lines,
)
from spices.config import H_LINE, O_LINE


class TestCheckOverlap:
    def test_overlapping_polygon(self):
        # A square straddling the water stability region at pH 7
        polygon = [(5, -0.5), (5, 1.5), (9, 1.5), (9, -0.5)]
        assert check_overlap(polygon, H_LINE, O_LINE, range_of_pH=(0, 14)) is True

    def test_non_overlapping_polygon(self):
        # A square well above the water stability region
        polygon = [(5, 3.0), (5, 4.0), (9, 4.0), (9, 3.0)]
        assert check_overlap(polygon, H_LINE, O_LINE, range_of_pH=(0, 14)) is False

    def test_non_overlapping_below(self):
        # A square well below the water stability region
        polygon = [(5, -3.0), (5, -2.0), (9, -2.0), (9, -3.0)]
        assert check_overlap(polygon, H_LINE, O_LINE, range_of_pH=(0, 14)) is False

    def test_partial_overlap(self):
        # Touches just the edge of the water stability region
        # At pH=0, H line is at V=0, O line is at V=1.23
        polygon = [(-1, 1.0), (-1, 1.5), (1, 1.5), (1, 1.0)]
        assert check_overlap(polygon, H_LINE, O_LINE, range_of_pH=(0, 14)) is True


class TestCheckOverlapWaterStability:
    def test_inside_window(self):
        # pH=7, V=0.5 should be inside the standard water window
        # H line at pH=7: V = -0.0591*7 = -0.4137
        # O line at pH=7: V = -0.0591*7 + 1.23 = 0.8163
        assert check_overlap_with_water_stability_lines(7, 0.5) is True

    def test_above_window(self):
        assert check_overlap_with_water_stability_lines(7, 2.0) is False

    def test_below_window(self):
        assert check_overlap_with_water_stability_lines(7, -2.0) is False

    def test_on_h_line(self):
        # Exactly on the H line at pH=0
        assert check_overlap_with_water_stability_lines(0, 0.0) is True

    def test_on_o_line(self):
        # Exactly on the O line at pH=0
        assert check_overlap_with_water_stability_lines(0, 1.23) is True

    def test_extended_window_oer_shift(self):
        # With OER_shift=0.5, the O line moves up by 0.5V
        # At pH=7, O line normally at 0.8163, shifted to 1.3163
        assert check_overlap_with_water_stability_lines(7, 1.2, OER_shift=0.5) is True
        assert check_overlap_with_water_stability_lines(7, 1.2, OER_shift=0.0) is False

    def test_extended_window_her_shift(self):
        # With HER_shift=-0.3, the H line moves down by 0.3V
        # At pH=7, H line normally at -0.4137, shifted to -0.7137
        assert check_overlap_with_water_stability_lines(7, -0.6, HER_shift=-0.3) is True
        assert check_overlap_with_water_stability_lines(7, -0.6, HER_shift=0.0) is False


class TestCheckOverlapExtended:
    def test_in_extended_only_region(self):
        # Point in the HER extended region (between standard H line and shifted H line)
        # At pH=7: H line at -0.4137, extended at -0.7137
        assert check_overlap_with_extended_water_stability_lines(7, -0.6) is True

    def test_in_standard_window(self):
        # Point clearly inside the standard window, not in extended region
        assert check_overlap_with_extended_water_stability_lines(7, 0.5) is False

    def test_outside_everything(self):
        assert check_overlap_with_extended_water_stability_lines(7, -3.0) is False

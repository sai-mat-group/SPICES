"""Tests for statistics module."""

import numpy as np
import pandas as pd
import pytest

from spices.screening.statistics import (
    get_stability_bool,
    _get_pH_range_bool,
    count_phases_with_voltage_stability,
)


class TestGetStabilityBool:
    def test_string_list_nonempty(self):
        assert get_stability_bool("[(7.0, 0.5), (7.5, 0.8)]") is True

    def test_string_list_empty(self):
        assert get_stability_bool("[]") is False

    def test_actual_list_nonempty(self):
        assert get_stability_bool([(7.0, 0.5)]) is True

    def test_actual_list_empty(self):
        assert get_stability_bool([]) is False

    def test_nan(self):
        assert get_stability_bool(float("nan")) is False

    def test_none(self):
        assert get_stability_bool(None) is False

    def test_bool_true(self):
        assert get_stability_bool(True) is True

    def test_bool_false(self):
        assert get_stability_bool(False) is False

    def test_zero(self):
        assert get_stability_bool(0) is False


class TestGetpHRangeBool:
    def test_full_range(self):
        assert _get_pH_range_bool("(7.0, 8.5)", 1.5) is True

    def test_partial_range(self):
        assert _get_pH_range_bool("(7.0, 8.0)", 1.5) is False

    def test_non_string(self):
        assert _get_pH_range_bool(float("nan"), 1.5) is False

    def test_exact_threshold(self):
        assert _get_pH_range_bool("(7.0, 8.5)", 1.5) is True


class TestCountPhasesWithVoltageStability:
    def _make_test_data(self):
        stability_df = pd.DataFrame({
            "MP id": ["mp-1", "mp-2", "mp-3", "mp-4"],
        })
        voltage_df = pd.DataFrame({
            "Charged MP id": ["mp-1", "mp-2"],
            "Discharged MP id": ["mp-2", "mp-3"],
            "Plot number": [0, 1],
            "pH range metastability criterion 4": [
                "(7.0, 8.5)", "(7.0, 8.0)",
            ],
            "Total metastability area fraction": [0.9, 0.5],
        })
        return voltage_df, stability_df

    def test_unfiltered_counts(self):
        vdf, sdf = self._make_test_data()
        result = count_phases_with_voltage_stability(vdf, sdf, 4)
        # MP IDs: mp-1 (charged), mp-2 (both), mp-3 (discharged) = 3 unique
        assert result["total_phases"] == 3
        assert result["total_plots"] == 2

    def test_pH_filter(self):
        vdf, sdf = self._make_test_data()
        result = count_phases_with_voltage_stability(
            vdf, sdf, 4, pH_window=(7.0, 8.5),
        )
        # Only first row has pH range of 1.5
        assert result["total_phases"] == 2  # mp-1 and mp-2
        assert result["total_plots"] == 1

    def test_area_filter(self):
        vdf, sdf = self._make_test_data()
        result = count_phases_with_voltage_stability(
            vdf, sdf, 4, min_area_fraction=0.8,
        )
        # Only first row has area >= 0.8
        assert result["total_plots"] == 1

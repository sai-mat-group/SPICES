"""Tests for the optional multi-criteria selection module."""

from spices.pourbaix.selection_criteria import (
    check_solid_entry_all_criteria,
    ALL_SELECTION_CRITERIA,
    SEAWATER_COLUMNS_ALL,
)


class TestConstants:
    def test_all_selection_criteria(self):
        assert ALL_SELECTION_CRITERIA == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_seawater_columns_all_count(self):
        assert len(SEAWATER_COLUMNS_ALL) == 9

    def test_seawater_columns_all_stability_present(self):
        assert "Seawater stability" in SEAWATER_COLUMNS_ALL

    def test_seawater_columns_all_criteria_present(self):
        for i in range(1, 9):
            col = f"Seawater metastability (Criterion {i})"
            assert col in SEAWATER_COLUMNS_ALL


class TestCheckSolidEntryAllCriteria:
    def test_callable(self):
        assert callable(check_solid_entry_all_criteria)

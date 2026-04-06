"""Tests for seawater metastability functions."""

from unittest.mock import MagicMock
import pytest

from spices.pourbaix.metastability import (
    is_metastable_phase_stable_in_seawater_search_by_mp_id,
    is_metastable_phase_stable_in_seawater_seach_by_mp_id,
)


def _make_solid_entry(entry_id, num_atoms=4):
    """Create a mock solid Pourbaix entry."""
    entry = MagicMock()
    entry.entry_id = entry_id
    entry.phase_type = "Solid"
    entry.name = f"solid-{entry_id}"
    entry.composition = MagicMock()
    entry.composition.num_atoms = num_atoms
    return entry


def _make_ion_entry(entry_id):
    """Create a mock ion Pourbaix entry."""
    entry = MagicMock()
    entry.entry_id = entry_id
    entry.phase_type = "Ion"
    entry.name = f"ion-{entry_id}"
    return entry


class TestIsMetastableInSeawaterSearchByMpId:
    def test_bare_seawater_pbx_required(self):
        sea_pbx = MagicMock()
        with pytest.raises(ValueError, match="bare_seawater_pbx is required"):
            is_metastable_phase_stable_in_seawater_search_by_mp_id(
                "mp-123", sea_pbx, bare_seawater_pbx=None,
            )

    def test_mp_id_not_in_diagram_returns_false(self):
        sea_pbx = MagicMock()
        sea_pbx.unprocessed_entries = [
            _make_solid_entry("mp-100-GGA"),
        ]
        bare_pbx = MagicMock()

        result = is_metastable_phase_stable_in_seawater_search_by_mp_id(
            "mp-999", sea_pbx, bare_seawater_pbx=bare_pbx,
        )
        assert result is False

    def test_mp_id_not_in_diagram_all_criteria_returns_dict(self):
        sea_pbx = MagicMock()
        sea_pbx.unprocessed_entries = [
            _make_solid_entry("mp-100-GGA"),
        ]
        bare_pbx = MagicMock()

        result = is_metastable_phase_stable_in_seawater_search_by_mp_id(
            "mp-999", sea_pbx,
            bare_seawater_pbx=bare_pbx,
            selection_criterion="all",
        )
        assert isinstance(result, dict)
        assert all(v is False for v in result.values())
        assert set(result.keys()) == {str(i) for i in range(1, 9)}

    def test_mp_id_not_in_diagram_area_mode_returns_empty_list(self):
        sea_pbx = MagicMock()
        sea_pbx.unprocessed_entries = [
            _make_solid_entry("mp-100-GGA"),
        ]
        bare_pbx = MagicMock()

        result = is_metastable_phase_stable_in_seawater_search_by_mp_id(
            "mp-999", sea_pbx,
            output_metastability_area=True,
            bare_seawater_pbx=bare_pbx,
        )
        assert isinstance(result, list)
        assert result == []

    def test_mp_id_not_in_diagram_area_mode_all_criteria_returns_empty_dict(self):
        sea_pbx = MagicMock()
        sea_pbx.unprocessed_entries = [
            _make_solid_entry("mp-100-GGA"),
        ]
        bare_pbx = MagicMock()

        result = is_metastable_phase_stable_in_seawater_search_by_mp_id(
            "mp-999", sea_pbx,
            output_metastability_area=True,
            bare_seawater_pbx=bare_pbx,
            selection_criterion="all",
        )
        assert isinstance(result, dict)
        assert all(isinstance(v, list) for v in result.values())
        assert all(len(v) == 0 for v in result.values())

    def test_backward_compat_alias_exists(self):
        assert (
            is_metastable_phase_stable_in_seawater_search_by_mp_id
            is is_metastable_phase_stable_in_seawater_seach_by_mp_id
        )

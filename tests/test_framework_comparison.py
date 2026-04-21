"""Tests for framework comparison module."""

import pytest
import pandas as pd
from unittest.mock import MagicMock

from spices.screening.framework_comparison import (
    FrameworkStructure,
    framework_set_intersection,
    multi_ion_common_frameworks,
    ion_selective_frameworks,
    filter_plateaus_by_pH_and_area,
    formula_to_mathtext,
    _get_working_ion_fraction,
    bfs_components,
    ION_COLORS,
)
from spices.intercalation.structure import DEFAULT_MATCHER


class TestFrameworkStructure:
    def test_hash_by_mp_id(self):
        """Hash is derived from mp_id; equality falls back to identity."""
        mock_struct = MagicMock()
        mock_struct.copy.return_value = mock_struct
        mock_struct.composition.reduced_formula = "CaFeO2"
        mock_struct.elements = []
        mock_struct.remove_oxidation_states = MagicMock()

        fw = FrameworkStructure("mp-1", mock_struct, "Ca", is_expt=False)
        assert hash(fw) == hash("mp-1")

    def test_uses_attempt_supercell_matcher(self):
        """Verify DEFAULT_MATCHER has attempt_supercell=True."""
        assert DEFAULT_MATCHER._supercell is True

    def test_is_expt_required(self):
        """is_expt has no default; omitting it raises TypeError."""
        mock_struct = MagicMock()
        mock_struct.copy.return_value = mock_struct
        mock_struct.composition.reduced_formula = "FeO2"
        mock_struct.elements = []
        mock_struct.remove_oxidation_states = MagicMock()

        with pytest.raises(TypeError):
            FrameworkStructure("mp-1", mock_struct, "Ca")

    def test_is_expt_set(self):
        mock_struct = MagicMock()
        mock_struct.copy.return_value = mock_struct
        mock_struct.composition.reduced_formula = "FeO2"
        mock_struct.elements = []
        mock_struct.remove_oxidation_states = MagicMock()

        fw = FrameworkStructure("mp-1", mock_struct, "Ca", is_expt=True)
        assert fw.is_expt is True


class TestSetOperations:
    def _make_fw(self, mp_id, formula):
        """Create a mock FrameworkStructure with given formula hash."""
        fw = MagicMock(spec=FrameworkStructure)
        fw.mp_id = mp_id
        fw.framework_reduced_formula = formula
        fw.__hash__ = MagicMock(return_value=hash(formula))
        fw.__eq__ = MagicMock(
            side_effect=lambda other: (
                other.framework_reduced_formula == formula
            )
        )
        return fw

    def test_intersection_finds_common(self):
        fw_a = self._make_fw("mp-1", "FeO2")
        fw_b = self._make_fw("mp-2", "FeO2")
        fw_c = self._make_fw("mp-3", "MnO2")

        result = framework_set_intersection(
            {fw_a}, {fw_a}, {fw_b}, {fw_c},
        )
        # fw_a and fw_b have same formula → should match
        assert len(result) >= 1

    def test_intersection_no_common(self):
        fw_a = self._make_fw("mp-1", "FeO2")
        fw_b = self._make_fw("mp-2", "TiO2")

        result = framework_set_intersection(
            {fw_a}, set(), {fw_b}, set(),
        )
        assert len(result) == 0

    def test_ion_selective_removes_common(self):
        fw_a = self._make_fw("mp-1", "FeO2")
        fw_b = self._make_fw("mp-2", "FeO2")
        fw_c = self._make_fw("mp-3", "MnO2")

        # fw_a (FeO2) is shared with fw_b, fw_c (MnO2) is unique
        result = ion_selective_frameworks(
            {fw_a, fw_c}, set(),
            [{fw_b}], [set()],
        )
        # Only MnO2 should remain
        assert fw_c in result
        assert fw_a not in result


class TestFilterPlateaus:
    def test_empty_df(self):
        df = pd.DataFrame(columns=["Plot number", "pH range metastability criterion 4",
                                    "Total metastability area fraction"])
        result = filter_plateaus_by_pH_and_area(df)
        assert len(result) == 0

    def test_filters_by_pH_and_area(self):
        df = pd.DataFrame({
            "Plot number": [0, 0, 1, 1],
            "pH range metastability criterion 4": [
                "(7.0, 8.5)", "(7.0, 8.5)", "(7.0, 8.0)", "(7.0, 8.0)",
            ],
            "Total metastability area fraction": [0.9, 0.9, 0.9, 0.9],
        })
        result = filter_plateaus_by_pH_and_area(df, pH_range=(7.0, 8.5))
        # Only plot 0 has full pH coverage (1.5 width)
        assert set(result["Plot number"].unique()) == {0}

    def test_filters_by_area(self):
        df = pd.DataFrame({
            "Plot number": [0, 1],
            "pH range metastability criterion 4": [
                "(7.0, 8.5)", "(7.0, 8.5)",
            ],
            "Total metastability area fraction": [0.9, 0.5],
        })
        result = filter_plateaus_by_pH_and_area(
            df, min_area_fraction=0.8,
        )
        assert set(result["Plot number"].unique()) == {0}


class TestFormulaToMathtext:
    def test_basic(self):
        result = formula_to_mathtext("Fe2O3")
        assert "Fe_{2}" in result
        assert "O_{3}" in result
        assert result.startswith("$")
        assert result.endswith("$")

    def test_single_element(self):
        result = formula_to_mathtext("Fe")
        assert result == "$Fe$"

    def test_parentheses(self):
        result = formula_to_mathtext("Fe2(SO4)3")
        assert "_{3}" in result


class TestWorkingIonFraction:
    def test_full_intercalation(self):
        # CaFeO2 has 1 Ca per FeO2 framework unit
        frac = _get_working_ion_fraction("CaFeO2", "FeO2", "Ca")
        assert abs(frac - 1.0) < 1e-6

    def test_no_working_ion(self):
        frac = _get_working_ion_fraction("FeO2", "FeO2", "Ca")
        assert frac == 0

    def test_partial_intercalation(self):
        # Ca2Fe2O4 = Ca(FeO2) has 1 Ca per FeO2 framework
        frac = _get_working_ion_fraction("Ca2Fe2O4", "FeO2", "Ca")
        assert abs(frac - 1.0) < 1e-6


class TestBfsComponents:
    def test_single_component(self):
        adj = {0: [1], 1: [0, 2], 2: [1]}
        components = bfs_components(adj, 3)
        assert len(components) == 1
        assert sorted(components[0]) == [0, 1, 2]

    def test_two_components(self):
        adj = {0: [1], 1: [0], 2: [3], 3: [2]}
        components = bfs_components(adj, 4)
        assert len(components) == 2

    def test_isolated_nodes(self):
        adj = {0: [], 1: [], 2: []}
        components = bfs_components(adj, 3)
        assert len(components) == 3


class TestIonColors:
    def test_all_elements_have_colors(self):
        for element in ["Li", "Na", "K", "Mg", "Ca"]:
            assert element in ION_COLORS

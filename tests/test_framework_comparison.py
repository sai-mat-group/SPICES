"""Tests for graph-based framework comparison module."""

import pytest
import pandas as pd
from unittest.mock import MagicMock

from spices.screening.framework_comparison import (
    FrameworkStructure,
    filter_plateaus_by_pH_and_area,
    formula_to_mathtext,
    bfs_components,
    reconnect_isolated_frameworks,
    save_graph_figures,
    make_networkx_graph,
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


class TestFilterPlateaus:
    def test_empty_df(self):
        df = pd.DataFrame(columns=[
            "Plot number",
            "pH range metastability criterion 4",
            "Total metastability area fraction",
        ])
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


class TestReconnectIsolated:
    def _fake_fw(self, mp_id, ion_symbol):
        fw = MagicMock(spec=FrameworkStructure)
        fw.mp_id = mp_id
        fw.framework_reduced_formula = f"FW_{mp_id}"
        # working_ion exposes .symbol
        wi = MagicMock()
        wi.symbol = ion_symbol
        fw.working_ion = wi
        return fw

    def test_isolated_reconnected_via_charged_partner(self):
        fws = [
            self._fake_fw("mp-A", "Li"),
            self._fake_fw("mp-B", "Li"),
            self._fake_fw("mp-X", "Na"),
        ]
        # A and B are isolated; df says they are a charged-discharged pair
        adjacency = {0: [], 1: [], 2: []}
        df_dict = {
            "Li": pd.DataFrame({
                "Charged MP id": ["mp-A"],
                "Discharged MP id": ["mp-B"],
            }),
            "Na": pd.DataFrame({
                "Charged MP id": ["mp-X"],
                "Discharged MP id": ["mp-X"],
            }),
        }
        adj = reconnect_isolated_frameworks(adjacency, fws, df_dict)
        assert 1 in adj[0]
        assert 0 in adj[1]

    def test_isolated_with_no_partner_logs_warning(self, caplog):
        fws = [self._fake_fw("mp-Z", "Li")]
        adjacency = {0: []}
        df_dict = {
            "Li": pd.DataFrame({
                "Charged MP id": ["mp-other"],
                "Discharged MP id": ["mp-third"],
            }),
        }
        with caplog.at_level("WARNING"):
            adj = reconnect_isolated_frameworks(adjacency, fws, df_dict)
        assert adj[0] == []  # still isolated
        assert any("not found" in rec.message for rec in caplog.records)


class TestSaveGraphFiguresMultiFormulaRaise:
    def test_raises_when_component_has_multiple_formulae(self, tmp_path):
        """Component-formula check (BUG 2 fix)."""
        # Build a NetworkX graph with two nodes whose framework formulas differ
        n1 = MagicMock(spec=FrameworkStructure)
        n1.mp_id = "mp-1"
        n1.framework_reduced_formula = "FeO2"
        n1.struct_reduced_formula = "LiFeO2"
        n1.is_expt = False
        wi1 = MagicMock(); wi1.symbol = "Li"
        n1.working_ion = wi1

        n2 = MagicMock(spec=FrameworkStructure)
        n2.mp_id = "mp-2"
        n2.framework_reduced_formula = "MnO2"  # different formula!
        n2.struct_reduced_formula = "LiMnO2"
        n2.is_expt = False
        wi2 = MagicMock(); wi2.symbol = "Li"
        n2.working_ion = wi2

        import networkx as nx
        G = nx.Graph()
        G.add_node(n1)
        G.add_node(n2)
        G.add_edge(n1, n2)

        with pytest.raises(ValueError, match="different framework reduced formulae"):
            save_graph_figures(
                [G], [n1, n2], df_dict={},
                save_dir=str(tmp_path / "graphs"),
            )


class TestIonColors:
    def test_all_elements_have_colors(self):
        for element in ["Li", "Na", "K", "Mg", "Ca"]:
            assert element in ION_COLORS

"""Tests for graph-based framework comparison module."""

import json
import pickle
import pytest
import networkx as nx
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
    build_structure_voltage_records,
    ION_COLORS,
    NODE_SIZE,
    NODE_BORDER_WIDTH,
    _MergedFrameworkNode,
    _collapse_graph_by_mpid,
    _save_cached_state,
    _load_cached_state,
    CACHE_SCHEMA_VERSION,
)
from spices.intercalation.structure import DEFAULT_MATCHER


class _Ion:
    """Picklable stand-in for ``pymatgen.core.Element`` (just the symbol)."""

    def __init__(self, symbol):
        self.symbol = symbol


class _PicklableFakeFW:
    """Picklable stand-in for ``FrameworkStructure`` exposing the same attrs."""

    def __init__(self, mp_id, ion_symbol, is_expt=False, framework_formula=None):
        self.mp_id = mp_id
        self.framework_reduced_formula = framework_formula or f"FW_{mp_id}"
        self.struct_reduced_formula = framework_formula or f"FW_{mp_id}"
        self.is_expt = is_expt
        self.working_ion = _Ion(ion_symbol)


def _fake_fw(mp_id, ion_symbol, is_expt=False, framework_formula=None):
    """Build a picklable stand-in FrameworkStructure for graph tests."""
    return _PicklableFakeFW(mp_id, ion_symbol, is_expt, framework_formula)


class _FakeComposition:
    """Composition stub: ``get_atomic_fraction(ion)`` reads a {symbol: frac}."""

    def __init__(self, fracs):
        self._fracs = fracs

    def get_atomic_fraction(self, ion):
        return self._fracs.get(getattr(ion, "symbol", ion), 0.0)


class _RecNode:
    """Fake FrameworkStructure for build_structure_voltage_records tests.

    ``frac`` is the node's OWN working-ion atomic fraction (0 = empty host);
    asked for any other ion the composition returns 0.
    """

    def __init__(self, mp_id, ion_symbol, frac, is_expt=False, formula=None):
        self.mp_id = mp_id
        self.working_ion = _Ion(ion_symbol)
        self.is_expt = is_expt
        self.struct_reduced_formula = formula or f"S_{mp_id}"
        self.struct = MagicMock()
        self.struct.composition = _FakeComposition({ion_symbol: frac})


def _voltage_df(rows):
    """rows: tuples ``(charged_mpid, discharged_mpid, V, area, pH_str)``."""
    return pd.DataFrame({
        "Charged MP id": [r[0] for r in rows],
        "Discharged MP id": [r[1] for r in rows],
        "V vs SHE (V)": [r[2] for r in rows],
        "Total metastability area fraction": [r[3] for r in rows],
        "pH range metastability criterion 4": [r[4] for r in rows],
    })


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
        assert r"\mathrm{Fe}_{2}" in result
        assert r"\mathrm{O}_{3}" in result
        assert result.startswith("$")
        assert result.endswith("$")

    def test_single_element(self):
        result = formula_to_mathtext("Fe")
        assert result == r"$\mathrm{Fe}$"

    def test_parentheses(self):
        result = formula_to_mathtext("Fe2(SO4)3")
        assert "_{3}" in result
        assert r"\mathrm{Fe}" in result
        assert r"\mathrm{SO}" in result


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


class TestNodeStylingConstants:
    def test_node_size(self):
        assert NODE_SIZE == 4500

    def test_border_width(self):
        assert NODE_BORDER_WIDTH == 1.0


class TestCollapseByMpid:
    def test_singletons_unchanged(self):
        """No duplicate mp_ids: collapsed graph keeps all nodes."""
        import networkx as nx
        a = _fake_fw("mp-A", "Li")
        b = _fake_fw("mp-B", "Li")
        G = nx.Graph()
        G.add_node(a)
        G.add_node(b)
        G.add_edge(a, b)

        edge_groups = {
            "diff_ion": [], "same_ion": [(a, b)], "no_voltage": [],
            "pH_filtered": [], "pH_area_filtered": [],
        }
        G_d, eg_d = _collapse_graph_by_mpid(G, edge_groups)
        assert len(G_d.nodes) == 2
        assert len(G_d.edges) == 1
        # Singleton node colors are ion colors
        colors = {G_d.nodes[n]["color"] for n in G_d.nodes}
        assert colors == {ION_COLORS["Li"]}

    def test_merge_same_mpid_to_lightgrey(self):
        """Two FrameworkStructures sharing mp_id collapse into one node."""
        import networkx as nx
        a_li = _fake_fw("mp-X", "Li", is_expt=False)
        a_na = _fake_fw("mp-X", "Na", is_expt=False)
        c = _fake_fw("mp-C", "Li", is_expt=False)
        G = nx.Graph()
        G.add_node(a_li)
        G.add_node(a_na)
        G.add_node(c)
        G.add_edge(a_li, a_na)  # diff-ion edge between the two mp-X
        G.add_edge(a_li, c)     # same-ion edge between mp-X and mp-C

        edge_groups = {
            "diff_ion": [(a_li, a_na)],
            "same_ion": [(a_li, c)],
            "no_voltage": [],
            "pH_filtered": [],
            "pH_area_filtered": [],
        }
        G_d, eg_d = _collapse_graph_by_mpid(G, edge_groups)

        # 3 originals -> 2 in the display graph (mp-X merged)
        assert len(G_d.nodes) == 2
        # The mp-X diff-ion edge becomes a self-loop and is dropped;
        # the mp-X--mp-C edge survives.
        assert len(G_d.edges) == 1
        assert eg_d["same_ion"] == [edge for edge in G_d.edges]
        assert eg_d["diff_ion"] == []

        # Merged node color is light-grey
        merged = next(
            n for n in G_d.nodes if isinstance(n, _MergedFrameworkNode)
        )
        assert G_d.nodes[merged]["color"] == "lightgrey"
        assert merged.working_ions == {"Li", "Na"}
        assert merged.mp_id == "mp-X"

    def test_edge_priority(self):
        """Parallel edges between same display-pair keep highest-priority."""
        import networkx as nx
        a_li = _fake_fw("mp-X", "Li")
        a_na = _fake_fw("mp-X", "Na")
        b_li = _fake_fw("mp-Y", "Li")
        b_na = _fake_fw("mp-Y", "Na")
        G = nx.Graph()
        for n in (a_li, a_na, b_li, b_na):
            G.add_node(n)
        # Two parallel post-collapse edges between {mp-X} and {mp-Y}:
        # one classified as same_ion (Li side), one as pH_area_filtered (Na side).
        # The Na pH_area edge should win.
        G.add_edge(a_li, b_li)
        G.add_edge(a_na, b_na)

        edge_groups = {
            "diff_ion": [],
            "same_ion": [(a_li, b_li)],
            "no_voltage": [],
            "pH_filtered": [],
            "pH_area_filtered": [(a_na, b_na)],
        }
        _, eg_d = _collapse_graph_by_mpid(G, edge_groups)
        assert len(eg_d["pH_area_filtered"]) == 1
        assert eg_d["same_ion"] == []

    def test_raises_on_mixed_is_expt(self):
        """Same mp_id with mixed is_expt is a data inconsistency -> ValueError."""
        import networkx as nx
        a_expt = _fake_fw("mp-X", "Li", is_expt=True)
        a_theo = _fake_fw("mp-X", "Na", is_expt=False)
        G = nx.Graph()
        G.add_node(a_expt)
        G.add_node(a_theo)

        edge_groups = {k: [] for k in (
            "diff_ion", "same_ion", "no_voltage",
            "pH_filtered", "pH_area_filtered",
        )}
        with pytest.raises(ValueError, match="inconsistent is_expt"):
            _collapse_graph_by_mpid(G, edge_groups)


class TestCacheRoundtrip:
    def test_save_load_roundtrip(self, tmp_path):
        cache_path = str(tmp_path / "step4_cache.pkl")
        meta_path = str(tmp_path / "step4_cache.meta.json")

        fw1 = _fake_fw("mp-A", "Li")
        fw2 = _fake_fw("mp-B", "Na")
        combined_fw = [fw1, fw2]
        adjacency = {0: [1], 1: [0]}
        stage3_dirs = {"Li": "/tmp/li", "Na": "/tmp/na"}

        _save_cached_state(
            cache_path, meta_path, combined_fw, adjacency,
            stage3_dirs, "criterion 4", reconnect_isolated=False,
        )
        # Sidecar holds the expected fields
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["schema_version"] == CACHE_SCHEMA_VERSION
        assert meta["stage3_dirs_keys"] == ["Li", "Na"]
        assert meta["sheet_name"] == "criterion 4"
        assert meta["reconnect_isolated"] is False
        assert meta["n_frameworks"] == 2

        loaded = _load_cached_state(
            cache_path, meta_path,
            stage3_dirs, "criterion 4", reconnect_isolated=False,
        )
        assert loaded is not None
        combined_fw_loaded, adjacency_loaded = loaded
        assert [fw.mp_id for fw in combined_fw_loaded] == ["mp-A", "mp-B"]
        assert adjacency_loaded == adjacency

    def test_load_detects_stage3_keys_drift(self, tmp_path):
        cache_path = str(tmp_path / "step4_cache.pkl")
        meta_path = str(tmp_path / "step4_cache.meta.json")
        fw1 = _fake_fw("mp-A", "Li")
        _save_cached_state(
            cache_path, meta_path, [fw1], {0: []},
            {"Li": "/tmp/li"}, "criterion 4", reconnect_isolated=False,
        )

        # Different ion order -> miss
        loaded = _load_cached_state(
            cache_path, meta_path,
            {"Na": "/tmp/na", "Li": "/tmp/li"},
            "criterion 4", reconnect_isolated=False,
        )
        assert loaded is None

    def test_load_detects_legacy_adjacency_only_cache(self, tmp_path):
        cache_path = str(tmp_path / "step4_cache.pkl")
        meta_path = str(tmp_path / "step4_cache.meta.json")
        # Write a legacy "adjacency only" cache (no combined_fw key) but a
        # current-schema sidecar.  Should be rejected.
        with open(cache_path, "wb") as f:
            pickle.dump({0: [1], 1: [0]}, f)  # bare dict, not bundle
        with open(meta_path, "w") as f:
            json.dump({
                "schema_version": CACHE_SCHEMA_VERSION,
                "stage3_dirs_keys": ["Li"],
                "sheet_name": "criterion 4",
                "reconnect_isolated": False,
                "mpids": ["mp-A"],
                "n_frameworks": 1,
                "n_adjacency_edges": 1,
            }, f)

        loaded = _load_cached_state(
            cache_path, meta_path,
            {"Li": "/tmp/li"}, "criterion 4", reconnect_isolated=False,
        )
        assert loaded is None

    def test_load_missing_returns_none(self, tmp_path):
        loaded = _load_cached_state(
            str(tmp_path / "absent.pkl"),
            str(tmp_path / "absent.meta.json"),
            {"Li": "/tmp/li"}, "criterion 4", reconnect_isolated=False,
        )
        assert loaded is None


class TestBuildStructureVoltageRecords:
    def test_qualify_solid_hashed_excluded_boundary(self):
        host = _RecNode("mp-host", "Li", 0.0, formula="FePO4")
        d1 = _RecNode("mp-d1", "Li", 0.2)
        d2 = _RecNode("mp-d2", "Li", 0.3)
        d3 = _RecNode("mp-d3", "Li", 0.4)
        d4 = _RecNode("mp-d4", "Li", 0.5)
        G = nx.Graph()
        G.add_edge(host, d1)
        G.add_edge(host, d2)
        G.add_edge(host, d3)
        G.add_edge(host, d4)
        df = _voltage_df([
            ("mp-host", "mp-d1", 0.30, 0.91, "(7.0, 8.5)"),  # solid
            ("mp-host", "mp-d2", 0.32, 0.75, "(7.0, 8.5)"),  # hashed
            ("mp-host", "mp-d3", 0.34, 0.65, "(7.0, 8.5)"),  # excluded (<0.70)
            ("mp-host", "mp-d4", 0.36, 0.70, "(7.0, 8.5)"),  # excluded (==0.70)
        ])
        rec = build_structure_voltage_records([G], {"Li": df})
        assert len(rec["components"]) == 1
        c = rec["components"][0]
        assert c["route"] == "ion_selective"
        assert c["n_ions"] == 1
        assert sorted(m["status"] for m in c["markers"]) == ["hashed", "solid"]
        assert c["label_key"] == "FePO4\nmp-host"  # empty host (frac 0)

    def test_multi_ion_with_faded_partner(self):
        li_host = _RecNode("mp-lh", "Li", 0.0, formula="FePO4")
        li_d = _RecNode("mp-ld", "Li", 0.3)
        na_host = _RecNode("mp-nh", "Na", 0.05)
        na_d = _RecNode("mp-nd", "Na", 0.4)
        G = nx.Graph()
        G.add_edge(li_host, li_d)       # Li qualifies
        G.add_edge(na_host, na_d)       # Na non-qualifying -> faded
        G.add_edge(li_host, na_host)    # diff-ion (shared skeleton)
        df_dict = {
            "Li": _voltage_df([("mp-lh", "mp-ld", 0.3, 0.90, "(7.0, 8.5)")]),
            "Na": _voltage_df([("mp-nh", "mp-nd", 0.2, 0.50, "(7.0, 8.5)")]),
        }
        rec = build_structure_voltage_records([G], df_dict)
        c = rec["components"][0]
        assert c["route"] == "multi_ion"
        assert c["n_ions"] == 2
        by_ion = {m["ion"]: m for m in c["markers"]}
        assert by_ion["Li"]["status"] == "solid"
        assert by_ion["Na"]["status"] == "faded"
        assert by_ion["Na"]["covers_pH"] is True
        assert c["label_key"] == "FePO4\nmp-lh"  # li_host frac 0 is the min

    def test_label_uses_each_nodes_own_working_ion(self):
        li_lo = _RecNode("mp-li-lo", "Li", 0.1, formula="HOST")
        li_hi = _RecNode("mp-li-hi", "Li", 0.5)
        na_lo = _RecNode("mp-na-lo", "Na", 0.3)
        na_hi = _RecNode("mp-na-hi", "Na", 0.6)
        G = nx.Graph()
        G.add_edge(li_lo, li_hi)
        G.add_edge(na_lo, na_hi)
        G.add_edge(li_lo, na_lo)
        df_dict = {
            "Li": _voltage_df([("mp-li-lo", "mp-li-hi", 0.3, 0.9, "(7.0, 8.5)")]),
            "Na": _voltage_df([("mp-na-lo", "mp-na-hi", 0.2, 0.9, "(7.0, 8.5)")]),
        }
        rec = build_structure_voltage_records([G], df_dict)
        # Li node at 0.1 is the true minimum; using one global ion would make
        # the Na nodes read fraction 0 and wrongly win.
        assert rec["components"][0]["label_key"] == "HOST\nmp-li-lo"

    def test_label_tiebreak_experimental_then_mpid(self, caplog):
        li_host = _RecNode("mp-a", "Li", 0.0, is_expt=False, formula="HOST")
        li_d = _RecNode("mp-li-d", "Li", 0.3)
        na_host = _RecNode("mp-z", "Na", 0.0, is_expt=True, formula="HOST")
        na_d = _RecNode("mp-na-d", "Na", 0.3)
        G = nx.Graph()
        G.add_edge(li_host, li_d)
        G.add_edge(na_host, na_d)
        G.add_edge(li_host, na_host)
        df_dict = {
            "Li": _voltage_df([("mp-a", "mp-li-d", 0.3, 0.9, "(7.0, 8.5)")]),
            "Na": _voltage_df([("mp-z", "mp-na-d", 0.2, 0.9, "(7.0, 8.5)")]),
        }
        with caplog.at_level("WARNING"):
            rec = build_structure_voltage_records([G], df_dict)
        # Tie at frac 0; experimental mp-z beats theoretical (lower) mp-a.
        assert rec["components"][0]["label_mpid"] == "mp-z"
        assert any("tie" in r.message.lower() for r in caplog.records)

    def test_component_dropped_when_none_qualify(self):
        host = _RecNode("mp-h", "Li", 0.0, formula="X")
        d = _RecNode("mp-d", "Li", 0.3)
        G = nx.Graph()
        G.add_edge(host, d)
        df = _voltage_df([("mp-h", "mp-d", 0.3, 0.65, "(7.0, 8.5)")])
        rec = build_structure_voltage_records([G], {"Li": df})
        assert rec["components"] == []

    def test_nan_area_excluded_and_json_safe(self):
        host = _RecNode("mp-h", "Li", 0.0, formula="X")
        good = _RecNode("mp-g", "Li", 0.3)
        bad = _RecNode("mp-b", "Li", 0.4)
        G = nx.Graph()
        G.add_edge(host, good)
        G.add_edge(host, bad)
        df = _voltage_df([
            ("mp-h", "mp-g", 0.3, 0.90, "(7.0, 8.5)"),
            ("mp-h", "mp-b", 0.3, float("nan"), "(7.0, 8.5)"),
        ])
        rec = build_structure_voltage_records([G], {"Li": df})
        markers = rec["components"][0]["markers"]
        assert [m["discharged_mpid"] for m in markers] == ["mp-g"]
        json.dumps(rec, allow_nan=False)  # must not raise

    def test_bad_ph_component_skipped_not_fatal(self):
        bad_h = _RecNode("mp-bh", "Li", 0.0, formula="BAD")
        bad_d = _RecNode("mp-bd", "Li", 0.3)
        g_bad = nx.Graph()
        g_bad.add_edge(bad_h, bad_d)
        good_h = _RecNode("mp-gh", "Li", 0.0, formula="GOOD")
        good_d = _RecNode("mp-gd", "Li", 0.3)
        g_good = nx.Graph()
        g_good.add_edge(good_h, good_d)
        df = _voltage_df([
            ("mp-bh", "mp-bd", 0.3, 0.9, float("nan")),  # NaN pH -> raises
            ("mp-gh", "mp-gd", 0.3, 0.9, "(7.0, 8.5)"),
        ])
        rec = build_structure_voltage_records([g_bad, g_good], {"Li": df})
        assert [c["label_key"] for c in rec["components"]] == ["GOOD\nmp-gh"]

    def test_schema_keys_present(self):
        host = _RecNode("mp-h", "Li", 0.0, formula="X")
        d = _RecNode("mp-d", "Li", 0.3, is_expt=True)
        G = nx.Graph()
        G.add_edge(host, d)
        df = _voltage_df([("mp-h", "mp-d", 0.3, 0.9, "(7.0, 8.5)")])
        rec = build_structure_voltage_records([G], {"Li": df})
        assert rec["schema_version"] == 1
        assert rec["qualify_area"] == 0.70 and rec["hatch_area"] == 0.80
        c = rec["components"][0]
        assert set(c) >= {"component_id", "route", "n_ions", "label_formula",
                          "label_mpid", "label_key", "markers"}
        m = c["markers"][0]
        assert set(m) >= {"ion", "V", "type", "status", "area", "covers_pH",
                          "charged_mpid", "discharged_mpid"}
        assert m["type"] == "Experimental"  # discharged is_expt=True

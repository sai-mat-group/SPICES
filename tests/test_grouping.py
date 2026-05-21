"""Tests for spices.intercalation.grouping."""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from spices.intercalation import grouping


# --------------------------------------------------------------------
# enrich_sheet_with_charged_mp_phases
# --------------------------------------------------------------------


class TestEnrichSheetWithChargedMpPhases:
    def _base_df(self):
        return pd.DataFrame({
            "MP id": ["mp-1", "mp-2"],
            "Formula": ["LiFeO2", "LiFeO2"],
            "Is experimental": [1, 0],
        })

    @patch("spices.dataframe_ops.update_seawater_metastability_columns_by_mp_id")
    @patch("spices.screening.mp_query.collect_mp_data_from_chemsys")
    @patch("spices.intercalation.structure.find_structures")
    @patch("spices.intercalation.structure.read_or_generate_charged_struct")
    def test_appends_new_rows_with_bool_columns(
        self, mock_read, mock_find, mock_collect, mock_update,
    ):
        mock_read.return_value = MagicMock(name="charged_struct")
        mock_find.side_effect = [
            ["mp-1", "mp-3"],      # mp-1's charged match set
            ["mp-2", "mp-4"],      # mp-2's charged match set
        ]
        mock_collect.return_value = [
            {"MP id": "mp-3", "Formula": "FeO2", "Is experimental": 0},
            {"MP id": "mp-4", "Formula": "FeO2", "Is experimental": 1},
        ]

        def _stub_update(row, **_):
            r = row.copy()
            r["Seawater metastability (Criterion 4)"] = [(7.5, 0.5)]
            return r

        mock_update.side_effect = _stub_update

        df = self._base_df()
        out = grouping.enrich_sheet_with_charged_mp_phases(
            df,
            working_ion="Li",
            mp_client=MagicMock(),
            cache=MagicMock(),
            element_str="Li",
        )

        # mp-3 and mp-4 appended; mp-1/mp-2 already existed
        assert set(out["MP id"]) == {"mp-1", "mp-2", "mp-3", "mp-4"}
        # _bool column added for each criterion col on new rows
        new_mask = out["MP id"].isin(["mp-3", "mp-4"])
        assert (
            out.loc[new_mask, "Seawater metastability (Criterion 4)_bool"]
            .astype(bool).all()
        )

    @patch("spices.intercalation.structure.find_structures")
    @patch("spices.intercalation.structure.read_or_generate_charged_struct")
    def test_no_new_mpids_returns_unchanged(self, mock_read, mock_find):
        mock_read.return_value = MagicMock(name="charged_struct")
        mock_find.return_value = ["mp-1"]     # already in df

        df = self._base_df()
        out = grouping.enrich_sheet_with_charged_mp_phases(
            df,
            working_ion="Li",
            mp_client=MagicMock(),
            cache=MagicMock(),
            element_str="Li",
        )

        pd.testing.assert_frame_equal(out, df)


# --------------------------------------------------------------------
# build_graph_components_from_sheet
# --------------------------------------------------------------------


class TestBuildGraphComponentsFromSheet:
    def _df(self):
        return pd.DataFrame({
            "MP id": ["mp-a", "mp-b", "mp-c"],
            "Formula": ["FeO2", "FeO2", "MnO2"],
            "Is experimental": [0, 1, 0],
        })

    @patch("spices.intercalation.grouping.make_networkx_graph")
    @patch("spices.intercalation.grouping.bfs_components")
    @patch("spices.intercalation.grouping.build_adjacency_dict")
    @patch("spices.intercalation.grouping.FrameworkStructure")
    @patch("spices.intercalation.grouping._load_row_struct")
    def test_partitions_and_subsets(
        self, mock_load, mock_fw, mock_adj,
        mock_bfs, mock_make_graph,
    ):
        # Three frameworks, bfs returns two components: [0, 1] and [2].
        mock_load.return_value = MagicMock(name="struct")
        mock_fw.side_effect = [
            MagicMock(name="fw_a"),
            MagicMock(name="fw_b"),
            MagicMock(name="fw_c"),
        ]
        mock_adj.return_value = {0: [1], 1: [0], 2: []}
        mock_bfs.return_value = [[0, 1], [2]]
        mock_make_graph.side_effect = [
            MagicMock(name="G0"),
            MagicMock(name="G1"),
        ]

        dfs, graphs = grouping.build_graph_components_from_sheet(
            self._df(), working_ion="Mg",
        )

        assert len(dfs) == 2 and len(graphs) == 2
        assert set(dfs[0]["MP id"]) == {"mp-a", "mp-b"}
        assert set(dfs[1]["MP id"]) == {"mp-c"}

    def test_missing_is_experimental_raises(self):
        df = pd.DataFrame({"MP id": ["mp-a"], "Formula": ["FeO2"]})
        with pytest.raises(KeyError, match="Is experimental"):
            grouping.build_graph_components_from_sheet(
                df, working_ion="Mg",
            )

    def test_missing_formula_raises(self):
        df = pd.DataFrame({"MP id": ["mp-a"], "Is experimental": [0]})
        with pytest.raises(KeyError, match="Formula"):
            grouping.build_graph_components_from_sheet(
                df, working_ion="Mg",
            )

    @patch("spices.intercalation.grouping.make_networkx_graph")
    @patch("spices.intercalation.grouping.bfs_components")
    @patch("spices.intercalation.grouping.build_adjacency_dict")
    @patch("spices.intercalation.grouping.FrameworkStructure")
    @patch("spices.intercalation.grouping._load_row_struct")
    def test_skips_nan_rows_with_warning(
        self, mock_load, mock_fw, mock_adj,
        mock_bfs, mock_make_graph, caplog,
    ):
        """NaN MP id / Formula rows are dropped before _load_row_struct."""
        mock_load.return_value = MagicMock(name="struct")
        mock_fw.return_value = MagicMock(name="fw")
        mock_adj.return_value = {0: []}
        mock_bfs.return_value = [[0]]
        mock_make_graph.return_value = MagicMock(name="G")

        df = pd.DataFrame({
            "MP id": ["mp-a", np.nan],
            "Formula": ["FeO2", np.nan],
            "Is experimental": [0, np.nan],
        })

        with caplog.at_level(
            logging.WARNING, logger="spices.intercalation.grouping",
        ):
            dfs, graphs = grouping.build_graph_components_from_sheet(
                df, working_ion="Mg",
            )

        assert mock_load.call_count == 1
        mock_load.assert_called_with("mp-a", "FeO2", "structs", "All_charged_structures")
        assert any(
            "Skipping 1 row(s) with NaN" in rec.message
            for rec in caplog.records
        )
        assert len(dfs) == 1
        assert set(dfs[0]["MP id"]) == {"mp-a"}


# --------------------------------------------------------------------
# _load_row_struct
# --------------------------------------------------------------------


class TestLoadRowStruct:
    def test_prefers_struct_dir(self, tmp_path):
        sd = tmp_path / "structs"
        sd.mkdir()
        (sd / "mp-1.cif").write_text("x")

        ac = tmp_path / "All_charged_structures" / "FeO2"
        ac.mkdir(parents=True)
        (ac / "mp-1.cif").write_text("y")

        with patch(
            "spices.intercalation.grouping.Structure.from_file"
        ) as mock_from_file:
            mock_from_file.return_value = "STRUCT_FROM_STRUCTS"
            result = grouping._load_row_struct(
                "mp-1", "FeO2", str(sd), str(tmp_path / "All_charged_structures"),
            )
            assert result == "STRUCT_FROM_STRUCTS"
            mock_from_file.assert_called_once_with(str(sd / "mp-1.cif"))

    def test_falls_back_to_all_charged(self, tmp_path):
        sd = tmp_path / "structs"
        sd.mkdir()     # empty

        ac = tmp_path / "All_charged_structures" / "FeO2"
        ac.mkdir(parents=True)
        (ac / "mp-1.cif").write_text("y")

        with patch(
            "spices.intercalation.grouping.Structure.from_file"
        ) as mock_from_file:
            mock_from_file.return_value = "STRUCT_FROM_CACHE"
            result = grouping._load_row_struct(
                "mp-1", "FeO2", str(sd), str(tmp_path / "All_charged_structures"),
            )
            assert result == "STRUCT_FROM_CACHE"
            mock_from_file.assert_called_once_with(str(ac / "mp-1.cif"))

    def test_raises_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            grouping._load_row_struct(
                "mp-1", "FeO2",
                str(tmp_path / "structs"),
                str(tmp_path / "All_charged_structures"),
            )


# --------------------------------------------------------------------
# Adjacency cache (step-3)
# --------------------------------------------------------------------


class TestAdjacencyCache:
    """Cache layer for ``build_graph_components_from_sheet`` adjacency dict.

    All cases mock out structure loading / framework construction / BFS /
    networkx so they isolate the cache path.
    """

    def _df(self):
        return pd.DataFrame({
            "MP id": ["mp-c", "mp-a", "mp-b"],   # intentionally unsorted
            "Formula": ["FeO2", "FeO2", "FeO2"],
            "Is experimental": [0, 1, 0],
        })

    def _patch_internals(self, adjacency_row=None):
        if adjacency_row is None:
            adjacency_row = {0: [1], 1: [0, 2], 2: [1]}
        load = patch("spices.intercalation.grouping._load_row_struct").start()
        fw = patch("spices.intercalation.grouping.FrameworkStructure").start()
        adj = patch(
            "spices.intercalation.grouping.build_adjacency_dict"
        ).start()
        bfs = patch("spices.intercalation.grouping.bfs_components").start()
        graph = patch(
            "spices.intercalation.grouping.make_networkx_graph"
        ).start()
        load.return_value = MagicMock(name="struct")
        fw.side_effect = lambda **kw: MagicMock(name=f"fw_{kw['mp_id']}")
        adj.return_value = adjacency_row
        bfs.return_value = [list(adjacency_row.keys())]
        graph.return_value = MagicMock(name="G")
        return load, fw, adj, bfs, graph

    def test_first_call_writes_pickle(self, tmp_path):
        import pickle
        cache_path = tmp_path / "FeO2.pkl"
        try:
            self._patch_internals()
            grouping.build_graph_components_from_sheet(
                self._df(),
                working_ion="Mg",
                adjacency_cache_path=str(cache_path),
            )
        finally:
            patch.stopall()

        assert cache_path.exists()
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        assert payload["schema_version"] == grouping.ADJACENCY_CACHE_SCHEMA_VERSION
        assert payload["mp_ids"] == ("mp-a", "mp-b", "mp-c")
        assert "adjacency" in payload
        assert "matcher_kwargs" in payload

    def test_second_call_loads_skips_build(self, tmp_path):
        cache_path = tmp_path / "FeO2.pkl"

        try:
            _, _, adj1, _, _ = self._patch_internals()
            grouping.build_graph_components_from_sheet(
                self._df(),
                working_ion="Mg",
                adjacency_cache_path=str(cache_path),
            )
            assert adj1.call_count == 1
        finally:
            patch.stopall()

        try:
            _, _, adj2, _, _ = self._patch_internals()
            adj2.side_effect = AssertionError(
                "build_adjacency_dict should not be called on cache hit",
            )
            grouping.build_graph_components_from_sheet(
                self._df(),
                working_ion="Mg",
                adjacency_cache_path=str(cache_path),
            )
            adj2.assert_not_called()
        finally:
            patch.stopall()

    def test_mp_ids_mismatch_rebuilds(self, tmp_path, caplog):
        import pickle
        cache_path = tmp_path / "FeO2.pkl"
        cache_path.write_bytes(pickle.dumps({
            "schema_version": grouping.ADJACENCY_CACHE_SCHEMA_VERSION,
            "mp_ids": ("mp-x", "mp-y", "mp-z"),    # stale
            "matcher_kwargs": grouping._matcher_signature(
                grouping.DEFAULT_MATCHER,
            ),
            "adjacency": {0: [], 1: [], 2: []},
        }))

        try:
            _, _, adj, _, _ = self._patch_internals()
            with caplog.at_level(
                logging.WARNING, logger="spices.intercalation.grouping",
            ):
                grouping.build_graph_components_from_sheet(
                    self._df(),
                    working_ion="Mg",
                    adjacency_cache_path=str(cache_path),
                )
            adj.assert_called_once()
        finally:
            patch.stopall()

        assert any(
            "mp_ids mismatch" in rec.message for rec in caplog.records
        )
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        assert payload["mp_ids"] == ("mp-a", "mp-b", "mp-c")

    def test_matcher_kwargs_mismatch_rebuilds(self, tmp_path, caplog):
        import pickle
        cache_path = tmp_path / "FeO2.pkl"
        cache_path.write_bytes(pickle.dumps({
            "schema_version": grouping.ADJACENCY_CACHE_SCHEMA_VERSION,
            "mp_ids": ("mp-a", "mp-b", "mp-c"),
            "matcher_kwargs": {"stale": True},
            "adjacency": {0: [], 1: [], 2: []},
        }))

        try:
            _, _, adj, _, _ = self._patch_internals()
            with caplog.at_level(
                logging.WARNING, logger="spices.intercalation.grouping",
            ):
                grouping.build_graph_components_from_sheet(
                    self._df(),
                    working_ion="Mg",
                    adjacency_cache_path=str(cache_path),
                )
            adj.assert_called_once()
        finally:
            patch.stopall()

        assert any(
            "matcher_kwargs mismatch" in rec.message
            for rec in caplog.records
        )

    def test_corrupt_pickle_rebuilds(self, tmp_path, caplog):
        cache_path = tmp_path / "FeO2.pkl"
        cache_path.write_bytes(b"not a valid pickle")

        try:
            _, _, adj, _, _ = self._patch_internals()
            with caplog.at_level(
                logging.WARNING, logger="spices.intercalation.grouping",
            ):
                grouping.build_graph_components_from_sheet(
                    self._df(),
                    working_ion="Mg",
                    adjacency_cache_path=str(cache_path),
                )
            adj.assert_called_once()
        finally:
            patch.stopall()

        assert any("unreadable" in rec.message for rec in caplog.records)

    def test_remap_sorted_to_row_order(self, tmp_path):
        """Shuffled row order produces row-indexed adjacency on load.

        Sorted mp_ids → (mp-a, mp-b, mp-c).  Cached adjacency in sorted
        index form: {0: [1], 1: [0, 2], 2: [1]} (a connects to b; b
        connects to a, c; c connects to b).  Row order in the df is
        (mp-c, mp-a, mp-b) → row indices 0=mp-c, 1=mp-a, 2=mp-b.  After
        remap, row-indexed adjacency should be {0: [2], 1: [2], 2: [1, 0]}.
        """
        import pickle
        cache_path = tmp_path / "FeO2.pkl"
        # Use the production signature so kwargs match on load.
        cache_path.write_bytes(pickle.dumps({
            "schema_version": grouping.ADJACENCY_CACHE_SCHEMA_VERSION,
            "mp_ids": ("mp-a", "mp-b", "mp-c"),
            "matcher_kwargs": grouping._matcher_signature(
                grouping.DEFAULT_MATCHER,
            ),
            "adjacency": {0: [1], 1: [0, 2], 2: [1]},
        }))

        captured = {}

        def _capture_bfs(adjacency, n):
            captured["adjacency"] = adjacency
            captured["n"] = n
            return [[0, 1, 2]]

        try:
            self._patch_internals()
            patch(
                "spices.intercalation.grouping.bfs_components",
                side_effect=_capture_bfs,
            ).start()
            grouping.build_graph_components_from_sheet(
                self._df(),
                working_ion="Mg",
                adjacency_cache_path=str(cache_path),
            )
        finally:
            patch.stopall()

        # mp-c -> row 0, mp-a -> row 1, mp-b -> row 2.
        # Sorted edges (mp-a, mp-b) → row edge (1, 2);
        # (mp-b, mp-c) → row edge (2, 0).
        assert sorted(captured["adjacency"][0]) == [2]
        assert sorted(captured["adjacency"][1]) == [2]
        assert sorted(captured["adjacency"][2]) == [0, 1]

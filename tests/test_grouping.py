"""Tests for spices.intercalation.grouping."""

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

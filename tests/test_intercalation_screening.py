"""Tests for the multi-worker sub-step 2 of intercalation screening."""

from concurrent.futures.process import BrokenProcessPool
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from spices.screening.intercalation import (
    SOURCE_FILE_COLUMN,
    _atomic_write_xlsx,
    _cleanup_orphan_tmp_xlsx,
    _merge_evaluated_new_rows,
    _phase1_discover_all_files,
    _phase3_redistribute_and_write,
    _seal_seawater_columns_for_evaluation,
    step2_enrich_charged_composition_sheets,
)
from spices.screening import intercalation as intercalation_mod


# --------------------------------------------------------------------
# _atomic_write_xlsx
# --------------------------------------------------------------------


class TestAtomicWriteXlsx:
    def test_writes_to_target_path(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2, 3]})
        target = tmp_path / "out.xlsx"
        _atomic_write_xlsx(df, str(target))
        assert target.exists()
        roundtrip = pd.read_excel(target)
        assert list(roundtrip["x"]) == [1, 2, 3]

    def test_no_tmp_file_left_behind(self, tmp_path):
        df = pd.DataFrame({"x": [1]})
        _atomic_write_xlsx(df, str(tmp_path / "out.xlsx"))
        leftover = list(tmp_path.glob("*.tmp.*"))
        assert leftover == []

    def test_replaces_existing_file(self, tmp_path):
        target = tmp_path / "out.xlsx"
        pd.DataFrame({"x": [1]}).to_excel(target, index=False)

        new_df = pd.DataFrame({"x": [9, 8, 7]})
        _atomic_write_xlsx(new_df, str(target))

        roundtrip = pd.read_excel(target)
        assert list(roundtrip["x"]) == [9, 8, 7]


# --------------------------------------------------------------------
# _cleanup_orphan_tmp_xlsx
# --------------------------------------------------------------------


class TestCleanupOrphanTmpXlsx:
    def test_removes_orphan_tmp_files(self, tmp_path):
        (tmp_path / "FeO2.xlsx.tmp.12345.xlsx").write_text("x")
        (tmp_path / "MnO2.xlsx.tmp.99999.xlsx").write_text("y")
        (tmp_path / "FeO2.xlsx").write_text("keep")  # not an orphan

        _cleanup_orphan_tmp_xlsx(str(tmp_path))

        assert not (tmp_path / "FeO2.xlsx.tmp.12345.xlsx").exists()
        assert not (tmp_path / "MnO2.xlsx.tmp.99999.xlsx").exists()
        assert (tmp_path / "FeO2.xlsx").exists()

    def test_handles_missing_directory(self, tmp_path):
        _cleanup_orphan_tmp_xlsx(str(tmp_path / "nonexistent"))


# --------------------------------------------------------------------
# _phase1_discover_all_files
# --------------------------------------------------------------------


class TestPhase1DiscoverAllFiles:
    def _write_sheet(self, path, mp_ids):
        pd.DataFrame({
            "MP id": mp_ids,
            "Formula": ["FeO2"] * len(mp_ids),
            "Is experimental": [0] * len(mp_ids),
        }).to_excel(path, index=False)

    @patch(
        "spices.screening.intercalation.discover_new_charged_rows_for_file"
    )
    def test_tags_new_rows_with_source_file(
        self, mock_discover, tmp_path,
    ):
        self._write_sheet(tmp_path / "FeO2.xlsx", ["mp-1", "mp-2"])
        self._write_sheet(tmp_path / "MnO2.xlsx", ["mp-3"])

        mock_discover.side_effect = [
            pd.DataFrame({
                "MP id": ["mp-10"], "Formula": ["FeO2"],
            }),
            pd.DataFrame({
                "MP id": ["mp-11", "mp-12"], "Formula": ["MnO2", "MnO2"],
            }),
        ]

        originals, all_new = _phase1_discover_all_files(
            str(tmp_path), working_ion="Li",
            mp_client=MagicMock(), element_str="Li",
            charged_struct_dir="cs", struct_dir="s",
            structure_cache_dir="acs",
        )

        assert set(originals.keys()) == {"FeO2.xlsx", "MnO2.xlsx"}
        # Sorted scan, so FeO2 first
        assert list(all_new[SOURCE_FILE_COLUMN]) == [
            "FeO2.xlsx", "MnO2.xlsx", "MnO2.xlsx",
        ]
        assert list(all_new["MP id"]) == ["mp-10", "mp-11", "mp-12"]

    @patch(
        "spices.screening.intercalation.discover_new_charged_rows_for_file"
    )
    def test_files_with_no_new_rows_excluded_from_new_df_but_kept_in_originals(
        self, mock_discover, tmp_path,
    ):
        self._write_sheet(tmp_path / "FeO2.xlsx", ["mp-1"])
        self._write_sheet(tmp_path / "ZeroMatch.xlsx", ["mp-2"])

        mock_discover.side_effect = [
            pd.DataFrame({"MP id": ["mp-9"], "Formula": ["FeO2"]}),
            pd.DataFrame(),  # empty for ZeroMatch
        ]

        originals, all_new = _phase1_discover_all_files(
            str(tmp_path), working_ion="Li",
            mp_client=MagicMock(), element_str="Li",
            charged_struct_dir="cs", struct_dir="s",
            structure_cache_dir="acs",
        )

        assert set(originals.keys()) == {"FeO2.xlsx", "ZeroMatch.xlsx"}
        assert list(all_new[SOURCE_FILE_COLUMN]) == ["FeO2.xlsx"]


# --------------------------------------------------------------------
# _phase3_redistribute_and_write
# --------------------------------------------------------------------


class TestPhase3RedistributeAndWrite:
    def test_writes_each_source_file(self, tmp_path):
        originals = {
            "FeO2.xlsx": pd.DataFrame({"MP id": ["mp-1"], "Formula": ["LiFeO2"]}),
            "MnO2.xlsx": pd.DataFrame({"MP id": ["mp-2"], "Formula": ["LiMnO2"]}),
        }
        for fname, df in originals.items():
            df.to_excel(tmp_path / fname, index=False)

        enriched = pd.DataFrame({
            "MP id": ["mp-9", "mp-10"],
            "Formula": ["FeO2", "MnO2"],
            "Seawater stability": [True, False],
            "Seawater metastability (Criterion 4)": [
                [(7.5, 0.5)], [],
            ],
            SOURCE_FILE_COLUMN: ["FeO2.xlsx", "MnO2.xlsx"],
        })

        _phase3_redistribute_and_write(originals, enriched, str(tmp_path))

        feo2 = pd.read_excel(tmp_path / "FeO2.xlsx")
        mno2 = pd.read_excel(tmp_path / "MnO2.xlsx")
        assert set(feo2["MP id"]) == {"mp-1", "mp-9"}
        assert set(mno2["MP id"]) == {"mp-2", "mp-10"}
        # _source_file column dropped from outputs
        assert SOURCE_FILE_COLUMN not in feo2.columns
        assert SOURCE_FILE_COLUMN not in mno2.columns
        # _bool companion column populated
        assert "Seawater metastability (Criterion 4)_bool" in feo2.columns

    def test_files_with_no_new_rows_still_rewritten(self, tmp_path):
        originals = {
            "Empty.xlsx": pd.DataFrame({"MP id": ["mp-only"], "Formula": ["X"]}),
        }
        originals["Empty.xlsx"].to_excel(tmp_path / "Empty.xlsx", index=False)
        enriched = pd.DataFrame()

        _phase3_redistribute_and_write(originals, enriched, str(tmp_path))

        roundtrip = pd.read_excel(tmp_path / "Empty.xlsx")
        assert list(roundtrip["MP id"]) == ["mp-only"]


# --------------------------------------------------------------------
# _seal_seawater_columns_for_evaluation
# --------------------------------------------------------------------


class TestSealSeawaterColumnsForEvaluation:
    def test_adds_missing_columns(self):
        df = pd.DataFrame({"MP id": ["mp-1"], "Formula": ["X"]})
        out = _seal_seawater_columns_for_evaluation(df)
        assert "Seawater stability" in out.columns
        assert "Seawater metastability (Criterion 4)" in out.columns
        assert pd.isna(out["Seawater stability"].iloc[0])

    def test_metastability_column_object_dtype(self):
        df = pd.DataFrame({"MP id": ["mp-1"], "Formula": ["X"]})
        out = _seal_seawater_columns_for_evaluation(df)
        # Object dtype lets us store list cells.
        out.at[0, "Seawater metastability (Criterion 4)"] = [(7.5, 0.5)]
        assert isinstance(
            out.at[0, "Seawater metastability (Criterion 4)"], list,
        )


# --------------------------------------------------------------------
# _merge_evaluated_new_rows
# --------------------------------------------------------------------


class TestMergeEvaluatedNewRows:
    def _target(self):
        return pd.DataFrame({
            SOURCE_FILE_COLUMN: ["FeO2.xlsx", "FeO2.xlsx", "MnO2.xlsx"],
            "MP id": ["mp-1", "mp-2", "mp-3"],
            "Seawater stability": [np.nan, np.nan, np.nan],
            "Seawater metastability (Criterion 4)": [None, None, None],
        })

    def test_fills_evaluated_rows_only(self):
        target = self._target()
        source = pd.DataFrame({
            SOURCE_FILE_COLUMN: ["FeO2.xlsx", "MnO2.xlsx"],
            "MP id": ["mp-1", "mp-3"],
            "Seawater stability": [True, False],
            "Seawater metastability (Criterion 4)": [
                [(7.5, 0.5)], [],
            ],
        })

        _merge_evaluated_new_rows(target, source)

        assert bool(target.loc[0, "Seawater stability"]) is True
        assert pd.isna(target.loc[1, "Seawater stability"])
        assert bool(target.loc[2, "Seawater stability"]) is False

    def test_skips_unevaluated_source_rows(self):
        target = self._target()
        source = pd.DataFrame({
            SOURCE_FILE_COLUMN: ["FeO2.xlsx"],
            "MP id": ["mp-1"],
            "Seawater stability": [np.nan],
            "Seawater metastability (Criterion 4)": [None],
        })
        _merge_evaluated_new_rows(target, source)
        assert pd.isna(target.loc[0, "Seawater stability"])

    def test_handles_empty_source(self):
        target = self._target()
        _merge_evaluated_new_rows(target, pd.DataFrame())
        assert target["Seawater stability"].isna().all()


# --------------------------------------------------------------------
# step2 end-to-end (mocked) — sequential mode
# --------------------------------------------------------------------


class TestStep2Sequential:
    def _setup_files(self, tmp_path):
        comp_dir = tmp_path / "unique_charged_composition_sheets"
        comp_dir.mkdir()
        pd.DataFrame({
            "MP id": ["mp-1"],
            "Formula": ["LiFeO2"],
            "Is experimental": [0],
        }).to_excel(comp_dir / "FeO2.xlsx", index=False)
        return comp_dir

    @patch(
        "spices.screening.intercalation."
        "update_seawater_metastability_columns_by_mp_id"
    )
    @patch(
        "spices.screening.intercalation.discover_new_charged_rows_for_file"
    )
    def test_sequential_mode_writes_enriched_files(
        self, mock_discover, mock_update, tmp_path,
    ):
        comp_dir = self._setup_files(tmp_path)
        mock_discover.return_value = pd.DataFrame({
            "MP id": ["mp-9"],
            "Formula": ["FeO2"],
            "Is experimental": [0],
        })

        def _stub_update(row, **_):
            r = row.copy()
            r["Seawater stability"] = True
            r["Seawater metastability (Criterion 4)"] = [(7.5, 0.5)]
            return r

        mock_update.side_effect = _stub_update

        step2_enrich_charged_composition_sheets(
            str(comp_dir),
            working_ion="Li",
            mp_client=MagicMock(),
            cache=MagicMock(),
            element_str="Li",
            n_workers=1,
            output_dir=str(tmp_path),
        )

        out = pd.read_excel(comp_dir / "FeO2.xlsx")
        assert set(out["MP id"]) == {"mp-1", "mp-9"}
        new_row = out[out["MP id"] == "mp-9"].iloc[0]
        assert bool(new_row["Seawater stability"]) is True
        # _bool companion column added
        assert (
            "Seawater metastability (Criterion 4)_bool"
            in out.columns
        )
        assert bool(new_row["Seawater metastability (Criterion 4)_bool"])


# --------------------------------------------------------------------
# step2 end-to-end (mocked) — multi-worker BrokenProcessPool recovery
# --------------------------------------------------------------------


class TestStep2BrokenPoolRecovery:
    def _setup_files(self, tmp_path):
        comp_dir = tmp_path / "unique_charged_composition_sheets"
        comp_dir.mkdir()
        for fname, mpid in [("FeO2.xlsx", "mp-1"), ("MnO2.xlsx", "mp-2")]:
            pd.DataFrame({
                "MP id": [mpid],
                "Formula": ["LiX"],
                "Is experimental": [0],
            }).to_excel(comp_dir / fname, index=False)
        return comp_dir

    @patch("spices.screening.intercalation.ProcessPoolExecutor")
    @patch(
        "spices.screening.intercalation._recover_chunk_for_core"
    )
    @patch(
        "spices.screening.intercalation._phase1_discover_all_files"
    )
    def test_broken_pool_recovers_via_checkpoint(
        self, mock_phase1, mock_recover, mock_pool_cls, tmp_path,
    ):
        comp_dir = self._setup_files(tmp_path)

        # Bypass Phase 1 entirely — return the originals + the
        # already-tagged new-rows DataFrame Phase 2 expects.  This keeps
        # the test focused on Phase 2's BrokenProcessPool recovery and
        # avoids consuming pool_inst.submit's side_effect for Phase 1.
        originals = {
            "FeO2.xlsx": pd.read_excel(comp_dir / "FeO2.xlsx"),
            "MnO2.xlsx": pd.read_excel(comp_dir / "MnO2.xlsx"),
        }
        all_new_rows = pd.DataFrame({
            "MP id": ["mp-9", "mp-10"],
            "Formula": ["FeO2", "MnO2"],
            "Is experimental": [0, 0],
            SOURCE_FILE_COLUMN: ["FeO2.xlsx", "MnO2.xlsx"],
        })
        mock_phase1.return_value = (originals, all_new_rows)

        # Simulate one future raising BrokenProcessPool; recovery returns
        # a fully-populated chunk from disk.
        broken_future = MagicMock()
        broken_future.result.side_effect = BrokenProcessPool

        ok_chunk = pd.DataFrame({
            "MP id": ["mp-10"],
            "Formula": ["MnO2"],
            "Is experimental": [0],
            "Seawater stability": [True],
            "Seawater metastability (Criterion 4)": [[(7.5, 0.5)]],
            SOURCE_FILE_COLUMN: ["MnO2.xlsx"],
        })
        ok_future = MagicMock()
        ok_future.result.return_value = ok_chunk

        pool_inst = MagicMock()
        pool_inst.__enter__ = MagicMock(return_value=pool_inst)
        pool_inst.__exit__ = MagicMock(return_value=False)
        pool_inst.submit.side_effect = [broken_future, ok_future]
        mock_pool_cls.return_value = pool_inst

        recovered_chunk = pd.DataFrame({
            "MP id": ["mp-9"],
            "Formula": ["FeO2"],
            "Is experimental": [0],
            "Seawater stability": [False],
            "Seawater metastability (Criterion 4)": [[]],
            SOURCE_FILE_COLUMN: ["FeO2.xlsx"],
        })
        mock_recover.return_value = recovered_chunk

        with patch(
            "spices.screening.intercalation.as_completed",
            lambda futs: [broken_future, ok_future],
        ):
            step2_enrich_charged_composition_sheets(
                str(comp_dir),
                working_ion="Li",
                mp_client=MagicMock(),
                cache=MagicMock(),
                element_str="Li",
                n_workers=2,
                output_dir=str(tmp_path),
            )

        # Recovery was called for the dead worker chunk.
        mock_recover.assert_called()
        # Both source files written.
        feo2 = pd.read_excel(comp_dir / "FeO2.xlsx")
        mno2 = pd.read_excel(comp_dir / "MnO2.xlsx")
        assert set(feo2["MP id"]) == {"mp-1", "mp-9"}
        assert set(mno2["MP id"]) == {"mp-2", "mp-10"}


# --------------------------------------------------------------------
# Source-file invariant
# --------------------------------------------------------------------


class TestSourceFileColumnPreserved:
    def test_chunk_xlsx_round_trip_preserves_source_file(self, tmp_path):
        chunk = pd.DataFrame({
            "MP id": ["mp-1"],
            "Formula": ["X"],
            "Seawater stability": [True],
            "Seawater metastability (Criterion 4)": [[(7.5, 0.5)]],
            SOURCE_FILE_COLUMN: ["FeO2.xlsx"],
        })
        path = tmp_path / "K_core0_final.xlsx"
        chunk.to_excel(path, index=False)
        roundtrip = pd.read_excel(path)
        assert SOURCE_FILE_COLUMN in roundtrip.columns
        assert list(roundtrip[SOURCE_FILE_COLUMN]) == ["FeO2.xlsx"]


# --------------------------------------------------------------------
# Phase 1 multi-worker dispatch + recovery
# --------------------------------------------------------------------


class TestPhase1MultiWorker:
    """Validate the multi-worker Phase 1 path of
    _phase1_discover_all_files.  ProcessPoolExecutor is mocked so
    workers never actually fork — we just check the splitting,
    dispatch, result aggregation, and BrokenProcessPool fallback.
    """

    def _write_sheet(self, path, mpid):
        pd.DataFrame({
            "MP id": [mpid],
            "Formula": ["LiX"],
            "Is experimental": [0],
        }).to_excel(path, index=False)

    def _setup_files(self, tmp_path, n=4):
        comp_dir = tmp_path / "comp"
        comp_dir.mkdir()
        for i in range(n):
            self._write_sheet(comp_dir / f"F{i}.xlsx", f"mp-{i}")
        return comp_dir

    @patch("spices.screening.intercalation.ProcessPoolExecutor")
    def test_chunks_split_round_robin(self, mock_pool_cls, tmp_path):
        comp_dir = self._setup_files(tmp_path, n=5)

        # Pool returns futures whose results are empty DataFrames.
        empty_future = MagicMock()
        empty_future.result.return_value = pd.DataFrame()
        empty_future.done.return_value = True

        pool_inst = MagicMock()
        pool_inst.__enter__ = MagicMock(return_value=pool_inst)
        pool_inst.__exit__ = MagicMock(return_value=False)
        pool_inst.submit.return_value = empty_future
        mock_pool_cls.return_value = pool_inst

        with patch(
            "spices.screening.intercalation.as_completed",
            lambda futs: list(futs),
        ):
            _phase1_discover_all_files(
                str(comp_dir),
                working_ion="Li", mp_client=MagicMock(),
                element_str="Li",
                charged_struct_dir="cs", struct_dir="s",
                structure_cache_dir="acs",
                n_workers=2,
                log_dir=str(tmp_path / "logs"),
            )

        # Two chunks dispatched (one per worker).  Round-robin slicing
        # gives chunk 0 = [F0, F2, F4], chunk 1 = [F1, F3].
        assert pool_inst.submit.call_count == 2
        chunk0 = pool_inst.submit.call_args_list[0][0][1]
        chunk1 = pool_inst.submit.call_args_list[1][0][1]
        assert chunk0 == ["F0.xlsx", "F2.xlsx", "F4.xlsx"]
        assert chunk1 == ["F1.xlsx", "F3.xlsx"]

    @patch("spices.screening.intercalation.ProcessPoolExecutor")
    def test_aggregates_worker_results(self, mock_pool_cls, tmp_path):
        comp_dir = self._setup_files(tmp_path, n=3)

        # Two chunks → two futures returning two distinct new-rows DFs.
        df_a = pd.DataFrame({
            "MP id": ["mp-9"], "Formula": ["LiA"],
            SOURCE_FILE_COLUMN: ["F0.xlsx"],
        })
        df_b = pd.DataFrame({
            "MP id": ["mp-10"], "Formula": ["LiB"],
            SOURCE_FILE_COLUMN: ["F1.xlsx"],
        })
        f_a = MagicMock()
        f_a.result.return_value = df_a
        f_a.done.return_value = True
        f_b = MagicMock()
        f_b.result.return_value = df_b
        f_b.done.return_value = True

        pool_inst = MagicMock()
        pool_inst.__enter__ = MagicMock(return_value=pool_inst)
        pool_inst.__exit__ = MagicMock(return_value=False)
        pool_inst.submit.side_effect = [f_a, f_b]
        mock_pool_cls.return_value = pool_inst

        with patch(
            "spices.screening.intercalation.as_completed",
            lambda futs: [f_a, f_b],
        ):
            originals, all_new = _phase1_discover_all_files(
                str(comp_dir),
                working_ion="Li", mp_client=MagicMock(),
                element_str="Li",
                charged_struct_dir="cs", struct_dir="s",
                structure_cache_dir="acs",
                n_workers=2,
                log_dir=str(tmp_path / "logs"),
            )

        assert set(originals.keys()) == {
            "F0.xlsx", "F1.xlsx", "F2.xlsx",
        }
        assert set(all_new["MP id"]) == {"mp-9", "mp-10"}
        assert set(all_new[SOURCE_FILE_COLUMN]) == {"F0.xlsx", "F1.xlsx"}

    @patch(
        "spices.screening.intercalation.discover_new_charged_rows_for_file"
    )
    @patch("spices.screening.intercalation.ProcessPoolExecutor")
    def test_broken_pool_falls_back_to_sequential(
        self, mock_pool_cls, mock_discover, tmp_path,
    ):
        comp_dir = self._setup_files(tmp_path, n=2)

        # One future succeeds; the other raises BrokenProcessPool.
        ok_df = pd.DataFrame({
            "MP id": ["mp-9"], "Formula": ["LiA"],
            SOURCE_FILE_COLUMN: ["F0.xlsx"],
        })
        f_ok = MagicMock()
        f_ok.result.return_value = ok_df
        f_ok.done.return_value = True

        f_dead = MagicMock()
        f_dead.result.side_effect = BrokenProcessPool
        f_dead.done.return_value = True

        pool_inst = MagicMock()
        pool_inst.__enter__ = MagicMock(return_value=pool_inst)
        pool_inst.__exit__ = MagicMock(return_value=False)
        pool_inst.submit.side_effect = [f_ok, f_dead]
        mock_pool_cls.return_value = pool_inst

        # Sequential fallback uses the unmocked _phase1_sequential_discover,
        # which calls discover_new_charged_rows_for_file once for the
        # dead chunk's file (F1.xlsx).
        mock_discover.return_value = pd.DataFrame({
            "MP id": ["mp-fallback"], "Formula": ["LiB"],
        })

        with patch(
            "spices.screening.intercalation.as_completed",
            lambda futs: [f_ok, f_dead],
        ):
            originals, all_new = _phase1_discover_all_files(
                str(comp_dir),
                working_ion="Li", mp_client=MagicMock(),
                element_str="Li",
                charged_struct_dir="cs", struct_dir="s",
                structure_cache_dir="acs",
                n_workers=2,
                log_dir=str(tmp_path / "logs"),
            )

        # Sequential recovery was invoked for the dead chunk's file.
        mock_discover.assert_called_once()
        # Both the OK worker's result and the recovered row appear.
        assert set(all_new["MP id"]) == {"mp-9", "mp-fallback"}

    def test_n_workers_one_takes_sequential_path(self, tmp_path):
        """Sanity: n_workers=1 must NOT touch ProcessPoolExecutor."""
        comp_dir = self._setup_files(tmp_path, n=2)
        with patch(
            "spices.screening.intercalation.ProcessPoolExecutor"
        ) as mock_pool, patch(
            "spices.screening.intercalation.discover_new_charged_rows_for_file"
        ) as mock_discover:
            mock_discover.return_value = pd.DataFrame()
            _phase1_discover_all_files(
                str(comp_dir),
                working_ion="Li", mp_client=MagicMock(),
                element_str="Li",
                charged_struct_dir="cs", struct_dir="s",
                structure_cache_dir="acs",
                n_workers=1,
            )
            mock_pool.assert_not_called()


class TestPhase1Init:
    """The pool initializer wires the shared progress counter into the
    worker module's global namespace.
    """
    def test_sets_module_global(self):
        sentinel = MagicMock()
        intercalation_mod._phase1_init(sentinel)
        assert intercalation_mod._PHASE1_COUNTER is sentinel
        # Restore for test isolation.
        intercalation_mod._PHASE1_COUNTER = None


# --------------------------------------------------------------------
# step3_create_voltage_plateau_sheets
# --------------------------------------------------------------------


class TestStep3CreateVoltagePlateauSheets:
    """Per-composition dispatch for step 3 (graph build + voltage plateaus).

    All cases mock the per-composition worker so they only exercise the
    dispatcher orchestration: task enumeration, adjacency_cache_dir
    creation, restart short-circuit, fail-fast on worker exception.
    """

    def _make_dirs(self, tmp_path):
        comp_dir = tmp_path / "unique_charged_composition_sheets"
        out_dir = tmp_path / "Li_intercalation" / "Voltage_plateau_data"
        comp_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)
        (comp_dir / "FeO2.xlsx").write_text("dummy")
        (comp_dir / "MnO2.xlsx").write_text("dummy")
        return comp_dir, out_dir

    def _kwargs(self):
        return dict(
            working_ion="Li",
            working_ion_valency=1,
            working_ion_voltage_vs_SHE=-3.05,
            working_ion_energy=-1.9,
            element_str="Li",
            selection_criteria=[4],
            decomp_energy_for_metastability=0.3,
            cache_dir=".",
        )

    @patch("spices.screening.intercalation._step3_process_composition")
    def test_sequential_writes_outputs(self, mock_worker, tmp_path):
        comp_dir, out_dir = self._make_dirs(tmp_path)
        intercalation_mod.step3_create_voltage_plateau_sheets(
            str(comp_dir), str(out_dir),
            n_workers=1, restart=False,
            **self._kwargs(),
        )
        called_fnames = sorted(call.args[0] for call in mock_worker.call_args_list)
        assert called_fnames == ["FeO2.xlsx", "MnO2.xlsx"]

    @patch("spices.screening.intercalation._dispatch_phase1_tasks")
    def test_parallel_forwards_n_workers(self, mock_dispatch, tmp_path):
        comp_dir, out_dir = self._make_dirs(tmp_path)
        intercalation_mod.step3_create_voltage_plateau_sheets(
            str(comp_dir), str(out_dir),
            n_workers=4, restart=False,
            **self._kwargs(),
        )
        mock_dispatch.assert_called_once()
        args, _ = mock_dispatch.call_args
        label, n_workers, tasks, worker_fn, args_for_task = args
        assert label == "Step 3"
        assert n_workers == 4
        assert sorted(tasks) == ["FeO2.xlsx", "MnO2.xlsx"]
        assert worker_fn is intercalation_mod._step3_process_composition

    @patch("spices.screening.intercalation._step3_process_composition")
    def test_first_worker_failure_aborts(self, mock_worker, tmp_path):
        comp_dir, out_dir = self._make_dirs(tmp_path)

        def _maybe_raise(fname, *args, **kwargs):
            if fname == "FeO2.xlsx":
                raise RuntimeError("boom")

        mock_worker.side_effect = _maybe_raise
        with pytest.raises(RuntimeError, match="boom"):
            intercalation_mod.step3_create_voltage_plateau_sheets(
                str(comp_dir), str(out_dir),
                n_workers=1, restart=False,
                **self._kwargs(),
            )

    def test_adjacency_cache_dir_created_at_sibling(self, tmp_path):
        comp_dir, out_dir = self._make_dirs(tmp_path)
        with patch(
            "spices.screening.intercalation._step3_process_composition"
        ):
            intercalation_mod.step3_create_voltage_plateau_sheets(
                str(comp_dir), str(out_dir),
                n_workers=1, restart=False,
                **self._kwargs(),
            )
        assert (out_dir.parent / "adjacency_cache").is_dir()

    def test_restart_skips_completed_composition(self, tmp_path):
        """When restart=True and all sc xlsxes exist, worker returns
        early without reading the input xlsx.

        Tests ``_step3_process_composition`` directly (not the
        dispatcher), since the short-circuit lives in the worker.
        """
        comp_dir, out_dir = self._make_dirs(tmp_path)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        adj_dir = tmp_path / "adjacency_cache"
        adj_dir.mkdir()

        # Pre-create sc_4/FeO2.xlsx so the restart check finds it.
        sc_dir = out_dir / "FeO2" / "sc_4"
        sc_dir.mkdir(parents=True)
        (sc_dir / "FeO2.xlsx").write_text("present")

        with patch(
            "spices.screening.intercalation.pd.read_excel"
        ) as mock_read:
            intercalation_mod._step3_process_composition(
                "FeO2.xlsx",
                str(comp_dir),
                str(out_dir),
                str(adj_dir),
                "Li", 1, -3.05, -1.9, "Li", [4],
                True, False, True,
                "charged_structs", "structs", "All_charged_structures",
                0.3, str(cache_dir), True,
            )
            mock_read.assert_not_called()

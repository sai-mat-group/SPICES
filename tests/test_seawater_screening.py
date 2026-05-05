"""Tests for seawater screening runner functions."""

import os
import pickle
import sys
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from spices.pourbaix.cache import PourbaixCache
from spices.screening.seawater import (
    SEAWATER_COLUMNS,
    _available_memory_gb,
    _dispatch_phase1_tasks,
    _filter_pending_chemsys,
    _filter_pending_sea_keys,
    _phase0_discover_unique_keys,
    _recover_chunk_for_core,
    _split_dataframe,
    run_seawater_screening,
)


class TestSplitDataframe:
    def _make_df(self, n_rows):
        return pd.DataFrame({"idx": range(n_rows)})

    def test_single_worker_returns_one_chunk(self):
        df = self._make_df(10)
        chunks = _split_dataframe(df, 1)
        assert len(chunks) == 1
        assert len(chunks[0]) == 10

    def test_round_robin_distribution(self):
        df = self._make_df(10)
        chunks = _split_dataframe(df, 3)
        # Worker 0: rows 0, 3, 6, 9
        assert list(chunks[0]["idx"]) == [0, 3, 6, 9]
        # Worker 1: rows 1, 4, 7
        assert list(chunks[1]["idx"]) == [1, 4, 7]
        # Worker 2: rows 2, 5, 8
        assert list(chunks[2]["idx"]) == [2, 5, 8]

    def test_all_rows_covered(self):
        df = self._make_df(100)
        chunks = _split_dataframe(df, 7)
        all_indices = []
        for chunk in chunks:
            all_indices.extend(chunk["idx"].tolist())
        assert sorted(all_indices) == list(range(100))

    def test_more_workers_than_rows(self):
        df = self._make_df(3)
        chunks = _split_dataframe(df, 10)
        # Only 3 non-empty chunks
        assert len(chunks) == 3
        total_rows = sum(len(c) for c in chunks)
        assert total_rows == 3

    def test_equal_distribution(self):
        df = self._make_df(48)
        chunks = _split_dataframe(df, 48)
        assert len(chunks) == 48
        for chunk in chunks:
            assert len(chunk) == 1

    def test_uneven_distribution(self):
        df = self._make_df(50)
        chunks = _split_dataframe(df, 48)
        # 2 workers get 2 rows, 46 workers get 1 row
        sizes = [len(c) for c in chunks]
        assert sum(sizes) == 50
        assert sizes.count(2) == 2
        assert sizes.count(1) == 46

    def test_reset_index(self):
        """Chunks should have reset index starting from 0."""
        df = self._make_df(10)
        chunks = _split_dataframe(df, 3)
        for chunk in chunks:
            assert chunk.index[0] == 0
            assert list(chunk.index) == list(range(len(chunk)))


class TestSeawaterColumns:
    def test_column_count(self):
        assert len(SEAWATER_COLUMNS) == 2

    def test_stability_column_present(self):
        assert "Seawater stability" in SEAWATER_COLUMNS

    def test_criterion_4_present(self):
        assert "Seawater metastability (Criterion 4)" in SEAWATER_COLUMNS


class TestPourbaixCacheLRU:
    def test_default_caps(self):
        cache = PourbaixCache()
        assert cache.max_entries == 8
        assert cache.max_diagrams == 8
        assert cache.max_sea_diagrams == 8

    def test_custom_caps(self):
        cache = PourbaixCache(max_entries=4, max_diagrams=2, max_sea_diagrams=1)
        assert cache.max_entries == 4
        assert cache.max_diagrams == 2
        assert cache.max_sea_diagrams == 1

    def test_evict_to_size_pops_oldest(self):
        cache = PourbaixCache(max_diagrams=3)
        for i in range(5):
            cache._pourbaix_diagrams[f"k{i}"] = i
            cache._evict_to_size(
                cache._pourbaix_diagrams, cache.max_diagrams,
            )
        assert list(cache._pourbaix_diagrams.keys()) == ["k2", "k3", "k4"]

    def test_move_to_end_protects_from_eviction(self):
        cache = PourbaixCache(max_diagrams=3)
        for i in range(3):
            cache._pourbaix_diagrams[f"k{i}"] = i
        # Touch k0 -> becomes most-recently-used.
        cache._pourbaix_diagrams.move_to_end("k0")
        cache._pourbaix_diagrams["k3"] = 3
        cache._evict_to_size(cache._pourbaix_diagrams, cache.max_diagrams)
        assert "k0" in cache._pourbaix_diagrams
        assert "k1" not in cache._pourbaix_diagrams


class TestRecoverChunkForCore:
    def _df(self, mp_id):
        return pd.DataFrame({"MP id": [mp_id], "Formula": ["X"]})

    def test_returns_original_when_runtime_dir_missing(self, tmp_path):
        original = self._df("mp-orig")
        out = _recover_chunk_for_core(
            str(tmp_path / "nope"), "K", 0, original,
        )
        pd.testing.assert_frame_equal(out, original)

    def test_returns_original_when_no_matching_files(self, tmp_path):
        original = self._df("mp-orig")
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        # Wrong element prefix.
        self._df("mp-other").to_excel(
            runtime / "Na_core0_final.xlsx", index=False,
        )
        out = _recover_chunk_for_core(str(runtime), "K", 0, original)
        pd.testing.assert_frame_equal(out, original)

    def test_prefers_final_over_part(self, tmp_path):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        self._df("mp-part").to_excel(
            runtime / "K_core0_part1.xlsx", index=False,
        )
        self._df("mp-final").to_excel(
            runtime / "K_core0_final.xlsx", index=False,
        )
        out = _recover_chunk_for_core(
            str(runtime), "K", 0, self._df("mp-orig"),
        )
        assert list(out["MP id"]) == ["mp-final"]

    def test_picks_highest_part_when_no_final(self, tmp_path):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        for n, mid in [(1, "mp-1"), (3, "mp-3"), (2, "mp-2")]:
            self._df(mid).to_excel(
                runtime / f"K_core0_part{n}.xlsx", index=False,
            )
        out = _recover_chunk_for_core(
            str(runtime), "K", 0, self._df("mp-orig"),
        )
        assert list(out["MP id"]) == ["mp-3"]


class TestAvailableMemoryGb:
    def test_returns_positive_float_on_linux(self):
        if not sys.platform.startswith("linux"):
            pytest.skip("requires /proc/meminfo")
        avail = _available_memory_gb()
        assert avail is not None
        assert avail > 0


class TestPhase0DiscoverUniqueKeys:
    """Phase 0 should dedup chemsys and sea_comp keys across rows."""

    def test_collapses_duplicate_formulas_to_one_key(self):
        df = pd.DataFrame({
            "Formula": ["LiFePO4", "LiFePO4", "LiFePO4"],
            "MP id": ["mp-1", "mp-2", "mp-3"],
        })
        chemsys, sea_keys = _phase0_discover_unique_keys(df)
        assert len(chemsys) == 1
        assert len(sea_keys) == 1

    def test_distinct_chemsys_emit_distinct_keys(self):
        df = pd.DataFrame({
            "Formula": ["LiFePO4", "LiCoO2"],
            "MP id": ["mp-1", "mp-2"],
        })
        chemsys, sea_keys = _phase0_discover_unique_keys(df)
        assert len(chemsys) == 2
        assert len(sea_keys) == 2

    def test_skips_invalid_formula(self):
        df = pd.DataFrame({
            "Formula": ["LiFePO4", "", None],
            "MP id": ["mp-1", "mp-2", "mp-3"],
        })
        chemsys, sea_keys = _phase0_discover_unique_keys(df)
        assert len(chemsys) == 1
        assert len(sea_keys) == 1

    def test_returns_representative_comp_dict(self):
        df = pd.DataFrame({
            "Formula": ["LiFePO4"],
            "MP id": ["mp-1"],
        })
        _, sea_keys = _phase0_discover_unique_keys(df)
        key, comp_dict = sea_keys[0]
        assert isinstance(comp_dict, dict)
        assert "Li" in comp_dict and "Fe" in comp_dict


class TestFilterPendingChemsys:
    def test_returns_all_when_dir_missing(self, tmp_path):
        result = _filter_pending_chemsys(
            ["Li-Fe-O", "Ca-Ti-O"],
            str(tmp_path / "nope"),
        )
        assert result == ["Li-Fe-O", "Ca-Ti-O"]

    def test_drops_chemsys_already_on_disk(self, tmp_path):
        entries = tmp_path / "entries"
        entries.mkdir()
        (entries / "Li-Fe-O_pbxentry.pkl").write_bytes(b"x")
        result = _filter_pending_chemsys(
            ["Li-Fe-O", "Ca-Ti-O"], str(entries),
        )
        assert result == ["Ca-Ti-O"]

    def test_ignores_unrelated_files(self, tmp_path):
        entries = tmp_path / "entries"
        entries.mkdir()
        (entries / "README.txt").write_text("ignore me")
        result = _filter_pending_chemsys(["Li-Fe-O"], str(entries))
        assert result == ["Li-Fe-O"]


class TestFilterPendingSeaKeys:
    def test_returns_all_when_dir_missing(self, tmp_path):
        keys = [("Li1 Fe1 P1", {"Li": 1}), ("Ca1 Ti1", {"Ca": 1})]
        result = _filter_pending_sea_keys(keys, str(tmp_path / "nope"))
        assert result == keys

    def test_drops_keys_already_on_disk(self, tmp_path):
        pbx = tmp_path / "pbx"
        pbx.mkdir()
        # Match by underscored reduced formula prefix
        (pbx / "Li1_Fe1_P1_pbx.pkl").write_bytes(b"x")
        keys = [
            ("Li1 Fe1 P1", {"Li": 1}),
            ("Ca1 Ti1", {"Ca": 1}),
        ]
        result = _filter_pending_sea_keys(keys, str(pbx))
        assert [k for k, _ in result] == ["Ca1 Ti1"]

    def test_handles_mp_id_suffixed_filenames(self, tmp_path):
        pbx = tmp_path / "pbx"
        pbx.mkdir()
        (pbx / "Li1_Fe1_P1_mp-12345_pbx.pkl").write_bytes(b"x")
        keys = [("Li1 Fe1 P1", {"Li": 1}), ("Ca1 Ti1", {"Ca": 1})]
        result = _filter_pending_sea_keys(keys, str(pbx))
        assert [k for k, _ in result] == ["Ca1 Ti1"]


class TestDispatchPhase1Sequential:
    """n_workers <= 1 path runs tasks inline; failures abort immediately."""

    def test_runs_all_tasks_in_order(self):
        seen = []

        def worker(task):
            seen.append(task)
            return task

        _dispatch_phase1_tasks(
            "Phase X", n_workers=1,
            tasks=["a", "b", "c"],
            worker_fn=worker,
            args_for_task=lambda t: (t,),
        )
        assert seen == ["a", "b", "c"]

    def test_aborts_on_first_failure(self):
        seen = []

        def worker(task):
            seen.append(task)
            if task == "b":
                raise RuntimeError("boom")
            return task

        with pytest.raises(RuntimeError, match="boom"):
            _dispatch_phase1_tasks(
                "Phase X", n_workers=1,
                tasks=["a", "b", "c"],
                worker_fn=worker,
                args_for_task=lambda t: (t,),
            )
        # 'c' should never run because 'b' failed
        assert seen == ["a", "b"]

    def test_empty_task_list_is_noop(self):
        called = []

        def worker(task):
            called.append(task)

        _dispatch_phase1_tasks(
            "Phase X", n_workers=4,
            tasks=[],
            worker_fn=worker,
            args_for_task=lambda t: (t,),
        )
        assert called == []


class TestRestartHardGuard:
    """restart=False must refuse to overwrite an output xlsx with progress."""

    def test_refuses_to_overwrite_populated_output(self, tmp_path):
        out_path = tmp_path / "out.xlsx"
        # Pre-populate with one evaluated row
        pd.DataFrame({
            "MP id": ["mp-1"],
            "Formula": ["LiFePO4"],
            "Seawater stability": [True],
            "Seawater metastability (Criterion 4)": [False],
        }).to_excel(out_path, index=False)

        in_path = tmp_path / "in.xlsx"
        pd.DataFrame({
            "MP id": ["mp-1"],
            "Formula": ["LiFePO4"],
        }).to_excel(in_path, index=False)

        with pytest.raises(RuntimeError, match="already has"):
            run_seawater_screening(
                element="Li",
                input_path=str(in_path),
                output_path=str(out_path),
                cache_dir=str(tmp_path),
                n_workers=1,
                restart=False,
            )

    def test_allows_overwrite_when_no_evaluated_rows(self, tmp_path):
        """A stale output with all-NaN seawater columns should not block."""
        out_path = tmp_path / "out.xlsx"
        pd.DataFrame({
            "MP id": ["mp-1"],
            "Formula": ["LiFePO4"],
            "Seawater stability": [None],
            "Seawater metastability (Criterion 4)": [None],
        }).to_excel(out_path, index=False)

        in_path = tmp_path / "in.xlsx"
        pd.DataFrame({"MP id": ["mp-1"], "Formula": ["LiFePO4"]}).to_excel(
            in_path, index=False,
        )

        # Mock Phase 1 + Phase 2 to avoid actual MP calls
        with patch(
            "spices.screening.seawater._run_phase1",
            MagicMock(return_value=None),
        ), patch(
            "spices.screening.seawater._process_chunk",
            side_effect=lambda chunk, *a, **kw: chunk,
        ), patch(
            "spices.screening.seawater.get_client",
            return_value=MagicMock(),
        ):
            # Should not raise
            run_seawater_screening(
                element="Li",
                input_path=str(in_path),
                output_path=str(out_path),
                cache_dir=str(tmp_path),
                n_workers=1,
                restart=False,
            )


class TestAtomicFinalWrite:
    """Final xlsx write must use tmp + os.replace; cleanup only on success."""

    def test_runtime_dir_kept_when_write_fails(self, tmp_path):
        in_path = tmp_path / "in.xlsx"
        pd.DataFrame({"MP id": ["mp-1"], "Formula": ["LiFePO4"]}).to_excel(
            in_path, index=False,
        )
        out_path = tmp_path / "out.xlsx"
        runtime = tmp_path / "runtime_sheets"

        # _process_chunk normally creates runtime_sheets/.  Simulate
        # that, then trigger a write failure on the final to_excel call.
        original_to_excel = pd.DataFrame.to_excel

        def failing_to_excel(self_df, path, *a, **kw):
            # Let the input file write succeed; fail only on the final
            # output write (which targets a path under tmp_path).
            if str(path).endswith(".tmp." + str(os.getpid()) + ".xlsx"):
                raise OSError("simulated disk failure")
            return original_to_excel(self_df, path, *a, **kw)

        with patch(
            "spices.screening.seawater._run_phase1",
            MagicMock(return_value=None),
        ), patch(
            "spices.screening.seawater._process_chunk",
            side_effect=lambda chunk, *a, **kw: (
                runtime.mkdir(parents=True, exist_ok=True) or chunk
            ),
        ), patch(
            "spices.screening.seawater.get_client",
            return_value=MagicMock(),
        ), patch.object(pd.DataFrame, "to_excel", failing_to_excel):
            with pytest.raises(OSError, match="simulated"):
                run_seawater_screening(
                    element="Li",
                    input_path=str(in_path),
                    output_path=str(out_path),
                    cache_dir=str(tmp_path),
                    n_workers=1,
                    restart=False,
                )

        # Even though we asked to clean up, runtime_sheets/ should still
        # exist because the final write failed.
        assert runtime.exists()
        # The actual output file should NOT exist (atomic write protected
        # against half-written output).
        assert not out_path.exists()

"""Tests for seawater screening runner functions."""

import sys

import pandas as pd
import pytest

from spices.pourbaix.cache import PourbaixCache
from spices.screening.seawater import (
    SEAWATER_COLUMNS,
    _available_memory_gb,
    _recover_chunk_for_core,
    _split_dataframe,
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

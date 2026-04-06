"""Tests for seawater screening runner functions."""

import pandas as pd
import pytest

from spices.screening.seawater import _split_dataframe, SEAWATER_COLUMNS


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
        assert len(SEAWATER_COLUMNS) == 9

    def test_stability_column_present(self):
        assert "Seawater stability" in SEAWATER_COLUMNS

    def test_all_criteria_present(self):
        for i in range(1, 9):
            col = f"Seawater metastability (Criterion {i})"
            assert col in SEAWATER_COLUMNS

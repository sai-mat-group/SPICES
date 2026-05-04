"""Tests for spices.intercalation.structure — charged-structure
caching and the multi-worker race fix in find_structures.
"""

import os
import threading
import time
from unittest.mock import MagicMock

import pytest

from spices.intercalation.structure import (
    POPULATE_SENTINEL,
    _match_against_saved,
    find_structures,
)


# --------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------


def _write_dummy_cif(path, formula="FeO2"):
    """Write a tiny valid CIF that pymatgen can parse."""
    cif = f"""data_{formula}
_cell_length_a   3.0
_cell_length_b   3.0
_cell_length_c   3.0
_cell_angle_alpha   90.0
_cell_angle_beta    90.0
_cell_angle_gamma   90.0
_space_group_name_H-M_alt  'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Fe1   0.0   0.0   0.0
O1    0.5   0.5   0.5
O2    0.5   0.0   0.0
"""
    with open(path, "w") as f:
        f.write(cif)


def _make_structure(formula="FeO2"):
    from pymatgen.core import Lattice, Structure
    return Structure(
        lattice=Lattice.cubic(3.0),
        species=["Fe", "O", "O"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0]],
    )


# --------------------------------------------------------------------
# _match_against_saved
# --------------------------------------------------------------------


class TestMatchAgainstSaved:
    def test_filters_by_cif_extension(self, tmp_path):
        formula_dir = tmp_path / "FeO2"
        formula_dir.mkdir()
        _write_dummy_cif(str(formula_dir / "mp-1.cif"))
        # Non-CIF files (including the sentinel) must be ignored.
        (formula_dir / POPULATE_SENTINEL).write_text("")
        (formula_dir / "README.md").write_text("ignore me")

        struct = _make_structure()
        matcher = MagicMock()
        matcher.fit.return_value = True

        out = _match_against_saved(struct, str(formula_dir), matcher)
        assert out == ["mp-1"]

    def test_returns_only_matched(self, tmp_path):
        formula_dir = tmp_path / "FeO2"
        formula_dir.mkdir()
        _write_dummy_cif(str(formula_dir / "mp-1.cif"))
        _write_dummy_cif(str(formula_dir / "mp-2.cif"))

        struct = _make_structure()
        matcher = MagicMock()
        matcher.fit.side_effect = [True, False]
        out = _match_against_saved(struct, str(formula_dir), matcher)
        assert out == ["mp-1"]


# --------------------------------------------------------------------
# find_structures — fast path / cold path / backward-compat path
# --------------------------------------------------------------------


class TestFindStructures:
    def _setup_mp_client(self, downloaded_mpids):
        """Mock client whose search() returns objects with material_id +
        a structure that has a .to_file method writing a small CIF.
        """
        mp = MagicMock()
        entries = []
        for mpid in downloaded_mpids:
            entry = MagicMock()
            entry.material_id = mpid

            def _to_file(filename, fmt="cif", _mpid=mpid):
                _write_dummy_cif(filename, formula=_mpid)
            entry.structure.to_file.side_effect = _to_file
            entries.append(entry)
        mp.materials.search.return_value = entries
        return mp

    def test_hot_path_with_sentinel_skips_search(self, tmp_path):
        formula_dir = tmp_path / "FeO2"
        formula_dir.mkdir()
        _write_dummy_cif(str(formula_dir / "mp-99.cif"))
        (formula_dir / POPULATE_SENTINEL).write_text("")

        mp = self._setup_mp_client(["mp-1"])
        matcher = MagicMock()
        matcher.fit.return_value = True

        out = find_structures(
            _make_structure(), mp, str(tmp_path), matcher=matcher,
        )
        # mp_client.search must NOT have been called (sentinel hot path).
        mp.materials.search.assert_not_called()
        assert out == ["mp-99"]

    def test_legacy_dir_without_sentinel_writes_one_and_skips_search(
        self, tmp_path,
    ):
        formula_dir = tmp_path / "FeO2"
        formula_dir.mkdir()
        _write_dummy_cif(str(formula_dir / "mp-77.cif"))

        mp = self._setup_mp_client(["mp-1"])
        matcher = MagicMock()
        matcher.fit.return_value = True

        out = find_structures(
            _make_structure(), mp, str(tmp_path), matcher=matcher,
        )
        mp.materials.search.assert_not_called()
        assert (formula_dir / POPULATE_SENTINEL).exists()
        assert out == ["mp-77"]

    def test_cold_path_downloads_writes_cifs_and_sentinel(self, tmp_path):
        mp = self._setup_mp_client(["mp-1", "mp-2"])
        matcher = MagicMock()
        matcher.fit.return_value = True

        out = find_structures(
            _make_structure(), mp, str(tmp_path), matcher=matcher,
        )
        formula_dir = tmp_path / "FeO2"
        assert (formula_dir / "mp-1.cif").exists()
        assert (formula_dir / "mp-2.cif").exists()
        assert (formula_dir / POPULATE_SENTINEL).exists()
        assert set(out) == {"mp-1", "mp-2"}

    def test_empty_mp_result_still_writes_sentinel(self, tmp_path):
        mp = self._setup_mp_client([])
        out = find_structures(
            _make_structure(), mp, str(tmp_path), matcher=MagicMock(),
        )
        formula_dir = tmp_path / "FeO2"
        assert (formula_dir / POPULATE_SENTINEL).exists()
        assert out == []

    def test_second_call_after_sentinel_is_fast_path(self, tmp_path):
        mp = self._setup_mp_client(["mp-1"])
        matcher = MagicMock()
        matcher.fit.return_value = True

        find_structures(
            _make_structure(), mp, str(tmp_path), matcher=matcher,
        )
        # Reset and second call.
        mp.materials.search.reset_mock()
        find_structures(
            _make_structure(), mp, str(tmp_path), matcher=matcher,
        )
        mp.materials.search.assert_not_called()


# --------------------------------------------------------------------
# Race fix — concurrent populate of the same formula
# --------------------------------------------------------------------


class TestFindStructuresLock:
    """Two threads concurrently populating the same formula must
    cooperate via the per-formula flock + sentinel: only one thread
    actually downloads; the other sees the sentinel after waiting.
    """

    def test_two_threads_same_formula_one_download(self, tmp_path):
        # Single shared mp client whose search() blocks briefly to widen
        # the race window so the second thread is guaranteed to enter
        # while the first holds the lock.
        mp = MagicMock()

        def _slow_search(*_args, **_kwargs):
            time.sleep(0.3)
            entry = MagicMock()
            entry.material_id = "mp-only"

            def _to_file(filename, fmt="cif"):
                _write_dummy_cif(filename, formula="mp-only")
            entry.structure.to_file.side_effect = _to_file
            return [entry]
        mp.materials.search.side_effect = _slow_search

        matcher = MagicMock()
        matcher.fit.return_value = True

        results = []
        errors = []

        def _worker():
            try:
                out = find_structures(
                    _make_structure(), mp,
                    str(tmp_path), matcher=matcher,
                )
                results.append(out)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=_worker)
        t2 = threading.Thread(target=_worker)
        t1.start()
        # Tiny stagger so t1 is guaranteed to enter the lock first.
        time.sleep(0.05)
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert not errors, f"Worker thread errored: {errors}"
        # The lock must have prevented a duplicate download.
        assert mp.materials.search.call_count == 1
        # Both threads got the same answer.
        assert results == [["mp-only"], ["mp-only"]]
        formula_dir = tmp_path / "FeO2"
        assert (formula_dir / POPULATE_SENTINEL).exists()

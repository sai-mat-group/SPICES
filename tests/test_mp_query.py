"""Tests for spices.screening.mp_query."""

import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pymatgen.core import Composition

from spices.screening.mp_query import MP_QUERY_COLUMNS, collect_mp_data_from_chemsys


def _stub_thermo_entry(
    mp_id="mp-1",
    formula="LiFeO2",
    chemsys="Fe-Li-O",
    energy_type="GGA+U",
    energy_per_atom=-7.0,
    energy_above_hull=0.0,
    crystal_system="Cubic",
    space_group="Fm-3m",
):
    entry = MagicMock()
    entry.material_id.string = mp_id
    entry.formula_pretty = formula
    entry.composition = Composition(formula)
    entry.chemsys = chemsys
    entry.energy_type = energy_type
    entry.energy_per_atom = energy_per_atom
    entry.energy_above_hull = energy_above_hull
    entry.symmetry.crystal_system.value = crystal_system
    entry.symmetry.symbol = space_group
    return entry


def _stub_summary(mp_id, theoretical=False):
    s = MagicMock()
    s.material_id.string = mp_id
    s.theoretical = theoretical
    return s


def _stub_mp_client(thermo_entries, summary_entries=None, charged_entries=None):
    """Build a MagicMock that mimics ``mp_client.materials.{thermo,summary}.search``."""
    if summary_entries is None:
        summary_entries = [_stub_summary(e.material_id.string) for e in thermo_entries]
    if charged_entries is None:
        charged_entries = []

    client = MagicMock()
    client.materials.thermo.search.return_value = thermo_entries
    client.materials.summary.search.return_value = summary_entries
    client.materials.search.return_value = charged_entries
    return client


class TestCollectMpDataFromChemsys:
    def test_schema_has_xc_functional_column(self):
        """Output schema is GGA(+U)-only — no Base XC / Correction XC / mixing columns."""
        client = _stub_mp_client([_stub_thermo_entry()])
        rows = collect_mp_data_from_chemsys(
            chemsys="Fe-Li-O",
            element_str="Li",
            mp_client=client,
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["XC Functional"] == "GGA+U"
        assert "Base XC Functional" not in row
        assert "Correction XC functional" not in row
        assert "GGA(+U)/R2SCAN mixing adjustment" not in row
        assert "Thermo-type" not in row
        assert set(row.keys()) == set(MP_QUERY_COLUMNS)

    def test_thermo_search_restricted_to_gga_gga_u(self):
        """Live API call must pass thermo_types=['GGA_GGA+U']."""
        client = _stub_mp_client([_stub_thermo_entry()])
        collect_mp_data_from_chemsys(
            chemsys="Fe-Li-O",
            element_str="Li",
            mp_client=client,
        )
        kwargs = client.materials.thermo.search.call_args.kwargs
        assert kwargs["thermo_types"] == ["GGA_GGA+U"]

    def test_dedup_warns_on_shrink(self, caplog):
        """A duplicate MP id in the thermo result triggers a logger warning."""
        dup = _stub_thermo_entry(mp_id="mp-dup")
        client = _stub_mp_client([dup, dup])
        with caplog.at_level(logging.WARNING, logger="spices.screening.mp_query"):
            rows = collect_mp_data_from_chemsys(
                chemsys="Fe-Li-O",
                element_str="Li",
                mp_client=client,
            )
        assert len(rows) == 1
        assert any(
            "step-0 dedup dropped" in rec.message for rec in caplog.records
        )

    def test_no_dedup_warning_when_unique(self, caplog):
        """Distinct MP ids do not trigger the warning."""
        client = _stub_mp_client([
            _stub_thermo_entry(mp_id="mp-1"),
            _stub_thermo_entry(mp_id="mp-2"),
        ])
        with caplog.at_level(logging.WARNING, logger="spices.screening.mp_query"):
            rows = collect_mp_data_from_chemsys(
                chemsys="Fe-Li-O",
                element_str="Li",
                mp_client=client,
            )
        assert len(rows) == 2
        assert not any(
            "step-0 dedup dropped" in rec.message for rec in caplog.records
        )

    def test_empty_thermo_list_returns_empty(self, caplog):
        """Empty thermo.search result returns [] (not a NaN sentinel row)."""
        client = _stub_mp_client([])
        with caplog.at_level(logging.WARNING, logger="spices.screening.mp_query"):
            rows = collect_mp_data_from_chemsys(
                chemsys=None,
                element_str="Li",
                mp_client=client,
                material_ids=["mp-nonexistent"],
            )
        assert rows == []
        assert any(
            "thermo.search returned 0 results" in rec.message
            for rec in caplog.records
        )

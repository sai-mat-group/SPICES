"""Tests for intercalation convex hull functions."""

import logging

import pandas as pd
import pytest
from pymatgen.core import Composition

from spices.intercalation.hull import (
    create_pd_entry,
    filter_df_to_get_on_hull_phases,
    get_working_ions_per_framework,
)


class TestCreatePdEntry:
    def test_creates_entry(self):
        entry = create_pd_entry("CaFe2O4", -30.0, name="mp-123")
        # pymatgen reduces CaFe2O4 to Ca(FeO2)2
        assert Composition(entry.composition).reduced_formula == Composition("CaFe2O4").reduced_formula
        assert entry.energy == -30.0
        assert entry.attribute == "mp-123"

    def test_no_name(self):
        entry = create_pd_entry("Fe2O3", -20.0)
        assert entry.attribute is None


class TestGetWorkingIonsPerFramework:
    def test_half_ca_per_framework(self):
        # CaFe2O4 → framework unit = FeO2 (reduced). 2 framework units.
        # 1 Ca / 2 framework units = 0.5
        result = get_working_ions_per_framework("CaFe2O4", "Ca")
        assert abs(result - 0.5) < 1e-10

    def test_one_ca_per_framework(self):
        # Ca2Fe2O4 → framework unit = FeO2. 2 framework units.
        # 2 Ca / 2 framework units = 1.0
        result = get_working_ions_per_framework("Ca2Fe2O4", "Ca")
        assert abs(result - 1.0) < 1e-10

    def test_no_working_ion(self):
        # Fe2O4 has 0 Ca
        result = get_working_ions_per_framework("Fe2O4", "Ca")
        assert abs(result - 0.0) < 1e-10

    def test_fractional_working_ions(self):
        # Ca3Fe4O8 → framework = FeO2 (reduced), Fe=4 in discharged, Fe=1 in charged
        # no_of_frameworks = 4/1 = 4, result = 3/4 = 0.75
        result = get_working_ions_per_framework("Ca3Fe4O8", "Ca")
        assert abs(result - 0.75) < 1e-10

    def test_li_working_ion(self):
        result = get_working_ions_per_framework("LiFePO4", "Li")
        assert abs(result - 1.0) < 1e-10

    def test_na_working_ion(self):
        result = get_working_ions_per_framework("Na2MnO3", "Na")
        assert abs(result - 2.0) < 1e-10


class TestFilterDfDedupWarning:
    def _df(self, mp_ids):
        return pd.DataFrame({
            "MP id": mp_ids,
            "Formula": ["LiFeO2"] * len(mp_ids),
            "E (eV/formula)": [-30.0] * len(mp_ids),
            "XC Functional": ["GGA+U"] * len(mp_ids),
        })

    def test_warns_when_duplicates_present(self, caplog):
        df = self._df(["mp-1", "mp-1", "mp-2"])
        with caplog.at_level(logging.WARNING, logger="spices.intercalation.hull"):
            filter_df_to_get_on_hull_phases(df, "Li")
        assert any(
            "hull dedup dropped" in rec.message for rec in caplog.records
        )

    def test_silent_when_unique(self, caplog):
        df = self._df(["mp-1", "mp-2"])
        with caplog.at_level(logging.WARNING, logger="spices.intercalation.hull"):
            filter_df_to_get_on_hull_phases(df, "Li")
        assert not any(
            "hull dedup dropped" in rec.message for rec in caplog.records
        )

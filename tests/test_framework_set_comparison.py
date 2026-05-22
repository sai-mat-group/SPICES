"""Tests for set-based framework comparison module."""

from unittest.mock import MagicMock

from spices.screening.framework_comparison import FrameworkStructure
from spices.screening.framework_set_comparison import (
    framework_set_intersection,
    multi_ion_common_frameworks,
    ion_selective_frameworks,
    _get_working_ion_fraction,
)


def _fake_fw(mp_id, ion_symbol="Li"):
    """Mock FrameworkStructure that is identity-distinct but mp_id-keyed."""
    fw = MagicMock(spec=FrameworkStructure)
    fw.mp_id = mp_id
    wi = MagicMock(); wi.symbol = ion_symbol
    fw.working_ion = wi
    return fw


class TestSetOperationsByMpid:
    """Verify the BUG 3 fix: set ops match by mp_id, not by identity."""

    def test_intersection_matches_by_mpid(self):
        """Frameworks loaded as separate instances but sharing mp_id match."""
        a1 = _fake_fw("mp-1", "Li")
        a2 = _fake_fw("mp-2", "Li")
        # Other ion: same mp-1 but distinct Python instance
        b1 = _fake_fw("mp-1", "Na")
        b3 = _fake_fw("mp-3", "Na")

        result = framework_set_intersection({a1, a2}, set(), {b1, b3}, set())
        # mp-1 is shared; should appear in result (from req side, i.e. a1)
        mp_ids = {fw.mp_id for fw in result}
        assert mp_ids == {"mp-1"}

    def test_intersection_no_common(self):
        a = _fake_fw("mp-1")
        b = _fake_fw("mp-2")
        result = framework_set_intersection({a}, set(), {b}, set())
        assert len(result) == 0

    def test_multi_ion_common(self):
        a = _fake_fw("mp-1", "Li")
        b = _fake_fw("mp-1", "Na")  # shared
        c = _fake_fw("mp-1", "K")   # shared with all
        d = _fake_fw("mp-2", "Li")  # only in Li

        result = multi_ion_common_frameworks(
            {a, d}, set(), [{b}, {c}], [set(), set()],
        )
        mp_ids = {fw.mp_id for fw in result}
        assert mp_ids == {"mp-1"}

    def test_ion_selective(self):
        a = _fake_fw("mp-1", "Li")  # shared
        b = _fake_fw("mp-1", "Na")  # shared
        c = _fake_fw("mp-2", "Li")  # selective

        result = ion_selective_frameworks(
            {a, c}, set(), [{b}], [set()],
        )
        mp_ids = {fw.mp_id for fw in result}
        assert mp_ids == {"mp-2"}


class TestWorkingIonFraction:
    def test_full_intercalation(self):
        # CaFeO2 has 1 Ca per FeO2 framework unit
        frac = _get_working_ion_fraction("CaFeO2", "FeO2", "Ca")
        assert abs(frac - 1.0) < 1e-6

    def test_no_working_ion(self):
        frac = _get_working_ion_fraction("FeO2", "FeO2", "Ca")
        assert frac == 0

    def test_partial_intercalation(self):
        # Ca2Fe2O4 = Ca(FeO2) has 1 Ca per FeO2 framework
        frac = _get_working_ion_fraction("Ca2Fe2O4", "FeO2", "Ca")
        assert abs(frac - 1.0) < 1e-6

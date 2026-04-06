"""Tests for spices.pourbaix.decomposition."""

from spices.pourbaix.decomposition import get_mp_id_from_pbx_entry_id


class TestGetMpIdFromPbxEntryId:
    def test_standard_format(self):
        assert get_mp_id_from_pbx_entry_id("mp-19017-GGA_GGA+U_R2SCAN") == "mp-19017"

    def test_simple_format(self):
        assert get_mp_id_from_pbx_entry_id("mp-4820-GGA") == "mp-4820"

    def test_ion_format(self):
        assert get_mp_id_from_pbx_entry_id("ion-0-Fe") == "ion-0"

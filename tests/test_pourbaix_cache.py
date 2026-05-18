"""Tests for the PourbaixCache disk/memory routing.

The regressions in this module pin the rule that ``generate_pourbaix_diagram``
and ``generate_sea_pourbaix_diagram`` MUST read entries from the on-disk
pickle (``{chemsys}_pbxentry.pkl``) before falling back to ``mp_client``.
A previous version called the memory-only ``get_or_download_pourbaix_entries``,
which caused fresh worker processes to rebuild entries from MP even though
Phase 1a had saved them to disk — a 3-hour wasted build per 9-element chemsys.
"""

import pickle
from unittest.mock import MagicMock, patch

import pytest
from pymatgen.core import Composition

from spices.pourbaix.cache import PourbaixCache
from spices.pourbaix.composition import (
    get_pourbaix_composition,
    get_sea_pourbaix_comp_dict,
)


class TestDiskCachedEntriesAreReused:
    """The diagram-generation paths must not call MP when entries are on disk."""

    def test_generate_sea_pourbaix_diagram_uses_disk_cached_entries(self, tmp_path):
        entries_dir = tmp_path / "saved_pbx_entries"
        entries_dir.mkdir()
        pbx_dir = tmp_path / "saved_pbx"
        pbx_dir.mkdir()

        # Compute the chemsys generate_sea_pourbaix_diagram will look for.
        cathode = Composition("Li4 V3 Cr2 Co1 P6 Cl1 Na1 Mg1 S1")
        pbx_comp = get_pourbaix_composition(cathode)
        sea_comp_dict = get_sea_pourbaix_comp_dict(pbx_comp, num_of_sea_elements=4)
        chemsys = Composition(sea_comp_dict).chemical_system

        sentinel = ["fake_entry_1", "fake_entry_2"]
        with open(entries_dir / f"{chemsys}_pbxentry.pkl", "wb") as f:
            pickle.dump(sentinel, f)

        mp_client = MagicMock()
        mp_client.get_pourbaix_entries.side_effect = AssertionError(
            "mp_client.get_pourbaix_entries must not be called when "
            f"{chemsys}_pbxentry.pkl is on disk"
        )

        cache = PourbaixCache(
            entries_dir=str(entries_dir), pbx_dir=str(pbx_dir),
        )

        with patch("spices.pourbaix.cache.PourbaixDiagram") as MockPbx:
            cache.generate_sea_pourbaix_diagram(
                cathode, mp_client,
                num_of_sea_elements=4, save_entries=True,
            )

        mp_client.get_pourbaix_entries.assert_not_called()
        # Pickle deserializes to an equal-but-not-identical list, so check by
        # value: the entries handed to PourbaixDiagram must be the disk pickle.
        assert MockPbx.call_args.args[0] == sentinel

    def test_generate_pourbaix_diagram_uses_disk_cached_entries(self, tmp_path):
        entries_dir = tmp_path / "saved_pbx_entries"
        entries_dir.mkdir()
        pbx_dir = tmp_path / "saved_pbx"
        pbx_dir.mkdir()

        cathode = Composition("Li4 V3 Cr2 Co1 P6")
        pbx_comp = get_pourbaix_composition(cathode)
        chemsys = pbx_comp.chemical_system

        sentinel = ["fake_entry_a", "fake_entry_b"]
        with open(entries_dir / f"{chemsys}_pbxentry.pkl", "wb") as f:
            pickle.dump(sentinel, f)

        mp_client = MagicMock()
        mp_client.get_pourbaix_entries.side_effect = AssertionError(
            "mp_client.get_pourbaix_entries must not be called when "
            f"{chemsys}_pbxentry.pkl is on disk"
        )

        cache = PourbaixCache(
            entries_dir=str(entries_dir), pbx_dir=str(pbx_dir),
        )

        with patch("spices.pourbaix.cache.PourbaixDiagram") as MockPbx:
            cache.generate_pourbaix_diagram(
                pbx_comp, mp_client, save_entries=True,
            )

        mp_client.get_pourbaix_entries.assert_not_called()
        assert MockPbx.call_args.args[0] == sentinel

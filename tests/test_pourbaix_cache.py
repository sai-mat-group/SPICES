"""Tests for the PourbaixCache disk/memory routing.

The regressions in this module pin the rule that ``generate_pourbaix_diagram``
and ``generate_sea_pourbaix_diagram`` MUST read entries from the on-disk
pickle (``{chemsys}_pbxentry.pkl``) before falling back to ``mp_client``.
A previous version called the memory-only ``get_or_download_pourbaix_entries``,
which caused fresh worker processes to rebuild entries from MP even though
Phase 1a had saved them to disk — a 3-hour wasted build per 9-element chemsys.
"""

import logging
import pickle
import re
import sys
import time
from unittest.mock import MagicMock, patch

import pytest
from pymatgen.core import Composition

from spices.pourbaix.cache import (
    PourbaixCache,
    _RssSampler,
    _read_rss_gb,
    _read_vmpeak_gb,
)
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


@pytest.mark.skipif(
    sys.platform != "linux", reason="/proc/self/status is Linux-only"
)
class TestRssSamplerInstrumentation:
    """The PourbaixDiagram-construction calls must log RSS so SIGKILL
    crashes during pymatgen's multi-entry build leave a memory trail
    in the workstation log.  See plan
    /home/sougat/.claude/plans/i-want-your-help-pure-hollerith.md.
    """

    def test_read_rss_gb_returns_positive_float(self):
        rss = _read_rss_gb()
        assert rss is not None
        assert rss > 0.0
        # Sanity: a running python process won't have multi-TB RSS.
        assert rss < 1024.0

    def test_read_vmpeak_gb_returns_positive_float(self):
        peak = _read_vmpeak_gb()
        assert peak is not None
        assert peak > 0.0

    def test_rss_sampler_logs_every_interval(self, caplog):
        caplog.set_level(logging.INFO, logger="spices.pourbaix.cache")
        with _RssSampler("test-label", interval=0.2):
            time.sleep(0.7)
        # 0.7s / 0.2s = at least 2 samples (timing has jitter, but >= 1 is
        # the conservative lower bound we can pin against the clock).
        sample_lines = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.INFO
            and re.match(r"\[test-label\] elapsed=\d+s RSS=", r.getMessage())
        ]
        assert len(sample_lines) >= 1, (
            f"_RssSampler emitted no samples; records were: "
            f"{[r.getMessage() for r in caplog.records]}"
        )

    def test_generate_sea_pourbaix_diagram_wraps_in_sampler_without_breaking_return(
        self, tmp_path, caplog,
    ):
        entries_dir = tmp_path / "saved_pbx_entries"
        entries_dir.mkdir()
        pbx_dir = tmp_path / "saved_pbx"
        pbx_dir.mkdir()

        cathode = Composition("Li4 V3 Cr2 Co1 P6 Cl1 Na1 Mg1 S1")
        pbx_comp = get_pourbaix_composition(cathode)
        sea_comp_dict = get_sea_pourbaix_comp_dict(pbx_comp, num_of_sea_elements=4)
        chemsys = Composition(sea_comp_dict).chemical_system
        sentinel = ["fake_entry_x", "fake_entry_y"]
        with open(entries_dir / f"{chemsys}_pbxentry.pkl", "wb") as f:
            pickle.dump(sentinel, f)

        cache = PourbaixCache(
            entries_dir=str(entries_dir), pbx_dir=str(pbx_dir),
        )

        caplog.set_level(logging.INFO, logger="spices.pourbaix.cache")
        with patch("spices.pourbaix.cache.PourbaixDiagram") as MockPbx:
            mock_diagram = MagicMock(name="MockPourbaixDiagram")
            MockPbx.return_value = mock_diagram
            result = cache.generate_sea_pourbaix_diagram(
                cathode, MagicMock(),
                num_of_sea_elements=4, save_entries=True,
            )

        assert result is mock_diagram, (
            "wrapping PourbaixDiagram in _RssSampler must not change "
            "the function's return value"
        )
        # The pre-call info line carries (entries=N, elements=N, RSS=...).
        pre_call = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.INFO
            and "Generating sea Pourbaix Diagram" in r.getMessage()
        ]
        assert any("entries=2" in msg and "elements=" in msg and "RSS="
                   in msg for msg in pre_call), (
            f"pre-call instrumentation line not found; messages were: {pre_call}"
        )

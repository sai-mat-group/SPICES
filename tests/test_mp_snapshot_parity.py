"""
Live-vs-snapshot parity tests for the SPICES MP snapshot backend.

Marked ``integration`` because they require (a) the bulk dump already
downloaded under ``SPICES_MP_SNAPSHOT_ROOT`` and (b) an ``MP_API_KEY``
for the live-side comparison.  Skip automatically when either is absent.
"""

import os

import pytest

from spices.mp_client import get_client


SNAPSHOT_ROOT = os.environ.get("SPICES_MP_SNAPSHOT_ROOT")
RELEASE = os.environ.get("SPICES_MP_SNAPSHOT_RELEASE", "2025-06-09")
HAS_API_KEY = bool(os.environ.get("MP_API_KEY"))


def _snapshot_ready():
    if not SNAPSHOT_ROOT:
        return False
    return os.path.isdir(os.path.join(SNAPSHOT_ROOT, RELEASE, "thermo"))


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def snapshot_client():
    if not _snapshot_ready():
        pytest.skip("Snapshot not downloaded under SPICES_MP_SNAPSHOT_ROOT")
    return get_client(backend="snapshot", snapshot_root=SNAPSHOT_ROOT, release=RELEASE)


@pytest.fixture(scope="module")
def live_client():
    if not HAS_API_KEY:
        pytest.skip("MP_API_KEY not set — live client unavailable")
    return get_client(backend="live")


def test_snapshot_imports_and_attributes(snapshot_client):
    assert snapshot_client.release == RELEASE
    assert snapshot_client.get_database_version() == RELEASE
    # Strict surface: unknown attribute must raise, not return None.
    from spices.mp_snapshot.client import SnapshotBackendError
    with pytest.raises(SnapshotBackendError):
        _ = snapshot_client.provenance  # not implemented on purpose


def test_snapshot_thermo_unsupported_type_raises(snapshot_client):
    from spices.mp_snapshot.client import SnapshotBackendError
    # SPICES restricts the snapshot to GGA_GGA+U only (parity with the
    # live MP API's hardcoded Pourbaix thermo_type).  R2SCAN-only and
    # GGA_GGA+U_R2SCAN must both be rejected.
    with pytest.raises(SnapshotBackendError):
        snapshot_client.materials.thermo.search(
            chemsys="Li-O",
            thermo_types=["R2SCAN"],
        )
    with pytest.raises(SnapshotBackendError):
        snapshot_client.materials.thermo.search(
            chemsys="Li-O",
            thermo_types=["GGA_GGA+U_R2SCAN"],
        )
    # GGA_GGA+U IS shipped — must not raise.
    snapshot_client.materials.thermo.search(
        chemsys="Li-O",
        thermo_types=["GGA_GGA+U"],
    )


def test_snapshot_thermo_returns_docs(snapshot_client):
    docs = snapshot_client.materials.thermo.search(chemsys="Li-O")
    assert len(docs) > 0
    d0 = docs[0]
    # Shape SPICES depends on:
    assert d0.material_id.string.startswith("mp-")
    assert d0.formula_pretty  # non-empty
    assert d0.thermo_type.value == "GGA_GGA+U"
    assert d0.symmetry is not None
    assert d0.symmetry.crystal_system.value  # e.g. "Cubic"
    assert d0.symmetry.symbol  # space group symbol
    # entries[xc].energy_adjustments is what find_correction iterates.
    xc = d0.energy_type
    adjs = d0.entries[xc].energy_adjustments
    for a in adjs:
        assert "name" in a.as_dict()
        assert "description" in a.as_dict()


def test_snapshot_thermo_returns_only_gga_gga_u(snapshot_client):
    """SPICES restricts the snapshot to thermo_type=GGA_GGA+U.

    The default search must return only that thermo_type, regardless of
    whether the GGA_GGA+U_R2SCAN subroot still exists on disk from a
    previous download — the loader's COLLECTIONS dict no longer indexes
    the R2SCAN subroot.
    """
    docs = snapshot_client.materials.thermo.search(chemsys="Li-Ti-O")
    thermo_types_seen = {d.thermo_type.value for d in docs}
    assert thermo_types_seen == {"GGA_GGA+U"}, (
        f"Expected only GGA_GGA+U thermo_type, got {sorted(thermo_types_seen)}"
    )

    # Explicit single-type call must yield the same set of mp-ids.
    explicit_docs = snapshot_client.materials.thermo.search(
        chemsys="Li-Ti-O",
        thermo_types=["GGA_GGA+U"],
    )
    default_ids = {d.material_id.string for d in docs}
    explicit_ids = {d.material_id.string for d in explicit_docs}
    assert default_ids == explicit_ids


def test_snapshot_materials_formula_reduction(snapshot_client):
    """Non-reduced formula strings from SPICES' rm_element must match."""
    raw = snapshot_client.materials.search(formula="Ti2 O4")
    reduced = snapshot_client.materials.search(formula="TiO2")
    raw_ids = {d.material_id.string for d in raw}
    red_ids = {d.material_id.string for d in reduced}
    assert len(raw_ids) > 0, "Reduced-formula lookup returned no hits"
    assert raw_ids == red_ids, (
        "Ti2 O4 and TiO2 should resolve to the same mp-id set"
    )


def test_snapshot_summary_theoretical(snapshot_client):
    docs = snapshot_client.materials.thermo.search(chemsys="Li-O")
    mids = [d.material_id.string for d in docs][:10]
    summaries = snapshot_client.materials.summary.search(material_ids=mids)
    assert len(summaries) == len(mids)
    for s in summaries:
        assert s.material_id.string.startswith("mp-")
        assert isinstance(s.theoretical, bool)


@pytest.mark.skipif(not HAS_API_KEY, reason="MP_API_KEY not set")
def test_live_vs_snapshot_thermo_chemsys(live_client, snapshot_client):
    """
    Material-ID-set parity for a small chemsys.  Energies should also
    match within tight tolerance for IDs present on both sides.
    """
    chemsys = "Li-Mn-O"
    live_docs = live_client.materials.thermo.search(
        chemsys=chemsys,
        thermo_types=["GGA_GGA+U"],
        fields=["material_id", "energy_per_atom", "energy_above_hull"],
    )
    # Restrict snapshot to the same thermo_type for an apples-to-apples
    # mp-id and energy compare; the snapshot now ships both prefixes.
    snap_docs = snapshot_client.materials.thermo.search(
        chemsys=chemsys,
        thermo_types=["GGA_GGA+U"],
    )

    live_ids = {d.material_id.string for d in live_docs}
    snap_ids = {d.material_id.string for d in snap_docs}

    # Snapshot is frozen at 2025-06-09.  Live DB may have added new
    # entries; it should NOT have removed entries (short of deprecation).
    # So snap_ids ⊆ live_ids modulo deprecations.  We only assert the
    # intersection is non-trivial.
    common = live_ids & snap_ids
    assert len(common) >= 1, (
        f"No common IDs for {chemsys}: live={len(live_ids)}, "
        f"snap={len(snap_ids)}"
    )
    # Energy parity on common IDs.
    live_e = {d.material_id.string: d.energy_per_atom for d in live_docs}
    snap_e = {d.material_id.string: d.energy_per_atom for d in snap_docs}
    for mid in common:
        assert abs(live_e[mid] - snap_e[mid]) < 1e-6, (
            f"Energy mismatch for {mid}: live={live_e[mid]}, "
            f"snap={snap_e[mid]}"
        )

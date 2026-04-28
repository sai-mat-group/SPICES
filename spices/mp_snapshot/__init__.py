"""
Offline Materials Project snapshot backend for SPICES.

Mirrors the subset of the ``mp_api.client.MPRester`` surface that SPICES
calls, but serves data from a local bulk dump of a frozen MP release
(e.g. ``2025-06-09``) under ``s3://materialsproject-build/collections/``.

The point is reproducibility: once MP's live database moves on, live-API
queries stop matching results from an earlier iteration.  Pointing SPICES
at a snapshot via
``spices.mp_client.get_client(backend="snapshot", snapshot_root=...)``
gives bit-identical query results for a fixed release.

Strict by construction: the client raises ``SnapshotBackendError`` on any
method or argument it does not implement, so the pipeline cannot silently
fall back to live MP and mix data from two database versions.
"""

# Single source of truth for the MP thermo_type SPICES uses.  Step 1/2
# Pourbaix diagrams are GGA(+U)-only by construction (live mp_api hardcodes
# this thermo_type for get_pourbaix_entries); steps 0 / 2 / 3 follow suit
# so the entire pipeline operates on one consistent functional family.
THERMO_TYPE = "GGA_GGA+U"


def __getattr__(name):
    # Lazy re-export so ``spices.mp_snapshot.download`` can be imported
    # (and run by the user) before the full client/loader modules exist.
    if name in ("SnapshotMPRester", "SnapshotBackendError"):
        from spices.mp_snapshot.client import (  # noqa: F401
            SnapshotMPRester, SnapshotBackendError,
        )
        return {"SnapshotMPRester": SnapshotMPRester,
                "SnapshotBackendError": SnapshotBackendError}[name]
    raise AttributeError(name)


__all__ = ["SnapshotMPRester", "SnapshotBackendError", "THERMO_TYPE"]

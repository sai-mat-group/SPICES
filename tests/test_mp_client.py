"""
Tests for ``spices.mp_client.get_client`` resolution.

The sticky-default fast path is the contract that lets a top-level
``get_client(backend="snapshot", ...)`` call carry through to inner
no-arg ``get_client()`` calls inside pipeline entry points like
``run_mp_query``.  An earlier version of the guard checked
``"MP_API_KEY" not in os.environ``, which silently disabled the
shortcut for any user with the env var exported (the standard live
setup).  These tests pin the corrected behavior.
"""

import os

import pytest

import spices.mp_client as mp_client_module
from spices.mp_client import get_client


SNAPSHOT_ROOT = os.environ.get("SPICES_MP_SNAPSHOT_ROOT")
RELEASE = os.environ.get("SPICES_MP_SNAPSHOT_RELEASE", "2025-06-09")


def _snapshot_ready():
    if not SNAPSHOT_ROOT:
        return False
    return os.path.isdir(os.path.join(SNAPSHOT_ROOT, RELEASE, "thermo"))


@pytest.fixture(autouse=True)
def _isolate_module_state(monkeypatch):
    """Reset ``_last_client`` / ``_clients`` per test so cached state
    from earlier tests cannot leak across cases."""
    monkeypatch.setattr(mp_client_module, "_last_client", None)
    monkeypatch.setattr(mp_client_module, "_clients", {})


@pytest.mark.integration
def test_sticky_default_returns_snapshot_when_MP_API_KEY_set(monkeypatch):
    """Regression: with MP_API_KEY exported, an inner no-arg
    ``get_client()`` must still return the snapshot client built at
    the top level — not silently fall through to a live MPRester."""
    if not _snapshot_ready():
        pytest.skip("Snapshot not downloaded under SPICES_MP_SNAPSHOT_ROOT")

    monkeypatch.setenv("MP_API_KEY", "fake-key-not-used")
    monkeypatch.delenv("SPICES_MP_BACKEND", raising=False)

    snap = get_client(
        backend="snapshot",
        snapshot_root=SNAPSHOT_ROOT,
        release=RELEASE,
    )
    assert type(snap).__name__ == "SnapshotMPRester"

    # Inner no-arg call must reuse the same instance.
    again = get_client()
    assert again is snap
    assert type(again).__name__ == "SnapshotMPRester"


def test_explicit_backend_live_overrides_sticky(monkeypatch):
    """Passing ``backend="live"`` explicitly must rebuild a live
    client even when the sticky-default already holds a snapshot."""
    sentinel_live = object()

    def fake_mprester(api_key):
        return sentinel_live

    monkeypatch.setenv("MP_API_KEY", "x" * 32)
    monkeypatch.setattr(mp_client_module, "MPRester", fake_mprester)

    # Seed the sticky-default with a fake snapshot client.
    sentinel_snapshot = object()
    monkeypatch.setattr(mp_client_module, "_last_client", sentinel_snapshot)

    live = get_client(backend="live")
    assert live is sentinel_live
    assert live is not sentinel_snapshot


def test_sticky_default_does_not_fire_with_explicit_args(monkeypatch):
    """Any explicit positional kwarg disables the sticky-default
    shortcut and falls through to backend resolution."""
    sentinel_live = object()

    def fake_mprester(api_key):
        return sentinel_live

    monkeypatch.setattr(mp_client_module, "MPRester", fake_mprester)

    sentinel_snapshot = object()
    monkeypatch.setattr(mp_client_module, "_last_client", sentinel_snapshot)

    # api_key=... means the call is *not* truly no-arg.
    result = get_client(api_key="x" * 32)
    assert result is sentinel_live
    assert result is not sentinel_snapshot

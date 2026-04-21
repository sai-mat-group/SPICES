"""
Centralized Materials Project client management.

Two backends are supported:

- ``"live"`` (default): the upstream ``mp_api.client.MPRester`` hitting
  the current Materials Project database.
- ``"snapshot"``: an offline :class:`spices.mp_snapshot.client.SnapshotMPRester`
  serving a frozen release (e.g. ``2025-06-09``) from a local bulk dump.
  Used to reproduce earlier iterations' results after MP's live DB has
  moved on.

Every SPICES function that queries MP accepts an ``mp_client`` parameter,
so downstream code doesn't care which backend it got — switching is a
single call-site change (or an env var).
"""

import os

from mp_api.client import MPRester

# Cache keyed by (backend, identifier) so repeated get_client calls with
# the same config reuse the same instance.  Identifier is the API key
# for "live" and (snapshot_root, release) for "snapshot".
_clients: dict = {}


def get_client(
    api_key=None,
    backend=None,
    snapshot_root=None,
    release=None,
):
    """
    Return an MP client — either live or a local snapshot backend.

    Backend resolution order:
      1. Explicit ``backend`` argument.
      2. ``SPICES_MP_BACKEND`` environment variable.
      3. ``"live"`` (default).

    For ``backend="live"``, API key resolution is:
      1. Explicit ``api_key`` argument.
      2. ``MP_API_KEY`` environment variable.
      3. ``.env`` file in CWD (via python-dotenv if installed).

    For ``backend="snapshot"``, snapshot location is:
      1. Explicit ``snapshot_root`` argument (or ``SPICES_MP_SNAPSHOT_ROOT``).
      2. Explicit ``release`` argument (or ``SPICES_MP_SNAPSHOT_RELEASE``,
         default ``"2025-06-09"``).

    Parameters
    ----------
    api_key : str or None
        Materials Project API key (live backend only).
    backend : {"live", "snapshot"} or None
        Which backend to use.  If None, resolved from the environment.
    snapshot_root : str or None
        Path to the local snapshot root (snapshot backend only).
    release : str or None
        Release label, e.g. ``"2025-06-09"`` (snapshot backend only).

    Returns
    -------
    MPRester or SnapshotMPRester
        Both expose the subset of the MPRester surface that SPICES uses.

    Raises
    ------
    ValueError
        If the live backend is selected but no API key can be found, or
        the snapshot backend is selected but no snapshot_root resolves.
    """
    if backend is None:
        backend = os.environ.get("SPICES_MP_BACKEND", "live").lower()

    if backend == "live":
        return _get_live_client(api_key)
    if backend == "snapshot":
        return _get_snapshot_client(snapshot_root, release)
    raise ValueError(
        f"Unknown backend '{backend}'. Use 'live' or 'snapshot'."
    )


def _get_live_client(api_key):
    if api_key is None:
        api_key = os.environ.get("MP_API_KEY")

    if api_key is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("MP_API_KEY")
        except ImportError:
            pass

    if api_key is None:
        raise ValueError(
            "No Materials Project API key found. Set the MP_API_KEY "
            "environment variable, create a .env file, or pass api_key "
            "explicitly."
        )

    key = ("live", api_key)
    if key not in _clients:
        _clients[key] = MPRester(api_key)
    return _clients[key]


def _get_snapshot_client(snapshot_root, release):
    if snapshot_root is None:
        snapshot_root = os.environ.get("SPICES_MP_SNAPSHOT_ROOT")
    if release is None:
        release = os.environ.get("SPICES_MP_SNAPSHOT_RELEASE", "2025-06-09")

    if snapshot_root is None:
        raise ValueError(
            "Snapshot backend selected but no snapshot_root given. Pass "
            "snapshot_root= to get_client() or set "
            "SPICES_MP_SNAPSHOT_ROOT in the environment."
        )

    # Local import so the snapshot subpackage stays optional at import
    # time — users on pure live-API setups never pay for boto3/etc.
    from spices.mp_snapshot.client import SnapshotMPRester

    key = ("snapshot", snapshot_root, release)
    if key not in _clients:
        _clients[key] = SnapshotMPRester(snapshot_root, release=release)
    return _clients[key]


def set_api_key(api_key):
    """
    Set (or replace) the default API key for this session.

    Only affects the live backend.  Snapshot instances cached for other
    backends are untouched.
    """
    os.environ["MP_API_KEY"] = api_key
    _clients[("live", api_key)] = MPRester(api_key)

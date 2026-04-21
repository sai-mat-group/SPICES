"""
SnapshotMPRester: drop-in replacement for ``mp_api.client.MPRester``
that serves the subset of calls SPICES makes from a local MP snapshot.

Strict by construction.  Every attribute or method not in the
implemented surface raises :class:`SnapshotBackendError`, so a pipeline
pointed at a snapshot can never silently fall through to the live
database and mix eras.  Pourbaix-diagram reconstruction is delegated to
:mod:`spices.mp_snapshot.pourbaix` so this module stays focused on the
MPRester surface contract.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from pymatgen.core import Composition, Structure

from spices.mp_snapshot.loader import SnapshotLoader

logger = logging.getLogger(__name__)

SUPPORTED_THERMO_TYPE = "GGA_GGA+U"


class SnapshotBackendError(RuntimeError):
    """Raised when SPICES asks the snapshot backend for something it
    cannot honor — unknown method, unsupported thermo_type, etc.

    Never silently fall through to live MP: that would mix two database
    versions in one pipeline run.
    """


# ----------------------------------------------------------------------
# Lightweight doc wrappers.  Expose only the attributes SPICES reads.
# ----------------------------------------------------------------------


class _MPID(str):
    """Mimics the ``MPID`` string-subclass used by ``mp_api``.

    SPICES variously reads ``.material_id`` as a plain string AND as
    ``.material_id.string``; making ``_MPID`` a ``str`` subclass with an
    added ``.string`` attribute satisfies both access patterns with one
    type.
    """

    def __new__(cls, value):
        inst = super().__new__(cls, value)
        return inst

    @property
    def string(self) -> str:
        return str(self)


@dataclass
class _EnumLike:
    """Mimics an enum value: exposes ``.value`` returning the raw string."""
    value: str


class _EnergyAdjustment:
    """Lazy wrapper over a serialized EnergyAdjustment dict.

    SPICES' ``find_correction`` calls ``adj.as_dict()["name"]`` and
    ``["description"]`` — so we expose an ``as_dict`` method that returns
    the stored dict verbatim.
    """

    __slots__ = ("_d",)

    def __init__(self, d: dict):
        self._d = d

    def as_dict(self) -> dict:
        return self._d


class _Entry:
    """Lazy wrapper over a serialized ComputedStructureEntry dict.

    SPICES reads ``.energy_adjustments`` only.  Other attributes (full
    Structure, parameters, etc.) are only materialized on demand by
    code that requests them (e.g. offline Pourbaix rebuild).
    """

    __slots__ = ("_d",)

    def __init__(self, d: dict):
        self._d = d

    @property
    def energy_adjustments(self):
        return [_EnergyAdjustment(a) for a in self._d.get("energy_adjustments", [])]

    @property
    def raw(self) -> dict:
        """Escape hatch — full monty-serialized CSE dict, for pourbaix.py."""
        return self._d


@dataclass
class _Symmetry:
    crystal_system: _EnumLike
    symbol: str


def _build_symmetry(raw: Optional[dict]) -> Optional[_Symmetry]:
    if not isinstance(raw, dict):
        return None
    cs = raw.get("crystal_system")
    return _Symmetry(
        crystal_system=_EnumLike(cs) if cs is not None else _EnumLike(""),
        symbol=raw.get("symbol", ""),
    )


# Top-level doc wrappers per collection.  Mirror the fields SPICES touches.

@dataclass
class ThermoDoc:
    material_id: _MPID
    composition: Composition
    chemsys: str
    formula_pretty: str
    thermo_type: _EnumLike
    energy_type: str
    entries: dict  # xc_key -> _Entry
    energy_per_atom: float
    energy_above_hull: float
    symmetry: Optional[_Symmetry]
    is_stable: bool = False


@dataclass
class SummaryDoc:
    material_id: _MPID
    theoretical: bool
    # Extras that occasionally show up in user code; kept accessible.
    formula_pretty: str = ""
    chemsys: str = ""


@dataclass
class MaterialsDoc:
    material_id: _MPID
    structure: Any  # pymatgen Structure (lazy; see property below)
    formula_pretty: str = ""
    _raw_structure: dict = field(default=None, repr=False)

    def __post_init__(self):
        # Structure is parsed eagerly when the doc is built (client-side).
        pass


def _thermo_from_doc(d: dict) -> ThermoDoc:
    entries = {k: _Entry(v) for k, v in (d.get("entries") or {}).items()}
    return ThermoDoc(
        material_id=_MPID(d["material_id"]),
        composition=Composition(d.get("composition") or {}),
        chemsys=d.get("chemsys", ""),
        formula_pretty=d.get("formula_pretty", ""),
        thermo_type=_EnumLike(d.get("thermo_type", SUPPORTED_THERMO_TYPE)),
        energy_type=d.get("energy_type", ""),
        entries=entries,
        energy_per_atom=d.get("energy_per_atom", float("nan")),
        energy_above_hull=d.get("energy_above_hull", float("nan")),
        symmetry=_build_symmetry(d.get("symmetry")),
        is_stable=bool(d.get("is_stable", False)),
    )


def _summary_from_doc(d: dict) -> SummaryDoc:
    return SummaryDoc(
        material_id=_MPID(d["material_id"]),
        theoretical=bool(d.get("theoretical", True)),
        formula_pretty=d.get("formula_pretty", ""),
        chemsys=d.get("chemsys", ""),
    )


def _material_from_doc(d: dict) -> MaterialsDoc:
    struct_dict = d.get("structure")
    structure = Structure.from_dict(struct_dict) if struct_dict else None
    return MaterialsDoc(
        material_id=_MPID(d["material_id"]),
        structure=structure,
        formula_pretty=d.get("formula_pretty", ""),
    )


# ----------------------------------------------------------------------
# Sub-namespaces emulating mp_api.client.MPRester's attribute tree.
# ----------------------------------------------------------------------


class _StrictNamespace:
    """Base class that raises SnapshotBackendError on unknown attributes."""

    _allowed: frozenset = frozenset()

    def __getattr__(self, name):
        if name.startswith("_") or name in self._allowed:
            raise AttributeError(name)
        raise SnapshotBackendError(
            f"{type(self).__name__} does not implement '{name}'. "
            f"Supported attributes: {sorted(self._allowed)}. "
            f"Extending the snapshot backend requires an explicit addition — "
            f"silent fallback to live MP is disabled."
        )


class _ThermoNS(_StrictNamespace):
    _allowed = frozenset({"search"})

    def __init__(self, loader: SnapshotLoader):
        self._loader = loader

    def search(
        self,
        chemsys=None,
        material_ids=None,
        is_stable=None,
        thermo_types=None,
        all_fields=False,
        fields=None,
    ):
        # Strict thermo_type guard: 2025-06-09 snapshot only ships GGA_GGA+U.
        if thermo_types is not None:
            requested = set(thermo_types)
            if not requested.issubset({SUPPORTED_THERMO_TYPE}):
                unsupported = requested - {SUPPORTED_THERMO_TYPE}
                raise SnapshotBackendError(
                    f"Snapshot backend only ships thermo_type='{SUPPORTED_THERMO_TYPE}'. "
                    f"Requested unsupported types: {sorted(unsupported)}. "
                    f"Mixing with later-added run types (e.g. R2SCAN) would "
                    f"silently combine data from different DB eras."
                )

        raw_docs = self._loader.search(
            "thermo",
            chemsys=chemsys,
            material_ids=material_ids,
        )
        out = [_thermo_from_doc(d) for d in raw_docs if d is not None]

        if is_stable is not None:
            out = [e for e in out if e.is_stable == bool(is_stable)]

        return out


class _SummaryNS(_StrictNamespace):
    _allowed = frozenset({"search"})

    def __init__(self, loader: SnapshotLoader):
        self._loader = loader

    def search(
        self,
        material_ids=None,
        chemsys=None,
        all_fields=False,
        fields=None,
    ):
        raw_docs = self._loader.search(
            "summary",
            chemsys=chemsys,
            material_ids=material_ids,
        )
        return [_summary_from_doc(d) for d in raw_docs if d is not None]


class _MaterialsNS(_StrictNamespace):
    _allowed = frozenset({"search", "thermo", "summary",
                          "get_structure_by_material_id"})

    def __init__(self, loader: SnapshotLoader):
        self._loader = loader
        self.thermo = _ThermoNS(loader)
        self.summary = _SummaryNS(loader)

    def search(
        self,
        formula=None,
        chemsys=None,
        crystal_system=None,
        material_ids=None,
        all_fields=False,
        fields=None,
    ):
        raw_docs = self._loader.search(
            "materials",
            chemsys=chemsys,
            material_ids=material_ids,
            formula=formula,
            crystal_system=crystal_system,
        )
        return [_material_from_doc(d) for d in raw_docs if d is not None]

    def get_structure_by_material_id(self, material_id):
        doc = self._loader.get_doc("materials", material_id)
        if doc is None:
            return []  # mp_api returns [] for missing IDs; SPICES handles this
        struct_dict = doc.get("structure")
        if not struct_dict:
            return []
        return Structure.from_dict(struct_dict)


# ----------------------------------------------------------------------
# Top-level client
# ----------------------------------------------------------------------


class SnapshotMPRester(_StrictNamespace):
    """
    Offline replacement for ``mp_api.client.MPRester``.

    Only the methods SPICES calls are implemented.  Everything else
    raises :class:`SnapshotBackendError` — preventing accidental live-MP
    fallback that would mix data from two database versions.
    """

    _allowed = frozenset({
        "materials",
        "get_pourbaix_entries",
        "get_database_version",
        "release",
        "snapshot_root",
    })

    def __init__(self, snapshot_root: str, release: str = "2025-06-09"):
        if not os.path.isdir(os.path.join(snapshot_root, release)):
            raise FileNotFoundError(
                f"Snapshot directory not found: "
                f"{os.path.join(snapshot_root, release)}. "
                f"Download it first via "
                f"`python -m spices.mp_snapshot.download {snapshot_root} "
                f"--release {release}`."
            )

        self.snapshot_root = snapshot_root
        self.release = release
        self._loader = SnapshotLoader(snapshot_root, release=release)
        self.materials = _MaterialsNS(self._loader)

    def get_database_version(self) -> str:
        """Return the frozen release label, mirroring the live API."""
        return self.release

    def get_pourbaix_entries(self, chemsys, **kwargs):
        """
        Offline rebuild of Pourbaix entries for *chemsys*.

        Delegates to :mod:`spices.mp_snapshot.pourbaix`; see there for
        the reconstruction procedure.
        """
        # Import locally so the module is optional at import time — the
        # heavy pymatgen imports only run when Pourbaix is actually used.
        from spices.mp_snapshot.pourbaix import build_pourbaix_entries

        if kwargs:
            raise SnapshotBackendError(
                f"Snapshot get_pourbaix_entries ignores keyword args: "
                f"{sorted(kwargs)}. Add support explicitly if needed."
            )

        return build_pourbaix_entries(self._loader, chemsys, release=self.release)

    # Context-manager parity with MPRester so notebooks using
    # ``with MPRester(...) as mpr:`` still work.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

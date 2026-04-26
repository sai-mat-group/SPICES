"""
Offline reconstruction of ``MPRester.get_pourbaix_entries(chemsys)``.

The live implementation is a server-computed endpoint that combines
three inputs:

1. Solid ``ComputedStructureEntry`` objects for the extended chemsys
   (target chemsys ∪ ion-reference solids' elements ∪ {O, H}) pulled
   from the ``thermo`` collection with ``thermo_type="GGA_GGA+U"``.
2. Aqueous ion reference data from the MPContribs
   ``ion_ref_data`` project — experimental ΔGᶠ, charges, reference
   solids.
3. MP's Aqueous compatibility scheme
   (``MaterialsProjectAqueousCompatibility`` on top of
   ``MaterialsProject2020Compatibility``) which adjusts solid DFT
   energies to be consistent with experimental ion energies.

We reproduce the algorithm from
``mp_api.client.MPRester.get_pourbaix_entries`` verbatim, replacing the
two network-backed data sources with local reads:

- Solid thermo → :class:`SnapshotLoader` over the local bulk dump.
- Ion reference data → a JSON dump written by ``download.py``.

Reproducibility caveat — read carefully
---------------------------------------
MPContribs' ``ion_ref_data`` project is **not** versioned alongside the
``collections/<release>/`` bulk dumps.  The dump is a single live
document that gets amended occasionally.  What ``download.py`` writes
under ``ion_ref_data/mpcontribs_ion_ref_data.json`` is a snapshot of
that document at download time — *not* what existed on 2025-06-09.

In practice this is usually fine: ion thermodynamics are empirical and
rarely revised.  But peer reviewers should be told that the ion_ref_data
freeze is dated to the download, not to the DB release.  The manifest
and this module's log output record the freeze date so that provenance
is recoverable.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import warnings

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.pourbaix_diagram import PourbaixEntry
from pymatgen.core import Element
from pymatgen.core.ion import Ion
from pymatgen.entries.compatibility import (
    MaterialsProject2020Compatibility,
    MaterialsProjectAqueousCompatibility,
)
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from pymatgen.analysis.pourbaix_diagram import IonEntry

from spices.mp_snapshot.loader import SnapshotLoader

logger = logging.getLogger(__name__)


def _load_ion_ref_data(release_root: str) -> list[dict]:
    """Read the frozen MPContribs ion_ref_data dump."""
    path = os.path.join(release_root, "ion_ref_data", "mpcontribs_ion_ref_data.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Ion reference data dump not found at {path}. "
            f"Run `python -m spices.mp_snapshot.download <root>` with an "
            f"MP_API_KEY in the environment so the MPContribs ion_ref_data "
            f"project can be fetched."
        )
    with open(path) as f:
        data = json.load(f)
    logger.info(
        "Loaded %d ion reference records from %s — ion_ref_data freeze is "
        "dated to download time, not to the snapshot release.",
        len(data), path,
    )
    return data


def _ion_data_for_chemsys(all_ion_data: list[dict], chemsys: list[str]) -> list[dict]:
    """Mirror of ``MPRester.get_ion_reference_data_for_chemsys`` filter."""
    return [d for d in all_ion_data if d["data"]["MajElements"] in chemsys]


def _deserialize_thermo_entries(raw_docs: list[dict]) -> list[ComputedStructureEntry]:
    """
    Extract ComputedStructureEntry objects from raw thermo JSONL docs.

    Each thermo doc's ``entries`` is a dict keyed by run_type (e.g.
    ``"GGA"``, ``"GGA+U"``) mapping to a serialized CSE dict.  We pick
    the entry whose key equals the doc's ``energy_type`` — the same
    selection the live API makes.
    """
    out = []
    for d in raw_docs:
        if d is None:
            continue
        xc = d.get("energy_type")
        entries = d.get("entries") or {}
        cse_dict = entries.get(xc)
        if cse_dict is None:
            # Fall back to the first entry if energy_type mismatch
            if not entries:
                continue
            cse_dict = next(iter(entries.values()))
        out.append(ComputedStructureEntry.from_dict(cse_dict))
    return out


def _build_ion_entries(pd: PhaseDiagram, ion_data: list[dict]) -> list[IonEntry]:
    """Verbatim port of ``MPRester.get_ion_entries`` body (no network)."""
    ion_entries = []
    for i_d in ion_data:
        ion = Ion.from_formula(i_d["formula"])
        refs = [
            e for e in pd.all_entries
            if e.composition.reduced_formula == i_d["data"]["RefSolid"]
        ]
        if not refs:
            logger.warning(
                "Reference solid %s for ion %s not in phase diagram — "
                "skipping this ion.",
                i_d["data"]["RefSolid"], i_d["formula"],
            )
            continue
        stable_ref = sorted(refs, key=lambda x: x.energy_per_atom)[0]
        rf = stable_ref.composition.get_reduced_composition_and_factor()[1]

        ref_unit = i_d["data"]["ΔGᶠRefSolid"]["unit"]
        if ref_unit == "kJ/mol":
            ref_solid_energy = i_d["data"]["ΔGᶠRefSolid"]["value"] / 96.485
        elif ref_unit == "MJ/mol":
            ref_solid_energy = i_d["data"]["ΔGᶠRefSolid"]["value"] / 96485
        else:
            raise ValueError(f"Unknown ΔGᶠRefSolid unit: {ref_unit}")

        solid_diff = pd.get_form_energy(stable_ref) - ref_solid_energy * rf
        elt = i_d["data"]["MajElements"]
        correction_factor = ion.composition[elt] / stable_ref.composition[elt]

        ion_unit = i_d["data"]["ΔGᶠ"]["unit"]
        if ion_unit == "kJ/mol":
            ion_free_energy = i_d["data"]["ΔGᶠ"]["value"] / 96.485
        elif ion_unit == "MJ/mol":
            ion_free_energy = i_d["data"]["ΔGᶠ"]["value"] / 96485
        else:
            raise ValueError(f"Unknown ΔGᶠ unit: {ion_unit}")

        energy = ion_free_energy + solid_diff * correction_factor
        ion_entries.append(IonEntry(ion, energy))

    return ion_entries


def build_pourbaix_entries(
    loader: SnapshotLoader,
    chemsys,
    release: str = "2025-06-09",
) -> list[PourbaixEntry]:
    """
    Offline equivalent of ``MPRester.get_pourbaix_entries(chemsys)``.

    Parameters
    ----------
    loader : SnapshotLoader
        Reader over the local snapshot.
    chemsys : str or list of str
        Target chemsys, e.g. ``"Li-Mn-O"`` or ``["Li", "Mn", "O"]``.
    release : str
        Release label (used only for log/context).

    Returns
    -------
    list of PourbaixEntry
        Same shape and semantics as the live API's return value.
    """
    # Normalize chemsys input
    if isinstance(chemsys, str):
        elements = chemsys.split("-")
    else:
        elements = list(chemsys)
    chemsys_list = sorted(e.capitalize() for e in elements)

    # 1. Load ion reference data and restrict to the target chemsys.
    all_ion_data = _load_ion_ref_data(loader.release_root)
    ion_data = _ion_data_for_chemsys(all_ion_data, chemsys_list)

    # 2. Collect the union of elements we need entries for: target chemsys
    #    plus all ion reference solid elements plus {O, H}.
    ion_ref_comps = [
        Ion.from_formula(d["data"]["RefSolid"]).composition for d in ion_data
    ]
    ion_ref_elts = set(
        itertools.chain.from_iterable(i.elements for i in ion_ref_comps)
    )
    extended_elts = (
        {str(e) for e in ion_ref_elts}
        | set(chemsys_list)
        | {"O", "H"}
    )
    logger.info(
        "Building Pourbaix for %s: extended entry search over %s (release=%s)",
        chemsys_list, sorted(extended_elts), release,
    )

    # 3. Pull thermo entries whose chemsys ⊆ extended_elts.
    raw_thermo = loader.search_subchemsys("thermo", sorted(extended_elts))
    ion_ref_entries = _deserialize_thermo_entries(raw_thermo)

    if not ion_ref_entries:
        raise RuntimeError(
            f"No thermo entries found in snapshot for chemsys subset "
            f"{sorted(extended_elts)}. Either the snapshot is incomplete "
            f"or the chemsys is outside MP's coverage."
        )

    # 4. Apply MP's Aqueous compatibility scheme.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You did not provide the required O2 and H2O energies.",
        )
        compat = MaterialsProjectAqueousCompatibility(
            solid_compat=MaterialsProject2020Compatibility(),
        )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Failed to guess oxidation states.*"
        )
        ion_ref_entries = compat.process_entries(ion_ref_entries)

    # 5. Build a PhaseDiagram over the processed entries.
    ion_ref_pd = PhaseDiagram(ion_ref_entries)

    # 6. Derive IonEntry objects anchored to the PD.
    ion_entries = _build_ion_entries(ion_ref_pd, ion_data)
    pbx_entries = [PourbaixEntry(e, f"ion-{n}") for n, e in enumerate(ion_entries)]

    # 7. Wrap solid entries as PourbaixEntries, filtering out pure OH /
    #    extraneous-element entries — same logic as mp_api source.
    extra_elts = (
        set(ion_ref_elts)
        - {Element(s) for s in chemsys_list}
        - {Element("H"), Element("O")}
    )
    for entry in ion_ref_entries:
        entry_elts = set(entry.composition.elements)
        if entry_elts <= {Element("H"), Element("O")}:
            continue
        if extra_elts.intersection(entry_elts):
            continue
        form_e = ion_ref_pd.get_form_energy(entry)
        new_entry = ComputedEntry(
            entry.composition, form_e, entry_id=entry.entry_id,
        )
        pbx_entries.append(PourbaixEntry(new_entry))

    logger.info(
        "Built %d Pourbaix entries for %s (ions=%d, solids=%d)",
        len(pbx_entries), chemsys_list, len(ion_entries),
        len(pbx_entries) - len(ion_entries),
    )
    return pbx_entries

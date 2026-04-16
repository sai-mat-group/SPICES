"""
Thermodynamic stability checks using Pourbaix diagrams.
"""

from pymatgen.core import Composition
from pymatgen.analysis.pourbaix_diagram import (
    PourbaixEntry, MultiEntry,
)

from spices.config import H_LINE, O_LINE, ANIONIC_ELEMENTS
from spices.pourbaix.geometry import check_overlap
from spices.pourbaix.decomposition import get_mp_id_from_pbx_entry_id


def is_stable_in_water(formula_string, charge, pourbaix_diagram,
                       range_of_pH=(0, 14)):
    """
    Check if a species is thermodynamically stable in water over the
    given pH range.

    Iterates over the stable-domain vertices of the Pourbaix diagram
    and tests whether any polygon for the target phase overlaps with
    the water stability region.

    Parameters
    ----------
    formula_string : str
        Chemical formula of the species.
    charge : int
        Charge state of the species in the diagram.
    pourbaix_diagram : PourbaixDiagram
        Pre-constructed Pourbaix diagram.
    range_of_pH : tuple of float
        (min, max) pH range for the overlap check.

    Returns
    -------
    bool
        True if the species is stable in water.
    """
    formula = Composition(formula_string).reduced_formula

    for entry, vertices in pourbaix_diagram._stable_domain_vertices.items():
        if isinstance(entry, PourbaixEntry) and not isinstance(entry, MultiEntry):
            if (entry.composition.reduced_formula == formula
                    and entry.charge == int(charge)):
                polygon_coords = [tuple(v) for v in vertices]
                if check_overlap(polygon_coords, H_LINE, O_LINE,
                                 range_of_pH=range_of_pH, show_plot=False):
                    return True
        else:
            # MultiEntry: check each sub-entry
            formulas = [e.composition.reduced_formula for e in entry.entry_list]
            charges = [e.charge for e in entry.entry_list]
            if formula in formulas:
                idx = formulas.index(formula)
                if int(charges[idx]) == int(charge):
                    polygon_coords = [tuple(v) for v in vertices]
                    if check_overlap(polygon_coords, H_LINE, O_LINE,
                                     range_of_pH=range_of_pH, show_plot=False):
                        return True

    return False


def check_solid_entry(entry_object, reactant_entry=None, mp_id=None,
                      pbx_diag=None, working_ion=None,
                      selection_criterion=4):
    """
    Check whether a stable Pourbaix entry contains a solid phase.

    In the simple case (``mp_id=None``), returns True if any sub-entry
    is a solid.  When ``mp_id`` is provided, applies **Criterion 4** by
    default: at least one solid decomposition product must contain a
    non-working-ion cation *and* an anionic element.

    Parameters
    ----------
    entry_object : PourbaixEntry or MultiEntry
        The stable entry at a given (pH, V) point.
    reactant_entry : PourbaixEntry or MultiEntry or None
        Stable entry from the bare seawater diagram (needed when
        ``mp_id`` is not None).
    mp_id : str or None
        Materials Project ID of the cathode material.
    pbx_diag : PourbaixDiagram or None
        Full Pourbaix diagram (needed with ``mp_id``).
    working_ion : Element or None
        The intercalating ion (Li, Na, K, Mg, Ca).
    selection_criterion : int
        Criterion for solid product selection (default 4).  For details
        on all 8 criteria, see
        ``spices.pourbaix.selection_criteria.check_solid_entry_all_criteria``.

    Returns
    -------
    bool or dict
        Boolean for criteria 1-8, dict of booleans for criterion 9.
    """
    if mp_id is None:
        return _check_solid_entry_simple(entry_object)

    if reactant_entry is None:
        raise ValueError("reactant_entry is required when mp_id is provided")

    return _check_solid_entry_strict(
        entry_object, reactant_entry, mp_id, pbx_diag, working_ion,
        selection_criterion,
    )


def _check_solid_entry_simple(entry_object):
    """Return True if any sub-entry of *entry_object* is a solid."""
    if isinstance(entry_object, MultiEntry):
        for entry in entry_object.entry_list:
            if entry.phase_type.strip() == "Solid":
                return True
    elif isinstance(entry_object, PourbaixEntry):
        if entry_object.phase_type.strip() == "Solid":
            return True
    return False


def _check_solid_entry_strict(entry_object, reactant_entry, mp_id,
                              pbx_diag, working_ion,
                              selection_criterion=4):
    """Strict solid-entry check.

    Default fast path evaluates Criterion 4 only.  Other criteria are
    delegated to ``spices.pourbaix.selection_criteria``.
    """
    if selection_criterion != 4:
        from spices.pourbaix.selection_criteria import (
            check_solid_entry_all_criteria,
        )
        return check_solid_entry_all_criteria(
            entry_object, reactant_entry, mp_id, pbx_diag, working_ion,
            selection_criterion,
        )

    # --- Criterion 4 fast path ---
    # Find element sets from the cathode mp_id
    for entry in pbx_diag.unprocessed_entries:
        if get_mp_id_from_pbx_entry_id(entry.entry_id) == mp_id:
            cationic_elements = set(entry.elements) - ANIONIC_ELEMENTS
            non_working_ion_cations = (
                cationic_elements - {working_ion} if working_ion else cationic_elements
            )
            break
    else:
        return False

    # Get list of reactant mp_ids to exclude
    if isinstance(reactant_entry, MultiEntry):
        reactant_mpids = [
            get_mp_id_from_pbx_entry_id(e.entry_id) for e in reactant_entry.entry_list
        ]
    else:
        reactant_mpids = [get_mp_id_from_pbx_entry_id(reactant_entry.entry_id)]

    entries = (
        entry_object.entry_list if isinstance(entry_object, MultiEntry)
        else [entry_object]
    )

    for entry in entries:
        entry_mpid = get_mp_id_from_pbx_entry_id(entry.entry_id)
        if entry.phase_type.strip() != "Solid" or entry_mpid in reactant_mpids:
            continue

        elems = entry.elements
        if (any(e in elems for e in non_working_ion_cations)
                and any(e in elems for e in ANIONIC_ELEMENTS)):
            return True

    return False


def get_all_stable_compounds_from_pourbaix_diagram(pourbaix_diagram):
    """
    Return a list of (formula, charge) tuples for all stable entries.

    Parameters
    ----------
    pourbaix_diagram : PourbaixDiagram

    Returns
    -------
    list of tuple
        Each tuple is (reduced_formula: str, charge: float).
    """
    results = []
    for entry in pourbaix_diagram.stable_entries:
        if isinstance(entry, MultiEntry):
            for sub in entry.entry_list:
                results.append((sub.composition.reduced_formula, sub.charge))
        else:
            results.append((entry.composition.reduced_formula, entry.charge))
    return results


def get_unique_stable_compounds_from_pourbaix_diagram(pourbaix_diagram):
    """
    Return the unique set of stable compound strings.

    Parameters
    ----------
    pourbaix_diagram : PourbaixDiagram

    Returns
    -------
    set of str
        Strings of the form ``"Fe2O3 charge: 0.0"``.
    """
    compound_set = set()
    for entry in pourbaix_diagram.stable_entries:
        if isinstance(entry, MultiEntry):
            for sub in entry.entry_list:
                s = f"{sub.composition.reduced_formula} charge: {float(sub.charge)}"
                compound_set.add(s)
        else:
            s = f"{entry.composition.reduced_formula} charge: {float(entry.charge)}"
            compound_set.add(s)
    return compound_set


def get_all_unique_compounds_from_pourbaix_diagram(pourbaix_diagram):
    """
    Return the unique set of all (not just stable) compound strings.

    Parameters
    ----------
    pourbaix_diagram : PourbaixDiagram

    Returns
    -------
    set of str
    """
    compound_set = set()
    for entry in pourbaix_diagram.unprocessed_entries:
        s = f"{entry.composition.reduced_formula} charge: {float(entry.charge)}"
        compound_set.add(s)
    return compound_set


def get_all_stable_solids_mp_id_from_pourbaix_diagram(pourbaix_diagram):
    """
    Return MP IDs of all stable solid phases in the diagram.

    Parameters
    ----------
    pourbaix_diagram : PourbaixDiagram

    Returns
    -------
    set of str
        E.g. ``{"mp-19017", "mp-4820"}``.
    """
    mp_id_set = set()
    for entry in pourbaix_diagram.stable_entries:
        if isinstance(entry, MultiEntry):
            for sub in entry.entry_list:
                id_ = get_mp_id_from_pbx_entry_id(sub.entry_id)
                if "mp" in id_:
                    mp_id_set.add(id_)
        else:
            id_ = get_mp_id_from_pbx_entry_id(entry.entry_id)
            if "mp" in id_:
                mp_id_set.add(id_)
    return mp_id_set

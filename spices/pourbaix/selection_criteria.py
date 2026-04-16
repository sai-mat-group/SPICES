"""
Optional module: full multi-criteria solid-entry selection for Pourbaix analysis.

Most users only need **Criterion 4** (the default in
``spices.pourbaix.stability.check_solid_entry``).  This module provides
the complete set of 8 selection criteria for advanced users who want to
explore how different definitions of "cathode preservation" affect
screening results.

Usage::

    from spices.pourbaix.selection_criteria import check_solid_entry_all_criteria

    # Single criterion (returns bool)
    result = check_solid_entry_all_criteria(entry, reactant, mp_id, pbx, ion,
                                           selection_criterion=3)

    # All criteria at once (returns dict of 8 bools)
    result = check_solid_entry_all_criteria(entry, reactant, mp_id, pbx, ion,
                                           selection_criterion=9)
    # result == {"1": True, "2": False, ..., "8": True}

Background
----------
When a cathode material decomposes in seawater at a given (pH, voltage),
the Pourbaix diagram predicts which phases are thermodynamically
stable.  The question is: *do those decomposition products look like the
cathode is being preserved?*  Each criterion defines a different answer
to that question, ranging from very permissive (Criterion 1) to very
strict (Criterion 6).

Element categories (derived from the cathode material):

- **Cationic elements**: all elements in the cathode *except* anionic ones.
- **Anionic elements**: {H, O, S, Se, F, Cl, Br, I, P, N, C, Si}.
- **Non-working-ion cations**: cationic elements minus the intercalating
  ion (e.g. minus Na+ for a Na-ion battery cathode).
- **Non-working-ion elements (excl. H, O)**: all cathode elements minus
  the working ion and H/O.
"""

from copy import deepcopy

from pymatgen.analysis.pourbaix_diagram import MultiEntry

from spices.config import ANIONIC_ELEMENTS, ELEMENTS_HO
from spices.pourbaix.decomposition import get_mp_id_from_pbx_entry_id

# All 8 selection criteria indices
ALL_SELECTION_CRITERIA = [1, 2, 3, 4, 5, 6, 7, 8]

# Column names for all 8 criteria (matches the Stage 2 output format)
SEAWATER_COLUMNS_ALL = [
    "Seawater stability",
    "Seawater metastability (Criterion 1)",
    "Seawater metastability (Criterion 2)",
    "Seawater metastability (Criterion 3)",
    "Seawater metastability (Criterion 4)",
    "Seawater metastability (Criterion 5)",
    "Seawater metastability (Criterion 6)",
    "Seawater metastability (Criterion 7)",
    "Seawater metastability (Criterion 8)",
]


def check_solid_entry_all_criteria(entry_object, reactant_entry, mp_id,
                                   pbx_diag, working_ion,
                                   selection_criterion=4):
    """
    Check decomposition-product solids against all 8 selection criteria.

    This is the full multi-criteria version of
    ``spices.pourbaix.stability.check_solid_entry``.  It evaluates which
    of the 8 criteria are satisfied by the solid phases in the
    decomposition products.

    Parameters
    ----------
    entry_object : PourbaixEntry or MultiEntry
        The stable entry at a given (pH, V) point.
    reactant_entry : PourbaixEntry or MultiEntry
        Stable entry from the bare seawater diagram.
    mp_id : str
        Materials Project ID of the cathode material.
    pbx_diag : PourbaixDiagram
        Full Pourbaix diagram.
    working_ion : Element or None
        The intercalating ion (Li, Na, K, Mg, Ca).
    selection_criterion : int
        Which criterion to evaluate (1-8 for a single bool, 9 for a
        dict of all 8).

    Returns
    -------
    bool or dict
        ``bool`` for criteria 1-8; ``dict`` mapping ``"1"``-``"8"`` to
        bools for criterion 9.

    Notes
    -----
    The 8 selection criteria, grouped by logic:

    **Existential criteria** — at least one solid decomposition product
    must satisfy the condition:

    1. Contains a **cationic element** of the cathode.
       *Most permissive — any cation from the cathode survives as a
       solid, including the working ion itself.*

    2. Contains a cationic element **AND** an anionic element.
       *The surviving solid is a compound (e.g. an oxide), not a bare
       metal.*

    3. Contains a **non-working-ion cation**.
       *The structural cation survives — not just the intercalating ion.
       E.g. for NaMnO2 with Na+ working ion, Mn must appear in a solid.*

    4. Contains a non-working-ion cation **AND** an anionic element.
       *The structural cation survives as a compound.  This is the
       default criterion used throughout the SPICES pipeline.*

    5. **All** non-working-ion elements (excl. H, O) appear somewhere
       in the solid decomposition products (can be spread across
       multiple solids).
       *Much stricter — requires comprehensive element retention.*

    6. Same as 5, but each element must appear in a solid that **also**
       contains an anionic element.
       *Strictest criterion — all framework elements retained as
       compounds.*

    7. Contains a **non-working-ion element (excl. H, O)**.
       *Like 3 but broader — includes non-cationic framework elements
       (e.g. P in a phosphate cathode).*

    8. Contains a non-working-ion element (excl. H, O) **AND** an
       anionic element.
       *Like 4 but for any non-working-ion element.*

    Permissiveness ranking (most to least permissive):
    1 > 3 > 7 > 2 > 4 > 8 > 5 > 6
    """
    # Find element sets from the cathode mp_id
    for entry in pbx_diag.unprocessed_entries:
        if get_mp_id_from_pbx_entry_id(entry.entry_id) == mp_id:
            cationic_elements = set(entry.elements) - ANIONIC_ELEMENTS
            non_working_ion_cations = (
                cationic_elements - {working_ion} if working_ion else cationic_elements
            )
            non_working_ion_elements_without_OH = (
                set(entry.elements) - ({working_ion} if working_ion else set()) - ELEMENTS_HO
            )
            break
    else:
        return False

    SC = {str(i): False for i in range(1, 9)}

    # Get list of reactant mp_ids to exclude
    if isinstance(reactant_entry, MultiEntry):
        reactant_mpids = [
            get_mp_id_from_pbx_entry_id(e.entry_id) for e in reactant_entry.entry_list
        ]
    else:
        reactant_mpids = [get_mp_id_from_pbx_entry_id(reactant_entry.entry_id)]

    # Tracking sets for criteria 5 and 6: start with all non-working-ion
    # elements and discard them as they are found in solid products.
    not_found_c5 = deepcopy(non_working_ion_elements_without_OH)
    not_found_c6 = deepcopy(non_working_ion_elements_without_OH)

    entries = (
        entry_object.entry_list if isinstance(entry_object, MultiEntry)
        else [entry_object]
    )

    for entry in entries:
        entry_mpid = get_mp_id_from_pbx_entry_id(entry.entry_id)
        if entry.phase_type.strip() != "Solid" or entry_mpid in reactant_mpids:
            continue

        elems = entry.elements

        # --- Criterion 1 ---
        # Does this solid contain ANY cationic element from the cathode?
        # Example: For NaMnO2 cathode, would pass if solid contains Na or Mn.
        if not SC["1"] and any(e in elems for e in cationic_elements):
            SC["1"] = True

        # --- Criterion 2 ---
        # Does this solid contain a cationic element AND an anionic element?
        # Ensures the product is a compound, not a bare metal.
        if not SC["2"]:
            if (any(e in elems for e in cationic_elements)
                    and any(e in elems for e in ANIONIC_ELEMENTS)):
                SC["2"] = True

        # --- Criterion 3 ---
        # Does this solid contain a non-working-ion cation?
        # For NaMnO2 with Na+ working ion: requires Mn in a solid.
        if not SC["3"] and any(e in elems for e in non_working_ion_cations):
            SC["3"] = True

        # --- Criterion 4 ---
        # Does this solid contain a non-working-ion cation AND an anionic element?
        # The DEFAULT criterion: structural cation survives as a compound.
        if not SC["4"]:
            if (any(e in elems for e in non_working_ion_cations)
                    and any(e in elems for e in ANIONIC_ELEMENTS)):
                SC["4"] = True

        # --- Criterion 5 ---
        # Are ALL non-working-ion elements (excl. H, O) found across the
        # solid products? Elements can be spread across multiple solids.
        if not SC["5"]:
            for elem in non_working_ion_elements_without_OH:
                if elem in elems:
                    not_found_c5.discard(elem)
            if len(not_found_c5) == 0:
                SC["5"] = True

        # --- Criterion 6 ---
        # Same as 5, but each element must appear in a solid that also
        # contains an anionic element (i.e. retained as a compound).
        if not SC["6"]:
            for elem in non_working_ion_elements_without_OH:
                if elem in elems and any(e in elems for e in ANIONIC_ELEMENTS):
                    not_found_c6.discard(elem)
            if len(not_found_c6) == 0:
                SC["6"] = True

        # --- Criterion 7 ---
        # Does this solid contain a non-working-ion element (excl. H, O)?
        # Broader than criterion 3: includes non-cationic framework elements.
        if not SC["7"] and any(e in elems for e in non_working_ion_elements_without_OH):
            SC["7"] = True

        # --- Criterion 8 ---
        # Does this solid contain a non-working-ion element (excl. H, O)
        # AND an anionic element?
        if not SC["8"]:
            if (any(e in elems for e in non_working_ion_elements_without_OH)
                    and any(e in elems for e in ANIONIC_ELEMENTS)):
                SC["8"] = True

    if selection_criterion == 9:
        return SC
    return SC.get(str(selection_criterion), False)

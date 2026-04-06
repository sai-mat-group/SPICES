"""
Decomposition energy calculations from Pourbaix diagrams.
"""

import logging
from itertools import combinations

from pymatgen.core import Composition

logger = logging.getLogger(__name__)


def get_mp_id_from_pbx_entry_id(pbx_entry_id):
    """
    Extract the Materials Project ID from a Pourbaix entry ID.

    Pourbaix entry IDs have the format ``"mp-12345-GGA_GGA+U_R2SCAN"``;
    this function returns ``"mp-12345"``.

    Parameters
    ----------
    pbx_entry_id : str

    Returns
    -------
    str
    """
    parts = pbx_entry_id.split("-")
    return f"{parts[0]}-{parts[1]}"


def get_decomposition_energy_from_mp_id(mp_id, pH, voltage,
                                        pourbaix_diagram):
    """
    Get the minimum decomposition energy for a material at given
    (pH, voltage).

    Parameters
    ----------
    mp_id : str
        Materials Project ID (e.g. ``"mp-19017"``).
    pH : float
    voltage : float
    pourbaix_diagram : PourbaixDiagram

    Returns
    -------
    float
        Minimum decomposition energy (eV/atom) across all matching
        entries.
    """
    energies = []
    for entry in pourbaix_diagram.unprocessed_entries:
        if get_mp_id_from_pbx_entry_id(entry.entry_id) == mp_id:
            energies.append(
                pourbaix_diagram.get_decomposition_energy(entry, pH, voltage)
            )
    return min(energies)


def get_decomposition_energy_from_formula(formula_string, pH, voltage,
                                          pourbaix_diagram):
    """
    Get the minimum decomposition energy for a formula at given
    (pH, voltage).

    Parameters
    ----------
    formula_string : str
        Chemical formula.
    pH : float
    voltage : float
    pourbaix_diagram : PourbaixDiagram

    Returns
    -------
    float
        Minimum decomposition energy (eV/atom).
    """
    target = Composition(formula_string).reduced_formula
    energies = []
    for entry in pourbaix_diagram.unprocessed_entries:
        if entry.composition.reduced_formula == target:
            energies.append(
                pourbaix_diagram.get_decomposition_energy(entry, pH, voltage)
            )
    return min(energies)


def get_seawater_decomposition_energy_from_mp_id(
    mp_id, pH, voltage, sea_pourbaix_diagram,
    bare_seawater_pbx,
    decomp_energy_for_metastability=None,
):
    """
    Get the decomposition energy for a material in a seawater Pourbaix
    diagram, accounting for seawater ion reactants.

    Unlike the dilute-water version, this function constructs
    multi-entries by combining the cathode entry with subsets of bare
    seawater reactant phases, then normalizes the decomposition energy
    by cathode atom count.

    Parameters
    ----------
    mp_id : str
        Materials Project ID (e.g. ``"mp-19017"``).
    pH : float
    voltage : float
    sea_pourbaix_diagram : PourbaixDiagram
        Seawater Pourbaix diagram (cathode + seawater elements).
    bare_seawater_pbx : PourbaixDiagram
        Bare seawater Pourbaix diagram (seawater elements only).
    decomp_energy_for_metastability : float or None
        If provided and a computed energy exceeds this threshold,
        return immediately (optimization for grid searches).

    Returns
    -------
    float
        Maximum normalized decomposition energy (eV/atom) across
        all balanced reaction pathways.

    Raises
    ------
    ValueError
        If *mp_id* is not found in the diagram.
    """
    energies = []

    stable_multientry = sea_pourbaix_diagram.get_stable_entry(pH, voltage)
    logger.debug("stable_multientry: %s", stable_multientry.name)

    for entry in sea_pourbaix_diagram.unprocessed_entries:
        if get_mp_id_from_pbx_entry_id(entry.entry_id) != mp_id:
            continue

        # Get possible reactants from bare seawater diagram
        bare_stable = bare_seawater_pbx.get_stable_entry(pH, voltage)
        entries_for_rxn_balance = list(bare_stable.entry_list)
        logger.debug(
            "entries_for_rxn_balance: %s",
            [e.name for e in entries_for_rxn_balance],
        )

        # Try each combination of seawater reactant entries
        for num_comb in range(len(entries_for_rxn_balance)):
            for comb in combinations(entries_for_rxn_balance, num_comb + 1):
                entry_list_for_decom = [entry] + list(comb)

                # Returns None if the aqueous reaction can't be balanced
                multi_entry_for_decom = sea_pourbaix_diagram.process_multientry(
                    entry_list_for_decom, stable_multientry.composition,
                )

                try:
                    decomp_e = sea_pourbaix_diagram.get_decomposition_energy(
                        multi_entry_for_decom, pH, voltage,
                    )
                except Exception:
                    continue

                # Normalize by cathode atom count
                # try/except for pymatgen version compatibility
                try:
                    num_atoms_in_cathode = (
                        entry.composition.num_atoms
                        * multi_entry_for_decom.weights[0]
                    )
                    decomp_e_normalized = (
                        decomp_e
                        * multi_entry_for_decom.composition.num_atoms
                        / num_atoms_in_cathode
                    )
                except Exception:
                    entry_num_atoms = sum(
                        entry.composition.as_dict().values()
                    )
                    multi_num_atoms = sum(
                        multi_entry_for_decom.composition.as_dict().values()
                    )
                    num_atoms_in_cathode = (
                        entry_num_atoms * multi_entry_for_decom.weights[0]
                    )
                    decomp_e_normalized = (
                        decomp_e * multi_num_atoms / num_atoms_in_cathode
                    )

                energies.append(decomp_e_normalized)
                logger.debug(
                    "Decomposition energy for %s: %.4f eV/atom (raw: %.4f)",
                    multi_entry_for_decom.name, decomp_e_normalized, decomp_e,
                )

                # Early return if energy exceeds threshold
                if decomp_energy_for_metastability is not None:
                    if decomp_e_normalized > decomp_energy_for_metastability:
                        return decomp_e_normalized

        return max(energies)

    raise ValueError(
        f"{mp_id} not found in the sea_pourbaix_diagram object"
    )

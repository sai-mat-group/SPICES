"""
Chemical composition manipulation utilities for Pourbaix analysis.
"""

from pymatgen.core import Composition

from spices.config import SEAWATER_COMP_DICT, SEAWATER_CONC_DICT


def rm_element(formula_string, element_str):
    """
    Remove a specified element from a chemical formula.

    This is used to obtain the de-intercalated (charged) composition
    of a cathode material by removing the working ion.

    Parameters
    ----------
    formula_string : str
        Chemical formula (e.g. ``"LiFePO4"``).
    element_str : str
        Element symbol to remove (e.g. ``"Li"``).

    Returns
    -------
    str
        Formula string with the element removed (e.g. ``"FePO4"``).

    Examples
    --------
    >>> rm_element("LiFePO4", "Li")
    'Fe1 P1 O4'
    >>> rm_element("CaTiO3", "Ca")
    'Ti1 O3'
    """
    comp = Composition(formula_string)
    comp_dict = comp.as_dict()
    comp_dict.pop(element_str, None)
    if not comp_dict:
        return ""
    return Composition.from_dict(comp_dict).formula


def make_chemsys_from_combinations(tuple1, tuple2, element_str):
    """
    Build a chemical system string from transition-metal and anion tuples.

    Parameters
    ----------
    tuple1 : tuple of str
        Transition metal elements (e.g. ``("Ti", "Fe")``).
    tuple2 : tuple of str
        Anion elements (e.g. ``("O", "P")``).
    element_str : str
        Working ion element (e.g. ``"Ca"``).

    Returns
    -------
    str
        Hyphen-separated chemical system string
        (e.g. ``"Ca-Ti-Fe-O-P"``).

    Examples
    --------
    >>> make_chemsys_from_combinations(("Ti", "Fe"), ("O",), "Li")
    'Li-Ti-Fe-O'
    """
    parts = [element_str] + list(tuple1) + list(tuple2)
    return "-".join(parts)


def get_pourbaix_composition(composition_obj):
    """
    Extract the Pourbaix-relevant composition by removing O and H.

    Pourbaix diagrams treat O and H implicitly (as water species),
    so only the remaining elements define the composition dict.

    Parameters
    ----------
    composition_obj : pymatgen.core.Composition
        Full composition including O and/or H.

    Returns
    -------
    pymatgen.core.Composition
        Composition with O and H removed.

    Examples
    --------
    >>> from pymatgen.core import Composition
    >>> get_pourbaix_composition(Composition("LiFePO4"))
    Comp: Li1 Fe1 P1
    """
    comp_dict = composition_obj.as_dict()
    comp_dict.pop("O", None)
    comp_dict.pop("H", None)
    return Composition.from_dict(comp_dict)


def get_sea_pourbaix_comp_dict(composition_obj, num_of_sea_elements=4):
    """
    Build the composition dict for a seawater Pourbaix diagram.

    Merges the Pourbaix-relevant composition of the material with
    seawater ions (Cl, Na, Mg, S, Ca, K in order of abundance).

    Parameters
    ----------
    composition_obj : pymatgen.core.Composition
        Material composition (may be a full Composition or a
        Pourbaix-reduced one).
    num_of_sea_elements : int
        Number of seawater ions to include (default 4: Cl, Na, Mg, S).

    Returns
    -------
    dict
        Composition dict suitable for ``PourbaixDiagram(comp_dict=...)``.
    """
    pbx_comp = get_pourbaix_composition(composition_obj)
    pbx_comp_dict = pbx_comp.as_dict()

    # Increment stoichiometry for overlapping seawater elements
    for key in list(pbx_comp_dict.keys()):
        if key in SEAWATER_COMP_DICT:
            pbx_comp_dict[key] += SEAWATER_COMP_DICT[key]

    # Add remaining seawater elements up to the requested count
    for index, key in enumerate(SEAWATER_COMP_DICT):
        if index >= num_of_sea_elements:
            break
        if key not in pbx_comp_dict:
            pbx_comp_dict[key] = SEAWATER_COMP_DICT[key]

    return pbx_comp_dict


def get_sea_pourbaix_conc_dict(composition_obj, num_of_sea_elements=4,
                               dilute_limit_conc=1e-06):
    """
    Build the concentration dict for a seawater Pourbaix diagram.

    Non-seawater elements get a dilute-limit concentration while
    seawater ions use their actual seawater molarities.

    Parameters
    ----------
    composition_obj : pymatgen.core.Composition
        Material composition.
    num_of_sea_elements : int
        Number of seawater ions to include (default 4).
    dilute_limit_conc : float
        Concentration (M) for non-seawater elements (default 1e-6).

    Returns
    -------
    dict
        Concentration dict suitable for ``PourbaixDiagram(conc_dict=...)``.
    """
    pbx_comp = get_pourbaix_composition(composition_obj)
    pbx_comp_dict = pbx_comp.as_dict()

    conc_dict = {key: dilute_limit_conc for key in pbx_comp_dict}

    # Override with seawater concentrations for matching elements
    for key in conc_dict:
        if key in SEAWATER_CONC_DICT:
            conc_dict[key] = SEAWATER_CONC_DICT[key]

    # Add seawater elements not already present
    for index, key in enumerate(SEAWATER_CONC_DICT):
        if index >= num_of_sea_elements:
            break
        conc_dict[key] = SEAWATER_CONC_DICT[key]

    return conc_dict

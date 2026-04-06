"""
Intercalation voltage calculation from DFT energies.
"""

import logging
import math

from pymatgen.core import Composition

logger = logging.getLogger(__name__)


def get_energy_of_working_ion(working_ion, mp_client):
    """
    Query the Materials Project for the stable elemental energy of the
    working ion metal under different XC functionals.

    Parameters
    ----------
    working_ion : str
        Element symbol (e.g. ``"Ca"``).
    mp_client : MPRester
        Materials Project API client.

    Returns
    -------
    dict
        Mapping from functional label to energy_per_atom (eV).
        Keys may include ``"GGA"``, ``"R2SCAN"``, ``"SCAN"``.

    Raises
    ------
    ValueError
        If no stable entry is found for the working ion.
    """
    entries = mp_client.materials.thermo.search(
        chemsys=working_ion,
        is_stable=True,
        thermo_types=["GGA_GGA+U", "GGA_GGA+U_R2SCAN", "R2SCAN"],
    )

    if not entries:
        raise ValueError(
            f"No stable thermo entries found for {working_ion}"
        )

    energies = {}
    for entry in entries:
        key = entry.energy_type.upper()
        if key in ("GGA", "GGA+U"):
            energies["GGA"] = entry.energy_per_atom
        elif key == "R2SCAN":
            energies["R2SCAN"] = entry.energy_per_atom
        elif key == "SCAN":
            energies["SCAN"] = entry.energy_per_atom

        logger.info(
            "%s %s energy per atom = %.6f eV",
            working_ion, entry.energy_type, entry.energy_per_atom,
        )

    return energies


def calculate_intercalation_voltage(
    discharged_formula,
    discharged_energy_per_formula,
    charged_formula,
    charged_energy_per_formula,
    functional_string,
    working_ion,
    working_ion_valency,
    working_ion_energies,
):
    """
    Compute the intercalation voltage from DFT total energies.

    The voltage is calculated using the Nernst-based formula::

        V = -(E_dis - E_char - Δn_ion · E_ion) / (z · Δn_ion)

    where energies are normalized to the same number of framework atoms
    using the LCM of a shared non-working-ion element.

    Parameters
    ----------
    discharged_formula : str
        Formula of the discharged (intercalated) phase.
    discharged_energy_per_formula : float
        DFT energy per formula unit of the discharged phase (eV).
    charged_formula : str
        Formula of the charged (de-intercalated) phase.
    charged_energy_per_formula : float
        DFT energy per formula unit of the charged phase (eV).
    functional_string : str
        XC functional label (``"GGA"``, ``"GGA+U"``, ``"GGA_U"``,
        or ``"R2SCAN"``).
    working_ion : str
        Element symbol of the working ion (e.g. ``"Ca"``).
    working_ion_valency : int
        Charge state of the working ion.
    working_ion_energies : dict
        Mapping from functional label to energy_per_atom, as returned
        by :func:`get_energy_of_working_ion`.

    Returns
    -------
    float
        Intercalation voltage vs. the working ion metal (V).
    """
    discharged_comp = Composition(discharged_formula).as_dict()
    charged_comp = Composition(charged_formula).as_dict()

    # Find a non-working-ion element for normalization
    non_working_element = None
    for element in discharged_comp:
        if element != working_ion:
            non_working_element = element
            break

    # LCM normalization to common framework atom count
    n_dis = discharged_comp[non_working_element]
    n_char = charged_comp[non_working_element]
    lcm = math.lcm(int(n_dis), int(n_char))

    E_discharged = discharged_energy_per_formula * lcm / n_dis
    E_charged = charged_energy_per_formula * lcm / n_char

    # Delta working ion between normalized structures
    ion_in_dis = discharged_comp.get(working_ion, 0) * lcm / n_dis
    ion_in_char = charged_comp.get(working_ion, 0) * lcm / n_char
    delta_working_ion = ion_in_dis - ion_in_char

    # Select reference energy based on functional
    func_upper = functional_string.upper().replace("_", "+")
    if func_upper in ("GGA", "GGA+U"):
        working_ion_energy = working_ion_energies["GGA"]
    elif func_upper == "R2SCAN":
        working_ion_energy = working_ion_energies["R2SCAN"]
    else:
        raise TypeError(
            f"Incompatible functional '{functional_string}'. "
            "Use GGA, GGA+U, GGA_U, or R2SCAN."
        )

    voltage = -(
        E_discharged - E_charged - delta_working_ion * working_ion_energy
    ) / (working_ion_valency * delta_working_ion)

    return voltage

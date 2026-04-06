"""
Thermodynamic entry utilities.
"""


def find_correction(thermo_entry_object):
    """
    Extract energy-adjustment metadata from a thermo entry.

    Parameters
    ----------
    thermo_entry_object : ThermoDoc
        A thermo entry returned by the MP API
        (``mpi.materials.thermo.search(...)``).

    Returns
    -------
    list of dict
        Each dict has keys ``"name"`` and ``"description"``
        describing an energy correction applied to the entry.
    """
    xc = thermo_entry_object.energy_type
    adjustments = thermo_entry_object.entries[xc].energy_adjustments
    return [
        {"name": adj.as_dict()["name"], "description": adj.as_dict()["description"]}
        for adj in adjustments
    ]


def is_most_stable(row, df):
    """
    Check if a DataFrame row has the lowest energy above hull for its
    formula.

    Parameters
    ----------
    row : pd.Series
        A row from the screening DataFrame.
    df : pd.DataFrame
        The full screening DataFrame.

    Returns
    -------
    bool
    """
    min_energy = df.loc[
        df["Formula"] == row["Formula"], "Energy Above Hull (eV)"
    ].min()
    return row["Energy Above Hull (eV)"] == min_energy

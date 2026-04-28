"""
Pseudo-binary convex hull construction and on-hull phase filtering.
"""

import logging

from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PDEntry, CompoundPhaseDiagram

logger = logging.getLogger(__name__)


def create_pd_entry(formula_string, energy_per_formula, name=None):
    """
    Create a pymatgen ``PDEntry`` from a formula and energy.

    Parameters
    ----------
    formula_string : str
    energy_per_formula : float
        Total energy per formula unit (eV).
    name : str or None
        Label (typically the MP ID).

    Returns
    -------
    PDEntry
    """
    return PDEntry(
        composition=Composition(formula_string),
        energy=energy_per_formula,
        attribute=name,
    )


def get_working_ions_per_framework(formula, working_ion):
    """
    Compute the number of working ions per framework unit.

    Parameters
    ----------
    formula : str
        Chemical formula (e.g. ``"Ca2Ti3O7"``).
    working_ion : str
        Element symbol (e.g. ``"Ca"``).

    Returns
    -------
    float
        Working ion count per framework unit.
    """
    discharged_comp = Composition(formula)

    # Build charged composition (remove working ion)
    charged_dict = {
        el: amt for el, amt in discharged_comp.as_dict().items()
        if el != working_ion
    }
    charged_comp = Composition(charged_dict).reduced_composition

    working_ion_count = discharged_comp[working_ion]

    # Number of framework units in the discharged formula
    first_element = list(charged_comp.as_dict().keys())[0]
    no_of_frameworks = (
        discharged_comp[first_element] / charged_comp[first_element]
    )

    return working_ion_count / no_of_frameworks


def filter_df_to_get_on_hull_phases(df, working_ion):
    """
    Build a pseudo-binary convex hull and return only on-hull phases.

    Constructs a ``CompoundPhaseDiagram`` between the most-charged
    and most-discharged terminal compositions, then filters the
    DataFrame to keep only thermodynamically stable (on-hull) entries.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``'MP id'``, ``'Formula'``,
        ``'E (eV/formula)'``, ``'XC Functional'``.
    working_ion : str

    Returns
    -------
    pd.DataFrame
        Filtered and sorted by ``'No of working ions per framework'``
        (ascending). Returns empty DataFrame if fewer than 2 distinct
        compositions exist.
    """
    # Deduplicate by MP id (keep first).  Under SPICES' single-thermo_type
    # restriction we expect each MP id to appear at most once already; the
    # warn-on-shrink check surfaces any unexpected duplicates rather than
    # silently masking them.
    df_copy = df.drop_duplicates(subset="MP id", keep="first").copy()
    if len(df_copy) != len(df):
        logger.warning(
            "hull dedup dropped %d duplicate-MP-id row(s) for "
            "working_ion=%s — investigate why duplicates exist after "
            "the GGA(+U)-only thermo_type filter",
            len(df) - len(df_copy), working_ion,
        )

    if len(df_copy) == 0:
        return df_copy.reset_index(drop=True)

    # Add working ions per framework column
    df_copy["No of working ions per framework"] = df_copy["Formula"].apply(
        lambda x: get_working_ions_per_framework(x, working_ion)
    )

    if len(df_copy) < 2:
        return df_copy.reset_index(drop=True)

    # Build PDEntry list
    pd_entry_list = [
        create_pd_entry(row["Formula"], row["E (eV/formula)"], name=row["MP id"])
        for _, row in df_copy.iterrows()
    ]

    # Terminal compositions
    charged_terminal = Composition(
        df_copy.loc[
            df_copy["No of working ions per framework"].idxmin()
        ]["Formula"]
    )
    discharged_terminal = Composition(
        df_copy.loc[
            df_copy["No of working ions per framework"].idxmax()
        ]["Formula"]
    )

    # Build pseudo-binary convex hull
    convex_hull = CompoundPhaseDiagram(
        entries=pd_entry_list,
        terminal_compositions=[charged_terminal, discharged_terminal],
    )

    # Keep only on-hull phases
    on_hull_mpids = {e.attribute for e in convex_hull.stable_entries}
    df_copy = df_copy[df_copy["MP id"].isin(on_hull_mpids)]
    df_copy = df_copy.sort_values(
        by="No of working ions per framework",
    )

    return df_copy.reset_index(drop=True)

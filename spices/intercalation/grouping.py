"""Step 3 graph-based subgrouping.

Per unique-charged-composition sheet, enrich with fully-charged MP
phases then partition into connected components by structural match of
the charged framework.  Replaces the legacy
``create_subgroups`` / ``add_charged_struct_info`` / ``merge_subgroups``
chain.
"""

import logging
import os

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from spices.screening.framework_comparison import (
    FrameworkStructure,
    build_adjacency_dict,
    bfs_components,
    make_networkx_graph,
)

logger = logging.getLogger(__name__)


def save_component_graphs(graphs, save_dir):
    """Render each NetworkX graph with uniform edges and save to PDF.

    One file per component is written to
    ``{save_dir}/component_{idx}.pdf``.

    Parameters
    ----------
    graphs : list of networkx.Graph
        One graph per connected component (as produced by
        :func:`spices.screening.framework_comparison.make_networkx_graph`).
    save_dir : str
        Target directory for the PDFs.  Created if missing.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    from spices.screening.framework_comparison import _draw_nodes_and_labels

    os.makedirs(save_dir, exist_ok=True)

    for idx, G in enumerate(graphs):
        pos = nx.spring_layout(G)

        plt.figure(figsize=(10, 8))
        ax = _draw_nodes_and_labels(G, pos)

        nx.draw_networkx_edges(
            G, pos, edge_color="tab:gray", width=1.5,
            style="solid", alpha=0.8,
        )

        ax.spines[:].set_visible(False)
        plt.grid(False)

        out_path = os.path.join(save_dir, f"component_{idx}.pdf")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        logger.info("Saved component graph %d to %s", idx, out_path)


def discover_new_charged_rows_for_file(
    df,
    working_ion,
    mp_client,
    element_str,
    charged_struct_dir="charged_structs",
    struct_dir="structs",
    structure_cache_dir="All_charged_structures",
    cache_dict=None,
    matcher=None,
):
    """Discover new charged MP phases for one composition sheet.

    Phase 1 of the parallel step-2 pipeline: for each discharged row,
    look up every MP id whose charged framework structurally matches
    via :func:`spices.intercalation.structure.find_structures` and
    fetch its MP thermo data.  Returns the new rows with NaN
    placeholders in the seawater columns — Phase 2 fills them in by
    calling :func:`update_seawater_metastability_columns_by_mp_id` per
    row.

    Parameters
    ----------
    df : pd.DataFrame
        Composition sheet (from step 1).  Must contain ``'MP id'``.
    working_ion : str
    mp_client : MPRester
    element_str : str
    charged_struct_dir, struct_dir, structure_cache_dir : str
    cache_dict : dict or None
        Shared structure cache for
        :func:`read_or_generate_charged_struct`.
    matcher : StructureMatcher or None

    Returns
    -------
    pd.DataFrame
        New rows (one per newly discovered MP id) with seawater columns
        as NaN.  Empty DataFrame if no new mp ids matched.
    """
    from spices.intercalation.structure import (
        read_or_generate_charged_struct,
        find_structures,
    )
    from spices.screening.mp_query import collect_mp_data_from_chemsys

    if cache_dict is None:
        cache_dict = {}

    existing_mpids = df["MP id"].tolist()
    new_mpids = []

    for discharged_mpid in df["MP id"]:
        charged_struct = read_or_generate_charged_struct(
            discharged_mpid, working_ion, mp_client,
            charged_struct_dir=charged_struct_dir,
            struct_dir=struct_dir,
            cache_dict=cache_dict,
        )
        if isinstance(charged_struct, list) and len(charged_struct) == 0:
            continue

        matched = find_structures(
            charged_struct, mp_client,
            structure_cache_dir=structure_cache_dir,
            matcher=matcher,
        )
        for mp_id in matched:
            if mp_id not in existing_mpids and mp_id not in new_mpids:
                new_mpids.append(mp_id)

    if not new_mpids:
        logger.info("No new charged structures to add")
        return pd.DataFrame()

    logger.info("Discovered %d new charged structures", len(new_mpids))

    data_dict_list = collect_mp_data_from_chemsys(
        chemsys=None,
        element_str=element_str,
        mp_client=mp_client,
        material_ids=new_mpids,
    )
    for d in data_dict_list:
        d["Structure stable in water"] = np.nan
        d["Structure metastable in water"] = np.nan

    return pd.DataFrame(data_dict_list)


def add_seawater_metastability_bool_columns(df):
    """Add ``{col}_bool`` columns alongside every Seawater metastability
    criterion column.

    For each column matching ``Seawater metastability (Criterion N)``,
    creates a boolean companion ``Seawater metastability (Criterion
    N)_bool`` whose value is True iff the cell is a non-empty
    list/tuple.  Modifies *df* in place and also returns it.
    """
    crit_cols = [
        c for c in df.columns
        if c.startswith("Seawater metastability (Criterion ")
        and not c.endswith("_bool")
    ]
    for col in crit_cols:
        df[f"{col}_bool"] = df[col].apply(
            lambda v: bool(isinstance(v, (list, tuple)) and len(v) > 0)
        )
    return df


def enrich_sheet_with_charged_mp_phases(
    df,
    working_ion,
    mp_client,
    cache,
    element_str,
    charged_struct_dir="charged_structs",
    struct_dir="structs",
    structure_cache_dir="All_charged_structures",
    cache_dict=None,
    matcher=None,
    decomp_energy_for_metastability=0.3,
):
    """Append fully-charged MP phases (with seawater metastability) to one sheet.

    Sequential composition of :func:`discover_new_charged_rows_for_file`
    and a per-row call to
    :func:`update_seawater_metastability_columns_by_mp_id`, followed by
    :func:`add_seawater_metastability_bool_columns`.  Used by the
    sequential (``n_workers <= 1``) path of
    :func:`spices.screening.intercalation.step2_enrich_charged_composition_sheets`.

    Parameters
    ----------
    df : pd.DataFrame
        Composition sheet (from step 1).  Must contain ``'MP id'``.
    working_ion : str
    mp_client : MPRester
    cache : PourbaixCache
    element_str : str
    charged_struct_dir, struct_dir, structure_cache_dir : str
    cache_dict : dict or None
        Shared structure cache for
        :func:`read_or_generate_charged_struct`.
    matcher : StructureMatcher or None
    decomp_energy_for_metastability : float

    Returns
    -------
    pd.DataFrame
        Enriched frame with new rows appended (``ignore_index=True``).
    """
    from spices.dataframe_ops import (
        update_seawater_metastability_columns_by_mp_id,
    )

    new_rows_df = discover_new_charged_rows_for_file(
        df,
        working_ion=working_ion,
        mp_client=mp_client,
        element_str=element_str,
        charged_struct_dir=charged_struct_dir,
        struct_dir=struct_dir,
        structure_cache_dir=structure_cache_dir,
        cache_dict=cache_dict,
        matcher=matcher,
    )
    if len(new_rows_df) == 0:
        return df

    enriched_rows = []
    for _, row in new_rows_df.iterrows():
        enriched_rows.append(
            update_seawater_metastability_columns_by_mp_id(
                row, cache=cache, mp_client=mp_client,
                element_str=element_str,
                metastability_points=True,
                decomp_energy_for_metastability=decomp_energy_for_metastability,
            )
        )
    enriched_df = pd.DataFrame(enriched_rows)
    add_seawater_metastability_bool_columns(enriched_df)

    return pd.concat([df, enriched_df], ignore_index=True)


def _load_row_struct(mp_id, formula, struct_dir, structure_cache_dir):
    """Locate and load the raw CIF for a row.

    Discharged rows (from step 1) have their struct cached by
    :func:`read_or_generate_charged_struct` at
    ``{struct_dir}/{mp_id}.cif``.  Fully-charged rows added by
    :func:`enrich_sheet_with_charged_mp_phases` have theirs at
    ``{structure_cache_dir}/{formula}/{mp_id}.cif``.
    """
    primary = os.path.join(struct_dir, f"{mp_id}.cif")
    if os.path.exists(primary):
        return Structure.from_file(primary)

    fallback = os.path.join(structure_cache_dir, formula, f"{mp_id}.cif")
    if os.path.exists(fallback):
        return Structure.from_file(fallback)

    raise FileNotFoundError(
        f"Struct for {mp_id} not found in {primary} or {fallback}",
    )


def build_graph_components_from_sheet(
    df_enriched,
    working_ion,
    charged_struct_dir="charged_structs",
    struct_dir="structs",
    structure_cache_dir="All_charged_structures",
    matcher=None,
):
    """Partition an enriched sheet into connected components.

    Each row becomes a
    :class:`spices.screening.framework_comparison.FrameworkStructure`
    node.  Edges are pairwise matcher.fit matches of the charged
    framework.  Connected components become the step-3 "subgroups".

    Parameters
    ----------
    df_enriched : pd.DataFrame
        Output of :func:`enrich_sheet_with_charged_mp_phases`.  Must
        contain ``'MP id'``, ``'Formula'`` and ``'Is experimental'``.
    working_ion : str
    charged_struct_dir, struct_dir, structure_cache_dir : str
    matcher : StructureMatcher or None

    Returns
    -------
    tuple
        ``(component_dfs, component_graphs)`` — parallel lists.  Each
        ``component_dfs[i]`` is a row-subsetted copy of *df_enriched*
        for the mp_ids in component *i*; each ``component_graphs[i]``
        is a ``networkx.Graph`` suitable for
        :func:`save_component_graphs`.
    """
    if "Is experimental" not in df_enriched.columns:
        raise KeyError(
            "'Is experimental' column missing from enriched sheet",
        )
    if "Formula" not in df_enriched.columns:
        raise KeyError("'Formula' column missing from enriched sheet")

    nan_mask = df_enriched["MP id"].isna() | df_enriched["Formula"].isna()
    if nan_mask.any():
        logger.warning(
            "Skipping %d row(s) with NaN MP id / Formula", int(nan_mask.sum()),
        )
        df_enriched = df_enriched.loc[~nan_mask].reset_index(drop=True)

    all_frameworks = []
    row_mp_ids = []
    for _, row in df_enriched.iterrows():
        mp_id = row["MP id"]
        formula = row["Formula"]
        is_expt = bool(row["Is experimental"])

        struct = _load_row_struct(
            mp_id, formula, struct_dir, structure_cache_dir,
        )
        fw = FrameworkStructure(
            mp_id=mp_id, struct=struct,
            working_ion=working_ion, is_expt=is_expt,
        )
        all_frameworks.append(fw)
        row_mp_ids.append(mp_id)

    adjacency = build_adjacency_dict(all_frameworks, matcher=matcher)
    component_indices = bfs_components(adjacency, len(all_frameworks))

    component_dfs = []
    component_graphs = []
    for indices in component_indices:
        mp_ids = [row_mp_ids[i] for i in indices]
        comp_df = df_enriched[
            df_enriched["MP id"].isin(mp_ids)
        ].reset_index(drop=True)
        component_dfs.append(comp_df)

        G = make_networkx_graph(indices, all_frameworks, adjacency)
        component_graphs.append(G)

    logger.info(
        "Partitioned %d rows into %d components",
        len(all_frameworks), len(component_indices),
    )
    return component_dfs, component_graphs

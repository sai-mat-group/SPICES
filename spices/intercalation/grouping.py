"""Step 3 graph-based subgrouping.

Per unique-charged-composition sheet, enrich with fully-charged MP
phases then partition into connected components by structural match of
the charged framework.  Replaces the legacy
``create_subgroups`` / ``add_charged_struct_info`` / ``merge_subgroups``
chain.
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from spices.intercalation.structure import DEFAULT_MATCHER
from spices.pourbaix.cache import _atomic_pickle_dump
from spices.screening.framework_comparison import (
    FrameworkStructure,
    build_adjacency_dict,
    bfs_components,
    make_networkx_graph,
)

logger = logging.getLogger(__name__)

ADJACENCY_CACHE_SCHEMA_VERSION = 1


def _matcher_signature(matcher):
    """Extract canonical kwargs from a ``StructureMatcher`` for cache keying.

    Pulls the constructor arguments that affect ``fit`` behavior.  The cache
    auto-invalidates when any of these change in
    :data:`spices.intercalation.structure.DEFAULT_MATCHER`.

    Attribute access tolerates both the public name (modern pymatgen) and
    the underscore-prefixed private name (older versions).
    """
    def _get(attr_public, attr_private):
        return getattr(
            matcher,
            attr_public,
            getattr(matcher, attr_private, None),
        )

    ignored = _get("ignored_species", "_ignored_species") or ()
    return {
        "ltol": getattr(matcher, "ltol", None),
        "stol": getattr(matcher, "stol", None),
        "angle_tol": getattr(matcher, "angle_tol", None),
        "primitive_cell": _get("primitive_cell", "_primitive_cell"),
        "attempt_supercell": _get("attempt_supercell", "_attempt_supercell"),
        "allow_subset": _get("allow_subset", "_allow_subset"),
        "supercell_size": _get("supercell_size", "_supercell_size"),
        "ignored_species": tuple(ignored),
        "comparator": type(_get("comparator", "_comparator")).__name__,
    }


def _load_adjacency_cache(path, row_mp_ids, matcher):
    """Load and validate the adjacency cache, remapping to row order.

    Returns the row-indexed adjacency ``dict[int, list[int]]`` on a valid
    cache hit, or ``None`` if the cache is missing, unreadable, or stale
    (schema/mp_ids/matcher mismatch).  Logs a warning on every mismatch
    branch so the user sees *why* a rebuild happened.
    """
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except (EOFError, pickle.UnpicklingError, OSError, AttributeError) as e:
        logger.warning(
            "Adjacency cache %s unreadable (%s); rebuilding", path, e,
        )
        return None

    if not isinstance(payload, dict):
        logger.warning(
            "Adjacency cache %s has unexpected payload type %s; rebuilding",
            path, type(payload).__name__,
        )
        return None

    if payload.get("schema_version") != ADJACENCY_CACHE_SCHEMA_VERSION:
        logger.warning(
            "Adjacency cache %s schema_version=%r != %d; rebuilding",
            path, payload.get("schema_version"),
            ADJACENCY_CACHE_SCHEMA_VERSION,
        )
        return None

    cached_mp_ids = payload.get("mp_ids")
    current_mp_ids_sorted = tuple(sorted(row_mp_ids))
    if cached_mp_ids != current_mp_ids_sorted:
        logger.warning(
            "Adjacency cache %s mp_ids mismatch "
            "(cache=%d, current=%d); rebuilding",
            path,
            len(cached_mp_ids) if cached_mp_ids else 0,
            len(current_mp_ids_sorted),
        )
        return None

    current_sig = _matcher_signature(matcher)
    if payload.get("matcher_kwargs") != current_sig:
        logger.warning(
            "Adjacency cache %s matcher_kwargs mismatch; rebuilding", path,
        )
        return None

    # Remap sorted-mp_id index → row-order index.
    sorted_idx_of_mp_id = {mp: i for i, mp in enumerate(current_mp_ids_sorted)}
    sorted_of_row = [sorted_idx_of_mp_id[mp] for mp in row_mp_ids]
    row_of_sorted = [0] * len(row_mp_ids)
    for row_i, sorted_i in enumerate(sorted_of_row):
        row_of_sorted[sorted_i] = row_i

    cached_adjacency = payload.get("adjacency") or {}
    adjacency = {i: [] for i in range(len(row_mp_ids))}
    for sorted_i, neighbors in cached_adjacency.items():
        row_i = row_of_sorted[sorted_i]
        adjacency[row_i] = [row_of_sorted[s] for s in neighbors]

    logger.info("Loaded cached adjacency dict from %s", path)
    return adjacency


def _save_adjacency_cache(path, row_adjacency, row_mp_ids, matcher):
    """Pickle the row-indexed adjacency dict to disk in sorted-index form."""
    sorted_mp_ids = sorted(row_mp_ids)
    sorted_idx_of_mp_id = {mp: i for i, mp in enumerate(sorted_mp_ids)}
    sorted_of_row = [sorted_idx_of_mp_id[mp] for mp in row_mp_ids]

    sorted_adjacency = {}
    for row_i, neighbors in row_adjacency.items():
        sorted_i = sorted_of_row[row_i]
        sorted_adjacency[sorted_i] = sorted(
            sorted_of_row[j] for j in neighbors
        )
    for i in range(len(row_mp_ids)):
        sorted_adjacency.setdefault(i, [])

    payload = {
        "schema_version": ADJACENCY_CACHE_SCHEMA_VERSION,
        "mp_ids": tuple(sorted_mp_ids),
        "matcher_kwargs": _matcher_signature(matcher),
        "adjacency": sorted_adjacency,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _atomic_pickle_dump(payload, path)
    logger.info("Saved adjacency cache to %s", path)


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
    adjacency_cache_path=None,
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
    adjacency_cache_path : str or None
        If provided, attempt to load the adjacency dict from this
        pickle path; on cache miss / corrupt pickle / stale schema
        rebuild via :func:`build_adjacency_dict` and atomically write
        the result back.  The cache key encodes the sorted tuple of
        row MP ids and the active matcher signature, so the cache
        invalidates automatically when either changes.

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

    # Duplicate MP ids in a sheet break the sorted-index remap; skip cache
    # for that edge case but still build adjacency normally.
    cache_usable = adjacency_cache_path is not None and len(
        set(row_mp_ids),
    ) == len(row_mp_ids)
    if adjacency_cache_path is not None and not cache_usable:
        logger.warning(
            "Skipping adjacency cache for %s: duplicate MP ids in sheet",
            adjacency_cache_path,
        )

    matcher_for_signature = matcher if matcher is not None else DEFAULT_MATCHER

    adjacency = None
    if cache_usable and os.path.exists(adjacency_cache_path):
        adjacency = _load_adjacency_cache(
            adjacency_cache_path, row_mp_ids, matcher_for_signature,
        )
    if adjacency is None:
        adjacency = build_adjacency_dict(all_frameworks, matcher=matcher)
        if cache_usable:
            _save_adjacency_cache(
                adjacency_cache_path,
                adjacency,
                row_mp_ids,
                matcher_for_signature,
            )

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

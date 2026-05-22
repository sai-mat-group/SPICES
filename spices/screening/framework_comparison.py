"""
Stage 4: Cross-element framework comparison.

Compares cathode frameworks across all 5 working ions (Ca, K, Li, Mg, Na)
to identify multi-ion platforms vs. ion-selective frameworks.

The primary analysis uses graph-based comparison (NetworkX), with optional
set-based comparison (pairwise intersections, Venn diagrams, voltage overlays).

Replaces per-element ``framework_comparison_utils.py`` and
``framework_graph_comparison_utils.py`` with a single parameterized module.
"""

import ast
import json
import logging
import os
import pickle
import re
from collections import defaultdict, deque

import pandas as pd
from pymatgen.core import Structure, Element

from spices.intercalation.structure import DEFAULT_MATCHER

logger = logging.getLogger(__name__)

# Ion-specific plot colors
ION_COLORS = {
    "Li": "#E24A33",
    "Na": "#FBC15E",
    "K": "#988ED5",
    "Mg": "#348ABD",
    "Ca": "#8EBA42",
}


# ====================================================================
# FrameworkStructure class
# ====================================================================


class FrameworkStructure:
    """
    Structure wrapper that identifies a framework by removing the working
    ion.  Hashable by ``mp_id``; equality falls back to Python identity.

    Parameters
    ----------
    mp_id : str
    struct : Structure
    working_ion : str or Element
    is_expt : bool
        Whether the structure is experimental.  Required.
    """

    def __init__(self, mp_id, struct, working_ion, is_expt):
        self.mp_id = mp_id
        self.struct = struct.copy()
        self.struct_reduced_formula = struct.composition.reduced_formula
        self.is_expt = is_expt

        if isinstance(working_ion, str):
            working_ion = Element(working_ion)
        self.working_ion = working_ion

        # Build framework by removing working ion
        framework = struct.copy()
        if self.working_ion in framework.elements:
            framework.remove_species([self.working_ion])
        framework.remove_oxidation_states()

        self.framework_struct = framework
        self.framework_reduced_formula = framework.composition.reduced_formula

    def __hash__(self):
        return hash(self.mp_id)

    def __repr__(self):
        return (
            f"MPID: {self.mp_id}, "
            f"Formula: {self.struct_reduced_formula}, "
            f"Framework: {self.framework_reduced_formula}, "
            f"Expt: {self.is_expt}"
        )


# ====================================================================
# Framework loading
# ====================================================================


def collect_framework(mp_id, struct_dir, working_ion, is_expt):
    """Load a CIF file as a ``FrameworkStructure``.

    Only the exact ``{mp_id}.cif`` filename is accepted.  Substring
    matching is rejected because mp-ids can share prefixes (e.g.
    ``mp-123`` vs ``mp-1234``).
    """
    cif_path = os.path.join(struct_dir, f"{mp_id}.cif")
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"{mp_id}.cif not found in {struct_dir}")

    return FrameworkStructure(
        mp_id=mp_id,
        struct=Structure.from_file(cif_path),
        working_ion=working_ion,
        is_expt=is_expt,
    )


def load_charged_frameworks(
    excel_path, sheet_name, charged_struct_dir,
    discharged_struct_dir, working_ion,
):
    """
    Build a set of ``FrameworkStructure`` objects from charged-phase
    MP IDs in a voltage plateau Excel file.

    The ``{mp_id}.cif`` is first looked up in ``discharged_struct_dir``
    and, if absent, in ``{charged_struct_dir}/{Charged Formula}/``.
    Missing structures raise ``FileNotFoundError`` rather than being
    silently skipped.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if "Charged Is experimental" not in df.columns:
        raise KeyError(
            f"'Charged Is experimental' column missing in {excel_path}",
        )

    charged_mpids = df["Charged MP id"].unique()

    frameworks = set()
    for mpid in charged_mpids:
        vals = df[df["Charged MP id"] == mpid][
            "Charged Is experimental"
        ].unique()
        if len(vals) != 1:
            raise ValueError(
                f"{mpid} has {len(vals)} distinct is_expt values",
            )
        is_expt = bool(vals[0])

        discharged_cif = os.path.join(
            discharged_struct_dir, f"{mpid}.cif",
        )
        if os.path.exists(discharged_cif):
            fw = collect_framework(
                mpid, discharged_struct_dir, working_ion, is_expt,
            )
            frameworks.add(fw)
            continue

        formula = df[df["Charged MP id"] == mpid].iloc[0]["Charged Formula"]
        subdir = os.path.join(charged_struct_dir, formula)
        fw = collect_framework(mpid, subdir, working_ion, is_expt)
        frameworks.add(fw)

    logger.info(
        "Loaded %d distinct charged frameworks from %d MP IDs",
        len(frameworks), len(charged_mpids),
    )
    return frameworks


def load_discharged_frameworks(
    excel_path, sheet_name, discharged_struct_dir, working_ion,
):
    """
    Build a set of ``FrameworkStructure`` objects from discharged-phase
    MP IDs in a voltage plateau Excel file.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    discharged_mpids = df["Discharged MP id"].unique()

    frameworks = set()
    for mpid in discharged_mpids:
        is_expt = False
        if "Discharged Is experimental" in df.columns:
            vals = df[df["Discharged MP id"] == mpid]["Discharged Is experimental"].unique()
            if len(vals) == 1:
                is_expt = bool(vals[0])

        fw = collect_framework(
            mpid, discharged_struct_dir, working_ion, is_expt,
        )
        frameworks.add(fw)

    logger.info(
        "Loaded %d distinct discharged frameworks from %d MP IDs",
        len(frameworks), len(discharged_mpids),
    )
    return frameworks


def load_all_frameworks_for_element(
    excel_path, sheet_name, discharged_struct_dir,
    charged_struct_dir, working_ion,
):
    """
    Load both framework sets and the voltage DataFrame for one element.

    Returns
    -------
    tuple
        ``(discharged_set, charged_set, voltage_df)``
    """
    dis_set = load_discharged_frameworks(
        excel_path, sheet_name, discharged_struct_dir, working_ion,
    )
    chg_set = load_charged_frameworks(
        excel_path, sheet_name, charged_struct_dir,
        discharged_struct_dir, working_ion,
    )
    voltage_df = pd.read_excel(excel_path, sheet_name=sheet_name)
    return dis_set, chg_set, voltage_df


# ====================================================================
# Graph-based comparison
# ====================================================================


def _get_all_frameworks_list(discharged_list, charged_list):
    """Combine framework lists, deduplicating by mp_id."""
    all_fw = []
    seen_mpids = set()
    for fw in list(discharged_list) + list(charged_list):
        if fw.mp_id not in seen_mpids:
            seen_mpids.add(fw.mp_id)
            all_fw.append(fw)
    return all_fw


def build_adjacency_dict(all_frameworks, matcher=None):
    """
    Build adjacency dict by pairwise structural comparison.

    Uses ``DEFAULT_MATCHER`` (``attempt_supercell=True``) by default.

    Parameters
    ----------
    all_frameworks : list of FrameworkStructure
    matcher : StructureMatcher or None

    Returns
    -------
    dict
        Maps each framework index to list of neighbor indices.
    """
    if matcher is None:
        matcher = DEFAULT_MATCHER

    n = len(all_frameworks)
    adjacency = defaultdict(list)

    logger.info("Building adjacency dict for %d frameworks (O(n^2))", n)
    # TODO: Within-composition pair-splitting (deferred follow-up).  For
    # very large compositions (e.g. V3P8O29 ~404 frameworks, ~81k pairs
    # → 3.5 h on a single core), the upper-triangle pair index space
    # could be sharded across workers and reduced into a single
    # adjacency dict.  See the "Step 3 parallelization" section in
    # claude-prompts/multi-worker-OOM-fix.txt for context.
    for i in range(n):
        for j in range(i + 1, n):
            if matcher.fit(
                all_frameworks[i].framework_struct,
                all_frameworks[j].framework_struct,
                symmetric=True,
            ):
                adjacency[i].append(j)
                adjacency[j].append(i)

    # Ensure all nodes exist in adjacency
    for i in range(n):
        if i not in adjacency:
            adjacency[i] = []

    return dict(adjacency)


def reconnect_isolated_frameworks(adjacency, all_frameworks, df_dict):
    """
    Manually reconnect isolated frameworks using voltage plateau data.

    Some charged↔discharged framework pairs that share a known voltage
    plateau in the Excel may not be matched by ``StructureMatcher.fit``
    (e.g. when ion removal causes lattice distortion). This helper looks
    up each isolated framework's same-ion partner in
    ``df_dict[working_ion]`` and adds a manual edge.

    Encodes the scientific assumption: "if two structures share a row in
    the voltage plateau Excel, they are the same framework." Use only
    when that assumption is intended.

    Parameters
    ----------
    adjacency : dict
        From ``build_adjacency_dict``. Mutated in place AND returned.
    all_frameworks : list of FrameworkStructure
    df_dict : dict
        Maps working ion symbol to its voltage plateau DataFrame.
    """
    mpid_to_idx = {fw.mp_id: i for i, fw in enumerate(all_frameworks)}
    isolated_idxs = [i for i, nbrs in adjacency.items() if not nbrs]
    edges_added = 0

    for i in isolated_idxs:
        if adjacency[i]:
            continue
        fw = all_frameworks[i]
        ion = fw.working_ion.symbol
        if ion not in df_dict:
            logger.warning(
                "Isolated %s (%s): no df_dict entry for ion %s",
                fw.mp_id, fw.framework_reduced_formula, ion,
            )
            continue

        df = df_dict[ion]
        partners = []
        if fw.mp_id in df["Charged MP id"].values:
            partners = list(
                df[df["Charged MP id"] == fw.mp_id]["Discharged MP id"].unique()
            )
        elif fw.mp_id in df["Discharged MP id"].values:
            partners = list(
                df[df["Discharged MP id"] == fw.mp_id]["Charged MP id"].unique()
            )
        else:
            logger.warning(
                "Isolated %s (%s, %s): not found in voltage plateau df",
                fw.mp_id, ion, fw.framework_reduced_formula,
            )
            continue

        for other_mpid in partners:
            j = mpid_to_idx.get(other_mpid)
            if j is None:
                logger.warning(
                    "Partner %s of isolated %s missing from framework list",
                    other_mpid, fw.mp_id,
                )
                continue
            if j not in adjacency[i]:
                adjacency[i].append(j)
            if i not in adjacency[j]:
                adjacency[j].append(i)
            edges_added += 1

    logger.info(
        "Reconnected %d edge(s) across %d isolated framework(s)",
        edges_added, len(isolated_idxs),
    )
    return adjacency


def bfs_components(adjacency, n):
    """
    Find connected components via BFS.

    Returns list of component index lists.
    """
    visited = [False] * n
    components = []

    for i in range(n):
        if visited[i]:
            continue

        queue = deque([i])
        visited[i] = True
        component = []

        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adjacency.get(node, []):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        components.append(component)

    return components


def make_networkx_graph(component_indices, all_frameworks, adjacency):
    """Create a NetworkX graph for one connected component."""
    import networkx as nx

    G = nx.Graph()
    for idx in component_indices:
        fw = all_frameworks[idx]
        G.add_node(
            fw,
            label=f"{fw.mp_id} {fw.struct_reduced_formula}",
            color=ION_COLORS.get(fw.working_ion.symbol, "gray"),
        )
    for idx in component_indices:
        for neighbor_idx in adjacency.get(idx, []):
            if neighbor_idx in component_indices:
                G.add_edge(
                    all_frameworks[idx], all_frameworks[neighbor_idx],
                )
    return G


def _get_df_value_for_edge(u, v, df_dict, column_name, criterion=None):
    """Look up a value from the voltage DataFrame for a same-ion edge."""
    working_ion = u.working_ion.symbol
    if working_ion not in df_dict:
        raise ValueError(f"No DataFrame for {working_ion}")

    df = df_dict[working_ion]

    # Determine charged/discharged
    u_frac = u.struct.composition.get_atomic_fraction(u.working_ion)
    v_frac = v.struct.composition.get_atomic_fraction(v.working_ion)
    if u_frac > v_frac:
        discharged, charged = u, v
    else:
        discharged, charged = v, u

    col = column_name
    if criterion is not None:
        col = f"{column_name} {criterion}"

    row = df[
        (df["Charged MP id"] == charged.mp_id)
        & (df["Discharged MP id"] == discharged.mp_id)
    ]

    if len(row) == 0:
        raise ValueError(
            f"No voltage plateau between {u.mp_id} and {v.mp_id}"
        )

    val = row[col].iloc[0]
    return val


def classify_edges(G, df_dict, pH_filter=(7, 8.5),
                   metastability_criterion=4,
                   min_area_fraction=0.80):
    """
    Categorize graph edges into 5 types.

    Returns dict with keys: ``'diff_ion'``, ``'same_ion'``,
    ``'no_voltage'``, ``'pH_filtered'``, ``'pH_area_filtered'``.
    """
    same_ion_edges = [
        (u, v) for u, v in G.edges()
        if u.working_ion.symbol == v.working_ion.symbol
    ]
    diff_ion_edges = [e for e in G.edges() if e not in same_ion_edges]

    pH_filtered = []
    no_voltage = []
    remaining_same = same_ion_edges.copy()

    req_pH_range = round(pH_filter[1] - pH_filter[0], 1)

    for u, v in same_ion_edges:
        try:
            pH_str = _get_df_value_for_edge(
                u, v, df_dict,
                "pH range metastability criterion",
                criterion=metastability_criterion,
            )
            pH_range = ast.literal_eval(str(pH_str))
            pH_min, pH_max = pH_range

            # Require full coverage of [pH_filter[0], pH_filter[1]]:
            # range fully inside the window AND width equal to the span.
            inside_window = (
                pH_min >= pH_filter[0] and pH_max <= pH_filter[1]
            )
            full_width = round(pH_max - pH_min, 1) == req_pH_range
            if inside_window and full_width:
                pH_filtered.append((u, v))
                remaining_same.remove((u, v))

        except ValueError as e:
            if "No voltage plateau" in str(e):
                no_voltage.append((u, v))
                remaining_same.remove((u, v))
            else:
                raise

    pH_area_filtered = []
    remaining_pH = pH_filtered.copy()

    for u, v in pH_filtered:
        area = _get_df_value_for_edge(
            u, v, df_dict, "Total metastability area fraction",
        )
        if float(area) >= min_area_fraction:
            pH_area_filtered.append((u, v))
            remaining_pH.remove((u, v))

    return {
        "diff_ion": diff_ion_edges,
        "same_ion": remaining_same,
        "no_voltage": no_voltage,
        "pH_filtered": remaining_pH,
        "pH_area_filtered": pH_area_filtered,
    }


def _draw_nodes_and_labels(G, pos):
    """Draw nodes (expt hex / theoretical circle, ion-colored), labels,
    and the two shared legends (working-ion color, phase shape).

    Shared between the stage-4 graph renderer and the step3 component
    plotter.  Returns the current Axes.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import networkx as nx

    expt_nodes = [n for n in G.nodes if n.is_expt]
    non_expt_nodes = [n for n in G.nodes if not n.is_expt]

    expt_colors = [
        ION_COLORS.get(n.working_ion.symbol, "gray")
        for n in expt_nodes
    ]
    non_expt_colors = [
        ION_COLORS.get(n.working_ion.symbol, "gray")
        for n in non_expt_nodes
    ]

    nx.draw_networkx_nodes(
        G, pos, nodelist=expt_nodes, node_size=2000,
        node_color=expt_colors, node_shape="H",
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=non_expt_nodes, node_size=2000,
        node_color=non_expt_colors, node_shape="o",
    )

    labels = {
        n: f"{formula_to_mathtext(n.struct_reduced_formula)}\n{n.mp_id}"
        for n in G.nodes
    }
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)

    ax = plt.gca()
    ion_handles = [
        mpatches.Patch(color=c, label=ion)
        for ion, c in ION_COLORS.items()
    ]
    leg1 = ax.legend(
        handles=ion_handles, title="Working Ion",
        loc="upper left", bbox_to_anchor=(1.01, 1.02), frameon=True,
    )
    ax.add_artist(leg1)

    ms = (2000 ** 0.5) * 0.3
    shape_handles = [
        Line2D([0], [0], marker="H", linestyle="None",
               fillstyle="none", markeredgecolor="black",
               markersize=ms, label="Experimental"),
        Line2D([0], [0], marker="o", linestyle="None",
               fillstyle="none", markeredgecolor="black",
               markersize=ms, label="Theoretical"),
    ]
    leg2 = ax.legend(
        handles=shape_handles, title="Phase type",
        loc="upper left", bbox_to_anchor=(1.01, 0.61), frameon=True,
    )
    ax.add_artist(leg2)
    return ax


def save_graph_figures(
    graphs, all_frameworks, df_dict,
    save_dir="./All_graphs",
    pH_filter=(7, 8.5),
    metastability_criterion=4,
    min_area_fraction=0.80,
):
    """Render and save graph PNGs organized by number of ions."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import networkx as nx

    os.makedirs(save_dir, exist_ok=True)

    for idx, G in enumerate(graphs):
        pos = nx.spring_layout(G)

        framework_formulae = {
            n.framework_reduced_formula for n in G.nodes
        }
        if len(framework_formulae) > 1:
            offending = [
                (n.mp_id, n.framework_reduced_formula) for n in G.nodes
            ]
            raise ValueError(
                f"Component {idx} contains {len(framework_formulae)} "
                f"different framework reduced formulae: {framework_formulae}. "
                f"This indicates a false-positive StructureMatcher match. "
                f"Nodes: {offending}"
            )
        framework_formula = next(iter(framework_formulae))

        plt.figure(figsize=(10, 8))
        ax = _draw_nodes_and_labels(G, pos)

        # Classify and draw edges
        edge_groups = classify_edges(
            G, df_dict, pH_filter, metastability_criterion,
            min_area_fraction,
        )

        nx.draw_networkx_edges(
            G, pos,
            edgelist=edge_groups["diff_ion"] + edge_groups["no_voltage"],
            edge_color="tab:gray", width=1.5,
            style=(0, (5, 3)), alpha=0.4,
        )
        nx.draw_networkx_edges(
            G, pos, edgelist=edge_groups["same_ion"],
            edge_color="tab:gray", width=1.5, style="solid", alpha=0.6,
        )
        nx.draw_networkx_edges(
            G, pos, edgelist=edge_groups["pH_filtered"],
            edge_color="tab:blue", width=1.5,
            style=(0, (5, 3)), alpha=1,
        )
        nx.draw_networkx_edges(
            G, pos, edgelist=edge_groups["pH_area_filtered"],
            edge_color="tab:blue", width=1.5, style="solid", alpha=1,
        )

        edge_handles = [
            Line2D([0], [0], color="tab:gray", alpha=0.4, lw=1.5,
                   linestyle=(0, (5, 3)),
                   label="Not (meta)stable / Diff. ions"),
            Line2D([0], [0], color="tab:gray", alpha=0.6, lw=1.5,
                   linestyle="solid", label="(Meta)stable"),
            Line2D([0], [0], color="tab:blue", lw=1.5,
                   linestyle=(0, (5, 3)),
                   label="(Meta)stable across pH 7-8.5"),
            Line2D([0], [0], color="tab:blue", lw=1.5,
                   linestyle="solid",
                   label="Area > 80% across pH 7-8.5"),
        ]
        ax.legend(
            handles=edge_handles, title="Edge type",
            loc="upper left", bbox_to_anchor=(1.01, 0.37),
            frameon=True, handlelength=2.8,
        )

        ax.spines[:].set_visible(False)
        plt.grid(False)

        num_ions = len({n.working_ion.symbol for n in G.nodes})
        sub_dir = os.path.join(save_dir, f"{num_ions}_ions")
        os.makedirs(sub_dir, exist_ok=True)
        plt.savefig(
            os.path.join(
                sub_dir, f"image_{idx}_{framework_formula}.png",
            ),
            dpi=600, bbox_inches="tight",
        )
        plt.close()


# ====================================================================
# pH / area filtering
# ====================================================================


def filter_plateaus_by_pH_and_area(
    df, pH_range=(7.0, 8.5), min_area_fraction=0.80,
    metastability_criterion=4,
):
    """
    Filter voltage plateaus by pH range coverage and metastability area.

    Parameters
    ----------
    df : pd.DataFrame
    pH_range : tuple of float
    min_area_fraction : float
    metastability_criterion : int

    Returns
    -------
    pd.DataFrame
        Filtered rows.
    """
    if len(df) == 0:
        return df

    req_pH_width = round(pH_range[1] - pH_range[0], 1)
    pH_col = f"pH range metastability criterion {metastability_criterion}"

    def _has_full_pH_coverage(group):
        for _, row in group.iterrows():
            val = row.get(pH_col)
            if pd.isna(val):
                return False
            if isinstance(val, str):
                val = ast.literal_eval(val)
            if isinstance(val, (list, tuple)) and len(val) == 2:
                pH_min, pH_max = val
                inside = pH_min >= pH_range[0] and pH_max <= pH_range[1]
                full_width = round(pH_max - pH_min, 1) == req_pH_width
                if not (inside and full_width):
                    return False
            else:
                return False
        return True

    # pH filter by plot number
    pH_pass = set()
    for plot_num, group in df.groupby("Plot number"):
        if _has_full_pH_coverage(group):
            pH_pass.add(plot_num)

    # Area filter
    area_pass = set(
        df[df["Total metastability area fraction"] >= min_area_fraction]["Plot number"]
    )

    combined = pH_pass.intersection(area_pass)
    return df[df["Plot number"].isin(combined)].reset_index(drop=True)


# ====================================================================
# Formula formatting (shared with set-based module)
# ====================================================================


def formula_to_mathtext(s):
    """Format chemical formula for LaTeX rendering."""
    s = s.replace("\u00b7", r"\cdot ").replace(".", r"\cdot ")
    s = re.sub(r"([A-Za-z\)])(\d+(?:\.\d+)?)", r"\1_{\2}", s)
    return f"${s}$"


# ====================================================================
# Top-level runner
# ====================================================================


def _load_cached_adjacency(cache_path, meta_path, combined_fw, reconnect_isolated):
    """Load adjacency from cache iff sidecar metadata matches."""
    if not (os.path.exists(cache_path) and os.path.exists(meta_path)):
        return None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Cache meta unreadable (%s); rebuilding", e)
        return None

    cur_mpids = [fw.mp_id for fw in combined_fw]
    if meta.get("mpids") != cur_mpids:
        logger.info("Cache mp_id list differs; rebuilding adjacency")
        return None
    if bool(meta.get("reconnect_isolated")) != bool(reconnect_isolated):
        logger.info(
            "Cache reconnect_isolated flag differs (cached=%s, current=%s); "
            "rebuilding adjacency",
            meta.get("reconnect_isolated"), reconnect_isolated,
        )
        return None

    with open(cache_path, "rb") as f:
        return pickle.load(f)


def _save_cached_adjacency(
    cache_path, meta_path, adjacency, combined_fw, reconnect_isolated,
):
    """Pickle adjacency and write sidecar metadata."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(adjacency, f)
    meta = {
        "mpids": [fw.mp_id for fw in combined_fw],
        "reconnect_isolated": bool(reconnect_isolated),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    logger.info("Saved adjacency cache to %s", cache_path)


def run_framework_comparison(
    stage3_dirs,
    output_dir=".",
    elements=None,
    sheet_name="criterion 4",
    pH_range=(7.0, 8.5),
    min_area_fraction=0.80,
    metastability_criterion=4,
    save_plots=True,
    cache_adjacency=True,
    adj_cache_path=None,
    reconnect_isolated=False,
):
    """
    Run the Stage 4 graph-based framework comparison pipeline.

    Loads frameworks for all ions in ``stage3_dirs``, builds one shared
    adjacency over the combined framework pool, identifies connected
    components, and renders one set of graph PNGs. Then, for each ion
    in ``elements``, partitions that ion's voltage plateau DataFrame
    into ``only_{ion}`` (frameworks unique to this ion's component) and
    ``multiple_ions`` (frameworks whose component spans more than one
    ion), applies the pH/area filter, and writes per-ion Excels.

    Parameters
    ----------
    stage3_dirs : dict
        Maps ion symbol to its Stage 3 output dir. Each dir must contain
        ``All_voltage_plateaus.xlsx``, ``structs/``, and
        ``All_charged_structures/`` (or ``All_charged_strutures/``).
    output_dir : str
        Root output directory (default: current directory). Produces
        ``{output_dir}/All_graphs/`` (shared) and
        ``{output_dir}/{ion}/pH_area_filtered_plateaus.xlsx`` per ion.
    elements : list of str or None
        Ions to produce per-ion Excels for. Default: all of
        ``stage3_dirs``.
    sheet_name : str
    pH_range : tuple of float
    min_area_fraction : float
    metastability_criterion : int
    save_plots : bool
    cache_adjacency : bool
    adj_cache_path : str or None
        Override path for the adjacency cache file. Default:
        ``{output_dir}/adjacency_cache.pkl`` (or
        ``adjacency_cache_reconnected.pkl`` when
        ``reconnect_isolated=True``). Sidecar
        ``adjacency_cache[.meta].json`` stores the mp_id list and the
        reconnect_isolated flag; loading is refused if these drift.
    reconnect_isolated : bool
        If True, after building adjacency call
        ``reconnect_isolated_frameworks`` to manually add edges between
        isolated frameworks and their voltage-plateau partner from the
        same-ion Excel. Encodes the assumption that two structures
        sharing a voltage plateau row ARE the same framework even if
        ``StructureMatcher.fit`` says otherwise.

    Returns
    -------
    dict
        ``{ion: {"only_{ion}": DataFrame, "multiple_ions": DataFrame}}``
    """
    if elements is None:
        elements = list(stage3_dirs.keys())

    os.makedirs(output_dir, exist_ok=True)

    # --- Load frameworks for all ions ---
    frameworks_dict = {}
    df_dict = {}

    for ion in stage3_dirs:
        s3_dir = stage3_dirs[ion]
        excel_path = os.path.join(s3_dir, "All_voltage_plateaus.xlsx")
        discharged_dir = os.path.join(s3_dir, "structs")

        # Handle typo in original directory name
        charged_dir = os.path.join(s3_dir, "All_charged_structures")
        if not os.path.exists(charged_dir):
            charged_dir = os.path.join(s3_dir, "All_charged_strutures")

        logger.info("Loading frameworks for %s from %s", ion, s3_dir)
        dis_set, chg_set, vdf = load_all_frameworks_for_element(
            excel_path, sheet_name, discharged_dir, charged_dir, ion,
        )
        frameworks_dict[ion] = (dis_set, chg_set)
        df_dict[ion] = vdf

    # --- Combine all ions' frameworks into one list ---
    all_fw_lists = {}
    for ion in frameworks_dict:
        dis, chg = frameworks_dict[ion]
        all_fw_lists[ion] = _get_all_frameworks_list(list(dis), list(chg))

    combined_fw = []
    for ion in all_fw_lists:
        combined_fw.extend(all_fw_lists[ion])

    logger.info(
        "Combined framework count: %d (across %d ions)",
        len(combined_fw), len(all_fw_lists),
    )

    # --- Build or load adjacency ---
    if adj_cache_path is None:
        cache_name = (
            "adjacency_cache_reconnected.pkl"
            if reconnect_isolated else "adjacency_cache.pkl"
        )
        adj_cache_path = os.path.join(output_dir, cache_name)
    meta_path = (
        os.path.splitext(adj_cache_path)[0] + ".meta.json"
    )

    adjacency = None
    if cache_adjacency:
        adjacency = _load_cached_adjacency(
            adj_cache_path, meta_path, combined_fw, reconnect_isolated,
        )
        if adjacency is not None:
            logger.info("Loaded cached adjacency from %s", adj_cache_path)

    if adjacency is None:
        adjacency = build_adjacency_dict(combined_fw)
        if reconnect_isolated:
            logger.info("Reconnecting isolated frameworks via plateau df")
            reconnect_isolated_frameworks(adjacency, combined_fw, df_dict)
        if cache_adjacency:
            _save_cached_adjacency(
                adj_cache_path, meta_path, adjacency,
                combined_fw, reconnect_isolated,
            )

    # --- Connected components & graphs ---
    components = bfs_components(adjacency, len(combined_fw))
    logger.info("Found %d connected components", len(components))

    graphs = [
        make_networkx_graph(comp, combined_fw, adjacency)
        for comp in components
    ]

    if save_plots:
        graph_dir = os.path.join(output_dir, "All_graphs")
        save_graph_figures(
            graphs, combined_fw, df_dict,
            save_dir=graph_dir,
            pH_filter=pH_range,
            metastability_criterion=metastability_criterion,
            min_area_fraction=min_area_fraction,
        )

    # --- Per-ion Excel partition (graph-component-based) ---
    # For each component, record which ions appear in it. An ion's mp_id
    # is "multi-ion" iff its component contains nodes from another ion.
    multi_ion_mpids_per_element = {ion: set() for ion in elements}
    for comp in components:
        ions_in_comp = {
            combined_fw[i].working_ion.symbol for i in comp
        }
        if len(ions_in_comp) <= 1:
            continue
        for i in comp:
            fw = combined_fw[i]
            sym = fw.working_ion.symbol
            if sym in multi_ion_mpids_per_element:
                multi_ion_mpids_per_element[sym].add(fw.mp_id)

    results = {}
    for element in elements:
        logger.info("=== Filtering voltage plateaus for %s ===", element)
        if element not in df_dict:
            logger.warning("No voltage df for %s; skipping", element)
            continue
        element_df = df_dict[element]
        multi_mpids = multi_ion_mpids_per_element[element]

        multi_mask = (
            element_df["Charged MP id"].isin(multi_mpids)
            | element_df["Discharged MP id"].isin(multi_mpids)
        )
        only_element_df = element_df[~multi_mask].copy()
        multiple_ions_df = element_df[multi_mask].copy()

        filtered_only = filter_plateaus_by_pH_and_area(
            only_element_df, pH_range, min_area_fraction,
            metastability_criterion,
        )
        filtered_multi = filter_plateaus_by_pH_and_area(
            multiple_ions_df, pH_range, min_area_fraction,
            metastability_criterion,
        )

        element_out = os.path.join(output_dir, element)
        os.makedirs(element_out, exist_ok=True)
        excel_path = os.path.join(
            element_out, "pH_area_filtered_plateaus.xlsx",
        )
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            filtered_only.to_excel(
                writer, sheet_name=f"only_{element}", index=False,
            )
            filtered_multi.to_excel(
                writer, sheet_name="multiple_ions", index=False,
            )
        logger.info(
            "%s: only_%s=%d rows, multiple_ions=%d rows -> %s",
            element, element, len(filtered_only),
            len(filtered_multi), excel_path,
        )

        results[element] = {
            f"only_{element}": filtered_only,
            "multiple_ions": filtered_multi,
        }

    return results

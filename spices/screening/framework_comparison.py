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
from spices.plotting import apply_publication_style

apply_publication_style()

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


NODE_SIZE = 4500
NODE_BORDER_WIDTH = 1.0
# Legend marker size is decoupled from NODE_SIZE so the legend remains
# visually proportional to the (fixed) legend font size even if NODE_SIZE
# is tuned for in-graph readability.
LEGEND_MARKER_SIZE = (2000 ** 0.5) * 0.3


def _node_color(G, n):
    """Resolve a node's display color: explicit attr, then ion lookup."""
    attr_color = G.nodes[n].get("color")
    if attr_color is not None:
        return attr_color
    if getattr(n, "working_ion", None) is not None:
        return ION_COLORS.get(n.working_ion.symbol, "lightgrey")
    return "lightgrey"


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

    expt_colors = [_node_color(G, n) for n in expt_nodes]
    non_expt_colors = [_node_color(G, n) for n in non_expt_nodes]

    expt_coll = nx.draw_networkx_nodes(
        G, pos, nodelist=expt_nodes, node_size=NODE_SIZE,
        node_color=expt_colors, node_shape="H",
        edgecolors="black", linewidths=NODE_BORDER_WIDTH,
    )
    non_expt_coll = nx.draw_networkx_nodes(
        G, pos, nodelist=non_expt_nodes, node_size=NODE_SIZE,
        node_color=non_expt_colors, node_shape="o",
        edgecolors="black", linewidths=NODE_BORDER_WIDTH,
    )
    # Markers are sized in display pt^2 and would otherwise be hard-clipped
    # at the axes spine when their centre sits near the data-range edge.
    # _pad_axes_to_fit_nodes (below) pads the view to fit them; clip_on=False
    # is belt-and-suspenders so a marker is never cut by the axes box.
    for coll in (expt_coll, non_expt_coll):
        if coll is not None:
            coll.set_clip_on(False)

    labels = {
        n: f"{formula_to_mathtext(n.struct_reduced_formula)}\n{n.mp_id}"
        for n in G.nodes
    }
    text_items = nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)
    for text in text_items.values():
        text.set_clip_on(False)

    ax = plt.gca()
    ion_handles = [
        mpatches.Patch(color=c, label=ion)
        for ion, c in ION_COLORS.items()
    ]
    ion_handles.append(
        mpatches.Patch(color="lightgrey", label="Multi-ion / none"),
    )
    leg1 = ax.legend(
        handles=ion_handles, title="Working Ion",
        loc="upper left", bbox_to_anchor=(1.01, 1.02), frameon=True,
    )
    ax.add_artist(leg1)

    shape_handles = [
        Line2D([0], [0], marker="H", linestyle="None",
               fillstyle="none", markeredgecolor="black",
               markersize=LEGEND_MARKER_SIZE, label="Experimental"),
        Line2D([0], [0], marker="o", linestyle="None",
               fillstyle="none", markeredgecolor="black",
               markersize=LEGEND_MARKER_SIZE, label="Theoretical"),
    ]
    leg2 = ax.legend(
        handles=shape_handles, title="Phase type",
        loc="upper left", bbox_to_anchor=(1.01, 0.61), frameon=True,
        labelspacing=1.8, borderpad=0.8,
    )
    ax.add_artist(leg2)

    _pad_axes_to_fit_nodes(ax, pos)
    return ax


def _pad_axes_to_fit_nodes(ax, pos, node_size=NODE_SIZE, frac=1.3):
    """Pad the axes view limits so fixed-size node markers fit fully.

    ``spring_layout`` positions are node *centres*; matplotlib autoscales
    to those points only, so a marker (sized in display pt^2) near the
    data-range edge spills past the axes box and gets clipped at the spine.
    This pads each axis by the node radius converted to data units, with a
    safety factor for the label that overflows the marker.

    ``ax.margins`` is unsuitable: it pads by a fraction of the data span,
    which is ~0 for the linear layout of a 2-node component.
    """
    import math

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Seed the view (with a floor for degenerate/linear layouts) so transData
    # is non-singular before we read the data-per-pixel scale from it.
    eps = 0.5
    ax.set_xlim(x_min - eps, x_max + eps)
    ax.set_ylim(y_min - eps, y_max + eps)

    r_px = math.sqrt(node_size / math.pi) / 72.0 * ax.figure.dpi
    inv = ax.transData.inverted()
    ox, oy = inv.transform((0.0, 0.0))
    rx, ry = inv.transform((r_px, r_px))
    pad_x = abs(rx - ox) * frac
    pad_y = abs(ry - oy) * frac

    # set_xlim/ylim disables autoscale, so edges drawn afterward by callers
    # (which sit between nodes) won't override these padded limits.
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)


class _MergedFrameworkNode:
    """Proxy node for a group of FrameworkStructure objects sharing one mp_id.

    Surfaces the same attribute interface used by ``_draw_nodes_and_labels``
    so the same renderer accepts both unmerged ``FrameworkStructure`` and
    merged proxies.
    """

    def __init__(self, mp_id, framework_reduced_formula, is_expt,
                 working_ions):
        self.mp_id = mp_id
        self.framework_reduced_formula = framework_reduced_formula
        # Merged node has no single intercalated composition; show the
        # bare framework formula in the label.
        self.struct_reduced_formula = framework_reduced_formula
        self.is_expt = is_expt
        self.working_ion = None
        self.working_ions = set(working_ions)


EDGE_PRIORITY = {
    "pH_area_filtered": 4,
    "pH_filtered": 3,
    "same_ion": 2,
    "diff_ion": 1,
    "no_voltage": 1,
}


def _collapse_graph_by_mpid(G, edge_groups):
    """Merge same-mp_id nodes into a single light-grey proxy node.

    A merged group MUST share ``is_expt`` — same mp_id has one provenance
    (experimental or theoretical).  Mismatch raises ``ValueError``.

    Parameters
    ----------
    G : networkx.Graph
        Original graph with ``FrameworkStructure`` nodes.
    edge_groups : dict
        Output of :func:`classify_edges`.

    Returns
    -------
    G_display : networkx.Graph
        Graph with merged proxy nodes.  Each node has a ``color``
        attribute: ion color for singletons, ``"lightgrey"`` for merged
        proxies.
    edge_groups_display : dict
        Edge groups remapped to the merged graph.  Self-loops dropped;
        parallel edges between the same display-node pair collapse to
        the single highest-priority category
        (``pH_area_filtered`` > ``pH_filtered`` > ``same_ion`` >
        ``diff_ion`` / ``no_voltage``).
    """
    import networkx as nx

    groups = defaultdict(list)
    for n in G.nodes:
        groups[n.mp_id].append(n)

    proxy = {}
    for mp_id, group in groups.items():
        if len(group) == 1:
            proxy[mp_id] = group[0]
            continue
        is_expt_values = {n.is_expt for n in group}
        if len(is_expt_values) > 1:
            # TODO: remove this raise once verified across the full
            # 5-ion dataset.  Same mp_id should have one provenance.
            raise ValueError(
                f"Merged node mp_id={mp_id} has inconsistent is_expt: "
                f"{sorted(is_expt_values)}"
            )
        proxy[mp_id] = _MergedFrameworkNode(
            mp_id=mp_id,
            framework_reduced_formula=group[0].framework_reduced_formula,
            is_expt=group[0].is_expt,
            working_ions={n.working_ion.symbol for n in group},
        )

    G_display = nx.Graph()
    for mp_id, p in proxy.items():
        if isinstance(p, _MergedFrameworkNode):
            color = "lightgrey"
        else:
            color = ION_COLORS.get(p.working_ion.symbol, "lightgrey")
        G_display.add_node(p, color=color)

    best_cat = {}
    for cat, edges in edge_groups.items():
        prio = EDGE_PRIORITY[cat]
        for u, v in edges:
            u_p = proxy[u.mp_id]
            v_p = proxy[v.mp_id]
            if u_p is v_p:
                continue
            key = frozenset((id(u_p), id(v_p)))
            current = best_cat.get(key)
            if current is None or current[0] < prio:
                best_cat[key] = (prio, cat, u_p, v_p)

    edge_groups_display = {cat: [] for cat in edge_groups}
    for _, cat, u_p, v_p in best_cat.values():
        G_display.add_edge(u_p, v_p)
        edge_groups_display[cat].append((u_p, v_p))

    return G_display, edge_groups_display


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

        edge_groups = classify_edges(
            G, df_dict, pH_filter, metastability_criterion,
            min_area_fraction,
        )

        G_display, edge_groups_display = _collapse_graph_by_mpid(
            G, edge_groups,
        )
        pos = nx.spring_layout(G_display)

        plt.figure(figsize=(7.2, 5.4))
        ax = _draw_nodes_and_labels(G_display, pos)

        nx.draw_networkx_edges(
            G_display, pos,
            edgelist=(
                edge_groups_display["diff_ion"]
                + edge_groups_display["no_voltage"]
            ),
            edge_color="tab:gray", width=1.0,
            style=(0, (5, 3)), alpha=0.4,
        )
        nx.draw_networkx_edges(
            G_display, pos, edgelist=edge_groups_display["same_ion"],
            edge_color="tab:gray", width=1.0, style="solid", alpha=0.6,
        )
        nx.draw_networkx_edges(
            G_display, pos, edgelist=edge_groups_display["pH_filtered"],
            edge_color="tab:blue", width=1.0,
            style=(0, (5, 3)), alpha=1,
        )
        nx.draw_networkx_edges(
            G_display, pos,
            edgelist=edge_groups_display["pH_area_filtered"],
            edge_color="tab:blue", width=1.0, style="solid", alpha=1,
        )

        edge_handles = [
            Line2D([0], [0], color="tab:gray", alpha=0.4, lw=1.0,
                   linestyle=(0, (5, 3)),
                   label="Not (meta)stable / Diff. ions"),
            Line2D([0], [0], color="tab:gray", alpha=0.6, lw=1.0,
                   linestyle="solid", label="(Meta)stable"),
            Line2D([0], [0], color="tab:blue", lw=1.0,
                   linestyle=(0, (5, 3)),
                   label="(Meta)stable across pH 7-8.5"),
            Line2D([0], [0], color="tab:blue", lw=1.0,
                   linestyle="solid",
                   label="Area > 80% across pH 7-8.5"),
        ]
        ax.legend(
            handles=edge_handles, title="Edge type",
            loc="upper left", bbox_to_anchor=(1.01, 0.37),
            frameon=True, handlelength=2.8,
        )

        ax.spines[:].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.grid(False)

        num_ions = len({
            n.working_ion.symbol for n in G.nodes
            if n.working_ion is not None
        })
        sub_dir = os.path.join(save_dir, f"{num_ions}_ions")
        os.makedirs(sub_dir, exist_ok=True)
        plt.savefig(
            os.path.join(
                sub_dir, f"image_{idx}_{framework_formula}.png",
            ),
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
    """Format chemical formula for LaTeX rendering with upright element symbols."""
    s = s.replace("\u00b7", r"\cdot ").replace(".", r"\cdot ")
    s = re.sub(r"([A-Za-z\)])(\d+(?:\.\d+)?)", r"\1_{\2}", s)
    s = re.sub(r"([A-Za-z]+)", r"\\mathrm{\1}", s)
    return f"${s}$"


# ====================================================================
# Top-level runner
# ====================================================================


CACHE_SCHEMA_VERSION = 2


def _diff_first_n(a, b, n=5):
    """Return (idx, a_val, b_val) for the first n positions where lists differ."""
    out = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            out.append((i, x, y))
            if len(out) >= n:
                return out
    if len(a) != len(b) and len(out) < n:
        if len(a) > len(b):
            for i in range(len(b), min(len(a), len(b) + n - len(out))):
                out.append((i, a[i], None))
        else:
            for i in range(len(a), min(len(b), len(a) + n - len(out))):
                out.append((i, None, b[i]))
    return out


def _load_cached_state(cache_path, meta_path, stage3_dirs, sheet_name,
                      reconnect_isolated):
    """Load (combined_fw, adjacency) bundle if sidecar metadata matches.

    Returns ``(combined_fw, adjacency)`` on success, ``None`` on miss
    (file absent, schema mismatch, key mismatch, or corruption).
    """
    if not (os.path.exists(cache_path) and os.path.exists(meta_path)):
        return None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Cache meta unreadable (%s); rebuilding", e)
        return None

    if meta.get("schema_version") != CACHE_SCHEMA_VERSION:
        logger.info(
            "Cache schema version mismatch (cached=%s, current=%s); "
            "rebuilding",
            meta.get("schema_version"), CACHE_SCHEMA_VERSION,
        )
        return None

    cur_keys = list(stage3_dirs.keys())
    if list(meta.get("stage3_dirs_keys", [])) != cur_keys:
        logger.info(
            "Cache stage3 ion order differs (cached=%s, current=%s); "
            "rebuilding",
            meta.get("stage3_dirs_keys"), cur_keys,
        )
        return None

    if meta.get("sheet_name") != sheet_name:
        logger.info(
            "Cache sheet_name differs (cached=%s, current=%s); rebuilding",
            meta.get("sheet_name"), sheet_name,
        )
        return None

    if bool(meta.get("reconnect_isolated")) != bool(reconnect_isolated):
        logger.info(
            "Cache reconnect_isolated flag differs "
            "(cached=%s, current=%s); rebuilding",
            meta.get("reconnect_isolated"), reconnect_isolated,
        )
        return None

    try:
        with open(cache_path, "rb") as f:
            bundle = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, AttributeError, OSError) as e:
        logger.warning("Cache pickle unreadable (%s); rebuilding", e)
        return None

    if not isinstance(bundle, dict) or "combined_fw" not in bundle \
            or "adjacency" not in bundle:
        logger.info(
            "Cache pickle is legacy adjacency-only format; rebuilding "
            "(framework pool was not cached)"
        )
        return None

    combined_fw = bundle["combined_fw"]
    adjacency = bundle["adjacency"]

    cached_mpids = meta.get("mpids", [])
    actual_mpids = [fw.mp_id for fw in combined_fw]
    if cached_mpids != actual_mpids:
        diffs = _diff_first_n(cached_mpids, actual_mpids)
        logger.warning(
            "Cache pickle mp_ids don't match sidecar; rebuilding. "
            "First diffs: %s", diffs,
        )
        return None

    return combined_fw, adjacency


def _save_cached_state(cache_path, meta_path, combined_fw, adjacency,
                       stage3_dirs, sheet_name, reconnect_isolated):
    """Pickle (combined_fw, adjacency) bundle and write sidecar metadata."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    bundle = {"combined_fw": combined_fw, "adjacency": adjacency}
    with open(cache_path, "wb") as f:
        pickle.dump(bundle, f)
    n_edges = sum(len(v) for v in adjacency.values()) // 2
    meta = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "stage3_dirs_keys": list(stage3_dirs.keys()),
        "sheet_name": sheet_name,
        "reconnect_isolated": bool(reconnect_isolated),
        "mpids": [fw.mp_id for fw in combined_fw],
        "n_frameworks": len(combined_fw),
        "n_adjacency_edges": n_edges,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    logger.info(
        "Saved step4 cache to %s (%d frameworks, %d edges)",
        cache_path, len(combined_fw), n_edges,
    )


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
    rerender_plots=False,
    force_rebuild=False,
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
    rerender_plots : bool
        If True, require a valid cache and skip both the framework load
        and the per-ion Excel writes; only regenerate PNGs.  Raises
        ``FileNotFoundError`` if no valid cache is found.  Use when
        iterating on plot styling.
    force_rebuild : bool
        If True, ignore any existing cache and rebuild from scratch.

    Returns
    -------
    dict
        ``{ion: {"only_{ion}": DataFrame, "multiple_ions": DataFrame}}``
        When ``rerender_plots=True``, returns an empty dict (no Excels).
    """
    if elements is None:
        elements = list(stage3_dirs.keys())

    os.makedirs(output_dir, exist_ok=True)

    # --- Resolve cache paths ---
    if adj_cache_path is None:
        cache_name = (
            "step4_cache_reconnected.pkl"
            if reconnect_isolated else "step4_cache.pkl"
        )
        adj_cache_path = os.path.join(output_dir, cache_name)
    meta_path = os.path.splitext(adj_cache_path)[0] + ".meta.json"

    # --- Try cache first (skips both load and adjacency build) ---
    cached = None
    if cache_adjacency and not force_rebuild:
        cached = _load_cached_state(
            adj_cache_path, meta_path,
            stage3_dirs, sheet_name, reconnect_isolated,
        )

    df_dict = {}
    if cached is not None:
        combined_fw, adjacency = cached
        logger.info(
            "Loaded %d frameworks + adjacency from cache %s",
            len(combined_fw), adj_cache_path,
        )
        # Still need df_dict for edge classification + per-ion Excel
        # writes; reading Excels is cheap.
        for ion in stage3_dirs:
            excel_path = os.path.join(
                stage3_dirs[ion], "All_voltage_plateaus.xlsx",
            )
            df_dict[ion] = pd.read_excel(excel_path, sheet_name=sheet_name)
    else:
        if rerender_plots:
            raise FileNotFoundError(
                f"--rerender-plots requires a valid cache at "
                f"{adj_cache_path}, but none was found or it failed "
                f"validation. Run the full pipeline once to build the "
                f"cache first."
            )

        # --- Full load: frameworks for all ions ---
        frameworks_dict = {}
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
            all_fw_lists[ion] = _get_all_frameworks_list(
                list(dis), list(chg),
            )

        combined_fw = []
        for ion in all_fw_lists:
            combined_fw.extend(all_fw_lists[ion])

        logger.info(
            "Combined framework count: %d (across %d ions)",
            len(combined_fw), len(all_fw_lists),
        )

        # --- Build adjacency ---
        adjacency = build_adjacency_dict(combined_fw)
        if reconnect_isolated:
            logger.info("Reconnecting isolated frameworks via plateau df")
            reconnect_isolated_frameworks(adjacency, combined_fw, df_dict)
        if cache_adjacency:
            _save_cached_state(
                adj_cache_path, meta_path,
                combined_fw, adjacency,
                stage3_dirs, sheet_name, reconnect_isolated,
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

    if rerender_plots:
        logger.info(
            "--rerender-plots: skipping per-ion Excel writes."
        )
        return {}

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

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
import logging
import os
import pickle
import re
import shutil
import warnings
from collections import defaultdict, deque
from itertools import combinations

import numpy as np
import pandas as pd
from pymatgen.core import Structure, Element, Composition

from spices.intercalation.structure import DEFAULT_MATCHER
from spices.config import SUPPORTED_ELEMENTS

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
# Set operations (optional, enabled by run_set_comparison=True)
# ====================================================================


def framework_set_intersection(
    req_charged, req_discharged, other_charged, other_discharged,
):
    """
    Frameworks from the required ion that are shared with another ion.
    Four-way intersection (charged-charged, charged-discharged, etc.)
    combined via union.
    """
    dd = {x for x in req_discharged if x in other_discharged}
    dc = {x for x in req_discharged if x in other_charged}
    cd = {x for x in req_charged if x in other_discharged}
    cc = {x for x in req_charged if x in other_charged}
    return dd.union(dc, cd, cc)


def multi_ion_common_frameworks(
    req_charged, req_discharged,
    other_charged_list, other_discharged_list,
):
    """Frameworks common to the required ion AND all other ions."""
    if len(other_charged_list) != len(other_discharged_list):
        raise ValueError("Charged and discharged lists must be same length")

    result = None
    for chg, dis in zip(other_charged_list, other_discharged_list):
        intersection = framework_set_intersection(
            req_charged, req_discharged, chg, dis,
        )
        if result is None:
            result = intersection
        else:
            result = result.intersection(intersection)
    return result or set()


def ion_selective_frameworks(
    req_charged, req_discharged,
    other_charged_list, other_discharged_list,
):
    """Frameworks unique to the required ion (not shared with any other)."""
    if len(other_charged_list) != len(other_discharged_list):
        raise ValueError("Charged and discharged lists must be same length")

    selective = req_discharged.union(req_charged)
    for chg, dis in zip(other_charged_list, other_discharged_list):
        common = multi_ion_common_frameworks(
            selective, set(), [chg], [dis],
        )
        selective = selective - common
    return selective


# ====================================================================
# Graph-based comparison (PRIMARY)
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
        framework_formula = list(framework_formulae)[0]

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
# Visualization helpers (set-based comparison)
# ====================================================================


def formula_to_mathtext(s):
    """Format chemical formula for LaTeX rendering."""
    s = s.replace("\u00b7", r"\cdot ").replace(".", r"\cdot ")
    s = re.sub(r"([A-Za-z\)])(\d+(?:\.\d+)?)", r"\1_{\2}", s)
    return f"${s}$"


def _get_working_ion_fraction(phase_formula, framework_formula, working_ion):
    """Compute working ion fraction x in M_x[Framework]."""
    framework_weight = Composition(framework_formula).get_reduced_formula_and_factor()[1]

    if working_ion in phase_formula:
        phase_comp = Composition(phase_formula)
        num_ion = phase_comp.get_el_amt_dict().get(working_ion, 0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fw_comp = phase_comp.replace({working_ion: {working_ion: 0}})

        num_fw = fw_comp.get_reduced_formula_and_factor()[1] / framework_weight
        return num_ion / num_fw

    return 0


def voltage_step_plot(
    plot_num, df, ion_name, fig=None,
    save=True, save_dir="./voltage_plots",
):
    """Generate a voltage vs. ion-fraction step plot."""
    import matplotlib.pyplot as plt

    plot_df = df[df["Plot number"] == plot_num]
    if plot_df.empty:
        raise ValueError(f"No data for plot number: {plot_num}")

    framework_formula = plot_df["Framework formula"].iloc[0]
    most_charged = plot_df["Charged Formula"].iloc[0]
    charged_frac = _get_working_ion_fraction(
        most_charged, framework_formula, ion_name,
    )
    discharged_fracs = [
        _get_working_ion_fraction(f, framework_formula, ion_name)
        for f in plot_df["Discharged Formula"].tolist()
    ]
    voltages = plot_df["V vs SHE (V)"].tolist()

    x_vals = [charged_frac] + discharged_fracs
    y_vals = [voltages[0]] + voltages

    if fig is None:
        fig, ax = plt.subplots(dpi=150)
    else:
        ax = fig.axes[0] if fig.axes else fig.add_subplot(1, 1, 1)

    ax.axhline(1.23, color="blue", linestyle="--", linewidth=1.5)
    ax.axhline(-0.71, color="blue", linestyle="--", linewidth=1.5)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.tick_params(direction="in", length=6, width=1.5, which="major")
    ax.tick_params(direction="in", length=3, width=1.0, which="minor")
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.minorticks_on()
    ax.grid(which="major", color="#CCCCCC", linestyle="-", linewidth=1.5)
    ax.grid(which="minor", color="#DDDDDD", linestyle=":", linewidth=1)

    ax.step(
        x_vals, y_vals,
        label=f"M = {ion_name}; plot {plot_num}",
        color=ION_COLORS.get(ion_name, "black"),
    )
    ax.set_ylim(-0.8, 1.3)
    ax.set_xlabel(
        r"$x$ in $M_x$" + formula_to_mathtext(framework_formula),
        fontsize=14,
    )
    ax.set_ylabel("Applied potential (V vs SHE)", fontsize=14)
    ax.tick_params(direction="in")
    legend = ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1),
        borderaxespad=0.5, frameon=True,
    )
    legend.get_frame().set_edgecolor("black")

    if save:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"voltage_plot_{framework_formula}_plot{plot_num}.png"
        fig.savefig(
            os.path.join(save_dir, fname), dpi=300, bbox_inches="tight",
        )
        plt.close(fig)

    return fig


def overlay_multi_ion_voltage_plots(
    req_framework_set, req_df,
    other_framework_sets, other_dfs,
    req_ion, other_ions,
    save_dir, save=True,
):
    """Overlay voltage plots for shared frameworks across multiple ions."""
    for framework in req_framework_set:
        plot_filter = (
            (req_df["Discharged MP id"] == framework.mp_id)
            | (req_df["Charged MP id"] == framework.mp_id)
        )
        plot_nums = req_df[plot_filter]["Plot number"].unique()

        for plot_num in plot_nums:
            figure = None
            for i, other_set in enumerate(other_framework_sets):
                matching = [f for f in other_set if framework == f]
                for match_fw in matching:
                    pf = (
                        (other_dfs[i]["Discharged MP id"] == match_fw.mp_id)
                        | (other_dfs[i]["Charged MP id"] == match_fw.mp_id)
                    )
                    other_plot_nums = other_dfs[i][pf]["Plot number"].unique()
                    for pn in other_plot_nums:
                        figure = voltage_step_plot(
                            pn, other_dfs[i], other_ions[i],
                            fig=figure, save=False,
                        )

            voltage_step_plot(
                plot_num, req_df, req_ion,
                fig=figure, save=save, save_dir=save_dir,
            )


def save_all_combination_voltage_plots(
    frameworks_dict, df_dict, ion_of_interest,
    n_ions, save_dir="./Voltage_plateaus",
):
    """Generate overlaid plots for all N-ion combinations."""
    other_ions = [ion for ion in frameworks_dict if ion != ion_of_interest]
    combs = combinations(other_ions, n_ions - 1)

    for comb in combs:
        req_set = multi_ion_common_frameworks(
            req_charged=frameworks_dict[ion_of_interest][1],
            req_discharged=frameworks_dict[ion_of_interest][0],
            other_charged_list=[frameworks_dict[i][1] for i in comb],
            other_discharged_list=[frameworks_dict[i][0] for i in comb],
        )
        if not req_set:
            continue

        other_sets = []
        other_dfs = []
        for other in comb:
            oset = multi_ion_common_frameworks(
                req_charged=frameworks_dict[other][1],
                req_discharged=frameworks_dict[other][0],
                other_charged_list=(
                    [frameworks_dict[i][1] for i in comb if i != other]
                    + [frameworks_dict[ion_of_interest][1]]
                ),
                other_discharged_list=(
                    [frameworks_dict[i][0] for i in comb if i != other]
                    + [frameworks_dict[ion_of_interest][0]]
                ),
            )
            other_sets.append(oset)
            other_dfs.append(df_dict[other])

        leaf = f"{ion_of_interest}-" + "-".join(comb)
        path = os.path.join(save_dir, f"{n_ions}-ions", leaf)

        overlay_multi_ion_voltage_plots(
            req_set, df_dict[ion_of_interest],
            other_sets, other_dfs,
            ion_of_interest, list(comb),
            save_dir=path, save=True,
        )


def venn_2set(
    frameworks_dict, ion_a, ion_b,
    intersection_set=None,
    save_dir="./Venn-diags", save=True,
):
    """Generate a 2-set Venn diagram comparing two ions."""
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2 as _venn2

    # Use .union() for "all frameworks" (bug fix: original used .intersection())
    set_a = frameworks_dict[ion_a][0].union(frameworks_dict[ion_a][1])
    set_b = frameworks_dict[ion_b][0].union(frameworks_dict[ion_b][1])

    if intersection_set is None:
        intersection_set = multi_ion_common_frameworks(
            frameworks_dict[ion_a][1], frameworks_dict[ion_a][0],
            [frameworks_dict[ion_b][1]], [frameworks_dict[ion_b][0]],
        )

    n_common = len(intersection_set)
    colors = {
        "Li": "crimson", "Na": "gold",
        "K": "#988ED5", "Mg": "#348ABD", "Ca": "#8EBA42",
    }

    _venn2(
        subsets=(len(set_a) - n_common, len(set_b) - n_common, n_common),
        set_labels=(ion_a, ion_b),
        set_colors=(colors.get(ion_a, "gray"), colors.get(ion_b, "gray")),
        alpha=0.3,
    )

    if save:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"{ion_a}-{ion_b}.pdf"), dpi=300,
        )
    plt.close()


# ====================================================================
# Top-level runner
# ====================================================================


def run_framework_comparison(
    element,
    stage3_dirs,
    output_dir=None,
    sheet_name="criterion 4",
    pH_range=(7.0, 8.5),
    min_area_fraction=0.80,
    metastability_criterion=4,
    n_ion_combinations=None,
    save_plots=True,
    save_venn=True,
    copy_filtered_plots=True,
    run_set_comparison=False,
    cache_adjacency=True,
):
    """
    Run the Stage 4 framework comparison pipeline.

    Graph-based comparison is the primary analysis. Set-based comparison
    (voltage overlays, Venn diagrams) is optional.

    Parameters
    ----------
    element : str
        Ion of interest (e.g. ``"Ca"``).
    stage3_dirs : dict
        Maps element symbol to Stage 3 output directory. Each dir must
        contain ``All_voltage_plateaus.xlsx``, ``structs/``, and
        ``All_charged_structures/`` (or ``All_charged_strutures/``).
    output_dir : str or None
    sheet_name : str
    pH_range : tuple of float
    min_area_fraction : float
    metastability_criterion : int
    n_ion_combinations : list of int or None
    save_plots : bool
    save_venn : bool
    copy_filtered_plots : bool
    run_set_comparison : bool
        If True, also run set-based comparison (voltage overlays, Venn).
    cache_adjacency : bool
        If True, save/load adjacency dict as pickle.

    Returns
    -------
    dict
        ``{"only_{element}": DataFrame, "multiple_ions": DataFrame}``
    """
    if output_dir is None:
        output_dir = f"{element}_framework_comparison"
    if n_ion_combinations is None:
        n_ion_combinations = [2, 3, 4, 5]

    os.makedirs(output_dir, exist_ok=True)

    # --- Load frameworks for all elements ---
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

    # --- Graph-based comparison (primary) ---
    logger.info("=== Running graph-based framework comparison ===")

    all_fw_lists = {}
    for ion in frameworks_dict:
        dis, chg = frameworks_dict[ion]
        all_fw_lists[ion] = _get_all_frameworks_list(list(dis), list(chg))

    combined_fw = []
    for ion in all_fw_lists:
        combined_fw.extend(all_fw_lists[ion])

    # Build or load adjacency dict
    adj_cache_path = os.path.join(output_dir, "adjacency_cache.pkl")
    if cache_adjacency and os.path.exists(adj_cache_path):
        logger.info("Loading cached adjacency dict from %s", adj_cache_path)
        with open(adj_cache_path, "rb") as f:
            adjacency = pickle.load(f)
    else:
        adjacency = build_adjacency_dict(combined_fw)
        if cache_adjacency:
            with open(adj_cache_path, "wb") as f:
                pickle.dump(adjacency, f)
            logger.info("Saved adjacency dict to %s", adj_cache_path)

    # BFS connected components → graphs
    components = bfs_components(adjacency, len(combined_fw))
    component_indices_sets = [set(comp) for comp in components]

    import networkx as nx
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

    # --- Set-based comparison (optional) ---
    if run_set_comparison:
        logger.info("=== Running set-based framework comparison ===")

        other_ions = [ion for ion in frameworks_dict if ion != element]

        # Pairwise intersections
        intersections = {}
        for other in other_ions:
            intersections[other] = framework_set_intersection(
                frameworks_dict[element][1], frameworks_dict[element][0],
                frameworks_dict[other][1], frameworks_dict[other][0],
            )

        # Voltage plots
        if save_plots:
            for n in n_ion_combinations:
                save_all_combination_voltage_plots(
                    frameworks_dict, df_dict, element, n,
                    save_dir=os.path.join(output_dir, "Voltage_plateaus"),
                )

        # Venn diagrams
        if save_venn:
            for other in other_ions:
                venn_2set(
                    frameworks_dict, element, other,
                    intersection_set=intersections[other],
                    save_dir=os.path.join(output_dir, "Venn-diags"),
                )

    # --- pH/area filtering ---
    logger.info("=== Filtering voltage plateaus ===")
    element_df = df_dict[element]

    # Split into ion-selective vs multi-ion
    other_ions = [ion for ion in frameworks_dict if ion != element]
    multi_ion_set = set()
    for other in other_ions:
        inter = framework_set_intersection(
            frameworks_dict[element][1], frameworks_dict[element][0],
            frameworks_dict[other][1], frameworks_dict[other][0],
        )
        multi_ion_set = multi_ion_set.union(inter)

    multi_ion_mpids = {fw.mp_id for fw in multi_ion_set}

    multi_mask = (
        element_df["Charged MP id"].isin(multi_ion_mpids)
        | element_df["Discharged MP id"].isin(multi_ion_mpids)
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

    # Export
    excel_path = os.path.join(output_dir, "pH_area_filtered_plateaus.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        filtered_only.to_excel(
            writer, sheet_name=f"only_{element}", index=False,
        )
        filtered_multi.to_excel(
            writer, sheet_name="multiple_ions", index=False,
        )
    logger.info("Filtered results saved to %s", excel_path)

    # Copy filtered plots
    if copy_filtered_plots and save_plots:
        _copy_filtered_plots(
            filtered_only, filtered_multi, element, output_dir,
        )

    return {
        f"only_{element}": filtered_only,
        "multiple_ions": filtered_multi,
    }


def _copy_filtered_plots(
    only_df, multi_df, element, output_dir,
):
    """Copy voltage plateau plots matching filtered plot numbers."""
    plots_dir = os.path.join(output_dir, "pH_area_filtered_plateaus")
    only_dir = os.path.join(plots_dir, f"only_{element}")
    multi_dir = os.path.join(plots_dir, "multiple_ions")
    os.makedirs(only_dir, exist_ok=True)
    os.makedirs(multi_dir, exist_ok=True)

    voltage_plots_dir = os.path.join(output_dir, "Voltage_plateaus")
    if not os.path.exists(voltage_plots_dir):
        return

    for target_dir, df in [(only_dir, only_df), (multi_dir, multi_df)]:
        if len(df) == 0:
            continue
        plot_nums = df["Plot number"].unique()
        for root, _, files in os.walk(voltage_plots_dir):
            for fname in files:
                if not fname.endswith(".png"):
                    continue
                for pn in plot_nums:
                    if f"plot{int(pn)}" in fname:
                        src = os.path.join(root, fname)
                        dst = os.path.join(target_dir, fname)
                        shutil.copy2(src, dst)

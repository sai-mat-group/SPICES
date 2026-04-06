"""
Visualization functions for Pourbaix diagrams and metastability maps.
"""

import numpy as np
import matplotlib.pyplot as plt
from pymatgen.analysis.pourbaix_diagram import PourbaixPlotter

from spices.pourbaix.decomposition import (
    get_decomposition_energy_from_mp_id,
    get_decomposition_energy_from_formula,
)


def get_metastability_heat_map_plot_by_mp_id(
    mp_id,
    pourbaix_diagram,
    ph_limits=(-2, 16),
    voltage_limits=(-4, 4),
    grid_size=18,
    decomp_energy_for_metastability=0.5,
    seawater_pbx=False,
    ax=None,
):
    """
    Plot a heat map of decomposition energy overlaid on the Pourbaix
    diagram for a specific MP ID.

    Parameters
    ----------
    mp_id : str
        Materials Project ID.
    pourbaix_diagram : PourbaixDiagram
    ph_limits : tuple
    voltage_limits : tuple
    grid_size : int
    decomp_energy_for_metastability : float
    seawater_pbx : bool
        If True, use seawater decomposition energy function.
    ax : matplotlib.axes.Axes or None
        Existing axes to plot on; creates new figure if None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    pH_range = np.linspace(ph_limits[0], ph_limits[1], grid_size)
    voltage_range = np.linspace(voltage_limits[0], voltage_limits[1], grid_size)
    energy_grid = np.zeros((grid_size, grid_size))

    for i, pH in enumerate(pH_range):
        for j, voltage in enumerate(voltage_range):
            energy_grid[j, i] = get_decomposition_energy_from_mp_id(
                mp_id, pH, voltage, pourbaix_diagram,
            )

    masked_grid = np.ma.masked_where(
        energy_grid >= decomp_energy_for_metastability, energy_grid,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(30, 24))
        fig.subplots_adjust(left=0, right=0.96, bottom=0, top=1)
    else:
        fig = ax.figure

    img = ax.imshow(
        masked_grid,
        extent=[pH_range.min(), pH_range.max(),
                voltage_range.min(), voltage_range.max()],
        origin="lower", cmap="coolwarm", alpha=0.5,
    )

    cbar_ax = fig.add_axes([0.98, 0.1, 0.05, 0.8])
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_label(f"{mp_id} Decomposition Energy (eV/atom)", fontsize=36)
    cbar.ax.tick_params(labelsize=35)

    plotter = PourbaixPlotter(pourbaix_diagram)
    plotter.get_pourbaix_plot(
        limits=[list(ph_limits), list(voltage_limits)],
        ax=ax, label_domains=True, show_neutral_axes=True,
    )

    ax.tick_params(axis="both", which="major", labelsize=35)
    ax.set_xlabel("pH", fontsize=40)
    ax.set_ylabel("E(V)", fontsize=40)

    return ax


def get_metastability_heat_map_plot(
    formula_string,
    pourbaix_diagram,
    ph_limits=(-2, 16),
    voltage_limits=(-4, 4),
    grid_size=18,
    decomp_energy_for_metastability=0.5,
    ax=None,
):
    """
    Plot a heat map of decomposition energy overlaid on the Pourbaix
    diagram for a formula string.

    Parameters
    ----------
    formula_string : str
    pourbaix_diagram : PourbaixDiagram
    ph_limits : tuple
    voltage_limits : tuple
    grid_size : int
    decomp_energy_for_metastability : float
    ax : matplotlib.axes.Axes or None

    Returns
    -------
    matplotlib.axes.Axes
    """
    pH_range = np.linspace(ph_limits[0], ph_limits[1], grid_size)
    voltage_range = np.linspace(voltage_limits[0], voltage_limits[1], grid_size)
    energy_grid = np.zeros((grid_size, grid_size))

    for i, pH in enumerate(pH_range):
        for j, voltage in enumerate(voltage_range):
            energy_grid[j, i] = get_decomposition_energy_from_formula(
                formula_string, pH, voltage, pourbaix_diagram,
            )

    masked_grid = np.ma.masked_where(
        energy_grid >= decomp_energy_for_metastability, energy_grid,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(30, 24))
        fig.subplots_adjust(left=0, right=0.96, bottom=0, top=1)
    else:
        fig = ax.figure

    img = ax.imshow(
        masked_grid,
        extent=[pH_range.min(), pH_range.max(),
                voltage_range.min(), voltage_range.max()],
        origin="lower", cmap="coolwarm", alpha=0.5,
    )

    cbar_ax = fig.add_axes([0.98, 0.1, 0.05, 0.8])
    cbar = fig.colorbar(img, cax=cbar_ax)
    cbar.set_label(
        f"{formula_string} Decomposition Energy (eV/atom)", fontsize=36,
    )
    cbar.ax.tick_params(labelsize=35)

    plotter = PourbaixPlotter(pourbaix_diagram)
    plotter.get_pourbaix_plot(
        limits=[list(ph_limits), list(voltage_limits)],
        ax=ax, label_domains=True, show_neutral_axes=True,
    )

    ax.tick_params(axis="both", which="major", labelsize=35)
    ax.set_xlabel("pH", fontsize=40)
    ax.set_ylabel("E(V)", fontsize=40)

    return ax


def get_phase_area_plot(formula_string, charge, pourbaix_diagram, ax=None):
    """
    Plot the Pourbaix stability region(s) of a given formula/charge,
    showing overlap with the water stability lines.

    Parameters
    ----------
    formula_string : str
    charge : int
    pourbaix_diagram : PourbaixDiagram
    ax : matplotlib.axes.Axes or None

    Returns
    -------
    matplotlib.axes.Axes
    """
    from pymatgen.core import Composition
    from pymatgen.analysis.pourbaix_diagram import PourbaixEntry, MultiEntry
    from shapely.geometry import Polygon, LineString
    from shapely.ops import unary_union
    from spices.config import H_LINE, O_LINE

    formula = Composition(formula_string).reduced_formula

    if ax is None:
        fig, ax = plt.subplots()

    for entry, vertices in pourbaix_diagram._stable_domain_vertices.items():
        matched = False
        if isinstance(entry, PourbaixEntry) and not isinstance(entry, MultiEntry):
            if (entry.composition.reduced_formula == formula
                    and entry.charge == int(charge)):
                matched = True
        else:
            formulas = [e.composition.reduced_formula for e in entry.entry_list]
            charges = [e.charge for e in entry.entry_list]
            if formula in formulas:
                idx = formulas.index(formula)
                if int(charges[idx]) == int(charge):
                    matched = True

        if matched:
            coords = [tuple(v) for v in vertices]
            polygon = Polygon(coords)
            x, y = polygon.exterior.coords.xy
            ax.plot(x, y, color="blue")

            # Water stability lines
            a1, b1, c1 = H_LINE
            a2, b2, c2 = O_LINE
            x_min, x_max = -2, 16
            ax.plot(
                [x_min, x_max],
                [-(a1 * x_min + c1) / b1, -(a1 * x_max + c1) / b1],
                linestyle="--", color="red",
            )
            ax.plot(
                [x_min, x_max],
                [-(a2 * x_min + c2) / b2, -(a2 * x_max + c2) / b2],
                linestyle="--", color="green",
            )
            ax.scatter(x, y, color="black", marker="o")

    ax.set_xlabel("pH")
    ax.set_ylabel("V (SHE)")
    return ax


class CustomPourbaixPlotter(PourbaixPlotter):
    """
    Extended Pourbaix plotter with seawater stability window and
    customizable domain coloring.

    Parameters
    ----------
    pourbaix_diagram : PourbaixDiagram
    """

    def get_pourbaix_plot(
        self,
        limits=None,
        title="",
        label_domains=True,
        label_fontsize=20,
        show_water_lines=True,
        show_neutral_axes=True,
        ax=None,
        highlight_entry=None,
        domain_color="#326771",
        highlight_color="skyblue",
        domain_boundary_color="black",
        water_line_color="#931621",
        domain_boundary_width=2,
        spine_width=2,
        separate_colours_for_ions_and_solids=False,
        solid_colour="#326771",
        ion_colour="#dce2c8",
        show_seawater_lines=False,
        seawater_lines_colour="blue",
        show_seawater_stability_window=False,
    ):
        """
        Plot Pourbaix diagram with extensive customization.

        Parameters
        ----------
        limits : tuple or None
            ``((xlo, xhi), (ylo, yhi))``. Default ``((0, 14), (-3, 3))``.
        title : str
        label_domains : bool
        label_fontsize : int
        show_water_lines : bool
        show_neutral_axes : bool
        ax : Axes or None
        highlight_entry : PourbaixEntry or None
        domain_color : str
        highlight_color : str
        domain_boundary_color : str
        water_line_color : str
        domain_boundary_width : int
        spine_width : int
        separate_colours_for_ions_and_solids : bool
        solid_colour : str
        ion_colour : str
        show_seawater_lines : bool
        seawater_lines_colour : str
        show_seawater_stability_window : bool
            If True (requires ``show_seawater_lines=True``), shade the
            seawater stability window (pH 7-8.5) with hatching.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from pymatgen.analysis.pourbaix_diagram import PREFAC
        from pymatgen.util.plotting import pretty_plot

        if limits is None:
            limits = ((0, 14), (-3, 3))

        ax = ax or pretty_plot(8, dpi=300)
        xlim, ylim = limits
        lw = domain_boundary_width

        # Water stability lines
        if show_water_lines:
            h_line = np.transpose([
                [xlim[0], -xlim[0] * PREFAC],
                [xlim[1], -xlim[1] * PREFAC],
            ])
            o_line = np.transpose([
                [xlim[0], -xlim[0] * PREFAC + 1.23],
                [xlim[1], -xlim[1] * PREFAC + 1.23],
            ])
            ax.plot(h_line[0], h_line[1], water_line_color,
                    linewidth=lw, linestyle="--")
            ax.plot(o_line[0], o_line[1], water_line_color,
                    linewidth=lw, linestyle="--")

        # Seawater stability lines
        if show_seawater_lines:
            h_shift, o_shift = -0.3, 0.5
            h_sea = np.transpose([
                [xlim[0], -xlim[0] * PREFAC + h_shift],
                [xlim[1], -xlim[1] * PREFAC + h_shift],
            ])
            o_sea = np.transpose([
                [xlim[0], -xlim[0] * PREFAC + 1.23 + o_shift],
                [xlim[1], -xlim[1] * PREFAC + 1.23 + o_shift],
            ])
            ax.plot(h_sea[0], h_sea[1], seawater_lines_colour,
                    linewidth=lw, linestyle="--")
            ax.plot(o_sea[0], o_sea[1], seawater_lines_colour,
                    linewidth=lw, linestyle="--")

        # Neutral axes
        if show_neutral_axes:
            ax.plot([7, 7], [ylim[0], ylim[1]], "k-.", linewidth=lw)
            ax.plot([xlim[0], xlim[1]], [0, 0], "k-.", linewidth=lw)

        # Domain boundaries and fills
        for entry, vertices in self._pbx._stable_domain_vertices.items():
            center = np.mean(vertices, axis=0)
            x, y = np.transpose(np.vstack([vertices, vertices[0]]))
            ax.plot(x, y, domain_boundary_color, linewidth=lw)

            # Domain coloring
            fill_color = domain_color
            if separate_colours_for_ions_and_solids:
                try:
                    phases = entry.entry_list
                    if any(p.phase_type == "Solid" for p in phases):
                        fill_color = solid_colour
                    else:
                        fill_color = ion_colour
                except AttributeError:
                    if entry.phase_type == "Solid":
                        fill_color = solid_colour
                    else:
                        fill_color = ion_colour

            if highlight_entry is not None and entry == highlight_entry:
                fill_color = highlight_color

            ax.fill(x, y, color=fill_color, alpha=0.5,
                    edgecolor=domain_boundary_color, linewidth=lw)

            if label_domains:
                ax.annotate(
                    self._generate_entry_label(entry), center,
                    ha="center", va="center", fontsize=label_fontsize,
                    color="black",
                ).draggable()

        ax.set_title(title, fontsize=20, fontweight="bold")
        label_size = 20
        ax.set_xlabel("pH", fontsize=label_size)
        ax.set_ylabel("Applied Potential (V vs SHE)", fontsize=label_size)
        ax.set(xlim=xlim, ylim=ylim)

        ax.tick_params(axis="both", direction="in", length=6, width=1.5,
                       which="both", top=False, right=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(spine_width)

        # Seawater stability window shading
        if show_seawater_stability_window and show_seawater_lines:
            x1, x2 = 7, 8.5
            h_y1 = -x1 * PREFAC - 0.3
            h_y2 = -x2 * PREFAC - 0.3
            o_y1 = -x1 * PREFAC + 1.23 + 0.5
            o_y2 = -x2 * PREFAC + 1.23 + 0.5

            ax.axvline(x=x1, color=seawater_lines_colour,
                       linestyle="--", linewidth=lw)
            ax.axvline(x=x2, color=seawater_lines_colour,
                       linestyle="--", linewidth=lw)

            ax.fill_between(
                [x1, x2], [h_y1, h_y2], [o_y1, o_y2],
                facecolor=seawater_lines_colour, alpha=0.25,
                hatch="///", edgecolor="black", linewidth=1.0,
            )

        return ax

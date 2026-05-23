"""
SPICES plotting style — matplotlib rcParams for publication-quality figures.

Sans-serif (Helvetica), 6-7 pt in-figure text, fonts embedded in PDF/EPS.

Each plotting module in SPICES imports and calls
:func:`apply_publication_style` at module load. Calls are idempotent.

figsize convention (assume single-column 89 mm = 3.5 in, double-column
183 mm = 7.2 in): the modules render at publication figsize directly, so
matplotlib point sizes equal printed point sizes — no post-export scaling
math required.
"""

import matplotlib as mpl


_APPLIED = False


def apply_publication_style():
    """Set matplotlib rcParams for SPICES publication-quality figures.

    Idempotent: safe to call from multiple modules.
    """
    global _APPLIED
    if _APPLIED:
        return

    mpl.rcParams.update({
        # Font: Helvetica primary; system fallbacks (Helvetica Neue,
        # Gill Sans) per user setup at ~/.local/share/fonts/. Matplotlib
        # appends its built-in DejaVu Sans automatically when none of
        # these resolve.
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Helvetica Neue", "Gill Sans"],

        # Embed fonts in vector outputs (TrueType, not Type 3).
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Sizes: 7 pt for body/labels/title, 6 pt for ticks/legend.
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 7,

        # Ticks: inward, on all four sides.
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,

        # Lines and axes.
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,

        # DPI: low for interactive, high for saved files.
        "figure.dpi": 100,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })

    _APPLIED = True

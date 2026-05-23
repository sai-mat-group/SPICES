"""Tests for the shared publication-style matplotlib rcParams."""

import matplotlib as mpl

from spices.plotting import apply_publication_style


class TestPublicationStyle:
    def test_font_family(self):
        apply_publication_style()
        assert mpl.rcParams["font.family"] == ["sans-serif"]
        assert mpl.rcParams["font.sans-serif"][0] == "Helvetica"

    def test_pdf_font_embedding(self):
        apply_publication_style()
        assert mpl.rcParams["pdf.fonttype"] == 42
        assert mpl.rcParams["ps.fonttype"] == 42

    def test_ticks_inward(self):
        apply_publication_style()
        assert mpl.rcParams["xtick.direction"] == "in"
        assert mpl.rcParams["ytick.direction"] == "in"
        assert mpl.rcParams["xtick.top"] is True
        assert mpl.rcParams["ytick.right"] is True

    def test_font_sizes(self):
        apply_publication_style()
        assert mpl.rcParams["font.size"] == 7
        assert mpl.rcParams["axes.labelsize"] == 7
        assert mpl.rcParams["xtick.labelsize"] == 6
        assert mpl.rcParams["legend.fontsize"] == 6

    def test_idempotent(self):
        """Calling twice does not raise and leaves state unchanged."""
        apply_publication_style()
        snapshot = dict(mpl.rcParams)
        apply_publication_style()
        # Spot-check that critical values are still set
        assert mpl.rcParams["font.sans-serif"][0] == "Helvetica"
        assert mpl.rcParams["font.size"] == snapshot["font.size"]

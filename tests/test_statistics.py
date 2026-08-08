"""Tests for the Stage-5 statistics module."""

import json

import matplotlib
matplotlib.use("Agg")  # headless for the figure smoke tests

import numpy as np
import pandas as pd
import pytest

from spices.screening.statistics import (
    get_stability_bool,
    _get_pH_range_bool,
    _ph_range_covers,
    _resolve_stage2_excel,
    _water_stability_ref_lines,
    count_phase_stabilities,
    count_phases_with_voltage_stability,
    build_structure_voltage_data,
    plot_pipeline_bar_chart,
    plot_periodic_table,
    plot_structure_voltages_combined,
    run_statistics,
)


class TestGetStabilityBool:
    def test_string_list_nonempty(self):
        assert get_stability_bool("[(7.0, 0.5), (7.5, 0.8)]") is True

    def test_string_list_empty(self):
        assert get_stability_bool("[]") is False

    def test_actual_list_nonempty(self):
        assert get_stability_bool([(7.0, 0.5)]) is True

    def test_actual_list_empty(self):
        assert get_stability_bool([]) is False

    def test_nan(self):
        assert get_stability_bool(float("nan")) is False

    def test_none(self):
        assert get_stability_bool(None) is False

    def test_bool_true(self):
        assert get_stability_bool(True) is True

    def test_bool_false(self):
        assert get_stability_bool(False) is False

    def test_zero(self):
        assert get_stability_bool(0) is False

    def test_int_and_float_agree(self):
        # Fix #11: float 2.0 must behave like int 2.
        assert get_stability_bool(2) is True
        assert get_stability_bool(2.0) is True
        assert get_stability_bool(0.0) is False
        assert get_stability_bool(np.float64(3.0)) is True


class TestGetpHRangeBool:
    def test_full_range(self):
        assert _get_pH_range_bool("(7.0, 8.5)", 1.5) is True

    def test_partial_range(self):
        assert _get_pH_range_bool("(7.0, 8.0)", 1.5) is False

    def test_non_string(self):
        assert _get_pH_range_bool(float("nan"), 1.5) is False

    def test_exact_threshold(self):
        assert _get_pH_range_bool("(7.0, 8.5)", 1.5) is True


class TestpHRangeCovers:
    def test_full_coverage(self):
        assert _ph_range_covers("(7.0, 8.5)", 7, 8.5) is True

    def test_near_miss_rejected(self):
        # Fix #5: width 1.46 rounds to 1.5 but does NOT cover [7, 8.5].
        assert _ph_range_covers("(7.0, 8.46)", 7, 8.5) is False

    def test_wrong_placement_rejected(self):
        # Width 2.0 but placed entirely below the window.
        assert _ph_range_covers("(3.0, 5.0)", 7, 8.5) is False

    def test_tol_absorbs_jitter(self):
        assert _ph_range_covers("(7.01, 8.49)", 7, 8.5) is True

    def test_non_string(self):
        assert _ph_range_covers(float("nan"), 7, 8.5) is False


class TestResolveStage2Excel:
    def test_canonical_wins_over_decoys(self, tmp_path):
        ion_dir = tmp_path / "Li"
        ion_dir.mkdir()
        (ion_dir / "Li_cathodes_seawater_metastability.xlsx").write_text("x")
        (ion_dir / "Li_cathodes_seawater_metastability_old.xlsx").write_text("x")
        (ion_dir / "Li_cathodes_mp_data.xlsx").write_text("x")
        resolved = _resolve_stage2_excel(str(ion_dir), "Li")
        assert resolved.endswith("Li_cathodes_seawater_metastability.xlsx")

    def test_recursive_find_in_subdir(self, tmp_path):
        ion_dir = tmp_path / "K"
        (ion_dir / "run2").mkdir(parents=True)
        (ion_dir / "run2" / "K_cathodes_seawater_metastability.xlsx").write_text("x")
        (ion_dir / "run1").mkdir()
        (ion_dir / "run1" / "K_seawater_results.xlsx").write_text("x")
        resolved = _resolve_stage2_excel(str(ion_dir), "K")
        assert resolved.endswith("K_cathodes_seawater_metastability.xlsx")

    def test_single_xlsx_fallback_warns(self, tmp_path, caplog):
        ion_dir = tmp_path / "Na"
        ion_dir.mkdir()
        (ion_dir / "something_else.xlsx").write_text("x")
        resolved = _resolve_stage2_excel(str(ion_dir), "Na")
        assert resolved.endswith("something_else.xlsx")

    def test_multiple_non_canonical_raises(self, tmp_path):
        ion_dir = tmp_path / "Mg"
        ion_dir.mkdir()
        (ion_dir / "a.xlsx").write_text("x")
        (ion_dir / "b.xlsx").write_text("x")
        with pytest.raises(ValueError):
            _resolve_stage2_excel(str(ion_dir), "Mg")

    def test_no_xlsx_raises(self, tmp_path):
        ion_dir = tmp_path / "Ca"
        ion_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            _resolve_stage2_excel(str(ion_dir), "Ca")


class TestCountPhaseStabilities:
    def test_honors_criteria(self):
        df = pd.DataFrame({
            "Seawater stability": [True, False, False],
            "Seawater metastability (Criterion 5)": [
                "[(7.0, 0.1)]", "[]", "[(8.0, 0.2)]",
            ],
        })
        counts = count_phase_stabilities(df, criteria=[5])
        assert counts["Seawater stability"] == 1
        assert counts["Seawater metastability (Criterion 5) bool"] == 2

    def test_missing_criterion_counts_zero(self):
        df = pd.DataFrame({"Seawater stability": [True, True]})
        counts = count_phase_stabilities(df, criteria=[4])
        assert counts["Seawater metastability (Criterion 4) bool"] == 0


class TestCountPhasesWithVoltageStability:
    def _make_test_data(self):
        stability_df = pd.DataFrame({
            "MP id": ["mp-1", "mp-2", "mp-3", "mp-4"],
        })
        voltage_df = pd.DataFrame({
            "Charged MP id": ["mp-1", "mp-2"],
            "Discharged MP id": ["mp-2", "mp-3"],
            "Plot number": [0, 1],
            "pH range metastability criterion 4": [
                "(7.0, 8.5)", "(7.0, 8.0)",
            ],
            "Total metastability area fraction": [0.9, 0.5],
        })
        return voltage_df, stability_df

    def test_unfiltered_counts(self):
        vdf, sdf = self._make_test_data()
        result = count_phases_with_voltage_stability(vdf, sdf, 4)
        assert result["total_phases"] == 3
        assert result["total_plots"] == 2

    def test_pH_filter_coverage(self):
        vdf, sdf = self._make_test_data()
        result = count_phases_with_voltage_stability(
            vdf, sdf, 4, pH_window=(7.0, 8.5),
        )
        # Only the first row covers [7, 8.5].
        assert result["total_phases"] == 2  # mp-1 and mp-2
        assert result["total_plots"] == 1

    def test_area_filter(self):
        vdf, sdf = self._make_test_data()
        result = count_phases_with_voltage_stability(
            vdf, sdf, 4, min_area_fraction=0.8,
        )
        assert result["total_plots"] == 1

    def test_pH_window_outside_seawater_raises(self):
        vdf, sdf = self._make_test_data()
        with pytest.raises(ValueError):
            count_phases_with_voltage_stability(
                vdf, sdf, 4, pH_window=(5.0, 8.5),
            )

    def test_experimental_filter_missing_column_raises(self):
        vdf, sdf = self._make_test_data()
        with pytest.raises(ValueError):
            count_phases_with_voltage_stability(
                vdf, sdf, 4, experimental=True,
            )


class TestWaterStabilityRefLines:
    def test_default_matches_intent(self):
        o_line, h_line = _water_stability_ref_lines()
        # At ref_pH=7.75 with OER/HER shifts -> ~ (1.272, -0.758).
        assert o_line == pytest.approx(1.272, abs=1e-2)
        assert h_line == pytest.approx(-0.758, abs=1e-2)

    def test_no_shifts(self):
        o_line, h_line = _water_stability_ref_lines(ref_pH=0.0, apply_shifts=False)
        assert o_line == pytest.approx(1.23)
        assert h_line == pytest.approx(0.0)


def _marker(ion, V, status="solid", type_="Theoretical", area=0.9,
            covers_pH=True, charged="mp-c", discharged="mp-d"):
    """Build one marker dict matching the records schema."""
    return {
        "ion": ion, "V": V, "type": type_, "status": status,
        "area": area, "covers_pH": covers_pH,
        "charged_mpid": charged, "discharged_mpid": discharged,
    }


def _ion_comp(component_id, route, n_ions, label_key, markers):
    """Build one component record; label_key is ``"formula\\nmp-id"``."""
    formula, mpid = label_key.split("\n")
    return {
        "component_id": component_id, "route": route, "n_ions": n_ions,
        "label_formula": formula, "label_mpid": mpid, "label_key": label_key,
        "markers": markers,
    }


def _write_records(tmp_path, components, **header):
    """Write a synthetic structure_voltage_records.json; return its path."""
    rec = {
        "schema_version": 1, "pH_filter": [7.0, 8.5],
        "metastability_criterion": 4, "qualify_area": 0.70, "hatch_area": 0.80,
        "components": components,
    }
    rec.update(header)
    path = tmp_path / "structure_voltage_records.json"
    with open(path, "w") as fh:
        json.dump(rec, fh)
    return str(path)


class TestBuildStructureVoltageData:
    def test_ion_selective_from_records(self, tmp_path):
        comps = [_ion_comp(
            0, "ion_selective", 1, "FePO4\nmp-1",
            [_marker("Li", 0.3, "solid", "Experimental")],
        )]
        path = _write_records(tmp_path, comps)
        data = build_structure_voltage_data(path, "ion_selective")
        v, typ, style = data["Li"]["FePO4\nmp-1"][0]
        assert v == pytest.approx(0.3)
        assert typ == "Experimental"
        assert style is None  # "solid" -> None

    def test_accepts_loaded_dict_and_hashed_style(self, tmp_path):
        comps = [_ion_comp(
            0, "ion_selective", 1, "FePO4\nmp-1",
            [_marker("Li", 0.3, "hashed", area=0.75)],
        )]
        path = _write_records(tmp_path, comps)
        with open(path) as fh:
            # Disable the anchor filter: this test asserts hashed-style
            # passthrough, not the filter behaviour.
            data = build_structure_voltage_data(
                json.load(fh), "ion_selective", anchor_filter=None)
        assert data["Li"]["FePO4\nmp-1"][0][2] == "hashed"

    def test_experimental_only_filter(self, tmp_path):
        comps = [
            _ion_comp(0, "ion_selective", 1, "FePO4\nmp-1",
                      [_marker("Li", 0.3, "solid", "Experimental")]),
            _ion_comp(1, "ion_selective", 1, "TiO2\nmp-2",
                      [_marker("Li", 0.1, "solid", "Theoretical")]),
        ]
        path = _write_records(tmp_path, comps)
        data = build_structure_voltage_data(path, "ion_selective")
        assert set(data["Li"]) == {"FePO4\nmp-1", "TiO2\nmp-2"}
        data = build_structure_voltage_data(
            path, "ion_selective", experimental_only_ions=("Li",))
        assert set(data["Li"]) == {"FePO4\nmp-1"}

    def test_ion_selective_order(self, tmp_path):
        comps = [
            _ion_comp(0, "ion_selective", 1, "ZnO\nmp-9", [_marker("Na", 0.2)]),
            _ion_comp(1, "ion_selective", 1, "BO\nmp-1", [_marker("Li", 0.3)]),
            _ion_comp(2, "ion_selective", 1, "AO\nmp-0", [_marker("Li", 0.4)]),
        ]
        path = _write_records(tmp_path, comps)
        data = build_structure_voltage_data(path, "ion_selective")
        assert list(data.keys()) == ["Li", "Na"]  # ION_COLORS order
        assert list(data["Li"].keys()) == ["AO\nmp-0", "BO\nmp-1"]  # by label

    def test_top_n_by_area(self, tmp_path):
        comps = [
            _ion_comp(i, "ion_selective", 1, f"F{i}\nmp-{i}",
                      [_marker("Li", 0.3, area=a)])
            for i, a in enumerate([0.75, 0.95, 0.85])
        ]
        path = _write_records(tmp_path, comps)
        data = build_structure_voltage_data(path, "ion_selective", top_n=2)
        assert set(data["Li"]) == {"F1\nmp-1", "F2\nmp-2"}

    def test_multi_ion_routing_and_styles(self, tmp_path):
        comps = [_ion_comp(
            0, "multi_ion", 2, "FePO4\nmp-1",
            [_marker("Li", 0.35, "hashed", "Experimental", area=0.75),
             _marker("Na", 0.30, "solid", "Experimental", area=0.95)],
        )]
        path = _write_records(tmp_path, comps)
        data, n_ions = build_structure_voltage_data(path, "multi_ion")
        key = "FePO4\nmp-1"
        assert n_ions[key] == 2
        assert set(data[key]) == {"Li", "Na"}
        assert data[key]["Li"][0][2] == "hashed"
        assert data[key]["Na"][0][2] is None

    def test_multi_ion_faded_modes(self, tmp_path):
        # solid+Experimental Li anchor keeps the component under the default
        # filter; the test then exercises faded_mode "all" vs "ph_covering".
        comps = [_ion_comp(
            0, "multi_ion", 2, "FePO4\nmp-1",
            [_marker("Li", 0.35, "solid", "Experimental",
                     area=0.95, covers_pH=True),
             _marker("Na", 0.30, "faded", area=0.5, covers_pH=True,
                     charged="mp-3", discharged="mp-4"),
             _marker("Na", 0.10, "faded", area=0.5, covers_pH=False,
                     charged="mp-5", discharged="mp-6")],
        )]
        path = _write_records(tmp_path, comps)
        key = "FePO4\nmp-1"
        data, _ = build_structure_voltage_data(path, "multi_ion",
                                               faded_mode="all")
        assert len(data[key]["Na"]) == 2
        assert all(e[2] == "faded" for e in data[key]["Na"])
        data, _ = build_structure_voltage_data(path, "multi_ion",
                                               faded_mode="ph_covering")
        assert len(data[key]["Na"]) == 1
        assert data[key]["Na"][0][0] == pytest.approx(0.30)

    def test_multi_ion_single_visible_marker_warns(self, tmp_path, caplog):
        # solid+Experimental anchor keeps the component; the Na faded marker
        # is then hidden by ph_covering, leaving Li as the lone visible ion.
        comps = [_ion_comp(
            0, "multi_ion", 2, "FePO4\nmp-1",
            [_marker("Li", 0.35, "solid", "Experimental",
                     area=0.95, covers_pH=True),
             _marker("Na", 0.10, "faded", area=0.5, covers_pH=False)],
        )]
        path = _write_records(tmp_path, comps)
        with caplog.at_level("WARNING"):
            data, _ = build_structure_voltage_data(
                path, "multi_ion", faded_mode="ph_covering")
        assert set(data["FePO4\nmp-1"]) == {"Li"}
        assert any("visible marker" in r.message for r in caplog.records)

    def test_mode_filters_components(self, tmp_path):
        comps = [
            _ion_comp(0, "ion_selective", 1, "A\nmp-1", [_marker("Li", 0.3)]),
            _ion_comp(1, "multi_ion", 2, "B\nmp-2",
                      [_marker("Li", 0.3), _marker("Na", 0.2)]),
        ]
        path = _write_records(tmp_path, comps)
        ion_data = build_structure_voltage_data(path, "ion_selective")
        assert set(ion_data.get("Li", {})) == {"A\nmp-1"}
        multi_data, _ = build_structure_voltage_data(path, "multi_ion")
        assert set(multi_data) == {"B\nmp-2"}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_structure_voltage_data(
                str(tmp_path / "nope.json"), "ion_selective")

    def test_schema_version_mismatch_raises(self, tmp_path):
        path = _write_records(tmp_path, [], schema_version=2)
        with pytest.raises(ValueError):
            build_structure_voltage_data(path, "ion_selective")

    def test_unknown_mode_raises(self, tmp_path):
        path = _write_records(tmp_path, [])
        with pytest.raises(ValueError):
            build_structure_voltage_data(path, "bogus")

    def test_anchor_filter_drops_hashed_only(self, tmp_path):
        # A component whose only marker is hashed (no solid marker at all)
        # is dropped under the default "solid" filter and kept when the
        # filter is disabled.
        comps = [_ion_comp(
            0, "ion_selective", 1, "FePO4\nmp-1",
            [_marker("Li", 0.3, "hashed", "Theoretical", area=0.75)],
        )]
        path = _write_records(tmp_path, comps)
        data = build_structure_voltage_data(path, "ion_selective")
        assert "Li" not in data
        data = build_structure_voltage_data(
            path, "ion_selective", anchor_filter=None)
        assert "FePO4\nmp-1" in data["Li"]

    def test_anchor_filter_keeps_pure_solid(self, tmp_path):
        # A component with only solid markers is kept under both anchor
        # modes, regardless of Experimental vs Theoretical... except that
        # "experimental" also fires when a weak marker is present (covered
        # in the tests below).
        comps = [
            _ion_comp(0, "ion_selective", 1, "FePO4\nmp-1",
                      [_marker("Li", 0.3, "solid", "Experimental")]),
            _ion_comp(1, "ion_selective", 1, "TiO2\nmp-2",
                      [_marker("Li", 0.1, "solid", "Theoretical")]),
        ]
        path = _write_records(tmp_path, comps)
        for mode in ("solid", "experimental"):
            data = build_structure_voltage_data(
                path, "ion_selective", anchor_filter=mode)
            assert set(data["Li"]) == {"FePO4\nmp-1", "TiO2\nmp-2"}

    def test_anchor_filter_solid_keeps_faded_partner(self, tmp_path):
        # The mp-540111 / mp-774537 shape: a solid THEORETICAL plateau plus
        # a faded partner. The partner's failure is context, not host
        # weakness, so the default "solid" filter keeps the column; the
        # historical "experimental" rule drops it (no Experimental anchor).
        comps = [_ion_comp(
            0, "multi_ion", 2, "FePO4\nmp-1",
            [_marker("Li", 0.35, "solid", "Theoretical", area=0.95),
             _marker("Na", 0.30, "faded", "Theoretical", area=0.5,
                     covers_pH=True)],
        )]
        path = _write_records(tmp_path, comps)
        data, _ = build_structure_voltage_data(path, "multi_ion")
        assert "FePO4\nmp-1" in data
        data, _ = build_structure_voltage_data(
            path, "multi_ion", anchor_filter="experimental")
        assert data == {}

    def test_anchor_filter_solid_keeps_solid_plus_hashed(self, tmp_path):
        # The MnFe(PO4)2 shape: the host's own solid plateau plus its own
        # borderline hashed one. "solid" keeps it; "experimental" drops it
        # because the hashed marker fires without an Experimental anchor.
        comps = [_ion_comp(
            0, "multi_ion", 2, "MnFe(PO4)2\nmp-1",
            [_marker("Mg", 0.2, "solid", "Theoretical", area=0.82),
             _marker("Li", 0.3, "hashed", "Theoretical", area=0.79)],
        )]
        path = _write_records(tmp_path, comps)
        data, _ = build_structure_voltage_data(path, "multi_ion")
        assert "MnFe(PO4)2\nmp-1" in data
        data, _ = build_structure_voltage_data(
            path, "multi_ion", anchor_filter="experimental")
        assert data == {}

    def test_anchor_filter_solid_drops_weak_only_multi_ion(self, tmp_path):
        # hashed + faded with no solid marker anywhere: dropped by default.
        comps = [_ion_comp(
            0, "multi_ion", 2, "TiO2\nmp-1",
            [_marker("Li", 0.3, "hashed", "Experimental", area=0.75),
             _marker("Mg", 0.1, "faded", "Theoretical", area=0.4,
                     covers_pH=False)],
        )]
        path = _write_records(tmp_path, comps)
        data, _ = build_structure_voltage_data(path, "multi_ion")
        assert data == {}
        data, _ = build_structure_voltage_data(
            path, "multi_ion", anchor_filter=None)
        assert "TiO2\nmp-1" in data

    def test_anchor_filter_experimental_keeps_anchored(self, tmp_path):
        # Historical rule preserved: solid+Experimental anchors a component
        # that also carries a faded partner.
        comps = [_ion_comp(
            0, "multi_ion", 2, "FePO4\nmp-1",
            [_marker("Li", 0.35, "solid", "Experimental", area=0.95),
             _marker("Na", 0.30, "faded", "Theoretical", area=0.5,
                     covers_pH=True)],
        )]
        path = _write_records(tmp_path, comps)
        data, _ = build_structure_voltage_data(
            path, "multi_ion", anchor_filter="experimental")
        assert "FePO4\nmp-1" in data

    def test_anchor_filter_monotone_under_new_partner(self, tmp_path):
        # Regression for the non-monotonicity flaw: the same host must be
        # kept both as a 1-ion ion_selective component and after Stage 4
        # re-routes it multi_ion with an added faded partner.
        alone = [_ion_comp(
            0, "ion_selective", 1, "Ti2(PO4)3\nmp-1",
            [_marker("Mg", 0.23, "solid", "Theoretical", area=0.995)],
        )]
        path = _write_records(tmp_path, alone)
        data = build_structure_voltage_data(path, "ion_selective")
        assert "Ti2(PO4)3\nmp-1" in data["Mg"]

        with_partner = [_ion_comp(
            0, "multi_ion", 2, "Ti2(PO4)3\nmp-1",
            [_marker("Mg", 0.23, "solid", "Theoretical", area=0.995),
             _marker("Li", -0.4, "faded", "Theoretical", area=0.39,
                     covers_pH=True)],
        )]
        path = _write_records(tmp_path, with_partner)
        data, _ = build_structure_voltage_data(path, "multi_ion")
        assert "Ti2(PO4)3\nmp-1" in data

    def test_anchor_filter_unknown_value_raises(self, tmp_path):
        path = _write_records(tmp_path, [])
        with pytest.raises(ValueError, match="anchor_filter"):
            build_structure_voltage_data(
                path, "ion_selective", anchor_filter="bogus")


class TestFigureSmoke:
    def test_periodic_table(self, tmp_path):
        out = tmp_path / "pt.pdf"
        fig = plot_periodic_table(save_path=str(out))
        assert out.exists()
        assert fig is not None

    @pytest.mark.parametrize("color_by", ["stage", "ion"])
    def test_bar_chart(self, tmp_path, color_by):
        # Three stages (Total / Metastable / Stable); Na has 0 stable phases
        # to exercise the skip-zero-height path on the log axis.
        stage_data = {
            "Li": {
                "Total electrodes": 1000,
                "Metastable in seawater": 500,
                "Stable in seawater": 6,
            },
            "Na": {
                "Total electrodes": 800,
                "Metastable in seawater": 400,
                "Stable in seawater": 0,
            },
        }
        out = tmp_path / f"bar_{color_by}.pdf"  # vector output
        fig = plot_pipeline_bar_chart(stage_data, save_path=str(out), color_by=color_by)
        assert out.exists()
        assert fig is not None
        legend = fig.axes[0].get_legend()
        assert legend is not None
        assert {t.get_text() for t in legend.get_texts()} == {
            "Total electrodes",
            "Metastable in seawater",
            "Stable in seawater",
        }

    def test_structure_voltages_combined(self, tmp_path):
        # Exercises both panels in one shot: ion-selective (top) with a
        # hashed overlay + ION_COLORS x-ordering, multi-ion (bottom) with
        # the n_ions grouping (3-ion host listed first to confirm it is
        # re-ordered after the 2-ion host), a faded partner marker, and
        # the combined right-side legend + mp-id x-ticks. The default
        # anchor_filter="solid" keeps every component here since each one
        # carries at least one solid marker.
        comps = [
            _ion_comp(0, "ion_selective", 1, "FePO4\nmp-1",
                      [_marker("Li", 0.3, "solid", "Experimental",
                               area=0.95)]),
            _ion_comp(1, "ion_selective", 1, "TiO2\nmp-3",
                      [_marker("Li", 0.1, "solid", "Experimental",
                               area=0.95),
                       _marker("Li", 0.05, "hashed", "Theoretical",
                               area=0.75)]),
            _ion_comp(2, "ion_selective", 1, "MnO2\nmp-2",
                      [_marker("Na", -0.2, "solid", "Experimental",
                               area=0.92),
                       _marker("Na", -0.5, "solid", "Theoretical",
                               area=0.95)]),
            _ion_comp(3, "multi_ion", 3, "MnO2\nmp-9",
                      [_marker("Li", 0.5, "solid", "Experimental",
                               area=0.95),
                       _marker("Na", 0.4, "solid", "Theoretical", area=0.9),
                       _marker("K", 0.2, "faded", "Theoretical",
                               area=0.5, covers_pH=True)]),
            _ion_comp(4, "multi_ion", 2, "FePO4\nmp-1",
                      [_marker("Li", 0.35, "solid", "Experimental",
                               area=0.95),
                       _marker("Na", 0.30, "hashed", "Theoretical",
                               area=0.75)]),
        ]
        path = _write_records(tmp_path, comps)
        out = tmp_path / "sv_combined.pdf"
        fig = plot_structure_voltages_combined(path, save_path=str(out))
        assert out.exists()
        assert fig is not None
        # Restyle: two full-width panels (no separate right-hand legend axis)
        # and a frameless top legend on the upper panel.
        assert len(fig.axes) == 2
        ax_top = fig.axes[0]
        assert ax_top.get_legend() is not None
        # One nude-gradient axvspan band per single-ion column: the only
        # difference in top-panel patches between gradient on/off (3 columns).
        plain = plot_structure_voltages_combined(path, gradient_background=False)
        assert len(ax_top.patches) - len(plain.axes[0].patches) == 3


class TestRunStatisticsStructureVoltage:
    """End-to-end wiring of the records JSON -> the combined PDF."""

    def test_writes_combined_figure_from_records(self, tmp_path):
        # The multi-ion component carries a solid marker, so the default
        # anchor_filter="solid" keeps it (faded partner included).
        comps = [
            _ion_comp(0, "ion_selective", 1, "FePO4\nmp-1",
                      [_marker("Li", 0.3, "solid", "Experimental")]),
            _ion_comp(1, "multi_ion", 2, "MnO2\nmp-2",
                      [_marker("Li", 0.4, "solid", "Experimental", area=0.95),
                       _marker("Na", 0.2, "faded", "Theoretical", area=0.5,
                               covers_pH=True)]),
        ]
        records_path = _write_records(tmp_path, comps)
        out = tmp_path / "stats_out"
        run_statistics(
            stage2_dirs={}, stage3_dirs={}, output_dir=str(out),
            stage4_records=records_path, make_periodic_table=False,
        )
        assert (out / "structure_voltages.pdf").exists()
        # The legacy per-panel PNGs are gone.
        assert not (out / "structure_voltages_ion_selective.png").exists()
        assert not (out / "structure_voltages_multi_ion.png").exists()

    def test_missing_records_skips_without_error(self, tmp_path, caplog):
        out = tmp_path / "stats_out2"
        with caplog.at_level("WARNING"):
            run_statistics(
                stage2_dirs={}, stage3_dirs={}, output_dir=str(out),
                stage4_records=str(tmp_path / "nope.json"),
                make_periodic_table=False,
            )
        assert not (out / "structure_voltages.pdf").exists()
        assert any("structure_voltage_records.json" in r.message
                   for r in caplog.records)

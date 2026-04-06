"""Tests for spices.pourbaix.composition."""

from pymatgen.core import Composition

from spices.pourbaix.composition import (
    rm_element,
    make_chemsys_from_combinations,
    get_pourbaix_composition,
    get_sea_pourbaix_comp_dict,
    get_sea_pourbaix_conc_dict,
)


class TestRmElement:
    def test_remove_li_from_lifepo4(self):
        result = rm_element("LiFePO4", "Li")
        assert Composition(result).reduced_formula == "FePO4"

    def test_remove_ca_from_catio3(self):
        result = rm_element("CaTiO3", "Ca")
        assert Composition(result).reduced_formula == "TiO3"

    def test_remove_na_from_na2mno3(self):
        result = rm_element("Na2MnO3", "Na")
        assert Composition(result).reduced_formula == "MnO3"

    def test_remove_absent_element(self):
        result = rm_element("FePO4", "Li")
        assert Composition(result).reduced_formula == "FePO4"

    def test_remove_mg(self):
        result = rm_element("MgFe2O4", "Mg")
        assert Composition(result).reduced_formula == "FeO2"


class TestMakeChemsys:
    def test_single_tm_single_anion(self):
        result = make_chemsys_from_combinations(("Ti",), ("O",), "Ca")
        assert result == "Ca-Ti-O"

    def test_multi_tm_multi_anion(self):
        result = make_chemsys_from_combinations(
            ("Ti", "Fe"), ("O", "P"), "Li"
        )
        assert result == "Li-Ti-Fe-O-P"

    def test_all_elements(self):
        result = make_chemsys_from_combinations(
            ("V", "Cr", "Mn"), ("S", "Se", "F"), "Na"
        )
        assert result == "Na-V-Cr-Mn-S-Se-F"


class TestGetPourbaixComposition:
    def test_removes_oxygen(self):
        comp = get_pourbaix_composition(Composition("LiFePO4"))
        assert "O" not in comp.as_dict()
        assert "Li" in comp.as_dict()
        assert "Fe" in comp.as_dict()
        assert "P" in comp.as_dict()

    def test_removes_hydrogen(self):
        comp = get_pourbaix_composition(Composition("LiOH"))
        assert "O" not in comp.as_dict()
        assert "H" not in comp.as_dict()
        assert "Li" in comp.as_dict()

    def test_no_oh_present(self):
        comp = get_pourbaix_composition(Composition("LiFeS2"))
        assert comp.as_dict() == Composition("LiFeS2").as_dict()


class TestSeaPourbaixDicts:
    def test_comp_dict_basic(self):
        result = get_sea_pourbaix_comp_dict(Composition("LiFePO4"), num_of_sea_elements=4)
        # Should have Li, Fe, P from material + Cl, Na, Mg, S from seawater
        assert "Cl" in result
        assert "Na" in result
        assert "Mg" in result
        assert "S" in result
        assert "Li" in result
        assert "Fe" in result

    def test_conc_dict_basic(self):
        result = get_sea_pourbaix_conc_dict(Composition("LiFePO4"), num_of_sea_elements=4)
        # Seawater elements should have seawater concentrations
        assert result["Cl"] > 0.5  # ~0.56 M
        assert result["Na"] > 0.4  # ~0.48 M
        # Non-seawater elements should have dilute limit
        assert result["Fe"] == 1e-6
        assert result["P"] == 1e-6

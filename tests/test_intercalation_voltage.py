"""Tests for intercalation voltage calculation."""

import pytest

from spices.intercalation.voltage import calculate_intercalation_voltage


class TestCalculateIntercalationVoltage:
    """Test the Nernst-based voltage calculation."""

    MOCK_ENERGIES = {"GGA": -2.0, "R2SCAN": -7.0}

    def test_basic_voltage(self):
        """Simple case: CaX → X with 1 Ca removed."""
        # CaFe2O4 (discharged) → Fe2O4 (charged)
        # Framework element Fe: 2 in both → LCM = 2
        # delta_Ca = 1 (1 Ca in discharged, 0 in charged)
        # V = -(E_dis - E_char - 1 * E_Ca) / (2 * 1)
        V = calculate_intercalation_voltage(
            discharged_formula="CaFe2O4",
            discharged_energy_per_formula=-30.0,
            charged_formula="Fe2O4",
            charged_energy_per_formula=-25.0,
            functional_string="GGA",
            working_ion="Ca",
            working_ion_valency=2,
            working_ion_energies=self.MOCK_ENERGIES,
        )
        # V = -(-30 - (-25) - 1*(-2)) / (2 * 1) = -(−30 + 25 + 2) / 2 = -(-3)/2 = 1.5
        assert abs(V - 1.5) < 1e-10

    def test_lcm_normalization(self):
        """When framework element counts differ, LCM normalizes them."""
        # Ca2Fe4O8 (discharged, Fe=4) → Fe2O4 (charged, Fe=2)
        # LCM of Fe: lcm(4, 2) = 4
        # E_dis_normalized = -60 * 4/4 = -60
        # E_char_normalized = -25 * 4/2 = -50
        # delta_Ca = 2 * 4/4 - 0 = 2
        # V = -(-60 - (-50) - 2*(-2)) / (2 * 2) = -(−60 + 50 + 4) / 4 = -(-6)/4 = 1.5
        V = calculate_intercalation_voltage(
            discharged_formula="Ca2Fe4O8",
            discharged_energy_per_formula=-60.0,
            charged_formula="Fe2O4",
            charged_energy_per_formula=-25.0,
            functional_string="GGA",
            working_ion="Ca",
            working_ion_valency=2,
            working_ion_energies=self.MOCK_ENERGIES,
        )
        assert abs(V - 1.5) < 1e-10

    def test_r2scan_functional(self):
        """R2SCAN functional uses R2SCAN reference energy."""
        V = calculate_intercalation_voltage(
            discharged_formula="CaFe2O4",
            discharged_energy_per_formula=-30.0,
            charged_formula="Fe2O4",
            charged_energy_per_formula=-25.0,
            functional_string="R2SCAN",
            working_ion="Ca",
            working_ion_valency=2,
            working_ion_energies=self.MOCK_ENERGIES,
        )
        # V = -(-30 - (-25) - 1*(-7)) / (2 * 1) = -(−30 + 25 + 7) / 2 = -(2)/2 = -1.0
        assert abs(V - (-1.0)) < 1e-10

    def test_gga_u_functional(self):
        """GGA+U should use GGA reference energy."""
        V_gga = calculate_intercalation_voltage(
            discharged_formula="CaFe2O4",
            discharged_energy_per_formula=-30.0,
            charged_formula="Fe2O4",
            charged_energy_per_formula=-25.0,
            functional_string="GGA",
            working_ion="Ca",
            working_ion_valency=2,
            working_ion_energies=self.MOCK_ENERGIES,
        )
        V_ggau = calculate_intercalation_voltage(
            discharged_formula="CaFe2O4",
            discharged_energy_per_formula=-30.0,
            charged_formula="Fe2O4",
            charged_energy_per_formula=-25.0,
            functional_string="GGA+U",
            working_ion="Ca",
            working_ion_valency=2,
            working_ion_energies=self.MOCK_ENERGIES,
        )
        assert V_gga == V_ggau

    def test_incompatible_functional_raises(self):
        with pytest.raises(TypeError, match="Incompatible functional"):
            calculate_intercalation_voltage(
                discharged_formula="CaFe2O4",
                discharged_energy_per_formula=-30.0,
                charged_formula="Fe2O4",
                charged_energy_per_formula=-25.0,
                functional_string="UNKNOWN",
                working_ion="Ca",
                working_ion_valency=2,
                working_ion_energies=self.MOCK_ENERGIES,
            )

    def test_monovalent_ion(self):
        """Test with Na (valency=1)."""
        V = calculate_intercalation_voltage(
            discharged_formula="NaFe2O4",
            discharged_energy_per_formula=-30.0,
            charged_formula="Fe2O4",
            charged_energy_per_formula=-25.0,
            functional_string="GGA",
            working_ion="Na",
            working_ion_valency=1,
            working_ion_energies=self.MOCK_ENERGIES,
        )
        # V = -(-30 - (-25) - 1*(-2)) / (1 * 1) = -(−3) / 1 = 3.0
        assert abs(V - 3.0) < 1e-10

    def test_partial_working_ion_in_charged(self):
        """Charged structure still has some working ions."""
        # Ca2Fe2O4 → CaFe2O4 (partial de-intercalation)
        # Fe: 2 in both, LCM=2. delta_Ca = 2 - 1 = 1
        V = calculate_intercalation_voltage(
            discharged_formula="Ca2Fe2O4",
            discharged_energy_per_formula=-35.0,
            charged_formula="CaFe2O4",
            charged_energy_per_formula=-30.0,
            functional_string="GGA",
            working_ion="Ca",
            working_ion_valency=2,
            working_ion_energies=self.MOCK_ENERGIES,
        )
        # V = -(-35 - (-30) - 1*(-2)) / (2 * 1) = -(−35+30+2)/2 = -(-3)/2 = 1.5
        assert abs(V - 1.5) < 1e-10

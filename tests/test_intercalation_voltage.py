"""Tests for intercalation voltage calculation."""

from spices.intercalation.voltage import calculate_intercalation_voltage


class TestCalculateIntercalationVoltage:
    """Test the Nernst-based voltage calculation.

    SPICES restricts every thermo query to ``thermo_type=GGA_GGA+U`` so
    the voltage helper is anchored to a single GGA(+U) reference energy.
    """

    GGA_REF_ENERGY = -2.0

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
            working_ion="Ca",
            working_ion_valency=2,
            working_ion_energy=self.GGA_REF_ENERGY,
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
            working_ion="Ca",
            working_ion_valency=2,
            working_ion_energy=self.GGA_REF_ENERGY,
        )
        assert abs(V - 1.5) < 1e-10

    def test_monovalent_ion(self):
        """Test with Na (valency=1)."""
        V = calculate_intercalation_voltage(
            discharged_formula="NaFe2O4",
            discharged_energy_per_formula=-30.0,
            charged_formula="Fe2O4",
            charged_energy_per_formula=-25.0,
            working_ion="Na",
            working_ion_valency=1,
            working_ion_energy=self.GGA_REF_ENERGY,
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
            working_ion="Ca",
            working_ion_valency=2,
            working_ion_energy=self.GGA_REF_ENERGY,
        )
        # V = -(-35 - (-30) - 1*(-2)) / (2 * 1) = -(−35+30+2)/2 = -(-3)/2 = 1.5
        assert abs(V - 1.5) < 1e-10

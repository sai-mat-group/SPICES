"""
Default constants and configuration for the SPICES package.
"""

from pymatgen.core import Element

# Nernst equation prefactor at 25C (RT/F * ln(10) ~ 0.0591 V)
PREFAC = 0.0591

# Water stability line coefficients: ax + by + c = 0
# H stability line: V = -PREFAC * pH  =>  PREFAC*pH + V = 0
H_LINE = (PREFAC, 1, 0)

# O stability line: V = -PREFAC * pH + 1.23  =>  PREFAC*pH + V - 1.23 = 0
O_LINE = (PREFAC, 1, -1.23)

# Default transition metals to screen
DEFAULT_TRANSITION_METALS = [
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Nb", "Mo", "Ru", "Pd", "Re", "Os", "Ir",
]

# Default anions to include in chemical system search
DEFAULT_ANIONS = ["O", "S", "Se", "F", "Cl", "Br", "I", "P", "N", "C", "Si"]

# Supported working ion elements
SUPPORTED_ELEMENTS = ["Ca", "K", "Li", "Mg", "Na"]

# Default decomposition energy threshold (eV) for metastability
DEFAULT_DECOMP_ENERGY_THRESHOLD = 0.5

# Default pH range for water stability check
DEFAULT_PH_RANGE = [0, 14]

# Default voltage range for metastability grid search
DEFAULT_VOLTAGE_RANGE = [-1, 1.5]

# Elements commonly classified as anionic in Pourbaix analysis
ANIONIC_ELEMENTS = frozenset({
    Element("H"), Element("O"), Element("S"), Element("Se"),
    Element("F"), Element("Cl"), Element("Br"), Element("I"),
    Element("P"), Element("N"), Element("C"), Element("Si"),
})

# H and O elements (frequently excluded from Pourbaix composition)
ELEMENTS_HO = frozenset({Element("H"), Element("O")})

# Seawater ion concentrations (molarity)
SEAWATER_CONC_DICT = {
    "Cl": 0.5588856000000001,
    "Na": 0.4800684,
    "Mg": 0.05404608,
    "S": 0.028865520000000002,
    "Ca": 0.010543080000000002,
    "K": 0.01044072,
}

# Seawater composition dict (unit stoichiometry for Pourbaix)
SEAWATER_COMP_DICT = {"Cl": 1, "Na": 1, "Mg": 1, "S": 1, "Ca": 1, "K": 1}

# --- Seawater screening (Stage 2) defaults ---
SEAWATER_PH_RANGE = (7, 8.5)
SEAWATER_VOLTAGE_RANGE = (-1.5, 2)
SEAWATER_DECOMP_ENERGY_THRESHOLD = 0.3
DEFAULT_OER_SHIFT = 0.5
DEFAULT_HER_SHIFT = -0.3
SEAWATER_PH_GRIDS = 7
SEAWATER_VOLTAGE_GRIDS = 41

# --- Intercalation voltage (Stage 3) defaults ---
# Working ion properties: valency and standard electrode potential vs SHE
WORKING_ION_PARAMS = {
    "Ca": {"valency": 2, "voltage_vs_SHE": -2.87},
    "K":  {"valency": 1, "voltage_vs_SHE": -2.93},
    "Li": {"valency": 1, "voltage_vs_SHE": -3.04},
    "Mg": {"valency": 2, "voltage_vs_SHE": -2.37},
    "Na": {"valency": 1, "voltage_vs_SHE": -2.71},
}

# Default selection criteria for voltage plateau analysis
DEFAULT_SELECTION_CRITERIA = [1, 2, 3, 4, 5, 6, 7, 8]

# Default decomposition energy for Stage 3 metastability checks
INTERCALATION_DECOMP_ENERGY_THRESHOLD = 0.3

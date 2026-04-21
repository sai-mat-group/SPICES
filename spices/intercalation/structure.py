"""
Charged-structure caching utilities for intercalation voltage analysis.
"""

import logging
import os

from pymatgen.core import Element, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

logger = logging.getLogger(__name__)

# Default matcher with attempt_supercell=True to catch supercell matches
# that the original notebooks missed (which used the default False).
DEFAULT_MATCHER = StructureMatcher(attempt_supercell=True)


def generate_charged_struct_from_MP(
    mp_id,
    working_ion,
    mp_client,
    struct_dir=None,
    charged_struct_dir=None,
):
    """
    Download a structure from MP and remove the working ion to produce
    the charged (de-intercalated) framework.

    Parameters
    ----------
    mp_id : str
        Materials Project ID.
    working_ion : str
        Element symbol to remove (e.g. ``"Ca"``).
    mp_client : MPRester
    struct_dir : str or None
        Directory to save the original structure CIF.
    charged_struct_dir : str or None
        Directory to save the charged structure CIF.

    Returns
    -------
    Structure or list
        The charged structure, or an empty list if the MP ID is not
        found.
    """
    logger.info("Generating charged structure for %s", mp_id)

    struct = mp_client.materials.get_structure_by_material_id(mp_id)

    # MP may return empty list for missing IDs
    if isinstance(struct, list) and len(struct) == 0:
        return struct

    # Save original structure
    if struct_dir is not None:
        os.makedirs(struct_dir, exist_ok=True)
        struct.to_file(
            filename=os.path.join(struct_dir, f"{mp_id}.cif"), fmt="cif",
        )

    # Remove working ion to get charged framework
    struct.remove_species([Element(working_ion)])

    # Defensive: ensure Element objects (not charged Species)
    struct.remove_oxidation_states()

    # Save charged structure
    if charged_struct_dir is not None:
        os.makedirs(charged_struct_dir, exist_ok=True)
        struct.to_file(
            filename=os.path.join(charged_struct_dir, f"{mp_id}.cif"),
            fmt="cif",
        )

    return struct


def read_or_generate_charged_struct(
    mp_id,
    working_ion,
    mp_client,
    charged_struct_dir="charged_structs",
    struct_dir="structs",
    cache_dict=None,
):
    """
    Retrieve a charged structure from memory cache, disk cache, or MP.

    Parameters
    ----------
    mp_id : str
    working_ion : str
    mp_client : MPRester
    charged_struct_dir : str
        Directory for disk-cached charged structure CIFs.
    struct_dir : str
        Directory for disk-cached original structure CIFs.
    cache_dict : dict or None
        In-memory cache mapping MP ID to Structure. If provided,
        structures are stored here for fast retrieval.

    Returns
    -------
    Structure or list
        The charged structure, or empty list if not found.
    """
    # Check memory cache
    if cache_dict is not None and mp_id in cache_dict:
        return cache_dict[mp_id]

    # Check disk cache
    if os.path.exists(charged_struct_dir):
        cif_path = os.path.join(charged_struct_dir, f"{mp_id}.cif")
        if os.path.exists(cif_path):
            struct = Structure.from_file(cif_path)
            # Defensive: ensure Element objects
            struct.remove_oxidation_states()
            if cache_dict is not None:
                cache_dict[mp_id] = struct
            return struct

    # Download and generate
    struct = generate_charged_struct_from_MP(
        mp_id, working_ion, mp_client,
        struct_dir=struct_dir,
        charged_struct_dir=charged_struct_dir,
    )
    if cache_dict is not None and not isinstance(struct, list):
        cache_dict[mp_id] = struct

    return struct


def find_structures(
    structure,
    mp_client,
    structure_cache_dir="All_charged_structures",
    matcher=None,
):
    """
    Find MP IDs of structures matching the given charged framework.

    Searches a disk cache of CIF files organized by reduced formula.
    If the cache directory for this formula doesn't exist, downloads
    all structures with that formula from MP and saves them.

    Parameters
    ----------
    structure : Structure
        The charged framework structure to match.
    mp_client : MPRester
    structure_cache_dir : str
        Root directory for cached structure CIFs.
    matcher : StructureMatcher or None
        Matcher to use. Defaults to ``DEFAULT_MATCHER``
        (``attempt_supercell=True``).

    Returns
    -------
    list of str
        Matching MP IDs.
    """
    if matcher is None:
        matcher = DEFAULT_MATCHER

    reduced_formula = structure.composition.reduced_formula
    mp_id_list = []

    os.makedirs(structure_cache_dir, exist_ok=True)
    formula_dir = os.path.join(structure_cache_dir, reduced_formula)

    if os.path.exists(formula_dir):
        # Match against saved structures
        for fname in os.listdir(formula_dir):
            if not fname.endswith(".cif"):
                continue
            saved_struct = Structure.from_file(
                os.path.join(formula_dir, fname),
            )
            saved_struct.remove_oxidation_states()
            if matcher.fit(struct1=structure, struct2=saved_struct, symmetric=True):
                mp_id_list.append(fname.split(".")[0])

        return mp_id_list

    # Download and cache all structures with this formula
    downloaded = mp_client.materials.search(
        formula=reduced_formula,
        all_fields=False,
        fields=["material_id", "structure"],
    )

    if not downloaded:
        return mp_id_list

    os.makedirs(formula_dir, exist_ok=True)
    for entry in downloaded:
        entry.structure.to_file(
            filename=os.path.join(formula_dir, f"{entry.material_id}.cif"),
            fmt="cif",
        )

    # Recurse to match against the newly saved structures
    return find_structures(
        structure, mp_client,
        structure_cache_dir=structure_cache_dir,
        matcher=matcher,
    )



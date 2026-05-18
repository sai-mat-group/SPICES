"""
Caching layer for Pourbaix entries and diagrams.

Two-layer cache:

* **On-disk pickle store** under ``entries_dir``/``pbx_dir`` is authoritative
  and persists across runs.  Built once, read forever.
* **In-memory LRU dicts** hold a bounded working set of fully-deserialized
  Python objects so repeated lookups in the same process don't hit disk.
  Capped via ``max_entries`` / ``max_diagrams`` / ``max_sea_diagrams`` to
  keep multi-worker runs from accumulating multi-GB of `PourbaixDiagram`
  references — eviction merely forces the next lookup to re-read the
  existing on-disk pickle (~0.1–1 s) instead of rebuilding from MP entries.

All disk writes go through :func:`_atomic_pickle_dump` (write to a
PID-tagged temp file then ``os.replace``) so concurrent workers can't leave
a truncated pickle that the next reader would hit as ``EOFError``.
"""

import os
import pickle
import logging
from collections import OrderedDict

from pymatgen.core import Composition
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram

from spices.pourbaix.composition import (
    get_pourbaix_composition,
    get_sea_pourbaix_comp_dict,
    get_sea_pourbaix_conc_dict,
)

logger = logging.getLogger(__name__)


def _atomic_pickle_dump(obj, path, protocol=pickle.HIGHEST_PROTOCOL):
    """Pickle *obj* to *path* via a PID-tagged temp file + ``os.replace``.

    ``os.replace`` is atomic on POSIX, so concurrent readers either see the
    old file or the fully-written new file — never a half-written one.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class PourbaixCache:
    """
    In-memory + on-disk cache for Pourbaix entries and diagrams.

    Parameters
    ----------
    entries_dir : str
        Directory for pickled Pourbaix entry lists.
    pbx_dir : str
        Directory for pickled Pourbaix diagram objects.
    max_entries : int
        Max number of chemsys -> entries kept in memory (default 8).
    max_diagrams : int
        Max number of formula -> dilute-water ``PourbaixDiagram`` kept in
        memory (default 8).
    max_sea_diagrams : int
        Max number of formula -> seawater ``PourbaixDiagram`` kept in
        memory (default 8).

    Notes
    -----
    The three caps bound per-worker RSS in multi-worker runs.  Each cached
    diagram can be 50–200 MB; with 12 workers and unbounded growth the
    OS will OOM-kill workers.  The on-disk pickle remains authoritative,
    so eviction is cheap.
    """

    def __init__(self, entries_dir="saved_pbx_entries", pbx_dir="saved_pbx",
                 max_entries=8, max_diagrams=8, max_sea_diagrams=8):
        self.entries_dir = entries_dir
        self.pbx_dir = pbx_dir

        self.max_entries = max_entries
        self.max_diagrams = max_diagrams
        self.max_sea_diagrams = max_sea_diagrams

        # In-memory LRU caches.  Newest at the right end; evict from the
        # left when len exceeds the cap.
        self._pourbaix_entries = OrderedDict()
        self._pourbaix_diagrams = OrderedDict()
        self._sea_pourbaix_diagrams = OrderedDict()

    @staticmethod
    def _evict_to_size(d, max_size):
        """Pop oldest items from *d* until ``len(d) <= max_size``."""
        while len(d) > max_size:
            d.popitem(last=False)

    # ------------------------------------------------------------------
    # Pourbaix entries
    # ------------------------------------------------------------------

    def get_or_download_pourbaix_entries(self, chemsys, mp_client,
                                        save=True):
        """
        Return Pourbaix entries for *chemsys*, downloading if needed.

        Parameters
        ----------
        chemsys : str
        mp_client : MPRester
        save : bool
            Whether to save newly downloaded entries to disk.

        Returns
        -------
        list of PourbaixEntry
        """
        if chemsys not in self._pourbaix_entries:
            logger.info("Downloading PourbaixEntry list for %s from MP", chemsys)
            self._pourbaix_entries[chemsys] = mp_client.get_pourbaix_entries(
                chemsys=chemsys
            )
            self._evict_to_size(self._pourbaix_entries, self.max_entries)
            if save:
                self._save_entries(chemsys)
        else:
            self._pourbaix_entries.move_to_end(chemsys)

        return self._pourbaix_entries[chemsys]

    def read_or_download_pourbaix_entries(self, chemsys, mp_client,
                                         save_new=True):
        """
        Read entries from disk cache, or download and save them.

        Parameters
        ----------
        chemsys : str
        mp_client : MPRester
        save_new : bool

        Returns
        -------
        list of PourbaixEntry
        """
        # Check disk cache
        if os.path.exists(self.entries_dir):
            for fname in os.listdir(self.entries_dir):
                if _get_chemsys_from_entry_name(fname) == chemsys:
                    fpath = os.path.join(self.entries_dir, fname)
                    logger.info("Found saved PourbaixEntry %s", fname)
                    with open(fpath, "rb") as f:
                        entries = pickle.load(f)
                    self._pourbaix_entries[chemsys] = entries
                    self._evict_to_size(self._pourbaix_entries, self.max_entries)
                    return entries
        else:
            os.makedirs(self.entries_dir, exist_ok=True)

        return self.get_or_download_pourbaix_entries(
            chemsys, mp_client, save=save_new,
        )

    def _save_entries(self, chemsys):
        os.makedirs(self.entries_dir, exist_ok=True)
        fpath = os.path.join(self.entries_dir, f"{chemsys}_pbxentry.pkl")
        _atomic_pickle_dump(self._pourbaix_entries[chemsys], fpath)

    # ------------------------------------------------------------------
    # Pourbaix diagrams (dilute / pure water)
    # ------------------------------------------------------------------

    def generate_pourbaix_diagram(self, pbx_comp, mp_client, save_entries=True):
        """
        Generate a new Pourbaix diagram from downloaded entries.

        Parameters
        ----------
        pbx_comp : Composition
            Pourbaix-relevant composition (O/H removed).
        mp_client : MPRester
        save_entries : bool

        Returns
        -------
        PourbaixDiagram
        """
        entries = self.read_or_download_pourbaix_entries(
            pbx_comp.chemical_system, mp_client, save_new=save_entries,
        )
        logger.info("Generating Pourbaix Diagram for %s", pbx_comp)
        return PourbaixDiagram(entries, comp_dict=pbx_comp.as_dict())

    def get_or_generate_pourbaix_diagram(self, composition_obj, mp_client,
                                         mp_id=None, save_pbx=False,
                                         save_entries=False):
        """
        Get a diagram from the in-memory cache, or generate and
        optionally save it.

        Parameters
        ----------
        composition_obj : Composition
            Full material composition.
        mp_client : MPRester
        mp_id : str or None
            Used in the pickle filename.
        save_pbx : bool
        save_entries : bool

        Returns
        -------
        PourbaixDiagram
        """
        pbx_comp = get_pourbaix_composition(composition_obj)
        key = pbx_comp.reduced_composition.formula

        if key not in self._pourbaix_diagrams:
            self._pourbaix_diagrams[key] = self.generate_pourbaix_diagram(
                pbx_comp, mp_client, save_entries=save_entries,
            )
            self._evict_to_size(self._pourbaix_diagrams, self.max_diagrams)
            if save_pbx:
                self._save_diagram(pbx_comp, mp_id)
        else:
            self._pourbaix_diagrams.move_to_end(key)

        return self._pourbaix_diagrams[key]

    def read_or_generate_pourbaix_diagram(self, composition_obj, mp_client,
                                          save_new_pbx=True,
                                          save_new_entries=True):
        """
        Read a diagram from disk, or generate it.

        Parameters
        ----------
        composition_obj : Composition
        mp_client : MPRester
        save_new_pbx : bool
        save_new_entries : bool

        Returns
        -------
        PourbaixDiagram
        """
        pbx_comp = get_pourbaix_composition(composition_obj)
        target_key = pbx_comp.reduced_composition.formula.replace(" ", "_")

        if os.path.exists(self.pbx_dir):
            for fname in os.listdir(self.pbx_dir):
                if _get_comp_from_pbx_name(fname) == target_key:
                    fpath = os.path.join(self.pbx_dir, fname)
                    logger.info("Found saved PourbaixDiagram %s", fname)
                    with open(fpath, "rb") as f:
                        diag = pickle.load(f)
                    self._pourbaix_diagrams[pbx_comp.reduced_composition.formula] = diag
                    self._evict_to_size(self._pourbaix_diagrams, self.max_diagrams)
                    return diag
        else:
            os.makedirs(self.pbx_dir, exist_ok=True)

        return self.get_or_generate_pourbaix_diagram(
            composition_obj, mp_client,
            save_pbx=save_new_pbx, save_entries=save_new_entries,
        )

    def _save_diagram(self, pbx_comp, mp_id=None):
        os.makedirs(self.pbx_dir, exist_ok=True)
        name = pbx_comp.reduced_composition.formula
        if mp_id:
            name = f"{name}_{mp_id}"
        fpath = os.path.join(
            self.pbx_dir, f"{name.replace(' ', '_')}_pbx.pkl"
        )
        key = pbx_comp.reduced_composition.formula
        _atomic_pickle_dump(self._pourbaix_diagrams[key], fpath)

    # ------------------------------------------------------------------
    # Seawater Pourbaix diagrams
    # ------------------------------------------------------------------

    def generate_sea_pourbaix_diagram(self, composition_obj, mp_client,
                                      num_of_sea_elements=4,
                                      save_entries=True):
        """
        Generate a seawater Pourbaix diagram.

        Parameters
        ----------
        composition_obj : Composition
        mp_client : MPRester
        num_of_sea_elements : int
        save_entries : bool

        Returns
        -------
        PourbaixDiagram
        """
        pbx_comp = get_pourbaix_composition(composition_obj)
        sea_comp_dict = get_sea_pourbaix_comp_dict(pbx_comp, num_of_sea_elements)
        sea_conc_dict = get_sea_pourbaix_conc_dict(pbx_comp, num_of_sea_elements)

        entries = self.read_or_download_pourbaix_entries(
            Composition(sea_comp_dict).chemical_system, mp_client,
            save_new=save_entries,
        )
        logger.info("Generating sea Pourbaix Diagram for %s", Composition(sea_comp_dict))
        return PourbaixDiagram(
            entries, comp_dict=sea_comp_dict, conc_dict=sea_conc_dict,
        )

    def get_or_generate_sea_pourbaix_diagram(self, composition_obj, mp_client,
                                              mp_id=None,
                                              num_of_sea_elements=4,
                                              save_pbx=False,
                                              save_entries=False):
        """Get or generate a seawater Pourbaix diagram."""
        pbx_comp = get_pourbaix_composition(composition_obj)
        sea_comp = Composition(get_sea_pourbaix_comp_dict(pbx_comp, num_of_sea_elements))
        key = sea_comp.reduced_composition.formula

        if key not in self._sea_pourbaix_diagrams:
            self._sea_pourbaix_diagrams[key] = self.generate_sea_pourbaix_diagram(
                composition_obj, mp_client,
                num_of_sea_elements=num_of_sea_elements,
                save_entries=save_entries,
            )
            self._evict_to_size(self._sea_pourbaix_diagrams, self.max_sea_diagrams)
            if save_pbx:
                os.makedirs(self.pbx_dir, exist_ok=True)
                name = sea_comp.reduced_composition.formula
                if mp_id:
                    name = f"{name}_{mp_id}"
                fpath = os.path.join(
                    self.pbx_dir, f"{name.replace(' ', '_')}_pbx.pkl"
                )
                _atomic_pickle_dump(self._sea_pourbaix_diagrams[key], fpath)
        else:
            self._sea_pourbaix_diagrams.move_to_end(key)

        return self._sea_pourbaix_diagrams[key]

    def read_or_generate_sea_pourbaix_diagram(self, composition_obj, mp_client,
                                               num_of_sea_elements=4,
                                               save_new_pbx=True,
                                               save_new_entries=True):
        """Read a seawater diagram from disk, or generate it."""
        pbx_comp = get_pourbaix_composition(composition_obj)
        sea_comp = Composition(get_sea_pourbaix_comp_dict(pbx_comp, num_of_sea_elements))
        target_key = sea_comp.reduced_composition.formula.replace(" ", "_")

        if os.path.exists(self.pbx_dir):
            for fname in os.listdir(self.pbx_dir):
                if _get_comp_from_pbx_name(fname) == target_key:
                    fpath = os.path.join(self.pbx_dir, fname)
                    logger.info("Found saved sea PourbaixDiagram %s", fname)
                    with open(fpath, "rb") as f:
                        diag = pickle.load(f)
                    self._sea_pourbaix_diagrams[sea_comp.reduced_composition.formula] = diag
                    self._evict_to_size(self._sea_pourbaix_diagrams, self.max_sea_diagrams)
                    return diag
        else:
            os.makedirs(self.pbx_dir, exist_ok=True)

        return self.get_or_generate_sea_pourbaix_diagram(
            composition_obj, mp_client,
            num_of_sea_elements=num_of_sea_elements,
            save_pbx=save_new_pbx, save_entries=save_new_entries,
        )

    def read_or_generate_bare_seawater_pbx(self, mp_client,
                                            num_of_seawater_elements=4,
                                            seawater_pbx=None):
        """
        Get a bare seawater Pourbaix diagram (no cathode elements).

        Parameters
        ----------
        mp_client : MPRester
        num_of_seawater_elements : int
        seawater_pbx : PourbaixDiagram or None
            If provided, used to detect additional seawater elements.

        Returns
        -------
        PourbaixDiagram
        """
        from spices.config import SEAWATER_COMP_DICT, SEAWATER_CONC_DICT

        conc_dict = {
            k: v for i, (k, v) in enumerate(SEAWATER_CONC_DICT.items())
            if i < num_of_seawater_elements
        }
        comp_dict = {
            k: v for i, (k, v) in enumerate(SEAWATER_COMP_DICT.items())
            if i < num_of_seawater_elements
        }

        # Add extra seawater elements from the full diagram
        if seawater_pbx is not None:
            for elem in seawater_pbx._elt_comp.keys():
                if elem in SEAWATER_COMP_DICT and elem not in comp_dict:
                    comp_dict[elem] = SEAWATER_COMP_DICT[elem]
                    conc_dict[elem] = SEAWATER_CONC_DICT[elem]

        target_key = Composition(comp_dict).reduced_composition.formula.replace(" ", "_")

        # Try reading from disk
        if os.path.exists(self.pbx_dir):
            for fname in os.listdir(self.pbx_dir):
                if _get_comp_from_pbx_name(fname) == target_key:
                    fpath = os.path.join(self.pbx_dir, fname)
                    with open(fpath, "rb") as f:
                        return pickle.load(f)

        # Generate
        entries = self.read_or_download_pourbaix_entries(
            Composition(comp_dict).chemical_system, mp_client,
        )
        bare_pbx = PourbaixDiagram(
            entries=entries, comp_dict=comp_dict, conc_dict=conc_dict,
        )

        # Save
        os.makedirs(self.pbx_dir, exist_ok=True)
        fpath = os.path.join(
            self.pbx_dir,
            f"{Composition(comp_dict).reduced_composition.formula.replace(' ', '_')}_pbx.pkl",
        )
        _atomic_pickle_dump(bare_pbx, fpath)

        return bare_pbx


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_chemsys_from_entry_name(fname):
    """``'Ca-Fe-O_pbxentry.pkl'`` -> ``'Ca-Fe-O'``"""
    return fname.split("_pbxentry")[0]


def _get_comp_from_pbx_name(pbx_name):
    """``'Ca1_Fe1_mp-12345_pbx.pkl'`` -> ``'Ca1_Fe1'``"""
    if "mp" in pbx_name:
        return pbx_name.split("_mp")[0]
    return pbx_name.split("_pbx")[0]

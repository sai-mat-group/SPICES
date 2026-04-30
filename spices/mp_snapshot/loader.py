"""
Indexed reader for a local Materials Project snapshot on disk.

Exists to serve SnapshotMPRester.  Responsibilities:

- Walk each collection's subtree (``summary/``, ``thermo/
  thermo_type=GGA_GGA+U/``, ``materials/``) and build forward +
  secondary indexes on first use; pickle them under ``<root>/indexes/``
  so subsequent runs are instant.
- Given a query (material_id, chemsys, formula, crystal_system), locate
  the shard, parse the target line, return the raw emmet-shaped dict.
- Cache parsed shards in an LRU so batch queries that hit the same shard
  repeatedly don't re-parse.

No wrapping into SPICES-facing attribute shapes happens here — that is
``client.py``'s job.  This layer deals strictly in dicts.
"""

import gzip
import json
import logging
import os
import pickle
from collections import OrderedDict, defaultdict

from spices.mp_snapshot import THERMO_TYPE

logger = logging.getLogger(__name__)

# Each collection maps to a list of subroots.  SPICES restricts thermo
# reads to the single GGA_GGA+U subroot to stay consistent with the live
# MP API's Pourbaix path (which hardcodes this thermo_type).
COLLECTIONS = {
    "summary": ["summary"],
    "thermo": [f"thermo/thermo_type={THERMO_TYPE}"],
    "materials": ["materials"],
}

# Files we never want to index (partition-level manifests).
SKIP_FILENAMES = {"manifest.jsonl.gz", "manifest.parquet"}

# Default size for the per-instance shard LRU.  Overridable via
# ``SPICES_SHARD_CACHE`` env var so multi-worker runs on tight-RAM
# machines can shrink the cache; HPC users with abundant RAM can grow
# it for batch jobs that hammer the same shards repeatedly.  Each cached
# shard holds a fully parsed list of JSON docs (typical thermo shard
# ~10–30 MB in memory).  Default 8 → ~80–240 MB ceiling per worker.
DEFAULT_SHARD_CACHE_SIZE = 8


def _atomic_pickle_dump(obj, path, protocol=pickle.HIGHEST_PROTOCOL):
    """Pickle *obj* to *path* via a PID-tagged temp file + ``os.replace``.

    Atomic on POSIX, so concurrent workers building the same index can't
    leave a truncated pickle that the next reader hits as ``EOFError``.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

# Bumped when on-disk pickle format changes incompatibly.  v2 added
# multi-subroot thermo and switched by_mid values from a single tuple
# to a list of tuples.  v3 added subroots_indexed so a freshly-downloaded
# subroot invalidates an older cached index (v2 pickles would otherwise
# silently miss the new shards).  Any pickle older than this triggers a
# rebuild.
INDEX_VERSION = 3


class Index:
    """
    In-memory index for a single collection.

    Attributes
    ----------
    by_mid : dict[str, list[tuple[str, int]]]
        material_id -> [(shard_relpath, line_number_zero_based), ...]
        List for forward-compat with multi-subroot indexing; the active
        thermo collection has a single subroot (GGA_GGA+U), so each
        mp-id appears once.
    by_chemsys : dict[str, list[str]]
        chemsys -> [material_id, ...]
    by_elemset : dict[frozenset, list[str]]
        frozenset(elements) -> [material_id, ...]
    by_formula : dict[str, list[str]]
        formula_pretty -> [material_id, ...]
    by_crystal_system : dict[str, list[str]]
        crystal_system -> [material_id, ...]
    """

    def __init__(self):
        self.version = INDEX_VERSION
        self.by_mid = defaultdict(list)
        self.by_chemsys = defaultdict(list)
        self.by_elemset = defaultdict(list)
        self.by_formula = defaultdict(list)
        self.by_crystal_system = defaultdict(list)
        # Subroots (relative to release_root) actually walked at build
        # time — used to detect a subroot appearing on disk after the
        # cache was written (e.g. a later `download` run fetched a new
        # thermo prefix), which would otherwise silently keep serving
        # the stale index.
        self.subroots_indexed = []

    def freeze(self):
        # convert defaultdicts to plain dicts so pickle is stable
        self.by_mid = dict(self.by_mid)
        self.by_chemsys = dict(self.by_chemsys)
        self.by_elemset = {k: v for k, v in self.by_elemset.items()}
        self.by_formula = dict(self.by_formula)
        self.by_crystal_system = dict(self.by_crystal_system)
        return self


class SnapshotLoader:
    """
    Reader for a local Materials Project snapshot.

    Parameters
    ----------
    snapshot_root : str
        Path to the root that contains ``<release>/{summary,thermo,materials}/``.
    release : str
        Release date string, e.g. ``"2025-06-09"``.
    """

    def __init__(self, snapshot_root, release="2025-06-09",
                 shard_cache_size=None):
        self.snapshot_root = snapshot_root
        self.release = release
        self.release_root = os.path.join(snapshot_root, release)
        self.index_dir = os.path.join(self.release_root, "indexes")

        if not os.path.isdir(self.release_root):
            raise FileNotFoundError(
                f"Snapshot release directory not found: {self.release_root}. "
                f"Run `python -m spices.mp_snapshot.download {snapshot_root} "
                f"--release {release}` first."
            )

        self._indexes = {}  # collection_name -> Index

        # Per-instance LRU for parsed shards.  Size: explicit kwarg >
        # SPICES_SHARD_CACHE env var > DEFAULT_SHARD_CACHE_SIZE.  After
        # fork() each worker gets its own (process-private) cache, so
        # the cap directly bounds per-worker RSS contribution from
        # snapshot reads.
        if shard_cache_size is None:
            shard_cache_size = int(
                os.environ.get("SPICES_SHARD_CACHE", DEFAULT_SHARD_CACHE_SIZE)
            )
        self._shard_cache_size = shard_cache_size
        self._shard_cache = OrderedDict()  # shard_relpath -> list[dict]

    # ------------------------------------------------------------------
    # Index build + load
    # ------------------------------------------------------------------

    def get_index(self, collection):
        """Return the Index for *collection*, building if needed."""
        if collection in self._indexes:
            return self._indexes[collection]

        idx_path = os.path.join(self.index_dir, f"{collection}.idx.pkl")
        idx = None
        if os.path.exists(idx_path):
            logger.info("Loading cached index: %s", idx_path)
            with open(idx_path, "rb") as f:
                idx = pickle.load(f)
            cached_version = getattr(idx, "version", 1)
            if cached_version < INDEX_VERSION:
                logger.info(
                    "Snapshot index v%d -> v%d: rebuilding (~30s)",
                    cached_version, INDEX_VERSION,
                )
                idx = None

        # Content-drift check: a subroot may have been downloaded after
        # the cached index was written (the version stamp alone can't
        # catch this).  Compare subroots present on disk now against
        # what was walked at build time; rebuild on any mismatch.
        if idx is not None:
            current = sorted(
                sub for sub in COLLECTIONS[collection]
                if os.path.isdir(os.path.join(self.release_root, sub))
            )
            indexed = sorted(getattr(idx, "subroots_indexed", []))
            if current != indexed:
                logger.info(
                    "Snapshot index for %s built from %s; disk now has %s "
                    "— rebuilding (~30s)",
                    collection, indexed, current,
                )
                idx = None

        if idx is None:
            idx = self._build_index(collection)
            os.makedirs(self.index_dir, exist_ok=True)
            _atomic_pickle_dump(idx, idx_path)
            logger.info("Wrote index %s", idx_path)

        self._indexes[collection] = idx
        return idx

    def _collection_subroots(self, collection):
        return [os.path.join(self.release_root, sub)
                for sub in COLLECTIONS[collection]]

    def _iter_shards(self, collection):
        for root in self._collection_subroots(collection):
            if not os.path.isdir(root):
                # Allow the snapshot to be partial (e.g. user hasn't yet
                # downloaded a newly-added prefix) — just skip it.
                logger.warning(
                    "Subroot missing for %s: %s (skipping). "
                    "Rerun `python -m spices.mp_snapshot.download` to fetch.",
                    collection, root,
                )
                continue
            for dirpath, _, fnames in os.walk(root):
                for fn in fnames:
                    if fn in SKIP_FILENAMES:
                        continue
                    if not fn.endswith(".jsonl.gz"):
                        continue
                    full = os.path.join(dirpath, fn)
                    rel = os.path.relpath(full, self.release_root)
                    yield rel, full

    def _build_index(self, collection):
        """
        Stream every shard once, record material_id -> [(shard, line_no), ...],
        and populate secondary indexes.
        """
        logger.info("Building index for collection=%s ...", collection)
        idx = Index()

        n_docs = 0
        n_shards = 0
        for rel, full in self._iter_shards(collection):
            n_shards += 1
            with gzip.open(full, "rt") as gz:
                for lineno, line in enumerate(gz):
                    if not line.strip():
                        continue
                    doc = json.loads(line)
                    mid = doc.get("material_id")
                    if not mid:
                        continue
                    idx.by_mid[mid].append((rel, lineno))

                    chemsys = doc.get("chemsys")
                    if chemsys:
                        idx.by_chemsys[chemsys].append(mid)
                        idx.by_elemset[frozenset(chemsys.split("-"))].append(mid)

                    fp = doc.get("formula_pretty")
                    if fp:
                        idx.by_formula[fp].append(mid)

                    sym = doc.get("symmetry")
                    if isinstance(sym, dict):
                        cs = sym.get("crystal_system")
                        if cs:
                            idx.by_crystal_system[cs].append(mid)

                    n_docs += 1
            if n_shards % 50 == 0:
                logger.info(
                    "  ... indexed %d shards, %d docs so far", n_shards, n_docs
                )

        idx.subroots_indexed = sorted(
            sub for sub in COLLECTIONS[collection]
            if os.path.isdir(os.path.join(self.release_root, sub))
        )
        idx.freeze()
        logger.info(
            "Index build for %s complete: %d shards, %d docs, %d chemsys, "
            "%d formulas, subroots=%s",
            collection, n_shards, n_docs,
            len(idx.by_chemsys), len(idx.by_formula),
            idx.subroots_indexed,
        )
        return idx

    # ------------------------------------------------------------------
    # Shard reads (with per-shard LRU cache)
    # ------------------------------------------------------------------

    def _load_shard(self, shard_relpath):
        """Parse an entire shard once; cache the list of dicts.

        LRU-bounded by ``self._shard_cache_size`` (env var
        ``SPICES_SHARD_CACHE``, default 8).  After fork() each worker
        owns its own cache, so the cap directly limits per-worker RSS.
        """
        if shard_relpath in self._shard_cache:
            self._shard_cache.move_to_end(shard_relpath)
            return self._shard_cache[shard_relpath]

        full = os.path.join(self.release_root, shard_relpath)
        docs = []
        with gzip.open(full, "rt") as gz:
            for line in gz:
                if not line.strip():
                    continue
                docs.append(json.loads(line))

        self._shard_cache[shard_relpath] = docs
        while len(self._shard_cache) > self._shard_cache_size:
            self._shard_cache.popitem(last=False)
        return docs

    def get_doc(self, collection, material_id):
        """
        Fetch the first doc matching *material_id* from *collection*.

        Use :meth:`get_docs_for_mid` when an mp-id could in principle map
        to multiple docs (multi-subroot indexing).  With the active
        single-subroot thermo collection this returns the same doc.
        """
        idx = self.get_index(collection)
        hits = idx.by_mid.get(material_id)
        if not hits:
            return None
        shard, lineno = hits[0]
        docs = self._load_shard(shard)
        return docs[lineno]

    def get_docs_for_mid(self, collection, material_id):
        """All docs matching *material_id* — one per (subroot) location."""
        idx = self.get_index(collection)
        hits = idx.by_mid.get(material_id) or []
        out = []
        for shard, lineno in hits:
            docs = self._load_shard(shard)
            out.append(docs[lineno])
        return out

    def get_docs(self, collection, material_ids):
        """
        Batch fetch.  Returns one doc per (mp-id, location) pair.  Output
        is flat; callers needing strict one-per-id should dedup
        themselves.
        """
        idx = self.get_index(collection)
        # Group by shard to avoid reopening each file for every ID.
        by_shard = defaultdict(list)
        for mid in material_ids:
            hits = idx.by_mid.get(mid) or []
            for shard, lineno in hits:
                by_shard[shard].append((lineno, mid))

        out = []
        for shard, items in by_shard.items():
            docs = self._load_shard(shard)
            for lineno, _mid in items:
                out.append(docs[lineno])
        return out

    # ------------------------------------------------------------------
    # Search helpers — mirror mp_client.materials.*.search semantics
    # ------------------------------------------------------------------

    def search(
        self,
        collection,
        chemsys=None,
        material_ids=None,
        formula=None,
        crystal_system=None,
    ):
        """
        Return docs matching the given filters.  Filters AND together.

        - ``chemsys`` is matched **exactly** (sorted dash-joined string,
          e.g. ``"Li-Mn-O"``).  A list is treated as union (match any).
        - ``material_ids`` restricts to the given ID list.
        - ``formula`` is matched against ``formula_pretty`` (union if list).
        - ``crystal_system`` matches ``symmetry.crystal_system`` (e.g.
          ``"Cubic"``; also accepts enum-like with ``.value``).
        """
        idx = self.get_index(collection)

        # Start with candidate material_ids from the most selective filter.
        candidates = None

        if material_ids is not None:
            mids = [material_ids] if isinstance(material_ids, str) else list(material_ids)
            candidates = set(mids)

        if chemsys is not None:
            cs_list = [chemsys] if isinstance(chemsys, str) else list(chemsys)
            cs_hits = set()
            for cs in cs_list:
                # Canonical form: MP stores chemsys as sorted dash-joined.
                canon = "-".join(sorted(cs.split("-")))
                cs_hits.update(idx.by_chemsys.get(canon, ()))
            candidates = cs_hits if candidates is None else candidates & cs_hits

        if formula is not None:
            # MP stores formula_pretty in reduced form ("TiO2"), but
            # callers (e.g. SPICES rm_element) pass raw composition
            # strings ("Ti2 O4").  Reduce input before lookup.
            from pymatgen.core import Composition
            f_list = [formula] if isinstance(formula, str) else list(formula)
            f_hits = set()
            for f in f_list:
                canon = Composition(f).reduced_formula
                f_hits.update(idx.by_formula.get(canon, ()))
            candidates = f_hits if candidates is None else candidates & f_hits

        if crystal_system is not None:
            cs_val = crystal_system.value if hasattr(crystal_system, "value") else str(crystal_system)
            cs_hits = set(idx.by_crystal_system.get(cs_val, ()))
            candidates = cs_hits if candidates is None else candidates & cs_hits

        if candidates is None:
            raise ValueError(
                "search() called with no filters — refusing to return the "
                "entire snapshot."
            )

        return self.get_docs(collection, sorted(candidates))

    def search_subchemsys(self, collection, element_list):
        """
        Return all docs whose chemsys elements are a **subset** of
        *element_list* — mirrors the ``get_entries_in_chemsys`` semantics
        used internally by ``MPRester.get_pourbaix_entries``.
        """
        idx = self.get_index(collection)
        target = frozenset(element_list)
        mids = []
        for elemset, mid_list in idx.by_elemset.items():
            if elemset <= target:
                mids.extend(mid_list)
        return self.get_docs(collection, mids)

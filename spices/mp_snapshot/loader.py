"""
Indexed reader for a local Materials Project snapshot on disk.

Exists to serve SnapshotMPRester.  Responsibilities:

- Walk the three collection subtrees (``summary/``, ``thermo/thermo_type=
  GGA_GGA+U/``, ``materials/``) and build forward + secondary indexes on
  first use; pickle them under ``<root>/indexes/`` so subsequent runs are
  instant.
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
from collections import defaultdict
from functools import lru_cache

logger = logging.getLogger(__name__)

COLLECTIONS = {
    "summary": "summary",
    "thermo": "thermo/thermo_type=GGA_GGA+U",
    "materials": "materials",
}

# Files we never want to index (partition-level manifests).
SKIP_FILENAMES = {"manifest.jsonl.gz", "manifest.parquet"}


class Index:
    """
    In-memory index for a single collection.

    Attributes
    ----------
    by_mid : dict[str, tuple[str, int]]
        material_id -> (shard_relpath, line_number_zero_based)
    by_chemsys : dict[str, list[str]]
        chemsys -> [material_id, ...]
    by_elemset : dict[frozenset, list[str]]
        frozenset(elements) -> [material_id, ...]  (same info, different key)
    by_formula : dict[str, list[str]]
        formula_pretty -> [material_id, ...]
    by_crystal_system : dict[str, list[str]]
        crystal_system -> [material_id, ...]  (populated for materials/summary)
    """

    def __init__(self):
        self.by_mid = {}
        self.by_chemsys = defaultdict(list)
        self.by_elemset = defaultdict(list)
        self.by_formula = defaultdict(list)
        self.by_crystal_system = defaultdict(list)

    def freeze(self):
        # convert defaultdicts to plain dicts so pickle is stable
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

    def __init__(self, snapshot_root, release="2025-06-09"):
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

    # ------------------------------------------------------------------
    # Index build + load
    # ------------------------------------------------------------------

    def get_index(self, collection):
        """Return the Index for *collection*, building if needed."""
        if collection in self._indexes:
            return self._indexes[collection]

        idx_path = os.path.join(self.index_dir, f"{collection}.idx.pkl")
        if os.path.exists(idx_path):
            logger.info("Loading cached index: %s", idx_path)
            with open(idx_path, "rb") as f:
                idx = pickle.load(f)
        else:
            idx = self._build_index(collection)
            os.makedirs(self.index_dir, exist_ok=True)
            with open(idx_path, "wb") as f:
                pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Wrote index %s", idx_path)

        self._indexes[collection] = idx
        return idx

    def _collection_root(self, collection):
        sub = COLLECTIONS[collection]
        return os.path.join(self.release_root, sub)

    def _iter_shards(self, collection):
        root = self._collection_root(collection)
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
        Stream every shard once, record material_id -> (shard, line_no),
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
                    idx.by_mid[mid] = (rel, lineno)

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

        idx.freeze()
        logger.info(
            "Index build for %s complete: %d shards, %d docs, %d chemsys, "
            "%d formulas",
            collection, n_shards, n_docs,
            len(idx.by_chemsys), len(idx.by_formula),
        )
        return idx

    # ------------------------------------------------------------------
    # Shard reads (with per-shard LRU cache)
    # ------------------------------------------------------------------

    @lru_cache(maxsize=32)
    def _load_shard(self, shard_relpath):
        """Parse an entire shard once; cache the list of dicts."""
        full = os.path.join(self.release_root, shard_relpath)
        docs = []
        with gzip.open(full, "rt") as gz:
            for line in gz:
                if not line.strip():
                    continue
                docs.append(json.loads(line))
        return docs

    def get_doc(self, collection, material_id):
        """Fetch one doc by material_id from *collection*."""
        idx = self.get_index(collection)
        hit = idx.by_mid.get(material_id)
        if hit is None:
            return None
        shard, lineno = hit
        docs = self._load_shard(shard)
        return docs[lineno]

    def get_docs(self, collection, material_ids):
        """Batch version of :meth:`get_doc`; preserves input order."""
        idx = self.get_index(collection)
        # Group by shard to avoid reopening each file for every ID.
        by_shard = defaultdict(list)
        order = {}
        for pos, mid in enumerate(material_ids):
            hit = idx.by_mid.get(mid)
            if hit is None:
                continue
            shard, lineno = hit
            by_shard[shard].append((lineno, mid, pos))
            order[mid] = pos

        out = [None] * len(material_ids)
        for shard, items in by_shard.items():
            docs = self._load_shard(shard)
            for lineno, mid, pos in items:
                out[pos] = docs[lineno]
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
            f_list = [formula] if isinstance(formula, str) else list(formula)
            f_hits = set()
            for f in f_list:
                f_hits.update(idx.by_formula.get(f, ()))
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

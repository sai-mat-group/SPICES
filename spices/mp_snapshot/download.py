"""
Download a frozen Materials Project release from AWS OpenData into a
local directory that ``SnapshotMPRester`` can serve from.

The public bucket ``s3://materialsproject-build`` hosts versioned
collections under ``collections/{release-date}/{collection}/``.  This
module copies the three collections SPICES needs
(``summary``, ``thermo/GGA_GGA+U``, ``materials``) plus the ion
reference data required to rebuild Pourbaix entries offline.

Default release is ``2025-06-09`` — the MP build that was live in
June 2025.
"""

import argparse
import datetime as _dt
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore import UNSIGNED
from botocore.client import Config

logger = logging.getLogger(__name__)

# Thread-local boto3 client so concurrent downloads don't serialize on a
# single HTTP connection pool.
_tls = threading.local()


def _thread_s3_client():
    if not hasattr(_tls, "client"):
        _tls.client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    return _tls.client

DEFAULT_RELEASE = "2025-06-09"
BUCKET = "materialsproject-build"

REQUIRED_PREFIXES = [
    "summary/",
    "thermo/thermo_type=GGA_GGA+U/",
    "materials/",
]


def _s3_client():
    """Anonymous S3 client — MP OpenData buckets allow no-sign-request."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _iter_keys(s3, bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            yield obj["Key"], obj["Size"]


def _download_one(args):
    """Thread-worker: fetch a single S3 object, return (bytes, was_skipped)."""
    bucket, key, local, size, skip_existing = args
    if skip_existing and os.path.exists(local) and os.path.getsize(local) == size:
        return size, True
    os.makedirs(os.path.dirname(local), exist_ok=True)
    s3 = _thread_s3_client()
    s3.download_file(bucket, key, local)
    return size, False


def download_collection(
    release, prefix, dest_root, skip_existing=True, max_workers=32,
):
    """
    Mirror ``s3://{BUCKET}/collections/{release}/{prefix}`` into
    ``{dest_root}/{release}/{prefix}`` using a thread pool.

    Parameters
    ----------
    release : str
    prefix : str
    dest_root : str
    skip_existing : bool
    max_workers : int
        Size of the thread pool that fetches shards in parallel.
    """
    s3 = _s3_client()
    s3_prefix = f"collections/{release}/{prefix}"
    local_base = os.path.join(dest_root, release, prefix)
    os.makedirs(local_base, exist_ok=True)

    # Pre-enumerate everything so we can show progress against a known total.
    jobs = []
    for key, size in _iter_keys(s3, BUCKET, s3_prefix):
        rel = key[len(s3_prefix):]
        if not rel:
            continue
        local = os.path.join(local_base, rel)
        jobs.append((BUCKET, key, local, size, skip_existing))

    logger.info("Collection %s: %d objects to mirror", prefix, len(jobs))

    total_bytes = 0
    downloaded = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_download_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            size, was_skipped = fut.result()
            if was_skipped:
                skipped += 1
            else:
                downloaded += 1
                total_bytes += size
            if i % 50 == 0 or i == len(jobs):
                logger.info(
                    "  [%s] %d/%d done (%d dl, %d skip, %.2f GB)",
                    prefix, i, len(jobs), downloaded, skipped, total_bytes / 1e9,
                )

    logger.info(
        "Collection %s complete: %d downloaded, %d skipped, %.2f GB",
        prefix, downloaded, skipped, total_bytes / 1e9,
    )
    return downloaded, skipped, total_bytes


def download_ion_ref_data(dest_root, release):
    """
    Freeze today's MPContribs ``ion_ref_data`` project to disk.

    There is no pymatgen-shipped ion reference JSON in recent versions,
    so MPContribs is the only source.  Note that MPContribs is not
    pinned alongside the ``collections/<release>/`` bulk dumps — this
    function snapshots whatever is live at download time.  That freeze
    date is written into the manifest for provenance.

    Requires ``MP_API_KEY`` in the environment.
    """
    ion_dir = os.path.join(dest_root, release, "ion_ref_data")
    os.makedirs(ion_dir, exist_ok=True)

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        logger.warning(
            "MP_API_KEY not set; skipping MPContribs ion_ref_data dump. "
            "Offline Pourbaix rebuild will be unavailable until this is "
            "fetched — rerun `python -m spices.mp_snapshot.download "
            "<root>` once the key is set."
        )
        return

    try:
        # The MPContribs client ships as a subpackage of mpcontribs-client
        # under two possible import paths; tolerate both.
        try:
            from mpcontribs.client import Client
        except ImportError:
            from mpcontribs_client import Client  # type: ignore

        client = Client(api_key)
        result = client.contributions.queryContributions(
            project="ion_ref_data",
            _fields=["identifier", "formula", "data"],
            _limit=500,
        ).result()
        records = list(result.get("data") or [])

        dump_path = os.path.join(ion_dir, "mpcontribs_ion_ref_data.json")
        with open(dump_path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        logger.info("Wrote %s (%d records)", dump_path, len(records))
    except Exception as exc:
        logger.warning(
            "MPContribs ion_ref_data dump failed: %s. Offline Pourbaix "
            "rebuild will raise until this file is obtained.",
            exc,
        )


def _pkg_version(pkg_name):
    try:
        from importlib.metadata import version
        return version(pkg_name)
    except Exception:
        return "unknown"


def write_manifest(dest_root, release, notes=""):
    """Record provenance for peer review."""
    manifest = {
        "release": release,
        "source": f"s3://{BUCKET}/collections/{release}/",
        "download_timestamp_utc": _dt.datetime.utcnow().isoformat(timespec="seconds"),
        "pymatgen_version": _pkg_version("pymatgen"),
        "mp_api_version": _pkg_version("mp-api"),
        "mpcontribs_client_version": _pkg_version("mpcontribs-client"),
        "prefixes": REQUIRED_PREFIXES,
        "notes": notes,
    }
    path = os.path.join(dest_root, release, "manifest.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest %s", path)


def download_snapshot(
    dest_root,
    release=DEFAULT_RELEASE,
    prefixes=None,
    skip_ion_ref_data=False,
):
    """
    Full snapshot download: collections + ion reference data + manifest.

    Parameters
    ----------
    dest_root : str
        Snapshot root on local disk.
    release : str
        MP release date to pull.
    prefixes : list of str or None
        Override the collection list (default: summary/, thermo/GGA_GGA+U/,
        materials/).
    skip_ion_ref_data : bool
        Skip the ion_ref_data fetch step.
    """
    prefixes = prefixes or REQUIRED_PREFIXES
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    for prefix in prefixes:
        download_collection(release, prefix, dest_root, skip_existing=True)

    if not skip_ion_ref_data:
        download_ion_ref_data(dest_root, release)

    write_manifest(dest_root, release)


def _parse_argv(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "dest_root",
        help="Local snapshot root, e.g. "
        "/home/sougat/Desktop/sougat_desktop_files/Sougat/"
        "fileserver_backup/Desalination/MP_snapshots",
    )
    p.add_argument("--release", default=DEFAULT_RELEASE)
    p.add_argument(
        "--skip-ion-ref-data",
        action="store_true",
        help="Skip the ion_ref_data dump (useful for re-runs of the bulk copy)",
    )
    p.add_argument(
        "--prefix",
        action="append",
        dest="prefixes",
        help="Override collection prefix list (can be given multiple times)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_argv(sys.argv[1:])
    download_snapshot(
        dest_root=args.dest_root,
        release=args.release,
        prefixes=args.prefixes,
        skip_ion_ref_data=args.skip_ion_ref_data,
    )

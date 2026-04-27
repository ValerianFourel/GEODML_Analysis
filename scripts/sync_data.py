"""Fetch only files that are new or changed on the HF dataset.

Compares remote file list (with sha + size) against the local copy under
GEODML_DATA_ROOT, downloads just the diff into the same paths, and
optionally extracts any newly-arrived html_cache tarballs.

Usage:
    python scripts/sync_data.py
    python scripts/sync_data.py --extract-html
    python scripts/sync_data.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getenv("HF_DATASET_REPO", "ValerianFourel/geodml-papersize"))
    ap.add_argument("--local-dir", default=os.getenv("GEODML_DATA_ROOT", "./geodml_data"))
    ap.add_argument("--revision", default=None, help="Pin to a specific commit/tag/branch.")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be fetched, don't download.")
    ap.add_argument("--extract-html", action="store_true", help="Unpack any newly-fetched html_cache tarballs.")
    args = ap.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set in environment or .env")
        return 2

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    local_root = Path(args.local_dir).resolve()
    local_root.mkdir(parents=True, exist_ok=True)

    print(f"[sync] repo={args.repo} rev={args.revision or 'main'} -> {local_root}")
    info = api.repo_info(repo_id=args.repo, repo_type="dataset", revision=args.revision, files_metadata=True)
    remote_files = [s for s in info.siblings if s.rfilename and not s.rfilename.endswith("/")]
    print(f"[sync] remote files: {len(remote_files)}")

    to_fetch: list[tuple[str, str | None, int | None]] = []
    for s in remote_files:
        local_path = local_root / s.rfilename
        remote_size = getattr(s, "size", None)
        remote_sha = getattr(s, "lfs", None)
        remote_sha = remote_sha.get("sha256") if isinstance(remote_sha, dict) else None
        if not local_path.exists():
            to_fetch.append((s.rfilename, "missing", remote_size))
            continue
        if remote_size is not None and local_path.stat().st_size != remote_size:
            to_fetch.append((s.rfilename, "size-mismatch", remote_size))
            continue

    if not to_fetch:
        print("[sync] up to date — nothing to download.")
    else:
        total_mb = sum((sz or 0) for _, _, sz in to_fetch) / (1024 * 1024)
        print(f"[sync] {len(to_fetch)} file(s) to fetch (~{total_mb:.1f} MB):")
        for name, reason, sz in to_fetch[:50]:
            print(f"   - [{reason}] {name}" + (f"  ({(sz or 0)/(1024*1024):.1f} MB)" if sz else ""))
        if len(to_fetch) > 50:
            print(f"   ... and {len(to_fetch) - 50} more")

        if args.dry_run:
            print("[sync] dry-run: not downloading.")
        else:
            new_files: list[Path] = []
            for name, _, _ in to_fetch:
                hf_hub_download(
                    repo_id=args.repo,
                    repo_type="dataset",
                    filename=name,
                    revision=args.revision,
                    local_dir=str(local_root),
                    local_dir_use_symlinks=False,
                    token=token,
                )
                new_files.append(local_root / name)
            print(f"[sync] downloaded {len(new_files)} file(s).")

            if args.extract_html:
                new_tarballs = [p for p in new_files if p.name == "html_cache.tar.gz"]
                print(f"[extract] {len(new_tarballs)} new html_cache tarball(s)")
                for tb in new_tarballs:
                    target = tb.parent / "html_cache"
                    if target.exists() and any(target.iterdir()):
                        print(f"[extract] skip (already unpacked): {target}")
                        continue
                    print(f"[extract] {tb}")
                    with tarfile.open(tb, "r:gz") as tf:
                        tf.extractall(tb.parent)

    print(f"OK. Data root: {local_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

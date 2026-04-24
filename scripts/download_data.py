"""Download the GEODML HuggingFace dataset to a local directory.

The dataset is ~3.5 GB without HTML caches, ~28 GB with.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --extract-html
    python scripts/download_data.py --local-dir /mnt/fast/geodml_data
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo", default=os.getenv("HF_DATASET_REPO", "ValerianFourel/geodml-papersize")
    )
    ap.add_argument(
        "--local-dir", default=os.getenv("GEODML_DATA_ROOT", "./geodml_data")
    )
    ap.add_argument(
        "--extract-html",
        action="store_true",
        help="After download, expand data/runs/*/phase2/html_cache.tar.gz in place.",
    )
    ap.add_argument(
        "--no-download",
        action="store_true",
        help="Skip download, only run post-processing (e.g. --extract-html).",
    )
    args = ap.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set. Copy .env.example to .env and fill it in.")
        return 2

    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_download:
        print(f"[download] repo={args.repo} -> {local_dir}")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=token,
            max_workers=8,
        )
        print(f"[download] done. Size:")
        subprocess.run(["du", "-sh", str(local_dir)], check=False)

    if args.extract_html:
        runs_dir = local_dir / "data" / "runs"
        if not runs_dir.exists():
            print(f"ERROR: {runs_dir} not found. Download first.")
            return 3
        tarballs = sorted(runs_dir.glob("*/phase2/html_cache.tar.gz"))
        print(f"[extract] found {len(tarballs)} html_cache tarballs")
        for tb in tarballs:
            target = tb.parent / "html_cache"
            if target.exists() and any(target.iterdir()):
                print(f"[extract] skip (already unpacked): {target}")
                continue
            print(f"[extract] {tb}")
            with tarfile.open(tb, "r:gz") as tf:
                tf.extractall(tb.parent)
        print("[extract] done.")

    print(f"OK. Data root: {local_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

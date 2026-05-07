#!/usr/bin/env python3
"""Login-node driver for Stage C (merge). Thin wrapper over
``interpretability.pipeline.merge`` so the operator doesn't have to remember
the module path.

Usage:
    python scripts/build_main_table.py --variant biased
    python scripts/build_main_table.py --variant neutral --external-features-parquet path/to/extra.parquet
    python scripts/build_main_table.py --variant biased --runs searxng_Llama-3.3-70B-Instruct_serp50_top10_biased

Pure pandas; ~30 s on full data; no GPU; safe to run on a JUWELS login node.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so `interpretability.*` resolves regardless
# of CWD or whether the user has set PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from interpretability.pipeline.merge import main


if __name__ == "__main__":
    raise SystemExit(main())

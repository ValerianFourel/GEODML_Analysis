#!/usr/bin/env bash
# Build a consolidated, push-ready dataset at ~/Hamburg/geodml-dataset/
# (sibling to ~/Hamburg/GEODML/ and ~/Hamburg/GEODML_Analysis/).
#
# Pulls together the three independent sources of truth into one tree:
#   1. Mac upstream     ~/Hamburg/GEODML/paperSizeExperiment/output/
#                       └── extracted html_caches (8 cells, ~16 GB)
#   2. Cluster snapshot $GEODML_DATA_ROOT (from the HF zips you unzipped)
#                       └── rerank checkpoints, order_probe, dataforseo, dml_results
#   3. Cluster Stage F  $REPO_ROOT/interpretability/output/
#                       └── ablation, saliency, weights, probing CSVs + plots
#
# Layout:
#   ~/Hamburg/geodml-dataset/
#     README.md                  ← what this is, how to use it
#     PROVENANCE.md              ← what came from where, when
#     refresh.sh                 ← rebuild from sources (this script, copied in)
#     data/
#       serp/                    ← phase0_*.parquet (from cluster)
#       dataforseo/              ← (from cluster)
#       runs/<run_id>/           ← rerank+order_probe outputs (from cluster)
#         phase2/html_cache/     ← extracted HTML (symlink → Mac upstream)
#         phase2/keywords.jsonl  ← rerank output (from cluster)
#         phase3/                ← legacy features (from cluster)
#       order_probe/             ← jsonls + summary (from cluster)
#       features/                ← Stage B (from local run)
#       main/                    ← Stage C (from local run)
#       dml_results/             ← Stage D (from local run) + legacy
#     interpretability/
#       output/                  ← Stage F CSVs + plots (from cluster)
#     archives/                  ← zip snapshots (from cluster + local)
#
# Optional env:
#   DATASET_ROOT      where to build (default ~/Hamburg/geodml-dataset)
#   LOCAL_HTML_SOURCE Mac upstream root (default ~/Hamburg/GEODML/paperSizeExperiment/output)
#   GEODML_DATA_ROOT  cluster-snapshot root (default $REPO_ROOT/geodml_data)
#   COPY_HTML         "1" to physically copy html_caches (~16 GB) instead of
#                     symlinking. Required if you plan to push to HF — symlinks
#                     don't transfer.
#   COPY_DATA         "1" to copy data/runs/* and interpretability/output/*
#                     instead of symlinking (also required for HF push)
#   FORCE             "1" to overwrite existing dataset

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

step() { printf '\n\033[1m══════ %s ══════\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
go()   { printf '  \033[36m→\033[0m %s\n' "$*"; }
skip() { printf '  \033[33m⊘\033[0m %s\n' "$*"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$*"; }

: "${DATASET_ROOT:=$HOME/Hamburg/geodml-dataset}"
: "${LOCAL_HTML_SOURCE:=$HOME/Hamburg/GEODML/paperSizeExperiment/output}"
: "${GEODML_DATA_ROOT:=$REPO_ROOT/geodml_data}"
: "${COPY_HTML:=}"
: "${COPY_DATA:=}"
: "${FORCE:=}"

DATE=$(date +%Y-%m-%d)
DATETIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Choose link or copy strategy
if [ -n "$COPY_DATA" ]; then
  link_dir() { go "copy: $1 → $2"; rsync -a --delete "$1/" "$2/"; }
else
  link_dir() {
    rm -rf "$2"
    ln -sfn "$1" "$2"
    ok "link: $2 → $1"
  }
fi

if [ -n "$COPY_HTML" ]; then
  link_html() {
    rm -rf "$2"   # remove any stale symlink so rsync doesn't follow it
    mkdir -p "$(dirname "$2")"
    go "copy: $1 → $2 (~$(du -sh "$1" | cut -f1))"
    rsync -a "$1/" "$2/"
  }
else
  link_html() {
    mkdir -p "$(dirname "$2")"
    rm -rf "$2"
    ln -sfn "$1" "$2"
    ok "link: $2 → $1"
  }
fi

step "Pre-flight"
echo "  DATASET_ROOT       = $DATASET_ROOT"
echo "  LOCAL_HTML_SOURCE  = $LOCAL_HTML_SOURCE"
echo "  GEODML_DATA_ROOT   = $GEODML_DATA_ROOT"
echo "  REPO_ROOT          = $REPO_ROOT"
echo "  COPY_HTML          = ${COPY_HTML:-(0; symlink)}"
echo "  COPY_DATA          = ${COPY_DATA:-(0; symlink)}"

[ -d "$LOCAL_HTML_SOURCE" ] || { fail "LOCAL_HTML_SOURCE missing: $LOCAL_HTML_SOURCE"; exit 2; }
[ -d "$GEODML_DATA_ROOT/data" ] || { fail "GEODML_DATA_ROOT/data missing: $GEODML_DATA_ROOT/data"; exit 2; }

if [ -d "$DATASET_ROOT" ] && [ -z "$FORCE" ]; then
  echo "  $DATASET_ROOT already exists. Re-running this script will refresh"
  echo "  links/copies but won't delete unrelated files. Use FORCE=1 to wipe."
fi

# ── 1. Skeleton ──────────────────────────────────────────────────────────────
step "1/6  Create skeleton"
[ -n "$FORCE" ] && [ -d "$DATASET_ROOT" ] && { go "wiping $DATASET_ROOT"; rm -rf "$DATASET_ROOT"; }
mkdir -p "$DATASET_ROOT"/{data/{serp,runs,order_probe,features,main,dml_results,dataforseo,logs},interpretability/output,archives}
ok "skeleton at $DATASET_ROOT"

# ── 2. Cluster data: serp, runs, order_probe, dataforseo, logs ───────────────
step "2/6  Cluster snapshot → data/"
for sub in serp dataforseo logs order_probe; do
  src="$GEODML_DATA_ROOT/data/$sub"
  dest="$DATASET_ROOT/data/$sub"
  if [ -d "$src" ]; then
    link_dir "$src" "$dest"
  else
    skip "no $src — leaving $dest empty"
  fi
done

# data/runs/ is special — we want the cluster's per-cell rerank/feature/order
# outputs, but we'll inject html_cache symlinks into each *cell-base* dir
# (un-suffixed) below. So copy/link each run dir, then add html_cache.
src="$GEODML_DATA_ROOT/data/runs"
dest="$DATASET_ROOT/data/runs"
if [ -d "$src" ]; then
  if [ -n "$COPY_DATA" ]; then
    go "copy: $src → $dest (per-run, ~$(du -sh "$src" | cut -f1))"
    rsync -a --exclude='*/phase2/html_cache' "$src/" "$dest/"
  else
    rm -rf "$dest"
    ln -sfn "$src" "$dest"
    ok "link: $dest → $src"
  fi
fi

# Already-produced local outputs (Stage B/C/D) get linked too if present
for sub in features main dml_results; do
  src="$GEODML_DATA_ROOT/data/$sub"
  dest="$DATASET_ROOT/data/$sub"
  if [ -d "$src" ] && [ -n "$(ls -A "$src" 2>/dev/null)" ]; then
    link_dir "$src" "$dest"
  fi
done

# ── 3. html_cache from Mac upstream → injected into per-cell run dirs ────────
step "3/6  Mac html_caches → data/runs/<engine>_<Model>_serp<N>_top10/phase2/html_cache/"
linked_html=0; missing_html=0
# When COPY_DATA=0, runs/ is a single symlink to GEODML_DATA_ROOT/data/runs.
# We don't want to mutate that source. So if linking, we materialise per-cell
# subdirs in DATASET_ROOT/data/runs/ for the html_cache injection only.
if [ -L "$DATASET_ROOT/data/runs" ]; then
  go "(materialising per-cell dirs so we can inject html_cache without mutating cluster snapshot)"
  rm "$DATASET_ROOT/data/runs"
  mkdir -p "$DATASET_ROOT/data/runs"
  shopt -s nullglob
  for cell in "$GEODML_DATA_ROOT"/data/runs/*/; do
    name=$(basename "$cell")
    mkdir -p "$DATASET_ROOT/data/runs/$name"
    # symlink everything inside except phase2/html_cache (we'll add our own)
    for sub in "$cell"*; do
      sub_name=$(basename "$sub")
      if [ "$sub_name" = "phase2" ]; then
        mkdir -p "$DATASET_ROOT/data/runs/$name/phase2"
        for inner in "$sub"/*; do
          inner_name=$(basename "$inner")
          [ "$inner_name" = "html_cache" ] && continue
          [ "$inner_name" = "html_cache.tar.gz" ] && continue
          ln -sfn "$inner" "$DATASET_ROOT/data/runs/$name/phase2/$inner_name"
        done
      else
        ln -sfn "$sub" "$DATASET_ROOT/data/runs/$name/$sub_name"
      fi
    done
  done
  shopt -u nullglob
fi

for ENGINE_LEGACY in searxng duckduckgo; do
  [ "$ENGINE_LEGACY" = "duckduckgo" ] && ENGINE_NEW="ddg" || ENGINE_NEW="$ENGINE_LEGACY"
  for MODEL in Llama-3.3-70B-Instruct Qwen2.5-72B-Instruct; do
    for POOL in 20 50; do
      SRC="$LOCAL_HTML_SOURCE/${ENGINE_LEGACY}_${MODEL}_serp${POOL}_top10/html_cache"
      CELL_DIR="$DATASET_ROOT/data/runs/${ENGINE_NEW}_${MODEL}_serp${POOL}_top10/phase2"
      DEST="$CELL_DIR/html_cache"
      label="${ENGINE_NEW}/${MODEL}/pool=${POOL}"
      if [ ! -d "$SRC" ]; then
        skip "$label: source missing ($SRC)"
        missing_html=$((missing_html + 1))
        continue
      fi
      mkdir -p "$CELL_DIR"
      link_html "$SRC" "$DEST"
      linked_html=$((linked_html + 1))
    done
  done
done
echo "  summary: linked=$linked_html, missing=$missing_html / 8"

# ── 4. Stage F outputs → interpretability/output/ ────────────────────────────
step "4/6  Stage F (interpretability) outputs"
src="$REPO_ROOT/interpretability/output"
dest="$DATASET_ROOT/interpretability/output"
if [ -d "$src" ]; then
  if [ -n "$COPY_DATA" ]; then
    go "copy: $src → $dest (excluding t7_chunks, checkpoints)"
    rsync -a \
      --exclude='*/t7_chunks*' \
      --exclude='*/__pycache__/*' \
      --exclude='*/checkpoint_*.json' \
      --exclude='*.tmp.npz' \
      "$src/" "$dest/"
  else
    rm -rf "$dest"
    ln -sfn "$src" "$dest"
    ok "link: $dest → $src"
  fi
else
  skip "no $src — Stage F outputs not present yet"
fi

# ── 5. Archives ──────────────────────────────────────────────────────────────
step "5/6  Archives (zip snapshots)"
src="$REPO_ROOT/archives"
dest="$DATASET_ROOT/archives"
if [ -d "$src" ] && [ -n "$(ls -A "$src" 2>/dev/null)" ]; then
  rsync -a "$src/" "$dest/"
  ok "synced $(ls "$src" | wc -l | tr -d ' ') archive(s) to $dest"
else
  skip "no archives at $src"
fi

# ── 6. README + PROVENANCE + refresh.sh + push helper ────────────────────────
step "6/6  Docs + helpers"
cat > "$DATASET_ROOT/README.md" <<README
# GEODML consolidated dataset

One tree, three sources. Built $DATETIME from
\`$REPO_ROOT/scripts/build_dataset_mirror.sh\`.

## Layout

\`\`\`
data/
  serp/                  ← cluster snapshot (HF dataset)
  dataforseo/            ← cluster snapshot
  runs/<cell>/phase2/    ← cluster rerank checkpoints + jsonls
                         (html_cache/ is per-cell symlink → Mac upstream)
  order_probe/           ← cluster outputs + summary parquet
  features/              ← Stage B (local CPU)
  main/                  ← Stage C (local CPU)
  dml_results/           ← Stage D (local CPU)
interpretability/
  output/                ← Stage F (cluster GPU): ablation, saliency, weights, probing
archives/
  geodml_data_*.zip      ← cluster snapshot zip
  interpretability_*.zip ← cluster Stage F zip
  local_results_*.zip    ← what your laptop produced (Stages B/C/D + figures)
\`\`\`

## How to use

\`\`\`bash
export GEODML_DATA_ROOT=$DATASET_ROOT
cd $REPO_ROOT
python scripts/audit_pipeline.py
\`\`\`

## Refresh from sources

\`\`\`bash
bash $DATASET_ROOT/refresh.sh
\`\`\`

## Push to HF

The default build uses **symlinks** for html_caches and large dirs (fast,
small disk). To push to HF you need real files:

\`\`\`bash
COPY_HTML=1 COPY_DATA=1 FORCE=1 bash $REPO_ROOT/scripts/build_dataset_mirror.sh
hf upload-large-folder ValerianFourel/geodml-papersize-full \\
  $DATASET_ROOT --repo-type dataset
\`\`\`

(Use a different repo name than \`geodml-papersize\` if you don't want to
overwrite the cluster snapshot — the consolidated tree contains the
extracted html_caches which are ~16 GB on top of the cluster snapshot.)

## What's incomplete

- **Stage F probing**: 2/8 cells. Needs cluster GPU; resume there.
- See PROVENANCE.md for the audit at build time.
README

cat > "$DATASET_ROOT/PROVENANCE.md" <<PROV
# Provenance

Generated $DATETIME by \`scripts/build_dataset_mirror.sh\`.

## Sources

| Path in dataset | Source |
|---|---|
| \`data/serp/\` | cluster snapshot (\`$GEODML_DATA_ROOT/data/serp\`) |
| \`data/dataforseo/\` | cluster snapshot |
| \`data/runs/<cell>/phase1\`, \`phase2\` (excl html_cache), \`phase3\` | cluster snapshot |
| \`data/runs/<cell>/phase2/html_cache/\` | **Mac upstream** (\`$LOCAL_HTML_SOURCE/<cell_legacy>/html_cache\`) — duckduckgo_ renamed to ddg_ |
| \`data/order_probe/\` | cluster snapshot |
| \`data/features/\`, \`data/main/\`, \`data/dml_results/\` | local CPU runs (\`continue_pipeline.sh\`, \`finish_via_api.sh\`) |
| \`interpretability/output/\` | cluster snapshot Stage F (\`$REPO_ROOT/interpretability/output\`) |
| \`archives/\` | both cluster zips and local result zips |

## Mac → cluster naming map

The Mac upstream dirs use \`duckduckgo_<Model>_serp<N>_top10/\`. The new
pipeline expects \`ddg_<Model>_serp<N>_top10/\`. \`build_dataset_mirror.sh\`
renames on link/copy.

## LLM config (matches cluster)

- \`LLM_TEMPERATURE = 0.1\`
- \`LLM_MAX_TOKENS = 500\`
- Source: \`interpretability/pipeline/config.py\`
- Both the cluster's \`LocalRanker\` and the local \`OpenAIRanker\` read from
  \`config.py\` directly, so any future re-run is comparable.

## Audit at build time

\`\`\`
$(cd "$REPO_ROOT" && GEODML_DATA_ROOT="$DATASET_ROOT" python scripts/audit_pipeline.py 2>&1 | tail -40 || echo "(audit failed; data may be incomplete)")
\`\`\`

## Git provenance

\`\`\`
$(git -C "$REPO_ROOT" log -1 --pretty=format:'%h %an %s' 2>/dev/null || echo n/a)
\`\`\`
PROV

# Embed a copy of this script for refresh.sh
cp "$REPO_ROOT/scripts/build_dataset_mirror.sh" "$DATASET_ROOT/refresh.sh"
chmod +x "$DATASET_ROOT/refresh.sh"

cat > "$DATASET_ROOT/push_to_hf.sh" <<'PUSH'
#!/usr/bin/env bash
# Push the consolidated dataset to HF.
# REQUIRES: real files (not symlinks). Run build with COPY_HTML=1 COPY_DATA=1
# first if anything in the tree is a symlink.
#
# Usage:
#   huggingface-cli login                # paste write-scoped token
#   REPO=ValerianFourel/geodml-papersize-full ./push_to_hf.sh

set -euo pipefail
DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=ValerianFourel/geodml-papersize-full}"

if find "$DATASET_ROOT" -type l -not -path '*/archives/*' | head -1 | grep -q . ; then
  echo "ERROR: tree contains symlinks — HF won't follow them. Rebuild with:"
  echo "  COPY_HTML=1 COPY_DATA=1 FORCE=1 bash <repo>/scripts/build_dataset_mirror.sh"
  exit 2
fi

hf upload-large-folder "$REPO" "$DATASET_ROOT" --repo-type dataset
PUSH
chmod +x "$DATASET_ROOT/push_to_hf.sh"

ok "wrote README.md, PROVENANCE.md, refresh.sh, push_to_hf.sh"

# ── Summary ──────────────────────────────────────────────────────────────────
step "Summary"
du -sh "$DATASET_ROOT" 2>/dev/null
du -sh "$DATASET_ROOT"/* 2>/dev/null
echo
echo "  Run audit against the new dataset:"
echo "    GEODML_DATA_ROOT=$DATASET_ROOT python scripts/audit_pipeline.py"
echo
echo "  To prepare for HF push (materialise everything):"
echo "    COPY_HTML=1 COPY_DATA=1 FORCE=1 bash scripts/build_dataset_mirror.sh"

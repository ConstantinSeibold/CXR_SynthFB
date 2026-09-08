#!/usr/bin/env bash
# generate_synthfb.sh — end-to-end SynthFB dataset generation
#
# Required env var:
#   MIMIC_CLEAN          path to MIMIC images root (expects p*/s*/*.jpg under it)
#
# Optional env vars:
#   MIMIC_META_CSV       source metadata CSV, only needed to build a NEW base-image pool
#                        (the pool shipped in data/ is used by default)
#   CUTOUTS_ROOT         RGBA cutouts dir (default: ./cutouts/labels)
#   SYNTHFB_CUTPASTE_PROB  per-annotation cut-paste probability (default 0.25)
#   SYNTHFB_DEVICE_PROB    per-annotation composite-device probability (default 0.25)
#
# CLI args:
#   -n N            number of synthetic images to generate (default 100)
#   -o OUT_DIR      output directory (default ./synthfb_out)
#   -s SEED         RNG seed (default 0)
#   --max-ann N     max annotations per image (default 12)
#   --split SPLIT   train|val|test (default: all). Source images disjoint across splits.
#                   When N > source pool size, samples with replacement.
#   --subset-csv P  explicit path to subset CSV. Overrides MIMIC_SUBSET_CSV env var;
#                   without either, falls back to <OUT_DIR>/mimic_fb_free_train.csv,
#                   then to the pool shipped in data/mimic_fb_free_pool.csv.
#   --skip-install  skip pip-install step
#   --skip-subset   skip prepare_mimic_subset.py (use existing CSV)
#   -h, --help      show this help
#
# Usage:
#   MIMIC_CLEAN=/data/mimic/images_512 ./generate_synthfb.sh -n 5000 -o ./synthfb_train
#
# Output layout:
#   <OUT_DIR>/
#     images/000001.png ...
#     masks/000001.png  ...        (uint16 instance maps)
#     annotations.json             (COCO + view + archetype per image, render_mode per annotation)
#     category_gallery.png         (with mask overlay)
#     category_gallery_clean.png   (no overlay, raw rendering)
#     category_examples/           (per-category PNGs, with mask)
#     category_examples_clean/     (per-category PNGs, clean render)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- defaults ----
N_IMAGES=100
OUT_DIR="./synthfb_out"
SEED=0
MAX_ANN=12
SPLIT=""
SKIP_INSTALL=0
SKIP_SUBSET=0
DEFAULT_META_CSV=""  # must be provided via MIMIC_META_CSV env var when subset CSV doesn't exist
DEFAULT_CUTOUTS="${SCRIPT_DIR}/cutouts/labels"

usage() {
  sed -n '2,33p' "$0"
  exit "${1:-0}"
}

SUBSET_CSV_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) N_IMAGES="$2"; shift 2 ;;
    -o) OUT_DIR="$2"; shift 2 ;;
    -s) SEED="$2"; shift 2 ;;
    --max-ann) MAX_ANN="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --subset-csv) SUBSET_CSV_ARG="$2"; shift 2 ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    --skip-subset)  SKIP_SUBSET=1;  shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown arg: $1" >&2; usage 1 ;;
  esac
done

if [[ -n "$SPLIT" && "$SPLIT" != "train" && "$SPLIT" != "val" && "$SPLIT" != "test" ]]; then
  echo "ERROR: --split must be one of: train, val, test (got '$SPLIT')" >&2
  exit 1
fi

# ---- validate ----
if [[ -z "${MIMIC_CLEAN:-}" ]]; then
  echo "ERROR: MIMIC_CLEAN env var not set." >&2
  echo "  Set it to your MIMIC images_512 root, e.g.:" >&2
  echo "    export MIMIC_CLEAN=/data/mimic/images_512" >&2
  exit 1
fi
if [[ ! -d "$MIMIC_CLEAN" ]]; then
  echo "ERROR: MIMIC_CLEAN dir does not exist: $MIMIC_CLEAN" >&2
  exit 1
fi

MIMIC_META_CSV="${MIMIC_META_CSV:-$DEFAULT_META_CSV}"
CUTOUTS_ROOT="${CUTOUTS_ROOT:-$DEFAULT_CUTOUTS}"
OUT_DIR_ABS="$(python -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$OUT_DIR")"
# Subset CSV resolution (in priority order):
#   1. --subset-csv CLI arg
#   2. MIMIC_SUBSET_CSV env var
#   3. <OUT_DIR>/mimic_fb_free_train.csv (previously built by this script)
#   4. the pool shipped with the repo (data/mimic_fb_free_pool.csv)
if [[ -n "$SUBSET_CSV_ARG" ]]; then
  SUBSET_CSV="$(python -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$SUBSET_CSV_ARG")"
elif [[ -n "${MIMIC_SUBSET_CSV:-}" ]]; then
  SUBSET_CSV="$(python -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$MIMIC_SUBSET_CSV")"
elif [[ -f "${OUT_DIR_ABS}/mimic_fb_free_train.csv" ]]; then
  SUBSET_CSV="${OUT_DIR_ABS}/mimic_fb_free_train.csv"
elif [[ -f "${SCRIPT_DIR}/data/mimic_fb_free_pool.csv" ]]; then
  SUBSET_CSV="${SCRIPT_DIR}/data/mimic_fb_free_pool.csv"
else
  SUBSET_CSV="${OUT_DIR_ABS}/mimic_fb_free_train.csv"
fi

mkdir -p "$OUT_DIR_ABS"

cat <<EOF
===== SynthFB generation =====
MIMIC_CLEAN:    $MIMIC_CLEAN
MIMIC_META_CSV: $MIMIC_META_CSV
CUTOUTS_ROOT:   $CUTOUTS_ROOT
OUT_DIR:        $OUT_DIR_ABS
SUBSET_CSV:     $SUBSET_CSV
N_IMAGES:       $N_IMAGES
MAX_ANN:        $MAX_ANN
SEED:           $SEED
SPLIT:          ${SPLIT:-(all)}
==============================
EOF

# ---- 1. install deps ----
if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  echo ">> installing deps (use --skip-install to skip)"
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r "${SCRIPT_DIR}/requirements.txt"
fi

# ---- 2. build the FB-free MIMIC train subset CSV ----
if [[ "$SKIP_SUBSET" -eq 0 ]]; then
  if [[ ! -f "$SUBSET_CSV" ]]; then
    if [[ ! -f "$MIMIC_META_CSV" ]]; then
      echo "ERROR: source meta CSV missing: $MIMIC_META_CSV" >&2
      echo "  Set MIMIC_META_CSV to point at your MIMIC metadata CSV (see README)" >&2
      exit 1
    fi
    echo ">> filtering source CSV -> $SUBSET_CSV"
    python "${SCRIPT_DIR}/prepare_mimic_subset.py" \
      --src     "$MIMIC_META_CSV" \
      --img-root "$MIMIC_CLEAN" \
      --out     "$SUBSET_CSV"
  else
    echo ">> reusing existing subset CSV"
  fi
fi

# ---- 3. locate generator script ----
SCRIPT_SRC="${SCRIPT_DIR}/SynthFB.py"
if [[ ! -f "$SCRIPT_SRC" ]]; then
  echo "ERROR: SynthFB.py missing from repo root." >&2
  exit 1
fi

# ---- 4. run the generator ----
echo ">> running generation"
export MIMIC_CLEAN
export MIMIC_SUBSET_CSV="$SUBSET_CSV"
export CUTOUTS_ROOT
export SYNTHFB_OUT="$OUT_DIR_ABS"
export SYNTHFB_N="$N_IMAGES"
export SYNTHFB_MAX_ANN="$MAX_ANN"
export SYNTHFB_SEED="$SEED"
export SYNTHFB_SPLIT="$SPLIT"
export MPLBACKEND="Agg"  # headless matplotlib

cd "$OUT_DIR_ABS"
python "$SCRIPT_SRC"

echo ""
echo ">> DONE: $OUT_DIR_ABS"
echo "   - images/      synthetic CXR PNGs"
echo "   - masks/       uint16 instance maps"
echo "   - annotations.json   COCO format (+ view + archetype + render_mode)"
echo "   - category_gallery*.png   per-class previews"

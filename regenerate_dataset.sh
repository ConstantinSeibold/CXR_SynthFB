#!/usr/bin/env bash
# regenerate_dataset.sh — one-command full SynthFB dataset rebuild.
#
# Wraps generate_synthfb.sh: filters the clean MIMIC meta CSV, then runs SynthFB.py
# 3x (train / val / test) into a single output root.
#
# Required env (with sensible defaults — override on the remote machine):
#   MIMIC_CLEAN       MIMIC images root containing p*/s*/*.jpg
#                     default: /data/mimic/images_512
#   MIMIC_META_CSV    source metadata CSV, only needed when building a NEW pool
#                     (the shipped pool in data/ is used by default).
#                     Must have columns: dataset_name, Mode, Support_Devices,
#                                        No_Finding, report, img_path, ViewDetailed
#                     default: $SCRIPT_DIR/mimic_metadata.csv
#   OUT_ROOT          output dataset root.
#                     default: $SCRIPT_DIR/synthfb_v2
#   N_TRAIN / N_VAL / N_TEST   per-split image counts.
#                     default: 5000 / 500 / 500
#   MAX_ANN           max annotations per image (default 12).
#   SEED              RNG seed (default 0).
#   ZIP_DIR           where to write per-split zip archives after generation.
#                     default: $OUT_ROOT/zips
#   FAST_MODE         1 = production speed flags (skip viz + gallery, JPG-95 images,
#                     fastest PNG compression). 0 = full debug outputs. default: 1.
#   CLEAN_POOL_CSV    pre-built clean base CSV (columns: img_path, split,
#                     optionally ViewDetailed; paths resolved against
#                     MIMIC_CLEAN). When this file exists,
#                     prepare_mimic_subset.py is skipped entirely and this CSV
#                     is shared across all three splits. When absent, the NLP
#                     report filter builds the pool from MIMIC_META_CSV.
#                     default: $SCRIPT_DIR/data/mimic_fb_free_pool.csv
#                     (the pool shipped with the repo)
#
# Usage on remote:
#   MIMIC_CLEAN=/data/mimic/images_512 \
#   MIMIC_META_CSV=/data/mimic/mimic_fb_subset.csv \
#   OUT_ROOT=/data/synthfb_v2 \
#   N_TRAIN=10000 N_VAL=1000 N_TEST=1000 \
#   ./regenerate_dataset.sh
#
# Output layout:
#   $OUT_ROOT/
#     mimic_fb_free_train.csv          (filtered subset, shared across splits)
#     train/                            images, masks, annotations.json, gallery
#     val/
#     test/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- defaults ----
MIMIC_CLEAN="${MIMIC_CLEAN:-/data/mimic/images_512}"
MIMIC_META_CSV="${MIMIC_META_CSV:-${SCRIPT_DIR}/mimic_metadata.csv}"
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/synthfb_v2}"
# Normalize to absolute path so subsequent `cd "$split_dir"` doesn't break
# relative ZIP_DIR resolution during the post-gen archive step.
OUT_ROOT="$(python -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$OUT_ROOT")"
N_TRAIN="${N_TRAIN:-5000}"
N_VAL="${N_VAL:-500}"
N_TEST="${N_TEST:-500}"
MAX_ANN="${MAX_ANN:-12}"
SEED="${SEED:-0}"
ZIP_DIR="${ZIP_DIR:-${OUT_ROOT}/zips}"
FAST_MODE="${FAST_MODE:-1}"
CLEAN_POOL_CSV="${CLEAN_POOL_CSV:-${SCRIPT_DIR}/data/mimic_fb_free_pool.csv}"

# Production speed flags — pass through to SynthFB.py via env
if [[ "$FAST_MODE" == "1" ]]; then
  export SYNTHFB_SKIP_VIZ=1
  export SYNTHFB_SKIP_GALLERY=1
  export SYNTHFB_JPG_QUALITY=95
  export SYNTHFB_PNG_COMPRESS=1
fi

# ---- validate ----
if [[ ! -d "$MIMIC_CLEAN" ]]; then
  echo "ERROR: MIMIC_CLEAN dir not found: $MIMIC_CLEAN" >&2
  echo "  Set MIMIC_CLEAN to your MIMIC images root." >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
FILTERED_CSV="${OUT_ROOT}/mimic_fb_free_train.csv"

# Source CSV selection: prefer a pre-built clean pool if one exists, otherwise
# fall back to NLP filtering of MIMIC_META_CSV. An image-side purity-filtered
# pool catches contaminated base images that the NLP gate misses, so supply
# one via CLEAN_POOL_CSV when available.
USE_CLEAN_POOL=0
if [[ -f "$CLEAN_POOL_CSV" ]]; then
  USE_CLEAN_POOL=1
fi

cat <<EOF
===== SynthFB v2 dataset regen =====
MIMIC_CLEAN:     $MIMIC_CLEAN
MIMIC_META_CSV:  $MIMIC_META_CSV
CLEAN_POOL_CSV:  $CLEAN_POOL_CSV  $( ((USE_CLEAN_POOL)) && echo '(present, will use)' || echo '(missing, falling back to NLP filter)' )
OUT_ROOT:        $OUT_ROOT
FILTERED_CSV:    $FILTERED_CSV
N_TRAIN/VAL/TEST: $N_TRAIN / $N_VAL / $N_TEST
MAX_ANN:         $MAX_ANN
SEED:            $SEED
====================================
EOF

# ---- 1. install deps once ----
echo ">> installing deps"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r "${SCRIPT_DIR}/requirements.txt"

# ---- 2. resolve source CSV (clean pool > NLP filter) ----
if (( USE_CLEAN_POOL )); then
  echo ">> using pre-built clean pool: $CLEAN_POOL_CSV"
  cp -f "$CLEAN_POOL_CSV" "$FILTERED_CSV"
elif [[ ! -f "$FILTERED_CSV" ]]; then
  if [[ ! -f "$MIMIC_META_CSV" ]]; then
    echo "ERROR: neither CLEAN_POOL_CSV ($CLEAN_POOL_CSV) nor MIMIC_META_CSV ($MIMIC_META_CSV) found." >&2
    echo "  Set CLEAN_POOL_CSV to your filtered pool (preferred), or MIMIC_META_CSV for NLP filter." >&2
    exit 1
  fi
  echo ">> filtering meta CSV -> $FILTERED_CSV"
  python "${SCRIPT_DIR}/prepare_mimic_subset.py" \
    --src      "$MIMIC_META_CSV" \
    --img-root "$MIMIC_CLEAN" \
    --out      "$FILTERED_CSV"
else
  echo ">> reusing existing filtered CSV"
fi

# ---- 3. run generator for each split ----
run_split() {
  local split="$1"
  local n="$2"
  local seed_offset="$3"
  local out_dir="${OUT_ROOT}/${split}"
  echo ""
  echo "===== SPLIT: $split (N=$n) ====="
  mkdir -p "$out_dir"
  # Per-split gen seed: insertion RNG must differ across splits so val/test
  # don't get the same insertion patterns as train (source images are already
  # disjoint via the CSV's split column).
  MIMIC_CLEAN="$MIMIC_CLEAN" \
  MIMIC_META_CSV="$MIMIC_META_CSV" \
    "${SCRIPT_DIR}/generate_synthfb.sh" \
      -n "$n" \
      -o "$out_dir" \
      -s "$((SEED + seed_offset))" \
      --max-ann "$MAX_ANN" \
      --split "$split" \
      --subset-csv "$FILTERED_CSV" \
      --skip-install \
      --skip-subset
  # Keep a copy of the pool CSV with each split for provenance (also lands in the zip).
  cp -f "$FILTERED_CSV" "$out_dir/mimic_fb_free_train.csv"
}

run_split train "$N_TRAIN" 0
run_split val   "$N_VAL"   1000
run_split test  "$N_TEST"  2000

# ---- 4. zip each split into ZIP_DIR ----
mkdir -p "$ZIP_DIR"
echo ""
echo ">> zipping splits -> $ZIP_DIR"
for split in train val test; do
  split_dir="${OUT_ROOT}/${split}"
  zip_path="${ZIP_DIR}/synthfb_${split}.zip"
  if [[ ! -d "$split_dir" ]]; then
    echo "   skip ${split} (no dir)"; continue
  fi
  echo "   -> ${zip_path}"
  # -j flatten? no — preserve images/ + masks/ subdir structure. -q quiet, -r recursive.
  # Include images/, masks/, annotations.json, mimic_fb_free_train.csv (split filter).
  (cd "$split_dir" && zip -q -r -1 "$zip_path" \
       images masks annotations.json \
       mimic_fb_free_train.csv 2>/dev/null || true)
  size_mb=$(du -m "$zip_path" 2>/dev/null | cut -f1)
  echo "      ${size_mb} MB"
done

echo ""
echo ">> DONE"
echo "   raw:  $OUT_ROOT/{train,val,test}/"
echo "   zips: $ZIP_DIR/synthfb_{train,val,test}.zip"

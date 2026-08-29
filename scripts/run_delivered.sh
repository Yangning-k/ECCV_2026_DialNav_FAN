#!/usr/bin/env bash
# Reproduce the submitted system on one split.
#
#   DATA_ROOT=/path/to/data bash scripts/run_delivered.sh val_unseen
#
# Eight shards, one GPU each, batch 8, seed 0, greedy decoding.  The run is
# deterministic: repeating it reproduces the reported metrics exactly.
#
# DTC_GT is the mean reference dialog length of the split, measured from the
# released annotations, and enters only the efficiency term of the score.
set -uo pipefail

SPLIT="${1:?usage: run_delivered.sh SPLIT, one of val_seen val_unseen test}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$REPO/output/delivered}"

case "$SPLIT" in
  val_seen) DTC_GT=1.9780 ;;
  val_unseen) DTC_GT=1.8423 ;;
  test) DTC_GT=1.5 ;;
  *) echo "unsupported split: $SPLIT" >&2; exit 1 ;;
esac

cd "$REPO" || exit 1
# shellcheck source=../configs/delivered.env
source configs/delivered.env

SHARD_DIR="${DIALNAV_SHARD_DIR:-$REPO/output/shards}"
python3 scripts/make_shards.py \
  --annotation "$DATA_ROOT/dataset/RAIN_holistic/$SPLIT.json" \
  --split "$SPLIT" --output_dir "$SHARD_DIR"
export DIALNAV_SHARD_DIR="$SHARD_DIR"

TAG="delivered_$SPLIT"
mkdir -p "$OUT_ROOT"
scripts/eval_shards.sh "$SPLIT" "$TAG" "$OUT_ROOT"

runs=()
for shard in 0 1 2 3 4 5 6 7; do runs+=("$OUT_ROOT/${TAG}_$shard"); done
python3 scripts/score_runs.py --split "$SPLIT" --dtc_gt "$DTC_GT" \
  --label "$TAG" --runs "${runs[@]}"

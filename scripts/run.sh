#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${DIALNAV_PYTHON:-python}"
MATTERPORT_BUILD="${DIALNAV_MATTERPORT_SIM_BUILD:-}"

SPLIT="${1:-val_unseen}"
GPU="${2:-0}"
OUT_ID="${3:-compliant_${SPLIT}}"
BATCH_SIZE="${4:-8}"
WTA_MODE="${5:-ct_0.6_cap_3}"
GUIDE_SEGMENT_STEPS="${6:-0}"
LOCAL_ANS_NAV="${7:-0}"
LOCAL_ANS_WEIGHT="${8:-0.4}"
MAX_ACTION_LEN="${9:-200}"
LOCAL_ANS_CKPT="${LOCAL_ANS_CKPT:-}"
if [[ "$LOCAL_ANS_NAV" == "1" && -z "$LOCAL_ANS_CKPT" ]]; then
  LOCAL_ANS_CKPT="$REPO_ROOT/assets/fan/local_grounding_final.pth"
fi

export PYTHONPATH="$REPO_ROOT/holistic:$REPO_ROOT:$MATTERPORT_BUILD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$MATTERPORT_BUILD")")/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${DIALNAV_HF_HOME:-$REPO_ROOT/.hf_cache}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
export GTL_SCENE_CACHE_DIR="${DIALNAV_CACHE_DIR:-$REPO_ROOT/.cache/scene_cache}"
export GTL_SCENE_CACHE_MAX="${GTL_SCENE_CACHE_MAX:-16}"
export GTL_PANO_CACHE_MAX="${GTL_PANO_CACHE_MAX:-500}"
export GUIDE_SEGMENT_STEPS="$GUIDE_SEGMENT_STEPS"
export LOCAL_ANS_NAV="$LOCAL_ANS_NAV"
export LOCAL_ANS_WEIGHT="$LOCAL_ANS_WEIGHT"
export LOCAL_ANS_CKPT="$LOCAL_ANS_CKPT"

mkdir -p "$GTL_SCENE_CACHE_DIR" "$HF_HOME" "$REPO_ROOT/output/$OUT_ID"

cd "$REPO_ROOT/holistic"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u main.py \
  --id "$OUT_ID" \
  --output_path "$REPO_ROOT/output/$OUT_ID" \
  --basepath "$REPO_ROOT" \
  --connectivity_dir "$REPO_ROOT/dataset/connectivity/" \
  --val_seen_anno_paths "$REPO_ROOT/dataset/RAIN_holistic/val_seen.json" \
  --val_unseen_anno_paths "$REPO_ROOT/dataset/RAIN_holistic/val_unseen.json" \
  --test_anno_paths "$REPO_ROOT/dataset/RAIN_holistic/test.json" \
  --qa_clip_tokenizer_path "$REPO_ROOT/dataset/modules/clip_tokenizer/bpe_simple_vocab_16e6.txt.gz" \
  --env_names "$SPLIT" \
  --batch_size "$BATCH_SIZE" \
  --max_action_len "$MAX_ACTION_LEN" \
  --nav_resume_file "$REPO_ROOT/dataset/checkpoints/nav_rainbow" \
  --nav_model DST \
  --nav_act_visited_nodes \
  --qg_resume_file "$REPO_ROOT/dataset/checkpoints/q_rainbow" \
  --wta_mode "$WTA_MODE" \
  --ag_resume_file "$REPO_ROOT/dataset/checkpoints/a_rainbow" \
  --loc_resume_file "$REPO_ROOT/dataset/checkpoints/loc_rainbow.pth" \
  --loc_model GTL

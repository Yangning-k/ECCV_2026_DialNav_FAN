#!/usr/bin/env bash
set -euo pipefail

# Run one complete DialNav split on eight pre-pinned GPU containers.
#
# Usage:
#   eval_shards_docker.sh <split> <run_name> <output_root>
#
# The container mapping is fixed to host GPUs 0..7:
#   dialnav_gpu0, dialnav_gpu1, dialnav_gpu2, dialnav_run2,
#   dialnav_gpu4, dialnav_gpu5, dialnav_gpu6, dialnav_run
#
# Every run intentionally uses batch_size=8 and max_action_len=50.  The
# containers expose one host GPU each, so CUDA_VISIBLE_DEVICES is 0 inside
# every container.

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <split> <run_name> <output_root>" >&2
  exit 2
fi

SPLIT="$1"
RUN_NAME="$2"
OUTPUT_ROOT="$3"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${DIALNAV_WORK:-/mnt/data-1/users/wangxiaofeng/dialnav_work}"
DATA_REPO="${DIALNAV_DATA_REPO:-$WORK/data/local_repo}"
PYTHON_BIN="${DIALNAV_PYTHON:-/mnt/data-1/users/wangxiaofeng/miniconda3/envs/navdp/bin/python}"
MATTERPORT_BUILD="${DIALNAV_MATTERPORT_SIM_BUILD:-/shared_disk/users/ning.yang/Codes/Matterport3DSimulator/build}"
SHARD_DIR="${DIALNAV_SHARD_DIR:-$ROOT/output/full_shards}"
SHARD_PATTERN="${DIALNAV_SHARD_PATTERN:-${SPLIT}8_%d.json}"
BATCH_SIZE=8
MAX_ACTION_LEN="${DIALNAV_MAX_ACTION_LEN:-50}"
NUM_SHARDS=8
CPU_CORES_PER_SHARD="${DIALNAV_CPU_CORES_PER_SHARD:-}"
CPU_OFFSET="${DIALNAV_CPU_OFFSET:-0}"

if [[ -n "$CPU_CORES_PER_SHARD" ]] &&
   ! [[ "$CPU_CORES_PER_SHARD" =~ ^[1-9][0-9]*$ ]]; then
  echo "DIALNAV_CPU_CORES_PER_SHARD must be a positive integer" >&2
  exit 2
fi
if ! [[ "$CPU_OFFSET" =~ ^[0-9]+$ ]]; then
  echo "DIALNAV_CPU_OFFSET must be a non-negative integer" >&2
  exit 2
fi
if [[ -n "$CPU_CORES_PER_SHARD" ]]; then
  # Plain `nproc` reports the cgroup CPU quota on this host (8), while
  # taskset can schedule across the full host CPU set (180 CPUs).  Validate
  # the CPU-id layout against the schedulable topology instead of the quota.
  cpu_count="$(nproc --all)"
  required_cpus=$((CPU_OFFSET + NUM_SHARDS * CPU_CORES_PER_SHARD))
  if (( required_cpus > cpu_count )); then
    echo "CPU shard layout needs $required_cpus CPUs, only $cpu_count available" >&2
    exit 2
  fi
fi

# The challenge rules do not cap navigation turns and the score ignores the
# agent's step count, so a longer budget only costs wall-clock time.  The
# allowlist stays in place to catch accidental budgets.
case "$MAX_ACTION_LEN" in
  50|200|300|400|600) ;;
  *)
    echo "DIALNAV_MAX_ACTION_LEN must be 50, 200, 300, 400, or 600; got: $MAX_ACTION_LEN" >&2
    exit 2
    ;;
esac

case "$SPLIT" in
  val_seen) SPLIT_FLAG="--val_seen_anno_paths" ;;
  val_unseen) SPLIT_FLAG="--val_unseen_anno_paths" ;;
  test) SPLIT_FLAG="--test_anno_paths" ;;
  *)
    echo "Unsupported split: $SPLIT (expected val_seen, val_unseen, or test)" >&2
    exit 2
    ;;
esac

CONTAINERS=(
  dialnav_gpu0
  dialnav_gpu1
  dialnav_gpu2
  dialnav_run2
  dialnav_gpu4
  dialnav_gpu5
  dialnav_gpu6
  dialnav_run
)
HOST_GPUS=(0 1 2 3 4 5 6 7)

if [[ ! -f "$ROOT/holistic/main.py" ]]; then
  echo "Holistic entry point not found: $ROOT/holistic/main.py" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"

PYTHONPATH_VALUE="$ROOT/holistic:$ROOT:$ROOT/modules/nav/DST/map_nav_src:$WORK/pylib:$MATTERPORT_BUILD"
if [[ -n "${DIALNAV_EXTRA_PYTHONPATH:-}" ]]; then
  PYTHONPATH_VALUE="$PYTHONPATH_VALUE:$DIALNAV_EXTRA_PYTHONPATH"
fi
LD_LIBRARY_PATH_VALUE="$(dirname "$(dirname "$MATTERPORT_BUILD")")/lib"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LD_LIBRARY_PATH_VALUE="$LD_LIBRARY_PATH_VALUE:$LD_LIBRARY_PATH"
fi

declare -a PIDS=()
declare -a OUT_DIRS=()
PASSTHROUGH_VARS=(
  DIALNAV_WTA_MODE
  GUIDE_SEGMENT_STEPS
  GUIDE_DESC_ENABLED
  GUIDE_DESCRIPTION_FILE
  GUIDE_ARRIVAL_CONFIRM
  GUIDE_ARRIVAL_JUDGE
  GUIDE_ARRIVAL_K
  GUIDE_ARRIVAL_TOPN
  GUIDE_ARRIVAL_MARGIN
  GUIDE_ARRIVAL_RENDER_DIR
  GUIDE_ARRIVAL_MODEL
  GUIDE_ARRIVAL_VIEWS
  GUIDE_ARRIVAL_MAX_CALLS
  GUIDE_ARRIVAL_WORKERS
  GUIDE_ARRIVAL_CACHE
  GUIDE_LOC_CONSISTENCY
  GUIDE_LOC_HISTORY_LIMIT
  GUIDE_LOC_MAX_JUMP
  GUIDE_PLAN_ENABLED
  GUIDE_PLAN_ANS_CKPTS
  GUIDE_PLAN_RERANK_CKPT
  GUIDE_PLAN_ALPHA
  GUIDE_PLAN_K
  GUIDE_PLAN_TEXTS
  GUIDE_PLAN_SCORE_TEXT
  GUIDE_PLAN_RERANK_BATCH
  GUIDE_PLAN_USE_RERANK
  GUIDE_PLAN_DEBUG
  GUIDE_CAPTION_SEARCH
  GUIDE_CAPTION_CONTRAST
  GUIDE_CAPTION_OBJECTS
  GUIDE_CAPTION_TOPK
  GUIDE_CAPTION_MAX_DISTRACTORS
  GUIDE_CAPTION_DEBUG
  GUIDE_CAPTION_TRUTH_PCT
  GUIDE_CAPTION_TEMPLATE
  GUIDE_CAPTION_VERIFY
  GUIDE_H14_FEATURES
  GUIDE_H14_TEXT
  DIALNAV_PROFILE_TIMING
  DIALNAV_TORCH_THREADS
  OMP_NUM_THREADS
  MKL_NUM_THREADS
  OPENBLAS_NUM_THREADS
  NUMEXPR_NUM_THREADS
  VECLIB_MAXIMUM_THREADS
  DIALNAV_CPU_CORES_PER_SHARD
  DIALNAV_CPU_OFFSET
  LOCAL_ANS_NAV
  LOCAL_ANS_CKPT
  LOCAL_ANS_WEIGHT
  LOCAL_ANS_WIDEN
  NAV_GEMINI_Q_ENABLED
  NAV_CONFIRM_KWAY
  NAV_CONFIRM_ASK
  NAV_CONFIRM_TOPK
  NAV_CONFIRM_RESERVE
  NAV_RETRO_STOP
  NAV_RETRO_MIN_STEP
  NAV_RETRO_RESERVE
  NAV_RETRO_CLIP
  NAV_RETRO_CLIP_TOPK
  NAV_RETRO_CLIP_THRESHOLD
  NAV_RETRO_CLIP_TEXT
  NAV_RETRO_GEMINI
  NAV_RETRO_GEMINI_K
  CLIP_STOP_ENABLED
  CLIP_STOP_APPLY_LOGITS
  CLIP_STOP_WEIGHTS
  CLIP_STOP_TOKENIZER
  CLIP_STOP_FEATURES
  NAV_SWEEP_ENABLED
  NAV_SWEEP_START
  NAV_SWEEP_NO_NEW_STEPS
  NAV_SWEEP_REPEAT_WINDOW
  NAV_SWEEP_MAX_REPEAT_NODES
  NAV_SWEEP_MAX_STEPS
  NAV_SWEEP_RESERVE
  GEMINI_STOP_ENABLED
  GEMINI_STOP_MODEL
  GEMINI_STOP_RENDER_DIR
  GEMINI_STOP_MAX_CANDIDATES
  GEMINI_STOP_VIEWS
  GEMINI_STOP_PRESCREEN_K
  GEMINI_STOP_MAX_CALLS
  GEMINI_STOP_CACHE
  GEMINI_STOP_MIN_STEP
  GEMINI_STOP_INTERVAL
  UPDATE_ANSWER_BEHIND
  ANSWER_FORMAT
  GTL_SCENE_CACHE_DIR
  GTL_SCENE_CACHE_MAX
  GTL_PANO_CACHE_MAX
  GTL_NO_CACHE
  HF_HOME
  TRANSFORMERS_OFFLINE
  HF_HUB_OFFLINE
  PYTORCH_CUDA_ALLOC_CONF
  HTTP_PROXY
  HTTPS_PROXY
  NO_PROXY
)
SECRET_VARS=(
  GEMINI_API_KEY
)

for ((index = 0; index < NUM_SHARDS; index++)); do
  annotation_file="$(printf "$SHARD_PATTERN" "$index")"
  annotation_path="$SHARD_DIR/$annotation_file"
  out_dir="$OUTPUT_ROOT/${RUN_NAME}_${index}"
  container="${CONTAINERS[$index]}"
  host_gpu="${HOST_GPUS[$index]}"
  OUT_DIRS+=("$out_dir")

  if [[ ! -f "$annotation_path" ]]; then
    echo "Missing annotation shard: $annotation_path" >&2
    exit 2
  fi
  if [[ -e "$out_dir/submit.json" && "${DIALNAV_OVERWRITE:-0}" != "1" ]]; then
    echo "Refusing to overwrite completed shard: $out_dir" >&2
    echo "Use a new run name or set DIALNAV_OVERWRITE=1." >&2
    exit 2
  fi
  if docker exec "$container" ps -eo args= 2>/dev/null |
      awk '$0 ~ /holistic\/main\.py/ { found = 1 }
           END { exit found ? 0 : 1 }'; then
    echo "Refusing to share $container: another holistic evaluation is running" >&2
    echo "Stop the stale process or use a free container before retrying." >&2
    exit 2
  fi

  mkdir -p "$out_dir"
  command=(
    "$PYTHON_BIN" -u "$ROOT/holistic/main.py"
    --id "${RUN_NAME}_${index}"
    --output_path "$out_dir"
    --basepath "$DATA_REPO"
    --connectivity_dir "$DATA_REPO/dataset/connectivity/"
    --val_seen_anno_paths ""
    --val_unseen_anno_paths ""
    --test_anno_paths ""
    "$SPLIT_FLAG" "$annotation_path"
    --env_names "$SPLIT"
    --batch_size "$BATCH_SIZE"
    --max_action_len "$MAX_ACTION_LEN"
    --nav_resume_file "$DATA_REPO/dataset/checkpoints/nav_rainbow"
    --nav_model DST
    --nav_act_visited_nodes
    --qg_resume_file "$DATA_REPO/dataset/checkpoints/q_rainbow"
    --wta_mode "${DIALNAV_WTA_MODE:-ct_0.6_cap_3}"
    --ag_resume_file "$DATA_REPO/dataset/checkpoints/a_rainbow"
    --loc_resume_file "$DATA_REPO/dataset/checkpoints/loc_rainbow.pth"
    --qa_clip_tokenizer_path "$DATA_REPO/dataset/modules/clip_tokenizer/bpe_simple_vocab_16e6.txt.gz"
    --loc_model GTL
    --benchmark dialnav
    --success_margin 0
    --error_margin 3.0
  )
  if [[ -n "$CPU_CORES_PER_SHARD" ]]; then
    cpu_start=$((CPU_OFFSET + index * CPU_CORES_PER_SHARD))
    cpu_end=$((cpu_start + CPU_CORES_PER_SHARD - 1))
    command=(taskset -c "$cpu_start-$cpu_end" "${command[@]}")
  fi
  printf '%q ' env CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="$PYTHONPATH_VALUE" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH_VALUE" \
    HF_HOME="${HF_HOME:-$WORK/hf_cache}" \
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:64}" \
    GTL_SCENE_CACHE_DIR="${GTL_SCENE_CACHE_DIR:-$WORK/scene_cache}" \
    GTL_SCENE_CACHE_MAX="${GTL_SCENE_CACHE_MAX:-64}" \
    GTL_PANO_CACHE_MAX="${GTL_PANO_CACHE_MAX:-500}" \
  > "$out_dir/command.sh"
  for name in "${PASSTHROUGH_VARS[@]}"; do
    if [[ ${!name+x} == x ]]; then
      value="${!name}"
      if [[ "$name" == "GUIDE_ARRIVAL_CACHE" ||
            "$name" == "GEMINI_STOP_CACHE" ]]; then
        value="${value//%d/$index}"
      fi
      printf '%q ' "$name=$value" >> "$out_dir/command.sh"
    fi
  done
  printf '%q ' "${command[@]}" >> "$out_dir/command.sh"
  printf '\n' >> "$out_dir/command.sh"
  {
    echo "container=$container"
    echo "host_gpu=$host_gpu"
    echo "batch_size=$BATCH_SIZE"
    echo "max_action_len=$MAX_ACTION_LEN"
    echo "cpu_cores_per_shard=${CPU_CORES_PER_SHARD:-unrestricted}"
    echo "cpu_offset=$CPU_OFFSET"
    echo "annotation=$annotation_path"
    echo "command=$out_dir/command.sh"
  } > "$out_dir/launcher.txt"

  exec_env=(
    "CUDA_VISIBLE_DEVICES=0"
    "PYTHONPATH=$PYTHONPATH_VALUE"
    "LD_LIBRARY_PATH=$LD_LIBRARY_PATH_VALUE"
    "HF_HOME=${HF_HOME:-$WORK/hf_cache}"
    "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}"
    "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
    "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:64}"
    "GTL_SCENE_CACHE_DIR=${GTL_SCENE_CACHE_DIR:-$WORK/scene_cache}"
    "GTL_SCENE_CACHE_MAX=${GTL_SCENE_CACHE_MAX:-64}"
    "GTL_PANO_CACHE_MAX=${GTL_PANO_CACHE_MAX:-500}"
  )
  for name in "${PASSTHROUGH_VARS[@]}"; do
    if [[ ${!name+x} == x ]]; then
      value="${!name}"
      if [[ "$name" == "GUIDE_ARRIVAL_CACHE" ||
            "$name" == "GEMINI_STOP_CACHE" ]]; then
        value="${value//%d/$index}"
      fi
      exec_env+=("$name=$value")
    fi
  done
  docker_exec_args=()
  for env_entry in "${exec_env[@]}"; do
    docker_exec_args+=(--env "$env_entry")
  done
  for name in "${SECRET_VARS[@]}"; do
    if [[ ${!name+x} == x ]]; then
      docker_exec_args+=(--env "$name")
    fi
  done
  printf -v command_text '%q ' "${command[@]}"
  log_path="$out_dir/run.log"
  nohup docker exec \
    "${docker_exec_args[@]}" \
    "$container" sh -lc "exec $command_text > $(printf '%q' "$log_path") 2>&1" \
    > "$out_dir/docker_exec.log" 2>&1 < /dev/null &
  pid="$!"
  echo "$pid" > "$out_dir/pid"
  PIDS+=("$pid")
  echo "launched shard=$index gpu=$host_gpu container=$container pid=$pid out=$out_dir"
done

status=0
for ((index = 0; index < NUM_SHARDS; index++)); do
  if wait "${PIDS[$index]}"; then
    echo "completed shard=$index out=${OUT_DIRS[$index]}"
  else
    echo "FAILED shard=$index out=${OUT_DIRS[$index]}" >&2
    status=1
  fi
done

exit "$status"

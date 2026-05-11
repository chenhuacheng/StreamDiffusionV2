#!/bin/bash
set -eu

# StreamDiffusionV2 14B Demo launcher (4-GPU pipe by default).
# Differs from start.sh: passes --config_path / --checkpoint_folder / --noise_scale
# through to main.py so we can target the 14B variant without editing config.py.
#
# Override via env: HOST PORT GPU_IDS STEP USE_TAEHV USE_TENSORRT FAST
#                   CONFIG_PATH CHECKPOINT_FOLDER NOISE_SCALE MODEL_TYPE
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

PORT="${PORT:-7862}"
HOST="${HOST:-0.0.0.0}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
STEP="${STEP:-2}"
MODEL_TYPE="${MODEL_TYPE:-T2V-14B}"
USE_TAEHV="${USE_TAEHV:-1}"
USE_TENSORRT="${USE_TENSORRT:-0}"
FAST="${FAST:-0}"
NOISE_SCALE="${NOISE_SCALE:-0.8}"

CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/wan_causal_dmd_v2v_14b.yaml}"
CHECKPOINT_FOLDER="${CHECKPOINT_FOLDER:-$PROJECT_ROOT/ckpts/wan_causal_dmd_v2v_14b}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"
LOCAL_GPU_IDS="$(seq 0 $((NUM_GPUS - 1)) | paste -sd, -)"

# Skip the frontend build if it has already been produced (the public/ folder
# is populated by `npm run build`); otherwise build it now.
PUBLIC_DIR="$FRONTEND_DIR/public"
if [ ! -f "$PUBLIC_DIR/build/bundle.js" ] && [ ! -f "$PUBLIC_DIR/index.html" ]; then
  cd "$FRONTEND_DIR"
  npm install --no-audit --no-fund
  npm run build
  echo "frontend build success"
else
  echo "frontend build already present, skipping rebuild"
fi

cd "$SCRIPT_DIR"
TAEHV_FLAG=""
case "$(printf '%s' "$USE_TAEHV" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on) TAEHV_FLAG="--use_taehv" ;;
esac

TENSORRT_FLAG=""
case "$(printf '%s' "$USE_TENSORRT" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on) TENSORRT_FLAG="--use_tensorrt" ;;
esac

FAST_FLAG=""
case "$(printf '%s' "$FAST" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on) FAST_FLAG="--fast" ;;
esac

echo "[start_14b] PORT=$PORT HOST=$HOST GPU_IDS=$GPU_IDS NUM_GPUS=$NUM_GPUS STEP=$STEP"
echo "[start_14b] CONFIG_PATH=$CONFIG_PATH"
echo "[start_14b] CHECKPOINT_FOLDER=$CHECKPOINT_FOLDER"
echo "[start_14b] flags: $TAEHV_FLAG $TENSORRT_FLAG $FAST_FLAG"

CUDA_VISIBLE_DEVICES="$GPU_IDS" python main.py \
  --port "$PORT" \
  --host "$HOST" \
  --num_gpus "$NUM_GPUS" \
  --gpu_ids "$LOCAL_GPU_IDS" \
  --step "$STEP" \
  --noise_scale "$NOISE_SCALE" \
  --model_type "$MODEL_TYPE" \
  --config_path "$CONFIG_PATH" \
  --checkpoint_folder "$CHECKPOINT_FOLDER" \
  $TAEHV_FLAG \
  $TENSORRT_FLAG \
  $FAST_FLAG

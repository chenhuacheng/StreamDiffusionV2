#!/bin/bash
set -eu

# StreamDiffusionV2 1.3B Demo launcher (4-GPU pipe by default).
# Mirror of start_14b.sh but targeting the 1.3B variant. Defaults pin the
# service to GPU 4,5,6,7 and port 7863 so it can run side-by-side with the
# 14B service on 0-3 / 7862.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

PORT="${PORT:-7863}"
HOST="${HOST:-0.0.0.0}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
STEP="${STEP:-2}"
MODEL_TYPE="${MODEL_TYPE:-T2V-1.3B}"
USE_TAEHV="${USE_TAEHV:-1}"
USE_TENSORRT="${USE_TENSORRT:-0}"
FAST="${FAST:-0}"
NOISE_SCALE="${NOISE_SCALE:-0.8}"
# Latency / scheduling knobs. Defaults below match the recommended setup
# for the 4-GPU 1.3B service: collect SLO metrics so the monitor can show
# TTFF + e2e distribution, and run one-shot block rebalancing during
# warmup so the pipeline stages settle on a balanced split.
ENABLE_METRICS="${ENABLE_METRICS:-1}"
TARGET_LATENCY="${TARGET_LATENCY:-1.0}"
SCHEDULE_BLOCK="${SCHEDULE_BLOCK:-1}"

CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/wan_causal_dmd_v2v.yaml}"
CHECKPOINT_FOLDER="${CHECKPOINT_FOLDER:-$PROJECT_ROOT/ckpts/wan_causal_dmd_v2v}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"
LOCAL_GPU_IDS="$(seq 0 $((NUM_GPUS - 1)) | paste -sd, -)"

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

METRICS_FLAG=""
case "$(printf '%s' "$ENABLE_METRICS" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on) METRICS_FLAG="--enable-metrics --target-latency $TARGET_LATENCY" ;;
esac

SCHEDULE_BLOCK_FLAG=""
case "$(printf '%s' "$SCHEDULE_BLOCK" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on) SCHEDULE_BLOCK_FLAG="--schedule_block" ;;
esac

echo "[start_1p3b] PORT=$PORT HOST=$HOST GPU_IDS=$GPU_IDS NUM_GPUS=$NUM_GPUS STEP=$STEP"
echo "[start_1p3b] CONFIG_PATH=$CONFIG_PATH"
echo "[start_1p3b] CHECKPOINT_FOLDER=$CHECKPOINT_FOLDER"
echo "[start_1p3b] flags: $TAEHV_FLAG $TENSORRT_FLAG $FAST_FLAG $METRICS_FLAG $SCHEDULE_BLOCK_FLAG"

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
  $FAST_FLAG \
  $METRICS_FLAG \
  $SCHEDULE_BLOCK_FLAG

#!/bin/bash
# Watchdog wrapper around start_14b.sh.
#
# StreamDiffusionV2's online 14B pipe path has a couple of edge-case crashes
# (e.g. KV-cache window collapse on user pause/resume/upload-done sequences:
#   RuntimeError: The expanded size of the tensor (0) must match the existing
#   size (1024) at non-singleton dimension 1
# in models/wan/causal_model.py kv_cache write).
# When the input rank crashes, the FastAPI front door stays up but the
# pipeline is decapitated. For a public demo we'd rather auto-recover.
#
# This wrapper:
#   1. Runs start_14b.sh; on exit, cleans up any leftover python/zombie procs
#      and any lingering CUDA memory by killing the process group.
#   2. Restarts the demo, with exponential-ish backoff (3s, 6s, 12s, capped 30s).
#   3. Logs every restart to outputs/demo_14b_4gpu.log with a banner.
set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/outputs/demo_14b_4gpu.log}"
mkdir -p "$(dirname "$LOG_FILE")"

# Defaults aligned with the main 14B experiment configuration.
export PORT="${PORT:-7862}"
export HOST="${HOST:-0.0.0.0}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export STEP="${STEP:-2}"
export USE_TAEHV="${USE_TAEHV:-1}"
export STREAMV2V_LOCK_RUNTIME_FLAGS="${STREAMV2V_LOCK_RUNTIME_FLAGS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Persist torch.compile / inductor / triton caches across restarts so the
# second-and-onward launch skips the (~60-90s) flex_attention max-autotune
# and Triton kernel codegen, leaving only model weight load on the critical
# path. Caches are scoped per-variant — 14B and 1.3B have different graph
# shapes, sharing them would either miss or hit and run wrong code.
_CACHE_ROOT="$PROJECT_ROOT/outputs/compile_cache/14b"
mkdir -p "$_CACHE_ROOT/inductor" "$_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$_CACHE_ROOT/inductor"
export TRITON_CACHE_DIR="$_CACHE_ROOT/triton"
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"
export TORCHINDUCTOR_AUTOGRAD_CACHE="${TORCHINDUCTOR_AUTOGRAD_CACHE:-1}"
# Persist .pyc bytecode for the giant streamv2v / wan module trees too.
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$PROJECT_ROOT/outputs/pycache}"

# Pin NCCL to loopback to prevent cross-service collisions when 14B and 1.3B
# demos run on the same host with separate process groups. Force-override
# any inherited NCCL_SOCKET_IFNAME from the parent shell.
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

# Model selector metadata — exposed to the frontend via /api/variant so the
# user can jump between 14B (this service on :7862) and 1.3B (:7863).
export DEMO_CURRENT_VARIANT="${DEMO_CURRENT_VARIANT:-14B}"
if [ -z "${DEMO_VARIANTS:-}" ]; then
    export DEMO_VARIANTS='[{"id":"14B","label":"14B (high quality)","port":7862},{"id":"1.3B","label":"1.3B (fast)","port":7863}]'
fi

backoff=3
max_backoff=30
attempt=0

cleanup_residue() {
    # Kill any orphan python / torch inductor children from a previous launch
    # so that GPUs are fully released before we relaunch.
    pkill -9 -f "main.py --port $PORT" 2>/dev/null || true
    pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null || true
    pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
    # Give the kernel a moment to release GPU contexts.
    sleep 2
}

while true; do
    attempt=$((attempt + 1))
    {
        echo
        echo "================================================================"
        echo "[watchdog] $(date '+%Y-%m-%d %H:%M:%S') attempt #$attempt: launching start_14b.sh"
        echo "[watchdog]   PORT=$PORT GPU_IDS=$GPU_IDS STEP=$STEP USE_TAEHV=$USE_TAEHV"
        echo "[watchdog]   STREAMV2V_LOCK_RUNTIME_FLAGS=$STREAMV2V_LOCK_RUNTIME_FLAGS"
        echo "================================================================"
    } >> "$LOG_FILE" 2>&1

    # Run the actual launcher; setsid so we can clean up the whole group on crash.
    setsid bash "$SCRIPT_DIR/start_14b.sh" >> "$LOG_FILE" 2>&1 &
    start_pid=$!

    # Liveness probe: detect the "fastapi alive but a worker is <defunct>"
    # state caused by the upstream causal_model KV-cache edge bug. If fewer
    # than 4 live (non-zombie) children remain under main.py, kill the tree
    # so watchdog can relaunch the full 4-GPU pipeline.
    (
        sleep 180  # allow FSDP + TAEHV load to finish
        while kill -0 "$start_pid" 2>/dev/null; do
            main_pid=$(pgrep -f "main.py --port $PORT" | head -1)
            if [ -n "$main_pid" ]; then
                live=$(ps --ppid "$main_pid" -o pid=,stat= 2>/dev/null | awk '$2 !~ /Z/ {c++} END{print c+0}')
                if [ "$live" -lt 4 ]; then
                    echo "[watchdog] $(date '+%F %T') only $live/4 live workers under main pid=$main_pid, triggering restart" >> "$LOG_FILE"
                    pkill -9 -f "main.py --port $PORT" 2>/dev/null
                    break
                fi
            fi
            sleep 20
        done
    ) &
    probe_pid=$!

    wait "$start_pid"
    rc=$?
    kill "$probe_pid" 2>/dev/null
    wait "$probe_pid" 2>/dev/null

    {
        echo
        echo "[watchdog] $(date '+%Y-%m-%d %H:%M:%S') start_14b.sh exited rc=$rc, cleaning up..."
    } >> "$LOG_FILE" 2>&1

    cleanup_residue

    # If user explicitly Ctrl-C'd the watchdog itself, rc is typically 130 and
    # we want to actually exit. We catch SIGINT/SIGTERM separately below.
    if [ "$rc" -eq 0 ]; then
        echo "[watchdog] start_14b exited rc=0; treating as transient and relaunching." >> "$LOG_FILE"
    fi

    {
        echo "[watchdog] sleeping ${backoff}s before relaunch..."
    } >> "$LOG_FILE" 2>&1
    sleep "$backoff"
    if [ "$backoff" -lt "$max_backoff" ]; then
        backoff=$((backoff * 2))
        if [ "$backoff" -gt "$max_backoff" ]; then
            backoff=$max_backoff
        fi
    fi
done

# Trap signals so killing the watchdog also kills the child.
trap 'echo "[watchdog] caught signal, terminating child group" >> "$LOG_FILE"; pkill -9 -P $$ 2>/dev/null; cleanup_residue; exit 0' INT TERM

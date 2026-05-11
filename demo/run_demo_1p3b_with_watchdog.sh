#!/bin/bash
# Watchdog wrapper around start_1p3b.sh (1.3B variant on port 7863, GPU 4-7).
# See run_demo_with_watchdog.sh for the rationale; this is its 1.3B twin.
set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/outputs/demo_1p3b_4gpu.log}"
mkdir -p "$(dirname "$LOG_FILE")"

export PORT="${PORT:-7863}"
export HOST="${HOST:-0.0.0.0}"
export GPU_IDS="${GPU_IDS:-4,5,6,7}"
export STEP="${STEP:-2}"
export USE_TAEHV="${USE_TAEHV:-1}"
export STREAMV2V_LOCK_RUNTIME_FLAGS="${STREAMV2V_LOCK_RUNTIME_FLAGS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Use a different torch.distributed rendezvous port than the 14B service.
export STREAMV2V_MASTER_PORT="${STREAMV2V_MASTER_PORT:-29510}"

# Persist torch.compile / inductor / triton caches across restarts (see 14B
# watchdog for full rationale). Use a 1.3B-only cache directory so cache
# entries can't be confused with 14B's.
_CACHE_ROOT="$PROJECT_ROOT/outputs/compile_cache/1p3b"
mkdir -p "$_CACHE_ROOT/inductor" "$_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$_CACHE_ROOT/inductor"
export TRITON_CACHE_DIR="$_CACHE_ROOT/triton"
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"
export TORCHINDUCTOR_AUTOGRAD_CACHE="${TORCHINDUCTOR_AUTOGRAD_CACHE:-1}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$PROJECT_ROOT/outputs/pycache}"
# Pin NCCL to loopback so the two demo services (each with their own
# 4-GPU process group) cannot accidentally find each other's sockets on
# the public network interfaces. Without this, NCCL on bond1 has been
# observed to connect across services and abort with ncclSystemError.
# Force-override any inherited NCCL_SOCKET_IFNAME from the parent shell.
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
# Cap NCCL collective-op timeout aggressively. Default is 30 min, but our
# 4-rank single-host comms should complete in seconds. Failing fast lets
# the watchdog/probe restart instead of holding GPUs hostage for 10 min.
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
# 60s — enough for NCCL bootstrap + first all-reduce, far below 600s.
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-60000}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1024}"

# Model selector metadata so both backends can expose /api/variant with a
# consistent peer list.
export DEMO_CURRENT_VARIANT="${DEMO_CURRENT_VARIANT:-1.3B}"
if [ -z "${DEMO_VARIANTS:-}" ]; then
    export DEMO_VARIANTS='[{"id":"14B","label":"14B (high quality)","port":7862},{"id":"1.3B","label":"1.3B (fast)","port":7863}]'
fi

backoff=3
max_backoff=30
attempt=0

cleanup_residue() {
    # Kill the obvious pieces by port/script name first.
    pkill -9 -f "main.py --port $PORT" 2>/dev/null || true
    pkill -9 -f "torch._inductor.compile_worker.*parent=" 2>/dev/null || true
    # Then kill any orphaned multiprocessing.spawn workers from the previous
    # main.py — these keep CUDA contexts open AND keep the NCCL ProcessGroup
    # alive on STREAMV2V_MASTER_PORT, causing the NEXT launch's BROADCAST to
    # hang for the full NCCL_TIMEOUT (we've seen 600s timeouts traced to
    # exactly this). We can't grep them by port (they have generic cmdlines),
    # so we kill all multiprocessing.spawn / resource_tracker / inductor
    # python processes outside the main.py tree.
    pkill -9 -f "from multiprocessing.spawn import spawn_main" 2>/dev/null || true
    pkill -9 -f "from multiprocessing.resource_tracker" 2>/dev/null || true
    pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null || true
    # Kill anything still holding a CUDA context on our GPU set, as a last
    # resort. fuser is the cleanest probe here.
    for dev in /dev/nvidia[4-7]; do
        [ -e "$dev" ] && fuser -k -9 "$dev" 2>/dev/null || true
    done
    sleep 2
}

# Always clean signal handlers up first so cleanup_residue is in scope.
trap 'echo "[watchdog-1p3b] received SIGTERM/SIGINT, cleaning up..." >> "$LOG_FILE"; pkill -9 -P $$ 2>/dev/null; cleanup_residue; exit 0' INT TERM

# Defensive pre-cleanup: if a previous watchdog crashed mid-flight it can
# leave orphaned NCCL workers that would deadlock the next BROADCAST.
echo "[watchdog-1p3b] $(date '+%F %T') pre-launch cleanup..." >> "$LOG_FILE"
cleanup_residue

while true; do
    attempt=$((attempt + 1))
    {
        echo
        echo "================================================================"
        echo "[watchdog-1p3b] $(date '+%Y-%m-%d %H:%M:%S') attempt #$attempt"
        echo "[watchdog-1p3b]   PORT=$PORT GPU_IDS=$GPU_IDS STEP=$STEP USE_TAEHV=$USE_TAEHV"
        echo "================================================================"
    } >> "$LOG_FILE" 2>&1

    setsid bash "$SCRIPT_DIR/start_1p3b.sh" >> "$LOG_FILE" 2>&1 &
    start_pid=$!

    # Background liveness probe: if the fastapi main.py is alive but one of
    # its 4 GPU workers has become <defunct> (known upstream KV-cache edge
    # bug) the service keeps listening on the port but is effectively dead.
    # Detect that and kill the whole tree so watchdog can relaunch.
    (
        # Give the service time to fully load (FSDP shards + TAEHV) before
        # we start expecting 4 live workers.
        sleep 180
        while kill -0 "$start_pid" 2>/dev/null; do
            main_pid=$(pgrep -f "main.py --port $PORT" | head -1)
            if [ -n "$main_pid" ]; then
                # Count live (non-zombie) python children of main.py
                live=$(ps --ppid "$main_pid" -o pid=,stat= 2>/dev/null | awk '$2 !~ /Z/ {c++} END{print c+0}')
                if [ "$live" -lt 4 ]; then
                    echo "[watchdog-1p3b] $(date '+%F %T') only $live/4 live workers under main pid=$main_pid, triggering restart" >> "$LOG_FILE"
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
        echo "[watchdog-1p3b] $(date '+%Y-%m-%d %H:%M:%S') start_1p3b.sh exited rc=$rc, cleaning up..."
    } >> "$LOG_FILE" 2>&1

    cleanup_residue

    if [ "$rc" -eq 0 ]; then
        # rc=0 alone is not enough — we've seen the demo's own internal
        # cleanup paths (e.g. a worker exit propagating through SIGTERM)
        # produce a clean rc=0 even when service is unhealthy. Only stop
        # the watchdog if start_1p3b returned 0 *and* uptime > 600s.
        # Otherwise treat it as a crash and relaunch.
        echo "[watchdog-1p3b] start_1p3b exited rc=0; treating as transient and relaunching." >> "$LOG_FILE"
    fi

    echo "[watchdog-1p3b] sleeping ${backoff}s before relaunch..." >> "$LOG_FILE"
    sleep "$backoff"
    if [ "$backoff" -lt "$max_backoff" ]; then
        backoff=$((backoff * 2))
        [ "$backoff" -gt "$max_backoff" ] && backoff=$max_backoff
    fi
done

# (trap registered above near cleanup_residue definition)

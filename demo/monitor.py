#!/usr/bin/env python3
"""
StreamDiffusionV2 demo 健康监控
- 零依赖（仅 Python 标准库）
- 采集：进程树 / GPU / 端口 / 日志 tail / API 探活 / watchdog 状态
- 提供：
    GET  /            -> HTML 面板（自动 5s 刷新）
    GET  /api/status  -> JSON 全量状态
    GET  /api/log     -> 最近 200 行日志
    GET  /healthz     -> 简版 200/503

Usage:
    python monitor.py --port 9999 \
        --demo-port 7863 --master-port 29510 \
        --gpus 4,5,6,7 \
        --log /root/StreamDiffusionV2/repo/outputs/demo_1p3b_4gpu.log
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime

# ------------------------ 全局配置 ------------------------
ARGS = None
START_TS = time.time()

# 滚动统计
HISTORY = deque(maxlen=720)         # 最近 720 个采样点（每 5s 一次 = 1h）
RESTART_EVENTS = deque(maxlen=50)   # watchdog 重启事件
LAST_STATE = {"snapshot": None, "ts": 0}


# ------------------------ 采集函数 ------------------------
def sh(cmd, timeout=5):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.returncode
    except Exception as e:
        return f"<err:{e}>", -1


def collect_gpus(gpu_ids):
    out, rc = sh(
        "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu "
        "--format=csv,noheader,nounits"
    )
    gpus = []
    if rc != 0:
        return gpus
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 5:
            continue
        idx = int(parts[0])
        if gpu_ids and idx not in gpu_ids:
            continue
        gpus.append({
            "idx": idx,
            "mem_used_mib": int(parts[1]),
            "mem_total_mib": int(parts[2]),
            "util": int(parts[3]),
            "temp": int(parts[4]),
        })
    return gpus


def collect_ports(ports):
    out, _ = sh("ss -ltn")
    listening = set()
    for line in out.splitlines():
        m = re.search(r":(\d+)\s", line)
        if m:
            listening.add(int(m.group(1)))
    return {p: (p in listening) for p in ports}


def collect_processes():
    """匹配 watchdog / start / main / spawn worker"""
    out, _ = sh(
        "ps -eo pid,ppid,etimes,stat,rss,cmd --no-headers"
    )
    rows = []
    pat = re.compile(r"watchdog|start_1p3b|start_14b|main\.py --port|spawn_main|run_demo_")
    for line in out.splitlines():
        if not pat.search(line):
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        try:
            pid = int(parts[0]); ppid = int(parts[1]); etime = int(parts[2])
            stat = parts[3]; rss = int(parts[4]); cmd = parts[5]
        except ValueError:
            continue
        # 过滤 grep / monitor 自己
        if "grep " in cmd or "monitor.py" in cmd:
            continue
        rows.append({
            "pid": pid, "ppid": ppid, "etime_s": etime,
            "stat": stat, "rss_mib": rss // 1024,
            "cmd": cmd[:160],
            "is_zombie": "Z" in stat,
        })
    return rows


def categorize_processes(procs):
    cat = {"watchdog": [], "launcher": [], "main": [], "worker": [], "zombie": []}
    for p in procs:
        cmd = p["cmd"]
        if p["is_zombie"]:
            cat["zombie"].append(p)
        if "watchdog" in cmd:
            cat["watchdog"].append(p)
        elif "start_1p3b" in cmd or "start_14b" in cmd:
            cat["launcher"].append(p)
        elif "main.py --port" in cmd:
            cat["main"].append(p)
        elif "spawn_main" in cmd:
            cat["worker"].append(p)
    return cat


def http_probe(url, timeout=3):
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "monitor"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(2048).decode("utf-8", errors="replace")
            return {"ok": True, "code": resp.status, "ms": int((time.time()-t0)*1000),
                    "body": body[:500]}
    except urllib.error.HTTPError as e:
        return {"ok": False, "code": e.code, "ms": int((time.time()-t0)*1000), "body": str(e)}
    except Exception as e:
        return {"ok": False, "code": 0, "ms": int((time.time()-t0)*1000), "body": str(e)[:200]}


def collect_metrics(demo_port, timeout=3):
    """Fetch /api/metrics/summary from the demo and return a normalised
    dict. Three outcomes:
      * {"status": "ok", ...payload}          -> metrics available
      * {"status": "disabled", "reason": ...} -> demo up but --enable-metrics off
      * {"status": "error",    "reason": ...} -> transport/decode failure
    We never raise: the monitor main loop must keep ticking.
    """
    url = f"http://127.0.0.1:{demo_port}/api/metrics/summary?window_size=500"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "monitor"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(65536).decode("utf-8", errors="replace")
        payload = json.loads(raw)
    except Exception as e:
        return {"status": "error", "reason": str(e)[:200]}

    if not payload.get("enabled", False):
        return {
            "status": "disabled",
            "reason": payload.get("reason", "metrics disabled in demo"),
        }
    return {
        "status": "ok",
        "active_users": payload.get("active_users", 0),
        "e2e": payload.get("e2e_latency", {}),
        "ttff": payload.get("first_frame_latency", {}),
        "hist": payload.get("e2e_histogram", {}),
        "pending": payload.get("pending_inputs_per_user", {}),
    }


def collect_stage_metrics(demo_port, timeout=3):
    """Fetch /api/stage_metrics from the demo.
    Returns the JSON payload on success, or a dict with status/reason on failure.
    """
    url = f"http://127.0.0.1:{demo_port}/api/stage_metrics?window_size=50"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "monitor"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(65536).decode("utf-8", errors="replace")
        payload = json.loads(raw)
    except Exception as e:
        return {"status": "error", "reason": str(e)[:200]}
    if not payload.get("enabled", False):
        return {"status": "disabled", "reason": payload.get("reason", "stage metrics disabled")}
    return {"status": "ok", "data": payload}


def tail_log(path, n=80):
    if not path or not os.path.exists(path):
        return []
    try:
        out, _ = sh(f"tail -n {n} {path!r} | tr '\\r' '\\n' | tail -n {n}")
        return out.splitlines()
    except Exception:
        return []


# ------------------------ Throughput from log ------------------------
# Format we parse:
# 2026-05-11 20:36:13,032 - INFO - [Metrics] Batch 382/1000: current_frames=4,
#    avg_latency=4.4208s, remaining=58, data_count=1524
#
# Semantics (verified against demo/main.py):
#   data_count : monotonic counter of *input* frames received from clients
#                (resets when worker restarts / batch_num rotates back to 1)
#   batch_num  : monotonic batch id, resets every 1000 (=> rotate)
#   remaining  : current queue length (pending input frames not yet consumed)
#   current_frames : frames packed into this batch
#
# From this we derive (rolling window over recent samples):
#   input  fps = Δdata_count / Δt        (real client push rate)
#   output fps = Δbatch_num * cf / Δt    (real worker drain rate)
#   queue      = latest `remaining`, plus its growth/s in window
_METRICS_RE = re.compile(
    r'^(\S+ \S+).*\[Metrics\] Batch (\d+)/\d+: '
    r'current_frames=(\d+), avg_latency=([\d.]+)s, '
    r'remaining=(\d+), data_count=(\d+)'
)


def parse_metrics_tail(path, max_lines=400):
    """Parse the last N [Metrics] lines from the log.

    Returns list of dicts (oldest first):
        {ts: float epoch, batch: int, cf: int, avg_lat_s: float,
         remaining: int, data_count: int}

    Cheap: we just `grep ... | tail -N` so we don't read the whole log.
    """
    if not path or not os.path.exists(path):
        return []
    # grep is much faster than reading the whole file in python
    out, rc = sh(
        f"grep -F '[Metrics] Batch' {path!r} | tail -n {max_lines}"
    )
    if rc != 0 or not out:
        return []
    rows = []
    for line in out.splitlines():
        m = _METRICS_RE.search(line)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
        except Exception:
            continue
        rows.append({
            "ts": ts,
            "batch": int(m.group(2)),
            "cf": int(m.group(3)),
            "avg_lat_s": float(m.group(4)),
            "remaining": int(m.group(5)),
            "data_count": int(m.group(6)),
        })
    return rows


def compute_throughput(rows, window_s=30.0):
    """Compute live input/output FPS + queue stats from recent metrics rows.

    Window: last `window_s` seconds of samples (or all rows if fewer).
    Robust to:
      - batch_num rotating back to 1 (every 1000 batches): we segment.
      - data_count resetting on worker restart: same.
    Returns dict (None values if not enough data):
        input_fps, output_fps, queue, queue_growth_per_s,
        per_frame_ms, samples, window_s, last_ts
    """
    blank = {
        "input_fps": None, "output_fps": None,
        "queue": None, "queue_growth_per_s": None,
        "per_frame_ms": None, "samples": 0,
        "window_s": window_s, "last_ts": None,
        "stale_s": None, "cf": None,
    }
    if not rows:
        return blank

    last_ts = rows[-1]["ts"]
    cutoff = last_ts - window_s
    win = [r for r in rows if r["ts"] >= cutoff]
    if len(win) < 2:
        # fall back: just report queue from the very last line
        blank.update({
            "queue": rows[-1]["remaining"],
            "cf": rows[-1]["cf"],
            "samples": 1,
            "last_ts": last_ts,
            "stale_s": max(0.0, time.time() - last_ts),
        })
        return blank

    # Drop counter resets inside window (keep only the latest contiguous run).
    # A reset is when batch number or data_count goes *backwards*.
    start = 0
    for i in range(1, len(win)):
        if win[i]["batch"] < win[i-1]["batch"] or win[i]["data_count"] < win[i-1]["data_count"]:
            start = i
    win = win[start:]
    if len(win) < 2:
        blank.update({
            "queue": rows[-1]["remaining"],
            "cf": rows[-1]["cf"],
            "samples": 1,
            "last_ts": last_ts,
            "stale_s": max(0.0, time.time() - last_ts),
        })
        return blank

    t0 = win[0]["ts"]; t1 = win[-1]["ts"]
    dt = t1 - t0
    if dt <= 0:
        return blank
    cf_counts = {}
    for r in win:
        cf_counts[r["cf"]] = cf_counts.get(r["cf"], 0) + 1
    cf_mode = max(cf_counts, key=cf_counts.get)
    n_batches = win[-1]["batch"] - win[0]["batch"]
    n_frames_in = win[-1]["data_count"] - win[0]["data_count"]
    n_frames_out = n_batches * cf_mode
    input_fps = n_frames_in / dt if dt > 0 else None
    output_fps = n_frames_out / dt if dt > 0 else None
    queue_growth = (win[-1]["remaining"] - win[0]["remaining"]) / dt
    per_frame_ms = (dt / n_frames_out * 1000.0) if n_frames_out > 0 else None

    return {
        "input_fps": round(input_fps, 2) if input_fps is not None else None,
        "output_fps": round(output_fps, 2) if output_fps is not None else None,
        "queue": rows[-1]["remaining"],
        "queue_growth_per_s": round(queue_growth, 2),
        "per_frame_ms": round(per_frame_ms, 1) if per_frame_ms is not None else None,
        "samples": len(win),
        "window_s": round(dt, 1),
        "last_ts": last_ts,
        "stale_s": round(max(0.0, time.time() - last_ts), 1),
        "cf": cf_mode,
    }


def scan_log_health(path, max_bytes=1_000_000):
    """看日志最后 1MB 的关键错误数 / restart 事件"""
    if not path or not os.path.exists(path):
        return {"size_mb": 0, "errors": {}, "restart_attempts": 0, "last_attempt_line": ""}
    sz = os.path.getsize(path)
    try:
        with open(path, "rb") as f:
            if sz > max_bytes:
                f.seek(-max_bytes, 2)
            data = f.read().decode("utf-8", errors="replace")
    except Exception:
        return {"size_mb": round(sz/1e6, 2), "errors": {}, "restart_attempts": 0, "last_attempt_line": ""}

    errors = {
        "EADDRINUSE": data.count("EADDRINUSE"),
        "Traceback": data.count("Traceback"),
        "RuntimeError": data.count("RuntimeError"),
        "NCCL timeout": data.count("NCCL") + data.count("Watchdog"),
        "OOM": data.count("OutOfMemoryError") + data.count("CUDA out of memory"),
    }
    attempts = re.findall(r"attempt\s*#?\s*(\d+)", data, re.IGNORECASE)
    last_attempt = ""
    m = list(re.finditer(r".*attempt.*$", data, re.IGNORECASE | re.MULTILINE))
    if m:
        last_attempt = m[-1].group(0)[-200:]
    return {
        "size_mb": round(sz/1e6, 2),
        "errors": errors,
        "restart_attempts": len(attempts),
        "last_attempt_line": last_attempt,
    }


def overall_status(snap):
    """汇总绿/黄/红
    关键指标：demo_port 在 listen + API 200 + main 进程 + 期望数量 worker + 无 zombie。
    master_port (NCCL TCPStore) 仅作 info：rendezvous 完成后会关闭，不算降级。
    """
    demo_port_ok = snap["ports"].get(ARGS.demo_port, False)
    api_ok = snap["api"]["root"]["ok"] if snap.get("api") else False
    main_ok = len(snap["procs_cat"]["main"]) >= 1
    workers_ok = len(snap["procs_cat"]["worker"]) >= snap["expected_workers"]
    zombies = len(snap["procs_cat"]["zombie"])
    wd_ok = len(snap["procs_cat"]["watchdog"]) >= 1

    if not (main_ok and api_ok and demo_port_ok):
        return "RED", "demo down (main/api/port lost)"
    if zombies > 0:
        return "RED", f"zombie workers={zombies}"
    if not workers_ok:
        return "YELLOW", f"only {len(snap['procs_cat']['worker'])}/{snap['expected_workers']} workers"
    if not wd_ok:
        return "YELLOW", "running but no watchdog (no auto-restart)"
    return "GREEN", "all systems nominal"


def collect_launch_config():
    """从 demo main.py 进程的 /proc/<pid>/environ 中提取启动配置参数。"""
    # 感兴趣的环境变量
    keys = [
        "STEP", "MODEL_TYPE", "GPU_IDS", "USE_TAEHV", "USE_TENSORRT",
        "FAST", "SCHEDULE_BLOCK", "STREAMV2V_LOCK_RUNTIME_FLAGS",
        "TARGET_LATENCY", "ENABLE_METRICS", "PORT", "HOST",
        "STREAMV2V_MASTER_PORT", "STREAMV2V_NCCL_TIMEOUT_SEC",
        "CONFIG_PATH", "CHECKPOINT_FOLDER", "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE",
        "PYTHONPATH",
    ]
    # 找 main.py 进程的 PID
    out, rc = sh("pgrep -f 'main.py --port'")
    if rc != 0 or not out.strip():
        return {}
    pid = out.strip().splitlines()[0]
    try:
        with open(f"/proc/{pid}/environ", "r") as f:
            env_data = f.read()
        env_pairs = env_data.split("\0")
        env_dict = {}
        for pair in env_pairs:
            if "=" in pair:
                k, v = pair.split("=", 1)
                env_dict[k] = v
        config = {}
        for k in keys:
            if k in env_dict:
                config[k] = env_dict[k]
        return config
    except Exception:
        return {}


# 缓存启动配置，按 demo PID 变化自动刷新（demo 重启后无需重启 monitor）
_LAUNCH_CONFIG_CACHE = None
_LAUNCH_CONFIG_PID = None


def _get_demo_pid():
    """获取当前 demo main.py 进程的 PID，找不到返回 None。"""
    out, rc = sh("pgrep -f 'main.py --port'")
    if rc != 0 or not out.strip():
        return None
    return out.strip().splitlines()[0]


def get_launch_config():
    global _LAUNCH_CONFIG_CACHE, _LAUNCH_CONFIG_PID
    current_pid = _get_demo_pid()
    # PID 变了（demo 重启）或从未采集过 → 重新读取
    if current_pid != _LAUNCH_CONFIG_PID or _LAUNCH_CONFIG_CACHE is None:
        _LAUNCH_CONFIG_CACHE = collect_launch_config()
        _LAUNCH_CONFIG_PID = current_pid
    return _LAUNCH_CONFIG_CACHE


def take_snapshot():
    snap = {"ts": int(time.time()), "iso": datetime.now().isoformat(timespec="seconds")}
    snap["launch_config"] = get_launch_config()
    snap["gpus"] = collect_gpus(ARGS.gpu_ids_set)
    snap["ports"] = collect_ports([ARGS.demo_port, ARGS.master_port])
    procs = collect_processes()
    snap["procs"] = procs
    snap["procs_cat"] = categorize_processes(procs)
    snap["expected_workers"] = len(ARGS.gpu_ids_set) if ARGS.gpu_ids_set else 4
    snap["api"] = {
        "root": http_probe(f"http://127.0.0.1:{ARGS.demo_port}/", timeout=3),
        "queue": http_probe(f"http://127.0.0.1:{ARGS.demo_port}/api/queue", timeout=3),
    }
    # Pull latency metrics from the demo itself. This endpoint only exists
    # when the demo was started with --enable-metrics; we handle three
    # situations:
    #   * 200 + enabled=true    -> full stats
    #   * 200 + enabled=false   -> metrics disabled (surface a friendly msg)
    #   * anything else         -> surface the transport error
    snap["metrics"] = collect_metrics(ARGS.demo_port)
    snap["stage_metrics"] = collect_stage_metrics(ARGS.demo_port)
    # Live throughput (input/output FPS + queue) parsed from the demo log.
    # Cheap: grep tail of the file. Window default 30s.
    metric_rows = parse_metrics_tail(ARGS.log, max_lines=400)
    snap["throughput"] = compute_throughput(metric_rows, window_s=ARGS.throughput_window_s)
    snap["log"] = scan_log_health(ARGS.log)
    snap["status"], snap["status_reason"] = overall_status(snap)
    snap["monitor_uptime_s"] = int(time.time() - START_TS)
    return snap


# ------------------------ 后台采集 ------------------------
def sampler_loop():
    last_attempts = 0
    while True:
        try:
            snap = take_snapshot()
            m = snap.get("metrics") or {}
            e2e = m.get("e2e") or {}
            ttff = m.get("ttff") or {}
            tp = snap.get("throughput") or {}
            HISTORY.append({
                "ts": snap["ts"],
                "status": snap["status"],
                "gpu_mem": [g["mem_used_mib"] for g in snap["gpus"]],
                "gpu_util": [g["util"] for g in snap["gpus"]],
                "workers": len(snap["procs_cat"]["worker"]),
                "zombies": len(snap["procs_cat"]["zombie"]),
                "api_root_ok": snap["api"]["root"]["ok"],
                "errors_total": sum(snap["log"]["errors"].values()),
                # NOTE: these are None / 0 when metrics disabled or no samples
                "e2e_p50_ms": e2e.get("p50_ms"),
                "e2e_p95_ms": e2e.get("p95_ms"),
                "e2e_p99_ms": e2e.get("p99_ms"),
                "ttff_last_ms": ttff.get("last_ms"),
                # live throughput (parsed from demo log)
                "input_fps":  tp.get("input_fps"),
                "output_fps": tp.get("output_fps"),
                "queue":      tp.get("queue"),
            })
            cur_attempts = snap["log"]["restart_attempts"]
            if cur_attempts > last_attempts:
                RESTART_EVENTS.append({
                    "ts": snap["ts"], "iso": snap["iso"],
                    "attempt_no": cur_attempts,
                    "line": snap["log"]["last_attempt_line"],
                })
                last_attempts = cur_attempts
            LAST_STATE["snapshot"] = snap
            LAST_STATE["ts"] = snap["ts"]
        except Exception as e:
            LAST_STATE["snapshot"] = {"error": str(e), "ts": int(time.time())}
        time.sleep(ARGS.interval)


# ------------------------ HTML 面板 ------------------------
HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>StreamDiffusionV2 monitor</title>
<style>
body{font:13px/1.4 -apple-system,Menlo,monospace;background:#0d1117;color:#c9d1d9;margin:0;padding:16px}
h1{font-size:16px;margin:0 0 12px;color:#58a6ff}
h2{font-size:13px;margin:18px 0 6px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.card{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:10px 12px}
.kv{display:flex;justify-content:space-between;padding:2px 0}
.kv span:first-child{color:#8b949e}
.ok{color:#3fb950}.warn{color:#d29922}.bad{color:#f85149}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-weight:600}
.badge.GREEN{background:#0f5132;color:#7ee2a8}
.badge.YELLOW{background:#664d03;color:#ffda6a}
.badge.RED{background:#842029;color:#ea868f}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:3px 6px;text-align:left;border-bottom:1px solid #21262d}
th{color:#8b949e;font-weight:500}
.bar{display:inline-block;width:80px;height:8px;background:#21262d;border-radius:4px;overflow:hidden;vertical-align:middle}
.bar>i{display:block;height:100%;background:#3fb950}
.bar>i.warn{background:#d29922}.bar>i.bad{background:#f85149}
pre{background:#0d1117;border:1px solid #30363d;border-radius:4px;padding:8px;max-height:300px;overflow:auto;font-size:11px}
.row{display:flex;gap:16px;flex-wrap:wrap;align-items:center}
small{color:#6e7681}
a{color:#58a6ff;text-decoration:none}
</style></head><body>
<h1>StreamDiffusionV2 demo monitor &nbsp;<span id="badge" class="badge"></span> &nbsp;<small id="meta"></small></h1>
<div class="row">
  <small>auto-refresh 5s</small>
  <small><a href="/api/status" target="_blank">JSON</a></small>
  <small><a href="/api/log" target="_blank">log tail</a></small>
  <small><a href="/healthz" target="_blank">healthz</a></small>
  <small><a href="/download/pipeline_analysis.md" download style="color:#3fb950;font-weight:600">📄 Pipeline 分析报告</a></small>
  <small><a href="/download/chat_summary.md" download="StreamDiffusionV2_多卡分析.md" style="background:#238636;color:#fff;padding:2px 10px;border-radius:4px;font-weight:600">⬇ 多卡并行分析文档</a></small>
</div>

<div class="grid" style="margin-top:12px">
  <div class="card"><h2>summary</h2><div id="summary"></div></div>
  <div class="card"><h2>ports & api</h2><div id="ports"></div></div>
  <div class="card"><h2>log health</h2><div id="loghealth"></div></div>
  <div class="card"><h2>live throughput</h2><div id="throughput"></div></div>
  <div class="card" style="grid-column:span 2"><h2>launch config (启动配置)</h2><div id="launchcfg"></div></div>
</div>

<h2>latency</h2>
<div class="grid">
  <div class="card"><h2>first-frame latency (TTFF)</h2><div id="ttff"></div></div>
  <div class="card" style="grid-column:span 2"><h2>end-to-end latency distribution</h2>
    <div id="e2estats" style="margin-bottom:8px"></div>
    <table id="histtab"><thead><tr><th>bucket (ms)</th><th>count</th><th style="width:55%">bar</th></tr></thead><tbody></tbody></table>
  </div>
</div>

<h2>18-stage pipeline breakdown</h2>
<div class="card" style="margin-bottom:12px">
  <div id="stagesummary" style="margin-bottom:8px"></div>
  <table id="stagetab"><thead><tr>
    <th>#</th><th>stage</th><th>type</th><th style="width:8%">mean</th><th style="width:8%">min</th><th style="width:8%">max</th><th style="width:40%">bar</th>
  </tr></thead><tbody></tbody></table>
</div>

<h2>GPUs</h2>
<div class="card"><table id="gputab"><thead><tr>
<th>idx</th><th>mem</th><th>used</th><th>util</th><th>temp</th></tr></thead><tbody></tbody></table></div>

<h2>processes</h2>
<div class="card"><table id="proctab"><thead><tr>
<th>role</th><th>pid</th><th>ppid</th><th>etime</th><th>stat</th><th>rss</th><th>cmd</th></tr></thead><tbody></tbody></table></div>

<h2>recent restart events</h2>
<div class="card" id="restarts"></div>

<h2>log tail (live)</h2>
<pre id="logtail">loading...</pre>

<script>
function fmtTime(s){if(!s)return '-';const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),x=s%60;return (h?h+'h':'')+m+'m'+x+'s'}
function bar(v,max,cls){const w=Math.min(100,Math.round(v/max*100));return `<span class="bar"><i class="${cls||''}" style="width:${w}%"></i></span>`}
async function refresh(){
  const r=await fetch('/api/status').then(r=>r.json()).catch(e=>({error:String(e)}));
  if(r.error){document.getElementById('summary').textContent=r.error;return}
  const s=r;
  document.getElementById('badge').textContent=s.status;
  document.getElementById('badge').className='badge '+s.status;
  document.getElementById('meta').textContent=`updated ${s.iso}  ·  monitor up ${fmtTime(s.monitor_uptime_s)}  ·  ${s.status_reason}`;

  // summary
  const cat=s.procs_cat;
  document.getElementById('summary').innerHTML=
    kv('watchdog',cat.watchdog.length, cat.watchdog.length?'ok':'bad')+
    kv('launcher',cat.launcher.length, cat.launcher.length?'ok':'warn')+
    kv('main.py',cat.main.length, cat.main.length?'ok':'bad')+
    kv('workers (live)',cat.worker.length+' / '+s.expected_workers, cat.worker.length>=s.expected_workers?'ok':'bad')+
    kv('zombies',cat.zombie.length, cat.zombie.length?'bad':'ok');

  // ports & api
  document.getElementById('ports').innerHTML=
    Object.entries(s.ports).map(([p,ok])=>kv('port '+p, ok?'LISTEN':'down', ok?'ok':'bad')).join('')+
    kv('GET /', `${s.api.root.code} (${s.api.root.ms}ms)`, s.api.root.ok?'ok':'bad')+
    kv('GET /api/queue', `${s.api.queue.code} (${s.api.queue.ms}ms)`, s.api.queue.ok?'ok':'bad');

  // log health
  const e=s.log.errors;
  document.getElementById('loghealth').innerHTML=
    kv('log size','&nbsp;'+s.log.size_mb+' MB')+
    kv('restart attempts',s.log.restart_attempts, s.log.restart_attempts>0?'warn':'ok')+
    kv('EADDRINUSE',e.EADDRINUSE, e.EADDRINUSE?'bad':'ok')+
    kv('Traceback',e.Traceback, e.Traceback?'warn':'ok')+
    kv('RuntimeError',e.RuntimeError, e.RuntimeError?'warn':'ok')+
    kv('OOM',e.OOM, e.OOM?'bad':'ok');

  // live throughput (input fps / output fps / queue) — parsed from demo log
  renderThroughput(s.throughput||{});

  // launch config
  renderLaunchConfig(s.launch_config||{});

  // latency: first-frame + e2e histogram
  renderLatency(s.metrics||{status:'error',reason:'no metrics payload'});

  // 18-stage pipeline breakdown
  renderStages(s.stage_metrics||{status:'error',reason:'no stage metrics'});

  // gpus
  document.querySelector('#gputab tbody').innerHTML=s.gpus.map(g=>{
    const pct=g.mem_used_mib/g.mem_total_mib;
    const cls=pct>0.9?'bad':(pct>0.7?'warn':'');
    return `<tr><td>${g.idx}</td><td>${bar(g.mem_used_mib,g.mem_total_mib,cls)}</td>
      <td>${g.mem_used_mib} / ${g.mem_total_mib} MiB</td>
      <td>${bar(g.util,100,g.util>90?'warn':'')} ${g.util}%</td>
      <td>${g.temp}°C</td></tr>`;
  }).join('');

  // procs
  const role=p=>{
    if(p.is_zombie)return '<span class="bad">ZOMBIE</span>';
    if(p.cmd.includes('watchdog'))return 'watchdog';
    if(p.cmd.includes('start_1p3b')||p.cmd.includes('start_14b'))return 'launcher';
    if(p.cmd.includes('main.py --port'))return 'main';
    if(p.cmd.includes('spawn_main'))return 'worker';
    return '?';
  };
  document.querySelector('#proctab tbody').innerHTML=s.procs.map(p=>
    `<tr><td>${role(p)}</td><td>${p.pid}</td><td>${p.ppid}</td>
     <td>${fmtTime(p.etime_s)}</td><td>${p.stat}</td><td>${p.rss_mib} MiB</td>
     <td><code>${p.cmd.replace(/</g,'&lt;')}</code></td></tr>`).join('');

  // restarts
  const r2=await fetch('/api/restarts').then(r=>r.json()).catch(_=>[]);
  document.getElementById('restarts').innerHTML = r2.length
    ? r2.slice(-10).reverse().map(x=>`<div><small>${x.iso}</small> attempt #${x.attempt_no} <code>${(x.line||'').replace(/</g,'&lt;')}</code></div>`).join('')
    : '<small>(no restarts recorded since monitor start)</small>';

  // log tail
  const t=await fetch('/api/log').then(r=>r.text()).catch(_=>'');
  const pre=document.getElementById('logtail');
  pre.textContent=t;
  pre.scrollTop=pre.scrollHeight;
}
function kv(k,v,cls){return `<div class="kv"><span>${k}</span><span class="${cls||''}">${v}</span></div>`}

function renderLaunchConfig(cfg){
  const el=document.getElementById('launchcfg');
  if(!cfg || Object.keys(cfg).length===0){
    el.innerHTML='<small>unable to read launch config (demo process not found)</small>';
    return;
  }
  // 高亮关键参数
  const highlights = {
    'STEP': v=> kv('STEP', v, 'ok'),
    'MODEL_TYPE': v=> kv('MODEL_TYPE', v),
    'GPU_IDS': v=> kv('GPU_IDS', v),
    'USE_TAEHV': v=> kv('USE_TAEHV', v, v==='1'?'ok':'warn'),
    'USE_TENSORRT': v=> kv('USE_TENSORRT', v, v==='1'?'ok':''),
    'SCHEDULE_BLOCK': v=> kv('SCHEDULE_BLOCK', v, v==='0'?'ok':'warn'),
    'STREAMV2V_LOCK_RUNTIME_FLAGS': v=> kv('LOCK_RUNTIME_FLAGS', v, v==='1'?'ok':'warn'),
    'TARGET_LATENCY': v=> kv('TARGET_LATENCY', v+'s'),
    'FAST': v=> kv('FAST', v),
    'ENABLE_METRICS': v=> kv('ENABLE_METRICS', v, v==='1'?'ok':'warn'),
  };
  // 优先显示关键参数
  const priority=['STEP','MODEL_TYPE','GPU_IDS','USE_TAEHV','USE_TENSORRT','SCHEDULE_BLOCK','STREAMV2V_LOCK_RUNTIME_FLAGS','TARGET_LATENCY','FAST','ENABLE_METRICS'];
  let html='';
  for(const k of priority){
    if(cfg[k]!==undefined){
      html += (highlights[k] || (v=>kv(k,v)))(cfg[k]);
    }
  }
  // 其余参数
  const rest=Object.keys(cfg).filter(k=>!priority.includes(k)).sort();
  for(const k of rest){
    html += kv(k, cfg[k].length>60 ? cfg[k].slice(0,60)+'…' : cfg[k]);
  }
  el.innerHTML=html;
}

function renderThroughput(t){
  const el=document.getElementById('throughput');
  if(!t || (t.input_fps===null && t.output_fps===null && t.queue===null)){
    el.innerHTML='<small>no [Metrics] log lines yet (waiting for first batch)</small>';
    return;
  }
  // Color hints:
  //   queue:        >50 bad, >20 warn
  //   queue growth: >+0.3/s bad (堆积), >+0.05 warn
  //   input vs output: input>output+0.3 -> warn (worker 跟不上)
  const q   = t.queue;
  const qg  = t.queue_growth_per_s;
  const inF = t.input_fps;
  const ouF = t.output_fps;
  const pf  = t.per_frame_ms;
  const stale = t.stale_s;

  const qCls  = (q===null) ? '' : (q>50?'bad':(q>20?'warn':'ok'));
  const qgCls = (qg===null)? '' : (qg>0.3?'bad':(qg>0.05?'warn':(qg<-0.05?'ok':'')));
  let inCls='', ouCls='';
  if(inF!==null && ouF!==null){
    if(inF > ouF + 0.3) { inCls='warn'; ouCls='warn'; }   // 输入快于输出 → 队列要堆
    else if (Math.abs(inF-ouF)<=0.3) { inCls='ok'; ouCls='ok'; }
  }
  const staleCls = (stale!==null && stale>15) ? 'warn' : '';

  el.innerHTML=
    kv('input fps (client push)',  inF===null?'-':inF.toFixed(2)+' fps', inCls)+
    kv('output fps (worker drain)',ouF===null?'-':ouF.toFixed(2)+' fps', ouCls)+
    kv('per-frame proc',           pf===null?'-':pf.toFixed(1)+' ms')+
    kv('queue (pending)',          q===null?'-':q+' frames', qCls)+
    kv('queue growth',             qg===null?'-':(qg>0?'+':'')+qg.toFixed(2)+' /s', qgCls)+
    kv('window',                   `${t.window_s||0}s · ${t.samples||0} samples` + (t.cf?` · cf=${t.cf}`:''))+
    kv('last log age',             stale===null?'-':stale.toFixed(0)+' s ago', staleCls);
}

function fmtMs(x){
  if(x===undefined||x===null) return '-';
  if(x<1000) return x.toFixed(1)+' ms';
  return (x/1000).toFixed(2)+' s';
}
function latencyClass(ms, target_ms){
  if(ms===undefined||ms===null) return '';
  if(!target_ms) target_ms=1000; // fallback 1s target
  if(ms > target_ms) return 'bad';
  if(ms > target_ms*0.7) return 'warn';
  return 'ok';
}
function renderLatency(m){
  const ttff=document.getElementById('ttff');
  const e2eStats=document.getElementById('e2estats');
  const histBody=document.querySelector('#histtab tbody');

  if(m.status!=='ok'){
    const msg = m.status==='disabled'
      ? `metrics disabled in demo &mdash; <small>${m.reason||''}</small>`
      : `metrics unavailable &mdash; <small>${m.reason||m.status||'unknown'}</small>`;
    ttff.innerHTML=`<div class="${m.status==='disabled'?'warn':'bad'}">${msg}</div>`;
    e2eStats.innerHTML='';
    histBody.innerHTML='';
    return;
  }

  // TTFF card
  const t=m.ttff||{};
  if(!t.count){
    ttff.innerHTML='<small>no first-frame samples yet (waiting for first session output)</small>';
  } else {
    const tgtMs=(m.e2e && m.e2e.deadline_s) ? m.e2e.deadline_s*1000 : 1000;
    ttff.innerHTML=
      kv('samples', t.count)+
      kv('last',    fmtMs(t.last_ms),   latencyClass(t.last_ms, tgtMs))+
      kv('mean',    fmtMs(t.mean_ms),   latencyClass(t.mean_ms, tgtMs))+
      kv('p50',     fmtMs(t.p50_ms),    latencyClass(t.p50_ms,  tgtMs))+
      kv('p95',     fmtMs(t.p95_ms),    latencyClass(t.p95_ms,  tgtMs))+
      kv('p99',     fmtMs(t.p99_ms),    latencyClass(t.p99_ms,  tgtMs))+
      kv('max',     fmtMs(t.max_ms),    latencyClass(t.max_ms,  tgtMs));
  }

  // E2E stats header
  const e=m.e2e||{};
  if(!e.count){
    e2eStats.innerHTML='<small>no e2e samples yet (connect a client + send frames)</small>';
    histBody.innerHTML='';
    return;
  }
  const tgtMs=(e.deadline_s||1.0)*1000;
  const missPct=(e.deadline_miss_rate*100).toFixed(2);
  const missCls=e.deadline_miss_rate>0.01 ? 'bad' : (e.deadline_miss_rate>0?'warn':'ok');
  e2eStats.innerHTML=
    `<div class="row">
       <span>n=${e.count}</span>
       <span>mean <b class="${latencyClass(e.mean_ms,tgtMs)}">${fmtMs(e.mean_ms)}</b></span>
       <span>p50 <b class="${latencyClass(e.p50_ms,tgtMs)}">${fmtMs(e.p50_ms)}</b></span>
       <span>p95 <b class="${latencyClass(e.p95_ms,tgtMs)}">${fmtMs(e.p95_ms)}</b></span>
       <span>p99 <b class="${latencyClass(e.p99_ms,tgtMs)}">${fmtMs(e.p99_ms)}</b></span>
       <span>max <b class="${latencyClass(e.max_ms,tgtMs)}">${fmtMs(e.max_ms)}</b></span>
       <span>deadline ${fmtMs(tgtMs)}  miss <b class="${missCls}">${missPct}%</b> (${Math.round(e.deadline_miss_rate*e.count)}/${e.count})</span>
     </div>`;

  // Histogram bars
  const h=m.hist||{};
  const edges=h.bin_edges_ms||[];
  const counts=h.counts||[];
  const total=counts.reduce((a,b)=>a+b,0)||1;
  const maxCnt=Math.max.apply(null,counts.concat([1]));
  let rows='';
  for(let i=0;i<edges.length;i++){
    const lo=edges[i];
    const hi=(i+1<edges.length)?edges[i+1]:null;
    const label=hi===null ? `&ge; ${lo}` : `${lo} &ndash; ${hi}`;
    const c=counts[i]||0;
    const pct=(c/total*100).toFixed(1);
    const w=Math.round(c/maxCnt*100);
    // Colour: bars past the deadline are red; bars at 70%+ of deadline are yellow.
    const cls = (lo >= tgtMs) ? 'bad' : (lo >= tgtMs*0.7 ? 'warn' : '');
    rows += `<tr><td>${label}</td><td>${c} <small>(${pct}%)</small></td>
             <td><span class="bar" style="width:95%"><i class="${cls}" style="width:${w}%"></i></span></td></tr>`;
  }
  histBody.innerHTML=rows;
}

// ── 18-stage pipeline breakdown ──
const STAGE_LABELS = {
  s01_camera_capture_ms:    '① Camera capture (getUserMedia)',
  s02_jpeg_encode_ms:       '② JPEG encode (canvas.toBlob)',
  s03_ws_recv_decode_ms:    '③ WS recv + JPEG decode + throttle',
  s04_update_data_ms:       '④ update_data (param merge)',
  s05_push_pipeline_ms:     '⑤ accept_new_params (push to pipeline)',
  s06_input_queue_wait_ms:  '⑥ input_queue wait (included in ⑦)',
  s07_read_queue_ms:        '⑦ read_images_from_queue (GPU side)',
  s08_vae_encode_ms:        '⑧ VAE encode (rank 0)',
  s09_dit_rank0_ms:         '⑨ DiT rank 0',
  s10_nccl_send_ms:         '⑩ NCCL send_latent_data',
  s11_dit_last_rank_ms:     '⑪ DiT last rank',
  s12_vae_decode_ms:        '⑫ VAE decode (last rank)',
  s13_output_queue_put_ms:  '⑬ output_queue.put',
  s14_produce_outputs_ms:   '⑭ produce_outputs (main thread)',
  s15_pil_to_frame_ms:      '⑮ pil_to_frame (JPEG encode)',
  s16_output_yield_ms:      '⑯ output_queue → MJPEG yield',
  s17_http_chunked_ms:      '⑰ HTTP chunked transfer',
  s18_frontend_decode_ms:   '⑱ Frontend decode + render',
};
const STAGE_ORDER = Object.keys(STAGE_LABELS);

function renderStages(sm) {
  const summary = document.getElementById('stagesummary');
  const tbody = document.querySelector('#stagetab tbody');
  if (!sm || sm.status !== 'ok') {
    const reason = (sm && sm.reason) || (sm && sm.status) || 'unavailable';
    summary.innerHTML = `<small class="warn">stage metrics: ${reason}</small>`;
    tbody.innerHTML = '';
    return;
  }
  const d = sm.data || {};
  const avgs = d.averages || {};
  const ta = d.total_avg || {};
  // Summary line
  summary.innerHTML =
    `<div class="row">
       <span>samples: <b>${d.samples||0}</b></span>
       <span>measured total: <b class="ok">${ta.measured_ms!=null?ta.measured_ms.toFixed(1)+' ms':'-'}</b></span>
       <span>estimated total: <b class="warn">${ta.estimated_ms!=null?ta.estimated_ms.toFixed(1)+' ms':'-'}</b> <small>(估算)</small></span>
       <span>grand total: <b>${ta.total_ms!=null?ta.total_ms.toFixed(1)+' ms':'-'}</b></span>
     </div>`;

  // Find max mean for bar scaling
  let maxMean = 1;
  for (const sk of STAGE_ORDER) {
    const a = avgs[sk];
    if (a && a.mean_ms != null && a.mean_ms > maxMean) maxMean = a.mean_ms;
  }

  let rows = '';
  for (let i = 0; i < STAGE_ORDER.length; i++) {
    const sk = STAGE_ORDER[i];
    const label = STAGE_LABELS[sk];
    const a = avgs[sk] || {};
    const stype = a.type || 'measured';
    const isEst = stype === 'estimated';
    const typeTag = isEst
      ? '<span class="warn">估算</span>'
      : (stype === 'gpu_measured' ? '<span class="ok">GPU</span>' : '<span class="ok">精确</span>');
    const mean = a.mean_ms != null ? a.mean_ms.toFixed(2) + ' ms' : '-';
    const mn = a.min_ms != null ? a.min_ms.toFixed(2) : '-';
    const mx = a.max_ms != null ? a.max_ms.toFixed(2) : '-';
    const w = a.mean_ms != null ? Math.round(a.mean_ms / maxMean * 100) : 0;
    const barCls = isEst ? 'warn' : (a.mean_ms != null && a.mean_ms > 50 ? 'warn' : '');
    rows += `<tr${isEst?' style="opacity:0.7"':''}>
      <td>${i+1}</td>
      <td>${label}</td>
      <td>${typeTag}</td>
      <td>${mean}</td>
      <td>${mn}</td>
      <td>${mx}</td>
      <td><span class="bar" style="width:95%"><i class="${barCls}" style="width:${w}%"></i></span></td>
    </tr>`;
  }
  tbody.innerHTML = rows;
}

refresh();setInterval(refresh,5000);
</script>
</body></html>"""


# ------------------------ HTTP handler ------------------------
class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):  # 静音
        return

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False, indent=2)
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _send_file_download(self, filepath, filename=None):
        """Serve a file as a download attachment."""
        if not os.path.isfile(filepath):
            self._send(404, {"error": "file not found"})
            return
        with open(filepath, "rb") as f:
            data = f.read()
        dl_name = filename or os.path.basename(filepath)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{dl_name}"')
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/" or path == "/index.html":
                self._send(200, HTML, "text/html")
            elif path.startswith("/download/"):
                # Serve files from the same directory as monitor.py
                fname = path[len("/download/"):]
                # Security: only allow simple filenames, no path traversal
                if "/" in fname or "\\" in fname or ".." in fname:
                    self._send(403, {"error": "forbidden"})
                    return
                base_dir = os.path.dirname(os.path.abspath(__file__))
                filepath = os.path.join(base_dir, fname)
                self._send_file_download(filepath)
            elif path == "/api/status":
                snap = LAST_STATE["snapshot"] or take_snapshot()
                self._send(200, snap)
            elif path == "/api/log":
                lines = tail_log(ARGS.log, n=200)
                self._send(200, "\n".join(lines), "text/plain")
            elif path == "/api/restarts":
                self._send(200, list(RESTART_EVENTS))
            elif path == "/api/history":
                self._send(200, list(HISTORY))
            elif path.startswith("/download/"):
                fname = path.split("/download/", 1)[1]
                # 安全：只允许下载同目录下的 .md 文件
                if ".." in fname or "/" in fname or not fname.endswith(".md"):
                    self._send(403, {"error": "forbidden"})
                    return
                fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
                if not os.path.isfile(fpath):
                    self._send(404, {"error": f"file not found: {fname}"})
                    return
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                b = content.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/markdown; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{fname}"')
                self.send_header("Content-Length", str(len(b)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(b)
            elif path == "/healthz":
                snap = LAST_STATE["snapshot"]
                if snap and snap.get("status") in ("GREEN", "YELLOW"):
                    self._send(200, {"ok": True, "status": snap["status"]})
                else:
                    self._send(503, {"ok": False, "status": (snap or {}).get("status", "UNKNOWN")})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": str(e)})


# ------------------------ main ------------------------
def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--demo-port", type=int, default=7863)
    ap.add_argument("--master-port", type=int, default=29510)
    ap.add_argument("--gpus", default="4,5,6,7", help="comma list of gpu indices")
    ap.add_argument("--log", default="/root/StreamDiffusionV2/repo/outputs/demo_1p3b_4gpu.log")
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--throughput-window-s", type=float, default=30.0,
                    help="rolling window (seconds) for input/output FPS estimation")
    ARGS = ap.parse_args()
    ARGS.gpu_ids_set = set(int(x) for x in ARGS.gpus.split(",") if x.strip())

    # 后台采集线程
    t = threading.Thread(target=sampler_loop, daemon=True)
    t.start()

    # 立刻先抓一次，避免首次访问空白
    try:
        LAST_STATE["snapshot"] = take_snapshot()
    except Exception as e:
        print(f"first snapshot failed: {e}", file=sys.stderr)

    server = ThreadingHTTPServer((ARGS.bind, ARGS.port), H)
    print(f"[monitor] listening on http://{ARGS.bind}:{ARGS.port}")
    print(f"[monitor] watching demo :{ARGS.demo_port}, master :{ARGS.master_port}, gpus {sorted(ARGS.gpu_ids_set)}")
    print(f"[monitor] log: {ARGS.log}")
    server.serve_forever()


if __name__ == "__main__":
    main()

from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request

import markdown2

import logging
import uuid
import time
from types import SimpleNamespace
import asyncio
import os
import mimetypes
import threading
import multiprocessing as mp
import signal
import sys
from collections import deque

from config import config, Args
from util import pil_to_frame, bytes_to_pil, is_firefox
from connection_manager import ConnectionManager, ServerFullException

# fix mime error on windows
mimetypes.add_type("application/javascript", ".js")

THROTTLE = 1.0 / 120
LOGGER = logging.getLogger(__name__)


class App:
    def __init__(self, config: Args, pipeline):
        self.args = config
        self.pipeline = pipeline
        self.app = FastAPI()
        self.conn_manager = ConnectionManager()
        self.prediction_workers = {}
        self.shutdown_event = asyncio.Event()
        self.demo_root = os.path.dirname(os.path.abspath(__file__))
        self.frontend_public_dir = os.path.join(self.demo_root, "frontend", "public")
        # Initialize metrics collection only if enabled
        self.enable_metrics = config.enable_metrics
        self.target_latency = config.target_latency  # Target latency in seconds for deadline miss rate
        self.step = config.step  # Pipeline step parameter
        self.gpu_ids = config.gpu_ids  # GPU IDs (e.g., "0,1" or "0")
        if self.enable_metrics:
            # Simple timestamp queue for input frames (FIFO)
            self.user_input_timestamps = {}  # user_id -> deque of input timestamps
            self.user_metrics_lock = threading.Lock()  # Lock for thread-safe timestamp tracking
            # Track metrics collection count per user (number of batches collected)
            self.user_batch_count = {}  # user_id -> count of batches collected
            self.user_latency_history = {}  # user_id -> list of latencies for statistics
            self.user_raw_data = {}  # user_id -> list of raw batch data (for logging)
            # First-frame latency tracking: for each user, remember the
            # timestamp when their *first* input frame entered the pipeline
            # queue (push_frames_to_pipeline).  When the matching output
            # frame is produced we emit a single TTFF ("time-to-first-
            # frame") sample per session.  This is stored globally (across
            # sessions) so the monitor can aggregate.
            self.user_first_input_ts = {}   # user_id -> float | None
            self.user_ttff_emitted = set()  # user_ids for which we already emitted TTFF
            self.first_frame_latencies = deque(maxlen=500)  # global ring of TTFF samples (seconds)
            self.metrics_log_dir = "./slo_metrics"
            os.makedirs(self.metrics_log_dir, exist_ok=True)
        self.init_app()

    def init_app(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.websocket("/api/ws/{user_id}")
        async def websocket_endpoint(user_id: uuid.UUID, websocket: WebSocket):
            try:
                await self.conn_manager.connect(
                    user_id, websocket, self.args.max_queue_size
                )
                await handle_websocket_data(user_id)
            except ServerFullException as e:
                logging.error(f"Server Full: {e}")
            finally:
                # Order matters:
                # 1) stop the prediction worker thread so it no longer drains
                #    pipeline output concurrently with our cleanup.
                # 2) THEN disconnect + signal pipeline end. We await this
                #    rather than fire-and-forget so that the
                #    input_queue-drain + restart_event.set path actually
                #    runs before the next client's handler touches the
                #    pipeline (avoids overlapping sessions racing on the
                #    shared GPU process group).
                try:
                    await self._stop_prediction_worker(user_id)
                except Exception as e:  # noqa: BLE001
                    logging.error(f"Error stopping prediction worker for {user_id}: {e}")
                try:
                    await asyncio.wait_for(
                        self.conn_manager.disconnect(user_id, self.pipeline),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    logging.warning(
                        "disconnect() for %s exceeded 5s; continuing cleanup.",
                        user_id,
                    )
                except Exception as e:  # noqa: BLE001
                    logging.error(f"Error during disconnect for {user_id}: {e}")
                # Clean up metrics and timestamp tracking for this user
                if self.enable_metrics:
                    with self.user_metrics_lock:
                        self.user_input_timestamps.pop(user_id, None)
                        self.user_batch_count.pop(user_id, None)
                        self.user_latency_history.pop(user_id, None)
                        self.user_raw_data.pop(user_id, None)
                        self.user_first_input_ts.pop(user_id, None)
                        self.user_ttff_emitted.discard(user_id)
                logging.info(f"User disconnected: {user_id}")

        async def handle_websocket_data(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                return HTTPException(status_code=404, detail="User not found")
            last_time = time.time()
            last_frame_time = None
            # 16 FPS throttling: minimum interval between frames (1/16 seconds)
            TARGET_FPS = 16.0
            min_frame_interval = 1.0 / TARGET_FPS
            last_frame_received_time = None
            try:
                while True:
                    if (
                        self.args.timeout > 0
                        and time.time() - last_time > self.args.timeout
                    ):
                        await self.conn_manager.send_json(
                            user_id,
                            {
                                "status": "timeout",
                                "message": "Your session has ended",
                            },
                        )
                        await self.conn_manager.disconnect(user_id, self.pipeline)
                        return
                    data = await self.conn_manager.receive_json(user_id)
                    # Refresh idle timer on any client control message
                    last_time = time.time()
                    # Handle stop/pause without closing socket: go idle and wait
                    if data and data.get("status") == "pause":
                        params = SimpleNamespace(**{"restart": True})
                        await self.conn_manager.update_data(user_id, params)
                        continue
                    if data and data.get("status") == "resume":
                        await self.conn_manager.send_json(user_id, {"status": "send_frame"})
                        continue
                    # Mark upload completion: after this, don't receive image bytes again
                    if data and data.get("status") == "upload_done":
                        self.conn_manager.set_video_upload_completed(user_id, True)
                        LOGGER.info("Upload completed for user %s", user_id)
                        await self.conn_manager.send_json(user_id, {"status": "upload_done_ack"})
                        continue
                    if not data or data.get("status") != "next_frame":
                        await asyncio.sleep(THROTTLE)
                        continue

                    params = await self.conn_manager.receive_json(user_id)
                    params = self.pipeline.InputParams(**params)
                    info = self.pipeline.Info()
                    params = self.pipeline.params_to_namespace(params)
                    
                    # Check if upload mode is enabled
                    is_upload_mode = params.__dict__.get('input_mode') == 'upload' or params.__dict__.get('upload_mode', False)
                    self.conn_manager.set_upload_mode(user_id, is_upload_mode)
                    if is_upload_mode:
                        LOGGER.debug("Upload mode detected for user %s", user_id)
                    
                    if info.input_mode == "image":
                        upload_completed = self.conn_manager.is_video_upload_completed(user_id)
                        # Only receive image bytes if not in upload mode, or upload not completed yet
                        if (not is_upload_mode) or (is_upload_mode and not upload_completed):
                            image_data = await self.conn_manager.receive_bytes(user_id)
                            if len(image_data) == 0:
                                await self.conn_manager.send_json(
                                    user_id, {"status": "send_frame"}
                                )
                                # await asyncio.sleep(sleep_time)
                                continue
                            
                            # 16 FPS throttling: only process frames at 16 FPS rate
                            current_time = time.time()
                            if last_frame_received_time is not None:
                                time_since_last_frame = current_time - last_frame_received_time
                                if time_since_last_frame < min_frame_interval:
                                    # Skip this frame to maintain 16 FPS
                                    await self.conn_manager.send_json(user_id, {"status": "send_frame"})
                                    continue
                            
                            last_frame_received_time = current_time
                            
                            # If upload mode and not completed, append frames to cache for later reuse
                            if is_upload_mode and not upload_completed:
                                await self.conn_manager.add_video_frame(user_id, image_data)
                                LOGGER.debug("Buffered uploaded frame for user %s", user_id)
                            # For camera mode, set current image directly
                            if not is_upload_mode:
                                params.image = bytes_to_pil(image_data)
                        else:
                            # Upload already completed: do not receive more bytes; image will be fed from cached frames
                            pass
                    await self.conn_manager.update_data(user_id, params)
                    await self.conn_manager.send_json(user_id, {"status": "wait"})
                    if last_frame_time is None:
                        last_frame_time = time.time()
                    else:
                        # print(f"Frame time: {time.time() - last_frame_time}")
                        last_frame_time = time.time()

            except Exception as e:
                logging.error(f"Websocket Error: {e}, {user_id} ")
                await self.conn_manager.disconnect(user_id, self.pipeline)

        @self.app.get("/api/queue")
        async def get_queue_size():
            queue_size = self.conn_manager.get_user_count()
            return JSONResponse({"queue_size": queue_size})

        # Histogram bin edges (ms) for end-to-end latency distribution.
        # Chosen to give useful resolution across the full range we care
        # about in a realtime video pipeline (sub-100ms "great" -> multi-
        # second "bad"). Keep these in sync with any frontend/monitor
        # consumer.
        E2E_HIST_BINS_MS = (
            0, 50, 100, 150, 200, 300, 500, 800,
            1200, 2000, 3000, 5000, 10000,
        )

        def _percentiles(values, pcts):
            """Tiny stdlib percentile helper (linear interpolation) so the
            /api/metrics/summary endpoint does not require numpy in the
            hot path. `values` must already be a list of floats."""
            if not values:
                return {f"p{p}": 0.0 for p in pcts}
            s = sorted(values)
            n = len(s)
            out = {}
            for p in pcts:
                if n == 1:
                    out[f"p{p}"] = s[0]
                    continue
                k = (p / 100.0) * (n - 1)
                lo = int(k)
                hi = min(lo + 1, n - 1)
                frac = k - lo
                out[f"p{p}"] = s[lo] * (1 - frac) + s[hi] * frac
            return out

        def _histogram_ms(values_sec, bin_edges_ms):
            """Bucket latencies (seconds) into fixed ms bins. Last bucket
            catches everything >= last edge (the '+inf' tail)."""
            counts = [0] * len(bin_edges_ms)  # len(edges) buckets: [e0,e1) ... [e_{n-1}, +inf)
            for v in values_sec:
                v_ms = v * 1000.0
                placed = False
                for i in range(len(bin_edges_ms) - 1):
                    if bin_edges_ms[i] <= v_ms < bin_edges_ms[i + 1]:
                        counts[i] += 1
                        placed = True
                        break
                if not placed:
                    # v_ms >= last edge, goes into the overflow bucket
                    counts[-1] += 1
            return counts

        # NOTE: /api/metrics/summary MUST be registered before
        # /api/metrics/{user_id}; otherwise FastAPI matches "summary" as a
        # user_id and rejects the request with HTTP 422 ("invalid UUID").
        @self.app.get("/api/metrics/summary")
        async def get_metrics_summary(window_size: int = 500):
            """Aggregate metrics across all active users for the demo
            monitor. Returns end-to-end latency stats + a fixed-bin
            histogram + first-frame (TTFF) stats. When metrics collection
            is disabled, returns `enabled=false` so the monitor can show
            a clear reason rather than treat it as an outage.
            """
            if not self.enable_metrics:
                return JSONResponse({
                    "enabled": False,
                    "reason": "start demo with --enable-metrics to populate",
                })
            try:
                with self.user_metrics_lock:
                    # Flatten the last `window_size` latencies per user, then
                    # globally clip again, so a single very chatty user does
                    # not drown out everyone else.
                    all_latencies = []
                    per_user = {}
                    for uid, hist in self.user_latency_history.items():
                        tail = list(hist[-window_size:])
                        per_user[str(uid)] = len(tail)
                        all_latencies.extend(tail)
                    all_latencies = all_latencies[-window_size * max(1, len(per_user)):]
                    ttff_samples = list(self.first_frame_latencies)
                    active_users = len(per_user)
                    pending_inputs = {
                        str(uid): len(q) for uid, q in self.user_input_timestamps.items()
                    }

                pcts = (50, 90, 95, 99)
                e2e_stats = {}
                if all_latencies:
                    e2e_stats = {
                        "count": len(all_latencies),
                        "mean_ms": (sum(all_latencies) / len(all_latencies)) * 1000.0,
                        "min_ms": min(all_latencies) * 1000.0,
                        "max_ms": max(all_latencies) * 1000.0,
                    }
                    e2e_stats.update({
                        f"{k}_ms": v * 1000.0
                        for k, v in _percentiles(all_latencies, pcts).items()
                    })
                    # Deadline miss rate against configured target
                    deadline = self.target_latency
                    missed = sum(1 for v in all_latencies if v > deadline)
                    e2e_stats["deadline_s"] = deadline
                    e2e_stats["deadline_miss_rate"] = missed / len(all_latencies)
                else:
                    e2e_stats = {"count": 0}

                ttff_stats = {"count": len(ttff_samples)}
                if ttff_samples:
                    ttff_stats.update({
                        "mean_ms": (sum(ttff_samples) / len(ttff_samples)) * 1000.0,
                        "min_ms": min(ttff_samples) * 1000.0,
                        "max_ms": max(ttff_samples) * 1000.0,
                        "last_ms": ttff_samples[-1] * 1000.0,
                    })
                    ttff_stats.update({
                        f"{k}_ms": v * 1000.0
                        for k, v in _percentiles(ttff_samples, pcts).items()
                    })

                histogram = {
                    "bin_edges_ms": list(E2E_HIST_BINS_MS),
                    # len(counts) == len(edges): last bucket is [last_edge, +inf)
                    "counts": _histogram_ms(all_latencies, E2E_HIST_BINS_MS),
                    "unit": "ms",
                }

                return JSONResponse({
                    "enabled": True,
                    "window_size": window_size,
                    "active_users": active_users,
                    "per_user_sample_count": per_user,
                    "pending_inputs_per_user": pending_inputs,
                    "e2e_latency": e2e_stats,
                    "first_frame_latency": ttff_stats,
                    "e2e_histogram": histogram,
                })
            except Exception as e:
                logging.error(f"Error getting summary metrics: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.get("/api/metrics/{user_id}")
        async def get_metrics(user_id: uuid.UUID, window_size: int = 100):
            """Get SLO metrics for a specific user"""
            if not self.enable_metrics:
                return JSONResponse({"error": "Metrics collection is not enabled"}, status_code=400)
            try:
                import numpy as np
                with self.user_metrics_lock:
                    if user_id not in self.user_latency_history or len(self.user_latency_history[user_id]) == 0:
                        return JSONResponse({"error": "No metrics data available"}, status_code=404)
                    
                    latencies = np.array(self.user_latency_history[user_id][-window_size:])
                    
                    stats = {
                        "mean_latency": float(np.mean(latencies)),
                        "median_latency": float(np.median(latencies)),
                        "p95_latency": float(np.percentile(latencies, 95)),
                        "p99_latency": float(np.percentile(latencies, 99)),
                        "min_latency": float(np.min(latencies)),
                        "max_latency": float(np.max(latencies)),
                        "std_latency": float(np.std(latencies)),
                        "sample_count": len(latencies),
                        "remaining_frames": len(self.user_input_timestamps.get(user_id, deque())),
                        "batch_count": self.user_batch_count.get(user_id, 0)
                    }
                    
                    return JSONResponse(stats)
            except Exception as e:
                logging.error(f"Error getting metrics: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        @self.app.get("/api/stream/{user_id}")
        async def stream(user_id: uuid.UUID, request: Request):
            try:
                async def push_frames_to_pipeline():
                    last_params = SimpleNamespace()
                    sleep_time = 1 / 20  # Initial guess
                    # 16 FPS throttling for upload mode
                    TARGET_FPS = 16.0
                    min_frame_interval = 1.0 / TARGET_FPS
                    last_frame_sent_time = None
                    while True:
                        # Check if upload mode is enabled
                        video_status = self.conn_manager.get_video_queue_status(user_id)
                        is_upload_mode = video_status.get("is_upload_mode", False)
                        
                        if is_upload_mode:
                            # Upload mode: get next frame from video queue with 16 FPS throttling
                            current_time = time.time()
                            if last_frame_sent_time is not None:
                                time_since_last_frame = current_time - last_frame_sent_time
                                if time_since_last_frame < min_frame_interval:
                                    # Wait to maintain 16 FPS
                                    await asyncio.sleep(min_frame_interval - time_since_last_frame)
                            
                            video_frame = await self.conn_manager.get_next_video_frame(user_id)
                            if video_frame:
                                last_frame_sent_time = time.time()
                                # Create params object with video frame
                                params = SimpleNamespace()
                                params.image = bytes_to_pil(video_frame)
                                # Copy other parameters
                                if vars(last_params):
                                    for key, value in last_params.__dict__.items():
                                        if key != 'image' and key != '_frame_id':
                                            setattr(params, key, value)
                                
                                if params.__dict__ != last_params.__dict__:
                                    # Record input timestamp when frame is added to pipeline queue
                                    if self.enable_metrics:
                                        input_timestamp = time.time()
                                        with self.user_metrics_lock:
                                            if user_id not in self.user_input_timestamps:
                                                self.user_input_timestamps[user_id] = deque()
                                            self.user_input_timestamps[user_id].append(input_timestamp)
                                            # Capture first-ever input ts for TTFF
                                            if user_id not in self.user_first_input_ts:
                                                self.user_first_input_ts[user_id] = input_timestamp
                                    
                                    last_params = params
                                    self.pipeline.accept_new_params(params)
                                    LOGGER.debug("Sent cached upload frame to pipeline for user %s", user_id)
                                # Yield control without delaying to maximize fluency
                                # await asyncio.sleep(sleep_time)
                            else:
                                # No frame available, wait a bit
                                await asyncio.sleep(sleep_time)
                        else:
                            # Camera mode: normal processing
                            params = await self.conn_manager.get_latest_data(user_id)
                            if params is None:
                                break
                            if vars(params) and params.__dict__ != last_params.__dict__:
                                last_params = params
                                # Record input timestamp when frame is added to pipeline queue
                                if self.enable_metrics:
                                    input_timestamp = time.time()
                                    with self.user_metrics_lock:
                                        if user_id not in self.user_input_timestamps:
                                            self.user_input_timestamps[user_id] = deque()
                                        self.user_input_timestamps[user_id].append(input_timestamp)
                                        if user_id not in self.user_first_input_ts:
                                            self.user_first_input_ts[user_id] = input_timestamp
                                self.pipeline.accept_new_params(params)
                            await self.conn_manager.send_json(
                                user_id, {"status": "send_frame"}
                            )
                            # Yield control without delaying
                            # await asyncio.sleep(sleep_time)

                async def generate():
                    MIN_FPS = 5
                    MAX_FPS = 30
                    SMOOTHING = 0.8  # EMA smoothing factor

                    last_burst_time = time.time()
                    last_queue_size = 0
                    sleep_time = 1 / 20  # Initial guess
                    last_frame_time = None
                    frame_time_list = []

                    # Initialize moving average frame interval
                    ema_frame_interval = sleep_time
                    while True:
                        queue_size = await self.conn_manager.get_output_queue_size(user_id)
                        if queue_size > last_queue_size:
                            current_burst_time = time.time()
                            elapsed = current_burst_time - last_burst_time

                            if queue_size > 0 and elapsed > 0:
                                raw_interval = elapsed / queue_size
                                ema_frame_interval = SMOOTHING * ema_frame_interval + (1 - SMOOTHING) * raw_interval
                                sleep_time = min(max(ema_frame_interval, 1 / MAX_FPS), 1 / MIN_FPS)

                            last_burst_time = current_burst_time

                        last_queue_size = queue_size
                        try:
                            frame = await self.conn_manager.get_frame(user_id)
                            if frame is None:
                                break
                            
                            # Output timestamp is already recorded in produce_predictions
                            
                            yield frame
                            if not is_firefox(request.headers.get("user-agent", "")):
                                yield frame
                            if last_frame_time is None:
                                last_frame_time = time.time()
                            else:
                                frame_time_list.append(time.time() - last_frame_time)
                                if len(frame_time_list) > 100:
                                    frame_time_list.pop(0)
                                last_frame_time = time.time()
                        except Exception as e:
                            LOGGER.error("Frame fetch error for user %s: %s", user_id, e)
                            break

                        await asyncio.sleep(sleep_time)

                def produce_predictions(user_id, loop, stop_event):
                    while not stop_event.is_set():
                        images = self.pipeline.produce_outputs()
                        if len(images) == 0:
                            time.sleep(THROTTLE)
                            continue
                        
                        # Calculate latency for each output frame using FIFO timestamp queue
                        if self.enable_metrics:
                            output_timestamp = time.time()
                            batch_latencies = []
                            
                            with self.user_metrics_lock:
                                if user_id in self.user_input_timestamps:
                                    # For each output frame, get corresponding input timestamp (FIFO)
                                    for _ in range(len(images)):
                                        if len(self.user_input_timestamps[user_id]) > 0:
                                            input_timestamp = self.user_input_timestamps[user_id].popleft()
                                            latency = output_timestamp - input_timestamp
                                            batch_latencies.append(latency)

                                            # Add to history for statistics
                                            if user_id not in self.user_latency_history:
                                                self.user_latency_history[user_id] = []
                                            self.user_latency_history[user_id].append(latency)

                                    # Emit TTFF (time-to-first-frame) once per session: the
                                    # latency between the user's *first* input frame and the
                                    # *first* output frame we produced for them.
                                    if (user_id not in self.user_ttff_emitted
                                            and len(batch_latencies) > 0
                                            and self.user_first_input_ts.get(user_id) is not None):
                                        ttff = output_timestamp - self.user_first_input_ts[user_id]
                                        self.first_frame_latencies.append(ttff)
                                        self.user_ttff_emitted.add(user_id)
                                        LOGGER.info(
                                            "[Metrics] TTFF user=%s first_frame_latency=%.3fs",
                                            user_id, ttff,
                                        )
                                    
                                    # Print batch statistics
                                    if len(batch_latencies) > 0:
                                        avg_latency = sum(batch_latencies) / len(batch_latencies)
                                        remaining_frames = len(self.user_input_timestamps[user_id])
                                        
                                        # Get batch count
                                        if user_id not in self.user_batch_count:
                                            self.user_batch_count[user_id] = 0
                                        self.user_batch_count[user_id] += 1
                                        batch_num = self.user_batch_count[user_id]
                                        
                                        # Prepare raw batch data
                                        raw_batch_data = {
                                            "batch_num": batch_num,
                                            "current_frames": len(batch_latencies),
                                            "avg_latency": avg_latency,
                                            "remaining": remaining_frames,
                                            "data_count": len(self.user_latency_history[user_id])
                                        }
                                        
                                        # Store raw data
                                        if user_id not in self.user_raw_data:
                                            self.user_raw_data[user_id] = []
                                        self.user_raw_data[user_id].append(raw_batch_data)
                                        
                                        LOGGER.info(
                                            "[Metrics] Batch %s/1000: current_frames=%s, avg_latency=%.4fs, remaining=%s, data_count=%s",
                                            batch_num,
                                            len(batch_latencies),
                                            avg_latency,
                                            remaining_frames,
                                            len(self.user_latency_history[user_id]),
                                        )
                                        
                                        # Log after 1000 batches
                                        if batch_num >= 1000:
                                            self._log_metrics_to_file(user_id)
                                            # Reset for next 1000 batches
                                            self.user_batch_count[user_id] = 0
                                            self.user_latency_history[user_id] = []
                                            self.user_raw_data[user_id] = []
                        
                        asyncio.run_coroutine_threadsafe(
                            self.conn_manager.put_frames_to_output_queue(
                                user_id,
                                list(map(pil_to_frame, images))
                            ),
                            loop
                        )

                await self._start_prediction_worker(
                    user_id,
                    produce_predictions,
                    asyncio.get_running_loop(),
                )
                asyncio.create_task(push_frames_to_pipeline())
                await self.conn_manager.send_json(user_id, {"status": "send_frame"})

                return StreamingResponse(
                    generate(),
                    media_type="multipart/x-mixed-replace;boundary=frame",
                    headers={"Cache-Control": "no-cache"},
                )

            except Exception as e:
                logging.error(f"Streaming Error: {e}, {user_id} ")
                # Stop prediction thread on error
                await self._stop_prediction_worker(user_id)
                return HTTPException(status_code=404, detail="User not found")

        # route to setup frontend
        @self.app.get("/api/settings")
        async def settings():
            info_schema = self.pipeline.Info.schema()
            info = self.pipeline.Info()
            if info.page_content:
                page_content = markdown2.markdown(info.page_content)

            input_params = self.pipeline.InputParams.schema()
            return JSONResponse(
                {
                    "info": info_schema,
                    "input_params": input_params,
                    "max_queue_size": self.args.max_queue_size,
                    "page_content": page_content if info.page_content else "",
                }
            )

        @self.app.get("/api/variant")
        async def variant():
            """Return which model variant this backend is serving and the
            peer variants available on other ports so the frontend can
            render a selector and redirect on switch.

            Configured via environment variables:
              DEMO_CURRENT_VARIANT : id of *this* backend (e.g. "14B" / "1.3B")
              DEMO_VARIANTS        : JSON list of
                  [{"id": "14B", "label": "14B", "port": 7862}, ...]
            """
            import json as _json
            current = os.environ.get("DEMO_CURRENT_VARIANT", "")
            raw = os.environ.get("DEMO_VARIANTS", "")
            variants = []
            if raw:
                try:
                    parsed = _json.loads(raw)
                    if isinstance(parsed, list):
                        variants = parsed
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Failed to parse DEMO_VARIANTS env: %s", exc)
            return JSONResponse({"current": current, "variants": variants})

        os.makedirs(self.frontend_public_dir, exist_ok=True)

        self.app.mount(
            "/", StaticFiles(directory=self.frontend_public_dir, html=True), name="public"
        )

        # Add shutdown event handler
        @self.app.on_event("shutdown")
        async def shutdown_event():
            LOGGER.info("Shutdown event triggered, cleaning up...")
            await self.cleanup()

    def _log_metrics_to_file(self, user_id: uuid.UUID):
        """Log metrics to file after collecting 1000 batches"""
        try:
            import json
            import numpy as np
            
            # Get latency history
            if user_id not in self.user_latency_history or len(self.user_latency_history[user_id]) == 0:
                LOGGER.info("[Metrics] No latency data to log for user %s", user_id)
                return
            
            latencies = np.array(self.user_latency_history[user_id])
            
            # Calculate statistics
            stats = {
                "mean_latency": float(np.mean(latencies)),
                "median_latency": float(np.median(latencies)),
                "p50_latency": float(np.percentile(latencies, 50)),
                "p90_latency": float(np.percentile(latencies, 90)),
                "p95_latency": float(np.percentile(latencies, 95)),
                "p99_latency": float(np.percentile(latencies, 99)),
                "p99_9_latency": float(np.percentile(latencies, 99.9)),
                "min_latency": float(np.min(latencies)),
                "max_latency": float(np.max(latencies)),
                "std_latency": float(np.std(latencies)),
                "sample_count": len(latencies)
            }
            
            # Calculate deadline miss rate using target latency
            deadline = self.target_latency
            missed = np.sum(latencies > deadline)
            deadline_stats = {
                "deadline_seconds": deadline,
                "deadline_miss_rate": float(missed / len(latencies)) if len(latencies) > 0 else 0.0,
                "missed_frames": int(missed),
                "total_frames": len(latencies)
            }
            
            # Calculate jitter (variation in consecutive latencies)
            if len(latencies) > 1:
                jitter = np.abs(np.diff(latencies))
                jitter_stats = {
                    "mean_jitter": float(np.mean(jitter)),
                    "std_jitter": float(np.std(jitter)),
                    "max_jitter": float(np.max(jitter)),
                    "min_jitter": float(np.min(jitter)),
                    "p50_jitter": float(np.percentile(jitter, 50)),
                    "p90_jitter": float(np.percentile(jitter, 90)),
                    "p95_jitter": float(np.percentile(jitter, 95)),
                    "p99_jitter": float(np.percentile(jitter, 99)),
                    "p99.9_jitter": float(np.percentile(jitter, 99.9)),
                    "jitter_variance": float(np.var(jitter))
                }
            else:
                jitter_stats = {}
            
            # Create timestamp folder (YYYYMMDD_HHMM_step{step}_gpu{gpu_ids} format)
            timestamp = time.strftime("%Y%m%d_%H%M")
            # Format GPU IDs: replace commas with underscores for folder naming
            gpu_str = self.gpu_ids.replace(",", "_")
            folder_name = f"{timestamp}_step{self.step}_gpu{gpu_str}"
            session_dir = os.path.join(self.metrics_log_dir, folder_name)
            os.makedirs(session_dir, exist_ok=True)
            
            # Prepare raw data file content
            raw_data_content = {
                "user_id": str(user_id),
                "timestamp": timestamp,
                "target_latency": self.target_latency,
                "batches": self.user_raw_data.get(user_id, [])
            }
            
            # Prepare statistics file content
            statistics_content = {
                "user_id": str(user_id),
                "timestamp": timestamp,
                "target_latency": self.target_latency,
                "batch_count": 1000,
                "total_frames": len(latencies),
                "latency_stats": stats,
                "deadline_miss_rate": deadline_stats,
                "jitter_distribution": jitter_stats,
                "tail_latency": {
                    "p90_latency": stats["p90_latency"],
                    "p95_latency": stats["p95_latency"],
                    "p99_latency": stats["p99_latency"],
                    "p99_9_latency": stats["p99_9_latency"],
                    "max_latency": stats["max_latency"],
                    "mean_latency": stats["mean_latency"],
                    "median_latency": stats["median_latency"]
                }
            }
            
            # Write raw data file
            raw_data_filename = os.path.join(session_dir, f"raw_data_{user_id}.json")
            with open(raw_data_filename, 'w') as f:
                json.dump(raw_data_content, f, indent=2)
            
            # Write statistics file
            statistics_filename = os.path.join(session_dir, f"statistics_{user_id}.json")
            with open(statistics_filename, 'w') as f:
                json.dump(statistics_content, f, indent=2)
            
            LOGGER.info("[Metrics] Logged metrics to %s/", session_dir)
            LOGGER.info("[Metrics]   - Raw data: raw_data_%s.json", user_id)
            LOGGER.info("[Metrics]   - Statistics: statistics_%s.json", user_id)
            LOGGER.info(
                "[Metrics] Summary: mean=%.4fs, p95=%.4fs, miss_rate=%.2f%%",
                stats["mean_latency"],
                stats["p95_latency"],
                deadline_stats["deadline_miss_rate"] * 100,
            )
            
        except Exception as e:
            logging.error(f"Error logging metrics to file: {e}")
    
    async def cleanup(self):
        """Clean up all resources on shutdown"""
        LOGGER.info("Starting cleanup process...")
        
        # Set shutdown event
        self.shutdown_event.set()
        
        # Stop all background tasks
        for user_id in list(self.prediction_workers):
            await self._stop_prediction_worker(user_id)
        LOGGER.info("Stopped prediction tasks")
        
        # Close all WebSocket connections and pipeline
        LOGGER.info("Closing %s active connections...", len(self.conn_manager.active_connections))
        try:
            await self.conn_manager.disconnect_all(self.pipeline)
        except Exception as e:
            LOGGER.error("Error during disconnect_all: %s", e)
        
        LOGGER.info("Cleanup completed")

    async def _start_prediction_worker(self, user_id, produce_predictions, loop):
        await self._stop_prediction_worker(user_id)
        stop_event = threading.Event()
        task = asyncio.create_task(
            asyncio.to_thread(produce_predictions, user_id, loop, stop_event)
        )
        self.prediction_workers[user_id] = (stop_event, task)

    async def _stop_prediction_worker(self, user_id):
        worker = self.prediction_workers.pop(user_id, None)
        if worker is None:
            return

        stop_event, task = worker
        stop_event.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# Global app instance for signal handler
app_instance = None

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    LOGGER.info("Received signal %s, shutting down gracefully...", signum)
    if app_instance:
        # Trigger cleanup in a separate thread to avoid blocking
        import threading
        def trigger_cleanup():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(app_instance.cleanup())
                loop.close()
            except Exception as e:
                LOGGER.error("Error during cleanup: %s", e)
        
        cleanup_thread = threading.Thread(target=trigger_cleanup)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        cleanup_thread.join(timeout=5)  # Wait up to 5 seconds for cleanup
    
    sys.exit(0)

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    mp.set_start_method("spawn", force=True)

    config.pretty_print()
    if config.num_gpus > 1:
        from vid2vid_pipe import MultiGPUPipeline
        pipeline = MultiGPUPipeline(config)
    else:
        from vid2vid import Pipeline
        pipeline = Pipeline(config)

    app_obj = App(config, pipeline)
    app = app_obj.app
    app_instance = app_obj  # Set global reference for signal handler

    try:
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            reload=False,
            ssl_certfile=config.ssl_certfile,
            ssl_keyfile=config.ssl_keyfile,
        )
    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt received, shutting down...")
        # Trigger cleanup
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(app_obj.cleanup())
            loop.close()
        except Exception as e:
            LOGGER.error("Error during cleanup: %s", e)
        sys.exit(0)
    except Exception as e:
        LOGGER.error("Fatal error: %s", e)
        sys.exit(1)

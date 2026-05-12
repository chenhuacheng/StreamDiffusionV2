import sys
import os
import time
from multiprocessing import Queue, Event, Process, Manager

DEMO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DEMO_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from streamv2v.inference_pipe import InferencePipelineManager, compute_default_block_distribution
from util import clear_queue, get_num_transformer_blocks, read_images_from_queue, resolve_worker_device

from datetime import timedelta

import torch
import torch.distributed as dist

from vid2vid import Pipeline, report_worker_error, set_config_value, wait_for_processes_ready


class MultiGPUPipeline(Pipeline):
    def prepare(self):
        self.total_blocks = get_num_transformer_blocks(self.args)
        self.total_block_num = compute_default_block_distribution(
            total_blocks=self.total_blocks,
            world_size=self.args.num_gpus,
        )

        self.input_queue = Queue()
        self.output_queue = Queue()
        self.prepare_events = [Event() for _ in range(self.args.num_gpus)]
        self.stop_event = Event()
        self.restart_event = Event()
        self.error_queue = Queue()
        self.runtime_state = Manager().dict()
        self.runtime_state["prompt"] = self.prompt
        self.runtime_state["use_taehv"] = bool(getattr(self.args, "use_taehv", False))
        self.runtime_state["use_tensorrt"] = bool(getattr(self.args, "use_tensorrt", False))
        self.p_input = Process(
            target=input_process,
            args=(0, self.total_block_num, self.total_blocks, self.args, self.runtime_state, self.prepare_events[0], self.restart_event, self.stop_event, self.input_queue, self.error_queue),
            daemon=True,
        )
        self.p_middles = [
            Process(
                target=middle_process,
                args=(i, self.total_block_num, self.total_blocks, self.args, self.runtime_state, self.prepare_events[i], self.stop_event, self.error_queue),
                daemon=True,
            )
            for i in range(1, self.args.num_gpus - 1)
        ]
        self.p_output = Process(
            target=output_process,
            args=(self.args.num_gpus - 1, self.total_block_num, self.total_blocks, self.args, self.runtime_state, self.prepare_events[-1], self.stop_event, self.output_queue, self.error_queue),
            daemon=True,
        )
        self.processes = [self.p_input] + self.p_middles + [self.p_output]

        for process in self.processes:
            process.start()

        wait_for_processes_ready(
            processes=self.processes,
            ready_events=self.prepare_events,
            error_queue=self.error_queue,
        )


def _runtime_flags_locked() -> bool:
    return os.environ.get("STREAMV2V_LOCK_RUNTIME_FLAGS", "").lower() in {"1", "true", "yes", "on"}


def _resolve_requested_runtime_flag(runtime_state, key: str, current: bool) -> bool:
    """Always return the launch-time value.

    use_taehv / use_tensorrt are pinned at startup and must NEVER be
    toggled at runtime — rebuilding the pipeline mid-session causes
    output stalls and is extremely expensive on 1.3B+ models.
    """
    return current


def _rebuild_pipeline_for_runtime_options(
    args,
    device,
    rank,
    world_size,
    current_manager,
    requested_use_taehv: bool,
    requested_use_tensorrt: bool,
):
    # ── HARD LOCK: never rebuild at runtime ──
    # use_taehv and use_tensorrt are set once at startup.  Rebuilding
    # the pipeline mid-session is prohibitively expensive and causes
    # output stalls.  Always return the existing manager unchanged.
    if current_manager is not None:
        current_manager.logger.info(
            "[HARD_LOCK] Ignoring rebuild request on rank %s "
            "(requested use_taehv=%s, use_tensorrt=%s); keeping launch-time settings.",
            rank,
            requested_use_taehv,
            requested_use_tensorrt,
        )
        return current_manager
    # When STREAMV2V_LOCK_RUNTIME_FLAGS=1, ignore frontend toggle of
    # use_taehv/use_tensorrt to avoid rebuilding the (very large) 14B FSDP
    # pipeline at runtime — rebuilds for 14B can OOM because old FSDP shards
    # are not always released cleanly before the new pipeline is allocated.
    # In that mode we keep the launch-time setting and just return the existing
    # manager so worker loops never re-enter the rebuild path.
    if os.environ.get("STREAMV2V_LOCK_RUNTIME_FLAGS", "").lower() in {"1", "true", "yes", "on"}:
        if current_manager is not None:
            current_manager.logger.info(
                "[LOCK_RUNTIME_FLAGS] Ignoring rebuild request on rank %s "
                "(requested use_taehv=%s, use_tensorrt=%s); keeping launch-time settings.",
                rank,
                requested_use_taehv,
                requested_use_tensorrt,
            )
        return current_manager
    if current_manager is not None:
        current_manager.logger.info(
            "Rebuilding demo rank %s for use_taehv=%s, use_tensorrt=%s",
            rank,
            requested_use_taehv,
            requested_use_tensorrt,
        )
        del current_manager
        torch.cuda.empty_cache()
    set_config_value(args, "use_taehv", requested_use_taehv)
    set_config_value(args, "use_tensorrt", requested_use_tensorrt)
    return prepare_pipeline(args, device, rank, world_size)


def input_process(rank, block_num, total_blocks, args, runtime_state, prepare_event, restart_event, stop_event, input_queue, error_queue):
    torch.set_grad_enabled(False)
    try:
        device = resolve_worker_device(args.gpu_ids, rank)
        torch.cuda.set_device(device)
        init_dist_tcp(rank, args.num_gpus, device=device)
        block_num = torch.tensor(block_num, dtype=torch.int64, device=device)

        current_use_taehv = bool(runtime_state.get("use_taehv", getattr(args, "use_taehv", False)))
        current_use_tensorrt = bool(runtime_state.get("use_tensorrt", getattr(args, "use_tensorrt", False)))
        set_config_value(args, "use_taehv", current_use_taehv)
        set_config_value(args, "use_tensorrt", current_use_tensorrt)
        pipeline_manager = prepare_pipeline(args, device, rank, args.num_gpus)
        num_steps = len(pipeline_manager.pipeline.denoising_step_list)
        chunk_size = pipeline_manager.get_demo_chunk_size()
        first_batch_num_frames = pipeline_manager.get_demo_first_batch_num_frames()
        is_running = False
        prompt = runtime_state["prompt"]
        schedule_block = args.schedule_block

        torch.cuda.memory._record_memory_history(max_entries=100000)

        prepare_event.set()

        while not stop_event.is_set():
            requested_use_taehv = _resolve_requested_runtime_flag(runtime_state, "use_taehv", current_use_taehv)
            requested_use_tensorrt = _resolve_requested_runtime_flag(runtime_state, "use_tensorrt", current_use_tensorrt)
            if requested_use_taehv != current_use_taehv or requested_use_tensorrt != current_use_tensorrt:
                if is_running and "session" in locals() and "denoised_pred" in locals() and "patched_x_shape" in locals():
                    pipeline_manager.send_demo_input_prompt_update(
                        prompt=runtime_state["prompt"],
                        device=device,
                        num_steps=num_steps,
                        chunk_idx=session.chunk_idx,
                        denoised_pred=denoised_pred,
                        patched_x_shape=patched_x_shape,
                        current_step=session.current_step,
                    )
                current_use_taehv = requested_use_taehv
                current_use_tensorrt = requested_use_tensorrt
                pipeline_manager = _rebuild_pipeline_for_runtime_options(
                    args,
                    device,
                    rank,
                    args.num_gpus,
                    pipeline_manager,
                    current_use_taehv,
                    current_use_tensorrt,
                )
                num_steps = len(pipeline_manager.pipeline.denoising_step_list)
                chunk_size = pipeline_manager.get_demo_chunk_size()
                first_batch_num_frames = pipeline_manager.get_demo_first_batch_num_frames()
                prompt = runtime_state["prompt"]
                schedule_block = args.schedule_block
                is_running = False
                clear_queue(input_queue)
                restart_event.clear()
                continue

            if is_running and (runtime_state["prompt"] != prompt or restart_event.is_set()):
                if restart_event.is_set():
                    clear_queue(input_queue)
                    restart_event.clear()
                prompt = runtime_state["prompt"]
                pipeline_manager.send_demo_input_prompt_update(
                    prompt=prompt,
                    device=device,
                    num_steps=num_steps,
                    chunk_idx=session.chunk_idx,
                    denoised_pred=denoised_pred,
                    patched_x_shape=patched_x_shape,
                    current_step=session.current_step,
                )
                is_running = False
                outstanding = []

            if not is_running:
                # First batch: downstream ranks are blocked on
                # _receive_initial_noise / dist.barrier waiting for rank 0
                # to send the very first collective. rank 0 cannot send
                # until the first frame arrives from a real client. We
                # therefore wait indefinitely for a client here (only
                # stop_event can break us out). The large NCCL timeout set
                # in init_dist_tcp (30 min default) gives this plenty of
                # headroom before the process group self-aborts.
                _s07_t0 = time.time()
                images = read_images_from_queue(
                    input_queue, first_batch_num_frames, device, stop_event,
                    idle_timeout_sec=None,
                )
                runtime_state["s07_read_queue_ms"] = (time.time() - _s07_t0) * 1000
                if images is None:
                    # stop_event was set => graceful shutdown.
                    return
                pipeline_manager.logger.info(f"Initializing rank {rank} first batch")
                session = pipeline_manager.start_demo_input_stream_session(
                    prompt=prompt,
                    images=images,
                    block_num=block_num[rank],
                    noise_scale=args.noise_scale,
                )
                outstanding = []
                pipeline_manager.logger.info(f"Starting rank {rank} inference loop")

            pipeline_manager.maybe_refresh_demo_input_window(session)

            if schedule_block:
                pipeline_manager._sync_for_timing(schedule_block)
                start_vae = time.time()

            if session.input_batch == 0:
                # Mid-session read: bound the wait so that if the client
                # disconnects we escape BEFORE NCCL watchdog fires on the
                # downstream ranks (which are blocked in recv_latent_data).
                # Default 15s is well under the 60s NCCL timeout we set
                # in init_dist_tcp.
                _s07_t0 = time.time()
                images = read_images_from_queue(
                    input_queue, chunk_size, device, stop_event,
                    idle_timeout_sec=float(os.environ.get("STREAMV2V_CHUNK_IDLE_SEC", "15")),
                )
                runtime_state["s07_read_queue_ms"] = (time.time() - _s07_t0) * 1000
                if images is None:
                    # CRITICAL: rank 0's input queue went idle mid-stream
                    # (WebSocket disconnect / pipeline.close()). The other
                    # ranks are blocked in receive_latent_data_async on the
                    # next chunk. If we simply `break`, they deadlock for
                    # the full NCCL timeout. Send a prompt-restart pill
                    # (chunk_idx=-1) so rank 1/2/3 fall out of the current
                    # session and wait for a fresh prompt on recv_prompt_async.
                    # This uses the existing, tested "prompt update" path.
                    pipeline_manager.logger.info(
                        "rank 0 input went idle mid-stream; sending chunk_idx=-1 "
                        "sentinel to unblock downstream ranks (session end)."
                    )
                    try:
                        if "denoised_pred" in locals() and "patched_x_shape" in locals():
                            pipeline_manager.send_demo_input_prompt_update(
                                prompt=runtime_state["prompt"],
                                device=device,
                                num_steps=num_steps,
                                chunk_idx=session.chunk_idx,
                                denoised_pred=denoised_pred,
                                patched_x_shape=patched_x_shape,
                                current_step=session.current_step,
                            )
                    except Exception as sentinel_exc:  # noqa: BLE001
                        pipeline_manager.logger.warning(
                            "Failed to send end-of-session sentinel to downstream "
                            "ranks (they may deadlock until NCCL timeout): %s",
                            sentinel_exc,
                        )
                    # After the sentinel, reset local state so the next user
                    # session starts cleanly without retrying NCCL ops that
                    # the downstream ranks will no longer answer on this
                    # sequence number.
                    is_running = False
                    outstanding = []
                    if stop_event.is_set():
                        return
                    continue
                pipeline_manager.prepare_demo_input_batch(session, images)

            if schedule_block:
                pipeline_manager._sync_for_timing(schedule_block)
                start_dit = time.time()
                t_vae = start_dit - start_vae
                # ── Stage 8: VAE encode timing from schedule_block ──
                runtime_state["s08_vae_encode_ms"] = t_vae * 1000

            # ── Stage 9: DiT rank0 ──
            _s09_t0 = time.time()
            denoised_pred, patched_x_shape = pipeline_manager.run_demo_input_step(
                session=session,
                block_num=block_num[rank],
                previous_latent_data=latent_data if "latent_data" in locals() else None,
            )
            runtime_state["s09_dit_rank0_ms"] = (time.time() - _s09_t0) * 1000

            if schedule_block:
                pipeline_manager._sync_for_timing(schedule_block)
                temp = time.time() - start_dit
                if temp < pipeline_manager.t_dit:
                    pipeline_manager.t_dit = temp

            pipeline_manager.processed += 1

            with torch.cuda.stream(pipeline_manager.com_stream):
                if pipeline_manager.processed >= pipeline_manager.world_size:
                    if "latent_data" in locals():
                        pipeline_manager.data_transfer.release_latent_data(latent_data)
                    latent_data = pipeline_manager.data_transfer.receive_latent_data_async(num_steps)

            torch.cuda.current_stream().wait_stream(pipeline_manager.com_stream)
            pipeline_manager._wait_for_outstanding(outstanding)

            # ── Stage 10: NCCL send_latent_data_async ──
            _s10_t0 = time.time()
            with torch.cuda.stream(pipeline_manager.com_stream):
                work_objects = pipeline_manager.data_transfer.send_latent_data_async(
                    chunk_idx=session.chunk_idx,
                    latents=denoised_pred,
                    original_latents=pipeline_manager.pipeline.hidden_states,
                    patched_x_shape=patched_x_shape,
                    current_start=pipeline_manager.pipeline.kv_cache_starts,
                    current_end=pipeline_manager.pipeline.kv_cache_ends,
                    current_step=session.current_step,
                )
                outstanding.append(work_objects)
                if schedule_block and pipeline_manager.processed >= pipeline_manager.schedule_step:
                    pipeline_manager._handle_block_scheduling(block_num, total_blocks=total_blocks)
                    schedule_block = False
            runtime_state["s10_nccl_send_ms"] = (time.time() - _s10_t0) * 1000

            if schedule_block:
                t_total = pipeline_manager.t_dit + t_vae
                if t_total < pipeline_manager.t_total:
                    pipeline_manager.t_total = t_total

            pipeline_manager.advance_demo_input_stream_session(session, images)
            is_running = True
    except Exception:
        report_worker_error(error_queue, f"multi_gpu_input_rank_{rank}")
        raise


def output_process(rank, block_num, total_blocks, args, runtime_state, prepare_event, stop_event, output_queue, error_queue):
    torch.set_grad_enabled(False)
    try:
        device = resolve_worker_device(args.gpu_ids, rank)
        torch.cuda.set_device(device)
        init_dist_tcp(rank, args.num_gpus, device=device)
        block_num = torch.tensor(block_num, dtype=torch.int64, device=device)

        current_use_taehv = bool(runtime_state.get("use_taehv", getattr(args, "use_taehv", False)))
        current_use_tensorrt = bool(runtime_state.get("use_tensorrt", getattr(args, "use_tensorrt", False)))
        set_config_value(args, "use_taehv", current_use_taehv)
        set_config_value(args, "use_tensorrt", current_use_tensorrt)
        pipeline_manager = prepare_pipeline(args, device, rank, args.num_gpus)
        num_steps = len(pipeline_manager.pipeline.denoising_step_list)
        prompt = runtime_state["prompt"]
        is_running = False
        need_update_prompt = False
        schedule_block = args.schedule_block
        prepare_event.set()

        while not stop_event.is_set():
            if need_update_prompt:
                prompt = pipeline_manager.data_transfer.recv_prompt_async()
                is_running = False
                need_update_prompt = False
                outstanding = []

            requested_use_taehv = _resolve_requested_runtime_flag(runtime_state, "use_taehv", current_use_taehv)
            requested_use_tensorrt = _resolve_requested_runtime_flag(runtime_state, "use_tensorrt", current_use_tensorrt)
            if requested_use_taehv != current_use_taehv or requested_use_tensorrt != current_use_tensorrt:
                current_use_taehv = requested_use_taehv
                current_use_tensorrt = requested_use_tensorrt
                pipeline_manager = _rebuild_pipeline_for_runtime_options(
                    args,
                    device,
                    rank,
                    args.num_gpus,
                    pipeline_manager,
                    current_use_taehv,
                    current_use_tensorrt,
                )
                num_steps = len(pipeline_manager.pipeline.denoising_step_list)
                prompt = runtime_state["prompt"]
                schedule_block = args.schedule_block
                is_running = False
                need_update_prompt = False
                clear_queue(output_queue)
                continue

            if not is_running:
                pipeline_manager.logger.info(f"Initializing rank {rank} first batch")
                images = pipeline_manager.prepare_demo_worker_session(
                    prompt=prompt,
                    block_mode="output",
                    block_num=block_num[rank],
                    decode_initial=True,
                )
                for image in images:
                    output_queue.put(image)
                outstanding = []
                pipeline_manager.logger.info(f"Starting rank {rank} inference loop")

            latent_data = pipeline_manager._receive_latent_data(latent_data if "latent_data" in locals() else None, num_steps)
            if latent_data.chunk_idx == -1:
                need_update_prompt = True
                continue
            schedule_block = pipeline_manager._maybe_schedule_blocks(
                schedule_block,
                pipeline_manager.schedule_step - rank,
                block_num,
                total_blocks=total_blocks,
            )

            if schedule_block:
                pipeline_manager._sync_for_timing(schedule_block)
                start_dit = time.time()

            # ── Stage 11: DiT last rank ──
            _s11_t0 = time.time()
            denoised_pred, _ = pipeline_manager._run_worker_stage("output", latent_data, block_num[rank])
            runtime_state["s11_dit_last_rank_ms"] = (time.time() - _s11_t0) * 1000

            if schedule_block:
                pipeline_manager._sync_for_timing(schedule_block)
                temp = time.time() - start_dit
                if temp < pipeline_manager.t_dit:
                    pipeline_manager.t_dit = temp

            pipeline_manager.processed += 1
            pipeline_manager._wait_for_outstanding(outstanding)
            pipeline_manager._send_worker_result("output", outstanding, latent_data, denoised_pred)

            if pipeline_manager.processed >= num_steps * pipeline_manager.world_size - 1:
                if schedule_block:
                    pipeline_manager._sync_for_timing(schedule_block)
                    start_vae = time.time()

                # ── Stage 12: VAE decode ──
                _s12_t0 = time.time()
                decoded_images = list(pipeline_manager._decode_prediction(denoised_pred))
                runtime_state["s12_vae_decode_ms"] = (time.time() - _s12_t0) * 1000

                # ── Stage 13: output_queue.put ──
                _s13_t0 = time.time()
                for image in decoded_images:
                    output_queue.put(image)
                runtime_state["s13_output_queue_put_ms"] = (time.time() - _s13_t0) * 1000

                torch.cuda.synchronize()

                if schedule_block:
                    t_vae = time.time() - start_vae
                    t_total = t_vae + pipeline_manager.t_dit
                    if t_total < pipeline_manager.t_total:
                        pipeline_manager.t_total = t_total

            is_running = True
    except Exception:
        report_worker_error(error_queue, f"multi_gpu_output_rank_{rank}")
        raise


def middle_process(rank, block_num, total_blocks, args, runtime_state, prepare_event, stop_event, error_queue):
    torch.set_grad_enabled(False)
    try:
        device = resolve_worker_device(args.gpu_ids, rank)
        torch.cuda.set_device(device)
        init_dist_tcp(rank, args.num_gpus, device=device)
        block_num = torch.tensor(block_num, dtype=torch.int64, device=device)

        current_use_taehv = bool(runtime_state.get("use_taehv", getattr(args, "use_taehv", False)))
        current_use_tensorrt = bool(runtime_state.get("use_tensorrt", getattr(args, "use_tensorrt", False)))
        set_config_value(args, "use_taehv", current_use_taehv)
        set_config_value(args, "use_tensorrt", current_use_tensorrt)
        pipeline_manager = prepare_pipeline(args, device, rank, args.num_gpus)
        num_steps = len(pipeline_manager.pipeline.denoising_step_list)
        prompt = runtime_state["prompt"]
        is_running = False
        need_update_prompt = False
        schedule_block = args.schedule_block

        prepare_event.set()

        while not stop_event.is_set():
            if need_update_prompt:
                prompt = pipeline_manager.data_transfer.recv_prompt_async()
                pipeline_manager.logger.info(f"Rank {rank} sending dummy data")
                pipeline_manager.send_demo_middle_prompt_update(
                    prompt=prompt,
                    device=device,
                    denoised_pred=denoised_pred,
                    latent_data=latent_data,
                )
                is_running = False
                need_update_prompt = False
                outstanding = []

            requested_use_taehv = _resolve_requested_runtime_flag(runtime_state, "use_taehv", current_use_taehv)
            requested_use_tensorrt = _resolve_requested_runtime_flag(runtime_state, "use_tensorrt", current_use_tensorrt)
            if requested_use_taehv != current_use_taehv or requested_use_tensorrt != current_use_tensorrt:
                current_use_taehv = requested_use_taehv
                current_use_tensorrt = requested_use_tensorrt
                pipeline_manager = _rebuild_pipeline_for_runtime_options(
                    args,
                    device,
                    rank,
                    args.num_gpus,
                    pipeline_manager,
                    current_use_taehv,
                    current_use_tensorrt,
                )
                num_steps = len(pipeline_manager.pipeline.denoising_step_list)
                prompt = runtime_state["prompt"]
                schedule_block = args.schedule_block
                is_running = False
                need_update_prompt = False
                continue

            if not is_running:
                pipeline_manager.logger.info(f"Initializing rank {rank} first batch")
                pipeline_manager.prepare_demo_worker_session(
                    prompt=prompt,
                    block_mode="middle",
                    block_num=block_num[rank],
                )
                outstanding = []
                pipeline_manager.logger.info(f"Starting rank {rank} inference loop")

            latent_data = pipeline_manager._receive_latent_data(latent_data if "latent_data" in locals() else None, num_steps)
            if latent_data.chunk_idx == -1:
                need_update_prompt = True
                continue
            schedule_block = pipeline_manager._maybe_schedule_blocks(
                schedule_block,
                pipeline_manager.schedule_step - rank,
                block_num,
                total_blocks=total_blocks,
            )

            if schedule_block:
                pipeline_manager._sync_for_timing(schedule_block)
                start_dit = time.time()

            denoised_pred, _ = pipeline_manager._run_worker_stage("middle", latent_data, block_num[rank])

            if schedule_block:
                pipeline_manager._sync_for_timing(schedule_block)
                temp = time.time() - start_dit
                if temp < pipeline_manager.t_dit:
                    pipeline_manager.t_dit = temp

            pipeline_manager.processed += 1
            pipeline_manager._wait_for_outstanding(outstanding)
            pipeline_manager._send_worker_result("middle", outstanding, latent_data, denoised_pred)

            torch.cuda.synchronize()

            if schedule_block:
                t_total = pipeline_manager.t_dit
                if t_total < pipeline_manager.t_total:
                    pipeline_manager.t_total = t_total

            is_running = True
    except Exception:
        report_worker_error(error_queue, f"multi_gpu_middle_rank_{rank}")
        raise


def init_dist_tcp(rank: int, world_size: int, master_addr: str = "127.0.0.1", master_port: int = 29500, device: torch.device = None):
    # Allow overriding the master port via env so multiple demo services can
    # coexist on the same host (e.g. 14B on :7862 and 1.3B on :7863 each need
    # their own torch.distributed rendezvous port).
    master_port = int(os.environ.get("STREAMV2V_MASTER_PORT", master_port))
    master_addr = os.environ.get("STREAMV2V_MASTER_ADDR", master_addr)
    # NCCL collective-op timeout. PyTorch's default is 10 minutes. We keep
    # it LONG here (30 min by default) because the downstream ranks block on
    # `_receive_initial_noise` / `dist.barrier` during first-batch setup
    # waiting for rank 0 to encode the first frame — which itself waits for
    # a real client to connect and push frames. If this timeout is shorter
    # than the "time to first client", the demo deadlocks even on a cold
    # start with no bugs.
    #
    # Mid-session deadlocks (client disconnects while rank 1/2/3 wait for
    # the next NCCL send from rank 0) are handled at a HIGHER layer by the
    # STREAMV2V_CHUNK_IDLE_SEC path in input_process(), which sends a
    # chunk_idx=-1 sentinel to unblock downstream ranks within ~15s — long
    # before NCCL would time out anyway.
    #
    # Env override: STREAMV2V_NCCL_TIMEOUT_SEC (seconds; default 1800).
    timeout_sec = int(os.environ.get("STREAMV2V_NCCL_TIMEOUT_SEC", "1800"))
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
        timeout=timedelta(seconds=timeout_sec),
    )


def prepare_pipeline(args, device, rank, world_size):
    pipeline_manager = InferencePipelineManager(args, device, rank, world_size)
    pipeline_manager.load_model(args.checkpoint_folder)
    return pipeline_manager

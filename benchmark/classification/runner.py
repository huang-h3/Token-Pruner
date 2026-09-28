"""Classification experiment runner for typed vision-encoder sessions."""

import time
from pathlib import Path

import decord
import torch

from token_pruner.progress import ProgressReporter
from token_pruner.records import (
    format_measurement_summary,
    weighted_metric_mean,
)
from token_pruner.measurements import SpanAccumulator
from token_pruner.task_io import exception_summary
from token_pruner.hevc_cache import prepare_hevc_batch_selection

from .data import append_skipped_video, build_filtered_loader, read_rgb_frames
from .measurements import ClassificationMeasurements, TIMING_FIELDS
from .profiling import (
    profile_record as cuda_profile_record,
    profiled_active_steps,
    profiler_ctx_func,
    sum_profiler_flops,
    warmup,
)


class _TokenReduceTimer:
    """Collect nested time without synchronizing inside the adapter."""

    def __init__(self, args, device):
        self.args = args
        self.spans = SpanAccumulator(device)

    def __call__(self, operation):
        with cuda_profile_record("token_reduce", self.args):
            return self.spans.record(operation)

    def elapsed_ms(self):
        return self.spans.total_ms()


def _timed_classification_forward(
    session,
    pixel_values,
    selector_outputs,
    variant,
    args,
    device,
):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    token_reduce = _TokenReduceTimer(args, device)
    with (
        torch.inference_mode(),
        cuda_profile_record("vision_classification_forward", args),
    ):
        torch.cuda.reset_peak_memory_stats(device)
        start_mem = torch.cuda.memory_allocated(device)
        start_event.record()
        if variant == "pruned":
            logits = session.forward_pruned(
                pixel_values,
                selector_outputs,
                token_reduce,
            )
        elif variant == "full":
            logits = session.forward_full(pixel_values)
        else:
            raise RuntimeError(f"Unknown classification variant {variant!r}.")
        end_event.record()
        torch.cuda.synchronize(device)
        peak_mem = torch.cuda.max_memory_allocated(device)
    return (
        logits,
        start_event.elapsed_time(end_event),
        token_reduce.elapsed_ms(),
        peak_mem,
        peak_mem - start_mem,
    )


def _hevc_selector(session, selector_indices, n_parallel, device):
    """The ``selector_fn`` that ``prepare_hevc_batch_selection`` calls once per batch."""

    def select(paths, kept_indices, cache_keys, score_cache_dir):
        return session.select(
            [str(path) for path in paths],
            [selector_indices[index] for index in kept_indices],
            n_parallel=n_parallel,
            device=device,
            cache_keys=cache_keys,
            score_cache_dir=score_cache_dir,
        )

    return select


def _warm_variant(session, pixel_values, selector_outputs, variant, device):
    with torch.inference_mode():
        if variant == "pruned":
            session.forward_pruned(pixel_values, selector_outputs)
        else:
            session.forward_full(pixel_values)
    torch.cuda.synchronize(device)


def _profiled_flops(profiler, profiler_steps):
    active_steps = profiled_active_steps(profiler_steps)
    return {
        "profiled_active_batches": active_steps,
        "profiled_total_flops": sum_profiler_flops(profiler),
        "profiled_vision_classification_forward_flops": sum_profiler_flops(
            profiler,
            "vision_classification_forward",
        ),
        "profiled_token_reduce_flops": sum_profiler_flops(
            profiler,
            "token_reduce",
        ),
    }


def _print_profiler_flops(profiler_flops, args):
    print("----------------------Profiler FLOPs----------------------")
    active = profiler_flops["profiled_active_batches"]
    if active <= 0:
        print("No active profiler batches were recorded.")
        return

    suffix = f"batch avg over {active} profiled batches, batch size {args.batch_size}"
    for key, label in (
        ("profiled_total_flops", "Total"),
        (
            "profiled_vision_classification_forward_flops",
            "Vision classification forward",
        ),
        ("profiled_token_reduce_flops", "Token reduction"),
    ):
        print(f"{label} FLOPs - {profiler_flops[key] / active / 1e9:.2f} GFLOPs, {suffix}")


def _variant_summary(measurements, variant):
    """Sample-weighted batch summary, matching what the CSV reports."""
    records = measurements.records_for(variant)
    return {
        name: weighted_metric_mean(records, name) or 0.0
        for name in (
            *TIMING_FIELDS,
            "peak_mem_mb",
            "delta_peak_mem_mb",
        )
    }


def _print_final(measurements, session, summary_label):
    print("----------------------Final results----------------------")
    for variant, enabled, label, top1, top5 in (
        (
            "pruned",
            session.run_pruned,
            summary_label,
            measurements.pruned_top1,
            measurements.pruned_top5,
        ),
        (
            "full",
            session.run_full,
            "Full",
            measurements.full_top1,
            measurements.full_top5,
        ),
    ):
        if not enabled:
            continue
        summary = _variant_summary(measurements, variant)
        timing_line, memory_line = format_measurement_summary(summary)
        print(
            f"{label}: top1={top1 / measurements.total:.4f}, "
            f"top5={top5 / measurements.total:.4f} | "
            f"{timing_line} | {memory_line}"
        )


def run_classification_inference(
    args,
    session,
    device,
    *,
    dataset,
    summary_label="Pruned",
    timing_schema="classification_batch_records_v1",
):
    dataset_filter_stats, loader, skipped_videos_path = build_filtered_loader(
        dataset,
        args.batch_size,
        args.skipped_videos_path,
    )
    warmup(args, session.model, session.num_frames, session.image_size, device)
    measurements = ClassificationMeasurements()
    profiler_steps = 0
    warmed_variants = set()

    print("\n----------------------Inference process-----------------------\n")
    print(f"\n {session.info} \n")

    with (
        profiler_ctx_func(args) as profiler,
        ProgressReporter(total=len(dataset), every=20, label="samples") as reporter,
    ):
        for step, batch in enumerate(loader, start=1):
            input_video_paths = [Path(path) for path in batch["video_paths"]]
            labels_cpu = batch["labels"]
            video_paths = []
            frames_batch = []
            sampled_frames_batch = []
            sampled_frame_indices_batch = []
            request_batch = []
            valid_labels = []
            batch_preprocessing_ms = 0.0

            for video_path, label in zip(input_video_paths, labels_cpu):
                start = time.perf_counter()
                try:
                    frames, frame_indices = read_rgb_frames(
                        video_path,
                        session.num_frames,
                    )
                except (
                    decord.DECORDError,
                    decord.DECORDLimitReachedError,
                    OSError,
                    ValueError,
                ) as exc:
                    batch_preprocessing_ms += (time.perf_counter() - start) * 1000.0
                    append_skipped_video(skipped_videos_path, video_path.name)
                    measurements.decode_skipped_samples += 1
                    print(
                        f"Skipping unreadable video {video_path.name}: "
                        f"{exception_summary(exc)}",
                        flush=True,
                    )
                    continue
                decode_ms = (time.perf_counter() - start) * 1000.0
                batch_preprocessing_ms += decode_ms
                video_paths.append(video_path)
                frames_batch.append(list(frames))
                sampled_frames_batch.append(frames)
                sampled_frame_indices_batch.append(frame_indices)
                request_batch.append(
                    {
                        "request_id": video_path.name,
                        "decode_ms": decode_ms,
                    }
                )
                valid_labels.append(int(label))

            if not valid_labels:
                if args.profiler:
                    profiler.step()
                    profiler_steps += 1
                continue

            selector_outputs = None
            hevc_encode_ms = 0.0
            hevc_encode_paid_ms = 0.0
            signal_to_mask_ms = 0.0
            signal_prepare_wall_ms = None
            if session.needs_hevc:
                full_video = session.hevc_encode_scope == "full-video"
                hevc_selector_indices = (
                    sampled_frame_indices_batch
                    if full_video
                    else [
                        list(range(len(frames)))
                        for frames in sampled_frames_batch
                    ]
                )
                hevc_sampled_frames = (
                    None if full_video else sampled_frames_batch
                )
                # Whole-stage wall clock; the remainder lands in signal_overhead_ms.
                signal_start = time.perf_counter()
                with cuda_profile_record("signal", args):
                    preparation = prepare_hevc_batch_selection(
                        video_paths,
                        args.hevc_dir,
                        _hevc_selector(
                            session, hevc_selector_indices, args.hevc_n_parallel, device
                        ),
                        device,
                        sampled_frames=hevc_sampled_frames,
                        sampled_gop_size=session.hevc_gop_size,
                        encode_scope=session.hevc_encode_scope,
                        hevc_permanent_dir=args.hevc_permanent_dir,
                    )
                for failed_index in preparation.failed_indices:
                    append_skipped_video(
                        skipped_videos_path,
                        video_paths[failed_index].name,
                    )
                measurements.hevc_skipped_samples += len(preparation.failed_indices)
                kept = preparation.kept_indices
                frames_batch = [frames_batch[index] for index in kept]
                valid_labels = [valid_labels[index] for index in kept]
                request_batch = [request_batch[index] for index in kept]
                effective_gop_sizes = preparation.hevc_effective_gop_sizes
                gop_history = session.hevc_effective_gop_sizes
                gop_history.extend(
                    value for value in effective_gop_sizes if value is not None
                )
                if gop_history:
                    unique_gops = set(gop_history)
                    session.hevc_effective_gop_size = (
                        next(iter(unique_gops)) if len(unique_gops) == 1 else None
                    )
                for request, encode_ms, paid_ms, cache_hit, effective_gop in zip(
                    request_batch,
                    preparation.hevc_encode_ms_per_sample,
                    preparation.hevc_encode_paid_ms_per_sample,
                    preparation.hevc_cache_hits,
                    effective_gop_sizes,
                ):
                    request.update(
                        {
                            "hevc_encode_ms": float(encode_ms),
                            "hevc_encode_paid_ms": float(paid_ms),
                            "hevc_cache_hit": bool(cache_hit),
                            "hevc_effective_gop_size": effective_gop,
                        }
                    )
                selector_outputs = preparation.selector_outputs
                if not valid_labels:
                    if args.profiler:
                        profiler.step()
                        profiler_steps += 1
                    continue
                hevc_encode_ms = preparation.hevc_encode_ms
                hevc_encode_paid_ms = preparation.hevc_encode_paid_ms
                signal_to_mask_ms = preparation.selector_compute_ms
                signal_prepare_wall_ms = (
                    time.perf_counter() - signal_start
                ) * 1000.0
            elif session.needs_signal:
                start = time.perf_counter()
                with cuda_profile_record("signal", args):
                    selector_outputs = session.select(
                        [str(path) for path in video_paths],
                        sampled_frame_indices_batch,
                        n_parallel=args.hevc_n_parallel,
                        device=device,
                    )
                    torch.cuda.synchronize(device)
                signal_to_mask_ms = (time.perf_counter() - start) * 1000.0
                signal_prepare_wall_ms = signal_to_mask_ms

            start = time.perf_counter()
            labels = torch.tensor(
                valid_labels,
                dtype=labels_cpu.dtype,
                device=device,
            )
            pixel_values = session.process_frames(frames_batch, device)
            torch.cuda.synchronize(device)
            batch_preprocessing_ms += (time.perf_counter() - start) * 1000.0

            variants = []
            if session.run_pruned:
                variants.append("pruned")
            if session.run_full:
                variants.append("full")
            if len(variants) == 2 and step % 2 == 0:
                variants.reverse()

            records = {}
            for variant in variants:
                if variant not in warmed_variants:
                    _warm_variant(
                        session,
                        pixel_values,
                        selector_outputs,
                        variant,
                        device,
                    )
                    warmed_variants.add(variant)
                (
                    logits,
                    forward_ms,
                    token_reduce_ms,
                    peak_mem,
                    delta_mem,
                ) = _timed_classification_forward(
                    session,
                    pixel_values,
                    selector_outputs,
                    variant,
                    args,
                    device,
                )
                records[variant] = measurements.record_batch(
                    variant=variant,
                    batch_index=step,
                    sample_count=labels.numel(),
                    preprocessing_ms=batch_preprocessing_ms,
                    hevc_encode_ms=hevc_encode_ms if variant == "pruned" else 0.0,
                    hevc_encode_paid_ms=(
                        hevc_encode_paid_ms if variant == "pruned" else 0.0
                    ),
                    signal_to_mask_ms=(
                        signal_to_mask_ms if variant == "pruned" else 0.0
                    ),
                    signal_prepare_wall_ms=(
                        signal_prepare_wall_ms if variant == "pruned" else None
                    ),
                    token_reduce_ms=token_reduce_ms,
                    vision_classification_forward_ms=forward_ms,
                    peak_mem_mb=peak_mem / (1024**2),
                    delta_peak_mem_mb=delta_mem / (1024**2),
                )
                measurements.record_accuracy(variant, logits, labels)

            single_request = labels.numel() == 1
            for sample_index, request in enumerate(request_batch):
                for variant, batch_record in records.items():
                    batch_fields = {
                        name: batch_record[name] if single_request else None
                        for name in TIMING_FIELDS
                    }
                    batch_fields.update(
                        {
                            name: batch_record[name] if single_request else None
                            for name in ("peak_mem_mb", "delta_peak_mem_mb")
                        }
                    )
                    if not single_request:
                        batch_fields["hevc_encode_ms"] = (
                            request.get("hevc_encode_ms", 0.0)
                            if variant == "pruned"
                            else 0.0
                        )
                    measurements.record_request(
                        {
                            "variant": variant,
                            "measurement_scope": "request",
                            "request_index": measurements.total + sample_index,
                            "request_id": request["request_id"],
                            "batch_index": step,
                            "batch_size_effective": labels.numel(),
                            "decode_ms": request["decode_ms"],
                            "hevc_encode_paid_ms": (
                                request.get("hevc_encode_paid_ms", 0.0)
                                if variant == "pruned"
                                else 0.0
                            ),
                            "hevc_cache_hit": (
                                request.get("hevc_cache_hit", False)
                                if variant == "pruned"
                                else False
                            ),
                            **batch_fields,
                        }
                    )

            measurements.total += labels.numel()
            reporter.update(
                records.get("pruned") or records["full"],
                count=labels.numel(),
            )
            if args.profiler:
                profiler.step()
                profiler_steps += 1

    profiler_flops = None
    if args.profiler:
        profiler_flops = _profiled_flops(profiler, profiler_steps)
        _print_profiler_flops(profiler_flops, args)
    if measurements.total == 0:
        raise RuntimeError("No valid videos were processed.")

    _print_final(measurements, session, summary_label)
    save_path = measurements.save(
        args,
        session,
        dataset,
        dataset_filter_stats,
        skipped_videos_path,
        timing_schema,
        profiler_flops,
    )
    print(f"Saved inference tensor to: {save_path}")
    return save_path

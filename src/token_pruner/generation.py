"""lmms-eval generation loop shared by the VLM adapters."""

import os
import random
import re
import time

import torch

# lmms-eval from https://github.com/EvolvingLMMs-Lab/lmms-eval is necessary to run this code.
from lmms_eval.protocol import ChatMessages

from .hevc import HEVC_CODEC_LAYOUT, HEVC_SIGNAL_POLICY, pick_encoder
from .measurements import (
    REQUEST_SCOPED_METRICS,
    build_batch_measurement,
    build_request_measurement,
    cuda_sync,
    new_generation_measurements,
    print_generation_summary,
    record_measurement,
    timed_inference_call,
)
from .progress import ProgressReporter
from .records import ResponseStore, resume_enabled, save_benchmark_result
from .selection import mode_semantics
from .task_io import (
    ANSWER_FORMAT_SUFFIX_ENV,
    GREEDY_DECODING_ENV,
    MAX_NEW_TOKENS_ENV,
    PROMPT_STYLE_ENV,
    QWEN3_VL_DECODING_POLICY_ENV,
    REASONING_SEGMENT_ENV,
    STOP_STRINGS_ENV,
    UNTRIMMED_SCORING_ENV,
    public_model_source,
    qwen3_vl_decoding_policy,
)
from .video_io import VIDEO_ABLATION_ENV


BATCHING_POLICY = "single-homogeneous-batch-v1"

#: Settings read from the environment that change what a run generates.
FINGERPRINT_ENVIRONMENT = (
    STOP_STRINGS_ENV,
    ANSWER_FORMAT_SUFFIX_ENV,
    UNTRIMMED_SCORING_ENV,
    REASONING_SEGMENT_ENV,
    MAX_NEW_TOKENS_ENV,
    GREEDY_DECODING_ENV,
    QWEN3_VL_DECODING_POLICY_ENV,
    PROMPT_STYLE_ENV,
    VIDEO_ABLATION_ENV,
    "HEVC_ENCODER",
)

#: A lettered option list marks MC; open-ended items keep their own wording.
_OPTION_LIST = re.compile(r"^\s*A[.)]\s.*^\s*B[.)]\s", re.MULTILINE | re.DOTALL)


def answer_format_suffix():
    """The configured answer-format instruction, or "" when disabled."""

    return os.environ.get(ANSWER_FORMAT_SUFFIX_ENV, "")


def apply_answer_format_suffix(hf_messages, suffix=None):
    """Append ``suffix`` to the last user turn of a multiple-choice prompt."""

    suffix = answer_format_suffix() if suffix is None else suffix
    if not suffix:
        return hf_messages
    for message in reversed(hf_messages):
        if message.get("role") != "user":
            continue
        blocks = [
            block
            for block in message.get("content", [])
            if block.get("type") == "text" and block.get("text")
        ]
        if not blocks:
            return hf_messages
        if not _OPTION_LIST.search("\n".join(block["text"] for block in blocks)):
            return hf_messages
        blocks[-1]["text"] = f"{blocks[-1]['text']}{suffix}"
        return hf_messages
    return hf_messages


def _qwen_decoding_policy(owner):
    return (
        qwen3_vl_decoding_policy()
        if str(owner.family).startswith("qwen3_vl")
        else None
    )


def _expected_records(owner, request_details):
    semantics = mode_semantics(
        owner.prune_mode,
        owner.i_mode,
        owner.p_mode,
        uniform_partitions=bool(owner.uniform_partitions),
        selector_id=owner.selector_id,
    )
    hevc_active = bool(
        owner.run_pruned
        and (
            owner.p_mode == "hevc"
            or (owner.p_mode == "shared_folding" and owner.folding_mode == "hevc-avg")
        )
    )
    metadata_anchor_policy = (
        "none" if hevc_active and owner.prune_mode == "shared"
        else owner.hevc_anchor_policy
    )
    configuration = {
        "run_mode": owner.run_mode,
        "prune_stage": owner.prune_stage,
        "prune_mode": owner.prune_mode,
        "prune_layer": owner.prune_layer,
        "i_mode": owner.i_mode if owner.run_pruned else None,
        "p_mode": owner.p_mode if owner.run_pruned else None,
        "selection_seed": (
            owner.selection_seed
            if owner.run_pruned and owner.p_mode == "random"
            else None
        ),
        "folding_mode": (
            owner.folding_mode
            if owner.run_pruned and owner.p_mode == "shared_folding"
            else None
        ),
        "fold_block_size": (
            owner.fold_block_size
            if owner.run_pruned and owner.p_mode == "shared_folding"
            else None
        ),
        "fold_pooling": (
            owner.fold_pooling
            if owner.run_pruned and owner.p_mode == "shared_folding"
            else None
        ),
        "k_keep_rate": owner.k_keep_rate,
        "model_source": public_model_source(owner.pretrained),
        "model_dtype": owner.model_dtype,
        "num_frames": owner.num_frames,
        "batch_size": owner.batch_size,
        "decoding_policy": _qwen_decoding_policy(owner),
        "score_reduce": owner.score_reduce,
        "hevc_n_parallel": owner.hevc_n_parallel,
        "hevc_encode_scope": owner.hevc_encode_scope if hevc_active else None,
        "hevc_gop_size": owner.hevc_gop_size if hevc_active else None,
        "hevc_anchor_policy": (
            metadata_anchor_policy if hevc_active else None
        ),
        "hevc_anchor_reducer": (
            None if owner.prune_mode == "shared" else owner.i_mode
        ) if hevc_active else None,
        "hevc_codec_layout": (HEVC_CODEC_LAYOUT if hevc_active else None),
        "hevc_signal_policy": (HEVC_SIGNAL_POLICY if hevc_active else None),
        "batching_policy": BATCHING_POLICY,
        "partition_rule": semantics.partition_rule if owner.run_pruned else None,
        "min_per_partition": (
            int(owner.min_cells_per_partition) if owner.run_pruned else 0
        ),
        "environment": {
            name: os.environ.get(name) for name in FINGERPRINT_ENVIRONMENT
        },
    }
    return [
        {
            "request_index": index,
            "doc_id": details[3],
            "task": details[4],
            "split": details[5],
            **configuration,
        }
        for index, details in enumerate(request_details)
    ]


def _restore_recorded_state(owner, records):
    for attribute, key in (
        ("effective_keep_totals", "effective_keep_total"),
        ("i_frame_counts", "i_frame_count"),
        ("p_keep_totals", "p_keep_total"),
    ):
        setattr(
            owner,
            attribute,
            [record[key] for record in records if record.get(key) is not None],
        )
    timing, memory = new_generation_measurements()
    batch_measurements = []
    seen_batches = set()
    for record in records:
        measurements = record.get("measurements") or {}
        record["measurements"] = measurements
        batch = record["batch_measurement"]
        new_batch = batch["batch_id"] not in seen_batches
        if new_batch:
            seen_batches.add(batch["batch_id"])
            batch_measurements.append(batch)
        for key in timing:
            if key in REQUEST_SCOPED_METRICS:
                value = measurements.get(key)
            elif new_batch:
                value = batch.get(key)
            else:
                continue
            if value is not None:
                timing[key].append(value)
        if new_batch:
            for key in memory:
                value = batch.get(key)
                if value is not None:
                    memory[key].append(value)
    return timing, memory, batch_measurements


def _batch_generation_seed(batch_details):
    """Stable RNG seed for one batch, shared by every arm."""

    key = "|".join(
        f"{task}|{split}|{doc_id}"
        for _context, _messages, _kwargs, doc_id, task, split in batch_details
    )
    return random.Random(key).getrandbits(63)


def _generate_batch(owner, inputs, generation_kwargs, stop_sequences, batch_details):
    # One task per batch, so `until` is shared.
    batch_stop_sequences = stop_sequences[0] if stop_sequences else None
    torch.manual_seed(_batch_generation_seed(batch_details))
    measurement = timed_inference_call(
        lambda: owner.backend.generate_tokens(
            inputs,
            generation_kwargs,
            batch_stop_sequences,
        ),
        owner.device,
    )
    response = owner.backend.decode_tokens(
        measurement.output,
        inputs,
        stop_sequences,
    )
    response_ready = time.perf_counter()
    batch_timing, request_timing = owner.backend.collect_batch_timing(
        measurement.output,
        inputs,
    )
    return response, measurement, batch_timing, request_timing, response_ready


def _save_results(
    owner,
    request_details,
    timing,
    memory,
    batch_measurements,
    *,
    replayed=False,
):
    pruning = owner.pruning
    pruning_config = (
        pruning.context.plan.config
        if owner.run_pruned and pruning.context.plan is not None
        else None
    )
    folding_options = (
        pruning_config.shared_folding if pruning_config is not None else None
    )
    semantics = mode_semantics(
        pruning.prune_mode,
        pruning.i_mode,
        pruning.p_mode,
        uniform_partitions=bool(owner.uniform_partitions),
        selector_id=owner.selector_id,
    )
    task_tag = ("+".join(sorted({str(d[4]) for d in request_details})) or "unknown_task")
    decoding_parameters = (
        owner.backend.decoding_parameters()
        if hasattr(owner.backend, "decoding_parameters")
        else None
    )
    params = {
        "model_type": owner.backend.family,
        "model_name": owner.backend.model_name,
        "model_path": public_model_source(owner.pretrained),
        "model_dtype": owner.model_dtype,
        "decoding_policy": _qwen_decoding_policy(owner),
        "decoding_parameters": decoding_parameters,
        "task": task_tag,
        "run_mode": owner.run_mode,
        "pruning_strategy": (
            ("shared-folding" if folding_options is not None else "grouped")
            if owner.run_pruned
            else None
        ),
        "folding_mode": (folding_options.mode if folding_options is not None else None),
        "fold_block_wise": (
            folding_options.block_wise if folding_options is not None else None
        ),
        "fold_block_size": (
            folding_options.block_size if folding_options is not None else None
        ),
        "fold_pooling": (
            folding_options.pooling if folding_options is not None else None
        ),
        "fold_slots_per_block": (
            folding_options.slots_per_block if folding_options is not None else None
        ),
        "prune_stage": pruning.prune_stage,
        "prune_mode": pruning.prune_mode,
        "prune_layer": pruning.prune_layer,
        "i_mode": pruning.i_mode if owner.run_pruned else None,
        "p_mode": pruning.p_mode if owner.run_pruned else None,
        "selection_seed": (
            owner.selection_seed
            if owner.run_pruned and pruning.p_mode == "random"
            else None
        ),
        "score_reduce": (
            owner.score_reduce
            if owner.run_pruned and pruning.prune_mode == "shared"
            else None
        ),
        "partition_rule": semantics.partition_rule if owner.run_pruned else None,
        "min_per_partition": (
            int(pruning_config.min_per_partition)
            if pruning_config is not None
            else 0
        ),
        "realized_keep_per_partition_min": (
            min(pruning.realized_keep_per_partition_min)
            if pruning.realized_keep_per_partition_min
            else None
        ),
        "realized_keep_per_partition_max": (
            max(pruning.realized_keep_per_partition_max)
            if pruning.realized_keep_per_partition_max
            else None
        ),
        "k_keep_rate": owner.k_keep_rate,
        "k_keep": pruning.keep_patches if owner.run_pruned else None,
        "total_patches_per_frame": pruning.total_patches,
        "target_keep_total": pruning.target_keep_total,
        "effective_keep_total": (
            sum(owner.effective_keep_totals) / len(owner.effective_keep_totals)
            if owner.effective_keep_totals
            else None
        ),
        "effective_keep_total_per_sample": owner.effective_keep_totals,
        "i_frame_count_per_sample": owner.i_frame_counts,
        "p_keep_total_per_sample": owner.p_keep_totals,
        "total_patches": pruning.total_patches_clip,
        "num_frames": owner.num_frames,
        "num_samples": len(request_details),
        "batch_size": owner.batch_size,
        "hevc_n_parallel": pruning.hevc_n_parallel,
        "hevc_encode_scope": (
            pruning.hevc_encode_scope if pruning.needs_hevc else None
        ),
        "hevc_gop_size": pruning.hevc_gop_size if pruning.needs_hevc else None,
        "hevc_effective_gop_size": (
            pruning.hevc_effective_gop_size if pruning.needs_hevc else None
        ),
        "hevc_effective_gop_sizes_per_sample": (
            list(pruning.hevc_effective_gop_sizes) if pruning.needs_hevc else []
        ),
        "hevc_artifact_keys_per_sample": (
            list(pruning.hevc_artifact_keys) if pruning.needs_hevc else []
        ),
        "hevc_anchor_policy": (
            "none" if pruning.prune_mode == "shared"
            else pruning.hevc_anchor_policy
        ) if pruning.needs_hevc else None,
        "hevc_anchor_reducer": (
            None if pruning.prune_mode == "shared" else pruning.i_mode
        ) if pruning.needs_hevc else None,
        "hevc_codec_layout": HEVC_CODEC_LAYOUT if pruning.needs_hevc else None,
        "hevc_signal_policy": HEVC_SIGNAL_POLICY if pruning.needs_hevc else None,
        "hevc_encoder": pick_encoder() if pruning.needs_hevc else None,
        "batching_policy": BATCHING_POLICY,
        "quantization": owner.backend.quantization,
        "vision_backend": owner.backend.vision_backend,
        "answers_path": owner.response_store.path.name,
        "timing_schema": "vlm_request_v1",
        "signal_definition": (
            "hevc_encode_ms_plus_signal_to_mask_ms_plus_signal_overhead_ms"
        ),
        "model_definition": "vlm_input_adapt_ms_plus_vlm_generate_ms",
        "inference_definition": "signal_ms_plus_model_ms",
        "e2e_definition": "preprocessing_ms_plus_inference_ms",
        "e2e_paid_definition": (
            "preprocessing_ms_plus_inference_ms_with_paid_hevc_encode_ms"
        ),
        "e2e_wall_definition": "physical_batch_wall_clock_ms",
        "responses_replayed": replayed,
        "replayed_without_timing": replayed and not any(timing.values()),
    }
    return save_benchmark_result(
        owner.results_path,
        timing=timing,
        memory=memory,
        params=params,
        request_measurements=[
            record.get("measurements", {})
            for record in owner.response_store.records
        ],
        batch_measurements=batch_measurements,
    )


def _media_payload(messages):
    images, videos, audios = messages.extract_media()
    return {
        "images": images,
        "videos": videos,
        "audios": audios,
    }


def run_generate_until(owner, requests):
    """Execute lmms-eval generation for a registered pruning-enabled VLM."""

    request_details = [request.args for request in requests]
    results_path = owner.results_path
    owner.response_store = ResponseStore(results_path) if results_path else None
    expected_records = _expected_records(owner, request_details)
    cached_prefix = (
        owner.response_store.load_prefix(expected_records)
        if owner.response_store is not None and resume_enabled()
        else []
    )
    if cached_prefix is not None and len(cached_prefix) == len(expected_records):
        timing, memory, batch_measurements = _restore_recorded_state(
            owner, owner.response_store.records
        )
        if not results_path.is_file():
            owner._ensure_runtime()
            _save_results(
                owner,
                request_details,
                timing,
                memory,
                batch_measurements,
                replayed=True,
            )
        for response, details in zip(cached_prefix, request_details):
            context, _, gen_kwargs, _, _, _ = details
            owner.cache_hook.add_partial(
                "generate_until",
                (context, gen_kwargs),
                response,
            )
        print(
            "Loaded completed responses for lmms-eval postprocessing: "
            f"{owner.response_store.path}"
        )
        return cached_prefix

    owner._ensure_runtime()
    if owner.response_store is not None:
        if cached_prefix is None:
            owner.response_store.begin()
            cached_prefix = []
        else:
            owner.response_store.begin(preserve=True)
    prefix_records = owner.response_store.records if owner.response_store else []
    timing, memory, batch_measurements = (
        _restore_recorded_state(owner, prefix_records)
        if prefix_records
        else (*new_generation_measurements(), [])
    )
    request_measurements = [
        record.get("measurements", {})
        for record in prefix_records
    ]
    results = list(cached_prefix)
    for response, details in zip(results, request_details):
        context, _, gen_kwargs, _, _, _ = details
        owner.cache_hook.add_partial(
            "generate_until",
            (context, gen_kwargs),
            response,
        )

    reporter = ProgressReporter(
        len(request_details),
        every=20,
        label="requests",
        enabled=owner.rank == 0,
    )
    for batch in batch_measurements:
        reporter.update(batch, count=max(1, int(batch.get("sample_count", 1))))

    for first in range(len(results), len(request_details), owner.batch_size):
        batch_indices = list(
            range(first, min(first + owner.batch_size, len(request_details)))
        )
        batch_details = [request_details[index] for index in batch_indices]
        batch_start = time.perf_counter()
        prepared_samples = []
        request_generation_kwargs = []
        stop_sequences = []
        for details in batch_details:
            context, doc_to_messages, gen_kwargs, doc_id, task, split = details
            doc = owner.task_dict[task][split][doc_id]
            messages = ChatMessages(messages=doc_to_messages(doc))
            prepared_samples.append(owner.backend.prepare_sample(
                apply_answer_format_suffix(messages.to_hf_messages()),
                _media_payload(messages),
                owner.num_frames,
            ))
            kwargs, stops = owner.backend.prepare_generation_kwargs(gen_kwargs)
            request_generation_kwargs.append(kwargs)
            stop_sequences.append(stops)

        cuda_sync(owner.device)
        preprocessing_ms = (time.perf_counter() - batch_start) * 1000.0
        preparation = None
        budgets = [(None, None, None)] * len(batch_indices)
        variant = "pruned" if owner.run_pruned else "full"

        signal_prepare_wall_ms = None
        if owner.run_pruned:
            # Whole-stage wall clock; the remainder lands in signal_overhead_ms.
            signal_start = time.perf_counter()
            preparation = owner.pruning.prepare_batch(
                [sample.video_path for sample in prepared_samples],
                [sample.frame_indices for sample in prepared_samples],
                [sample.sampled_frames for sample in prepared_samples],
            )
            cuda_sync(owner.device)
            signal_prepare_wall_ms = (time.perf_counter() - signal_start) * 1000.0
            budgets = owner.pruning.current_keep_budgets(len(batch_indices))
            owner.pruning.enable()
            for effective_keep_total, i_frame_count, p_keep_total in budgets:
                owner.effective_keep_totals.append(effective_keep_total)
                owner.i_frame_counts.append(i_frame_count)
                owner.p_keep_totals.append(p_keep_total)
        else:
            owner.pruning.disable()
            owner.pruning.context.reset_timing()

        # One generate call per batch: the first request's arguments and budget.
        generation_kwargs = request_generation_kwargs[0]
        effective_keep_total = budgets[0][0]
        input_start = time.perf_counter()
        inputs = owner.backend.prepare_batch_inputs(prepared_samples)
        cuda_sync(owner.device)
        preprocessing_ms += (time.perf_counter() - input_start) * 1000.0

        vlm_input_adapt_ms = 0.0
        if owner.run_pruned:
            cuda_sync(owner.device)
            adapt_start = time.perf_counter()
            inputs = owner.pruning.prepare_inputs(
                owner.backend,
                inputs,
                effective_keep_total,
            )
            cuda_sync(owner.device)
            vlm_input_adapt_ms = (time.perf_counter() - adapt_start) * 1000.0

        pre_generate_ms = (time.perf_counter() - batch_start) * 1000.0
        (
            responses,
            inference,
            fine,
            request_fine,
            response_ready,
        ) = _generate_batch(
            owner,
            inputs,
            generation_kwargs,
            stop_sequences,
            batch_details,
        )
        for item in request_fine:
            model_ttft_ms = item.get("model_ttft_ms")
            item["ttft_ms"] = (
                pre_generate_ms + model_ttft_ms
                if model_ttft_ms is not None
                else None
            )
        raw_responses = owner.backend.last_untrimmed_texts()
        token_reduce_ms = owner.pruning.token_reduce_ms() if owner.run_pruned else 0.0
        e2e_wall_ms = (response_ready - batch_start) * 1000.0
        batch_id = first // owner.batch_size
        batch_measurement = build_batch_measurement(
            variant=variant,
            preprocessing_ms=preprocessing_ms,
            inference=inference,
            fine=fine,
            preparation=preparation,
            vlm_input_adapt_ms=vlm_input_adapt_ms,
            token_reduce_ms=token_reduce_ms,
            signal_prepare_wall_ms=signal_prepare_wall_ms,
            e2e_wall_ms=e2e_wall_ms,
            sample_count=len(batch_indices),
        )
        batch_measurement.update(
            {
                "batch_id": batch_id,
                "request_indices": batch_indices,
                "batch_size": len(batch_indices),
            }
        )
        for key in ("model_ttft_ms", "ttft_ms", "tpot_ms"):
            values = [item[key] for item in request_fine if item[key] is not None]
            batch_measurement[key] = sum(values) / len(values) if values else None
        batch_output_tokens = sum(int(item["output_tokens"]) for item in request_fine)
        batch_measurement["output_tokens"] = batch_output_tokens / len(batch_indices)
        batch_measurement["batch_output_tokens"] = batch_output_tokens
        batch_measurements.append(batch_measurement)
        record_measurement(timing, memory, batch_measurement)

        response_records = []
        for local_index, (request_index, details, response) in enumerate(
            zip(batch_indices, batch_details, responses)
        ):
            context, _, gen_kwargs, doc_id, task, split = details
            effective_keep_total, i_frame_count, p_keep_total = budgets[local_index]
            request_measurement = build_request_measurement(
                variant=variant,
                preprocessing_ms=preprocessing_ms,
                inference=inference,
                fine={**fine, **request_fine[local_index]},
                preparation=preparation if len(batch_indices) == 1 else None,
                vlm_input_adapt_ms=vlm_input_adapt_ms,
                token_reduce_ms=token_reduce_ms,
                signal_prepare_wall_ms=(
                    signal_prepare_wall_ms if len(batch_indices) == 1 else None
                ),
                e2e_wall_ms=e2e_wall_ms,
                batch_scoped=len(batch_indices) > 1,
            )
            if (
                owner.run_pruned
                and preparation is not None
                and local_index < len(preparation.hevc_encode_ms_per_sample)
            ):
                request_measurement.update(
                    {
                        "hevc_encode_ms": preparation.hevc_encode_ms_per_sample[
                            local_index
                        ],
                        "hevc_encode_paid_ms": (
                            preparation.hevc_encode_paid_ms_per_sample[local_index]
                        ),
                        "hevc_cache_hit": preparation.hevc_cache_hits[local_index],
                    }
                )
            request_measurement.update(
                {
                    "request_index": request_index,
                    "doc_id": doc_id,
                    "task": task,
                    "split": split,
                    "batch_id": batch_id,
                    "batch_size": len(batch_indices),
                }
            )
            results.append(response)
            request_measurements.append(request_measurement)
            record_measurement(timing, memory, request_measurement)
            if owner.response_store is not None:
                response_records.append(
                    {
                        **expected_records[request_index],
                        "response": response,
                        "raw_response": (
                            raw_responses[local_index]
                            if local_index < len(raw_responses)
                            else None
                        ),
                        "effective_keep_total": effective_keep_total,
                        "i_frame_count": i_frame_count,
                        "p_keep_total": p_keep_total,
                        "measurements": request_measurement,
                        "batch_measurement": batch_measurement,
                    }
                )
            owner.cache_hook.add_partial(
                "generate_until",
                (context, gen_kwargs),
                response,
            )
        if owner.response_store is not None:
            owner.response_store.append_many(response_records)
        reporter.update(batch_measurement, count=len(batch_indices))
    reporter.close()

    saved_path = (
        _save_results(
            owner,
            request_details,
            timing,
            memory,
            batch_measurements,
        )
        if results_path is not None
        else None
    )
    print("----------------------Final results----------------------")
    print(f"Processed={len(results)}")
    print_generation_summary(
        timing,
        memory,
        batch_measurements=batch_measurements,
        request_measurements=request_measurements,
    )
    if saved_path is not None:
        print(f"Saved inference tensor to: {saved_path}")
    if owner.response_store is not None:
        print(f"Saved incremental responses to: {owner.response_store.path}")
    return results

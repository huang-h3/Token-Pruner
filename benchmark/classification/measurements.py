"""Classification accuracy and batch-scoped benchmark measurements."""

from dataclasses import dataclass, field
from pathlib import Path

import torch

from token_pruner.records import save_benchmark_result
from token_pruner.hevc import pick_encoder


TIMING_FIELDS = (
    # Stage totals: e2e = preprocessing + inference; inference = signal + model.
    "preprocessing_ms",
    "signal_ms",
    "model_ms",
    "inference_ms",
    "e2e_ms",
    # Signal breakdown (all three sum to signal_ms).
    "hevc_encode_ms",
    "signal_to_mask_ms",
    "signal_overhead_ms",
    # Model breakdown (token_reduce_ms is nested in the forward).
    "vision_classification_forward_ms",
    "token_reduce_ms",
    # Encode time actually paid on this run; zero on a cache hit.
    "hevc_encode_paid_ms",
    "signal_paid_ms",
    "inference_paid_ms",
    "e2e_paid_ms",
)
MEMORY_FIELDS = ("peak_mem_mb", "delta_peak_mem_mb")
VARIANT_CODES = {"full": 0, "pruned": 1}


def topk_correct(logits, labels):
    """Per-sample top-1/top-5 correctness for one batch."""

    prediction = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
    correctness = prediction.eq(labels.view(-1, 1))
    return (
        correctness[:, :1].any(dim=1).to(torch.bool).cpu(),
        correctness.any(dim=1).to(torch.bool).cpu(),
    )


@dataclass
class ClassificationMeasurements:
    """One batch-scoped record per executed model variant, plus accuracy counts."""

    records: list[dict] = field(default_factory=list)
    request_records: list[dict] = field(default_factory=list)
    total: int = 0
    pruned_top1: int = 0
    pruned_top5: int = 0
    full_top1: int = 0
    full_top5: int = 0
    # Per-sample correctness in evaluation order, one entry per scored sample.
    per_sample: dict[str, dict[str, list[bool]]] = field(
        default_factory=lambda: {
            variant: {"top1": [], "top5": []} for variant in ("pruned", "full")
        }
    )
    hevc_skipped_samples: int = 0
    decode_skipped_samples: int = 0

    def record_batch(
        self,
        *,
        variant,
        batch_index,
        sample_count,
        preprocessing_ms,
        hevc_encode_ms=0.0,
        hevc_encode_paid_ms=None,
        signal_to_mask_ms=0.0,
        signal_prepare_wall_ms=None,
        token_reduce_ms=0.0,
        vision_classification_forward_ms,
        peak_mem_mb,
        delta_peak_mem_mb,
    ):
        if variant not in VARIANT_CODES:
            raise RuntimeError(f"Unknown classification variant {variant!r}.")
        if hevc_encode_paid_ms is None:
            hevc_encode_paid_ms = hevc_encode_ms
        hevc_encode_ms = float(hevc_encode_ms)
        hevc_encode_paid_ms = float(hevc_encode_paid_ms)
        signal_to_mask_ms = float(signal_to_mask_ms)
        # The signal stage's remainder still belongs to it, so e2e stays additive.
        signal_overhead_ms = (
            max(
                0.0,
                float(signal_prepare_wall_ms)
                - hevc_encode_paid_ms
                - signal_to_mask_ms,
            )
            if signal_prepare_wall_ms is not None
            else 0.0
        )
        signal_ms = hevc_encode_ms + signal_to_mask_ms + signal_overhead_ms
        signal_paid_ms = (
            hevc_encode_paid_ms + signal_to_mask_ms + signal_overhead_ms
        )
        # token_reduce_ms is nested in the complete model forward.
        model_ms = float(vision_classification_forward_ms)
        record = {
            "variant": variant,
            "measurement_scope": "batch",
            "batch_index": int(batch_index),
            "sample_count": int(sample_count),
            "preprocessing_ms": float(preprocessing_ms),
            "hevc_encode_ms": hevc_encode_ms,
            "hevc_encode_paid_ms": hevc_encode_paid_ms,
            "signal_to_mask_ms": signal_to_mask_ms,
            "signal_overhead_ms": signal_overhead_ms,
            "signal_ms": signal_ms,
            "signal_paid_ms": signal_paid_ms,
            "token_reduce_ms": float(token_reduce_ms),
            "vision_classification_forward_ms": model_ms,
            "model_ms": model_ms,
            "inference_ms": signal_ms + model_ms,
            "inference_paid_ms": signal_paid_ms + model_ms,
            "peak_mem_mb": float(peak_mem_mb),
            "delta_peak_mem_mb": float(delta_peak_mem_mb),
        }
        record["e2e_ms"] = record["preprocessing_ms"] + record["inference_ms"]
        record["e2e_paid_ms"] = (
            record["preprocessing_ms"] + record["inference_paid_ms"]
        )
        self.records.append(record)
        return record

    def records_for(self, variant):
        return [record for record in self.records if record["variant"] == variant]

    def record_request(self, record):
        """Append a typed request record; unavailable batched metrics stay None."""

        self.request_records.append(dict(record))
        return self.request_records[-1]

    def record_accuracy(self, variant, logits, labels):
        top1, top5 = topk_correct(logits, labels)
        if variant not in self.per_sample:
            raise RuntimeError(f"Unknown classification variant {variant!r}.")
        if variant == "pruned":
            self.pruned_top1 += int(top1.sum())
            self.pruned_top5 += int(top5.sum())
        else:
            self.full_top1 += int(top1.sum())
            self.full_top5 += int(top5.sum())
        self.per_sample[variant]["top1"].extend(bool(value) for value in top1)
        self.per_sample[variant]["top5"].extend(bool(value) for value in top5)

    def per_sample_correct(self, session):
        """Correctness vectors per variant, aligned with evaluation order."""

        variants = []
        if session.run_pruned:
            variants.append("pruned")
        if session.run_full:
            variants.append("full")
        return {
            variant: {
                key: torch.tensor(self.per_sample[variant][key], dtype=torch.bool)
                for key in ("top1", "top5")
            }
            for variant in variants
        }

    def timing(self):
        """Numeric view; batch_measurements is authoritative."""

        return {
            "variant_code": [VARIANT_CODES[record["variant"]] for record in self.records],
            "sample_count": [record["sample_count"] for record in self.records],
            **{
                name: [record[name] for record in self.records]
                for name in TIMING_FIELDS
            },
        }

    def memory(self):
        return {
            name: [record[name] for record in self.records]
            for name in MEMORY_FIELDS
        }

    def accuracy(self, session):
        values = {}
        if session.run_pruned:
            values["pruned"] = [
                self.pruned_top1 / self.total,
                self.pruned_top5 / self.total,
            ]
        if session.run_full:
            values["full"] = [
                self.full_top1 / self.total,
                self.full_top5 / self.total,
            ]
        return values

    def build_params(
        self,
        args,
        session,
        dataset,
        dataset_filter_stats,
        skipped_videos_path,
        timing_schema,
    ):
        params = dict(session.params)
        params.update(
            {
                "model_type": session.model_type,
                "num_samples": self.total,
                "requested_samples": int(dataset_filter_stats["total"]),
                "skipped_samples": (
                    int(dataset_filter_stats["skipped"])
                    + self.hevc_skipped_samples
                    + self.decode_skipped_samples
                ),
                "valid_dataset_samples": len(dataset),
                "skipped_videos_file": (
                    skipped_videos_path.name
                    if skipped_videos_path is not None
                    else None
                ),
                "hevc_skipped_samples": self.hevc_skipped_samples,
                "decode_skipped_samples": self.decode_skipped_samples,
                "dataset_filter": dataset_filter_stats,
                "batch_size": args.batch_size,
                "measurement_scope": "batch",
                "variant_codes": dict(VARIANT_CODES),
                "num_frames": session.num_frames,
                "hevc_encoder": pick_encoder() if session.needs_hevc else None,
                "hevc_effective_gop_sizes_per_sample": (
                    list(session.hevc_effective_gop_sizes)
                    if session.needs_hevc
                    else []
                ),
                "device": str(session.device),
                "model_name": session.model_name,
                "dataset": dataset.name,
                "dataset_split": dataset.split,
                "run_mode": args.run_mode,
                "timing_schema": timing_schema,
                "signal_definition": (
                    "hevc_encode_ms_plus_signal_to_mask_ms_plus_signal_overhead_ms"
                ),
                "model_definition": "vision_classification_forward_ms",
                "inference_definition": "signal_ms_plus_model_ms",
                "e2e_definition": "preprocessing_ms_plus_inference_ms",
                "e2e_paid_definition": (
                    "preprocessing_ms_plus_inference_ms_with_paid_hevc_encode_ms"
                ),
            }
        )
        return params

    def save(
        self,
        args,
        session,
        dataset,
        dataset_filter_stats,
        skipped_videos_path,
        timing_schema,
        profiler_flops=None,
    ):
        accuracy = self.accuracy(session)
        extra = {}
        if session.run_pruned:
            extra[session.accuracy_key] = torch.tensor(
                accuracy["pruned"],
                dtype=torch.float32,
            )
        if session.run_full:
            extra["full"] = torch.tensor(accuracy["full"], dtype=torch.float32)
        # Per-sample correctness supports paired comparisons between runs.
        extra["per_sample_correct"] = self.per_sample_correct(session)
        return save_benchmark_result(
            Path(args.results_path),
            timing=self.timing(),
            memory=self.memory(),
            params=self.build_params(
                args,
                session,
                dataset,
                dataset_filter_stats,
                skipped_videos_path,
                timing_schema,
            ),
            accuracy=accuracy,
            flops=profiler_flops,
            request_measurements=self.request_records,
            batch_measurements=self.records,
            extra=extra,
        )

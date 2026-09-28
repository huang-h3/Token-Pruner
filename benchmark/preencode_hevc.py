"""Populate the inference HEVC store from lmms-eval tasks, without models."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

from . import bootstrap  # noqa: F401
from .paths import HEVC_STORE_DIR
from token_pruner.hevc import resolve_hevc_encode_scope
from token_pruner.hevc_cache import HevcArtifactStore
from token_pruner.records import atomic_write_text


def task_names(values):
    return list(dict.fromkeys(
        name for value in values for name in value.replace(",", " ").split()
    ))


def collect_videos(tasks, limit):
    """Use the same leaf tasks, document limit and chat media as inference."""
    from lmms_eval.evaluator_utils import get_sample_size
    from lmms_eval.protocol import ChatMessages

    videos = {}
    counts = {}
    for output in tasks:
        name, task = output.task_name, output.task
        if name in counts:
            continue
        count = 0
        for doc_id, doc in task.doc_iterator(limit=get_sample_size(task, limit)):
            messages = ChatMessages(messages=task.doc_to_messages(doc))
            _, media, _ = messages.extract_media()
            if not media:
                raise ValueError(f"{name}/{doc_id}: no video in task messages")
            for raw in media:
                if not isinstance(raw, (str, Path)):
                    raise TypeError(
                        f"{name}/{doc_id}: expected a local video path, "
                        f"got {type(raw).__name__}"
                    )
                path = Path(raw).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"{name}/{doc_id}: video not found: {path}")
                videos.setdefault(path, []).append({"task": name, "doc_id": int(doc_id)})
            count += 1
        counts[name] = count
    return videos, counts


def encode_videos(videos, store, gop_size):
    from tqdm import tqdm

    results = []
    for path in tqdm(videos, desc="HEVC preencode"):
        row = {"source_path": str(path)}
        try:
            artifact = store.get_or_encode(
                path, encode_scope="full-video", sampled_gop_size=gop_size,
            )
            if artifact is None:
                raise RuntimeError("encoder did not produce a valid artifact")
            row.update(
                status="hit" if artifact.cache_hit else "encoded",
                artifact_key=artifact.cache_key,
                path=str(artifact.path),
                actual_encoder=artifact.actual_encoder,
                paid_encode_ms=artifact.paid_encode_ms,
            )
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"{path}: {row['error']}", file=sys.stderr, flush=True)
        results.append(row)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=[os.getenv("TASKS", "")])
    parser.add_argument(
        "--limit", type=float, default=os.getenv("LIMIT") or None,
        help="Same per-leaf-task limit as lmms-eval; omitted means all documents.",
    )
    parser.add_argument(
        "--permanent-dir", type=Path,
        default=Path(os.getenv("HEVC_PERMANENT_DIR", str(HEVC_STORE_DIR))),
    )
    parser.add_argument(
        "--gop-size", type=int, default=int(os.getenv("HEVC_SAMPLED_GOP_SIZE", "32")),
    )
    parser.add_argument(
        "--encode-scope", choices=["full-video"],
        default=os.getenv("HEVC_ENCODE_SCOPE", "full-video"),
    )
    parser.add_argument("--report", type=Path, default=os.getenv("HEVC_PREENCODE_REPORT"))
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Enumerate and validate source videos; do not encode.",
    )
    args = parser.parse_args(argv)
    names = task_names(args.tasks)
    if not names:
        parser.error("set TASKS or pass --tasks")
    if args.limit is not None and (
        not math.isfinite(args.limit) or (args.limit <= 0 and args.limit != -1)
    ):
        parser.error("--limit must be positive or -1 (all documents)")
    if args.gop_size < 0:
        parser.error("--gop-size must be non-negative")
    if resolve_hevc_encode_scope(args.encode_scope) != "full-video":
        parser.error("preencoding requires full-video with VIDEO_ABLATION=none")
    if os.getenv("LMMS_EVAL_ROOT"):
        sys.path.insert(0, str(Path(os.environ["LMMS_EVAL_ROOT"]).expanduser()))
    permanent = args.permanent_dir.expanduser().resolve()
    report_path = (args.report or permanent / "preencode_summary.json").expanduser()
    report = dict(
        tasks=names, limit=args.limit, encode_scope="full-video",
        gop_size=args.gop_size, permanent_dir=str(permanent), dry_run=args.dry_run,
    )
    try:
        from lmms_eval.tasks import TaskManager, get_task_dict
        from lmms_eval.evaluator_utils import get_task_list

        tasks = get_task_list(
            get_task_dict(names, task_manager=TaskManager(), task_type="chat")
        )
        videos, counts = collect_videos(tasks, args.limit)
        if not videos:
            raise ValueError("no videos selected; refusing to report successful preparation")
        report.update(documents_per_task=counts, unique_videos=len(videos))
        print(f"{sum(counts.values())} documents, {len(videos)} unique video paths", flush=True)
        report["videos"] = (
            [dict(source_path=str(path), status="planned") for path in videos]
            if args.dry_run
            else encode_videos(videos, HevcArtifactStore(permanent), args.gop_size)
        )
        totals = {
            status: sum(row["status"] == status for row in report["videos"])
            for status in ("hit", "encoded", "failed", "planned")
        }
        report.update(
            totals=totals,
            status=(
                "failed" if totals["failed"]
                else "planned" if args.dry_run
                else "complete"
            ),
        )
    except (Exception, SystemExit) as exc:
        cause = exc.last_attempt.exception() if hasattr(exc, "last_attempt") else exc
        report.update(status="failed", error=f"{type(cause).__name__}: {cause}")
        print(report["error"], file=sys.stderr)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(report_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"{report['status']}: {report.get('totals', {})}; report: {report_path}", flush=True)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())

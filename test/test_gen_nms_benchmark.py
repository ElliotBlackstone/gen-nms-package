# to run, use
# python ./test/test_gen_nms_benchmark.py --n 500 --repeats 10 --csv diou_benchmark_500.csv
# change value for n, repeats as desired

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torchvision

import gen_nms

import pandas as pd



Tensor = torch.Tensor


@dataclass
class BenchmarkResult:
    device: str
    mode: str
    implementation: str
    avg_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    repeats: int
    num_boxes: int
    num_kept: int
    exact_match: Optional[bool]
    tolerant_match: Optional[bool]
    match_note: str


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def old_diou_nms(
    boxes: Tensor,
    scores: Tensor,
    diou_threshold: float,
    class_agnostic: bool = True,
    idxs: Optional[Tensor] = None,
) -> Tensor:
    """
    Baseline DIoU-NMS implemented in Python using torchvision.ops.distance_box_iou.

    Note: class-agnostic=False is undefined without class labels, so this function
    accepts `idxs` as an additional optional argument.
    """
    if boxes.ndim != 2 or boxes.size(-1) != 4:
        raise ValueError(f"boxes must have shape [N, 4], got {tuple(boxes.shape)}")
    if scores.ndim != 1:
        raise ValueError(f"scores must have shape [N], got {tuple(scores.shape)}")
    if boxes.size(0) != scores.size(0):
        raise ValueError("boxes and scores must have the same length")
    if not class_agnostic:
        if idxs is None:
            raise ValueError("idxs must be provided when class_agnostic=False")
        if idxs.ndim != 1 or idxs.size(0) != boxes.size(0):
            raise ValueError("idxs must have shape [N]")

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    if class_agnostic:
        return _old_diou_nms_single_class(boxes, scores, diou_threshold)

    keep_mask = torch.zeros(scores.shape[0], dtype=torch.bool, device=boxes.device)
    unique_classes = torch.unique(idxs)
    for class_id in unique_classes:
        curr_indices = torch.where(idxs == class_id)[0]
        curr_keep = _old_diou_nms_single_class(
            boxes[curr_indices], scores[curr_indices], diou_threshold
        )
        keep_mask[curr_indices[curr_keep]] = True

    keep_indices = torch.where(keep_mask)[0]
    sorted_keep = keep_indices[scores[keep_indices].argsort(descending=True)]
    return sorted_keep


@torch.no_grad()
def _old_diou_nms_single_class(boxes: Tensor, scores: Tensor, diou_threshold: float) -> Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep: list[int] = []

    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        current_box = boxes[i].unsqueeze(0)
        rest_boxes = boxes[rest]
        diou_vals = torchvision.ops.distance_box_iou(current_box, rest_boxes).squeeze(0)
        order = rest[diou_vals <= diou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


@torch.no_grad()
def run_new_diou_nms(boxes: Tensor, scores: Tensor, diou_threshold: float, **_: object) -> Tensor:
    return gen_nms.diou_nms(boxes, scores, diou_threshold)


@torch.no_grad()
def run_new_batched_diou_nms(
    boxes: Tensor,
    scores: Tensor,
    diou_threshold: float,
    idxs: Tensor,
    **_: object,
) -> Tensor:
    return gen_nms.batched_diou_nms(boxes, scores, idxs, diou_threshold)


@torch.no_grad()
def generate_test_data(
    n: int,
    num_classes: int,
    seed: int,
    scale: float = 1024.0,
    num_clusters: int = 24,
) -> tuple[Tensor, Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)

    cluster_centers = 0.15 + 0.7 * torch.rand((num_clusters, 2), generator=g)
    assignments = torch.randint(0, num_clusters, (n,), generator=g)
    centers = cluster_centers[assignments] + 0.05 * torch.randn((n, 2), generator=g)
    centers = centers.clamp(0.02, 0.98)

    sizes = 0.05 + 0.20 * torch.rand((n, 2), generator=g)
    xy1 = (centers - sizes / 2).clamp(0.0, 0.99)
    xy2 = (centers + sizes / 2).clamp(0.01, 1.0)
    xy2 = torch.maximum(xy2, xy1 + 1e-3)
    boxes = torch.cat([xy1, xy2], dim=1) * scale

    # Exactly unique scores to avoid CPU/GPU tie-breaking ambiguity as much as possible.
    scores = torch.randperm(n, generator=g, dtype=torch.int64).to(torch.float32) / float(max(n, 1))
    idxs = torch.randint(0, num_classes, (n,), generator=g, dtype=torch.int64)
    return boxes.contiguous(), scores.contiguous(), idxs.contiguous()


@torch.no_grad()
def compare_keeps(
    boxes: Tensor,
    scores: Tensor,
    keep_a: Tensor,
    keep_b: Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> tuple[bool, bool, str]:
    ka = keep_a.detach().cpu().to(torch.long)
    kb = keep_b.detach().cpu().to(torch.long)

    if torch.equal(ka, kb):
        return True, True, "exact index match"

    if ka.numel() != kb.numel():
        return False, False, f"different number kept: {ka.numel()} vs {kb.numel()}"

    boxes_cpu = boxes.detach().cpu().to(torch.float32)
    scores_cpu = scores.detach().cpu().to(torch.float32)

    ba = boxes_cpu[ka]
    bb = boxes_cpu[kb]
    sa = scores_cpu[ka]
    sb = scores_cpu[kb]

    order_a = sa.argsort(descending=True)
    order_b = sb.argsort(descending=True)
    sa = sa[order_a]
    sb = sb[order_b]
    ba = ba[order_a]
    bb = bb[order_b]

    scores_close = torch.allclose(sa, sb, atol=atol, rtol=rtol)
    boxes_close = torch.allclose(ba, bb, atol=atol, rtol=rtol)
    if scores_close and boxes_close:
        return False, True, "kept boxes/scores match within tolerance"

    return False, False, "kept indices and kept boxes differ"


@torch.no_grad()
def time_one(
    fn: Callable[..., Tensor],
    *,
    boxes: Tensor,
    scores: Tensor,
    diou_threshold: float,
    idxs: Optional[Tensor],
    class_agnostic: bool,
    warmup: int,
    repeats: int,
) -> tuple[Tensor, list[float]]:
    device = boxes.device

    last_out = None
    for _ in range(warmup):
        last_out = fn(
            boxes=boxes,
            scores=scores,
            diou_threshold=diou_threshold,
            idxs=idxs,
            class_agnostic=class_agnostic,
        )
        synchronize_if_needed(device)

    times_ms: list[float] = []
    for _ in range(repeats):
        synchronize_if_needed(device)
        t0 = time.perf_counter()
        last_out = fn(
            boxes=boxes,
            scores=scores,
            diou_threshold=diou_threshold,
            idxs=idxs,
            class_agnostic=class_agnostic,
        )
        synchronize_if_needed(device)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    assert last_out is not None
    return last_out, times_ms


def summarize_times(times_ms: list[float]) -> tuple[float, float, float, float]:
    return (
        statistics.mean(times_ms),
        statistics.median(times_ms),
        min(times_ms),
        max(times_ms),
    )


@torch.no_grad()
def benchmark_mode(
    *,
    device: torch.device,
    mode: str,
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    diou_threshold: float,
    warmup: int,
    repeats: int,
) -> list[BenchmarkResult]:
    boxes_d = boxes.to(device)
    scores_d = scores.to(device)
    idxs_d = idxs.to(device)

    if mode == "class_agnostic":
        old_fn = old_diou_nms
        new_fn = run_new_diou_nms
        class_agnostic = True
        old_name = "old_diou_nms"
        new_name = "gen_nms.diou_nms"
    elif mode == "per_class":
        old_fn = old_diou_nms
        new_fn = run_new_batched_diou_nms
        class_agnostic = False
        old_name = "old_diou_nms(per-class)"
        new_name = "gen_nms.batched_diou_nms"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    old_keep, old_times = time_one(
        old_fn,
        boxes=boxes_d,
        scores=scores_d,
        diou_threshold=diou_threshold,
        idxs=idxs_d,
        class_agnostic=class_agnostic,
        warmup=warmup,
        repeats=repeats,
    )
    new_keep, new_times = time_one(
        new_fn,
        boxes=boxes_d,
        scores=scores_d,
        diou_threshold=diou_threshold,
        idxs=idxs_d,
        class_agnostic=class_agnostic,
        warmup=warmup,
        repeats=repeats,
    )

    exact, tolerant, note = compare_keeps(boxes_d, scores_d, old_keep, new_keep)

    old_avg, old_med, old_min, old_max = summarize_times(old_times)
    new_avg, new_med, new_min, new_max = summarize_times(new_times)

    return [
        BenchmarkResult(
            device=device.type,
            mode=mode,
            implementation=old_name,
            avg_ms=old_avg,
            median_ms=old_med,
            min_ms=old_min,
            max_ms=old_max,
            repeats=repeats,
            num_boxes=int(boxes_d.size(0)),
            num_kept=int(old_keep.numel()),
            exact_match=None,
            tolerant_match=None,
            match_note="baseline",
        ),
        BenchmarkResult(
            device=device.type,
            mode=mode,
            implementation=new_name,
            avg_ms=new_avg,
            median_ms=new_med,
            min_ms=new_min,
            max_ms=new_max,
            repeats=repeats,
            num_boxes=int(boxes_d.size(0)),
            num_kept=int(new_keep.numel()),
            exact_match=exact,
            tolerant_match=tolerant,
            match_note=note,
        ),
    ]


@torch.no_grad()
def gpu_vs_cpu_check(
    *,
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    diou_threshold: float,
) -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        return []

    cpu_diou = gen_nms.diou_nms(boxes, scores, diou_threshold)
    gpu_diou = gen_nms.diou_nms(boxes.cuda(), scores.cuda(), diou_threshold)
    cpu_exact, cpu_tol, cpu_note = compare_keeps(boxes, scores, cpu_diou, gpu_diou.cpu())

    cpu_batch = gen_nms.batched_diou_nms(boxes, scores, idxs, diou_threshold)
    gpu_batch = gen_nms.batched_diou_nms(boxes.cuda(), scores.cuda(), idxs.cuda(), diou_threshold)
    bat_exact, bat_tol, bat_note = compare_keeps(boxes, scores, cpu_batch, gpu_batch.cpu())

    return [
        {
            "device": "cpu_vs_gpu",
            "mode": "class_agnostic",
            "implementation": "gen_nms.diou_nms",
            "avg_ms": math.nan,
            "median_ms": math.nan,
            "min_ms": math.nan,
            "max_ms": math.nan,
            "repeats": 0,
            "num_boxes": int(boxes.size(0)),
            "num_kept": int(cpu_diou.numel()),
            "exact_match": cpu_exact,
            "tolerant_match": cpu_tol,
            "match_note": cpu_note,
        },
        {
            "device": "cpu_vs_gpu",
            "mode": "per_class",
            "implementation": "gen_nms.batched_diou_nms",
            "avg_ms": math.nan,
            "median_ms": math.nan,
            "min_ms": math.nan,
            "max_ms": math.nan,
            "repeats": 0,
            "num_boxes": int(boxes.size(0)),
            "num_kept": int(cpu_batch.numel()),
            "exact_match": bat_exact,
            "tolerant_match": bat_tol,
            "match_note": bat_note,
        },
    ]


def build_table(rows: list[BenchmarkResult | dict[str, object]]) -> object:
    normalized = [r.__dict__ if isinstance(r, BenchmarkResult) else r for r in rows]
    for row in normalized:
        if row["implementation"].startswith("gen_nms") and row["device"] in {"cpu", "cuda"}:
            baseline_name = "old_diou_nms" if row["mode"] == "class_agnostic" else "old_diou_nms(per-class)"
            baseline = next(
                r for r in normalized
                if r["device"] == row["device"] and r["mode"] == row["mode"] and r["implementation"] == baseline_name
            )
            row["speedup_vs_old_x"] = baseline["avg_ms"] / row["avg_ms"] if row["avg_ms"] > 0 else math.nan
        else:
            row["speedup_vs_old_x"] = math.nan

    if pd is not None:
        df = pd.DataFrame(normalized)
        ordered_cols = [
            "device",
            "mode",
            "implementation",
            "num_boxes",
            "num_kept",
            "repeats",
            "avg_ms",
            "median_ms",
            "min_ms",
            "max_ms",
            "speedup_vs_old_x",
            "exact_match",
            "tolerant_match",
            "match_note",
        ]
        return df[ordered_cols]
    return normalized


def print_table(table: object) -> None:
    if pd is not None and hasattr(table, "to_string"):
        print(table.to_string(index=False, float_format=lambda x: f"{x:,.3f}" if isinstance(x, float) and math.isfinite(x) else str(x)))
        return

    for row in table:
        print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark gen_nms against a torchvision-based DIoU-NMS baseline.")
    parser.add_argument("--n", type=int, default=1500, help="Number of boxes to generate.")
    parser.add_argument("--num-classes", type=int, default=8, help="Number of classes for per-class NMS.")
    parser.add_argument("--threshold", type=float, default=0.5, help="DIoU suppression threshold.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per benchmark.")
    parser.add_argument("--repeats", type=int, default=10, help="Timed iterations per benchmark.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--csv", type=str, default="", help="Optional path to save the result table as CSV.")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    boxes, scores, idxs = generate_test_data(
        n=args.n,
        num_classes=args.num_classes,
        seed=args.seed,
    )

    rows: list[BenchmarkResult | dict[str, object]] = []

    rows.extend(
        benchmark_mode(
            device=torch.device("cpu"),
            mode="class_agnostic",
            boxes=boxes,
            scores=scores,
            idxs=idxs,
            diou_threshold=args.threshold,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    )
    rows.extend(
        benchmark_mode(
            device=torch.device("cpu"),
            mode="per_class",
            boxes=boxes,
            scores=scores,
            idxs=idxs,
            diou_threshold=args.threshold,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    )

    if torch.cuda.is_available():
        rows.extend(
            benchmark_mode(
                device=torch.device("cuda"),
                mode="class_agnostic",
                boxes=boxes,
                scores=scores,
                idxs=idxs,
                diou_threshold=args.threshold,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )
        rows.extend(
            benchmark_mode(
                device=torch.device("cuda"),
                mode="per_class",
                boxes=boxes,
                scores=scores,
                idxs=idxs,
                diou_threshold=args.threshold,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )
        rows.extend(
            gpu_vs_cpu_check(
                boxes=boxes,
                scores=scores,
                idxs=idxs,
                diou_threshold=args.threshold,
            )
        )
    else:
        print("CUDA not available; GPU benchmarks skipped.")

    table = build_table(rows)
    print_table(table)

    if args.csv:
        if pd is None:
            raise RuntimeError("pandas is required to write CSV output")
        table.to_csv(args.csv, index=False)
        print(f"\nSaved CSV to: {args.csv}")


if __name__ == "__main__":
    main()

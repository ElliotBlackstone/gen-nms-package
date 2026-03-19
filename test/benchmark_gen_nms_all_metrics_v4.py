# to run, use
# python benchmark_gen_nms_all_metrics_v4.py --n 1500 --repeats 10
# or 
# python benchmark_gen_nms_all_metrics_v4.py --n 1500 --repeats 10 --csv-prefix benchmark_results
# to save a .csv file with the results
from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torchvision.ops as tvops
import gen_nms

import pandas as pd


def _pairwise_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    union = area1 + area2 - inter
    return inter / union


def _pairwise_giou_one_to_many(box: torch.Tensor, boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    c_lt = torch.minimum(box[:2], boxes[:, :2])
    c_rb = torch.maximum(box[2:], boxes[:, 2:])
    c_wh = (c_rb - c_lt).clamp(min=0)
    c_area = c_wh[:, 0] * c_wh[:, 1]

    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    union = area1 + area2 - inter

    return inter / union - (c_area - union) / c_area


def _pairwise_diou_one_to_many(box: torch.Tensor, boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    iou = _pairwise_iou_one_to_many(box, boxes, eps=eps)

    center1 = (box[:2] + box[2:]) * 0.5
    center2 = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    rho2 = ((center1 - center2) ** 2).sum(dim=1)

    c_lt = torch.minimum(box[:2], boxes[:, :2])
    c_rb = torch.maximum(box[2:], boxes[:, 2:])
    c_wh = (c_rb - c_lt).clamp(min=0)
    c2 = (c_wh ** 2).sum(dim=1)

    return iou - rho2 / c2


def _pairwise_ciou_one_to_many(box: torch.Tensor, boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    iou = _pairwise_iou_one_to_many(box, boxes, eps=eps)
    center1 = (box[:2] + box[2:]) * 0.5
    center2 = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    rho2 = ((center1 - center2) ** 2).sum(dim=1)

    c_lt = torch.minimum(box[:2], boxes[:, :2])
    c_rb = torch.maximum(box[2:], boxes[:, 2:])
    c_wh = (c_rb - c_lt).clamp(min=0)
    c2 = (c_wh ** 2).sum(dim=1)

    w1 = (box[2] - box[0]).clamp(min=eps)
    h1 = (box[3] - box[1]).clamp(min=eps)
    w2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=eps)
    h2 = (boxes[:, 3] - boxes[:, 1]).clamp(min=eps)

    v = (4.0 / (math.pi ** 2)) * (torch.atan(w1 / h1) - torch.atan(w2 / h2)) ** 2
    alpha = v / (1.0 - iou + v)

    diou = iou - rho2 / c2
    return diou - alpha * v


PAIRWISE_METRICS: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "iou": _pairwise_iou_one_to_many,
    "giou": _pairwise_giou_one_to_many,
    "diou": _pairwise_diou_one_to_many,
    "ciou": _pairwise_ciou_one_to_many,
}


def pure_python_nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float, metric: str) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    metric_fn = PAIRWISE_METRICS[metric]
    order = scores.argsort(descending=True)
    keep: List[torch.Tensor] = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        rest = order[1:]
        vals = metric_fn(boxes[i], boxes[rest])
        order = rest[vals <= threshold]

    return torch.stack(keep)


GEN_NMS_FUNCS: Dict[str, Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = {
    "iou": gen_nms.iou_nms,
    "giou": gen_nms.giou_nms,
    "diou": gen_nms.diou_nms,
    "ciou": gen_nms.ciou_nms,
}



def _is_cuda_call(args: Tuple) -> bool:
    return any(isinstance(a, torch.Tensor) and a.is_cuda for a in args)



def _sync_if_needed(is_cuda: bool) -> None:
    if is_cuda:
        torch.cuda.synchronize()



def benchmark_once(fn: Callable, *args) -> Tuple[torch.Tensor, float]:
    output = None
    is_cuda = _is_cuda_call(args)

    with torch.inference_mode():
        _sync_if_needed(is_cuda)
        if is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = fn(*args)
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            output = fn(*args)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0

    assert output is not None
    return output, elapsed_ms



def median_ms(values: List[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def matches_python_cpu(ref_idx: torch.Tensor, test_idx: torch.Tensor) -> bool:
    ref_idx_cpu = ref_idx.detach().cpu().to(torch.long)
    test_idx_cpu = test_idx.detach().cpu().to(torch.long)

    if ref_idx_cpu.numel() != test_idx_cpu.numel():
        return False

    return torch.equal(
        ref_idx_cpu.sort().values,
        test_idx_cpu.sort().values,
    )


def make_random_boxes(
    n: int,
    image_size: float = 1024.0,
    min_wh: float = 4.0,
    max_wh: float = 128.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    xy1 = torch.rand(n, 2, dtype=dtype) * (image_size - max_wh - 1.0)
    wh = min_wh + torch.rand(n, 2, dtype=dtype) * (max_wh - min_wh)
    xy2 = xy1 + wh
    return torch.cat([xy1, xy2], dim=1)


@dataclass(frozen=True)
class TaskKey:
    metric: str
    impl: str
    device: str


@dataclass
class BenchmarkTask:
    key: TaskKey
    fn: Callable
    args: Tuple


@dataclass
class BenchResult:
    output: torch.Tensor
    time_ms: float


@dataclass
class Row:
    group: str
    metric: str
    impl: str
    device: str
    time_ms: Optional[float]
    speedup_vs_python_cpu: Optional[float]
    speedup_vs_torchvision_same_device: Optional[float]
    matches_python_cpu: Optional[bool]
    num_kept: Optional[int]
    note: str = ""


def rows_to_frame(rows: List[Row]):
    data = [r.__dict__ for r in rows]
    if pd is not None:
        return pd.DataFrame(data)
    return data


def print_frame(frame, title: str) -> None:
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))
    if pd is not None:
        print(frame.to_string(index=False))
    else:
        for row in frame:
            print(row)


def global_gpu_prewarm(boxes_gpu: torch.Tensor, scores_gpu: torch.Tensor, threshold: float) -> None:
    with torch.inference_mode():
        for fn in [gen_nms.iou_nms, gen_nms.giou_nms, gen_nms.diou_nms, gen_nms.ciou_nms, tvops.nms]:
            for _ in range(10):
                fn(boxes_gpu, scores_gpu, threshold)
        torch.cuda.synchronize()


def build_tasks(
    boxes_cpu: torch.Tensor,
    scores_cpu: torch.Tensor,
    boxes_gpu: Optional[torch.Tensor],
    scores_gpu: Optional[torch.Tensor],
    threshold: float,
) -> List[BenchmarkTask]:
    tasks: List[BenchmarkTask] = []

    for metric in ["iou", "giou", "diou", "ciou"]:
        py_fn = lambda b, s, t, m=metric: pure_python_nms(b, s, t, m)
        tasks.append(BenchmarkTask(TaskKey(metric, "pure_python", "cpu"), py_fn, (boxes_cpu, scores_cpu, threshold)))

        gen_fn = GEN_NMS_FUNCS[metric]
        tasks.append(BenchmarkTask(TaskKey(metric, "gen_nms", "cpu"), gen_fn, (boxes_cpu, scores_cpu, threshold)))

        if boxes_gpu is not None and scores_gpu is not None:
            tasks.append(BenchmarkTask(TaskKey(metric, "gen_nms", "gpu"), gen_fn, (boxes_gpu, scores_gpu, threshold)))

    tasks.append(BenchmarkTask(TaskKey("iou", "torchvision_nms", "cpu"), tvops.nms, (boxes_cpu, scores_cpu, threshold)))
    if boxes_gpu is not None and scores_gpu is not None:
        tasks.append(BenchmarkTask(TaskKey("iou", "torchvision_nms", "gpu"), tvops.nms, (boxes_gpu, scores_gpu, threshold)))

    return tasks


def run_tasks(
    tasks: List[BenchmarkTask],
    repeats: int,
    warmup: int,
    seed: int,
    order_mode: str,
    order_display: str,
) -> Dict[TaskKey, BenchResult]:
    rng = random.Random(seed + 1729)
    task_list = list(tasks)

    print("\nBenchmark methodology")
    print("---------------------")
    print("Interleaved warmup/measurement across tasks so no implementation is always first or last.")

    outputs: Dict[TaskKey, torch.Tensor] = {}
    timings: Dict[TaskKey, List[float]] = {task.key: [] for task in task_list}

    for stage_name, count in (("warmup", warmup), ("measure", repeats)):
        for rep in range(count):
            ordered_tasks = list(task_list)
            if order_mode == "randomized":
                rng.shuffle(ordered_tasks)

            if rep == 0 and order_display == "display":
                print(f"\nFirst {stage_name} pass order:")
                for t in ordered_tasks:
                    print(f"  metric={t.key.metric:>4} impl={t.key.impl:<15} device={t.key.device}")

            for task in ordered_tasks:
                out, ms = benchmark_once(task.fn, *task.args)
                outputs[task.key] = out
                if stage_name == "measure":
                    timings[task.key].append(ms)

    return {
        task.key: BenchResult(output=outputs[task.key], time_ms=median_ms(timings[task.key]))
        for task in task_list
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark gen_nms vs pure Python and torchvision")
    parser.add_argument("--n", type=int, default=1500, help="number of boxes")
    parser.add_argument("--threshold", type=float, default=0.5, help="NMS threshold")
    parser.add_argument("--repeats", type=int, default=20, help="benchmark repeats")
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--csv-prefix", type=str, default="", help="optional prefix for writing CSV files")
    parser.add_argument("--order-mode", choices=["randomized", "fixed"], default="randomized",
                        help="order used inside each interleaved warmup/measurement pass")
    parser.add_argument("--order-display", choices=["silent", "display"], default="silent",
                        help="prints the order used if desired")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("Environment")
    print("-----------")
    print("torch.__version__:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("gen_nms has ops:",
          {name: hasattr(torch.ops.gen_nms, f"{name}_nms") for name in ["iou", "giou", "diou", "ciou"]})

    boxes_cpu = make_random_boxes(args.n, dtype=torch.float32)
    scores_cpu = torch.rand(args.n, dtype=torch.float32)
    scores_cpu = scores_cpu + torch.linspace(0.0, 1e-6, args.n, dtype=torch.float32)

    has_cuda = torch.cuda.is_available()
    boxes_gpu = boxes_cpu.cuda() if has_cuda else None
    scores_gpu = scores_cpu.cuda() if has_cuda else None

    if has_cuda:
        global_gpu_prewarm(boxes_gpu, scores_gpu, args.threshold)

    tasks = build_tasks(boxes_cpu, scores_cpu, boxes_gpu, scores_gpu, args.threshold)
    results = run_tasks(tasks, repeats=args.repeats, warmup=args.warmup, seed=args.seed, order_mode=args.order_mode, order_display=args.order_display)

    metric_rows: List[Row] = []
    iou_vs_tv_rows: List[Row] = []

    for metric in ["iou", "giou", "diou", "ciou"]:
        ref = results[TaskKey(metric, "pure_python", "cpu")]
        ref_idx_cpu = ref.output
        py_time_ms = ref.time_ms

        metric_rows.append(Row(
            group="all-metrics",
            metric=metric.upper(),
            impl="pure_python",
            device="cpu",
            time_ms=py_time_ms,
            speedup_vs_python_cpu=1.0,
            speedup_vs_torchvision_same_device=None,
            matches_python_cpu=True,
            num_kept=int(ref_idx_cpu.numel()),
            note="reference",
        ))

        gen_cpu = results[TaskKey(metric, "gen_nms", "cpu")]
        metric_rows.append(Row(
            group="all-metrics",
            metric=metric.upper(),
            impl="gen_nms",
            device="cpu",
            time_ms=gen_cpu.time_ms,
            speedup_vs_python_cpu=(py_time_ms / gen_cpu.time_ms) if gen_cpu.time_ms > 0 else None,
            speedup_vs_torchvision_same_device=None,
            matches_python_cpu=matches_python_cpu(ref_idx_cpu, gen_cpu.output),
            num_kept=int(gen_cpu.output.numel()),
            note="",
        ))

        if has_cuda:
            gen_gpu = results[TaskKey(metric, "gen_nms", "gpu")]
            metric_rows.append(Row(
                group="all-metrics",
                metric=metric.upper(),
                impl="gen_nms",
                device="gpu",
                time_ms=gen_gpu.time_ms,
                speedup_vs_python_cpu=(py_time_ms / gen_gpu.time_ms) if gen_gpu.time_ms > 0 else None,
                speedup_vs_torchvision_same_device=None,
                matches_python_cpu=matches_python_cpu(ref_idx_cpu, gen_gpu.output),
                num_kept=int(gen_gpu.output.numel()),
                note="",
            ))

    ref_iou = results[TaskKey("iou", "pure_python", "cpu")]
    gen_iou_cpu = results[TaskKey("iou", "gen_nms", "cpu")]
    tv_iou_cpu = results[TaskKey("iou", "torchvision_nms", "cpu")]


    iou_vs_tv_rows.append(Row(
        group="iou-vs-torchvision",
        metric="IOU",
        impl="pure_python",
        device="cpu",
        time_ms=ref_iou.time_ms,
        speedup_vs_python_cpu=1.0,
        speedup_vs_torchvision_same_device=None,
        matches_python_cpu=True,
        num_kept=int(ref_iou.output.numel()),
        note="reference",
    ))
    iou_vs_tv_rows.append(Row(
        group="iou-vs-torchvision",
        metric="IOU",
        impl="gen_nms",
        device="cpu",
        time_ms=gen_iou_cpu.time_ms,
        speedup_vs_python_cpu=(ref_iou.time_ms / gen_iou_cpu.time_ms) if gen_iou_cpu.time_ms > 0 else None,
        speedup_vs_torchvision_same_device=(tv_iou_cpu.time_ms / gen_iou_cpu.time_ms) if gen_iou_cpu.time_ms > 0 else None,
        matches_python_cpu=matches_python_cpu(ref_iou.output, gen_iou_cpu.output),
        num_kept=int(gen_iou_cpu.output.numel()),
        note="",
    ))
    iou_vs_tv_rows.append(Row(
        group="iou-vs-torchvision",
        metric="IOU",
        impl="torchvision_nms",
        device="cpu",
        time_ms=tv_iou_cpu.time_ms,
        speedup_vs_python_cpu=(ref_iou.time_ms / tv_iou_cpu.time_ms) if tv_iou_cpu.time_ms > 0 else None,
        speedup_vs_torchvision_same_device=1.0,
        matches_python_cpu=matches_python_cpu(ref_iou.output, tv_iou_cpu.output),
        num_kept=int(tv_iou_cpu.output.numel()),
        note="",
    ))

    if has_cuda:
        gen_iou_gpu = results[TaskKey("iou", "gen_nms", "gpu")]
        tv_iou_gpu = results[TaskKey("iou", "torchvision_nms", "gpu")]

        iou_vs_tv_rows.append(Row(
            group="iou-vs-torchvision",
            metric="IOU",
            impl="gen_nms",
            device="gpu",
            time_ms=gen_iou_gpu.time_ms,
            speedup_vs_python_cpu=(ref_iou.time_ms / gen_iou_gpu.time_ms) if gen_iou_gpu.time_ms > 0 else None,
            speedup_vs_torchvision_same_device=(tv_iou_gpu.time_ms / gen_iou_gpu.time_ms) if gen_iou_gpu.time_ms > 0 else None,
            matches_python_cpu=matches_python_cpu(ref_iou.output, gen_iou_gpu.output),
            num_kept=int(gen_iou_gpu.output.numel()),
            note="",
        ))
        iou_vs_tv_rows.append(Row(
            group="iou-vs-torchvision",
            metric="IOU",
            impl="torchvision_nms",
            device="gpu",
            time_ms=tv_iou_gpu.time_ms,
            speedup_vs_python_cpu=(ref_iou.time_ms / tv_iou_gpu.time_ms) if tv_iou_gpu.time_ms > 0 else None,
            speedup_vs_torchvision_same_device=1.0,
            matches_python_cpu=matches_python_cpu(ref_iou.output, tv_iou_gpu.output),
            num_kept=int(tv_iou_gpu.output.numel()),
            note="",
        ))

    frame_all = rows_to_frame(metric_rows)
    frame_iou = rows_to_frame(iou_vs_tv_rows)

    print_frame(frame_all, "Python vs gen_nms for IoU / GIoU / DIoU / CIoU")
    print_frame(frame_iou, "IoU NMS: pure Python vs gen_nms vs torchvision.ops.nms")

    if args.csv_prefix:
        prefix = Path(args.csv_prefix)
        all_path = prefix.with_name(prefix.name + "_all_metrics.csv")
        iou_path = prefix.with_name(prefix.name + "_iou_vs_torchvision.csv")

        if pd is not None:
            frame_all.to_csv(all_path, index=False)
            frame_iou.to_csv(iou_path, index=False)
        else:
            import csv
            with open(all_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].__dict__.keys()))
                writer.writeheader()
                for row in metric_rows:
                    writer.writerow(row.__dict__)

            with open(iou_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(iou_vs_tv_rows[0].__dict__.keys()))
                writer.writeheader()
                for row in iou_vs_tv_rows:
                    writer.writerow(row.__dict__)

        print(f"\nWrote CSV files:\n  {all_path}\n  {iou_path}")


if __name__ == "__main__":
    main()

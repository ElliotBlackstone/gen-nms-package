# gen-nms

A small PyTorch extension package that provides **DIoU-based non-maximum suppression (NMS)** (with CUDA support) as a standalone package, without modifying `torchvision`.

The package exposes two main functions:

- `gen_nms.diou_nms(...)`: class-agnostic DIoU NMS
- `gen_nms.batched_diou_nms(...)`: per-class DIoU NMS

This package was derived in part from `torchvision` NMS code and adapted to use **DIoU** instead of IoU. See [`LICENSE.txt`](LICENSE.txt) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for licensing and attribution details.

---

## Why this package exists

`torchvision` includes efficient compiled NMS operators, but it does not expose a DIoU-based NMS operator out of the box. This package provides that functionality in a separate package so you can:

- keep using an unmodified `torchvision`
- call DIoU-based NMS from your own code
- use CPU or CUDA backends through a compiled PyTorch extension

---

## Features

- Standalone PyTorch extension package
- DIoU-based NMS instead of IoU-based suppression
- CPU and CUDA implementations
- Class-agnostic and per-class batched APIs
- Compatible with ordinary PyTorch tensors in `xyxy` format

---

## Installation

### Option 1: install from a local clone

```bash
python -m pip install -e . --no-build-isolation
```

### Option 2: install from GitHub

```bash
python -m pip install git+https://github.com/ElliotBlackstone/gen-nms-package.git
```

### Build notes

This package builds a compiled PyTorch extension. For CUDA builds:

- your local CUDA toolkit should match the CUDA version that your installed PyTorch build expects
- a working C++ compiler and `nvcc` are required
- a virtual environment is strongly recommended

A quick check inside Python is:

```python
import torch
print(torch.__version__)
print(torch.version.cuda)
```

If you are building a CUDA extension, mismatches between the installed CUDA toolkit and `torch.version.cuda` can cause the build to fail.

---

## Quick start

```python
import torch
import gen_nms

boxes = torch.tensor([
    [0.0, 0.0, 10.0, 10.0],
    [1.0, 1.0, 11.0, 11.0],
    [50.0, 50.0, 60.0, 60.0],
], dtype=torch.float32)

scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
labels = torch.tensor([0, 0, 1], dtype=torch.int64)

keep_class_agnostic = gen_nms.diou_nms(boxes, scores, 0.5)
keep_per_class = gen_nms.batched_diou_nms(boxes, scores, labels, 0.5)

print(keep_class_agnostic)
print(keep_per_class)
```

---

## API

### `gen_nms.diou_nms(boxes, scores, diou_threshold)`

Performs class-agnostic DIoU NMS on one pool of boxes.

**Arguments**

- `boxes`: `Tensor[N, 4]` in `(x1, y1, x2, y2)` format
- `scores`: `Tensor[N]`
- `diou_threshold`: `float`

**Returns**

- `Tensor[K]` containing kept indices, ordered by decreasing score

### `gen_nms.batched_diou_nms(boxes, scores, idxs, diou_threshold)`

Performs DIoU NMS independently per class.

**Arguments**

- `boxes`: `Tensor[N, 4]` in `(x1, y1, x2, y2)` format
- `scores`: `Tensor[N]`
- `idxs`: `Tensor[N]` containing class/category indices
- `diou_threshold`: `float`

**Returns**

- `Tensor[K]` containing kept indices, ordered by decreasing score

### Difference between the two

- `diou_nms(...)` allows any box to suppress any other box
- `batched_diou_nms(...)` only suppresses boxes within the same class label

---

## CPU and GPU behavior

The package supports both CPU and CUDA tensors.

- CPU tensors dispatch to the CPU implementation
- CUDA tensors dispatch to the CUDA implementation

The intended semantics are the same, but exact kept indices can differ between CPU and GPU when multiple boxes have identical scores and more than one valid suppression order exists.

In ordinary distinct-score cases, CPU and GPU should usually agree.

---

## Benchmark snapshot

On one benchmark (using [`test_gen_nms_benchmark.py`](./test/test_gen_nms_benchmark.py)) with **1,500 boxes**, the `gen_nms` package exactly matched the DIoU baseline results and delivered the following speedups relative to a Python-level baseline built around `torchvision.ops.distance_box_iou`:

| Mode | Device | Baseline | New op | Speedup |
|---|---:|---:|---:|---:|
| Class-agnostic | CPU | 185.54 ms | 5.97 ms | 31.09x |
| Per-class | CPU | 263.83 ms | 2.39 ms | 110.16x |
| Class-agnostic | CUDA | 1253.79 ms | 0.99 ms | 1266.13x |
| Per-class | CUDA | 1201.22 ms | 1.57 ms | 764.40x |

On that same benchmark, the CPU and GPU outputs of the new package matched exactly.

These results are workload-dependent and should be treated as a benchmark example, not a universal performance guarantee.

---

## Verifying the installation

A quick smoke test:

```python
import torch
import gen_nms

print(hasattr(torch.ops.gen_nms, "diou_nms"))
print(torch.ops.gen_nms.diou_nms)
```

Expected outcome:

- `True`
- an operator object such as `gen_nms.diou_nms`

You can also run a simple functional check:

```python
import torch
import gen_nms

boxes = torch.tensor([
    [0.0, 0.0, 10.0, 10.0],
    [1.0, 1.0, 11.0, 11.0],
    [50.0, 50.0, 60.0, 60.0],
], dtype=torch.float32)

scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
labels = torch.tensor([0, 0, 1], dtype=torch.int64)

print(gen_nms.diou_nms(boxes, scores, 0.5))
print(gen_nms.batched_diou_nms(boxes, scores, labels, 0.5))
```

---

## Development notes

If you are modifying the extension source:

- rebuild the package after changing `.cpp` or `.cu` files
- Python-only changes are easier to test in an editable install
- avoid deleting files recursively inside `.venv/` when cleaning build artifacts

A safe clean-and-reinstall sequence is:

```bash
find . -path './.venv' -prune -o -name '*.o' -delete
find . -path './.venv' -prune -o -name '*.so' -delete
find . -path './.venv' -prune -o -name '*.egg-info' -exec rm -rf {} +
find . -path './.venv' -prune -o -name 'build' -exec rm -rf {} +
python -m pip install -e . --no-build-isolation
```

---

## License

This repository includes code derived from `torchvision` and is distributed under BSD-3-Clause terms. See:

- [`LICENSE.txt`](LICENSE.txt)
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)

---

# gen-nms

A small PyTorch extension package that provides **IoU, generalized IoU (GIoU), distance IoU (DIoU), complete IoU (CIoU)-based non-maximum suppression (NMS)** (with CUDA support) as a standalone package, without modifying `torchvision`.

The package exposes eight main functions:

- `gen_nms.iou_nms(...)`: class-agnostic IoU NMS
- `gen_nms.batched_iou_nms(...)`: per-class IoU NMS
- `gen_nms.giou_nms(...)`: class-agnostic GIoU NMS
- `gen_nms.batched_giou_nms(...)`: per-class GIoU NMS
- `gen_nms.diou_nms(...)`: class-agnostic DIoU NMS
- `gen_nms.batched_diou_nms(...)`: per-class DIoU NMS
- `gen_nms.ciou_nms(...)`: class-agnostic CIoU NMS
- `gen_nms.batched_ciou_nms(...)`: per-class CIoU NMS

The `iou_nms`, `batched_iou_nms` functions are identical to `torchvision` functions `nms`, `batched_nms` and are included for completeness and benchmarking purposes. This package was derived in part from `torchvision` NMS code and adapted to use **GIoU, DIoU, CIoU**. See [`LICENSE.txt`](LICENSE.txt) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for licensing and attribution details.

---

## Why this package exists

`torchvision` includes efficient compiled NMS operators, but it does not expose GIoU, DIoU, CIoU-based NMS operators out of the box. This package provides that functionality in a separate package so you can:

- keep using an unmodified `torchvision`
- call GIoU, DIoU, CIoU-based NMS from your own code
- use CPU or CUDA backends through a compiled PyTorch extension

---

## Features

- Standalone PyTorch extension package
- GIoU, DIoU, CIoU-based NMS as well as IoU-based suppression
- CPU and CUDA implementations
- Class-agnostic and per-class batched APIs
- Compatible with ordinary PyTorch tensors in `xyxy` format

---

## Installation

`gen-nms` is currently installed from source. Install a compatible PyTorch build first, then install this package.

### 1. Install PyTorch first

Check your PyTorch build inside Python:

```python
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
```

### 2. Install `gen-nms`

From a local clone:
```bash
python -m pip install -e . --no-build-isolation
```

From GitHub:
```bash
python -m pip install --no-build-isolation git+https://github.com/ElliotBlackstone/gen-nms-package.git
```


### 3. CUDA build notes

-Install a local CUDA toolkit 
-Install PyTorch with CUDA support (that matches your toolkit version)
-Ensure a working C++ compiler and `nvcc` are available
-On Windows, build from an x64 Visual Studio 2022 developer/native tools shell

---

## Verifying the installation

### Minimal smoke test

```python
import torch
import gen_nms

print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

for name in ["iou_nms", "giou_nms", "diou_nms", "ciou_nms"]:
    assert hasattr(gen_nms, name), f"missing Python API: {name}"
    assert hasattr(torch.ops.gen_nms, name), f"missing registered op: {name}"

boxes = torch.tensor([
    [0.0, 0.0, 10.0, 10.0],
    [1.0, 1.0, 11.0, 11.0],
    [50.0, 50.0, 60.0, 60.0],
], dtype=torch.float32)

scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
labels = torch.tensor([0, 0, 1], dtype=torch.int64)

print("diou_nms:", gen_nms.diou_nms(boxes, scores, 0.5))
print("batched_diou_nms:", gen_nms.batched_diou_nms(boxes, scores, labels, 0.5))

if torch.cuda.is_available():
    boxes_cuda = boxes.cuda()
    scores_cuda = scores.cuda()
    print("diou_nms (cuda):", gen_nms.diou_nms(boxes_cuda, scores_cuda, 0.5))
```

---

## API

### `gen_nms.iou_nms(boxes, scores, iou_threshold)`

Performs class-agnostic IoU NMS on one pool of boxes.

**Arguments**

- `boxes`: `Tensor[N, 4]` in `(x1, y1, x2, y2)` format
- `scores`: `Tensor[N]`
- `iou_threshold`: `float`

**Returns**

- `Tensor[K]` containing kept indices, ordered by decreasing score

### `gen_nms.batched_iou_nms(boxes, scores, idxs, iou_threshold)`

Performs IoU NMS independently per class.

**Arguments**

- `boxes`: `Tensor[N, 4]` in `(x1, y1, x2, y2)` format
- `scores`: `Tensor[N]`
- `idxs`: `Tensor[N]` containing class/category indices
- `iou_threshold`: `float`

**Returns**

- `Tensor[K]` containing kept indices, ordered by decreasing score

### `gen_nms.giou_nms(boxes, scores, giou_threshold)`

Performs class-agnostic GIoU NMS on one pool of boxes.

**Arguments**

- `boxes`: `Tensor[N, 4]` in `(x1, y1, x2, y2)` format
- `scores`: `Tensor[N]`
- `giou_threshold`: `float`

**Returns**

- `Tensor[K]` containing kept indices, ordered by decreasing score

### `gen_nms.batched_giou_nms(boxes, scores, idxs, giou_threshold)`

Performs GIoU NMS independently per class.

**Arguments**

- `boxes`: `Tensor[N, 4]` in `(x1, y1, x2, y2)` format
- `scores`: `Tensor[N]`
- `idxs`: `Tensor[N]` containing class/category indices
- `giou_threshold`: `float`

**Returns**

- `Tensor[K]` containing kept indices, ordered by decreasing score

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

### `gen_nms.ciou_nms(boxes, scores, ciou_threshold)`

Performs class-agnostic CIoU NMS on one pool of boxes.

**Arguments**

- `boxes`: `Tensor[N, 4]` in `(x1, y1, x2, y2)` format
- `scores`: `Tensor[N]`
- `ciou_threshold`: `float`

**Returns**

- `Tensor[K]` containing kept indices, ordered by decreasing score

### `gen_nms.batched_ciou_nms(boxes, scores, idxs, ciou_threshold)`

Performs CIoU NMS independently per class.

**Arguments**

- `boxes`: `Tensor[N, 4]` in `(x1, y1, x2, y2)` format
- `scores`: `Tensor[N]`
- `idxs`: `Tensor[N]` containing class/category indices
- `ciou_threshold`: `float`

**Returns**

- `Tensor[K]` containing kept indices, ordered by decreasing score

### Difference between the two

- `diou_nms(...)` allows any box to suppress any other box
- `batched_diou_nms(...)` only suppresses boxes within the same class label

DIoU was used above as an example.  This relation holds for IoU, GIoU, and CIoU.

---

## CPU and GPU behavior

The package supports both CPU and CUDA tensors.

- CPU tensors dispatch to the CPU implementation
- CUDA tensors dispatch to the CUDA implementation

The intended semantics are the same, but exact kept indices can differ between CPU and GPU when multiple boxes have identical scores and more than one valid suppression order exists.

In ordinary distinct-score cases, CPU and GPU should usually agree.

---

## Benchmark snapshot

On one benchmark (using [`test_gen_nms_all_metrics_v4.py`](./test/test_gen_nms_all_metrics_v4.py)) with **1000 boxes**, the `gen_nms` package exactly matched and delivered the following speedups relative to a Python-level baseline built around `torchvision.ops.distance_box_iou`:

| Metric | Implementation | Device | Time (ms) | Speedup vs Python CPU | Matches Python CPU | Boxes Kept |
| ------ | -------------- | -----: | --------: | --------------------: | ------------------ | ---------: |
| IoU    | pure_python    |    cpu |    70.828 |                 1.00× | Yes                |        885 |
| IoU    | gen_nms        |    cpu |     2.136 |                33.17× | Yes                |        885 |
| IoU    | gen_nms        |    gpu |     2.290 |                30.93× | Yes                |        885 |
| GIoU   | pure_python    |    cpu |    96.142 |                 1.00× | Yes                |        917 |
| GIoU   | gen_nms        |    cpu |     2.401 |                40.04× | Yes                |        917 |
| GIoU   | gen_nms        |    gpu |     2.439 |                39.42× | Yes                |        917 |
| DIoU   | pure_python    |    cpu |   127.262 |                 1.00× | Yes                |        903 |
| DIoU   | gen_nms        |    cpu |     2.746 |                46.35× | Yes                |        903 |
| DIoU   | gen_nms        |    gpu |     2.345 |                54.27× | Yes                |        903 |
| CIoU   | pure_python    |    cpu |   179.205 |                 1.00× | Yes                |        905 |
| CIoU   | gen_nms        |    cpu |     3.110 |                57.63× | Yes                |        905 |
| CIoU   | gen_nms        |    gpu |     2.490 |                71.96× | Yes                |        905 |


These results are workload-dependent and should be treated as a benchmark example, not a universal performance guarantee.  More benchmark results are available in the ['test'](./test/) folder.  Unsurprisingly, GPU outperforms CPU for larger values of n, and vice versa for smaller values of n.

### IoU NMS vs `torchvision.ops.nms` (n = 1000)

The following table demonstrates that `gen_nms.iou_nms` performs as well as `torchvision.ops.nms`.

| Implementation  | Device | Time (ms) | Speedup vs Python CPU | Relative to `torchvision` | Matches Python CPU | Boxes Kept |
| --------------- | -----: | --------: | --------------------: | ------------------------: | ------------------ | ---------: |
| pure_python     |    cpu |    70.828 |                 1.00× |                         — | Yes                |        885 |
| gen_nms         |    cpu |     2.136 |                33.17× |                    1.011× | Yes                |        885 |
| torchvision_nms |    cpu |     2.159 |                32.81× |                    1.000× | Yes                |        885 |
| gen_nms         |    gpu |     2.290 |                30.93× |                    0.999× | Yes                |        885 |
| torchvision_nms |    gpu |     2.287 |                30.98× |                    1.000× | Yes                |        885 |




---

## Verifying the installation

A quick smoke test:

```python
import torch
import gen_nms

OPS = [
    torch.ops.gen_nms.iou_nms,
    torch.ops.gen_nms.giou_nms,
    torch.ops.gen_nms.diou_nms,
    torch.ops.gen_nms.ciou_nms,
]

def sample_inputs(device):
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [50.0, 50.0, 60.0, 60.0],
            [52.0, 52.0, 61.0, 61.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float32, device=device)
    return (boxes, scores, 0.5)

for op in OPS:
    torch.library.opcheck(op, sample_inputs("cpu"))
    print(f"{op} CPU opcheck passed")

if torch.cuda.is_available():
    for op in OPS:
        torch.library.opcheck(op, sample_inputs("cuda"))
        print(f"{op} CUDA opcheck passed")
```

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

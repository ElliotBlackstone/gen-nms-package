# gen-nms

A small PyTorch extension package that provides **IoU, generalized IoU (GIoU), distance IoU (DIoU), and complete IoU (CIoU)-based non-maximum suppression (NMS)** as a standalone package, implemented in native C++ on CPU and CUDA on GPU, without modifying `torchvision`.

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
python -m pip install . --no-build-isolation
```

From GitHub:
```bash
python -m pip install --no-build-isolation git+https://github.com/ElliotBlackstone/gen-nms-package.git
```


### 3. CUDA build notes

- Install a local CUDA toolkit 
- Install PyTorch with CUDA support (that matches your toolkit version)
- Ensure a working C++ compiler and `nvcc` are available
- On Windows, build from an x64 Visual Studio 2022 developer/native tools shell

---

## Verifying the installation

Run the smoke test:

```bash
python test/smoke_test.py
```

Run the operator checks:
```bash
python test/test_opcheck.py
```


---

## API

All functions return a `Tensor[K]` of kept indices, ordered by decreasing score.

### Common arguments

- `boxes`: `Tensor[N, 4]` in `(x1, y1, x2, y2)` format
- `scores`: `Tensor[N]`
- `idxs`: `Tensor[N]` containing class/category indices (batched functions only)
- `*_threshold`: `float` threshold for the corresponding overlap metric

### Function summary

| Function | Suppression scope | Overlap metric |
| --- | --- | --- |
| `gen_nms.iou_nms(boxes, scores, iou_threshold)` | class-agnostic | IoU |
| `gen_nms.batched_iou_nms(boxes, scores, idxs, iou_threshold)` | per-class | IoU |
| `gen_nms.giou_nms(boxes, scores, giou_threshold)` | class-agnostic | GIoU |
| `gen_nms.batched_giou_nms(boxes, scores, idxs, giou_threshold)` | per-class | GIoU |
| `gen_nms.diou_nms(boxes, scores, diou_threshold)` | class-agnostic | DIoU |
| `gen_nms.batched_diou_nms(boxes, scores, idxs, diou_threshold)` | per-class | DIoU |
| `gen_nms.ciou_nms(boxes, scores, ciou_threshold)` | class-agnostic | CIoU |
| `gen_nms.batched_ciou_nms(boxes, scores, idxs, ciou_threshold)` | per-class | CIoU |

### Class-agnostic vs batched

- The class-agnostic functions allow any box to suppress any other box.
- The batched functions only allow suppression among boxes with the same value in `idxs`.

### Notes

- `gen_nms.iou_nms` and `gen_nms.batched_iou_nms` are included for completeness and benchmarking against the corresponding `torchvision` IoU-based APIs.
- CPU tensors dispatch to the CPU implementation, and CUDA tensors dispatch to the CUDA implementation.

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


These results are workload-dependent and should be treated as a benchmark example, not a universal performance guarantee.  More benchmark results are available in the ['test/'](./test/) folder.  Unsurprisingly, GPU outperforms CPU for larger values of n, and vice versa for smaller values of n.

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

## Development notes

If you are modifying the extension source:

- rebuild the package after changing `.cpp` or `.cu` files
- Python-only changes are easier to test in an editable install
- avoid deleting files recursively inside `.venv/` when cleaning build artifacts

A safe clean-and-reinstall sequence is (assuming your environment name is `.venv`):

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

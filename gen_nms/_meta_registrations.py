"""FakeTensor / Meta registrations for gen_nms custom operators.

These registrations let torch.compile / torch.export / FX reason about the
metadata of gen_nms custom ops without executing the real CPU / CUDA kernels.

Operator schemas:
    gen_nms::iou_nms(Tensor boxes, Tensor scores, float threshold) -> Tensor
    gen_nms::giou_nms(Tensor boxes, Tensor scores, float threshold) -> Tensor
    gen_nms::diou_nms(Tensor boxes, Tensor scores, float threshold) -> Tensor
    gen_nms::ciou_nms(Tensor boxes, Tensor scores, float threshold) -> Tensor

Each operator returns a 1-D LongTensor of kept indices whose length depends on
input data, so the fake kernel returns a symbolic-length tensor.
"""

from __future__ import annotations

import functools

import torch
from torch import Tensor

try:
    # Import the extension first so the C++ TORCH_LIBRARY definitions exist
    # before we add Python-side fake / meta registrations.
    from . import _C  # noqa: F401
except ImportError:
    # Allow the module to be imported in environments where the extension is not
    # built yet. Registration will be skipped if the ops are unavailable.
    _C = None


@functools.lru_cache(maxsize=None)
def _get_meta_lib(namespace: str) -> torch.library.Library:
    return torch.library.Library(namespace, "IMPL", "Meta")


@functools.lru_cache(maxsize=None)
def _op_exists(qualname: str) -> bool:
    namespace, opname = qualname.split("::", 1)
    try:
        getattr(getattr(torch.ops, namespace), opname)
        return True
    except (AttributeError, RuntimeError):
        return False


def _new_dynamic_size() -> int | torch.SymInt:
    if hasattr(torch.library, "get_ctx"):
        return torch.library.get_ctx().new_dynamic_size()
    return torch._custom_ops.get_ctx().create_unbacked_symint()


def _register_fake_if_available(qualname: str):
    def decorator(fn):
        if not _op_exists(qualname):
            return fn

        if hasattr(torch.library, "register_fake"):
            return torch.library.register_fake(qualname)(fn)

        namespace, opname = qualname.split("::", 1)
        _get_meta_lib(namespace).impl(opname, fn)
        return fn

    return decorator


def _check_nms_inputs(boxes: Tensor, scores: Tensor) -> None:
    torch._check(boxes.dim() == 2, lambda: f"boxes must be 2D, got {boxes.dim()}D")
    torch._check(boxes.size(1) == 4, lambda: f"boxes must have shape [N, 4], got {tuple(boxes.shape)}")
    torch._check(scores.dim() == 1, lambda: f"scores must be 1D, got {scores.dim()}D")
    torch._check(
        boxes.size(0) == scores.size(0),
        lambda: f"boxes and scores must agree in dim 0, got {boxes.size(0)} and {scores.size(0)}",
    )
    torch._check(
        boxes.device == scores.device,
        lambda: f"boxes and scores must be on the same device, got {boxes.device} and {scores.device}",
    )


def _meta_nms_common(boxes: Tensor, scores: Tensor, threshold: float) -> Tensor:
    del threshold  # threshold does not affect output metadata
    _check_nms_inputs(boxes, scores)
    num_to_keep = _new_dynamic_size()
    return boxes.new_empty((num_to_keep,), dtype=torch.long)


@_register_fake_if_available("gen_nms::iou_nms")
def meta_iou_nms(boxes: Tensor, scores: Tensor, threshold: float) -> Tensor:
    return _meta_nms_common(boxes, scores, threshold)


@_register_fake_if_available("gen_nms::giou_nms")
def meta_giou_nms(boxes: Tensor, scores: Tensor, threshold: float) -> Tensor:
    return _meta_nms_common(boxes, scores, threshold)


@_register_fake_if_available("gen_nms::diou_nms")
def meta_diou_nms(boxes: Tensor, scores: Tensor, threshold: float) -> Tensor:
    return _meta_nms_common(boxes, scores, threshold)


@_register_fake_if_available("gen_nms::ciou_nms")
def meta_ciou_nms(boxes: Tensor, scores: Tensor, threshold: float) -> Tensor:
    return _meta_nms_common(boxes, scores, threshold)


__all__ = [
    "meta_iou_nms",
    "meta_giou_nms",
    "meta_diou_nms",
    "meta_ciou_nms",
]

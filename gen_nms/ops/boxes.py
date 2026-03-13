# This file is derived from torchvision's boxes.py and has been modified.
# Copyright for original portions belongs to the torchvision contributors.
# Modifications Copyright (c) 2026 Elliot Blackstone.
# Licensed under the BSD-3-Clause License. See LICENSE and THIRD_PARTY_NOTICES.md.

import torch
from torch import Tensor

from ..utils import _log_api_usage_once


def diou_nms(boxes: Tensor, scores: Tensor, diou_threshold: float) -> Tensor:
    """
    Performs class agnostic non-maximum suppression (NMS) on the boxes according
    to their distance-intersection-over-union (DIoU).

    NMS iteratively removes lower scoring boxes which have a
    DIoU greater than ``diou_threshold`` with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the DIoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        diou_threshold (float): discards all overlapping boxes with DIoU > diou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(diou_nms)
    return torch.ops.gen_nms.diou_nms(boxes, scores, diou_threshold)


def batched_diou_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    diou_threshold: float,
) -> Tensor:
    """
    Performs class aware non-maximum suppression in a batched fashion,
    using DIoU instead of IoU.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        diou_threshold (float): discards all overlapping boxes with DIoU > diou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """    
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(batched_diou_nms)
    # Benchmarks that drove the following thresholds are at
    # https://github.com/pytorch/vision/issues/1311#issuecomment-781329339
    # and https://github.com/pytorch/vision/pull/8925
    if boxes.numel() > (4000 if boxes.device.type == "cpu" else 100_000) and not torch.jit.is_tracing():
        return _batched_diou_nms_vanilla(boxes, scores, idxs, diou_threshold)
    else:
        return _batched_diou_nms_coordinate_trick(boxes, scores, idxs, diou_threshold)
    



@torch.jit._script_if_tracing
def _batched_diou_nms_coordinate_trick(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    diou_threshold: float,
) -> Tensor:
    # strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = diou_nms(boxes_for_nms, scores, diou_threshold)
    return keep


@torch.jit._script_if_tracing
def _batched_diou_nms_vanilla(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    diou_threshold: float,
) -> Tensor:
    # Based on Detectron2 implementation, just manually call diou_nms() on each class independently
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(idxs):
        curr_indices = torch.where(idxs == class_id)[0]
        curr_keep_indices = diou_nms(boxes[curr_indices], scores[curr_indices], diou_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    return keep_indices[scores[keep_indices].sort(descending=True)[1]]

# This file is derived from torchvision's boxes.py and has been modified.
# Copyright for original portions belongs to the torchvision contributors.
# Modifications Copyright (c) 2026 Elliot Blackstone.
# Licensed under the BSD-3-Clause License. See LICENSE and THIRD_PARTY_NOTICES.md.

import torch
from torch import Tensor

from ..utils import _log_api_usage_once


def iou_nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    Performs class agnostic non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have a
    IoU greater than ``iou_threshold`` with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the IoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(iou_nms)
    return torch.ops.gen_nms.iou_nms(boxes, scores, iou_threshold)

def giou_nms(boxes: Tensor, scores: Tensor, giou_threshold: float) -> Tensor:
    """
    Performs class agnostic non-maximum suppression (NMS) on the boxes according
    to their generalized-intersection-over-union (GIoU).

    NMS iteratively removes lower scoring boxes which have a
    GIoU greater than ``giou_threshold`` with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the GIoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        giou_threshold (float): discards all overlapping boxes with GIoU > giou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(giou_nms)
    return torch.ops.gen_nms.giou_nms(boxes, scores, giou_threshold)

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

def ciou_nms(boxes: Tensor, scores: Tensor, ciou_threshold: float) -> Tensor:
    """
    Performs class agnostic non-maximum suppression (NMS) on the boxes according
    to their complete-intersection-over-union (CIoU).

    NMS iteratively removes lower scoring boxes which have a
    CIoU greater than ``ciou_threshold`` with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the CIoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        ciou_threshold (float): discards all overlapping boxes with CIoU > ciou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(ciou_nms)
    return torch.ops.gen_nms.ciou_nms(boxes, scores, ciou_threshold)


def batched_iou_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """
    Performs class aware non-maximum suppression in a batched fashion,
    using IoU.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """    
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(batched_iou_nms)
    # Benchmarks that drove the following thresholds are at
    # https://github.com/pytorch/vision/issues/1311#issuecomment-781329339
    # and https://github.com/pytorch/vision/pull/8925
    if boxes.numel() > (4000 if boxes.device.type == "cpu" else 100_000) and not torch.jit.is_tracing():
        return _batched_nms_vanilla(iou_nms, boxes, scores, idxs, iou_threshold)
    else:
        return _batched_nms_coordinate_trick(iou_nms, boxes, scores, idxs, iou_threshold)

def batched_giou_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    giou_threshold: float,
) -> Tensor:
    """
    Performs class aware non-maximum suppression in a batched fashion,
    using GIoU instead of IoU.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        giou_threshold (float): discards all overlapping boxes with GIoU > giou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """    
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(batched_giou_nms)
    # Benchmarks that drove the following thresholds are at
    # https://github.com/pytorch/vision/issues/1311#issuecomment-781329339
    # and https://github.com/pytorch/vision/pull/8925
    if boxes.numel() > (4000 if boxes.device.type == "cpu" else 100_000) and not torch.jit.is_tracing():
        return _batched_nms_vanilla(giou_nms, boxes, scores, idxs, giou_threshold)
    else:
        return _batched_nms_coordinate_trick(giou_nms, boxes, scores, idxs, giou_threshold)
    
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
        return _batched_nms_vanilla(diou_nms, boxes, scores, idxs, diou_threshold)
    else:
        return _batched_nms_coordinate_trick(diou_nms, boxes, scores, idxs, diou_threshold)

def batched_ciou_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    ciou_threshold: float,
) -> Tensor:
    """
    Performs class aware non-maximum suppression in a batched fashion,
    using CIoU instead of IoU.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        ciou_threshold (float): discards all overlapping boxes with CIoU > ciou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """    
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(batched_ciou_nms)
    # Benchmarks that drove the following thresholds are at
    # https://github.com/pytorch/vision/issues/1311#issuecomment-781329339
    # and https://github.com/pytorch/vision/pull/8925
    if boxes.numel() > (4000 if boxes.device.type == "cpu" else 100_000) and not torch.jit.is_tracing():
        return _batched_nms_vanilla(ciou_nms, boxes, scores, idxs, ciou_threshold)
    else:
        return _batched_nms_coordinate_trick(ciou_nms, boxes, scores, idxs, ciou_threshold)



@torch.jit._script_if_tracing
def _batched_nms_coordinate_trick(op, boxes: Tensor, scores: Tensor, idxs: Tensor, threshold: float):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    return op(boxes_for_nms, scores, threshold)

@torch.jit._script_if_tracing
def _batched_nms_vanilla(op, boxes: Tensor, scores: Tensor, idxs: Tensor, threshold: float):
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(idxs):
        curr_indices = torch.where(idxs == class_id)[0]
        curr_keep = op(boxes[curr_indices], scores[curr_indices], threshold)
        keep_mask[curr_indices[curr_keep]] = True
    keep_indices = torch.where(keep_mask)[0]
    return keep_indices[scores[keep_indices].sort(descending=True)[1]]

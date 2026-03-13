# This file is derived from torchvision's __init__.py and has been modified.
# Copyright for original portions belongs to the torchvision contributors.
# Modifications Copyright (c) 2026 Elliot Blackstone.
# Licensed under the BSD-3-Clause License. See LICENSE and THIRD_PARTY_NOTICES.md.

from pathlib import Path
import torch

_so_files = list(Path(__file__).parent.glob("_C*.so"))
if len(_so_files) != 1:
    raise RuntimeError(f"Expected exactly one compiled library matching _C*.so, found: {_so_files}")

torch.ops.load_library(str(_so_files[0]))

from .ops import diou_nms, batched_diou_nms

__all__ = ["diou_nms", "batched_diou_nms"]
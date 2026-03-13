# This file is derived from torchvision's __init__.py and has been modified.
# Copyright for original portions belongs to the torchvision contributors.
# Modifications Copyright (c) 2026 Elliot Blackstone.
# Licensed under the BSD-3-Clause License. See LICENSE and THIRD_PARTY_NOTICES.md.

from .boxes import (
    diou_nms,
    batched_diou_nms,
)





__all__ = [
    "diou_nms",
    "batched_diou_nms",
]

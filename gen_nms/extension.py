# This file is derived from torchvision's extension.py and has been modified.
# Copyright for original portions belongs to the torchvision contributors.
# Modifications Copyright (c) 2026 Elliot Blackstone.
# Licensed under the BSD-3-Clause License. See LICENSE and THIRD_PARTY_NOTICES.md.

import os
import torch

from ._internally_replaced_utils import _get_extension_path


def _load_library(lib_name):
    try:
        lib_path = _get_extension_path(lib_name)
        torch.ops.load_library(lib_path)
        return True
    except (ImportError, OSError) as e:
        if os.environ.get("GEN_NMS_WARN_WHEN_EXTENSION_LOADING_FAILS"):
            import warnings
            warnings.warn(f"Failed to load '{lib_name}' extension: {type(e).__name__}: {e}")
        return False


def _has_ops():
    return False


if _load_library("_C"):
    def _has_ops():  # noqa: F811
        return True


def _assert_has_ops():
    if not _has_ops():
        raise RuntimeError(
            "Couldn't load custom C++ ops for gen_nms. "
            "This can happen if your PyTorch install is incompatible with this wheel, "
            "or if there were errors while compiling the extension. "
            "Set GEN_NMS_WARN_WHEN_EXTENSION_LOADING_FAILS=1 and retry to get more details."
        )
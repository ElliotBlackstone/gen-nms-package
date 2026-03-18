// This file is derived from torchvision's iou_nms.h and has been modified.
// Copyright for original portions belongs to the torchvision contributors.
// Modifications Copyright (c) 2026 Elliot Blackstone.
// Licensed under the BSD-3-Clause License. See LICENSE and THIRD_PARTY_NOTICES.md.

#pragma once

#include <ATen/ATen.h>

namespace gen_nms {
namespace ops {

at::Tensor ciou_nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double ciou_threshold);

} // namespace ops
} // namespace gen_nms

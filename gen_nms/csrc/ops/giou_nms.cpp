// This file is derived from torchvision's iou_nms.cpp and has been modified.
// Copyright for original portions belongs to the torchvision contributors.
// Modifications Copyright (c) 2026 Elliot Blackstone.
// Licensed under the BSD-3-Clause License. See LICENSE and THIRD_PARTY_NOTICES.md.

#include "giou_nms.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace gen_nms {
namespace ops {

at::Tensor giou_nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double giou_threshold) {
  C10_LOG_API_USAGE_ONCE("gen_nms.csrc.ops.giou_nms.giou_nms");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("gen_nms::giou_nms", "")
                       .typed<decltype(giou_nms)>();
  return op.call(dets, scores, giou_threshold);
}

TORCH_LIBRARY_FRAGMENT(gen_nms, m) {
  m.def("gen_nms::giou_nms(Tensor dets, Tensor scores, float giou_threshold) -> Tensor");
}

} // namespace ops
} // namespace gen_nms

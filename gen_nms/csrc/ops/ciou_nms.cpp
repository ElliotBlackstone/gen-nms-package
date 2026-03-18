// This file is derived from torchvision's iou_nms.cpp and has been modified.
// Copyright for original portions belongs to the torchvision contributors.
// Modifications Copyright (c) 2026 Elliot Blackstone.
// Licensed under the BSD-3-Clause License. See LICENSE and THIRD_PARTY_NOTICES.md.

#include "ciou_nms.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <torch/types.h>

namespace gen_nms {
namespace ops {

at::Tensor ciou_nms(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double ciou_threshold) {
  C10_LOG_API_USAGE_ONCE("gen_nms.csrc.ops.ciou_nms.ciou_nms");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("gen_nms::ciou_nms", "")
                       .typed<decltype(ciou_nms)>();
  return op.call(dets, scores, ciou_threshold);
}

TORCH_LIBRARY_FRAGMENT(gen_nms, m) {
  m.def("gen_nms::ciou_nms(Tensor dets, Tensor scores, float ciou_threshold) -> Tensor");
}

} // namespace ops
} // namespace gen_nms

// This file is derived from torchvision's iou_nms_kernel.cpp and has been modified.
// Copyright for original portions belongs to the torchvision contributors.
// Modifications Copyright (c) 2026 Elliot Blackstone.
// Licensed under the BSD-3-Clause License. See LICENSE and THIRD_PARTY_NOTICES.md.

#include <ATen/ATen.h>
#include <torch/library.h>

namespace gen_nms {
namespace ops {

namespace {

template <typename scalar_t>
at::Tensor nms_kernel_impl(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double diou_threshold) {
  TORCH_CHECK(dets.is_cpu(), "dets must be a CPU tensor");
  TORCH_CHECK(scores.is_cpu(), "scores must be a CPU tensor");
  TORCH_CHECK(dets.scalar_type() == scores.scalar_type(),
              "dets should have the same type as scores");
  TORCH_CHECK(dets.dim() == 2 && dets.size(1) == 4, "dets must be Nx4");
  TORCH_CHECK(scores.dim() == 1 && scores.size(0) == dets.size(0), "scores must be N");

  const auto ndets = dets.size(0);
  if (ndets == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong));
  }

  // Split coords
  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();

  // Stable descending sort
  auto order_t = std::get<1>(scores.sort(/*stable=*/true, /*dim=*/0, /*descending=*/true));

  auto suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte));
  auto keep_t = at::zeros({ndets}, dets.options().dtype(at::kLong));

  auto* suppressed = suppressed_t.data_ptr<uint8_t>();
  auto* keep = keep_t.data_ptr<int64_t>();
  auto* order = order_t.data_ptr<int64_t>();

  auto* x1p = x1_t.data_ptr<scalar_t>();
  auto* y1p = y1_t.data_ptr<scalar_t>();
  auto* x2p = x2_t.data_ptr<scalar_t>();
  auto* y2p = y2_t.data_ptr<scalar_t>();

  using acc_t = double;
  const acc_t eps = 1e-7;

  int64_t num_to_keep = 0;

  for (int64_t _i = 0; _i < ndets; ++_i) {
    const int64_t i = order[_i];
    if (suppressed[i]) {
      continue;
    }
    keep[num_to_keep++] = i;

    // box i (cast once)
    const acc_t ix1 = (acc_t)x1p[i];
    const acc_t iy1 = (acc_t)y1p[i];
    const acc_t ix2 = (acc_t)x2p[i];
    const acc_t iy2 = (acc_t)y2p[i];

    const acc_t iw0 = ix2 - ix1;
    const acc_t ih0 = iy2 - iy1;
    const acc_t iarea = iw0 * ih0;

    // center of i
    const acc_t cix = (ix1 + ix2) * 0.5;
    const acc_t ciy = (iy1 + iy2) * 0.5;

    for (int64_t _j = _i + 1; _j < ndets; ++_j) {
      const int64_t j = order[_j];
      if (suppressed[j]) {
        continue;
      }

      // box j
      const acc_t jx1 = (acc_t)x1p[j];
      const acc_t jy1 = (acc_t)y1p[j];
      const acc_t jx2 = (acc_t)x2p[j];
      const acc_t jy2 = (acc_t)y2p[j];

      const acc_t jw0 = jx2 - jx1;
      const acc_t jh0 = jy2 - jy1;
      const acc_t jarea = jw0 * jh0;

      // intersection
      const acc_t xx1 = std::max(ix1, jx1);
      const acc_t yy1 = std::max(iy1, jy1);
      const acc_t xx2 = std::min(ix2, jx2);
      const acc_t yy2 = std::min(iy2, jy2);

      const acc_t w = std::max((acc_t)0, xx2 - xx1);
      const acc_t h = std::max((acc_t)0, yy2 - yy1);
      const acc_t inter = w * h;

      // IoU
      const acc_t uni = iarea + jarea - inter;
      if (uni <= (acc_t)0) {
        continue;  // degenerate; do not suppress
      }
      const acc_t iou = inter / uni;

      // center distance squared rho^2
      const acc_t cjx = (jx1 + jx2) * 0.5;
      const acc_t cjy = (jy1 + jy2) * 0.5;
      const acc_t dx = cix - cjx;
      const acc_t dy = ciy - cjy;
      const acc_t rho2 = dx * dx + dy * dy;

      // enclosing diagonal squared c^2 (+eps)
      const acc_t ex1 = std::min(ix1, jx1);
      const acc_t ey1 = std::min(iy1, jy1);
      const acc_t ex2 = std::max(ix2, jx2);
      const acc_t ey2 = std::max(iy2, jy2);
      const acc_t cw = ex2 - ex1;
      const acc_t ch = ey2 - ey1;
      const acc_t c2 = cw * cw + ch * ch + eps;

      // DIoU
      const acc_t diou = iou - (rho2 / c2);

      if (diou > (acc_t)diou_threshold) {
        suppressed[j] = 1;
      }
    }
  }

  return keep_t.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep);
}



at::Tensor diou_nms_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double diou_threshold) {
  TORCH_CHECK(
      dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(
      dets.size(1) == 4,
      "boxes should have 4 elements in dimension 1, got ",
      dets.size(1));
  TORCH_CHECK(
      scores.dim() == 1,
      "scores should be a 1d tensor, got ",
      scores.dim(),
      "D");
  TORCH_CHECK(
      dets.size(0) == scores.size(0),
      "boxes and scores should have same number of elements in ",
      "dimension 0, got ",
      dets.size(0),
      " and ",
      scores.size(0));

  auto result = at::empty({0}, dets.options());

  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "diou_nms_kernel", [&] {
    result = nms_kernel_impl<scalar_t>(dets, scores, diou_threshold);
  });
  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(gen_nms, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("gen_nms::diou_nms"), TORCH_FN(diou_nms_kernel));
}

} // namespace ops
} // namespace gen_nms

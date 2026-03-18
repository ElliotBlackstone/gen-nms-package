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
at::Tensor ciou_nms_kernel_impl(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double ciou_threshold) {
  TORCH_CHECK(dets.is_cpu(), "dets must be a CPU tensor");
  TORCH_CHECK(scores.is_cpu(), "scores must be a CPU tensor");
  TORCH_CHECK(dets.scalar_type() == scores.scalar_type(),
              "dets should have the same type as scores");
  TORCH_CHECK(dets.dim() == 2 && dets.size(1) == 4, "dets must be Nx4");
  TORCH_CHECK(scores.dim() == 1 && scores.size(0) == dets.size(0), "scores must be N");

  auto ndets = dets.size(0);
  if (ndets == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong));
  }

  // Split coords
  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();

  at::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

  // centers
  at::Tensor cx_t = (x2_t + x1_t) * 0.5;
  at::Tensor cy_t = (y2_t + y1_t) * 0.5;

  // aspect ratio
  at::Tensor atan_t = atan((x2_t - x1_t) / (y2_t - y1_t));

  // Stable descending sort
  auto order_t = std::get<1>(scores.sort(/*stable=*/true, /*dim=*/0, /*descending=*/true));

  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte));
  at::Tensor keep_t = at::zeros({ndets}, dets.options().dtype(at::kLong));

  auto suppressed = suppressed_t.data_ptr<uint8_t>();
  auto keep = keep_t.data_ptr<int64_t>();
  auto order = order_t.data_ptr<int64_t>();

  auto x1 = x1_t.data_ptr<scalar_t>();
  auto y1 = y1_t.data_ptr<scalar_t>();
  auto x2 = x2_t.data_ptr<scalar_t>();
  auto y2 = y2_t.data_ptr<scalar_t>();
  auto areas = areas_t.data_ptr<scalar_t>();
  auto cx = cx_t.data_ptr<scalar_t>();
  auto cy = cy_t.data_ptr<scalar_t>();
  auto angles = atan_t.data_ptr<scalar_t>();

  int64_t num_to_keep = 0;

  for (int64_t _i = 0; _i < ndets; ++_i) {
    const int64_t i = order[_i];
    if (suppressed[i]) {
      continue;
    }
    keep[num_to_keep++] = i;

    // box i
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    // center of i
    auto cix = cx[i];
    auto ciy = cy[i];

    for (int64_t _j = _i + 1; _j < ndets; ++_j) {
      const int64_t j = order[_j];
      if (suppressed[j]) {
        continue;
      }

      // box j
      auto jx1 = x1[j];
      auto jy1 = y1[j];
      auto jx2 = x2[j];
      auto jy2 = y2[j];

      auto xx1 = std::max(ix1, jx1);
      auto yy1 = std::max(iy1, jy1);
      auto xx2 = std::min(ix2, jx2);
      auto yy2 = std::min(iy2, jy2);
      
      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1);
      auto inter = w * h;

      // IoU
      auto uni = iarea + areas[j] - inter;
      if (uni <= 0) {
        continue;  // degenerate; do not suppress
      }
      auto iou = inter / uni;

      // center distance squared rho^2
      auto dx = cix - cx[j];
      auto dy = ciy - cy[j];
      auto rho2 = dx * dx + dy * dy;

      // enclosing diagonal squared c^2
      auto ex1 = std::min(ix1, jx1);
      auto ey1 = std::min(iy1, jy1);
      auto ex2 = std::max(ix2, jx2);
      auto ey2 = std::max(iy2, jy2);
      auto cw = ex2 - ex1;
      auto ch = ey2 - ey1;
      auto c2 = cw * cw + ch * ch;

      // aspect ratio
      double pi = 3.14159265358979323846;
      auto v = (4 / (pi * pi)) * (angles[i] - angles[j]) * (angles[i] - angles[j]);
      auto alpha = v / (1 - iou + v);

      // CIoU
      auto ciou = iou - (rho2 / c2) - alpha * v;

      if (ciou > ciou_threshold) {
        suppressed[j] = 1;
      }
    }
  }

  return keep_t.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep);
}



at::Tensor ciou_nms_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double ciou_threshold) {
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

  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "ciou_nms_kernel", [&] {
    result = ciou_nms_kernel_impl<scalar_t>(dets, scores, ciou_threshold);
  });
  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(gen_nms, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("gen_nms::ciou_nms"), TORCH_FN(ciou_nms_kernel));
}

} // namespace ops
} // namespace gen_nms

import torch
import gen_nms


def main() -> None:
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())

    op_names = ["iou_nms", "giou_nms", "diou_nms", "ciou_nms"]
    for name in op_names:
        assert hasattr(gen_nms, name), f"missing Python API: {name}"
        assert hasattr(torch.ops.gen_nms, name), f"missing registered op: {name}"

    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [50.0, 50.0, 60.0, 60.0],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1], dtype=torch.int64)

    keep = gen_nms.diou_nms(boxes, scores, 0.5)
    keep_batched = gen_nms.batched_diou_nms(boxes, scores, labels, 0.5)

    assert keep.dtype == torch.long
    assert keep_batched.dtype == torch.long

    print("CPU diou_nms:", keep)
    print("CPU batched_diou_nms:", keep_batched)

    if torch.cuda.is_available():
        keep_cuda = gen_nms.diou_nms(boxes.cuda(), scores.cuda(), 0.5)
        assert keep_cuda.dtype == torch.long
        print("CUDA diou_nms:", keep_cuda)

    print("smoke test passed")


if __name__ == "__main__":
    main()
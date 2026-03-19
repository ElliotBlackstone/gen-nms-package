import torch
import gen_nms


OPS = [
    torch.ops.gen_nms.iou_nms,
    torch.ops.gen_nms.giou_nms,
    torch.ops.gen_nms.diou_nms,
    torch.ops.gen_nms.ciou_nms,
]


def sample_inputs(device: str):
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [50.0, 50.0, 60.0, 60.0],
            [52.0, 52.0, 61.0, 61.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float32, device=device)
    return (boxes, scores, 0.5)


def test_opcheck_cpu():
    print("---------- CPU op check ----------")
    for op in OPS:
        print(f"{op}, CPU: {torch.library.opcheck(op, sample_inputs("cpu"))}")
        print()


def test_opcheck_cuda():
    print("---------- CUDA op check ----------")
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    for op in OPS:
        print(f"{op}, CUDA: {torch.library.opcheck(op, sample_inputs("cuda"))}")
        print()


if __name__ == "__main__":
    test_opcheck_cpu()
    test_opcheck_cuda()
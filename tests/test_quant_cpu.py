import torch
from edge0.quant_cpu import unpack_int4_affine

def test_unpack_affine_int4():
    # Eight 4-bit values packed into one uint32 word.
    vals = torch.tensor([0, 1, 2, 3, 4, 5, 6, 15], dtype=torch.int64)
    packed = sum((int(v) << (4*i)) for i, v in enumerate(vals)).to_bytes(4, "little")
    p = torch.tensor([int.from_bytes(packed, "little")], dtype=torch.uint32).reshape(1, 1, 1)
    scales = torch.tensor([[[2.0]]])
    biases = torch.tensor([[[1.0]]])
    w = unpack_int4_affine(p, scales, biases, group_size=8)
    expected = (vals.float() - 1.0) * 2.0
    assert torch.equal(w[0, 0], expected)

"""CPU implementation of Edge0's 4-bit affine expert matmul."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def _as_bf16(x):
    if x is None:
        return None
    if x.dtype == torch.uint16:
        return x.view(torch.bfloat16)
    return x

def unpack_int4_affine(packed, scales, biases, group_size=64):
    if packed.dtype not in (torch.uint32, torch.int32, torch.int64):
        return packed
    # Checkpoint layout: [E, out, in/8], 8 nibbles per uint32.
    q = packed.to(torch.int64).unsqueeze(-1)
    shifts = (torch.arange(8, device=packed.device, dtype=torch.int64) * 4).view(1, 1, 1, 8)
    q = ((q >> shifts) & 0xF).reshape(packed.shape[0], packed.shape[1], -1).float()
    scales = _as_bf16(scales).float()
    biases = _as_bf16(biases).float() if biases is not None else None
    groups = q.shape[-1] // group_size
    s = scales.repeat_interleave(group_size, dim=-1)[..., :q.shape[-1]]
    # MLX affine quantization is q = round(w / scale + bias), hence
    # dequantization is (q - bias) * scale.
    if biases is None:
        return q * s
    b = biases.repeat_interleave(group_size, dim=-1)[..., :q.shape[-1]]
    return (q - b) * s

def expert_linear(x, packed, scales=None, biases=None, group_size=64):
    w = unpack_int4_affine(packed, scales, biases, group_size)
    return F.linear(x, w)

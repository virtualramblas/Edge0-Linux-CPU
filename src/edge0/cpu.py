"""Linux CPU reference primitives for Edge0-8B."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import torch
from safetensors import safe_open

@dataclass(frozen=True)
class BailingConfig:
    hidden_size: int = 1536
    num_hidden_layers: int = 24
    vocab_size: int = 157184
    num_experts: int = 128
    num_experts_per_tok: int = 8
    n_group: int = 8
    topk_group: int = 4
    routed_scaling_factor: float = 2.5
    intermediate_size: int = 512
    moe_intermediate_size: int = 512
    moe_shared_expert_intermediate_size: int = 512
    num_shared_experts: int = 1
    layer_group_size: int = 4
    first_k_dense_replace: int = 1
    short_conv_kernel_size: int = 4
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    q_lora_rank: int = 256
    kv_lora_rank: int = 512
    rope_theta: float = 6000000.0
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    kda_safe_gate: bool = True
    kda_lower_bound: float = -5.0

    @property
    def qk_head_dim(self): return self.qk_nope_head_dim + self.qk_rope_head_dim

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text())
        names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in names})

    def is_mla_layer(self, idx):
        full = self.num_hidden_layers // self.layer_group_size * self.layer_group_size
        return (idx + 1) % self.layer_group_size == 0 or idx >= full

def group_router(logits, expert_bias=None, *, top_k=8, n_group=8,
                 topk_group=4, routed_scaling=2.5):
    scores = torch.sigmoid(logits.float())
    select = scores if expert_bias is None else scores + expert_bias
    shape = select.shape
    groups = select.reshape(*shape[:-1], n_group, shape[-1] // n_group)
    group_scores = groups.topk(2, dim=-1).values.sum(dim=-1)
    group_idx = group_scores.topk(topk_group, dim=-1).indices
    keep = torch.zeros_like(groups, dtype=torch.bool)
    keep.scatter_(-2, group_idx.unsqueeze(-1).expand(*group_idx.shape, groups.shape[-1]), True)
    flat = groups.masked_fill(~keep, float("-inf")).reshape(*shape)
    idx = flat.topk(top_k, dim=-1).indices
    weights = scores.gather(-1, idx)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return idx, weights * routed_scaling

def load_safetensors_weights(model_dir):
    root = Path(model_dir)
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files in {root}")
    out = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                out[key] = f.get_tensor(key).contiguous()
    return out

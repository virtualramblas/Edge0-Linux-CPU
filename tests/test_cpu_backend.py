import torch
from edge0.cpu import BailingConfig, group_router

def test_edge0_8b_router_shape_and_normalization():
    cfg = BailingConfig()
    logits = torch.arange(cfg.num_experts, dtype=torch.float32).reshape(1, -1)
    idx, weights = group_router(logits, top_k=8, n_group=8, topk_group=4,
                                routed_scaling=2.5)
    assert idx.shape == (1, 8)
    assert torch.allclose(weights.sum(-1), torch.tensor([2.5]))

def test_router_bias_changes_selection():
    logits = torch.zeros(1, 128)
    bias = torch.zeros(128)
    bias[0] = 100
    idx, _ = group_router(logits, bias)
    assert 0 in idx[0].tolist()

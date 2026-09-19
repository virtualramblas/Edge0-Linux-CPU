import torch
from edge0.checkpoint import sanitize_bailing_weights, build_state_dict

def test_sanitize_stacks_experts_and_converts_conv():
    w = {}
    for e in range(4):
        w[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"] = torch.full((2, 3), float(e))
    w["model.layers.1.attention.q_conv1d.weight"] = torch.zeros(8, 1, 4)
    s = sanitize_bailing_weights(w, num_layers=2, num_experts=4)
    assert s["model.layers.1.mlp.experts.gate_proj.weight"].shape == (4, 2, 3)
    assert s["model.layers.1.mlp.experts.gate_proj.weight"][3, 0, 0].item() == 3
    assert s["model.layers.1.attention.q_conv1d.conv.weight"].shape == (8, 1, 4)

def test_state_key_mapping():
    w = {
        "model.embed_tokens.weight": torch.empty(2, 3),
        "model.layers.0.input_layernorm.weight": torch.empty(3),
        "lm_head.weight": torch.empty(2, 3),
    }
    s = build_state_dict(w)
    assert set(s) == {
        "word_embeddings.weight",
        "layers.0.input_layernorm.weight",
        "lm_head.weight",
    }

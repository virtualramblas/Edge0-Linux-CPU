"""Checkpoint normalization for the Edge0-8B CPU reference path."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import torch
from safetensors import safe_open

def load_raw_safetensors(model_dir):
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files in {model_dir}")
    out = {}
    for p in files:
        with safe_open(str(p), framework="pt", device="cpu") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k).contiguous()
    return out

def sanitize_bailing_weights(weights, num_layers=24, num_experts=128):
    """Mirror the upstream MLX sanitize pass without importing MLX."""
    weights = {k: v for k, v in weights.items()
               if ".mtp_" not in k and "mtp." not in k}
    out = dict(weights)
    for li in range(num_layers):
        prefix = f"model.layers.{li}.mlp.experts"
        marker = f"{prefix}.0.gate_proj.weight"
        if marker not in out and f"{prefix}.0.gate_proj.scales" not in out:
            continue
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for part in ("weight", "scales", "biases"):
                keys = [f"{prefix}.{e}.{proj}.{part}" for e in range(num_experts)]
                if keys[0] not in out:
                    continue
                out[f"{prefix}.{proj}.{part}"] = torch.stack([out.pop(k) for k in keys])
    # Upstream checkpoint conv layout is [C,1,K] (or [C,K]); PyTorch Conv1d
    # consumes [C,1,K]. Keep [C,1,K] and expand the flat form.
    for k in list(out):
        if k.endswith("_conv1d.weight"):
            w = out.pop(k)
            if w.ndim == 2:
                w = w.unsqueeze(1)
            out[k[:-len(".weight")] + ".conv.weight"] = w
    return out

def cpu_key_map():
    return {
        "model.embed_tokens.weight": "word_embeddings.weight",
        "model.word_embeddings.weight": "word_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

def map_backbone_key(k):
    if k in cpu_key_map():
        return cpu_key_map()[k]
    if k.startswith("model.layers."):
        rest = k[len("model.layers."):]
        li, tail = rest.split(".", 1)
        prefix = f"layers.{li}."
        aliases = {
            "input_layernorm.": "input_layernorm.",
            "post_attention_layernorm.": "post_attention_layernorm.",
            "attention.q_proj.": "attention.q_proj.",
            "attention.k_proj.": "attention.k_proj.",
            "attention.v_proj.": "attention.v_proj.",
            "attention.q_conv1d.": "attention.q_conv1d.",
            "attention.k_conv1d.": "attention.k_conv1d.",
            "attention.v_conv1d.": "attention.v_conv1d.",
            "attention.f_proj.": "attention.f_proj.",
            "attention.g_proj.": "attention.g_proj.",
            "attention.b_proj.": "attention.b_proj.",
            "attention.A_log": "attention.A_log",
            "attention.dt_bias": "attention.dt_bias",
            "attention.o_norm.": "attention.o_norm.",
            "attention.o_proj.": "attention.o_proj.",
            "attention.q_a_proj.": "attention.q_a_proj.",
            "attention.q_a_layernorm.": "attention.q_a_layernorm.",
            "attention.q_b_proj.": "attention.q_b_proj.",
            "attention.kv_a_proj_with_mqa.": "attention.kv_a_proj_with_mqa.",
            "attention.kv_a_layernorm.": "attention.kv_a_layernorm.",
            "attention.kv_b_proj.": "attention.kv_b_proj.",
            "attention.g_proj.": "attention.g_proj.",
            "attention.dense.": "attention.dense.",
            "mlp.gate.weight": "mlp.gate.weight",
            "mlp.experts.gate_proj.": "mlp.experts.gate_proj.",
            "mlp.experts.up_proj.": "mlp.experts.up_proj.",
            "mlp.experts.down_proj.": "mlp.experts.down_proj.",
            "mlp.shared_experts.gate_proj.": "mlp.shared_experts.gate_proj.",
            "mlp.shared_experts.up_proj.": "mlp.shared_experts.up_proj.",
            "mlp.shared_experts.down_proj.": "mlp.shared_experts.down_proj.",
        }
        for src, dst in aliases.items():
            if tail.startswith(src):
                return prefix + dst + tail[len(src):]
    return k

def build_state_dict(weights):
    return {map_backbone_key(k): v for k, v in weights.items()}

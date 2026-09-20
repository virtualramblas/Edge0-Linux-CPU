"""Checkpoint normalization for the Edge0-8B CPU reference path."""
from __future__ import annotations

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
    """
    Normalize Edge0 checkpoint weights.

    Supports both:
      1. Per-expert checkpoint tensors:
           experts.0.gate_proj.weight
           experts.1.gate_proj.weight
           ...

      2. The real Edge0-8B checkpoint layout, where experts are already
         stacked:
           experts.gate_proj.weight
           shape [num_experts, out_features, packed_features]
    """

    # Remove MTP tensors, matching the upstream sanitization behavior.
    out = {
        k: v
        for k, v in weights.items()
        if ".mtp_" not in k and "mtp." not in k
    }

    for li in range(num_layers):
        prefix = f"model.layers.{li}.mlp.experts"

        for proj in ("gate_proj", "up_proj", "down_proj"):
            for part in ("weight", "scales", "biases"):
                stacked_key = f"{prefix}.{proj}.{part}"

                # ---------------------------------------------------------
                # Case 1: already stacked -- this is the real checkpoint.
                # ---------------------------------------------------------
                if stacked_key in out:
                    tensor = out[stacked_key]

                    if tensor.ndim < 1:
                        raise ValueError(
                            f"{stacked_key}: invalid tensor shape "
                            f"{tuple(tensor.shape)}"
                        )

                    if tensor.shape[0] != num_experts:
                        raise ValueError(
                            f"{stacked_key}: expected {num_experts} experts, "
                            f"got first dimension {tensor.shape[0]}"
                        )

                    continue

                # ---------------------------------------------------------
                # Case 2: individual experts -- used by the unit tests and
                # older checkpoint layouts.
                # ---------------------------------------------------------
                keys = [
                    f"{prefix}.{e}.{proj}.{part}"
                    for e in range(num_experts)
                ]

                if not all(key in out for key in keys):
                    continue

                out[stacked_key] = torch.stack(
                    [out.pop(key) for key in keys]
                )

    # Upstream checkpoint conv layout is [C, 1, K] or [C, K].
    # PyTorch Conv1d consumes [C, 1, K].
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
    """
    Map checkpoint names into the CPU model's logical state-dict names.

    This intentionally keeps LoRA suffixes such as .lora_A and .lora_B
    unchanged because the CPU model must consume those tensors explicitly.
    """

    if k in cpu_key_map():
        return cpu_key_map()[k]

    if not k.startswith("model.layers."):
        return k

    rest = k[len("model.layers."):]
    li, tail = rest.split(".", 1)

    prefix = f"layers.{li}."

    aliases = {
        "input_layernorm.": "input_layernorm.",
        "post_attention_layernorm.": "post_attention_layernorm.",

        # KDA attention
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

        # MLA attention
        "attention.q_a_proj.": "attention.q_a_proj.",
        "attention.q_a_layernorm.": "attention.q_a_layernorm.",
        "attention.q_b_proj.": "attention.q_b_proj.",
        "attention.kv_a_proj_with_mqa.": "attention.kv_a_proj_with_mqa.",
        "attention.kv_a_layernorm.": "attention.kv_a_layernorm.",
        "attention.kv_b_proj.": "attention.kv_b_proj.",
        "attention.dense.": "attention.dense.",

        # MoE router
        "mlp.gate.weight": "mlp.gate.weight",
        "mlp.gate.expert_bias": "mlp.gate.expert_bias",

        # Routed experts
        "mlp.experts.gate_proj.": "mlp.experts.gate_proj.",
        "mlp.experts.up_proj.": "mlp.experts.up_proj.",
        "mlp.experts.down_proj.": "mlp.experts.down_proj.",

        # Shared expert
        "mlp.shared_experts.gate_proj.": "mlp.shared_experts.gate_proj.",
        "mlp.shared_experts.up_proj.": "mlp.shared_experts.up_proj.",
        "mlp.shared_experts.down_proj.": "mlp.shared_experts.down_proj.",

        # Other MLP-related tensors
        "mlp.shared_expert_gate.": "mlp.shared_expert_gate.",
    }

    for src, dst in aliases.items():
        if tail.startswith(src):
            return prefix + dst + tail[len(src):]

    return k


def build_state_dict(weights):
    return {
        map_backbone_key(k): v
        for k, v in weights.items()
    }
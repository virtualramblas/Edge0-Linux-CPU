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

    Supports:
      1. Per-expert tensors:
           experts.0.gate_proj.weight
           experts.1.gate_proj.weight
           ...

      2. Real Edge0-8B stacked tensors:
           experts.gate_proj.weight
           shape [num_experts, out_features, packed_features]
    """

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

                # Real checkpoint: already stacked.
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

                # Older/per-expert layout.
                keys = [
                    f"{prefix}.{e}.{proj}.{part}"
                    for e in range(num_experts)
                ]

                if not all(key in out for key in keys):
                    continue

                out[stacked_key] = torch.stack(
                    [out.pop(key) for key in keys]
                )

    # Upstream Conv1d layout is [C, K] or [C, 1, K].
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
    Map checkpoint names into CPU model state-dict names.

    Important:
      checkpoint: model.layers.N.attention.*
      CPU model:  layers.N.attn.*

    LoRA suffixes are preserved:
      .lora_A -> .lora_A
      .lora_B -> .lora_B
    """

    direct = cpu_key_map()
    if k in direct:
        return direct[k]

    if not k.startswith("model.layers."):
        return k

    rest = k[len("model.layers."):]
    li, tail = rest.split(".", 1)

    prefix = f"layers.{li}."

    # Layer norms.
    if tail.startswith("input_layernorm."):
        return prefix + tail

    if tail.startswith("post_attention_layernorm."):
        return prefix + tail

    # ------------------------------------------------------------
    # Attention
    #
    # Checkpoint namespace:
    #   attention.q_proj.weight
    #
    # CPU model namespace:
    #   attn.q_proj.weight
    #
    # The remainder is copied verbatim, including .lora_A/B.
    # ------------------------------------------------------------
    if tail.startswith("attention."):
        return prefix + "attn." + tail[len("attention."):]

    # ------------------------------------------------------------
    # MoE / MLP
    #
    # The CPU model keeps the checkpoint's MLP substructure, so these
    # can also be copied verbatim after the layer prefix.
    # ------------------------------------------------------------
    if tail.startswith("mlp."):
        return prefix + tail

    return k


def build_state_dict(weights):
    state = {}

    for k, v in weights.items():
        mapped = map_backbone_key(k)

        if mapped in state:
            raise KeyError(
                f"duplicate mapped checkpoint key:\n"
                f"  {mapped}\n"
                f"while processing original key:\n"
                f"  {k}"
            )

        state[mapped] = v

    return state
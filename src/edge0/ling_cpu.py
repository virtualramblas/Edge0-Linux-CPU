from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant_cpu import QuantizedLinear, QuantizedLoRALinear


# ---------------------------------------------------------------------------
# Basic normalization
# ---------------------------------------------------------------------------

def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    y = x_float * torch.rsqrt(variance + eps)
    return y.to(dtype=x.dtype) * weight.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Standard dense linear layer with an optional LoRA residual.

    y = base(x) + scaling * B(A(x))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        rank: int = 0,
        alpha: float = 1.0,
    ):
        super().__init__()

        self.base = nn.Linear(
            in_features,
            out_features,
            bias=bias,
        )

        self.rank = int(rank)
        self.alpha = float(alpha)

        if self.rank > 0:
            self.lora_A = nn.Linear(
                in_features,
                self.rank,
                bias=False,
            )
            self.lora_B = nn.Linear(
                self.rank,
                out_features,
                bias=False,
            )
            self.scaling = self.alpha / self.rank
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)

        if self.rank > 0:
            y = y + self.scaling * self.lora_B(self.lora_A(x))

        return y


# ---------------------------------------------------------------------------
# Short convolution
# ---------------------------------------------------------------------------

class ShortConv1d(nn.Module):
    """
    Depthwise causal short convolution.

    Input:
        x: [B, T, C]

    Returns:
        y: [B, T, C]
        state: [B, C, K-1]
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
    ):
        super().__init__()

        self.channels = int(channels)
        self.kernel_size = int(kernel_size)

        self.conv = nn.Conv1d(
            self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            groups=self.channels,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(
                f"ShortConv1d expects [B,T,C], got {tuple(x.shape)}"
            )

        bsz, seq_len, channels = x.shape

        if channels != self.channels:
            raise ValueError(
                f"ShortConv1d expected {self.channels} channels, "
                f"got {channels}"
            )

        k = self.kernel_size

        x_cf = x.transpose(1, 2)

        if state is None:
            if k > 1:
                prefix = torch.zeros(
                    bsz,
                    channels,
                    k - 1,
                    dtype=x.dtype,
                    device=x.device,
                )
                x_cf = torch.cat([prefix, x_cf], dim=-1)
        else:
            expected = (bsz, channels, k - 1)

            if tuple(state.shape) != expected:
                raise ValueError(
                    "invalid ShortConv state shape: "
                    f"{tuple(state.shape)}; expected {expected}"
                )

            x_cf = torch.cat([state, x_cf], dim=-1)

        y_cf = self.conv(x_cf)
        y = y_cf.transpose(1, 2)

        if k > 1:
            new_state = x_cf[..., -(k - 1):].detach()
        else:
            new_state = x_cf[..., :0].detach()

        return y, new_state


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def rope_interleave(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    """
    Apply interleaved RoPE.

    Expected x layout:

        [B, H, T, D]

    position_ids:

        [T] or [B, T]
    """

    if x.ndim != 4:
        raise ValueError(
            f"rope_interleave expects [B,H,T,D], got {tuple(x.shape)}"
        )

    bsz, _, seq_len, dim = x.shape

    if dim % 2 != 0:
        raise ValueError(
            f"RoPE dimension must be even, got {dim}"
        )

    if position_ids.ndim == 1:
        if position_ids.shape[0] != seq_len:
            raise ValueError(
                "position_ids length does not match sequence length: "
                f"{position_ids.shape[0]} != {seq_len}"
            )

        positions = position_ids.reshape(
            1, 1, seq_len, 1
        )

    elif position_ids.ndim == 2:
        if position_ids.shape[-1] != seq_len:
            raise ValueError(
                "position_ids sequence length does not match x: "
                f"{position_ids.shape[-1]} != {seq_len}"
            )

        if position_ids.shape[0] not in (1, bsz):
            raise ValueError(
                "position_ids batch dimension must be 1 or B; got "
                f"{position_ids.shape[0]} for B={bsz}"
            )

        positions = position_ids.reshape(
            position_ids.shape[0],
            1,
            seq_len,
            1,
        )

    else:
        raise ValueError(
            "position_ids must have shape [T] or [B,T], got "
            f"{tuple(position_ids.shape)}"
        )

    positions = positions.to(
        device=x.device,
        dtype=x.dtype,
    )

    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(
                0,
                dim,
                2,
                device=x.device,
                dtype=x.dtype,
            )
            / dim
        )
    )

    angles = positions * inv_freq.reshape(
        1, 1, 1, -1
    )

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos

    y = torch.empty_like(x)
    y[..., 0::2] = y_even
    y[..., 1::2] = y_odd

    return y


# ---------------------------------------------------------------------------
# KDA gate
# ---------------------------------------------------------------------------

def kda_gate(
    f: torch.Tensor,
    A: torch.Tensor,
    dt: torch.Tensor,
    lower_bound: float = -5.0,
    safe_gate: bool = True,
) -> torch.Tensor:
    """
    Compute the KDA decay gate.

    Shapes:

        f  : [B,T,H,D]
        A  : [H]
        dt : [H,D]

    Returns:

        g  : [B,T,H,D]

    The gate is negative. safe_gate clamps the lower end.
    """

    if f.ndim != 4:
        raise ValueError(
            f"kda_gate expects f=[B,T,H,D], got {tuple(f.shape)}"
        )

    _, _, n_heads, head_dim = f.shape

    if A.numel() != n_heads:
        raise ValueError(
            f"A must contain {n_heads} values, got {A.numel()}"
        )

    if tuple(dt.shape) != (n_heads, head_dim):
        raise ValueError(
            "dt must have shape "
            f"({n_heads},{head_dim}), got {tuple(dt.shape)}"
        )

    f_float = f.float()

    A = A.float().reshape(
        1, 1, n_heads, 1
    )

    dt = dt.float().reshape(
        1, 1, n_heads, head_dim
    )

    # Negative continuous-time decay.
    #
    # For A=0 and dt=0:
    #
    #     g = -softplus(f)
    #
    # so g is strictly negative.
    g = -torch.exp(A) * F.softplus(
        dt + f_float
    )

    if safe_gate:
        g = torch.clamp(
            g,
            min=float(lower_bound),
        )

    return g.to(dtype=f.dtype)


# ---------------------------------------------------------------------------
# KDA recurrence
# ---------------------------------------------------------------------------

def kda_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    KDA recurrent update.

    Inputs:

        q     [B,T,H,D]
        k     [B,T,H,D]
        v     [B,T,H,D]
        g     [B,T,H,D]
        beta  [B,T,H]

    State:

        [B,H,D,D]
    """

    if q.ndim != 4:
        raise ValueError(
            f"q must be [B,T,H,D], got {tuple(q.shape)}"
        )

    if k.shape != q.shape:
        raise ValueError(
            f"k shape {tuple(k.shape)} does not match q "
            f"{tuple(q.shape)}"
        )

    if v.shape != q.shape:
        raise ValueError(
            f"v shape {tuple(v.shape)} does not match q "
            f"{tuple(q.shape)}"
        )

    if g.shape != q.shape:
        raise ValueError(
            f"g shape {tuple(g.shape)} does not match q "
            f"{tuple(q.shape)}"
        )

    bsz, seq_len, n_heads, head_dim = q.shape

    if tuple(beta.shape) != (
        bsz,
        seq_len,
        n_heads,
    ):
        raise ValueError(
            "beta must have shape "
            f"({bsz},{seq_len},{n_heads}), got "
            f"{tuple(beta.shape)}"
        )

    expected_state = (
        bsz,
        n_heads,
        head_dim,
        head_dim,
    )

    if state is None:
        state = torch.zeros(
            *expected_state,
            device=q.device,
            dtype=q.dtype,
        )
    elif tuple(state.shape) != expected_state:
        raise ValueError(
            f"invalid KDA state shape: {tuple(state.shape)}; "
            f"expected {expected_state}"
        )

    outputs = []

    for t in range(seq_len):
        qt = q[:, t]
        kt = k[:, t]
        vt = v[:, t]
        gt = g[:, t]
        bt = beta[:, t]

        decay = torch.exp(gt)

        state = (
            decay.unsqueeze(-1) * state
        )

        update = (
            kt.unsqueeze(-1)
            * vt.unsqueeze(-2)
        )

        state = state + (
            bt.unsqueeze(-1).unsqueeze(-1)
            * update
        )

        yt = torch.matmul(
            qt.unsqueeze(-2),
            state,
        ).squeeze(-2)

        outputs.append(yt)

    if seq_len == 0:
        output = q.new_empty(
            bsz,
            0,
            n_heads,
            head_dim,
        )
    else:
        output = torch.stack(
            outputs,
            dim=1,
        )

    return output, state


# ---------------------------------------------------------------------------
# KDA attention
# ---------------------------------------------------------------------------

class KDAAttention(nn.Module):
    """
    CPU reference implementation of KDA attention.

    The constructor accepts either explicit dimensions or a BailingConfig:

        KDAAttention(config)

    This preserves the existing model.py wiring:

        KDAReference(c)
    """

    def __init__(
        self,
        hidden_size=1536,
        num_heads=16,
        head_dim=128,
        conv_kernel=4,
        q_lora_rank=16,
        lora_alpha=32.0,
        safe_gate_lower=-5.0,
        rope_theta=6_000_000.0,
        safe_gate=True,
    ):
        super().__init__()

        # BailingConfig compatibility.
        if not isinstance(hidden_size, int):
            c = hidden_size

            hidden_size = c.hidden_size
            num_heads = c.num_attention_heads
            head_dim = c.head_dim
            conv_kernel = c.short_conv_kernel_size

            # KDA's projection LoRA rank is 16 in the Edge0 checkpoint.
            q_lora_rank = 16
            lora_alpha = 32.0

            safe_gate_lower = c.kda_lower_bound
            safe_gate = c.kda_safe_gate
            rope_theta = c.rope_theta

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_size = (
            self.num_heads * self.head_dim
        )

        self.conv_kernel = int(conv_kernel)
        self.rope_theta = float(rope_theta)
        self.safe_gate = bool(safe_gate)
        self.safe_gate_lower = float(
            safe_gate_lower
        )

        # KDA q/k/v.
        self.q_proj = QuantizedLoRALinear(
            self.hidden_size,
            self.inner_size,
            rank=q_lora_rank,
            alpha=lora_alpha,
        )

        self.k_proj = QuantizedLoRALinear(
            self.hidden_size,
            self.inner_size,
            rank=q_lora_rank,
            alpha=lora_alpha,
        )

        self.v_proj = QuantizedLoRALinear(
            self.hidden_size,
            self.inner_size,
            rank=q_lora_rank,
            alpha=lora_alpha,
        )

        self.q_conv = ShortConv1d(
            self.inner_size,
            self.conv_kernel,
        )

        self.k_conv = ShortConv1d(
            self.inner_size,
            self.conv_kernel,
        )

        self.v_conv = ShortConv1d(
            self.inner_size,
            self.conv_kernel,
        )

        self.f_proj = QuantizedLoRALinear(
            self.hidden_size,
            self.inner_size,
            rank=q_lora_rank,
            alpha=lora_alpha,
        )

        self.g_proj = QuantizedLoRALinear(
            self.hidden_size,
            self.inner_size,
            rank=q_lora_rank,
            alpha=lora_alpha,
        )

        self.b_proj = QuantizedLoRALinear(
            self.hidden_size,
            self.num_heads,
            rank=q_lora_rank,
            alpha=lora_alpha,
        )

        self.o_norm = RMSNorm(
            self.head_dim
        )

        self.o_proj = QuantizedLoRALinear(
            self.inner_size,
            self.hidden_size,
            rank=q_lora_rank,
            alpha=lora_alpha,
        )

        self.A_log = nn.Parameter(
            torch.zeros(self.num_heads)
        )

        self.dt_bias = nn.Parameter(
            torch.zeros(self.inner_size)
        )

    def _reshape_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x.reshape(
            x.shape[0],
            x.shape[1],
            self.num_heads,
            self.head_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
    ):
        if x.ndim != 3:
            raise ValueError(
                "KDAAttention expects [B,T,C], got "
                f"{tuple(x.shape)}"
            )

        bsz, seq_len, _ = x.shape

        q = self._reshape_heads(
            self.q_proj(x)
        )

        k = self._reshape_heads(
            self.k_proj(x)
        )

        v = self._reshape_heads(
            self.v_proj(x)
        )

        q_flat = q.reshape(
            bsz,
            seq_len,
            self.inner_size,
        )

        k_flat = k.reshape(
            bsz,
            seq_len,
            self.inner_size,
        )

        v_flat = v.reshape(
            bsz,
            seq_len,
            self.inner_size,
        )

        q_flat, q_state = self.q_conv(
            q_flat,
            None if state is None
            else state.get("q_conv"),
        )

        k_flat, k_state = self.k_conv(
            k_flat,
            None if state is None
            else state.get("k_conv"),
        )

        v_flat, v_state = self.v_conv(
            v_flat,
            None if state is None
            else state.get("v_conv"),
        )

        q = self._reshape_heads(
            q_flat
        )

        k = self._reshape_heads(
            k_flat
        )

        v = self._reshape_heads(
            v_flat
        )

        # Per-head q/k normalization.
        q = F.normalize(
            q.float(),
            dim=-1,
        ).to(q.dtype)

        k = F.normalize(
            k.float(),
            dim=-1,
        ).to(k.dtype)

        if position_ids is not None:
            q_rope = q.permute(
                0, 2, 1, 3
            )

            k_rope = k.permute(
                0, 2, 1, 3
            )

            q_rope = rope_interleave(
                q_rope,
                position_ids,
                self.rope_theta,
            )

            k_rope = rope_interleave(
                k_rope,
                position_ids,
                self.rope_theta,
            )

            q = q_rope.permute(
                0, 2, 1, 3
            )

            k = k_rope.permute(
                0, 2, 1, 3
            )

        f = self._reshape_heads(
            self.f_proj(x)
        )

        A = -torch.exp(
            self.A_log.float()
        )

        dt_bias = self.dt_bias.reshape(
            self.num_heads,
            self.head_dim,
        )

        g = kda_gate(
            f,
            A,
            dt_bias,
            lower_bound=self.safe_gate_lower,
            safe_gate=self.safe_gate,
        )

        beta = torch.sigmoid(
            self.b_proj(x)
        )

        previous_state = (
            None
            if state is None
            else state.get("kda")
        )

        y, recurrent_state = kda_recurrence(
            q,
            k,
            v,
            g,
            beta,
            previous_state,
        )

        y = self.o_norm(y)

        gate = torch.sigmoid(
            self._reshape_heads(
                self.g_proj(x)
            )
        )

        y = y * gate

        y = y.reshape(
            bsz,
            seq_len,
            self.inner_size,
        )

        y = self.o_proj(y)

        new_state = {
            "q_conv": q_state,
            "k_conv": k_state,
            "v_conv": v_state,
            "kda": recurrent_state,
        }

        return y, new_state


# ---------------------------------------------------------------------------
# MLA reference attention
# ---------------------------------------------------------------------------

class MLAAttention(nn.Module):
    """
    CPU reference implementation of MLA.

    Accepts either explicit dimensions or BailingConfig:

        MLAAttention(config)
    """

    def __init__(
        self,
        hidden_size=1536,
        num_heads=16,
        head_dim=128,
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        lora_rank=16,
        lora_alpha=32.0,
        rope_theta=6_000_000.0,
    ):
        super().__init__()

        # BailingConfig compatibility.
        if not isinstance(hidden_size, int):
            c = hidden_size

            hidden_size = c.hidden_size
            num_heads = c.num_attention_heads
            head_dim = c.head_dim
            q_lora_rank = c.q_lora_rank
            kv_lora_rank = c.kv_lora_rank
            qk_nope_head_dim = c.qk_nope_head_dim
            qk_rope_head_dim = c.qk_rope_head_dim
            v_head_dim = c.v_head_dim
            rope_theta = c.rope_theta

            # LoRA rank/alpha for MLA projection adapters.
            lora_rank = 16
            lora_alpha = 32.0

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)

        self.q_lora_rank = int(
            q_lora_rank
        )

        self.kv_lora_rank = int(
            kv_lora_rank
        )

        self.qk_nope_head_dim = int(
            qk_nope_head_dim
        )

        self.qk_rope_head_dim = int(
            qk_rope_head_dim
        )

        self.v_head_dim = int(
            v_head_dim
        )

        self.rope_theta = float(
            rope_theta
        )

        # Compressed query.
        self.q_a_proj = QuantizedLinear(
            self.hidden_size,
            self.q_lora_rank,
        )

        self.qn = RMSNorm(
            self.q_lora_rank
        )

        self.q_b_proj = QuantizedLoRALinear(
            self.q_lora_rank,
            self.num_heads
            * (
                self.qk_nope_head_dim
                + self.qk_rope_head_dim
            ),
            rank=lora_rank,
            alpha=lora_alpha,
        )

        # Compressed KV + MQA RoPE component.
        self.kv_a_proj_with_mqa = QuantizedLinear(
            self.hidden_size,
            self.kv_lora_rank
            + self.qk_rope_head_dim,
        )

        self.kn = RMSNorm(
            self.kv_lora_rank
        )

        self.kv_b_proj = QuantizedLoRALinear(
            self.kv_lora_rank,
            self.num_heads
            * (
                self.qk_nope_head_dim
                + self.v_head_dim
            ),
            rank=lora_rank,
            alpha=lora_alpha,
        )

        self.g_proj = QuantizedLoRALinear(
            self.hidden_size,
            self.num_heads,
            rank=lora_rank,
            alpha=lora_alpha,
        )

        self.dense = QuantizedLoRALinear(
            self.num_heads
            * self.v_head_dim,
            self.hidden_size,
            rank=lora_rank,
            alpha=lora_alpha,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        state: Optional[dict] = None,
    ):
        del state

        if x.ndim != 3:
            raise ValueError(
                "MLAAttention expects [B,T,C], got "
                f"{tuple(x.shape)}"
            )

        bsz, seq_len, _ = x.shape

        q_compressed = self.q_a_proj(x)
        q_compressed = self.qn(
            q_compressed
        )

        q = self.q_b_proj(
            q_compressed
        )

        q = q.reshape(
            bsz,
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim
            + self.qk_rope_head_dim,
        )

        q_nope = q[
            ...,
            :self.qk_nope_head_dim,
        ]

        q_rope = q[
            ...,
            self.qk_nope_head_dim:,
        ]

        kv = self.kv_a_proj_with_mqa(x)

        kv_compressed = kv[
            ...,
            :self.kv_lora_rank,
        ]

        k_rope = kv[
            ...,
            self.kv_lora_rank:,
        ]

        kv_compressed = self.kn(
            kv_compressed
        )

        kv_expanded = self.kv_b_proj(
            kv_compressed
        )

        kv_expanded = kv_expanded.reshape(
            bsz,
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim
            + self.v_head_dim,
        )

        k_nope = kv_expanded[
            ...,
            :self.qk_nope_head_dim,
        ]

        v = kv_expanded[
            ...,
            self.qk_nope_head_dim:,
        ]

        # MQA RoPE component is shared across heads.
        k_rope = k_rope.unsqueeze(2).expand(
            bsz,
            seq_len,
            self.num_heads,
            self.qk_rope_head_dim,
        )

        if position_ids is not None:
            q_rope = rope_interleave(
                q_rope.permute(
                    0, 2, 1, 3
                ),
                position_ids,
                self.rope_theta,
            ).permute(
                0, 2, 1, 3
            )

            k_rope = rope_interleave(
                k_rope.permute(
                    0, 2, 1, 3
                ),
                position_ids,
                self.rope_theta,
            ).permute(
                0, 2, 1, 3
            )

        q_full = torch.cat(
            [q_nope, q_rope],
            dim=-1,
        )

        k_full = torch.cat(
            [k_nope, k_rope],
            dim=-1,
        )

        qh = q_full.permute(
            0, 2, 1, 3
        )

        kh = k_full.permute(
            0, 2, 1, 3
        )

        vh = v.permute(
            0, 2, 1, 3
        )

        scale = 1.0 / math.sqrt(
            q_full.shape[-1]
        )

        scores = torch.matmul(
            qh,
            kh.transpose(-2, -1),
        ) * scale

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(
            causal_mask,
            torch.finfo(
                scores.dtype
            ).min,
        )

        probs = torch.softmax(
            scores.float(),
            dim=-1,
        ).to(scores.dtype)

        y = torch.matmul(
            probs,
            vh,
        )

        y = y.permute(
            0, 2, 1, 3
        )

        gate = torch.sigmoid(
            self.g_proj(x)
        ).unsqueeze(-1)

        y = y * gate

        y = y.reshape(
            bsz,
            seq_len,
            self.num_heads
            * self.v_head_dim,
        )

        y = self.dense(y)

        return y, None


# ---------------------------------------------------------------------------
# Compatibility aliases
# ---------------------------------------------------------------------------

KDAReference = KDAAttention
MLAReference = MLAAttention

KDA = KDAAttention
MLA = MLAAttention
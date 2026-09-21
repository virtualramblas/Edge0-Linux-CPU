"""CPU reference implementations of Edge0 attention components.

This module intentionally contains no MLX code.

The implementation is designed for the M1 CPU/RAM path:
    - PyTorch CPU tensors
    - INT4 affine-quantized checkpoint weights
    - LoRA adapters
    - KDA recurrence
    - ShortConv1d
    - MLA reference attention
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant_cpu import (
    QuantizedLinear,
    QuantizedLoRALinear,
)


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Dense floating-point linear layer with a LoRA update.

    Used only when the checkpoint's base weight is floating point.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: int = 32,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        self.lora_A = nn.Parameter(
            torch.zeros(rank, in_features)
        )

        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)
        )

        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight)

        delta = F.linear(
            F.linear(x, self.lora_A),
            self.lora_B,
        )

        return base + self.scaling * delta


# ---------------------------------------------------------------------------
# Short convolution
# ---------------------------------------------------------------------------

class ShortConv1d(nn.Module):
    """Depthwise causal 1-D convolution used by KDA.

    The checkpoint stores the convolution weight as either:
        [channels, kernel]
    or:
        [channels, 1, kernel]

    The normalized model representation is:
        [channels, 1, kernel]
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 4,
    ):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            groups=channels,
            bias=False,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ):
        """
        x:
            [batch, seq, channels]

        state:
            optional [batch, channels, kernel_size - 1]

        Returns:
            y:
                [batch, seq, channels]

            new_state:
                [batch, channels, kernel_size - 1]
        """

        if x.ndim != 3:
            raise ValueError(
                f"ShortConv1d expects [B, T, C], got {tuple(x.shape)}"
            )

        b, t, c = x.shape

        if c != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {c}"
            )

        x_cf = x.transpose(1, 2)

        if state is None:
            state = torch.zeros(
                b,
                c,
                self.kernel_size - 1,
                dtype=x.dtype,
                device=x.device,
            )
        else:
            if state.shape != (
                b,
                c,
                self.kernel_size - 1,
            ):
                raise ValueError(
                    "invalid ShortConv state shape: "
                    f"{tuple(state.shape)}"
                )

        padded = torch.cat(
            [state, x_cf],
            dim=-1,
        )

        y = self.conv(padded)

        new_state = padded[
            :,
            :,
            -(self.kernel_size - 1):,
        ].detach()

        return y.transpose(1, 2), new_state


# ---------------------------------------------------------------------------
# KDA helpers
# ---------------------------------------------------------------------------

def kda_gate(
    x: torch.Tensor,
    lower: float = -5.0,
) -> torch.Tensor:
    """Stable bounded KDA gate.

    The safe lower bound prevents the exponential gate parameterization
    from becoming numerically pathological.
    """

    return torch.sigmoid(
        torch.clamp(x, min=lower)
    )


def kda_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    *,
    state: torch.Tensor | None = None,
):
    """Reference KDA recurrence.

    Shapes:
        q, k, v:
            [B, T, H, D]

        decay:
            [B, T, H, D]

        state:
            [B, H, D, D]

    Returns:
        output:
            [B, T, H, D]

        final_state:
            [B, H, D, D]
    """

    if q.ndim != 4:
        raise ValueError(
            f"q must be [B,T,H,D], got {tuple(q.shape)}"
        )

    if k.shape != q.shape:
        raise ValueError(
            f"k shape {tuple(k.shape)} != q shape {tuple(q.shape)}"
        )

    if v.shape != q.shape:
        raise ValueError(
            f"v shape {tuple(v.shape)} != q shape {tuple(q.shape)}"
        )

    if decay.shape != q.shape:
        raise ValueError(
            "decay shape "
            f"{tuple(decay.shape)} != q shape {tuple(q.shape)}"
        )

    b, t, h, d = q.shape

    if state is None:
        state = torch.zeros(
            b,
            h,
            d,
            d,
            dtype=q.dtype,
            device=q.device,
        )

    if state.shape != (b, h, d, d):
        raise ValueError(
            f"invalid KDA state shape: {tuple(state.shape)}"
        )

    outputs = []

    for i in range(t):
        qi = q[:, i]
        ki = k[:, i]
        vi = v[:, i]
        di = decay[:, i]

        # State decay is applied independently for each feature dimension.
        state = state * di.unsqueeze(-1)

        # Outer product k^T v:
        # [B,H,D,1] @ [B,H,1,D]
        state = state + (
            ki.unsqueeze(-1)
            * vi.unsqueeze(-2)
        )

        yi = torch.einsum(
            "bhd,bhde->bhe",
            qi,
            state,
        )

        outputs.append(yi)

    return torch.stack(outputs, dim=1), state


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def rope_interleave(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float = 6_000_000.0,
) -> torch.Tensor:
    """Apply interleaved RoPE.

    The final dimension is interpreted as pairs:
        (x0, x1), (x2, x3), ...

    This is the corrected even/odd implementation.
    """

    if x.shape[-1] % 2 != 0:
        raise ValueError(
            "RoPE dimension must be even"
        )

    dim = x.shape[-1]

    device = x.device
    dtype = x.dtype

    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(
                0,
                dim,
                2,
                device=device,
                dtype=torch.float32,
            )
            / dim
        )
    )

    positions = position_ids.to(
        device=device,
        dtype=torch.float32,
    )

    freqs = torch.einsum(
        "...t,d->...td",
        positions,
        inv_freq,
    )

    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)

    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos

    y = torch.empty_like(x)
    y[..., 0::2] = y_even
    y[..., 1::2] = y_odd

    return y


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------

def causal_mask(
    q_len: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Construct an additive causal attention mask."""

    q_positions = torch.arange(
        q_len,
        device=device,
    )

    kv_positions = torch.arange(
        kv_len,
        device=device,
    )

    # For cached decoding, the query positions correspond to the final
    # q_len positions in the KV sequence.
    q_positions = q_positions + (
        kv_len - q_len
    )

    allowed = (
        kv_positions.unsqueeze(0)
        <= q_positions.unsqueeze(1)
    )

    mask = torch.zeros(
        q_len,
        kv_len,
        dtype=dtype,
        device=device,
    )

    mask = mask.masked_fill(
        ~allowed,
        torch.finfo(dtype).min,
    )

    return mask


# ---------------------------------------------------------------------------
# KDA
# ---------------------------------------------------------------------------

class KDAReference(nn.Module):
    """Reference CPU implementation of Edge0 KDA attention."""

    def __init__(
        self,
        hidden_size: int = 1536,
        num_heads: int = 16,
        head_dim: int = 128,
        conv_kernel: int = 4,
        rope_theta: float = 6_000_000.0,
        safe_gate_lower: float = -5.0,
    ):
        super().__init__()

        h = hidden_size
        nh = num_heads
        hd = head_dim
        p = nh * hd

        self.hidden_size = h
        self.num_heads = nh
        self.head_dim = hd
        self.proj_dim = p
        self.rope_theta = rope_theta
        self.safe_gate_lower = safe_gate_lower

        # All seven KDA projections are INT4 in the real checkpoint.
        # All seven also have LoRA A/B tensors in the checkpoint.
        self.q_proj = QuantizedLoRALinear(
            h,
            p,
            rank=16,
            alpha=32,
        )

        self.k_proj = QuantizedLoRALinear(
            h,
            p,
            rank=16,
            alpha=32,
        )

        self.v_proj = QuantizedLoRALinear(
            h,
            p,
            rank=16,
            alpha=32,
        )

        self.q_conv1d = ShortConv1d(
            p,
            kernel_size=conv_kernel,
        )

        self.k_conv1d = ShortConv1d(
            p,
            kernel_size=conv_kernel,
        )

        self.v_conv1d = ShortConv1d(
            p,
            kernel_size=conv_kernel,
        )

        self.f_proj = QuantizedLoRALinear(
            h,
            p,
            rank=16,
            alpha=32,
        )

        self.g_proj = QuantizedLoRALinear(
            h,
            p,
            rank=16,
            alpha=32,
        )

        self.b_proj = QuantizedLoRALinear(
            h,
            nh,
            rank=16,
            alpha=32,
        )

        self.A_log = nn.Parameter(
            torch.zeros(nh)
        )

        self.dt_bias = nn.Parameter(
            torch.zeros(nh, hd)
        )

        self.o_norm = nn.RMSNorm(
            hd,
            eps=1e-6,
        )

        self.o_proj = QuantizedLoRALinear(
            p,
            h,
            rank=16,
            alpha=32,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        state: dict | None = None,
    ):
        if x.ndim != 3:
            raise ValueError(
                f"KDA expects [B,T,H], got {tuple(x.shape)}"
            )

        b, t, h = x.shape

        if h != self.hidden_size:
            raise ValueError(
                f"expected hidden size {self.hidden_size}, got {h}"
            )

        if position_ids is None:
            position_ids = torch.arange(
                t,
                device=x.device,
            ).unsqueeze(0).expand(b, -1)

        if state is None:
            state = {}

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q, q_conv_state = self.q_conv1d(
            q,
            state.get("q_conv"),
        )

        k, k_conv_state = self.k_conv1d(
            k,
            state.get("k_conv"),
        )

        v, v_conv_state = self.v_conv1d(
            v,
            state.get("v_conv"),
        )

        q = q.reshape(
            b,
            t,
            self.num_heads,
            self.head_dim,
        )

        k = k.reshape(
            b,
            t,
            self.num_heads,
            self.head_dim,
        )

        v = v.reshape(
            b,
            t,
            self.num_heads,
            self.head_dim,
        )

        # Normalize q/k independently.
        q = F.normalize(
            q,
            dim=-1,
        )

        k = F.normalize(
            k,
            dim=-1,
        )

        # KDA's RoPE applies to the rotary feature dimensions.
        q = rope_interleave(
            q,
            position_ids,
            theta=self.rope_theta,
        )

        k = rope_interleave(
            k,
            position_ids,
            theta=self.rope_theta,
        )

        # f controls the feature-wise decay.
        f = self.f_proj(x)

        f = f.reshape(
            b,
            t,
            self.num_heads,
            self.head_dim,
        )

        # dt_bias is stored as [heads, head_dim].
        f = f.float()

        if (
            self.dt_bias.numel()
            != self.num_heads * self.head_dim
        ):
            raise ValueError(
                "dt_bias has wrong number of elements: "
                f"{self.dt_bias.numel()} != "
                f"{self.num_heads * self.head_dim}"
            )

        dt_bias = self.dt_bias.float().reshape(
            self.num_heads,
            self.head_dim,
        )

        f = f + dt_bias.reshape(
            1,
            1,
            self.num_heads,
            self.head_dim,
        )

        f = kda_gate(
            f,
            lower=self.safe_gate_lower,
        )

        # A_log supplies an additional learned decay scale.
        a = -torch.exp(
            self.A_log.float()
        )

        decay = torch.exp(
            a.reshape(1, 1, self.num_heads, 1)
            * f
        )

        # b_proj produces a per-head modulation.
        b_gate = self.b_proj(x)

        b_gate = torch.sigmoid(
            b_gate
        ).reshape(
            b,
            t,
            self.num_heads,
            1,
        )

        v = v * b_gate

        y, recurrent_state = kda_recurrence(
            q,
            k,
            v,
            decay,
            state=state.get("recurrent"),
        )

        # Output normalization operates per head.
        y = self.o_norm(y)

        # g_proj provides the output gate.
        g = self.g_proj(x)

        g = g.reshape(
            b,
            t,
            self.num_heads,
            self.head_dim,
        )

        g = torch.sigmoid(g)

        y = y * g

        y = y.reshape(
            b,
            t,
            self.proj_dim,
        )

        y = self.o_proj(y)

        new_state = {
            "q_conv": q_conv_state,
            "k_conv": k_conv_state,
            "v_conv": v_conv_state,
            "recurrent": recurrent_state,
        }

        return y, new_state


# ---------------------------------------------------------------------------
# MLA
# ---------------------------------------------------------------------------

class MLAReference(nn.Module):
    """Reference CPU implementation of Edge0 MLA attention."""

    def __init__(
        self,
        hidden_size: int = 1536,
        num_heads: int = 16,
        head_dim: int = 128,
        rope_theta: float = 6_000_000.0,
    ):
        super().__init__()

        h = hidden_size
        nh = num_heads
        hd = head_dim

        self.hidden_size = h
        self.num_heads = nh
        self.head_dim = hd
        self.rope_theta = rope_theta

        # ---------------------------------------------------------------
        # Query compression path.
        #
        # q_a_proj is quantized but has no LoRA.
        # q_b_proj is quantized and has LoRA.
        # ---------------------------------------------------------------

        self.q_a_proj = QuantizedLinear(
            h,
            256,
        )

        self.q_a_layernorm = nn.RMSNorm(
            256,
            eps=1e-6,
        )

        self.q_b_proj = QuantizedLoRALinear(
            256,
            3072,
            rank=16,
            alpha=32,
        )

        # ---------------------------------------------------------------
        # KV compression path.
        # ---------------------------------------------------------------

        self.kv_a_proj_with_mqa = QuantizedLinear(
            h,
            576,
        )

        self.kv_a_layernorm = nn.RMSNorm(
            512,
            eps=1e-6,
        )

        self.kv_b_proj = QuantizedLoRALinear(
            512,
            4096,
            rank=16,
            alpha=32,
        )

        # ---------------------------------------------------------------
        # Output gate.
        # ---------------------------------------------------------------

        self.g_proj = QuantizedLoRALinear(
            h,
            16,
            rank=16,
            alpha=32,
        )

        # ---------------------------------------------------------------
        # Output projection.
        # ---------------------------------------------------------------

        self.dense = QuantizedLoRALinear(
            2048,
            h,
            rank=16,
            alpha=32,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        state: dict | None = None,
    ):
        if x.ndim != 3:
            raise ValueError(
                f"MLA expects [B,T,H], got {tuple(x.shape)}"
            )

        b, t, h = x.shape

        if h != self.hidden_size:
            raise ValueError(
                f"expected hidden size {self.hidden_size}, got {h}"
            )

        if position_ids is None:
            position_ids = torch.arange(
                t,
                device=x.device,
            ).unsqueeze(0).expand(b, -1)

        # ---------------------------------------------------------------
        # Query.
        # ---------------------------------------------------------------

        q_latent = self.q_a_proj(x)

        q_latent = self.q_a_layernorm(
            q_latent
        )

        q = self.q_b_proj(q_latent)

        q = q.reshape(
            b,
            t,
            self.num_heads,
            192,
        )

        # 128 non-RoPE dimensions + 64 RoPE dimensions.
        q_nope = q[..., :128]
        q_rope = q[..., 128:]

        q_rope = rope_interleave(
            q_rope,
            position_ids,
            theta=self.rope_theta,
        )

        # ---------------------------------------------------------------
        # KV path.
        # ---------------------------------------------------------------

        kv_latent = self.kv_a_proj_with_mqa(x)

        kv_content = kv_latent[..., :512]
        k_rope = kv_latent[..., 512:]

        kv_content = self.kv_a_layernorm(
            kv_content
        )

        kv = self.kv_b_proj(kv_content)

        kv = kv.reshape(
            b,
            t,
            self.num_heads,
            256,
        )

        k_nope = kv[..., :128]
        v = kv[..., 128:]

        k_rope = k_rope.reshape(
            b,
            t,
            1,
            64,
        )

        k_rope = rope_interleave(
            k_rope,
            position_ids,
            theta=self.rope_theta,
        )

        # Broadcast MQA rotary key to all heads.
        k_rope = k_rope.expand(
            -1,
            -1,
            self.num_heads,
            -1,
        )

        # ---------------------------------------------------------------
        # Attention.
        # ---------------------------------------------------------------

        q = torch.cat(
            [q_nope, q_rope],
            dim=-1,
        )

        k = torch.cat(
            [k_nope, k_rope],
            dim=-1,
        )

        q = q.transpose(
            1,
            2,
        )

        k = k.transpose(
            1,
            2,
        )

        v = v.transpose(
            1,
            2,
        )

        scale = 1.0 / math.sqrt(
            q.shape[-1]
        )

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) * scale

        mask = causal_mask(
            t,
            t,
            x.device,
            scores.dtype,
        )

        scores = scores + mask

        probs = torch.softmax(
            scores,
            dim=-1,
        )

        y = torch.matmul(
            probs,
            v,
        )

        y = y.transpose(
            1,
            2,
        )

        # ---------------------------------------------------------------
        # Output gate.
        # ---------------------------------------------------------------

        gate = self.g_proj(x)

        gate = torch.sigmoid(
            gate
        )

        y = y.reshape(
            b,
            t,
            2048,
        )

        # Expand the 16-dimensional gate over each head's 128 dimensions.
        gate = gate.unsqueeze(-1).expand(
            -1,
            -1,
            self.num_heads,
            self.head_dim,
        ).reshape(
            b,
            t,
            2048,
        )

        y = y * gate

        y = self.dense(y)

        return y, state
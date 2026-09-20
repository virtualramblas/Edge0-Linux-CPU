"""Faithful PyTorch CPU primitives for the Ling 3.0/Bailing 8B backbone.

This module deliberately mirrors the upstream MLX implementation at the
algorithmic level. It is used by the RAM-resident M1 path; no MLX or SSD
streaming is involved.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class ShortConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              groups=channels, bias=False)

    def forward(self, x, state=None):
        # x: [B,T,C], checkpoint convention is depthwise [C,1,K] or [C,K,1]
        b, t, c = x.shape
        if state is None:
            state = x.new_zeros((b, self.kernel_size - 1, c))
        inp = torch.cat([state, x], dim=1)
        y = F.silu(self.conv(inp.transpose(1, 2)).transpose(1, 2))
        new_state = inp[:, -(self.kernel_size - 1):, :] if self.kernel_size > 1 else inp[:, :0]
        return y, new_state

def kda_gate(f, A_log, dt_bias, lower_bound=-5.0, safe_gate=True):
    f = f.float()

    if dt_bias.numel() != f.shape[-2] * f.shape[-1]:
        raise ValueError(
            f"KDA dt_bias shape {tuple(dt_bias.shape)} is incompatible "
            f"with f shape {tuple(f.shape)}"
        )

    dt_bias = dt_bias.float().reshape(f.shape[-2], f.shape[-1])
    f = f + dt_bias.reshape(1, 1, f.shape[-2], f.shape[-1])
    a = torch.exp(A_log.float()).reshape(1, 1, -1, 1)
    if safe_gate:
        return lower_bound * torch.sigmoid(a * f)
    return -a * F.softplus(f)

def kda_recurrence(q, k, v, g_log, beta, state=None):
    # Exact delta-rule structure used by the MLX fallback ops.
    # q/k/v: [B,T,H,D], g_log: [B,T,H,D], beta: [B,T,H].
    b, t, h, d = q.shape
    S = q.new_zeros((b, h, d, d), dtype=torch.float32) if state is None else state.float()
    outs = []
    for i in range(t):
        decay = torch.exp(g_log[:, i].float()).unsqueeze(-1)
        S = S * decay
        kt = k[:, i].float()
        vt = v[:, i].float()
        qt = q[:, i].float()
        pred = torch.einsum("bhd,bhde->bhe", kt, S)
        err = vt - pred
        S = S + beta[:, i].float().unsqueeze(-1).unsqueeze(-1) * torch.einsum("bhd,bhe->bhde", kt, err)
        outs.append(torch.einsum("bhde,bhd->bhe", S, qt))
    return torch.stack(outs, dim=1), S

def rope_interleave(x, positions, rope_theta):
    """
    Apply interleaved rotary position embedding.

    Expected x shape:
        [batch, heads, sequence, rotary_dim]

    positions:
        [sequence]

    rotary_dim must be even.

    Dimensions are paired as:
        (0, 1), (2, 3), ...

    so each pair is rotated as:
        x0' = x0 * cos - x1 * sin
        x1' = x0 * sin + x1 * cos
    """
    rotary_dim = x.shape[-1]

    if rotary_dim % 2 != 0:
        raise ValueError(
            f"RoPE dimension must be even, got {rotary_dim}"
        )

    if positions.ndim != 1:
        raise ValueError(
            f"positions must be 1-D, got shape {tuple(positions.shape)}"
        )

    half_dim = rotary_dim // 2

    # Match the usual interleaved RoPE frequency layout.
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(
                0,
                half_dim,
                device=x.device,
                dtype=torch.float32,
            )
            / half_dim
        )
    )

    angles = positions.to(torch.float32)[:, None] * inv_freq[None, :]

    cos = torch.cos(angles).to(dtype=x.dtype)
    sin = torch.sin(angles).to(dtype=x.dtype)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # [sequence, half_dim] -> broadcast over batch/head.
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos

    y = torch.empty_like(x)
    y[..., 0::2] = y_even
    y[..., 1::2] = y_odd

    return y

def causal_mask(query_len, key_len, device):
    qpos = torch.arange(key_len - query_len, key_len, device=device)
    kpos = torch.arange(key_len, device=device)
    return kpos.unsqueeze(0) <= qpos.unsqueeze(1)

class KDAReference(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, p = cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim
        self.nh, self.d, self.p = cfg.num_attention_heads, cfg.head_dim, p
        self.q_proj = nn.Linear(h, p, bias=False)
        self.k_proj = nn.Linear(h, p, bias=False)
        self.v_proj = nn.Linear(h, p, bias=False)
        self.q_conv1d = ShortConv1d(p, cfg.short_conv_kernel_size)
        self.k_conv1d = ShortConv1d(p, cfg.short_conv_kernel_size)
        self.v_conv1d = ShortConv1d(p, cfg.short_conv_kernel_size)
        self.f_proj = nn.Linear(h, p, bias=False)
        self.g_proj = nn.Linear(h, p, bias=False)
        self.b_proj = nn.Linear(h, self.nh, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.nh))
        self.dt_bias = nn.Parameter(torch.zeros(p))
        self.o_norm = nn.RMSNorm(self.d, eps=cfg.rms_norm_eps, elementwise_affine=True)
        self.o_proj = nn.Linear(p, h, bias=False)
        self.lower_bound = cfg.kda_lower_bound
        self.safe_gate = cfg.kda_safe_gate

    def forward(self, x, state=None):
        state = state or {}
        q, qs = self.q_conv1d(self.q_proj(x), state.get("q"))
        k, ks = self.k_conv1d(self.k_proj(x), state.get("k"))
        v, vs = self.v_conv1d(self.v_proj(x), state.get("v"))
        b, t, _ = q.shape
        q = q.reshape(b, t, self.nh, self.d)
        k = k.reshape(b, t, self.nh, self.d)
        v = v.reshape(b, t, self.nh, self.d)
        qf, kf = q.float(), k.float()
        q = qf / (qf.norm(dim=-1, keepdim=True) + 1e-6) * (self.d ** -0.5)
        k = kf / (kf.norm(dim=-1, keepdim=True) + 1e-6)
        f = self.f_proj(x).reshape(b, t, self.nh, self.d)
        glog = kda_gate(f, self.A_log, self.dt_bias,
                        self.lower_bound, self.safe_gate)
        beta = torch.sigmoid(self.b_proj(x).float())
        out, ssm = kda_recurrence(q, k, v, glog, beta, state.get("S"))
        gate = torch.sigmoid(self.g_proj(x).reshape(b, t, self.nh, self.d))
        out = self.o_norm(out.to(x.dtype)) * gate
        return self.o_proj(out.reshape(b, t, -1)), {"q": qs, "k": ks, "v": vs, "S": ssm}

class MLAReference(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, nh = cfg.hidden_size, cfg.num_attention_heads
        qd = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
        self.nh, self.qno, self.qrope, self.qd, self.vd = nh, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, qd, cfg.v_head_dim
        self.theta = cfg.rope_theta
        self.qa = nn.Linear(h, cfg.q_lora_rank, bias=False)
        self.qn = nn.RMSNorm(cfg.q_lora_rank, eps=cfg.rms_norm_eps)
        self.qb = nn.Linear(cfg.q_lora_rank, nh * qd, bias=False)
        self.kva = nn.Linear(h, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False)
        self.kn = nn.RMSNorm(cfg.kv_lora_rank, eps=cfg.rms_norm_eps)
        self.kvb = nn.Linear(cfg.kv_lora_rank, nh * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False)
        self.g = nn.Linear(h, nh, bias=False)
        self.dense = nn.Linear(nh * cfg.v_head_dim, h, bias=False)

    def forward(self, x, state=None):
        b, t, _ = x.shape
        q = self.qb(self.qn(self.qa(x))).reshape(b, t, self.nh, self.qd).transpose(1, 2)
        qno, qrope = q.split([self.qno, self.qrope], dim=-1)
        z = self.kva(x)
        lat, krope = z.split([self.kn.normalized_shape[0], self.qrope], dim=-1)
        lat = self.kn(lat)
        kv = self.kvb(lat).reshape(b, t, self.nh, self.qno + self.vd).transpose(1, 2)
        kn, v = kv.split([self.qno, self.vd], dim=-1)
        past = 0 if state is None else state["k"].shape[2]
        pos = torch.arange(t, device=x.device) + past
        qrope = rope_interleave(qrope, pos, self.theta)
        krope = rope_interleave(krope.reshape(b, 1, t, self.qrope), pos, self.theta).expand(b, self.nh, t, self.qrope)
        q = torch.cat([qno, qrope], dim=-1)
        k = torch.cat([kn, krope], dim=-1)
        if state is not None:
            k = torch.cat([state["k"], k], dim=2)
            v = torch.cat([state["v"], v], dim=2)
        att = torch.matmul(q, k.transpose(-2, -1)) * (self.qd ** -0.5)
        mask = causal_mask(t, k.shape[2], x.device)
        att = att.masked_fill(~mask[None, None], torch.finfo(att.dtype).min)
        p = torch.softmax(att, dim=-1)
        out = torch.matmul(p, v).transpose(1, 2).reshape(b, t, self.nh, self.vd)
        out = out * torch.sigmoid(self.g(x)).unsqueeze(-1)
        return self.dense(out.reshape(b, t, -1)), {"k": k.detach(), "v": v.detach()}
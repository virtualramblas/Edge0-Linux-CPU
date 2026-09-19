import torch
from edge0.cpu import BailingConfig, group_router
from edge0.ling_cpu import ShortConv1d, rope_interleave, kda_gate, kda_recurrence

def test_ling_layer_schedule():
    c = BailingConfig()
    assert [c.is_mla_layer(i) for i in range(8)] == [False, False, False, True, False, False, False, True]

def test_rope_position_zero_is_identity():
    x = torch.randn(2, 16, 3, 64)
    p = torch.zeros(3, dtype=torch.long)
    y = rope_interleave(x, p, 6_000_000.0)
    assert torch.allclose(y, x, atol=1e-6, rtol=1e-6)

def test_shortconv_prefill_then_decode_matches_single_pass():
    torch.manual_seed(0)
    m = ShortConv1d(8, 4)
    x = torch.randn(1, 5, 8)
    full, _ = m(x)
    a, state = m(x[:, :3])
    b, _ = m(x[:, 3:], state)
    assert torch.allclose(full[:, 3:], b, atol=1e-6, rtol=1e-6)

def test_kda_gate_bounds():
    f = torch.randn(2, 3, 16, 128)
    A = torch.zeros(16)
    dt = torch.zeros(16, 128)
    g = kda_gate(f, A, dt, lower_bound=-5.0, safe_gate=True)
    assert torch.all(g < 0)
    assert torch.all(g > -5.0)

def test_kda_recurrence_chunking_is_exact():
    torch.manual_seed(1)
    q = torch.randn(1, 6, 2, 4)
    k = torch.randn(1, 6, 2, 4)
    v = torch.randn(1, 6, 2, 4)
    g = torch.randn(1, 6, 2, 4) * -0.2
    beta = torch.sigmoid(torch.randn(1, 6, 2))
    whole, s1 = kda_recurrence(q, k, v, g, beta)
    first, s = kda_recurrence(q[:, :3], k[:, :3], v[:, :3], g[:, :3], beta[:, :3])
    second, s2 = kda_recurrence(q[:, 3:], k[:, 3:], v[:, 3:], g[:, 3:], beta[:, 3:], s)
    assert torch.allclose(whole, torch.cat([first, second], 1), atol=1e-6, rtol=1e-6)
    assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-6)

"""RAM-resident Edge0-8B model assembly."""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from .cpu import BailingConfig, group_router
from .ling_cpu import KDAReference, MLAReference, LoRALinear
from .checkpoint import sanitize_bailing_weights, build_state_dict
from .quant_cpu import QuantizedLinear, QuantizedLoRALinear

class RMSNorm(nn.Module):
    def __init__(self,n,eps=1e-6):
        super().__init__(); self.weight=nn.Parameter(torch.ones(n)); self.eps=eps
    def forward(self,x):
        y=x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+self.eps)
        return y.to(x.dtype)*self.weight

class DenseMLP(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        use_lora=False,
        use_quantized=False,
    ):
        super().__init__()

        if use_quantized and use_lora:
            Linear = QuantizedLoRALinear
        elif use_quantized:
            Linear = QuantizedLinear
        elif use_lora:
            Linear = LoRALinear
        else:
            Linear = nn.Linear

        if use_lora and use_quantized:
            kwargs = {
                "rank": 16,
                "alpha": 32,
            }
        else:
            kwargs = {}

        self.gate_proj = Linear(
            hidden_size,
            intermediate_size,
            **kwargs,
        )

        self.up_proj = Linear(
            hidden_size,
            intermediate_size,
            **kwargs,
        )

        self.down_proj = Linear(
            intermediate_size,
            hidden_size,
            **kwargs,
        )

        self.mlp = DenseMLP(
          hidden_size,
          intermediate_size,
          use_lora=True,
          use_quantized=True,
      )

    def forward(self, x):
        return self.down_proj(
            F.silu(self.gate_proj(x))
            * self.up_proj(x)
        )

class ExpertStore(nn.Module):
    def __init__(self,c,layer):
        super().__init__(); self.c=c; self.layer=layer
        self.weights=nn.ParameterDict(); self.scales={}; self.biases={}
    def load(self,tensors):
        prefix=f"model.layers.{self.layer}.mlp.experts"
        for p in ("gate_proj","up_proj","down_proj"):
            w=tensors.get(f"{prefix}.{p}.weight")
            if w is not None: self.weights[p]=nn.Parameter(w,requires_grad=False)
            for n,store in (("scales",self.scales),("biases",self.biases)):
                z=tensors.get(f"{prefix}.{p}.{n}")
                if z is not None: store[p]=z
    def _weight(self,p,e):
        w=self.weights[p][e]
        if w.dtype not in (torch.uint32,torch.int32,torch.int64): return w
        shifts=(torch.arange(8,dtype=torch.int64).view(1,1,8)*4)
        q=((w.to(torch.int64).unsqueeze(-1)>>shifts)&15).reshape(w.shape[0],-1).float()
        s=self.scales[p][e].float().repeat_interleave(64,-1)[...,:q.shape[-1]]
        b=0 if p not in self.biases else self.biases[p][e].float().repeat_interleave(64,-1)[...,:q.shape[-1]]
        return (q-b)*s
    def forward(self,x,idx,weights):
        flat=x.reshape(-1,x.shape[-1]); ids=idx.reshape(-1,idx.shape[-1]); ws=weights.reshape(-1,weights.shape[-1]); out=torch.zeros_like(flat)
        for e in range(self.c.num_experts):
            pos=(ids==e).nonzero(as_tuple=False)
            if pos.numel()==0: continue
            tok=pos[:,0]; slot=pos[:,1]; xx=flat[tok]
            gate=F.silu(F.linear(xx,self._weight("gate_proj",e)))
            up=F.linear(xx,self._weight("up_proj",e))
            y=F.linear(gate*up,self._weight("down_proj",e))
            out.index_add_(0,tok,y*ws[tok,slot].to(y.dtype).unsqueeze(-1))
        return out.reshape_as(x)

class Router(nn.Module):
    def __init__(self,c):
        super().__init__(); self.weight=nn.Parameter(torch.empty(c.num_experts,c.hidden_size)); self.bias=nn.Parameter(torch.zeros(c.num_experts),requires_grad=False)
    def forward(self,x): return group_router(F.linear(x,self.weight),self.bias)

class Layer(nn.Module):
    def __init__(self,c,i,tensors):
        super().__init__(); self.is_mla=c.is_mla_layer(i); self.attn=MLAReference(c) if self.is_mla else KDAReference(c)
        self.input_layernorm=RMSNorm(c.hidden_size,c.rms_norm_eps); self.post_attention_layernorm=RMSNorm(c.hidden_size,c.rms_norm_eps)
        self.experts=None if i<c.first_k_dense_replace else ExpertStore(c,i)
        if self.experts is not None: self.experts.load(tensors)
        self.router=None if self.experts is None else Router(c)
        self.mlp=DenseMLP(c.hidden_size,c.intermediate_size) if self.experts is None else None
        self.shared=DenseMLP(c.hidden_size,c.moe_shared_expert_intermediate_size)
    def forward(self,x,cache=None):
        a,cs=self.attn(self.input_layernorm(x),cache); h=x+a; m=self.post_attention_layernorm(h)
        y=self.mlp(m) if self.experts is None else self.experts(m,*self.router(m))+self.shared(m)
        return h+y,cs

class BailingCPUModel(nn.Module):
    def __init__(self,c,tensors):
        super().__init__(); self.c=c; self.word_embeddings=nn.Embedding(c.vocab_size,c.hidden_size)
        self.layers=nn.ModuleList([Layer(c,i,tensors) for i in range(c.num_hidden_layers)])
        self.norm=RMSNorm(c.hidden_size,c.rms_norm_eps); self.lm_head=nn.Linear(c.hidden_size,c.vocab_size,bias=False)
    def forward(self,input_ids,past=None):
        h=self.word_embeddings(input_ids); new=[]
        for i,l in enumerate(self.layers):
            h,cs=l(h,None if past is None else past[i]); new.append(cs)
        return self.lm_head(self.norm(h)),new

def load_ram_model(model_dir):
    root = Path(model_dir)
    cfg = BailingConfig.from_json(root / "config.json")

    raw = {}
    for p in sorted(root.glob("*.safetensors")):
        with safe_open(str(p), framework="pt", device="cpu") as f:
            for k in f.keys():
                raw[k] = f.get_tensor(k).contiguous()

    tensors = sanitize_bailing_weights(
        raw,
        cfg.num_hidden_layers,
        cfg.num_experts,
    )

    model = BailingCPUModel(cfg, tensors)

    state = build_state_dict(tensors)
    own = dict(model.named_parameters())

    loaded = []
    skipped = []

    for k, v in state.items():
        if k in own:
            if own[k].shape == v.shape:
                loaded.append((k, tuple(v.shape)))
            else:
                skipped.append(
                    (
                        k,
                        tuple(v.shape),
                        "shape mismatch",
                        tuple(own[k].shape),
                    )
                )
        else:
            skipped.append(
                (
                    k,
                    tuple(v.shape),
                    "missing model parameter",
                    None,
                )
            )

    print("\n=== LOAD DIAGNOSTIC ===")
    print("checkpoint entries:", len(state))
    print("model parameters:", len(own))
    print("shape/name matches:", len(loaded))
    print("skipped:", len(skipped))

    from collections import Counter

    def skipped_component(key):
        parts = key.split(".")

        if parts[0] == "layers":
            # layers.0.attn.q_proj.lora_A -> attn.q_proj.lora_A
            return ".".join(parts[2:])

        return key


    print("\n=== SKIPPED COMPONENTS ===")

    counts = Counter(skipped_component(x[0]) for x in skipped)

    for k, n in counts.most_common():
        print(f"{n:4d}  {k}")


    print("\n=== FIRST 150 SKIPPED ===")

    for x in skipped[:150]:
        print(x)



    print("\n=== MATCHES ===")
    for item in loaded:
        print(item)

    print("\n=== FIRST 100 SKIPPED ===")
    for item in skipped[:100]:
        print(item)

    return model, cfg, loaded, skipped
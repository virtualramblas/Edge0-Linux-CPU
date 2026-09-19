"""Pure PyTorch CPU Ling/Bailing reference for Edge0-8B.

The implementation is intentionally eager and RAM-resident.  It is a correctness
reference for M1, not the SSD streaming path.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from .cpu import group_router

class RMSNorm(nn.Module):
    def __init__(self, n, eps=1e-6):
        super().__init__(); self.weight=nn.Parameter(torch.ones(n)); self.eps=eps
    def forward(self,x):
        y=x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+self.eps)
        return y.to(x.dtype)*self.weight

class Linear(nn.Module):
    def __init__(self, out_features, in_features, bias=False):
        super().__init__(); self.weight=nn.Parameter(torch.empty(out_features,in_features)); self.bias=nn.Parameter(torch.zeros(out_features)) if bias else None
    def forward(self,x): return F.linear(x,self.weight,self.bias)

class QuantExpertStore:
    def __init__(self, n=128):
        self.n=n; self.weight={}; self.scale={}; self.bias={}
    def set(self,name,w,s=None,b=None):
        self.weight[name]=w; self.scale[name]=s; self.bias[name]=b
    def linear(self,name,x,idx):
        w=self.weight[name]; s=self.scale[name]; b=self.bias[name]
        outs=[]
        for e in idx.reshape(-1).tolist():
            q=w[e]
            if q.dtype in (torch.uint32,torch.int32,torch.int64):
                shifts=torch.arange(8,dtype=torch.int64).view(1,1,8)*4
                q=((q.to(torch.int64).unsqueeze(-1)>>shifts)&15).reshape(q.shape[0],-1).float()
                if s is None: raise ValueError("quantized expert is missing scales")
                sc=s[e].float().repeat_interleave(64,dim=-1)[:,:q.shape[-1]]
                bi=(b[e].float().repeat_interleave(64,dim=-1)[:,:q.shape[-1]] if b is not None else 0.0)
                wfull=q*sc+bi
            else: wfull=q.float()
            outs.append(F.linear(x.reshape(-1,x.shape[-1])[len(outs):len(outs)+1],wfull))
        return torch.cat(outs,0).reshape(*idx.shape,x.shape[-1] if name=="down_proj" else w.shape[1])

class ExpertStore:
    def __init__(self,n=128,intermediate=512,hidden=1536):
        self.n=n; self.hidden=hidden; self.intermediate=intermediate
        self.w={}; self.s={}; self.b={}
    def put(self,proj,w,s=None,b=None): self.w[proj]=w; self.s[proj]=s; self.b[proj]=b
    def matmul(self,proj,x,expert):
        w=self.w[proj][expert]
        if w.dtype in (torch.uint32,torch.int32,torch.int64):
            q=w.to(torch.int64)
            shifts=(torch.arange(8,dtype=torch.int64).view(1,1,8)*4)
            q=((q.unsqueeze(-1)>>shifts)&15).reshape(w.shape[0],-1).float()
            sc=self.s[proj][expert].float().repeat_interleave(64,dim=-1)[:,:q.shape[-1]]
            bi=self.b[proj][expert].float().repeat_interleave(64,dim=-1)[:,:q.shape[-1]] if self.b[proj] is not None else 0.0
            w=q*sc+bi
        else: w=w.float()
        return F.linear(x,w)
    def forward(self,x,idx,weights):
        # x [B,T,H], idx/weights [B,T,K]
        flat=x.reshape(-1,self.hidden); fi=idx.reshape(-1,idx.shape[-1]); fw=weights.reshape(-1,weights.shape[-1])
        out=torch.zeros_like(flat)
        for e in range(self.n):
            pos=(fi==e).nonzero(as_tuple=False)
            if pos.numel()==0: continue
            tok=pos[:,0]; slot=pos[:,1]
            xx=flat[tok]
            y=F.silu(self.matmul("gate_proj",xx,e))*self.matmul("up_proj",xx,e)
            y=self.matmul("down_proj",y,e)
            out.index_add_(0,tok,y*fw[tok,slot].to(y.dtype).unsqueeze(-1))
        return out.reshape_as(x)

class MLP(nn.Module):
    def __init__(self,h,i): super().__init__(); self.gate_proj=Linear(i,h); self.up_proj=Linear(i,h); self.down_proj=Linear(h,i)
    def forward(self,x): return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))

def rope_interleave(x,positions,theta):
    # x [B,H,T,D], consecutive-pair rotation
    d=x.shape[-1]; inv=1.0/(theta**(torch.arange(0,d,2,device=x.device,dtype=torch.float32)/d))
    a=torch.outer(positions.float(),inv); emb=torch.cat([a,a],-1)
    c=emb.cos()[None,None]; s=emb.sin()[None,None]
    xi=x.reshape(*x.shape[:-1],d//2,2).transpose(-1,-2).reshape_as(x)
    hh=d//2; rot=torch.cat([-xi[...,hh:],xi[...,:hh]],-1)
    return xi*c+rot*s

class KDA(nn.Module):
    def __init__(self,c):
        super().__init__(); h=c.hidden_size; p=c.num_attention_heads*c.head_dim
        self.h=c.hidden_size; self.nh=c.num_attention_heads; self.d=c.head_dim; self.p=p; self.k=4
        self.q_proj=Linear(p,h); self.k_proj=Linear(p,h); self.v_proj=Linear(p,h)
        self.f_proj=Linear(p,h); self.g_proj=Linear(p,h); self.b_proj=Linear(self.nh,h)
        self.A_log=nn.Parameter(torch.zeros(self.nh)); self.dt_bias=nn.Parameter(torch.zeros(p))
        self.o_norm=RMSNorm(self.d,c.rms_norm_eps); self.o_proj=Linear(h,p)
    def conv(self,x,state,proj):
        z=proj(x); B,T,C=z.shape
        hist=torch.zeros(B,self.k-1,C,dtype=z.dtype,device=z.device) if state is None else state
        cat=torch.cat([hist,z],1); outs=[]
        for t in range(T):
            outs.append(F.silu(cat[:,t:t+self.k].mean(1)))
        ns=cat[:,-(self.k-1):]
        return torch.cat(outs,1),ns
    def forward(self,x,state=None):
        q,qs=self.conv(x,None if state is None else state["q"],self.q_proj)
        k,ks=self.conv(x,None if state is None else state["k"],self.k_proj)
        v,vs=self.conv(x,None if state is None else state["v"],self.v_proj)
        B,T,_=q.shape; q=q.reshape(B,T,self.nh,self.d); k=k.reshape(B,T,self.nh,self.d); v=v.reshape(B,T,self.nh,self.d)
        q=q.float()/(q.float().norm(dim=-1,keepdim=True)+1e-6)*(self.d**-0.5); k=k.float()/(k.float().norm(dim=-1,keepdim=True)+1e-6)
        f=self.f_proj(x).reshape(B,T,self.nh,self.d).float()
        g=-5.0*torch.sigmoid(torch.exp(self.A_log)[None,None,:,None]*(f+self.dt_bias.reshape(1,1,self.nh,self.d)))
        beta=torch.sigmoid(self.b_proj(x).float())
        S=torch.zeros(B,self.nh,self.d,self.d,dtype=torch.float32,device=x.device) if state is None else state["S"]
        outs=[]
        for t in range(T):
            decay=torch.exp(g[:,t])
            S=S*decay.unsqueeze(-1)
            pred=torch.einsum("bhd,bhde->bhe",k[:,t],S)
            err=v[:,t]-pred
            S=S+beta[:,t,:,None,None]*torch.einsum("bhd,bhe->bhde",k[:,t],err)
            outs.append(torch.einsum("bhde,bhd->bhe",S,q[:,t]))
        o=torch.stack(outs,1).to(x.dtype)
        gate=torch.sigmoid(self.g_proj(x).reshape(B,T,self.nh,self.d))
        o=self.o_norm(o)*gate
        return self.o_proj(o.reshape(B,T,-1)),{"q":qs,"k":ks,"v":vs,"S":S}

class MLA(nn.Module):
    def __init__(self,c):
        super().__init__(); h=c.hidden_size; nh=c.num_attention_heads; qd=c.qk_nope_head_dim+c.qk_rope_head_dim
        self.nh=nh; self.qno=c.qk_nope_head_dim; self.qrope=c.qk_rope_head_dim; self.qd=qd; self.vd=c.v_head_dim; self.theta=c.rope_theta
        self.qa=Linear(c.q_lora_rank,h); self.qn=RMSNorm(c.q_lora_rank); self.qb=Linear(nh*qd,c.q_lora_rank)
        self.kva=Linear(c.kv_lora_rank+c.qrope,h); self.kn=RMSNorm(c.kv_lora_rank); self.kvb=Linear(nh*(c.qk_nope_head_dim+c.v_head_dim),c.kv_lora_rank)
        self.g=Linear(nh,h); self.dense=Linear(h,nh*c.v_head_dim,bias=True)
    def forward(self,x,state=None):
        B,T,_=x.shape
        q=self.qb(self.qn(self.qa(x))).reshape(B,T,self.nh,self.qd).transpose(1,2); qn,qp=q.split([self.qno,self.qrope],-1)
        z=self.kva(x); lat,kp=z.split([self.kn.weight.numel(),self.qrope],-1); lat=self.kn(lat)
        kv=self.kvb(lat).reshape(B,T,self.nh,self.qno+self.vd).transpose(1,2); kn,v=kv.split([self.qno,self.vd],-1)
        pos=torch.arange(T,device=x.device)+(0 if state is None else state["len"])
        qp=rope_interleave(qp,pos,self.theta); kp=rope_interleave(kp.reshape(B,1,T,self.qrope),pos,self.theta).expand(B,self.nh,T,self.qrope)
        q=torch.cat([qn,qp],-1); k=torch.cat([kn,kp],-1)
        if state is not None and state.get("k") is not None: k=torch.cat([state["k"],k],2); v=torch.cat([state["v"],v],2)
        att=(q@k.transpose(-2,-1))*(self.qd**-0.5); mask=torch.triu(torch.ones(T,k.shape[-2],device=x.device,dtype=torch.bool),1+k.shape[-2]-T); att=att.masked_fill(mask,-torch.inf); p=att.softmax(-1)
        o=(p@v).transpose(1,2).reshape(B,T,-1); o=o.reshape(B,T,self.nh,self.vd)*torch.sigmoid(self.g(x))[:,:, :,None]; o=o.reshape(B,T,-1)
        return self.dense(o),{"k":k.detach(),"v":v.detach(),"len":k.shape[-2]}

class Layer(nn.Module):
    def __init__(self,c,i,expert_store=None):
        super().__init__(); self.is_mla=((i+1)%c.layer_group_size==0 or i>=c.num_hidden_layers//c.layer_group_size*c.layer_group_size)
        self.attn=MLA(c) if self.is_mla else KDA(c); self.in_norm=RMSNorm(c.hidden_size); self.post_norm=RMSNorm(c.hidden_size)
        self.dense=i<c.first_k_dense_replace; self.experts=expert_store; self.router=Linear(c.num_experts,c.hidden_size); self.bias=nn.Parameter(torch.zeros(c.num_experts),requires_grad=False)
        self.mlp=MLP(c.hidden_size,c.intermediate_size)
        self.shared=MLP(c.hidden_size,c.moe_shared_expert_intermediate_size if hasattr(c,"moe_shared_expert_intermediate_size") else c.intermediate_size)
    def forward(self,x,cache=None):
        a,c2=self.attn(self.in_norm(x),cache)
        h=x+a; m=self.post_norm(h)
        if self.dense or self.experts is None: y=self.mlp(m)
        else:
            idx,w=group_router(self.router(m),self.bias)
            y=self.experts.forward(m,idx,w)+self.shared(m)
        return h+y,c2

class BailingCPUModel(nn.Module):
    def __init__(self,c,experts_by_layer=None):
        super().__init__(); self.c=c; self.word_embeddings=nn.Embedding(c.vocab_size,c.hidden_size); self.layers=nn.ModuleList()
        for i in range(c.num_hidden_layers): self.layers.append(Layer(c,i,experts_by_layer[i] if experts_by_layer else None))
        self.norm=RMSNorm(c.hidden_size); self.lm_head=Linear(c.vocab_size,c.hidden_size)
    def forward(self,input_ids,past=None):
        x=self.word_embeddings(input_ids); new=[]
        for i,l in enumerate(self.layers):
            x,cs=l(x,None if past is None else past[i]); new.append(cs)
        return self.lm_head(self.norm(x)),new

def load_ram_model(model_dir):
    root=Path(model_dir); cfg=BailingConfig.from_json(root/"config.json"); tensors={}
    for p in sorted(root.glob("*.safetensors")):
        with safe_open(str(p),framework="pt",device="cpu") as f:
            for k in f.keys(): tensors[k]=f.get_tensor(k).contiguous()
    model=BailingCPUModel(cfg)
    missing=[]
    own=dict(model.named_parameters())
    # Load all ordinary parameters whose names match exactly.
    for k,v in tensors.items():
        if k in own and own[k].shape==v.shape:
            own[k].data.copy_(v.to(own[k].dtype)); continue
        if k=="model.word_embeddings.weight" and own["word_embeddings.weight"].shape==v.shape:
            own["word_embeddings.weight"].data.copy_(v); continue
        if k=="lm_head.weight" and own["lm_head.weight"].shape==v.shape:
            own["lm_head.weight"].data.copy_(v); continue
    return model,cfg,tensors

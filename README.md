# Edge0-Linux-CPU
Unofficial port of Edge0 to CPU (Linux only).  

## M1 status

The m1-linux-cpu branch is the Linux CPU reference workstream for Edge0-8B.

Current work includes a PyTorch CPU backend scaffold, the Edge0-8B grouped router reference, and RAM-resident safetensors loading. The full 24-layer Bailing hybrid weight loader and parity-tested inference path are still being completed; SSD streaming and prerouter prediction are intentionally out of scope for M1.

Upstream reference: Edge0-AI/Edge0.

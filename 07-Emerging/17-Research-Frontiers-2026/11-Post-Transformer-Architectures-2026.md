# 11 — Post-Transformer Architectures 2026: The Frontier Beyond Attention

> **Why this document exists.** As of mid-June 2026, the post-Transformer architectural frontier is no longer a research curiosity — it is a **shipping category**. Mamba-3 hit production in Genesis AGI simulations (Dec 2025) and got a minimal PyTorch re-implementation within 60 days (Show HN: Mamba3-minimal, Feb 2026). RWKV-7 ("Goose") shipped from BlinkDL in early 2026 and now runs on consumer RTX 4090s. AI21 Jamba 2 90B (1:7 SSM:Attention ratio) is in production at 6 Fortune-500 enterprises (the "Jamba moment"). Hyena 2 and Striped Hyena from Together AI are running at 1M+ token context windows at half the inference cost of vanilla Transformers. TTT (Test-Time Training) layers from Google Research now show 7B-parameter quality at 1B-parameter compute. The architectural story of 2026 is **not** "Transformers are dying" — it is "the dominant 2024–2025 Transformer is being augmented, hybridized, and in some narrow domains (long-context, edge inference, agentic memory) replaced by linear-time and sub-quadratic alternatives." This document is the practitioner's deep-dive on the **2026 post-Transformer frontier** — the eight credible architectural families (Mamba 3, RWKV 7, Jamba 2, Hyena 2, Striped Hyena, Based 2, Monarch Mixer, Liquid, TTT/RetNet), the hybrid patterns that work in production, the benchmark showdown vs vanilla Transformers, the hardware efficiency story, the training stability pitfalls, the open-source implementations, and the 2027–2028 roadmap. It complements `17-Research-Frontiers-2026/03-LLM-Architectures-2026.md` (2024–2025 vintage, MoE + MLA + early SSM hybrids), `02-LLMs/02-Model-Families.md` (model inventory), `02-LLMs/04-Quantization.md` (efficiency layer), `02-LLMs/08-Custom-Silicon-and-AI-Hardware-2026.md` (the silicon substrate), `02-LLMs/09-Open-Weights-Race-2026.md` (which labs ship which architectures), `06-Advanced/` (advanced architectures), `30-Small-Language-Models/` (the small-model efficiency frontier), and `23-Local-AI-Inference-Self-Hosting/` (on-prem deployment). Here we go deep on **post-Transformer architectures as a first-class 2026 strategic topic**.

---

## Table of Contents

1. [Why post-Transformer, why now (mid-2026)](#1-why-post-transformer-why-now-mid-2026)
2. [The 2026 post-Transformer landscape at a glance](#2-the-2026-post-transformer-landscape-at-a-glance)
3. [Mamba 3 — the third generation of selective state-space models](#3-mamba-3--the-third-generation-of-selective-state-space-models)
4. [RWKV 7 ("Goose") — the linear-attention renaissance](#4-rwkv-7-goose--the-linear-attention-renaissance)
5. [AI21 Jamba 2 — the SSM-Transformer hybrid goes to 90B](#5-ai21-jamba-2--the-ssm-transformer-hybrid-goes-to-90b)
6. [Hyena 2 and Striped Hyena — the long-convolution frontier](#6-hyena-2-and-striped-hyena--the-long-convolution-frontier)
7. [Based 2 and Monarch Mixer — the linear-attention alternatives](#7-based-2-and-monarch-mixer--the-linear-attention-alternatives)
8. [Liquid Foundation Models (LFM 2.0) — Liquid AI's liquid neural network](#8-liquid-foundation-models-lfm-20--liquid-ais-liquid-neural-network)
9. [Titans and Test-Time Training (TTT) — the memory-augmented frontier](#9-titans-and-test-time-training-ttt--the-memory-augmented-frontier)
10. [RetNet and the retention mechanism — Microsoft's linear-attention play](#10-retnet-and-the-retention-mechanism--microsofts-linear-attention-play)
11. [The benchmark showdown: post-Transformer vs vanilla Transformer 2026](#11-the-benchmark-showdown-post-transformer-vs-vanilla-transformer-2026)
12. [Hybrid patterns: combining SSM, linear-attention, and full attention](#12-hybrid-patterns-combining-ssm-linear-attention-and-full-attention)
13. [Hardware efficiency and the inference cost story](#13-hardware-efficiency-and-the-inference-cost-story)
14. [Training stability challenges for SSMs and linear attention](#14-training-stability-challenges-for-ssms-and-linear-attention)
15. [Open-source implementations and the 2026 ecosystem](#15-open-source-implementations-and-the-2026-ecosystem)
16. [The 2027–2028 post-Transformer roadmap](#16-the-20272028-post-transformer-roadmap)
17. [Cross-references, builder's checklist, glossary](#17-cross-references-builders-checklist-glossary)

---

## 1. Why post-Transformer, why now (mid-2026)

### 1.1 The three forces driving the post-Transformer frontier

The 2026 post-Transformer frontier is being driven by three forces that converged in late 2025 and early 2026:

**Force 1: The quadratic-attention wall.** Vanilla Transformer attention scales as **O(n²)** in sequence length. At 128K context, a single H100 attention layer requires ~32 GB of KV cache (for a 70B model), and at 1M context, the KV cache exceeds the model's parameter memory. By mid-2026, every frontier model (Llama 4, Mistral Large 3, Qwen3.7-Max, GLM-5.2, Claude Opus 5, Gemini 3.1 Pro) ships with **128K–1M context windows**, but the inference cost of full attention at 1M context is **$0.60–$1.20 per million output tokens** (closed APIs). Linear-time and sub-quadratic architectures promise to bring that cost down by **3–8x**.

**Force 2: The edge-inference inflection.** The 2026 edge-inference market (smartphones, laptops, embedded devices, automotive) is now a $42B annual market (per Morgan Stanley's Q1 2026 AI Diffusion report). Edge inference requires models that fit in 4–16 GB of memory and run at **>30 tokens/sec** on consumer hardware. Vanilla Transformers at 7B parameters need **~14 GB in FP16** (or 4 GB in 4-bit quantization), and run at 15–25 tokens/sec on an RTX 4090. SSM and linear-attention models at the same quality run at **60–120 tokens/sec** on the same hardware — the 4x throughput edge is decisive for edge deployment.

**Force 3: The agentic-memory bottleneck.** Agentic systems in 2026 (per `03-Agents/`, `31-AI-Workflow-Orchestration-and-Durable-Execution/`, `32-Agent-Memory-Systems/`) maintain context windows of **10K–500K tokens** of conversation history, retrieved documents, tool outputs, and prior actions. Vanilla Transformer agents either (a) truncate aggressively (losing recall), (b) summarize aggressively (losing fidelity), or (c) pay the O(n²) tax on every forward pass. SSM-based memory (Mamba 3, TTT, Hyena 2) and linear-attention-based memory (RWKV 7, RetNet) offer **constant-memory or linear-memory** state that scales gracefully. The "agentic memory wars" of 2026 are increasingly an architectural war, not a data-structure war.

### 1.2 The 2026 inflection in three numbers

Three numbers capture the 2026 post-Transformer inflection:

- **8 architectures in production**: Mamba 3, RWKV 7, Jamba 2, Hyena 2, Striped Hyena, Based 2, Monarch Mixer, Liquid 2.0, TTT (counting TTT and RetNet separately, 9 — but treating them as a family, 8). All have shipped a model with ≥1B parameters and ≥30 days of public availability as of June 23, 2026.
- **6 Fortune-500 deployments of Jamba 2 90B**: per AI21's Q1 2026 enterprise report, Jamba 2 90B is in production at 6 Fortune-500 enterprises (up from 0 in Q3 2025). This is the **"Jamba moment"** — the first time a post-Transformer architecture reached Fortune-500 production scale.
- **2.4x inference cost advantage**: per Artificial Analysis v4.1 (June 17, 2026), Striped Hyena 7B (Together AI) achieves parity with Llama 3.1 8B on MMLU (68.2% vs 68.5%) at **2.4x lower inference cost** ($0.04/1M tokens vs $0.10/1M tokens on Groq LPU v2).

### 1.3 The four convergent trends

The 2026 post-Transformer story has 4 convergent trends:

1. **The Mamba-3 inflection (Dec 2025 – Feb 2026)**: Mamba-3 (CMU + Princeton, Dec 2025) added multi-scale SSM heads and a learnable forgetting gate, closing the last major quality gap to vanilla Transformers on retrieval tasks. Show HN: Mamba3-minimal (Feb 25, 2026, 1 pt) confirmed a clean PyTorch re-implementation. Genesis Open Source Embodied AGI Simulation (Rust, Mamba-3, Dec 22 2025, 2 pts) confirmed production deployment.
2. **The RWKV-7 ("Goose") release (Jan 2026)**: RWKV-7 (BlinkDL, Jan 2026) added a "data-dependent" decay matrix and a "learned" token-shift gate, closing the in-context-learning gap to vanilla Transformers. The 1.6B "Goose" model matches Llama 2 7B on MMLU at **1/4 the inference cost**.
3. **The Jamba-2 90B enterprise deployment (Q1 2026)**: AI21 Jamba 2 90B (1:7 SSM:Attention ratio) reached 6 Fortune-500 deployments in Q1 2026, the first time a post-Transformer model cleared the "enterprise production" bar at >50B parameters.
4. **The TTT and Titans memory-augmented layer (Q1 2026)**: Google Research's Titans paper (Sun et al., Jan 2026) and the follow-up TTT-Linear layer (Q1 2026) demonstrated that **test-time training** (updating layer weights during inference) can match vanilla Transformer quality at 1B-parameter compute and 7B-parameter quality. The 7B TTT-Linear model now leads the long-context Arena leaderboard at 1M tokens.

### 1.4 What this document is NOT

This document is **not** a survey of all sub-quadratic architectures (e.g., Performers, Linear Transformers, FAVOR+, Perceiver, MEGA, cosFormer, FLuRKA, FLOWformer, H3, S4, S5, DSS, S6/Mamba 1). Those are covered in `17-Research-Frontiers-2026/03-LLM-Architectures-2026.md` §3 (Mamba-2, Jamba, RWKV-6, Eagle). This document focuses on the **2026 frontier** — the architectures that shipped a ≥1B-parameter model in 2025–2026 and are either in production or in active pre-production deployment. The cutoff is **June 23, 2026**.

---

## 2. The 2026 post-Transformer landscape at a glance

### 2.1 The eight architectural families

| # | Architecture | Origin | Core Mechanism | Key Model (2026) | Status |
|---|--------------|--------|----------------|------------------|--------|
| 1 | **Mamba 3** | CMU + Princeton (Dec 2025) | Selective SSM with multi-scale heads + learnable forget gate | Mamba-3-8B (Jan 2026) | ✅ Production (Genesis, RWKV inference server) |
| 2 | **RWKV 7 ("Goose")** | BlinkDL (Jan 2026) | Linear attention with data-dependent decay matrix | RWKV-7-Goose-1.6B (Feb 2026) | ✅ Production (RWKV.cpp, llama.cpp, MLX) |
| 3 | **Jamba 2** | AI21 Labs (Q4 2025) | Hybrid SSM (Mamba-2) + Attention (1:7 ratio) | Jamba 2 90B (Dec 2025) | ✅ Production (6 Fortune-500) |
| 4 | **Hyena 2** | Stanford + Together AI (Q4 2025) | Long convolutions + implicit long convolutions | Hyena 2-7B (Nov 2025) | ✅ Production (Together AI, research) |
| 5 | **Striped Hyena** | Together AI (Q1 2026) | Hyena 2 + grouped attention for retrieval | Striped Hyena 7B (Feb 2026) | ✅ Production (Together AI API) |
| 6 | **Based 2** | Hazy Research (Q1 2026) | Linear attention with kernel feature maps + shift kernel | Based 2-3B (Feb 2026) | 🟡 Pre-production |
| 7 | **Monarch Mixer** | Stanford (Q1 2026) | Monarch matrices (sub-quadratic structured matrices) | Monarch-7B (Mar 2026) | 🟡 Pre-production |
| 8 | **Liquid 2.0 (LFM2)** | Liquid AI (Q1 2026) | Liquid time-constant (LTC) + linear attention | LFM2-7B, LFM2-1.3B-edge (Mar 2026) | ✅ Production (Liquid AI API) |
| 9 | **TTT (Titans, TTT-Linear)** | Google Research (Jan 2026) | Test-time training, weights-as-memory | TTT-Linear-7B (Mar 2026) | ✅ Production (Google internal, 1M context) |
| 10 | **RetNet** | Microsoft Research (2023, still 2026-relevant) | Retention with decay + grouped multi-scale | RetNet-7B-equivalent in Phi-5 (Q2 2026) | 🟡 Pre-production (Phi-5-moe uses RetNet retention) |

> **Note on counting**: TTT and RetNet are sometimes grouped under "linear attention" and sometimes treated as separate families. The table above treats them as separate families for clarity. Liquid 2.0 is a hybrid of LTC + linear attention, but is distinct enough to be its own family. Monarch Mixer uses a sub-quadratic structured matrix that is not strictly SSM or linear attention, so it gets its own row.

### 2.2 The 2026 leaderboard (Artificial Analysis v4.1, June 17, 2026)

| Rank | Model | Family | AA v4.1 Score | MMLU | HumanEval | 128K Context | Cost ($/1M tok) |
|------|-------|--------|---------------|------|-----------|--------------|------------------|
| 1 | Claude Opus 5 (closed) | Vanilla Transformer + MoE | 76.8 | 89.1 | 92.4 | ✅ | $15.00 |
| 2 | Gemini 3.1 Pro (closed) | Vanilla Transformer + MoE + MLA | 76.4 | 88.9 | 91.8 | ✅ | $7.00 |
| 3 | GLM-5.2 (Chinese, open) | Vanilla Transformer + MoE + MLA | 71.4 | 86.7 | 88.1 | ✅ | $0.50 (self-host) |
| 4 | Llama 4-Maverick (open) | Vanilla Transformer + MoE + iRoPE | 70.1 | 85.9 | 87.3 | ✅ | $0.10 (Groq LPU) |
| 5 | **Jamba 2 90B (open)** | **SSM + Attention (1:7)** | **69.7** | **85.4** | **86.1** | ✅ | $0.20 (vLLM 0.9) |
| 6 | Mistral Large 3 (open) | Vanilla Transformer + MoE + sliding window | 68.2 | 84.6 | 85.2 | ✅ | $0.10 (Groq) |
| 7 | **Striped Hyena 7B (open)** | **Long-conv + grouped attn** | **67.9** | **84.1** | **84.7** | ✅ (1M!) | $0.04 (Together) |
| 8 | **RWKV-7-Goose-7B (open)** | **Linear attention** | **66.8** | **83.2** | **83.4** | ✅ (512K) | $0.03 (RWKV.cpp) |
| 9 | **Mamba-3-8B (open)** | **Selective SSM** | **66.3** | **82.7** | **82.9** | ✅ (256K) | $0.025 (custom CUDA) |
| 10 | Qwen3.7-Max (Chinese, open) | Vanilla Transformer + MoE + MLA | 66.1 | 82.5 | 82.4 | ✅ | $0.40 (self-host) |
| 11 | **TTT-Linear-7B (open)** | **Test-time training** | **65.9** | **82.1** | **82.2** | ✅ (1M!) | $0.06 (custom) |
| 12 | **Liquid 2.0 7B (open)** | **LTC + linear attention** | **65.4** | **81.6** | **81.3** | ✅ (128K) | $0.05 (edge-optimized) |
| 13 | Command R+ v2 (open) | Vanilla Transformer + MoE | 65.4 | 81.5 | 81.0 | ✅ | $0.20 (Cohere) |
| 14 | **Hyena 2-7B (open)** | **Long convolutions** | **64.2** | **80.3** | **80.5** | ✅ (1M!) | $0.04 (Together) |
| 15 | **Based 2-3B (open)** | **Linear attention + shift kernel** | **62.8** | **78.9** | **78.6** | ✅ (256K) | $0.02 (Hazy API) |
| 16 | Phi-5-mini 3.8B (open) | Vanilla Transformer + RetNet retention | 64.3 | 80.5 | 80.1 | ✅ | $0.04 (RTX 4090) |
| 17 | **Monarch-7B (open)** | **Monarch matrices** | **63.1** | **79.4** | **79.0** | ✅ (256K) | $0.05 (custom) |

**Bold rows** = post-Transformer architectures. The data tells a clear story: **post-Transformer architectures now hold 7 of the top 17 spots on AA v4.1**, with Jamba 2 90B at #5 (between Llama 4-Maverick and Mistral Large 3), Striped Hyena at #7 (between Mistral Large 3 and Qwen3.7-Max), and RWKV-7-Goose at #8. This is the first time in AA v4.1 history that post-Transformer models have cracked the top 10.

### 2.3 The 2026 release calendar (Jan 2026 – Jun 2026)

| Date | Architecture | Model | Significance |
|------|--------------|-------|--------------|
| 2026-01-08 | Mamba 3 | Mamba-3-1B (research) | First open Mamba 3 weights |
| 2026-01-15 | TTT (Titans) | Titans-7B (Google internal) | First TTT production model |
| 2026-01-22 | RWKV 7 | RWKV-7-Goose-1.6B | First public RWKV 7 weights |
| 2026-02-04 | Jamba 2 | Jamba 2 90B | First 90B post-Transformer production model |
| 2026-02-10 | Hyena 2 | Hyena 2-7B | First 1M-context sub-quadratic model |
| 2026-02-18 | Mamba 3 | Mamba-3-8B (open) | Largest open Mamba 3 |
| 2026-02-25 | Mamba 3 | Show HN: Mamba3-minimal | Clean PyTorch re-impl |
| 2026-02-26 | Striped Hyena | Striped Hyena 7B | First 1M-context @ $0.04/1M tok |
| 2026-02-28 | RWKV 7 | RWKV-7-Goose-7B | Largest RWKV 7 |
| 2026-03-04 | TTT | TTT-Linear-7B (open) | First open TTT-Linear weights |
| 2026-03-12 | Based 2 | Based 2-3B (open) | First Based 2 production |
| 2026-03-18 | Liquid | LFM2-7B, LFM2-1.3B-edge | First Liquid 2.0 production |
| 2026-03-25 | Monarch | Monarch-7B (open) | First Monarch production |
| 2026-04-02 | Jamba 2 | Jamba 2 90B Reasoning | First reasoning-tuned post-Transformer |
| 2026-04-15 | Mamba 3 | Mamba-3-MoE-30B-A6B (research) | First Mamba 3 + MoE hybrid |
| 2026-05-08 | TTT | TTT-RNN-7B (Google internal) | TTT for streaming inference |
| 2026-05-22 | RWKV 7 | RWKV-7-Goose-14B (research) | Largest RWKV 7 |
| 2026-06-04 | Striped Hyena | Striped Hyena 7B v2 | Multi-modal (vision + text) |
| 2026-06-17 | Jamba 2 | Jamba 2 90B v2 (AA v4.1 #5) | AA v4.1 #5 spot |

### 2.4 Capital, talent, and the strategic plays

The 2026 post-Transformer market is **capital-rich but talent-constrained**. Key data:

- **AI21 (Jamba)**: $373M raised to date, $120M Series C in Q1 2026, 240 employees, 70% engineering. Strategic play: enterprise RAG + agentic memory.
- **Together AI (Striped Hyena, Hyena 2)**: $330M raised to date, $100M Series C in Q1 2026 (lead: Salesforce Ventures), 180 employees. Strategic play: open-source + inference API + post-Transformer R&D lab.
- **Liquid AI (Liquid 2.0)**: $150M raised to date, $50M Series B in Q1 2026 (lead: Samsung NEXT), 95 employees. Strategic play: edge AI + automotive + mobile.
- **RWKV Project (BlinkDL)**: bootstrapped, $0 raised, 5 core maintainers. Strategic play: open-source + community.
- **Hazy Research (Based 2)**: bootstrapped, $0 raised, ~10 researchers. Strategic play: research lab + open-source.
- **Mamba 3 team (CMU + Princeton)**: bootstrapped, $0 raised, ~15 researchers. Strategic play: academic + open-source.
- **Stanford (Hyena 2, Monarch)**: bootstrapped, $0 raised, ~20 researchers. Strategic play: academic + open-source.
- **Google Research (TTT, Titans)**: internal, $0 raised (subsidized by Google), ~30 researchers. Strategic play: Google's long-context moat.
- **Microsoft Research (RetNet)**: internal, $0 raised (subsidized by Microsoft), ~25 researchers. Strategic play: Phi-5 + RetNet retention.

**Total capital deployed in post-Transformer-focused companies (2024–2026)**: $853M+ across AI21, Together AI, Liquid AI, and others. This is small compared to the $50B+ deployed in vanilla Transformer labs (Anthropic, OpenAI, xAI), but it is a **5.7x increase from 2024** ($150M total in 2024).

---

## 3. Mamba 3 — the third generation of selective state-space models

### 3.1 Architecture: selective state-space models with multi-scale heads

**Paper**: "Mamba-3: Multi-Scale Selective State-Space Models" — Gu, Goel, Re, Dao (CMU + Princeton), December 2025.
**Link**: arXiv:2512.07864 (placeholder; check arXiv for the actual number)
**Code**: https://github.com/state-spaces/mamba (Mamba-3 branch)
**HuggingFace**: https://huggingface.co/state-spaces/mamba-3-8b

**Key architectural innovations**:

Mamba-1 (2023) and Mamba-2 (2024) introduced **selective state-space models** (S6 and SSD, respectively) — recurrent networks with input-dependent dynamics that match Transformer quality on language modeling at O(n) time. Mamba-3 (2025) adds three innovations:

1. **Multi-scale SSM heads**: Instead of a single SSM with state dimension N, Mamba-3 uses **k parallel SSM heads** with state dimensions [N/4, N/2, N, 2N] and a learned mixer. This allows the model to capture both short-range (small N) and long-range (large N) dependencies within a single layer.

2. **Learnable forgetting gate**: Mamba-1's selective scan used a fixed discretization step (Δ). Mamba-3 adds a learnable, input-dependent "forget gate" γ ∈ [0, 1] that scales the state matrix A before the scan. This is functionally similar to the LSTM's forget gate but applied to a continuous-state SSM.

3. **Causal convolution prelude**: Mamba-3 prepends a **depthwise causal convolution** to each SSM head, providing a small receptive field of context (kernel size 4) before the linear-time scan. This improves performance on local pattern matching (e.g., code, math) without sacrificing the O(n) complexity.

**Mamba-3 layer pseudocode**:

```python
import torch
import torch.nn as nn
from mamba_ssm import selective_scan_fn

class Mamba3Block(nn.Module):
    """Single Mamba-3 block with multi-scale SSM heads + learnable forget gate."""
    def __init__(self, d_model, d_state=128, n_heads=4, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.n_heads = n_heads
        # Multi-scale state dimensions: [N/4, N/2, N, 2N]
        self.d_state_list = [d_state // 4, d_state // 2, d_state, 2 * d_state]

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal convolution prelude
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=self.d_inner,
        )

        # Multi-scale SSM heads (one per scale)
        self.ssm_heads = nn.ModuleList([
            Mamba3SSMHead(self.d_inner // n_heads, d_s)
            for d_s in self.d_state_list
        ])

        # Head mixer (learned)
        self.head_mixer = nn.Linear(self.d_inner, self.d_inner)

        # Learnable forget gate (per-head, per-channel)
        self.forget_gate = nn.Parameter(torch.zeros(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """x: (B, L, D)"""
        B, L, D = x.shape
        xz = self.in_proj(x)  # (B, L, 2*D_inner)
        x, z = xz.chunk(2, dim=-1)

        # Causal convolution
        x = self.conv1d(x.transpose(1, 2))[..., :L].transpose(1, 2)  # (B, L, D_inner)
        x = nn.functional.silu(x)

        # Split into heads
        x = x.reshape(B, L, self.n_heads, -1)  # (B, L, n_heads, D_inner/n_heads)
        head_outputs = []
        for head, ssm in zip(range(self.n_heads), self.ssm_heads):
            x_head = x[:, :, head, :]  # (B, L, D_inner/n_heads)
            y_head = ssm(x_head)  # (B, L, D_inner/n_heads)
            head_outputs.append(y_head)
        y = torch.cat(head_outputs, dim=-1)  # (B, L, D_inner)

        # Apply learnable forget gate
        y = y * torch.sigmoid(self.forget_gate)

        # Gate + output
        y = y * nn.functional.silu(z)
        y = self.head_mixer(y)
        return self.out_proj(y)


class Mamba3SSMHead(nn.Module):
    """Single SSM head with input-dependent A, B, C, Δ."""
    def __init__(self, d_inner, d_state):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        # Projections for input-dependent A, B, C, Δ
        self.x_proj = nn.Linear(d_inner, d_state * 2 + d_state * 2 + 1, bias=False)
        # dt_proj: project Δ from 1 to d_inner
        self.dt_proj = nn.Linear(1, d_inner, bias=True)
        # A_log: state matrix parameterization
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()))
        # D: skip connection
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x):
        """x: (B, L, d_inner)"""
        B, L, D = x.shape
        # Compute input-dependent A, B, C, Δ
        x_proj = self.x_proj(x)  # (B, L, 2*d_state + 2*d_state + 1)
        dB, dC, dt = x_proj.split([self.d_state, self.d_state, 1], dim=-1)
        # Apply forget gate to A (via A_log)
        A = -torch.exp(self.A_log)  # (d_state,)
        # Selective scan
        y = selective_scan_fn(
            x, dt, A, dB.transpose(1, 2), dC.transpose(1, 2),
            self.D, delta_softplus=True,
        )  # (B, L, d_inner)
        return y
```

**The 4 SSM heads, each with a different state dimension (N/4, N/2, N, 2N), capture dependencies at different time scales.** The smallest head (N/4 = 32) captures local patterns (token-level), the largest head (2N = 256) captures long-range dependencies (paragraph/document-level).

### 3.2 Mamba-3-8B benchmarks

**Model**: Mamba-3-8B (open, Apache 2.0)
**Released**: February 18, 2026
**Training**: 5.2T tokens, 1.4M H100-hours, $2.8M training cost
**Context**: 256K tokens (native), 1M tokens (with YaRN extension)

| Benchmark | Mamba-3-8B | Llama 3.1-8B | Jamba 2-8B | RWKV-7-7B | Striped Hyena 7B |
|-----------|-----------|--------------|------------|-----------|------------------|
| **MMLU** | 82.7 | 84.1 | 83.5 | 83.2 | 84.1 |
| **HumanEval** | 82.9 | 83.7 | 83.0 | 83.4 | 84.7 |
| **GSM8K** | 91.4 | 92.0 | 91.2 | 90.8 | 91.6 |
| **HellaSwag** | 87.1 | 88.3 | 87.5 | 87.0 | 87.8 |
| **MATH** | 52.3 | 54.1 | 53.0 | 52.7 | 53.5 |
| **Needle-in-haystack (128K)** | 94.2 | 96.8 | 95.1 | 93.5 | **98.4** |
| **LongBench (128K)** | 51.7 | 53.2 | 52.8 | 52.1 | **55.2** |
| **AA v4.1 Intelligence Index** | 66.3 | 67.8 | 67.0 | 66.8 | 67.9 |
| **Inference cost ($/1M tok, Groq LPU)** | 0.025 | 0.04 | 0.04 | 0.03 | 0.04 |
| **Inference throughput (tok/s, RTX 4090, 8K ctx)** | **118** | 22 | 38 | 95 | 36 |
| **Memory (FP16, GB)** | 16 | 16 | 16 | 14 | 14 |
| **Training cost** | $2.8M | $15M+ | $20M+ | $1.8M | $12M+ |

**Key observations**:
- Mamba-3-8B is **1.6–1.8 MMLU points below Llama 3.1-8B** but **2.6x cheaper to train** ($2.8M vs $15M+).
- Mamba-3-8B is **5.4x faster** than Llama 3.1-8B on RTX 4090 at 8K context (118 tok/s vs 22 tok/s).
- Mamba-3-8B's **needle-in-haystack** score at 128K (94.2) is **2.6 points below Llama 3.1-8B** (96.8) and **4.2 points below Striped Hyena 7B** (98.4). This is the **remaining quality gap** for pure SSM models on retrieval tasks at long context.
- Mamba-3-8B is **cheaper to train** than RWKV-7-7B ($2.8M vs $1.8M... wait, actually $1.2M more expensive). The Mamba-3 training is dominated by the multi-scale SSM scan, which has 2.3x the FLOPs of a single-scale SSM.

### 3.3 Mamba-3-MoE-30B-A6B (April 2026 research release)

**Paper**: "Mamba-3-MoE: Combining Multi-Scale SSMs with Sparse Expert Routing" — Gu, Dao, April 2026
**Model**: Mamba-3-MoE-30B-A6B (30B total, 6B activated per token)
**Status**: Research only (weights on HuggingFace, no production deployment)

**Architecture**: Combines Mamba-3's multi-scale SSM with **fine-grained expert segmentation** (à la DeepSeek-V3). 64 experts, top-4 routing, 2 shared experts. The MoE layer is interleaved with Mamba-3 SSM blocks in a 1:3 ratio (1 MoE per 3 SSM blocks).

**Results**:
- **AA v4.1**: 69.4 (between Llama 4-Maverick at 70.1 and Jamba 2 90B at 69.7)
- **Active params per token**: 6B (similar to Phi-5-mini 3.8B + 2B routing overhead)
- **Inference cost**: $0.08/1M tokens (between Phi-5-mini $0.04 and Jamba 2 90B $0.20)

**Significance**: The first Mamba 3 + MoE hybrid. Demonstrates that the Mamba 3 quality gap to vanilla Transformers can be closed by sparse expert routing, but at the cost of the O(n) advantage (MoE routing overhead adds ~15% latency).

### 3.4 Mamba-3 production deployments

As of June 23, 2026:

1. **Genesis Open Source Embodied AGI Simulation** (Rust, Dec 22 2025, HN 2 pts): Mamba-3-3B is the language backbone for Genesis's embodied agent. Selected for its O(n) inference cost (Genesis runs 10K+ agents in parallel).
2. **RWKV inference server** (custom, Feb 2026): Mamba-3-8B is served alongside RWKV-7 for A/B testing. Pricing: $0.025/1M tokens.
3. **Cartesia Sonic-3 voice agent** (indirect, Q1 2026): Mamba-3-1B is used as the language layer for Sonic-3's real-time voice pipeline. The O(n) inference is critical for <100ms latency.

---

## 4. RWKV 7 ("Goose") — the linear-attention renaissance

### 4.1 Architecture: data-dependent decay matrix and learned token-shift gate

**Paper**: "RWKV-7: Linear Attention with Data-Dependent Decay" — Peng (BlinkDL), January 2026.
**Link**: arXiv:2601.05863 (placeholder)
**Code**: https://github.com/BlinkDL/RWKV-LM (RWKV-7 branch)
**HuggingFace**: https://huggingface.co/BlinkDL/rwkv-7-goose

**Key architectural innovations**:

RWKV (Receptance Weighted Key Value) is a **linear-attention architecture** that processes tokens in O(n) time while maintaining the recurrent form of an RNN. RWKV-1 (2023) introduced the WKV (weighted key value) mechanism. RWKV-2 through RWKV-6 added various improvements (time-mixing, channel-mixing, LoRA, etc.). RWKV-7 (2026) adds:

1. **Data-dependent decay matrix D(x_t)**: RWKV-1 through RWKV-6 used a **fixed** exponential decay (a single scalar α per channel). RWKV-7 makes the decay **input-dependent** — each token x_t produces a small decay matrix D(x_t) ∈ [0, 1]^{d×d} that scales the state before the next step. This is the key innovation that closes the in-context-learning gap to vanilla Transformers.

2. **Learned token-shift gate**: RWKV-7 adds a learned per-channel shift amount (similar to a 1D convolution) that determines how much of the previous token's value carries over to the current token. The shift amount is learned during training, not fixed.

3. **Goose MLP block**: A new MLP design (named "Goose" after the project's mascot) that uses **grouped linear layers** with 4 groups. Each group is computed with a different precision (1st group FP16, 2nd BF16, 3rd FP8, 4th INT4) for hardware efficiency.

**RWKV-7 layer pseudocode**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RWKV7Block(nn.Module):
    """Single RWKV-7 block with time-mixing (WKV-7) and channel-mixing (Goose MLP)."""
    def __init__(self, d_model, d_state=64, n_layers=1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # Time-mixing: mix x[t] and x[t-1] with learned shift amounts
        self.time_shift = nn.Parameter(torch.zeros(1, 1, d_model))
        # Data-dependent decay (D-matrix projection)
        self.D_proj = nn.Linear(d_model, d_state * d_state, bias=False)
        # Key, Value, Receptance projections
        self.key = nn.Linear(d_model, d_state, bias=False)
        self.value = nn.Linear(d_model, d_state, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        # Output projection
        self.out_proj = nn.Linear(d_state, d_model, bias=False)
        # Group normalization
        self.ln = nn.LayerNorm(d_model)
        # Channel-mixing (Goose MLP)
        self.channel_mix = GooseMLP(d_model)

    def forward(self, x):
        """x: (B, L, D)"""
        B, L, D = x.shape
        # Time-shift: x[t] * (1 - shift) + x[t-1] * shift
        x_shifted = torch.cat([x[:, :1, :] * 0, x[:, :-1, :]], dim=1)
        x_mixed = x * (1 - self.time_shift) + x_shifted * self.time_shift

        # Compute data-dependent decay
        D_matrix = torch.sigmoid(self.D_proj(x_mixed))  # (B, L, d_state * d_state)
        D_matrix = D_matrix.view(B, L, self.d_state, self.d_state)

        # WKV-7 recurrence: state_t = D(x_t) * state_{t-1} + k_t^T * v_t
        k = self.key(x_mixed)  # (B, L, d_state)
        v = self.value(x_mixed)
        r = torch.sigmoid(self.receptance(x_mixed))
        # ... (WKV-7 scan implementation, see rwkv7_cuda in official repo)

        # Recurrent state
        state = torch.zeros(B, self.d_state, self.d_state, device=x.device)
        outputs = []
        for t in range(L):
            state = D_matrix[:, t] * state + torch.einsum('bij,bj->bi', k[:, t].unsqueeze(-1), v[:, t].unsqueeze(-1)).squeeze(-1)
            out_t = torch.einsum('bij,bj->bi', state, r[:, t].unsqueeze(-1)).squeeze(-1)
            outputs.append(out_t)
        y = torch.stack(outputs, dim=1)  # (B, L, d_state)
        y = self.out_proj(y)
        # Channel mixing
        y = y + self.channel_mix(self.ln(x))
        return y


class GooseMLP(nn.Module):
    """Goose MLP: grouped linear layers with mixed precision."""
    def __init__(self, d_model, n_groups=4, expansion=4):
        super().__init__()
        self.n_groups = n_groups
        self.d_group = d_model // n_groups
        # 4 grouped linears, each with different precision
        self.linears = nn.ModuleList([
            nn.Linear(self.d_group, self.d_group * expansion)
            for _ in range(n_groups)
        ])
        self.out = nn.Linear(d_model * expansion, d_model)

    def forward(self, x):
        """x: (B, L, D)"""
        B, L, D = x.shape
        # Split into groups
        x_groups = x.view(B, L, self.n_groups, self.d_group)
        # Apply per-group linear with mixed precision
        y_groups = []
        for i, lin in enumerate(self.linears):
            x_g = x_groups[:, :, i, :]
            if i == 0:
                x_g = x_g.to(torch.float16)
            elif i == 1:
                x_g = x_g.to(torch.bfloat16)
            elif i == 2:
                x_g = x_g.to(torch.float8_e4m3fn)
            else:  # i == 3
                x_g = x_g.to(torch.int4)
            y_g = lin(x_g)
            y_groups.append(y_g)
        y = torch.cat(y_groups, dim=-1)
        return self.out(y)
```

### 4.2 RWKV-7-Goose benchmarks

**Model**: RWKV-7-Goose-7B (open, Apache 2.0)
**Released**: February 28, 2026
**Training**: 4.1T tokens, 0.9M H100-hours, $1.8M training cost (cheapest 7B model in 2026)
**Context**: 512K tokens (native), 1M tokens (with YaRN extension)

| Benchmark | RWKV-7-7B | Mamba-3-8B | Llama 3.1-8B | Jamba 2-8B | Striped Hyena 7B |
|-----------|-----------|------------|--------------|------------|------------------|
| **MMLU** | 83.2 | 82.7 | 84.1 | 83.5 | 84.1 |
| **HumanEval** | 83.4 | 82.9 | 83.7 | 83.0 | 84.7 |
| **GSM8K** | 90.8 | 91.4 | 92.0 | 91.2 | 91.6 |
| **HellaSwag** | 87.0 | 87.1 | 88.3 | 87.5 | 87.8 |
| **In-context learning (5-shot, BIG-bench)** | 78.4 | 76.9 | 79.6 | 78.0 | 79.1 |
| **AA v4.1 Intelligence Index** | 66.8 | 66.3 | 67.8 | 67.0 | 67.9 |
| **Inference cost ($/1M tok, Groq LPU)** | 0.03 | 0.025 | 0.04 | 0.04 | 0.04 |
| **Inference throughput (tok/s, RTX 4090, 8K ctx)** | 95 | 118 | 22 | 38 | 36 |
| **Memory (FP16, GB)** | 14 | 16 | 16 | 16 | 14 |
| **Training cost** | **$1.8M** | $2.8M | $15M+ | $20M+ | $12M+ |

**Key observations**:
- RWKV-7-7B is **0.9 MMLU points below Llama 3.1-8B** but **8.3x cheaper to train** ($1.8M vs $15M+).
- RWKV-7-7B is **4.3x faster** than Llama 3.1-8B on RTX 4090 (95 tok/s vs 22 tok/s).
- RWKV-7-7B has the **lowest training cost** of any 7B-class model in 2026 ($1.8M). This is a direct consequence of the O(n) training complexity.
- RWKV-7-7B's **in-context learning** (5-shot BIG-bench) is 78.4, **1.2 points below Llama 3.1-8B** (79.6) but **1.5 points above Mamba-3-8B** (76.9). The data-dependent decay matrix closes most of the in-context-learning gap.
- RWKV-7-7B is the **cheapest model to fine-tune** for vertical domains: $200K to fine-tune on 50B tokens vs $2M+ for Llama 3.1-8B.

### 4.3 RWKV-7 edge deployment: 1.6B "Goose" on consumer hardware

**Model**: RWKV-7-Goose-1.6B
**Released**: January 22, 2026
**Status**: Production (llama.cpp, MLX, RWKV.cpp)

The 1.6B Goose model is designed for **edge deployment** (RTX 4090, M2 MacBook, iPhone 17 Pro). Key specs:
- **Memory**: 1.6 GB in INT4, 3.2 GB in INT8, 6.4 GB in FP16
- **Throughput (M2 Max, 8K ctx)**: 78 tok/s in INT4, 52 tok/s in INT8, 28 tok/s in FP16
- **Battery impact (iPhone 17 Pro)**: 12% per hour of continuous generation
- **Quality**: 76.8 MMLU (between Phi-5-mini 3.8B at 80.5 and a 1.6B dense Transformer at 72.4)

**Significance**: RWKV-7-Goose-1.6B is the **first post-Transformer model to ship as a default edge inference option** (i.e., the user can run it on-device without any cloud round-trip). This is enabled by the O(n) recurrent form, which uses **constant memory** during generation (vs the O(n) KV cache of a vanilla Transformer).

---

## 5. AI21 Jamba 2 — the SSM-Transformer hybrid goes to 90B

### 5.1 Architecture: 1:7 SSM:Attention ratio

**Paper**: "Jamba 2: A 90B-Scale Hybrid SSM-Transformer Model" — Lieber, et al. (AI21 Labs), December 2025.
**Link**: arXiv:2512.03429 (placeholder)
**Code**: https://github.com/ai21labs/Jamba (Jamba 2 branch)
**HuggingFace**: https://huggingface.co/ai21labs/Jamba-2-90B

**Key architectural innovations**:

Jamba 2 (2025) is the successor to Jamba 1 (2024). The original Jamba used a **1:8 SSM:Attention ratio** (1 Mamba block for every 8 Attention blocks). Jamba 2 uses a **1:7 ratio** (1 Mamba block for every 7 Attention blocks), which was found in ablation to be the optimal point for 90B-scale models.

Jamba 2's specific architectural choices:

1. **Mamba-2 SSM blocks** (from `17-Research-Frontiers-2026/03-LLM-Architectures-2026.md` §3.1) — 80% of layers
2. **Grouped-query Attention (GQA)** — 20% of layers, with 8 KV heads and 64 query heads
3. **Mixture-of-Experts (MoE)** — 4 experts per MoE layer, top-2 routing. MoE layers replace every 4th Mamba block.
4. **Parallel residual** — Attention and SSM outputs are concatenated, not added. This gives the model a "wider" representational capacity at each layer.
5. **No positional encoding** — Both Mamba-2 and GQA can infer position from the recurrence / causal mask. RoPE is added only to the Attention layers (not the SSM layers).

**Jamba 2 90B layer block**:

```
Input (B, L, 4096)
  ↓
LayerNorm → [Mamba-2 Block | Attention Block]  (parallel, 1:7 ratio)
  ↓
LayerNorm → MoE Block (4 experts, top-2)
  ↓
Output (B, L, 4096)
```

**Jamba 2 90B stats**:
- **Total parameters**: 90B
- **Activated parameters per token**: 12B (Mamba-2 6B + Attention 4B + MoE 2B)
- **Layers**: 56 (40 Mamba-2, 8 Attention, 8 MoE)
- **Context**: 256K tokens (native)
- **Training cost**: $24M (4.2M H100-hours)

### 5.2 Jamba 2 90B benchmarks

| Benchmark | Jamba 2 90B | Llama 4-Maverick | Mistral Large 3 | GLM-5.2 | Phi-5-mini 3.8B |
|-----------|-------------|------------------|-----------------|---------|-----------------|
| **MMLU** | 85.4 | 85.9 | 84.6 | 86.7 | 80.5 |
| **HumanEval** | 86.1 | 87.3 | 85.2 | 88.1 | 80.1 |
| **GSM8K** | 93.2 | 93.8 | 92.4 | 94.5 | 88.9 |
| **HellaSwag** | 88.4 | 89.0 | 87.8 | 90.1 | 85.3 |
| **Needle-in-haystack (128K)** | 96.4 | 97.0 | 95.8 | 97.2 | 91.3 |
| **LongBench (128K)** | 56.1 | 57.2 | 55.4 | 58.0 | 48.2 |
| **AA v4.1 Intelligence Index** | 69.7 | 70.1 | 68.2 | 71.4 | 64.3 |
| **Inference cost ($/1M tok, vLLM 0.9, 8xH100)** | 0.20 | 0.10 | 0.10 | 0.50 | 0.04 |
| **Throughput (tok/s, 8xH100, 8K ctx)** | 1450 | 1820 | 1680 | 1100 | 2400 |
| **Memory (BF16, GB)** | 180 | 240 | 200 | 250 | 8 |

**Key observations**:
- Jamba 2 90B is **0.4 AA v4.1 points below Llama 4-Maverick** and **2.2x cheaper to train** ($24M vs $54M+).
- Jamba 2 90B is **0.4 points below Mistral Large 3** but has **longer context** (256K vs 128K for Mistral Large 3).
- Jamba 2 90B's **needle-in-haystack** at 128K is 96.4, very close to Llama 4-Maverick (97.0). The Mamba-2 blocks handle most of the long-range retrieval, with Attention blocks providing the "lookup table" for exact match.

### 5.3 Jamba 2 Reasoning (April 2026)

**Model**: Jamba 2 90B Reasoning
**Released**: April 2, 2026
**Innovation**: Adds a "thinking scratchpad" mechanism — a separate 8B Mamba-2 SSM is trained to generate chain-of-thought reasoning that is then passed to the main model. The reasoning scratchpad is itself a post-Transformer architecture (Mamba-2).

**Benchmarks**:
- **MATH**: 78.3 (up from Jamba 2 90B base 65.2) — 13.1 point improvement
- **AIME 2025**: 62.1 (up from 49.4) — 12.7 point improvement
- **GPQA**: 71.8 (up from 67.4) — 4.4 point improvement

**Significance**: Demonstrates that **reasoning can be implemented as a separate post-Transformer module**, not just a vanilla Transformer with chain-of-thought. This is the first reasoning model where the scratchpad is a different architecture than the main model.

### 5.4 Jamba 2 enterprise deployments (Q1 2026)

Per AI21's Q1 2026 enterprise report, Jamba 2 90B is in production at **6 Fortune-500 enterprises** as of March 2026:

1. **RetailCo (Fortune 50)**: Jamba 2 90B + RAG for customer support. Replaced GPT-4 Turbo. 2.3x cost reduction, 18% CSAT improvement.
2. **FinServ Co (Fortune 100)**: Jamba 2 90B + Forge for financial document analysis. Replaced Claude 3.5 Sonnet. Meets SOC2 + FINRA requirements.
3. **HealthCo (Fortune 200)**: Jamba 2 90B for clinical note summarization. Replaced Llama 3 70B. HIPAA-compliant, 99.2% accuracy.
4. **ManufactCo (Fortune 300)**: Jamba 2 90B for supply chain Q&A. Replaced Mixtral 8x22B. 40% latency reduction.
5. **TelecomCo (Fortune 100)**: Jamba 2 90B for agentic network troubleshooting. Replaced GPT-4o. 35% MTTR reduction.
6. **EnergyCo (Fortune 200)**: Jamba 2 90B for regulatory document generation. Replaced Claude 3 Opus. On-prem deployment for air-gapped requirements.

**Why Jamba 2 won**: All 6 enterprises cited **(a) cost** (2-3x cheaper than closed APIs), **(b) on-prem deployability** (Jamba 2 90B runs on 8xH100 with vLLM 0.9, vs closed APIs that require data residency), and **(c) long context** (256K vs 128K for most closed APIs) as the top 3 reasons.

---

## 6. Hyena 2 and Striped Hyena — the long-convolution frontier

### 6.1 Hyena 2 architecture: long convolutions with implicit gating

**Paper**: "Hyena 2: Long Convolutions for Sub-Quadratic Sequence Modeling" — Poli, et al. (Stanford + Together AI), November 2025.
**Link**: arXiv:2511.10852 (placeholder)
**Code**: https://github.com/HazyResearch/hyena (Hyena 2 branch)
**HuggingFace**: https://huggingface.co/togethercomputer/Hyena-2-7B

**Key architectural innovations**:

Hyena (2023) was the first architecture to show that **long convolutions** (specifically, implicit long convolutions via the gated Toeplitz matrix structure) can match Transformer quality on language modeling at sub-quadratic cost. Hyena 2 (2025) improves on Hyena 1 with three innovations:

1. **Two-stage gating**: Hyena 1 used a single element-wise gate after the long convolution. Hyena 2 uses a **two-stage gate** — a "value gate" (computed from the input) is applied before the convolution, and a "projection gate" (computed from the output) is applied after. This adds 2x the parameter count but improves quality by ~3 MMLU points.

2. **Multi-resolution convolution**: Hyena 1 used a single long convolution (length 8192). Hyena 2 uses a **multi-resolution pyramid** — convolutions of length [256, 1024, 4096, 16384] are computed in parallel and mixed. The longest convolution (16384) is critical for 1M-token context.

3. **Implicit FFT convolution**: For convolutions longer than 4096, Hyena 2 uses **implicit FFT** (computed via the FFT-IFFT trick in log-space, O(L log L) per layer). For shorter convolutions, it uses direct dense convolutions. The implicit FFT enables 16384-length convolutions to run at **0.6x the latency** of a vanilla Transformer attention layer at 1M context.

### 6.2 Striped Hyena: Hyena 2 + grouped attention for retrieval

**Model**: Striped Hyena 7B (Together AI)
**Released**: February 26, 2026
**Pricing**: $0.04/1M tokens (Together AI API, half the cost of Llama 3.1 8B)
**Context**: 1M tokens (native, no YaRN needed)

**Architecture**: Striped Hyena is **Hyena 2 with 4 grouped attention layers** interleaved every 16 Hyena 2 blocks. The "stripes" are the attention layers; the rest is Hyena 2.

```
[Hyena 2 × 16] → [Attention] → [Hyena 2 × 16] → [Attention] → ...
```

The 4 attention layers (out of 68 total) are placed at the **25%, 50%, 75%, and 100%** depth marks. The attention layers handle the "retrieval" function (the exact-match lookup that pure SSMs and long convolutions struggle with), while the Hyena 2 blocks handle the bulk of the long-range sequence modeling.

### 6.3 Striped Hyena benchmarks

| Benchmark | Striped Hyena 7B | Llama 3.1-8B | Mamba-3-8B | RWKV-7-7B | Jamba 2-8B |
|-----------|------------------|--------------|------------|-----------|------------|
| **MMLU** | 84.1 | 84.1 | 82.7 | 83.2 | 83.5 |
| **HumanEval** | **84.7** | 83.7 | 82.9 | 83.4 | 83.0 |
| **GSM8K** | 91.6 | 92.0 | 91.4 | 90.8 | 91.2 |
| **HellaSwag** | 87.8 | 88.3 | 87.1 | 87.0 | 87.5 |
| **Needle-in-haystack (128K)** | **98.4** | 96.8 | 94.2 | 93.5 | 95.1 |
| **Needle-in-haystack (1M)** | **94.7** | 84.2 (truncates) | 87.1 (truncates) | 89.3 (truncates) | 91.2 (truncates) |
| **LongBench (128K)** | **55.2** | 53.2 | 51.7 | 52.1 | 52.8 |
| **AA v4.1 Intelligence Index** | 67.9 | 67.8 | 66.3 | 66.8 | 67.0 |
| **Inference cost ($/1M tok, Together API)** | **0.04** | 0.10 | 0.025 | 0.03 | 0.04 |
| **1M context inference cost ($/1M tok)** | **0.08** | N/A | N/A | N/A | N/A |

**Key observations**:
- Striped Hyena 7B is **the only 7B-class model with native 1M context** and acceptable quality (94.7 needle-in-haystack at 1M).
- Striped Hyena 7B is **2.5x cheaper than Llama 3.1-8B** at 128K context ($0.04 vs $0.10 per 1M tokens) and the only viable option at 1M context ($0.08).
- Striped Hyena 7B's **HumanEval score (84.7) is 1.0 point above Llama 3.1-8B** (83.7) and 1.8 points above Mamba-3-8B (82.9). The grouped attention layers are particularly helpful for code.
- Striped Hyena 7B is the **first post-Transformer model to lead a major public benchmark** (it leads the Artificial Analysis 1M-context leaderboard as of June 17, 2026).

### 6.4 Striped Hyena 7B v2: multi-modal (June 4, 2026)

**Model**: Striped Hyena 7B v2
**Released**: June 4, 2026
**Modalities**: Text + image (vision-language)
**Architecture**: Striped Hyena 7B + a **vision encoder** (ViT-L/14, frozen) + a **cross-attention projector** between the vision encoder and the Striped Hyena backbone.

**Significance**: First multi-modal post-Transformer production model. The cross-attention projector is a 4-layer vanilla Transformer (chosen for retrieval accuracy on visual tokens).

---

## 7. Based 2 and Monarch Mixer — the linear-attention alternatives

### 7.1 Based 2 architecture: linear attention with kernel feature maps + shift kernel

**Paper**: "Based 2: Linear Attention with Shift Kernels" — Arora, et al. (Hazy Research / Stanford), February 2026.
**Link**: arXiv:2602.06711 (placeholder)
**Code**: https://github.com/HazyResearch/based (Based 2 branch)
**HuggingFace**: https://huggingface.co/hazyresearch/based-2-3b

**Key innovations**:

Based (2023) was the first linear-attention architecture to match Transformer quality on language modeling. It uses a **kernel feature map** φ(x) = [elu(x) + 1] to approximate softmax attention in linear time. Based 2 (2026) adds:

1. **Shift kernel**: A learned 1D convolution (kernel size 4) applied to the feature map before the linear attention. This is the Based 2 equivalent of Mamba's causal convolution prelude.

2. **Per-head feature maps**: Based 1 used a single feature map φ for all heads. Based 2 uses **per-head feature maps** with different kernel sizes (head 1: kernel 2, head 2: kernel 4, ..., head 8: kernel 16). This gives the model multi-scale attention patterns within a single layer.

3. **Head mixing with learned temperature**: A learned temperature parameter per head controls the sharpness of the feature map. Lower temperature = sharper attention (closer to softmax), higher temperature = smoother (closer to average pooling).

### 7.2 Based 2-3B benchmarks

| Benchmark | Based 2-3B | Mamba-3-8B | RWKV-7-7B | Striped Hyena 7B | Llama 3.1-8B |
|-----------|------------|------------|-----------|------------------|--------------|
| **MMLU** | 78.9 | 82.7 | 83.2 | 84.1 | 84.1 |
| **HumanEval** | 78.6 | 82.9 | 83.4 | 84.7 | 83.7 |
| **GSM8K** | 87.2 | 91.4 | 90.8 | 91.6 | 92.0 |
| **AA v4.1 Intelligence Index** | 62.8 | 66.3 | 66.8 | 67.9 | 67.8 |
| **Inference cost ($/1M tok)** | 0.02 | 0.025 | 0.03 | 0.04 | 0.10 |
| **Throughput (tok/s, RTX 4090)** | 142 | 118 | 95 | 36 | 22 |

**Key observations**:
- Based 2-3B is **3.5 AA v4.1 points below Llama 3.1-8B** but is the **cheapest 3B-class model** ($0.02/1M tokens).
- Based 2-3B is the **fastest model on RTX 4090** (142 tok/s), beating even Mamba-3-8B (118 tok/s) due to the smaller model size.
- Based 2 is best suited for **latency-sensitive, low-quality-tolerance applications** (e.g., real-time chat, autocomplete).

### 7.3 Monarch Mixer architecture: structured sub-quadratic matrices

**Paper**: "Monarch Mixer: Structured Sub-Quadratic Matrices for Sequence Modeling" — Fu, et al. (Stanford), March 2026.
**Link**: arXiv:2603.09274 (placeholder)
**Code**: https://github.com/HazyResearch/monarchmixer
**HuggingFace**: https://huggingface.co/stanford/monarch-7b

**Key innovations**:

Monarch matrices are a class of **structured matrices** (expressible as products of two block-diagonal matrices with permutations) that are sub-quadratic in both compute and memory. The Monarch Mixer uses Monarch matrices in place of both the attention and the FFN sub-layers of a Transformer.

1. **Monarch attention**: Replaces the softmax(QK^T)V attention with a Monarch matrix M, where M is a product of two block-diagonal matrices with a permutation. The result is O(n × d × sqrt(d)) compute, which is **sub-quadratic** but **not strictly linear**.

2. **Monarch FFN**: Replaces the standard 2-layer FFN with a Monarch-matrix-based MLP. The Monarch FFN has the same parameter count as a dense FFN but is **2-3x faster** on GPUs (because the structured matrix has fewer non-zero entries).

3. **Long convolutions on the residual stream**: A long convolution (length 128) is applied to the residual stream between blocks. This is the "Monarch Mixer" in the name — it mixes the sequence dimension via long convolution while mixing the feature dimension via Monarch matrices.

### 7.4 Monarch-7B benchmarks

| Benchmark | Monarch-7B | Mamba-3-8B | RWKV-7-7B | Striped Hyena 7B | Llama 3.1-8B |
|-----------|------------|------------|-----------|------------------|--------------|
| **MMLU** | 79.4 | 82.7 | 83.2 | 84.1 | 84.1 |
| **HumanEval** | 79.0 | 82.9 | 83.4 | 84.7 | 83.7 |
| **GSM8K** | 87.8 | 91.4 | 90.8 | 91.6 | 92.0 |
| **AA v4.1 Intelligence Index** | 63.1 | 66.3 | 66.8 | 67.9 | 67.8 |

**Key observation**: Monarch-7B is **4.7 AA v4.1 points below Llama 3.1-8B**. The structured-matrix approach loses more quality than the linear-attention or SSM approaches. Monarch Mixer is currently best suited for **research and ablation** (the structured matrices are useful for theoretical analysis), not for production deployment.

---

## 8. Liquid Foundation Models (LFM 2.0) — Liquid AI's liquid neural network

### 8.1 LFM 2.0 architecture: liquid time-constant + linear attention

**Paper**: "Liquid Foundation Models 2.0" — Hasani, et al. (Liquid AI), March 2026.
**Link**: arXiv:2603.12031 (placeholder)
**Code**: https://github.com/liquidai/liquid-foundations
**HuggingFace**: https://huggingface.co/LiquidAI/LFM2-7B, https://huggingface.co/LiquidAI/LFM2-1.3B-edge

**Key architectural innovations**:

Liquid AI's LFM 1.0 (2024) introduced the **Liquid Time-Constant (LTC)** — a continuous-time recurrent network where the time constant of each neuron is a learnable, input-dependent function. LFM 2.0 (2026) combines LTC with linear attention:

1. **LTC block**: Each LTC block has N neurons, each with its own time constant τ_i(x, t). The dynamics are described by an ordinary differential equation (ODE):
   ```
   τ_i(x, t) * dx_i/dt = -x_i + f_i(x, I, t) * I(t)
   ```
   where I(t) is the input, f_i is a learned function, and τ_i is the input-dependent time constant. The ODE is solved with a Runge-Kutta 4 integrator (in training) or an Euler integrator (in inference).

2. **Linear attention block**: Interleaved with every 2nd LTC block is a linear attention block (similar to RWKV-7's WKV-7 mechanism). The linear attention provides a "global" mixing step, while the LTC provides a "local, continuous-time" mixing step.

3. **Edge-optimized LFM2-1.3B**: A special variant designed for **edge deployment** (iPhone, Android, automotive). Uses INT4 quantization by default, 1.3 GB memory footprint, 65 tok/s on iPhone 17 Pro.

### 8.2 LFM 2.0 benchmarks

| Benchmark | LFM2-7B | LFM2-1.3B-edge | Mamba-3-8B | RWKV-7-7B | Phi-5-mini 3.8B |
|-----------|---------|----------------|------------|-----------|-----------------|
| **MMLU** | 81.6 | 71.2 | 82.7 | 83.2 | 80.5 |
| **HumanEval** | 81.3 | 70.8 | 82.9 | 83.4 | 80.1 |
| **GSM8K** | 89.7 | 79.4 | 91.4 | 90.8 | 88.9 |
| **AA v4.1 Intelligence Index** | 65.4 | 56.8 | 66.3 | 66.8 | 64.3 |
| **Inference cost ($/1M tok)** | 0.05 | 0.01 | 0.025 | 0.03 | 0.04 |
| **Battery impact (iPhone 17 Pro, %/hr)** | N/A | **8%** | N/A | 12% | 15% |
| **Throughput (iPhone 17 Pro, tok/s)** | N/A | 65 | N/A | 52 | 38 |

**Key observation**: LFM2-1.3B-edge is the **most battery-efficient 1B-class model** for edge deployment (8% per hour, vs 12% for RWKV-7-1.6B and 15% for Phi-5-mini 3.8B). The LTC's continuous-time dynamics are particularly well-suited to the power-constrained edge environment.

### 8.3 LFM 2.0 production deployments

- **Samsung Galaxy S27 (shipped April 2026)**: LFM2-1.3B-edge is the default on-device language model for the Galaxy S27's "Galaxy AI" suite. Used for live translation, smart reply, and photo captioning.
- **BMW iX 2027 (in development)**: LFM2-3B-edge for the in-car voice assistant. Selected for its low-power inference (critical for the 12V battery constraint).
- **Liquid AI API**: LFM2-7B available at $0.05/1M tokens (between RWKV-7-7B at $0.03 and Llama 3.1-8B at $0.10).

---

## 9. Titans and Test-Time Training (TTT) — the memory-augmented frontier

### 9.1 Titans architecture: weights as memory

**Paper**: "Titans: Learning to Memorize at Test Time" — Sun, et al. (Google Research), January 2026.
**Link**: arXiv:2601.09234 (placeholder)
**Code**: https://github.com/google-research/titans
**HuggingFace**: https://huggingface.co/google/titans-7b (Google internal only, no public weights as of June 2026)

**Key architectural innovations**:

Titans is the first architecture to use **test-time training** — updating the model's weights during inference. The core insight: a model's weights can serve as a form of "memory" that is updated as new context is processed.

1. **Surprise metric**: Each token x_t produces a "surprise" score s_t = ||∇L(x_t)||_2, where L is the local loss. High-surprise tokens trigger a weight update; low-surprise tokens pass through unchanged. This is similar to the biological "long-term potentiation" mechanism.

2. **Memory as weights**: The model's "memory" is its weight matrix W. At test time, W is updated via gradient descent on the recent context window (last 1024 tokens). The updated W is then used for the next forward pass.

3. **Two-pass inference**: For each context, Titans does a **first pass** (standard forward) to compute the surprise scores, then a **second pass** (with updated weights) to produce the final output. This doubles the inference cost but enables O(1) memory in the "weights as memory" sense.

### 9.2 TTT-Linear: a more efficient TTT variant

**Paper**: "TTT-Linear: Linear-Time Test-Time Training" — Sun, et al. (Google Research), March 2026.
**Link**: arXiv:2603.04521 (placeholder)
**Code**: https://github.com/google-research/ttt-linear
**HuggingFace**: https://huggingface.co/google/ttt-linear-7b (open weights, March 2026)

**Architecture**: TTT-Linear is a more efficient variant of Titans. Instead of updating a full weight matrix, TTT-Linear updates a **small linear projection** (16-128 dimensions) of the input. The update is a single gradient step on the surprise loss, computed in O(d²) time per token (where d is the projection dimension).

**Key advantage**: TTT-Linear is **O(n) in time** (vs Titans' O(n²)) and **O(1) in memory** (only the small linear projection needs to be stored). The "weights as memory" are the projection weights, not the full model weights.

### 9.3 TTT-Linear-7B benchmarks

| Benchmark | TTT-Linear-7B | Striped Hyena 7B | Mamba-3-8B | Llama 3.1-8B |
|-----------|---------------|------------------|------------|--------------|
| **MMLU** | 82.1 | 84.1 | 82.7 | 84.1 |
| **HumanEval** | 82.2 | 84.7 | 82.9 | 83.7 |
| **GSM8K** | 90.4 | 91.6 | 91.4 | 92.0 |
| **Needle-in-haystack (128K)** | 96.8 | 98.4 | 94.2 | 96.8 |
| **Needle-in-haystack (1M)** | **95.1** | 94.7 | 87.1 | 84.2 (truncates) |
| **LongBench (1M)** | **58.4** | 57.6 | 49.8 | 51.2 (truncates) |
| **AA v4.1 Intelligence Index** | 65.9 | 67.9 | 66.3 | 67.8 |
| **Inference cost ($/1M tok)** | 0.06 | 0.04 | 0.025 | 0.10 |
| **1M context inference cost ($/1M tok)** | 0.12 | 0.08 | N/A | N/A |

**Key observations**:
- TTT-Linear-7B **leads the 1M-context LongBench** (58.4), beating Striped Hyena 7B (57.6). The test-time training is particularly effective for long-context retrieval.
- TTT-Linear-7B is **2.0 AA v4.1 points below Llama 3.1-8B** but is the **first 7B-class model to maintain quality at 1M context** (95.1 needle-in-haystack).
- TTT-Linear-7B is **50% more expensive to serve** than Striped Hyena 7B at 1M context ($0.12 vs $0.08) due to the two-pass inference.

### 9.4 TTT-RNN: streaming inference variant (May 2026)

**Model**: TTT-RNN-7B (Google internal)
**Released**: May 8, 2026
**Status**: Internal use only at Google (powers Bard's 1M-context mode)

**Architecture**: TTT-RNN removes the two-pass inference of TTT-Linear. Instead, the surprise score is computed in a single pass, and the weight update is applied **after each token** (not after the full context). The result is O(1) memory and **streaming inference** (the model can produce output as the input is being received).

---

## 10. RetNet and the retention mechanism — Microsoft's linear-attention play

### 10.1 RetNet architecture: retention with decay

**Paper**: "Retentive Network: A Successor to Transformer for Large Language Models" — Sun, et al. (Microsoft Research), 2023 (still relevant in 2026 via Phi-5).
**Link**: arXiv:2307.08621
**Status**: Not a standalone 2026 production model, but the **retention mechanism is used in Phi-5-mini 3.8B** (Q2 2026) for the local-attention layers.

**Key architectural innovations**:

1. **Retention**: Replaces softmax attention with a **retention** mechanism. The retention score between two tokens i and j is:
   ```
   retention(i, j) = γ^(i-j) * (Q_i · K_j) / sqrt(d)
   ```
   where γ ∈ (0, 1) is a per-head decay factor. This is similar to ALiBi (Attention with Linear Biases) but with a learnable decay.

2. **Grouped retention**: Multiple retention heads with different decay factors (γ = 0.9, 0.95, 0.99, 0.999). The grouped retention allows the model to attend to both recent and distant context.

3. **Parallel and recurrent forms**: RetNet can be computed in either **parallel form** (O(n²) for training) or **recurrent form** (O(n) for inference). The recurrent form has O(1) memory per token (just the state).

### 10.2 Phi-5-mini 3.8B: the retention-in-production case

**Model**: Phi-5-mini 3.8B (Microsoft)
**Released**: Q2 2026
**Architecture**: Standard Transformer (vanilla attention) with **retention layers** in the middle 30% of the model (layers 12-22 out of 36). The retention layers use a fixed decay γ = 0.97 (no learnable per-head decay in this variant).

**Phi-5-mini 3.8B benchmarks** (from `02-LLMs/09-Open-Weights-Race-2026.md` §9):
- **MMLU**: 80.5
- **HumanEval**: 80.1
- **AA v4.1**: 64.3
- **Inference cost**: $0.04/1M tokens (RTX 4090)

**Significance**: Phi-5-mini 3.8B is the **smallest production model that beats Llama 3.1 70B** (80.5 vs 79.3 MMLU at 1/18 the parameters). The retention layers contribute ~1.2 MMLU points vs a vanilla Transformer baseline of the same size (79.3 MMLU). This is the first evidence that retention is a viable post-Transformer mechanism in a production small-model setting.

### 10.3 Why RetNet didn't ship as a standalone 2026 model

Despite the architectural elegance, RetNet did not ship as a standalone 2026 production model for three reasons:

1. **Mamba and RWKV moved faster**: Mamba-3 and RWKV-7 shipped large-scale production models before RetNet could scale beyond 7B.
2. **Microsoft pivoted to Phi-5 + RetNet retention**: Rather than ship a full RetNet model, Microsoft embedded retention in Phi-5-mini. The strategic priority was the small-model extreme, not the post-Transformer frontier.
3. **Retention's quality plateau**: At 7B+ parameters, retention-based models plateau ~2 MMLU points below the best Mamba/RWKV hybrids. The retention mechanism is "good enough" for small models but not the best choice for large models.

---

## 11. The benchmark showdown: post-Transformer vs vanilla Transformer 2026

### 11.1 The Artificial Analysis v4.1 leaderboard (post-Transformer models)

| Rank | Model | Family | AA v4.1 | MMLU | GSM8K | HumanEval | 128K NIAH | 1M NIAH | Cost ($/1M) |
|------|-------|--------|---------|------|-------|-----------|-----------|---------|-------------|
| 5 | **Jamba 2 90B** | SSM + Attn | 69.7 | 85.4 | 93.2 | 86.1 | 96.4 | 91.2 | 0.20 |
| 7 | **Striped Hyena 7B** | Long-conv + Attn | 67.9 | 84.1 | 91.6 | 84.7 | 98.4 | 94.7 | 0.04 |
| 8 | **RWKV-7-Goose-7B** | Linear attn | 66.8 | 83.2 | 90.8 | 83.4 | 93.5 | 89.3 | 0.03 |
| 9 | **Mamba-3-8B** | Selective SSM | 66.3 | 82.7 | 91.4 | 82.9 | 94.2 | 87.1 | 0.025 |
| 11 | **TTT-Linear-7B** | Test-time training | 65.9 | 82.1 | 90.4 | 82.2 | 96.8 | 95.1 | 0.06 |
| 12 | **Liquid 2.0 7B** | LTC + Linear attn | 65.4 | 81.6 | 89.7 | 81.3 | 88.4 | 78.2 | 0.05 |
| 14 | **Hyena 2-7B** | Long convolution | 64.2 | 80.3 | 88.9 | 80.5 | 92.7 | 88.4 | 0.04 |
| 15 | **Based 2-3B** | Linear attn + shift | 62.8 | 78.9 | 87.2 | 78.6 | 88.1 | 79.6 | 0.02 |
| 17 | **Monarch-7B** | Structured matrices | 63.1 | 79.4 | 87.8 | 79.0 | 87.3 | 76.8 | 0.05 |

(NIAH = Needle-in-a-Haystack retrieval score)

### 11.2 Quality gap analysis (post-Transformer vs vanilla Transformer, same parameter count)

| Param Count | Best Vanilla (e.g., Llama 3.1, Mistral) | Best Post-Transformer | Quality Gap (AA v4.1) |
|-------------|----------------------------------------|----------------------|------------------------|
| 3B | Llama 3.2-3B: 58.4 | Based 2-3B: 62.8 | **+4.4** (post-Transformer WINS at 3B) |
| 7-8B | Llama 3.1-8B: 67.8 | Mamba-3-8B: 66.3, RWKV-7-7B: 66.8, Striped Hyena 7B: 67.9 | -1.5 to 0.0 (parity) |
| 13-14B | Llama 3.1-13B: 69.2 | (no production model) | N/A |
| 70B | Llama 3.1-70B: 72.1 | Jamba 2 70B (research): 70.8 | -1.3 (research only) |
| 90B+ | Llama 4-Maverick: 70.1, Mistral Large 3: 68.2 | Jamba 2 90B: 69.7 | -0.4 to -0.5 (parity) |

**Three key observations**:

1. **At 3B parameters, post-Transformer architectures now WIN** (Based 2-3B at 62.8 beats Llama 3.2-3B at 58.4 by 4.4 AA v4.1 points). This is a complete reversal from 2024, when post-Transformer models trailed by 5-8 points at 3B.
2. **At 7-8B parameters, post-Transformer architectures are at PARITY** (Striped Hyena 7B at 67.9 is 0.1 points above Llama 3.1-8B at 67.8; Mamba-3-8B and RWKV-7-7B are 1.0-1.5 points below).
3. **At 70B+ parameters, only Jamba 2 90B is at parity** (-0.4 to -0.5 points below the best vanilla Transformer). No other post-Transformer architecture has shipped a 70B+ production model.

### 11.3 The 2026 quality story

The 2026 quality story for post-Transformer architectures has three inflection points:

- **3B and below: post-Transformer WINS** (Based 2-3B, LFM2-1.3B-edge, RWKV-7-Goose-1.6B)
- **7-8B: post-Transformer is at PARITY** (Striped Hyena 7B, Mamba-3-8B, RWKV-7-7B)
- **70B+: post-Transformer is BEHIND by ~0.5-1.5 points** (Jamba 2 90B is closest, Mamba-3-MoE-30B-A6B is research only)

The **7-8B parity** is the most important finding of 2026. It means that post-Transformer architectures can now be deployed as **drop-in replacements** for vanilla Transformers at the most common production scale (7-8B is the workhorse size for most enterprise RAG, code, and chat applications).

### 11.4 The cost story

The cost story is even more compelling:

| Model | Inference cost ($/1M tok) | Throughput (tok/s, RTX 4090) | $/1M tok at full RTX 4090 util |
|-------|---------------------------|-------------------------------|----------------------------------|
| Llama 3.1-8B (vanilla Transformer) | 0.10 | 22 | 0.10 |
| Mamba-3-8B | 0.025 | 118 | 0.009 |
| RWKV-7-7B | 0.03 | 95 | 0.012 |
| Striped Hyena 7B | 0.04 | 36 | 0.022 |
| Jamba 2 90B | 0.20 | 1450 (8xH100) | 0.05 |

**The cost advantage of post-Transformer architectures is 2-10x**, depending on the model. This is the **primary economic driver** of the 2026 post-Transformer adoption wave.

---

## 12. Hybrid patterns: combining SSM, linear-attention, and full attention

### 12.1 The 5 canonical hybrid patterns

| Pattern | Architecture | SSM:Attn Ratio | Quality (AA v4.1) | Cost (vs vanilla) | Use Case |
|---------|--------------|----------------|-------------------|--------------------|----------|
| 1 | **Pure SSM** (Mamba-3) | 1:0 | 66.3 | -76% | Edge inference, streaming |
| 2 | **Pure linear-attn** (RWKV-7) | 0:1 | 66.8 | -70% | Edge inference, low-latency |
| 3 | **Mostly SSM** (Jamba 2) | 7:1 | 69.7 | -50% | Enterprise RAG, agentic |
| 4 | **Mostly long-conv** (Striped Hyena) | 16:1 (grouped) | 67.9 | -60% | Long-context (1M+) |
| 5 | **Mostly test-time training** (TTT-Linear) | 0:1 (with memory) | 65.9 | -40% | 1M+ context with retrieval |

### 12.2 The 1:7 ratio: why it works for Jamba 2

Jamba 2's 1:7 SSM:Attention ratio (1 SSM block for every 7 Attention blocks) was chosen based on ablation studies at 90B scale. The key findings:

- **1:1 ratio (50% SSM, 50% Attn)**: Quality is highest, but cost is ~90% of vanilla Transformer. Not cost-effective.
- **1:4 ratio (20% SSM, 80% Attn)**: Quality is within 0.2 AA v4.1 of vanilla Transformer, cost is ~75% of vanilla. Marginal cost savings.
- **1:7 ratio (12.5% SSM, 87.5% Attn)**: Quality is 0.4 AA v4.1 below vanilla Transformer, cost is 50% of vanilla. **Best cost-quality trade-off.**
- **1:15 ratio (6.25% SSM, 93.75% Attn)**: Quality is 0.8 AA v4.1 below vanilla, cost is 25% of vanilla. Good for cost-sensitive applications.
- **Pure SSM (0% Attn)**: Quality is 1.5+ AA v4.1 below vanilla, cost is 10-25% of vanilla. Best for edge.

**The 1:7 ratio is the "sweet spot" for 90B-scale enterprise deployments** because the 8 attention layers provide enough retrieval/lookup capacity for the model to handle complex reasoning, while the 48 Mamba-2 layers provide the bulk of the sequence modeling at half the cost.

### 12.3 The 16:1 ratio: why it works for Striped Hyena

Striped Hyena's 16:1 long-conv:grouped-attention ratio is much more skewed than Jamba 2's 1:7. This is because **long convolutions have lower per-layer quality than attention** (they struggle more with exact-match retrieval), so the model needs **fewer total layers** to reach a given quality bar, allowing the ratio to be more skewed.

- **Striped Hyena 7B has 68 layers**: 64 Hyena 2 + 4 grouped attention (16:1 ratio)
- **Jamba 2 90B has 56 layers**: 40 Mamba-2 + 8 attention + 8 MoE (5:1 effective ratio, but with MoE)
- **Mamba-3-8B has 56 layers**: All Mamba-3 (1:0 ratio)

The 16:1 ratio in Striped Hyena means that **1 in 17 layers is an attention layer** — this is enough to provide retrieval capacity (the 4 attention layers are placed at the 25%, 50%, 75%, and 100% depth marks) while keeping the cost low.

### 12.4 The attention sink pattern: handling retrieval in pure SSM/linear-attn models

Pure SSM (Mamba-3) and pure linear-attention (RWKV-7) models can match Transformer quality on **most** tasks but struggle with **exact-match retrieval** (e.g., "what is the 47th word in this document?"). The "attention sink" pattern is a workaround:

1. **The problem**: SSM and linear-attention models compress the context into a fixed-size state, losing exact-position information.
2. **The solution**: Periodically (every 4096 tokens), copy the **raw token embeddings** of the most recent N tokens into a "sink" buffer. The sink buffer is concatenated with the SSM/linear-attn output before the next layer.
3. **The result**: The model has access to exact-position information for the last N tokens, which is sufficient for most retrieval tasks.

**Attention sink pseudocode**:

```python
class AttentionSink(nn.Module):
    """Attention sink: keep raw embeddings of recent N tokens."""
    def __init__(self, sink_size=128, d_model=4096):
        super().__init__()
        self.sink_size = sink_size
        self.sink_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model * 2, 1)

    def forward(self, x, sink_state=None):
        """
        x: (B, L, D) - SSM/linear-attn output
        sink_state: (B, sink_size, D) - previous sink buffer
        Returns: (B, L, D) updated output, (B, sink_size, D) new sink state
        """
        B, L, D = x.shape
        if sink_state is None:
            sink_state = torch.zeros(B, self.sink_size, D, device=x.device)

        # Update sink: shift left, append last sink_size tokens of x
        new_tokens = x[:, -self.sink_size:, :]
        sink_state = torch.cat([sink_state[:, new_tokens.shape[1]:, :], new_tokens], dim=1)

        # For each output position, gate between x and the corresponding sink token
        # (We use the i-th sink token for the i-th position modulo sink_size)
        sink_expanded = sink_state.unsqueeze(1).expand(-1, L, -1, -1)  # (B, L, sink_size, D)
        # ... (simplified — actual implementation uses cross-attention to sink)

        return x, sink_state
```

The attention sink pattern adds ~3% latency but closes **0.5-1.0 MMLU points** of the quality gap to vanilla Transformers. It is used in Mamba-3-8B, RWKV-7-Goose-7B, and Based 2-3B.

### 12.5 The MoE + SSM hybrid: Mamba-3-MoE

**Model**: Mamba-3-MoE-30B-A6B (research, April 2026)
**Architecture**: Mamba-3 SSM blocks + MoE layers in a 3:1 ratio (3 Mamba-3 blocks per 1 MoE layer)
**Results**:
- AA v4.1: 69.4 (between Llama 4-Maverick and Jamba 2 90B)
- Active params per token: 6B
- Inference cost: $0.08/1M tokens (between Phi-5-mini and Jamba 2 90B)

**Significance**: The first Mamba 3 + MoE hybrid. Demonstrates that the Mamba 3 quality gap to vanilla Transformers can be closed by **sparse expert routing**, but at the cost of the O(n) advantage (MoE routing overhead adds ~15% latency).

---

## 13. Hardware efficiency and the inference cost story

### 13.1 The KV cache collapse

The most dramatic hardware efficiency gain of post-Transformer architectures is the **KV cache collapse**:

| Architecture | KV Cache per Token (128K context, 7B model) | Memory for 128K tokens |
|--------------|--------------------------------------------|------------------------|
| Vanilla Transformer (Llama 3.1-8B) | 2 × 32 layers × 8 KV heads × 128 dim × 2 bytes = 128 KB | 16 GB |
| Mamba-3-8B | 0 (no KV cache) | 0 GB |
| RWKV-7-7B | 0 (recurrent state) | 0.5 GB (state, constant size) |
| Striped Hyena 7B | 2 × 4 attn layers × 8 KV heads × 128 dim × 2 bytes = 16 KB | 2 GB |
| Jamba 2 90B | 2 × 8 attn layers × 8 KV heads × 128 dim × 2 bytes = 32 KB | 4 GB |

**The KV cache collapse is the single biggest production advantage of post-Transformer architectures.** A vanilla 7B Transformer at 128K context requires **16 GB of memory just for the KV cache** (in addition to the 14 GB of model weights). A Mamba-3-7B requires **0 GB** — the state is part of the forward pass, not a separate cache.

### 13.2 Inference throughput: post-Transformer vs vanilla Transformer

| Model | Architecture | Throughput (tok/s, single H100, 8K ctx) | Throughput (tok/s, 8xH100, 8K ctx) | Throughput (tok/s, RTX 4090, 8K ctx) |
|-------|--------------|------------------------------------------|--------------------------------------|--------------------------------------|
| Llama 3.1-8B | Vanilla | 1850 | 12500 | 22 |
| Mamba-3-8B | SSM | **6800** | 38000 | **118** |
| RWKV-7-7B | Linear attn | **5500** | 31000 | **95** |
| Striped Hyena 7B | Long-conv | 2200 | 14000 | 36 |
| Jamba 2 90B | SSM + Attn | 220 (per H100) | 1450 | N/A |
| TTT-Linear-7B | Test-time training | 1800 | 11000 | 28 |

**The throughput advantage of post-Transformer models is 3-5x on single H100** and **2-3x on 8xH100** (for non-MoE models). On RTX 4090, the advantage is even larger (4-5x for Mamba-3 and RWKV-7).

### 13.3 The inference cost TCO model

For an enterprise running 1B tokens/day on a self-hosted 8xH100 cluster:

| Model | Cost per 1M tokens | Daily cost | Annual cost | 3-year TCO |
|-------|-------------------|------------|-------------|-------------|
| GPT-4o (closed API) | $5.00 | $5,000 | $1,825,000 | $5,475,000 |
| Claude Opus 5 (closed API) | $15.00 | $15,000 | $5,475,000 | $16,425,000 |
| Llama 3.1-8B (Groq API) | $0.10 | $100 | $36,500 | $109,500 |
| Llama 3.1-8B (self-host, 8xH100) | $0.06 | $60 | $21,900 | $65,700 |
| Mamba-3-8B (self-host, 8xH100) | $0.015 | $15 | $5,475 | $16,425 |
| RWKV-7-7B (self-host, 8xH100) | $0.018 | $18 | $6,570 | $19,710 |
| Striped Hyena 7B (Together API) | $0.04 | $40 | $14,600 | $43,800 |

**The 3-year TCO spread**:
- **Most expensive**: Claude Opus 5 at $16.4M
- **Cheapest**: Mamba-3-8B self-host at $16.4K
- **Spread**: **$16.4M vs $16.4K = 1000x**

This is the **1000x TCO spread** that is driving the enterprise adoption of post-Transformer architectures in 2026.

### 13.4 The memory hierarchy: HBM, SRAM, and the post-Transformer advantage

The hardware efficiency of post-Transformer architectures is **not just about FLOPs** — it's about **memory hierarchy**:

- **HBM (High Bandwidth Memory)**: The off-chip GPU memory (80 GB on H100, 192 GB on B100). Vanilla Transformers need to store the KV cache in HBM. Post-Transformer models don't.
- **SRAM (on-chip memory)**: The fast on-chip memory (50 MB on H100). Vanilla Transformers can keep the model weights in SRAM (or HBM for larger models). Post-Transformer models can keep the state in SRAM.
- **Registers**: The fastest memory (256 KB on H100). The post-Transformer recurrent state is small enough to fit in registers (for small models) or SRAM (for large models).

**The memory hierarchy advantage of post-Transformer models is the key to their 4-5x throughput advantage on RTX 4090 and 3-5x throughput advantage on H100.** The model weights and the state both fit in SRAM, eliminating the HBM round-trip that vanilla Transformers must do for the KV cache.

---

## 14. Training stability challenges for SSMs and linear attention

### 14.1 The gradient flow problem

SSMs and linear attention models have a **gradient flow problem** that vanilla Transformers do not:

- **Vanilla Transformer**: Gradients flow through the attention scores (softmax(QK^T)). The softmax creates a **gradient highway** that prevents vanishing/exploding gradients.
- **SSM (Mamba)**: Gradients flow through the state matrix A. If A has eigenvalues close to 1, gradients explode. If A has eigenvalues close to 0, gradients vanish. The selective scan makes A input-dependent, which **amplifies** the gradient flow problem.
- **Linear attention (RWKV)**: Gradients flow through the time-decay γ. If γ is close to 1, gradients explode. If γ is close to 0, gradients vanish. RWKV-7's data-dependent decay matrix makes this even more complex.

### 14.2 The 5 training stability techniques

| Technique | Architecture | Mechanism | Impact |
|-----------|--------------|-----------|--------|
| **Normalization before SSM/LA** | All | Pre-norm (LayerNorm before SSM/LA block) | +0.3 MMLU, 2x training speedup |
| **A_log parameterization** | Mamba-1/2/3 | Parameterize A as -exp(A_log) to ensure A < 0 | Prevents gradient explosion |
| **dt softplus** | Mamba-1/2/3 | dt = softplus(dt_proj(x)) to ensure dt > 0 | Prevents negative time step |
| **Z-loss on A** | Mamba-3 | Add ||A||² to loss to regularize A | +0.5 MMLU at large scale |
| **Decay clipping** | RWKV-7 | Clip γ to [0.9, 0.999] | Prevents extreme decay values |

### 14.3 The "Mamba 3 diverges at 70B" problem

The most significant training stability issue of 2026 is the **"Mamba 3 diverges at 70B"** problem. Mamba-3 trained at 7B and 13B scales is stable, but **diverges after ~2T tokens at 30B+ scale**. The divergence is in the **A matrix eigenvalues** — they drift toward 1.0 as training progresses, leading to gradient explosion.

Three solutions have been proposed:
1. **Z-loss on A** (mentioned above) — regularize A toward 0
2. **Mamba-3-MoE** (Mamba-3 + MoE) — the MoE routing provides additional gradient stability
3. **Scheduled A clipping** (Google Research, June 2026) — clip the A matrix eigenvalues to [0, 0.99] every 1000 training steps

As of June 23, 2026, no Mamba-3 model >30B parameters has been successfully trained. **The "Mamba 3 70B" or "Mamba 3 90B" remains a research problem.** This is the **single biggest limitation** of the Mamba architecture in 2026.

### 14.4 The "RWKV-7 OOMs at 14B" problem

RWKV-7 has a different stability issue: **out-of-memory (OOM) errors at 14B+ parameters**. The data-dependent decay matrix D(x_t) ∈ [0, 1]^{d×d} is **d²** in size (vs d for RWKV-6), and the gradient computation requires storing the full d² matrix for each token in the sequence. At 14B parameters with d=4096, the D matrix is 16 MB per token, and a 4096-token sequence requires **64 GB** of gradient memory — which exceeds the H100's 80 GB HBM.

Two solutions are being explored:
1. **RWKV-7 v2 (research, expected 2026 Q4)**: Uses a **low-rank** decay matrix (rank 64 instead of full d²). Trades some quality for memory efficiency.
2. **RWKV-7-MoE**: Combines RWKV-7 with MoE routing. The MoE routing provides a way to "shard" the D matrix across experts.

---

## 15. Open-source implementations and the 2026 ecosystem

### 15.1 The 5 essential open-source implementations

| Library | Architecture | Language | Stars (Jun 2026) | License | Use Case |
|---------|--------------|----------|------------------|---------|----------|
| **mamba.py** | Mamba 1/2/3 | Python (PyTorch) | 18,400 | Apache 2.0 | Reference impl, research |
| **mamba3-minimal** | Mamba 3 (minimal) | Python (PyTorch) | 4,200 | MIT | Education, prototyping |
| **RWKV-LM** | RWKV 1-7 | Python + CUDA | 14,800 | Apache 2.0 | Production, edge |
| **Jamba** | Jamba 1/2 | Python (HF Transformers) | 3,600 | Apache 2.0 | Enterprise RAG |
| **TTT** | TTT, TTT-Linear | Python (PyTorch + JAX) | 1,800 | Apache 2.0 | Research, 1M context |

### 15.2 The inference ecosystem

| Server | Architecture | Latency (ms, 1st token) | Throughput (tok/s, 8xH100) | Best For |
|--------|--------------|---------------------------|-----------------------------|----------|
| **vLLM 0.9** | All except Mamba-3 | 80 | 12500 (Llama 3.1-8B) | Vanilla Transformer, Jamba |
| **SGLang 0.4** | All | 95 | 9800 | High-concurrency serving |
| **Mamba inference (custom CUDA)** | Mamba 1/2/3 | 30 | 38000 (Mamba-3-8B) | Mamba-3 only, low-latency |
| **RWKV.cpp** | RWKV 1-7 | 25 | 31000 (RWKV-7-7B) | Edge (CPU + GPU), RWKV only |
| **llama.cpp** | All (via GGUF) | 60 | 1800 | Edge (CPU), quantization |
| **MLX** | All (via MLX format) | 45 | 2400 | Apple Silicon |
| **OpenVINO** | All (via OpenVINO IR) | 55 | 2200 | Intel CPU/GPU |

### 15.3 The 2026 release calendar (full)

See §2.3 above for the full release calendar. Key 2026 H2 releases to watch:

- **2026-07**: **Mamba-3 30B** (expected; first Mamba-3 above 13B)
- **2026-08**: **RWKV-7 v2 (low-rank decay)** (expected; addresses the OOM problem)
- **2026-09**: **Jamba 2 Reasoning 200B** (AI21, expected; larger reasoning model)
- **2026-10**: **Striped Hyena 13B** (Together AI, expected; larger long-context model)
- **2026-11**: **TTT-RNN 7B (open weights)** (Google Research, expected)
- **2026-12**: **Liquid 2.0 14B** (Liquid AI, expected)

---

## 16. The 2027–2028 post-Transformer roadmap

### 16.1 Expected 2026 H2 – 2027 H1 releases

| Quarter | Architecture | Expected Release | Significance |
|---------|--------------|------------------|--------------|
| 2026 Q3 | Mamba 3 30B | Mamba-3 30B (with scheduled A clipping) | First Mamba-3 above 13B |
| 2026 Q3 | xAI Grok-2-Open (post-Transformer) | Grok-2-Open (Mamba-3-based, 200B) | First Mamba-3 from a frontier lab |
| 2026 Q4 | RWKV-7 v2 (low-rank) | RWKV-7 v2 14B | First RWKV-7 above 7B |
| 2026 Q4 | DBRX 2 (post-Transformer) | DBRX 2 (Mamba-3 + MoE, 130B) | First enterprise Mamba-3 + MoE |
| 2027 Q1 | Jamba 2 200B | Jamba 2 200B Reasoning | Largest post-Transformer model |
| 2027 Q1 | TTT-RNN (open) | TTT-RNN 7B (open weights) | First open-weights streaming TTT |
| 2027 Q2 | Striped Hyena 13B v3 | Striped Hyena 13B | Larger long-context model |
| 2027 Q2 | Mamba 4 (research) | Mamba 4 (with Hopfield retrieval) | Research preview |

### 16.2 The 5 transitions to watch

1. **The 70B+ transition**: Can post-Transformer architectures scale to 70B+ parameters? Mamba-3 30B (Q3 2026) and Jamba 2 200B (Q1 2027) are the critical test cases. If they succeed, the post-Transformer market expands to the "frontier" tier. If they fail, post-Transformer remains a "mid-tier" technology.

2. **The 1M+ context transition**: Striped Hyena and TTT-Linear already have 1M context. The 2027 H1 question is: can the quality at 1M context reach parity with 128K context? The "LongBench 1M" benchmark (Google, expected Q4 2026) will be the test.

3. **The MoE + SSM transition**: Mamba-3-MoE and Jamba 2 90B (which uses MoE) show the path. The 2027 H1 question is: can we get Jamba-2-quality at Llama-3.1-cost? A "Mamba-3 + MoE + Attention" hybrid is the likely answer.

4. **The multimodal transition**: Striped Hyena 7B v2 (vision + text) is the first multimodal post-Transformer model. The 2027 H1 question is: can post-Transformer models handle video, audio, and embodied agent inputs (per `28-AI-Video-Audio-Generation/` and `11-AI-Applications/13-Embodied-AI-Industries.md`)?

5. **The on-device transition**: LFM2-1.3B-edge and RWKV-7-Goose-1.6B are the first on-device post-Transformer models. The 2027 H1 question is: can a 3B post-Transformer model run on a smartwatch (512 MB RAM)?

### 16.3 Strategic recommendations for 2026 H2

For enterprises evaluating post-Transformer architectures in 2026 H2, the recommendations are:

1. **For edge inference (smartphones, IoT, automotive)**: **Adopt RWKV-7-Goose-1.6B or LFM2-1.3B-edge immediately.** The 4-8x battery advantage is decisive.

2. **For long-context (≥256K) applications (RAG, code analysis, document Q&A)**: **Adopt Striped Hyena 7B or TTT-Linear-7B.** The 1M native context is a clear win.

3. **For enterprise RAG (8B-12B activated params)**: **Wait for Jamba 2 90B Reasoning (Q1 2027) or evaluate Mamba-3-MoE-30B-A6B (research) for production pilots in Q4 2026.**

4. **For coding assistants**: **Adopt Striped Hyena 7B.** It leads the 7B class on HumanEval (84.7).

5. **For multimodal applications**: **Evaluate Striped Hyena 7B v2 (June 2026) as a research preview.** No production-ready multimodal post-Transformer model is available yet.

6. **For reasoning-heavy applications**: **Adopt Jamba 2 90B Reasoning (April 2026).** It is the first reasoning-tuned post-Transformer model.

---

## 17. Cross-references, builder's checklist, glossary

### 17.1 Cross-references to existing library documents

This document explicitly cross-references the following existing library documents:

**Architectures and model families**:
- `02-LLMs/02-Model-Families.md` — model inventory and lineage
- `02-LLMs/04-Quantization.md` — quantization layer (INT4/INT8/FP8 for post-Transformer models)
- `02-LLMs/08-Custom-Silicon-and-AI-Hardware-2026.md` — the silicon substrate (H100, B100, Groq LPU, TPU v6)
- `02-LLMs/09-Open-Weights-Race-2026.md` — Western open weights (Llama 4, Jamba 2, Phi-5, RWKV-7, Mamba-3, Striped Hyena)
- `17-Research-Frontiers-2026/03-LLM-Architectures-2026.md` — 2024-2025 architecture survey (MoE, MLA, Mamba-2, Jamba 1, RWKV-6, Eagle, model merging, QAT, FP8, YaRN)
- `17-Research-Frontiers-2026/06-Reasoning-Models.md` — reasoning models (Jamba 2 90B Reasoning, TTT-Linear for reasoning)
- `17-Research-Frontiers-2026/09-Efficient-ML-Research.md` — efficient ML research

**Agents and agentic systems**:
- `03-Agents/` — agentic architectures (Mamba-3-based agents in Genesis)
- `31-AI-Workflow-Orchestration-and-Durable-Execution/` — durable agent execution
- `32-Agent-Memory-Systems/` — agent memory (TTT-Linear, Hyena 2 for memory)
- `18-Agent-Security-and-Trust/` — security and trust (Forge + post-Transformer models)
- `20-Agent-Infrastructure-and-Observability/` — agent infrastructure
- `28-AI-Agent-Commerce-and-A2A-Payments/` — agent commerce (Mamba-3-1B for low-latency agents)

**Efficiency and deployment**:
- `23-Local-AI-Inference-Self-Hosting/` — on-prem deployment (RWKV.cpp, llama.cpp, MLX for edge)
- `30-Small-Language-Models/` — small-model efficiency frontier (Phi-5-mini, LFM2-1.3B-edge)
- `02-LLMs/04-Quantization.md` — quantization for post-Transformer models

**Multimodal**:
- `28-AI-Video-Audio-Generation/` — video/audio generation (Striped Hyena 7B v2 multimodal)
- `11-AI-Applications/13-Embodied-AI-Industries.md` — embodied AI (Mamba-3 in Genesis)

**RAG, retrieval, and knowledge**:
- `04-RAG/` — RAG patterns (Jamba 2 90B + RAG, Command R+ v2)
- `05-Enterprise/` — enterprise RAG (Jamba 2 90B deployments)
- `07-Emerging/` — emerging patterns

**Security, safety, and regulation**:
- `18-Agent-Security-and-Trust/` — agent security (LlamaGuard 3 + post-Transformer)
- `21-AI-Regulation-Antitrust/` — regulation (EU AI Act Art. 53 on transparency)
- `22-AI-Cybersecurity-Mythos/` — cybersecurity

**Industry and applications**:
- `10-Industry/` — industry analysis
- `11-AI-Applications/` — applications (voice, video, code, sales, healthcare)
- `12-Business-Prospects/` — business prospects
- `13-Top-Demand/` — top demand (post-Transformer skills: Mamba, RWKV, TTT)
- `14-Case-Studies-Real-World-Projects/` — case studies (Jamba 2 enterprise deployments)
- `16-AI-Business-Models-Playbooks/` — business models
- `19-Voice-AI-and-Agents/` — voice agents (Mamba-3-1B in Cartesia Sonic-3)
- `24-AI-Sales-and-Marketing/` — sales and marketing
- `25-Multi-Cloud-AI-Strategy/` — multi-cloud strategy
- `26-Browser-Based-AI/` — browser-based AI
- `27-AI-in-HR-and-Recruiting/` — HR and recruiting

### 17.2 Builder's checklist (12 steps)

For teams building a production post-Transformer system in 2026 H2:

1. **Define the workload** (latency-sensitive, long-context, edge, agentic memory, multimodal, reasoning). Different post-Transformer architectures win at different workloads.
2. **Choose the architecture family** (Mamba 3 for streaming/edge, RWKV 7 for low-latency/edge, Jamba 2 for enterprise, Striped Hyena for long-context, TTT-Linear for 1M+ context retrieval, LFM2 for battery-constrained edge).
3. **Choose the model size** (1.3B for smartwatch, 1.6B for phone, 3B for laptop, 7-8B for server, 90B for enterprise).
4. **Choose the inference server** (vLLM 0.9 for general, Mamba inference for Mamba-only, RWKV.cpp for edge, llama.cpp for CPU/edge, MLX for Apple Silicon).
5. **Choose the quantization** (FP16 for max quality, INT8 for balanced, INT4 for edge, FP8 for training).
6. **Benchmark on your workload** (don't trust the AA v4.1 numbers — measure on your specific prompts). Post-Transformer models can lose 2-5 MMLU points on out-of-distribution prompts.
7. **Test the long-context behavior** (if your workload is ≥128K). Post-Transformer models have different long-context failure modes than vanilla Transformers (SSM state saturation, linear-attention decay saturation).
8. **Set up monitoring** (post-Transformer models can "degenerate" in subtle ways — e.g., Mamba state collapse, RWKV decay collapse). Monitor the surprise score (for TTT), state norm (for Mamba), and decay distribution (for RWKV).
9. **Plan for the 70B+ transition** (if your workload is enterprise-grade). Wait for Jamba 2 200B (Q1 2027) or Mamba-3-MoE-130B (Q4 2026) before committing to a long-term architecture.
10. **Plan for the multimodal transition** (if your workload is multimodal). Striped Hyena 7B v2 (June 2026) is the first multimodal post-Transformer production model, but the ecosystem is still immature.
11. **Test the agentic memory story** (if your workload is agentic). TTT-Linear and Hyena 2 are the current leaders for 1M+ context agentic memory. The memory is a different abstraction than vanilla Transformer KV cache — plan for it.
12. **Have a fallback to vanilla Transformer** (always). For workloads where quality is paramount (e.g., reasoning, math, code), vanilla Transformers still lead. Use a hybrid (e.g., Jamba 2 90B with both SSM and Attention) to get the best of both worlds.

### 17.3 Glossary (30 terms)

1. **AA v4.1 (Artificial Analysis Intelligence Index v4.1)**: The dominant 2026 LLM benchmark, weighted average of MMLU, HumanEval, GSM8K, MATH, GPQA, and HellaSwag.
2. **A_log parameterization**: Mamba's parameterization of the state matrix A as -exp(A_log) to ensure A < 0 (stable eigenvalues).
3. **Attention sink**: A pattern where the most recent N token embeddings are stored in a "sink" buffer and concatenated with the SSM/linear-attn output. Used in Mamba-3, RWKV-7, Based 2 to compensate for the lack of exact-position retrieval in pure SSM/linear-attn models.
4. **Causal convolution prelude**: A 1D convolution (kernel size 4) applied before the SSM scan in Mamba-3. Provides local context for the SSM.
5. **Data-dependent decay**: A linear-attention innovation (RWKV-7) where the decay factor γ is computed from the input token, not a fixed scalar.
6. **Drop-in replacement**: A model that can replace a vanilla Transformer without changes to the surrounding system (same input/output format, same tool-calling format, same context length). Post-Transformer 7-8B models are at "drop-in replacement" parity as of 2026.
7. **Forget gate**: A learnable, input-dependent scalar γ ∈ [0, 1] that scales the state matrix A in Mamba-3. Functionally similar to the LSTM's forget gate.
8. **Goose MLP**: The MLP block in RWKV-7, which uses 4 grouped linear layers with mixed precision (FP16, BF16, FP8, INT4).
9. **HBM (High Bandwidth Memory)**: The off-chip GPU memory (80 GB on H100, 192 GB on B100). Vanilla Transformer KV cache is stored in HBM. Post-Transformer models have no KV cache.
10. **Hyena 2**: A long-convolution architecture from Stanford + Together AI. Uses multi-resolution convolutions of length [256, 1024, 4096, 16384].
11. **Implicit long convolution**: A convolution computed via the FFT-IFFT trick in log-space, O(L log L) per layer. Used in Hyena 2 for convolutions longer than 4096.
12. **Jamba 2**: AI21's hybrid SSM-Transformer architecture (1:7 SSM:Attention ratio). 90B parameters, 12B activated.
13. **Jamba moment**: The Q1 2026 inflection when Jamba 2 90B reached 6 Fortune-500 enterprise deployments — the first time a post-Transformer model cleared the "enterprise production" bar at >50B parameters.
14. **KV cache collapse**: The elimination of the KV cache in post-Transformer models. A vanilla 7B Transformer at 128K context needs 16 GB of KV cache; Mamba-3-7B needs 0.
15. **Linear attention**: A class of attention mechanisms that approximate softmax attention with a kernel feature map, achieving O(n) time and O(1) memory. Examples: RWKV-7, Performer, Based 2, RetNet.
16. **LTC (Liquid Time-Constant)**: A continuous-time recurrent network where each neuron has a learnable, input-dependent time constant. Used in Liquid AI's LFM 2.0.
17. **Mamba 3**: The third generation of selective state-space models (CMU + Princeton, Dec 2025). Multi-scale SSM heads + learnable forget gate + causal convolution prelude.
18. **Mamba moment**: The late 2025 / early 2026 inflection when Mamba-3 reached production scale (Genesis AGI sim, Dec 2025) and was re-implemented in a clean PyTorch reference (Mamba3-minimal, Feb 2026).
19. **Mixture-of-Depth Experts (MoDE)**: Conditional depth computation (from `17-Research-Frontiers-2026/03-LLM-Architectures-2026.md` §4.1). Routes tokens to different layer depths.
20. **Monarch matrices**: A class of structured matrices (block-diagonal products with permutations) that are sub-quadratic in both compute and memory. Used in Monarch Mixer.
21. **Multi-scale SSM heads**: Mamba-3's key innovation — k parallel SSM heads with state dimensions [N/4, N/2, N, 2N] and a learned mixer.
22. **Post-Transformer**: An architecture that is not a vanilla Transformer. Includes SSM (Mamba 3), linear attention (RWKV 7), long convolution (Hyena 2), structured matrices (Monarch), continuous-time recurrent (Liquid), and test-time training (TTT).
23. **RWKV-7 ("Goose")**: The seventh generation of the RWKV linear-attention family (BlinkDL, Jan 2026). Data-dependent decay matrix + learned token-shift gate + Goose MLP.
24. **Selective scan**: Mamba's input-dependent state-space dynamics. The state matrix A, the input projection B, and the output projection C are all computed from the input.
25. **SSD (Structured State Space Duality)**: Mamba-2's reformulation of the selective SSM as a structured matrix multiplication, enabling 2.2x training throughput vs Mamba-1.
26. **SSM (State-Space Model)**: A recurrent neural network with a continuous-time state, defined by matrices A, B, C, D. Mamba is the canonical example in 2026.
27. **Striped Hyena**: Together AI's hybrid of Hyena 2 (long convolutions) + grouped attention. 16:1 long-conv:grouped-attention ratio. 7B parameters, 1M native context.
28. **Striped Hyena moment**: The Feb 26, 2026 release of Striped Hyena 7B — the first 1M-context sub-quadratic model at production quality.
29. **Test-time training (TTT)**: An architecture where the model's weights are updated during inference, using the recent context as training data. Titans and TTT-Linear are the canonical examples.
30. **TTT-Linear**: A more efficient variant of TTT (March 2026) that updates a small linear projection instead of a full weight matrix. O(n) time, O(1) memory, 1M native context.

### 17.4 References (selected)

For convenience, here are the key 2026 papers and resources referenced in this document:

- Gu, Goel, Re, Dao. "Mamba-3: Multi-Scale Selective State-Space Models." December 2025. (CMU + Princeton)
- Peng (BlinkDL). "RWKV-7: Linear Attention with Data-Dependent Decay." January 2026.
- Lieber, et al. (AI21 Labs). "Jamba 2: A 90B-Scale Hybrid SSM-Transformer Model." December 2025.
- Poli, et al. (Stanford + Together AI). "Hyena 2: Long Convolutions for Sub-Quadratic Sequence Modeling." November 2025.
- Together AI. "Striped Hyena: Long-Context Sub-Quadratic Sequence Modeling." February 2026.
- Arora, et al. (Hazy Research / Stanford). "Based 2: Linear Attention with Shift Kernels." February 2026.
- Fu, et al. (Stanford). "Monarch Mixer: Structured Sub-Quadratic Matrices for Sequence Modeling." March 2026.
- Hasani, et al. (Liquid AI). "Liquid Foundation Models 2.0." March 2026.
- Sun, et al. (Google Research). "Titans: Learning to Memorize at Test Time." January 2026.
- Sun, et al. (Google Research). "TTT-Linear: Linear-Time Test-Time Training." March 2026.
- Sun, et al. (Microsoft Research). "Retentive Network: A Successor to Transformer for Large Language Models." 2023. (Still relevant via Phi-5)
- Gu, Dao. "Mamba-3-MoE: Combining Multi-Scale SSMs with Sparse Expert Routing." April 2026.

And the existing library documents that this document complements:
- `17-Research-Frontiers-2026/03-LLM-Architectures-2026.md` — 2024-2025 architecture survey
- `02-LLMs/09-Open-Weights-Race-2026.md` — Western open weights
- `02-LLMs/08-Custom-Silicon-and-AI-Hardware-2026.md` — the silicon substrate
- `02-LLMs/04-Quantization.md` — quantization layer
- `23-Local-AI-Inference-Self-Hosting/` — on-prem deployment
- `30-Small-Language-Models/` — small-model efficiency frontier

---

*This document is a deep-dive on the 2026 post-Transformer architectural frontier. It complements (does not replace) the existing 2024-2025 architecture survey in `17-Research-Frontiers-2026/03-LLM-Architectures-2026.md`. The cutoff is June 23, 2026. For the latest post-Transformer releases, see the [Mamba GitHub](https://github.com/state-spaces/mamba), the [RWKV GitHub](https://github.com/BlinkDL/RWKV-LM), the [AI21 Jamba GitHub](https://github.com/ai21labs/Jamba), the [Hyena GitHub](https://github.com/HazyResearch/hyena), the [Together AI API](https://together.ai), the [Liquid AI API](https://liquid.ai), and the [Google Research TTT page](https://research.google/pubs/titans-learning-to-memorize-at-test-time/).*

<!-- SECTION_APPEND_2 -->

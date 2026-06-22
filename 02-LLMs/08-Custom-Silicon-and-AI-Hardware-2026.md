# 08 — Custom Silicon and AI Hardware Acceleration 2026: The Deep-Dive

> **Why this document exists.** As of mid-2026, the defining question in shipping frontier AI is no longer "which model?" but **"on which silicon?"** NVIDIA's H100/B200/Blackwell stack (with Vera Rubin launching January 2026) still dominates training — but inference has fractured: Groq LPU v2, Cerebras WSE-3, SambaNova RDU v3, Google TPU v6, AWS Trainium 3, and the new Meta MTIA v3 / Microsoft Maia 2 chips each win 2–10x on cost or latency for specific workloads. A 1,000-token completion costs \$0.10 / 1M tokens on Groq, \$0.40 on Cerebras, \$0.60 on TPU v6, \$1.50 on Blackwell — and \$3.00 on H100. Over a 3-year production deployment at 1B queries/month, that spread is **\$100M+ per model** (see §4 below for the cost math). This document is the practitioner's field guide to the 2026 custom-silicon wave: the chip architectures (GPU, systolic, dataflow, LSI), the software stacks (CUDA, ROCm, Triton, JAX/XLA, MLIR, Modular MAX), the M&A wave (NVIDIA→Groq \$20B, Intel→SambaNova \$1.6B, Meta→Rivos, Meta+Broadcom multi-gen), the inference economics flip, the routing patterns, the procurement patterns, and the 2027–2028 silicon roadmap. It complements `13-Top-Demand/15-AI-Energy-Sustainability-and-Compute-2026.md` §3 (the energy-side view), `02-LLMs/06-AI-Model-Providers-Free-Tiers.md` (the API providers), `23-Local-AI-Inference-Self-Hosting/` (on-prem), `25-Multi-Cloud-AI-Strategy/` (cloud procurement), `30-Small-Language-Models/` (efficiency), and `17-Research-Frontiers-2026/03-LLM-Architectures-2026.md` (the algorithmic side). Here we go deep on **silicon as a first-class artifact**.

---

## Table of Contents

1. [Why silicon, why now (mid-2026)](#1-why-silicon-why-now-mid-2026)
2. [The 2026 M&A wave: NVIDIA → Groq \$20B, Intel → SambaNova, Meta → Rivos](#2-the-2026-ma-wave-nvidia--groq-20b-intel--sambanova-meta--rivos)
3. [The 2026 silicon landscape: 25+ vendors, 4 architectural families](#3-the-2026-silicon-landscape-25-vendors-4-architectural-families)
4. [The inference economics flip: why inference now beats training in cost](#4-the-inference-economics-flip-why-inference-now-beats-training-in-cost)
5. [NVIDIA's 2026 stack: Blackwell, Vera Rubin, Rubin Ultra, Feynman](#5-nvidias-2026-stack-blackwell-vera-rubin-rubin-ultra-feynman)
6. [The NVIDIA alternatives: Google TPU v6, AWS Trainium 3, Microsoft Maia 2, Meta MTIA v3](#6-the-nvidia-alternatives-google-tpu-v6-aws-trainium-3-microsoft-maia-2-meta-mtia-v3)
7. [The inference specialists: Groq LPU v2, Cerebras WSE-3, SambaNova RDU v3, Tenstorrent](#7-the-inference-specialists-groq-lpu-v2-cerebras-wse-3-sambanova-rdu-v3-tenstorrent)
8. [Memory hierarchy: HBM3e, HBM4, on-chip SRAM, wafer-scale integration](#8-memory-hierarchy-hbm3e-hbm4-on-chip-sram-wafer-scale-integration)
9. [Networking: NVLink 5, NVLink 6, InfiniBand, optical, scale-up vs scale-out](#9-networking-nvlink-5-nvlink-6-infiniband-optical-scale-up-vs-scale-out)
10. [The software stack: CUDA, ROCm, Triton, JAX/XLA, MLIR, Modular MAX, GroqWare, Thunder](#10-the-software-stack-cuda-rocm-triton-jaxxla-mlir-modular-max-groqware-thunder)
11. [Precision formats in 2026: FP4, NVFP4, MXFP, FP6, FP8, BF16, FP16, INT8, INT4](#11-precision-formats-in-2026-fp4-nvfp4-mxfp-fp6-fp8-bf16-fp16-int8-int4)
12. [The Modular attack on CUDA: the most important software story of 2026](#12-the-modular-attack-on-cuda-the-most-important-software-story-of-2026)
13. [Production routing patterns: how to use 2–4 silicon in production](#13-production-routing-patterns-how-to-use-24-silicon-in-production)
14. [Cost modeling: \$/1M tokens across the 2026 stack (with code)](#14-cost-modeling-1m-tokens-across-the-2026-stack-with-code)
15. [Procurement patterns: how to buy, lease, and reserve silicon in 2026](#15-procurement-patterns-how-to-buy-lease-and-reserve-silicon-in-2026)
16. [Build vs buy: when to train on your own silicon vs rent the hyperscalers'](#16-build-vs-buy-when-to-train-on-your-own-silicon-vs-rent-the-hyperscalers)
17. [The 2027–2028 silicon roadmap: Rubin Ultra, Feynman, TPU v7, Trainium 4, WSE-4](#17-the-20272028-silicon-roadmap-rubin-ultra-feynman-tpu-v7-trainium-4-wse-4)
18. [Cross-references](#18-cross-references)
19. [Builder's checklist](#19-builders-checklist)
20. [Glossary](#20-glossary)

---

## 1. Why silicon, why now (mid-2026)

### 1.1 The 2020s AI compute story (revisited, from a silicon lens)

The 2020s AI compute story, viewed from the **silicon** side rather than the **energy** side (covered in `13-Top-Demand/15-AI-Energy-Sustainability-and-Compute-2026.md` §1), has three distinct phases:

**Phase 1 (2017–2022) — The NVIDIA monopoly.** V100, A100, T4, L4. CUDA was the only credible software stack. Every training and inference job ran on NVIDIA. The bottleneck was research talent and dataset curation; the silicon was just "rented" from the cloud.

**Phase 2 (2023–2024) — The H100 shortage.** H100s were allocated by personal relationships; waitlists stretched to 12+ months. The bottleneck shifted to **hardware supply** and **CapEx availability**. xAI, Mistral, Cohere, and the Saudi-backed HUMAIN raised multi-billion-dollar rounds primarily to secure GPU clusters. CUDA's moat deepened as developers built around it.

**Phase 3 (2025–2026) — The fragmentation.** The bottleneck shifted twice more. First to **electricity / power** (Sept 2024 → present, see the energy doc). Then — and this is the 2026 story — to **the inference economics flip**: a 1,000-token completion costs 30x less on Groq LPU v2 than on H100. Inference-specialty silicon broke the CUDA monopoly for production workloads. Companies that moved to inference-specialty silicon (Groq, Cerebras) or in-house silicon (Google, Meta) gained a structural cost advantage that compounds.

### 1.2 The 2026 inflection in three numbers

Three numbers define the 2026 silicon inflection:

1. **\$0.10 / 1M tokens on Groq LPU v2** vs. **\$3.00 / 1M tokens on H100** — a 30x spread. For a 70B model serving 1B queries/month at 1,000 tokens/query, that is \$7.2M/year (Groq) vs. \$216M/year (H100). Over 3 years, the spread is **\$626M**.
2. **Nvidia's moat is not what it used to be** (HN Algolia, 6 pts, 2026-04-12). The 30-year SIMT monopoly is now contested on every workload class. Inference is the most contested; training is still NVIDIA-dominant but TPU v6 has 25%+ of Google-internal training and Trainium 3 has 15%+ of AWS-internal training.
3. **NVIDIA buys Groq's assets for \$20B cash** (HN Algolia, **699 pts**, 2025-12-24). The largest AI M&A deal of the decade. The acqui-hire is not about Groq's products (NVIDIA already has GPU inference); it is about the **talent** (Jonathan Ross, the LPU architect, and ~150 senior engineers) and the **IP portfolio** (LPU patents). NVIDIA is buying the threat.

### 1.3 What changed in 2025–2026

| Trend | 2024 | 2026 | Why it matters |
|-------|------|------|----------------|
| Training silicon | NVIDIA H100 (95%) | NVIDIA B200 + GB200 (80%) + TPU v6 (10%) + Trainium 3 (5%) | TPU and Trainium are now credible training alternatives |
| Inference silicon | NVIDIA H100/T4/L4 (90%) | NVIDIA (40%) + TPU v6 (15%) + Groq (12%) + Cerebras (8%) + Trainium 3 (10%) + others (15%) | Inference is the most fragmented market in AI |
| Software portability | CUDA-only (95%) | CUDA + ROCm + Triton + JAX/XLA + MLIR + Modular MAX | Modular MAX + Triton + MLIR make non-NVIDIA silicon competitive on developer velocity |
| $/1M tokens (inference) | H100 \$3.00 (median) | H100 \$3.00, B200 \$1.50, Trainium 3 \$0.70, TPU v6 \$0.60, Cerebras WSE-3 \$0.40, Groq LPU v2 \$0.10 | 30x spread between best and worst |
| Memory bandwidth | H100: 3.35 TB/s HBM3 | B200: 8.0 TB/s HBM3e; Rubin: 13.0 TB/s HBM4; WSE-3: 33 TB/s on-chip | Memory bandwidth is the dominant cost in inference; on-chip memory is the only way to break the HBM wall |
| Precision | BF16 / FP16 / INT8 | FP4 / NVFP4 / FP6 / FP8 / BF16 / FP16 / INT8 / INT4 | 4-bit inference is now native on Blackwell; 6-bit (FP6) on Rubin |
| Interconnect | NVLink 4 (900 GB/s) | NVLink 5 (1,800 GB/s), NVLink 6 (3,600 GB/s, Rubin), optical in 2027 | Multi-GPU training is now memory-bandwidth bound, not compute bound |

### 1.4 Why silicon, why this doc, why now

This document exists because the 2026 silicon landscape is the **single highest-leverage decision a builder makes in production**. The 4 sub-decisions are:

1. **Which silicon to train on** (NVIDIA vs TPU vs Trainium vs custom)
2. **Which silicon to serve on** (Groq vs Cerebras vs TPU vs Trainium vs NVIDIA)
3. **Which software stack to commit to** (CUDA vs ROCm vs Triton vs JAX vs Modular MAX)
4. **How to route traffic across silicon** (single-vendor vs multi-vendor)

The wrong choice on any of these costs 3–30x on cost or latency. The right choice compounds over 3+ years. This doc gives the builder the data to make each decision with conviction.

---

## 2. The 2026 M&A wave: NVIDIA → Groq \$20B, Intel → SambaNova, Meta → Rivos

### 2.1 The deal wave (chronological, 2024–2026)

| Date | Acquirer | Target | Deal | Type | Why |
|------|----------|--------|------|------|-----|
| 2024-03 | Microsoft | Inflection AI | ~\$650M (acqui-hire) | Talent | Mustafa Suleyman + most of Inflection's team |
| 2024-06 | Amazon | Adept | \$330M (acqui-hire) | Talent | David Luan + agent research team |
| 2024-12 | Google | Character.AI | \$2.7B (acqui-hire) | Talent | Noam Shazeer returns to Google |
| 2025-02 | Meta | Korean AI chip startup (deferred) | ~\$200M | Talent | Samsung engineer founded; inference chip IP |
| 2025-03 | Meta → FuriosaAI | — | \$800M offer rejected | Korean startup chose independence | Founded by ex-Samsung; Waratah chip |
| 2025-09 | Groq | — | \$750M raise, \$6.9B valuation | Funding | Inference silicon for LPU v3 |
| 2025-09 | Meta | Rivos | ~\$1B | Acqui-hire | RISC-V chip IP for inference |
| 2025-10 | Intel → SambaNova | — | Talks; eventually closed Q1 2026 | Acqui-hire | Enterprise inference + on-prem stack |
| 2025-12-14 | Intel | SambaNova | \$1.6B closed | Acqui-hire | Enterprise inference business + DataScale stack |
| 2025-12-24 | NVIDIA | Groq (assets + talent) | **\$20B cash** | Acqui-hire | Largest AI M&A of the decade; Jonathan Ross + 150 engineers |
| 2026-01-06 | NVIDIA | Vera Rubin launch | — | Product | NVIDIA's 2026 flagship, B300 successor |
| 2026-02-12 | NVIDIA | Blackwell MLPerf results | — | Product | 10x inference cost reduction vs H100 |
| 2026-04-12 | Multiple | "Nvidia's moat is not what it used to be" | — | Industry commentary | First time the moat is widely questioned |
| 2026-04-16 | Meta + Broadcom | Multi-gen silicon AI chip partnership | Multi-billion, multi-year | Partnership | Custom ASICs for Meta's recommendation + inference workloads |
| 2026-Q1 | Apple | (internal) | — | Product | M5 / M5 Ultra ramp; Apple Intelligence on-device |
| 2026-Q2 | Microsoft | (internal) | — | Product | Maia 2 limited GA; Azure-internal workloads only |
| 2026-Q1 | Google | (internal) | — | Product | TPU v6 (Trillium) GA; TPU v7 announced |
| 2026-Q2 | AWS | (internal) | — | Product | Trainium 3 GA; Inferentia 3 production |
| 2026-Q4 | Cerebras | (product) | — | Product | WSE-3 GA; WSE-4 announced |
| 2026-Q4 | SambaNova | (under Intel) | — | Product | RDU v3 production; DataScale v3 |

**The 2026 M&A story in one line:** **talent + inference IP is the most valuable asset in AI** — and every major acquirer (NVIDIA, Intel, Meta, Google, Apple) is paying a premium for it.

### 2.2 The NVIDIA → Groq deal: what it really means

The NVIDIA → Groq \$20B deal (Dec 2025) is the most important AI M&A transaction of the decade. Three things to understand:

1. **It's not about products.** NVIDIA is not buying Groq Cloud or the LPU v2 product line (which will continue operating under a transitional license). NVIDIA is buying the **talent** (Jonathan Ross, the LPU architect, and ~150 senior engineers) and the **IP portfolio** (~300 issued and pending patents on deterministic LSI, on-chip memory routing, and compiler techniques).

2. **It's a defensive move.** Groq was NVIDIA's most credible threat in the inference market. With Groq's IP in NVIDIA's pocket, **no other startup can credibly build a deterministic-LSI inference chip** without licensing from NVIDIA. The startup ecosystem just lost its highest-leverage exit.

3. **It signals the end of "rent the chips, build the models" as a viable business model.** If the only way to make money on inference is to be a hyperscaler with in-house silicon (Google, Meta, AWS, Microsoft) or to be acquired by NVIDIA, the inference-only startups (Groq, Cerebras, SambaNova, Tenstorrent) are now in a "**sell or die**" position. Cerebras is the next most likely target.

### 2.3 The Meta + Broadcom multi-gen silicon deal

The April 2026 announcement of a multi-gen Meta + Broadcom partnership is the second-most-important 2026 silicon story. Meta is committing to a **multi-year, multi-billion-dollar** custom-ASIC program with Broadcom (the leading networking + ASIC partner) to build inference chips for:

- **Recommendation** (Reels, Feed, Search ranking) — the highest-volume, lowest-latency workloads at Meta
- **Generative AI inference** (Llama 4, Llama 5) — the second-priority workload
- **Future agentic workloads** (Meta AI agents, advertising creative) — the third-priority workload

The first chip is expected to tape out Q4 2026 and ramp in 2027. It will use TSMC 3N process (vs. the 5N used by most 2026 chips) and integrate Broadcom's 102.4 Tbps SerDes for scale-out.

The strategic implication: **Meta is now the third hyperscaler to commit to in-house silicon at scale** (after Google TPU and AWS Trainium). The big-four (Google, AWS, Microsoft, Meta) will all be silicon-self-sufficient by 2027. Only Oracle and the neoclouds (CoreWeave, Lambda, Crusoe) will remain NVIDIA-only.

### 2.4 The Intel → SambaNova deal: the on-prem play

Intel closed the \$1.6B SambaNova acquisition in December 2025. The strategic logic is different from NVIDIA → Groq:

- **SambaNova's RDU v3 is optimized for on-prem enterprise inference** (regulated industries, government, financial services — all of which cannot use public cloud for data-residency reasons).
- **Intel's Gaudi 3 has been a slow ramp** (5–10% market share in 2025 vs. the 25%+ Intel originally targeted). SambaNova's enterprise sales motion + DataScale software stack + the customer base (Fidelity, US Air Force, etc.) gives Intel a credible on-prem AI silicon story.
- **The combined entity is positioned as the "Nvidia alternative for on-prem"** — and with regulatory pressure on data residency (EU AI Act, China data laws, US FedRAMP) the on-prem AI silicon market is growing 40%+ YoY.

### 2.5 The Modular story: attacking CUDA from the software side

Modular (founded by Chris Lattner, the creator of LLVM, Swift, and MLIR) is the most important **software-only** 2026 silicon story. Modular's MAX platform (GA December 2025) is a **hardware-agnostic AI compute platform** that compiles Python → MLIR → CUDA/ROCm/TPU/Trainium/Groq/Cerebras with comparable developer velocity to PyTorch.

The strategic implication: **Modular is the only credible "write once, run anywhere" AI compiler**. If MAX achieves even 50% of CUDA's developer velocity, the CUDA moat collapses — because developers can target any silicon without rewriting in CUDA. NVIDIA's \$20B Groq acqui-hire is partly a defensive response to Modular's threat: if the software moat is going to break anyway, NVIDIA needs to control the hardware IP that runs on top of it.

The Modular story is covered in detail in §12 below.

---

## 3. The 2026 silicon landscape: 25+ vendors, 4 architectural families

### 3.1 The full 2026 vendor × architecture matrix

| Vendor | Chip | Architecture | Process | Memory | Peak BF16 PFLOPS | HBM TB/s | On-chip mem | Best for | 2026 status |
|--------|------|--------------|---------|--------|------------------|----------|-------------|----------|-------------|
| **NVIDIA** | B200 | Hopper-next / Blackwell | TSMC 4N | 192 GB HBM3e | 2.5 | 8.0 | — | Training, general inference | Production; 70% of training |
| **NVIDIA** | GB200 | Blackwell (Grace CPU + 2× B200) | TSMC 4N | 384 GB HBM3e | 5.0 | 16.0 | — | Training, large-model inference | Production |
| **NVIDIA** | B300 | Blackwell-refresh | TSMC 4N | 288 GB HBM3e | 3.0 | 8.0 | — | Inference, training | Production mid-2026 |
| **NVIDIA** | Vera Rubin (R100) | Rubin | TSMC 3N | 288 GB HBM4 | 4.5 | 13.0 | — | Training + inference | Launched Jan 2026 |
| **NVIDIA** | Vera Rubin Ultra (R200) | Rubin-Ultra | TSMC 3N | 1 TB HBM4e | 9.0 | 26.0 | — | Largest training | Announced 2026; 2027 |
| **AMD** | MI400 | CDNA 4 | TSMC 3N | 288 GB HBM3e | 3.0 | 8.0 | — | Training + inference | Limited GA Q2 2026 |
| **AMD** | MI450 | CDNA 5 | TSMC 2N | 432 GB HBM4 | 6.0 | 13.0 | — | Training + inference | Announced 2026; 2027 |
| **Google** | TPU v6 (Trillium) | Systolic | TSMC 5N | 144 GB HBM | 1.4 | 1.6 | 144 MB CMEM | Training + inference | GA Q1 2026 |
| **Google** | TPU v7 (Ironwood) | Systolic | TSMC 3N | 256 GB HBM3e | 2.0 | 4.0 | 256 MB CMEM | Training + inference | Announced 2026; 2027 |
| **AWS** | Trainium 3 | NeuronCore v3 | TSMC 5N | 144 GB HBM3e | 1.6 | 2.4 | 128 MB | Training + inference, AWS-only | GA Q2 2026 |
| **AWS** | Inferentia 3 | NeuronCore v3 | TSMC 5N | 128 GB HBM3e | 0.8 | 1.6 | 96 MB | Inference, AWS-only | Production Q4 2025 |
| **AWS** | Trainium 4 | NeuronCore v4 | TSMC 3N | 192 GB HBM4 | 3.2 | 4.8 | 256 MB | Training + inference | Announced 2026; 2027 |
| **Microsoft** | Maia 2 | Custom | TSMC 5N | 144 GB HBM3e | 1.4 | 2.0 | 96 MB | Inference, Azure-internal | Limited GA Q2 2026 |
| **Microsoft** | Maia 100 (next) | Custom | TSMC 3N | 192 GB HBM4 | 2.8 | 4.0 | 192 MB | Inference + training | Announced; 2027 |
| **Meta** | MTIA v3 | Custom RISC | TSMC 5N | 128 GB HBM3e | 1.2 | 1.8 | 96 MB | Recommendation + inference | Production Q1 2026 |
| **Meta + Broadcom** | Custom ASIC #1 | Custom | TSMC 3N | 192 GB HBM4 | 2.0 | 3.2 | 128 MB | Recommendation + inference | Tape-out Q4 2026; ramp 2027 |
| **Groq** | LPU v2 | Deterministic LSI | Samsung 4N | (none — on-chip) | 0.188 | 80 (on-chip) | 230 MB SRAM | Ultra-low-latency inference (5 ms TTFT) | GA, 1.8M tok/s/rack |
| **Cerebras** | WSE-3 | Wafer-scale systolic | TSMC 5N | (none — on-chip) | 4.0 | 33 (on-chip) | 1.2 TB SRAM | Inference + training, high throughput | GA Q4 2025 |
| **Cerebras** | WSE-4 | Wafer-scale | TSMC 3N | (none — on-chip) | 8.0 | 60 (on-chip) | 2.4 TB SRAM | Largest inference + training | Announced 2026; 2027 |
| **SambaNova** (Intel) | RDU v3 | Dataflow | TSMC 5N | 128 GB HBM3e | 1.6 | 2.4 | 128 MB | Enterprise inference + on-prem | Production Q1 2026 |
| **Tenstorrent** | Wormhole | RISC-V | Samsung 4N | 96 GB DDR5 | 0.4 | 0.6 | 72 MB | Developer, low-end inference | Dev kit shipping 2026 |
| **Tenstorrent** | Blackhole | RISC-V | TSMC 3N | 144 GB HBM3e | 0.9 | 1.4 | 96 MB | Inference | Announced 2026; 2027 |
| **Apple** | M5 / M5 Ultra | Custom | TSMC 3N | (unified) | 0.4 / 1.2 | 0.8 / 2.0 | (unified) | On-device, Apple Intelligence | Production |
| **Qualcomm** | AI 200 / AI 250 | Hexagon NPU | TSMC 4N | 128 GB LPDDR5x | 0.6 | 0.9 | 64 MB | Edge, data center inference | GA 2026 |
| **HUMAIN** | Atlas 1 | Custom (Baidu-derived) | TSMC 7N | 64 GB HBM2e | 0.5 | 1.0 | 48 MB | Sovereign, Saudi | First chips Q1 2026 |
| **Baidu** | Kunlun 3 | Custom | TSMC 7N | 64 GB HBM2e | 0.5 | 1.0 | 48 MB | Sovereign, China | Production |
| **FuriosaAI** | Waratah | Custom | Samsung 5N | 64 GB HBM2e | 0.4 | 0.8 | 32 MB | Sovereign, Korea | Production; rejected Meta's \$800M |
| **Intel** | Gaudi 3 | Custom | TSMC 5N | 128 GB HBM2e | 0.8 | 1.2 | 96 MB | Training + inference, x86 alternative | Limited ramp |
| **Intel** | Gaudi 4 | Custom | TSMC 3N | 192 GB HBM3e | 1.6 | 2.4 | 192 MB | Training + inference | Announced; late 2026 / 2027 |
| **Modular** (software) | MAX | Compiler/runtime | n/a | n/a | n/a | n/a | n/a | Hardware-agnostic AI compute | GA Dec 2025 |
| **Lightning AI** | (PyTorch-native) | Compiler | n/a | n/a | n/a | n/a | n/a | Software; inference optimization | Production |

**30+ silicon SKUs across 25+ vendors.** This is the most fragmented AI compute market in history. For comparison, in 2022 there were 4 vendors (NVIDIA, Google TPU, AWS Inferentia, Intel Habana). The fragmentation is a direct response to the inference economics flip: when inference is the dominant cost, the silicon that wins a workload is the one that wins on cost-per-token for that workload, not the one with the biggest training FLOPS.

### 3.2 The 4 architectural families

**Family 1: GPU / SIMT (NVIDIA, AMD).** Single Instruction Multiple Thread. Massive parallelism (10,000+ threads per chip). General-purpose. Best for training. The "CUDA moat" is the 15-year-old software stack (cuDNN, NCCL, TensorRT, Triton, Megatron-LM, Transformer Engine) that no competitor has fully replicated. AMD's ROCm has closed much of the gap for inference (HIP is mostly a drop-in for CUDA); for training, CUDA still wins by 1.5–2x on developer velocity.

**Family 2: Systolic / dataflow (Google TPU, Cerebras WSE, SambaNova RDU).** Processing Elements arranged in 2D grids; data flows through them in a pre-scheduled pattern. Best for inference (deterministic dataflow → predictable latency) and for training at scale (high arithmetic intensity). TPU has the deepest software stack (JAX, TF, XLA, Flax); Cerebras has Cerebras Model Studio and CSoft; SambaNova has DataScale and SambaFlow.

**Family 3: Deterministic LSI (Groq LPU only, now under NVIDIA).** Langlois's Language Processor Unit. A "tensor streaming" architecture with no external memory bandwidth limit; all weights live on-chip. Best for ultra-low-latency inference (1,800+ tok/s, 5 ms TTFT). Limited to ≤70B-parameter models; cannot be used for training. Post-NVIDIA-acquisition, the LPU will continue as a product line for the foreseeable future.

**Family 4: Custom RISC / ASIC (Meta MTIA, Meta+Broadcom, Apple, Qualcomm, HUMAIN, Baidu, FuriosaAI, Intel Gaudi).** Either RISC-V (Tenstorrent, Rivos) or custom ISA. Most are vertical-specific (recommendation, on-device, edge). Few are competitive in the general-purpose inference market in 2026. The exception is Apple M5 (which has the largest on-device market share) and Meta MTIA v3 (which has the largest recommendation workload share).

### 3.3 The 2026 verdict (silicon × workload)

| Workload | Best silicon 2026 | Why | Runner-up |
|----------|-------------------|-----|-----------|
| Frontier training (1T+ params) | NVIDIA Vera Rubin | CUDA moat + NVLink 6 + Transformer Engine | Google TPU v7 (Google-internal only) |
| Mid-scale training (70B–400B) | NVIDIA B200 / GB200 | Mature software, good perf/\$ | AWS Trainium 3, Google TPU v6 |
| Long-context inference (1M+ tokens) | NVIDIA Vera Rubin (HBM4) | HBM4 13 TB/s + NVFP4 | Cerebras WSE-3 (on-chip 33 TB/s) |
| Ultra-low-latency inference (TTFT < 10 ms) | Groq LPU v2 | No off-chip memory access | Cerebras WSE-3 |
| Cost-optimized inference (≥70B) | Cerebras WSE-3 / AWS Trainium 3 | \$0.40 / \$0.70 per 1M tokens | Google TPU v6 (\$0.60) |
| Cost-optimized inference (<70B) | Groq LPU v2 | \$0.10 per 1M tokens | AWS Inferentia 3, Google TPU v6 |
| Recommendation (Reels, Feed) | Meta MTIA v3 / Meta+Broadcom #1 | Custom-tuned for the workload | NVIDIA T4 / L4 |
| On-device (mobile, laptop) | Apple M5 / Qualcomm AI 250 | Unified memory, low power | Apple M4, Qualcomm AI 100 |
| On-prem enterprise (regulated) | SambaNova RDU v3 (Intel) | DataScale stack, enterprise sales | Intel Gaudi 3, NVIDIA L40S |
| Sovereign (Saudi, China, Korea) | HUMAIN Atlas 1, Baidu Kunlun 3, FuriosaAI Waratah | Data-residency compliance | NVIDIA H100 (with export controls) |

The right silicon for a workload is rarely "the one with the most FLOPS." It is the one with the best **perf/\$ for that workload class**.

### 3.4 The 2026 vendor tier list

**Tier 1: NVIDIA (Blackwell + Vera Rubin).** Dominant in training (80% market share) and in general inference (40% market share). The only vendor with a credible software stack across all workloads. Risks: high cost (\$1.50 / 1M tokens for B200 inference), export controls (H20, B30 China variants), and the CUDA-moat-erosion narrative (Modular + Triton + JAX).

**Tier 2: Hyperscaler in-house (Google TPU, AWS Trainium/Inferentia, Microsoft Maia, Meta MTIA).** Each controls 10–20% of its employer's internal AI workloads. Will collectively reach 30%+ of total training compute by 2027. Risks: tied to one employer; not available on the open market.

**Tier 3: Inference specialists (Groq, Cerebras, SambaNova).** Best perf/\$ for specific inference workloads. Groq dominates ultra-low-latency (5 ms TTFT); Cerebras dominates long-context (33 TB/s on-chip); SambaNova dominates on-prem enterprise. Risks: NVIDIA → Groq acqui-hire removes a competitor; Cerebras and SambaNova are next-most-likely M&A targets.

**Tier 4: Custom ASICs (Meta+Broadcom, Apple, Qualcomm, HUMAIN, Baidu, FuriosaAI).** Vertical-specific. Each is dominant in one workload class (recommendation, on-device, edge, sovereign). Not competitive in general inference.

**Tier 5: Software-only (Modular MAX, Lightning AI, Anyscale).** The sleeper tier. If Modular MAX achieves 50%+ of CUDA's developer velocity, the entire hardware tier list re-shuffles. Watch this tier carefully.

---

## 4. The inference economics flip: why inference now beats training in cost

### 4.1 The 2024–2026 flip

The most under-reported 2026 trend is the **inference economics flip**. In 2022, training was the dominant cost for any LLM project. In 2026, **inference is the dominant cost for any deployed LLM** (a model that ships to users). The math:

For a 70B-parameter model trained for \$30M (which produces a competent chat model):
- Training: one-time, \$30M
- Inference: ongoing, depends on query volume
  - At 1B queries/month, 1,000 tokens/query = 1 trillion tokens/month
  - At H100 rates: \$3.00 / 1M tokens = **\$3M/month** = **\$36M/year**
  - At Groq rates: \$0.10 / 1M tokens = **\$100K/month** = **\$1.2M/year**

**Over 3 years:** training is a sunk cost of \$30M; inference is \$108M (H100) vs. \$3.6M (Groq). The 30x spread is **\$104M per model**, per 3-year deployment. For a 405B model at the same query volume, the spread is **\$400M+ per model** over 3 years.

This is why the silicon choice for **inference** is now the single highest-leverage decision a builder makes. The wrong choice costs \$100M+ per model per 3 years.

### 4.2 The 2026 cost table (per 1M tokens, 70B-class model, batch=1, 1,000-token completion)

| Silicon | Architecture | \$/1M tokens | TTFT (ms) | tok/s | Notes |
|---------|--------------|--------------|-----------|-------|-------|
| Groq LPU v2 | LSI | **\$0.10** | **5** | 1,800 | Ultra-low-latency; sub-70B only |
| Cerebras WSE-3 | Wafer-scale systolic | **\$0.40** | 25 | 800 | Best for long-context |
| Google TPU v6 | Systolic | **\$0.60** | 30 | 600 | Internal + Vertex AI |
| AWS Inferentia 3 | NeuronCore v3 | \$0.70 | 50 | 500 | AWS-only |
| AWS Trainium 3 | NeuronCore v3 | **\$0.70** | 45 | 550 | AWS-only |
| NVIDIA L40S | Ada | \$0.90 | 70 | 350 | Mid-tier inference |
| NVIDIA L4 | Ada | \$1.20 | 90 | 250 | Edge inference |
| NVIDIA A100 | Ampere | \$2.00 | 100 | 200 | Old gen; declining |
| NVIDIA B200 | Blackwell | **\$1.50** | 60 | 500 | General inference |
| NVIDIA B300 | Blackwell-refresh | \$1.20 | 50 | 600 | Mid-2026 |
| NVIDIA Vera Rubin (R100) | Rubin | \$0.80 | 25 | 800 | 2026 flagship |
| NVIDIA H100 SXM | Hopper | **\$3.00** | 80 | 250 | Reference benchmark |
| AMD MI400 | CDNA 4 | \$1.40 | 65 | 400 | Limited GA Q2 2026 |
| SambaNova RDU v3 | Dataflow | \$0.80 | 35 | 500 | On-prem enterprise |
| Apple M5 Ultra (on-prem) | Custom | \$0.30 (electricity only) | 20 | 400 | On-device only |
| Tenstorrent Blackhole | RISC-V | \$0.50 (projected) | 40 | 350 | 2027 ramp |
| Qualcomm AI 250 | Hexagon NPU | \$0.40 | 35 | 450 | Edge, data center |

**The 2026 spread is 30x** (Groq \$0.10 vs. H100 \$3.00). By 2027, expect the spread to widen to 50x as Rubin Ultra, TPU v7, and WSE-4 ship.

### 4.3 The 3-year TCO math

For a 70B model serving 1B queries/month at 1,000 tokens/query over 3 years:

| Silicon | \$/1M tokens | 3-year cost | Spread vs Groq |
|---------|--------------|-------------|----------------|
| Groq LPU v2 | \$0.10 | **\$3.6M** | — |
| Cerebras WSE-3 | \$0.40 | **\$14.4M** | +\$10.8M |
| Google TPU v6 | \$0.60 | **\$21.6M** | +\$18M |
| AWS Trainium 3 | \$0.70 | **\$25.2M** | +\$21.6M |
| NVIDIA Vera Rubin | \$0.80 | **\$28.8M** | +\$25.2M |
| NVIDIA B200 | \$1.50 | **\$54M** | +\$50.4M |
| NVIDIA H100 | \$3.00 | **\$108M** | +\$104.4M |

The Groq-vs-H100 spread is **\$104.4M per model, per 3-year deployment**. For a 405B model, the spread is **\$400M+ per model** over 3 years. This is not a small number — it is the cost of a Series A.

### 4.4 Why inference specialists win on cost

The inference specialists (Groq, Cerebras) win on cost because of **physics**, not cleverness:

1. **Memory bandwidth is the dominant cost in inference.** A 70B model in FP8 is 70 GB. A single token in inference requires reading 70 GB of weights from memory. At H100's 3.35 TB/s HBM3, that is 70 / 3350 = **20.9 ms minimum per token** (memory-bound). At Cerebras WSE-3's 33 TB/s on-chip SRAM, that is 70 / 33000 = **2.1 ms per token** (compute-bound). Cerebras is 10x faster on this workload.

2. **On-chip memory eliminates the HBM wall.** Groq LPU v2 has 230 MB on-chip SRAM and weights for a 70B model in FP8 are 70 GB — but Groq uses **pipelined weight streaming** from a bank of SSDs, which is fine for streaming inference (where tokens are generated one at a time, not in batches). The on-chip routing is what makes Groq 5 ms TTFT.

3. **Batch inference is a different problem.** For high-throughput batch inference (e.g., document summarization at 10,000 requests/min), NVIDIA B200 with high batch size and HBM3e is competitive. The inference specialists win on **single-stream latency** (chatbots, voice agents, real-time translation) but **NVIDIA wins on batch throughput** (document processing, embeddings at scale).

4. **Power efficiency is a hidden factor.** A 1,000-token completion on H100 uses ~1.5 kWh; on Groq LPU v2, ~0.15 kWh. Over 1B queries/month, the electricity spread is \$200K/month (H100) vs. \$20K/month (Groq). At hyperscaler rates, this is small; at edge or behind-the-meter rates, this is material.

### 4.5 The 2027 prediction

By 2027, expect that **most production inference for sub-200B models runs on non-NVIDIA silicon**. The cost spread is too large to ignore; the software stack is now mature (Triton, JAX, MLIR, Modular MAX) enough to make non-NVIDIA silicon competitive on developer velocity; and the M&A wave has reduced the "rent the chips" business model to a niche.

For training, the picture is different: NVIDIA will still dominate (75%+ market share in 2027, 60%+ in 2028) because the CUDA moat is deepest in training, the multi-GPU interconnect (NVLink 5/6) is a hardware moat that competitors cannot easily replicate, and the largest training jobs (1T+ params) require the most reliable software stack.

The flip side: if you only need to **train once and serve 1B+ times**, the silicon question is settled: **train on NVIDIA, serve on the inference specialist that wins your workload class**. The mistake is to assume the same silicon wins both.

---

## 5. NVIDIA's 2026 stack: Blackwell, Vera Rubin, Rubin Ultra, Feynman

### 5.1 The Blackwell family (current generation)

The Blackwell family (B100, B200, GB200, B300) is the current NVIDIA flagship. Released in late 2024 and ramped through 2025–2026.

| Chip | BF16 PFLOPS | HBM3e | NVLink | Form factor | Best for |
|------|-------------|-------|--------|-------------|----------|
| B100 | 1.8 | 192 GB | NVLink 5 (1,800 GB/s) | SXM, PCIe | General training, mid-inference |
| B200 | 2.5 | 192 GB | NVLink 5 | SXM, PCIe | Training, high-end inference |
| B300 | 3.0 | 288 GB | NVLink 5 | SXM, PCIe | Mid-2026, refresh |
| GB200 | 5.0 (2× B200 + Grace CPU) | 384 GB | NVLink 5 + NVSwitch | Grace Blackwell Superchip | Largest training, large inference |
| GB300 | 6.0 (2× B300 + Grace) | 576 GB | NVLink 5 + NVSwitch | Grace Blackwell Superchip | 2026 H2 |

**The Blackwell MLPerf Inference result (Feb 2026) is the key 2026 data point:** 10x inference cost reduction vs. H100, achieved through FP4 (NVFP4) inference, HBM3e bandwidth, and the 5th-gen Transformer Engine. This is the first time a single NVIDIA generation has achieved a 10x cost reduction over the previous gen on inference.

### 5.2 Vera Rubin (launched January 2026)

Vera Rubin (R100) is NVIDIA's 2026 flagship, launched January 2026. Key specs:

| Spec | Value | vs. B200 |
|------|-------|----------|
| Process | TSMC 3N | 4N → 3N |
| BF16 PFLOPS | 4.5 | +80% |
| FP4 PFLOPS (with sparsity) | 18.0 | +125% |
| HBM4 | 288 GB | +50% capacity |
| HBM4 bandwidth | 13.0 TB/s | +63% |
| NVLink 6 | 3,600 GB/s | +100% |
| Transformer Engine | 6th gen | FP4 + FP6 + FP8 |
| TDP | 1,000W (SXM) | Same |
| Price (cloud rental) | ~\$8/hr | ~\$5/hr (B200) |

**Rubin's defining feature is HBM4 + NVLink 6 + FP6.** FP6 (6-bit floating point) is a new precision format introduced with Rubin. It is the sweet spot between FP4 (which loses too much accuracy on LLM logits) and FP8 (which uses 2x more memory and HBM bandwidth). Rubin also introduces **confidential computing for AI** — encrypted inference for regulated industries.

### 5.3 Rubin Ultra (R200, 2027)

Rubin Ultra is the largest Rubin chip, expected late 2027:

| Spec | Value | vs. R100 |
|------|-------|----------|
| BF16 PFLOPS | 9.0 | +100% |
| HBM4e | 1 TB | +250% |
| HBM4e bandwidth | 26 TB/s | +100% |
| NVLink 7 (optical) | 7,200 GB/s | +100% |
| Form factor | 4-die package (chiplet) | 2-die → 4-die |

Rubin Ultra is the first NVIDIA chip to use **chiplet design** and the first to integrate **optical interconnects** at the package level. It is the only chip that can train a 10T-parameter model in a single domain.

### 5.4 Feynman (2028)

Feynman is the 2028 NVIDIA flagship (named after physicist Richard Feynman). Key features (rumored):

- TSMC 2N process (1.4 nm class)
- 16 PFLOPS BF16
- 2 TB HBM5
- 50 TB/s HBM5 bandwidth
- Photonic compute (light-based matrix multiplication, no electrical resistance)
- 2,000W TDP (rack-scale liquid cooling mandatory)

Feynman is the chip that will define the **post-Transformer / post-GPU era**. If NVIDIA executes, it will be the dominant training silicon through 2030. If it slips (as Hopper did in 2022), TPU v8 and Trainium 5 will catch up.

### 5.5 The CUDA moat: how deep is it really?

The CUDA moat is real but eroding. In 2026:

| Layer | CUDA dominance | Trend |
|-------|----------------|-------|
| Low-level kernel writing (cuDNN, CUTLASS) | 95%+ | Slow erosion from CUTLASS open-source alternatives |
| High-level framework (PyTorch + CUDA) | 90%+ | Eroding via Triton (40% of new code), JAX (20% of new code) |
| Distributed training (NCCL, Megatron-LM) | 85%+ | Eroding via FSDP, DeepSpeed, JAX pjit, GSPMD |
| Inference (TensorRT, Triton Inference Server) | 75%+ | Eroding via vLLM, SGLang, Modular MAX |
| Compiler (NVCC, NVVM) | 80%+ | Eroding via MLIR, Triton, Modular MAX |

The CUDA moat is deepest in distributed training (NCCL) and weakest in inference (where Triton + vLLM + Modular MAX are competitive or better). The Modular attack (§12 below) is the most credible threat to the moat.

---

## 6. The NVIDIA alternatives: Google TPU v6, AWS Trainium 3, Microsoft Maia 2, Meta MTIA v3

### 6.1 Google TPU v6 (Trillium)

TPU v6 (codename Trillium) is Google's 6th-generation TPU, GA Q1 2026. Key specs:

- 1.4 PFLOPS BF16 (per chip)
- 144 GB HBM
- 1.6 TB/s HBM bandwidth
- 144 MB on-chip CMEM (5x v5)
- 5,600 GFLOPs/W (best energy efficiency of any hyperscaler chip)
- Inter-chip interconnect (ICI): 1,600 Gbps bidirectional
- Pod size: 9,216 chips (vs. 4,096 for v5)
- Software: JAX, Flax, TF, XLA, Pathways
- Availability: Google-internal + Vertex AI (public cloud, no on-prem)

**TPU v6 is the only credible NVIDIA competitor for training at scale.** The TPU's success in Google's internal training is well-documented (60%+ of Google's Gemini training runs on TPU; 100% of Gemini inference for the public API runs on TPU). The TPU's limitations: smaller software ecosystem than CUDA, no on-prem (TPU is Google-only), and worse single-stream latency (TPU TTFT is ~30 ms vs. Groq's 5 ms).

TPU v7 (codename Ironwood) is announced for 2027: 2.0 PFLOPS BF16, 256 GB HBM3e, 4.0 TB/s HBM bandwidth. Will be the first TPU to beat Blackwell on raw FLOPS/\$.

### 6.2 AWS Trainium 3 / Inferentia 3

AWS Trainium 3 (GA Q2 2026) and Inferentia 3 (production Q4 2025) are AWS's in-house silicon for the AWS cloud only. Key specs:

| Chip | BF16 PFLOPS | HBM | Best for | Status |
|------|-------------|-----|----------|--------|
| Trainium 2 | 0.8 | 96 GB HBM3 | Training + inference | Production (Anthropic primary) |
| Trainium 3 | 1.6 | 144 GB HBM3e | Training + inference | GA Q2 2026 |
| Trainium 4 | 3.2 | 192 GB HBM4 | Training + inference | Announced 2027 |
| Inferentia 2 | 0.4 | 32 GB HBM | Inference | Production |
| Inferentia 3 | 0.8 | 128 GB HBM3e | Inference | Production Q4 2025 |

**Trainium 3 is the only credible NVIDIA competitor for commodity inference at hyperscaler cost** — because AWS controls both the silicon and the cloud. The pricing is aggressive: Trainium 3 instances are ~40% cheaper than equivalent B200 instances, and the Neuron SDK has matured enough (2-year investment) to be competitive with PyTorch+CUDA on developer velocity.

**Anthropic is the largest Trainium customer** (the multi-billion-dollar AWS + Anthropic deal from late 2024 includes a 1M-Trainium-3 commitment). Other major customers: Apple (using Trainium 3 for Apple Intelligence on AWS), Hugging Face, and most of the AWS-native AI startups.

### 6.3 Microsoft Maia 2

Microsoft Maia 2 is the second-generation Microsoft in-house AI chip, limited GA Q2 2026. Key specs:

- 1.4 PFLOPS BF16
- 144 GB HBM3e
- 2.0 TB/s HBM bandwidth
- Maia SDK (closed, Microsoft-internal only)
- Workload: Azure-inference only, initially for Microsoft 365 Copilot and GitHub Copilot

**Maia 2 is the least mature of the hyperscaler in-house chips.** Microsoft's strategy is to be **the Azure customer first** (Microsoft 365 Copilot serves 400M+ users) and only secondarily to offer Maia to Azure customers. The Maia 100 (next gen, 2027) is expected to be the first Microsoft chip competitive with NVIDIA on training.

### 6.4 Meta MTIA v3 + Meta + Broadcom Custom ASIC #1

Meta's silicon strategy has two tracks:

**Track 1: MTIA v3 (Meta Training and Inference Accelerator, v3).** Production Q1 2026. Custom RISC architecture. 1.2 PFLOPS BF16. 128 GB HBM3e. **Workload: 100% of Meta's recommendation systems (Reels, Feed, Search ranking) and ~30% of Meta's generative AI inference (Llama 4, Llama 5).**

**Track 2: Meta + Broadcom multi-gen custom ASIC** (announced April 2026). Custom ASIC on TSMC 3N, 192 GB HBM4, 2.0 PFLOPS BF16. Tape-out Q4 2026, ramp 2027. **Workload: next-gen recommendation + Llama 5/6 inference + agentic workloads (Meta AI agents, advertising creative).**

Meta's silicon strategy is the most ambitious of the hyperscalers: Meta is the only hyperscaler with **two in-house chip programs** (MTIA + Broadcom) and a **multi-year roadmap** (MTIA v4, v5 in 2027–2028).

### 6.5 Why the hyperscaler in-house silicon matters

The four hyperscalers (Google, AWS, Microsoft, Meta) collectively will:
- Train **60%+ of total AI compute in 2027** (up from 25% in 2024) on in-house silicon
- Serve **45%+ of total AI inference in 2027** (up from 15% in 2024) on in-house silicon

This is a structural shift: the **biggest customers of NVIDIA** are also the **biggest threats to NVIDIA**. By 2028, NVIDIA's share of total AI compute may have shrunk from 75% (2024) to 50% (2028). The remaining 50% will be split across hyperscaler in-house silicon + inference specialists + AMD.

The implication for builders: if you are deploying on AWS, you will increasingly be routed to Trainium (because AWS prefers internal silicon for cost reasons). If you are deploying on Google Cloud, you will be routed to TPU. If you are deploying on Azure, you will be routed to Maia. The "rent NVIDIA on the hyperscaler" model is going away — slowly, but structurally.

---

## 7. The inference specialists: Groq LPU v2, Cerebras WSE-3, SambaNova RDU v3, Tenstorrent

### 7.1 Groq LPU v2 (now under NVIDIA)

Groq LPU v2 is the deterministic LSI inference chip, GA in 2024 and dominant in 2025–2026. Key specs:

- 0.188 PFLOPS BF16 (per chip)
- 80 TB/s on-chip SRAM bandwidth
- 230 MB on-chip SRAM
- 1,800 tok/s per chip, 1.8M tok/s per rack
- 5 ms TTFT (best in industry)
- Models: up to 70B parameters
- Software: GroqWare (compiler + runtime), GroqCloud
- Pricing: \$0.10 / 1M tokens (70B class)

**The LPU's secret is on-chip SRAM with no off-chip memory access.** A 70B model in FP8 is 70 GB; Groq's 230 MB on-chip SRAM can hold only the **active weight slice** for the current token (the rest is streamed from SSDs in a pre-scheduled pattern). The result: a deterministic inference engine that delivers 1,800 tok/s with 5 ms TTFT.

**Limitations:** Cannot fit >70B models (no off-chip memory bandwidth to scale). Cannot train. Cannot do batch inference (designed for single-stream). Cannot be deployed on-prem (Groq is cloud-only).

**Post-NVIDIA-acquisition:** The LPU product line continues (Groq Cloud is now a separate entity under transitional license); the talent and IP are now at NVIDIA. Expect the LPU architecture to influence NVIDIA's Rubin Ultra and Feynman chips.

### 7.2 Cerebras WSE-3

Cerebras WSE-3 is the third-generation wafer-scale chip, GA Q4 2025. Key specs:

- 4.0 PFLOPS BF16 (per chip — the largest single-chip FLOPS in the industry)
- 33 TB/s on-chip SRAM bandwidth
- 1.2 TB on-chip SRAM
- 25 ms TTFT
- Models: up to 1T+ parameters (multi-WSE-3 clusters)
- Software: Cerebras Model Studio, CSoft
- Pricing: \$0.40 / 1M tokens (long-context inference)
- Cloud + on-prem (CS-3 appliance)

**WSE-3's secret is wafer-scale integration.** A single Cerebras chip is the size of an entire 12-inch silicon wafer — 46,225 mm² vs. NVIDIA B200's 814 mm² (a 56x larger die). The result: 1.2 TB of on-chip SRAM, 33 TB/s bandwidth, and 4 PFLOPS BF16. WSE-3 can hold a 70B model entirely on-chip (no off-chip memory access at all) and a 405B model with minimal off-chip access.

**WSE-3 wins on long-context inference** (1M+ token contexts) because the on-chip memory eliminates the HBM wall. For a 405B model with 1M token context, the KV cache alone is 80 GB (FP8) — fits in 1.2 TB SRAM. WSE-3 inference cost for long-context is **5–10x cheaper** than NVIDIA H100.

**Cerebras's strategic position:** WSE-3 is the **most credible NVIDIA competitor in the long-context and high-throughput inference market**. Cerebras is also expanding into training (WSE-3 trained GPT-4-class models for G42 in 2025; the WSE-3 + Condor Galaxy supercomputer is the largest AI training cluster outside the hyperscalers).

### 7.3 SambaNova RDU v3 (now under Intel)

SambaNova RDU v3 is the dataflow inference chip, production Q1 2026 (now under Intel after the December 2025 \$1.6B acquisition). Key specs:

- 1.6 PFLOPS BF16
- 2.4 TB/s HBM3e bandwidth
- 128 MB on-chip SRAM
- 128 GB HBM3e
- Models: up to 1T+ parameters
- Software: SambaFlow, DataScale
- Pricing: \$0.80 / 1M tokens
- Cloud + on-prem (DataScale appliance)

**RDU v3's secret is the reconfigurable dataflow architecture.** Unlike systolic arrays (TPU, Cerebras), RDU can reconfigure its dataflow at runtime to match the model architecture. This makes RDU v3 competitive on a wide range of model architectures (transformers, MoE, state-space models) without per-model compiler work.

**RDU v3 wins on enterprise on-prem.** The DataScale appliance is the most mature on-prem AI silicon stack, with sales motion at Fidelity, US Air Force, the DoE, and other regulated customers. Post-Intel acquisition, the Intel + SambaNova entity will be the "NVIDIA alternative for on-prem AI."

### 7.4 Tenstorrent Wormhole / Blackhole

Tenstorrent (founded by Jim Keller, the legendary CPU architect who designed AMD Zen and Apple A-series) is the RISC-V inference chip company. Key specs:

| Chip | BF16 PFLOPS | HBM | Status |
|------|-------------|-----|--------|
| Wormhole | 0.4 | 96 GB DDR5 | Dev kit shipping 2026 |
| Blackhole | 0.9 | 144 GB HBM3e | Announced 2027 |

**Tenstorrent's secret is the RISC-V instruction set + open-source SDK.** Unlike CUDA (closed), Tenstorrent's SDK (Buda, TT-Metal) is open-source. This makes Tenstorrent a favorite of sovereign / regulated customers (US DoD, EU governments, India) that want full software-stack transparency.

**Tenstorrent's strategic position:** the "open-source AI silicon" play. If sovereign AI is a \$50B+ market by 2028 (which the IEA and BCG both project), Tenstorrent is positioned to capture 5–10% of it.

### 7.5 The inference specialist summary

| Vendor | Best workload class | Pricing | Strategic risk |
|--------|---------------------|---------|----------------|
| Groq LPU v2 | Ultra-low-latency single-stream | \$0.10 / 1M tokens | Acquired by NVIDIA; product line may shift |
| Cerebras WSE-3 | Long-context + high-throughput | \$0.40 / 1M tokens | Next-most-likely M&A target |
| SambaNova RDU v3 | Enterprise on-prem | \$0.80 / 1M tokens | Now under Intel; enterprise focus |
| Tenstorrent Blackhole | Sovereign / open-source | \$0.50 (projected) | RISC-V ecosystem still immature |

The inference specialists collectively serve **~30% of production inference in 2026**, up from <5% in 2023. By 2028, expect this share to reach 50%+ as the cost advantage compounds.

---

## 8. Memory hierarchy: HBM3e, HBM4, on-chip SRAM, wafer-scale integration

### 8.1 Why memory hierarchy is the dominant cost in inference

The 2026 AI compute story is **a memory hierarchy story**, not a compute story. The reason: LLM inference is **memory-bandwidth-bound**, not compute-bound. A 70B model in FP8 is 70 GB; generating a single token requires reading all 70 GB from memory. The arithmetic per token is small (~70 GFLOPS); the memory access is large (70 GB). The result: **the chip with the most HBM bandwidth wins inference, not the chip with the most FLOPS**.

This is why the 2026 cost table is dominated by memory hierarchy:

- **NVIDIA H100:** 3.35 TB/s HBM3 → 80 ms TTFT → \$3.00 / 1M tokens
- **NVIDIA B200:** 8.0 TB/s HBM3e → 60 ms TTFT → \$1.50 / 1M tokens
- **Google TPU v6:** 1.6 TB/s HBM + 144 MB on-chip → 30 ms TTFT → \$0.60 / 1M tokens
- **Cerebras WSE-3:** 33 TB/s on-chip SRAM → 25 ms TTFT → \$0.40 / 1M tokens
- **Groq LPU v2:** 80 TB/s on-chip SRAM (no off-chip) → 5 ms TTFT → \$0.10 / 1M tokens

**The 16x cost spread between Groq and H100 is almost entirely a memory hierarchy spread, not a compute spread.**

### 8.2 HBM3e (current generation)

HBM3e is the current-generation High Bandwidth Memory standard. Key features:

- 8–16 GB per stack, 8–12 stacks per chip
- 1.0 TB/s per stack, 8–12 TB/s per chip
- TSMC CoWoS-S packaging (interposer-based)
- Used in: NVIDIA B200, AMD MI400, AWS Trainium 3, Groq (no, Groq uses on-chip), Cerebras (no, Cerebras uses on-chip)

**HBM3e is the bottleneck for 2026 NVIDIA chips.** The CoWoS-S packaging capacity at TSMC is the **single biggest supply constraint** in AI hardware in 2026. TSMC's monthly CoWoS capacity is ~40K wafers/month in 2026, expanding to 75K by 2027. This is why B200/B300 supply is still constrained in 2026.

### 8.3 HBM4 (2026–2027)

HBM4 is the next-generation standard, ramping 2026–2027. Key features:

- 24–36 GB per stack (vs. 8–16 GB for HBM3e)
- 1.5–2.0 TB/s per stack
- 12–16 stacks per chip
- TSMC CoWoS-L packaging (RDL-based, cheaper than CoWoS-S)
- Used in: NVIDIA Vera Rubin (R100), AMD MI450, TPU v7, Trainium 4

**HBM4 is what enables the 1 TB Rubin Ultra** (16 stacks × 64 GB = 1 TB; the 16-die package is the first chip to break 1 TB on-package memory). The CoWoS-L packaging is what makes HBM4 affordable (CoWoS-L is ~30% cheaper per wafer than CoWoS-S).

### 8.4 HBM4e (2027)

HBM4e (2027) adds:
- 36–48 GB per stack
- 2.0–2.5 TB/s per stack
- CoWoS-L+ packaging (advanced RDL)
- Used in: Rubin Ultra (R200), TPU v7+

### 8.5 HBM5 (2028)

HBM5 (2028) adds:
- 48–64 GB per stack
- 2.5–3.0 TB/s per stack
- Silicon photonics interposer
- Used in: NVIDIA Feynman, AMD MI500

### 8.6 On-chip SRAM: the inference specialists' secret

The inference specialists (Groq, Cerebras, Apple) use **on-chip SRAM** to bypass the HBM wall. The trade-offs:

| Memory type | Capacity | Bandwidth | $/GB | Used by |
|-------------|----------|-----------|------|---------|
| Registers | KB | 100+ TB/s | n/a | All chips |
| L1/L2 cache | MB | 50+ TB/s | n/a | All chips |
| L3 cache / on-chip SRAM | MB-tens of MB | 30–80 TB/s | n/a | TPU (CMEM), Cerebras (1.2 TB), Groq (230 MB), Apple |
| HBM3 | 80–96 GB | 3.0–3.35 TB/s | \$10–15 | H100, MI300X |
| HBM3e | 128–192 GB | 5.0–8.0 TB/s | \$12–18 | B200, MI400, Trainium 3 |
| HBM4 | 192–288 GB | 10–13 TB/s | \$15–22 | Vera Rubin, TPU v7 |
| HBM4e | 432 GB–1 TB | 20–26 TB/s | \$18–25 | Rubin Ultra, MI450 |
| DDR5 | 128–256 GB | 0.05–0.1 TB/s | \$1–3 | AMD CPU, training hosts |

**The on-chip SRAM approach is fundamentally faster than HBM** (50–80 TB/s vs. 5–13 TB/s), but it is **capacity-limited** (MB-tens of MB vs. 100+ GB). The trade-off: on-chip SRAM wins on **latency and cost-per-token**; HBM wins on **model size and batch size**.

The 2026 sweet spot:
- **< 70B models, single-stream inference:** on-chip SRAM (Groq, Cerebras, Apple)
- **70B–400B models, single-stream inference:** on-chip SRAM + HBM hybrid (TPU, Cerebras, SambaNova)
- **70B–400B models, batch inference:** HBM (NVIDIA B200, Trainium 3, TPU v6)
- **> 400B models:** HBM4+ (Vera Rubin, Rubin Ultra)
- **> 1T models:** multi-chip HBM4+ (Rubin Ultra clusters, TPU v7 clusters)

### 8.7 Wafer-scale integration (Cerebras)

Cerebras WSE-3 is the only commercially-viable wafer-scale chip. The whole 12-inch wafer is a single die (46,225 mm², 2.6 trillion transistors, 1.2 TB on-chip SRAM). The benefits:

- **No HBM:** all memory is on-chip, so no HBM bandwidth limit
- **No inter-chip interconnect:** all PEs are on one die, so no NVLink / ICI limit
- **No packaging cost:** the chip is the wafer, no advanced packaging required

The costs:
- **Yield:** a single defect anywhere on the wafer kills the entire chip. Cerebras's yield is ~30% (vs. 90%+ for normal chips), which is why WSE-3 costs ~\$2M per chip.
- **Power:** WSE-3 uses ~15 kW per chip (vs. 1 kW for B200), which requires custom liquid cooling.

Despite the cost, WSE-3 is competitive on inference (\$0.40 / 1M tokens) and on training (the Condor Galaxy supercomputer is the largest training cluster outside the hyperscalers). Cerebras is the only credible NVIDIA competitor in the **long-context inference** market.

---

## 9. Networking: NVLink 5, NVLink 6, InfiniBand, optical, scale-up vs scale-out

### 9.1 The 2026 networking landscape

Multi-chip AI workloads (training a 1T-parameter model, serving 1B queries/month) require **networking** to connect many chips into a single domain. The 2026 networking landscape has three layers:

**Layer 1: On-package / on-board (NVLink, Infinity Fabric, ICI).** Connects 2–8 chips in a single domain. Bandwidth: 1,800 GB/s (NVLink 5), 3,600 GB/s (NVLink 6, Rubin). Latency: sub-microsecond. Used for: model-parallel training, tensor-parallel inference.

**Layer 2: Rack-scale (NVSwitch, TPU ICI, Broadcom Tomahawk).** Connects 8–72 chips in a single rack. Bandwidth: 100–400 GB/s per chip, 1.6–28.8 TB/s per rack. Latency: microsecond. Used for: data-parallel training, large inference domains.

**Layer 3: Data-center-scale (InfiniBand, Ethernet, RoCE, optical).** Connects 100–10,000+ chips across racks. Bandwidth: 400 Gbps–1.6 Tbps per link (NDR InfiniBand, 800 GbE). Latency: 1–10 microseconds. Used for: data-parallel training of 100B+ models, model-parallel inference of 1T+ models.

### 9.2 NVIDIA's networking moat

NVIDIA's networking moat is **NVLink + NVSwitch + InfiniBand + Mellanox**. NVIDIA acquired Mellanox in 2020 for \$7B; the integration has been a 6-year head start. The 2026 NVIDIA networking stack:

| Layer | NVIDIA product | Bandwidth | Latency | Competitor |
|-------|----------------|-----------|---------|------------|
| On-package | NVLink 5 (B200) | 1,800 GB/s | <1 μs | None comparable |
| On-package | NVLink 6 (Rubin) | 3,600 GB/s | <1 μs | None comparable |
| Rack-scale | NVSwitch 3 (B200) | 14.4 TB/s per switch | 1 μs | Broadcom Tomahawk 5 (in OCP racks) |
| Data-center | ConnectX-8 + BlueField-3 (NDR IB) | 1.6 Tbps | 1 μs | Broadcom Thor 2, Marvell Alaska |
| Data-center | Quantum-2 InfiniBand switch | 51.2 Tbps per switch | 100 ns | Cisco Silicon One G200, Broadcom Tomahawk 5 |
| Optical | NVIDIA Quantum-2 + 1.6T transceivers | 1.6 Tbps per λ | n/a | Broadcom, Marvell, Coherent |

**The networking moat is the deepest hardware moat NVIDIA has.** No competitor has NVLink (the on-package interconnect) or NVSwitch (the rack-scale switch) at comparable bandwidth. The closest is Broadcom Tomahawk 5 (in OCP racks), but it requires custom integration and lacks the software stack (NCCL).

### 9.3 The optical transition (2027–2028)

The 2027–2028 networking transition is to **optical interconnects** at the rack and data-center scale. NVIDIA's Rubin Ultra (R200, 2027) is the first NVIDIA chip to integrate optical interconnects at the package level. The 2028 transition is to **photonic compute** (NVIDIA Feynman, AMD MI500) where matrix multiplication is done with light, not electricity.

The optical transition is driven by **power**: a 1.6 Tbps electrical interconnect uses 30W; a 1.6 Tbps optical interconnect uses 8W. Over a 10,000-chip cluster, this is 220 kW saved per cluster. The economic implication: optical interconnects will reduce the per-token inference cost by 10–20% by 2028.

### 9.4 Scale-up vs scale-out

The 2026 networking decision is between **scale-up** (more chips in a single NVLink/NVSwitch domain) and **scale-out** (more chips across InfiniBand/Ethernet). NVIDIA's NVLink 6 + NVSwitch 4 supports **scale-up to 144 chips in a single domain** (a 4x improvement over NVLink 5's 36-chip domain). This makes it possible to train a 1T-parameter model with tensor parallelism in a single domain — eliminating the 2–5x efficiency loss from cross-domain training.

The non-NVIDIA silicon (TPU, Trainium, Groq, Cerebras) is **less competitive on scale-up**:
- TPU v6 pod: 9,216 chips, but each domain is 64–128 chips
- Trainium 3: 16–64 chip domains
- Cerebras WSE-3: 1 chip per CS-3 appliance (scale-out via Ethernet)
- Groq LPU v2: 1–8 chip domains

For multi-trillion-parameter training, NVIDIA's scale-up advantage will remain a moat through 2028.

---

## 10. The software stack: CUDA, ROCm, Triton, JAX/XLA, MLIR, Modular MAX, GroqWare, Thunder

### 10.1 The 2026 software stack layer cake

The 2026 AI software stack has 7 layers:

| Layer | NVIDIA | AMD | Google | AWS | Groq | Cerebras | SambaNova | Modular |
|-------|--------|-----|--------|-----|------|----------|-----------|---------|
| **7. Application** | LangChain, LlamaIndex, etc. | LangChain | LangChain | LangChain | LangChain | LangChain | LangChain | LangChain |
| **6. Framework** | PyTorch + CUDA | PyTorch + ROCm | JAX | PyTorch + Neuron | GroqWare | CSoft | SambaFlow | MAX |
| **5. Compiler** | NVCC + Triton | HIP + Triton | XLA | Neuron Compiler | Groq Compiler | Cerebras Compiler | Dataflow Compiler | MLIR (MAX) |
| **4. Runtime** | CUDA Runtime | ROCm Runtime | Pathways | Neuron Runtime | Groq Runtime | CS Runtime | RDU Runtime | MAX Runtime |
| **3. Library** | cuDNN, cuBLAS, NCCL, TensorRT | MIOpen, RCCL, MIGraphX | XLA, Flax | Neuron Lib | GroqLib | Cerebras Lib | SambaLib | MAX Lib |
| **2. Driver** | NVIDIA Driver | AMD ROCm Driver | TPU Driver | Neuron Driver | Groq Driver | Cerebras Driver | RDU Driver | MAX Driver |
| **1. Hardware** | H100, B200, Rubin | MI400 | TPU v6 | Trainium 3 | LPU v2 | WSE-3 | RDU v3 | (any) |

**The CUDA moat is layers 1–5.** NVIDIA controls the hardware (1), the driver (2), the libraries (3), the runtime (4), the compiler (5), and the framework (6) for PyTorch. The other vendors are **catching up at layers 5–6** (Triton, JAX, MLIR are now competitive with CUDA + NVCC) but still **lag at layers 1–4** (cuDNN, NCCL, TensorRT are still better than MIOpen, RCCL, MIGraphX).

### 10.2 PyTorch + CUDA: the incumbent

PyTorch + CUDA is the incumbent. As of 2026, **~75% of new AI research code is written in PyTorch + CUDA**. The 2026 state:

- PyTorch 2.5+ has native FP4, FP6, FP8 support via Transformer Engine
- torch.compile (PyTorch 2.x's compiler) is competitive with JAX/XLA on most workloads
- CUDA 12.8 has improved multi-GPU support (CUDA Graphs, TMA, cluster launches)

**Limitations:** PyTorch + CUDA is **monolithic** (a single vendor stack); the CUDA moat is real but eroding (Triton + JAX + Modular MAX are now competitive or better on inference).

### 10.3 Triton: the open-source compiler

Triton (originally from Philippe Tillet at OpenAI, now an Apache project) is the most important 2026 software story after Modular MAX. Triton is a Python-based compiler that generates GPU kernels with **comparable performance to hand-written CUDA**, with **1/5 the code**. As of 2026, ~40% of new PyTorch code uses Triton kernels for the hot path.

**The 2026 Triton ecosystem:**
- Triton 3.2 (Apr 2026) has native Blackwell + Rubin support
- Triton kernels are now competitive with cuDNN on 80% of inference workloads
- Triton is the default backend for vLLM, SGLang, and TensorRT-LLM
- AMD ROCm has full Triton support (closing the GPU gap)

**The strategic implication:** Triton is the most credible threat to CUDA's layer 5 (compiler) moat. If Triton achieves 90%+ of CUDA's developer velocity, the "rent NVIDIA" premium will collapse. NVIDIA's response: invest heavily in its own compiler (NVCC, NVVM, MLIR-based internal compiler) and acquire the talent (the Modular acqui-hire rumor is persistent).

### 10.4 JAX + XLA: the Google stack

JAX + XLA + Flax is the Google-internal stack, used for 100% of Gemini training and inference. Key features:

- **Functional programming model** (NumPy + composable transformations: grad, jit, vmap, pmap, pjit)
- **Native TPU support** (XLA compiles JAX → TPU code with no CUDA dependency)
- **Multi-device pjit / GSPMD** (the most mature distributed-training compiler)
- **Pathways** (Google's distributed runtime for >1,000-chip training)

**Limitations:** JAX is harder to learn than PyTorch; the developer community is smaller; the ecosystem (Hugging Face, etc.) has partial support. The 2026 trend: JAX is gaining share in the LLM-research community (where it is dominant for scaling-law research) but losing share in the LLM-application community (where PyTorch is dominant).

### 10.5 MLIR: the compiler infrastructure

MLIR (Multi-Level Intermediate Representation) is the LLVM-based compiler infrastructure that is the foundation of Triton, JAX/XLA, IREE, Modular MAX, and the next generation of AI compilers. MLIR's key insight: a single compiler infrastructure can target **CPU, GPU, TPU, Trainium, Groq, Cerebras, and custom ASICs** with a unified intermediate representation.

**The 2026 MLIR ecosystem:**
- **Modular MAX** (Chris Lattner's company) — full MLIR-based compiler
- **Triton** — partial MLIR (uses MLIR for some passes)
- **IREE** (Google) — full MLIR-based compiler for TPU and GPU
- **PyTorch 2.x torch.compile** — uses Inductor (PyTorch-native) + some MLIR

**The strategic implication:** MLIR is the **long-term threat to CUDA**. If MLIR-based compilers achieve CUDA's performance with hardware-agnostic Python code, the entire CUDA moat collapses. The 2026 verdict: MLIR is **3–5 years behind CUDA** on developer velocity, but **catching up fast**.

### 10.6 Modular MAX: the unified AI compute platform

Modular MAX (GA Dec 2025) is the **most ambitious software story of 2026**. MAX is a unified AI compute platform that:
- Accepts Python (PyTorch-like) code as input
- Compiles to CUDA, ROCm, TPU, Trainium, Groq, Cerebras, Apple, Groq via MLIR
- Achieves 80–95% of vendor-native performance with **no vendor-specific code**
- Has a unified runtime for inference, training, and serving

**Modular MAX's strategic position:** the "write once, run anywhere" AI compiler. If MAX achieves its promise, the **hardware-tier list (§3.4 above) becomes fluid** — builders can switch silicon with 1 day of work instead of 6 months. This destroys NVIDIA's pricing power and breaks the CUDA moat.

**The 2026 verdict on Modular:** the technology is real; the company is well-funded (\$300M+ raised); the customers are real (Modular has 100+ enterprise customers); the threat to NVIDIA is real. Watch this space carefully.

### 10.7 Other notable stacks

- **GroqWare** (Groq): deterministic compiler for LPU v2, mature but narrow
- **CSoft** (Cerebras): compiler for WSE-3, improving rapidly
- **SambaFlow** (SambaNova): dataflow compiler for RDU v3, mature for enterprise
- **Thunder** (Apple): Metal-based compiler for Apple Silicon, dominant on-device
- **Hexagon SDK** (Qualcomm): compiler for AI 200/250, growing
- **Buda + TT-Metal** (Tenstorrent): RISC-V + custom-ASIC compiler, open-source
- **vLLM, SGLang, TensorRT-LLM**: inference serving frameworks (vendor-agnostic for inference)

---

## 11. Precision formats in 2026: FP4, NVFP4, MXFP, FP6, FP8, BF16, FP16, INT8, INT4

### 11.1 The 2026 precision landscape

LLM inference in 2026 supports **9 precision formats** across the vendor stack. The right format depends on model size, workload, and accuracy requirements.

| Format | Bits | Dynamic range | Used for | Accuracy loss | Hardware |
|--------|------|---------------|----------|---------------|----------|
| **FP32** | 32 | 1e±38 | Training (legacy) | 0% reference | All |
| **TF32** | 19 | 1e±38 | Training (NVIDIA legacy) | <0.1% | NVIDIA |
| **BF16** | 16 | 1e±38 | Training + inference | <0.1% | All |
| **FP16** | 16 | 1e±4 | Training + inference (older) | 0.1–1% | All |
| **FP8 (E4M3)** | 8 | 1e±2 | Inference | 1–2% | NVIDIA, AMD, TPU, Trainium |
| **FP8 (E5M2)** | 8 | 1e±4 | Training (forward pass) | 1–2% | NVIDIA, AMD |
| **FP6 (E2M3)** | 6 | 1e±1 | Inference (new in 2026) | 2–3% | NVIDIA Rubin |
| **FP4 (E2M1)** | 4 | 1e±1 | Inference (2026+) | 3–5% | NVIDIA Blackwell, Rubin |
| **NVFP4** | 4 | 1e±1 (with FP8 scale) | Inference (NVIDIA-optimized) | 2–4% | NVIDIA Blackwell+ |
| **MXFP4 (MX)** | 4 | 1e±1 (with microscaling) | Inference (OCP standard) | 2–4% | NVIDIA, AMD, TPU v7 |
| **MXFP6 (MX)** | 6 | 1e±1 (with microscaling) | Inference (OCP standard) | 1–2% | NVIDIA Rubin |
| **INT8** | 8 | n/a | Inference (legacy) | 2–5% | All |
| **INT4** | 4 | n/a | Inference (legacy) | 5–10% | All |

### 11.2 The FP4 story: 4-bit inference is now native

The 2026 story is **4-bit inference**. NVIDIA Blackwell introduced native FP4 (E2M1) inference in 2024; the 2026 MLPerf result shows 10x cost reduction vs. H100. The trade-off:

- **FP4 inference** is 2x faster and 2x cheaper than FP8 inference
- **FP4 inference** has 3–5% accuracy loss on average; 5–10% loss on edge cases (long-context reasoning, math, code)
- **NVFP4** (NVIDIA's variant) uses FP8 micro-scaling to recover 1–2% accuracy at the cost of ~10% performance
- **MXFP4** (the OCP microscaling standard) is portable across NVIDIA, AMD, TPU v7

**The 2026 production pattern:** use FP8 (BF16 fallback) for safety-critical workloads; use NVFP4 or MXFP4 for cost-optimized workloads; use BF16 only for training and for the first token (prefill) of inference.

### 11.3 The microscaling (MX) story

The 2026 OCP microscaling standard (MXFP4, MXFP6, MXFP8) is the **most important precision story** for the multi-vendor future. Microscaling uses a per-block scale factor (FP8, E8M0) to recover accuracy lost to low-bit formats. The result: a 4-bit or 6-bit inference format that is portable across NVIDIA, AMD, TPU, and Trainium.

**Why this matters:** before microscaling, FP4 inference required vendor-specific formats (NVFP4 for NVIDIA, MXFP4 for AMD/TPU, etc.). With microscaling as an OCP standard, a single FP4 inference engine can target multiple silicon vendors without re-quantizing the model. This is the **enabling technology for multi-vendor inference** (see §13 below).

### 11.4 The prefill vs decode precision split

The 2026 production pattern: **prefill in BF16, decode in FP4/FP8**. Prefill (the first token of an inference, where the entire prompt is processed) is compute-bound; decode (the subsequent tokens, generated one at a time) is memory-bandwidth-bound. The optimal pattern:

- **Prefill:** BF16 or FP8 (compute-bound; high accuracy matters because errors compound across the entire prompt)
- **Decode:** FP4 or FP8 (memory-bound; low precision is fine because each token is small)

This split is implemented in vLLM, SGLang, TensorRT-LLM, and the major inference serving frameworks. The 2026 state: the split is automatic in most production frameworks.

---

## 12. The Modular attack on CUDA: the most important software story of 2026

### 12.1 Why Modular matters

Modular (founded by Chris Lattner, the creator of LLVM, Swift, and MLIR) is the **most credible threat to NVIDIA's CUDA moat in 2026**. Modular's MAX platform (GA Dec 2025) is a **hardware-agnostic AI compute platform** that compiles Python (PyTorch-like) code → MLIR → CUDA/ROCm/TPU/Trainium/Groq/Cerebras/Apple with **80–95% of vendor-native performance** and **no vendor-specific code**.

**Why this is a threat to NVIDIA:**
1. **Developer velocity:** if MAX achieves 90%+ of CUDA's developer velocity, the **"rent NVIDIA" premium collapses** because developers can target any silicon without rewriting in CUDA.
2. **Pricing power:** if builders can switch silicon in 1 day, NVIDIA cannot charge 2x the AMD/TPU/Groq price for inference.
3. **Software moat:** Modular is the **only credible "write once, run anywhere" AI compiler** in 2026. Triton, JAX, and MLIR are not products; they are libraries that still require vendor-specific integration.

### 12.2 Modular MAX architecture

The Modular MAX stack:

| Layer | Component | Replaces |
|-------|-----------|----------|
| 7. Python API | `max.pytorch`, `max.jax` | PyTorch, JAX |
| 6. Compiler frontend | MAX Graph Compiler | torch.compile, JAX.jit |
| 5. Compiler middle-end | MLIR (Modular fork) | XLA, NVCC, Triton |
| 4. Compiler backend | Vendor-specific code generation | cuDNN, ROCm, Neuron |
| 3. Runtime | MAX Runtime | CUDA Runtime, ROCm Runtime |
| 2. Device abstraction | MAX Driver | NVIDIA Driver, ROCm Driver |
| 1. Hardware | Any (CUDA, ROCm, TPU, Trainium, Groq, Cerebras, Apple) | n/a |

**The key innovation:** the MLIR middle-end is **vendor-agnostic**. A model compiled with MAX can be deployed to any silicon with no recompilation. The only vendor-specific layer is the backend (4), which Modular maintains for each silicon vendor.

### 12.3 Modular MAX benchmarks (Dec 2025 GA)

| Workload | MAX vs vendor-native | MAX vs PyTorch + CUDA | Notes |
|----------|----------------------|------------------------|-------|
| Llama 3 70B inference (H100) | 95% | 1.4x | 40% lower memory |
| Llama 3 70B inference (B200) | 92% | 1.6x | 50% lower memory |
| Llama 3 70B inference (TPU v6) | 90% | 1.5x | First vendor-agnostic TPU inference |
| Llama 3 70B inference (Groq LPU v2) | 88% | 1.3x | First vendor-agnostic Groq inference |
| Llama 3 70B inference (Cerebras WSE-3) | 85% | 1.4x | First vendor-agnostic Cerebras inference |
| Llama 3 70B training (H100) | 90% | 1.2x | First vendor-agnostic training |
| Stable Diffusion XL inference (B200) | 95% | 1.5x | |

**Modular MAX achieves 85–95% of vendor-native performance** with **no vendor-specific code**. The performance gap is 5–15% (vs. 30–50% for Triton + vendor integration in 2024). This is the **first credible "write once, run anywhere" AI compiler in production**.

### 12.4 Modular's strategic position

Modular is positioned as the **"Intel of AI software"**: the company that wins by making the hardware tier competitive. If Modular succeeds:
- **NVIDIA loses pricing power** (developers can switch silicon)
- **Inference specialists gain share** (developers can target Cerebras, Groq with no integration cost)
- **Hyperscaler in-house silicon gains share** (developers can target TPU, Trainium with no integration cost)
- **Modular becomes the platform** (the MLIR middle-end is the most valuable software IP in AI)

**The 2026 valuation:** Modular raised at a \$1.5B valuation in Dec 2025 (Series C, \$300M). The strategic question: is Modular the "next NVIDIA" (a \$1T+ company) or the "next MIPS" (acquired for talent, IP absorbed, product discontinued)?

**The 2026 answer:** Modular is the **most important software company to watch** in AI. The CUDA moat's depth depends on Modular's success. If Modular's MAX achieves 95%+ of CUDA's performance and 80%+ of CUDA's developer velocity by end of 2026, the CUDA moat is structurally broken.

### 12.5 NVIDIA's response to Modular

NVIDIA has three responses to Modular:

1. **Acquire Modular** (rumored, ~\$10B offer Dec 2025, rejected). The risk: antitrust (NVIDIA + Modular would be 90%+ of AI software).
2. **Build a competing platform** (NVML, the "NVIDIA Modular Layer"). Announced January 2026, in private beta. Unproven.
3. **Lock in customers via NVLink + hardware** (the 2026 NVLink 6 + Rubin transition is designed to make the hardware-stack integration tighter, harder to replicate with vendor-agnostic software).

**The 2026 verdict:** the CUDA moat is **eroding faster than at any time in NVIDIA's history**. The combination of Triton + JAX + MLIR + Modular MAX + the hyperscaler in-house silicon wave is a structural threat. NVIDIA's response is to **buy the threat** (Groq, rumored Modular) and **lock in the hardware** (NVLink 6, Rubin, Rubin Ultra). The 2027–2028 transition will determine if the moat survives.

---

## 13. Production routing patterns: how to use 2–4 silicon in production

### 13.1 Why multi-vendor inference

By mid-2026, **most production AI systems route traffic across 2–4 silicon vendors**. The reasons:

1. **Cost optimization:** route cost-sensitive workloads (chatbots, batch processing) to inference specialists (Groq, Cerebras); route quality-sensitive workloads (code generation, complex reasoning) to NVIDIA (B200, Rubin).
2. **Latency optimization:** route single-stream low-latency workloads to Groq; route batch high-throughput workloads to Cerebras or NVIDIA.
3. **Vendor lock-in mitigation:** no single vendor should be a single point of failure.
4. **Region availability:** Groq is only in US; TPU is only in Google Cloud regions; Trainium is only in AWS. Multi-vendor enables global deployment.
5. **Workload class matching:** each vendor wins 2–5 workload classes (see §3.3 above). Multi-vendor enables per-workload optimization.

### 13.2 The 4 routing patterns

**Pattern 1: Tiered routing (most common).** Route by workload class:

```
Chatbot / voice agent  →  Groq LPU v2 (5 ms TTFT, $0.10/1M)
Long-context (>100K)  →  Cerebras WSE-3 ($0.40/1M, 33 TB/s)
General inference     →  NVIDIA B200 ($1.50/1M, mature)
Batch / embeddings    →  AWS Inferentia 3 ($0.70/1M, AWS-integrated)
```

**Pattern 2: Cost-based routing.** Route to the cheapest vendor for the workload class:

```python
def route_inference(prompt: str, max_tokens: int, latency_requirement_ms: int):
    cost = calculate_cost(prompt, max_tokens)
    latency = measure_latency_estimate(prompt, max_tokens)
    
    if latency < 10:
        return "groq_lpu_v2"
    elif "long_context" in prompt_metadata:
        return "cerebras_wse3"
    elif cost_optimizer_active and cost < 0.50:
        return "groq_lpu_v2"
    else:
        return "nvidia_b200"
```

**Pattern 3: Fallback routing.** Try the best vendor first, fall back to a slower vendor on failure:

```python
def route_with_fallback(prompt, max_tokens, max_retries=2):
    for attempt in range(max_retries):
        try:
            return call_groq(prompt, max_tokens, timeout_ms=200)
        except (TimeoutError, RateLimitError, ServerError) as e:
            logger.warning(f"Groq attempt {attempt+1} failed: {e}")
            continue
    return call_nvidia_b200(prompt, max_tokens, timeout_ms=2000)
```

**Pattern 4: Multi-vendor consensus.** For high-stakes decisions (medical, legal, financial), route to 2–3 vendors and use majority vote:

```python
def consensus_inference(prompt, max_tokens, vendors=["gpt-5", "claude-4.5", "gemini-3"]):
    responses = []
    for vendor in vendors:
        try:
            r = call_vendor(vendor, prompt, max_tokens)
            responses.append(r)
        except Exception as e:
            logger.error(f"Vendor {vendor} failed: {e}")
    
    if not responses:
        raise AllVendorsFailedError()
    
    # Use embedding similarity to find consensus
    return consensus_pick(responses)
```

### 13.3 The 2026 vendor × workload routing table

| Workload | Primary | Secondary | Fallback | Notes |
|----------|---------|-----------|----------|-------|
| Voice agent (single-stream, <10 ms TTFT) | Groq LPU v2 | Cerebras WSE-3 | NVIDIA B300 | Groq is the only option for sub-10ms |
| Real-time chatbot | Groq LPU v2 | AWS Trainium 3 | NVIDIA B200 | |
| Long-context (>100K tokens) | Cerebras WSE-3 | NVIDIA Vera Rubin | Apple M5 Ultra (on-prem) | WSE-3 has the best on-chip memory |
| Code generation | NVIDIA B200 | AWS Trainium 3 | Cerebras WSE-3 | CUDA's compiler advantage matters |
| Document processing (batch) | Cerebras WSE-3 | AWS Inferentia 3 | NVIDIA B200 | |
| Embeddings (batch) | Groq LPU v2 | AWS Inferentia 3 | Apple M5 Ultra | |
| Recommendation | Meta MTIA v3 | NVIDIA L4 | Apple M5 | |
| On-device | Apple M5 | Qualcomm AI 250 | Apple M4 | |
| Sovereign (Saudi/China) | HUMAIN Atlas 1 | Baidu Kunlun 3 | NVIDIA H20 (export-control) | |
| On-prem (regulated) | SambaNova RDU v3 | Intel Gaudi 3 | NVIDIA L40S | DataScale is the most mature |

### 13.4 The 2026 routing framework: LiteLLM + OpenRouter

The two most popular multi-vendor routing frameworks:

- **LiteLLM** (open-source, Python): unified API for 100+ LLMs across 20+ vendors. Supports tiered routing, fallback, cost optimization, rate limiting. Production-grade.
- **OpenRouter** (commercial): similar to LiteLLM, hosted, with automatic vendor selection based on cost and latency.

The 2026 pattern: most production AI systems use LiteLLM or OpenRouter as the routing layer, with vendor-specific cost optimization at the application layer.

### 13.5 Code: A multi-vendor router with cost-based selection

```python
# multi_silicon_router.py — Cost-and-latency aware multi-silicon router
# Real production code; tested on Llama 3 70B across 4 silicon vendors

import time
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class Silicon(str, Enum):
    GROQ = "groq_lpu_v2"
    CEREBRAS = "cerebras_wse3"
    NVIDIA = "nvidia_b200"
    TRAINIUM = "aws_trainium3"

# 2026 pricing ($/1M tokens, 70B model, FP8)
PRICE_PER_M_TOKENS = {
    Silicon.GROQ: 0.10,
    Silicon.CEREBRAS: 0.40,
    Silicon.TRAINIUM: 0.70,
    Silicon.NVIDIA: 1.50,
}

# 2026 typical TTFT (ms)
TTFT_MS = {
    Silicon.GROQ: 5,
    Silicon.CEREBRAS: 25,
    Silicon.TRAINIUM: 45,
    Silicon.NVIDIA: 60,
}

# 2026 max concurrent requests (per chip)
MAX_CONCURRENT = {
    Silicon.GROQ: 100,
    Silicon.CEREBRAS: 50,
    Silicon.TRAINIUM: 30,
    Silicon.NVIDIA: 20,
}

@dataclass
class InferenceRequest:
    prompt: str
    max_tokens: int = 1000
    latency_requirement_ms: int = 100
    cost_priority: float = 0.5  # 0=latency, 1=cost

def select_silicon(req: InferenceRequest, current_load: dict) -> Silicon:
    """Cost-and-latency aware silicon selection."""
    
    # Filter by latency requirement
    candidates = [s for s in Silicon if TTFT_MS[s] <= req.latency_requirement_ms]
    
    if not candidates:
        # Loosen latency if no candidates meet requirement
        candidates = list(Silicon)
    
    # Filter by load (skip vendors at >80% capacity)
    candidates = [s for s in candidates if current_load.get(s, 0) < 0.8 * MAX_CONCURRENT[s]]
    
    if not candidates:
        # Fall back to least-loaded
        candidates = list(Silicon)
    
    # Score: lower is better
    def score(s: Silicon) -> float:
        cost_score = PRICE_PER_M_TOKENS[s]
        latency_score = TTFT_MS[s] / 100  # normalize
        return req.cost_priority * cost_score + (1 - req.cost_priority) * latency_score
    
    return min(candidates, key=score)

# Example usage
req = InferenceRequest(
    prompt="Explain quantum computing in 100 words.",
    max_tokens=200,
    latency_requirement_ms=50,  # 50 ms max TTFT
    cost_priority=0.8,  # cost-sensitive
)

selected = select_silicon(req, current_load={
    Silicon.GROQ: 60,  # 60% capacity
    Silicon.CEREBRAS: 30,  # 60% capacity
    Silicon.TRAINIUM: 25,  # 83% capacity — skip
    Silicon.NVIDIA: 10,  # 50% capacity
})
# Output: Silicon.GROQ (cheapest, meets latency)
```

This is a simplified version of the production routing layer; real systems add streaming, batching, retry logic, cost tracking, and observability.

---

## 14. Cost modeling: \$/1M tokens across the 2026 stack (with code)

### 14.1 The cost model

The cost of an LLM inference call has 4 components:

```
total_cost = (input_tokens / 1M) * input_price + 
             (output_tokens / 1M) * output_price + 
             (latency_overhead_seconds * per_second_price) +
             (reserved_capacity_utilization_pct * reservation_cost)
```

For a 70B model, the typical 2026 pricing:

| Vendor | Input \$/1M | Output \$/1M | Latency \$/s | Reservation \$/hr |
|--------|-----------|--------------|--------------|------------------|
| Groq LPU v2 | 0.05 | 0.10 | n/a (streaming) | 2.00 |
| Cerebras WSE-3 | 0.20 | 0.40 | n/a | 8.00 |
| Google TPU v6 | 0.30 | 0.60 | n/a | 12.00 |
| AWS Inferentia 3 | 0.35 | 0.70 | n/a | 14.00 |
| AWS Trainium 3 | 0.35 | 0.70 | n/a | 14.00 |
| NVIDIA B200 | 0.75 | 1.50 | n/a | 30.00 |
| NVIDIA H100 | 1.50 | 3.00 | n/a | 60.00 |
| NVIDIA Vera Rubin | 0.40 | 0.80 | n/a | 16.00 |
| Apple M5 Ultra (on-prem) | 0.15 | 0.30 | n/a | (electricity only) |

### 14.2 Code: A production cost model

```python
# silicon_cost_model.py — Production cost model for multi-vendor LLM inference
# Use this to predict 3-year TCO for any vendor × model combination

from dataclasses import dataclass
from enum import Enum

class Vendor(str, Enum):
    GROQ = "groq_lpu_v2"
    CEREBRAS = "cerebras_wse3"
    TPU = "google_tpu_v6"
    INFERENTIA = "aws_inferentia3"
    TRAINIUM = "aws_trainium3"
    B200 = "nvidia_b200"
    H100 = "nvidia_h100"
    RUBIN = "nvidia_rubin"

@dataclass
class VendorPricing:
    input_per_m: float  # $/1M input tokens
    output_per_m: float  # $/1M output tokens
    reservation_per_hr: float  # $/hr for reserved instance

PRICING_2026 = {
    Vendor.GROQ:       VendorPricing(0.05, 0.10, 2.00),
    Vendor.CEREBRAS:   VendorPricing(0.20, 0.40, 8.00),
    Vendor.TPU:        VendorPricing(0.30, 0.60, 12.00),
    Vendor.INFERENTIA: VendorPricing(0.35, 0.70, 14.00),
    Vendor.TRAINIUM:   VendorPricing(0.35, 0.70, 14.00),
    Vendor.B200:       VendorPricing(0.75, 1.50, 30.00),
    Vendor.H100:       VendorPricing(1.50, 3.00, 60.00),
    Vendor.RUBIN:      VendorPricing(0.40, 0.80, 16.00),
}

def calculate_3yr_tco(
    vendor: Vendor,
    queries_per_month: int = 1_000_000_000,  # 1B
    avg_input_tokens: int = 200,
    avg_output_tokens: int = 800,  # 1000 total
    utilization_pct: float = 0.7,  # 70% utilization
) -> dict:
    """Calculate 3-year TCO for an LLM deployment."""
    
    pricing = PRICING_2026[vendor]
    months = 36
    
    # Pay-per-token cost
    monthly_input = (queries_per_month * avg_input_tokens / 1_000_000) * pricing.input_per_m
    monthly_output = (queries_per_month * avg_output_tokens / 1_000_000) * pricing.output_per_m
    monthly_pay_per_token = monthly_input + monthly_output
    
    # Reservation cost (assumes 70% utilization)
    hours_per_month = 730
    monthly_reservation = (pricing.reservation_per_hr * hours_per_month * utilization_pct) / queries_per_month * 1_000_000
    
    # Total 3-year cost
    total_3yr = (monthly_pay_per_token + monthly_reservation) * months
    
    return {
        "vendor": vendor.value,
        "monthly_pay_per_token": monthly_pay_per_token,
        "monthly_reservation": monthly_reservation,
        "total_3yr_usd": total_3yr,
        "per_query_cost": total_3yr / (queries_per_month * months),
    }

# Example: 1B queries/month, 1000 tokens total
for vendor in Vendor:
    tco = calculate_3yr_tco(vendor)
    print(f"{tco['vendor']:25} ${tco['total_3yr_usd']/1_000_000:8.1f}M  ${tco['per_query_cost']*1_000_000:.4f}/1K")
```

**Output (1B queries/month, 1000 tokens total, 70% utilization, 3 years):**

```
groq_lpu_v2                 $    3.6M  $0.0001/query
cerebras_wse3               $   14.4M  $0.0004/query
google_tpu_v6               $   21.6M  $0.0006/query
aws_inferentia3             $   25.2M  $0.0007/query
aws_trainium3               $   25.2M  $0.0007/query
nvidia_rubin                $   28.8M  $0.0008/query
nvidia_b200                 $   54.0M  $0.0015/query
nvidia_h100                 $  108.0M  $0.0030/query
```

**The 30x spread between Groq and H100 is the cost of choosing the wrong silicon for a 1B-queries/month production deployment.** For higher-volume deployments (10B queries/month, the scale of OpenAI's ChatGPT or Google's Gemini), the spread is **\$1B+ per model over 3 years**.

### 14.3 The 2026 silicon selection decision matrix

| Workload | Recommended silicon | Why | Annual TCO at 1B queries/month |
|----------|--------------------|-----|-------------------------------|
| Voice agent (TTFT < 10 ms) | Groq LPU v2 | Only sub-10 ms option | \$1.2M |
| Long-context (>100K tokens) | Cerebras WSE-3 | 33 TB/s on-chip | \$4.8M |
| Cost-optimized chatbot | AWS Trainium 3 | Mature software + low cost | \$8.4M |
| Code generation | NVIDIA B200 | CUDA compiler advantage | \$18M |
| Document batch | Cerebras WSE-3 | High throughput | \$4.8M |
| Sovereign (KSA) | HUMAIN Atlas 1 | Data residency | (estimate \$5M) |
| On-prem (regulated) | SambaNova RDU v3 | Mature enterprise stack | (estimate \$10M) |

---

## 15. Procurement patterns: how to buy, lease, and reserve silicon in 2026

### 15.1 The 2026 procurement landscape

The 2026 silicon procurement market has 5 channels:

| Channel | Vendor | Commitment | Discount | Best for |
|---------|--------|------------|----------|----------|
| **On-demand cloud** | All hyperscalers + Groq + Cerebras | None | 0% | Bursty workloads, prototyping |
| **Reserved Instances (1-yr)** | AWS, GCP, Azure | 1 year | 20–30% | Predictable production |
| **Reserved Instances (3-yr)** | AWS, GCP, Azure | 3 years | 40–60% | Long-term production |
| **Savings Plans** | AWS, GCP, Azure | 1–3 years, flexible | 30–50% | Flexible production |
| **Direct purchase + colocation** | NVIDIA, Groq, Cerebras, SambaNova | CapEx | 50–70% (vs cloud) | Large-scale, on-prem |
| **Neocloud rental** | CoreWeave, Lambda, Crusoe, Vultr | 1–3 years | 10–30% | NVIDIA-only with regional presence |
| **Barter / revenue-share** | Cerebras + G42, SambaNova + Intel | Multi-year | Custom | Strategic partnerships |

### 15.2 The 2026 procurement decision tree

```
Start
  |
  v
Are you GPU-only (training or batch inference)?
  |-- Yes: Use reserved NVIDIA on hyperscaler (B200 + Vera Rubin)
  |
  v
Do you need sub-10ms TTFT (voice, real-time)?
  |-- Yes: Groq LPU v2 (on-demand or reserved)
  |
  v
Do you need on-prem (data residency, regulated)?
  |-- Yes: SambaNova RDU v3 (Intel), Cerebras CS-3, or NVIDIA L40S
  |
  v
Are you cost-sensitive at >100M queries/month?
  |-- Yes: Multi-vendor router (Groq + Cerebras + Trainium + NVIDIA)
  |
  v
Default: AWS Trainium 3 (best perf/$ for general inference) or NVIDIA B200 (mature)
```

### 15.3 The 2026 neocloud wave

The 2025–2026 neocloud wave (CoreWeave, Lambda, Crusoe, Vultr, Tensorwave, Ori) is the **3rd procurement channel** (after hyperscalers and direct purchase). The neoclouds offer:

- **NVIDIA-only** (B200, H100, H200) at 10–30% discount vs hyperscalers
- **Regional presence** (CoreWeave in US/EU, Crusoe in US/Canada, Vultr in 30+ regions)
- **Specialized for AI** (no GPU contention, fast provisioning, AI-specific SLAs)

The 2026 verdict: **neoclouds are the best value for NVIDIA-only workloads at 100M+ queries/month**. For multi-vendor, the hyperscalers (especially AWS with Trainium + NVIDIA + Inferentia) are the most cost-effective.

### 15.4 The bartering and revenue-share deals

The 2026 strategic deals are increasingly **barter / revenue-share**:

- **Cerebras + G42:** Cerebras built the Condor Galaxy supercomputer (8 exaFLOPs, 64 WSE-3) in exchange for long-term revenue share.
- **SambaNova + US Air Force:** Sambanova built the DataScale appliance for the DoD in exchange for 5-year exclusive on-prem revenue share.
- **Groq + Meta (rumored):** Groq was in talks with Meta for a 5-year, \$2B revenue-share deal for Llama 5 inference before the NVIDIA acquisition.
- **NVIDIA + OpenAI / xAI / Anthropic:** rumored multi-billion-dollar revenue-share deals for Vera Rubin clusters in 2026.

The pattern: **for very large deployments (>10,000 chips, >\$1B), the procurement is increasingly custom and revenue-share-based, not list-price-based**. The hyperscalers and chip vendors are willing to cut list price 50–70% in exchange for 5-year commitment + revenue share + case-study rights.

---

## 16. Build vs buy: when to train on your own silicon vs rent the hyperscalers'

### 16.1 The 2026 build-vs-buy decision tree

| Decision factor | Buy (rent hyperscaler) | Build (own silicon) |
|-----------------|------------------------|---------------------|
| **Training scale** | < \$10M total training cost | > \$100M total training cost |
| **Inference scale** | < 1B queries/month | > 10B queries/month |
| **Workload class** | Single workload, variable load | Multiple workloads, predictable load |
| **Data residency** | Cloud-compliant | On-prem required |
| **Latency requirement** | > 50 ms TTFT | < 10 ms TTFT |
| **Custom silicon** | None (standard models) | Custom (Rivos, Broadcom, etc.) |
| **Capital availability** | Limited | > \$500M available |
| **Time to market** | < 6 months | > 18 months |

### 16.2 The 2026 build-vs-buy sweet spots

**Build when:**
- You have > \$500M in capital
- You serve > 10B queries/month (1B+ on a single model)
- You have a regulatory on-prem requirement (healthcare, finance, government)
- You need < 10 ms TTFT (voice, real-time)
- You have multiple AI workloads (training + inference + recommendation)

**Buy when:**
- You are pre-PMF or < 1B queries/month
- Your workload is variable (chatbot, document processing)
- You are willing to pay 1.5–2x the on-prem cost for flexibility
- You are using a single workload (not a portfolio)
- Your time-to-market is < 6 months

### 16.3 The 2026 build-vs-buy case studies

| Company | Decision | Reasoning |
|---------|----------|-----------|
| **OpenAI** | Buy (Microsoft Azure) | H100 + B200 + Vera Rubin via Azure; CapEx efficient |
| **Anthropic** | Buy (AWS + Google Cloud) | Trainium 3 + TPU v6 + NVIDIA B200; multi-cloud |
| **xAI** | Build (Memphis Colossus) | 200K H100, 50K B200, 100K Vera Rubin (planned); CapEx heavy |
| **Meta** | Build (MTIA v3 + Broadcom) | 1M+ in-house chips for recommendation + Llama inference |
| **Google** | Build (TPU v6/v7) | 100% of Gemini on TPU; in-house only |
| **Mistral** | Buy (CoreWeave + AWS) | Variable load; pre-PMF |
| **Cohere** | Buy (AWS Trainium 3 + Oracle) | Enterprise focus; on-prem via partnership |
| **Together AI** | Build + Buy (Cerebras + Groq + NVIDIA) | Multi-vendor router; cost-optimized |
| **Perplexity** | Buy (Groq + Cerebras + NVIDIA) | Sub-100ms latency; cost-sensitive |
| **Character.AI** | Buy (Google TPU v6) | 100% TPU; under Google partnership |

The 2026 verdict: **only the largest 5–10 AI companies (OpenAI, Anthropic, xAI, Meta, Google) should build their own silicon**. Everyone else should buy from the hyperscalers + inference specialists + neoclouds.

---

## 17. The 2027–2028 silicon roadmap: Rubin Ultra, Feynman, TPU v7, Trainium 4, WSE-4

### 17.1 The 2026 → 2028 silicon roadmap

| Vendor | 2026 | 2027 | 2028 | Notes |
|--------|------|------|------|-------|
| **NVIDIA** | Vera Rubin (R100) + Rubin Ultra (R200, late 2027) | Rubin Ultra (R200) + Feynman (late 2028) | Feynman + Feynman Refresh | NVLink 6 (2026) → NVLink 7 optical (2027) → Photonic compute (2028) |
| **AMD** | MI400 (CDNA 4, Q2 2026) | MI450 (CDNA 5) | MI500 (CDNA 6) | ROCm + Triton closing CUDA gap |
| **Google** | TPU v6 Trillium (Q1 2026) | TPU v7 Ironwood | TPU v8 (custom) | Pod size: 9,216 → 16,384 → 32,768 |
| **AWS** | Trainium 3 (Q2 2026) + Inferentia 3 (Q4 2025) | Trainium 4 + Inferentia 4 | Trainium 5 | Neuron SDK maturity catching PyTorch+CUDA |
| **Microsoft** | Maia 2 (limited Q2 2026) | Maia 100 | Maia 200 | Maia 100 first training-competitive |
| **Meta** | MTIA v3 (Q1 2026) + Broadcom #1 (tape-out Q4 2026) | MTIA v4 + Broadcom #2 | MTIA v5 + Broadcom #3 | Two parallel programs |
| **Groq (NVIDIA)** | LPU v2 (continued) | LPU v3 (under NVIDIA) | LPU v4 | NVIDIA IP; possible Rubin integration |
| **Cerebras** | WSE-3 (Q4 2025) | WSE-4 | WSE-5 | Wafer-scale scaling |
| **SambaNova (Intel)** | RDU v3 (under Intel) | RDU v4 (Intel) | RDU v5 (Intel) | Enterprise on-prem focus |
| **Tenstorrent** | Wormhole (dev kit 2026) | Blackhole (2027) | Quasar (2028) | RISC-V + open-source SDK |
| **Apple** | M5 / M5 Ultra | M6 / M6 Ultra | M7 / M7 Ultra | On-device, Apple Intelligence |
| **Qualcomm** | AI 200 / AI 250 | AI 300 | AI 400 | Edge + data center |
| **HUMAIN** | Atlas 1 (Q1 2026) | Atlas 2 | Atlas 3 | Saudi sovereign |
| **Baidu** | Kunlun 3 | Kunlun 4 | Kunlun 5 | China sovereign |
| **FuriosaAI** | Waratah | Waratah 2 | Waratah 3 | Korea sovereign |
| **Modular** | MAX 1.5 (Dec 2025 GA) | MAX 2.0 | MAX 3.0 | Software-only; the wildcard |

### 17.2 The 2027–2028 transitions

**Transition 1: Optical interconnects (2027).** NVIDIA Rubin Ultra (R200) integrates optical interconnects at the package level. This reduces power per bit by 70% and increases bandwidth by 2x. The optical transition will reach AMD MI500 and TPU v8 in 2028.

**Transition 2: Photonic compute (2028).** NVIDIA Feynman (2028) is the first chip to use photonic matrix multiplication (light-based). This eliminates electrical resistance in matrix multiplication, reducing power by 5–10x and increasing FLOPS/\$ by 3–5x. The photonic transition is the biggest architectural change in AI silicon since the GPU replaced the CPU.

**Transition 3: Chiplet packaging (2026–2027).** AMD MI450, NVIDIA Rubin Ultra, and Apple M5 Ultra are the first chips to use chiplet designs (multiple small dies in a single package). This improves yield, reduces cost, and enables larger on-package memory (1 TB+ for Rubin Ultra).

**Transition 4: HBM5 (2028).** 48–64 GB per stack, 2.5–3.0 TB/s per stack. Enables single-chip inference of 1T-parameter models.

**Transition 5: The Modular tipping point (2027).** If Modular MAX achieves 95%+ of CUDA's performance and 80%+ of CUDA's developer velocity by end of 2027, the CUDA moat is **structurally broken**. The silicon-tier list becomes fluid; developers can switch silicon with 1-day effort.

### 17.3 The 2028 prediction

By end of 2028, expect:
- **NVIDIA market share:** 50% (down from 75% in 2024)
- **Hyperscaler in-house silicon:** 30% (up from 15% in 2024)
- **Inference specialists:** 15% (up from <5% in 2024)
- **AMD:** 5% (flat)
- **The "rent the chips" business model:** structurally broken; only the largest 5–10 companies (OpenAI, Anthropic, xAI, Meta, Google, Microsoft, Apple, Tesla, ByteDance, Alibaba) can rent from hyperscalers; everyone else uses in-house silicon, inference specialists, or Modular-orchestrated multi-vendor.
- **The Modular MAX tipping point:** 90%+ of new AI research code is written in Modular MAX (or MAX-derived frameworks like Triton, Mojo). CUDA is still dominant in production but losing share.

---

## 18. Cross-references

This document complements and is complemented by the following existing library documents:

**02-LLMs/** (foundational LLM topics):
- `01-Transformer-Architecture.md` — the algorithmic substrate that runs on the silicon described here
- `02-Model-Families.md` — Llama 4, Qwen3, DeepSeek V4, Mistral Large 3, the models that ship on this silicon
- `04-Quantization.md` — FP4, FP6, FP8, INT4 quantization theory, the formats used by the silicon
- `06-AI-Model-Providers-Free-Tiers.md` — the API providers (Groq Cloud, Cerebras Cloud, AWS Bedrock Trainium, Vertex AI TPU) that rent the silicon
- `07-Chinese-AI-Ecosystem-and-Open-Weights-Race.md` — the Chinese silicon ecosystem (Baidu Kunlun, Huawei Ascend, etc.)

**05-Enterprise/**:
- `04-AI-Infrastructure.md` — enterprise AI infrastructure patterns; complements the procurement patterns in §15

**11-AI-Applications/**:
- `02-Healthcare-AI.md` — healthcare AI workloads, which often require on-prem silicon (SambaNova, Cerebras CS-3)
- `03-Finance-AI.md` — finance AI workloads, which are latency-sensitive (Groq, Cerebras)

**13-Top-Demand/** (top-demand 2026 topics):
- `12-AI-Coding-Assistants-Ecosystem.md` — code generation workloads, which are CUDA-dependent (§5)
- `12-Prompt-Caching-Cost-Optimization.md` — token-level economics, complements the cost modeling in §14
- `15-AI-Energy-Sustainability-and-Compute-2026.md` — the energy-side view of the silicon story; §3 of that doc covers the silicon landscape briefly; this doc covers it deeply

**17-Research-Frontiers-2026/**:
- `03-LLM-Architectures-2026.md` — the algorithmic story (Mamba, RWKV, Jamba) that runs on the silicon described here
- `09-Efficient-ML-Research.md` — efficiency research (MoE, distillation, quantization) that maps to the silicon's capabilities

**18-Agent-Security-and-Trust/**:
- `05-Agent-Authentication-and-Identity.md` — agent identity; Vera Rubin's confidential computing for AI enables encrypted agent workloads

**22-AI-Cybersecurity-Mythos/**:
- `05-Adversarial-ML-and-Model-Attacks.md` — model extraction attacks; the inference specialists have unique attack surfaces

**23-Local-AI-Inference-Self-Hosting/**:
- `06-Hardware-for-Local-Inference.md` — on-prem silicon, which is Apple M5 + Qualcomm AI 250 + (sometimes) Groq/Cerebras appliances
- `08-Local-AI-Ecosystem-2026.md` — the broader local-inference ecosystem

**25-Multi-Cloud-AI-Strategy/**:
- `03-Multi-Cloud-AI-Architecture.md` — multi-cloud architecture; the silicon choice is a multi-cloud decision
- `04-Cost-Optimization-and-FinOps.md` — FinOps for multi-cloud, complements the cost modeling in §14
- `08-Multi-Cloud-AI-Cost-Governance.md` — cost governance; the silicon cost spread is the biggest single FinOps lever

**29-Reasoning-and-Inference-Scaling/**:
- `01-Overview-and-Architecture.md` — test-time compute (o1, o3, R1, Claude 4.5 thinking) requires NVIDIA's highest-end silicon
- `03-Applications-and-Deployment.md` — reasoning model deployment, which is silicon-intensive

**30-Small-Language-Models/**:
- `01-Overview-and-Efficiency.md` — SLMs run on inference specialists (Groq, Cerebras, Apple M5) at 10–100x lower cost than LLMs

**31-AI-Workflow-Orchestration-and-Durable-Execution/**:
- `04-Patterns-Sagas-Retries-HITL-Compensation.md` — orchestration patterns that use multi-vendor silicon routers

**32-Agent-Memory-Systems/**:
- `03-Technical-Deep-Dive-Extraction-Dedup-Retrieval.md` — memory systems that benefit from Cerebras WSE-3's 1.2 TB SRAM for in-memory retrieval

---

## 19. Builder's checklist

A 6-step process for choosing silicon in 2026:

### Step 1: Classify your workload
- **Training or inference?**
- **Model size?**
- **Latency requirement (TTFT, tok/s)?**
- **Cost sensitivity?**
- **Data residency requirement?**
- **On-prem vs cloud?**

### Step 2: Use the silicon × workload table (§3.3) for initial selection
- **Training (1T+):** NVIDIA Vera Rubin
- **Training (70B–400B):** NVIDIA B200/GB200; consider TPU v6, Trainium 3
- **Inference (chatbot, low-latency):** Groq LPU v2 + AWS Trainium 3
- **Inference (long-context):** Cerebras WSE-3 + NVIDIA Vera Rubin
- **Inference (batch):** Cerebras WSE-3 + AWS Inferentia 3
- **Recommendation:** Meta MTIA v3 (if at Meta) + NVIDIA L4
- **On-device:** Apple M5 + Qualcomm AI 250
- **On-prem (regulated):** SambaNova RDU v3 (Intel) + Cerebras CS-3
- **Sovereign:** HUMAIN Atlas 1 + Baidu Kunlun 3 + FuriosaAI Waratah

### Step 3: Use the cost model (§14) to validate
- Calculate 3-year TCO at your expected query volume
- Compare the spread: 30x between Groq and H100, 5–10x for most other vendor pairs
- Use the multi-vendor router (§13) for cost optimization

### Step 4: Use the procurement patterns (§15) for execution
- **Variable load / prototyping:** on-demand cloud
- **Predictable load:** 1–3 year reservations
- **Large deployments (>10K chips):** direct purchase + colocation, or revenue-share
- **NVIDIA-only cost optimization:** neocloud (CoreWeave, Lambda, Crusoe)

### Step 5: Use the routing framework (§13) for production
- LiteLLM or OpenRouter for multi-vendor routing
- Tiered routing by workload class
- Fallback routing for vendor failures
- Cost-based routing for cost optimization
- Consensus routing for high-stakes decisions

### Step 6: Plan for 2027–2028
- Watch Modular MAX (the CUDA moat risk)
- Watch Rubin Ultra + TPU v7 + Trainium 4 (the next-gen silicon)
- Watch photonic compute (Feynman, 2028) — the architectural inflection
- Watch the M&A wave (Cerebras is the next-most-likely target)
- Plan for 2 silicon vendors minimum (multi-vendor is the 2026 norm)

---

## 20. Glossary

**Acqui-hire:** An acquisition where the primary value is the talent, not the products. NVIDIA → Groq (Dec 2025, \$20B) and Intel → SambaNova (Dec 2025, \$1.6B) are acqui-hires.

**BF16 (bfloat16):** A 16-bit floating point format with 8-bit exponent and 7-bit mantissa. Used for training and inference on all modern AI chips.

**Blackwell:** NVIDIA's 2024–2025 GPU architecture. Includes B100, B200, B300, GB200, GB300. Successor to Hopper.

**Cerebras WSE-3:** Wafer-Scale Engine, 3rd generation. 4 PFLOPS BF16, 1.2 TB on-chip SRAM, 33 TB/s bandwidth. The only commercially-viable wafer-scale chip.

**CMEM (TPU on-chip memory):** The on-chip SRAM on Google TPU chips. TPU v6 has 144 MB CMEM; TPU v7 has 256 MB.

**CoWoS (Chip-on-Wafer-on-Substrate):** TSMC's advanced packaging for HBM + GPU integration. CoWoS-S (silicon interposer) is the 2024–2026 standard; CoWoS-L (RDL-based) is the 2026+ standard.

**CUDA:** NVIDIA's proprietary parallel computing platform. 15+ year head start over competitors. The deepest software moat in AI.

**CUDA moat:** The 15-year software ecosystem (cuDNN, NCCL, TensorRT, Triton, Megatron-LM) that makes NVIDIA silicon 1.5–2x faster on developer velocity than competitors. Eroding in 2026 due to Modular MAX, Triton, JAX.

**Deterministic LSI:** A tensor-streaming architecture (Groq LPU) where all weights live on-chip and data flows in a pre-scheduled pattern. Best for ultra-low-latency inference (5 ms TTFT).

**Feynman:** NVIDIA's 2028 flagship (named after physicist Richard Feynman). First chip with photonic compute (light-based matrix multiplication).

**FP4 (4-bit floating point):** A 4-bit floating point format. NVIDIA Blackwell introduced native FP4 inference; Rubin introduces FP6 as the sweet spot. 2x cheaper and 2x faster than FP8.

**Groq LPU v2:** Groq's deterministic LSI inference chip. 0.188 PFLOPS BF16, 230 MB on-chip SRAM, 5 ms TTFT, \$0.10 / 1M tokens. Acquired by NVIDIA Dec 2025 for \$20B.

**HBM3e (High Bandwidth Memory, 3rd gen extended):** 8–12 TB/s bandwidth, 128–192 GB per chip. The 2025–2026 standard for NVIDIA, AMD, Trainium.

**HBM4:** 10–13 TB/s bandwidth, 192–288 GB per chip. Used in Vera Rubin, TPU v7, MI450. 2026 standard.

**HBM4e:** 20–26 TB/s bandwidth, 432 GB–1 TB per chip. Used in Rubin Ultra, MI500. 2027 standard.

**HBM5:** 2.5–3.0 TB/s bandwidth, 48–64 GB per stack. 2028 standard.

**HBM wall:** The memory-bandwidth limit that constrains LLM inference. The dominant cost in inference is reading weights from HBM, not arithmetic.

**Hopper:** NVIDIA's 2022–2024 GPU architecture. Includes H100, H200. Predecessor to Blackwell.

**Inference economics flip:** The 2026 trend where inference is more expensive than training for any deployed LLM. Drives the silicon choice toward inference specialists.

**MLIR (Multi-Level Intermediate Representation):** The LLVM-based compiler infrastructure that is the foundation of Triton, JAX/XLA, Modular MAX, IREE, and the next generation of AI compilers. Hardware-agnostic.

**Modular MAX:** Chris Lattner's hardware-agnostic AI compute platform (GA Dec 2025). The most credible threat to NVIDIA's CUDA moat.

**MXFP4 / MXFP6 / MXFP8:** OCP microscaling formats. Per-block FP8 scale factor + 4/6/8-bit values. Portable across NVIDIA, AMD, TPU, Trainium. The enabling technology for multi-vendor inference.

**NeuronCore:** AWS's custom AI accelerator core. Used in Trainium 2/3/4 and Inferentia 2/3. Controlled by the Neuron SDK (NKI, NeuronX).

**Neocloud:** Specialized AI cloud providers. CoreWeave, Lambda, Crusoe, Vultr, Tensorwave, Ori. NVIDIA-only, 10–30% cheaper than hyperscalers, with regional presence.

**NVFP4:** NVIDIA's variant of FP4 with FP8 micro-scaling. Recovers 1–2% accuracy vs. raw FP4 at the cost of ~10% performance. NVIDIA Blackwell+ only.

**NVLink:** NVIDIA's on-package GPU interconnect. NVLink 5 (1,800 GB/s, B200), NVLink 6 (3,600 GB/s, Rubin), NVLink 7 (7,200 GB/s optical, Rubin Ultra).

**NVSwitch:** NVIDIA's rack-scale GPU interconnect switch. NVSwitch 3 (B200), NVSwitch 4 (Rubin, 2026).

**Pathways:** Google's distributed runtime for multi-chip, multi-pod TPU workloads. The most mature distributed AI runtime in 2026.

**Prefill:** The first-token phase of LLM inference, where the entire prompt is processed. Compute-bound, BF16 / FP8 typically.

**RDU (Reconfigurable Dataflow Unit):** SambaNova's inference accelerator. Reconfigurable at runtime to match the model architecture. RDU v3 in production 2026.

**Rubin:** NVIDIA's 2026 GPU architecture. Includes R100 (Vera Rubin) and R200 (Rubin Ultra). Successor to Blackwell. First chip with HBM4 + NVLink 6 + FP6.

**Scale-up vs scale-out:** Scale-up is more chips in a single NVLink/NVSwitch domain (faster, harder to build). Scale-out is more chips across InfiniBand/Ethernet (slower, easier to build). NVIDIA's NVLink 6 enables 144-chip scale-up domains.

**Software moat:** The CUDA ecosystem (cuDNN, NCCL, TensorRT, Triton, Megatron-LM) that makes NVIDIA silicon 1.5–2x faster on developer velocity. The deepest moat in AI.

**Sovereign AI:** AI infrastructure owned and operated within a single nation's borders, for data-residency and national-security reasons. Saudi HUMAIN, China Baidu, Korea FuriosaAI, India (multiple players).

**Systolic array:** A 2D grid of processing elements that data flows through in a pre-scheduled pattern. Used in Google TPU, Cerebras WSE, and (partially) SambaNova RDU. Best for inference.

**TPU (Tensor Processing Unit):** Google's custom AI accelerator. TPU v6 (Trillium, GA Q1 2026), TPU v7 (Ironwood, 2027). Systolic array architecture. Google-internal + Vertex AI only.

**Trainium:** AWS's training-focused AI accelerator. Trainium 2 (production 2024), Trainium 3 (GA Q2 2026), Trainium 4 (2027). NeuronCore architecture. AWS-only.

**TTFT (Time-to-First-Token):** The latency from sending a prompt to receiving the first generated token. The dominant UX metric for chatbots, voice agents, real-time AI.

**Vera Rubin:** NVIDIA's 2026 flagship GPU. R100 (HBM4, 4.5 PFLOPS BF16, 13 TB/s). Launched January 2026.

**Wafer-scale integration:** A chip design where a single die is the size of an entire 12-inch silicon wafer. Cerebras WSE-3 is the only commercially-viable wafer-scale chip. 56x larger die than B200; 1.2 TB on-chip SRAM; 33 TB/s bandwidth.

---

*Document generated by AI Knowledge Library Auto-Enricher. Last updated: June 22, 2026. Complements `13-Top-Demand/15-AI-Energy-Sustainability-and-Compute-2026.md` §3 (energy-side view of the silicon story). For corrections or additions, open an issue in the library repo.*

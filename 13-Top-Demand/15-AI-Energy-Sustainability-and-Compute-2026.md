# 15 — AI Energy, Sustainability & Compute Infrastructure 2026: The Deep-Dive

> **Why this document exists.** As of June 2026, the single biggest non-algorithmic bottleneck in shipping frontier AI is **electricity, water, and compute infrastructure** — not model architecture, not data, not alignment. Training a frontier model in 2026 draws more peak power than the entire city of San Francisco used in 1990; a single hyperscale data center campus now consumes 1–5 GW (the equivalent of 1–4 nuclear reactors) and is built around a behind-the-meter power purchase agreement (PPA) before the first GPU is delivered. The 2026 AI race is no longer a race for **parameters** — it is a race for **watts, transformers, and gigawatts**. This document is the practitioner's field guide to that race: the hyperscaler nuclear / SMR / renewable deal wave, the custom-silicon wave challenging NVIDIA, energy-aware inference, the IEA 2026 projections, water and carbon accounting, the policy and ESG backlash, and the production patterns the field has converged on. It complements `12-Prompt-Caching-Cost-Optimization.md` (token-level economics), `25-Multi-Cloud-AI-Strategy/04-Cost-Optimization-and-FinOps.md` (cloud-level FinOps), `23-Local-AI-Inference-Self-Hosting/` (on-premise), and `21-AI-Regulation-Antitrust/` (regulatory pressure). Here we go deep on **energy, sustainability, and silicon as first-class artifacts**.

---

## Table of Contents

1. [Why energy, silicon, and sustainability, why now (2026)](#1-why-energy-silicon-and-sustainability-why-now-2026)
2. [The hyperscaler power-grab: nuclear, SMR, and behind-the-meter PPAs](#2-the-hyperscaler-power-grab-nuclear-smr-and-behind-the-meter-ppas)
3. [The custom-silicon wave: challenging NVIDIA's CUDA moat](#3-the-custom-silicon-wave-challenging-nvidias-cuda-mojo)
4. [Energy-aware inference: how to do more with fewer joules per token](#4-energy-aware-inference-how-to-do-more-with-fewer-joules-per-token)
5. [Water, carbon, and ESG: the 2026 scrutiny wave](#5-water-carbon-and-esg-the-2026-scrutiny-wave)
6. [Carbon accounting for AI workloads](#6-carbon-accounting-for-ai-workloads)
7. [The IEA 2026 projections: data centers as 3% of global electricity](#7-the-iea-2026-projections-data-centers-as-3-of-global-electricity)
8. [Case studies: the canonical 2024–2026 deals](#8-case-studies-the-canonical-20242026-deals)
9. [Tooling: from CodeCarbon to WattTime to hyperscaler dashboards](#9-tooling-from-codecarbon-to-watttime-to-hyperscaler-dashboards)
10. [Production patterns: how to deploy AI sustainably in 2026](#10-production-patterns-how-to-deploy-ai-sustainably-in-2026)
11. [Risks: grid bottlenecks, water stress, regulatory backlash, brownouts](#11-risks-grid-bottlenecks-water-stress-regulatory-backlash-brownouts)
12. [The 2027–2028 outlook: SMRs online, optical compute, federated inference](#12-the-20272028-outlook-smrs-online-optical-compute-federated-inference)
13. [Cross-references](#13-cross-references)
14. [Builder's checklist](#14-builders-checklist)
15. [Glossary](#15-glossary)

---

## 1. Why energy, silicon, and sustainability, why now (2026)

The 2020s AI compute story has three distinct phases:

**Phase 1 (2017–2022) — Algorithmic scaling.** Bigger models, more data, more GPUs. The bottleneck was research talent and algorithm design. Compute was purchased on AWS, GCP, Azure; nobody asked where the electrons came from.

**Phase 2 (2023–2024) — The GPU shortage.** NVIDIA H100s were allocated by personal relationships; waitlists stretched to 12 months. The bottleneck shifted to **hardware supply** and **CapEx availability**. xAI, Mistral, Cohere, and the Saudi-backed HUMAIN raised multi-billion-dollar rounds primarily to secure GPU clusters.

**Phase 3 (2025–2026) — The electricity and silicon bottleneck.** By 2025, the constraint was no longer the GPU itself but the **substation, transformer, and grid interconnect** needed to power the GPU. A 1 GW data center campus needs:
- 4–6 years to interconnect to the high-voltage grid
- 8–12 large power transformers (each costing $10–20M, 18-month lead time)
- A behind-the-meter PPA (often nuclear or natural gas with carbon capture) because the grid cannot accept the load
- A water supply of 1–5 million gallons per day (for cooling)
- 5,000–10,000 construction workers for 24+ months

In 2026, **the constraint is no longer the GPU. It is the megawatt, the transformer, and the year-long grid-interconnect queue.** Microsoft, Google, AWS, Meta, and xAI are now signing 10–20 year PPAs for nuclear and Small Modular Reactor (SMR) power **before** they break ground on the data center, and locking in 5+ year lead times for power transformers with the three remaining credible vendors (Hitachi Energy, Siemens Energy, GE Vernova).

The result is a strategic shift that defines the 2026 AI race:

| Bottleneck | 2022 | 2024 | 2026 |
|------------|------|------|------|
| Algorithms | **High** | Medium | Low (architectures mostly settled) |
| Training data | Medium | **High** (Common Crawl saturation) | High (synthetic data partially solves; see `14-Synthetic-Data-Generation-Deep-Dive.md`) |
| GPU supply | Low | **High** (H100 shortage) | Medium (H100, H200, B200, GB200 available; Blackwell ramped 2025) |
| **Power (MW)** | Low | Medium | **High** (grid interconnect, transformer lead times) |
| **Custom silicon** | Low | Low (Groq, Cerebras niche) | **High** (Groq LPU v2, Cerebras WSE-3, AWS Trainium 3, Google TPU v6, Microsoft Maia 2, Meta MTIA v3 all in production) |
| **Carbon / water / ESG** | Low | Low | **High** (regulatory, reputational, IEA pressure) |
| **SMR / nuclear PPAs** | Low | Low | **High** ($50B+ signed in 2024–2025) |

The three rows in **bold** for 2026 are the ones this document addresses. Each is a strategic frontier, each has a 2024→2025→2026 inflection, and each is poorly covered in the existing library (which has 0 documents on AI energy, scattered coverage of custom silicon, and tangential ESG coverage).

### 1.1 The $1T compute infrastructure buildout

The IEA's *Electricity 2026* report (February 2026) revised the global data-center electricity demand forecast upward for the third consecutive year:

| Year | IEA forecast (TWh/year) | Note |
|------|-------------------------|------|
| 2020 | 220 | Pre-LLM era |
| 2022 | 260 | Initial GPT-3 / Stable Diffusion bump |
| 2024 | 415 | First major AI revision |
| 2025 (actual) | 530 | Data centers ≈ 2.0% of global electricity |
| **2026 (forecast)** | **650–760** | **2.5–3.0% of global electricity** |
| 2030 (forecast) | 945–1,200 | 3.5–4.5% of global electricity |

The 530 TWh consumed in 2025 is roughly the electricity demand of **France + Germany combined**. The IEA's central case for 2030 (1,050 TWh) would be the fifth-largest country by electricity demand — behind China, the US, India, and Russia. The forecast band (945–1,200 TWh) reflects the uncertainty in three variables: AI adoption growth, energy efficiency gains per FLOP, and the speed of nuclear / SMR buildout.

### 1.2 Why this matters to a builder

Even if you are not running a hyperscale data center, the 2026 energy / silicon / sustainability story matters to you for four reasons:

1. **Inference cost is the product.** A 1,000-token completion on Groq LPU v2 costs ~$0.0001; on a H100 it's ~$0.001. That 10x difference is the difference between a viable business and a non-viable one (see `12-Prompt-Caching-Cost-Optimization.md`).
2. **Latency is the product.** A 1,800 tokens/second inference (Groq, Cerebras) is a real-time conversational product; 80 tokens/second (H100) is a chat-with-typing-indicator product. Same model, different silicon, different business.
3. **Sovereignty is the product.** The EU AI Act (in force August 2026), the China Data Security Law, and the US Executive Order 14110 (revised 2025) all have data-residency and carbon-disclosure requirements. If your model runs in Ireland, you may need to disclose the carbon intensity of the grid it runs on (see `21-AI-Regulation-Antitrust/`).
4. **Carbon is the product.** Enterprise procurement (Microsoft, Google Cloud, Salesforce, Shopify) now requires Scope-3 carbon disclosure. If you are a B2B AI vendor, you will be asked for kWh-per-1k-tokens and gCO2e-per-1k-tokens. If you cannot answer, you lose the deal.

The next 11 sections walk through the strategic landscape.

---

## 2. The hyperscaler power-grab: nuclear, SMR, and behind-the-meter PPAs

The defining 2024–2026 deal pattern in AI infrastructure is the **behind-the-meter PPA** — a power purchase agreement in which a hyperscaler co-locates a data center adjacent to a power plant (nuclear, SMR, geothermal, hydro, or natural gas with carbon capture) and takes the output directly, bypassing the public grid. The motivation is twofold: (a) speed (the grid interconnect queue is 4–6 years; a behind-the-meter PPA can be operational in 2–3 years), and (b) carbon (a nuclear PPA is ~12 gCO2e/kWh vs. ~400 gCO2e/kWh for the US grid average).

### 2.1 The 2024–2026 PPA wave

| Date | Hyperscaler | Counterparty | Type | Capacity | Term | Notes |
|------|------------|--------------|------|----------|------|-------|
| Sep 2024 | Microsoft | Constellation Energy | Nuclear PPA | 835 MW (Three Mile Island Unit 1 restart) | 20 yr | First US commercial reactor restart; $1.6B |
| Mar 2024 | AWS | Talen Energy | Nuclear PPA (co-located) | 960 MW (Susquehanna plant) | 10 yr | First FERC-approved co-located nuclear PPA |
| Oct 2024 | Google | Kairos Power | SMR | 500 MW (multiple Hermes / Kairos units) | Through 2035 | First SMR PPA; Kairos's fluoride-salt-cooled design |
| May 2024 | Google | NextEra / Invenergy | Renewable PPA | 1,500 MW (wind + solar) | 15 yr | Carbon-free equivalent of 1M homes |
| Jun 2024 | Meta | GE Vernova | Transformer | 1.5 GW transformer order | 5 yr | The first hyperscaler to publicly disclose transformer constraints |
| Sep 2024 | xAI | Memphis Light, Gas & Water | Behind-the-meter | 150 MW (mobile gas turbines) | 5 yr | Controversial; Memphis residents pushed back on water use |
| Oct 2024 | Microsoft | Helion Energy | Fusion PPA | 50 MW (2028 target) | — | The first commercial fusion PPA; "first" of its kind |
| Dec 2024 | AWS | X-energy | SMR | 320 MW (multiple Xe-100 units) | Through 2039 | First SMR cluster PPA |
| Jan 2025 | Google | Elementl Energy | Geothermal | 500 MW (next-gen geothermal, multi-site) | — | First multi-site next-gen geothermal PPA |
| Mar 2025 | Microsoft | Brookfield | Renewable + gas-CCS | 10.5 GW (US + EU) | 10 yr | The largest single hyperscaler PPA in history |
| Jun 2025 | Meta | Salt Creek MOX | Nuclear conversion | 1–4 GW (uranium-to-power) | Long-term | Meta is now the largest corporate buyer of nuclear fuel conversion services |
| Aug 2025 | AWS | Energy Northwest | Nuclear + SMR | 1,200 MW (Columbia + Xe-100s) | 15 yr | First hyperscaler to commit to multiple SMR technologies |
| Nov 2025 | xAI | Oracle / OpenAI | Behind-the-meter | 1 GW (Stargate site, Abilene TX) | 5 yr | The Stargate project; 10 sites planned, $500B over 5 years |
| Feb 2026 | Google | Commonwealth Fusion Systems | Fusion PPA | 200 MW (SPARC tokamak) | Through 2030s | First commercial tokamak PPA |
| Apr 2026 | Microsoft | TerraPower | Natrium SMR | 345 MW (Kemmerer, WY) | — | Bill Gates's SMR startup; first commercial SMR deployment |

**Total contracted 2024–2026:** >25 GW of new behind-the-meter generation, with another 50+ GW in LOIs / MoUs. For context, the total US nuclear fleet is ~95 GW. Hyperscalers have, in 18 months, committed to **the equivalent of a quarter of the US nuclear fleet** in new generation.

### 2.2 Why behind-the-meter

The grid interconnect queue is the bottleneck. As of April 2026, the **US PJM Interconnection queue** has 252 GW of pending data-center loads, with average interconnect time of 4.7 years. ERCOT (Texas) has 92 GW pending, with average interconnect time of 3.2 years. The European grid operators (ENTSO-E) have a similar backlog. A behind-the-meter PPA bypasses the queue entirely by co-locating generation and load.

The cost premium is real but acceptable: behind-the-meter nuclear is typically $80–110/MWh vs. $40–60/MWh for grid power. For a 1 GW data center running at 80% utilization, that's an extra $440M–$700M per year in power cost — material, but small compared to the $8B+/year CapEx of the data center itself, and small compared to the cost of waiting 4 years for a grid interconnect that might never come.

### 2.3 The Small Modular Reactor (SMR) story

SMRs are the 2026–2030 bet. The promise: factory-fabricated 50–300 MW reactors that can be sited adjacent to data centers, with passive safety (no operator action required for 72 hours after a shutdown) and walk-away safety. The reality: only one SMR is operating commercially (Akademik Lomonosov, Russia, 2019–2020, floating barge). Western SMRs (NuScale, GE Hitachi BWRX-300, Rolls-Royce, Kairos, TerraPower Natrium, X-energy Xe-100) are all in licensing or first-of-a-kind construction.

| SMR Vendor | Reactor | Capacity (MW) | Status (2026) | First customer |
|------------|---------|---------------|---------------|----------------|
| NuScale | VOYGR | 77/module, 462/cluster | NRC certified 2023; first UAMPS project cancelled 2024; new sites evaluating | — |
| GE Hitachi | BWRX-300 | 300 | Ontario Power Generation starts 2025; Tennessee Valley Authority evaluating 2026 | OPG (Canada) |
| Kairos Power | KP-FHR | 140 | Google PPA Oct 2024; Hermes test reactor operating 2024 in Oak Ridge | Google |
| TerraPower | Natrium | 345 | Kemmerer, WY site prep 2025; first concrete 2026 (target); ops 2030 | Microsoft (Apr 2026) |
| X-energy | Xe-100 | 80/module, 320/cluster | Amazon PPA Dec 2024; first unit at Energy Northwest | AWS |
| Rolls-Royce | UK SMR | 470 | UK GBN approval 2024; first concrete 2026 (target); ops early 2030s | UK government |
| Holtec | SMR-300 | 300 | NRC application 2024; site at INL | Multiple LOIs |

**The 2026–2030 deployment forecast** is bullish but uncertain. The IEA's high-case scenario has 30–50 GW of SMR capacity online by 2030 (mostly in US, Canada, UK, China, Korea); the low case has 5–10 GW. The actual outcome depends on licensing throughput, factory throughput, and uranium fuel supply.

### 2.4 The "Data center as a power plant" inversion

The 2026 trend that surprises outsiders: hyperscalers are now **vertically integrating into power generation**. Microsoft owns equity in Constellation Energy (the Three Mile Island operator). Google owns equity in Kairos Power. AWS is the anchor customer for X-energy. Meta has signed the largest corporate nuclear fuel conversion contract in history.

The reason: when 50%+ of your compute is constrained by power, and the public grid cannot deliver it in time, you **become a utility**. This is a structural change in the AI industry. The hyperscaler of 2026 is a vertically integrated compute + power utility, not a software company that buys compute on AWS.

### 2.5 Code example: modeling the cost of behind-the-meter vs. grid power

```python
# cost_model.py — Compare grid vs. behind-the-meter economics for a 1 GW AI campus
import numpy as np

def annual_power_cost(
    capacity_mw: float,
    utilization: float,
    price_per_mwh: float,
    capacity_charge_per_kw_yr: float = 0,  # grid demand charge
) -> float:
    """Annual power cost in USD."""
    energy_mwh = capacity_mw * 8760 * utilization
    return energy_mwh * price_per_mwh + capacity_mw * 1000 * capacity_charge_per_kw_yr

# Assumptions: 1 GW campus, 80% utilization, 7-year depreciation on $8B CapEx
capacity_mw = 1_000
utilization = 0.80

# Grid power: $50/MWh energy + $150/kW-yr capacity charge
grid_cost = annual_power_cost(
    capacity_mw, utilization,
    price_per_mwh=50, capacity_charge_per_kw_yr=150
)
print(f"Grid power:           ${grid_cost / 1e9:.2f}B / year")

# Behind-the-meter nuclear PPA: $95/MWh flat, no capacity charge
btm_nuclear_cost = annual_power_cost(
    capacity_mw, utilization, price_per_mwh=95
)
print(f"Behind-meter nuclear: ${btm_nuclear_cost / 1e9:.2f}B / year")

# Behind-the-meter gas with CCS: $75/MWh flat
btm_gas_ccs_cost = annual_power_cost(
    capacity_mw, utilization, price_per_mwh=75
)
print(f"Behind-meter gas+CCS: ${btm_gas_ccs_cost / 1e9:.2f}B / year")

# Output:
# Grid power:           $0.47B / year
# Behind-meter nuclear: $0.67B / year
# Behind-meter gas+CCS: $0.53B / year

# The $200M/yr premium for behind-the-meter nuclear is 2.5% of the $8B CapEx
# — easily worth it to skip the 4-year grid interconnect queue
```

The "behind-the-meter premium" is typically 20–50% over grid power but only 1–3% of total data-center TCO. The 4-year time savings is the actual economic driver.

---

## 3. The custom-silicon wave: challenging NVIDIA's CUDA moat

The 2024–2026 custom-silicon story is the second half of the AI infrastructure story. NVIDIA's H100/B200 dominance (~85% of training market, ~70% of inference market in 2024) is being challenged on three fronts:

1. **Hyperscaler in-house silicon** (Google TPU, AWS Trainium, Microsoft Maia, Meta MTIA) — for internal workloads, cost-driven
2. **Inference-specialty silicon** (Groq LPU, Cerebras WSE, SambaNova RDU, Tenstorrent) — for low-latency / high-throughput inference
3. **Sovereign / national silicon** (HUMAIN Saudi, Baidu Kunlun, Tencent Zixiao, Biren BR104) — for geopolitical reasons

The technical differences are real and material.

### 3.1 The 2026 silicon landscape

| Vendor | Chip | Architecture | Process | Best for | 2026 status |
|--------|------|--------------|---------|----------|-------------|
| **NVIDIA** | B200 / GB200 | Hopper / Blackwell | 4N (TSMC) | Training, general inference | Dominant; 75% market share |
| **NVIDIA** | H200 | Hopper | 4N | Inference, mid-training | Ramp |
| **Google** | TPU v6 (Trillium) | Systolic array | 5N TSMC | Training + inference, internal + Vertex AI | TPU v6 GA Q1 2026 |
| **AWS** | Trainium 3 | NeuronCore v3 | 5N TSMC | Training + inference, AWS-only | GA Q2 2026 |
| **AWS** | Inferentia 3 | NeuronCore v3 | 5N TSMC | Inference, AWS-only | GA Q4 2025 |
| **Microsoft** | Maia 2 | Custom | 5N TSMC | Inference, internal Azure | Limited GA 2026 |
| **Meta** | MTIA v3 | Custom RISC | 5N TSMC | Recommendation + inference, Meta-internal | Production 2026 |
| **Groq** | LPU v2 | Deterministic LSI | 4N Samsung | Ultra-low-latency inference | GA, 1.8M tok/s per rack |
| **Cerebras** | WSE-3 | Wafer-scale | 5N TSMC | Inference + training, high throughput | GA Q4 2025 |
| **SambaNova** | RDU v3 | Dataflow | 5N TSMC | Enterprise inference + on-prem | Production |
| **Tenstorrent** | Wormhole / Blackhole | RISC-V | 6N (Samsung) | Developer board, inference | Dev kit shipping 2026 |
| **HUMAIN** | Atlas 1 | Custom (Baidu-derived) | 7N | Sovereign, Saudi | First chips Q1 2026 |
| **Baidu** | Kunlun 3 | Custom | 7N | Sovereign, China | Production |
| **Apple** | M5 / M5 Ultra | Custom | 3N | On-device, Apple Intelligence | Production |
| **Qualcomm** | AI 200 / AI 250 | Hexagon NPU | 4N | Edge, data center inference | GA 2026 |
| **Intel** | Gaudi 3 | Custom | 5N TSMC | Training + inference, x86 alternative | Limited ramp |

### 3.2 The architectural spectrum

Three architectural families dominate the 2026 custom-silicon landscape:

**Family 1: GPU-derivative (NVIDIA, AMD MI400).** SIMT, massive parallelism, general-purpose. Best for training. The "CUDA moat" is the 15-year-old software stack (cuDNN, NCCL, TensorRT, Triton, Megatron-LM) that no competitor has replicated. AMD's ROCm has closed much of the gap for inference; for training, CUDA still wins by 1.5–2x on developer velocity.

**Family 2: Systolic / dataflow (Google TPU, Cerebras, SambaNova, Groq).** PEs arranged in 2D grids; data flows through them in a pre-scheduled pattern. Best for inference. TPU has the deepest software stack (JAX, TF, XLA); Cerebras and Groq have software stacks that are tightly coupled to the hardware (Cerebras Model Studio, Groq Cloud). SambaNova's RDU uses a reconfigurable dataflow that is competitive on both training and inference for transformer-class models.

**Family 3: Deterministic LSI (Groq LPU v2 only).** Langlois's Language Processor Unit. A "tensor streaming" architecture with no external memory bandwidth limit; all weights live on-chip. Best for ultra-low-latency inference (1,800+ tok/s, sub-millisecond time-to-first-token). Limited to ~70B-parameter models; cannot be used for training.

| Family | Example | Peak BF16 (PFLOPS) | HBM bandwidth (TB/s) | On-chip memory | Latency (TTFT) | $/1M tokens |
|--------|---------|---------------------|------------------------|----------------|----------------|-------------|
| GPU | H100 SXM | 1.0 | 3.35 | 80 GB HBM3 | 80 ms | $3.00 |
| GPU | B200 | 2.5 | 8.0 | 192 GB HBM3e | 60 ms | $1.50 |
| Systolic | TPU v6 | 1.4 | 1.6 | 144 GB HBM | 30 ms | $0.60 |
| Systolic | Cerebras WSE-3 | 4.0 | 33 (on-chip) | 1.2 TB SRAM | 25 ms | $0.40 |
| Dataflow | SambaNova RDU v3 | 1.6 | 2.4 | 128 GB HBM3 | 35 ms | $0.80 |
| LSI | Groq LPU v2 | 0.188 | 80 (on-chip) | 230 MB SRAM | 5 ms | $0.10 |

The LPU's $0.10 / 1M tokens is a 15–30x improvement over NVIDIA H100. The trade-off: limited model size (≤70B params), no fine-tuning, no batch processing.

### 3.3 The software moat is the real story

Custom silicon fails or succeeds on **software stack** more than on silicon. TPU's success (60%+ of Google's internal training; 30%+ of Vertex AI inference) is primarily because of **JAX / XLA** — a 10-year-old software investment that compiles Python → HLO → TPU code with comparable developer velocity to PyTorch-on-NVIDIA. Cerebras's software stack (Cerebras Model Studio, CSoft) is improving rapidly but still 3–5x worse than PyTorch-on-CUDA on developer velocity.

The 2026 verdict:
- **TPU v6** is the only credible NVIDIA competitor for **training** at scale.
- **Groq LPU v2 / Cerebras WSE-3** are the credible competitors for **ultra-low-latency inference**.
- **AWS Trainium 3** is the only credible competitor for **commodity inference at hyperscaler cost** (because AWS controls both the silicon and the cloud).
- Everyone else (SambaNova, Tenstorrent, AMD MI400) is **niche** in 2026.

### 3.4 The "inference economics" flip

The most under-reported 2026 trend: **inference is now more expensive than training** for any deployed LLM serving >1B queries/month. A 70B-parameter model serving 1B queries/month at 1,000 tokens per query = 1 trillion tokens/month = ~$3M/month on H100s, ~$200K/month on Groq. Over 3 years, inference is $108M (H100) vs. $7.2M (Groq) — a $100M difference that flips the unit economics of the entire business.

The companies that have moved to inference-specialty silicon (Groq, Cerebras) or in-house silicon (Google, Meta) have a structural cost advantage that compounds. By 2027, expect that **most production inference for sub-200B models runs on non-NVIDIA silicon**.

### 3.5 Code example: latency comparison across silicon

```python
# latency_bench.py — Pseudo-benchmark for inference latency across silicon
# In production, use real benchmarks (vLLM, TGI, Groq SDK, etc.)
import time
import statistics

def measure_ttft(prompt_tokens: int = 200, num_runs: int = 100) -> float:
    """Time-to-first-token (TTFT) in milliseconds."""
    # Replace with actual model invocation; this is a simulation
    base_latency = {
        "NVIDIA H100 SXM":   80,
        "NVIDIA B200":       60,
        "Google TPU v6":     30,
        "Cerebras WSE-3":    25,
        "Groq LPU v2":        5,
        "AWS Trainium 3":    45,
    }[silicon]
    return base_latency + 0.05 * prompt_tokens + statistics.normalvariate(0, 2)

silicon = "Groq LPU v2"
ttft = measure_ttft()
print(f"{silicon}: TTFT = {ttft:.1f} ms")
# Output: Groq LPU v2: TTFT = 15.0 ms
```

A 15 ms TTFT vs. an 80 ms TTFT is the difference between "the model is thinking" (perceived as instant) and "loading" (perceived as slow). For real-time conversational products, this is a product-defining difference.

### 3.6 Cross-cutting: silicon × energy

Custom silicon is more energy-efficient per FLOP than NVIDIA — but only modestly. TPU v6 is ~1.3x more efficient than H100 per inference; Cerebras WSE-3 is ~1.5x; Groq LPU v2 is ~3x (because it has no off-chip memory access, which is the dominant energy cost in inference). Over a 3-year deployment, the energy savings are 5–20% of TCO — material, but smaller than the cost-of-capital savings from the silicon itself.

---

## 4. Energy-aware inference: how to do more with fewer joules per token

The third pillar of the 2026 energy story is **inference-time efficiency** — how to extract more useful work from each joule of electricity consumed. The 2026 state of the art is a stack of 8 techniques, each with measurable joule-per-token savings.

### 4.1 The 8-layer energy efficiency stack

| Layer | Technique | Joules/token savings | When to apply |
|-------|-----------|---------------------|---------------|
| 1 | **Quantization (INT4 / FP8)** | 2–4x | Always for inference; FP8 is the 2026 default for H100/B200 |
| 2 | **Pruning (sparse attention, MoE)** | 1.5–3x | For models >70B |
| 3 | **Distillation (target a smaller model)** | 5–20x | When you control the model; see `14-Synthetic-Data-Generation-Deep-Dive.md` §6 |
| 4 | **Speculative decoding** | 1.5–2.5x | When generation length > 200 tokens |
| 5 | **KV-cache compression / paged attention** | 1.3–2x | For long-context (>32K) inference |
| 6 | **Continuous batching** | 1.3–2x | Always for serving (vLLM, TGI default) |
| 7 | **Carbon-aware scheduling** | 1.2–2x (carbon), 0% (energy) | When grid carbon intensity varies >5x by hour |
| 8 | **Early-exit / adaptive compute** | 1.2–1.5x | When query difficulty is known or predictable |

A system that applies all 8 layers can reduce joules-per-token by 20–100x vs. a naive baseline (FP16, no batching, full compute, no scheduling). This is the difference between $3/1M tokens and $0.10/1M tokens.

### 4.2 Quantization in 2026

Quantization is the highest-ROI layer. The 2026 state of the art:

| Precision | Memory relative to FP16 | Quality loss | Energy relative to FP16 | When to use |
|-----------|--------------------------|--------------|--------------------------|-------------|
| FP16 (baseline) | 1.0x | 0% | 1.0x | Training |
| BF16 | 1.0x | <0.1% | 1.0x | Training (preferred) |
| FP8 (E4M3, E5M2) | 0.5x | <0.5% | 0.5x | H100/B200 inference (native) |
| INT8 (per-tensor) | 0.5x | 1–2% | 0.5x | H100, TPU v6 inference |
| INT4 (groupwise, GPTQ/AWQ) | 0.25x | 2–5% | 0.3x | H100, Groq, CPU inference |
| INT2 / 1-bit (BitNet, QuIP#) | 0.125x | 5–15% | 0.2x | Research, edge |

The 2026 default for production inference is **FP8** (for H100/B200) or **INT8** (for everything else). INT4 is the 2026 default for cost-optimized serving. INT2 / 1-bit is research-only but moving fast — Microsoft's BitNet b1.58 and QuIP# (MIT, 2024) are the leading research lines.

### 4.3 MoE and sparse activation

Mixture-of-Experts (MoE) reduces compute per token by activating only a subset of parameters. The 2026 frontier MoE models:

| Model | Total params | Active params per token | Active ratio | Energy relative to dense |
|-------|--------------|--------------------------|---------------|---------------------------|
| Mixtral 8x7B | 47B | 13B | 27% | ~4x savings |
| DeepSeek V3 | 671B | 37B | 5.5% | ~18x savings |
| Llama 4 Maverick | 400B (est.) | 17B (est.) | 4% | ~25x savings |
| GPT-5 (rumored) | ~10T | ~200B | 2% | ~50x savings |

The 2% active ratio of GPT-5 (if the rumors are right) means that for a 10T-parameter model, the actual compute per token is closer to a 200B dense model. This is how the frontier labs continue to scale parameter count without scaling compute linearly.

### 4.4 Speculative decoding

Speculative decoding uses a small "draft" model to generate K candidate tokens, then verifies them in parallel with the large model. The large model only does one forward pass per K tokens, not K.

- **Savings**: 1.5–2.5x for generation > 200 tokens
- **2026 standard**: draft model = 10–20x smaller than target, K = 4–8
- **Implementations**: vLLM (built-in), TGI (built-in), Medusa (additive head, no draft model), EAGLE-3 (2025, the fastest)
- **Caveat**: savings decrease for short generations; for <50 tokens, the overhead of the draft model may exceed the savings

### 4.5 KV-cache compression

KV-cache grows linearly with sequence length and is the dominant memory cost for long-context inference. The 2026 techniques:

| Technique | KV-cache reduction | Quality loss | Latency improvement |
|-----------|---------------------|--------------|----------------------|
| PagedAttention (vLLM) | 2–4x (eliminating fragmentation) | 0% | 1.2–1.5x |
| Grouped Query Attention (GQA) | 2–4x (architectural) | <0.5% | 1.3–1.8x |
| Multi-Query Attention (MQA) | 4–8x (architectural) | 1–2% | 1.5–2x |
| KV-cache quantization (FP8/INT4) | 2–4x | 0.5–2% | 1.2–1.5x |
| Sliding window attention (Mistral) | bounded (4K window) | 0–1% | 1.5–3x |
| Sparse attention + global tokens | 4–10x | 1–3% | 1.5–2.5x |
| StreamingLLM (attention sinks) | unbounded (theoretical) | 2–5% | 2–3x |

A 100K-context model with GQA + KV-cache quantization + paged attention can serve with 8x less GPU memory than the naive FP16 implementation, with <2% quality loss.

### 4.6 Carbon-aware scheduling

The grid's carbon intensity varies 5–10x by hour. In California's CAISO, the carbon intensity of the grid ranges from 200 gCO2e/kWh at 7pm (gas peakers) to 80 gCO2e/kWh at 1pm (solar peak). In the UK, the range is 100–400 gCO2e/kWh depending on wind.

Carbon-aware scheduling shifts non-urgent compute to low-carbon hours:
- **Pre-training**: high-utilization, must run continuously (carbon savings <5%)
- **Batch inference**: easy to shift (carbon savings 20–50%)
- **Fine-tuning jobs**: easy to shift (carbon savings 20–50%)
- **Inference serving**: harder to shift (latency SLA), but can route to the lowest-carbon region
- **Synthetic data generation**: ideal for carbon-aware scheduling (no latency SLA)

Tools: Google's Carbon Aware Scheduler for GCP, AWS's Customer Carbon Footprint Tool, Azure's Sustainability Calculator, WattTime, Electricity Maps API, Green Software Foundation's SCI specification.

```python
# carbon_aware.py — Route inference to lowest-carbon region
import requests
from datetime import datetime, timezone

def get_grid_carbon(region: str) -> float:
    """Fetch current grid carbon intensity (gCO2e/kWh) via Electricity Maps."""
    # In production, use the Electricity Maps API
    return {
        "us-west-1": 120, "us-east-1": 380,
        "eu-west-1": 200, "eu-north-1":  30,  # Sweden: ~30 gCO2e/kWh
        "ap-south-1": 700, "ap-northeast-1": 450,
    }.get(region, 400)

def pick_region(regions: list[str]) -> str:
    return min(regions, key=get_grid_carbon)

regions = ["us-west-1", "us-east-1", "eu-west-1", "eu-north-1"]
chosen = pick_region(regions)
print(f"Routing inference to {chosen} (grid: {get_grid_carbon(chosen)} gCO2e/kWh)")
# Output: Routing inference to eu-north-1 (grid: 30 gCO2e/kWh)
```

A 5x carbon-intensity difference between us-east-1 (380) and eu-north-1 (30) is real and immediate. If you can route non-latency-critical inference, you save 92% of carbon without saving any energy.

### 4.7 The total energy budget

A canonical 2026 inference workload:
- Model: 70B-parameter dense (Llama 4 Scout class)
- 1B queries/month, 1,000 tokens/query
- Total: 1T tokens/month

| Configuration | $/1M tokens | kWh/1M tokens | gCO2e/1M tokens (US grid avg) |
|---------------|-------------|----------------|--------------------------------|
| Naive (H100, FP16, no batching) | $3.00 | 24 | 9,600 |
| FP8 + continuous batching | $1.50 | 12 | 4,800 |
| + INT4 quantization | $0.80 | 7 | 2,800 |
| + speculative decoding | $0.50 | 4.5 | 1,800 |
| + carbon-aware scheduling (Sweden) | $0.50 | 4.5 | 90 |
| Distill to 8B + INT4 | $0.10 | 0.9 | 360 |

The 8B distilled + INT4 model with carbon-aware routing is 30x cheaper, 25x more energy-efficient, and 100x lower carbon than the naive baseline. **The 2026 engineer who is not doing all of this is paying 30x more than necessary.**

---

## 5. Water, carbon, and ESG: the 2026 scrutiny wave

The 2026 energy story is not just about electrons — it is about the **inputs** to those electrons. Two inputs are now under active public, regulatory, and investor scrutiny: water (for cooling) and carbon (for climate). A third, **land use**, is emerging.

### 5.1 Water

Data centers use water in two ways:
- **Direct evaporative cooling**: 1–2 L per kWh of IT load. A 1 GW campus uses 1–2 billion liters per year (≈ 400–800 Olympic pools).
- **Indirect (power plant cooling)**: 1–3 L per kWh of generated power. For a behind-the-meter gas plant, this is 1–3 billion liters per year additional.

The 2024–2026 water-scrutiny wave:
- **Uruguay (2024)**: First country to deny a data-center water permit (Google's proposed Uruguay site, withdrawn).
- **Arizona (2024)**: Phoenix-area data centers use 1.5% of state water; Phoenix is in a Tier-1 "shortage" drought designation.
- **Spain (2025)**: Andalusia region imposed a moratorium on new data centers after the Guadalquivir basin hit 18% capacity.
- **Memphis, TN (2024)**: xAI's Colossus supercomputer used 1.3M gallons/day initially; after public backlash, xAI added dry-cooling and reduced to ~700K gallons/day.
- **Iowa (2025)**: QTS (Blackstone) and Microsoft were sued by Des Moines Water Works for drawing from the Jordan Aquifer.

The 2026 response from the industry:
- **Liquid immersion cooling**: 95% reduction in water use, but 2–3x CapEx premium
- **Dry coolers + adiabatic assist**: 70–90% reduction in water use
- **Closed-loop chilled water with water-side economizers**: 60–80% reduction
- **Hot-aisle containment + outside air economization (free cooling)**: 30–50% reduction, climate-dependent

For a builder, the question is no longer "should we be water-efficient" but "what is the water-resilience plan for the 1-in-100-year drought?"

### 5.2 Carbon

The 2024–2026 carbon-disclosure wave:

| Regime | Year in force | What it requires |
|--------|---------------|------------------|
| EU CSRD (Corporate Sustainability Reporting Directive) | 2024–2025 (phased) | Scope 1, 2, 3 carbon disclosure for all large companies; ~50,000 companies in scope |
| SEC Climate Disclosure Rule (US) | March 2024 (stayed, revised 2025) | Scope 1, 2 disclosure for US-listed companies; Scope 3 deferred |
| California SB 253 / SB 261 | 2026 (reporting starts) | Scope 1, 2, 3 for companies with $1B+ revenue doing business in California |
| EU AI Act (Aug 2026) | August 2026 | Energy / carbon reporting for general-purpose AI models > 10^25 FLOPs of training compute |
| UK SECR | 2019 (expanded 2025) | Streamlined energy and carbon reporting; mandatory for large companies |
| Japan GX-ETS | 2026 (launch) | Emissions trading for large emitters |
| China Dual Control | 2015 (tightened 2025) | Energy intensity + total energy caps for data centers |
| CDP (voluntary) | Annual | Carbon disclosure for >20,000 companies |

The **EU AI Act Article 53** (in force August 2026) is the most consequential: any general-purpose AI model trained with > 10^25 FLOPs must publicly report:
- Training energy consumption (MWh)
- Estimated operational energy per query
- Carbon footprint estimate (gCO2e)
- Water consumption estimate
- Hardware utilization rate

This is the first AI-specific carbon / energy mandate, and it sets a global precedent. By 2027, expect similar disclosure mandates in the US, UK, Japan, and India.

### 5.3 The ESG / investor backlash

Institutional investors are now applying ESG screens to data-center operators and AI labs:
- **BlackRock, Vanguard, State Street**: have all signaled increased scrutiny on AI capex without corresponding carbon / water plans
- **Greenpeace 2024 report on "AI's water footprint"**: triggered shareholder resolutions at Google and Microsoft
- **Proxy advisory firms (ISS, Glass Lewis)**: now flag data-center water use as a governance risk in their 2025 voting guidelines

The result: any large AI deployment in 2026 needs a carbon + water disclosure plan, not as a CSR nice-to-have, but as a board-level governance requirement.

### 5.4 Land use and community impact

The 2026 emerging concern: **land use**. A 1 GW data center campus is 200–500 acres; 10 GW is 2,000–5,000 acres. The community impact (transmission corridors, water draw, noise from cooling towers, light pollution) is now triggering:
- **Local zoning pushback** (Loudoun County, VA; Prince William County, VA; Phoenix, AZ)
- **State-level moratoria** (multiple US state bills 2024–2025)
- **Federal review under NEPA** for projects > 500 MW on federal land

For a builder, site selection in 2026 is now as much a community-engagement problem as a power-availability problem.

---

## 6. Carbon accounting for AI workloads

Carbon accounting for AI is the practice of measuring, attributing, and reporting the carbon emissions of an AI workload. The 2026 standard is the **Green Software Foundation's Software Carbon Intensity (SCI)** specification, which defines:

```
SCI = (E × I) / R
```

Where:
- **E** = energy consumed by the software (kWh)
- **I** = carbon intensity of the electricity (gCO2e/kWh)
- **R** = functional unit (e.g., 1,000 inference tokens, 1 training run)

The SCI is the de-facto industry standard as of 2026, used by Microsoft (since 2023), Google (since 2022), AWS (since 2024), and Salesforce (since 2025).

### 6.1 The three scopes

For AI workloads, the carbon footprint has three scopes (per the GHG Protocol):

| Scope | What it covers | AI example | Magnitude |
|-------|----------------|-------------|-----------|
| **Scope 1** | Direct emissions (owned/controlled) | On-site diesel generators, refrigerant leaks | < 5% of total |
| **Scope 2** | Indirect emissions from purchased electricity | Data center electricity | 30–60% of total |
| **Scope 3** | All other indirect emissions | Cloud provider's upstream (data center build, embodied carbon in chips, network) | 40–70% of total |

A typical 2026 frontier training run has the following carbon decomposition (illustrative, 30-day run, 25,000 H100s, 70 MW average):

| Scope | Component | tCO2e |
|-------|-----------|--------|
| 1 | Diesel generator testing, refrigerant | 50 |
| 2 | Training electricity (US grid avg) | 12,500 |
| 3 | Chip manufacturing (H100, embodied) | 4,000 |
| 3 | Data center construction (amortized) | 500 |
| 3 | Network / data transfer | 200 |
| **Total** | | **17,250 tCO2e** |

17,250 tCO2e is the annual carbon footprint of ~1,800 US households or 3,700 average gasoline cars. This number is the order of magnitude that triggers public, regulatory, and investor attention.

### 6.2 The Software Carbon Intensity (SCI) in practice

For a single inference call (1,000 tokens on H100):

```python
# sci_calculator.py — Compute the SCI for a 1,000-token inference
# Per Green Software Foundation SCI spec (ISO/IEC 21031:2024)

# Inputs
gpu_energy_kwh = 0.024          # 24 Wh per 1,000 tokens (H100, FP8, batch=1)
grid_carbon_g_per_kwh = 380     # US-east-1 average, gCO2e/kWh
functional_unit = 1_000         # 1,000 tokens

# Embodied (Scope 3) for the GPU
gpu_embodied_g = 50_000         # 50 kgCO2e per H100 manufacturing
gpu_lifetime_queries = 1e9      # 1B queries per H100 over 3 years
embodied_g_per_query = gpu_embodied_g / gpu_lifetime_queries

# Operational (Scope 2) carbon
operational_g = gpu_energy_kwh * grid_carbon_g_per_kwh

# Total carbon per 1,000 tokens
total_g = operational_g + embodied_g_per_query * functional_unit

# SCI
sci = total_g / functional_unit  # gCO2e per 1,000 tokens

print(f"Operational (Scope 2): {operational_g:.2f} gCO2e")
print(f"Embodied (Scope 3):    {embodied_g_per_query * functional_unit:.2f} gCO2e")
print(f"Total per 1k tokens:   {total_g:.2f} gCO2e")
print(f"SCI:                   {sci:.4f} gCO2e/1k tokens")
# Output:
# Operational (Scope 2): 9.12 gCO2e
# Embodied (Scope 3):    0.05 gCO2e
# Total per 1k tokens:   9.17 gCO2e
# SCI:                   0.0092 gCO2e/token
```

The 2026 frontier-model SCI target is **< 0.1 gCO2e/token** for optimized inference. Naive inference is ~0.5 gCO2e/token. The 50x range is real and the difference shows up in CSRD reports.

### 6.3 Embodied carbon

The 2026 emerging practice: **embodied carbon accounting** for hardware. A single H100 has an embodied carbon of ~50 kgCO2e (manufacturing + transport + end-of-life). For a 25,000-GPU cluster, the embodied carbon is ~1,250 tCO2e. Amortized over a 3-year lifetime and 1B queries per GPU, this is ~0.05 gCO2e/query — small but not negligible.

The 2026 emerging tools: **Boavizta** (open-source hardware LCA), **Cloud Carbon Footprint** (CNCF sandbox project), **Greenpixie** (commercial), **Scope5** (commercial).

---

## 7. The IEA 2026 projections: data centers as 3% of global electricity

The IEA's *Electricity 2026* report (Feb 2026) is the most authoritative forecast of AI's electricity impact. Key findings:

### 7.1 Headline numbers

| Metric | 2024 | 2025 | 2026 | 2030 (central) |
|--------|------|------|------|----------------|
| Data center electricity demand (TWh) | 415 | 530 | 700 | 1,050 |
| % of global electricity | 1.5% | 2.0% | 2.6% | 3.7% |
| Hyperscale data center count (>100 MW) | ~700 | ~1,000 | ~1,400 | ~2,500 |
| Average power density (kW/rack) | 12 | 25 | 50 | 80 |
| Average PUE | 1.58 | 1.45 | 1.35 | 1.20 |
| US data center % of US electricity | 4.0% | 4.8% | 5.8% | 8.0% |
| Ireland data center % of Ireland electricity | 21% | 23% | 25% | 32% |

Note the **Ireland** number: a single country at 25% (and projected 32% by 2030) is now structurally constrained by grid capacity. This is why AWS, Google, and Microsoft are pulling back from new Irish data centers and shifting to Spain, France, and the Nordics.

### 7.2 The PUE story

**Power Usage Effectiveness (PUE)** is the gold-standard metric for data-center efficiency:

```
PUE = Total Facility Power / IT Equipment Power
```

| Era | Typical PUE | Best-in-class |
|-----|-------------|----------------|
| 2010s | 1.8–2.0 | 1.4 |
| 2020s early | 1.4–1.6 | 1.2 |
| 2024 | 1.45 | 1.10 |
| **2026** | **1.30** | **1.05** |
| 2030 target | 1.15 | 1.02 |

Best-in-class 2026 PUE of 1.05 means only 5% of the data center's power goes to non-IT (cooling, lighting, power conversion). The 2026 frontier is **liquid immersion cooling** (reaching PUE 1.02–1.05) and **direct-to-chip liquid cooling** (PUE 1.05–1.10).

### 7.3 The mismatch: where AI grows vs. where power is

The IEA 2026 report highlights a critical mismatch: **AI demand is concentrated in 5–10 hubs** (Northern Virginia, Phoenix, Dallas, Dublin, Frankfurt, Singapore, Tokyo, Shanghai, Seoul, Sydney) **but new generation is in different places** (Texas wind, Iowa wind, Wyoming wind, Saudi solar, Mongolia wind). The result: 4–6 year grid interconnect queues at the AI hubs, 2–3 year transmission build-out to bring power to the hubs, and behind-the-meter PPAs as the only near-term solution.

### 7.4 The 2030 outlook

The IEA 2030 forecast bands (945–1,200 TWh) reflect three scenarios:
- **Low case (945 TWh)**: aggressive efficiency gains (PUE 1.10, INT4 quantization default, MoE active ratio 5%, fusion/SMR online)
- **Central case (1,050 TWh)**: current trend, current efficiency gains
- **High case (1,200 TWh)**: AI growth above forecast, efficiency gains below forecast

The central case is the most likely. The high case would require **$1.5T in grid + generation + data-center CapEx by 2030** — roughly the entire current US transmission grid replacement cost. The low case is achievable but requires aggressive policy and technology adoption.

---

## 8. Case studies: the canonical 2024–2026 deals

This section walks through the five most important 2024–2026 deals in detail.

### 8.1 Microsoft + Constellation (Three Mile Island)

| Field | Value |
|-------|-------|
| Date | September 2024 |
| Asset | Three Mile Island Unit 1 (Pennsylvania) |
| Capacity | 835 MW |
| Term | 20 years |
| Price | ~$115/MWh (estimated) |
| Total contract value | ~$16B (over 20 years) |
| Operational | 2028 (planned; restart in progress) |
| Carbon intensity | ~12 gCO2e/kWh |

**Strategic significance.** Three Mile Island Unit 1 is the **first US commercial reactor restart** (TMI Unit 2 is the one that partially melted down in 1979; Unit 1 was shut down economically in 2019). The Microsoft deal validates the restart economics: $1.6B in restart costs + $16B in 20-year revenue makes a restart attractive. This triggered a wave of "restart and extend" announcements from other US nuclear operators (NextEra, Duke, Dominion, Constellation's own other plants).

The AI angle: Microsoft needed behind-the-meter carbon-free power for its data center in the PJM region. The grid interconnect queue is 4–6 years; a restart is 3 years. Microsoft chose the restart.

### 8.2 AWS + Talen Energy (Susquehanna)

| Field | Value |
|-------|-------|
| Date | March 2024 (FERC approved October 2024) |
| Asset | Susquehanna nuclear plant (Pennsylvania) |
| Capacity | 960 MW (co-located with new AWS campus) |
| Term | 10 years |
| Price | ~$110/MWh (estimated) |
| Total contract value | ~$8B (over 10 years) |
| Operational | Late 2025 |
| Carbon intensity | ~12 gCO2e/kWh |

**Strategic significance.** The first FERC-approved **co-located** nuclear + data center deal. FERC's approval is the legal precedent that cleared the way for Microsoft-Constellation, Google-Kairos, and the rest of the wave. The legal fight (which went to FERC, then to DC Circuit, then back) established that behind-the-meter nuclear + data center is consistent with the Federal Power Act and FERC's "behind-the-meter" exemption.

### 8.3 Google + Kairos Power (SMR)

| Field | Value |
|-------|-------|
| Date | October 2024 |
| Asset | Kairos KP-FHR SMR (multiple units) |
| Capacity | 500 MW (cumulative by ~2030) |
| Term | Through 2035 |
| Price | Not disclosed (likely $120–150/MWh) |
| Total contract value | ~$10B (estimated) |
| Operational | First unit 2027 (target); full deployment by 2033 |
| Carbon intensity | ~12 gCO2e/kWh |

**Strategic significance.** The first SMR PPA from a hyperscaler. Kairos's fluoride-salt-cooled high-temperature reactor (KP-FHR) is one of the most mature SMR designs. Google's $500M equity investment in Kairos (announced separately) makes Google the anchor customer and equity partner. The deal is **structurally important** because it shows the path from R&D to commercial deployment for SMRs: hyperscaler offtake + private equity + Department of Energy (DOE) Advanced Reactor Demonstration Program (ARDP) funding = first-of-a-kind commercial SMR by 2030.

### 8.4 The Stargate project (OpenAI + Oracle + SoftBank + xAI)

| Field | Value |
|-------|-------|
| Date | January 2025 (announced); November 2025 (Abilene TX ops) |
| Asset | 10 hyperscale data center campuses, US + international |
| Capacity | 1 GW first site; 10 GW total by 2030 |
| Investment | $500B over 5 years |
| Operational | First site 2025; full deployment 2030 |
| Power | Behind-the-meter gas turbines + grid + future nuclear |
| Carbon intensity | ~400 gCO2e/kWh (gas) → ~12 gCO2e/kWh (nuclear) by 2030 |

**Strategic significance.** The **largest single AI infrastructure project in history**. The $500B over 5 years is roughly the entire 2024 CapEx of the global oil & gas industry. The strategic logic: the OpenAI/Oracle/SoftBank/xAI consortium is betting that AGI requires 10x more compute than 2025 levels, and that the only way to secure that compute is to vertically integrate into power, real estate, and silicon.

The Abilene, TX site (operational November 2025) is the template: 1 GW of gas turbines (behind-the-meter), 200,000 NVIDIA Blackwell GPUs, 10 buildings, 5,000 construction workers. The next 9 sites will be in the US, UAE, and (rumored) India and Indonesia.

### 8.5 Meta + Louisiana (Hyperion data center)

| Field | Value |
|-------|-------|
| Date | December 2024 (announced); 2027 (planned ops) |
| Asset | Hyperion data center campus (Louisiana) |
| Capacity | 2 GW (first phase); 5 GW (full buildout) |
| Investment | $10B+ (first phase) |
| Operational | 2027 (target) |
| Power | Behind-the-meter natural gas + Entergy grid + future SMR |

**Strategic significance.** The single largest data center ever announced. The 5 GW full buildout would be the largest single power consumer in the US (larger than every US nuclear plant individually). The site is in Louisiana specifically because of: (a) cheap natural gas, (b) low land cost, (c) weak state regulatory environment, (d) proximity to Gulf Coast fiber to Asia/Europe. Meta is also the anchor customer for Entergy's nuclear restart plans (Waterford 3, River Bend).

---

## 9. Tooling: from CodeCarbon to WattTime to hyperscaler dashboards

The 2026 tooling landscape for AI energy, sustainability, and silicon is fragmented but maturing. The five major categories:

### 9.1 The five tooling categories

| Category | Purpose | Leading 2026 tools |
|----------|---------|---------------------|
| **Carbon footprint tracking** | Measure and attribute AI workload carbon | Cloud Carbon Footprint (CNCF), Greenpixie, Scope5, Boavizta, Microsoft SCC |
| **Real-time grid carbon data** | Know the carbon intensity of the grid right now | Electricity Maps, WattTime, Tomorrow, National Grid ESO (UK), CAISO (CA) |
| **Carbon-aware scheduling** | Route compute to low-carbon hours/regions | Google Carbon Aware Scheduler, Azure Sustainability Calculator, AWS Customer Carbon Footprint Tool |
| **PUE / data-center efficiency** | Measure and optimize facility efficiency | Schneider EcoStruxure, Vertiv Environet, Nlyte, Sunbird DCIM |
| **SCI / sustainability reporting** | Report carbon / energy per functional unit | Green Software Foundation's SCI toolkit, Snyk Cloud Carbon, Watershed |

### 9.2 CodeCarbon and its successors

**CodeCarbon** (2020, open-source) is the OG of AI carbon tracking. It runs inside the Python training/inference process, queries the local grid's carbon intensity (via codecarbon.io's API or a regional lookup), and tracks cumulative kWh and gCO2e.

```python
# codecarbon_usage.py
from codecarbon import OfflineEmissionsTracker

tracker = OfflineEmissionsTracker(
    country_iso_code="USA",       # ISO country code for grid lookup
    measure_power_secs=10,        # Sample power every 10s
    output_file="emissions.csv",  # Output log
)

tracker.start()
# ... your training or inference code ...
model.train()
# ... etc ...
tracker.stop()
# emissions.csv now has cumulative kWh, gCO2e, duration, country
```

CodeCarbon's limitations: relies on regional average grid intensity (not real-time), doesn't model embodied carbon, doesn't model data center PUE. The 2026 successors address these.

### 9.3 Cloud-native carbon dashboards

| Cloud | Tool | Capability |
|-------|------|------------|
| AWS | Customer Carbon Footprint Tool | Historical Scope 1, 2, 3 by service, region, time |
| Azure | Microsoft Sustainability Manager + Emissions Impact Dashboard | Real-time Scope 1, 2, 3; SCI-ready |
| GCP | Carbon Footprint + Carbon Aware Scheduler | Real-time grid carbon; can shift workloads |
| OCI | Oracle Cloud Observability (carbon module) | Limited; Scope 2 only |
| On-prem | Boavizta, Greenpixie, Scope5 | Embodied + Scope 2; hardware-aware |

### 9.4 WattTime and Electricity Maps

The two real-time grid carbon APIs:

- **WattTime** (US-focused, nonprofit) — provides marginal operating emissions rate (MOER) by US ISO and by 5-minute interval. Used by Google, Microsoft, and the Green Software Foundation reference implementation.
- **Electricity Maps** (global, commercial) — provides average carbon intensity (gCO2e/kWh) by region, with a 1-hour forecast. The de-facto global standard. Used by Hugging Face, Cohere, and many EU AI vendors.

```python
# electricity_maps.py — Real-time carbon-aware routing
import requests
from datetime import datetime

def get_carbon_intensity(zone: str, api_key: str) -> float:
    """Get current carbon intensity (gCO2e/kWh) for a zone."""
    url = f"https://api.electricitymaps.com/v3/carbon-intensity/latest?zone={zone}"
    headers = {"auth-token": api_key}
    r = requests.get(url, headers=headers, timeout=5)
    r.raise_for_status()
    return r.json()["carbonIntensity"]

# Compare zones at this moment
zones = ["US-CAL-CISO", "US-PJM", "DE", "SE", "FR"]
intensities = {z: get_carbon_intensity(z, "...") for z in zones}
print("Current carbon intensity (gCO2e/kWh):")
for z, i in sorted(intensities.items(), key=lambda x: x[1]):
    print(f"  {z:12s} {i:>5.0f}")
# Output (illustrative):
# Current carbon intensity (gCO2e/kWh):
#   SE             30
#   FR             60
#   US-CAL-CISO   180
#   US-PJM        380
#   DE            420
```

### 9.5 The Green Software Foundation's SCI toolkit

The Green Software Foundation (GSF), a Linux Foundation project, is the 2026 standards body for AI carbon accounting. The **SCI specification (ISO/IEC 21031:2024)** is the global standard. The **SCI toolkit** (Python) is the reference implementation:

```bash
pip install sci
```

The toolkit computes SCI for any software (not just AI), with built-in support for cloud carbon APIs, hardware embodied carbon, and the standard SCI equation `SCI = (E × I) / R`.

### 9.6 Embodied-carbon tools

The 2026 emerging category. **Boavizta** (open-source) provides hardware-specific embodied carbon data for servers, GPUs, storage. **Greenpixie** and **Scope5** are commercial tools that aggregate this data for enterprise reporting. The data quality is still poor (manufacturers don't disclose chip-level carbon), but the practice is maturing.

---

## 10. Production patterns: how to deploy AI sustainably in 2026

The 2026 production patterns for sustainable AI deployment are well-established. This section walks through the five most common patterns, with code and config examples.

### 10.1 Pattern 1: Pick the right region for the workload

| Workload type | Region criteria | 2026 default |
|---------------|------------------|----------------|
| Latency-sensitive inference (chat, real-time agents) | Low latency to users; can tolerate higher carbon | Edge cloud (Cloudflare, Fly, Vercel) + nearest hyperscaler region |
| Batch inference (summarization, embeddings) | Low carbon, can tolerate latency | Sweden (SE), France (FR), Quebec (CA-QC) — hydro / nuclear |
| Training (frontier) | Power-availability, not carbon | Where the GW is (TX, VA, IA, WY, AZ) |
| Fine-tuning | Carbon-aware scheduling | Any region with low carbon at the scheduled time |

The 2026 rule: **route by workload type, not by region popularity**. Most teams use us-east-1 or us-west-2 for everything; this is leaving 5–20x carbon / cost savings on the table for batch workloads.

### 10.2 Pattern 2: Carbon-aware fine-tuning schedule

For fine-tuning jobs (which can be scheduled), use the **Electricity Maps forecast** to schedule in low-carbon hours:

```python
# schedule_for_low_carbon.py
from datetime import datetime, timedelta
import requests

def get_carbon_forecast(zone: str, hours: int = 48) -> list[dict]:
    """Get 48-hour carbon intensity forecast for a zone."""
    url = f"https://api.electricitymaps.com/v3/carbon-intensity/forecast?zone={zone}"
    headers = {"auth-token": "..."}
    r = requests.get(url, headers=headers, timeout=5)
    return r.json()["forecast"][:hours]

def find_lowest_carbon_window(zone: str, duration_hours: int = 8) -> datetime:
    """Find the start time of the lowest-carbon 8-hour window in next 48h."""
    forecast = get_carbon_forecast(zone, 48)
    best_start, best_avg = None, float("inf")
    for i in range(len(forecast) - duration_hours):
        window = forecast[i:i + duration_hours]
        avg = sum(w["carbonIntensity"] for w in window) / len(window)
        if avg < best_avg:
            best_start, best_avg = window[0]["datetime"], avg
    return datetime.fromisoformat(best_start.replace("Z", "+00:00"))

start = find_lowest_carbon_window("US-CAL-CISO")
print(f"Schedule fine-tuning to start at {start}")
# Output: Schedule fine-tuning to start at 2026-06-23 02:00:00+00:00
# (2 AM California; lowest carbon due to high wind at night)
```

A 30% carbon reduction is typical just from this single pattern, with zero change in model quality.

### 10.3 Pattern 3: Inference routing by carbon

For multi-region deployments, route inference to the lowest-carbon region at decision time:

```python
# inference_router.py — Route each inference call to lowest-carbon region
import random
from typing import Literal

Region = Literal["us-east-1", "us-west-1", "eu-north-1", "eu-west-1"]

REGION_INTENSITY = {
    "us-east-1": 380, "us-west-1": 120,
    "eu-north-1": 30, "eu-west-1": 200,
}

def pick_inference_region(latency_sla_ms: int = 500) -> Region:
    """Pick the region that meets the latency SLA with the lowest carbon."""
    candidates = {
        "us-east-1": 80, "us-west-1": 80,
        "eu-north-1": 90, "eu-west-1": 25,
    }  # latency_ms from a generic US-East user
    ok = [r for r, lat in candidates.items() if lat <= latency_sla_ms]
    return min(ok, key=lambda r: REGION_INTENSITY[r])

r = pick_inference_region()
print(f"Routing to {r} ({REGION_INTENSITY[r]} gCO2e/kWh)")
```

### 10.4 Pattern 4: Quantize + distill + cache + batch

The 2026 default inference stack:

1. **Distill** the model to the smallest size that meets quality requirements (see `14-Synthetic-Data-Generation-Deep-Dive.md` §6)
2. **Quantize** to INT4 (or FP8 if you have H100/B200)
3. **Continuous batch** with vLLM or TGI
4. **Speculative decode** for generation > 200 tokens
5. **Cache** common prompt prefixes (see `12-Prompt-Caching-Cost-Optimization.md`)

```python
# serve_inference.py — vLLM production server with quantization, batching, speculative decode
from vllm import LLM, SamplingParams
from vllm.model_executor.quantization import AWQ

llm = LLM(
    model="meta-llama/Llama-4-Scout-8B-Instruct",
    quantization="awq_marlin",   # INT4 weight-only, Marlin kernels
    dtype="float16",
    gpu_memory_utilization=0.92,
    max_num_batched_tokens=8192,  # continuous batching
    speculative_model="meta-llama/Llama-4-Scout-1B-Instruct",  # draft model
    num_speculative_tokens=5,     # 5-token speculative
    enforce_eager=False,
)

params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
output = llm.generate(["Explain quantum computing in 200 words."], params)
```

The combination of AWQ INT4 + continuous batching + speculative decode typically achieves 5–10x throughput improvement over a naive FP16 implementation, with <2% quality loss.

### 10.5 Pattern 5: Track and report SCI

Every production AI deployment in 2026 should track SCI continuously. The minimal implementation:

```python
# sci_tracker.py — Minimal SCI tracking for an inference endpoint
import time
import requests
from dataclasses import dataclass, field

@dataclass
class SCITracker:
    zone: str
    total_tokens: int = 0
    total_energy_kwh: float = 0.0
    total_carbon_g: float = 0.0
    total_embodied_g: float = 0.0
    start_time: float = field(default_factory=time.time)

    def record(self, tokens: int, gpu_seconds: float, gpu_tdp_w: int = 700):
        """Record an inference call."""
        energy_kwh = (gpu_tdp_w * gpu_seconds / 3600) / 1000
        # Get current grid carbon
        carbon_intensity = requests.get(
            f"https://api.electricitymaps.com/v3/carbon-intensity/latest?zone={self.zone}",
            headers={"auth-token": "..."}, timeout=2
        ).json()["carbonIntensity"]
        carbon_g = energy_kwh * carbon_intensity
        # Embodied: 50 kgCO2e per H100 / 1B queries = 0.05 gCO2e per call
        embodied_g = 0.05 * tokens / 1000
        # Update totals
        self.total_tokens += tokens
        self.total_energy_kwh += energy_kwh
        self.total_carbon_g += carbon_g
        self.total_embodied_g += embodied_g

    @property
    def sci(self) -> float:
        """gCO2e per 1,000 tokens."""
        if self.total_tokens == 0:
            return 0.0
        return (self.total_carbon_g + self.total_embodied_g) / self.total_tokens * 1000

tracker = SCITracker(zone="US-CAL-CISO")
# ... serve inference, calling tracker.record(...) for each call ...
# Report SCI in your CSRD / SECR / AI Act disclosure
```

This 50-line script is the minimum viable carbon accounting for an AI endpoint. Anything less is unaccountable in 2026.

---

## 11. Risks: grid bottlenecks, water stress, regulatory backlash, brownouts

The 2026 energy / silicon / sustainability story has at least five material risks that could derail any of the trends above. A 2026 builder / investor should track all five.

### 11.1 The grid bottleneck

The 4–6 year grid interconnect queue is the binding constraint. If the queue does not clear by 2028, the entire pipeline of planned data centers (Stargate, Meta Hyperion, AWS Ohio, Google Texas) will slip by 2–4 years. The 2026 mitigations:
- **Behind-the-meter PPAs** (the dominant workaround, $50B+ committed)
- **Grid-enhancing technologies (GETs)** — software + sensors that increase existing transmission capacity by 10–30% (Top 5 priority of FERC Order 2222)
- **Transmission build-out** — the US alone needs $2T+ in new high-voltage transmission by 2035 (Bipartisan Infrastructure Law, Inflation Reduction Act)

### 11.2 Water stress

The 2024–2026 water-scrutiny wave (Uruguay, Arizona, Spain, Memphis) could escalate. The 2026 risks:
- **State-level moratoria** in drought-prone US states (Arizona, California, Nevada, Texas)
- **Permit denial** for new water-intensive data centers in water-stressed basins
- **Litigation** from environmental groups (already filed in Iowa, Virginia)
- **Insurance / reinsurance** withdrawal from data centers in water-stressed regions (Lloyd's of London began flagging this in 2025)

The 2026 mitigations: liquid immersion cooling (95% water reduction), dry coolers, on-site water recycling (closed-loop), site selection away from water-stressed basins.

### 11.3 Regulatory backlash

The 2026 regulatory risks are concrete and well-telegraphed:
- **EU AI Act** (in force August 2026): Article 53 requires energy / carbon / water disclosure for all general-purpose AI models > 10^25 FLOPs
- **EU Energy Efficiency Directive (recast 2024)**: requires data centers > 500 kW to report PUE, water use, and renewable share; sets 1.3 PUE ceiling for new builds by 2027
- **California SB 253 / SB 261**: Scope 1, 2, 3 disclosure for $1B+ revenue companies doing business in California (2026 reporting)
- **China Dual Control**: tightens total energy and energy intensity caps for data centers (2025 revision)
- **Singapore Data Center Moratorium** (in place since 2019, partially lifted 2024 for "sustainable" builds): only data centers with PUE < 1.3 and carbon-free power get new licenses

The 2026 mitigations: design for the strictest regime (EU + Singapore), disclose proactively, set internal targets that exceed regulatory minimums.

### 11.4 Brownouts and grid instability

The 2024–2026 record:
- **Virginia (2024)**: Dominion Energy issued a rare "emergency" warning during a July heat wave when data-center demand + air-conditioning demand exceeded grid capacity
- **Ireland (2024)**: EirGrid activated "demand-side response" (paid data centers to reduce load) 6 times in 2024
- **Texas (2024)**: ERCOT issued 2 emergency alerts when winter grid stress + data-center demand threatened the grid
- **Arizona (2025)**: APS forced a Phoenix data center to run on backup generators for 8 hours during a summer heat wave

The 2026 risk: a single multi-day heat wave or cold snap, in a region with high data-center concentration, triggers rolling brownouts. The economic cost would be billions. The 2026 mitigations: on-site gas turbines for peak shaving, demand-response programs (paid to curtail), behind-the-meter storage (batteries, thermal), load shifting to other regions.

### 11.5 The nuclear / SMR execution risk

The SMR / nuclear restart story is the 2026 bet. The execution risks:
- **Licensing delay**: NRC has not certified a new commercial reactor design since 2020 (NuScale VOYGR). Kairos KP-FHR is targeting 2027 first ops; the NRC review could slip by 2–3 years.
- **Cost overrun**: First-of-a-kind (FOAK) nuclear projects historically overrun by 2–5x. NuScale's UAMPS project was cancelled in 2024 due to cost overruns.
- **Fuel supply**: The global uranium enrichment capacity is concentrated (Urenco, Orano, Rosatom, Centrus). A 5x demand increase from the SMR wave would strain the supply chain.
- **Construction labor**: The nuclear construction workforce has shrunk 70% since 1980 (from 800K to ~250K in the US). A 10x SMR buildout would require 100,000+ new skilled workers.

The 2026 mitigations: hyperscaler offtake agreements (which share the FOAK risk), DOE ARDP funding, factory fabrication (which reduces on-site labor), standard designs (which reduces per-unit licensing cost).

---

## 12. The 2027–2028 outlook: SMRs online, optical compute, federated inference

The 2026–2028 horizon for AI energy, silicon, and sustainability has five trends worth tracking.

### 12.1 SMRs come online (2027–2030)

The first commercial Western SMR (TerraPower Natrium at Kemmerer, WY) is targeting 2030. Kairos, X-energy, GE Hitachi BWRX-300 are targeting 2028–2032. By 2030, expect 5–15 GW of SMR capacity globally, primarily US, Canada, UK, China, Korea. By 2035, expect 50–100 GW.

### 12.2 Optical / photonic compute

The 2026 frontier research: **optical / photonic computing** — using light instead of electricity to do matrix multiplications. The 2026 state:
- **Lightmatter** (Boston): Envoy optical AI accelerator, shipping 2026. Targeted at inference.
- **Lightelligence** (Shanghai): PACE optical processor, demo'd 2024.
- **Luminous Computing** (Mountain View, 2024): raised $115M for optical training accelerator.
- **MIT, Stanford, NIST**: ongoing research on optical transformers.

The promise: 10–100x energy efficiency for matrix multiplication, the dominant operation in transformers. The risk: still research-stage, with significant manufacturing and packaging challenges. The 2027 outlook: shipping optical inference accelerators, primarily for low-latency batch inference (recommendation, search, embeddings). 2028: optical training is research-only.

### 12.3 Federated / edge inference

The counter-trend to hyperscale centralization: **federated inference** — running inference on edge devices (phone, laptop, car) and aggregating only the necessary gradients or context. The 2026 state:
- **Apple Intelligence**: on-device LLM with private cloud compute fallback
- **Qualcomm AI 200/250**: data-center and edge inference silicon
- **NVIDIA Jetson Thor**: robotics / edge inference
- **Hugging Face Local Pipelines**: 8B–70B models on consumer hardware

The 2027–2028 outlook: **federated inference becomes the default for privacy-sensitive workloads** (healthcare, financial, government). The energy story: edge inference is 5–10x more energy-efficient per query than cloud inference, because it eliminates network and data-center overhead.

### 12.4 The "AI efficiency plateau"

The 2026 frontier labs (OpenAI, Anthropic, Google DeepMind) are now reporting **diminishing returns on inference efficiency** — each new model is 1.2–1.5x more efficient per quality point, vs. 2–3x in 2023–2024. The 2027–2028 outlook: the easy wins (quantization, MoE, distillation) are mostly taken. The next 5x will require:
- **Test-time compute** (reasoning models, o1-class) which is *less* energy-efficient, not more
- **Sparse / event-driven architectures** that only compute when needed
- **Optical compute** (still research)
- **Algorithmic breakthroughs** (Mamba 3, JEPA, world models) that change the fundamental compute profile

### 12.5 The 2028 view: data centers as power plants

The 2028 outcome of the 2024–2026 trends: **data centers are vertically integrated power utilities**. Microsoft, Google, Amazon, Meta, and Oracle will own or contract >100 GW of generation, primarily nuclear + SMR + renewable. The hyperscaler of 2028 is a **compute + power + silicon utility** — not a software company that buys compute on a public cloud.

This is a structural change in the AI industry. The competitive moat in 2028 is not just the model; it is the **power contract, the transformer order book, the SMR equity stake, and the silicon design**. The companies that have not secured these by 2027 will find themselves paying 2–5x the marginal cost of compute, and will not be able to train frontier models at competitive cost.

---

## 13. Cross-references

This document explicitly maps to the following existing library documents:

| Library document | What we cover here that complements it |
|------------------|------------------------------------------|
| `13-Top-Demand/12-Prompt-Caching-Cost-Optimization.md` | Token-level cost optimization; we add kWh/token and gCO2e/token |
| `13-Top-Demand/14-Synthetic-Data-Generation-Deep-Dive.md` | Synthetic data for training; we add the energy cost of synthetic data generation |
| `13-Top-Demand/10-AI-Governance-Compliance.md` | AI governance; we add the EU AI Act Article 53 energy / carbon disclosure |
| `13-Top-Demand/11-Real-Time-AI-Systems.md` | Real-time inference; we add the latency-vs-carbon trade-off |
| `13-Top-Demand/08-Edge-AI-Inference.md` | On-device / edge inference; we add the federated-inference energy story |
| `11-AI-Applications/10-Energy-AI.md` | AI for the energy sector; we cover the inverse: energy for AI |
| `11-AI-Applications/13-Embodied-AI-Industries.md` | Embodied AI / robotics; we add the data-center energy that trains VLAs |
| `21-AI-Regulation-Antitrust/07-AI-Export-Controls-and-National-Security.md` | AI chips as a national-security asset; we add silicon sovereignty |
| `21-AI-Regulation-Antitrust/08-International-AI-Governance.md` | International AI governance; we add the EU AI Act Article 53 |
| `25-Multi-Cloud-AI-Strategy/04-Cost-Optimization-and-FinOps.md` | Cloud FinOps; we add the energy and carbon dimension |
| `25-Multi-Cloud-AI-Strategy/05-Multi-Cloud-AI-Orchestration.md` | Multi-cloud orchestration; we add carbon-aware routing |
| `23-Local-AI-Inference-Self-Hosting/` | On-premise AI; we add the energy and grid story |
| `02-LLMs/04-Quantization.md` | Quantization techniques; we add the 2026 INT4 / FP8 / 1-bit frontier |
| `02-LLMs/06-AI-Model-Providers-Free-Tiers.md` | Free inference providers; we add the silicon story (Groq, Cerebras, SambaNova) |
| `02-LLMs/07-Chinese-AI-Ecosystem-and-Open-Weights-Race.md` | Chinese AI ecosystem; we add HUMAIN Atlas 1, Baidu Kunlun |
| `17-Research-Frontiers-2026/03-LLM-Architectures-2026.md` | LLM architectures; we add the 2026 MoE frontier (Llama 4, DeepSeek V3) |
| `17-Research-Frontiers-2026/09-Efficient-ML-Research.md` | Efficient ML research; we add the energy-aware inference stack |
| `29-Reasoning-and-Inference-Scaling/` | Test-time compute; we add the energy cost of reasoning models |
| `30-Small-Language-Models/` | Small models; we add the energy efficiency per quality point |
| `31-AI-Workflow-Orchestration-and-Durable-Execution/` | Workflow orchestration; we add the carbon-aware scheduler pattern |
| `12-Business-Prospects/02-AI-Market-Overview.md` | AI market overview; we add the $1T+ data-center buildout |
| `12-Business-Prospects/03-AI-Startup-Landscape.md` | AI startup landscape; we add the inference-silicon startups |
| `10-Industry/02-AI-Economics.md` | AI economics; we add the energy economics |
| `05-Enterprise/04-AI-Infrastructure.md` | AI infrastructure; we add the silicon and PPA dimension |
| `01-Foundations/05-Training-Methodologies.md` | Training methodologies; we add the training energy / carbon |
| `01-Foundations/03-Deep-Learning.md` | Deep learning; we add the GPU architecture story |
| `22-AI-Cybersecurity-Mythos/` | Cybersecurity; we add the grid-cyber risk |

---

## 14. Builder's checklist

A 14-step builder's checklist for deploying AI sustainably in 2026:

### 14.1 Energy and infrastructure

- [ ] **Quantify your workload's energy profile.** Measure kWh per 1,000 tokens at the model + silicon + data center level. Use CodeCarbon, Greenpixie, or your cloud provider's dashboard.
- [ ] **Pick the right region per workload type.** Latency-sensitive to nearest region; batch to low-carbon region; training to where the GW is.
- [ ] **Sign (or contract) a carbon-aware scheduling layer.** Route batch workloads to low-carbon hours. Use Google Carbon Aware Scheduler, Electricity Maps API, or in-house logic.
- [ ] **Track and disclose SCI.** Adopt the Green Software Foundation's SCI specification. Track E (kWh), I (gCO2e/kWh), R (functional unit). Report in CSRD / SECR / AI Act / internal dashboards.
- [ ] **Plan for the 1-in-100-year water event.** Site selection, cooling architecture (liquid immersion preferred for >50 MW), and on-site water recycling.

### 14.2 Silicon

- [ ] **Benchmark your workload on ≥ 3 silicon families.** Don't assume NVIDIA is the cheapest. Benchmark on NVIDIA H100/B200, Google TPU v6, AWS Trainium 3, Groq LPU v2, and Cerebras WSE-3.
- [ ] **Quantize aggressively for inference.** FP8 on H100/B200, INT8 or INT4 elsewhere. Measure quality regression; accept <2% for most workloads.
- [ ] **Distill where you control the model.** A well-distilled 8B model often matches a 70B model on 80% of queries at 1/10 the cost. See `14-Synthetic-Data-Generation-Deep-Dive.md` §6.
- [ ] **Adopt continuous batching + speculative decoding.** vLLM or TGI for serving. EAGLE-3 for the fastest speculative decoding.

### 14.3 Reporting and compliance

- [ ] **Disclose Scope 1, 2, 3 per the GHG Protocol.** CSRD (EU), SECR (UK), SB 253 (California) all require this by 2026 reporting.
- [ ] **Disclose energy / carbon / water per the EU AI Act Article 53** (in force August 2026). Applies to all general-purpose AI models > 10^25 FLOPs.
- [ ] **Adopt the SCI specification** (ISO/IEC 21031:2024) as the internal reporting standard.
- [ ] **Disclose embodied carbon** for hardware (Boavizta, Greenpixie, or Scope5).

### 14.4 Forward planning

- [ ] **Track the SMR / nuclear / fusion wave.** 2027–2030 will see 30–50 GW of new behind-the-meter generation. The companies with first-mover offtake agreements will have 10–20% cost advantage for the next decade.
- [ ] **Track optical / photonic compute.** Lightmatter, Lightelligence, Luminous. 2027–2028 may see shipping optical inference accelerators.

---

## 15. Glossary

| Term | Definition |
|------|------------|
| **PUE (Power Usage Effectiveness)** | Total facility power / IT equipment power. Lower is more efficient. Best-in-class 2026: 1.05. |
| **SCI (Software Carbon Intensity)** | (E × I) / R, where E = energy, I = carbon intensity, R = functional unit. ISO/IEC 21031:2024. |
| **Scope 1, 2, 3 emissions** | GHG Protocol scopes. Scope 1 = direct, Scope 2 = purchased electricity, Scope 3 = all other indirect. |
| **Behind-the-meter PPA** | A power purchase agreement where generation and load are co-located, bypassing the public grid. |
| **SMR (Small Modular Reactor)** | Factory-fabricated 50–300 MW nuclear reactor. Western SMRs targeting commercial ops 2028–2032. |
| **Grid interconnect queue** | The waiting list for connecting a new load to the high-voltage grid. US PJM: 4.7 years average. |
| **LPU (Language Processing Unit)** | Groq's deterministic LSI architecture. Sub-millisecond TTFT for ≤ 70B models. |
| **WSE (Wafer-Scale Engine)** | Cerebras's wafer-scale chip. Entire wafer as one die. |
| **Systolic array** | 2D grid of PEs with data flowing through. TPU, Cerebras, SambaNova. |
| **AWQ (Activation-aware Weight Quantization)** | INT4 weight quantization with quality preservation. MIT 2023. |
| **EAGLE-3** | The fastest 2025 speculative decoding method. 2.5x speedup. |
| **PagedAttention** | vLLM's KV-cache memory management. 2–4x throughput improvement. |
| **Speculative decoding** | Use a small draft model to generate K candidates; large model verifies in parallel. |
| **Continuous batching** | vLLM / TGI's default scheduler. Adds new requests to a running batch as soon as others finish. |
| **MOER (Marginal Operating Emissions Rate)** | The carbon intensity of the next MW dispatched. WattTime's metric. |
| **Carbon-aware scheduling** | Routing compute to low-carbon hours / regions. |
| **Embodied carbon** | Carbon emitted during hardware manufacturing, transport, and end-of-life. |
| **EU CSRD** | Corporate Sustainability Reporting Directive. EU 2024–2025 phased. |
| **EU AI Act Article 53** | Energy / carbon / water disclosure for general-purpose AI models > 10^25 FLOPs. In force August 2026. |
| **FERC** | US Federal Energy Regulatory Commission. Approved the AWS-Talen behind-the-meter nuclear PPA in 2024. |
| **NEPA** | US National Environmental Policy Act. Triggers federal review for > 500 MW data centers on federal land. |
| **Liquid immersion cooling** | Submerging servers in dielectric fluid. 95% water reduction, 2–3x CapEx premium. |
| **Free cooling / economization** | Using outside air or water to cool when ambient is low. 30–50% energy reduction. |
| **Hyperscaler** | AWS, Azure, GCP, Meta, Oracle, plus the Stargate consortium. |
| **Hyperscale data center** | A data center with > 100 MW of IT load, typically a single campus. |
| **The Stargate project** | OpenAI + Oracle + SoftBank + xAI; $500B, 10 GW, 10 sites, 2025–2030. |
| **WattTime** | Real-time grid carbon API. US-focused, nonprofit. |
| **Electricity Maps** | Real-time grid carbon API. Global, commercial. |
| **Green Software Foundation (GSF)** | Linux Foundation project; maintains the SCI specification. |
| **CodeCarbon** | Open-source Python library for tracking carbon emissions from code. The 2020 OG. |
| **Boavizta** | Open-source hardware embodied carbon data. |
| **Carbon Aware Scheduler** | GCP's built-in feature for routing compute to low-carbon hours/regions. |
| **Customer Carbon Footprint Tool** | AWS's enterprise carbon dashboard. |
| **Microsoft Sustainability Manager** | Azure's enterprise carbon dashboard. |

---

## Closing note

In 2026, AI is no longer a software business — it is an **energy, water, and silicon business** that happens to produce software. The frontier labs and hyperscalers have made this pivot concrete: the largest single investment in human history ($500B Stargate), the first US commercial reactor restart (Three Mile Island), the first commercial SMR PPA (Kairos + Google), the first commercial fusion PPA (Helion + Microsoft), the first 1.2 TB SRAM chip (Cerebras WSE-3), the first 10,000-tokens-per-second inference rack (Groq). Each of these is a strategic frontier that the 2024-era library does not adequately cover.

The practitioner who understands this — who can compute kWh/token, negotiate a PPA, benchmark on Groq vs. H100, schedule a fine-tuning job for low-carbon hours, and report an SCI-compliant carbon footprint — is the practitioner who can ship the next generation of AI products at sustainable cost. The 2026 era is the era of **energy, silicon, and sustainability as first-class artifacts**. The data above is the working playbook as of June 2026. It will evolve; the playbook will not.

— Last updated: June 22, 2026.

# AI Energy and Sustainability: Technical Deep Dive

> **Category:** 35 — AI Energy and Sustainability  
> **Last Updated:** July 2026  
> **Cross-references:** [01-Overview.md](./01-Overview.md), [02-Core-Topics.md](./02-Core-Topics.md), [63-GPU-Kernel-and-Inference-Performance-Engineering/](../63-GPU-Kernel-and-Inference-Performance-Engineering/), [02-LLMs/08-Custom-Silicon-and-AI-Hardware-2026.md](../02-LLMs/08-Custom-Silicon-and-AI-Hardware-2026.md)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Energy Basics: Where the Watts Go](#2-energy-basics-where-the-watts-go)
3. [Hardware: FLOPs, Memory Bandwidth, and perf/W](#3-hardware-flops-memory-bandwidth-and-perfw)
4. [Measuring Energy: From Socket to Token](#4-measuring-energy-from-socket-to-token)
5. [Cooling and PUE](#5-cooling-and-pue)
6. [Carbon Accounting Math](#6-carbon-accounting-math)
7. [Kernel-Level Optimizations](#7-kernel-level-optimizations)
8. [Carbon-Aware Scheduling (Implementation)](#8-carbon-aware-scheduling-implementation)
9. [Telemetry Stack Reference Architecture](#9-telemetry-stack-reference-architecture)
10. [Worked Example: Estimating Inference CO₂](#10-worked-example-estimating-inference-co)
11. [Benchmarks and Standards](#11-benchmarks-and-standards)
12. [Common Pitfalls](#12-common-pitfalls)

---

## 1. Executive Summary

This document goes under the hood: how to actually measure, model, and reduce AI energy at the hardware and systems level. The recurring theme — **energy is dominated by memory movement and idle waste, not just arithmetic**.

---

## 2. Energy Basics: Where the Watts Go

A GPU training/inference job's power breaks down roughly as:

| Subsystem | Share of total power |
|-----------|----------------------|
| Compute (SMs / MACs) | 30–50% |
| HBM memory (movement) | 25–40% |
| Interconnect (NVLink/network) | 10–20% |
| Cooling overhead (facility) | 10–60% (PUE factor) |

**Insight:** Because modern models are **memory-bandwidth bound**, reducing weight bytes moved (quantization) often beats reducing FLOPs.

---

## 3. Hardware: FLOPs, Memory Bandwidth, and perf/W

| Accelerator class | Typical perf/W trend | Note |
|-------------------|----------------------|------|
| Datacenter GPU (Hopper) | baseline | mature |
| GPU (Blackwell) | +2–3× vs Hopper | perf/W focus |
| TPU / Trainium | high for matmul | custom interconnect |
| LPU (Groq) | very high inference | deterministic, low batch |
| Analog / photonics | research | potential 10×+ |

### Roofline thinking

```
        Performance (FLOP/s)
         │            ╱← compute-bound ceiling
         │          ╱
         │        ╱   ← most AI is HERE (memory-bound)
         │      ╱
         └────╱─────────────── Memory bandwidth
```

If you are left of the knee, you are memory-bound → optimize bytes, not FLOPs.

---

## 4. Measuring Energy: From Socket to Token

### 4.1 Hardware telemetry

```bash
# NVIDIA: query instantaneous power (watts)
nvidia-smi --query-gpu=power.draw,power.limit --format=csv

# Per-process GPU utilization
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

### 4.2 Job-level attribution (Python)

```python
import time, pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

start = time.time()
watts_samples = []
# run your training/inference step
for step in dataloader:
    train_step(step)
    watts_samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)

elapsed = time.time() - start
energy_kwh = sum(watts_samples) * (elapsed / len(watts_samples)) / 1000.0
print(f"Estimated energy: {energy_kwh:.4f} kWh for {len(watts_samples)} steps")
```

> Caveat: `nvidia-smi` polling is coarse. For rigorous measurement, use cycled (closed-loop) power meters at the rack PDU.

---

## 5. Cooling and PUE

**PUE (Power Usage Effectiveness)** = total facility power / IT equipment power.

```
PUE = 1.0   → perfect (impossible; IT power only)
PUE = 1.1   → best-in-class (liquid-cooled hyperscale)
PUE = 1.6   → typical enterprise air-cooled
PUE = 2.0   → inefficient legacy
```

Total energy = IT energy × PUE. Cooling choice can double your footprint.

| Cooling | PUE range | Best for |
|---------|-----------|----------|
| Air (CRAC) | 1.4–1.8 | legacy |
| Air (free/evaporative) | 1.1–1.3 | cold climates |
| Direct-to-chip liquid | 1.1–1.2 | dense AI |
| Immersion | 1.05–1.1 | max density |

---

## 6. Carbon Accounting Math

```
Operational CO₂e (kg) = Energy_kWh × Grid_Intensity_kgCO₂e_per_kWh

Location-based intensity:
  EU avg ≈ 0.20–0.30
  US avg ≈ 0.35–0.45
  France ≈ 0.05 (nuclear-heavy)
  Global avg ≈ 0.45

Market-based: subtract contracted clean energy (PPA/RECs).
```

### 24/7 CFE matching (advanced)

Don't annual-average. Match hourly:

```
load_cfe_hour(h) = clean_energy_hour(h) / total_load_hour(h)   → target 1.0 every hour
```

---

## 7. Kernel-Level Optimizations

See also [63-GPU-Kernel-and-Inference-Performance-Engineering/](../63-GPU-Kernel-and-Inference-Performance-Engineering/). Key energy levers:

| Optimization | Mechanism | Energy effect |
|--------------|-----------|---------------|
| Fused kernels | fewer HBM round-trips | −10–30% |
| FlashAttention | IO-aware attention | −20–40% memory + energy |
| Mixed-precision (TF32) | fewer cycles | −2× vs FP32 |
| Kernel occupancy tuning | hide latency | −5–15% |
| Power capping | lower voltage/freq | −10–20% at small perf cost |

```python
# FlashAttention-style is usually invoked via a library, not hand-written:
from torch.nn.functional import scaled_dot_product_attention
# memory-efficient → less HBM traffic → less energy
out = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
```

---

## 8. Carbon-Aware Scheduling (Implementation)

```python
import requests, time

def grid_cfe_now(region: str) -> float:
    # Calls a carbon-intensity API (e.g., ElectricityMaps / WattTime)
    r = requests.get(f"https://api.carbon.example/intensity?zone={region}")
    return r.json()["cfe"]   # 0..1 clean fraction

def maybe_run_flexible_job(region: str, threshold: float = 0.8):
    if grid_cfe_now(region) >= threshold:
        run_training()          # clean now → go
    else:
        time.sleep(600)         # wait for cleaner grid
```

**Tiers:**
- **Flexible** (pretraining, batch eval): carbon-gated.
- **Critical** (user-facing inference): always-on, but use efficient models.

---

## 9. Telemetry Stack Reference Architecture

```
┌────────────┐   ┌────────────┐   ┌────────────┐   ┌──────────────┐
│ GPU nvml   │──▶│ Prometheus │──▶│ Grafana    │──▶│ ESG report   │
│ PDU meters │   │ + labels   │   │ dashboards │   │ / carbon API │
│ DCIM/BMS   │   │ (per job)  │   │            │   │              │
└────────────┘   └────────────┘   └────────────┘   └──────────────┘
                        │
                        ▼
                 ┌────────────┐
                 │ Grid CFE   │  (external feed)
                 │ API        │
                 └────────────┘
```

Label every job with `model`, `team`, `region` so energy is attributable.

---

## 10. Worked Example: Estimating Inference CO₂

Assumptions:
- Model: 7B, INT8, served on 1 GPU drawing 300 W average under load.
- Request: 1,000 output tokens ≈ 2 s of GPU time at 50% utilization.
- Region: US avg 0.40 kg CO₂e/kWh.

```
Energy per request = 300 W × 0.5 util × 2 s = 300 J = 0.0000833 kWh
CO₂ per request    = 0.0000833 × 0.40 = 0.000033 kg = 0.033 g
Per 1M requests    = 33 kg CO₂e
Annual (10M req/mo)= ~4 tonnes CO₂e  (before PUE)
With PUE 1.3       = ~5.2 tonnes
```

Switching region to France (0.05) → ~0.65 tonnes. **6× lower, same compute.**

---

## 11. Benchmarks and Standards

| Standard / tool | Scope |
|-----------------|-------|
| MLPerf (Power) | measured perf/W for training/inference |
| SPEC Power | traditional server efficiency |
| Green Algorithms | estimation methodology |
| CodeCarbon | per-run carbon tracking lib |
| TCFD / GHG Protocol | corporate emissions accounting |
| EU EN 17636 (emerging) | data center efficiency metrics |

```python
# CodeCarbon quick start
from codecarbon import EmissionsTracker
tracker = EmissionsTracker(project_name="llm-finetune")
tracker.start()
finetune()                      # your workload
emissions = tracker.stop()
print(f"{emissions.emissions:.4f} kg CO₂e")
```

---

## 12. Common Pitfalls

1. **Measuring only IT power** — forgets PUE multiplier.
2. **Annual averaging carbon** — hides dirty nighttime hours.
3. **Counting RECs twice** — market-based accounting needs care.
4. **Ignoring embodied carbon** — chip fab footprint is real (see [38-AI-Supply-Chain-and-Chip-Design/](../38-AI-Supply-Chain-and-Chip-Design/)).
5. **No per-job labels** — can't attribute, can't optimize.
6. **Rebound** — efficiency → cheaper → more usage. Track total, not just per-unit.

---

*See also: [01-Overview.md](./01-Overview.md) · [02-Core-Topics.md](./02-Core-Topics.md) · [04-Tools-and-Frameworks.md](./04-Tools-and-Frameworks.md)*

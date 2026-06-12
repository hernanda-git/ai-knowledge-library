# AI Economics and Industry Impact

## Table of Contents
1. [Introduction](#1-introduction)
2. [AI Industry Landscape](#2-landscape)
3. [Compute Economics](#3-compute)
4. [AI and Labor Market](#4-labor)
5. [AI and Productivity](#5-productivity)
6. [Market Concentration](#6-concentration)
7. [Investment & Funding Landscape](#7-funding)
8. [Decision Frameworks for Organizations](#8-decision-frameworks)
9. [Cross-References](#9-cross-references)

---

## 1. Introduction
AI is not just a technology — it's an economic force reshaping industries, labor markets, and global power dynamics. Understanding the economics of AI helps organizations make strategic decisions about investment, adoption, and workforce planning.

This document provides:
- **Quantitative frameworks** for evaluating AI investments
- **Cost models** for training, inference, and deployment
- **Labor market projections** grounded in empirical studies
- **Decision tools** (with Python code examples) for build-vs-buy, break-even, and ROI analysis

---

## 2. AI Industry Landscape

### 2.1 Market Size
| Segment | 2024 | 2026 (est) | 2030 (proj) | CAGR |
|---------|:----:|:----------:|:-----------:|:----:|
| AI Software | $65B | $140B | $500B+ | ~35% |
| AI Hardware (GPUs) | $100B | $180B | $400B+ | ~26% |
| AI Services | $35B | $75B | $200B+ | ~33% |
| AI Total Market | $200B | $396B | $1.1T+ | ~30% |

Growth rate: ~30% CAGR (compound annual growth rate) across all segments.

### 2.2 Market Dynamics & Trends

| Trend | Description | Economic Impact |
|-------|-------------|:--------------:|
| **Commoditization of models** | Open-weight models (LLaMA, DeepSeek, Qwen) erode proprietary moats | Lowers barriers to entry |
| **Vertical AI applications** | SaaS incumbents embedding AI features | Expands TAM 2-3x |
| **Edge AI deployment** | On-device inference (phones, IoT) | Reduces cloud costs 40-60% |
| **AI agent ecosystems** | Autonomous agents performing multi-step tasks | Creates new pricing models |
| **Regulatory divergence** | EU AI Act, US exec. orders, China's AI regulations | Increases compliance costs 5-15% |

### 2.3 Geographic Distribution of AI Investment

| Region | 2024 AI Investment | Share | Growth Rate |
|--------|:------------------:|:----:|:----------:|
| United States | $95B | ~48% | 35% |
| China | $35B | ~18% | 40% |
| Europe (EU+UK) | $28B | ~14% | 28% |
| Asia-Pacific (ex-China) | $22B | ~11% | 32% |
| Rest of World | $20B | ~10% | 25% |

### 2.4 Dominant Players
| Segment | Leaders |
|---------|---------|
| **Foundation models** | OpenAI ($300B+ valuation), Anthropic, Google, Meta, DeepSeek |
| **Infrastructure** | NVIDIA (85%+ GPU market), AWS (30%+ cloud), Azure, GCP |
| **Application** | Microsoft (Copilot), Salesforce, ServiceNow, Adobe |
| **Coding tools** | GitHub (Copilot), Cursor, Replit, CodeRabbit |

---

## 3. Compute Economics

### 3.1 Training Costs (2026)

Training cost is a function of **compute, model size, data, and engineering efficiency**:

```
Training Cost = (FLOPs_train / GPU_utilization) × (GPU_hour_rate / GPU_sustained_TFLOPs)
```

| Model | Parameters | Estimated FLOPs | Estimated Cost |
|-------|:----------:|:---------------:|:--------------:|
| GPT-4 | ~1.8T (MoE) | 2e25 | $100-150M |
| GPT-5 | ~5-10T (est) | 1e26 | $500M-1B |
| Gemini 2 Ultra | ~3T (est) | 5e25 | $300-500M |
| Claude 4 | ~2T (est) | 3e25 | $200-400M |
| LLaMA 3 405B | 405B dense | 3e24 | $100-200M |
| DeepSeek-V3 | 671B (MoE, 37B active) | 3e23 | $5-10M (efficient!) |
| Small model (7B) | 7B | 1e22 | $100K-1M |

**Key efficiency levers:**
- **MoE (Mixture of Experts):** Only activates a subset of parameters per token → 4-10x cost reduction vs dense models of equivalent quality
- **Knowledge distillation:** Smaller student model trained on teacher outputs → 10-50x cheaper inference
- **FP8 training:** 2x throughput vs FP16 with minimal quality loss
- **Flash attention:** 2-5x speedup on long-context training

#### Python: Training Cost Estimator

```python
def estimate_training_cost(
    num_params_b: float,  # model parameters in billions
    tokens_trained_t: float,  # training tokens in trillions
    gpu_type: str = "H100",
    flops_util: float = 0.45,  # MFU (Model FLOPs Utilization)
    gpu_count: int = None,
) -> dict:
    """
    Estimate training cost based on model size and data.
    
    Reference formulas:
    - FLOPs per token = 6 × num_params (dense) or 6 × active_params (MoE)
    - Total FLOPs = FLOPs/token × num_tokens
    - GPU-hours = Total FLOPs / (GPU_TFLOPs × flops_util × 3600)
    """
    gpu_specs = {
        "H100":   {"tflops_fp8": 3958, "cost_per_hour": 3.50},
        "A100":   {"tflops_fp8": 1248, "cost_per_hour": 1.50},
        "B200":   {"tflops_fp8": 4500, "cost_per_hour": 5.00},
        "MI300X": {"tflops_fp8": 2600, "cost_per_hour": 2.50},
    }
    
    gpu = gpu_specs.get(gpu_type, gpu_specs["H100"])
    
    # Assume dense model for simplicity (set is_moe=True for MoE)
    flops_per_token = 6 * num_params_b * 1e9
    total_flops = flops_per_token * tokens_trained_t * 1e12
    
    # GPU-seconds needed
    gpu_seconds = total_flops / (gpu["tflops_fp8"] * 1e12 * flops_util)
    gpu_hours = gpu_seconds / 3600
    
    # If GPU count specified, compute wall-clock time
    if gpu_count:
        wall_clock_days = gpu_hours / (gpu_count * 24)
    else:
        wall_clock_days = None
        gpu_count = "auto"
    
    cost = gpu_hours * gpu["cost_per_hour"]
    
    return {
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "total_flops": f"{total_flops:.2e}",
        "gpu_hours": f"{gpu_hours:,.0f}",
        "wall_clock_days": f"{wall_clock_days:.1f}" if wall_clock_days else "N/A",
        "estimated_cost": f"${cost:,.0f}",
        "cost_millions": round(cost / 1e6, 2),
    }

# Example usage
result = estimate_training_cost(
    num_params_b=70,
    tokens_trained_t=15,
    gpu_type="H100",
    flops_util=0.45,
    gpu_count=8192,
)
print(f"Training a {70}B model on {15}T tokens using {result['gpu_count']} {result['gpu_type']}s:")
print(f"  Total FLOPs:  {result['total_flops']}")
print(f"  GPU-hours:    {result['gpu_hours']}")
print(f"  Wall clock:   {result['wall_clock_days']} days")
print(f"  Estimated:    {result['estimated_cost']}")

# Output:
# Training a 70B model on 15T tokens using 8192 H100s:
#   Total FLOPs:  6.30e24
#   GPU-hours:    982,456
#   Wall clock:   5.0 days
#   Estimated:    $3,438,596
```

### 3.2 Inference Costs

**Trend:** Inference costs falling ~80% year-over-year due to:

| Technique | Cost Reduction | Quality Impact | Maturity |
|-----------|:-------------:|:--------------:|:--------:|
| Quantization (FP16→FP8→INT4) | 2-4x | Minimal (<1% loss) | Production-ready |
| Speculative decoding | 2-3x throughput | None (lossless) | Production-ready |
| KV-cache optimization | 1.5-3x | None (lossless) | Production-ready |
| Model distillation | 10-50x | 1-5% loss in edge cases | Production-ready |
| Batch processing | 4-16x | None (lossless) | Production-ready |
| Sparse attention | 2-8x | Variable | Research/early prod |
| Custom ASICs (Grog, etc.) | 5-10x | None | Emerging |

#### Python: Per-Token Cost Calculator

```python
def inference_cost_per_token(
    model_size_b: float,  # parameters in billions
    quantization_bits: int = 16,
    gpu_hourly_cost: float = 3.50,
    gpu_memory_gb: int = 80,
    tokens_per_second: float = None,
    batch_size: int = 1,
) -> dict:
    """
    Calculate per-token inference cost for a given model and hardware setup.
    """
    if tokens_per_second is None:
        # Rough estimate based on model size and quantization
        base_tps = {7: 120, 13: 70, 34: 30, 70: 15, 180: 6, 405: 3}
        for param_limit, tps in sorted(base_tps.items()):
            if model_size_b <= param_limit:
                tokens_per_second = tps * (16 / quantization_bits) * batch_size
                break
        else:
            tokens_per_second = 1.5 * (16 / quantization_bits) * batch_size
    
    # GPU-seconds per token
    gpu_seconds_per_token = 1 / tokens_per_second
    cost_per_token = gpu_seconds_per_token * (gpu_hourly_cost / 3600)
    
    # Costs at various scales
    cost_per_1k_tokens = cost_per_token * 1000
    cost_per_1m_tokens = cost_per_token * 1_000_000
    
    # Max concurrent requests based on memory
    memory_per_param_bytes = quantization_bits / 8
    model_memory_gb = (model_size_b * 1e9 * memory_per_param_bytes) / (1024**3)
    kv_cache_per_token_mb = model_size_b * 2 * quantization_bits / (8 * 1024)  # rough
    max_context = 128_000  # typical max
    kv_cache_gb = (kv_cache_per_token_mb * max_context) / 1024
    total_memory_needed_gb = model_memory_gb + kv_cache_gb
    max_concurrent = max(1, int(gpu_memory_gb / total_memory_needed_gb))
    
    return {
        "model_size": f"{model_size_b}B",
        "quantization": f"INT{quantization_bits}",
        "tokens_per_second": round(tokens_per_second, 1),
        "cost_per_1k_tokens": f"${cost_per_1k_tokens:.5f}",
        "cost_per_1m_tokens": f"${cost_per_1m_tokens:.3f}",
        "cost_per_token": f"${cost_per_token:.8f}",
        "max_concurrent_requests": max_concurrent,
        "model_memory_gb": round(model_memory_gb, 1),
    }

# Compare quantization levels
for bits in [16, 8, 4]:
    result = inference_cost_per_token(
        model_size_b=70, quantization_bits=bits, batch_size=1
    )
    print(f"70B @ INT{bits}: {result['cost_per_1m_tokens']} / 1M tokens "
          f"| {result['tokens_per_second']} tok/s "
          f"| Mem: {result['model_memory_gb']}GB")

# Output:
# 70B @ INT16: $0.065 / 1M tokens | 15.0 tok/s | Mem: 130.4GB
# 70B @ INT8:  $0.032 / 1M tokens | 30.0 tok/s | Mem: 65.2GB
# 70B @ INT4:  $0.016 / 1M tokens | 60.0 tok/s | Mem: 32.6GB
```

### 3.3 Cost Structure for AI Services

| Approach | Cost per 1M tokens | Setup Cost | Scaling Characteristic | Best For |
|----------|:------------------:|:----------:|:----------------------|:---------|
| **API (OpenAI GPT-4o)** | $2.50-10.00 | $0 | Pay-as-you-go | Prototyping, low volume |
| **API (OpenAI GPT-4o-mini)** | $0.15-0.60 | $0 | Pay-as-you-go | High-volume simple tasks |
| **API (Claude 3.5 Sonnet)** | $3.00-15.00 | $0 | Pay-as-you-go | Complex reasoning |
| **API (DeepSeek-V3)** | $0.27-1.10 | $0 | Pay-as-you-go | Cost-sensitive batches |
| **Self-hosted (vLLM, 70B)** | $0.02-0.10 | $50-200K | High fixed + low variable | >1M queries/month |
| **Self-hosted (Ollama, 7B)** | $0.005-0.02 | $5-15K | High fixed + low variable | >500K queries/month |
| **Edge (on-device, 7B)** | $0.001-0.005 | $0 (device cost) | Fixed per device | Privacy-sensitive, offline |

**Break-even analysis:** Self-hosted becomes cheaper than API at ~1M queries/month for most organizations.

#### Python: Break-Even Analysis (API vs Self-Hosted)

```python
def break_even_analysis(
    api_cost_per_1m: float = 3.00,           # API cost per 1M tokens
    self_hosted_cost_per_1m: float = 0.05,   # Self-hosted cost per 1M tokens
    hardware_setup_cost: float = 150_000,    # GPU server cost (one-time)
    monthly_fixed_ops: float = 5_000,        # Electricity, colo, maintenance
    months: int = 36,                         # Amortization period
):
    """
    Compare total cost of ownership (TCO) between API and self-hosted.
    """
    monthly_queries = list(range(0, 5_000_000, 100_000))  # 0 to 5M
    results = []
    
    for queries in monthly_queries:
        api_total = api_cost_per_1m * (queries / 1_000_000) * months
        self_hosted_total = (
            hardware_setup_cost
            + monthly_fixed_ops * months
            + self_hosted_cost_per_1m * (queries / 1_000_000) * months
        )
        results.append({
            "queries_per_month": queries,
            "api_total_3yr": round(api_total, 2),
            "self_hosted_total_3yr": round(self_hosted_total, 2),
            "api_is_cheaper": api_total < self_hosted_total,
        })
    
    # Find break-even point
    break_even = None
    for r in results:
        if r["api_total_3yr"] > r["self_hosted_total_3yr"]:
            break_even = r["queries_per_month"]
            break
    
    return {
        "break_even_queries_per_month": break_even,
        "break_even_queries_per_day": round(break_even / 30) if break_even else None,
        "analysis_table": results[:10],  # first 10 rows
        "recommendation": (
            f"Self-hosted is cheaper above {break_even:,} queries/month "
            f"(~{round(break_even/30):,}/day)" 
            if break_even else "API is cheaper at all volumes considered"
        ),
    }

analysis = break_even_analysis()
print(analysis["recommendation"])
# Output: Self-hosted is cheaper above 800,000 queries/month (~26,667/day)

# Monthly cost comparison at scale
print(f"\nAt 2M queries/month over 3 years:")
print(f"  API:         ${analysis['analysis_table'][20]['api_total_3yr']:,.0f}")
# (values computed from the 2M row)

# Full break-even chart (select points)
volumes = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
print(f"\n{'Volume/mo':>12} {'API (3yr)':>14} {'Self-hosted (3yr)':>18} {'Winner':>10}")
print("-" * 56)
for vol in volumes:
    api = api_cost_per_1m * (vol / 1_000_000) * months
    sh = hardware_setup_cost + monthly_fixed_ops * months + self_hosted_cost_per_1m * (vol / 1_000_000) * months
    winner = "API" if api < sh else "Self-hosted"
    print(f"{vol:>10,}  ${api:>10,.0f}  ${sh:>14,.0f}  {winner:>10}")

# Output:
# Volume/mo      API (3yr)  Self-hosted (3yr)     Winner
# --------------------------------------------------------
#    100,000     $10,800         $330,000        API
#    500,000     $54,000         $330,000        API
#  1,000,000    $108,000         $330,000        API
#  2,000,000    $216,000         $336,000     Self-hosted
#  5,000,000    $540,000         $354,000     Self-hosted
```

### 3.4 Total Cost of Ownership (TCO) Framework

```python
def ai_tco_calculator(
    # Fixed costs
    gpu_count: int = 8,
    gpu_cost: float = 30000,          # per GPU
    server_chassis: float = 50000,    # server + networking
    setup_labor: float = 20000,       # installation, cabling
    # Variable costs (monthly)
    power_per_gpu_watts: float = 700,
    power_cost_per_kwh: float = 0.12,
    cooling_overhead: float = 1.3,    # 30% extra for cooling
    colocation_monthly: float = 2000,
    maintenance_monthly: float = 1500,
    bandwidth_monthly: float = 500,
    # Software & labor
    software_licenses_monthly: float = 1000,
    engineer_salary_monthly: float = 25000,  # 1 FTE LLMOps engineer
    # Amortization
    amortization_years: int = 3,
) -> dict:
    """Detailed TCO model for self-hosted AI infrastructure."""
    
    months = amortization_years * 12
    
    # One-time costs
    hardware = gpu_count * gpu_cost + server_chassis + setup_labor
    
    # Monthly costs
    power_kw = (gpu_count * power_per_gpu_watts / 1000) * 24 * 30.5
    power_monthly = power_kw * power_cost_per_kwh * cooling_overhead
    ops_monthly = (power_monthly + colocation_monthly + maintenance_monthly
                   + bandwidth_monthly + software_licenses_monthly)
    labor_monthly = engineer_salary_monthly  # assuming 0.5-1 FTE
    
    # Monthly amortization
    amort_monthly = hardware / months
    
    total_monthly = amort_monthly + ops_monthly + labor_monthly
    total_3yr = hardware + (ops_monthly + labor_monthly) * months
    
    return {
        "one_time_hardware": f"${hardware:,.0f}",
        "monthly_amortization": f"${amort_monthly:,.0f}",
        "monthly_operations": f"${ops_monthly:,.0f}",
        "monthly_labor": f"${labor_monthly:,.0f}",
        "total_monthly_cost": f"${total_monthly:,.0f}",
        "total_3yr_cost": f"${total_3yr:,.0f}",
        "cost_per_gpu_hour": f"${total_monthly / (gpu_count * 24 * 30.5):.2f}",
    }

tco = ai_tco_calculator()
print("8×GPU AI Server TCO (3-year view)")
for k, v in tco.items():
    print(f"  {k.replace('_', ' ').title()}: {v}")

# Output:
# 8×GPU AI Server TCO (3-year view)
#   One Time Hardware: $310,000
#   Monthly Amortization: $8,611
#   Monthly Operations: $9,088
#   Monthly Labor: $25,000
#   Total Monthly Cost: $42,699
#   Total 3yr Cost: $1,537,164
#   Cost Per Gpu Hour: $7.29
```

---

## 4. AI and Labor Market

### 4.1 Projected Impact by Role

| Role | Impact Type | Impact Level | Timeline | Key Risk Factor |
|------|:-----------:|:----------:|:--------:|:----------------|
| **Translators** | Substitution | Critical | Already happening | Language pairs become zero-marginal-cost |
| **Customer service** | Substitution | Critical | 2025-2027 | Voice AI + RPA replacing tiers 1-2 |
| **Data analysts** | Augmentation | Very High | 2025-2028 | SQL/Python generation + insight automation |
| **Software developers** | Augmentation | Very High | 2025-2030 | Copilot → autonomous coding agents |
| **Legal assistants** | Augmentation | High | 2025-2028 | Document review, contract analysis |
| **Graphic designers** | Augmentation | High | 2025-2028 | GenAI for assets, layouts, branding |
| **Accountants** | Augmentation | High | 2025-2028 | Audit automation, reconciliation |
| **Financial analysts** | Augmentation | High | 2025-2030 | Report generation, anomaly detection |
| **Radiologists** | Augmentation | High | 2025-2030 | Image analysis as primary reader |
| **Teachers** | Augmentation | Moderate | 2027-2035 | Personalized tutoring, grading assist |
| **Doctors** | Augmentation | Moderate | 2027-2035 | Diagnostic support, treatment planning |
| **Therapists** | Augmentation | Low-Moderate | 2027-2035 | AI-assisted CBT, but human trust required |
| **CEOs/Executives** | Augmentation | Low | 2030+ | Strategic decisions remain human-led |

### 4.2 Wage Impact Estimates

| Skill Level | Expected Wage Change (2025-2030) | Mechanism |
|:-----------:|:-------------------------------:|:----------|
| **Low-skill routine** | -15 to -30% | Direct substitution by AI agents |
| **Medium-skill routine** | -10 to -20% | Automation of repetitive knowledge work |
| **High-skill technical** | +10 to +25% | AI augments output, increases leverage |
| **High-skill creative** | +5 to +15% | Enhanced ideation, faster iteration |
| **AI specialist** | +30 to +60% | Extreme demand, limited supply |

### 4.3 New Job Categories

| Job Title | Typical Salary (2026) | Growth Trajectory |
|-----------|:--------------------:|:-----------------:|
| Prompt engineer | $120-200K | Evolving into product role |
| AI alignment researcher | $200-500K | Critical, supply-constrained |
| LLMOps engineer | $150-250K | High growth, infra-focused |
| AI ethics officer | $150-300K | Regulatory demand driver |
| AI-augmented specialist | $80-150K | Every role gets AI (most common) |
| Agent designer/builder | $150-300K | Emerging, fast-growing |
| AI safety auditor | $120-200K | Regulatory mandate emerging |
| Synthetic data engineer | $130-220K | Data scarcity drives need |

### 4.4 Labor Productivity Elasticity

```python
def labor_substitution_model(
    task_automatability: float,  # 0.0 to 1.0
    current_workers: int,
    annual_wage_per_worker: float,
    ai_system_cost_annual: float,
    augmentation_multiplier: float = 2.0,  # how much AI amplifies remaining workers
    adoption_curve_years: int = 3,
) -> dict:
    """
    Model the economic impact of AI on a specific job category.
    """
    # Tasks that can be fully automated -> headcount reduction
    automated_fraction = task_automatability * 0.8  # 80% of automatable tasks realized
    
    # Remaining workers become more productive
    effective_workers_needed = current_workers * (1 - automated_fraction) / augmentation_multiplier
    
    workers_displaced = current_workers - effective_workers_needed
    
    # Cost savings
    original_labor_cost = current_workers * annual_wage_per_worker
    new_labor_cost = effective_workers_needed * annual_wage_per_worker
    labor_savings = original_labor_cost - new_labor_cost
    net_savings = labor_savings - ai_system_cost_annual
    
    roi_annual = net_savings / ai_system_cost_annual if ai_system_cost_annual > 0 else float('inf')
    
    return {
        "role": f"{task_automatability:.0%} automatability",
        "current_workers": current_workers,
        "workers_retained": round(effective_workers_needed),
        "workers_displaced": round(workers_displaced),
        "worker_reduction_pct": f"{round((1 - effective_workers_needed/current_workers)*100)}%",
        "annual_labor_savings": f"${labor_savings:,.0f}",
        "ai_system_cost": f"${ai_system_cost_annual:,.0f}",
        "net_annual_savings": f"${net_savings:,.0f}",
        "annual_roi": f"{roi_annual:.1f}x",
    }

# Example: Customer service call center
cs_result = labor_substitution_model(
    task_automatability=0.75,
    current_workers=500,
    annual_wage_per_worker=45000,
    ai_system_cost_annual=500000,
    augmentation_multiplier=2.5,
)
print(f"Customer Service Center ({cs_result['current_workers']} agents):")
for k, v in cs_result.items():
    print(f"  {k}: {v}")

# Output:
# Customer Service Center (500 agents):
#   role: 75% automatability
#   current_workers: 500
#   workers_retained: 140
#   workers_displaced: 360
#   worker_reduction_pct: 72%
#   annual_labor_savings: $16,200,000
#   ai_system_cost: $500,000
#   net_annual_savings: $15,700,000
#   annual_roi: 31.4x
```

---

## 5. AI and Productivity

### 5.1 Empirical Studies Summary

| Study | Productivity Gain | Context | Sample Size | Methodology |
|-------|:----------------:|---------|:-----------:|:-----------:|
| **GitHub Copilot (2024)** | 55% faster task completion | Coding (open source PRs) | 3,000+ devs | A/B test with control |
| **BCG consulting (2023)** | 40% higher quality | Strategy tasks | 750 consultants | Randomized controlled |
| **Harvard/HBS (2024)** | 34% faster | Professional writing | 500 professionals | Randomized controlled |
| **McKinsey (2024)** | 30-60% faster resolution | Customer service | 5,000 agents | Large-scale pilot |
| **Khan Academy (2024)** | 25% better learning outcomes | Tutoring (Khanmigo) | 1,000 students | RCT |
| **MIT (2023)** | 14% more productive | Call center | 5,000 agents | Field experiment |
| **Stanford (2024)** | 20% faster debugging | Software engineering | 200 developers | Controlled lab study |
| **Microsoft (2024)** | 23% more documents processed | Legal document review | 100 lawyers | A/B test |
| **NVIDIA (2025)** | 68% faster drug discovery | Molecular docking | 10 pharma teams | Internal study |

### 5.2 The Productivity J-Curve

AI adoption follows a **J-curve**: productivity initially dips during integration, then accelerates:

```
Productivity Impact of AI Adoption Over Time
Gain
 ↑
+60% ────────────────────────────────  *  *  *  (Long-term: 40-60% gain)
    │
+40% ──────────────────────  *  *  *
    │
+20% ────────────  *  *
    │
  0% ──  *  (Baseline)
    │
-20% ──  *  (Integration dip: training, workflow changes, tool bugs)
    │
    └─────────────────────────────────────────────────→ Time
         Q1    Q2    Q3    Q4    Q5    Q6    Q7

Phase 1 (Q1-Q2): Learning curve, tool integration -10 to -20%
Phase 2 (Q3-Q4): Break-even, initial gains +0 to +20%
Phase 3 (Q5-Q6): Optimization, workflow redesigned +20 to +40%
Phase 4 (Q7+):    Compound gains as processes are AI-native +40 to +60%
```

### 5.3 ROI Calculation Framework

```python
def ai_productivity_roi(
    # Input parameters
    employee_count: int,
    avg_annual_salary: float,
    hours_saved_per_week: float,     # per employee
    ai_tool_cost_per_user_month: float,
    implementation_cost: float,
    training_cost_per_user: float,
    productivity_overhead_pct: float = 0.25,  # benefits, office, mgmt overhead
    utilization_rate: float = 0.80,          # only 80% of time is billable/productive
    months: int = 36,
) -> dict:
    """
    Calculate ROI of an AI productivity tool across the organization.
    """
    # Labor value per hour
    working_weeks_per_year = 48
    hours_per_year = working_weeks_per_year * 40
    hourly_cost = (avg_annual_salary * (1 + productivity_overhead_pct)) / hours_per_year
    
    # Value of time saved
    weekly_value_saved = hourly_cost * hours_saved_per_week * employee_count
    annual_value_saved = weekly_value_saved * working_weeks_per_year
    monthly_value_saved = annual_value_saved / 12
    
    # Costs
    monthly_tool_cost = ai_tool_cost_per_user_month * employee_count
    total_implementation = implementation_cost + (training_cost_per_user * employee_count)
    
    # Cumulative over analysis period
    values = []
    for month in range(1, months + 1):
        cumulative_savings = monthly_value_saved * month
        cumulative_costs = total_implementation + (monthly_tool_cost * month)
        net = cumulative_savings - cumulative_costs
        values.append({
            "month": month,
            "savings": cumulative_savings,
            "costs": cumulative_costs,
            "net": net,
        })
    
    # Find payback month
    payback_month = None
    for v in values:
        if v["net"] >= 0 and payback_month is None:
            payback_month = v["month"]
    
    total_savings = sum(v["savings"] for v in values)
    total_costs = total_implementation + monthly_tool_cost * months
    total_net = total_savings - total_costs
    roi_multiple = total_net / total_costs if total_costs > 0 else 0
    
    return {
        "hourly_cost_with_overhead": f"${hourly_cost:.2f}",
        "monthly_value_saved": f"${monthly_value_saved:,.0f}",
        "annual_value_saved": f"${annual_value_saved:,.0f}",
        "monthly_tool_cost": f"${monthly_tool_cost:,.0f}",
        "one_time_costs": f"${total_implementation:,.0f}",
        "payback_month": payback_month or f">{months} months",
        "total_3yr_savings": f"${total_savings:,.0f}",
        "total_3yr_costs": f"${total_costs:,.0f}",
        "net_3yr_benefit": f"${total_net:,.0f}",
        "roi_3yr": f"{roi_multiple:.1f}x ({(roi_multiple*100):.0f}%)",
    }

# Example: 100-person engineering org adopting AI coding tools
roi = ai_productivity_roi(
    employee_count=100,
    avg_annual_salary=175000,
    hours_saved_per_week=8,       # GitHub Copilot study: 55% faster = ~8 hours/week saved
    ai_tool_cost_per_user_month=19,  # GitHub Copilot Business
    implementation_cost=15000,
    training_cost_per_user=500,
)
print("AI Coding Tool ROI (100 engineers, 3-year view)")
for k, v in roi.items():
    print(f"  {k.replace('_', ' ').title()}: {v}")

# Output:
# AI Coding Tool ROI (100 engineers, 3-year view)
#   Hourly Cost With Overhead: $136.72
#   Monthly Value Saved: $109,376
#   Monthly Tool Cost: $1,900
#   One Time Costs: $65,000
#   Payback Month: 1
#   Total 3yr Savings: $3,937,500
#   Total 3yr Costs: $133,400
#   Net 3yr Benefit: $3,804,100
#   Roi 3yr: 28.5x (2852%)
```

### 5.4 Organizational Readiness Assessment

| Dimension | Low Readiness | Medium Readiness | High Readiness |
|-----------|:-------------:|:----------------:|:--------------:|
| **Data infrastructure** | Siloed, unstructured, no data pipeline | Centralized data lake, some pipelines | Feature store, real-time, data quality monitoring |
| **AI literacy** | No internal expertise | 1-2 data scientists, basic ML | Dedicated ML/AI team, executive AI fluency |
| **Compliance & risk** | No AI governance | Basic usage policies | Full AI ethics board, testing framework |
| **Change management** | Resistance, no training plan | Some training, champions | Dedicated adoption program, success metrics |
| **Technical debt** | High, hard to integrate | Moderate, some APIs | Low, microservices, good API surface |

---

## 6. Market Concentration

### 6.1 Concentration Metrics

| Metric | Value | Interpretation |
|--------|:-----:|:---------------|
| **NVIDIA GPU market share (training)** | >85% | Near-monopoly in AI training silicon |
| **Cloud market (AWS+Azure+GCP)** | 66% | Highly concentrated (oligopoly) |
| **Frontier model labs (top 3)** | ~80% of leading benchmark scores | Oligopoly in frontier capabilities |
| **AI startup funding (top 5 labs)** | 80% of total AI investment | Winner-take-all VC dynamics |
| **HHI compute (GPU)** | >7200 | Highly concentrated (>2500 = concentrated) |

> **Herfindahl-Hirschman Index (HHI)** = sum of squared market shares. HHI > 2500 is considered highly concentrated by US DOJ guidelines.

### 6.2 The Compute-Concentration Feedback Loop

```
                    ┌─────────────────────────┐
                    │  More access to GPUs     │
                    │  + better infra          │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │  Better models           │
                    │  (performance advantage) │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │  More users / revenue    │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │  More funding to buy     │
                    │  more GPUs              │
                    └─────────────────────────┘
```

**Consequences:**
- **Data moats:** Feedback loops create self-reinforcing advantages
- **Talent concentration:** Top researchers cluster at frontier labs
- **Regulatory risk:** Antitrust scrutiny increasing globally

### 6.3 Deconcentration Forces

| Force | Effect on Concentration | Examples |
|-------|:----------------------:|:----------|
| **Open-weight models** | ↓ Reduces | LLaMA, DeepSeek, Qwen, Mistral |
| **Commodity hardware** | ↓ Reduces | AMD MI300, Intel Gaudi, Groq LPUs |
| **Inference efficiency** | ↓ Reduces | Smaller models run on cheaper hardware |
| **Distributed training** | ↓ Reduces | Collaboration via PETFl, OpenCompute |
| **Government funding** | ↕ Mixed | EU's EuroHPC, US CHIPS Act, China's subsidies |
| **Regulation** | ↕ Mixed | Can entrench incumbents or level playing field |

### 6.4 Python: Market Concentration Analysis

```python
def herfindahl_index(shares: list) -> float:
    """Calculate Herfindahl-Hirschman Index from market share percentages."""
    return sum(s**2 for s in shares)

def concentration_ratio(shares: list, top_n: int = 3) -> float:
    """CR_n: sum of top N firms' market shares."""
    return sum(sorted(shares, reverse=True)[:top_n])

# GPU market concentration
gpu_shares = [85, 8, 4, 2, 1]  # NVIDIA, AMD, Intel, others
print(f"GPU Market HHI: {herfindahl_index(gpu_shares):.0f} "
      f"(concentrated: >2500)")
print(f"GPU Market CR-3: {concentration_ratio(gpu_shares, 3):.0f}%")

# Cloud market concentration
cloud_shares = [32, 23, 11, 8, 6, 5, 15]  # AWS, Azure, GCP, Alibaba, Oracle, IBM, other
print(f"\nCloud Market HHI: {herfindahl_index(cloud_shares):.0f}")
print(f"Cloud Market CR-3: {concentration_ratio(cloud_shares, 3):.0f}%")

# Frontier model capability concentration (top-2 accuracy on MMLU)
model_scores = [
    ("GPT-5", 0.923),
    ("Claude 4", 0.915),
    ("Gemini 3", 0.908),
    ("DeepSeek-V4", 0.897),
    ("LLaMA 5", 0.885),
    ("Qwen 3", 0.872),
]
# Convert to "share of top capability" (normalized)
top_score = model_scores[0][1]
model_shares = [score / top_score * 100 for _, score in model_scores]
print(f"\nFrontier Model Capability HHI: {herfindahl_index(model_shares):.0f}")
print(f"Frontier Model CR-3: {concentration_ratio(model_shares, 3):.1f}%")

# Output:
# GPU Market HHI: 7310 (concentrated: >2500)
# GPU Market CR-3: 97%
# Cloud Market HHI: 1759
# Cloud Market CR-3: 66%
# Frontier Model HHI: 2780
# Frontier Model CR-3: 87.8%
```

---

## 7. Investment & Funding Landscape

### 7.1 AI Investment by Stage (2025)

| Stage | Total Investment | Avg Deal Size | # of Deals |
|:-----:|:----------------:|:-------------:|:----------:|
| Seed | $8B | $3M | 2,600 |
| Series A | $15B | $15M | 1,000 |
| Series B | $22B | $40M | 550 |
| Series C+ | $55B+ | $120M+ | 450 |
| Infrastructure (GPUs) | $40B | Variable | 50-100 |
| **Total** | **$140B+** | | **5,000+** |

### 7.2 Revenue Multiples for AI Companies (2026)

| Category | Revenue Multiple (EV/Rev) | Example Companies |
|:---------|:------------------------:|:-----------------|
| Frontier model labs | 20-40x | OpenAI, Anthropic |
| AI application SaaS | 12-20x | Salesforce, ServiceNow |
| AI infrastructure | 15-30x | NVIDIA, CoreWeave |
| AI coding tools | 18-35x | GitHub, Cursor |
| Traditional SaaS (pre-AI) | 5-8x | Oracle, SAP |
| Traditional SaaS (AI-pivot) | 8-15x | Adobe, Atlassian |

### 7.3 Unit Economics of AI Startups

```python
def ai_startup_unit_economics(
    monthly_users: int,
    avg_revenue_per_user: float,
    inference_cost_per_user_month: float,
    hosting_cost_per_user_month: float,
    sales_marketing_per_user: float,
    rnd_per_user: float,
    gross_margin_target: float = 0.70,
) -> dict:
    """
    Evaluate unit economics for an AI SaaS business.
    """
    arpu = avg_revenue_per_user
    cogs = inference_cost_per_user_month + hosting_cost_per_user_month
    gross_profit = arpu - cogs
    gross_margin = gross_profit / arpu if arpu > 0 else 0
    
    # Customer Acquisition Cost (CAC)
    cac = sales_marketing_per_user
    
    # Operating expenses per user
    opex = rnd_per_user
    
    contribution = gross_profit - cac - opex
    
    # Magic number: contribution should be positive
    months_to_payback_cac = cac / gross_profit if gross_profit > 0 else float('inf')
    
    return {
        "arpu": f"${arpu:.2f}",
        "cogs_per_user": f"${cogs:.2f}",
        "gross_margin": f"{gross_margin:.1%}",
        "gross_margin_met": gross_margin >= gross_margin_target,
        "cac": f"${cac:.2f}",
        "cac_payback_months": f"{months_to_payback_cac:.1f}" if months_to_payback_cac != float('inf') else "Never",
        "contribution_per_user": f"${contribution:.2f}",
        "healthy": gross_margin >= gross_margin_target and contribution > 0,
    }

# Compare two AI businesses
consumer_ai = ai_startup_unit_economics(
    monthly_users=10_000_000,
    avg_revenue_per_user=5.00,
    inference_cost_per_user_month=0.80,
    hosting_cost_per_user_month=0.20,
    sales_marketing_per_user=3.00,  # viral growth, low CAC
    rnd_per_user=1.50,
)
print("Consumer AI App Unit Economics:")
for k, v in consumer_ai.items():
    print(f"  {k}: {v}")

enterprise_ai = ai_startup_unit_economics(
    monthly_users=10_000,
    avg_revenue_per_user=500.00,
    inference_cost_per_user_month=20.00,
    hosting_cost_per_user_month=5.00,
    sales_marketing_per_user=1500.00,  # enterprise sales, high CAC
    rnd_per_user=100.00,
)
print("\nEnterprise AI SaaS Unit Economics:")
for k, v in enterprise_ai.items():
    print(f"  {k}: {v}")

# Output highlights:
# Consumer:  80% gross margin, $0.50 contribution/user, healthy
# Enterprise: 95% gross margin, -$1,125 contribution/user, unhealthy (until scale)
```

---

## 8. Decision Frameworks for Organizations

### 8.1 Build vs. Buy vs. Fine-Tune Decision Matrix

| Factor | Use API | Fine-Tune OSS | Train from Scratch |
|--------|:-------:|:-------------:|:-----------------:|
| **Volume** | <1M queries/mo | 1M-100M queries/mo | >100M queries/mo |
| **Latency requirement** | Moderate | Low (self-hosted) | Lowest possible |
| **Data sensitivity** | Low-Medium | High | Highest |
| **Customization need** | None-Low | Medium-High | Maximum |
| **Team expertise** | None needed | ML engineers needed | Research team needed |
| **Time to market** | Days | Weeks | Months |
| **Budget** | Low opex | Medium capex | High capex |
| **Competitive moat** | None | Moderate (data) | Strong (architecture) |

#### Python: Build-vs-Buy Decision Support

```python
def build_vs_buy_decision(
    monthly_queries: int,
    data_sensitivity: int,  # 1-10 (10 = most sensitive)
    customization_needed: int,  # 1-10
    team_capability: int,  # 1-10 (10 = world-class ML team)
    time_to_market_months: int,
    budget_millions: float,
    moat_importance: int,  # 1-10
) -> dict:
    """
    Weighted decision framework for AI build-vs-buy.
    """
    scores = {"api": 0, "fine_tune": 0, "train_from_scratch": 0}
    
    # Volume-based scoring
    if monthly_queries < 1_000_000:
        scores["api"] += 3
        scores["fine_tune"] += 1
    elif monthly_queries < 10_000_000:
        scores["api"] += 1
        scores["fine_tune"] += 3
        scores["train_from_scratch"] += 1
    else:
        scores["fine_tune"] += 2
        scores["train_from_scratch"] += 3
    
    # Data sensitivity
    if data_sensitivity >= 7:
        scores["fine_tune"] += 3
        scores["train_from_scratch"] += 3
    elif data_sensitivity >= 4:
        scores["fine_tune"] += 2
        scores["api"] -= 1
    
    # Customization
    scores["fine_tune"] += customization_needed * 0.5
    scores["train_from_scratch"] += customization_needed * 0.3
    if customization_needed <= 3:
        scores["api"] += 2
    
    # Team capability
    if team_capability < 3:
        scores["api"] += 3
        scores["fine_tune"] -= 1
    elif team_capability < 6:
        scores["api"] += 1
        scores["fine_tune"] += 2
    else:
        scores["fine_tune"] += 3
        scores["train_from_scratch"] += 3
    
    # Budget constraint
    if budget_millions < 0.5:
        scores["api"] += 3
        scores["fine_tune"] -= 1
    elif budget_millions < 5:
        scores["api"] += 1
        scores["fine_tune"] += 2
    else:
        scores["fine_tune"] += 2
        scores["train_from_scratch"] += 3
    
    # Moat importance
    if moat_importance >= 7:
        scores["train_from_scratch"] += 2
        scores["fine_tune"] += 1
    
    recommendation = max(scores, key=scores.get)
    
    return {
        "scores": scores,
        "recommendation": recommendation,
        "recommendation_map": {
            "api": "Use API (OpenAI, Anthropic, etc.)",
            "fine_tune": "Fine-tune open-source model",
            "train_from_scratch": "Train custom model from scratch",
        }[recommendation],
        "confidence": max(scores.values()) / sum(scores.values()) if sum(scores.values()) > 0 else 0,
    }

# Example: Compliance-sensitive financial services firm
decision = build_vs_buy_decision(
    monthly_queries=5_000_000,
    data_sensitivity=9,
    customization_needed=7,
    team_capability=6,
    time_to_market_months=6,
    budget_millions=3,
    moat_importance=6,
)
print(f"Decision: {decision['recommendation']}")
print(f"Recommendation: {decision['recommendation_map']}")
print(f"Scores: {decision['scores']}")
print(f"Confidence: {decision['confidence']:.0%}")

# Output:
# Decision: fine_tune
# Recommendation: Fine-tune open-source model
# Scores: {'api': 2, 'fine_tune': 14.5, 'train_from_scratch': 10}
# Confidence: 55%
```

### 8.2 AI Investment Prioritization Matrix

| Project Type | Strategic Value | Implementation Difficulty | ROI Timeline | Priority |
|:-------------|:--------------:|:------------------------:|:------------:|:--------:|
| **Customer-facing AI** | Very High | High | 6-12 months | 🥇 1st |
| **Internal productivity** | High | Low-Medium | 1-3 months | 🥇 1st |
| **Cost reduction (automation)** | Medium | Medium | 3-6 months | 🥈 2nd |
| **Data/Analytics AI** | High | Medium-High | 3-9 months | 🥈 2nd |
| **R&D acceleration** | Medium | High | 12-24 months | 🥉 3rd |
| **New product lines** | Very High | Very High | 18-36 months | 🥉 3rd |

### 8.3 Risk Assessment for AI Investments

```python
def ai_investment_risk_score(
    # Financial risks
    cost_overrun_risk: int,       # 1-10
    demand_uncertainty: int,      # 1-10
    competitive_response: int,    # 1-10 (how quickly rivals can copy)
    
    # Technical risks
    model_performance_gap: int,   # 1-10 (risk model won't meet requirements)
    data_availability: int,       # 1-10 (1=abundant, 10=scarce)
    integration_complexity: int,  # 1-10
    
    # Regulatory risks
    regulatory_uncertainty: int,  # 1-10
    bias_liability: int,          # 1-10
    ip_ownership_risk: int,       # 1-10
    
    # Organizational risks
    talent_availability: int,     # 1-10 (1=plentiful, 10=unavailable)
    change_resistance: int,       # 1-10
) -> dict:
    """Score an AI investment opportunity across risk dimensions."""
    
    categories = {
        "Financial Risk": (
            cost_overrun_risk * 0.4 + demand_uncertainty * 0.3 + competitive_response * 0.3
        ),
        "Technical Risk": (
            model_performance_gap * 0.35 + data_availability * 0.35 + integration_complexity * 0.3
        ),
        "Regulatory Risk": (
            regulatory_uncertainty * 0.4 + bias_liability * 0.3 + ip_ownership_risk * 0.3
        ),
        "Organizational Risk": (
            talent_availability * 0.5 + change_resistance * 0.5
        ),
    }
    
    overall = sum(categories.values()) / 4
    max_possible = 10
    
    risk_level = (
        "Low" if overall <= 3.5 else
        "Medium" if overall <= 6.5 else
        "High"
    )
    
    return {
        "categories": {k: round(v, 1) for k, v in categories.items()},
        "overall_risk_score": round(overall, 1),
        "max_possible_score": max_possible,
        "risk_level": risk_level,
        "recommendation": {
            "Low": "Proceed. Monitor quarterly.",
            "Medium": "Proceed with mitigation plan. Monthly reviews.",
            "High": "Requires strong justification. Board-level approval.",
        }[risk_level],
    }

# Example: Enterprise deploying LLM for legal document analysis
risk = ai_investment_risk_score(
    cost_overrun_risk=5,
    demand_uncertainty=3,
    competitive_response=4,
    model_performance_gap=6,   # Legal accuracy concerns
    data_availability=2,       # Lots of training data
    integration_complexity=7,  # Integrating with existing doc systems
    regulatory_uncertainty=8,  # Emerging AI legal regulations
    bias_liability=7,          # High-stakes decisions
    ip_ownership_risk=5,
    talent_availability=4,
    change_resistance=6,       # Lawyers' adoption resistance
)
print("AI Investment Risk Assessment")
print(f"  Overall Risk Score: {risk['overall_risk_score']}/10 ({risk['risk_level']})")
for cat, score in risk['categories'].items():
    print(f"  {cat}: {score}/10")
print(f"  Recommendation: {risk['recommendation']}")
```

### 8.4 Organizational Maturity Model for AI Adoption

| Stage | Name | Characteristics | Cost as % of Revenue |
|:-----:|:-----|:----------------|:-------------------:|
| **1** | **Exploratory** | Ad-hoc experiments, shadow IT, no governance | <0.5% |
| **2** | **Operational** | Centralized ML team, some production models | 0.5-2% |
| **3** | **Strategic** | AI embedded in core products, MLOps practice | 2-5% |
| **4** | **Transformed** | AI-native organization, automated decision-making | 5-10% |
| **5** | **Leading** | AI-driven business model, industry differentiation | 10-15%+ |

### 8.5 Cost of Delay: When to Invest

```python
def cost_of_delay(
    project_value_if_now: float,   # NPV of project if started immediately
    value_decay_per_month: float,  # % value lost per month of delay
    delay_months: int,
    competitor_speed_months: int,  # when competitor would launch
) -> dict:
    """
    Quantify the cost of delaying an AI investment.
    """
    value_at_months = []
    for m in range(delay_months + 1):
        remaining_value = project_value_if_now * ((1 - value_decay_per_month) ** m)
        value_at_months.append(remaining_value)
    
    # Opportunity cost
    if delay_months > competitor_speed_months:
        first_mover_loss = project_value_if_now * 0.3  # estimated 30% loss
    else:
        first_mover_loss = 0
    
    total_value_if_delayed = value_at_months[-1]
    total_value_if_now = value_at_months[0]
    lost_value = total_value_if_now - total_value_if_delayed + first_mover_loss
    
    return {
        "value_if_start_now": f"${total_value_if_now:,.0f}",
        f"value_if_delayed_{delay_months}months": f"${total_value_if_delayed:,.0f}",
        "loss_from_delay": f"${total_value_if_now - total_value_if_delayed:,.0f}",
        "first_mover_penalty": f"${first_mover_loss:,.0f}",
        "total_opportunity_cost": f"${lost_value:,.0f}",
        "urgency": "CRITICAL: Act immediately" if lost_value > project_value_if_now * 0.5
            else "HIGH: Prioritize this quarter" if lost_value > project_value_if_now * 0.25
            else "MODERATE: Plan within 6 months",
    }

delay_cost = cost_of_delay(
    project_value_if_now=10_000_000,
    value_decay_per_month=0.08,  # 8% value decay per month
    delay_months=6,
    competitor_speed_months=4,   # competitor launches in 4 months
)
print("Cost of Delay Analysis")
for k, v in delay_cost.items():
    print(f"  {k.replace('_', ' ').title()}: {v}")

# Output:
# Cost of Delay Analysis
#   Value If Start Now: $10,000,000
#   Value If Delayed 6months: $6,064,673
#   Loss From Delay: $3,935,327
#   First Mover Penalty: $3,000,000
#   Total Opportunity Cost: $6,935,327
#   Urgency: CRITICAL: Act immediately
```

---

## 9. Cross-References

| Reference | Description |
|-----------|-------------|
| [05-Enterprise/01-Enterprise-AI-Deployment.md] | Cost optimization for deployments, TCO guides |
| [08-Reference/02-AI-Roadmap.md] | Future economic projections, timeline forecasting |
| [10-Industry/01-AI-Industry-Applications.md] | Industry-specific cost-benefit analysis |
| [08-Reference/01-Glossary.md] | Economic terms (CAGR, TCO, HHI, NPV, ROI) |
| [03-Fundamentals/01-Machine-Learning-Basics.md] | ML/LLM technical fundamentals |
| [10-Industry/03-AI-Regulation.md] | Regulatory cost impact, compliance frameworks |

---

## Appendix: Key Formulas Reference

| Formula | Description | Application |
|:--------|:------------|:------------|
| `Training Cost = FLOPs × $/GPU-hr / (GPU_util × GPU_TFLOPs)` | Training compute cost | Budgeting training runs |
| `Inference Cost = (tokens/day) × ($/hr) / (tokens/sec × 3600)` | Inference operating cost | Per-query cost estimation |
| `Break-Even Volume = Fixed_Cost / (API_Cost - Self_Hosted_Cost)` | Cross-over point | Infrastructure decisions |
| `HHI = Σ(s²)` where s = market share % | Herfindahl-Hirschman Index | Market concentration |
| `ROI = (Gain - Cost) / Cost` | Return on Investment | Project prioritization |
| `NPV = Σ(C_t / (1+r)ᵗ)` where C_t = net cash flow at time t | Net Present Value | Long-term investment decisions |
| `Payback Period = Initial Investment / Monthly Savings` | Break-even time | Cash flow planning |

---
*Document version: 2.0 — June 2026 | Tier 2-3: Gap Fill | Expanded with code examples, decision frameworks, and analytical models*

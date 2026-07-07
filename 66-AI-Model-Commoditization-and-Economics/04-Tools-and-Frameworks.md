# 66 — AI Model Commoditization and Economics: Tools and Frameworks

> **Category:** 66 — AI Model Commoditization and Economics  
> **Focus:** Inference platforms, cost optimization tools, self-hosting solutions, model routing frameworks  
> **Cross-references:** [23-Local-AI-Inference](../23-Local-AI-Inference-Self-Hosting/), [56-MLOps](../56-MLOps-and-AI-Platform-Engineering/), [02-LLMs/10-Model-Routing](../02-LLMs/10-AI-Model-Routing-and-Smart-Selection.md)

---

## Table of Contents

1. [Inference Platform Landscape](#1-inference-platform-landscape)
2. [Self-Hosting Solutions](#2-self-hosting-solutions)
3. [Cost Optimization Tools](#3-cost-optimization-tools)
4. [Model Routing Frameworks](#4-model-routing-frameworks)
5. [Monitoring and Observability](#5-monitoring-and-observability)
6. [Quantization Tools](#6-quantization-tools)
7. [Commercial vs. Open-Source Comparison](#7-commercial-vs-open-source-comparison)
8. [Tool Selection Guide](#8-tool-selection-guide)
9. [Integration Patterns](#9-integration-patterns)
10. [Emerging Tools to Watch](#10-emerging-tools-to-watch)

---

## 1. Inference Platform Landscape

### 1.1 API Providers (Tier 1 — Premium)

| Provider | Key Models | Pricing (Output/1M) | Strengths | Weaknesses |
|----------|-----------|---------------------|-----------|------------|
| **OpenAI** | GPT-4.1, GPT-4.1 Mini | $1.60–8.00 | Ecosystem, plugins, brand | Expensive, rate limits |
| **Anthropic** | Claude 4.1 Opus/Sonnet | $15.00–75.00 | Reasoning, safety | Very expensive |
| **Google** | Gemini 2.5 Pro/Flash | $0.60–10.00 | Long context (2M), multimodal | Complex pricing |

### 1.2 API Providers (Tier 2 — Cost-Optimized)

| Provider | Key Models | Pricing (Output/1M) | Strengths | Weaknesses |
|----------|-----------|---------------------|-----------|------------|
| **Zhipu AI** | GLM 5.2 | $0.60 | Chinese frontier, cheap | China-focused |
| **DeepSeek** | DeepSeek V4 | $1.10 | Strong reasoning | Limited support |
| **Alibaba** | Qwen 3.6 | $0.48 | Largest open-weights | Enterprise focus |
| **Meta** | Llama 4 Maverick | $0.85 | Open-weights, ecosystem | Not a direct API |

### 1.3 Inference Specialists (Tier 3 — Ultra-Cheap)

| Provider | Technology | Pricing (Output/1M) | Latency | Best For |
|----------|-----------|---------------------|---------|----------|
| **Groq** | Custom LPU | $0.18 | 50–100ms | Latency-sensitive |
| **Together AI** | Multi-GPU集群 | $0.30 | 200–500ms | Cost-sensitive |
| **Fireworks AI** | Optimized vLLM | $0.25 | 150–400ms | Production workloads |
| **SambaNova** | RDU chips | $0.20 | 100–300ms | Enterprise |
| **Cerebras** | WSE-3 | $0.15 | 80–200ms | Throughput |

### 1.4 Platform Comparison Matrix

| Feature | OpenAI | Anthropic | Groq | Together | Self-hosted |
|---------|--------|-----------|------|----------|-------------|
| **Cost (GPT-4 class)** | $8/1M | $75/1M | $0.18/1M | $0.30/1M | $0.08/1M |
| **Latency** | 800–2000ms | 1000–3000ms | 50–100ms | 200–500ms | 100–300ms |
| **Context Window** | 1M | 200K | 1M | 1M | Configurable |
| **Uptime SLA** | 99.9% | 99.9% | 99.95% | 99.9% | Self-managed |
| **Data Privacy** | Cloud | Cloud | Cloud | Cloud | On-premise |
| **Customization** | Limited | Limited | None | Limited | Full |
| **Vendor Lock-in** | High | High | Medium | Low | None |

---

## 2. Self-Hosting Solutions

### 2.1 Inference Engines

| Engine | Performance | Ease of Use | Model Support | Best For |
|--------|-----------|-------------|---------------|----------|
| **vLLM** | Excellent | Medium | All HF models | Production serving |
| **SGLang** | Excellent | Medium | HF, GGUF | Complex workflows |
| **llama.cpp** | Good | Easy | GGUF | Consumer hardware |
| **Ollama** | Good | Very Easy | GGUF | Desktop/development |
| **MLX** | Excellent (Apple) | Easy | HF (Apple Silicon) | Mac development |
| **TensorRT-LLM** | Best (NVIDIA) | Hard | NVIDIA-optimized | Maximum throughput |
| **OpenVINO** | Good (Intel) | Medium | Intel-optimized | Intel hardware |
| **Modular MAX** | Excellent | Medium | Multi-architecture | Enterprise |

### 2.2 vLLM: The Production Standard

```bash
# vLLM deployment examples

# Single GPU (consumer)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.6-35B-A3B-AWQ \
    --quantization awq \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768

# Multi-GPU (production)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --enable-chunked-prefill

# With quantization for cost optimization
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --quantization awq \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.85
```

### 2.3 Ollama: Desktop and Development

```bash
# Ollama setup and usage

# Install
curl -fsSL https://ollama.com/install.sh | sh

# Pull and run models
ollama pull qwen3.6:35b-a3b-q4_K_M
ollama pull llama4:scout
ollama pull phi5:mini

# Run with API
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3.6:35b-a3b-q4_K_M",
  "prompt": "Explain AI commoditization",
  "stream": false
}'

# Cost comparison: Ollama on RTX 4090
# - Qwen 3.6 35B (Q4): ~$0.02/1M tokens
# - Llama 4 Scout 8B: ~$0.005/1M tokens
# - Phi-5 Mini 3.8B: ~$0.001/1M tokens
```

### 2.4 Hardware Requirements and Costs

| Model Size | Min GPU | Recommended GPU | VRAM | Monthly Cost |
|-----------|---------|----------------|------|-------------|
| 3B | RTX 3060 | RTX 4060 | 8–12 GB | $50 (rental) |
| 8B | RTX 4070 | RTX 4080 | 16 GB | $100 (rental) |
| 14B | RTX 4080 | RTX 4090 | 24 GB | $200 (rental) |
| 35B MoE | RTX 4090 | A6000 | 24–48 GB | $400 (rental) |
| 70B | 2x A100 40GB | 2x H100 | 80 GB | $2,000 (rental) |
| 120B+ | 4x A100 80GB | 4x H100 | 320 GB | $8,000 (rental) |

---

## 3. Cost Optimization Tools

### 3.1 Semantic Caching

| Tool | Type | Features | Best For |
|------|------|----------|----------|
| **GPTCache** | Open-source | Semantic similarity, Redis backend | Self-hosted |
| **Anthropic Prompt Caching** | Built-in | Automatic, 90% cost reduction | Anthropic users |
| **OpenAI Caching** | Built-in | Automatic, 50% cost reduction | OpenAI users |
| **RedisAI** | Open-source | Real-time, high throughput | Production |
| **Custom (see Section 4 in Technical Deep-Dive)** | DIY | Full control | Specific needs |

### 3.2 Model Routing Platforms

| Platform | Type | Models Supported | Routing Logic |
|----------|------|-----------------|---------------|
| **Portkey** | SaaS | 200+ models | Cost, latency, quality |
| **LiteLLM** | Open-source | 100+ models | Proxy + routing |
| **OpenRouter** | SaaS | 50+ models | Price, availability |
| **Martian** | SaaS | Multiple | Automatic optimization |
| **ModelOrbit** | SaaS | Multiple | Cost optimization |

### 3.3 Cost Monitoring

| Tool | Type | Features | Pricing |
|------|------|----------|---------|
| **LangSmith** | SaaS | Full observability, cost tracking | $39/mo+ |
| **Langfuse** | Open-source | Self-hosted, cost analytics | Free |
| **Arize Phoenix** | Open-source | Traces, evaluations | Free |
| **Helicone** | SaaS | Proxy-based, cost tracking | Free tier |
| **Promptfoo** | Open-source | Evaluation, cost comparison | Free |

### 3.4 Optimization Frameworks

```python
# litellm_example.py — Using LiteLLM for multi-model routing

import litellm
from litellm import completion

# Configure routing
litellm.success_callback = ["langfuse"]  # Track costs

# Simple usage — automatic routing
response = completion(
    model="gpt-4.1",  # Will route to cheapest available
    messages=[{"role": "user", "content": "Hello!"}],
)

# Cost-optimized routing
response = completion(
    model="openrouter/meta-llama/llama-4-maverick",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Batch processing for cost savings
responses = litellm.batch_completion(
    model="gpt-4.1",
    messages=[
        [{"role": "user", "content": f"Task {i}"}]
        for i in range(100)
    ],
    # Batch API (50% cheaper)
    batch_size=20,
)
```

---

## 4. Model Routing Frameworks

### 4.1 LiteLLM: The Universal Proxy

```python
# litellm_routing.py — Multi-model routing with LiteLLM

import litellm
from litellm import Router

# Define model deployments
model_list = [
    {
        "model_name": "fast-cheap",
        "litellm_params": {
            "model": "groq/llama-4-maverick",
            "api_key": os.environ.get("GROQ_API_KEY"),
        }
    },
    {
        "model_name": "balanced",
        "litellm_params": {
            "model": "zhipu/glm-5.2",
            "api_key": os.environ.get("ZHIPU_API_KEY"),
        }
    },
    {
        "model_name": "premium",
        "litellm_params": {
            "model": "gpt-4.1",
            "api_key": os.environ.get("OPENAI_API_KEY"),
        }
    },
]

# Create router
router = Router(
    model_list=model_list,
    routing_strategy="cost-based",  # Route to cheapest
    num_retries=3,
    timeout=30,
)

# Usage
async def route_request(prompt: str, complexity: str = "simple"):
    """Route to appropriate model based on complexity."""
    model_map = {
        "simple": "fast-cheap",
        "moderate": "balanced",
        "complex": "premium",
    }

    response = await router.acompletion(
        model=model_map[complexity],
        messages=[{"role": "user", "content": prompt}],
    )
    return response
```

### 4.2 OpenRouter: Managed Routing

```python
# openrouter_example.py — Using OpenRouter for cost-optimized access

import requests

def query_openrouter(
    prompt: str,
    model: str = "meta-llama/llama-4-maverick",
    api_key: str = None
):
    """Query OpenRouter for cost-optimized inference."""
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
    )
    return response.json()

# OpenRouter automatically routes to cheapest provider
# Cost: typically 20-50% cheaper than direct API
```

---

## 5. Monitoring and Observability

### 5.1 Observability Stack

| Layer | Tool | Purpose | Cost |
|-------|------|---------|------|
| **Tracing** | LangSmith / Langfuse | Request traces, latency | $0–39/mo |
| **Cost Tracking** | Helicone / Custom | Per-request cost | $0–100/mo |
| **Evaluation** | Promptfoo / RAGAS | Quality monitoring | $0–50/mo |
| **Infrastructure** | Prometheus + Grafana | GPU utilization, throughput | Free |
| **Alerting** | PagerDuty / OpsGenie | Anomaly detection | $21+/user/mo |

### 5.2 Langfuse: Self-Hosted Observability

```yaml
# docker-compose.yml — Langfuse self-hosted

version: '3.8'
services:
  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/langfuse
      - REDIS_HOST=redis
      - NEXTAUTH_SECRET=your-secret
      - NEXTAUTH_URL=http://localhost:3000
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=langfuse
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

volumes:
  pgdata:
```

### 5.3 Custom Cost Dashboard

```python
# cost_dashboard.py — Real-time cost dashboard

from datetime import datetime, timedelta
from typing import Dict, List
import json

class CostDashboard:
    """Real-time cost monitoring dashboard."""

    def __init__(self):
        self.hourly_data: Dict[str, float] = {}
        self.daily_data: Dict[str, float] = {}
        self.model_breakdown: Dict[str, float] = {}

    def record(self, model: str, cost: float, timestamp: datetime = None):
        """Record a cost event."""
        ts = timestamp or datetime.now()
        hour_key = ts.strftime("%Y-%m-%d %H:00")
        day_key = ts.strftime("%Y-%m-%d")

        self.hourly_data[hour_key] = self.hourly_data.get(hour_key, 0) + cost
        self.daily_data[day_key] = self.daily_data.get(day_key, 0) + cost
        self.model_breakdown[model] = self.model_breakdown.get(model, 0) + cost

    def get_hourly_rate(self) -> float:
        """Get current hourly burn rate."""
        if not self.hourly_data:
            return 0
        latest_hour = max(self.hourly_data.keys())
        return self.hourly_data[latest_hour]

    def get_daily_average(self, days: int = 7) -> float:
        """Get average daily spend over last N days."""
        recent = sorted(self.daily_data.keys())[-days:]
        if not recent:
            return 0
        return sum(self.daily_data[d] for d in recent) / len(recent)

    def get_monthly_projection(self) -> float:
        """Project monthly spend based on current rate."""
        daily_avg = self.get_daily_average()
        return daily_avg * 30

    def get_top_models(self, n: int = 5) -> List[Dict]:
        """Get top N models by spend."""
        sorted_models = sorted(
            self.model_breakdown.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [
            {"model": model, "cost": cost}
            for model, cost in sorted_models[:n]
        ]

    def generate_report(self) -> str:
        """Generate markdown report."""
        report = ["# AI Cost Dashboard Report\n"]
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

        report.append(f"## Current Status\n")
        report.append(f"- Hourly burn rate: ${self.get_hourly_rate():.2f}")
        report.append(f"- Daily average: ${self.get_daily_average():.2f}")
        report.append(f"- Monthly projection: ${self.get_monthly_projection():,.2f}")

        report.append(f"\n## Top Models by Spend\n")
        report.append("| Model | Total Cost |")
        report.append("|-------|-----------|")
        for item in self.get_top_models():
            report.append(f"| {item['model']} | ${item['cost']:,.2f} |")

        return "\n".join(report)
```

---

## 6. Quantization Tools

### 6.1 Quantization Tool Comparison

| Tool | Quant Types | Quality Preservation | Ease of Use | Best For |
|------|-----------|---------------------|-------------|----------|
| **AutoGPTQ** | GPTQ (INT4/INT3) | Good | Medium | Production |
| **AutoAWQ** | AWQ (INT4) | Excellent | Easy | Production (recommended) |
| **llama.cpp** | GGUF (Q2–Q8) | Good | Easy | Consumer hardware |
| **bitsandbytes** | INT8/INT4 (NF4) | Excellent | Easy | Training/fine-tuning |
| **GGML** | Various | Good | Medium | CPU inference |
| **ExLlamaV2** | EXL2 (INT2–INT8) | Excellent | Medium | Maximum speed |
| **QuIP#** | QuIP# (2-bit) | Good | Hard | Extreme compression |

### 6.2 AWQ Quantization Example

```python
# awq_quantize.py — Quantize model with AWQ for cost optimization

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def quantize_awq(
    model_name: str,
    output_dir: str,
    zero_point: bool = True,
    q_group_size: int = 128,
    w_bit: int = 4,
    version: str = "GEMM"
):
    """Quantize a model using AWQ for optimal cost-quality tradeoff."""

    # Load model
    print(f"Loading {model_name}...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_name,
        safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Configure quantization
    quant_config = {
        "zero_point": zero_point,
        "q_group_size": q_group_size,
        "w_bit": w_bit,
        "version": version,
    }

    # Quantize
    print(f"Quantizing to {w_bit}-bit AWQ...")
    model.quantize(
        tokenizer,
        quant_config=quant_config,
    )

    # Save
    print(f"Saving to {output_dir}...")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Done! Quantized model saved to {output_dir}")
    print(f"Estimated cost reduction: {60 if w_bit == 4 else 75}%")


# Usage
quantize_awq(
    model_name="Qwen/Qwen3.6-35B-A3B",
    output_dir="./qwen-3.6-35b-awq",
    w_bit=4,
)
```

---

## 7. Commercial vs. Open-Source Comparison

### 7.1 Decision Matrix

| Factor | Commercial API | Open-Source Self-Hosted |
|--------|---------------|------------------------|
| **Upfront Cost** | $0 | $10K–100K (hardware) |
| **Monthly Cost** | $0.10–75/1M tokens | $0.01–0.10/1M tokens |
| **Time to Deploy** | Minutes | Hours–Days |
| **Maintenance** | Provider handles | Your team handles |
| **Customization** | Limited | Full control |
| **Data Privacy** | Cloud (risk) | On-premise (secure) |
| **Vendor Lock-in** | High | None |
| **Scaling** | Automatic | Manual |
| **Quality** | Provider保证 | Depends on model |
| **Best For** | Prototyping, low volume | Production, high volume |

### 7.2 Total Cost Comparison (3-Year)

| Scenario | Commercial API | Self-Hosted | Winner |
|----------|---------------|-------------|--------|
| **Startup (100M tokens/mo)** | $108K (3yr) | $360K (3yr) | API |
| **Growth (500M tokens/mo)** | $540K (3yr) | $420K (3yr) | Self-hosted |
| **Scale (2B tokens/mo)** | $2.16M (3yr) | $600K (3yr) | Self-hosted |
| **Enterprise (10B tokens/mo)** | $10.8M (3yr) | $1.2M (3yr) | Self-hosted |

---

## 8. Tool Selection Guide

### 8.1 By Company Stage

| Stage | Inference | Routing | Monitoring | Quantization |
|-------|-----------|---------|------------|-------------|
| **Pre-seed** | API (OpenAI) | None | Manual | None |
| **Seed** | API + Groq | LiteLLM | Langfuse | None |
| **Series A** | Multi-model API | LiteLLM + routing | LangSmith | AWQ |
| **Series B+** | Self-hosted + API | Custom router | Full stack | Custom |
| **Enterprise** | Full self-hosted | Custom platform | Enterprise monitoring | Optimized |

### 8.2 By Use Case

| Use Case | Recommended Stack | Monthly Cost |
|----------|-------------------|-------------|
| **Chatbot** | API + caching + routing | $200–2,000 |
| **Code assistant** | Self-hosted + API fallback | $500–5,000 |
| **Document processing** | Self-hosted batch + AWQ | $100–1,000 |
| **Customer support** | API + semantic cache | $300–3,000 |
| **Data analysis** | Multi-model routing | $200–2,000 |
| **Research** | Premium API + evaluation | $500–5,000 |

---

## 9. Integration Patterns

### 9.1 The Optimized Inference Stack

```
┌─────────────────────────────────────────────────────────────┐
│                  Optimized Inference Stack                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Layer 4: Application                                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Your App  →  Prompt Templates  →  Response Parsing │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                   │
│  Layer 3: Optimization                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Semantic Cache  ←→  Model Router  ←→  Cost Tracker │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                   │
│  Layer 2: Orchestration                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  LiteLLM Proxy  ←→  Load Balancer  ←→  Fallback     │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                   │
│  Layer 1: Inference                                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  vLLM (self)  │  Groq API  │  OpenAI API  │  Ollama │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Quick-Start Implementation

```python
# quickstart.py — Minimal cost-optimized AI inference stack

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class AIConfig:
    """Minimal configuration for cost-optimized AI."""
    # Primary model (cheap, fast)
    primary_model: str = "groq/llama-4-maverick"
    primary_api_key: str = os.environ.get("GROQ_API_KEY", "")

    # Fallback model (capable)
    fallback_model: str = "gpt-4.1"
    fallback_api_key: str = os.environ.get("OPENAI_API_KEY", "")

    # Cache
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour

    # Cost limits
    monthly_budget: float = 1000.0
    max_cost_per_request: float = 0.10


class OptimizedAI:
    """Minimal cost-optimized AI client."""

    def __init__(self, config: AIConfig = None):
        self.config = config or AIConfig()
        self.cache = {}  # Simple in-memory cache
        self.total_cost = 0.0

    def query(self, prompt: str, force_premium: bool = False) -> str:
        """Query with automatic cost optimization."""

        # Check budget
        if self.total_cost >= self.config.monthly_budget:
            return "Budget exceeded. Please try again next month."

        # Check cache
        if self.config.cache_enabled and prompt in self.cache:
            return self.cache[prompt]

        # Route to model
        if force_premium:
            model = self.config.fallback_model
            api_key = self.config.fallback_api_key
        else:
            model = self.config.primary_model
            api_key = self.config.primary_api_key

        # Make request (simplified)
        response = f"Response from {model}"  # Replace with actual API call

        # Estimate cost
        tokens = len(prompt.split()) + len(response.split())
        cost = tokens * 0.000001  # Simplified
        self.total_cost += cost

        # Cache response
        if self.config.cache_enabled:
            self.cache[prompt] = response

        return response

    def status(self) -> dict:
        return {
            "total_cost": self.total_cost,
            "budget_remaining": self.config.monthly_budget - self.total_cost,
            "cache_size": len(self.cache),
        }


# Usage
ai = OptimizedAI()
response = ai.query("What is AI commoditization?")
print(ai.status())
```

---

## 10. Emerging Tools to Watch

### 10.1 New Entrants (2026)

| Tool | Category | Promise | Status |
|------|----------|---------|--------|
| **Unsloth** | Fine-tuning | 2x faster, 60% less memory | Production |
| **Axolotl** | Fine-tuning | Easy fine-tuning | Production |
| **Guardrails AI** | Safety | Output validation | Production |
| **Lakera** | Safety | Prompt injection defense | Production |
| **Instructor** | Structured output | Type-safe LLM responses | Production |
| **Marvin** | AI functions | Natural language functions | Beta |
| **DSPy** | Prompt optimization | Automatic prompt engineering | Research |

### 10.2 Technology Trends

| Trend | Impact | Timeline |
|-------|--------|----------|
| **Speculative decoding** | 2–3x faster inference | Now |
| **Continuous batching** | 30–50% better utilization | Now |
| **Mixture of Experts** | 3–5x cheaper at same quality | Now |
| **Quantization-aware training** | Better INT4 quality | 2026–2027 |
| **Edge inference** | Run models on phones | 2026–2027 |
| **Neuromorphic chips** | 100x energy efficiency | 2027–2028 |

---

## Key Tool Selection Principles

1. **Start with APIs, migrate to self-hosting.** Begin with Groq/OpenAI for prototyping; move to self-hosted vLLM when volume exceeds 500M tokens/month.

2. **LiteLLM is the Swiss Army knife.** It provides a unified interface to 100+ providers with built-in routing and cost tracking.

3. **AWQ is the best quantization for production.** It offers the best quality/cost tradeoff with minimal engineering effort.

4. **Langfuse for observability.** Self-hosted, free, and comprehensive — the default choice for cost-conscious teams.

5. **Monitor costs from day one.** The cost tracking and alerting tools are free and prevent budget surprises.

---

*See also: [05-Future-Outlook.md](05-Future-Outlook.md) for emerging trends and [01-Overview.md](01-Overview.md) for strategic context.*

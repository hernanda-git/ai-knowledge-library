# 06 — AI Model Procurement and Gateway

> **Model procurement strategy: evaluating models across providers (capabilities, pricing, latency, data handling), API gateway for multi-model routing (LiteLLM, Portkey, OpenRouter), fallback strategies, load balancing across providers. Cost optimization: spot instances for inference, reserved capacity, commit discounts. Model arbitration (which model for which task).**

---

## Table of Contents

1. [Introduction to AI Model Procurement](#1-introduction-to-ai-model-procurement)
2. [Model Evaluation Framework](#2-model-evaluation-framework)
   - [2.1 Capability Assessment](#21-capability-assessment)
   - [2.2 Pricing Comparison](#22-pricing-comparison)
   - [2.3 Latency & Throughput](#23-latency--throughput)
   - [2.4 Data Handling & Privacy](#24-data-handling--privacy)
   - [2.5 Vendor Lock-In Risk](#25-vendor-lock-in-risk)
3. [Pricing Model Deep Dive](#3-pricing-model-deep-dive)
   - [3.1 On-Demand Pricing](#31-on-demand-pricing)
   - [3.2 Provisioned Throughput / Reserved Capacity](#32-provisioned-throughput--reserved-capacity)
   - [3.3 Batch Inference](#33-batch-inference)
   - [3.4 Spot / Preemptible Inference](#34-spot--preemptible-inference)
4. [API Gateway Solutions](#4-api-gateway-solutions)
   - [4.1 LiteLLM](#41-litellm)
   - [4.2 Portkey](#42-portkey)
   - [4.3 OpenRouter](#43-openrouter)
   - [4.4 Custom Gateway Architecture](#44-custom-gateway-architecture)
   - [4.5 Feature Comparison Matrix](#45-feature-comparison-matrix)
5. [Fallback & Failover Strategies](#5-fallback--failover-strategies)
   - [5.1 Passive Failover](#51-passive-failover)
   - [5.2 Active Failover (Load Balancing)](#52-active-failover-load-balancing)
   - [5.3 Cascading Fallback](#53-cascading-fallback)
   - [5.4 Degradation Strategy](#54-degradation-strategy)
6. [Load Balancing Across Providers](#6-load-balancing-across-providers)
   - [6.1 Latency-Based Routing](#61-latency-based-routing)
   - [6.2 Cost-Based Routing](#62-cost-based-routing)
   - [6.3 Quality-Based Routing](#63-quality-based-routing)
   - [6.4 Hybrid Routing Strategies](#64-hybrid-routing-strategies)
7. [Cost Optimization Strategies](#7-cost-optimization-strategies)
   - [7.1 Commitment Discounts](#71-commitment-discounts)
   - [7.2 Model Tiering](#72-model-tiering)
   - [7.3 Semantic Caching](#73-semantic-caching)
   - [7.4 Prompt Optimization](#74-prompt-optimization)
   - [7.5 Spot Instance Inference](#75-spot-instance-inference)
8. [Model Arbitration — Which Model for Which Task?](#8-model-arbitration--which-model-for-which-task)
   - [8.1 Task-Model Mapping Matrix](#81-task-model-mapping-matrix)
   - [8.2 Complexity-Based Routing](#82-complexity-based-routing)
   - [8.3 Multi-Model Orchestration Patterns](#83-multi-model-orchestration-patterns)
9. [Model Procurement Process](#9-model-procurement-process)
   - [9.1 Request for Information (RFI)](#91-request-for-information-rfi)
   - [9.2 Evaluation & Benchmarking](#92-evaluation--benchmarking)
   - [9.3 Commercial Negotiation](#93-commercial-negotiation)
   - [9.4 Ongoing Governance](#94-ongoing-governance)
10. [Implementation Guide](#10-implementation-guide)
    - [10.1 LiteLLM Deployment on Kubernetes](#101-litellm-deployment-on-kubernetes)
    - [10.2 Portkey Configuration](#102-portkey-configuration)
    - [10.3 Custom Gateway in Python](#103-custom-gateway-in-python)
11. [Monitoring & Observability for Model Procurement](#11-monitoring--observability-for-model-procurement)
12. [References](#12-references)

---

## 1. Introduction to AI Model Procurement

AI model procurement in 2026 is fundamentally different from traditional software procurement. Models are not static products — they are continuously updated, priced dynamically, and available through multiple competing channels. An effective procurement strategy must account for:

- **Multi-channel availability:** The same model (e.g., Claude 4 Opus) may be available through AWS Bedrock, GCP Vertex AI, Anthropic direct API, and third-party resellers — each with different pricing, SLAs, and data handling.
- **Rapid evolution:** Model generations improve every 3–6 months. Procurement must be agile enough to switch as better options emerge.
- **Complex pricing:** Per-token pricing, reserved capacity, batch discounts, and usage tiers create complex cost tradeoffs.
- **Data governance constraints:** Not all model providers offer the same data residency, encryption, and compliance guarantees.

This document provides a framework for evaluating models, selecting the right procurement channel, and operating a multi-provider AI gateway.

---

## 2. Model Evaluation Framework

### 2.1 Capability Assessment

When evaluating a model for a specific workload, consider:

**Task-Specific Benchmarks:**
| Benchmark | Measures | Relevant For |
|---|---|---|
| MMLU-Pro | Knowledge & reasoning (57 subjects) | General-purpose chatbots |
| HumanEval | Code generation accuracy | Code assistants |
| MATH-500 | Mathematical reasoning | STEM applications |
| GPQA | Graduate-level science reasoning | Research analytics |
| SimpleQA | Factual accuracy | Enterprise Q&A |
| Chatbot Arena Elo | Human preference ranking | Customer-facing chatbots |
| RAG-HIT | Retrieval augmented generation | Document Q&A |
| BLEU / ROUGE | Translation & summarization | Content generation |
| HELM (Lite) | Holistic evaluation | Comparing models across metrics |

**Capability Dimensions:**
1. **Reasoning depth:** Can the model handle multi-step, complex reasoning tasks? (e.g., o3, Claude 4 Opus, Gemini 2.0 Pro)
2. **Context window:** How many tokens can the model process in a single prompt? (Gemini leads at 1M+; GPT-4o at 128K; Claude 4 at 200K)
3. **Multimodal support:** Can the model process images, video, audio? (Gemini 2.0, GPT-4o, Claude 4)
4. **Tool use / function calling:** Can the model call external APIs? (all frontier models support this)
5. **Code generation:** How accurate is code output in target languages? (Claude 4 and GPT-4o lead)
6. **Multilingual capability:** How many languages does the model support? (GPT-4o: 100+, Gemini: 100+)
7. **Instruction following:** How consistently does the model follow complex instructions?
8. **Steerability:** Can the model be guided with system prompts, temperature, and top-p settings?

### 2.2 Pricing Comparison

**Standard Pricing Per 1M Tokens (approximate 2026):**

| Model | Provider | Input Cost | Output Cost | 1M Input Context Extra |
|---|---|---|---|---|
| GPT-4o | Azure OpenAI | $2.50 | $10.00 | $7.50/1M additional tokens |
| GPT-4o-mini | Azure OpenAI | $0.15 | $0.60 | — |
| o3 | Azure OpenAI | $10.00 | $40.00 | — |
| o4-mini | Azure OpenAI | $1.10 | $4.40 | — |
| Claude 4 Opus | AWS Bedrock | $15.00 | $75.00 | — |
| Claude 4 Sonnet | AWS Bedrock | $3.00 | $15.00 | — |
| Claude 4 Haiku | AWS Bedrock | $0.25 | $1.25 | — |
| Gemini 2.0 Pro | GCP Vertex AI | $2.50 | $10.00 | — |
| Gemini 2.0 Flash | GCP Vertex AI | $0.35 | $1.40 | — |
| Gemini 1.5 Flash | GCP Vertex AI | $0.075 | $0.30 | — |
| Llama 4 70B | Bedrock/Vertex | $0.55 | $2.10 | — |
| Llama 4 8B | Bedrock/Vertex | $0.10 | $0.40 | — |
| Mistral Large 2 | Bedrock/Vertex | $2.00 | $6.00 | — |
| Titan Premier | AWS Bedrock | $2.50 | $10.00 | — |
| text-embedding-3-large | Azure OpenAI | $0.13 | — | — |
| Titan Embeddings | AWS Bedrock | $0.02/1K | — | — |

**Effective Cost Comparison for a Typical RAG Query:**

Assume: 2,000 input tokens (user query + 3 retrieved chunks), 500 output tokens.

| Model | Input Cost | Output Cost | Total Cost per Query |
|---|---|---|---|
| GPT-4o-mini | $0.0000003 | $0.0000003 | **$0.0000006** |
| Gemini 2.0 Flash | $0.0000007 | $0.0000007 | **$0.0000014** |
| GPT-4o | $0.000005 | $0.000005 | **$0.00001** |
| Gemini 2.0 Pro | $0.000005 | $0.000005 | **$0.00001** |
| Claude 4 Sonnet | $0.000006 | $0.0000075 | **$0.0000135** |
| Claude 4 Opus | $0.00003 | $0.0000375 | **$0.0000675** |

**Key Insight:** Cost difference between cheapest and most expensive model for a single query is ~100x. For 10 million queries/month, that is $6 vs $675.

### 2.3 Latency & Throughput

**Time to First Token (TTFT) — P50 (approximate):**

| Model | Provider | Short Context (<500 tokens) | Long Context (10K+ tokens) |
|---|---|---|---|
| GPT-4o-mini | Azure OpenAI | 200ms | 500ms |
| Gemini 2.0 Flash | GCP Vertex AI | 250ms | 600ms |
| GPT-4o | Azure OpenAI | 400ms | 1200ms |
| Claude 4 Haiku | AWS Bedrock | 300ms | 700ms |
| Claude 4 Sonnet | AWS Bedrock | 600ms | 1500ms |
| Gemini 2.0 Pro | GCP Vertex AI | 500ms | 2000ms |
| Claude 4 Opus | AWS Bedrock | 1000ms | 3000ms |
| o4-mini | Azure OpenAI | 800ms | 2500ms |

**Throughput (output tokens per second):**

| Model | Provider | P50 Throughput |
|---|---|---|
| GPT-4o-mini | Azure OpenAI | 200 tokens/sec |
| Gemini 2.0 Flash | GCP Vertex AI | 150 tokens/sec |
| Claude 4 Haiku | AWS Bedrock | 100 tokens/sec |
| GPT-4o | Azure OpenAI | 60 tokens/sec |
| Claude 4 Sonnet | AWS Bedrock | 45 tokens/sec |
| Gemini 2.0 Pro | GCP Vertex AI | 40 tokens/sec |
| Claude 4 Opus | AWS Bedrock | 20 tokens/sec |

**Latency Budgeting:**
- Real-time chat: Target <2s total response time.
- Streaming use: First token <500ms, then steady stream.
- Batch processing: Latency less important; optimize for throughput/cost.

### 2.4 Data Handling & Privacy

| Model | Provider Compliance | Data Used for Training | Encryption Options | Data Residency |
|---|---|---|---|---|
| GPT-4o | SOC 2, HIPAA, FedRAMP | No | CMK available | 30+ regions; data-resident SKU |
| Claude 4 | SOC 2, HIPAA, ISO 27001 | No | KMS encryption | Via cloud provider region |
| Gemini 2.0 | SOC 2, HIPAA, FedRAMP | No (with CMEK) | CMEK available | Via GCP region |
| Llama 4 | Self-managed | N/A (self-host) | Full control | Anywhere |

**Data Handling Assessment Checklist:**
1. Does the provider train on customer data? (Must be "No" for enterprise.)
2. Is specific opt-out needed for abuse monitoring? (Azure → yes, AWS → yes, GCP → yes.)
3. Is CMK/Customer-managed encryption available?
4. Can data residency be guaranteed at the contract level?
5. Is there a BAA/HIPAA agreement available?
6. Is FedRAMP authorization available (where required)?
7. Are data deletion processes documented and audited?

### 2.5 Vendor Lock-In Risk

**Assessing Lock-In for Each Model Channel:**

| Factor | Low Risk | High Risk |
|---|---|---|
| **Model format** | Open-weight (Llama, Mistral) | Proprietary (GPT-4, Claude 4, Gemini) |
| **API compatibility** | OpenAI-compatible API | Proprietary API |
| **Provider dependence** | Available through multiple clouds | Exclusive to one cloud |
| **Fine-tuning portability** | Model is standard (Hugging Face format) | Custom fine-tuning format |
| **Prompt format** | Standard chat template | Custom system prompt format |

**Mitigation Strategies:**
- Prefer models available through multiple cloud providers (Claude 4 on AWS + GCP; Llama 4 on all three).
- Use an abstraction layer (LiteLLM) to normalize API differences.
- Maintain compatibility with multiple model families for critical workloads.
- Keep prompt templates model-agnostic where possible.
- Regularly benchmark alternatives to maintain optionality.

---

## 3. Pricing Model Deep Dive

### 3.1 On-Demand Pricing

**Characteristics:**
- Pay per token (input and output).
- No upfront commitment.
- Ideal for: Variable workloads, prototyping, low volume, overflow.

**Pricing Structure:**
```
Total Cost = (Input_Tokens × Input_Price) + (Output_Tokens × Output_Price)

Example: GPT-4o, 10K input + 2K output
  = (10,000 × $2.50/1M) + (2,000 × $10.00/1M)
  = $0.025 + $0.02
  = $0.045 per request
```

**When to use on-demand:**
- Workloads with volatile or unpredictable traffic.
- < 50M tokens/month per model (break-even with PTU varies).
- Prototyping and experimentation.
- Overflow traffic during bursts.

### 3.2 Provisioned Throughput / Reserved Capacity

**Azure Provisioned Throughput Units (PTUs):**
- 1 PTU of GPT-4o ≈ 30,000 input tokens/min throughput.
- $8,500/month (1-month commitment) or $6,000/month (12-month commitment).
- Discount: ~20% (1-month) to ~40% (12-month) vs. on-demand at full utilization.

**AWS Provisioned Throughput:**
- Model units with specified throughput (varies by model).
- 1-month or 6-month commitment.
- Discount: ~20% (1-month) to ~35% (6-month).

**GCP Vertex AI Reserved Resources:**
- Reserve GPU/TPU instances for training and inference.
- 1-year or 3-year commitment.
- Discount: up to 70% for 3-year commitments.

**Break-Even Calculation:**
```
Monthly On-Demand Cost = (Monthly_Tokens × Price) / 1,000,000
Monthly PTU Cost = PTU_Units × Monthly_PTU_Price

Break-even: Monthly_Tokens > (PTU_Units × Monthly_PTU_Price × 1,000,000) / Price
```

**Example:** GPT-4o on Azure
- 1 PTU = 30K tokens/min = 1,296M tokens/month (assuming 100% utilization).
- On-demand at full utilization: 1,296M × $2.50/1M = $3,240 (input only).
- PTU cost: $8,500/month (1-month).
- **Break-even utilization:** $8,500 / $3,240 = 262% — PTU only makes sense if you consistently exceed 62% of max throughput.
- Actual practical break-even: ~50M–100M tokens/month for GPT-4o.

### 3.3 Batch Inference

Batch inference offers 50% discount for asynchronous, non-real-time processing:

| Provider | Batch Discount | Turnaround SLA |
|---|---|---|
| Azure OpenAI | 50% off on-demand | Within 24 hours |
| AWS Bedrock | 50% off on-demand | Best effort (usually <1 hour) |
| GCP Vertex AI | No separate batch pricing | — |

**Best for:**
- Bulk document processing.
- Data enrichment pipelines.
- Model evaluation.
- Backfill and migration tasks.

### 3.4 Spot / Preemptible Inference

For custom models (not managed APIs), spot/preemptible instances offer significant savings:

| Provider | Instance Type | Discount | Interruption Risk |
|---|---|---|---|
| AWS | Spot instances (SageMaker) | 60–70% | Moderate (2-min warning) |
| Azure | Low-priority VMs | 60–80% | High (30-sec warning) |
| GCP | Preemptible VMs | 60–80% | High (24-hour max lifetime) |

**Architecture for Spot Inference:**

```yaml
# Kubernetes deployment for spot/preemptible inference
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spot-model-server
spec:
  replicas: 10
  template:
    spec:
      # Use spot nodes for cost savings
      nodeSelector:
        lifecycle: spot
      containers:
      - name: model
        image: vllm/vllm-openai:latest
        # Add graceful shutdown handler
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 30 && drain_requests"]
---
# Regular (on-demand) backup for reliability
apiVersion: apps/v1
kind: Deployment
metadata:
  name: on-demand-backup
spec:
  replicas: 2  # Small number for reliability
  template:
    spec:
      nodeSelector:
        lifecycle: on-demand
```

---

## 4. API Gateway Solutions

### 4.1 LiteLLM

LiteLLM is an open-source Python library that provides a unified interface for 100+ LLM providers.

**Key Features:**
- OpenAI-compatible API for all supported providers.
- Load balancing across multiple deployments of the same model.
- Fallback chains (try provider A, then B, then C).
- Spending tracking and budget limits per key/user.
- Rate limiting and retry logic built in.
- Proxy mode: Deploy as a standalone server (LiteLLM Proxy).

**Configuration Example (Proxy Mode):**

```yaml
general_settings:
  master_key: sk-your-master-key
  database_url: postgresql://user:pass@host:5432/litellm

model_list:
  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4o
      api_base: https://my-openai.openai.azure.com
      api_key: os.environ/AZURE_OPENAI_KEY
      rpm: 4500  # Rate limit per minute
    
  - model_name: gpt-4
    litellm_params:
      model: gcp/gemini-1.5-pro
      vertex_project: my-project
      vertex_location: us-central1
    # Fallback for gpt-4 requests
    
  - model_name: claude-3.5
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20240620
      aws_region_name: us-east-1
    model_info:
      mode: completion

router_settings:
  routing_strategy: "latency-based"  # or "cost-based" or "usage-based"
  allowed_fails: 3
  num_retries: 2
  fallback_strategy: "next-provider"
  cooldown: 60  # seconds before retrying failed provider
```

### 4.2 Portkey

Portkey is a managed AI gateway with observability and cost control features.

**Key Features:**
- Request/response logging and tracing.
- Cost tracking per user, model, application.
- Cache management (semantic caching).
- Fallbacks and retries.
- A/B testing between models.
- Prompt management and versioning.
- User rate limiting and throttling.

**Configuration Example:**

```python
import portkey

# Configure Portkey gateway
portkey.Config(
    api_key="pk-...",
    mode="proxy",
    config={
        "strategy": {
            "mode": "fallback",
            "primary": "azure-gpt-4o",
            "fallbacks": [
                "bedrock-claude-4-sonnet",
                "vertex-gemini-1.5-pro"
            ]
        },
        "cache": {
            "mode": "semantic",
            "similarity_threshold": 0.95
        },
        "retry": {
            "attempts": 3,
            "backoff": "exponential"
        }
    }
)

# Use as drop-in replacement for OpenAI client
response = portkey.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    metadata={"user": "user_123", "application": "chat"}
)
```

### 4.3 OpenRouter

OpenRouter is a managed service that aggregates multiple model providers.

**Key Features:**
- Single API key for 100+ models.
- Automatic fallback and load balancing.
- Free credits for experimentation.
- Model comparison and benchmarking.
- Simple OpenAI-compatible API.

**Usage:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-...",
)

response = client.chat.completions.create(
    model="openai/gpt-4o",  # or "anthropic/claude-4-opus", "google/gemini-2.0-pro"
    messages=[{"role": "user", "content": "Hello"}]
)
```

**Limitations:**
- Third-party intermediary — data passes through OpenRouter.
- Limited control over model deployment configuration.
- Higher latency due to extra hop.
- Not suitable for data-sensitive enterprise workloads.

### 4.4 Custom Gateway Architecture

For organizations needing maximum control, a custom gateway can be built:

```python
# Simplified custom AI gateway
import asyncio
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProviderConfig:
    name: str
    endpoint: str
    api_key: str
    model: str
    weight: int = 100
    max_retries: int = 3
    timeout_ms: int = 30000
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0

class AIGateway:
    def __init__(self, providers: list[ProviderConfig]):
        self.providers = providers
        self.failure_counts = {p.name: 0 for p in providers}
        self.cooldowns = {p.name: 0 for p in providers}
    
    async def route_request(self, request, strategy="cost"):
        """Route request to the best provider based on strategy."""
        if strategy == "cost":
            candidates = sorted(
                self.providers, 
                key=lambda p: p.cost_per_1m_input
            )
        elif strategy == "latency":
            candidates = sorted(
                self.providers,
                key=lambda p: self._estimate_latency(p)
            )
        else:  # weighted random
            total_weight = sum(p.weight for p in self.providers)
            r = random.uniform(0, total_weight)
            cumulative = 0
            candidates = []
            for p in self.providers:
                cumulative += p.weight
                if r <= cumulative:
                    candidates = [p]
                    break
        
        # Try providers in order with fallback
        for provider in candidates:
            if self.cooldowns.get(provider.name, 0) > time.time():
                continue
            
            try:
                response = await self._call_provider(provider, request)
                self.failure_counts[provider.name] = 0
                return response
            except Exception as e:
                self.failure_counts[provider.name] += 1
                if self.failure_counts[provider.name] >= 3:
                    self.cooldowns[provider.name] = time.time() + 60
        
        raise Exception("All providers failed")
    
    async def _call_provider(self, provider, request):
        """Make API call to specific provider."""
        # Provider-specific implementation
        ...
```

### 4.5 Feature Comparison Matrix

| Feature | LiteLLM | Portkey | OpenRouter | Custom |
|---|---|---|---|---|
| **Open source** | ✓ | Partial | ✗ | ✓ |
| **Self-hosted** | ✓ | Enterprise only | ✗ | ✓ |
| **Provider count** | 100+ | 50+ | 100+ | Any |
| **Load balancing** | ✓ | ✓ | ✓ | Custom |
| **Fallback chains** | ✓ | ✓ | ✓ | Custom |
| **Cost tracking** | ✓ | ✓ | ✓ | Custom |
| **Semantic caching** | Partial | ✓ | ✗ | Custom |
| **Rate limiting** | ✓ | ✓ | ✓ | Custom |
| **A/B testing** | Manual | ✓ | ✗ | Custom |
| **Prompt management** | ✗ | ✓ | ✗ | Custom |
| **User management** | ✓ | ✓ | ✗ | Custom |
| **On-prem deployment** | ✓ | Enterprise | ✗ | ✓ |
| **Cost** | Free | Paid | Paid (usage %) | Dev time |

---

## 5. Fallback & Failover Strategies

### 5.1 Passive Failover (Cold Standby)

```
Primary: Azure OpenAI (GPT-4o)
  ↓ (on failure)
Secondary: AWS Bedrock (Claude 4 Sonnet)
  ↓ (on failure)
Tertiary: GCP Vertex (Gemini 2.0 Pro)
```

**Implementation:**
```python
providers = ["azure/gpt-4o", "bedrock/claude-4-sonnet", "vertex/gemini-2.0-pro"]

for provider in providers:
    try:
        response = gateway.call(provider, request)
        return response
    except (RateLimitError, ServiceUnavailableError) as e:
        log.warning(f"Provider {provider} failed: {e}. Trying next.")
        continue

raise AllProvidersExhaustedError("All AI providers failed")
```

### 5.2 Active Failover (Load Balanced)

```
Traffic split:
  - 70% → Azure OpenAI (GPT-4o) [Primary]
  - 20% → AWS Bedrock (Claude 4 Sonnet) [Active secondary]
  - 10% → GCP Vertex (Gemini 2.0 Pro) [Active tertiary]
  
On Azure failure:
  - 80% → AWS Bedrock
  - 20% → GCP Vertex
```

**Implementation with LiteLLM:**
```yaml
router_settings:
  routing_strategy: "usage-based"
  model_group_alias:
    gpt-4:
      - model: azure/gpt-4o
        weight: 70
      - model: bedrock/claude-4-sonnet
        weight: 20
      - model: vertex/gemini-2.0-pro
        weight: 10
  cooldown: 300  # 5 min cooldown after failure
```

### 5.3 Cascading Fallback

```python
def generate_with_fallback(request, quality_threshold=0.8):
    """Try progressively cheaper/more available models."""
    
    # Tier 1: Best quality
    response = try_provider("azure/o3", request)
    if assess_quality(response) >= quality_threshold:
        return response
    
    # Tier 2: Good quality, more available
    response = try_provider("azure/gpt-4o", request)
    if assess_quality(response) >= quality_threshold:
        return response
    
    # Tier 3: Decent quality, widely available
    response = try_provider("bedrock/claude-4-sonnet", request)
    if assess_quality(response) >= quality_threshold:
        return response
    
    # Tier 4: Always available (smallest model)
    return try_provider("bedrock/claude-4-haiku", request)
```

### 5.4 Degradation Strategy

When all primary models fail:

1. **Degrade model quality:** Switch from GPT-4o → GPT-4o-mini or Claude 4 Opus → Haiku.
2. **Degrade response length:** Reduce max_tokens to minimize cost/latency.
3. **Switch to deterministic:** If generative AI unavailable, fall back to rule-based or retrieval-only responses.
4. **Queue requests:** For batch processing, queue requests and retry when services recover.
5. **Human escalation:** For critical use cases, flag requests for manual handling.

---

## 6. Load Balancing Across Providers

### 6.1 Latency-Based Routing

```
For each user request:
  1. Ping each provider (or use historical latency data).
  2. Route to the provider with lowest current latency.
  3. Exclude providers with latency > threshold (e.g., 5s).
  4. Rebalance if latency degrades mid-stream.
```

**Pros:** Best user experience.
**Cons:** Can route to expensive providers; may not use reserved capacity efficiently.

### 6.2 Cost-Based Routing

```
For each user request:
  1. Calculate expected cost for each provider (based on token count).
  2. Route to the cheapest eligible provider.
  3. Eligible = meets quality threshold, not in cooldown.
```

**Pros:** Minimizes inference cost.
**Cons:** May send simple queries to expensive "cheapest" provider if not properly tiered.

### 6.3 Quality-Based Routing

```
For each user request:
  1. Classify request complexity (simple/medium/complex).
  2. Route simple → cheapest model (GPT-4o-mini, Claude Haiku).
  3. Route medium → balanced model (GPT-4o, Claude Sonnet).
  4. Route complex → best model (Claude 4 Opus, o3, Gemini Pro).
```

**Pros:** Optimizes cost-to-quality ratio.
**Cons:** Requires complexity classification; quality assessment overhead.

### 6.4 Hybrid Routing Strategies

**Weighted Multi-Factor Routing:**
```python
def score_provider(provider, request):
    cost_score = normalize(-provider.expected_cost, 0, 1)
    latency_score = normalize(-provider.expected_latency, 0, 1) 
    quality_score = provider.quality_score(request.complexity)
    utilization_score = 1 - provider.current_utilization
    
    score = (
        0.3 * cost_score +
        0.2 * latency_score +
        0.3 * quality_score +
        0.2 * utilization_score
    )
    return score

# Route to highest-scored provider
best_provider = max(providers, key=lambda p: score_provider(p, request))
```

---

## 7. Cost Optimization Strategies

### 7.1 Commitment Discounts

| Provider | Commitment Type | Discount | Minimum Term |
|---|---|---|---|
| Azure OpenAI | PTU (1-month) | 20% | 1 month |
| Azure OpenAI | PTU (12-month) | 40% | 12 months |
| AWS Bedrock | Provisioned Throughput (1-month) | 20% | 1 month |
| AWS Bedrock | Provisioned Throughput (6-month) | 35% | 6 months |
| AWS SageMaker | Savings Plans | Up to 44% | 1–3 years |
| GCP Vertex AI | Committed Use (CUD) | Up to 70% | 1–3 years |

**Strategy:**
1. Identify the baseline workload (minimum tokens/month you will always use).
2. Cover baseline with long-term commitments (12-month or 3-year).
3. Use short-term commitments for seasonal peaks.
4. Use on-demand for overflow, variability, and experimentation.

### 7.2 Model Tiering

Implement a tiered model architecture:

```python
MODEL_TIERS = {
    "tier1": {  # Best quality, highest cost
        "complex_reasoning": "azure/o3",
        "complex_coding": "bedrock/claude-4-opus", 
        "complex_analysis": "vertex/gemini-2.0-pro",
    },
    "tier2": {  # Good quality, moderate cost
        "general_chat": "azure/gpt-4o",
        "document_qa": "bedrock/claude-4-sonnet",
        "content_gen": "vertex/gemini-1.5-pro",
    },
    "tier3": {  # Decent quality, low cost
        "simple_qa": "azure/gpt-4o-mini",
        "classification": "bedrock/claude-4-haiku",
        "summarization": "vertex/gemini-1.5-flash",
    },
    "tier4": {  # Good enough, lowest cost
        "bulk_embedding": "azure/text-embedding-3-small",
        "batch_tagging": "bedrock/titan-text-lite",
        "bulk_translation": "vertex/gemini-1.5-flash",
    }
}

def route_to_tier(request):
    complexity = classify_complexity(request)
    
    if complexity == "simple":
        target = MODEL_TIERS["tier3"]
    elif complexity == "medium":
        target = MODEL_TIERS["tier2"]
    else:  # complex
        target = MODEL_TIERS["tier1"]
    
    # Select provider based on task type
    provider = target.get(request.task_type, target["general_chat"])
    return call_provider(provider, request)
```

### 7.3 Semantic Caching

Cache semantically similar queries to avoid redundant API calls:

```python
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self, threshold=0.95, max_size=10000):
        self.threshold = threshold
        self.max_size = max_size
        self.cache = {}  # embedding_hash -> (response, embedding)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    def get(self, query):
        query_emb = self.encoder.encode(query)
        
        # Check all cached entries for similarity
        for cached_query, (response, cached_emb) in self.cache.items():
            similarity = np.dot(query_emb, cached_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(cached_emb)
            )
            if similarity >= self.threshold:
                return response
        
        return None
    
    def set(self, query, response):
        if len(self.cache) < self.max_size:
            self.cache[query] = (response, self.encoder.encode(query))
```

**Cache Hit Rate:** Typically 20–40% for customer-facing chatbots, 50–70% for internal knowledge base Q&A.

**Cost Savings:** 20–70% reduction in API costs depending on query diversity.

### 7.4 Prompt Optimization

Strategies to reduce token consumption:

1. **Trim conversation history:** Summarize older messages rather than including them as-is.
2. **Use structured output:** Request concise, structured responses (JSON, bullet points).
3. **Remove redundant context:** Only include relevant knowledge base chunks.
4. **Optimize system prompts:** Shorter, more specific system prompts use fewer tokens.
5. **Use response compression:** Set lower max_tokens for simpler tasks.
6. **Batch similar requests:** Combine multiple simple queries into one API call.

**Token Reduction Techniques:**
| Technique | Typical Savings | Impact on Quality |
|---|---|---|
| History summarization | 30–60% | Minor |
| Relevant chunk selection | 40–60% | Moderate (if good retriever) |
| Concise system prompt | 10–30% | Minor to none |
| Dynamic max_tokens | 20–50% | Only for deterministic tasks |
| Input compression (lossy) | 30–50% | Variable (test first) |

### 7.5 Spot Instance Inference

For self-hosted models on cloud infrastructure:

```yaml
# AWS: SageMaker Spot Training
TrainingJob:
  ResourceConfig:
    InstanceType: ml.p4d.24xlarge
    InstanceCount: 4
    SpotPercentage: 100  # Up to 70% savings
  StoppingCondition:
    MaxRuntimeInSeconds: 86400
  CheckpointConfig:
    S3Uri: s3://my-bucket/checkpoints/
    LocalPath: /opt/ml/checkpoints/

# GCP: Preemptible TPU for training
TrainingJob:
  WorkerPoolSpecs:
    - MachineSpec:
        AcceleratorType: TPU_V5P
        AcceleratorCount: 4
      Preemptible: True  # Up to 80% savings
```

---

## 8. Model Arbitration — Which Model for Which Task?

### 8.1 Task-Model Mapping Matrix

| Task Category | Best Model(s) | Good Alternative | Budget Option |
|---|---|---|---|
| **Complex reasoning** | o3, Claude 4 Opus | GPT-4o, Gemini 2.0 Pro | Claude 4 Sonnet |
| **Code generation** | Claude 4 Opus, GPT-4o | Gemini 2.0 Pro | Claude 4 Sonnet |
| **Creative writing** | GPT-4o, Claude 4 Sonnet | Gemini 2.0 Pro | GPT-4o-mini |
| **Document analysis** | Claude 4 Opus (200K ctx) | Gemini 2.0 Pro (1M ctx) | Claude 4 Sonnet |
| **Customer support** | GPT-4o-mini (fast+cheap) | Claude 4 Haiku | Gemini 1.5 Flash |
| **Image understanding** | Gemini 2.0 Pro, GPT-4o | Claude 4 Opus | GPT-4o-mini |
| **Video analysis** | Gemini 2.0 Pro | GPT-4o (frames) | — |
| **Audio understanding** | GPT-4o (native audio) | Gemini 2.0 Pro | — |
| **Embeddings** | text-embedding-3-large | Cohere Embed v3 | text-embedding-3-small |
| **Classification** | GPT-4o-mini, Claude Haiku | Titan Text Lite | Gemini 1.5 Flash |
| **Summarization** | Claude 4 Sonnet, GPT-4o | Gemini 1.5 Pro | GPT-4o-mini |
| **Translation** | Gemini 2.0 Pro, GPT-4o | Google Translate API | GPT-4o-mini |
| **Extraction (structured)** | GPT-4o, Claude 4 Sonnet | Gemini 1.5 Pro | GPT-4o-mini |
| **Sentiment analysis** | GPT-4o-mini, Claude Haiku | Titan Text Lite | Comprehend API |

### 8.2 Complexity-Based Routing

```python
def classify_complexity(request) -> str:
    """Classify request complexity for model routing."""
    
    # Simple rules-based classification
    simple_patterns = [
        r"what is|who is|where is",      # Simple Q&A
        r"summarize|summarize this",      # Summarization
        r"translate (to|from)",           # Translation
        r"(yes|no) question",            # Binary questions
        r"define|meaning of",            # Definitions
    ]
    
    complex_patterns = [
        r"write code|implement|function", # Code generation
        r"analyze|compare|contrast",     # Analysis
        r"multi-step|chain of thought",  # Reasoning
        r"debug|find bug|fix error",     # Debugging
        r"design architecture",          # Architecture
    ]
    
    text = request.get("messages", [{}])[-1].get("content", "")
    
    # Check for complex patterns first
    for pattern in complex_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "complex"
    
    # Then check for simple patterns
    for pattern in simple_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "simple"
    
    # Default to medium complexity
    return "medium"
```

### 8.3 Multi-Model Orchestration Patterns

**Pattern 1: Classifier + Generator**
```
User Input → Classifier (small LLM: GPT-4o-mini)
                │
    ┌───────────┼───────────┐
    │           │           │
  Code Q&A   Document   General Chat
  Use Claude  Use Gemini  Use GPT-4o
```

**Pattern 2: Generator + Evaluator**
```
Generator (Claude 4 Opus) → Evaluator (GPT-4o-mini)
                                │
                          Quality Check:
                          - Score > 0.8 → Return
                          - Score < 0.8 → Regenerate
```

**Pattern 3: Parallel Voting**
```
User Input → GPT-4o → Answer A
User Input → Claude 4 → Answer B  
User Input → Gemini → Answer C
                  │
           Voting/Ensemble
           (Select best answer
            via GPT-4o-mini judge)
```

---

## 9. Model Procurement Process

### 9.1 Request for Information (RFI)

**Template for Model Provider RFI:**

```
1. Model Specifications
   - Model name, version, architecture
   - Context window (tokens)
   - Supported input/output modalities
   - Training data cutoff date
   - Languages supported

2. Performance
   - Benchmarks (MMLU, HumanEval, MATH, GPQA)
   - Latency (TTFT, tokens/sec) at various load levels
   - Throughput limits (requests per minute per deployment)

3. Pricing
   - Input/output token pricing
   - Reserved capacity pricing and terms
   - Batch pricing
   - Volume discount tiers

4. Compliance
   - SOC 2 Type II report availability
   - HIPAA BAA availability
   - FedRAMP authorization level
   - GDPR data processing agreement
   - Data residency options
   - Data used for training? (Yes/No)

5. Security
   - Encryption at rest (BYOK/CMK support)
   - Encryption in transit
   - Private network access (VPC/Private Link)
   - Audit logging capabilities

6. SLA
   - Uptime guarantee
   - Latency guarantee
   - Support tiers and response times
   - Credits for SLA breaches
```

### 9.2 Evaluation & Benchmarking

**Evaluation Process:**
1. **Define evaluation dataset:** 500–1000 domain-specific examples with expected outputs.
2. **Run blinded evaluation:** Compare models without knowing which is which.
3. **Score on multiple dimensions:** Accuracy, latency, cost, safety.
4. **Edge case testing:** Test adversarial inputs, long contexts, multi-turn conversations.
5. **Production simulation:** Run a week-long shadow mode evaluation with real traffic.

**Evaluation Metrics:**

```python
evaluation_results = {
    "model": "claude-4-opus",
    "accuracy": 0.94,
    "hallucination_rate": 0.02,
    "avg_latency_ms": 1500,
    "p95_latency_ms": 3000,
    "cost_per_1k_request": 0.045,
    "safety_failures": 1,  # out of 1000
    "data_residency_compliant": True,
    "client_approval_score": 4.2  # out of 5
}
```

### 9.3 Commercial Negotiation

**Negotiation Leverage Points:**
- **Volume commitment:** Larger committed volumes → better pricing.
- **Multi-year terms:** 2–3 year commitments → deeper discounts.
- **Multiple providers:** Having alternatives strengthens your position.
- **Reference customer:** Being named as a reference case can unlock discounts.
- **Joint GTM:** If you have a platform that can drive usage to the provider.

**Negotiation Targets:**
- Standard on-demand pricing: Baseline.
- 12-month PTU commitment: 40% off on-demand (Azure).
- 24-month commitment: 50%+ off.
- Enterprise agreement: Additional 10–20% on top of published discounts.

### 9.4 Ongoing Governance

**Model Procurement Governance Board:**
- Monthly review of model performance and cost.
- Quarterly review of new models and providers.
- Annual re-negotiation of major contracts.
- Continuous monitoring of SLA compliance.

**Key Performance Indicators (KPIs):**
- Cost per query (by model, by workload).
- Quality score (human-rated or LLM-as-judge).
- Availability (uptime, error rate).
- Latency (P50, P95, P99).
- Provider diversity (percentage of spend on single provider — target <50%).

---

## 10. Implementation Guide

### 10.1 LiteLLM Deployment on Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm-proxy
  namespace: ai-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: litellm-proxy
  template:
    metadata:
      labels:
        app: litellm-proxy
    spec:
      serviceAccountName: litellm-sa
      containers:
      - name: litellm
        image: ghcr.io/berriai/litellm:main-latest
        args: ["--port", "4000", "--config", "/app/config.yaml"]
        ports:
        - containerPort: 4000
        env:
        - name: AZURE_OPENAI_KEY
          valueFrom:
            secretKeyRef:
              name: ai-credentials
              key: azure-openai-key
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: ai-credentials
              key: aws-access-key-id
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: ai-credentials
              key: aws-secret-access-key
        - name: DATABASE_URL
          value: postgresql://litellm:pass@postgres:5432/litellm
        volumeMounts:
        - name: config
          mountPath: /app/config.yaml
          subPath: config.yaml
      volumes:
      - name: config
        configMap:
          name: litellm-config
---
apiVersion: v1
kind: Service
metadata:
  name: litellm-proxy
  namespace: ai-gateway
spec:
  selector:
    app: litellm-proxy
  ports:
  - port: 4000
    targetPort: 4000
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: litellm-proxy-hpa
  namespace: ai-gateway
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: litellm-proxy
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 10.2 Portkey Configuration

```javascript
// Portkey configuration for multi-cloud routing
const portkeyConfig = {
  provider: "portkey",
  apiKey: process.env.PORTKEY_API_KEY,
  config: {
    strategy: {
      mode: "weighted-fallback",
      targets: [
        {
          provider: "azure-openai",
          model: "gpt-4o",
          weight: 70,
          config: {
            apiKey: process.env.AZURE_OPENAI_KEY,
            baseUrl: process.env.AZURE_OPENAI_ENDPOINT
          }
        },
        {
          provider: "anthropic",
          model: "claude-4-sonnet",
          weight: 30,
          config: {
            apiKey: process.env.ANTHROPIC_API_KEY
          }
        }
      ],
      fallbackStrategy: "cascading",
      fallbackStatusCodes: [429, 500, 502, 503, 504]
    },
    cache: {
      mode: "semantic",
      threshold: 0.92,
      ttl: 3600
    },
    retry: {
      attempts: 3,
      initialDelayMs: 1000,
      maxDelayMs: 10000,
      backoffFactor: 2
    }
  }
};
```

### 10.3 Custom Gateway in Python

```python
# Gateway implementation with all major features
import asyncio
import hashlib
import json
import time
from typing import Optional, Callable
import httpx
from dataclasses import dataclass, field

@dataclass
class GatewayConfig:
    providers: dict
    cache_enabled: bool = True
    cache_similarity: float = 0.95
    max_retries: int = 3
    fallback_enabled: bool = True
    cost_tracking: bool = True
    
class AIGateway:
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=30.0)
        self.cache = {}
        self.cost_tracker = CostTracker()
        self.failure_tracker = {}
    
    async def chat_completion(self, model_alias: str, messages: list, **kwargs):
        """Main entry point for chat completions."""
        
        # Check cache
        if self.config.cache_enabled:
            cache_key = self._cache_key(messages, kwargs)
            cached = self._check_cache(cache_key)
            if cached:
                return cached
        
        # Get provider chain
        providers = self.config.providers.get(model_alias, [])
        
        for attempt in range(self.config.max_retries):
            for provider in providers:
                if self._is_cooldown(provider["name"]):
                    continue
                
                try:
                    start = time.time()
                    response = await self._call_provider(provider, messages, kwargs)
                    latency = time.time() - start
                    
                    # Track cost
                    if self.config.cost_tracking:
                        self.cost_tracker.log(
                            provider=provider["name"],
                            model=provider["model"],
                            tokens=response["usage"],
                            latency=latency
                        )
                    
                    # Cache if appropriate
                    if self.config.cache_enabled and not kwargs.get("stream", False):
                        self._set_cache(cache_key, response)
                    
                    return response
                    
                except Exception as e:
                    self.failure_tracker[provider["name"]] = {
                        "last_failure": time.time(),
                        "count": self.failure_tracker.get(provider["name"], {}).get("count", 0) + 1
                    }
            
            # Wait before retry if we exhausted all providers
            await asyncio.sleep(2 ** attempt)
        
        raise GatewayExhaustionError("All providers failed after retries")
    
    def _cache_key(self, messages, kwargs):
        content = json.dumps({"messages": messages, **kwargs}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _check_cache(self, key):
        if key in self.cache:
            entry = self.cache[key]
            if entry["expires"] > time.time():
                return entry["response"]
        return None
    
    def _set_cache(self, key, response, ttl=3600):
        self.cache[key] = {
            "response": response,
            "expires": time.time() + ttl
        }
    
    def _is_cooldown(self, provider_name):
        tracker = self.failure_tracker.get(provider_name, {})
        last_failure = tracker.get("last_failure", 0)
        count = tracker.get("count", 0)
        
        if count >= 3 and (time.time() - last_failure) < 300:  # 5 min cooldown
            return True
        return False
    
    async def _call_provider(self, provider, messages, kwargs):
        # Provider-specific implementation
        ...
```

---

## 11. Monitoring & Observability for Model Procurement

**Key Metrics to Track:**

| Metric | Source | Actionable Insight |
|---|---|---|
| Cost per 1K requests | Gateway | Which provider/model is most cost-effective |
| P50/P99 latency | Gateway | User experience; identify slow providers |
| Error rate by provider | Gateway | Identify unreliable providers |
| Token utilization | Provider metrics | Are PTUs being fully utilized? |
| Cache hit rate | Gateway | Cache effectiveness; potential savings |
| Quality score | Evaluation pipeline | Model degradation over time |
| Provider availability | Health checks | SLA compliance |
| Spend by model/team | Cost tracking | Budget allocation; cost attribution |

**Dashboard Recommendations:**
- Real-time: Latency, error rate, requests per second.
- Daily: Cost by provider, cost by model, cache hit rate.
- Weekly: Quality scores, provider performance comparison.
- Monthly: Savings from commitments, negotiation leverage points.

---

## 12. References

1. LiteLLM. (2026). *LiteLLM Documentation*. https://docs.litellm.ai/
2. Portkey. (2026). *Portkey AI Gateway Documentation*. https://docs.portkey.ai/
3. OpenRouter. (2026). *OpenRouter API Documentation*. https://openrouter.ai/docs
4. OpenAI. (2026). *Azure OpenAI Service Pricing*. https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/
5. AWS. (2026). *Amazon Bedrock Pricing*. https://aws.amazon.com/bedrock/pricing/
6. Google Cloud. (2026). *Vertex AI Pricing*. https://cloud.google.com/vertex-ai/pricing
7. Anthropic. (2026). *Claude API Documentation*. https://docs.anthropic.com/
8. FinOps Foundation. (2025). *FinOps for AI: Managing ML and LLM Costs*.

---

> **Next:** [07 — Data Sovereignty and Compliance Multi-Cloud](07-Data-Sovereignty-and-Compliance-Multi-Cloud.md)

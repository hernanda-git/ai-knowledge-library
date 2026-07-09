# 66 — AI Model Commoditization and Economics: Technical Deep-Dive

> **Category:** 66 — AI Model Commoditization and Economics  
> **Focus:** Implementation patterns, cost optimization code, self-hosting architecture, model routing systems  
> **Cross-references:** [02-LLMs/10-Model-Routing](../02-LLMs/10-AI-Model-Routing-and-Smart-Selection.md), [23-Local-AI-Inference](../23-Local-AI-Inference-Self-Hosting/), [56-MLOps](../56-MLOps-and-AI-Platform-Engineering/)

---

## Table of Contents

1. [Cost Monitoring and Optimization Pipeline](#1-cost-monitoring-and-optimization-pipeline)
2. [Model Routing Implementation](#2-model-routing-implementation)
3. [Self-Hosting Architecture](#3-self-hosting-architecture)
4. [Semantic Caching System](#4-semantic-caching-system)
5. [Batch Processing Optimization](#5-batch-processing-optimization)
6. [Quantization and Efficiency](#6-quantization-and-efficiency)
7. [Multi-Model Orchestration](#7-multi-model-orchestration)
8. [Cost Alerting and Governance](#8-cost-alerting-and-governance)
9. [Performance Benchmarking](#9-performance-benchmarking)
10. [Production Deployment Patterns](#10-production-deployment-patterns)

---

## 1. Cost Monitoring and Optimization Pipeline

### 1.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Cost Optimization Pipeline                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Request  │───>│  Model   │───>│  Inference│              │
│  │ Router   │    │ Selector │    │  Engine   │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│       │               │               │                     │
│       ▼               ▼               ▼                     │
│  ┌──────────────────────────────────────────┐              │
│  │           Cost Tracker                    │              │
│  │  - Token count (input/output)             │              │
│  │  - Model used                             │              │
│  │  - Latency                                │              │
│  │  - Cost per request                       │              │
│  └──────────────────────────────────────────┘              │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────────────────────────────┐              │
│  │           Analytics Dashboard             │              │
│  │  - Real-time cost monitoring              │              │
│  │  - Optimization recommendations          │              │
│  │  - Anomaly detection                      │              │
│  │  - Budget alerts                          │              │
│  └──────────────────────────────────────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Cost Tracking Implementation

```python
# cost_tracker.py — Real-time AI cost tracking

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List
import json

@dataclass
class InferenceCost:
    """Track cost for a single inference call."""
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_per_1m_input: float
    cost_per_1m_output: float
    latency_ms: float
    timestamp: datetime
    task_type: Optional[str] = None
    cached: bool = False

    @property
    def total_cost(self) -> float:
        """Calculate total cost for this inference call."""
        input_cost = (self.input_tokens / 1_000_000) * self.cost_per_1m_input
        output_cost = (self.output_tokens / 1_000_000) * self.cost_per_1m_output
        return input_cost + output_cost

    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.total_cost,
            "latency_ms": self.latency_ms,
            "task_type": self.task_type,
            "cached": self.cached,
            "timestamp": self.timestamp.isoformat()
        }


class CostTracker:
    """Aggregate cost tracking and optimization analytics."""

    # Pricing table (July 2026)
    PRICING = {
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "claude-4.1-opus": {"input": 15.00, "output": 75.00},
        "claude-4.1-sonnet": {"input": 3.00, "output": 15.00},
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
        "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
        "glm-5.2": {"input": 0.15, "output": 0.60},
        "qwen-3.6-35b": {"input": 0.12, "output": 0.48},
        "llama-4-maverick": {"input": 0.20, "output": 0.85},
        "groq-llama-4": {"input": 0.05, "output": 0.18},
        "self-hosted-8b": {"input": 0.01, "output": 0.02},
    }

    def __init__(self):
        self.calls: List[InferenceCost] = []
        self.monthly_budget: float = 10000.0  # $10K/month default

    def track(self, cost: InferenceCost) -> Dict:
        """Track an inference call and return analytics."""
        self.calls.append(cost)

        # Calculate running totals
        total_cost = sum(c.total_cost for c in self.calls)
        calls_by_model = {}
        for c in self.calls:
            if c.model not in calls_by_model:
                calls_by_model[c.model] = {"count": 0, "cost": 0.0}
            calls_by_model[c.model]["count"] += 1
            calls_by_model[c.model]["cost"] += c.total_cost

        # Calculate savings vs. all-GPT-4 baseline
        gpt4_baseline = sum(
            (c.input_tokens / 1_000_000) * 8.00 +
            (c.output_tokens / 1_000_000) * 30.00
            for c in self.calls
        )
        savings = gpt4_baseline - total_cost
        savings_pct = (savings / gpt4_baseline * 100) if gpt4_baseline > 0 else 0

        return {
            "total_cost": total_cost,
            "savings_vs_gpt4": savings,
            "savings_percentage": savings_pct,
            "calls_by_model": calls_by_model,
            "budget_remaining": self.monthly_budget - total_cost,
            "budget_utilization": (total_cost / self.monthly_budget) * 100
        }

    def get_optimization_recommendations(self) -> List[Dict]:
        """Analyze usage patterns and suggest optimizations."""
        recommendations = []

        # Check for overuse of expensive models
        model_costs = {}
        for c in self.calls:
            if c.model not in model_costs:
                model_costs[c.model] = {"cost": 0, "tokens": 0}
            model_costs[c.model]["cost"] += c.total_cost
            model_costs[c.model]["tokens"] += c.output_tokens

        for model, data in model_costs.items():
            if data["cost"] > self.monthly_budget * 0.3:
                recommendations.append({
                    "type": "model_optimization",
                    "priority": "high",
                    "message": f"Model {model} accounts for {data['cost']/self.monthly_budget*100:.0f}% of budget. Consider routing some requests to cheaper alternatives.",
                    "estimated_savings": data["cost"] * 0.5
                })

        # Check for uncached repeated queries
        cached_ratio = sum(1 for c in self.calls if c.cached) / len(self.calls) if self.calls else 0
        if cached_ratio < 0.3:
            recommendations.append({
                "type": "caching",
                "priority": "medium",
                "message": f"Cache hit rate is only {cached_ratio*100:.0f}%. Implementing semantic caching could reduce costs by 30-50%.",
                "estimated_savings": sum(c.total_cost for c in self.calls) * 0.3
            })

        return recommendations
```

### 1.3 Usage

```python
# Example: Track inference costs and get recommendations
tracker = CostTracker()

# Track a GPT-4.1 call
tracker.track(InferenceCost(
    model="gpt-4.1",
    provider="openai",
    input_tokens=500,
    output_tokens=200,
    cost_per_1m_input=2.00,
    cost_per_1m_output=8.00,
    latency_ms=1200,
    timestamp=datetime.now(),
    task_type="complex_reasoning"
))

# Track a cheaper alternative
tracker.track(InferenceCost(
    model="glm-5.2",
    provider="zhipu",
    input_tokens=500,
    output_tokens=200,
    cost_per_1m_input=0.15,
    cost_per_1m_output=0.60,
    latency_ms=800,
    timestamp=datetime.now(),
    task_type="classification"
))

# Get analytics
analytics = tracker.track(InferenceCost(...))
print(f"Savings: ${analytics['savings_vs_gpt4']:.2f} ({analytics['savings_percentage']:.0f}%)")

# Get recommendations
recs = tracker.get_optimization_recommendations()
for r in recs:
    print(f"[{r['priority']}] {r['message']}")
```

---

## 2. Model Routing Implementation

### 2.1 The Model Router

```python
# model_router.py — Intelligent model selection based on task requirements

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import hashlib

class TaskComplexity(Enum):
    """Classify task complexity for model routing."""
    TRIVIAL = "trivial"      # Classification, extraction, routing
    SIMPLE = "simple"        # Summarization, Q&A, translation
    MODERATE = "moderate"    # Coding, analysis, writing
    COMPLEX = "complex"      # Reasoning, planning, multi-step
    FRONTIER = "frontier"    # Novel tasks, creative, ambiguous

@dataclass
class ModelConfig:
    """Configuration for a model in the routing table."""
    name: str
    provider: str
    cost_per_1m_input: float
    cost_per_1m_output: float
    quality_score: float  # 0-100 (e.g., Artificial Analysis score)
    latency_p50_ms: float
    context_window: int
    max_batch_size: int = 32
    supports_vision: bool = False
    supports_tools: bool = False

    @property
    def cost_efficiency(self) -> float:
        """Quality per dollar."""
        avg_cost = (self.cost_per_1m_input + self.cost_per_1m_output) / 2
        return self.quality_score / avg_cost if avg_cost > 0 else 0


class ModelRouter:
    """Route requests to optimal model based on task requirements."""

    # Model registry (July 2026 pricing)
    MODELS = {
        "gpt-4.1": ModelConfig("gpt-4.1", "openai", 2.00, 8.00, 75.0, 1200, 1_000_000, supports_tools=True),
        "claude-4.1-opus": ModelConfig("claude-4.1-opus", "anthropic", 15.00, 75.00, 78.0, 2500, 200_000, supports_tools=True),
        "claude-4.1-sonnet": ModelConfig("claude-4.1-sonnet", "anthropic", 3.00, 15.00, 73.0, 1500, 200_000, supports_tools=True),
        "glm-5.2": ModelConfig("glm-5.2", "zhipu", 0.15, 0.60, 71.4, 800, 128_000, supports_tools=True),
        "qwen-3.6-35b": ModelConfig("qwen-3.6-35b", "alibaba", 0.12, 0.48, 70.8, 900, 128_000, supports_tools=True),
        "llama-4-maverick": ModelConfig("llama-4-maverick", "meta", 0.20, 0.85, 70.1, 700, 1_000_000, supports_tools=True),
        "gemini-2.5-flash": ModelConfig("gemini-2.5-flash", "google", 0.15, 0.60, 68.0, 400, 1_000_000, supports_vision=True),
        "phi-5-mini": ModelConfig("phi-5-mini", "microsoft", 0.005, 0.02, 64.3, 100, 128_000),
        "qwen-2.5-3b": ModelConfig("qwen-2.5-3b", "alibaba", 0.002, 0.008, 60.0, 50, 32_000),
    }

    def __init__(self, strategy: str = "cost_optimized"):
        """
        Initialize router with strategy.

        Strategies:
        - "cost_optimized": Minimize cost while meeting quality threshold
        - "quality_first": Maximize quality within budget
        - "balanced": Best quality/cost ratio
        - "latency_first": Minimize latency
        """
        self.strategy = strategy
        self.cache: Dict[str, str] = {}

    def classify_task(self, prompt: str, context: Dict = None) -> TaskComplexity:
        """Classify task complexity based on prompt analysis."""
        prompt_lower = prompt.lower()
        prompt_len = len(prompt.split())

        # Heuristic classification
        if any(kw in prompt_lower for kw in ["think step by step", "reason", "analyze deeply", "plan"]):
            return TaskComplexity.COMPLEX
        if any(kw in prompt_lower for kw in ["write code", "implement", "debug", "refactor"]):
            return TaskComplexity.MODERATE
        if any(kw in prompt_lower for kw in ["classify", "extract", "route", "label", "tag"]):
            return TaskComplexity.TRIVIAL
        if any(kw in prompt_lower for kw in ["summarize", "translate", "rewrite", "what is"]):
            return TaskComplexity.SIMPLE
        if prompt_len > 500:
            return TaskComplexity.MODERATE
        return TaskComplexity.SIMPLE

    def route(self, prompt: str, context: Dict = None) -> str:
        """Route request to optimal model."""
        # Check cache for repeated queries
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]

        complexity = self.classify_task(prompt, context)

        # Quality thresholds by complexity
        quality_thresholds = {
            TaskComplexity.TRIVIAL: 55.0,
            TaskComplexity.SIMPLE: 60.0,
            TaskComplexity.MODERATE: 68.0,
            TaskComplexity.COMPLEX: 72.0,
            TaskComplexity.FRONTIER: 76.0,
        }

        threshold = quality_thresholds[complexity]

        # Filter models meeting quality threshold
        eligible = [
            (name, model) for name, model in self.MODELS.items()
            if model.quality_score >= threshold
        ]

        if not eligible:
            # Fall back to best available
            eligible = list(self.MODELS.items())

        # Apply strategy
        if self.strategy == "cost_optimized":
            # Select cheapest eligible model
            selected = min(eligible, key=lambda x: (x[1].cost_per_1m_input + x[1].cost_per_1m_output) / 2)
        elif self.strategy == "quality_first":
            # Select highest quality eligible model
            selected = max(eligible, key=lambda x: x[1].quality_score)
        elif self.strategy == "balanced":
            # Select best cost efficiency
            selected = max(eligible, key=lambda x: x[1].cost_efficiency)
        elif self.strategy == "latency_first":
            # Select fastest eligible model
            selected = min(eligible, key=lambda x: x[1].latency_p50_ms)
        else:
            selected = eligible[0]

        self.cache[cache_key] = selected[0]
        return selected[0]

    def estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a given model and token counts."""
        model = self.MODELS[model_name]
        return (input_tokens / 1_000_000 * model.cost_per_1m_input +
                output_tokens / 1_000_000 * model.cost_per_1m_output)


# Example usage
router = ModelRouter(strategy="cost_optimized")

# Trivial task → routed to cheapest model
model = router.route("Classify this email as spam or not spam")
print(f"Trivial task → {model}")  # qwen-2.5-3b

# Complex task → routed to capable model
model = router.route("Think step by step: Design a distributed caching system")
print(f"Complex task → {model}")  # glm-5.2 or qwen-3.6-35b

# Frontier task → routed to best model
model = router.route("Analyze the game theory implications of AI pricing wars")
print(f"Frontier task → {model}")  # gpt-4.1 or claude-4.1-opus
```

---

## 3. Self-Hosting Architecture

### 3.1 Production Self-Hosting Stack

```
┌─────────────────────────────────────────────────────────────┐
│                  Self-Hosting Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Load     │───>│ vLLM     │───>│ GPU      │              │
│  │ Balancer │    │ Inference│    │ Cluster  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│       │               │               │                     │
│       ▼               ▼               ▼                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Model    │    │ KV-Cache │    │ Health   │              │
│  │ Registry │    │ Manager  │    │ Checker  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 vLLM Deployment Configuration

```yaml
# docker-compose.yml — Production vLLM deployment

version: '3.8'

services:
  vllm-primary:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
    command: >
      --model meta-llama/Llama-4-Maverick-17B-128E-Instruct
      --host 0.0.0.0
      --port 8000
      --tensor-parallel-size 2
      --max-model-len 131072
      --gpu-memory-utilization 0.9
      --enable-prefix-caching
      --enable-chunked-prefill
      --quantization awq
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    volumes:
      - model-cache:/root/.cache/huggingface
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  vllm-secondary:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    command: >
      --model Qwen/Qwen3.6-35B-A3B-AWQ
      --host 0.0.0.0
      --port 8001
      --tensor-parallel-size 1
      --max-model-len 65536
      --gpu-memory-utilization 0.85
      --quantization awq
    ports:
      - "8001:8001"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - vllm-primary
      - vllm-secondary

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  model-cache:
```

### 3.3 Cost Comparison: Self-Hosted vs. API

```python
# cost_comparison.py — Compare self-hosted vs API costs

@dataclass
class HostingOption:
    name: str
    monthly_fixed_cost: float  # Hardware depreciation + electricity
    cost_per_1m_tokens: float  # Variable cost
    setup_cost: float  # One-time setup
    setup_time_hours: float

OPTIONS = {
    "openai_api": HostingOption(
        name="OpenAI GPT-4.1 API",
        monthly_fixed_cost=0,
        cost_per_1m_tokens=10.00,  # Average of input/output
        setup_cost=0,
        setup_time_hours=0.5
    ),
    "groq_api": HostingOption(
        name="Groq Llama 4 API",
        monthly_fixed_cost=0,
        cost_per_1m_tokens=0.25,
        setup_cost=0,
        setup_time_hours=0.5
    ),
    "self_hosted_2gpu": HostingOption(
        name="Self-hosted Llama 4 (2x H100)",
        monthly_fixed_cost=5000,
        cost_per_1m_tokens=0.08,
        setup_cost=10000,
        setup_time_hours=20
    ),
    "self_hosted_8gpu": HostingOption(
        name="Self-hosted GLM 5.2 (8x H100)",
        monthly_fixed_cost=20000,
        cost_per_1m_tokens=0.10,
        setup_cost=30000,
        setup_time_hours=40
    ),
}


def compare_costs(monthly_tokens_millions: float, options: dict = OPTIONS) -> dict:
    """Compare total costs for different hosting options at given usage."""
    results = {}
    for name, option in options.items():
        monthly_cost = (
            option.monthly_fixed_cost +
            monthly_tokens_millions * option.cost_per_1m_tokens
        )
        annual_cost = monthly_cost * 12 + option.setup_cost
        cost_per_token = monthly_cost / (monthly_tokens_millions * 1_000_000)

        results[name] = {
            "monthly_cost": monthly_cost,
            "annual_cost": annual_cost,
            "cost_per_token": cost_per_token,
            "setup_time": option.setup_time_hours,
        }

    # Find cheapest
    cheapest = min(results.items(), key=lambda x: x[1]["annual_cost"])
    results["recommendation"] = cheapest[0]

    return results


# Example: 1B tokens/month
results = compare_costs(1000)  # 1000M = 1B tokens
for name, data in results.items():
    if name != "recommendation":
        print(f"{name}: ${data['monthly_cost']:,.0f}/month, ${data['annual_cost']:,.0f}/year")

print(f"\nRecommendation: {results['recommendation']}")
```

---

## 4. Semantic Caching System

### 4.1 Architecture

```python
# semantic_cache.py — Semantic caching for AI inference

import hashlib
import json
import time
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class CacheEntry:
    prompt_hash: str
    response: str
    model: str
    tokens_used: int
    cost_saved: float
    created_at: float
    ttl_seconds: int = 86400  # 24 hours default

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


class SemanticCache:
    """Cache semantically similar prompts to reduce API costs."""

    def __init__(self, similarity_threshold: float = 0.92, max_size: int = 10000):
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.total_saved: float = 0.0

    def _hash_prompt(self, prompt: str) -> str:
        """Generate deterministic hash for exact-match caching."""
        return hashlib.sha256(prompt.lower().strip().encode()).hexdigest()[:16]

    def _compute_embedding(self, prompt: str) -> np.ndarray:
        """Compute simple TF-IDF-like embedding for similarity."""
        # Simplified embedding (production would use a real embedder)
        words = prompt.lower().split()
        vocab = list(set(w for entries in [list(self.embeddings.keys())] for w in entries))
        if not vocab:
            vocab = words
        embedding = np.zeros(len(vocab))
        for word in words:
            if word in vocab:
                embedding[vocab.index(word)] += 1
        # Normalize
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def _find_similar(self, embedding: np.ndarray) -> Optional[str]:
        """Find most similar cached entry above threshold."""
        best_score = 0
        best_hash = None

        for hash_key, cached_emb in self.embeddings.items():
            if hash_key in self.cache and not self.cache[hash_key].is_expired:
                score = np.dot(embedding, cached_emb)
                if score > best_score and score >= self.threshold:
                    best_score = score
                    best_hash = hash_key

        return best_hash

    def get(self, prompt: str) -> Optional[Tuple[str, float]]:
        """
        Look up cached response.
        Returns (response, cost_saved) or None if cache miss.
        """
        # Exact match first
        prompt_hash = self._hash_prompt(prompt)
        if prompt_hash in self.cache:
            entry = self.cache[prompt_hash]
            if not entry.is_expired:
                return (entry.response, entry.cost_saved)

        # Semantic similarity match
        embedding = self._compute_embedding(prompt)
        similar_hash = self._find_similar(embedding)
        if similar_hash:
            entry = self.cache[similar_hash]
            return (entry.response, entry.cost_saved)

        return None

    def set(self, prompt: str, response: str, model: str, tokens: int, cost: float):
        """Store a new cache entry."""
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.items(), key=lambda x: x[1].created_at)
            del self.cache[oldest[0]]
            if oldest[0] in self.embeddings:
                del self.embeddings[oldest[0]]

        prompt_hash = self._hash_prompt(prompt)
        self.cache[prompt_hash] = CacheEntry(
            prompt_hash=prompt_hash,
            response=response,
            model=model,
            tokens_used=tokens,
            cost_saved=cost,
            created_at=time.time()
        )
        self.embeddings[prompt_hash] = self._compute_embedding(prompt)
        self.total_saved += cost

    def stats(self) -> Dict:
        """Return cache statistics."""
        total_entries = len(self.cache)
        active_entries = sum(1 for e in self.cache.values() if not e.is_expired)
        return {
            "total_entries": total_entries,
            "active_entries": active_entries,
            "total_cost_saved": self.total_saved,
            "hit_rate": active_entries / max(total_entries, 1),
        }
```

### 4.2 Expected Savings from Caching

| Scenario | Cache Hit Rate | Monthly Cost (No Cache) | Monthly Cost (With Cache) | Savings |
|----------|---------------|------------------------|--------------------------|---------|
| Customer support | 40% | $5,000 | $3,000 | $2,000 (40%) |
| Document processing | 30% | $8,000 | $5,600 | $2,400 (30%) |
| API serving | 25% | $12,000 | $9,000 | $3,000 (25%) |
| Internal tools | 50% | $3,000 | $1,500 | $1,500 (50%) |

---

## 5. Batch Processing Optimization

### 5.1 Batch vs. Real-Time Cost Comparison

| Provider | Real-Time $/1M | Batch $/1M | Savings |
|----------|---------------|-----------|---------|
| OpenAI | $8.00 (output) | $4.00 (output) | 50% |
| Anthropic | $15.00 (output) | $7.50 (output) | 50% |
| Google | $10.00 (output) | $5.00 (output) | 50% |
| Self-hosted | $0.18 (output) | $0.10 (output) | 44% |

### 5.2 Batch Processing Implementation

```python
# batch_processor.py — Batch AI inference for cost optimization

import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
import time

@dataclass
class BatchJob:
    job_id: str
    prompts: List[str]
    model: str
    priority: str = "normal"  # "low", "normal", "high"
    max_tokens: int = 1000

class BatchProcessor:
    """Process AI inference requests in batches for cost savings."""

    def __init__(self, batch_size: int = 32, max_wait_seconds: float = 5.0):
        self.batch_size = batch_size
        self.max_wait = max_wait_seconds
        self.queue: List[BatchJob] = []
        self.results: Dict[str, List[str]] = {}

    async def submit(self, job: BatchJob) -> str:
        """Submit a batch job for processing."""
        self.queue.append(job)
        return job.job_id

    async def process_batch(self, batch: List[BatchJob]) -> Dict[str, List[str]]:
        """Process a batch of jobs (would call inference API in production)."""
        # Simulate batch processing
        results = {}
        for job in batch:
            results[job.job_id] = [f"Response to: {p[:50]}..." for p in job.prompts]
        return results

    async def run(self):
        """Main processing loop."""
        while True:
            if len(self.queue) >= self.batch_size:
                # Process full batch
                batch = self.queue[:self.batch_size]
                self.queue = self.queue[self.batch_size:]
                results = await self.process_batch(batch)
                self.results.update(results)
            elif self.queue and time.time() % self.max_wait < 1:
                # Process remaining with smaller batch
                results = await self.process_batch(self.queue)
                self.results.update(results)
                self.queue = []
            await asyncio.sleep(0.1)

# Cost savings calculation
def calculate_batch_savings(
    monthly_prompts: int,
    avg_tokens_per_prompt: int,
    model: str = "gpt-4.1"
) -> Dict:
    """Calculate potential savings from batch processing."""
    real_time_cost = {
        "gpt-4.1": 8.00,
        "claude-4.1-sonnet": 15.00,
        "glm-5.2": 0.60,
    }
    batch_discount = 0.50  # 50% discount for batch

    monthly_tokens_millions = (monthly_prompts * avg_tokens_per_prompt) / 1_000_000
    real_time_monthly = monthly_tokens_millions * real_time_cost.get(model, 8.00)
    batch_monthly = real_time_monthly * (1 - batch_discount)

    return {
        "monthly_prompts": monthly_prompts,
        "monthly_tokens_millions": monthly_tokens_millions,
        "real_time_cost": real_time_monthly,
        "batch_cost": batch_monthly,
        "monthly_savings": real_time_monthly - batch_monthly,
        "annual_savings": (real_time_monthly - batch_monthly) * 12,
        "savings_percentage": batch_discount * 100,
    }
```

---

## 6. Quantization and Efficiency

### 6.1 Quantization Decision Matrix

| Use Case | Recommended Format | Quality Impact | Cost Reduction | When to Use |
|----------|-------------------|---------------|----------------|-------------|
| Production, quality critical | FP16 / BF16 | None | 0% | When quality is paramount |
| Production, balanced | INT8 (WOQ) | <0.5% | 33% | Most production workloads |
| Production, cost optimized | INT4 (AWQ) | 1–2% | 60% | High-volume, cost-sensitive |
| Edge / mobile | INT4 (GGUF Q4_K_M) | 2–3% | 65% | Consumer hardware |
| Extreme compression | INT2 (GGUF Q2_K) | 5–8% | 80% | Embedded, very constrained |
| Experimental | Binary (BBT) | 15–25% | 90% | Research only |

### 6.2 Quantization Code Example

```python
# quantize_model.py — Quantize model for cost optimization

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def quantize_model(
    model_name: str,
    output_dir: str,
    quant_type: str = "int4",
    compute_dtype: str = "float16"
):
    """Quantize a model for cost-optimized deployment."""

    # Configure quantization
    if quant_type == "int8":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    elif quant_type == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, compute_dtype),
        )
    else:
        raise ValueError(f"Unsupported quant type: {quant_type}")

    # Load and quantize
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save quantized model
    print(f"Saving quantized model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Report savings
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"Quantized model saved. Approximate size: {original_size:.1f} GB")
    print(f"Estimated cost reduction: {60 if quant_type == 'int4' else 33}%")

    return model, tokenizer


# Example: Quantize Qwen 3.6 for cost-optimized deployment
# quantize_model("Qwen/Qwen3.6-35B-A3B", "./qwen-3.6-int4", quant_type="int4")
```

---

## 7. Multi-Model Orchestration

### 7.1 Orchestration Architecture

```python
# multi_model_orchestrator.py — Orchestrate multiple models for optimal cost/quality

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio

@dataclass
class ModelEndpoint:
    name: str
    url: str
    api_key: str
    cost_per_1m_input: float
    cost_per_1m_output: float
    max_concurrent: int = 10
    current_load: int = 0

class MultiModelOrchestrator:
    """Orchestrate multiple model endpoints for cost optimization."""

    def __init__(self):
        self.endpoints: Dict[str, ModelEndpoint] = {}
        self.routing_rules: List[Dict] = []

    def register_endpoint(self, endpoint: ModelEndpoint):
        """Register a model endpoint."""
        self.endpoints[endpoint.name] = endpoint

    def add_routing_rule(
        self,
        task_type: str,
        preferred_model: str,
        fallback_model: str,
        max_cost_per_token: float
    ):
        """Add a routing rule for a task type."""
        self.routing_rules.append({
            "task_type": task_type,
            "preferred": preferred_model,
            "fallback": fallback_model,
            "max_cost": max_cost_per_token,
        })

    async def route_request(
        self,
        prompt: str,
        task_type: str,
        context: Dict = None
    ) -> Dict[str, Any]:
        """Route request to optimal model based on rules and load."""
        # Find applicable rule
        rule = next(
            (r for r in self.routing_rules if r["task_type"] == task_type),
            None
        )

        if rule:
            preferred = self.endpoints.get(rule["preferred"])
            fallback = self.endpoints.get(rule["fallback"])

            # Check if preferred model has capacity
            if preferred and preferred.current_load < preferred.max_concurrent:
                return await self._call_model(preferred, prompt)
            elif fallback and fallback.current_load < fallback.max_concurrent:
                return await self._call_model(fallback, prompt)

        # Default: pick least loaded endpoint
        available = [
            ep for ep in self.endpoints.values()
            if ep.current_load < ep.max_concurrent
        ]
        if available:
            least_loaded = min(available, key=lambda x: x.current_load)
            return await self._call_model(least_loaded, prompt)

        raise RuntimeError("All endpoints at capacity")

    async def _call_model(self, endpoint: ModelEndpoint, prompt: str) -> Dict:
        """Call a model endpoint (simulated)."""
        endpoint.current_load += 1
        try:
            # In production, this would make an actual API call
            response = f"Response from {endpoint.name}"
            return {
                "model": endpoint.name,
                "response": response,
                "cost": len(prompt) * endpoint.cost_per_1m_input / 1_000_000,
            }
        finally:
            endpoint.current_load -= 1


# Example setup
orchestrator = MultiModelOrchestrator()

# Register endpoints
orchestrator.register_endpoint(ModelEndpoint(
    name="gpt-4.1",
    url="https://api.openai.com/v1/chat/completions",
    api_key="sk-...",
    cost_per_1m_input=2.00,
    cost_per_1m_output=8.00,
    max_concurrent=20
))

orchestrator.register_endpoint(ModelEndpoint(
    name="glm-5.2",
    url="https://api.zhipu.com/v1/chat/completions",
    api_key="...",
    cost_per_1m_input=0.15,
    cost_per_1m_output=0.60,
    max_concurrent=50
))

# Add routing rules
orchestrator.add_routing_rule("classification", "glm-5.2", "gpt-4.1", 0.001)
orchestrator.add_routing_rule("reasoning", "gpt-4.1", "glm-5.2", 0.01)
```

---

## 8. Cost Alerting and Governance

### 8.1 Alert Configuration

```python
# cost_alerts.py — Automated cost alerting and governance

from enum import Enum
from dataclasses import dataclass
from typing import List, Callable
import time

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class CostAlert:
    name: str
    threshold: float  # Percentage of budget
    severity: AlertSeverity
    action: str  # "notify", "throttle", "block"
    cooldown_seconds: int = 3600

class CostGovernance:
    """Automated cost governance with alerts and controls."""

    def __init__(self, monthly_budget: float):
        self.budget = monthly_budget
        self.spent: float = 0.0
        self.alerts: List[CostAlert] = []
        self.alert_history: dict = {}
        self.callbacks: List[Callable] = []

        # Default alerts
        self.alerts.extend([
            CostAlert("50% budget", 50.0, AlertSeverity.INFO, "notify"),
            CostAlert("75% budget", 75.0, AlertSeverity.WARNING, "notify"),
            CostAlert("90% budget", 90.0, AlertSeverity.CRITICAL, "throttle"),
            CostAlert("100% budget", 100.0, AlertSeverity.EMERGENCY, "block"),
        ])

    def record_usage(self, cost: float, model: str, task_type: str) -> List[dict]:
        """Record usage and check alerts."""
        self.spent += cost
        utilization = (self.spent / self.budget) * 100

        triggered = []
        for alert in self.alerts:
            if utilization >= alert.threshold:
                # Check cooldown
                last_triggered = self.alert_history.get(alert.name, 0)
                if time.time() - last_triggered > alert.cooldown_seconds:
                    triggered.append({
                        "alert": alert.name,
                        "severity": alert.severity.value,
                        "utilization": utilization,
                        "action": alert.action,
                        "message": f"Budget utilization at {utilization:.1f}% ({alert.name})",
                    })
                    self.alert_history[alert.name] = time.time()

                    # Execute action
                    if alert.action == "block":
                        raise RuntimeError(f"Budget exceeded: {utilization:.1f}% of ${self.budget:,.0f}")

        return triggered

    def get_status(self) -> dict:
        """Get current governance status."""
        return {
            "budget": self.budget,
            "spent": self.spent,
            "remaining": self.budget - self.spent,
            "utilization_pct": (self.spent / self.budget) * 100,
            "days_in_month": 30,
            "projected_monthly": self.spent * 30 / max(time.time() % (30 * 86400) / 86400, 1),
            "alerts_triggered": len(self.alert_history),
        }


# Example usage
governance = CostGovernance(monthly_budget=10000.0)

# Simulate usage
for _ in range(100):
    alerts = governance.record_usage(cost=50.0, model="gpt-4.1", task_type="coding")
    for alert in alerts:
        print(f"[{alert['severity']}] {alert['message']}")

print(governance.get_status())
```

---

## 9. Performance Benchmarking

### 9.1 Cost-Performance Benchmarking Framework

```python
# benchmark.py — Benchmark model cost-performance

from dataclasses import dataclass
from typing import List, Dict
import time

@dataclass
class BenchmarkResult:
    model: str
    task_type: str
    quality_score: float  # 0-100
    latency_p50_ms: float
    latency_p99_ms: float
    cost_per_1m_tokens: float
    tokens_per_second: float

    @property
    def cost_efficiency(self) -> float:
        """Quality per dollar."""
        return self.quality_score / self.cost_per_1m_tokens

    @property
    def latency_efficiency(self) -> float:
        """Quality per millisecond."""
        return self.quality_score / self.latency_p50_ms


class CostPerformanceBenchmark:
    """Benchmark models for cost-performance optimization."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def rank_models(
        self,
        task_type: str,
        ranking_criteria: str = "cost_efficiency"
    ) -> List[Dict]:
        """Rank models by specified criteria for a task type."""
        filtered = [r for r in self.results if r.task_type == task_type]

        if ranking_criteria == "cost_efficiency":
            ranked = sorted(filtered, key=lambda x: x.cost_efficiency, reverse=True)
        elif ranking_criteria == "latency":
            ranked = sorted(filtered, key=lambda x: x.latency_p50_ms)
        elif ranking_criteria == "quality":
            ranked = sorted(filtered, key=lambda x: x.quality_score, reverse=True)
        else:
            ranked = filtered

        return [
            {
                "rank": i + 1,
                "model": r.model,
                "quality": r.quality_score,
                "latency_p50": r.latency_p50_ms,
                "cost_per_1m": r.cost_per_1m_tokens,
                "cost_efficiency": r.cost_efficiency,
            }
            for i, r in enumerate(ranked)
        ]

    def generate_report(self) -> str:
        """Generate a markdown report of benchmark results."""
        report = ["# Cost-Performance Benchmark Report\n"]

        task_types = set(r.task_type for r in self.results)
        for task_type in sorted(task_types):
            report.append(f"\n## {task_type.title()}\n")
            rankings = self.rank_models(task_type)
            report.append("| Rank | Model | Quality | Latency (ms) | Cost/1M | Efficiency |")
            report.append("|------|-------|---------|-------------|---------|-----------|")
            for r in rankings:
                report.append(
                    f"| {r['rank']} | {r['model']} | {r['quality']:.1f} | "
                    f"{r['latency_p50']:.0f} | ${r['cost_per_1m']:.2f} | "
                    f"{r['cost_efficiency']:.1f} |"
                )

        return "\n".join(report)
```

---

## 10. Production Deployment Patterns

### 10.1 Deployment Pattern Comparison

| Pattern | Complexity | Cost Savings | Latency | Best For |
|---------|-----------|-------------|---------|----------|
| **Single model API** | Low | 0% (baseline) | Network-dependent | Prototyping |
| **Multi-model routing** | Medium | 60–80% | Same as single | Production workloads |
| **Self-hosted + API fallback** | Medium | 50–70% | Lower (local) | High-volume |
| **Semantic caching layer** | Low | 30–50% | Lower (cached) | Repeated queries |
| **Batch processing** | Medium | 40–50% | Higher (delayed) | Non-real-time |
| **Full optimization stack** | High | 70–90% | Variable | Enterprise |

### 10.2 Recommended Architecture by Company Size

| Company Size | Recommended Stack | Monthly Budget | Expected Savings |
|-------------|-------------------|---------------|-----------------|
| Startup (1–10) | API + caching | $500–5,000 | 20–30% |
| Small (10–50) | Multi-model routing + caching | $5,000–20,000 | 40–60% |
| Mid-size (50–200) | Self-hosted + routing + caching | $20,000–100,000 | 60–80% |
| Enterprise (200+) | Full stack + custom optimization | $100,000+ | 70–90% |

---

## Key Implementation Insights

1. **Start with cost tracking.** You cannot optimize what you do not measure. Implement cost tracking before any optimization.

2. **Model routing is the highest-ROI optimization.** Routing 80% of volume to cheaper models saves 60–80% with minimal engineering effort.

3. **Semantic caching is low-hanging fruit.** Implement caching for any workload with >20% repeated queries.

4. **Self-hosting is viable above 500M tokens/month.** Below that threshold, API providers are typically cheaper.

5. **Quantization is essentially free quality.** INT4 quantization reduces costs 60% with <2% quality loss for most tasks.

---

*See also: [04-Tools-and-Frameworks.md](04-Tools-and-Frameworks.md) for tool recommendations and [05-Future-Outlook.md](05-Future-Outlook.md) for future trends.*

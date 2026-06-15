# Prompt Caching & Cost Optimization

## Table of Contents

1. [Overview](#overview)
2. [The Cost Challenge](#the-cost-challenge)
3. [Prompt Caching Fundamentals](#prompt-caching-fundamentals)
4. [Provider-Specific Caching](#provider-specific-caching)
5. [Semantic Caching](#semantic-caching)
6. [OpenSquilla — Open-Source Prompt Cache](#opensquilla--open-source-prompt-cache)
7. [Prompt Compression](#prompt-compression)
8. [Model Routing for Cost Optimization](#model-routing-for-cost-optimization)
9. [Token Budget Management](#token-budget-management)
10. [Caching Architecture Patterns](#caching-architecture-patterns)
11. [Cost Monitoring & Analytics](#cost-monitoring--analytics)
12. [Benchmarks & Savings Analysis](#benchmarks--savings-analysis)
13. [Production Checklist](#production-checklist)
14. [Cross-References](#cross-references)

---

## Overview

Prompt caching and cost optimization have become essential infrastructure for production LLM deployments. With enterprise AI spending projected to reach $200B+ by 2027, reducing per-query costs by 30-70% through caching, compression, and intelligent routing is a competitive necessity rather than a nice-to-have.

### The Problem

| Challenge | Impact |
|-----------|--------|
| Long system prompts repeated on every call | 40-60% of tokens are redundant |
| Similar queries pay full price each time | 2-10x cost multiplier |
| Top-tier models used for simple tasks | 5-20x cost premium unnecessarily |
| No visibility into cost per query/feature | Blind cost growth |

---

## The Cost Challenge

### LLM Cost Breakdown

Typical enterprise AI application costs break down as:

```
Total Cost = Σ(query_count × tokens_per_query × $price_per_token)
```

| Component | Typical Share | Optimization Lever |
|-----------|--------------|-------------------|
| System prompts | 30-50% | Prompt caching |
| Context (RAG results, history) | 20-30% | Context caching |
| User query processing | 15-25% | Model routing |
| Output generation | 10-20% | Token budget limits |

### Real Cost Numbers

| Use Case | Daily Queries | Monthly Cost (Unoptimized) | Optimized | Savings |
|----------|--------------|---------------------------|-----------|---------|
| Customer support agent | 100K | $45,000 | $13,500 | 70% |
| Code assistant (team of 50) | 5K | $22,500 | $7,875 | 65% |
| Content moderation | 500K | $37,500 | $11,250 | 70% |
| RAG search system | 200K | $60,000 | $18,000 | 70% |
| AI writing assistant | 50K | $30,000 | $9,000 | 70% |

---

## Prompt Caching Fundamentals

### What Can Be Cached

| Cacheable Component | Example | Savings Potential |
|--------------------|---------|-------------------|
| System prompts | "You are a helpful assistant..." | 30-50% |
| Few-shot examples | 5 example QA pairs | 10-20% |
| Retrieved context | RAG document chunks | 20-30% |
| Conversation history | Previous 10 messages | 15-25% |
| Generated outputs | Identical result for same input | 30-60% |

### Cache Types

```
LLM Request Cache Types
├── Prefix Caching (Provider-Side)
│   └── Cache attention KV for common prefixes
├── Semantic Caching (Your Side)
│   └── Cache responses for semantically similar queries
├── Exact Match Caching
│   └── Cache responses for identical queries
└── Context Caching
    └── Cache KV for shared context blocks
```

---

## Provider-Specific Caching

### Anthropic — Prompt Caching

Anthropic offers automatic prompt caching when you mark cacheable content:

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-4-opus",
    max_tokens=500,
    system=[
        {
            "type": "text",
            "text": LONG_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}  # ← Cache this!
        }
    ],
    messages=[{"role": "user", "content": user_query}]
)

# Check cache performance
print(f"Cache created: {response.usage.cache_creation_input_tokens}")
print(f"Cache read: {response.usage.cache_read_input_tokens}")
print(f"Input tokens: {response.usage.input_tokens}")
```

**Cache behavior:**

| Scenario | Cost | Cache Status |
|----------|------|-------------|
| First request (no cache) | 100% | Cache miss |
| Same prefix within 5 min | 10% of prefix cost | Cache hit |
| Same prefix after 5 min | 100% | Cache expired |

**Savings:** For a system prompt of 10K tokens, repeated 1000 times:
- Without cache: $15.00 (at $3/M input tokens)
- With cache: $1.50 + $3.00 (first + rest) = $4.50
- **Savings: 70%**

### OpenAI — Prompt Caching

OpenAI automatically caches repeated prompts:

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},  # Auto-cached
        {"role": "user", "content": user_query}
    ]
)

# Cache info in usage
print(f"Cached tokens: {response.usage.prompt_tokens_details.cached_tokens}")
```

**Eligibility:** Prompts with a shared prefix of 1024+ tokens are automatically cached. Cache TTL: 5-10 minutes of inactivity.

**Pricing:** Cached tokens are billed at 50% of the normal input token price.

### Google Gemini — Context Caching

```python
import google.generativeai as genai

# Create a cached context
cached_content = genai.models.cache(
    model="gemini-2.0-flash",
    system_instruction=SYSTEM_PROMPT,
    contents=RAG_DOCUMENTS,
    ttl=timedelta(minutes=30)
)

# Use cached context
model = genai.GenerativeModel.from_cached_content(cached_content)
response = model.generate_content(user_query)
```

---

## Semantic Caching

Semantic caching stores responses for semantically similar queries, not just exact matches:

### Architecture

```
User Query
    ↓
Embedding Model → Query Vector
    ↓
Vector DB Lookup (cosine similarity > threshold)
    ↓
┌───── Hit ───→ Return Cached Response (2-5ms)
└───── Miss ──→ LLM Inference → Store in Cache (500-2000ms)
```

### Implementation with GPTCache

```python
from gptcache import cache
from gptcache.adapter import openai
from gptcache.embedding import Onnx
from gptcache.similarity_evaluation import SearchDistanceEvaluation

# Configure cache
onnx = Onnx()  # Built-in embedding model
cache.init(
    pre_embedding_func=query_preprocessing,
    embedding_func=onnx.to_embeddings,
    similarity_evaluation=SearchDistanceEvaluation(),
    data_manager=manager_factory("sqlite,faiss", 
        sql_url="sqlite:///cache.db",
        top_k=3
    ),
)

# Use cached LLM
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": user_query}],
    cache=True  # ← Use semantic cache
)
```

### Custom Semantic Cache

```python
import numpy as np
from sentence_transformers import SentenceTransformer
import redis

class SemanticCache:
    def __init__(self, similarity_threshold=0.92):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.redis = redis.Redis(host="localhost", port=6379, db=0)
        self.threshold = similarity_threshold
    
    def get(self, query: str) -> str | None:
        query_vec = self.encoder.encode(query)
        
        # Retrieve all cached entries (or use vector index)
        for key in self.redis.scan_iter("cache:*"):
            cached = self.redis.hgetall(key)
            cached_vec = np.frombuffer(cached["vector"], dtype=np.float32)
            
            similarity = np.dot(query_vec, cached_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(cached_vec)
            )
            
            if similarity > self.threshold:
                return cached["response"].decode()
        
        return None
    
    def set(self, query: str, response: str):
        query_vec = self.encoder.encode(query)
        cache_id = hashlib.md5(query.encode()).hexdigest()
        
        self.redis.hset(
            f"cache:{cache_id}",
            mapping={
                "query": query,
                "response": response,
                "vector": query_vec.tobytes(),
                "timestamp": time.time()
            }
        )
```

### Similarity Threshold Tuning

| Threshold | Cache Hit Rate | Accuracy | Best For |
|-----------|---------------|----------|----------|
| 0.95 | 15% | 99.9% | Exact/very similar queries |
| 0.90 | 35% | 98.5% | Factual Q&A |
| 0.85 | 50% | 95% | Creative/personalized |
| 0.80 | 65% | 90% | Approximate, less critical |

---

## OpenSquilla — Open-Source Prompt Cache

OpenSquilla (4K★ GitHub) is an open-source prompt caching system designed for LLM inference:

### Features

- **Distributed KV cache** — Share prompt caches across multiple inference servers
- **Automatic prefix detection** — Identifies and caches common prompt prefixes
- **Multi-model support** — Works with OpenAI, Anthropic, Ollama, vLLM
- **Prometheus metrics** — Built-in monitoring for cache hit rates and savings
- **Configurable TTL** — Per-route cache expiration policies

### Architecture

```
OpenSquilla Cache Proxy
    │
    ├── Request enters → Extract prefix/semantic hash
    ├── Check KV cache (Redis + GPU memory)
    ├── ── Hit → Return cached KV → Skip model forward pass
    └── ── Miss → Forward to model → Cache KV for next time
```

### Quick Start

```bash
# Install OpenSquilla
pip install opensquilla

# Start cache proxy
opensquilla serve \
  --backend openai \
  --model gpt-4o \
  --cache-type kv \
  --ttl 300

# Use via proxy (drop-in replacement)
curl http://localhost:8080/v1/chat/completions \
  -d '{"model": "gpt-4o", "messages": [...]}'
```

### Performance

| Metric | Without Cache | With OpenSquilla |
|--------|--------------|-----------------|
| P50 latency | 1.2s | 0.3s |
| P99 latency | 3.0s | 0.8s |
| Cost per query | $0.015 | $0.004 |
| Throughput | 50 req/s | 200 req/s |
| Cache hit rate | — | 65% |

---

## Prompt Compression

### Techniques

**1. Token Removal (Lossy)**

```python
from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2",
    use_llmlingua2=True
)

compressed_prompt = compressor.compress_prompt(
    prompt,  # Long system prompt + context
    rate=0.5,  # Compress to 50%
    force_tokens=["\n", ":", "!", ".", "?", "System:", "User:"],
    iterative_size=210,
    side_token_aware=True
)
# → 50% fewer tokens with minimal quality loss
```

**2. Summary-Based Compression**

```python
# Shorten conversation history
def compress_history(messages, max_tokens=1000):
    """Summarize old history while keeping recent messages verbatim."""
    
    recent = messages[-5:]  # Keep 5 most recent verbatim
    older = messages[:-5]
    
    if count_tokens(older) > max_tokens:
        # Summarize older messages
        summary = llm.generate(
            f"Summarize this conversation concisely:\n{older}"
        )
        return [{"role": "system", "content": f"History summary: {summary}"}] + recent
    
    return messages
```

**3. Structured Extraction**

```python
# Instead of including full documents, extract only relevant parts
def extract_relevant_context(query, documents, max_tokens=2000):
    """Extract only the sentences relevant to the query."""
    
    # Embed query and all sentences
    query_vec = embed(query)
    sentences = [s for doc in documents for s in sent_tokenize(doc)]
    sentence_vecs = embed(sentences)
    
    # Keep only top-k most relevant sentences
    similarities = cosine_similarity([query_vec], sentence_vecs)[0]
    top_indices = np.argsort(similarities)[-MAX_SENTENCES:]
    
    return " ".join([sentences[i] for i in top_indices])
```

### Compression Savings

| Technique | Token Reduction | Quality Impact | Use Case |
|-----------|----------------|---------------|----------|
| LLMLingua-2 | 50-70% | Minimal (<5%) | Long context |
| History summarization | 60-80% | Moderate (10%) | Chat agents |
| Structured extraction | 70-90% | Low (5%) | RAG systems |
| Selective token removal | 30-50% | Minimal (<3%) | System prompts |

---

## Model Routing for Cost Optimization

Not every query needs a top-tier model. Route intelligently:

### Router Architecture

```
Query → Classifier → ┌── Simple (<100 tokens, factual) → gpt-4o-mini ($0.15/M)
                     ├── Medium (analysis) → gpt-4o ($2.50/M)
                     ├── Complex (reasoning) → claude-4-opus ($15/M)
                     └── Code/Technical → gpt-4o ($2.50/M)
```

### Implementation

```python
class ModelRouter:
    def __init__(self):
        self.routes = {
            "simple": {"model": "gpt-4o-mini", "cost_per_1k": 0.00015},
            "medium": {"model": "gpt-4o", "cost_per_1k": 0.0025},
            "complex": {"model": "claude-4-opus", "cost_per_1k": 0.015},
            "code": {"model": "gpt-4o", "cost_per_1k": 0.0025},
        }
        self.router_model = "gpt-4o-mini"  # Cheap router
    
    def classify(self, query: str, context_tokens: int) -> str:
        """Classify query complexity."""
        
        # Heuristic shortcuts
        if context_tokens > 50000:
            return "complex"
        if len(query.split()) < 10:
            return "simple"
        
        # ML-based classification
        response = openai.chat.completions.create(
            model=self.router_model,
            messages=[{
                "role": "system",
                "content": "Classify as: simple, medium, complex, or code. Reply with one word."
            }, {
                "role": "user", 
                "content": query
            }],
            max_tokens=5
        )
        
        route = response.choices[0].message.content.strip().lower()
        return route if route in self.routes else "medium"
    
    def route(self, query: str, context: str):
        route = self.classify(query, count_tokens(context))
        model = self.routes[route]
        
        response = openai.chat.completions.create(model=model["model"], ...)
        return response, model["cost_per_1k"]
```

### Cost Comparison

| Routing Strategy | Avg Cost/Query | Quality | Complexity |
|-----------------|---------------|---------|------------|
| All gpt-4o | $0.015 | High | None |
| All gpt-4o-mini | $0.001 | Low | None |
| Heuristic routing | $0.004 | Medium | Low |
| ML-based routing | $0.005 | High | Medium |
| Cascade (try cheap, escalate) | $0.003 | High | Medium |

---

## Token Budget Management

### Per-Feature Budgets

```python
TOKEN_BUDGETS = {
    "customer_support": {
        "max_input": 4000,
        "max_output": 500,
        "model_tier": "medium"
    },
    "document_analysis": {
        "max_input": 32000, 
        "max_output": 2000,
        "model_tier": "complex"
    },
    "simple_classification": {
        "max_input": 500,
        "max_output": 50,
        "model_tier": "simple"
    }
}

def apply_token_budget(messages, feature):
    budget = TOKEN_BUDGETS[feature]
    
    # Truncate oldest messages if exceeding budget
    while count_tokens(messages) > budget["max_input"]:
        if messages[0]["role"] == "system":
            # Keep system prompt, truncate user/assistant
            messages.pop(1)  # Remove oldest user message
        else:
            messages.pop(0)
    
    # Set max_tokens in API call
    return messages, budget
```

---

## Caching Architecture Patterns

### Pattern 1: Two-Layer Cache

```
Query → L1: Exact Match Cache (Redis, TTL=24h)
         ↓ Miss
         L2: Semantic Cache (Vector DB, threshold=0.92)
         ↓ Miss
         LLM Inference → Populate L1 + L2
```

### Pattern 2: Provider Cache + Your Cache

```
Query → Provider Cache (KV prefix cache, free)
         ↓ (partial hit — input tokens discounted)
         API call to provider
         ↓
         Response → Semantic Cache (your side, full response caching)
```

### Pattern 3: Distributed Cache for Multi-Region

```yaml
cache:
  backend: redis
  cluster: true
  nodes:
    - redis://cache-us:6379
    - redis://cache-eu:6379
    - redis://cache-asia:6379
  
  replication: async  # Eventually consistent
  
  ttl:
    exact_match: 86400  # 24h
    semantic: 3600       # 1h
    kv_cache: 300        # 5min
  
  eviction: lru
  max_memory: 16gb
```

---

## Cost Monitoring & Analytics

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

llm_requests_total = Counter(
    'llm_requests_total', 'Total LLM requests',
    ['model', 'feature', 'cache_hit']
)

llm_cost_total = Counter(
    'llm_cost_total', 'Total LLM cost in USD',
    ['model', 'feature']
)

llm_tokens_total = Counter(
    'llm_tokens_total', 'Total tokens consumed',
    ['model', 'type']  # type: input, output, cached_input
)

cache_hit_ratio = Gauge(
    'cache_hit_ratio', 'Cache hit ratio',
    ['cache_type']
)

def track_llm_call(model, feature, tokens_input, tokens_output, cached_tokens=0, cache_hit=False):
    llm_requests_total.labels(model=model, feature=feature, cache_hit=str(cache_hit)).inc()
    llm_tokens_total.labels(model=model, type="input").inc(tokens_input)
    llm_tokens_total.labels(model=model, type="output").inc(tokens_output)
    
    if cached_tokens:
        llm_tokens_total.labels(model=model, type="cached_input").inc(cached_tokens)
    
    cost = calculate_cost(model, tokens_input, tokens_output, cached_tokens)
    llm_cost_total.labels(model=model, feature=feature).inc(cost)
```

### Grafana Dashboard

```json
{
  "panels": [
    {
      "title": "Cost Per Day",
      "type": "graph",
      "targets": [
        {
          "expr": "sum(increase(llm_cost_total[24h]))",
          "legendFormat": "Total Cost"
        }
      ]
    },
    {
      "title": "Cache Hit Rate",
      "type": "gauge",
      "targets": [
        {
          "expr": "cache_hit_ratio{cache_type='semantic'}",
          "legendFormat": "Semantic Cache"
        }
      ]
    },
    {
      "title": "Cost by Model",
      "type": "pie",
      "targets": [
        {
          "expr": "sum by (model) (increase(llm_cost_total[24h]))",
          "legendFormat": "{{model}}"
        }
      ]
    }
  ]
}
```

---

## Benchmarks & Savings Analysis

### Real-World Results

| Company/Use Case | Implementation | Monthly Cost Before | After | Savings |
|-----------------|---------------|-------------------|-------|---------|
| E-commerce support | Prefix caching + routing | $45,000 | $13,500 | 70% |
| SaaS analytics | Semantic cache (GPTCache) | $12,000 | $3,600 | 70% |
| Legal document review | Prompt compression | $28,000 | $11,200 | 60% |
| Code generation | Model routing | $22,000 | $8,800 | 60% |
| Content moderation | Exact match cache | $37,000 | $14,800 | 60% |
| Customer email triage | Full stack (all techniques) | $18,000 | $4,500 | 75% |

### Cumulative Savings Potential

```
Unoptimized: $100,000/month
├── Prompt caching          → -$30,000
├── Semantic caching        → -$15,000  (cumulative: -$45,000)
├── Model routing           → -$10,000  (cumulative: -$55,000)
├── Prompt compression      → -$8,000   (cumulative: -$63,000)
├── Token budgets           → -$5,000   (cumulative: -$68,000)
├── Monitoring + iteration  → -$2,000   (cumulative: -$70,000)
                                ─────────
Optimized: $30,000/month     = 70% savings
```

---

## Production Checklist

- [ ] Implement provider prompt caching (Anthropic/OpenAI prefix cache)
- [ ] Add semantic cache with vector DB (GPTCache, custom Redis)
- [ ] Set up model routing (heuristic or ML-based tier assignment)
- [ ] Apply prompt compression for long-context queries
- [ ] Implement token budgets per feature/endpoint
- [ ] Add distributed KV cache with OpenSquilla or similar
- [ ] Monitor cache hit rates and cost trends
- [ ] Set up budget alerts and anomaly detection
- [ ] A/B test caching impact on quality
- [ ] Document cache invalidation strategy
- [ ] Plan for cache warm-up after deployments
- [ ] Review and tune similarity thresholds monthly

---

## Cross-References

- **Local Inference** → [23-Local-AI-Inference-Self-Hosting/03-Ollama-Local-Inference.md](../23-Local-AI-Inference-Self-Hosting/03-Ollama-Local-Inference.md) — Local model cost
- **Structured Output** → [06-Advanced/12-Structured-Output-Controlled-Generation.md](../06-Advanced/12-Structured-Output-Controlled-Generation.md) — Token-efficient generation
- **Real-Time Systems** → [13-Top-Demand/11-Real-Time-AI-Systems.md](../13-Top-Demand/11-Real-Time-AI-Systems.md) — Streaming optimization
- **Agent Cost Tracking** → [20-Agent-Infrastructure-and-Observability/05-Agent-Cost-Tracking-and-Optimization.md](../20-Agent-Infrastructure-and-Observability/05-Agent-Cost-Tracking-and-Optimization.md) — Agent cost optimization
- **Fine-Tuning** → [13-Top-Demand/07-Fine-Tuning-Custom-Models.md](../13-Top-Demand/07-Fine-Tuning-Custom-Models.md) — Fine-tuning to reduce token usage
- **Business Models** → [16-AI-Business-Models-Playbooks/02-AI-SaaS-Playbook.md](../16-AI-Business-Models-Playbooks/02-AI-SaaS-Playbook.md) — Unit economics of AI SaaS

---

*Last updated: June 2026 | 450+ lines covering prompt caching, semantic caching, model routing, compression, and cost optimization patterns*

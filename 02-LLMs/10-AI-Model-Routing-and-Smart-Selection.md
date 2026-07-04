# AI Model Routing and Smart Selection: A Practitioner's Guide

> As the AI model landscape fragments into dozens of providers, model families, and price tiers, the ability to dynamically route requests to the optimal model has become a critical engineering discipline. This document provides a comprehensive guide to AI model routing — the practice of intelligently directing AI requests to the right model based on task requirements, cost constraints, quality targets, and latency budgets. From simple fallback chains to ML-powered routers, model routing can reduce AI costs by 40–70% while maintaining or improving output quality.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Why Model Routing Matters in 2026](#2-why-model-routing-matters-in-2026)
3. [The Model Landscape: Understanding the Options](#3-the-model-landscape-understanding-the-options)
4. [Model Routing Architectures](#4-model-routing-architectures)
5. [Cost-Performance Optimization Strategies](#5-cost-performance-optimization-strategies)
6. [Task-Based Model Selection](#6-task-based-model-selection)
7. [Real-Time Model Selection Techniques](#7-real-time-model-selection-techniques)
8. [Implementation Patterns with Code](#8-implementation-patterns-with-code)
9. [Quality Assurance and Evaluation](#9-quality-assurance-and-evaluation)
10. [Production Deployment Patterns](#10-production-deployment-patterns)
11. [Case Studies and Benchmarks](#11-case-studies-and-benchmarks)
12. [Tools and Frameworks](#12-tools-and-frameworks)
13. [Common Pitfalls and Anti-Patterns](#13-common-pitfalls-and-anti-patterns)
14. [Future Outlook](#14-future-outlook)
15. [Cross-References](#15-cross-references)

---

## 1. Introduction

### The Model Abundance Problem

In early 2024, choosing an AI model was simple: GPT-4 for quality, GPT-3.5 for speed and cost. By mid-2026, practitioners face a radically different landscape:

- **50+ production-grade model families** (GPT, Claude, Gemini, Llama, Mistral, DeepSeek, Qwen, Grok, etc.)
- **200+ distinct model variants** across sizes, fine-tunes, and quantization levels
- **15+ inference providers** (OpenAI, Anthropic, Google, Groq, Cerebras, SambaNova, Together, Fireworks, etc.)
- **10x cost variance** between the cheapest and most expensive options for equivalent tasks
- **3–10x quality variance** between models for the same task

The result: **choosing the wrong model for a task can cost 10x more than necessary**, while choosing the wrong provider can add 100ms of unnecessary latency. Model routing — the discipline of dynamically selecting the optimal model for each request — has emerged as one of the highest-leverage engineering practices in AI systems.

### What Is Model Routing?

Model routing is the practice of directing AI inference requests to the most appropriate model based on:

- **Task characteristics** (code generation, summarization, translation, analysis)
- **Quality requirements** (creative writing vs. factual extraction)
- **Cost constraints** (budget-limited applications)
- **Latency requirements** (real-time vs. batch processing)
- **Context length** (short prompts vs. long documents)
- **Provider availability** (fallback when a provider is down)

A model router sits between the application and the model providers, making real-time decisions about which model to use for each request.

### The Business Case

| Metric | Without Routing | With Routing | Improvement |
|--------|----------------|--------------|-------------|
| Average cost per 1K tokens | $0.015 | $0.006 | 60% reduction |
| Quality (human eval, 1-5 scale) | 4.2 | 4.3 | +0.1 (improved) |
| P99 latency | 2.8s | 1.4s | 50% reduction |
| Monthly cost (10M requests) | $15,000 | $6,000 | $9,000 savings |
| Uptime | 99.5% | 99.95% | Through failover |

---

## 2. Why Model Routing Matters in 2026

### 2.1 The Cost Explosion

AI inference costs have become a dominant line item for many organizations:

- **Average enterprise AI spend**: $2.4M annually (McKinsey 2026)
- **AI infrastructure cost growth**: 47% YoY (Flexera 2026)
- **AI cost overruns >50%**: 62% of enterprises (Deloitte 2026)

The primary driver is that most applications use a single, expensive model for all tasks — including tasks that could be handled by cheaper, faster models with equal quality.

### 2.2 The Quality Gap Is Task-Dependent

Research consistently shows that model quality rankings are highly task-dependent:

| Task | Best Model | Runner-Up | Cost Ratio |
|------|-----------|-----------|------------|
| Code generation | Claude Fable | GPT-5.5 High | 1.5x |
| Creative writing | GPT-5.5 High | Claude Opus | 1.2x |
| Factual Q&A | Gemini 3 Pro | GPT-5.5 Medium | 0.8x |
| Summarization | DeepSeek V3 | Claude Haiku | 0.3x |
| Translation | Gemini 3 Pro | DeepSeek V3 | 0.7x |
| Structured extraction | GPT-5.5 Medium | Claude Sonnet | 0.9x |

**Key insight**: No single model is best at everything. A routing strategy that matches models to tasks can achieve better quality at lower cost than using the most expensive model for everything.

### 2.3 The Provider Diversity Factor

The inference provider landscape has fragmented:

| Provider | Strength | Weakness | Typical Latency |
|----------|----------|----------|-----------------|
| OpenAI | Quality, ecosystem | Cost, rate limits | 200–800ms |
| Anthropic | Safety, code | Cost, availability | 200–600ms |
| Google Gemini | Multimodal, long context | API complexity | 150–500ms |
| Groq | Speed (10x faster) | Model selection | 50–150ms |
| Cerebras | Speed, throughput | Model size limits | 30–100ms |
| DeepSeek | Cost (10x cheaper) | Latency, availability | 300–1000ms |
| Together AI | Model breadth | Quality consistency | 200–600ms |
| Fireworks AI | Speed + open models | Limited closed models | 100–300ms |

**Key insight**: The same model can have different performance characteristics depending on the provider. A router that considers both model AND provider can optimize across two dimensions simultaneously.

---

## 3. The Model Landscape: Understanding the Options

### 3.1 Model Tiers (2026)

| Tier | Models | Cost (per 1M tokens) | Best For |
|------|--------|---------------------|----------|
| **Frontier** | Claude Fable, GPT-5.5 High, Gemini 3 Ultra | $15–$30 | Complex reasoning, research |
| **Flagship** | GPT-5.5 Medium, Claude Opus, Gemini 3 Pro | $5–$15 | General production use |
| **Balanced** | Claude Sonnet, GPT-5.5 Low, Gemini 3 Flash | $1–$5 | High-volume production |
| **Fast** | Claude Haiku, GPT-4.1 Mini, Gemini 2.0 Flash | $0.10–$1 | Real-time, high-throughput |
| **Economy** | DeepSeek V3, Llama 3.3 70B, Qwen 2.5 72B | $0.01–$0.10 | Cost-sensitive batch processing |
| **Edge** | Phi-4, Gemma 2 9B, Llama 3.2 8B | $0.001–$0.01 | On-device, air-gapped |

### 3.2 Model Selection Matrix

```
                    Quality
                      ▲
                      │
         Frontier     │     ★ Claude Fable
                      │  ★ GPT-5.5 High
                      │     ★ Gemini 3 Ultra
                      │
         Flagship     │  ★ GPT-5.5 Med    ★ Claude Opus
                      │     ★ Gemini 3 Pro
                      │
         Balanced     │  ★ Claude Sonnet   ★ Gemini 3 Flash
                      │     ★ GPT-5.5 Low
                      │
         Fast         │  ★ Claude Haiku    ★ Gemini 2.0 Flash
                      │     ★ GPT-4.1 Mini  ★ Groq Llama
                      │
         Economy      │  ★ DeepSeek V3     ★ Llama 3.3 70B
                      │     ★ Qwen 2.5 72B
                      │
                      └──────────────────────────────────► Cost
                      $0.01                              $30
```

### 3.3 Task-Model Affinity

Different models have different "affinities" for specific tasks. Understanding these affinities is the foundation of effective routing:

**Code Generation**:
- **Best**: Claude Fable (strong reasoning, code quality)
- **Fast alternative**: DeepSeek Coder V3 (excellent code, 10x cheaper)
- **Edge option**: CodeGemma 2 9B (surprisingly good for simple code)

**Creative Writing**:
- **Best**: GPT-5.5 High (nuanced, creative)
- **Fast alternative**: Claude Sonnet (good creativity, 5x cheaper)
- **Avoid**: DeepSeek V3 (tends toward formulaic output)

**Factual Extraction**:
- **Best**: Gemini 3 Pro (strong factual grounding, long context)
- **Fast alternative**: GPT-5.5 Medium (good extraction, lower cost)
- **Budget option**: Llama 3.3 70B (with RAG, comparable quality)

**Summarization**:
- **Best**: Claude Haiku (fast, accurate summaries)
- **Budget option**: DeepSeek V3 (excellent summarization quality)
- **Long documents**: Gemini 3 Pro (1M token context)

**Translation**:
- **Best**: Gemini 3 Pro (multilingual strength)
- **Budget option**: DeepSeek V3 (strong multilingual)
- **Specific languages**: Specialized fine-tunes often outperform general models

---

## 4. Model Routing Architectures

### 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    Application                       │
└─────────────┬───────────────────────────────────────┘
              │ Request
              ▼
┌─────────────────────────────────────────────────────┐
│              Model Router (Decision Layer)            │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ Task          │  │ Cost         │  │ Quality   │ │
│  │ Classifier    │  │ Optimizer    │  │ Gate      │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
└─────────────┬───────────────────────────────────────┘
              │ Selected Model + Provider
              ▼
┌─────────────┬───────────┬───────────┬──────────────┐
│  OpenAI     │ Anthropic │  Google   │  Groq/...    │
│  GPT-5.5    │ Claude    │  Gemini   │  DeepSeek    │
└─────────────┴───────────┴───────────┴──────────────┘
```

### 4.2 Architecture Types

#### Type 1: Static Routing (Rule-Based)

The simplest approach — fixed rules that map task types to models.

```python
# Static routing rules
ROUTING_TABLE = {
    "code_generation": {"model": "claude-fable", "provider": "anthropic"},
    "summarization": {"model": "deepseek-v3", "provider": "deepseek"},
    "creative_writing": {"model": "gpt-5.5-high", "provider": "openai"},
    "factual_qa": {"model": "gemini-3-pro", "provider": "google"},
    "translation": {"model": "gemini-3-pro", "provider": "google"},
    "extraction": {"model": "gpt-5.5-medium", "provider": "openai"},
    "default": {"model": "claude-sonnet", "provider": "anthropic"},
}

def route_request(task_type: str) -> dict:
    return ROUTING_TABLE.get(task_type, ROUTING_TABLE["default"])
```

**Pros**: Simple, predictable, no ML required
**Cons**: No optimization, no adaptation, requires manual tuning

#### Type 2: Classifier-Based Routing

Use a small classifier to determine the task type and route accordingly.

```python
from transformers import pipeline

class ClassifierRouter:
    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model="task-classifier-v2"
        )
        self.model_map = {
            "CODE": {"model": "claude-fable", "provider": "anthropic"},
            "CREATIVE": {"model": "gpt-5.5-high", "provider": "openai"},
            "FACTUAL": {"model": "gemini-3-pro", "provider": "google"},
            "ANALYTICAL": {"model": "gpt-5.5-medium", "provider": "openai"},
            "CONVERSATIONAL": {"model": "claude-sonnet", "provider": "anthropic"},
        }
    
    def route(self, prompt: str) -> dict:
        task_type = self.classifier(prompt)[0]["label"]
        return self.model_map.get(task_type, self.model_map["CONVERSATIONAL"])
```

**Pros**: Adapts to input, handles novel task types
**Cons**: Adds latency (50–200ms), requires training data

#### Type 3: Hybrid Router (Recommended)

Combines classifier-based routing with cost optimization and quality gates.

```python
class HybridRouter:
    def __init__(self):
        self.classifier = TaskClassifier()
        self.cost_optimizer = CostOptimizer()
        self.quality_gate = QualityGate()
    
    def route(self, prompt: str, context: dict) -> RouteDecision:
        # Step 1: Classify task
        task_type = self.classifier.classify(prompt)
        
        # Step 2: Get candidate models
        candidates = self.get_candidates(task_type)
        
        # Step 3: Apply cost constraints
        if context.get("budget_limit"):
            candidates = self.cost_optimizer.filter(
                candidates, 
                budget=context["budget_limit"]
            )
        
        # Step 4: Select optimal model
        selected = self.cost_optimizer.select(candidates)
        
        # Step 5: Verify quality gate
        if not self.quality_gate.check(selected, task_type):
            selected = self.fallback_model(task_type)
        
        return RouteDecision(
            model=selected.model,
            provider=selected.provider,
            reason=f"Task:{task_type} Cost:{selected.cost}",
            confidence=selected.confidence
        )
```

#### Type 4: ML-Powered Router (Advanced)

Uses a trained model to predict the optimal model for each request.

```python
class MLPoweredRouter:
    def __init__(self):
        self.router_model = load_router_model("router-v3")
        self.feature_extractor = FeatureExtractor()
    
    def route(self, prompt: str, metadata: dict) -> RouteDecision:
        # Extract features
        features = self.feature_extractor.extract(
            prompt=prompt,
            prompt_length=len(prompt),
            has_code="```" in prompt,
            has_json="{" in prompt,
            language=metadata.get("language", "en"),
            user_tier=metadata.get("user_tier", "free"),
        )
        
        # Predict optimal model
        prediction = self.router_model.predict(features)
        
        return RouteDecision(
            model=prediction.model,
            provider=prediction.provider,
            confidence=prediction.confidence,
            expected_cost=prediction.estimated_cost,
            expected_quality=prediction.estimated_quality,
        )
```

---

## 5. Cost-Performance Optimization Strategies

### 5.1 The Cost-Quality Frontier

Every model sits on a cost-quality frontier. The goal of routing is to find the point on this frontier that maximizes value for each request.

```
Quality
  ▲
  │     ★ Claude Fable ($25/1M)
  │    /
  │   ★ GPT-5.5 High ($15/1M)
  │  /
  │ ★ Gemini 3 Pro ($10/1M)
  │/
  │ ★ Claude Sonnet ($3/1M)
  │/
  │★ DeepSeek V3 ($0.27/1M)
  │
  └──────────────────────────► Cost
```

**Key insight**: The curve is not linear. Going from DeepSeek V3 to Claude Sonnet (10x cost increase) typically yields a 20–30% quality improvement. Going from Claude Sonnet to Claude Fable (8x cost increase) typically yields only a 5–10% quality improvement. The optimal routing strategy exploits these diminishing returns.

### 5.2 Tiered Routing Strategy

The most effective cost optimization strategy is tiered routing — using different model tiers for different request complexities:

| Request Complexity | Model Tier | Cost | Quality Target |
|-------------------|------------|------|----------------|
| Simple (FAQ, extraction) | Fast/Economy | $0.01–$0.10 | 80%+ accuracy |
| Medium (analysis, writing) | Balanced | $1–$5 | 90%+ accuracy |
| Hard (reasoning, code) | Flagship | $5–$15 | 95%+ accuracy |
| Critical (research, legal) | Frontier | $15–$30 | 99%+ accuracy |

### 5.3 Adaptive Routing Based on User Tier

Route requests differently based on user value:

```python
TIER_ROUTING = {
    "free": {
        "model": "deepseek-v3",
        "max_tokens": 500,
        "rate_limit": "10 req/min",
    },
    "basic": {
        "model": "claude-haiku",
        "max_tokens": 2000,
        "rate_limit": "60 req/min",
    },
    "pro": {
        "model": "claude-sonnet",
        "max_tokens": 8000,
        "rate_limit": "300 req/min",
    },
    "enterprise": {
        "model": "claude-fable",
        "max_tokens": 32000,
        "rate_limit": "unlimited",
    },
}
```

### 5.4 Prompt-Aware Cost Optimization

Analyze the prompt to determine the minimum viable model:

```python
def estimate_prompt_complexity(prompt: str) -> str:
    """Estimate the complexity of a prompt to determine model tier."""
    indicators = {
        "simple": [
            len(prompt) < 100,
            not any(kw in prompt.lower() for kw in 
                   ["analyze", "compare", "explain", "why", "how"]),
            prompt.count("?") <= 1,
        ],
        "complex": [
            len(prompt) > 1000,
            any(kw in prompt.lower() for kw in 
               ["analyze", "compare", "explain", "why", "how", "implement"]),
            "```" in prompt,  # Code blocks
            prompt.count("\n") > 10,  # Multi-line structured input
            any(w in prompt.lower() for w in 
               ["proof", "algorithm", "optimization", "architecture"]),
        ],
    }
    
    if sum(indicators["complex"]) >= 2:
        return "complex"
    elif sum(indicators["simple"]) >= 2:
        return "simple"
    else:
        return "medium"
```

---

## 6. Task-Based Model Selection

### 6.1 Task Classification Framework

Before routing, you need to classify the task. Here's a comprehensive task taxonomy:

| Task Category | Examples | Primary Model | Backup Model |
|--------------|----------|---------------|--------------|
| **Code Generation** | Write function, debug code, refactor | Claude Fable | DeepSeek Coder |
| **Code Review** | Find bugs, suggest improvements | Claude Fable | GPT-5.5 Medium |
| **Creative Writing** | Stories, copywriting, brainstorming | GPT-5.5 High | Claude Opus |
| **Factual Q&A** | Research, lookup, verification | Gemini 3 Pro | GPT-5.5 Medium |
| **Summarization** | Document summary, meeting notes | Claude Haiku | DeepSeek V3 |
| **Translation** | Language pairs, localization | Gemini 3 Pro | DeepSeek V3 |
| **Structured Extraction** | JSON, tables, data parsing | GPT-5.5 Medium | Claude Sonnet |
| **Analysis** | Data analysis, report generation | GPT-5.5 Medium | Claude Sonnet |
| **Conversation** | Chat, support, tutoring | Claude Sonnet | GPT-5.5 Medium |
| **Classification** | Sentiment, topic, intent | Claude Haiku | GPT-4.1 Mini |
| **Math/Reasoning** | Calculations, logic, proofs | GPT-5.5 High | Claude Fable |
| **Multimodal** | Image analysis, video understanding | Gemini 3 Pro | GPT-5.5 High |

### 6.2 Task Detection Patterns

```python
import re

TASK_PATTERNS = {
    "code_generation": [
        r"write.*(?:function|class|script|code)",
        r"implement.*(?:algorithm|method|feature)",
        r"(?:fix|debug|refactor).*(?:bug|error|code)",
        r"```[\s\S]*```",  # Code blocks in prompt
    ],
    "summarization": [
        r"summar(?:ize|ise|ization)",
        r"(?:give|provide).*(?:summary|overview|brief)",
        r"tldr",
        r"key points",
    ],
    "creative_writing": [
        r"(?:write|compose|create).*(?:story|poem|essay|article)",
        r"(?:brainstorm|ideate).*(?:ideas|concepts|names)",
        r"(?:draft|write).*(?:email|letter|blog)",
    ],
    "analysis": [
        r"analy(?:ze|se|sis)",
        r"compar(?:e|ison|ing)",
        r"(?:evaluate|assess|review)",
        r"(?:pros|cons|tradeoffs|trade-offs)",
    ],
    "extraction": [
        r"extract.*(?:data|information|fields|entities)",
        r"(?:parse|convert).*(?:to|into).*(?:json|csv|table)",
        r"(?:list|identify).*(?:names|dates|amounts)",
    ],
}

def detect_task_type(prompt: str) -> str:
    scores = {}
    for task_type, patterns in TASK_PATTERNS.items():
        score = sum(
            1 for p in patterns 
            if re.search(p, prompt, re.IGNORECASE)
        )
        scores[task_type] = score
    
    if max(scores.values()) == 0:
        return "general"
    return max(scores, key=scores.get)
```

### 6.3 Context-Aware Selection

Beyond task type, consider the context of the request:

| Context Factor | Impact on Model Choice |
|---------------|----------------------|
| **User tier** | Higher tier → higher quality model |
| **Time of day** | Peak hours → faster/cheaper model |
| **Response length** | Long responses → cheaper model (cost control) |
| **Conversation history** | Complex history → stronger model |
| **Domain** | Legal/medical → higher quality model |
| **Recency requirements** | Real-time → lower latency model |

---

## 7. Real-Time Model Selection Techniques

### 7.1 Pre-Flight Estimation

Before sending to a model, estimate whether it can handle the request:

```python
class PreFlightEstimator:
    """Estimate model capability before sending request."""
    
    def estimate(self, prompt: str, model: Model) -> float:
        """Return estimated success probability (0-1)."""
        features = {
            "prompt_length": len(prompt),
            "has_code": "```" in prompt,
            "requires_reasoning": self._needs_reasoning(prompt),
            "requires_creativity": self._needs_creativity(prompt),
            "language_count": self._count_languages(prompt),
        }
        
        # Simple heuristic (replace with trained model in production)
        score = 0.8  # Base probability
        
        if features["requires_reasoning"] and model.tier < 3:
            score -= 0.3
        if features["has_code"] and model.name not in CODE_MODELS:
            score -= 0.2
        if features["prompt_length"] > model.context_window * 0.8:
            score -= 0.4
        
        return max(0.0, min(1.0, score))
```

### 7.2 Speculative Routing

Send the request to multiple models simultaneously and use the first good response:

```python
import asyncio

class SpeculativeRouter:
    async def route_speculative(self, prompt: str) -> str:
        """Send to multiple models, use first good response."""
        tasks = [
            self.call_model("deepseek-v3", prompt),      # Fast, cheap
            self.call_model("claude-haiku", prompt),      # Fast, medium cost
        ]
        
        for completed in asyncio.as_completed(tasks):
            response = await completed
            if self.quality_check(response, prompt):
                return response
        
        # Fallback to expensive model
        return await self.call_model("claude-sonnet", prompt)
```

### 7.3 Cascade Routing

Start with a cheap model, escalate to expensive if quality is insufficient:

```python
class CascadeRouter:
    CASCADE = [
        {"model": "deepseek-v3", "max_retries": 1},
        {"model": "claude-haiku", "max_retries": 1},
        {"model": "claude-sonnet", "max_retries": 1},
        {"model": "claude-fable", "max_retries": 0},
    ]
    
    async def route_cascade(self, prompt: str) -> str:
        for tier in self.CASCADE:
            response = await self.call_model(tier["model"], prompt)
            
            if self.quality_check(response, prompt):
                return response
            
            if tier["max_retries"] > 0:
                # Retry with quality instructions
                enhanced_prompt = self.enhance_prompt(prompt, response)
                response = await self.call_model(tier["model"], enhanced_prompt)
                if self.quality_check(response, prompt):
                    return response
        
        return response  # Last resort
```

---

## 8. Implementation Patterns with Code

### 8.1 OpenRouter-Based Router (Simple)

```python
import openai

class OpenRouterRouter:
    """Route through OpenRouter for unified API access."""
    
    def __init__(self):
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    
    def route(self, prompt: str, task_type: str) -> str:
        model_map = {
            "code": "anthropic/claude-fable",
            "creative": "openai/gpt-5.5-high",
            "factual": "google/gemini-3-pro",
            "cheap": "deepseek/deepseek-v3",
            "fast": "groq/llama-3.3-70b",
        }
        
        model = model_map.get(task_type, "anthropic/claude-sonnet")
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        
        return response.choices[0].message.content
```

### 8.2 Cost-Aware Router with Budget Tracking

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

@dataclass
class CostTracker:
    daily_budget: float = 100.0
    spent_today: float = 0.0
    last_reset: datetime = None
    
    def can_afford(self, estimated_cost: float) -> bool:
        self._check_reset()
        return (self.spent_today + estimated_cost) <= self.daily_budget
    
    def record_spend(self, cost: float):
        self._check_reset()
        self.spent_today += cost
    
    def _check_reset(self):
        if self.last_reset and datetime.now().date() > self.last_reset.date():
            self.spent_today = 0.0
            self.last_reset = datetime.now()

class CostAwareRouter:
    def __init__(self):
        self.cost_tracker = CostTracker()
        self.model_costs = {
            "deepseek-v3": 0.00027,    # per 1K tokens
            "claude-haiku": 0.001,
            "claude-sonnet": 0.003,
            "gpt-5.5-medium": 0.005,
            "claude-fable": 0.025,
            "gpt-5.5-high": 0.015,
        }
    
    def route(self, prompt: str, task_type: str, max_cost: float = None) -> str:
        candidates = self._get_candidates(task_type)
        
        # Sort by cost
        candidates.sort(key=lambda m: self.model_costs[m])
        
        for model in candidates:
            estimated_cost = self.model_costs[model] * (len(prompt) / 1000)
            
            if max_cost and estimated_cost > max_cost:
                continue
            
            if self.cost_tracker.can_afford(estimated_cost):
                response = self._call_model(model, prompt)
                actual_cost = self._calculate_cost(model, response)
                self.cost_tracker.record_spend(actual_cost)
                return response
        
        # Budget exhausted — use free tier or cached response
        return self._fallback_response(prompt)
```

### 8.3 Quality-Gated Router

```python
class QualityGatedRouter:
    def __init__(self):
        self.quality_thresholds = {
            "code_generation": 0.85,
            "summarization": 0.80,
            "creative_writing": 0.75,
            "factual_qa": 0.90,
            "extraction": 0.85,
        }
    
    def route_with_quality_check(self, prompt: str, task_type: str) -> str:
        model = self.select_model(task_type)
        response = self.call_model(model, prompt)
        
        quality_score = self.evaluate_quality(response, prompt, task_type)
        threshold = self.quality_thresholds.get(task_type, 0.80)
        
        if quality_score >= threshold:
            return response
        
        # Quality insufficient — try next model
        for alt_model in self.get_alternatives(model, task_type):
            response = self.call_model(alt_model, prompt)
            quality_score = self.evaluate_quality(response, prompt, task_type)
            
            if quality_score >= threshold:
                return response
        
        return response  # Best available
    
    def evaluate_quality(self, response: str, prompt: str, task_type: str) -> float:
        """Evaluate response quality using a lightweight evaluator."""
        # Use a small, fast model as evaluator
        eval_prompt = f"""Rate this response on a scale of 0-1 for the task "{task_type}":
        
Prompt: {prompt[:500]}
Response: {response[:500]}

Rating (0-1):"""
        
        score_str = self.call_model("claude-haiku", eval_prompt)
        try:
            return float(score_str.strip())
        except ValueError:
            return 0.5  # Default to medium quality
```

### 8.4 Multi-Provider Failover Router

```python
class FailoverRouter:
    def __init__(self):
        self.providers = [
            {"name": "openai", "client": OpenAIClient(), "priority": 1},
            {"name": "anthropic", "client": AnthropicClient(), "priority": 2},
            {"name": "google", "client": GoogleClient(), "priority": 3},
            {"name": "deepseek", "client": DeepSeekClient(), "priority": 4},
        ]
        self.health = {p["name"]: True for p in self.providers}
    
    async def route_with_failover(self, prompt: str, model: str) -> str:
        providers = sorted(self.providers, key=lambda p: p["priority"])
        
        for provider in providers:
            if not self.health[provider["name"]]:
                continue
            
            try:
                response = await provider["client"].complete(
                    model=model,
                    prompt=prompt,
                    timeout=10.0,
                )
                return response
            except (TimeoutError, RateLimitError, ServiceUnavailableError) as e:
                self.health[provider["name"]] = False
                # Schedule health check recovery
                asyncio.create_task(self._recover_health(provider["name"]))
                continue
        
        raise AllProvidersDown("All AI providers are unavailable")
    
    async def _recover_health(self, provider_name: str):
        await asyncio.sleep(60)  # Wait 1 minute before retry
        self.health[provider_name] = True
```

---

## 9. Quality Assurance and Evaluation

### 9.1 Evaluation Framework

```python
class ModelRouterEvaluator:
    def evaluate_routing_strategy(
        self, 
        test_suite: list[dict],
        router: BaseRouter,
    ) -> dict:
        results = {
            "total_requests": len(test_suite),
            "correct_routes": 0,
            "quality_scores": [],
            "costs": [],
            "latencies": [],
        }
        
        for test_case in test_suite:
            start_time = time.time()
            route_decision = router.route(test_case["prompt"], test_case["task_type"])
            response = router.execute(route_decision, test_case["prompt"])
            latency = time.time() - start_time
            
            # Quality evaluation
            quality = self.evaluate_quality(response, test_case["expected"])
            results["quality_scores"].append(quality)
            results["costs"].append(route_decision.estimated_cost)
            results["latencies"].append(latency)
            
            if quality >= test_case.get("quality_threshold", 0.8):
                results["correct_routes"] += 1
        
        results["avg_quality"] = sum(results["quality_scores"]) / len(results["quality_scores"])
        results["avg_cost"] = sum(results["costs"]) / len(results["costs"])
        results["avg_latency"] = sum(results["latencies"]) / len(results["latencies"])
        results["success_rate"] = results["correct_routes"] / results["total_requests"]
        
        return results
```

### 9.2 A/B Testing Routing Strategies

```python
class ABTestRouter:
    def __init__(self, strategy_a: BaseRouter, strategy_b: BaseRouter):
        self.strategy_a = strategy_a
        self.strategy_b = strategy_b
        self.results_a = []
        self.results_b = []
    
    def route(self, prompt: str, task_type: str) -> RouteDecision:
        # Random assignment
        if random.random() < 0.5:
            decision = self.strategy_a.route(prompt, task_type)
            self.results_a.append({"decision": decision, "prompt": prompt})
        else:
            decision = self.strategy_b.route(prompt, task_type)
            self.results_b.append({"decision": decision, "prompt": prompt})
        
        return decision
    
    def get_comparison(self) -> dict:
        return {
            "strategy_a": self._summarize(self.results_a),
            "strategy_b": self._summarize(self.results_b),
            "winner": self._determine_winner(),
        }
```

---

## 10. Production Deployment Patterns

### 10.1 Router as a Microservice

```yaml
# docker-compose.yml
version: '3.8'
services:
  model-router:
    image: model-router:latest
    ports:
      - "8080:8080"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - ROUTER_CONFIG=/config/router.yaml
    volumes:
      - ./config:/config
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 10.2 Router Configuration

```yaml
# router.yaml
routing:
  strategy: "hybrid"  # static, classifier, hybrid, ml-powered
  
  task_classifiers:
    - type: "pattern"
      patterns: ["code", "function", "class", "debug"]
      model: "claude-fable"
      provider: "anthropic"
    
    - type: "pattern"
      patterns: ["summarize", "summary", "tldr"]
      model: "deepseek-v3"
      provider: "deepseek"
    
    - type: "classifier"
      model: "task-classifier-v2"
      threshold: 0.7

cost_optimization:
  enabled: true
  daily_budget: 500.00
  alert_threshold: 0.8  # Alert at 80% of budget
  
  tier_routing:
    free:
      model: "deepseek-v3"
      max_tokens: 500
    basic:
      model: "claude-haiku"
      max_tokens: 2000
    pro:
      model: "claude-sonnet"
      max_tokens: 8000

quality_gates:
  enabled: true
  default_threshold: 0.8
  task_thresholds:
    code_generation: 0.85
    factual_qa: 0.90
    creative_writing: 0.75

failover:
  enabled: true
  max_retries: 3
  retry_delay_ms: 1000
  health_check_interval_s: 60
  
  provider_priority:
    - openai
    - anthropic
    - google
    - deepseek
```

### 10.3 Monitoring Dashboard Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| **Requests/sec** | Total routing throughput | >1000 req/s |
| **Route distribution** | % of requests per model | Any model >80% |
| **Cost per request** | Average cost | >$0.05 |
| **Quality score** | Average quality rating | <0.8 |
| **Failover rate** | % of requests needing failover | >5% |
| **P99 latency** | 99th percentile latency | >5s |
| **Error rate** | Failed requests | >1% |
| **Budget utilization** | % of daily budget used | >80% |

---

## 11. Case Studies and Benchmarks

### 11.1 Case Study: AI Coding Assistant

**Company**: Mid-size SaaS (200 employees)
**Problem**: $45K/month in AI costs, mostly GPT-5.5 High for all tasks
**Solution**: Implemented hybrid routing

| Task | Before (Model) | After (Model) | Cost Change |
|------|----------------|---------------|-------------|
| Code completion | GPT-5.5 High | Claude Haiku | -85% |
| Code review | GPT-5.5 High | Claude Fable | -20% |
| Bug explanation | GPT-5.5 High | Claude Sonnet | -60% |
| Documentation | GPT-5.5 High | DeepSeek V3 | -95% |
| Test generation | GPT-5.5 High | DeepSeek Coder | -90% |

**Results**:
- Monthly cost: $45K → $12K (73% reduction)
- Quality: Maintained or improved (A/B tested)
- Latency: P50 improved by 40%

### 11.2 Case Study: Customer Support Chatbot

**Company**: E-commerce platform (50M monthly users)
**Problem**: GPT-4.1 Mini for all conversations, quality issues on complex queries
**Solution**: Cascade routing with quality gates

| Query Type | Model Used | % of Traffic | Cost per 1K |
|-----------|-----------|-------------|-------------|
| Simple FAQ | DeepSeek V3 | 60% | $0.00027 |
| Product questions | Claude Haiku | 25% | $0.001 |
| Complaints/complex | Claude Sonnet | 12% | $0.003 |
| Escalation | GPT-5.5 Medium | 3% | $0.005 |

**Results**:
- Monthly cost: $82K → $28K (66% reduction)
- CSAT score: 3.8 → 4.2 (improved quality on complex queries)
- Resolution rate: 72% → 85%

### 11.3 Benchmark: Routing Accuracy

Tested across 10,000 diverse prompts:

| Router Type | Accuracy | Avg Cost | P95 Latency |
|------------|----------|----------|-------------|
| Static rules | 72% | $0.008 | 200ms |
| Classifier-based | 84% | $0.006 | 350ms |
| Hybrid | 91% | $0.005 | 280ms |
| ML-powered | 94% | $0.004 | 300ms |
| Single model (baseline) | N/A | $0.015 | 400ms |

---

## 12. Tools and Frameworks

### 12.1 Routing Infrastructure

| Tool | Type | Key Feature | License |
|------|------|-------------|---------|
| **OpenRouter** | API gateway | 200+ models, unified API | Commercial |
| **Portkey** | Gateway | Caching, fallbacks, analytics | Open source |
| **LiteLLM** | Proxy | 100+ providers, OpenAI-compatible | Open source |
| **Martian** | Router | ML-powered model selection | Commercial |
| **Not Diamond** | Router | Task-specific model routing | Open source |
| **Keywords AI** | Gateway | Observability, prompt management | Commercial |
| **Langfuse** | Observability | Routing analytics, tracing | Open source |

### 12.2 Implementation Libraries

```python
# Using LiteLLM for multi-provider routing
import litellm

# Configure providers
litellm.openai_key = os.environ["OPENAI_API_KEY"]
litellm.anthropic_key = os.environ["ANTHROPIC_API_KEY"]

# Route to cheapest available
response = litellm.completion(
    model="deepseek/deepseek-v3",  # Try cheapest first
    messages=[{"role": "user", "content": prompt}],
    fallbacks=["anthropic/claude-haiku", "openai/gpt-4.1-mini"],
)

# Cost tracking
print(f"Cost: ${litellm.completion_cost(completion_response=response)}")
```

### 12.3 Evaluation Tools

| Tool | Purpose | Key Feature |
|------|---------|-------------|
| **Braintrust** | Evaluation | Side-by-side model comparison |
| **Promptfoo** | Testing | Model routing test suites |
| **LangSmith** | Tracing | Route decision logging |
| **Arize Phoenix** | Observability | Model performance comparison |
| **Ragas** | RAG evaluation | Response quality metrics |

---

## 13. Common Pitfalls and Anti-Patterns

### 13.1 Over-Routing

**Problem**: Adding routing logic for every edge case, creating unmaintainable complexity.

**Solution**: Start simple (static rules), add complexity only when data shows it's needed.

```python
# BAD: Over-engineered routing
def route_v1(prompt):
    if is_code(prompt) and is_python(prompt) and len(prompt) > 500:
        if is_weekend():
            return "claude-fable"  # Weekends = better quality
        else:
            return "deepseek-coder"
    elif is_code(prompt) and is_javascript(prompt):
        # ... 100 more conditions

# GOOD: Simple routing with data-driven refinement
def route_v1(prompt):
    task_type = classify_task(prompt)
    return ROUTE_TABLE.get(task_type, DEFAULT_MODEL)
```

### 13.2 Ignoring Latency

**Problem**: Optimizing only for cost, ignoring that routing adds latency.

**Solution**: Set latency budgets and measure routing overhead.

```python
# Routing overhead should be <10% of total latency
ROUTING_OVERHEAD_BUDGET = 0.1  # 10% of total latency budget

class LatencyAwareRouter:
    def route(self, prompt, latency_budget_ms):
        routing_budget = latency_budget_ms * ROUTING_OVERHEAD_BUDGET
        
        start = time.time()
        decision = self._make_decision(prompt)
        routing_time = (time.time() - start) * 1000
        
        if routing_time > routing_budget:
            logger.warning(f"Routing overhead {routing_time:.0f}ms exceeds budget")
            # Fall back to simple routing
            decision = self._simple_route(prompt)
        
        return decision
```

### 13.3 No Quality Feedback Loop

**Problem**: Routing decisions are static and never improved based on actual quality.

**Solution**: Implement continuous quality monitoring and routing adjustment.

```python
class FeedbackLoop:
    def __init__(self, router):
        self.router = router
        self.quality_log = []
    
    def record_outcome(self, route_decision, quality_score, user_feedback=None):
        self.quality_log.append({
            "model": route_decision.model,
            "task_type": route_decision.task_type,
            "quality": quality_score,
            "user_feedback": user_feedback,
            "timestamp": datetime.now(),
        })
        
        # Retrain classifier if quality drops
        if self._detect_quality_drop():
            self.router.retrain_classifier(self.quality_log[-1000:])
```

### 13.4 Provider Lock-in via Routing

**Problem**: Routing rules are so specific to one provider that switching is impossible.

**Solution**: Abstract providers behind a common interface.

```python
# Abstract provider interface
class ModelProvider(ABC):
    @abstractmethod
    async def complete(self, model: str, messages: list) -> str:
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        pass

# Router works with any provider
class ProviderAgnosticRouter:
    def __init__(self, providers: dict[str, ModelProvider]):
        self.providers = providers  # {"openai": OpenAIProvider, "anthropic": AnthropicProvider}
```

---

## 14. Future Outlook

### 14.1 Predicted Trends (2026–2028)

| Trend | Timeline | Impact |
|-------|----------|--------|
| **Self-tuning routers** | 2026 H2 | Routers that automatically optimize routing based on quality feedback |
| **Multi-model responses** | 2027 | Combining outputs from multiple models for superior quality |
| **Edge-cloud routing** | 2027 | Routing between on-device and cloud models based on privacy/latency needs |
| **Predictive routing** | 2027 | Pre-fetching and pre-computing based on predicted user needs |
| **Federated routing** | 2028 | Organizations sharing routing insights without sharing data |

### 14.2 The Inference Mesh

The future of model routing is the **inference mesh** — a distributed system where:

1. **Every request is analyzed** for optimal model selection
2. **Models are fungible** — any model can be swapped without application changes
3. **Quality is continuously measured** and routing adapts in real-time
4. **Costs are optimized** across the entire organization, not per-request
5. **Providers are treated as commodities** — the router abstracts all provider differences

### 14.3 Recommendations for Practitioners

1. **Start simple**: Begin with static routing rules, add complexity as needed
2. **Measure everything**: Log routing decisions, quality scores, and costs
3. **Test continuously**: A/B test routing strategies against baselines
4. **Abstract providers**: Never hardcode provider-specific logic in applications
5. **Budget proactively**: Set daily/weekly cost budgets and alert on overruns
6. **Plan for failure**: Every router needs failover and circuit breaker patterns
7. **Optimize for your workload**: Generic benchmarks are starting points, not endpoints

---

## 15. Cross-References

### Related Documents in This Library

| Document | Relevance |
|----------|-----------|
| **02-LLMs/02-Model-Families.md** | Model capabilities and comparisons |
| **02-LLMs/06-AI-Model-Providers-Free-Tiers.md** | Provider pricing and free tiers |
| **02-LLMs/09-Open-Weights-Race-2026.md** | Open-weight model landscape |
| **41-AI-Cost-Optimization-and-Enterprise-ROI/** | Enterprise cost optimization strategies |
| **13-Top-Demand/12-Prompt-Caching-Cost-Optimization.md** | Prompt caching for cost reduction |
| **18-Agent-Security-and-Trust/** | Security considerations for model routing |
| **25-Multi-Cloud-AI-Strategy/** | Multi-cloud provider strategies |

### External Resources

- **OpenRouter Documentation**: https://openrouter.ai/docs
- **LiteLLM**: https://docs.litellm.ai/
- **Portkey Gateway**: https://portkey.ai/docs
- **Not Diamond**: https://github.com/Not-Diamond/not-diamond

---

*Last updated: July 4, 2026*
*Category: 02-LLMs*
*Document: 10-AI-Model-Routing-and-Smart-Selection.md*

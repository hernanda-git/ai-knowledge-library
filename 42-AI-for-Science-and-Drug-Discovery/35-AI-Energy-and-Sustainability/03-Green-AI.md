# 03 — Green AI: Sustainable Practices for Model Development and Deployment

> **Category:** 35-AI-Energy-and-Sustainability
> **Last updated:** June 29, 2026
> **Cross-references:** `30-Small-Language-Models/01-Overview-and-Efficiency.md`, `06-Advanced/04-Prompt-Engineering.md`, `02-LLMs/04-Quantization.md`, `01-Overview.md`

---

## Table of Contents

1. [What Is Green AI?](#1-what-is-green-ai)
2. [The Carbon Footprint of AI Models](#2-the-carbon-footprint-of-ai-models)
3. [Model Efficiency Techniques](#3-model-efficiency-techniques)
4. [Carbon-Aware Computing](#4-carbon-aware-computing)
5. [Green Prompt Engineering](#5-green-prompt-engineering)
6. [Efficient Inference Strategies](#6-efficient-inference-strategies)
7. [Sustainable Training Practices](#7-sustainable-training-practices)
8. [Measuring and Reporting AI Carbon Footprint](#8-measuring-and-reporting-ai-carbon-footprint)
9. [Tools and Frameworks for Green AI](#9-tools-and-frameworks-for-green-ai)
10. [Industry Initiatives and Standards](#10-industry-initiatives-and-standards)
11. [The Business Case for Green AI](#11-the-business-case-for-green-ai)
12. [Cross-References](#12-cross-references)

---

## 1. What Is Green AI?

Green AI refers to the practice of developing, training, and deploying AI systems with minimal environmental impact. It encompasses:

1. **Energy-efficient model design**: Architectures that require less compute
2. **Carbon-aware deployment**: Running workloads when/where clean energy is available
3. **Efficient inference**: Optimizing serving to minimize energy per query
4. **Sustainable infrastructure**: Using renewable energy, efficient cooling, and responsible hardware lifecycle
5. **Measurement and transparency**: Tracking and reporting AI's environmental footprint

### The Green AI Spectrum

```
Level 0: No consideration (default)
    ↓
Level 1: Awareness (tracking energy/carbon)
    ↓
Level 2: Optimization (reducing energy per task)
    ↓
Level 3: Carbon-aware (routing to clean energy)
    ↓
Level 4: Carbon-negative (offsets exceeding emissions)
    ↓
Level 5: Net-positive (generating clean energy)
```

Most organizations are at Level 0-1. Leading companies are at Level 2-3. No one has achieved Level 5.

---

## 2. The Carbon Footprint of AI Models

### Training Carbon Emissions

| Model | Parameters | Training Time | GPU Hours | Est. CO₂ (tonnes) | Equiv. |
|-------|-----------|---------------|-----------|-------------------|--------|
| BERT-base | 110M | 2 days | 1,000 | 0.7 | 1 car × 1 week |
| GPT-3 | 175B | 34 days | 3,640,000 | 552 | 5 cars × 1 year |
| PaLM | 540B | 64 days | 11,000,000 | 3,800 | 37 cars × 1 year |
| LLaMA-2 (70B) | 70B | 21 days | 1,700,000 | 310 | 3 cars × 1 year |
| GPT-4 | ~1.8T (est.) | ~100 days | ~25,000,000 | ~10,000 | 1,000 cars × 1 year |
| GPT-5 | ~10T (est.) | ~180 days | ~100,000,000 | ~50,000 | 5,000 cars × 1 year |

### Inference Carbon Emissions

Inference now dominates total AI carbon footprint:

| Query Type | Tokens (in+out) | Energy (Wh) | CO₂ (g) | Daily Queries | Daily CO₂ (kg) |
|-----------|-----------------|-------------|---------|---------------|----------------|
| Simple chat | 500 | 0.005 | 0.002 | 1M | 2 |
| Code generation | 2,000 | 0.02 | 0.008 | 100K | 0.8 |
| RAG pipeline | 5,000 | 0.05 | 0.02 | 50K | 1 |
| Image generation | N/A | 0.1 | 0.04 | 100K | 4 |
| Video generation | N/A | 5.0 | 2.0 | 10K | 20 |

### The Cumulative Problem

Even small per-query emissions add up at scale:

```
10 billion queries/day × 0.002g CO₂/query = 20 tonnes CO₂/day
= 7,300 tonnes CO₂/year
= ~1,500 cars for a year
```

This is why inference efficiency is the most impactful lever for green AI.

---

## 3. Model Efficiency Techniques

### Quantization

Reducing model precision from FP32/FP16 to INT8/INT4:

| Precision | Memory Reduction | Speedup | Quality Loss | Energy Savings |
|-----------|-----------------|---------|--------------|----------------|
| FP16 → INT8 | 2x | 1.5-2x | Negligible | 40-50% |
| FP16 → INT4 | 4x | 2-3x | Small | 60-75% |
| FP16 → INT2 | 8x | 3-4x | Moderate | 80-90% |

**Implementation example (bitsandbytes):**

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Load model with INT4 quantization (75% memory reduction)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # Nested quantization
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B",
    quantization_config=bnb_config,
    device_map="auto",
)

# Energy impact: 70B model reduced from ~140GB to ~35GB
# Inference energy reduced by ~60%
```

### Knowledge Distillation

Training a smaller model to mimic a larger one:

| Teacher → Student | Size Reduction | Quality Retention | Energy Savings |
|-------------------|---------------|-------------------|----------------|
| GPT-4 → Phi-3 | 100x | 85-90% | 90%+ |
| LLaMA-2-70B → LLaMA-2-7B | 10x | 80-85% | 85%+ |
| Claude-3 Opus → Claude-3 Haiku | 50x | 80-90% | 88%+ |

**Distillation pipeline:**

```python
from transformers import Trainer, TrainingArguments

# Distill large teacher into small student
training_args = TrainingArguments(
    output_dir="./distilled_model",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=5e-5,
    weight_decay=0.01,
    # Temperature for soft label distribution
    distillation_temperature=4.0,
    distillation_alpha=0.5,  # Balance between teacher soft labels and hard labels
)

# Energy savings: Training 7B model is ~10x cheaper than 70B
# Inference: 7B is ~10x cheaper than 70B
# Total lifecycle: ~100x energy savings
```

### Pruning and Sparsity

Removing unnecessary weights from models:

| Sparsity Level | Quality Impact | Energy Savings | Memory Savings |
|---------------|---------------|----------------|----------------|
| 10% | Negligible | 10-15% | 10% |
| 30% | Small | 25-35% | 30% |
| 50% | Moderate | 40-55% | 50% |
| 70% | Significant | 60-75% | 70% |
| 90% | Large | 80-90% | 90% |

### Mixture of Experts (MoE)

Only activating a subset of parameters per query:

```
Traditional dense model: 100% parameters active per query
MoE model: 10-25% parameters active per query

Energy reduction: 4-10x per query
Quality: Comparable to dense model of same total size
```

**Example: Mixtral 8x7B**
- Total parameters: 46.7B
- Active parameters per query: 12.9B
- Quality: Matches GPT-3.5
- Energy per query: ~4x less than equivalent dense model

### Early Exit and Adaptive Computation

Skipping computation for easy inputs:

```python
class AdaptiveTransformer(nn.Module):
    def __init__(self, num_layers, confidence_threshold=0.95):
        self.layers = nn.ModuleList([TransformerLayer() for _ in range(num_layers)])
        self.confidence_head = nn.Linear(hidden_size, vocab_size)
        self.threshold = confidence_threshold
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Check if we can exit early
            logits = self.confidence_head(x)
            probs = F.softmax(logits, dim=-1)
            max_prob = probs.max(dim=-1).values
            
            if max_prob.mean() > self.threshold:
                # Easy input — exit early, save 20-40% compute
                return logits
        
        return self.confidence_head(x)
```

---

## 4. Carbon-Aware Computing

### The Concept

Carbon-aware computing routes AI workloads to times and locations where clean energy is available:

```
User Request → Carbon-Aware Router → {Clean Grid Location, Right Time} → Execute → Return
```

### Implementation Architecture

```python
class CarbonAwareRouter:
    def __init__(self, regions):
        self.regions = regions  # List of data center regions
        self.grid_carbon = {}  # Real-time carbon intensity (g CO₂/kWh)
    
    async def get_carbon_intensity(self, region):
        """Fetch real-time carbon intensity from grid operator."""
        # APIs: electricityMaps, WattTime, gridstatus.io
        response = await self.api.get(f"/carbon/{region}")
        return response.carbon_intensity  # g CO₂/kWh
    
    async def route_request(self, request, max_latency_ms=200):
        """Route request to lowest-carbon region within latency budget."""
        candidates = []
        
        for region in self.regions:
            carbon = await self.get_carbon_intensity(region.name)
            latency = await self.measure_latency(region)
            
            if latency <= max_latency_ms:
                candidates.append({
                    'region': region,
                    'carbon': carbon,
                    'latency': latency,
                    'score': carbon * latency,  # Minimize carbon × latency
                })
        
        # Pick lowest carbon × latency score
        best = min(candidates, key=lambda x: x['score'])
        return best['region']
    
    async def batch_route(self, requests, delay_budget_seconds=30):
        """For non-urgent workloads, wait for clean energy windows."""
        for request in requests:
            region = await self.route_request(request)
            
            # If all regions are dirty right now, wait up to delay_budget
            carbon = await self.get_carbon_intensity(region.name)
            if carbon > self.threshold:
                wait_time = await self.estimate_clean_window(region)
                if wait_time <= delay_budget_seconds:
                    await asyncio.sleep(wait_time)
                    region = await self.route_request(request)
            
            await self.execute(request, region)
```

### Real-Time Carbon Intensity Data

| Source | Coverage | Update Frequency | API Cost |
|--------|----------|-----------------|----------|
| electricityMaps | Global | 5 min | Free tier available |
| WattTime | US, EU, AU | 5 min | Free tier available |
| gridstatus.io | US | 5 min | Free tier available |
| ENTSO-E | Europe | 1 hour | Free |
| EIA | US | 1 hour | Free |

### Carbon-Aware Scheduling Example

```python
# Schedule training job during clean energy windows
scheduler = CarbonAwareScheduler()

# Analyze grid carbon patterns
carbon_forecast = await scheduler.get_forecast(
    region="us-west-2",
    hours_ahead=48,
)

# Find the cleanest 8-hour window for training
best_window = scheduler.find_clean_window(
    carbon_forecast,
    duration_hours=8,
    max_carbon_intensity=50,  # g CO₂/kWh
)

print(f"Best window: {best_window.start} - {best_window.end}")
print(f"Average carbon: {best_window.avg_carbon} g CO₂/kWh")
print(f"Savings vs. worst window: {best_window.savings_percent}%")

# Schedule the job
await scheduler.schedule_training_job(
    job=training_job,
    window=best_window,
    region="us-west-2",
)
```

---

## 5. Green Prompt Engineering

### Optimizing Token Usage

Every unnecessary token wastes energy. Green prompt engineering minimizes tokens while maintaining quality:

| Prompt Style | Token Count | Quality | Energy Impact |
|-------------|-------------|---------|---------------|
| Verbose, unstructured | 500 | Good | Baseline |
| Structured, focused | 200 | Good | -60% |
| Optimized with examples | 150 | Excellent | -70% |
| Compressed with system prompt | 100 | Good | -80% |

### Techniques

**1. System Prompt Optimization**

```python
# Wasteful prompt (500 tokens)
user_prompt = """
I need you to help me write a function. Please write a Python function 
that takes a list of numbers and returns the average. Make sure to handle 
edge cases like empty lists. Please include docstrings and type hints. 
Also, please make sure the function is efficient and follows PEP 8 style.
"""

# Optimized prompt (120 tokens)
system_prompt = "You are a Python developer. Write clean, typed, PEP 8 functions with docstrings."
user_prompt = """
def average(nums: list[float]) -> float:
    """Return mean of nums; 0.0 if empty."""
"""
# 76% token reduction with same quality
```

**2. Few-Shot Example Compression**

```python
# Verbose few-shot (800 tokens)
examples = [
    {"input": "The weather today is...", "output": "Sentiment: Positive"},
    {"input": "I hate this product...", "output": "Sentiment: Negative"},
    # ... 10 more examples
]

# Compressed few-shot (200 tokens)
examples = "POS: good/great/excellent | NEG: bad/hate/terrible | NEU: okay/fine"
# 75% token reduction
```

**3. Response Length Control**

```python
# Uncontrolled response
response = model.generate(
    prompt="Summarize this article",
    max_tokens=1000,  # May generate 1000 tokens
)

# Controlled response
response = model.generate(
    prompt="Summarize this article in 2-3 sentences",
    max_tokens=100,  # Will generate ~80 tokens
    temperature=0.3,  # Lower temperature = more concise
)
# 90% token reduction
```

### The "Token Budget" Approach

Assign token budgets to different parts of your AI pipeline:

```python
class TokenBudget:
    def __init__(self, total_budget=500):
        self.total = total_budget
        self.system_prompt = 50  # 10%
        self.context = 200       # 40%
        self.user_query = 50     # 10%
        self.response = 200      # 40%
    
    def validate(self, messages):
        total_tokens = count_tokens(messages)
        if total_tokens > self.total:
            # Truncate context (oldest messages first)
            messages = self.truncate_context(messages, self.total)
        return messages
```

---

## 6. Efficient Inference Strategies

### Batching

Grouping multiple requests to share computation:

```python
# Naive: process one at a time
for request in requests:
    response = model.generate(request)  # GPU underutilized

# Efficient: batch processing
batch = collate_requests(requests, max_batch_size=32)
responses = model.generate(batch)  # GPU fully utilized

# Energy impact: 3-5x better energy efficiency
```

### KV-Cache Optimization

The Key-Value cache is critical for efficient inference:

```python
# Without KV-cache: O(n²) computation per token
# With KV-cache: O(n) computation per token

# Optimization: PagedAttention (vLLM)
# - Eliminates memory fragmentation
# - Enables larger batch sizes
# - 2-4x throughput improvement

# Example vLLM configuration
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B",
    tensor_parallel_size=4,  # Split across 4 GPUs
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    enable_prefix_caching=True,  # Cache common prefixes
    use_v2_block_manager=True,   # PagedAttention
)

# Energy impact: 2-3x more queries per GPU-hour
```

### Model Cascading

Using different models for different complexity levels:

```python
class ModelCascade:
    def __init__(self):
        self.models = [
            {"model": "phi-3-mini", "cost": 0.00001, "quality": "good"},
            {"model": "llama-3.1-8b", "cost": 0.00005, "quality": "better"},
            {"model": "llama-3.1-70b", "cost": 0.0005, "quality": "best"},
        ]
    
    async def route(self, query):
        # Start with cheapest model
        for model_info in self.models:
            response = await self.generate(query, model_info["model"])
            confidence = self.evaluate_confidence(response)
            
            if confidence > 0.9:
                return response  # Good enough, stop here
            
            # Try next (more expensive) model
        
        return response  # Return best model's response

# Energy impact: 60-80% reduction for easy queries
# Quality: Maintained through escalation
```

### Caching Strategies

```python
class SemanticCache:
    """Cache responses for semantically similar queries."""
    
    def __init__(self, similarity_threshold=0.95):
        self.cache = {}
        self.embeddings = {}
        self.threshold = similarity_threshold
    
    async def get_or_generate(self, query, model):
        # Check cache
        query_embedding = await self.embed(query)
        
        for cached_query, cached_response in self.cache.items():
            similarity = cosine_similarity(query_embedding, self.embeddings[cached_query])
            if similarity > self.threshold:
                return cached_response  # Cache hit — 0 energy
        
        # Cache miss — generate response
        response = await model.generate(query)
        
        # Store in cache
        self.cache[query] = response
        self.embeddings[query] = query_embedding
        
        return response

# Energy impact: 30-70% reduction for repetitive workloads
# Quality: Maintained (cached responses are identical)
```

---

## 7. Sustainable Training Practices

### Checkpointing and Resumption

```python
# Save checkpoints frequently to avoid wasted compute
training_args = TrainingArguments(
    output_dir="./checkpoints",
    save_strategy="steps",
    save_steps=500,  # Save every 500 steps
    save_total_limit=3,  # Keep only last 3 checkpoints
    
    # Resume from checkpoint if training fails
    resume_from_checkpoint=True,
)

# Energy impact: If training fails at step 4999, only 500 steps are wasted
# vs. 4999 steps without checkpointing
```

### Mixed Precision Training

```python
# Use FP16/BF16 for training (2x memory reduction, 1.5-2x speedup)
training_args = TrainingArguments(
    fp16=True,  # or bf16=True for Ampere+ GPUs
    # Energy impact: 30-40% reduction in training energy
)

# With automatic mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():  # Automatically uses FP16 where safe
        outputs = model(batch)
        loss = outputs.loss
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Quality impact: Negligible for most tasks
# Energy savings: 30-40%
```

### Efficient Data Loading

```python
# Inefficient: Loading all data to memory
dataset = load_dataset("large_dataset")  # Loads everything

# Efficient: Streaming
dataset = load_dataset("large_dataset", streaming=True)  # Loads on-demand

# Efficient: Preprocessing
from datasets import Dataset

# Tokenize once, save, reuse
tokenized = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=8,  # Parallel processing
    cache_file_name="tokenized_cache.arrow",  # Cache to disk
)

# Energy impact: 20-30% reduction in I/O overhead
```

### Distributed Training Optimization

```python
# Efficient distributed training
training_args = TrainingArguments(
    # FSDP (Fully Sharded Data Parallel)
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        "backward_prefetch": "backward_pre",
        "forward_prefetch": "forward_prefetch",
    },
    
    # Gradient checkpointing (trade compute for memory)
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    # DeepSpeed ZeRO Stage 3
    deepspeed="ds_config.json",
)

# Energy impact: Enables training larger models with same hardware
# Reduces total training time by 20-30%
```

---

## 8. Measuring and Reporting AI Carbon Footprint

### CodeCarbon

The most popular tool for tracking AI carbon emissions:

```python
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(
    output_dir="./carbon_logs",
    output_file="emissions.csv",
    country_iso_code="USA",
    log_level="INFO",
)

tracker.start()

# Your AI training code
model = train_model(data)
predictions = model.predict(test_data)

emissions = tracker.stop()
print(f"Total emissions: {emissions:.4f} kg CO₂")
print(f"Equivalent to: {emissions * 2.5:.2f} km driving")
```

### ML CO₂ Impact

```python
from mlco2 import Impact

impact = Impact(
    training_co2_kg=552,  # GPT-3 training emissions
    country_code="USA",
)

# Get breakdown
print(f"Direct emissions: {impact.direct_emissions:.2f} kg CO₂")
print(f"Indirect emissions: {impact.indirect_emissions:.2f} kg CO₂")
print(f"Total lifecycle: {impact.total_lifecycle:.2f} kg CO₂")
```

### Reporting Dashboard

```python
class GreenAIDashboard:
    def __init__(self):
        self.metrics = {
            "total_energy_kwh": 0,
            "total_carbon_kg": 0,
            "total_queries": 0,
            "avg_carbon_per_query": 0,
            "renewable_percentage": 0,
            "efficiency_trend": [],
        }
    
    def log_query(self, tokens_in, tokens_out, energy_wh, carbon_g):
        self.metrics["total_energy_kwh"] += energy_wh / 1000
        self.metrics["total_carbon_kg"] += carbon_g / 1000
        self.metrics["total_queries"] += 1
        self.metrics["avg_carbon_per_query"] = (
            self.metrics["total_carbon_kg"] / self.metrics["total_queries"]
        )
    
    def generate_report(self):
        return {
            "summary": self.metrics,
            "comparison": self.benchmark(),
            "recommendations": self.optimize(),
            "trend": self.trend_analysis(),
        }
```

---

## 9. Tools and Frameworks for Green AI

### Carbon Tracking

| Tool | Type | Features | Best For |
|------|------|----------|----------|
| CodeCarbon | Python library | Real-time tracking, CSV export | Research teams |
| ML CO₂ Impact | Python library | Lifecycle analysis | Academic research |
| Green Algorithms | Web calculator | Quick estimation | Quick estimates |
| electricityMaps API | API | Real-time grid carbon | Carbon-aware routing |
| WattTime API | API | US grid carbon data | US-based workloads |

### Efficiency Optimization

| Tool | Type | Features | Best For |
|------|------|----------|----------|
| vLLM | Inference server | PagedAttention, batching | High-throughput serving |
| TensorRT-LLM | Inference | NVIDIA optimization | GPU inference |
| ONNX Runtime | Inference | Cross-platform | Edge inference |
| bitsandbytes | Training | INT4/INT8 quantization | Memory-efficient training |
| DeepSpeed | Training | ZeRO, FSDP | Distributed training |
| Axolotl | Training | Fine-tuning | Custom model training |

### Green AI Frameworks

| Framework | Focus | Key Feature |
|-----------|-------|-------------|
| Sustainable AI | Full lifecycle | Carbon accounting, reporting |
| Greenflow | Pipeline optimization | Energy-aware workflow scheduling |
| EcoAI | Model selection | Choosing efficient models |
| CarbonBudget | Experiment tracking | Per-experiment carbon limits |

---

## 10. Industry Initiatives and Standards

### Green AI Pledges

| Initiative | Members | Commitment |
|-----------|---------|------------|
| Climate Pledge (Amazon) | 400+ companies | Net-zero by 2040 |
| Renewable Energy Buyers Alliance | 300+ companies | 100% renewable by 2030 |
| Science Based Targets initiative | 5000+ companies | Paris-aligned targets |
| AI for Climate (UN) | 50+ countries | AI for climate action |

### Emerging Standards

| Standard | Scope | Status |
|----------|-------|--------|
| ISO 14064 (GHG accounting) | Organizational emissions | Published |
| ISO 50001 (energy management) | Energy efficiency | Published |
| EU AI Act (energy reporting) | AI-specific | Enacted 2025 |
| GHG Protocol (Scope 3) | Supply chain emissions | Published |
| AI Sustainability Scorecard | AI model assessment | Draft |

### Certification Programs

- **LEED for Data Centers**: Building sustainability certification
- **ENERGY STAR for Data Centers**: Energy efficiency certification
- **ISO 14001**: Environmental management system
- **Carbon Trust Standard**: Carbon footprint certification
- **SBTi**: Science-based targets validation

---

## 11. The Business Case for Green AI

### Cost Savings

| Efficiency Measure | Energy Savings | Cost Savings (annual) | Implementation Cost |
|-------------------|---------------|----------------------|-------------------|
| INT4 quantization | 60-75% | $500K-5M | Low (days) |
| Model cascading | 60-80% | $300K-3M | Medium (weeks) |
| Caching | 30-70% | $200K-2M | Low (days) |
| Carbon-aware scheduling | 20-40% | $100K-1M | Medium (weeks) |
| Liquid cooling | 40-60% | $500K-5M | High (months) |

### Competitive Advantage

- **Brand reputation**: 73% of consumers prefer environmentally responsible companies
- **Regulatory compliance**: Avoid carbon taxes and energy regulations
- **Investor demand**: ESG-focused investors prefer green AI companies
- **Talent attraction**: Engineers prefer companies with sustainability commitments
- **Cost leadership**: Lower energy costs = lower prices or higher margins

### Risk Mitigation

- **Energy price volatility**: Less exposure to energy price spikes
- **Regulatory risk**: Prepared for carbon taxes and energy regulations
- **Supply chain risk**: Reduced dependence on specific energy sources
- **Reputation risk**: Avoiding "AI greenwashing" accusations

---

## 12. Cross-References

| Document | Relevance |
|----------|-----------|
| `30-Small-Language-Models/01-Overview-and-Efficiency.md` | SLM efficiency techniques |
| `02-LLMs/04-Quantization.md` | Quantization deep dive |
| `06-Advanced/04-Prompt-Engineering.md` | Prompt optimization |
| `01-Overview.md` | AI energy crisis overview |
| `02-Data-Center-Energy.md` | Data center power and cooling |
| `04-Nuclear-and-Renewable.md` | Clean energy for AI |
| `20-Agent-Infrastructure-and-Observability/05-Agent-Cost-Tracking.md` | Cost tracking includes energy |

---

*Previous: [02-Data-Center-Energy.md](02-Data-Center-Energy.md) | Next: [04-Nuclear-and-Renewable.md](04-Nuclear-and-Renewable.md)*

---
**See also:**
- [Enterprise AI Deployment: Production Infrastructure and Operations](05-Enterprise/01-Enterprise-AI-Deployment.md)
- [Small Language Models — Efficiency, Edge Deployment & On-Device AI](30-Small-Language-Models/01-Overview-and-Efficiency.md)
- [Applied Reasoning — Use Cases, Distillation & Deployment](29-Reasoning-and-Inference-Scaling/03-Applications-and-Deployment.md)

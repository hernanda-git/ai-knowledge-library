# MLOps & AI Platform Engineering — Core Topics

> **Category:** 56 — MLOps & AI Platform Engineering  
> **Last Updated:** July 2026  
> **Cross-references:** [01-Overview.md](01-Overview.md), [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md)

---

## Table of Contents

1. [Model Lifecycle Management](#1-model-lifecycle-management)
2. [Training Pipeline Architecture](#2-training-pipeline-architecture)
3. [Model Serving Patterns](#3-model-serving-patterns)
4. [GPU Resource Management](#4-gpu-resource-management)
5. [Feature Engineering at Scale](#5-feature-engineering-at-scale)
6. [Model Monitoring & Observability](#6-model-monitoring--observability)
7. [Cost Management & Optimization](#7-cost-management--optimization)
8. [Security & Compliance](#8-security--compliance)
9. [Multi-Model Orchestration](#9-multi-model-orchestration)
10. [CI/CD for AI Systems](#10-cicd-for-ai-systems)

---

## 1. Model Lifecycle Management

### The Model Lifecycle (2026)

```
Research → Prototype → Validation → Staging → Production → Monitoring → Retraining
   │           │            │           │           │            │           │
   ▼           ▼            ▼           ▼           ▼            ▼           ▼
 Notebooks  Experiments  Benchmarks  Canary   Progressive   Drift      Automated
            Tracking      Suite      Release   Rollout     Detection    Retraining
```

### Model Registry Best Practices

A model registry is the source of truth for all models:

```python
# Model Registration Example (MLflow 3.0)
import mlflow
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import numpy as np

# Define model signature
input_schema = Schema([
    TensorSpec(np.dtype(np.float32), (-1, 512), name="input_features"),
    TensorSpec(np.dtype(np.int32), (-1,), name="attention_mask")
])
output_schema = Schema([
    TensorSpec(np.dtype(np.float32), (-1, 1024), name="embeddings")
])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Register model
model_info = mlflow.pyfunc.log_model(
    artifact_path="embedding-model",
    model=embedding_model,
    signature=signature,
    registered_model_name="production-embeddings",
    tags={
        "team": "search",
        "framework": "sentence-transformers",
        "gpu_required": "true",
        "cost_tier": "medium",
        "compliance": "internal-approved"
    },
    metadata={
        "training_data_version": "v2.3.1",
        "training_date": "2026-07-01",
        "evaluation_metrics": {
            "mrr@10": 0.89,
            "recall@100": 0.95,
            "latency_p99_ms": 12
        }
    }
)
```

### Model Versioning Strategy

| Version Type | Example | When to Use |
|-------------|---------|-------------|
| Major | v1 → v2 | Breaking API changes |
| Minor | v1.0 → v1.1 | New features, non-breaking |
| Patch | v1.0.0 → v1.0.1 | Bug fixes, minor improvements |
| Experiment | v1.0.0-exp-42 | Research experiments |
| Canary | v1.1.0-canary | Pre-production testing |

### Model Promotion Process

```
Development → Staging → Production
     │            │           │
     ▼            ▼           ▼
  Automatic    Automated   Manual + Automated
  (on merge)   (quality    (approval gate)
               gates pass)
```

---

## 2. Training Pipeline Architecture

### Pipeline Types

#### Batch Training Pipeline
```python
# Example: Batch training pipeline with Kubeflow
from kfp import dsl
from kfp.components import load_component_from_text

@dsl.pipeline(
    name='llm-fine-tuning-pipeline',
    description='Fine-tune LLM for domain-specific task'
)
def training_pipeline(
    dataset_path: str,
    base_model: str,
    hyperparams: dict
):
    # Step 1: Data validation
    validate_data = load_component_from_text('''
      name: Validate Data
      implementation:
        container:
          image: company/data-validator:latest
          args:
          - --path={{input}}
          - --schema=training_schema.yaml
          - --min-rows=10000
          - --max-null-ratio=0.05
    ''')
    data_validation = validate_data(input=dataset_path)
    
    # Step 2: Preprocessing
    preprocess = load_component_from_text('''
      name: Preprocess
      implementation:
        container:
          image: company/data-preprocessor:latest
          args:
          - --input={{data}}
          - --output={{output}}
          - --tokenizer={{base_model}}
          - --max-length=2048
          - --packing=true
    ''')
    preprocessed = preprocess(
        data=data_validation.output,
        base_model=base_model
    )
    
    # Step 3: Training
    train = load_component_from_text('''
      name: Train
      implementation:
        container:
          image: company/llm-trainer:latest
          args:
          - --base-model={{base_model}}
          - --data={{preprocessed}}
          - --output={{output}}
          - --epochs={{epochs}}
          - --learning-rate={{lr}}
          - --batch-size={{batch_size}}
          - --gpu=nvidia-h100-80gb
    ''')
    model = train(
        preprocessed=preprocessed.output,
        base_model=base_model,
        epochs=hyperparams['epochs'],
        lr=hyperparams['learning_rate'],
        batch_size=hyperparams['batch_size']
    )
    
    # Step 4: Evaluation
    evaluate = load_component_from_text('''
      name: Evaluate
      implementation:
        container:
          image: company/model-evaluator:latest
          args:
          - --model={{model}}
          - --eval-dataset={{eval_dataset}}
          - --metrics=accuracy,latency,throughput
          - --thresholds=accuracy>0.85,latency_p99<100ms
    ''')
    evaluation = evaluate(model=model.output)
    
    # Step 5: Register
    register = load_component_from_text('''
      name: Register Model
      implementation:
        container:
          image: company/model-registry:latest
          args:
          - --model={{model}}
          - --metrics={{evaluation}}
          - --register-to=production-registry
          - --version-type=minor
    ''')
    register(model=model.output, evaluation=evaluation.output)
```

### Training Infrastructure Patterns

| Pattern | Use Case | Complexity | Cost |
|---------|----------|-----------|------|
| **Single GPU** | Small models, experiments | Low | Low |
| **Multi-GPU (single node)** | Medium models, fine-tuning | Medium | Medium |
| **Multi-node** | Large models, pre-training | High | High |
| **Spot/Preemptible** | Cost-sensitive training | Medium | Low |
| **Serverless** | Burst training, CI/CD | Low | Pay-per-use |

### Training Cost Calculator

```python
def estimate_training_cost(
    model_params_b: float,
    dataset_tokens_m: float,
    epochs: int,
    gpu_type: str = "h100"
):
    """Estimate training cost in USD"""
    gpu_costs = {
        "h100": 2.50,    # $/hour
        "a100": 1.50,
        "l4": 0.50,
        "t4": 0.20
    }
    
    # Rough estimate: 1B params needs ~4GB memory
    # Training throughput depends on model size
    tokens_per_second = {
        "h100": 500_000,
        "a100": 300_000,
        "l4": 100_000,
        "t4": 50_000
    }
    
    total_tokens = dataset_tokens_m * 1_000_000 * epochs
    hours_needed = total_tokens / tokens_per_second[gpu_type]
    cost = hours_needed * gpu_costs[gpu_type]
    
    return {
        "hours": round(hours_needed, 1),
        "cost_usd": round(cost, 2),
        "cost_per_epoch": round(cost / epochs, 2)
    }

# Example: Fine-tune 7B model on 10M tokens
result = estimate_training_cost(
    model_params_b=7,
    dataset_tokens_m=10,
    epochs=3,
    gpu_type="h100"
)
# {'hours': 60.0, 'cost_usd': 150.0, 'cost_per_epoch': 50.0}
```

---

## 3. Model Serving Patterns

### Serving Architecture Comparison

| Pattern | Latency | Throughput | Cost | Use Case |
|---------|---------|-----------|------|----------|
| **Real-time** | Low (< 100ms) | Medium | High | Chatbots, search |
| **Batch** | High (minutes) | Very High | Low | Analytics, ETL |
| **Streaming** | Medium | High | Medium | Live feeds, recommendations |
| **Edge** | Very Low | Low | Very Low | Mobile, IoT |

### Real-Time Serving with vLLM

```python
# vLLM deployment configuration
from vllm import LLM, SamplingParams

# Server configuration
llm = LLM(
    model="meta-llama/Llama-3-70B-Instruct",
    tensor_parallel_size=4,      # 4 GPUs
    pipeline_parallel_size=1,
    max_model_len=8192,
    gpu_memory_utilization=0.9,
    max_num_batched_tokens=8192,
    max_num_seqs=256,
    trust_remote_code=True,
    quantization="awq",           # AWQ quantization
    enable_prefix_caching=True,   # KV cache optimization
    block_size=16,
)

# Inference
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048,
    repetition_penalty=1.1,
    stop=["</s>", "Human:"],
)

outputs = llm.generate(prompts, sampling_params)
```

### Multi-Model Router

```python
# Intelligent model routing
class ModelRouter:
    def __init__(self):
        self.models = {
            "fast": {"model": "llama-3-8b", "cost": 0.0001, "latency": 20},
            "balanced": {"model": "llama-3-70b", "cost": 0.001, "latency": 80},
            "powerful": {"model": "gpt-4o", "cost": 0.01, "latency": 200},
        }
        self.classifier = load_classifier()
    
    def route(self, query: str, requirements: dict) -> str:
        """Route query to optimal model based on complexity and requirements"""
        complexity = self.classifier.predict(query)
        
        if requirements.get("latency_ms", 1000) < 50:
            return "fast"
        elif requirements.get("budget_per_query", 1.0) < 0.0005:
            return "fast"
        elif complexity > 0.8 or requirements.get("quality", 0) > 0.9:
            return "powerful"
        else:
            return "balanced"
    
    def fallback_chain(self, query: str) -> str:
        """Try models in order until success"""
        for model_name in ["fast", "balanced", "powerful"]:
            try:
                result = self.models[model_name]["instance"].generate(query)
                return result
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                continue
        raise Exception("All models failed")
```

### Canary Deployment

```yaml
# Progressive rollout configuration
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llm-router
spec:
  predictor:
    canaryTrafficPercent: 10
    model:
      modelFormat:
        name: pytorch
      storageUri: gs://models/llama-3-70b-v2
      resources:
        limits:
          nvidia.com/gpu: "4"
        requests:
          nvidia.com/gpu: "4"
  canaryTrafficPercent: 10
```

---

## 4. GPU Resource Management

### GPU Utilization Challenges

The #1 problem: **GPUs are expensive and often underutilized**.

```
Average GPU utilization in production:
  - Training: 40-60%
  - Inference: 20-40%
  - Ideal: 70-90%

This means:
  - $1M in GPUs → $400K-600K wasted
  - With optimization: $1M → $100K-300K wasted
```

### GPU Sharing Strategies

#### Time-Sharing
```python
# MIG (Multi-Instance GPU) for A100/H100
# Split one GPU into multiple instances
# Good for: Multiple small models

# A100 80GB can be split into:
# - 7x 1g.10gb instances (7 small models)
# - 3x 3g.20gb instances (3 medium models)
# - 1x 7g.40gb instance (1 large model)
# - Various combinations
```

#### Memory-Sharing
```python
# CUDA MPS (Multi-Process Service)
# Multiple processes share GPU compute
# Good for: Multiple models on same GPU

# Enable MPS
import subprocess
subprocess.run(["nvidia-cuda-mps-control", "-d", "default"])
```

#### Dynamic Partitioning
```python
# NVIDIA Multi-Process Service with dynamic partitioning
# Automatically allocate GPU resources based on demand

class GPUScheduler:
    def __init__(self, total_gpus: int):
        self.total_gpus = total_gpus
        self.allocations = {}
        self.queue = []
    
    def allocate(self, job_id: str, gpu_count: int, priority: int):
        """Allocate GPUs to a job"""
        if self.available_gpus() >= gpu_count:
            self.allocations[job_id] = {
                "gpu_count": gpu_count,
                "priority": priority,
                "start_time": time.time()
            }
            return True
        else:
            self.queue.append({
                "job_id": job_id,
                "gpu_count": gpu_count,
                "priority": priority
            })
            return False
    
    def preempt(self, job_id: str):
        """Preempt a low-priority job"""
        if job_id in self.allocations:
            del self.allocations[job_id]
            self._schedule_pending()
```

### Spot Instance Strategy

```python
# Cost optimization with spot instances
SPOT_DISCOUNT = {
    "aws": 0.70,   # 70% discount
    "gcp": 0.80,   # 80% discount
    "azure": 0.60  # 60% discount
}

class SpotAwareTrainer:
    def __init__(self, use_spot: bool = True):
        self.use_spot = use_spot
        self.checkpoint_interval = 300  # seconds
    
    def train_with_spot(self, model, dataset):
        """Train with automatic checkpointing for spot interruption"""
        import signal
        import threading
        
        def checkpoint_handler(signum, frame):
            """Handle spot interruption notice"""
            self.save_checkpoint(model)
            logger.info("Spot interruption: checkpoint saved")
        
        # Register signal handler (AWS provides 2-minute warning)
        signal.signal(signal.SIGTERM, checkpoint_handler)
        signal.signal(signal.SIGINT, checkpoint_handler)
        
        # Training loop
        for epoch in range(epochs):
            for batch in dataset:
                loss = model.training_step(batch)
                self.optimizer.step()
                
                # Periodic checkpointing
                if time.time() - self.last_checkpoint > self.checkpoint_interval:
                    self.save_checkpoint(model)
            
            # Save checkpoint at end of epoch
            self.save_checkpoint(model)
```

---

## 5. Feature Engineering at Scale

### Feature Store Architecture

```
┌─────────────────────────────────────────────────┐
│              Feature Store                       │
├─────────────────────────────────────────────────┤
│  Online Store (Redis/DynamoDB)                  │
│  ├── Low-latency (< 10ms)                       │
│  ├── Real-time features                         │
│  └── Current feature values                     │
├─────────────────────────────────────────────────┤
│  Offline Store (S3/GCS/Delta Lake)              │
│  ├── Historical features                        │
│  ├── Training data                              │
│  └── Point-in-time correctness                  │
├─────────────────────────────────────────────────┤
│  Feature Pipeline (Spark/Flink)                 │
│  ├── Batch computation                          │
│  ├── Stream processing                          │
│  └── Feature transformation                     │
└─────────────────────────────────────────────────┘
```

### Feature Definitions

```python
# Feast feature definitions
from feast import FeatureView, Field, Entity
from feast.types import Float32, Int64, String
from feast.infra.offline_stores.file_source import FileSource
from datetime import timedelta

# Entity definition
user = Entity(
    name="user_id",
    description="User identifier",
    value_type=Int64
)

# Feature views
user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="avg_query_length", dtype=Float32),
        Field(name="total_queries_30d", dtype=Int64),
        Field(name="preferred_language", dtype=String),
        Field(name="embedding_preference", dtype=String),
    ],
    online=True,
    source=FileSource(
        path="s3://features/user_features.parquet",
        timestamp_field="event_timestamp"
    )
)

# Streaming feature view
user_realtime_features = FeatureView(
    name="user_realtime_features",
    entities=[user],
    ttl=timedelta(minutes=5),
    schema=[
        Field(name="queries_last_5m", dtype=Int64),
        Field name="avg_response_time_5m", dtype=Float32),
    ],
    online=True,
    source=KafkaSource(
        kafka_bootstrap_servers="kafka:9092",
        topic="user_events",
        timestamp_field="event_timestamp"
    )
)
```

### Real-Time Feature Computation

```python
# Real-time feature computation with Flink
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

env = StreamExecutionEnvironment.get_execution_environment()
t_env = StreamTableEnvironment.create(env)

# Define streaming SQL for real-time features
t_env.execute_sql("""
    CREATE TABLE user_events (
        user_id BIGINT,
        event_type STRING,
        query STRING,
        response_time_ms DOUBLE,
        event_time TIMESTAMP(3),
        WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'user_events',
        'properties.bootstrap.servers' = 'kafka:9092',
        'format' = 'json'
    )
""")

# Compute real-time features
t_env.execute_sql("""
    CREATE TABLE user_realtime_features WITH (
        'connector' = 'redis',
        'host' = 'redis',
        'port' = '6379'
    ) AS
    SELECT
        user_id,
        COUNT(*) as queries_last_5m,
        AVG(response_time_ms) as avg_response_time_5m,
        MAX(event_time) as last_active
    FROM user_events
    WHERE event_time > CURRENT_TIMESTAMP - INTERVAL '5' MINUTE
    GROUP BY user_id
""")
```

---

## 6. Model Monitoring & Observability

### Monitoring Dimensions

| Dimension | What to Monitor | Why | How |
|-----------|----------------|-----|-----|
| **Performance** | Latency, throughput, errors | User experience | Prometheus, Datadog |
| **Quality** | Accuracy, F1, BLEU, hallucination rate | Model effectiveness | Evidently, WhyLabs |
| **Data** | Feature drift, schema changes, nulls | Data pipeline health | Great Expectations, Evidently |
| **Cost** | GPU utilization, cost per request | Budget management | Kubecost, CloudHealth |
| **Fairness** | Bias metrics across groups | Compliance, ethics | Fairlearn, AIF360 |
| **Security** | Adversarial inputs, prompt injection | Safety | Guardrails AI, Lakera |

### Data Drift Detection

```python
# Evidently AI for data drift monitoring
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset
)

def check_data_drift(reference_data, current_data, column_mapping):
    """Check for data drift between reference and current data"""
    report = Report(metrics=[
        DataDriftPreset(stattest='ks', stattest_threshold=0.05),
        DataQualityPreset(),
    ])
    
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    
    result = report.as_dict()
    
    # Check for drift
    drift_detected = any(
        metric['result']['drift_detected']
        for metric in result['metrics']
        if 'drift_detected' in metric.get('result', {})
    )
    
    if drift_detected:
        alert_drift(result)
    
    return result
```

### LLM-Specific Monitoring

```python
# LLM monitoring with Langfuse
from langfuse import Langfuse

langfuse = Langfuse()

@langfuse.observe()
def llm_inference(prompt: str, model: str) -> str:
    """Monitor LLM inference with full trace"""
    response = llm.generate(prompt)
    
    # Log generation details
    langfuse.score(
        name="hallucination_score",
        value=check_hallucination(prompt, response),
        comment="Automated hallucination check"
    )
    
    langfuse.score(
        name="relevance_score", 
        value=check_relevance(prompt, response),
        comment="Automated relevance check"
    )
    
    return response
```

### Alerting Strategy

```yaml
# Alerting rules (Prometheus/Grafana)
groups:
  - name: llm_serving_alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.99, llm_latency_seconds) > 0.5
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "LLM serving latency SLO breach"
          
      - alert: HighErrorRate
        expr: rate(llm_errors_total[5m]) / rate(llm_requests_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
          
      - alert: LowGPUUtilization
        expr: avg(gpu_utilization) < 0.3
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "GPU utilization below 30% for 30 minutes"
          
      - alert: CostSpike
        expr: ai_cost_per_hour > ai_cost_baseline * 2
        for: 15m
        labels:
          severity: warning
          
      - alert: ModelDrift
        expr: model_quality_score < 0.8
        for: 1h
        labels:
          severity: warning
```

---

## 7. Cost Management & Optimization

### Cost Attribution

```python
# Cost tracking decorator
import functools
from datetime import datetime

def track_cost(team: str, project: str, model: str):
    """Decorator to track AI costs"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            start_cost = get_current_cost(team)
            
            result = func(*args, **kwargs)
            
            end_time = datetime.now()
            end_cost = get_current_cost(team)
            
            cost = end_cost - start_cost
            duration = (end_time - start_time).total_seconds()
            
            # Log to cost tracking system
            log_cost({
                "team": team,
                "project": project,
                "model": model,
                "cost_usd": cost,
                "duration_seconds": duration,
                "timestamp": start_time.isoformat(),
                "function": func.__name__
            })
            
            return result
        return wrapper
    return decorator

@track_cost(team="search", project="semantic-search", model="embedding-v2")
def embed_query(query: str):
    return embedding_model.encode(query)
```

### Cost Optimization Strategies

| Strategy | Savings | Implementation | Risk |
|----------|---------|---------------|------|
| **Quantization** | 40-60% | AWQ, GPTQ, GGUF | Quality loss |
| **Model distillation** | 60-80% | Train smaller model | Complex setup |
| **Caching** | 20-40% | Redis, in-memory | Stale results |
| **Batching** | 10-30% | Dynamic batching | Latency increase |
| **Spot instances** | 60-80% | AWS/GCP spot | Interruption |
| **Right-sizing** | 20-40% | Match GPU to workload | Under-provisioning |
| **Autoscaling** | 30-50% | Scale to demand | Cold starts |

### Cost Dashboard

```python
# Cost dashboard data generation
def generate_cost_report(team: str, period: str = "30d"):
    """Generate cost report for a team"""
    costs = query_cost_data(team, period)
    
    report = {
        "team": team,
        "period": period,
        "total_cost": sum(c["cost"] for c in costs),
        "breakdown": {
            "gpu": sum(c["gpu_cost"] for c in costs),
            "storage": sum(c["storage_cost"] for c in costs),
            "network": sum(c["network_cost"] for c in costs),
            "api": sum(c["api_cost"] for c in costs),
        },
        "by_project": group_by_project(costs),
        "by_model": group_by_model(costs),
        "trend": calculate_trend(costs),
        "optimization_opportunities": find_optimizations(costs)
    }
    
    return report
```

---

## 8. Security & Compliance

### AI Security Threats

| Threat | Description | Mitigation |
|--------|-------------|------------|
| **Prompt Injection** | Malicious prompts that bypass safety | Input validation, Guardrails AI |
| **Data Poisoning** | Corrupted training data | Data validation, provenance tracking |
| **Model Theft** | Stealing model weights | Access controls, encryption |
| **Adversarial Inputs** | Inputs designed to fool model | Adversarial training, input validation |
| **Privacy Leakage** | Model memorizes training data | Differential privacy, data anonymization |

### Security Best Practices

```python
# Secure model serving
from guardrails import Guard
from guardrails.hub import Toxicity, Relevance, Hallucination

# Input validation
guard = Guard().use_many(
    Toxicity(threshold=0.5),
    Relevance(reference=expected_topic),
    Hallucination(model="gpt-3.5-turbo")
)

def secure_inference(prompt: str) -> str:
    """Inference with security checks"""
    # Validate input
    validation_result = guard.validate(prompt)
    if not validation_result.passed:
        raise SecurityError(f"Input validation failed: {validation_result.error}")
    
    # Run inference
    response = model.generate(prompt)
    
    # Validate output
    output_validation = guard.validate(response)
    if not output_validation.passed:
        response = "I cannot provide that information."
    
    return response
```

### Compliance Requirements

| Regulation | Requirements | Implementation |
|-----------|-------------|----------------|
| **EU AI Act** | Risk assessment, documentation, human oversight | Model cards, audit logs |
| **GDPR** | Data protection, right to explanation | Privacy-preserving ML, explainability |
| **HIPAA** | Healthcare data protection | Encryption, access controls |
| **SOC 2** | Security controls, audit trail | Logging, monitoring, access controls |

---

## 9. Multi-Model Orchestration

### Orchestration Patterns

#### Sequential Pipeline
```
Input → Model A → Model B → Model C → Output
```

#### Ensemble
```
Input → Model A ─┐
       → Model B ─┼→ Aggregator → Output
       → Model C ─┘
```

#### Mixture of Experts
```
Input → Router → Expert 1 ─┐
                       Expert 2 ─┼→ Combiner → Output
                       Expert 3 ─┘
```

#### Cascading
```
Input → Fast Model → (low confidence?) → Powerful Model → Output
```

### Implementation Example

```python
# Multi-model orchestration
class ModelOrchestrator:
    def __init__(self):
        self.models = {
            "router": load_model("router-model"),
            "fast": load_model("llama-3-8b"),
            "balanced": load_model("llama-3-70b"),
            "powerful": load_model("gpt-4o"),
        }
    
    def orchestrate(self, query: str, strategy: str = "cascade") -> dict:
        """Orchestrate multiple models"""
        if strategy == "cascade":
            return self._cascade(query)
        elif strategy == "ensemble":
            return self._ensemble(query)
        elif strategy == "moe":
            return self._mixture_of_experts(query)
    
    def _cascade(self, query: str) -> dict:
        """Try fast model first, escalate if needed"""
        # Try fast model
        fast_result = self.models["fast"].generate(query)
        confidence = self._check_confidence(fast_result)
        
        if confidence > 0.9:
            return {"model": "fast", "result": fast_result, "cost": 0.0001}
        
        # Try balanced model
        balanced_result = self.models["balanced"].generate(query)
        confidence = self._check_confidence(balanced_result)
        
        if confidence > 0.95:
            return {"model": "balanced", "result": balanced_result, "cost": 0.001}
        
        # Fall back to powerful model
        powerful_result = self.models["powerful"].generate(query)
        return {"model": "powerful", "result": powerful_result, "cost": 0.01}
```

---

## 10. CI/CD for AI Systems

### AI CI/CD Pipeline

```yaml
# GitHub Actions workflow for AI
name: AI CI/CD Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'models/**'
      - 'training/**'
      - 'serving/**'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run unit tests
        run: pytest tests/unit/
        
      - name: Run data validation
        run: python scripts/validate_data.py --data-path data/
        
      - name: Run model quality tests
        run: python scripts/test_model_quality.py --model-path models/
        
      - name: Security scan
        run: trivy image --severity HIGH,CRITICAL ai-serving:latest

  evaluate:
    needs: test
    runs-on: [self-hosted, gpu]
    steps:
      - name: Run evaluation suite
        run: |
          python scripts/evaluate.py \
            --model-path models/ \
            --eval-dataset eval/ \
            --metrics accuracy,latency,throughput \
            --thresholds accuracy>0.85,latency_p99<100ms
            
      - name: Check cost estimate
        run: python scripts/estimate_cost.py --model-path models/

  deploy:
    needs: evaluate
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to staging
        run: |
          kubectl set image deployment/ai-serving \
            ai-serving=registry/model:${{ github.sha }}
            
      - name: Run canary tests
        run: python scripts/canary_test.py --traffic=10%
        
      - name: Promote to production
        if: success()
        run: |
          kubectl set image deployment/ai-serving \
            ai-serving=registry/model:${{ github.sha }}
```

### Quality Gates

| Gate | Criteria | Action on Failure |
|------|----------|-------------------|
| **Data Quality** | No nulls > 5%, schema valid | Block pipeline |
| **Model Quality** | Accuracy > threshold | Block deployment |
| **Security** | No HIGH/CRITICAL vulnerabilities | Block deployment |
| **Cost** | Within budget | Warn, allow override |
| **Latency** | p99 < threshold | Block deployment |
| **Fairness** | Bias metrics within limits | Block deployment |

---

## Cross-References

- **[02-Core-Topics.md](02-Core-Topics.md)** — This file (core topics)
- **[03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md)** — Technical implementation details
- **[04-Tools-and-Frameworks.md](04-Tools-and-Frameworks.md)** — Tool landscape
- **[05-Future-Outlook.md](05-Future-Outlook.md)** — Future trends

---

*Next: [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md) — Deep technical implementation*

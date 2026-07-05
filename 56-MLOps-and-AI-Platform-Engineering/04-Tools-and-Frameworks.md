# MLOps & AI Platform Engineering — Tools and Frameworks

> **Category:** 56 — MLOps & AI Platform Engineering  
> **Last Updated:** July 2026  
> **Cross-references:** [02-Core-Topics.md](02-Core-Topics.md), [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md)

---

## Table of Contents

1. [Tool Landscape Overview](#1-tool-landscape-overview)
2. [Orchestration & Pipelines](#2-orchestration--pipelines)
3. [Experiment Tracking & Model Registry](#3-experiment-tracking--model-registry)
4. [Model Serving](#4-model-serving)
5. [Feature Stores](#5-feature-stores)
6. [Monitoring & Observability](#6-monitoring--observability)
7. [Data Versioning & Quality](#7-data-versioning--quality)
8. [GPU Management](#8-gpu-management)
9. [Cost Management](#9-cost-management)
10. [Platform Solutions](#10-platform-solutions)
11. [Selection Guide](#11-selection-guide)

---

## 1. Tool Landscape Overview

### The MLOps Stack (2026)

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│  MLflow UI │ W&B Dashboard │ Grafana │ Custom Portal         │
├─────────────────────────────────────────────────────────────┤
│                    ORCHESTRATION LAYER                       │
│  Kubeflow │ Argo Workflows │ Prefect │ Dagster │ Airflow    │
├─────────────────────────────────────────────────────────────┤
│                    TRACKING LAYER                            │
│  MLflow │ W&B │ ClearML │ Neptuna │ Comet ML               │
├─────────────────────────────────────────────────────────────┤
│                    SERVING LAYER                             │
│  vLLM │ TGI │ BentoML │ Seldon Core │ KServe │ Triton     │
├─────────────────────────────────────────────────────────────┤
│                    DATA LAYER                                │
│  Feast │ Tecton │ DVC │ LakeFS │ Delta Lake │ Great Expect.│
├─────────────────────────────────────────────────────────────┤
│                    INFRASTRUCTURE LAYER                      │
│  Kubernetes │ Docker │ Terraform │ Helm │ Crossplane         │
├─────────────────────────────────────────────────────────────┤
│                    OBSERVABILITY LAYER                       │
│  Prometheus │ Grafana │ Evidently │ WhyLabs │ Langfuse      │
└─────────────────────────────────────────────────────────────┘
```

### Tool Count by Category

| Category | Tools Count | maturity | Trend |
|----------|-------------|----------|-------|
| Orchestration | 8+ | High | Consolidating |
| Experiment Tracking | 6+ | High | Merging with platform |
| Model Serving | 10+ | Medium | vLLM dominant |
| Feature Store | 6+ | Medium | Real-time focus |
| Monitoring | 8+ | Medium | LLM-specific emerging |
| Data Versioning | 5+ | Medium | Delta Lake leading |
| GPU Management | 4+ | Low | Growing fast |
| Cost Management | 5+ | Low | New category |

---

## 2. Orchestration & Pipelines

### Kubeflow Pipelines

**Best for:** Complex ML pipelines, Kubernetes-native environments

```python
# Kubeflow Pipeline Example
from kfp import dsl
from kfp.components import load_component_from_text

@dsl.pipeline(name='llm-pipeline')
def llm_pipeline(
    data_path: str,
    model_name: str,
    epochs: int = 3
):
    # Component definitions...
    validate = load_component_from_text('''
      name: Validate Data
      implementation:
        container:
          image: company/validator:latest
          args: [--path={{input}}]
    ''')
    
    train = load_component_from_text('''
      name: Train Model
      implementation:
        container:
          image: company/trainer:latest
          args: [--model={{model}}, --epochs={{epochs}}]
    ''')
    
    # Pipeline flow
    validated = validate(input=data_path)
    model = train(data=validated.output, model=model_name, epochs=epochs)

# Compile and run
from kfp.compiler import Compiler
Compiler().compile(llm_pipeline, 'pipeline.yaml')
```

| Feature | Kubeflow | Argo | Prefect | Dagster |
|---------|----------|------|---------|---------|
| **K8s Native** | ✅ | ✅ | ❌ | ❌ |
| **UI** | ✅ | ✅ | ✅ | ✅ |
| **Versioning** | ✅ | ✅ | ✅ | ✅ |
| **GPU Support** | ✅ | ✅ | ⚠️ | ⚠️ |
| **Complexity** | High | Medium | Low | Medium |
| **Learning Curve** | Steep | Moderate | Easy | Easy |

### Prefect

**Best for:** Python-native workflows, ease of use

```python
# Prefect Pipeline Example
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(retries=3, cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def validate_data(data_path: str) -> dict:
    """Validate training data"""
    import pandas as pd
    df = pd.read_parquet(data_path)
    
    checks = {
        'row_count': len(df) > 10000,
        'null_ratio': df.isnull().sum().sum() / df.size < 0.05,
        'schema_valid': all(col in df.columns for col in required_columns)
    }
    
    return checks

@task(retries=2)
def train_model(data_path: str, model_name: str, epochs: int) -> str:
    """Train the model"""
    # Training code here
    model_path = f"models/{model_name}/checkpoint.pt"
    return model_path

@flow(name="llm-training-pipeline")
def training_flow(
    data_path: str,
    model_name: str = "llama-3-7b",
    epochs: int = 3
):
    """Main training pipeline"""
    # Validate
    checks = validate_data(data_path)
    if not all(checks.values()):
        raise ValueError(f"Data validation failed: {checks}")
    
    # Train
    model_path = train_model(data_path, model_name, epochs)
    
    # Evaluate
    metrics = evaluate_model(model_path)
    
    # Register if quality passes
    if metrics['accuracy'] > 0.85:
        register_model(model_path, metrics)
    
    return model_path

# Run pipeline
training_flow(
    data_path="s3://data/training.parquet",
    model_name="llama-3-7b",
    epochs=3
)
```

### Dagster

**Best for:** Data-aware orchestration, software-defined assets

```python
# Dagster Asset Example
from dagster import asset, AssetExecutionContext, Output
import pandas as pd

@asset(
    name="training_data",
    group_name="data",
    deps=["raw_data"],
    metadata={"quality": "high"}
)
def training_data(context: AssetExecutionContext) -> pd.DataFrame:
    """Clean and prepare training data"""
    raw = context.resources.io_manager.load_input("raw_data")
    
    # Clean data
    clean = raw.dropna()
    clean = clean[clean['text'].str.len() > 100]
    
    context.log.info(f"Processed {len(clean)} rows")
    
    return clean

@asset(
    name="trained_model",
    group_name="training",
    deps=["training_data"]
)
def trained_model(context: AssetExecutionContext):
    """Train the model"""
    data = context.resources.io_manager.load_input("training_data")
    
    # Training loop
    model = train(data)
    
    # Save model
    model.save("models/latest")
    
    return Output(
        value=model,
        metadata={"accuracy": model.accuracy}
    )
```

---

## 3. Experiment Tracking & Model Registry

### MLflow

**Best for:** All-in-one tracking + registry + deployment

```python
# MLflow Experiment Tracking
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# Start experiment
mlflow.set_experiment("llm-fine-tuning")

with mlflow.start_run(run_name="llama-3-7b-run-1"):
    # Log parameters
    mlflow.log_params({
        "model": "meta-llama/Llama-3-7B",
        "epochs": 3,
        "learning_rate": 2e-5,
        "batch_size": 32,
        "warmup_steps": 100,
        "weight_decay": 0.01
    })
    
    # Training loop
    for epoch in range(3):
        train_loss, val_loss, accuracy = train_epoch(model, dataloader)
        
        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy
        }, step=epoch)
    
    # Log model
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name="production-llm",
        signature=model_signature
    )
    
    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("training_curves.png")
```

### Weights & Biases

**Best for:** Advanced visualization, collaboration, sweeps

```python
# W&B Experiment Tracking
import wandb

# Initialize run
run = wandb.init(
    project="llm-fine-tuning",
    name="llama-3-7b-run-1",
    config={
        "model": "meta-llama/Llama-3-7B",
        "epochs": 3,
        "learning_rate": 2e-5,
        "batch_size": 32
    }
)

# Training loop
for epoch in range(3):
    train_loss, val_loss, accuracy = train_epoch(model, dataloader)
    
    # Log metrics
    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "accuracy": accuracy,
        "learning_rate": scheduler.get_last_lr()[0]
    })
    
    # Log model checkpoint
    wandb.save(f"checkpoints/epoch_{epoch}.pt")

# Log model artifact
artifact = wandb.Artifact("production-llm", type="model")
artifact.add_dir("model")
run.log_artifact(artifact)

# Finish run
run.finish()
```

### Comparison

| Feature | MLflow | W&B | ClearML | Neptuna |
|---------|--------|-----|---------|---------|
| **Open Source** | ✅ | ⚠️ | ✅ | ✅ |
| **Model Registry** | ✅ | ✅ | ✅ | ✅ |
| **Visualization** | Basic | Advanced | Good | Basic |
| **Hyperparameter Sweeps** | ❌ | ✅ | ✅ | ✅ |
| **GPU Tracking** | ⚠️ | ✅ | ✅ | ❌ |
| **Collaboration** | Basic | Advanced | Good | Basic |
| **Pricing** | Free | Freemium | Freemium | Free |

---

## 4. Model Serving

### vLLM

**Best for:** High-performance LLM serving, OpenAI-compatible API

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --quantization awq \
    --host 0.0.0.0 \
    --port 8000
```

```python
# Client usage (OpenAI-compatible)
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3-70B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing."}
    ],
    temperature=0.7,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### TGI (Text Generation Inference)

**Best for:** Production reliability, Hugging Face integration

```bash
# Start TGI server
docker run --gpus all \
    -p 8080:80 \
    -v /data/models:/models \
    ghcr.io/huggingface/text-generation-inference:1.4 \
    --model-id /models/llama-3-70b \
    --quantize awq \
    --max-input-length 4096 \
    --max-total-tokens 8192
```

### BentoML

**Best for:** Custom serving, multi-model, Python-native

```python
# BentoML Service
import bentoml
from bentoml.io import JSON, Text

@bentoml.service(
    gpu=1,
    resources={"memory": "8Gi"},
    traffic={"timeout": 60}
)
class LLMService:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3-7B",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3-7B"
        )
    
    @bentoml.api
    def generate(
        self,
        prompt: Text,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> JSON:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return {"response": response}

# Save and serve
bentoml_service = LLMService()
saved_model = bentoml_service.save()
```

### Comparison

| Feature | vLLM | TGI | BentoML | Seldon Core | KServe |
|---------|------|-----|---------|-------------|--------|
| **Performance** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| **OpenAI API** | ✅ | ✅ | ⚠️ | ❌ | ❌ |
| **Quantization** | ✅ | ✅ | ✅ | ⚠️ | ⚠️ |
| **Batching** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Multi-Model** | ⚠️ | ❌ | ✅ | ✅ | ✅ |
| **K8s Native** | ⚠️ | ✅ | ⚠️ | ✅ | ✅ |
| **Complexity** | Low | Low | Medium | High | High |

---

## 5. Feature Stores

### Feast

**Best for:** Open-source, self-hosted, flexibility

```python
# Feast Setup
from feast import FeatureStore
from feast import Entity, Feature, FeatureView, FileSource
from datetime import timedelta

# Define entity
driver = Entity(
    name="driver_id",
    value_type=ValueType.INT64,
    description="Driver identifier"
)

# Define feature view
driver_stats = FeatureView(
    name="driver_stats",
    entities=["driver_id"],
    ttl=timedelta(hours=1),
    features=[
        Feature(name="avg_speed", dtype=ValueType.FLOAT),
        Feature(name="total_trips_24h", dtype=ValueType.INT64),
        Feature name="rating", dtype=ValueType.FLOAT),
    ],
    online=True,
    batch_source=FileSource(
        path="s3://features/driver_stats.parquet",
        event_timestamp_column="event_timestamp"
    )
)

# Get features
store = FeatureStore(repo_path=".")
feature_vector = store.get_online_features(
    features=[
        "driver_stats:avg_speed",
        "driver_stats:total_trips_24h",
    ],
    entity_rows=[{"driver_id": 1001}]
).to_dict()
```

### Tecton

**Best for:** Managed platform, real-time features

```python
# Tecton Feature Definition
from tecton import FeatureView, BatchFeatureView, StreamFeatureView
from tecton.types import Float32, Int64
from tecton import AggregationFunction

# Batch feature view
user_features = BatchFeatureView(
    name="user_features",
    source=ParquetDataSource(path="s3://data/user_events/"),
    entities=[user_id],
    freshness_policy=FreshnessPolicy(max_staleness=timedelta(days=1)),
    features=[
        Feature(name="total_purchases_30d", dtype=Int64),
        Feature(name="avg_order_value", dtype=Float32),
    ],
    aggregations=[
        AggregationFunction(function="count", column="order_id", time_window=timedelta(days=30)),
        AggregationFunction(function="mean", column="order_value", time_window=timedelta(days=30)),
    ]
)

# Stream feature view
user_realtime = StreamFeatureView(
    name="user_realtime",
    source=KafkaSource(topic="user_events"),
    entities=[user_id],
    freshness_policy=FreshnessPolicy(max_staleness=timedelta(minutes=5)),
    features=[
        Feature(name="events_last_5m", dtype=Int64),
        Feature(name="session_duration", dtype=Float32),
    ]
)
```

### Comparison

| Feature | Feast | Tecton | Hopsworks | Databricks Feature Store |
|---------|-------|--------|-----------|--------------------------|
| **Open Source** | ✅ | ❌ | ✅ | ❌ |
| **Real-Time** | ✅ | ✅ | ✅ | ✅ |
| **Managed** | ❌ | ✅ | ✅ | ✅ |
| **Point-in-Time** | ✅ | ✅ | ✅ | ✅ |
| **Streaming** | ✅ | ✅ | ✅ | ⚠️ |
| **Pricing** | Free | $$$ | $$ | $$ (part of Databricks) |

---

## 6. Monitoring & Observability

### Evidently AI

**Best for:** Data drift, model quality, open-source

```python
# Evidently Monitoring
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
    ClassificationPreset
)

# Create monitoring report
report = Report(metrics=[
    DataDriftPreset(stattest='ks', stattest_threshold=0.05),
    DataQualityPreset(),
    ClassificationPreset(),
])

report.run(
    reference_data=training_data,
    current_data=production_data,
    column_mapping=ColumnMapping(
        target='label',
        prediction='prediction',
        numerical_features=['feature1', 'feature2'],
        categorical_features=['category']
    )
)

# Get results
result = report.as_dict()
drift_detected = any(
    metric['result']['drift_detected']
    for metric in result['metrics']
    if 'drift_detected' in metric.get('result', {})
)

# Save report
report.save_html("monitoring_report.html")
```

### Langfuse

**Best for:** LLM-specific monitoring, traces, evaluation

```python
# Langfuse LLM Monitoring
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse()

@observe(as_type="generation")
def llm_generate(prompt: str, model: str) -> str:
    """Monitor LLM generation with full trace"""
    # Call LLM
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Log generation details
    langfuse_context.update_current_observation(
        model=model,
        input=prompt,
        output=response.choices[0].message.content,
        usage={
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    )
    
    return response.choices[0].message.content

@observe()
def rag_pipeline(query: str) -> str:
    """Full RAG pipeline with tracing"""
    # Retrieve documents
    docs = retrieve_documents(query)
    
    # Generate answer
    answer = llm_generate(
        prompt=f"Based on {docs}, answer: {query}",
        model="gpt-4o"
    )
    
    # Score quality
    langfuse.score(
        name="relevance",
        value=check_relevance(query, docs, answer)
    )
    
    return answer
```

### WhyLabs

**Best for:** Managed platform, comprehensive monitoring

```python
# WhyLabs Monitoring
import whylogs as why
from whylogs.api.writer.whylabs import WhyLabsWriter

# Create whylog profile
profile = why.log pandas(df)

# Upload to WhyLabs
writer = WhyLabsWriter()
writer.write(file=profile.view().serialize(), file_name="dataset_profile.bin")

# Set up monitoring
from whylabs.client import WhyLabsClient

client = WhyLabsClient()
client.upload_profile(
    profile=profile.view(),
    dataset_id="your-dataset-id",
    segment="production"
)
```

### Comparison

| Feature | Evidently | Langfuse | WhyLabs | Arize Phoenix |
|---------|-----------|----------|---------|---------------|
| **Open Source** | ✅ | ✅ | ❌ | ✅ |
| **LLM-Specific** | ⚠️ | ✅ | ✅ | ✅ |
| **Data Drift** | ✅ | ❌ | ✅ | ✅ |
| **Model Quality** | ✅ | ✅ | ✅ | ✅ |
| **Traces** | ❌ | ✅ | ⚠️ | ✅ |
| **Managed Option** | ❌ | ✅ | ✅ | ✅ |

---

## 7. Data Versioning & Quality

### DVC (Data Version Control)

**Best for:** Git-like data versioning, open-source

```bash
# DVC Setup
dvc init
dvc remote add -d storage s3://mybucket/dvc

# Track data
dvc add data/training.parquet
git add data/training.parquet.dvc .gitignore
git commit -m "Add training data v1"

# Version data
dvc push

# Get data version
dvc checkout
```

### Great Expectations

**Best for:** Data quality validation

```python
# Great Expectations Validation
import great_expectations as gx

context = gx.get_context()

# Create expectation suite
suite = context.add_expectation_suite("training_data_validation")

# Add expectations
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(column="text")
)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="text_length",
        min_value=100,
        max_value=10000
    )
)
suite.add_expectation(
    gx.expectations.ExpectTableColumnsToMatchOrderedList(
        column_list=["id", "text", "label", "timestamp"]
    )
)

# Validate data
result = context.run_checkpoint(
    checkpoint_name="training_data_checkpoint",
    batch_request={
        "datasource_name": "s3_datasource",
        "data_asset_name": "training_data.parquet"
    }
)

if not result.success:
    raise ValueError("Data validation failed!")
```

### Delta Lake

**Best for:** ACID transactions, time travel, schema evolution

```python
# Delta Lake Operations
from delta.tables import DeltaTable
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .getOrCreate()

# Write Delta table
df.write.format("delta").save("s3://data/features")

# Time travel
df_v1 = spark.read.format("delta").load("s3://data/features").option("versionAsOf", 0)
df_latest = spark.read.format("delta").load("s3://data/features")

# Schema evolution
spark.read.format("delta").load("s3://data/features") \
    .withColumn("new_feature", F.lit(1.0)) \
    .write.format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .save("s3://data/features")
```

---

## 8. GPU Management

### Run:ai

**Best for:** Enterprise GPU orchestration, time-sharing

```yaml
# Run:ai GPU allocation
apiVersion: run.ai/v2
kind: GpuAllocation
metadata:
  name: training-gpu
spec:
  gpuType: nvidia-a100-80gb
  gpuCount: 4
  timeShare:
    enabled: true
    maxGpuUtilization: 80%
  priority: high
  team: ml-platform
```

### NVIDIA MPS

**Best for:** GPU sharing at driver level

```bash
# Enable NVIDIA MPS
nvidia-cuda-mps-control -d default

# Set memory limit per process
export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=1024

# Run multiple processes
python train_model_1.py &
python train_model_2.py &
```

### Slurm (for HPC)

**Best for:** HPC environments, batch scheduling

```bash
#!/bin/bash
#SBATCH --job-name=llm-training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --qos=normal

# Load modules
module load cuda/12.2
module load nccl/2.18

# Launch training
torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --master_addr=$SLURM_MASTER_ADDR \
    --master_port=$SLURM_MASTER_PORT \
    train.py
```

---

## 9. Cost Management

### Kubecost

**Best for:** Kubernetes cost allocation, open-source

```yaml
# Kubecost deployment
helm upgrade -i kubecost cost-analyzer \
    --namespace kubecost \
    --set kubecostToken="YOUR_TOKEN" \
    --set prometheus.server.global.external_labels.cluster_id="ai-production" \
    --set grafana.enabled=true
```

```python
# Kubecost API for cost queries
import requests

def get_namespace_cost(namespace: str, days: int = 30) -> dict:
    """Get cost for a Kubernetes namespace"""
    response = requests.get(
        f"http://kubecost:9090/model/allocation",
        params={
            "window": f"{days}d",
            "aggregate": "namespace",
            "filter": f"namespace={namespace}"
        }
    )
    return response.json()

def get_gpu_cost(team: str) -> dict:
    """Get GPU cost for a team"""
    response = requests.get(
        f"http://kubecost:9090/model/allocation",
        params={
            "window": "30d",
            "aggregate": "label",
            "label": "team"
        }
    )
    return response.json()
```

### CloudHealth / Spot by NetApp

**Best for:** Multi-cloud cost optimization

```python
# Cost optimization recommendations
def get_optimization_recommendations():
    """Get cost optimization recommendations"""
    # Spot.io recommendations
    recommendations = {
        "spot_instances": {
            "eligible_workloads": ["training-batch", "inference-spiky"],
            "estimated_savings": "60-80%",
            "risk": "low for fault-tolerant workloads"
        },
        "right_sizing": {
            "over_provisioned": ["inference-stable"],
            "estimated_savings": "30-40%",
            "recommendation": "Downsize from A100 to L4"
        },
        "reserved_instances": {
            "steady_workloads": ["inference-stable"],
            "estimated_savings": "40-50%",
            "commitment": "1 year"
        }
    }
    return recommendations
```

---

## 10. Platform Solutions

### Cloud Platforms

#### AWS SageMaker

```python
# SageMaker Training
import sagemaker
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    source_dir=".",
    role=sagemaker.get_execution_role(),
    instance_count=4,
    instance_type="ml.p5.48xlarge",
    framework_version="2.1",
    py_version="py310",
    distribution={
        "pytorchddp": {"enabled": True}
    },
    hyperparameters={
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 2e-5
    }
)

estimator.fit({"training": "s3://data/training/"})
```

#### Google Vertex AI

```python
# Vertex AI Training
from google.cloud import aiplatform

aiplatform.init(project="my-project", region="us-central1")

job = aiplatform.CustomTrainingJob(
    display_name="llm-fine-tuning",
    script_path="train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1:latest",
    requirements=["transformers", "datasets"],
)

model = job.run(
    replica_count=4,
    machine_type="a2-highgpu-8g",
    args=["--epochs", "3", "--batch-size", "32"]
)
```

### AI-Native Platforms

#### Modal

```python
# Modal Serverless GPU
import modal

app = modal.App("llm-serving")

@app.function(
    gpu="H100",
    memory=128000,
    timeout=3600
)
def run_inference(prompt: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3-70B-Instruct",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-70B-Instruct")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Deploy
@app.local_entrypoint()
def main():
    result = run_inference.remote("Explain quantum computing")
    print(result)
```

---

## 11. Selection Guide

### Decision Matrix

| Use Case | Recommended Tools | Why |
|----------|------------------|-----|
| **Startup (< 10 engineers)** | MLflow + vLLM + Prefect | Simple, open-source, low overhead |
| **Mid-size (10-50 engineers)** | W&B + BentoML + Dagster | Better collaboration, more features |
| **Enterprise (50+ engineers)** | Kubeflow + Seldon + Evidently + Custom Platform | Scalable, customizable |
| **Cloud-native (AWS)** | SageMaker + CloudWatch + S3 | Deep AWS integration |
| **Cloud-native (GCP)** | Vertex AI + BigQuery + GCS | Deep GCP integration |
| **Cost-sensitive** | vLLM + Spot instances + Kubecost | Maximize cost efficiency |
| **Regulated industry** | Custom platform + audit logging + compliance | Full control |

### Quick Start Recommendation

**For most teams starting out:**

1. **Tracking:** MLflow (free, all-in-one)
2. **Serving:** vLLM (fast, OpenAI-compatible)
3. **Orchestration:** Prefect (easy, Python-native)
4. **Monitoring:** Evidently (open-source, LLM-aware)
5. **Feature Store:** Feast (when ready)

### Migration Paths

```
Starting → Scaling → Enterprise

MLflow → MLflow + W&B → Custom Platform
vLLM → vLLM + TGI → Seldon/KServe
Prefect → Dagster → Kubeflow
Evidently → WhyLabs → Custom Monitoring
Feast → Tecton → Custom Feature Store
```

---

## Cross-References

- **[01-Overview.md](01-Overview.md)** — High-level overview
- **[02-Core-Topics.md](02-Core-Topics.md)** — Core topics and patterns
- **[03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md)** — Technical implementation
- **[05-Future-Outlook.md](05-Future-Outlook.md)** — Future trends

---

*Next: [05-Future-Outlook.md](05-Future-Outlook.md) — Future trends and predictions*

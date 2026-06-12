# 10 — AI Tools Ecosystem

> **Purpose:** Comprehensive overview of the AI tools ecosystem — from development through deployment, monitoring, data management, vector databases, and compute.
> **Last Updated:** 2026-06-12
> **Maintainer:** AI Knowledge Base Team

---

## Table of Contents

1. [Introduction](#introduction)
2. [Development Tools](#development-tools)
3. [Model Serving & Inference](#model-serving--inference)
4. [Deployment & Orchestration](#deployment--orchestration)
5. [Monitoring & Observability](#monitoring--observability)
6. [Data Tools](#data-tools)
7. [Vector Databases](#vector-databases)
8. [Compute & Infrastructure](#compute--infrastructure)
9. [Experiment Tracking](#experiment-tracking)
10. [CI/CD for ML](#cicd-for-ml)
11. [Security & Governance](#security--governance)
12. [Cost Management](#cost-management)
13. [Integrated Platforms](#integrated-platforms)
14. [Decision Guides](#decision-guides)
15. [Further Reading](#further-reading)

---

## Introduction

The AI tools ecosystem has matured rapidly. This document provides a structured overview of the tools you need at each stage of the AI development lifecycle.

### The AI Development Lifecycle

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Develop    │───▶│   Train     │───▶│   Evaluate  │───▶│   Deploy    │
│  - IDE      │    │  - Compute  │    │  - Benchmarks│   │  - Serving  │
│  - Notebook │    │  - Tracking │    │  - Evals    │    │  - CI/CD    │
│  - Prompt   │    │  - Data     │    │  - Safety   │    │  - Scale    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  Iterate    │◀───│   Monitor   │◀───│    Serve    │◀────────┘
│  - Feedback │    │  - Logging  │    │  - API      │
│  - Retrain  │    │  - Alerts   │    │  - Vector DB│
│  - Improve  │    │  - Cost     │    │  - Cache    │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Tool Selection Principles

1. **Start simple** — Use the simplest tool that solves your problem
2. **Match your stack** — Python? TypeScript? .NET? Choose tools in your ecosystem
3. **Consider total cost** — Open-source has infra costs; SaaS has per-token costs
4. **Check community health** — Stars, commits, contributors, responsiveness
5. **Plan for production** — What works in a notebook may not scale

---

## Development Tools

### IDEs & Editors

| Tool | Best For | Key Features | Cost | AI Features |
|------|----------|-------------|------|-------------|
| **VS Code** | All-purpose | Extensive extensions, remote dev | Free | GitHub Copilot, Continue.dev |
| **Cursor** | AI-first coding | Built-in AI, agent mode, context | $20/month | Native AI, diff application |
| **Windsurf** | AI-first coding | Cascade agent, multi-file edits | $20/month | Deep code understanding |
| **PyCharm** | Python development | Refactoring, debugging, profiling | Free/Paid | JetBrains AI Assistant |
| **Vim/Neovim** | Terminal power users | Keyboard-driven, extensible | Free | Copilot plugin, Cursorless |
| **Jupyter Lab** | Data exploration | Interactive notebooks, visualization | Free | AI extensions available |
| **Google Colab** | Quick experimentation | Free GPU, cloud notebooks | Free/Paid | Gemini integration |
| **GitHub Codespaces** | Cloud development | Pre-configured dev environments | Free tier | Copilot baked in |

### VS Code Extensions for AI

| Extension | Purpose | Installations |
|-----------|---------|---------------|
| **GitHub Copilot** | Code completion, chat | 30M+ |
| **Continue.dev** | Open-source AI assistant | 2M+ |
| **Cursorless** | Voice-driven coding | 500K+ |
| **AI Code Reviewer** | PR review automation | 1M+ |
| **GitLens** | Git blame, code insights | 25M+ |
| **Markdown Preview Mermaid** | Diagram rendering | 5M+ |
| **Rainbow CSV** | CSV file handling | 5M+ |
| **Jupyter** | Notebook support | 20M+ |
| **Python** | Python language support | 50M+ |

### Prompt Engineering Tools

| Tool | Description | Cost |
|------|-------------|------|
| **OpenAI Playground** | Test prompts with all parameters | Free with API key |
| **Anthropic Console** | Prompt testing and evaluation | Free |
| **LangSmith** | Prompt versioning, testing, monitoring | Free tier |
| **PromptLayer** | Prompt logging and analysis | Free tier |
| **Agenta** | Prompt management and A/B testing | Open-source |
| **GPTForge** | Prompt experimentation | Open-source |
| **Portkey** | Prompt versioning and deployments | Free tier |
| **DSPy** | Programmatic prompt optimization | Open-source |

### Notebooks & Prototyping

| Tool | Description | Best For |
|------|-------------|----------|
| **JupyterLab** | Classic notebook environment | Data exploration, research |
| **DeepNote** | Cloud notebook with GPUs | Collaborative research |
| **Hex** | Collaborative data notebook | Data science teams |
| **Observable** | Reactive JavaScript notebooks | Visualization |
| **Noteable** | Cloud notebooks for teams | Education, collaboration |
| **Saturn Cloud** | Python + R data science platform | Enterprise data science |

---

## Model Serving & Inference

### Inference Engines

| Tool | Language | Hardware | Throughput | Key Feature |
|------|----------|----------|------------|-------------|
| **vLLM** | Python | GPU | ⭐⭐⭐⭐⭐ | PagedAttention, continuous batching |
| **TGI (Text Generation Inference)** | Rust/Python | GPU | ⭐⭐⭐⭐ | HuggingFace ecosystem integration |
| **Ollama** | Go | CPU/GPU | ⭐⭐⭐ | Easiest local inference |
| **llama.cpp** | C/C++ | CPU/GPU | ⭐⭐⭐⭐ | Maximum hardware compatibility |
| **ExLlamaV2** | Python/CUDA | GPU | ⭐⭐⭐⭐⭐ | Fastest for Llama-family models |
| **CTranslate2** | C++/Python | CPU/GPU | ⭐⭐⭐⭐ | Intel CPU optimized |
| **TensorRT-LLM** | C++/Python | NVIDIA GPU | ⭐⭐⭐⭐⭐ | Maximum NVIDIA performance |
| **OpenVINO** | C++/Python | Intel CPU/GPU | ⭐⭐⭐⭐ | Intel hardware optimized |
| **Marlin** | CUDA | GPU | ⭐⭐⭐⭐⭐ | Optimized 4-bit inference |
| **SGLang** | Python | GPU | ⭐⭐⭐⭐⭐ | Structured generation, RadixAttention |

### Quick Comparison: Inference Engines

| Feature | vLLM | TGI | Ollama | llama.cpp |
|---------|------|-----|--------|-----------|
| **Continuous batching** | ✅ | ✅ | ❌ | ❌ |
| **PagedAttention** | ✅ | ❌ | ❌ | ❌ |
| **Quantization** | FP8, INT4, GPTQ, AWQ | FP8, GPTQ, AWQ | GGUF | GGUF |
| **Multimodal** | ✅ | ✅ | ✅ | Limited |
| **Function calling** | ✅ | ✅ | ✅ | ❌ |
| **Streaming** | ✅ | ✅ | ✅ | ✅ |
| **OpenAI-compatible API** | ✅ | ✅ | ✅ | ✅ |
| **Distributed inference** | ✅ | ✅ | ❌ | ❌ |
| **Ease of setup** | Medium | Medium | Very Easy | Medium |

### Serving Frameworks

| Tool | Description | Best For |
|------|-------------|----------|
| **BentoML** | Unified model serving framework | Python model serving |
| **Ray Serve** | Scalable serving on Ray | Distributed serving |
| **KServe** | Kubernetes-native model serving | K8s deployments |
| **Seldon Core** | ML deployment on K8s | Enterprise MLOps |
| **NVIDIA Triton** | Multi-framework inference server | GPU-optimized serving |
| **TorchServe** | PyTorch model serving | PyTorch models |
| **MLflow Serving** | MLflow model deployment | MLflow-integrated teams |
| **FastAPI** | Generic Python web serving | Simple API endpoints |

### API Gateways & Routing

| Tool | Description | Key Feature |
|------|-------------|-------------|
| **Portkey** | AI gateway with guardrails | Observability + cost control |
| **OpenRouter** | Multi-provider LLM API | Access 100+ models |
| **LiteLLM** | OpenAI-compatible proxy for 100+ providers | Drop-in replacement |
| **AI Gateway (Portkey)** | Routing, fallbacks, caching | Production gateway |
| **Zuplo** | API gateway for AI | Rate limiting, auth |

---

## Deployment & Orchestration

### Containerization & Runtime

| Tool | Description | When to Use |
|------|-------------|-------------|
| **Docker** | Container standard | Always — packaging |
| **Podman** | Daemonless container engine | Rootless containers |
| **NVIDIA Container Toolkit** | GPU access in containers | Any GPU workload |
| **Apptainer** | HPC containers | Academic/HPC clusters |

### Orchestration

| Tool | Description | Learning Curve |
|------|-------------|----------------|
| **Kubernetes (K8s)** | Container orchestration | Steep |
| **Docker Compose** | Multi-container local setup | Easy |
| **Nomad (HashiCorp)** | Simple orchestrator | Medium |
| **Airflow** | DAG-based workflow scheduling | Medium |
| **Prefect** | Modern workflow orchestration | Easy-Medium |
| **Dagster** | Data orchestration | Medium |
| **Kubeflow** | ML workflows on K8s | Steep |
| **ZenML** | ML pipeline orchestration | Medium |

### Serverless & Edge

| Service | Description | Best For |
|---------|-------------|----------|
| **AWS Lambda** | Serverless functions | Event-driven AI |
| **Cloudflare Workers** | Edge computing | Low-latency inference |
| **Vercel Edge** | Edge functions | Lightweight AI APIs |
| **Modal** | Serverless GPU compute | Batch inference, training |
| **Banana** | Serverless GPU inference | ML model hosting |
| **Replicate** | Cloud ML model hosting | Quick deployments |
| **Beam** | Serverless GPU compute | Python ML workloads |

### Platform-as-a-Service (PaaS)

| Platform | Description | Cost Model |
|----------|-------------|------------|
| **Modal** | Serverless GPU platform | Per-second billing |
| **Replicate** | ML model hosting + API | Per-prediction |
| **HuggingFace Inference Endpoints** | Managed model hosting | Per-hour GPU |
| **Banana** | Serverless GPU inference | Per-second |
| **Beam** | Serverless Python ML | Per-second |
| **BuildShip** | Visual backend builder | Freemium |

---

## Monitoring & Observability

### LLM Observability Platforms

| Tool | Description | Cost | Self-Host |
|------|-------------|------|-----------|
| **LangSmith** | LangChain-native tracing, evaluation | Free tier | ❌ |
| **Weights & Biases (W&B) Prompts** | LLM prompt monitoring | Free tier | ❌ |
| **MLflow Tracing** | MLflow LLM tracing | Free | ✅ |
| **WhyLabs** | AI observability, drift detection | Free tier | ❌ |
| **Arize Phoenix** | Open-source LLM observability | Free | ✅ |
| **Helicone** | LLM API monitoring | Free tier | ❌ |
| **LangFuse** | Open-source LLM observability | Free | ✅ |
| **TruLens** | LLM evaluation and monitoring | Free | ✅ |
| **SigNoz** | Open-source APM | Free | ✅ |
| **Datadog (LLM Observability)** | Enterprise APM | Paid | ❌ |

### What to Monitor

| Metric | Tooling | Alert Threshold |
|--------|---------|-----------------|
| **Latency (P50, P95, P99)** | LangSmith, Datadog | P95 > 5s |
| **Token usage** | Helicone, Portkey | Per-budget limit |
| **Error rate** | Any APM | > 1% |
| **Cost per request** | Helicone, Portkey | Per-budget limit |
| **Drift (input/output)** | WhyLabs, Arize | PSI > 0.1 |
| **Hallucination rate** | TruLens, RAGAS | > 5% |
| **User satisfaction** | Custom | < 4/5 |
| **Safety violations** | Custom guardrails | Any violation |

### Monitoring Stack Example

```yaml
# Production monitoring stack
monitoring:
  metrics:
    - LangSmith (LLM-specific tracing)
    - Prometheus (system metrics)
    - Grafana (dashboards)
  
  logging:
    - Structured JSON logging
    - ELK stack or Loki (log aggregation)
  
  alerting:
    - PagerDuty / OpsGenie (incident response)
    - Slack webhooks (notifications)
  
  evaluation:
    - RAGAS (weekly RAG quality eval)
    - TruLens (LLM output quality)
    - Custom evaluation pipeline (nightly)
  
  cost:
    - Helicone (per-request cost tracking)
    - Custom budget dashboards
```

---

## Data Tools

### Data Labeling & Annotation

| Tool | Description | Type | Pricing |
|------|-------------|------|---------|
| **Label Studio** | Multi-format data labeling | Open-source | Free / Cloud paid |
| **Argilla** | LLM data annotation | Open-source | Free |
| **Scale AI** | Enterprise data labeling | SaaS | Per-task |
| **Snorkel AI** | Programmatic labeling | Platform | Enterprise |
| **Prodigy** | Active learning annotation | Library | License |
| **CVAT** | Computer vision annotation | Open-source | Free |
| **Segments.ai** | Multi-modal labeling | SaaS | Free tier |
| **Doccano** | Text annotation | Open-source | Free |
| **Lionbridge** | Managed annotation | Service | Per-project |

### Data Processing & Pipelines

| Tool | Description | Best For |
|------|-------------|----------|
| **Apache Spark** | Distributed data processing | Large-scale ETL |
| **Polars** | Fast DataFrame library | Python data processing |
| **Pandas** | Data analysis library | General data work |
| **Dask** | Parallel computing | Scaling pandas workflows |
| **Ray Data** | Distributed data loading | ML training pipelines |
| **Airbyte** | Data integration platform | ELT pipelines |
| **dbt** | Data transformation | Analytics engineering |
| **Great Expectations** | Data quality validation | Data testing |
| **DVC** | Data version control | Versioning datasets |
| **LFS (Git LFS)** | Large file storage | Model/dataset versioning |

### Data Quality & Validation

| Tool | Description | Key Feature |
|------|-------------|-------------|
| **Great Expectations** | Data quality expectations | Automated validation suites |
| **Cleanlab** | Data error detection | Find label errors |
| **Deequ** | AWS data quality library | Scala/PySpark |
| **Pandera** | DataFrame validation | Schema enforcement |
| **TensorFlow Data Validation** | TFX data validation | ML-specific |
| **YData Profiling** | Data profiling reports | Quick EDA reports |

---

## Vector Databases

### Comparison Table

| Feature | Pinecone | Weaviate | Qdrant | Chroma | Milvus | pgvector |
|---------|----------|----------|--------|--------|--------|----------|
| **Type** | Managed | Self-hosted/Cloud | Self-hosted/Cloud | Embedded/Server | Self-hosted/Cloud | PostgreSQL extension |
| **Written in** | - | Go | Rust | Python | Go/Go | C |
| **Open-source** | ❌ | ✅ (BSD) | ✅ (Apache 2.0) | ✅ (Apache 2.0) | ✅ (Apache 2.0) | ✅ (PostgreSQL) |
| **Filtering** | ✅ | ✅ (rich) | ✅ (rich) | ✅ (basic) | ✅ (rich) | ✅ (SQL) |
| **Hybrid search** | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Multi-tenancy** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Index types** | Proprietary | HNSW | HNSW, SCANN | Flat, HNSW | IVF, HNSW, DiskANN | IVFFlat, HNSW |
| **GPU support** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Cloud native** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Ease of setup** | Very Easy | Medium | Medium | Very Easy | Complex | Easy |

### Vector Database Selection Guide

```
What's your primary need?
│
├─ Need a managed, zero-ops solution?
│   └─ Pinecone (simplest, but expensive at scale)
│
├─ Need self-hosted, developer-friendly?
│   ├─ Chroma (prototyping, small scale)
│   └─ Qdrant (production, high performance)
│
├─ Need rich filtering + hybrid search?
│   └─ Weaviate (best filtering capabilities)
│
├─ Need to integrate with existing PostgreSQL?
│   └─ pgvector (use existing DB, zero new infra)
│
├─ Need massive scale (100M+ vectors)?
│   └─ Milvus (distributed, cloud-native)
│
└─ Need GPU-accelerated?
    └─ Milvus (only GPU-enabled option)
```

### Vector Database Quick Start

```python
# Chroma — Quickest to start
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")
collection.add(
    documents=["Document 1 content", "Document 2 content"],
    ids=["doc1", "doc2"]
)
results = collection.query(query_texts=["search query"], n_results=2)
print(results)
```

```python
# Qdrant — Production choice
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient("localhost", port=6333)
client.recreate_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

client.upsert(
    collection_name="my_collection",
    points=[PointStruct(id=1, vector=[0.1]*384, payload={"doc": "content"})],
)

results = client.search(
    collection_name="my_collection",
    query_vector=[0.1]*384,
    limit=10,
)
```

---

## Compute & Infrastructure

### GPU Cloud Providers

| Provider | GPU Types | Pricing Model | Best For |
|----------|-----------|---------------|----------|
| **AWS (EC2)** | A100, H100, L40S, A10G | Per-hour reserved/spot | Enterprise |
| **GCP (Cloud TPU + GPU)** | A100, H100, TPU v5e | Per-second | Google ecosystem |
| **Azure** | A100, H100, ND-series | Per-hour reserved/spot | Microsoft ecosystem |
| **Lambda Labs** | A100, H100, H200 | Per-hour (competitive) | Deep learning |
| **RunPod** | RTX 4090, A100, H100 | Per-second billing | Best value |
| **Vast.ai** | Wide variety (RTX to H100) | Per-hour bidding | Cheapest spot pricing |
| **JarvisLabs** | RTX 4090, A100, H100 | Per-hour | Easy setup |
| **Paperspace** | RTX 4000–6000, A100 | Per-hour + fixed | User-friendly |
| **CoreWeave** | H100, A100 | Per-hour (large clusters) | Kubernetes-native |
| **Crusoe Cloud** | A100, L40S | Per-hour | Low-carbon compute |

### GPU Pricing Comparison (Approximate, June 2026)

| Provider | RTX 4090 (24GB) | A100 (80GB) | H100 (80GB) | Spot Available |
|----------|-----------------|-------------|-------------|----------------|
| **Lambda Labs** | $0.25/hr | $1.10/hr | $2.50/hr | ❌ |
| **RunPod** | $0.20/hr | $0.80/hr | $1.50/hr | ✅ |
| **Vast.ai** | $0.15/hr | $0.50/hr | $1.20/hr | ✅ |
| **AWS (on-demand)** | ~$0.60/hr (L4) | ~$3.00/hr | ~$4.00/hr | ✅ |
| **JarvisLabs** | $0.30/hr | $0.90/hr | $2.00/hr | ❌ |
| **Paperspace** | $0.35/hr | $1.50/hr | $2.80/hr | ✅ |

### Inference-Specific Providers

| Provider | Focus | Key Differentiator |
|----------|-------|-------------------|
| **Groq** | Ultra-low latency | LPU architecture, fast token generation |
| **together.ai** | Broad model access | 100+ open models, fine-tuning API |
| **Fireworks AI** | Fast inference | Optimized for speed, function calling |
| **Anyscale** | Ray-native serving | Best for distributed inference |
| **Replicate** | Model API marketplace | Easy API, many models |

### Training Infrastructure

| Service | Description | Best For |
|---------|-------------|----------|
| **NVIDIA DGX Cloud** | Managed supercomputing | Enterprise training |
| **Lambda GPU Cloud** | Large cluster training | Deep learning labs |
| **RunPod** | Affordable training pods | Independent researchers |
| **Modal** | Serverless GPU training | Batch jobs, fine-tuning |
| **Anyscale** | Ray-based distributed training | Large-scale training |
| **FSDP + DeepSpeed** | Training libraries | Custom distributed training |
| **NeMo (NVIDIA)** | LLM training framework | Enterprise model training |

### Local Inference Hardware

| Hardware | Performance | Models | Power | Cost |
|----------|------------|--------|-------|------|
| **Apple M4 Max (128GB)** | Excellent | Up to 70B 4-bit | 40W | $4,500+ |
| **Apple M3 Ultra** | Excellent | Up to 120B 4-bit | 60W | $7,000+ |
| **RTX 4090 (24GB)** | Very Good | Up to 34B 4-bit | 450W | $1,600 |
| **RTX 5090 (32GB?)** | Excellent | Up to 45B 4-bit | 600W | $2,000+ |
| **RTX 6000 Ada (48GB)** | Excellent | Up to 70B 4-bit | 300W | $6,800 |
| **2x RTX 3090 (48GB)** | Good | Up to 70B 4-bit | 700W | $1,200 used |
| **Mac Studio M2 Ultra** | Very Good | Up to 70B 4-bit | 90W | $4,000+ |

---

## Experiment Tracking

### Tracking Platforms

| Tool | Description | Self-Host | Cost |
|------|-------------|-----------|------|
| **Weights & Biases** | Most popular experiment tracker | ❌ (Cloud) | Free tier, paid teams |
| **MLflow** | Open-source ML lifecycle | ✅ | Free |
| **Neptune.ai** | ML metadata store | ❌ (Cloud) | Free tier, paid teams |
| **Comet** | Experiment tracking | ❌ (Cloud) | Free tier, paid teams |
| **ClearML** | Open-source ML platform | ✅ | Free |
| **DVC** | ML experiment management | ✅ | Free |
| **Sacred** | Experiment config + logging | ✅ | Free |
| **Aim** | Open-source experiment tracker | ✅ | Free |
| **Kubeflow** | K8s ML workflows | ✅ | Free / Infrastructure cost |

### What to Track

```
Every experiment should capture:
├─ Parameters (hyperparameters, configs)
├─ Metrics (accuracy, loss, F1, etc.)
├─ Artifacts (model weights, checkpoints)
├─ Source code (commit hash)
├─ Environment (dependencies, OS, CUDA version)
├─ Dataset (version, hash, split)
├─ Duration (training time)
├─ Cost (compute cost, API cost)
└─ Notes (what changed, why, observations)
```

### MLflow Quick Start

```python
import mlflow

# Set experiment
mlflow.set_experiment("fine-tune-llama")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model", "Llama-3.1-8B")
    mlflow.log_param("learning_rate", 2e-4)
    mlflow.log_param("batch_size", 4)
    mlflow.log_param("lora_r", 16)
    
    # Log metrics
    mlflow.log_metric("train_loss", 0.45)
    mlflow.log_metric("eval_loss", 0.52)
    mlflow.log_metric("eval_accuracy", 0.87)
    
    # Log model
    mlflow.transformers.log_model(
        transformers_model=model,
        artifact_path="model",
        task="text-generation"
    )
    
    # Log dataset info
    mlflow.log_artifact("dataset_info.json")
    
    # Log notes
    mlflow.set_tag("notes", "Tried higher learning rate, eval loss improved")
```

---

## CI/CD for ML (MLOps)

### CI/CD Tools

| Tool | Description | Stage |
|------|-------------|-------|
| **GitHub Actions** | CI/CD integrated with GitHub | All |
| **GitLab CI** | Integrated CI/CD | All |
| **Jenkins** | Classic CI/CD | Enterprise |
| **CircleCI** | Cloud CI/CD | Build/test |
| **ArgoCD** | Kubernetes deployment | CD |
| **Flux** | GitOps for K8s | CD |
| **Atlantis** | Terraform PR automation | Infra changes |

### ML Pipeline Orchestration

| Tool | Description | Abstraction Level |
|------|-------------|-------------------|
| **Kubeflow Pipelines** | K8s-native ML pipelines | High |
| **ZenML** | ML pipeline framework | Medium-High |
| **Prefect** | Workflow orchestration | Medium |
| **Airflow** | DAG-based scheduling | Medium |
| **Flyte** | ML-aware workflow engine | Medium-High |
| **Metaflow** | Human-centric ML framework | High |
| **TFX (TensorFlow Extended)** | Production ML pipelines | High |

### Feature Stores

| Tool | Description | Integration |
|------|-------------|-------------|
| **Feast** | Open-source feature store | Offline/online serving |
| **Tecton** | Enterprise feature platform | Cloud-native |
| **Hopsworks** | Feature store + platform | Full stack |
| **Vertex AI Feature Store** | GCP-managed | GCP ecosystem |
| **SageMaker Feature Store** | AWS-managed | AWS ecosystem |

### MLOps Stack Examples

```yaml
# Startup MLOps Stack (~$500/month)
development:
  - IDE: VS Code + Continue.dev
  - Tracking: MLflow (self-hosted)
  - Compute: RunPod spot instances

production:
  - Serving: vLLM + FastAPI
  - Monitoring: LangFuse (self-hosted)
  - CI/CD: GitHub Actions
  - Vector DB: Qdrant (self-hosted)
  - Data: Label Studio (self-hosted)
```

```yaml
# Enterprise MLOps Stack (~$10K+/month)
development:
  - IDE: Cursor / PyCharm Professional
  - Tracking: Weights & Biases Enterprise
  - Compute: AWS SageMaker / GCP Vertex AI

production:
  - Serving: Ray Serve + vLLM
  - Monitoring: Datadog + WhyLabs
  - CI/CD: ArgoCD + GitHub Actions
  - Orchestration: Kubeflow
  - Vector DB: Pinecone / Weaviate Cloud
  - Feature Store: Tecton / Feast
  - Data: Scale AI (labeling) + Great Expectations (quality)
```

---

## Security & Governance

### Guardrails & Safety Tools

| Tool | Description | Type |
|------|-------------|------|
| **Guardrails AI** | Input/output validation | Open-source |
| **NVIDIA NeMo Guardrails** | Safety guardrails toolkit | Open-source |
| **Lakera Guard** | LLM security (jailbreak detection) | SaaS |
| **Rebuff** | Prompt injection detection | Open-source |
| **Garak** | LLM vulnerability scanning | Open-source |
| **LLM Guard** | Security scanning framework | Open-source |
| **NeMo Curator** | Data curation for safety | Open-source |

### Secrets & Key Management

| Tool | Description | Type |
|------|-------------|------|
| **HashiCorp Vault** | Secrets management | Self-hosted/Cloud |
| **AWS Secrets Manager** | AWS secrets | Cloud |
| **Azure Key Vault** | Azure secrets | Cloud |
| **GCP Secret Manager** | GCP secrets | Cloud |
| **Doppler** | Universal secrets management | Cloud |
| **1Password CLI** | Developer secrets | Cloud/Desktop |
| **dotenv** | Local env variables | Library |

---

## Cost Management

### Cost Tracking Tools

| Tool | Description | Key Feature |
|------|-------------|-------------|
| **Helicone** | LLM API cost tracking | Per-request cost |
| **LangSmith** | LLM cost + usage analytics | Integrated with LangChain |
| **Portkey** | Cost + usage dashboard | Multi-provider |
| **Vantage** | Cloud cost management | All cloud providers |
| **CloudZero** | Engineering cost intelligence | Per-feature costs |

### Cost Optimization Strategies

| Strategy | Savings | Complexity |
|----------|---------|------------|
| **Model tiering** (use small model for simple tasks) | 40-70% | Low |
| **Caching** (cache common requests) | 30-60% | Medium |
| **Batching** (batch API calls) | 20-40% | Low |
| **Prompt compression** (reduce token count) | 20-40% | Low |
| **Quantization** (INT4/FP8) | 50-75% (GPU) | Low |
| **Spot instances** (for training) | 60-90% | Medium |
| **Semantic caching** (cache similar queries) | 20-50% | High |
| **Autoscaling** (scale to zero when idle) | 30-70% | Medium |

### Budget Tracking Template

```python
# Simple cost tracking script
costs = {
    "openai_gpt4o": {"per_input_tokens": 0.00001, "per_output_tokens": 0.00003},
    "openai_gpt4o_mini": {"per_input_tokens": 0.0000015, "per_output_tokens": 0.000006},
    "anthropic_claude_op4": {"per_input_tokens": 0.000015, "per_output_tokens": 0.000075},
    "together_mixtral": {"per_input_tokens": 0.0000009, "per_output_tokens": 0.0000009},
}

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = costs[model]
    input_cost = input_tokens * pricing["per_input_tokens"]
    output_cost = output_tokens * pricing["per_output_tokens"]
    return input_cost + output_cost

# Daily estimate
daily_usage = {
    "openai_gpt4o": {"input": 500_000, "output": 100_000},
    "openai_gpt4o_mini": {"input": 2_000_000, "output": 500_000},
}

total_daily_cost = sum(
    estimate_cost(model, data["input"], data["output"])
    for model, data in daily_usage.items()
)
print(f"Daily cost: ${total_daily_cost:.2f}")
print(f"Monthly cost: ${total_daily_cost * 30:.2f}")
```

---

## Integrated Platforms

### All-in-One AI Platforms

| Platform | Description | Best For |
|----------|-------------|----------|
| **HuggingFace** | Models, datasets, Spaces, Inference | Full AI ecosystem |
| **Replicate** | Model API marketplace | Deploying open models |
| **Google Vertex AI** | GCP AI platform | GCP users |
| **AWS SageMaker** | AWS ML platform | AWS users |
| **Azure AI Studio** | Microsoft AI platform | Azure/Microsoft users |
| **Databricks** | Data + AI platform | Large-scale data + ML |
| **Anyscale** | Ray-based ML platform | Distributed workloads |
| **Weights & Biases** | Experiment tracking + more | ML development teams |
| **Domino Data Lab** | Enterprise MLOps | Regulated industries |

### Low-Code / No-Code Platforms

| Platform | Description | Best For |
|----------|-------------|----------|
| **Dify** | Visual LLM app builder | Quick prototyping |
| **Flowise** | Visual LangChain builder | LangChain without code |
| **Langflow** | Visual flow builder | Custom AI workflows |
| **BuildShip** | Visual backend builder | Full-stack AI apps |
| **Zapier AI** | AI automation | Business automation |
| **n8n** | Workflow automation | AI + 400 integrations |

---

## Decision Guides

### Tool Selection by Task

| Task | Primary Tool | Alternative |
|------|-------------|-------------|
| **Prototype an LLM app** | LangChain / Streamlit | Dify / Gradio |
| **Fine-tune a model** | Axolotl / Unsloth | PEFT / TRL |
| **Deploy an LLM API** | vLLM + FastAPI | TGI / Ollama |
| **Build a RAG system** | LangChain / LlamaIndex | Haystack / Dify |
| **Monitor an LLM app** | LangSmith / LangFuse | W&B Prompts / Helicone |
| **Track experiments** | MLflow / W&B | Neptune / Comet |
| **Process training data** | HuggingFace Datasets | Argilla / Distilabel |
| **Manage vector data** | Qdrant / Chroma | Pinecone / Weaviate |

### Tool Selection by Scale

| Scale | Users | Tools |
|-------|-------|-------|
| **Individual** | 1-10 | Ollama, Chroma, MLflow, Streamlit |
| **Team** | 10-100 | vLLM, Qdrant, W&B, LangSmith |
| **Organization** | 100-10K | Ray Serve, Milvus/Pinecone, Datadog, Kubeflow |
| **Enterprise** | 10K+ | Triton, K8s, ArgoCD, WhyLabs, Vault |

### Tool Selection by Budget

| Budget | Compute | Platforms | Monitoring |
|--------|---------|-----------|------------|
| **$0-$100/month** | RunPod spot, Colab Pro | HuggingFace, Replicate (free tier) | LangFuse (self-host) |
| **$100-$1K/month** | Lambda Labs, RunPod | Dify (self-host), Modal | LangSmith (dev tier) |
| **$1K-$10K/month** | AWS/GCP reserved | SageMaker/Vertex AI | Datadog, WhyLabs |
| **$10K+/month** | Dedicated clusters | Enterprise platforms | Full observability stack |

---

## Further Reading

- [04-Agent-Toolkits.md](04-Agent-Toolkits.md) — Agent framework comparisons
- [06-Awesome-AI-Repos.md](06-Awesome-AI-Repos.md) — GitHub repositories for these tools
- [07-AI-2026-Roadmap.md](07-AI-2026-Roadmap.md) — When to learn which tools
- [09-Community-Forums-Events.md](09-Community-Forums-Events.md) — Communities for each tool
- [State of AI Report](https://www.stateof.ai/) — Annual AI landscape overview
- [MLOps.toys](https://mlops.toys/) — Visual MLOps tool comparison

---

*Document version 1.0 — Last updated 2026-06-12*

# MLOps & AI Platform Engineering — Overview

> **Category:** 56 — MLOps & AI Platform Engineering  
> **Last Updated:** July 2026  
> **Cross-references:** [05-Enterprise/04-AI-Infrastructure.md](../05-Enterprise/04-AI-Infrastructure.md), [33-AI-Native-Software-Development/](../33-AI-Native-Software-Development/), [20-Agent-Infrastructure-and-Observability/](../20-Agent-Infrastructure-and-Observability/), [41-AI-Cost-Optimization/](../41-AI-Cost-Optimization-and-Enterprise-ROI/)

---

## Table of Contents

1. [What is MLOps & AI Platform Engineering?](#1-what-is-mlops--ai-platform-engineering)
2. [Why It Matters in 2026](#2-why-it-matters-in-2026)
3. [Core Principles](#3-core-principles)
4. [Maturity Model](#4-maturity-model)
5. [Key Components](#5-key-components)
6. [Platform Engineering vs MLOps](#6-platform-engineering-vs-mlops)
7. [The AI Platform Stack](#7-the-ai-platform-stack)
8. [Industry Landscape 2026](#8-industry-landscape-2026)
9. [Getting Started Guide](#9-getting-started-guide)
10. [Common Anti-Patterns](#10-common-anti-patterns)

---

## 1. What is MLOps & AI Platform Engineering?

**MLOps** (Machine Learning Operations) is a set of practices that combines Machine Learning, DevOps, and Data Engineering to deploy and maintain ML systems in production reliably and efficiently.

**AI Platform Engineering** extends MLOps into a broader discipline focused on building internal developer platforms (IDPs) that enable teams to build, deploy, monitor, and iterate on AI/ML applications with minimal friction.

### The Convergence

In 2026, these two disciplines have converged:

```
Traditional MLOps (2020-2024)
  ├── Model training pipelines
  ├── Experiment tracking
  ├── Model versioning
  └── Basic monitoring

AI Platform Engineering (2025-2026)
  ├── Everything in MLOps
  ├── Internal Developer Platform (IDP)
  ├── Self-service infrastructure
  ├── GPU resource management
  ├── Cost attribution & governance
  ├── Multi-model orchestration
  ├── Agent deployment pipelines
  └── AI-native CI/CD
```

### Key Metrics (2026 Industry Benchmarks)

| Metric | Good | Great | Elite |
|--------|------|-------|-------|
| Model-to-production time | < 2 weeks | < 3 days | < 4 hours |
| Training pipeline success rate | > 90% | > 97% | > 99.5% |
| Model serving uptime | > 99% | > 99.9% | > 99.99% |
| Inference latency (p99) | < 500ms | < 100ms | < 20ms |
| GPU utilization | > 40% | > 65% | > 85% |
| Cost per inference | Baseline | -30% | -60% |
| Experiment reproducibility | > 80% | > 95% | > 99% |
| Time to detect model drift | < 24h | < 1h | < 5 min |
| Rollback time | < 1h | < 10min | < 30s |

---

## 2. Why It Matters in 2026

### The Production Crisis

> "95% of AI agents never reach production" — this stat from early 2026 has become a rallying cry for the MLOps community.

The gap between prototype and production has never been wider:

- **Training costs** have dropped 10x since 2024 (inference cost per token is 92% lower)
- **Deployment complexity** has increased 5x (multi-modal, multi-agent, multi-cloud)
- **Regulatory requirements** have exploded (EU AI Act, state-level regulations)
- **Expectations** are astronomical (real-time, always-on, self-healing)

### The Platform War

Hacker News front page (July 2026): "We'll fight the platform war against big AI"

This reflects a real industry trend:
- **Cloud providers** (AWS, GCP, Azure) competing for AI platform dominance
- **AI-native platforms** (Modal, Together, Anyscale, Fireworks) offering purpose-built infrastructure
- **Open-source platforms** (BentoML, Ray, Kubeflow) providing alternatives
- **Enterprise platforms** (Databricks, Snowflake, Palantir) integrating AI natively

### Cost Reality Check

"Performance per dollar is getting faster and cheaper" (Wafer.ai, HN front page July 2026)

But cost optimization requires platform engineering:
- **GPU costs**: $2-4/hour per H100, $0.50-1/hour per L4
- **Inference costs**: GPT-4o at $2.50/1M input tokens, Claude Opus at $15/1M input tokens
- **Training costs**: Fine-tuning a 7B model: ~$50-200; Pre-training a 70B model: ~$2-5M
- **Hidden costs**: Data storage, monitoring, logging, compliance, engineering time

---

## 3. Core Principles

### 3.1 Infrastructure as Code (IaC) for AI

Every AI resource must be reproducible:

```yaml
# Example: AI Platform Resource Definition (2026 style)
apiVersion: ai.platform/v1
kind: ModelServing
metadata:
  name: production-llm-router
  namespace: ai-platform
  labels:
    team: ml-platform
    cost-center: engineering
    tier: critical
spec:
  model:
    registry: ghcr.io/company/models
    name: llama-3-70b-instruct
    version: "2.1.0"
    checksum: sha256:abc123...
  serving:
    framework: vllm
    replicas: 3
    resources:
      gpu: nvidia-h100-80gb
      gpuCount: 2
      memory: 128Gi
    autoscaling:
      minReplicas: 1
      maxReplicas: 10
      targetLatencyMs: 100
      targetGPUUtilization: 70
  monitoring:
    metrics:
      - inference_latency_p99
      - throughput_tokens_per_sec
      - gpu_utilization
      - cost_per_1k_tokens
    alerts:
      - name: latency_slo_breach
        threshold: 200ms
        severity: critical
      - name: cost_spike
        threshold: 2x_baseline
        severity: warning
  routing:
    strategy: canary
    trafficSplit:
      stable: 90
      canary: 10
```

### 3.2 Self-Service Everything

Platform teams build; product teams consume:

```
Developer Experience Layer:
  ├── One-click deployment (CLI or UI)
  ├── Environment templates
  ├── Auto-generated APIs
  ├── Built-in monitoring dashboards
  └── Cost visibility

Platform Layer:
  ├── GPU resource pooling
  ├── Model registry
  ├── Feature store
  ├── Experiment tracking
  └── Pipeline orchestration

Infrastructure Layer:
  ├── Kubernetes + GPU operators
  ├── Object storage (S3/GCS)
  ├── Message queues (Kafka/Redpanda)
  ├── Databases (PostgreSQL/Redis)
  └── Observability stack
```

### 3.3 Shift-Left on AI Quality

AI quality cannot be an afterthought:

| Traditional CI/CD | AI-Native CI/CD |
|-------------------|-----------------|
| Unit tests | + Model quality tests |
| Integration tests | + Data quality tests |
| Security scans | + Adversarial robustness tests |
| Performance tests | + Inference latency benchmarks |
| Code review | + Model card review |
| Dependency audit | + Training data audit |

### 3.4 Cost Transparency

Every AI operation must have a cost label:

```
cost_attribution:
  team: "search-team"
  project: "semantic-search-v3"
  environment: "production"
  model: "embedding-model-v2"
  
Cost breakdown per request:
  - GPU compute: $0.0002
  - API calls: $0.0001
  - Storage: $0.00001
  - Monitoring: $0.00005
  - Total: $0.00036/request
```

---

## 4. Maturity Model

### Level 0: Manual
- Models trained on laptops
- Manual deployment via SSH
- No monitoring
- No version control for models or data

### Level 1: Pipeline
- Automated training pipelines
- Basic experiment tracking (MLflow/W&B)
- Model registry
- Manual deployment with approval

### Level 2: Platform
- Self-service model deployment
- Automated testing (data + model)
- Centralized monitoring
- Cost attribution
- Feature store

### Level 3: Product
- Internal Developer Platform (IDP)
- One-click everything
- Real-time monitoring with auto-remediation
- Multi-model orchestration
- A/B testing built-in
- Full cost governance

### Level 4: Autonomous
- Self-healing systems
- Automated model retraining
- Predictive scaling
- Full auditability for compliance
- AI managing AI infrastructure

---

## 5. Key Components

### 5.1 The MLOps Toolchain (2026)

| Layer | Tools | Purpose |
|-------|-------|---------|
| **Orchestration** | Kubeflow, Argo Workflows, Prefect, Dagster | Pipeline management |
| **Experiment Tracking** | MLflow, W&B, ClearML, Neptuna | Track experiments |
| **Model Registry** | MLflow Model Registry, Vertex AI Model Registry | Version models |
| **Feature Store** | Feast, Tecton, Hopsworks, Databricks Feature Store | Feature management |
| **Serving** | vLLM, TGI, BentoML, Seldon Core, KServe | Deploy models |
| **Monitoring** | Evidently, WhyLabs, Arize Phoenix, Langfuse | Monitor quality |
| **Data Versioning** | DVC, LakeFS, Delta Lake | Track data |
| **GPU Management** | NVIDIA GPU Operator, Run:ai, Grid.ai | GPU allocation |
| **Cost Management** | Kubecost, CloudHealth, Spot by NetApp | Cost optimization |
| **Security** | Snyk, Trivy, Falco | Security scanning |

### 5.2 Platform Components

```
┌─────────────────────────────────────────────────────┐
│                  Developer Portal                    │
│  (Backstage, Port, Custom UI)                       │
├─────────────────────────────────────────────────────┤
│                Self-Service Layer                    │
│  ├── Environment Scaffolding                        │
│  ├── Model Deployment Wizard                        │
│  ├── Pipeline Templates                             │
│  ├── Monitoring Dashboard Generator                 │
│  └── Cost Attribution Dashboard                     │
├─────────────────────────────────────────────────────┤
│               Platform Services                     │
│  ├── Model Registry     ├── Feature Store           │
│  ├── Experiment Tracker  ├── Pipeline Orchestrator   │
│  ├── Model Serving       ├── Monitoring Stack        │
│  ├── GPU Scheduler       ├── Cost Manager            │
│  └── Secret Manager      └── Compliance Gateway      │
├─────────────────────────────────────────────────────┤
│              Infrastructure Layer                   │
│  ├── Kubernetes (EKS/GKE/AKS)                       │
│  ├── GPU Nodes (H100, A100, L4, T4)                 │
│  ├── Storage (S3, EFS, Local SSD)                   │
│  ├── Networking (Service Mesh, Load Balancers)       │
│  └── Observability (Prometheus, Grafana, Loki)       │
└─────────────────────────────────────────────────────┘
```

### 5.3 The Platform Team

The platform team structure has evolved:

```
AI Platform Team (2026)
├── Platform Engineers (2-4)
│   ├── Kubernetes & GPU management
│   ├── CI/CD pipeline maintenance
│   └── Developer tooling
├── ML Engineers (2-3)
│   ├── Model serving optimization
│   ├── Training infrastructure
│   └── Feature engineering
├── Data Engineers (1-2)
│   ├── Data pipeline reliability
│   ├── Feature store management
│   └── Data quality monitoring
├── SRE/Reliability (1-2)
│   ├── Monitoring & alerting
│   ├── Incident response
│   └── Capacity planning
└── Product Manager (1)
    ├── Developer experience
    ├── Roadmap prioritization
    └── Stakeholder management
```

---

## 6. Platform Engineering vs MLOps

### Differences

| Aspect | MLOps | AI Platform Engineering |
|--------|-------|------------------------|
| **Focus** | ML lifecycle | Developer experience |
| **Users** | ML engineers | All AI developers |
| **Goal** | Reliable ML systems | Self-service AI platform |
| **Scope** | Training → serving | Full AI development lifecycle |
| **Metrics** | Model performance | Developer productivity |
| **Approach** | Tool-centric | Product-centric |

### How They Overlap

Both share:
- Model versioning and registry
- Pipeline orchestration
- Monitoring and observability
- Cost management
- Security and compliance

### The 2026 Reality

Most organizations need both:
- **MLOps** for the technical ML workflow
- **Platform Engineering** for the developer experience layer

The platform team builds the MLOps platform; the MLOps team uses it.

---

## 7. The AI Platform Stack

### Complete Stack (2026)

```
┌────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                       │
│  Chatbots │ Copilots │ Agents │ Search │ Recommendation   │
├────────────────────────────────────────────────────────────┤
│                     API LAYER                              │
│  Gateway │ Rate Limiting │ Auth │ Caching │ Routing        │
├────────────────────────────────────────────────────────────┤
│                   MODEL LAYER                              │
│  LLM Router │ Ensemble │ Fallback │ A/B Test │ Canary      │
├────────────────────────────────────────────────────────────┤
│                  SERVING LAYER                             │
│  vLLM │ TGI │ TensorRT-LLM │ Triton │ BentoML │ KServe   │
├────────────────────────────────────────────────────────────┤
│                ORCHESTRATION LAYER                         │
│  Kubernetes │ Argo │ Kubeflow │ Ray Serve │ Modal          │
├────────────────────────────────────────────────────────────┤
│                  RESOURCE LAYER                            │
│  GPU Pool │ CPU Pool │ Memory │ Storage │ Network          │
├────────────────────────────────────────────────────────────┤
│                 OBSERVABILITY LAYER                        │
│  Metrics │ Logs │ Traces │ Alerts │ Dashboards │ Cost      │
├────────────────────────────────────────────────────────────┤
│                  DATA LAYER                                │
│  Feature Store │ Vector DB │ Time Series │ Object Store    │
└────────────────────────────────────────────────────────────┘
```

### Component Deep-Dives

| Component | 2024 State | 2026 State | Key Change |
|-----------|-----------|-----------|------------|
| **Model Serving** | Single model per endpoint | Multi-model routing, dynamic batching | AI-native serving |
| **GPU Management** | Manual allocation | Dynamic partitioning, time-sharing | GPU as a service |
| **Monitoring** | Latency + errors | Quality + cost + drift + fairness | Holistic observability |
| **CI/CD** | Deploy model | Test data + model + infra + cost | Full lifecycle |
| **Feature Store** | Batch features | Real-time + streaming features | Sub-second features |
| **Experiment Tracking** | Log metrics | Full lineage + reproducibility | Audit trail |

---

## 8. Industry Landscape 2026

### Major Platform Players

| Platform | Type | Strengths | Weaknesses |
|----------|------|-----------|------------|
| **AWS SageMaker** | Cloud-native | Deep AWS integration, broad features | AWS lock-in, complex pricing |
| **Google Vertex AI** | Cloud-native | BigQuery integration, AutoML | GCP lock-in |
| **Azure ML** | Cloud-native | Enterprise compliance, Office integration | Azure lock-in |
| **Databricks** | Data + AI | Lakehouse, Unity Catalog, MLflow | Premium pricing |
| **Modal** | AI-native | Serverless GPU, fast iteration | Limited enterprise features |
| **Together AI** | AI-native | Open-source focus, fast inference | Narrower platform |
| **Anyscale (Ray)** | Distributed | Scalable compute, flexible | Complex setup |
| **BentoML** | Open-source | Model serving, self-hosted | DIY platform |
| **Kubeflow** | Open-source | Kubernetes-native, flexible | Complex, steep learning curve |
| **Weights & Biases** | Experiment tracking | Best-in-class tracking | Not a full platform |

### Open Source Trends

The open-source AI platform stack is maturing:

1. **Kubeflow 2.0** — Simplified, production-ready
2. **Ray 3.0** — Native GPU scheduling, improved observability
3. **MLflow 3.0** — Full lifecycle, LLM-native tracking
4. **Feast 1.0** — Real-time feature serving
5. **BentoML 2.0** — Multi-model serving, GPU optimization
6. **OpenLLM** — Open-source LLM serving
7. **vLLM** — High-performance LLM inference

### Emerging Trends

1. **AI Platform as Internal Product** — Treating the platform like a product with users
2. **GPU-as-a-Service** — Dynamic GPU allocation with time-sharing
3. **Serverless AI** — Pay-per-inference, no infrastructure management
4. **Multi-Cloud AI** — Avoiding lock-in with portable platforms
5. **AI-Native Observability** — Purpose-built monitoring for AI systems

---

## 9. Getting Started Guide

### Step 1: Assess Current State

```markdown
## AI Maturity Assessment

□ Do you have a model registry?
□ Are training pipelines automated?
□ Do you track experiments?
□ Is there a feature store?
□ Is model serving containerized?
□ Do you monitor model performance in production?
□ Is there cost attribution for AI workloads?
□ Can developers self-service deploy models?
□ Is there automated testing for models?
□ Are there rollback procedures?
```

### Step 2: Pick Your Platform Path

**Path A: Build (100+ engineers)**
- Use open-source tools (Kubeflow, MLflow, Ray)
- Custom internal developer platform
- Full control, full responsibility

**Path B: Buy (10-100 engineers)**
- Cloud platform (SageMaker, Vertex AI, Azure ML)
- Managed services (Modal, Together AI)
- Faster start, vendor lock-in risk

**Path C: Hybrid (most common)**
- Cloud for training, self-hosted for serving
- Open-source for core, commercial for ancillary
- Balance of control and speed

### Step 3: Build incrementally

**Month 1-2: Foundation**
- Set up model registry (MLflow)
- Basic training pipeline
- Experiment tracking

**Month 3-4: Serving**
- Containerized model serving
- Basic monitoring
- Health checks

**Month 5-6: Platform**
- Self-service deployment
- Cost attribution
- Automated testing

**Month 7-12: Optimize**
- Advanced monitoring
- Auto-scaling
- Full governance

---

## 10. Common Anti-Patterns

### Anti-Pattern: The Notebook Graveyard
**Symptom:** Jupyter notebooks everywhere, nothing reproducible
**Solution:** Enforce code-based ML development, use experiment tracking

### Anti-Pattern: The "Works on My Machine" Model
**Symptom:** Models that work in training but fail in production
**Solution:** Environment parity, containerization, automated testing

### Anti-Pattern: The Monitoring Black Hole
**Symptom:** No visibility into production model performance
**Solution:** Structured monitoring with SLOs, alerting, dashboards

### Anti-Pattern: The Cost Surprise
**Symptom:** $100K GPU bill with no idea who spent what
**Solution:** Cost attribution, budget alerts, resource quotas

### Anti-Pattern: The Manual Deploy
**Symptom:** Deploying models requires a human clicking buttons
**Solution:** Automated CI/CD, self-service deployment

### Anti-Pattern: The Single Point of Failure
**Symptom:** One person knows how everything works
**Solution:** Documentation, runbooks, cross-training

### Anti-Pattern: The Vendor Lock-In Trap
**Symptom:** All AI tooling is from one vendor, switching is impossible
**Solution:** Standards-based approaches, abstraction layers, multi-cloud

### Anti-Pattern: The Security Afterthought
**Symptom:** Models deployed without security review
**Solution:** Security in CI/CD, model scanning, access controls

---

## Cross-References

- **[05-Enterprise/04-AI-Infrastructure.md](../05-Enterprise/04-AI-Infrastructure.md)** — Hardware and infrastructure details
- **[33-AI-Native-Software-Development/03-AI-Native-CI-CD-and-DevOps.md](../33-AI-Native-Software-Development/03-AI-Native-CI-CD-and-DevOps.md)** — CI/CD patterns for AI
- **[20-Agent-Infrastructure-and-Observability/](../20-Agent-Infrastructure-and-Observability/)** — Agent-specific infrastructure
- **[41-AI-Cost-Optimization/](../41-AI-Cost-Optimization-and-Enterprise-ROI/)** — Cost management strategies
- **[03-Agents/03-Agentic-Frameworks.md](../03-Agents/03-Agentic-Frameworks.md)** — Frameworks that need MLOps support
- **[53-AI-Model-Cascading/](../53-AI-Model-Cascading-and-Multi-Model-Orchestration/)** — Multi-model deployment patterns

---

*Next: [02-Core-Topics.md](02-Core-Topics.md) — Deep dive into MLOps core topics*

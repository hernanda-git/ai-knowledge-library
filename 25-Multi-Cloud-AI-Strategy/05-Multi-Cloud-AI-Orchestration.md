# 05 — Multi-Cloud AI Orchestration

> **Orchestrating AI across clouds: Kubeflow for multi-cloud ML pipelines, MLflow for experiment tracking, cross-cloud model deployment, data gravity considerations, network latency between clouds, unified IAM. Architecture patterns: primary cloud + failover, best-of-breed per model type, data residency routing. Includes YAML for multi-cloud k8s AI deployment.**

---

## Table of Contents

1. [Introduction to Multi-Cloud AI Orchestration](#1-introduction-to-multi-cloud-ai-orchestration)
2. [Platform Comparison: Kubeflow vs. MLflow vs. Native Tools](#2-platform-comparison-kubeflow-vs-mlflow-vs-native-tools)
3. [Kubeflow for Multi-Cloud ML Pipelines](#3-kubeflow-for-multi-cloud-ml-pipelines)
   - [3.1 Kubeflow Architecture](#31-kubeflow-architecture)
   - [3.2 Multi-Cloud Kubeflow Deployment](#32-multi-cloud-kubeflow-deployment)
   - [3.3 Pipeline Portability](#33-pipeline-portability)
   - [3.4 Multi-Cloud Pipeline Example](#34-multi-cloud-pipeline-example)
4. [MLflow for Experiment Tracking & Model Registry](#4-mlflow-for-experiment-tracking--model-registry)
   - [4.1 MLflow Architecture](#41-mlflow-architecture)
   - [4.2 Cross-Cloud Tracking Server](#42-cross-cloud-tracking-server)
   - [4.3 Model Registry Across Clouds](#43-model-registry-across-clouds)
   - [4.4 MLflow Deployment Examples](#44-mlflow-deployment-examples)
5. [Cross-Cloud Model Deployment](#5-cross-cloud-model-deployment)
   - [5.1 Orchestration Layer Patterns](#51-orchestration-layer-patterns)
   - [5.2 API Gateway for Multi-Cloud Inference](#52-api-gateway-for-multi-cloud-inference)
   - [5.3 Unified Observability](#53-unified-observability)
   - [5.4 Terraform & Infrastructure as Code](#54-terraform--infrastructure-as-code)
6. [Data Gravity & Network Considerations](#6-data-gravity--network-considerations)
   - [6.1 Data Gravity Principle](#61-data-gravity-principle)
   - [6.2 Cross-Cloud Data Movement](#62-cross-cloud-data-movement)
   - [6.3 Network Latency Between Clouds](#63-network-latency-between-clouds)
   - [6.4 Data Transfer Cost Analysis](#64-data-transfer-cost-analysis)
7. [Unified Identity & Access Management](#7-unified-identity--access-management)
8. [Architecture Patterns](#8-architecture-patterns)
   - [8.1 Primary + Failover Pattern](#81-primary--failover-pattern)
   - [8.2 Best-of-Breed per Model Type](#82-best-of-breed-per-model-type)
   - [8.3 Data Residency Routing Pattern](#83-data-residency-routing-pattern)
   - [8.4 Global Active-Active Load Balancing](#84-global-active-active-load-balancing)
9. [Multi-Cloud Kubernetes AI Deployment YAML](#9-multi-cloud-kubernetes-ai-deployment-yaml)
   - [9.1 Multi-Cloud K8s Cluster Setup](#91-multi-cloud-k8s-cluster-setup)
   - [9.2 Model Inference Deployment on EKS](#92-model-inference-deployment-on-eks)
   - [9.3 Model Inference Deployment on AKS](#93-model-inference-deployment-on-aks)
   - [9.4 Model Inference Deployment on GKE](#94-model-inference-deployment-on-gke)
   - [9.5 Cross-Cloud Service Mesh Configuration](#95-cross-cloud-service-mesh-configuration)
10. [Observability & Logging Across Clouds](#10-observability--logging-across-clouds)
11. [CI/CD for Multi-Cloud AI](#11-cicd-for-multi-cloud-ai)
12. [Challenges & Mitigations](#12-challenges--mitigations)
13. [References](#13-references)

---

## 1. Introduction to Multi-Cloud AI Orchestration

Multi-cloud AI orchestration is the practice of managing AI/ML workloads across two or more cloud providers using a unified operational layer. The goal is to abstract the complexity of individual cloud environments while enabling portability, resilience, and cost optimization.

**Why orchestration matters:**

Without an orchestration layer, each cloud AI deployment is isolated, requiring separate:
- CI/CD pipelines
- Monitoring dashboards
- IAM configurations
- Cost tracking systems
- Incident response procedures

This fragmentation increases operational overhead and defeats the purpose of multi-cloud — instead of flexibility, you get multiplied complexity.

**The orchestration stack typically includes:**

1. **ML Pipeline Orchestration:** Kubeflow, Flyte, or Airflow for portable ML pipelines.
2. **Experiment Tracking:** MLflow or Weights & Biases for cross-platform tracking.
3. **Model Registry:** Centralized model catalog with multi-cloud deployment support.
4. **API Gateway:** Unified inference endpoint that routes to optimal providers (LiteLLM, Portkey).
5. **Service Mesh:** Istio, Linkerd, or Consul for cross-cloud communication.
6. **Infrastructure as Code:** Terraform, Pulumi, or Crossplane for multi-cloud provisioning.
7. **Observability:** Prometheus + Grafana, Datadog, or OpenTelemetry for unified metrics.

---

## 2. Platform Comparison: Kubeflow vs. MLflow vs. Native Tools

| Feature | Kubeflow | MLflow | Cloud Native (SageMaker/Azure ML/Vertex AI) |
|---|---|---|---|
| **Pipeline orchestration** | ✓ (DAG-based KFP) | ✗ (experiments only) | ✓ (native pipelines) |
| **Experiment tracking** | ✓ (via MLMD) | ✓ (core feature) | ✓ (native) |
| **Model registry** | ✓ (via KFServing) | ✓ (core feature) | ✓ (native) |
| **Multi-cloud portability** | ✓ (runs on any K8s) | ✓ (tracking server anywhere) | ✗ (locked to cloud) |
| **Kubernetes native** | ✓ | ✗ (runs on VMs/K8s) | ✓ (via EKS/AKS/GKE) |
| **Learning curve** | Steep | Moderate | Low (within cloud) |
| **Community** | Large (CNCF) | Very large | Vendor-specific |
| **Open source** | ✓ | ✓ | Partial |

**Recommendation:** Use Kubeflow for pipeline orchestration + MLflow for experiment tracking across all three clouds. Use native tools (SageMaker Pipelines, Azure ML, Vertex AI Pipelines) for cloud-specific workloads where portability is not needed.

---

## 3. Kubeflow for Multi-Cloud ML Pipelines

### 3.1 Kubeflow Architecture

Kubeflow is a CNCF-incubated project that makes ML workflows on Kubernetes portable, scalable, and reproducible. Its architecture:

```
┌──────────────────────────────────────────────────┐
│                   Kubeflow Dashboard              │
├──────────────────────────────────────────────────┤
│  Notebooks │ Pipelines │ Katib │ Kserve │ Registry│
├──────────────────────────────────────────────────┤
│               Kubernetes Cluster                  │
│     (EKS / AKS / GKE / On-Prem / Multi-Cloud)    │
├──────────────────────────────────────────────────┤
│    Istio (Service Mesh)   │   Knative (Serverless)│
├──────────────────────────────────────────────────┤
│       K8s Storage (PVC, CSI)   │   K8s Network    │
└──────────────────────────────────────────────────┘
```

**Key Components:**
- **Kubeflow Pipelines (KFP):** DAG-based pipeline orchestration with reusable components.
- **Katib:** Hyperparameter tuning with Bayesian, random, and grid search.
- **KServe (formerly KFServing):** Model serving with canary deployments, multi-model serving, and auto-scaling.
- **Model Registry:** Central model catalog with versioning and metadata.
- **TensorBoard & Metadata:** Visualization and experiment tracking.
- **Notebooks:** Jupyter-based development environment.

### 3.2 Multi-Cloud Kubeflow Deployment

Kubeflow can be deployed across multiple cloud Kubernetes services:

**Deployment Options:**

| Cloud | Kubernetes Service | Kubeflow Installation |
|---|---|---|
| AWS | Amazon EKS | `kubeflow-aws` distribution |
| Azure | Azure Kubernetes Service (AKS) | `kubeflow-aks` manifest |
| GCP | Google Kubernetes Engine (GKE) | `kubeflow-gcp` marketplace app |

**Multi-Cluster Kubeflow:**

For true multi-cloud orchestration, deploy Kubeflow on a management cluster that coordinates pipelines across cloud clusters:

```yaml
# Multi-cloud Kubeflow architecture
Management Cluster:
  - Kubeflow Central Dashboard
  - Pipeline Scheduler
  - MLflow Tracking Server
  - Model Registry (global)

Worker Clusters:
  - AWS EKS (us-east-1):
    - GPU nodes (p4d/p5 instances)
    - S3 access for training data
  - Azure AKS (westeurope):
    - GPU nodes (ND-series)
    - Blob Storage access
  - GCP GKE (europe-west4):
    - TPU nodes (TPU v5p)
    - Cloud Storage access
```

### 3.3 Pipeline Portability

To make pipelines portable across clouds:

1. **Use containerized components:** Each pipeline step is a Docker container. Store containers in a multi-region registry (ECR + ACR + GCR or Docker Hub).

2. **Abstract cloud-specific dependencies:**
   - Use environment variables for cloud credentials.
   - Use K8s Secrets for cloud-specific configuration.
   - Use PVCs or cloud-agnostic object storage interfaces.

3. **Use Python SDK for pipeline definition:**
```python
from kfp.v2 import dsl

@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn"]
)
def train_model(
    dataset_path: str,
    model_output: dsl.Output[dsl.Model]
) -> None:
    """Cloud-agnostic training component."""
    # Code is identical across clouds
    # dataset_path is an abstract URI (s3://, gs://, azure://)
    ...

@dsl.pipeline(
    name="multi-cloud-training",
    description="Portable ML pipeline"
)
def ml_pipeline(
    data_uri: str = "s3://my-bucket/training/data.csv",
    learning_rate: float = 0.01
):
    preprocess = preprocess_data(data_uri=data_uri)
    train = train_model(
        dataset_path=preprocess.outputs["processed_data"]
    )
    evaluate = evaluate_model(
        model=train.outputs["model_output"]
    )
```

### 3.4 Multi-Cloud Pipeline Example

**Scenario:** Train a model using AWS for data storage, GCP for TPU training, and Azure for model deployment.

```
Step 1: Data Preparation (AWS)
  - Access data in S3
  - Preprocess with SageMaker Processing / K8s job
  - Store processed data in S3

Step 2: TPU Training (GCP)
  - Copy data from S3 → GCS (or use Transfer Service)
  - Train model on TPU v5p via GKE
  - Log metrics to MLflow
  - Store model artifact in GCS (replicated to S3)

Step 3: Model Evaluation (GCP)
  - Evaluate on Vertex AI
  - Register in MLflow Model Registry (multi-cloud)
  - If quality > threshold → deploy

Step 4: Model Deployment (Azure)
  - Deploy to Azure ML endpoint
  - Also deploy to AWS SageMaker as fallback
  - Configure LiteLLM gateway to route traffic
```

---

## 4. MLflow for Experiment Tracking & Model Registry

### 4.1 MLflow Architecture

MLflow is an open-source platform for the ML lifecycle:

```
┌─────────────────────────────────────────────────┐
│                     MLflow                       │
├────────────┬──────────┬───────────┬──────────────┤
│ Tracking   │ Models   │ Model     │ Projects     │
│ Server     │ Registry │ Serving   │ (Deployment) │
├────────────┴──────────┴───────────┴──────────────┤
│              Backend Store (RDBMS)                │
│              Artifact Store (S3/GCS/Azure Blob)   │
└──────────────────────────────────────────────────┘
```

**Components:**
- **MLflow Tracking:** Log parameters, metrics, artifacts across experiments.
- **MLflow Models:** Standard format for packaging ML models.
- **MLflow Model Registry:** Central model catalog with versioning and stages.
- **MLflow Projects:** Reproducible ML code packaging.

### 4.2 Cross-Cloud Tracking Server

Deploy MLflow Tracking Server accessible from all clouds:

**Option 1: MLflow on K8s with Multi-Cloud Access**
```yaml
# MLflow deployment on EKS with cross-cloud access
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
      - name: mlflow
        image: ghcr.io/mlflow/mlflow:v2.15.0
        args: ["mlflow", "server",
          "--host", "0.0.0.0",
          "--port", "5000",
          "--backend-store-uri", "postgresql://mlflow:password@rds.aws:5432/mlflow",
          "--default-artifact-root", "s3://mlflow-artifacts/"]
        env:
        - name: MLFLOW_S3_ENDPOINT_URL
          value: "https://s3.amazonaws.com"
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: access-key
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: secret-key
```

**Option 2: Managed MLflow (Databricks):**
Databricks offers a managed MLflow experience across AWS, Azure, and GCP, providing a unified tracking server.

### 4.3 Model Registry Across Clouds

Use MLflow Model Registry as a single source of truth:

```
Model Registry (MLflow)
    │
    ├── Model: sentiment-classifier
    │   ├── v1 (Staging)     →   Deployed on AWS SageMaker
    │   ├── v2 (Production)  →   Deployed on Azure ML
    │   └── v3 (Production)  →   Deployed on AWS + Azure (A/B test)
    │
    ├── Model: llm-gpt4o-wrapper
    │   ├── v1 (Production)  →   Azure OpenAI Gateway
    │   └── v2 (Archived)    →   Previously on Vertex AI
    │
    └── Model: embedding-model
        ├── v3 (Production)  →   Deployed on GCP Vertex AI
        └── v4 (Staging)     →   Evaluating on AWS Bedrock
```

**Model Registration Flow:**

```python
import mlflow

with mlflow.start_run():
    # Train model
    model = train(X_train, y_train)
    
    # Log parameters and metrics
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.94)
    
    # Log model
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="sentiment-classifier"
    )

# Transition model stage
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="sentiment-classifier",
    version=2,
    stage="Production"
)
```

### 4.4 MLflow Deployment Examples

**Deploy MLflow model to SageMaker:**
```python
import mlflow.sagemaker

mlflow.sagemaker.deploy(
    app_name="sentiment-classifier",
    model_uri="models:/sentiment-classifier/Production",
    execution_role_arn="arn:aws:iam::123456:role/sagemaker-execution",
    bucket_name="mlflow-models",
    instance_type="ml.m5.large",
    region_name="us-east-1"
)
```

**Deploy MLflow model to Azure ML:**
```python
import mlflow.azureml

azure_workspace = mlflow.azureml.get_or_create_workspace(
    subscription_id="...",
    resource_group="ml-rg",
    workspace_name="mlflow-workspace"
)

mlflow.azureml.deploy(
    model_uri="models:/sentiment-classifier/Production",
    workspace=azure_workspace,
    deployment_name="sentiment-classifier-aks",
    deployment_type="aks",
    instance_type="Standard_NC6s_v3"
)
```

**Deploy MLflow model to Vertex AI:**
```python
from mlflow.gcp import deploy

mlflow.gcp.deploy(
    model_uri="models:/sentiment-classifier/Production",
    project_id="my-gcp-project",
    region="us-central1",
    machine_type="n1-standard-4",
    endpoint_name="sentiment-classifier"
)
```

---

## 5. Cross-Cloud Model Deployment

### 5.1 Orchestration Layer Patterns

**Pattern 1: Centralized Orchestration**
```
Control Plane (single region)
    │
    ├── Orchestrator (Step Functions / Logic Apps / Cloud Workflows)
    │
    ├── AWS Workers (SageMaker, Bedrock)
    ├── Azure Workers (ML, OpenAI)
    └── GCP Workers (Vertex AI)
```

**Pattern 2: Distributed Orchestration**
```
AWS Region 1    Azure Region 2    GCP Region 3
    │                │                │
Orchestrator     Orchestrator     Orchestrator
    │                │                │
 AWS AI svcs     Azure AI svcs    GCP AI svcs
    │                │                │
    └────────────────┼────────────────┘
                     │
              Global State Store
              (Cosmos DB / Spanner / DynamoDB Global)
```

**Pattern 3: Event-Driven Orchestration**
```
Event Source → Event Bus (EventBridge / Event Grid / Pub-Sub)
                    │
            ┌───────┴───────┐
            │               │
       AWS Pipeline    Azure Pipeline
            │               │
       Bedrock Call    OpenAI Call
            │               │
            └───────┬───────┘
                    │
            Unified Results Store
```

### 5.2 API Gateway for Multi-Cloud Inference

A unified API gateway abstracts provider-specific endpoints:

**LiteLLM Configuration Example:**
```yaml
# config.yaml for LiteLLM multi-cloud gateway
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: azure/gpt-4o
      api_base: https://my-openai.openai.azure.com/
      api_key: os.environ/AZURE_OPENAI_KEY
    model_info:
      mode: completion
      supports_function_calling: true

  - model_name: claude-4-opus
    litellm_params:
      model: bedrock/anthropic.claude-4-opus
      aws_access_key_id: os.environ/AWS_ACCESS_KEY
      aws_secret_access_key: os.environ/AWS_SECRET_KEY
      aws_region_name: us-east-1
    model_info:
      mode: completion

  - model_name: gemini-1.5-pro
    litellm_params:
      model: vertex_ai/gemini-1.5-pro
      vertex_project: my-gcp-project
      vertex_location: us-central1
```

**Routing Rules (Portkey Example):**
```json
{
  "strategy": {
    "mode": "fallback",
    "primary": "azure/gpt-4o",
    "fallbacks": [
      "bedrock/claude-4-sonnet",
      "vertex_ai/gemini-1.5-pro"
    ],
    "conditions": {
      "on_status_code": [429, 500, 503],
      "on_latency_ms": 5000
    }
  },
  "targets": [
    {
      "provider": "azure/gpt-4o",
      "weight": 70
    },
    {
      "provider": "bedrock/claude-4-sonnet",
      "weight": 20
    },
    {
      "provider": "vertex_ai/gemini-1.5-pro",
      "weight": 10
    }
  ]
}
```

### 5.3 Unified Observability

**OpenTelemetry for Multi-Cloud AI Monitoring:**

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.exporter import OTLPSpanExporter

# Configure OpenTelemetry to send traces to a central collector
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("llm_inference") as span:
    span.set_attribute("provider", "azure")
    span.set_attribute("model", "gpt-4o")
    span.set_attribute("input_tokens", 150)
    span.set_attribute("output_tokens", 45)
    span.set_attribute("latency_ms", 1200)
    span.set_attribute("cost", 0.00375)
    
    # Make the API call
    response = client.chat.completions.create(...)
```

**Key Metrics to Track:**
- **Per-provider:** Request count, latency, error rate, token throughput.
- **Per-model:** Quality scores, cost per request, cache hit rate.
- **Per-workload:** End-user satisfaction, task completion rate.
- **Financial:** Daily/weekly/monthly spend per provider, per team, per project.

**Recommended Stack:**
- **Metrics:** Prometheus + Grafana (multi-cloud Prometheus federation).
- **Logs:** Cloud-agnostic log aggregation (ELK, Datadog, Grafana Loki).
- **Traces:** OpenTelemetry + Jaeger or Datadog APM.
- **Cost:** CloudHealth, Vantage, or proprietary cost analytics.

### 5.4 Terraform & Infrastructure as Code

Terraform is the standard for multi-cloud AI infrastructure provisioning:

**Multi-Cloud Terraform Structure:**
```
terraform/
├── environments/
│   ├── dev/
│   │   ├── aws/
│   │   ├── azure/
│   │   └── gcp/
│   ├── staging/
│   └── prod/
├── modules/
│   ├── ai-gateway/        # LiteLLM/Portkey deployment
│   ├── bedrock/           # AWS Bedrock configuration
│   ├── azure-openai/      # Azure OpenAI deployment
│   ├── vertex-ai/         # Vertex AI endpoint
│   ├── mlflow/            # MLflow tracking server
│   └── monitoring/        # Prometheus/Grafana
├── providers.tf
├── main.tf
└── outputs.tf
```

**Example: Terraform for Multi-Cloud AI Gateway:**
```hcl
# Deploy LiteLLM across multiple clouds
resource "kubernetes_deployment" "litellm" {
  provider = kubernetes.aws
  metadata {
    name      = "litellm-gateway"
    namespace = "ai-gateway"
  }
  spec {
    replicas = 3
    selector { match_labels = { app = "litellm" } }
    template {
      metadata { labels = { app = "litellm" } }
      spec {
        container {
          image = "ghcr.io/berriai/litellm:main-latest"
          name  = "litellm"
          env {
            name  = "DATABASE_URL"
            value = "postgresql://..."  
          }
        }
      }
    }
  }
}

# Also deploy to Azure and GCP using different provider aliases
resource "kubernetes_deployment" "litellm_azure" {
  provider = kubernetes.azure
  # ... similar configuration
}

resource "kubernetes_deployment" "litellm_gcp" {
  provider = kubernetes.gcp
  # ... similar configuration
}
```

---

## 6. Data Gravity & Network Considerations

### 6.1 Data Gravity Principle

Data gravity states that data attracts applications, services, and workflows. The larger the data mass, the stronger the gravitational pull.

**Implications for Multi-Cloud AI:**
- **Training data determines training location:** Don't move petabytes of training data — train where the data lives.
- **Cross-cloud inference adds latency:** If your application is on AWS and your model on GCP, every inference incurs cross-cloud latency.
- **Vector databases create gravity:** Embedding stores should be in the same cloud as the application serving RAG queries.

### 6.2 Cross-Cloud Data Movement Strategies

**Strategy 1: Object Storage Replication**
```
AWS S3 (us-east-1) 
    │
    ├── S3 Cross-Region Replication → S3 (eu-west-1)
    │
    ├── S3 Batch Replication → GCS (us-central1)
    │
    └── S3 → Azure Blob Storage (via AzCopy or Storage Transfer Service)
```

**Strategy 2: Cloud-Agnostic Storage**
- Use MinIO or other S3-compatible storage deployed on K8s across clouds.
- Provides a single interface for all cloud object storage.

**Strategy 3: Data Mesh Architecture**
- Each cloud owns its data domain.
- Data products are published to a shared catalog.
- Consumers access data in situ (no central data lake).

### 6.3 Network Latency Between Clouds

Typical cross-cloud latency (P99):

| Source → Destination | Latency (ms) |
|---|---|
| AWS us-east-1 → Azure eastus | 8–12 ms |
| AWS us-east-1 → GCP us-east4 | 5–8 ms |
| AWS eu-west-1 → Azure westeurope | 5–10 ms |
| AWS ap-southeast-1 → Azure southeastasia | 8–15 ms |
| AWS us-west-2 → GCP us-west1 | 8–12 ms |
| Azure eastus → GCP us-central1 | 10–18 ms |

**Mitigations:**
- Co-locate cloud regions in the same metro area for lowest latency.
- Use cloud interconnect / direct peering for dedicated bandwidth.
- Deploy inference endpoints in the same cloud region as consuming applications.
- Cache inference results to avoid repeated cross-cloud calls.

### 6.4 Data Transfer Cost Analysis

| Transfer Direction | Cost (per GB) |
|---|---|
| AWS S3 → Internet | $0.09 (first 10TB) |
| Azure Blob → Internet | $0.087 (first 10TB) |
| GCP Cloud Storage → Internet | $0.12 (first 10TB) |
| AWS → Azure (Direct) | $0.02–$0.05 (via interconnect) |
| AWS → GCP (Direct) | $0.02–$0.05 (via interconnect) |
| AWS → Azure (Internet) | $0.09 + $0.087 = $0.177 |

**Key Insight:** Cross-cloud data transfer over the public internet costs ~2–3x more than using cloud interconnects. For data-intensive AI workloads, always use direct peering or dedicated interconnects.

---

## 7. Unified Identity & Access Management

**Challenge:** Each cloud has its own IAM system. Managing identities across AWS IAM, Azure RBAC/Entra ID, and GCP IAM adds complexity.

**Solution: Federated Identity with External Identity Provider:**

```
External IdP (Okta / Azure AD / Google Workspace)
    │
    ├── AWS IAM Identity Center (SSO)
    │   ├── AWS IAM Roles → Bedrock, SageMaker
    │   └── Permissions via SCPs
    │
    ├── Azure Entra ID
    │   ├── Azure RBAC → OpenAI, Cognitive Services
    │   └── Conditional Access Policies
    │
    └── GCP Workforce Identity Federation
        ├── GCP IAM Roles → Vertex AI
        └── Organization Policies
```

**Best Practices:**
1. Use an external identity provider (Okta, Azure AD, Google Workspace) as the single source of truth.
2. Configure federation in each cloud to trust the external IdP.
3. Assign users to groups in the IdP; map groups to cloud roles.
4. Use short-lived credentials (AWS STS, Azure Managed Identity, GCP Service Account impersonation).
5. Implement least-privilege — users get only the AI permissions they need.

---

## 8. Architecture Patterns

### 8.1 Primary + Failover Pattern

```
Normal Operation:              Failover:
User → AWS Bedrock (Claude)    User → Azure OpenAI (GPT-4o)
       ↑ primary                      ↑ fallback
       ↑ healthy                      ↑ primary failed
```

**Components:**
- Primary cloud handles all traffic.
- Secondary cloud is warm (models deployed, endpoints ready, 0 traffic).
- Health checks probe primary at sub-minute frequency.
- On primary failure, DNS/Traffic Manager routes to secondary.
- Models should be functionally equivalent (e.g., Claude 4 ↔ GPT-4o).

**Terraform Implementation:**
```hcl
# Health check + failover routing
resource "aws_route53_health_check" "primary" {
  fqdn              = "primary-api.example.com"
  port              = 443
  type              = "HTTPS"
  resource_path     = "/health"
  failure_threshold = 3
  request_interval  = 30
}

resource "aws_route53_record" "ai_endpoint" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "ai-api.example.com"
  type    = "A"
  
  failover_routing_policy {
    type = "PRIMARY"
  }
  
  set_identifier = "primary"
  alias {
    name    = aws_lb.primary.dns_name
    zone_id = aws_lb.primary.zone_id
  }
}
```

### 8.2 Best-of-Breed per Model Type

```
User → LiteLLM Gateway
         │
    ┌────┴────────────────────────────┐
    │          │                      │
Azure OpenAI    AWS Bedrock         GCP Vertex
(GPT-4o)        (Claude 4)          (Gemini 1.5 Pro)
│               │                    │
Chat & Code     Document AI         Video & Image
Customer        Contract Review     Visual Question
Support         Legal Analysis      Answering
```

**Routing Logic:**
- Chat/completions → Azure OpenAI (GPT-4o, best conversational quality).
- Document analysis → AWS Bedrock (Claude 4, best document understanding).
- Vision/multimodal → GCP Vertex (Gemini, best native multimodal).
- Cost-sensitive classification → Any cloud (Llama 4, cheapest).
- Embeddings → Azure OpenAI (best price/quality for text embeddings).

### 8.3 Data Residency Routing Pattern

```
Request → Geo-IP Service → LiteLLM Gateway
    │                           │
    ├── EU User                 ├── Azure OpenAI (France Central)
    ├── US User                 ├── AWS Bedrock (us-east-1)
    ├── APAC User               ├── GCP Vertex (asia-southeast1)
    └── Unknown                 └── Azure OpenAI (US) [default]
```

**Implementation:**
```python
def route_request(user_request):
    region = resolve_user_region(user_request)
    
    router = {
        "EU": {
            "provider": "azure",
            "endpoint": "https://francecentral.openai.azure.com",
            "model": "gpt-4o",
            "guardrails": "eu-ai-act-compliant"
        },
        "US": {
            "provider": "aws",
            "endpoint": "bedrock-us-east-1",
            "model": "claude-4-sonnet",
            "guardrails": "hipaa-compliant"
        },
        "APAC": {
            "provider": "gcp",
            "endpoint": "vertex-asia-southeast1",
            "model": "gemini-1.5-pro"
        }
    }
    
    return router.get(region, router["US"])
```

### 8.4 Global Active-Active Load Balancing

```
Global Load Balancer (Cloudflare / Azure Front Door / GCP GLB)
    │
    ├── 50% traffic → AWS (us-east-1) → Bedrock
    ├── 30% traffic → Azure (eastus)  → OpenAI
    └── 20% traffic → GCP (us-central1) → Vertex AI
```

**Requirements:**
- Stateless applications (session state in distributed cache).
- Consistent model quality across providers.
- Unified observability for cross-cloud performance comparison.
- Cost-weighted routing (more load to cheaper provider).

---

## 9. Multi-Cloud Kubernetes AI Deployment YAML

### 9.1 Multi-Cloud K8s Cluster Setup

**Cross-Cluster Communication via Service Mesh (Istio):**

```yaml
# Istio ServiceMesh for multi-cloud AI
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  name: multi-cloud-mesh
spec:
  meshConfig:
    accessLogFile: /dev/stdout
    enableTracing: true
    defaultConfig:
      proxyMetadata:
        AWS_REGION: us-east-1
  components:
    pilot:
      k8s:
        env:
          - name: PILOT_ENABLE_CROSS_CLUSTER
            value: "true"
  values:
    global:
      meshID: multi-cloud-ai
      multiCluster:
        clusterName: eks-us-east-1
      network: aws-us-east-1
---
# Primary cluster: EKS (us-east-1)
# Secondary clusters: AKS (eastus), GKE (us-central1)
# Connected via Istio multi-cluster mesh
```

### 9.2 Model Inference Deployment on EKS

```yaml
# Deploy model inference on AWS EKS
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama4-inference
  namespace: ai-inference
  labels:
    app: llama4
    cloud: aws
    model-type: text-generation
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llama4
  template:
    metadata:
      labels:
        app: llama4
        cloud: aws
    spec:
      nodeSelector:
        node-type: gpu
        cloud.google.com/gke-accelerator: nvidia-h100
      containers:
      - name: model-server
        image: vllm/vllm-openai:latest
        args:
          - "--model"
          - "meta-llama/Llama-4-8B"
          - "--tensor-parallel-size"
          - "1"
          - "--max-model-len"
          - "8192"
          - "--gpu-memory-utilization"
          - "0.9"
        ports:
        - containerPort: 8000
          name: http
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 64Gi
            cpu: 8
          requests:
            nvidia.com/gpu: 1
            memory: 48Gi
            cpu: 4
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token
              key: token
---
apiVersion: v1
kind: Service
metadata:
  name: llama4-inference
  namespace: ai-inference
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "external"
    service.beta.kubernetes.io/aws-load-balancer-nlb-target-type: "ip"
    service.beta.kubernetes.io/aws-load-balancer-scheme: "internal"
spec:
  selector:
    app: llama4
  type: LoadBalancer
  ports:
  - port: 8000
    targetPort: 8000
    name: http
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llama4-hpa
  namespace: ai-inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama4-inference
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: request_duration_ms
      target:
        type: AverageValue
        averageValue: 500
```

### 9.3 Model Inference Deployment on AKS

```yaml
# Deploy model inference on Azure AKS
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpt4o-proxy
  namespace: ai-inference
  labels:
    app: gpt4o-proxy
    cloud: azure
    model-type: chat
spec:
  replicas: 5
  selector:
    matchLabels:
      app: gpt4o-proxy
  template:
    metadata:
      labels:
        app: gpt4o-proxy
        cloud: azure
    spec:
      nodeSelector:
        agentpool: gpu
      containers:
      - name: openai-proxy
        image: myregistry.azurecr.io/openai-proxy:latest
        ports:
        - containerPort: 8080
        env:
        - name: AZURE_OPENAI_ENDPOINT
          value: "https://my-openai.openai.azure.com"
        - name: AZURE_OPENAI_DEPLOYMENT
          value: "gpt-4o"
        - name: AZURE_OPENAI_API_VERSION
          value: "2025-12-01"
        - name: USE_MANAGED_IDENTITY
          value: "true"
        resources:
          limits:
            memory: 4Gi
            cpu: 2
          requests:
            memory: 2Gi
            cpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: gpt4o-proxy
  namespace: ai-inference
  annotations:
    service.beta.kubernetes.io/azure-load-balancer-internal: "true"
spec:
  selector:
    app: gpt4o-proxy
  type: LoadBalancer
  ports:
  - port: 8080
    targetPort: 8080
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gpt4o-proxy-hpa
  namespace: ai-inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gpt4o-proxy
  minReplicas: 3
  maxReplicas: 30
  metrics:
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: 100
```

### 9.4 Model Inference Deployment on GKE

```yaml
# Deploy model inference on GCP GKE
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gemini-proxy
  namespace: ai-inference
  labels:
    app: gemini-proxy
    cloud: gcp
    model-type: multimodal
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gemini-proxy
  template:
    metadata:
      labels:
        app: gemini-proxy
        cloud: gcp
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-h100
      serviceAccountName: gemini-sa
      containers:
      - name: gemini-proxy
        image: us-central1-docker.pkg.dev/my-project/gemini-proxy:latest
        ports:
        - containerPort: 8080
        env:
        - name: GOOGLE_PROJECT_ID
          value: "my-gcp-project"
        - name: GOOGLE_LOCATION
          value: "us-central1"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
            cpu: 8
          requests:
            nvidia.com/gpu: 1
            memory: 16Gi
            cpu: 4
---
apiVersion: v1
kind: Service
metadata:
  name: gemini-proxy
  namespace: ai-inference
  annotations:
    cloud.google.com/load-balancer-type: "Internal"
spec:
  selector:
    app: gemini-proxy
  type: LoadBalancer
  ports:
  - port: 8080
    targetPort: 8080
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gemini-proxy-hpa
  namespace: ai-inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gemini-proxy
  minReplicas: 2
  maxReplicas: 15
  metrics:
  - type: Pods
    pods:
      metric:
        name: aiengine_gemini_request_latency_seconds
      target:
        type: AverageValue
        averageValue: 2.0
```

### 9.5 Cross-Cloud Service Mesh Configuration

```yaml
# Istio VirtualService for multi-cloud routing
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ai-inference-routing
  namespace: ai-inference
spec:
  hosts:
  - ai-gateway.example.com
  gateways:
  - ai-gateway
  http:
  - match:
    - headers:
        x-region:
          exact: EU
    route:
    - destination:
        host: gpt4o-proxy.ai-inference.svc.cluster.local
        port:
          number: 8080
      weight: 100
  - match:
    - headers:
        x-model-tier:
          exact: cheap
    route:
    - destination:
        host: llama4-inference.ai-inference.svc.cluster.local
        port:
          number: 8000
      weight: 100
  - route:
    - destination:
        host: gpt4o-proxy.ai-inference.svc.cluster.local
        port:
          number: 8080
      weight: 70
    - destination:
        host: gemini-proxy.ai-inference.svc.cluster.local
        port:
          number: 8080
      weight: 30
```

---

## 10. Observability & Logging Across Clouds

**Centralized Logging Architecture:**

```
AWS CloudWatch     Azure Monitor     GCP Cloud Logging
    │                    │                  │
    └────────────────────┼──────────────────┘
                         │
              Central Log Aggregator
              (Grafana Loki / Elasticsearch / Datadog)
                         │
                  Grafana Dashboard
```

**Prometheus Multi-Cloud Federation:**

```yaml
# Prometheus federation configuration
scrape_configs:
  - job_name: 'federate-aws'
    scrape_interval: 30s
    honor_labels: true
    metrics_path: '/federate'
    params:
      'match[]':
        - '{job=~"ai-inference.*"}'
        - '{job=~"litellm.*"}'
    static_configs:
      - targets: ['aws-prometheus.example.com:9090']

  - job_name: 'federate-azure'
    scrape_interval: 30s
    honor_labels: true
    metrics_path: '/federate'
    params:
      'match[]': ['{job=~".*ai.*"}']
    static_configs:
      - targets: ['azure-prometheus.example.com:9090']

  - job_name: 'federate-gcp'
    scrape_interval: 30s
    honor_labels: true
    metrics_path: '/federate'
    params:
      'match[]': ['{job=~".*vertex.*"}']
    static_configs:
      - targets: ['gcp-prometheus.example.com:9090']
```

---

## 11. CI/CD for Multi-Cloud AI

**GitOps Pipeline:**

```
Developer Push → GitHub/GitLab
    │
    ├── CI: GitHub Actions / GitLab CI
    │   ├── Lint code
    │   ├── Run unit tests
    │   ├── Build Docker image
    │   └── Push to registry (ECR + ACR + GCR)
    │
    └── CD: ArgoCD / Flux
        ├── Deploy to AWS EKS (staging)
        │   └── Integration tests
        ├── Deploy to Azure AKS (staging)
        │   └── Integration tests
        ├── Deploy to GCP GKE (staging)
        │   └── Integration tests
        └── Promote to production (all clouds)
```

**Model-Specific CI/CD Pipeline:**

```yaml
# GitHub Actions workflow for multi-cloud AI deployment
name: Multi-Cloud AI Deploy
on:
  push:
    branches: [main]
    paths:
      - 'models/**'
      - 'pipelines/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Evaluate model quality
        run: python evaluate_model.py --model-uri ${{ env.MODEL_URI }}
      - name: Register in MLflow
        run: python register_model.py
  
  deploy-aws:
    needs: evaluate
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to SageMaker
        run: |
          python deploy_to_sagemaker.py \
            --model-uri ${{ env.MODEL_URI }} \
            --region us-east-1
  
  deploy-azure:
    needs: evaluate
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Azure ML
        run: |
          python deploy_to_azureml.py \
            --model-uri ${{ env.MODEL_URI }} \
            --workspace my-workspace
  
  smoke-test:
    needs: [deploy-aws, deploy-azure]
    runs-on: ubuntu-latest
    steps:
      - name: Run multi-cloud smoke tests
        run: |
          python smoke_test.py \
            --aws-endpoint ${{ secrets.AWS_ENDPOINT }} \
            --azure-endpoint ${{ secrets.AZURE_ENDPOINT }}
```

---

## 12. Challenges & Mitigations

| Challenge | Impact | Mitigation |
|---|---|---|
| **API Inconsistency** | Different API formats, auth, rate limits per cloud | Use LiteLLM/Portkey abstraction layer |
| **Data Transfer Costs** | Moving data between clouds is expensive | Minimize cross-cloud data movement; use interconnects |
| **Latency** | Cross-cloud inference adds 5–20ms | Co-locate services; cache results; use edge caching |
| **IAM Complexity** | Managing identities across 3 clouds is error-prone | Federated identity with external IdP |
| **Configuration Drift** | K8s manifests may differ across clouds unnecessarily | Use Kustomize overlays or Helm for cloud-specific differences |
| **Skill Requirements** | Need expertise in 3 cloud platforms + K8s | Invest in training; use abstraction tools |
| **Compliance** | Each cloud has different compliance boundaries | Use data residency routing; audit regularly |
| **Observability Fragmentation** | Siloed monitoring per cloud | OpenTelemetry + central observability |
| **Cost Tracking** | Hard to track per-workload costs across clouds | Unified tagging strategy; FinOps tools |

---

## 13. References

1. Kubeflow. (2026). *Kubeflow Documentation*. https://www.kubeflow.org/docs/
2. MLflow. (2026). *MLflow Documentation*. https://mlflow.org/docs/
3. Litellm. (2026). *LiteLLM Documentation*. https://docs.litellm.ai/
4. Portkey. (2026). *Portkey AI Gateway Documentation*. https://docs.portkey.ai/
5. Istio. (2026). *Multi-Cluster Deployment Models*. https://istio.io/latest/docs/setup/install/multicluster/
6. Terraform. (2026). *Multi-Cloud Infrastructure*. https://developer.hashicorp.com/terraform/tutorials/multi-cloud
7. ArgoCD. (2026). *Multi-Cluster Deployment*. https://argo-cd.readthedocs.io/en/stable/operator-manual/declarative-setup/
8. OpenTelemetry. (2026). *OpenTelemetry Documentation*. https://opentelemetry.io/docs/

---

> **Next:** [06 — AI Model Procurement and Gateway](06-AI-Model-Procurement-and-Gateway.md)

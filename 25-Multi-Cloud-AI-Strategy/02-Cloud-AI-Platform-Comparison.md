# 02 — Cloud AI Platform Comparison

## Comprehensive Comparison of Major AI Cloud Platforms

This document provides a deep, detailed comparison of the major cloud AI platforms available in 2026. It covers model availability, pricing structures, latency characteristics, regional presence, feature sets, and data privacy considerations for AWS Bedrock/SageMaker, Azure OpenAI Service/AI Studio, Google Cloud Vertex AI/Gemini API, Oracle Cloud AI Services, and IBM watsonx.

---

## Table of Contents

1. Overview of Major Platforms
2. AWS AI Platform — Bedrock & SageMaker
3. Azure AI Platform — OpenAI Service & AI Studio
4. Google Cloud AI Platform — Vertex AI & Gemini API
5. Oracle Cloud AI — OCI AI Services
6. IBM Cloud AI — watsonx
7. Model Availability Comparison Table
8. Pricing Comparison — Per-Token and Per-Hour
9. Latency and Performance Comparison
10. Regional Availability Matrix
11. Feature Comparison — Fine-Tuning, RAG, Agents
12. Data Privacy and Security
13. Specialized Hardware Comparison
14. MLOps and Tooling
15. Migration Difficulty Assessment
16. Decision Framework

---

## 1. Overview of Major Platforms

The AI platform market in 2026 is dominated by three hyperscalers (AWS, Azure, GCP) with two significant challengers (Oracle, IBM). Each offers a fundamentally different approach:

- **AWS** provides the broadest infrastructure-level AI toolkit, with both managed services (Bedrock) and DIY ML (SageMaker).
- **Azure** leverages the exclusive OpenAI relationship and deep enterprise integration to offer the most seamless GPT-4o experience.
- **GCP** focuses on its strengths in data analytics and custom hardware (TPUs) with Vertex AI as a unified platform.
- **Oracle** differentiates on price, GPU availability, and database-native AI.
- **IBM** targets regulated industries with governance-first AI (watsonx).

---

## 2. AWS AI Platform — Bedrock & SageMaker

### Amazon Bedrock

Bedrock is AWS's fully managed service for foundation models. It provides API access to models from:
- **Anthropic** — Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku, Claude 4 (projected)
- **Amazon** — Titan Text (G1 - Premier, Lite, Express), Titan Embeddings (Text V2, Multimodal)
- **Meta** — Llama 3.1 8B, 70B, 405B, Llama 4 (projected)
- **Cohere** — Command R+, Command R, Embed (English, Multilingual)
- **Stability AI** — Stable Diffusion XL, SD3
- **Mistral AI** — Mistral Large 2, Mixtral 8x22B

**Key Features:**
- Model evaluation (automated and human)
- Guardrails for content filtering (Denied Topics, Word Filters, Sensitive Information Filters)
- Knowledge bases (RAG with Aurora, Pinecone, Redis)
- Agents (multi-step orchestration with action groups)
- Custom model imports (import your own fine-tuned models)
- Provisioned throughput (dedicated capacity with per-hour pricing)
- Batch inference (async bulk processing)

**Strengths:** Broadest model selection, robust security (IAM, KMS, VPC), strong integration with AWS data services (S3, Glue, Redshift), mature agent framework.

**Weaknesses:** Complex pricing (many levers), no native OpenAI models, some services require significant configuration.

### Amazon SageMaker

SageMaker is AWS's end-to-end ML platform for training and deploying custom models:

- SageMaker Studio — Integrated development environment
- SageMaker Training — Managed training with distributed libraries
- SageMaker Hyperpod — Multi-node training for foundation models
- SageMaker Inference — Real-time, serverless, and batch inference
- SageMaker JumpStart — Pre-trained models, notebooks, and solutions
- SageMaker Model Registry — Versioning and lineage
- SageMaker Pipelines — MLOps pipelines with DAG
- SageMaker Clarify — Bias detection and explainability
- SageMaker Feature Store — Feature management

**Hardware:** P5 (H100), P5e (B200), Trn1 (Trainium), Trn2 (Trainium2), Inf2 (Inferentia2) instances.

---

## 3. Azure AI Platform — OpenAI Service & AI Studio

### Azure OpenAI Service

The exclusive enterprise provider of OpenAI models on Microsoft infrastructure:

- **GPT-4o** (multimodal, text+vision+audio)
- **GPT-4o mini** (cost-effective, fast)
- **o1** (reasoning model with chain-of-thought)
- **o3** (advanced reasoning, projected)
- **GPT-4 Turbo** (legacy)
- **GPT-3.5 Turbo** (legacy)
- **DALL-E 3** (image generation)
- **Whisper** (speech to text)
- **TTS** (text to speech)
- **Embeddings** (text-embedding-3-large, text-embedding-3-small)

**Key Features:**
- Data residency with SLA (training data not used to improve models)
- Content filtering (Azure AI Content Safety integrated)
- Fine-tuning (GPT-4o, GPT-4o mini, GPT-3.5 Turbo)
- Real-time audio
- Batch processing (50% discount)
- Provisioned throughput units (PTUs)
- Responsible AI tooling

### Azure AI Studio

A unified platform for building, testing, and deploying AI applications:

- Model catalog (OpenAI, Meta, Mistral, Cohere, Llama, Phi-3)
- Prompt flow — Visual designer for AI workflows
- Content safety — Built-in moderation and safety
- AI Search — Vector + hybrid search for RAG
- AI Agents — Multi-agent orchestration (projected)
- Evaluation flows — Model evaluation and testing
- Model monitoring — Drift detection, performance tracking

**Note:** Azure also hosts non-OpenAI models through the Model Catalog and Azure AI Studio, but the primary differentiator remains the OpenAI partnership.

**Strengths:** Best GPT-4 experience (lowest latency among cloud providers for GPT-4o), strong enterprise SLAs, seamless integration with Microsoft 365 and Power Platform, excellent content safety tools.

**Weaknesses:** OpenAI dependency (single point of failure), GPU capacity constraints, less flexible for non-OpenAI workloads.

---

## 4. Google Cloud AI Platform — Vertex AI & Gemini API

### Vertex AI

Google Cloud's unified AI platform:

- **Gemini API** — Gemini 2.0 Flash, Gemini 2.0 Pro, Gemini 1.5 Pro/Flash
- **Model Garden** — 150+ models from Google, Anthropic, Meta, Mistral, Gemma, Llama, Claude
- **Vertex AI Studio** — Prompt design, testing, and iteration
- **Vertex AI Agent Builder** — Agent development platform
- **Vertex AI Search** — Enterprise search with grounding
- **Vertex AI RAG Engine** — Managed RAG with data connectors
- **Vertex AI Vector Search** — Scalable vector database
- **Vertex AI Model Registry** — Model versioning and lineage
- **Vertex AI Pipelines** — Kubeflow-based MLOps
- **Vertex AI Prediction** — Managed inference

**Gemini Models:**
- Gemini 2.0 Flash — Fast, cost-effective, multimodal
- Gemini 2.0 Pro — Most capable, longer context (1M+ tokens)
- Gemini 1.5 Pro — Context window up to 2M tokens
- Gemini 1.5 Flash — Balanced performance and cost

**Key Features:**
- Native multimodal (text, images, video, audio, code)
- Longest context window (2M tokens)
- Strong grounding (Google Search, enterprise data)
- WekaFM integration for domain-specific fine-tuning
- Colab Enterprise integration
- TensorFlow and JAX native support

**Hardware:** TPU v5p, TPU v5e, TPU Trillium, GPU (A100, H100, L4).

**Strengths:** Best native search/RAG (grounding with Google Search), longest context windows, strongest custom hardware (TPUs), best data integration (BigQuery).

**Weaknesses:** Smaller enterprise footprint, Gemini not always competitive on leaderboards, fewer enterprise-grade security certifications compared to AWS/Azure.

---

## 5. Oracle Cloud AI — OCI AI Services

### OCI AI Services

Oracle's approach to AI emphasizes cost competitiveness and database integration:

- **OCI Generative AI** — Cohere Command R+, Llama 3.1, Meta Llama 4
- **OCI AI Services** — Pre-built AI (language, vision, speech, document understanding, anomaly detection)
- **OCI Data Science** — Managed notebooks, jobs, and models
- **OCI Model Deployment** — Real-time and batch inference
- **Oracle Database 23ai** — Integrated AI (AI Vector Search, JSON Relational Duality, Graph)
- **OCI Compute** — Bare metal and VM with H100, A100, Ampere

**Key Differentiators:**
- **Pricing.** OCI is typically 30–50% cheaper than hyperscalers for equivalent GPU capacity.
- **GPU Availability.** OCI has better GPU availability during shortages due to smaller customer base.
- **Database-Native AI.** Oracle 23ai integrates vector search directly into the database.
- **Sovereignty.** OCI offers unique sovereignty regions (EU, US Government).
- **SLAs.** Strong SLAs on compute and GPU availability.

**Strengths:** Lowest cost among major providers, excellent GPU availability, best integration with Oracle Database, strong in regulated industries already on Oracle.

**Weaknesses:** Fewest model choices, smallest AI ecosystem, less mature MLOps tooling, limited global region presence.

---

## 6. IBM Cloud AI — watsonx

### IBM watsonx Platform

IBM's AI platform focused on governance, openness, and hybrid cloud:

- **watsonx.ai** — Studio for training, validating, and deploying models (Granite, open-source models)
- **watsonx.governance** — AI governance, bias detection, explainability, compliance
- **watsonx.data** — Data lakehouse for AI workloads
- **watsonx Assistant** — Enterprise conversational AI
- **watsonx Orchestrate** — AI automation for business processes

**IBM Granite Models:**
- Granite 3.0 — 8B and 34B parameter models
- Granite Guardian — Safety guardrail models
- Granite MoE — Mixture of experts models
- Granite Vision — Multimodal models
- Granite TimeSeries — Time series forecasting

**Key Differentiators:**
- **Open-source first.** Granite models are Apache 2.0 licensed.
- **Governance built-in.** watsonx.governance is the most comprehensive AI governance tool.
- **Hybrid cloud.** Runs on IBM Cloud, AWS, Azure, GCP, and on-premises.
- **Industry focus.** Strong in financial services, healthcare, and government.
- **Open standard support.** Commitment to ONNX, OpenShift, and open ecosystems.

**Strengths:** Best governance capabilities in the market, open-source models reduce lock-in, strong hybrid cloud story, deep industry solutions for finance and healthcare.

**Weaknesses:** Smallest market share among these five, slower model release cadence, fewer pre-built integrations, limited developer community.

---

## 7. Model Availability Comparison Table

| Model Category | AWS Bedrock | Azure OpenAI | GCP Vertex AI | OCI AI | IBM watsonx |
|---|---|---|---|---|---|
| **GPT-4o** | ❌ | ✅ Native | ✅ (via Catalog) | ❌ | ❌ |
| **Claude 3.5 Sonnet** | ✅ Native | ✅ (via Catalog) | ✅ (via Catalog) | ❌ | ❌ |
| **Claude 3 Opus** | ✅ | ✅ (via Catalog) | ✅ (via Catalog) | ❌ | ❌ |
| **Gemini 2.0 Flash** | ❌ | ❌ | ✅ Native | ❌ | ❌ |
| **Llama 3.1 405B** | ✅ | ✅ (via Catalog) | ✅ | ✅ | ✅ |
| **Llama 4** | ✅ | ✅ (via Catalog) | ✅ | ✅ | ✅ |
| **Mistral Large 2** | ✅ | ✅ (via Catalog) | ✅ | ❌ | ❌ |
| **Cohere Command R+** | ✅ | ✅ (via Catalog) | ✅ | ✅ | ❌ |
| **Stable Diffusion SD3** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **IBM Granite 3.0** | ❌ | ❌ | ❌ | ❌ | ✅ Native |
| **Amazon Titan** | ✅ Native | ❌ | ❌ | ❌ | ❌ |
| **Phi-3** | ❌ | ✅ Native | ✅ | ❌ | ❌ |
| **Gemma 2** | ❌ | ❌ | ✅ Native | ❌ | ✅ |
| **DALL-E 3** | ❌ | ✅ Native | ❌ | ❌ | ❌ |

---

## 8. Pricing Comparison — Per-Token and Per-Hour

### Per-Token Pricing (Input / Output per 1M tokens, USD)

| Model | AWS Bedrock | Azure OpenAI | GCP Vertex AI | OCI AI | IBM watsonx |
|---|---|---|---|---|---|
| **GPT-4o** | N/A | $2.50 / $10.00 | Via catalog | N/A | N/A |
| **GPT-4o mini** | N/A | $0.15 / $0.60 | Via catalog | N/A | N/A |
| **Claude 3.5 Sonnet** | $3.00 / $15.00 | Via catalog | $3.00 / $15.00 | N/A | N/A |
| **Claude 3 Opus** | $15.00 / $75.00 | Via catalog | $15.00 / $75.00 | N/A | N/A |
| **Gemini 2.0 Flash** | N/A | N/A | $0.10 / $0.40 | N/A | N/A |
| **Gemini 2.0 Pro** | N/A | N/A | $2.00 / $8.00 | N/A | N/A |
| **Llama 3.1 405B** | $5.00 / $15.00 | $5.00 / $15.00 | $5.00 / $15.00 | $3.50 / $10.50 | $4.00 / $12.00 |
| **Llama 3.1 70B** | $0.65 / $2.10 | $0.65 / $2.10 | $0.65 / $2.10 | $0.45 / $1.47 | $0.52 / $1.68 |
| **Mistral Large 2** | $2.00 / $6.00 | Via catalog | $2.00 / $6.00 | N/A | N/A |
| **Cohere R+** | $2.50 / $10.00 | Via catalog | $2.50 / $10.00 | $1.75 / $7.00 | N/A |

### Per-Hour Pricing (Provisioned Throughput / Dedicated Capacity)

| Platform | Unit | Price (USD/hour) |
|---|---|---|
| AWS Bedrock PT (Claude 3.5 Sonnet) | 1 model unit (~100k TPM) | $39.60 |
| Azure PTU (GPT-4o) | 1 PTU | ~$8.00–15.00/hr |
| GCP Vertex AI (Llama 3 70B) | 1 node (L4 GPU) | $1.50 |
| GCP Vertex AI (Gemini Pro) | 1 base unit | $0.60 |
| OCI AI (Llama 3 70B) | 1 GPU (A100) | $1.10 |
| IBM watsonx (Granite 3.0) | 1 vGPU | $0.85 |

### Training Compute Pricing (Per GPU-Hour, USD)

| GPU Type | AWS | Azure | GCP | OCI | IBM |
|---|---|---|---|---|---|
| **H100 (80GB)** | $4.50–5.50 | $4.00–5.00 | $4.00–5.00 | $3.00–4.00 | $3.50–4.50 |
| **H100 (spot)** | $1.35–1.65 | $1.20–1.50 | $1.20–1.50 | $0.90–1.20 | $1.05–1.35 |
| **A100 (80GB)** | $3.06–3.82 | $2.85–3.50 | $2.85–3.50 | $2.00–2.80 | $2.40–3.00 |
| **A100 (spot)** | $0.92–1.15 | $0.85–1.05 | $0.85–1.05 | $0.60–0.84 | $0.72–0.90 |
| **L4** | $0.94 | $0.85 | $0.80 | $0.55 | $0.65 |
| **Trainium2 (trn2)** | $4.97–12.98 | N/A | N/A | N/A | N/A |
| **TPU v5p** | N/A | N/A | $4.20 | N/A | N/A |

**Note:** All prices approximate as of early 2026 and subject to change. Reserved/committed prices can be 30–50% lower than on-demand.

---

## 9. Latency and Performance Comparison

### Inference Latency (P50, milliseconds, for 500-token completion)

| Model | AWS Bedrock | Azure OpenAI | GCP Vertex AI | OCI AI |
|---|---|---|---|---|
| GPT-4o | N/A | 1800 | N/A | N/A |
| Claude 3.5 Sonnet | 2100 | Via catalog: ~2200 | Via catalog: ~2150 | N/A |
| Gemini 2.0 Flash | N/A | N/A | 800 | N/A |
| Llama 3.1 70B | 950 | 900 | 850 | 1100 |
| Llama 3.1 405B | 3200 | 3100 | 3000 | 3500 |

### Performance Characteristics

| Criteria | AWS Bedrock | Azure OpenAI | GCP Vertex AI | OCI AI |
|---|---|---|---|---|
| **Cold start time** | 2–5s (new endpoint) | 1–3s (PTU), 5–15s (serverless) | 1–4s | 3–8s |
| **Max throughput per region** | Very high (many regions) | Moderate (capacity constrained) | High | Moderate |
| **Streaming latency (first token)** | 300–800ms | 200–600ms | 200–500ms | 400–1000ms |
| **Burst capacity** | High (autoscaling) | Requires PTU planning | High (autoscaling) | Moderate (request-based) |

---

## 10. Regional Availability Matrix

### AI Service Regions

| Platform | US | Europe | Asia-Pacific | South America | Middle East | Africa |
|---|---|---|---|---|---|---|
| **AWS Bedrock** | 6 regions | 6 regions | 8 regions | 2 regions | 2 regions | 1 region |
| **Azure OpenAI** | 6 regions | 5 regions | 6 regions | 1 region | 1 region | 0 |
| **Azure AI Studio** | 6 regions | 5 regions | 6 regions | 1 region | 1 region | 0 |
| **GCP Vertex AI** | 5 regions | 4 regions | 5 regions | 1 region | 1 region | 0 |
| **OCI Generative AI** | 3 regions | 3 regions | 3 regions | 0 | 0 | 0 |
| **IBM watsonx** | 4 regions | 3 regions | 3 regions | 1 region | 0 | 0 |

### Data Residency Considerations

| Requirement | AWS | Azure | GCP | OCI | IBM |
|---|---|---|---|---|---|
| **EU data residency** | ✅ (Frankfurt, Ireland, London, Paris, Stockholm, Milan) | ✅ (Netherlands, Ireland, Switzerland, UK, France) | ✅ (Belgium, Frankfurt, London, Zurich) | ✅ (Frankfurt, London) | ✅ (Frankfurt, London) |
| **GDPR compliance** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **HIPAA eligible** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **FedRAMP (High)** | ✅ | ✅ | ✅ | ✅ | Pending |
| **PCI DSS** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **SOC 2 Type II** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **ISO 27001** | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 11. Feature Comparison — Fine-Tuning, RAG, Agents

### Fine-Tuning Capabilities

| Feature | AWS Bedrock | Azure OpenAI | GCP Vertex AI | OCI AI | IBM watsonx |
|---|---|---|---|---|---|
| **Model fine-tuning** | ✅ (Titan, Llama, Cohere) | ✅ (GPT-4o, GPT-4o mini, GPT-3.5) | ✅ (Gemini, Llama, Mistral) | ✅ (Llama, Cohere) | ✅ (Granite) |
| **LoRA/QLoRA support** | ✅ (SageMaker) | ❌ (only full FT) | ✅ (Vertex AI) | ✅ | ✅ |
| **DPO/RLHF** | ✅ (SageMaker) | ❌ | ✅ (Vertex AI) | ❌ | ✅ |
| **Continuous pre-training** | ✅ (SageMaker) | ❌ | ✅ (Vertex AI) | ❌ | ✅ |
| **Custom model import** | ✅ | ✅ (custom) | ✅ | ✅ | ✅ |
| **AutoML fine-tuning** | ✅ | ❌ | ✅ (AutoSxS) | ❌ | ❌ |
| **Hyperparameter tuning** | ✅ (SageMaker) | ❌ | ✅ (Vizier) | ✅ | ✅ |

### RAG (Retrieval-Augmented Generation)

| Feature | AWS Bedrock | Azure AI Studio | GCP Vertex AI | OCI AI | IBM watsonx |
|---|---|---|---|---|---|
| **Managed RAG** | ✅ (Knowledge Bases) | ✅ (AI Search + Prompt Flow) | ✅ (RAG Engine) | ❌ | ✅ |
| **Vector database options** | Aurora, Pinecone, Redis, OpenSearch | AI Search Vector, Cosmos DB | Vector Search, AlloyDB | Oracle 23ai Vector | watsonx.data, Chroma |
| **Hybrid search** | ✅ | ✅ (semantic + keyword) | ✅ | ✅ (Oracle DB) | ✅ |
| **Multi-modal RAG** | ✅ (Titan Multimodal) | ✅ (GPT-4o vision) | ✅ (Gemini) | ❌ | ❌ |
| **Document parsing** | ✅ (S3 + Lambda) | ✅ (Azure AI Document Intelligence) | ✅ (Document AI) | ✅ (OCI Document Understanding) | ✅ (watsonx Discovery) |
| **Chunking strategies** | ✅ (customizable) | ✅ (customizable) | ✅ (customizable) | ⚠️ (basic) | ✅ (customizable) |
| **Metadata filtering** | ✅ | ✅ | ✅ | ⚠️ | ✅ |

### Agent Capabilities

| Feature | AWS Bedrock/Agents | Azure AI Studio | GCP Vertex AI Agent Builder | OCI Digital Assistant | IBM watsonx Orchestrate |
|---|---|---|---|---|---|
| **Multi-step reasoning** | ✅ | ✅ | ✅ | ⚠️ | ✅ |
| **Tool/function calling** | ✅ (Action Groups) | ✅ (Tool Definition) | ✅ (OpenAPI) | ✅ | ✅ |
| **Code interpreter** | ✅ | ✅ (Code Interpreter) | ✅ | ❌ | ❌ |
| **Multi-agent orchestration** | ✅ | ✅ (Semantic Kernel) | ✅ (Agent Builder) | ❌ | ✅ |
| **Memory/persistence** | ✅ (session state) | ✅ (semantic memory) | ✅ (conversation history) | ✅ | ✅ |
| **Guardrails** | ✅ (Bedrock Guardrails) | ✅ (Content Safety) | ✅ (Safety filters) | ⚠️ | ✅ (watsonx.governance) |
| **Human-in-the-loop** | ⚠️ (manual) | ✅ (Copilot Studio) | ✅ (Agent Builder) | ❌ | ✅ |

---

## 12. Data Privacy and Security

### Training Data Usage

| Platform | Data Used for Training? | Data Retention | Opt-Out Available? |
|---|---|---|---|
| **AWS Bedrock** | No (models isolated) | Per customer config | N/A |
| **Azure OpenAI** | No (with SLA) | 30 days (abuse monitoring), opt-out available | ✅ |
| **GCP Vertex AI** | No (by default) | Per customer config | ✅ |
| **OCI AI** | No | Per customer config | N/A |
| **IBM watsonx** | No (Granite models) | Per customer config | N/A |

### Encryption

| Feature | AWS | Azure | GCP | OCI | IBM |
|---|---|---|---|---|---|
| **Encryption at rest** | AES-256 (KMS) | AES-256 (Azure Key Vault) | AES-256 (Cloud KMS) | AES-256 (Vault) | AES-256 (Key Protect) |
| **Encryption in transit** | TLS 1.3 | TLS 1.3 | TLS 1.3 | TLS 1.3 | TLS 1.3 |
| **BYOK** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **VPC/Private endpoints** | ✅ (PrivateLink) | ✅ (Private Endpoint) | ✅ (Private Service Connect) | ✅ (OCI VCN) | ✅ (IBM Cloud VPC) |
| **Data isolation** | Per-account | Per-account | Per-account | Per-tenant | Per-account |

---

## 13. Specialized Hardware Comparison

| Hardware | AWS | Azure | GCP | OCI | IBM |
|---|---|---|---|---|---|
| **H100** (Hopper) | ✅ P5 instances | ✅ ND H100 v5 | ✅ A3 High | ✅ BM.GPU.H100.8 | ✅ |
| **B200** (Blackwell) | ✅ P5e instances | ✅ ND B200 (projected) | ✅ A4 (projected) | Projected | ❌ |
| **H200** | ✅ Hpc7a | ✅ ND H200 v5 | Projected | ❌ | ❌ |
| **A100** | ✅ P4d/P4de | ✅ ND A100 v4 | ✅ A2 | ✅ BM.GPU.A100 | ✅ |
| **Trainium 2** | ✅ Trn2/Trn2 Ultra | ❌ | ❌ | ❌ | ❌ |
| **Inferentia 2** | ✅ Inf2 | ❌ | ❌ | ❌ | ❌ |
| **TPU v5p** | ❌ | ❌ | ✅ | ❌ | ❌ |
| **TPU Trillium** | ❌ | ❌ | ✅ (2025+) | ❌ | ❌ |
| **Ampere Altra** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **L4** | ✅ G6 | ✅ NC L4s | ✅ G2 | ✅ GPU.L4 | ❌ |

---

## 14. MLOps and Tooling

| Capability | AWS SageMaker | Azure ML | GCP Vertex AI | OCI Data Science | IBM watsonx |
|---|---|---|---|---|---|
| **Experiment tracking** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Pipeline orchestration** | ✅ (Pipelines) | ✅ (Pipelines) | ✅ (Kubeflow) | ✅ (Jobs) | ✅ (Pipelines) |
| **Model registry** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Feature store** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Model monitoring** | ✅ (Model Monitor) | ✅ (Model Monitor) | ✅ (Model Monitoring) | ⚠️ (basic) | ✅ (watsonx.governance) |
| **Explainability** | ✅ (Clarify) | ✅ (Responsible AI) | ✅ (Explainable AI) | ❌ | ✅ (bias detection) |
| **CI/CD integration** | ✅ (CodePipeline) | ✅ (Azure DevOps) | ✅ (Cloud Build) | ✅ (OCI DevOps) | ✅ (Toolchain) |
| **Kubernetes native** | ✅ (EKS + SageMaker) | ✅ (AKS + Azure ML) | ✅ (GKE + Kubeflow) | ✅ (OKE + OCI DS) | ✅ (OpenShift) |
| **AutoML** | ✅ (Autopilot) | ✅ (AutoML) | ✅ (AutoML) | ❌ | ✅ |

---

## 15. Migration Difficulty Assessment

| Migration Path | Difficulty | Key Challenges |
|---|---|---|
| Single cloud -> Multi-cloud (greenfield) | Medium | Architecture design, abstraction layer build-out, multi-cloud networking |
| AWS -> Azure (GPT workloads) | Medium | API differences, IAM mapping, data transfer costs |
| Azure -> AWS (non-OpenAI) | Low-Medium | Model portability (ONNX), pipeline reconfiguration |
| GCP -> AWS (TPU workloads) | High | TPU vs GPU code changes, TensorFlow/JAX adaptation |
| Any -> OCI (cost optimization) | Low | Primarily API compatibility, data location |
| Any -> IBM (governance) | Low-Medium | Governance tooling integration, model format compatibility |

---

## 16. Decision Framework

### When to Choose Each Platform as Primary

| Use Case | Recommended Primary | Secondary |
|---|---|---|
| Enterprise Microsoft shop, need GPT-4o | **Azure** | AWS (for non-OpenAI models) |
| Deep data analytics, need custom training | **GCP** | AWS (for breadth) |
| Broadest model selection, flexible workloads | **AWS** | GCP (for specific models) |
| Cost-sensitive, good GPU availability | **OCI** | AWS (spot instances) |
| Regulated industry, need governance | **IBM** | Azure (compliance tools) |
| Research/academic (budget constrained) | **GCP (free tier)** | OCI (low-cost GPU) |
| Production agent systems | **AWS/Azure** (tie) | GCP (Gemini if relevant) |

### Decision Matrix (Score 1-5)

| Criteria | AWS | Azure | GCP | OCI | IBM |
|---|---|---|---|---|---|
| Model breadth | 5 | 4 | 4 | 2 | 3 |
| OpenAI models | 1 | 5 | 2 | 1 | 1 |
| Cost (inference) | 3 | 3 | 4 | 5 | 3 |
| Cost (training) | 4 | 3 | 4 | 5 | 3 |
| Latency | 4 | 4 | 5 | 3 | 3 |
| Regional availability | 5 | 4 | 4 | 2 | 3 |
| MLOps tooling | 5 | 4 | 5 | 2 | 4 |
| Governance/Compliance | 4 | 4 | 3 | 3 | 5 |
| Enterprise integration | 4 | 5 | 3 | 3 | 4 |
| Agent framework | 4 | 5 | 4 | 2 | 3 |
| RAG capabilities | 4 | 5 | 5 | 3 | 4 |
| Fine-tuning flexibility | 5 | 3 | 5 | 3 | 4 |
| Community/ecosystem | 5 | 4 | 4 | 2 | 3 |
| Migration ease | 3 | 3 | 3 | 4 | 4 |
| **Total** | **56** | **54** | **55** | **40** | **46** |

**Note:** These scores are relative and will vary based on specific organizational requirements. Use this as a starting framework, not a definitive ranking.

---

## Appendix: API Endpoint Reference

### AWS Bedrock
```
# Invoke model
POST https://bedrock-runtime.{region}.amazonaws.com/model/{modelId}/invoke
POST https://bedrock-runtime.{region}.amazonaws.com/model/{modelId}/invoke-with-response-stream

# Knowledge base
POST https://bedrock-agent-runtime.{region}.amazonaws.com/knowledgebases/{kbId}/retrieve
POST https://bedrock-agent-runtime.{region}.amazonaws.com/knowledgebases/{kbId}/retrieveAndGenerate
```

### Azure OpenAI
```
# Chat completions
POST https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version=2025-01-01-preview

# Embeddings
POST https://{resource}.openai.azure.com/openai/deployments/{deployment}/embeddings?api-version=2025-01-01-preview

# Fine-tuning
POST https://{resource}.openai.azure.com/openai/fine-tunes?api-version=2025-01-01-preview
```

### GCP Vertex AI
```
# Predict (Gemini)
POST https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/google/models/{model}:generateContent
POST https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/google/models/{model}:streamGenerateContent

# Predict (third-party models)
POST https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/endpoints/{endpoint}:predict
```

### OCI Generative AI
```
# Chat
POST https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/chat
# Summarize
POST https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/summarizeText
# Embed
POST https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText
```

### IBM watsonx
```
# Generate
POST https://{region}.ml.cloud.ibm.com/ml/v1-beta/generation/text?version=2025-01-01
# Tokenize
POST https://{region}.ml.cloud.ibm.com/ml/v1-beta/tokenization?version=2025-01-01
# Embed
POST https://{region}.ml.cloud.ibm.com/ml/v1-beta/embeddings?version=2025-01-01
```

This concludes the Cloud AI Platform Comparison document. Refer to Document 03 for guidance on architecting multi-cloud AI solutions that leverage the strengths covered here.

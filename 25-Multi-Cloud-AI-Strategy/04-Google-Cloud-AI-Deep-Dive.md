# 04 — Google Cloud AI Deep Dive

> **Google Cloud AI: Vertex AI (Gemini, Claude, Llama), Model Garden, AI Platform, AutoML, Document AI, Dialogflow, Translation API, Speech-to-Text. Google-specific strengths: TPU access, BigQuery ML, Gemini 1M context, multimodal strength. Vertex AI Agent Builder, Vertex AI Search. Comparison with AWS/Azure.**

---

## Table of Contents

1. [Google Cloud AI Landscape](#1-google-cloud-ai-landscape)
2. [Vertex AI — Unified ML Platform](#2-vertex-ai--unified-ml-platform)
   - [2.1 Platform Overview](#21-platform-overview)
   - [2.2 Model Garden](#22-model-garden)
   - [2.3 Training & Custom Models](#23-training--custom-models)
   - [2.4 Model Deployment & Prediction](#24-model-deployment--prediction)
   - [2.5 AutoML](#25-automl)
   - [2.6 Vertex AI Pipelines](#26-vertex-ai-pipelines)
   - [2.7 Model Registry & Governance](#27-model-registry--governance)
3. [Gemini Models — Deep Dive](#3-gemini-models--deep-dive)
   - [3.1 Gemini 2.0 Model Family](#31-gemini-20-model-family)
   - [3.2 Gemini 1.5 Pro / Flash](#32-gemini-15-pro--flash)
   - [3.3 Multimodal Capabilities](#33-multimodal-capabilities)
   - [3.4 1M+ Token Context Window](#34-1m-token-context-window)
   - [3.5 Gemini API & SDKs](#35-gemini-api--sdks)
4. [Vertex AI Agent Builder & Search](#4-vertex-ai-agent-builder--search)
   - [4.1 Agent Builder](#41-agent-builder)
   - [4.2 Vertex AI Search](#42-vertex-ai-search)
   - [4.3 Conversational AI with Dialogflow](#43-conversational-ai-with-dialogflow)
5. [BigQuery ML](#5-bigquery-ml)
6. [Specialized AI Services](#6-specialized-ai-services)
   - [6.1 Document AI](#61-document-ai)
   - [6.2 Speech-to-Text & Text-to-Speech](#62-speech-to-text--text-to-speech)
   - [6.3 Translation API](#63-translation-api)
   - [6.4 Vision API](#64-vision-api)
   - [6.5 Natural Language API](#65-natural-language-api)
   - [6.6 Video Intelligence API](#66-video-intelligence-api)
7. [TPU Technology & Performance](#7-tpu-technology--performance)
8. [Architecture Patterns on GCP](#8-architecture-patterns-on-gcp)
   - [8.1 Vertex AI + BigQuery ML Pipeline](#81-vertex-ai--bigquery-ml-pipeline)
   - [8.2 RAG with Vertex AI Search](#82-rag-with-vertex-ai-search)
   - [8.3 Multi-Region Global Inference](#83-multi-region-global-inference)
9. [Pricing & Cost Optimization](#9-pricing--cost-optimization)
10. [Security & Compliance](#10-security--compliance)
11. [Comparison with AWS and Azure](#11-comparison-with-aws-and-azure)
12. [References](#12-references)

---

## 1. Google Cloud AI Landscape

Google Cloud AI leverages Google's deep research heritage in machine learning and artificial intelligence. The portfolio is anchored by:

**Tier 1 — Foundation Models & Platform:**
- **Vertex AI:** Unified platform for the entire ML lifecycle — model training, deployment, evaluation, and monitoring.
- **Gemini Models:** Google's flagship multimodal foundation models (text, image, video, audio, code).
- **Model Garden:** Catalog of 150+ foundation models from Google, Anthropic, Meta, Mistral, and open-source communities.

**Tier 2 — ML Infrastructure:**
- **TPU (Tensor Processing Units):** Custom-designed ASICs for ML training and inference.
- **GPU Access:** A100, H100, L4, and upcoming B200 availability.
- **BigQuery ML:** Train and deploy ML models directly on data in BigQuery using SQL.

**Tier 3 — Specialized AI Services:**
- **Document AI:** Enterprise document processing with specialized processors.
- **Dialogflow:** Conversational AI for chatbots and contact centers.
- **Translation API:** Neural machine translation in 130+ languages.
- **Speech-to-Text / Text-to-Speech:** Audio transcription and synthesis.
- **Vision API:** Image analysis and detection.
- **Video Intelligence API:** Video content analysis.
- **Natural Language API:** Entity recognition, sentiment, and syntax analysis.
- **Vertex AI Agent Builder:** No-code AI agent creation.
- **Vertex AI Search:** Enterprise search with semantic understanding.
- **Recommendations AI:** Personalized product recommendations.
- **Healthcare Natural Language API:** Medical NLP (HIPAA-eligible).

This deep dive covers Vertex AI (the central platform), Gemini models (Google's unique differentiator), TPU access, BigQuery ML, and specialized services — with particular emphasis on what distinguishes GCP from AWS and Azure.

---

## 2. Vertex AI — Unified ML Platform

### 2.1 Platform Overview

Vertex AI is Google Cloud's unified machine learning platform, bringing together all GCP AI services under a single interface. It is positioned as a direct competitor to AWS SageMaker and Azure ML.

**Core Capabilities:**
- **Data Preparation:** Connect to BigQuery, Cloud Storage, or external sources. Label data with Vertex AI Labeling (human-in-the-loop).
- **AutoML:** Build custom models for tabular, image, text, and video data without writing ML code.
- **Custom Training:** Train models using any ML framework with managed infrastructure.
- **Model Evaluation:** Evaluate model performance, fairness, and explainability.
- **Model Deployment:** Deploy models to endpoints with automatic scaling.
- **Model Monitoring:** Detect drift, data skew, and performance degradation.
- **ML Governance:** Model Registry, lineage tracking, and approval workflows.
- **Gen AI Studio:** Prompt design, experimentation, and tuning for foundation models.
- **Agent Builder:** Build AI agents with no-code and custom-code options.

**Key Differentiators:**
- **Unified API:** A single API surface for all ML tasks (training, prediction, evaluation).
- **BigQuery Integration:** Native integration with Google's data warehouse.
- **TPU Support:** First cloud provider to offer custom TPU hardware for ML.
- **Open Source Leadership:** Deep integration with TensorFlow, JAX, PyTorch, and Kubeflow.

### 2.2 Model Garden

Model Garden is Google's marketplace for foundation models, offering 150+ models from multiple sources:

**Google's Models:**
- **Gemini 2.0 Pro:** Flagship multimodal model (text, image, video, audio, code).
- **Gemini 2.0 Flash:** Fast, cost-optimized model for high-throughput workloads.
- **Gemini 1.5 Pro:** Prior generation, still widely deployed for stability.
- **Gemini 1.5 Flash:** Cost-optimized prior generation.
- **Gemma 2 / Gemma 3:** Open-weight models for self-hosting.
- **Med-Gemini:** Healthcare-specific model (HIPAA-eligible).
- **Codey:** Code generation and completion models.
- **Imagen:** Text-to-image generation.
- **Chirp:** Speech-to-text models.

**Third-Party Models (available in Model Garden):**
- **Anthropic:** Claude 4 Opus, Claude 3.5 Sonnet.
- **Meta:** Llama 4, Llama 3.1.
- **Mistral:** Mistral Large 2, Mistral 7B, Mixtral 8x22B.
- **Cohere:** Command R+, Command R.
- **NVIDIA:** Nemotron-4, Llama-based models.
- **Hugging Face:** Thousands of open-source models through Vertex AI Model Registry.

**Model Garden Capabilities:**
- **Model comparison:** Side-by-side comparison of model performance and pricing.
- **One-click deployment:** Deploy any model to a managed endpoint.
- **Fine-tuning:** Support for supervised fine-tuning (SFT) and RLHF.
- **Model evaluation:** Built-in evaluation benchmarks and custom evaluation.

### 2.3 Training & Custom Models

**Managed Training Infrastructure:**
- **Vertex AI Training:** Managed training jobs with automatic infrastructure provisioning.
- **Distributed Training:** Support for data parallelism, model parallelism, and pipeline parallelism.
- **Accelerator Options:**
  - **TPU v6 (Ironwood):** 8,064 TFLOPS per chip, up to 8,960 TPU chips in a pod.
  - **TPU v5p:** 459 TFLOPS per chip, 8,960 chips per pod.
  - **TPU v4:** 275 TFLOPS per chip, 4,096 chips per pod.
  - **GPU:** A100 (80GB), H100 (80GB), L4 (24GB), T4 (16GB).
- **Hyperparameter Tuning:** Vizier-based automatic HPO with early stopping.
- **Experiment Tracking:** Vertex AI Experiments for tracking training runs.

**Supported Frameworks:**
- TensorFlow (2.x)
- PyTorch (with Lightning, FSDP support)
- JAX (native TPU support)
- scikit-learn
- XGBoost
- Hugging Face Transformers

**Custom Containers:**
- Bring your own Docker container for complete flexibility.
- Pre-built containers for common frameworks.
- Custom container versions for specific CUDA/cuDNN requirements.

### 2.4 Model Deployment & Prediction

**Deployment Options:**
- **Vertex AI Endpoints:** Managed, auto-scaling prediction service.
  - Supports CPU, GPU, and TPU model serving.
  - Automatic scaling based on request load.
  - Configurable traffic splitting for A/B testing.
- **Vertex AI Online Prediction:** Low-latency inference for real-time applications.
- **Vertex AI Batch Prediction:** Process large datasets asynchronously.
- **Vertex AI Private Endpoints:** Deploy within a private network (VPC-SC).

**Prediction Features:**
- **Explainable AI:** Feature importance (SHAP, Integrated Gradients) for predictions.
- **Model Monitoring:** Track prediction quality, feature drift, and data skew.
- **Autoscaling:** Scale to zero when not in use (saving costs).
- **Multi-model endpoints:** Host multiple models on a single endpoint.

**Serving Infrastructure:**
- **TensorFlow Serving:** Optimized for TensorFlow models.
- **PyTorch Serve:** TorchServe integration.
- **NVIDIA Triton Inference Server:** Multi-framework serving.
- **JAX Serving:** Native TPU serving for JAX models.

### 2.5 AutoML

Vertex AI AutoML enables building custom models without ML expertise:

**AutoML Capabilities:**
- **Tabular:** Regression, classification, forecasting (AutoML Tables).
- **Image:** Classification, object detection, segmentation.
- **Text:** Classification, entity extraction, sentiment analysis.
- **Video:** Classification, action recognition, object tracking.

**AutoML Workflow:**
1. Upload labeled data (CSV, JSONL, or BigQuery table).
2. Define the target column and objective.
3. AutoML automatically:
   - Performs feature engineering and selection.
   - Trains multiple model architectures.
   - Optimizes hyperparameters.
   - Selects the best-performing model.
4. Deploy the model to an endpoint with one click.

**Limitations:**
- Less flexibility than custom training.
- Higher per-prediction cost vs. custom models at scale.
- Limited to GCP-supported architectures.

### 2.6 Vertex AI Pipelines

Vertex AI Pipelines is a managed ML workflow orchestration service based on Kubeflow Pipelines (KFP) and TensorFlow Extended (TFX):

**Pipeline Features:**
- **DAG-based workflows:** Define pipeline as a directed acyclic graph of components.
- **Pre-built components:** Google-provided components for data processing, training, evaluation, deployment.
- **Custom components:** Package any Python code as a pipeline component.
- **Caching:** Automatic caching of component outputs (expensive re-execution avoided).
- **Parameterization:** Parameterize pipelines for reuse across environments.
- **Artifact tracking:** Track inputs and outputs of each pipeline step.
- **Scheduled execution:** Run pipelines on a schedule (daily, weekly, event-driven).

**Integration with CI/CD:**
- Cloud Build integration for ML CI/CD.
- Vertex AI Pipelines triggered by Cloud Build, Cloud Scheduler, or Eventarc.
- Artifact Registry for storing pipeline components.

**Example Pipeline Component:**

```python
from kfp.v2 import dsl

@dsl.component(
    base_image="python:3.10",
    packages_to_install=["pandas", "sklearn"]
)
def train_model(
    dataset: dsl.Dataset,
    model: dsl.Output[dsl.Model],
    accuracy: dsl.Output[Metrics]
):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    
    df = pd.read_csv(dataset.path)
    X = df.drop("target", axis=1)
    y = df["target"]
    
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)
    
    import joblib
    joblib.dump(clf, model.path)
    
    accuracy.log_metric("accuracy", clf.score(X, y))
```

### 2.7 Model Registry & Governance

Vertex AI Model Registry provides model cataloging and governance:

**Features:**
- **Model versions:** Track multiple versions of the same model.
- **Model aliases:** Assign aliases (e.g., "production", "staging") to model versions.
- **Model lineage:** Automatic tracking of training data, pipeline, and parameters.
- **Approval workflow:** Model sign-off process (development → staging → production).
- **Model monitoring:** Automatic detection of performance degradation.
- **Compliance documentation:** Export model cards and lineage reports.

**Integration:**
- Vertex AI Experiments for experiment tracking.
- Vertex AI Pipelines for automated model registration.
- Cloud Audit Logs for compliance.

---

## 3. Gemini Models — Deep Dive

Gemini is Google's most capable foundation model family, representing Google's primary competitive differentiator in the AI market.

### 3.1 Gemini 2.0 Model Family

**Gemini 2.0 Pro (Flagship):**
- **Context window:** Up to 2 million tokens.
- **Modalities:** Text, image, video, audio, code.
- **Performance:** Leading scores on MMLU, HumanEval, MATH, and multimodal benchmarks.
- **Features:** Native tool use, code execution, Google Search grounding, function calling.
- **Pricing:** $2.50/1M input tokens, $10.00/1M output tokens (approximate).

**Gemini 2.0 Flash (Cost-Optimized):**
- **Context window:** Up to 1 million tokens.
- **Modalities:** Same as Pro (text, image, video, audio, code).
- **Performance:** ~95% of Pro quality at ~15% of the cost.
- **Features:** Same as Pro.
- **Pricing:** $0.35/1M input tokens, $1.40/1M output tokens (approximate).

**Gemini 2.0 Nano (On-Device):**
- **Context window:** Up to 128K tokens.
- **Modalities:** Text-only (optimized).
- **Use case:** On-device inference for mobile and edge.
- **Pricing:** Not available as cloud API.

### 3.2 Gemini 1.5 Pro / Flash

Still available for customers needing stability or specific behavior:

**Gemini 1.5 Pro:**
- 1M token context window.
- Strong multimodal capabilities.
- Available for deployment in Vertex AI.
- Pricing: $1.25/1M input, $5.00/1M output.

**Gemini 1.5 Flash:**
- 1M token context window.
- ~90% of Pro quality at lower cost.
- Pricing: $0.075/1M input, $0.30/1M output.

### 3.3 Multimodal Capabilities

Gemini's multimodal processing is a key differentiator:

**Text:**
- Natural language understanding and generation.
- Support for 100+ languages.
- Translation, summarization, analysis.

**Image:**
- Direct image understanding (not just metadata).
- Object detection, scene understanding, OCR.
- Diagram and chart interpretation.
- Visual question answering.

**Video:**
- Native video understanding without frame extraction.
- Temporal reasoning (understanding events across time).
- Up to 1 hour of video in a single prompt.

**Audio:**
- Audio understanding (speech, music, environmental sounds).
- Transcriptions and audio analysis.
- Speaker identification.

**Code:**
- Code generation in 20+ languages.
- Code explanation, debugging, optimization.
- Multi-file codebase understanding.

### 3.4 1M+ Token Context Window

Gemini's long context window is a transformative capability:

**What 1M+ Token Context Enables:**
- **Full codebase analysis:** Process an entire codebase (100K+ LOC) in a single prompt.
- **Long document processing:** Analyze whole books, legal contracts, or research papers.
- **Multi-turn conversations:** Maintain full conversation history without summarization.
- **Video analysis:** Process hours of video content holistically.

**Performance Characteristics:**
- Context retrieval accuracy: ~99.7% for 512K tokens; ~99.2% for 1M+ tokens.
- Latency increases linearly with context length.
- Cost scales with input token count (all tokens in the context window are billed).

**Use Cases:**
- Legal document review (entire contracts in one prompt).
- Code review (full repositories).
- Customer support (entire conversation history).
- Research analysis (full papers + supplementary materials).

### 3.5 Gemini API & SDKs

**API Access:**
- **Vertex AI Gemini API:** Enterprise API with full security, governance, and SLA.
- **Gemini API (ai.google.dev):** Developer-focused API for experimentation and prototyping.
- **Google AI Studio:** Web-based environment for prompt engineering and model testing.

**SDKs:**
- Python: `google-cloud-aiplatform` and `google-generativeai`
- Node.js: `@google-ai/generativelanguage`
- Go: `cloud.google.com/go/aiplatform`
- Java: `com.google.cloud:google-cloud-aiplatform`
- REST and gRPC APIs

**Key API Features:**
- Streaming responses for real-time interaction.
- Function calling (tool use) for connecting to external APIs.
- Safety settings configurable per request.
- System instructions for consistent behavior.

---

## 4. Vertex AI Agent Builder & Search

### 4.1 Agent Builder

Vertex AI Agent Builder (formerly Gen AI Agent Builder) enables building AI agents:

**Capabilities:**
- **No-code agent creation:** Build conversational agents through a web UI.
- **Custom code extension:** Extend agents with Python, Node.js, or any REST API.
- **Multi-agent orchestration:** Coordinate multiple specialized agents.
- **Human hand-off:** Escalate to human agents when confidence is low.
- **Channel integration:** Deploy to web, mobile, Google Chat, and third-party platforms.
- **Data source connection:** Connect to websites, documents, databases, and APIs.

**Agent Features:**
- Dynamic routing based on intent and context.
- Memory and session management.
- Flywheel data for continuous improvement.
- Monitoring and analytics dashboard.

### 4.2 Vertex AI Search

Vertex AI Search (formerly Enterprise Search on Gen App Builder) provides AI-powered search:

**Capabilities:**
- **Semantic search:** Understands query intent, not just keywords.
- **Hybrid search:** Combines semantic and keyword search.
- **Multi-modal search:** Search across text, images, and structured data.
- **Enterprise data source connectors:**
  - Cloud Storage, BigQuery, Spanner.
  - Websites (public and authenticated).
  - Third-party: Salesforce, Confluence, Jira, ServiceNow.
- **Unstructured & structured data:** Web pages, PDFs, HTML, Office docs, databases.
- **Summarization:** AI-generated answers with citations.

**Key Features:**
- **Natural language queries:** Users can ask questions conversationally.
- **Result ranking:** ML-based relevance scoring.
- **Faceted search:** Filter by category, date, source, etc.
- **Security:** Results respect document-level access control (if configured).

### 4.3 Conversational AI with Dialogflow

Dialogflow is GCP's conversational AI platform:

**Dialogflow CX:**
- Advanced conversational AI for complex use cases.
- Visual flow builder for conversation design.
- State-based architecture for multi-turn conversations.
- Version management for conversation flows.
- Test coverage, A/B testing, and analytics.
- Contact center integration (with CCAIP).

**Dialogflow ES:**
- Simpler, intent-based conversational AI.
- Suitable for FAQ bots and simple chatbots.
- Pre-built agents for common use cases.

**Generative AI Integration:**
- Dialogflow CX Agent Builder includes generative AI fallback.
- When intents don't match, Gemini generates a response.
- Human agent hand-off when needed.

---

## 5. BigQuery ML

BigQuery ML enables training and deploying ML models directly on data in BigQuery using SQL:

**Capabilities:**
- **Training:** Train models using SQL syntax — no code required.
- **Model types supported:**
  - Linear regression and logistic regression.
  - Boosted trees (XGBoost) and random forests.
  - Deep neural networks (DNN).
  - Time-series models (ARIMA, Exponential Smoothing).
  - Matrix factorization for recommendation systems.
  - PCA and k-means clustering.
  - AutoML (automated model selection and tuning).
- **Model evaluation:** Evaluate models using standard metrics.
- **Prediction:** Generate predictions with SQL or export to Vertex AI endpoints.

**How It Works:**
```sql
-- Train a model
CREATE OR REPLACE MODEL `my_project.my_dataset.sales_forecast`
OPTIONS(
  model_type='ARIMA_PLUS',
  time_series_timestamp_col='date',
  time_series_data_col='sales',
  time_series_id_col='product_id'
) AS
SELECT date, product_id, sales
FROM `my_project.my_dataset.historical_sales`;

-- Generate predictions
SELECT *
FROM ML.FORECAST(
  MODEL `my_project.my_dataset.sales_forecast`,
  STRUCT(30 AS horizon, 0.8 AS confidence_level)
);
```

**Advantages:**
- **No data movement:** Train models where data lives.
- **SQL interface:** Accessible to analysts and data engineers.
- **Automatic scaling:** BigQuery handles infrastructure.
- **Integrated with Vertex AI:** Export models to Vertex AI for advanced deployment.

---

## 6. Specialized AI Services

### 6.1 Document AI

Document AI is Google's enterprise document processing platform:

**Processors (Pre-built):**
- **OCR Processor:** Extract text from documents (handwriting and print).
- **Form Parser:** Extract key-value pairs from forms.
- **Invoice Parser:** Extract line items, totals, vendor details.
- **Receipt Parser:** Extract merchant, date, total, line items.
- **Identity Document Parser:** Extract data from passports, driver's licenses.
- **Contract Parser:** Extract parties, dates, obligations, clauses.
- **Procurement Document Parser:** Purchase orders, invoices.
- **US Healthcare:** HIPAA-eligible medical document processing.

**Custom Processors:**
- Train Document AI on your specific document types.
- Use AI Platform for custom model training.

**Workflow Builder:**
- Visual workflow for document processing pipelines.
- Validation, enrichment, and human review stages.
- Integration with Cloud Functions, Pub/Sub, and BigQuery.

**Pricing:**
- OCR: $1.50 per 1,000 pages.
- Specialized processors: $5.00–$15.00 per 1,000 pages.
- Custom processors: Based on training compute hours.

### 6.2 Speech-to-Text & Text-to-Speech

**Speech-to-Text:**
- **Standard:** 125+ languages, real-time and batch transcription.
- **Enhanced:** Improved accuracy for specific domains (phone calls, video, medical).
- **Medical:** HIPAA-eligible transcription with medical terminology support.
- **Custom:** Domain-specific vocabulary and language models.
- **Features:** Speaker diarization, word-level timestamps, profanity filtering.
- **Pricing:** $0.006–$0.024 per minute (standard); $0.036–$0.060 per minute (enhanced).

**Text-to-Speech:**
- **WaveNet voices:** 380+ voices across 70+ languages (natural sounding).
- **Studio voices:** Professional-quality voices for production use.
- **Custom voice:** Create a custom voice from a few hours of audio.
- **SSML support:** Fine-grained control over pronunciation, pitch, speed.
- **Pricing:** $0.000004–$0.000016 per character (WaveNet/Studio).

### 6.3 Translation API

**Google Cloud Translation:**
- **Standard:** Neural machine translation for 130+ languages.
- **Advanced (Premium):** Higher quality, custom models, glossary support.
- **AutoML Translation:** Train custom translation models on domain-specific parallel data.
- **Document Translation:** Translate entire documents (PDF, Word, HTML) preserving formatting.
- **Batch Translation:** Large-scale batch processing.
- **Pricing:** $20.00 per 1M characters (standard); $80.00 per 1M characters (advanced).

### 6.4 Vision API

**Cloud Vision API:**
- **Label detection:** General object and scene recognition (10,000+ categories).
- **Text detection (OCR):** Printed and handwritten text in 50+ languages.
- **Facial detection:** Detect faces and facial attributes.
- **Landmark detection:** Identify famous landmarks.
- **Logo detection:** Recognize company logos.
- **Safe search:** Detect adult content.
- **Image properties:** Dominant colors, cropping suggestions.
- **Web detection:** Find similar images on the web.
- **Pricing:** $0.50–$1.50 per 1,000 images (standard features).

**Video Intelligence API:**
- **Shot detection:** Scene changes.
- **Label detection:** Objects and concepts in video.
- **Explicit content detection:** Adult content.
- **Person detection:** Track people across frames.
- **Text detection:** On-screen text.
- **Logo detection:** Brand appearance tracking.
- **Speech transcription:** ASR for video audio tracks.
- **Pricing:** $0.05–$0.15 per minute of video.

### 6.5 Natural Language API

**Cloud Natural Language API:**
- **Entity analysis:** Identify and tag entities (people, places, organizations).
- **Sentiment analysis:** Overall and per-entity sentiment.
- **Content classification:** Classify content into 700+ categories.
- **Syntax analysis:** Parts of speech, dependency trees.
- **Entity sentiment:** Sentiment toward specific entities.
- **Pricing:** $1.00 per 1,000 text records (entity/sentiment).

**Healthcare Natural Language API:**
- Medical entity extraction (conditions, medications, procedures).
- Relationship extraction (dosage, frequency, route).
- HIPAA-eligible.
- Integration with HL7 FHIR.

### 6.6 Video Intelligence API

**Cloud Video Intelligence:**
- **Shot change detection:** Identify boundaries between shots.
- **Label detection:** Identify objects, places, and actions.
- **Explicit content detection:** Flag adult content.
- **Face detection:** Track faces across frames.
- **Text detection:** OCR on video frames.
- **Logo detection:** Identify brand logos.
- **Speech transcription:** Transcribe audio track.
- **Object tracking:** Track objects across video frames.
- **Pricing:** $0.05–$0.15 per minute of video.

---

## 7. TPU Technology & Performance

Google's Tensor Processing Units (TPUs) are a major differentiator:

**TPU v6 (Ironwood) — 2025:**
- Performance: 8,064 TFLOPS (FP16) per chip.
- Memory: 64 GB HBM2e per chip.
- Pod size: Up to 8,960 chips.
- Pod performance: 72+ PFLOPS.
- Best for: Large-scale training and inference.

**TPU v5p — 2024:**
- Performance: 459 TFLOPS (FP16) per chip.
- Memory: 48 GB HBM2 per chip.
- Pod size: Up to 8,960 chips.
- Pod performance: 4.1 PFLOPS.
- Best for: Training large models (100B+ parameters).

**TPU v5e — 2023:**
- Performance: 197 TFLOPS (FP16) per chip.
- Memory: 24 GB HBM2 per chip.
- Pod size: Up to 256 chips.
- Cost-optimized for training and inference.

**TPU Advantages:**
- **Cost efficiency:** TPUs often deliver better price/performance than GPUs for large-scale ML.
- **Scalability:** TPU pods with thousands of interconnected chips.
- **Optimized for Google models:** Gemini models are natively designed for TPU architecture.
- **Availability:** Less constrained than GPU supply for large deployments.

**TPU Limitations:**
- **Framework support:** Best with JAX and TensorFlow; PyTorch support improving but not as mature.
- **Model compatibility:** Not all model architectures benefit from TPU optimization.
- **Geographic availability:** TPU availability is limited to select regions.

---

## 8. Architecture Patterns on GCP

### 8.1 Vertex AI + BigQuery ML Pipeline

```
BigQuery (Data Warehouse)
    │
    ├── BigQuery ML (simple models via SQL)
    │   └── Predictions back to BigQuery
    │
    └── Vertex AI (complex models)
        │
        ├── AutoML / Custom Training
        ├── Model Registry
        ├── Endpoint (Online Prediction)
        └── Batch Prediction → BigQuery / Cloud Storage
```

**Use case:** Organizations with data already in BigQuery that want to minimize data movement.

### 8.2 RAG with Vertex AI Search

```
User Query → App (Cloud Run / Compute Engine)
                │
        ┌───────▼────────┐
        │ Gemini API      │
        │ (Embedding)     │
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │ Vertex AI Search│
        │ (Hybrid Search) │
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │ Gemini API      │
        │ (GPT-4o class) │
        │ + Context       │
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │  Response with  │
        │  Citations      │
        └────────────────┘
```

**Components:**
1. Index documents in Vertex AI Search (from Cloud Storage, website, or BigQuery).
2. Accept user query — generate embedding using Gemini Embedding API.
3. Retrieve relevant documents from Vertex AI Search (hybrid search).
4. Construct prompt with retrieved context.
5. Generate response using Gemini (with Gemini's built-in grounding).
6. Return response with citations.

### 8.3 Multi-Region Global Inference

```
Google Cloud Load Balancing (Global)
         │
    ┌────┴────────────┐
    │                 │
us-central1      europe-west4
    │                 │
Vertex AI         Vertex AI
Endpoint          Endpoint
(Gemini Pro)      (Gemini Pro)
    │                 │
Cloud Storage    Cloud Storage
(us-central1)    (europe-west4)
```

**Architecture:**
- Deploy Vertex AI endpoints in multiple regions (us, eu, apac).
- Use Google Cloud External HTTP(S) Load Balancing for global traffic distribution.
- Route requests to the nearest region based on latency.
- Each region accesses regional Cloud Storage for data residency compliance.
- Global Cloud Spanner or Firestore for shared state.

---

## 9. Pricing & Cost Optimization

### Vertex AI Pricing Comparison

| Model | On-Demand Input (1M tokens) | On-Demand Output (1M tokens) | Notes |
|---|---|---|---|
| Gemini 2.0 Pro | $2.50 | $10.00 | Flagship model |
| Gemini 2.0 Flash | $0.35 | $1.40 | Cost-optimized |
| Gemini 1.5 Pro | $1.25 | $5.00 | Legacy, still available |
| Gemini 1.5 Flash | $0.075 | $0.30 | Cheapest Gemini |
| Claude 4 Opus (via Model Garden) | $15.00 | $75.00 | Same as AWS |
| Claude 3.5 Sonnet | $3.00 | $15.00 | Same as AWS |
| Llama 4 8B | $0.10 | $0.40 | Same as AWS |
| Imagen 3 | $0.050 per image | — | Image generation |

### Cost Optimization Strategies for GCP

1. **Use Gemini 2.0 Flash as default** — 95% of Pro quality at 15% of cost.
2. **Leverage TPUs for training** — better price/performance than GPUs for compatible workloads.
3. **Use Vertex AI Prediction with autoscaling to zero** — no cost when not used.
4. **BigQuery ML for simple models** — avoid moving data out of BigQuery.
5. **Committed use discounts** — 1-year or 3-year commitments for GPU/TPU (up to 70% discount).
6. **Batch prediction for bulk processing** — lower cost than real-time endpoints.
7. **Model Garden with provisioned throughput** — reduce per-token cost for consistent workloads.
8. **Use context caching** — cache common context tokens to reduce costs for repeated queries.
9. **Preemptible TPUs/GPUs** for training — 60–80% discount (reclaimable, best-effort).
10. **Gemini 1.5 Flash for high-volume RAG** — 40x cheaper than Pro for similar retrieval tasks.

---

## 10. Security & Compliance

**Network Security:**
- **VPC Service Controls (VPC-SC):** Prevent data exfiltration by controlling access to GCP services.
- **Private Google Access:** Access Google APIs over the internal network.
- **Cloud NEG / Private Service Connect:** Private access to Vertex AI endpoints.
- **Identity-Aware Proxy (IAP):** Control access to AI applications.

**Data Protection:**
- **Customer-Managed Encryption Keys (CMEK):** Encrypt model artifacts and data with your keys.
- **Customer-Supplied Encryption Keys (CSEK):** Provide your own encryption keys.
- **Data Loss Prevention (DLP):** Inspect and redact sensitive data.
- **Confidential VM:** Encrypt data in use for sensitive workloads.

**Access Control:**
- **Cloud IAM:** Granular roles for Vertex AI, Gemini, and other services.
- **Resource Manager:** Organization policies for AI service restrictions.
- **Service Accounts:** Identity for applications using AI services.
- **Workload Identity Federation:** Access GCP AI services from external identity providers.

**Compliance Certifications:**
- SOC 1/2/3
- HIPAA
- FedRAMP Moderate/High
- PCI DSS
- ISO 27001
- GDPR
- C5 (Germany)
- IRAP (Australia)

---

## 11. Comparison with AWS and Azure

| Dimension | GCP | AWS | Azure |
|---|---|---|---|
| **Flagship Model** | Gemini 2.0 Pro | Titan Premier | GPT-4o (Azure OpenAI) |
| **Context Window** | 1M–2M tokens | 200K (Claude on Bedrock) | 128K (GPT-4o) |
| **Custom Accelerators** | TPU v6 (Ironwood) | Trainium 2 | No (GPU only) |
| **Multi-model Access** | Model Garden (150+) | Bedrock (7 providers) | AI Foundry (200+) |
| **Data Warehouse ML** | BigQuery ML | Redshift ML | Synapse ML |
| **Document AI** | Document AI | Amazon Textract | Document Intelligence |
| **Conversational AI** | Dialogflow CX | Amazon Lex | Copilot Studio |
| **Speech** | Speech-to-Text/WaveNet | Polly/Transcribe | Azure Speech |
| **Search** | Vertex AI Search | Kendra | AI Search |
| **Pricing (Flagship)** | Gemini Pro: $2.50/$10.00 | Titan Premier: $2.50/$10.00 | GPT-4o: $2.50/$10.00 |
| **Enterprise Integration** | Google Workspace | AWS Ecosystem | Microsoft 365 |
| **Open-Source Models** | Best (Gemma, JAX, TensorFlow) | Good | Good |

---

## 12. References

1. Google Cloud. (2026). *Vertex AI Documentation*. https://cloud.google.com/vertex-ai/docs
2. Google Cloud. (2026). *Gemini API Documentation*. https://cloud.google.com/vertex-ai/generative-ai/docs
3. Google Cloud. (2026). *Model Garden Overview*. https://cloud.google.com/model-garden
4. Google Cloud. (2026). *TPU Documentation*. https://cloud.google.com/tpu/docs
5. Google Cloud. (2026). *BigQuery ML Documentation*. https://cloud.google.com/bigquery-ml/docs
6. Google Cloud. (2026). *Document AI Documentation*. https://cloud.google.com/document-ai/docs
7. Google Cloud. (2026). *Dialogflow Documentation*. https://cloud.google.com/dialogflow/docs
8. Google Cloud. (2026). *Vertex AI Pricing*. https://cloud.google.com/vertex-ai/pricing
9. Google Cloud. (2026). *Cloud Architecture Center — AI and ML*. https://cloud.google.com/architecture/ai-ml

---

> **Next:** [05 — Multi-Cloud AI Orchestration](05-Multi-Cloud-AI-Orchestration.md)

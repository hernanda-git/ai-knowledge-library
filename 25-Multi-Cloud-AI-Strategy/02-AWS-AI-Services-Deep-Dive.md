# 02 — AWS AI Services Deep Dive

> **AWS AI services: Amazon Bedrock (Claude, Llama, Mistral, Titan), SageMaker (training, hosting, pipelines), Amazon Q, Lex, Polly, Rekognition, Textract, Comprehend. Architecture patterns, pricing models, integration with AWS ecosystem, security best practices.**

---

## Table of Contents

1. [AWS AI/ML Service Landscape](#1-aws-aiml-service-landscape)
2. [Amazon Bedrock — Foundational Model Service](#2-amazon-bedrock--foundational-model-service)
   - [2.1 Supported Models](#21-supported-models)
   - [2.2 Bedrock Capabilities](#22-bedrock-capabilities)
   - [2.3 Pricing Models](#23-pricing-models)
   - [2.4 Security & VPC Integration](#24-security--vpc-integration)
   - [2.5 Bedrock Agents & Knowledge Bases](#25-bedrock-agents--knowledge-bases)
   - [2.6 Bedrock Guardrails](#26-bedrock-guardrails)
3. [Amazon SageMaker — End-to-End ML Platform](#3-amazon-sagemaker--end-to-end-ml-platform)
   - [3.1 SageMaker Studio & Canvas](#31-sagemaker-studio--canvas)
   - [3.2 Training: Managed, Distributed, Spot](#32-training-managed-distributed-spot)
   - [3.3 SageMaker Pipelines](#33-sagemaker-pipelines)
   - [3.4 Model Hosting & Inference](#34-model-hosting--inference)
   - [3.5 Model Registry & Governance](#35-model-registry--governance)
   - [3.6 Clarify, Debugger, and Monitor](#36-clarify-debugger-and-monitor)
4. [Amazon Q — AI Assistant](#4-amazon-q--ai-assistant)
5. [Specialized AI Services](#5-specialized-ai-services)
   - [5.1 Amazon Lex — Conversational AI](#51-amazon-lex--conversational-ai)
   - [5.2 Amazon Polly — Text-to-Speech](#52-amazon-polly--text-to-speech)
   - [5.3 Amazon Rekognition — Image & Video Analysis](#53-amazon-rekognition--image--video-analysis)
   - [5.4 Amazon Textract — Document Processing](#54-amazon-textract--document-processing)
   - [5.5 Amazon Comprehend — NLP](#55-amazon-comprehend--nlp)
   - [5.6 Amazon Translate & Transcribe](#56-amazon-translate--transcribe)
6. [Architecture Patterns on AWS](#6-architecture-patterns-on-aws)
   - [6.1 Bedrock + Lambda + S3 RAG Pattern](#61-bedrock--lambda--s3-rag-pattern)
   - [6.2 SageMaker + Bedrock Hybrid Pipeline](#62-sagemaker--bedrock-hybrid-pipeline)
   - [6.3 Real-Time Inference with Multi-Model Endpoints](#63-real-time-inference-with-multi-model-endpoints)
7. [Pricing Comparisons & Optimization](#7-pricing-comparisons--optimization)
8. [Security & Compliance Best Practices](#8-security--compliance-best-practices)
9. [Integration with AWS Ecosystem](#9-integration-with-aws-ecosystem)
10. [Comparison with Azure and GCP](#10-comparison-with-azure-and-gcp)
11. [References](#11-references)

---

## 1. AWS AI/ML Service Landscape

Amazon Web Services offers the broadest portfolio of AI and machine learning services among the three major cloud providers. The ecosystem can be categorized into three tiers:

**Tier 1 — Foundation Model Access:**
- **Amazon Bedrock:** Managed service for accessing and fine-tuning foundation models from multiple providers.
- **Amazon Q:** AI-powered assistant for developers and business users.

**Tier 2 — ML Platform:**
- **Amazon SageMaker:** Comprehensive platform covering the full ML lifecycle — data labeling, feature engineering, training, tuning, deployment, monitoring, and governance.

**Tier 3 — Specialized AI Services:**
- **Conversational AI:** Amazon Lex (chatbots, IVR).
- **Speech:** Amazon Polly (TTS), Amazon Transcribe (STT).
- **Vision:** Amazon Rekognition (image/video analysis), Amazon Lookout for Vision (industrial defect detection).
- **Document Processing:** Amazon Textract (OCR + document understanding), Amazon DocumentDB (for document workflows).
- **NLP:** Amazon Comprehend (entity extraction, sentiment, key phrases), Amazon Comprehend Medical (HIPAA-eligible).
- **Translation:** Amazon Translate (neural machine translation).
- **Search:** Amazon Kendra (enterprise search with AI).
- **Code:** Amazon CodeWhisperer (AI code generation, now part of Amazon Q Developer).

This deep dive focuses on the most strategically important services for multi-cloud AI architects: Bedrock (consuming foundation models), SageMaker (building custom models), and Amazon Q (assistants/copilots), with coverage of specialized services where they offer unique capabilities.

---

## 2. Amazon Bedrock — Foundational Model Service

Amazon Bedrock is AWS's fully managed service that provides access to foundation models from leading AI companies through a single API. It was launched in April 2023 and has rapidly become AWS's primary vehicle for generative AI workloads.

### 2.1 Supported Models

As of 2026, Amazon Bedrock offers models from the following providers:

**Amazon Titan Models (AWS proprietary):**
- **Titan Text Lite:** Fast, cost-effective text generation for simple tasks.
- **Titan Text Express:** Balanced text generation for general-purpose workloads.
- **Titan Text Premier:** Highest-quality text generation for complex reasoning.
- **Titan Embeddings G1 — Text:** For semantic search and RAG.
- **Titan Multimodal Embeddings:** Embeddings combining text and images.
- **Titan Image Generator:** Text-to-image generation.
- **Titan Image Generator v2:** Enhanced image editing and generation.

**Anthropic (Claude):**
- **Claude 4 (Opus, Sonnet, Haiku):** Latest generation — Opus for complex reasoning, Sonnet for balanced performance, Haiku for low-latency.
- **Claude 3.5 (Opus, Sonnet, Haiku):** Prior generation, still available for stability.
- **Claude 3 (Opus, Sonnet, Haiku):** Legacy tier for existing workflows.
- **Claude Instant:** Budget-friendly, fast responses.

**Meta (Llama):**
- **Llama 4 (405B, 70B, 8B):** Latest open-weight models with strong performance.
- **Llama 3.1 (405B, 70B, 8B):** Widely adopted, strong benchmarks.
- **Llama 3 (8B, 70B):** Legacy, still supported.
- **Llama 2:** Earlier generation, available for compatibility.

**Mistral AI:**
- **Mistral Large 2:** High-performance multilingual model.
- **Mistral Large:** Strong general-purpose model.
- **Mistral 7B:** Lightweight, fast inference.

**Cohere:**
- **Command R+:** Optimized for RAG and tool use.
- **Command R:** Balanced retrieval-augmented generation.
- **Embed v3:** High-quality multilingual embeddings.

**AI21 Labs:**
- **Jurassic-2 Ultra:** Large-scale text generation.
- **Jurassic-2 Mid:** Balanced performance.

**Stability AI:**
- **Stable Diffusion XL 1.0:** Image generation.
- **Stable Image Ultra/Ultra 2:** Enhanced image generation.

### 2.2 Bedrock Capabilities

**Inference:**
- InvokeModel / InvokeModelWithResponseStream for text, image, and multimodal models.
- Converse API: Unified interface for conversational patterns across supported models.
- Batch inference: Submit large volumes of prompts asynchronously at lower cost.

**Fine-tuning:**
- Custom model training using your data while AWS keeps the base model weights.
- Supports continued pre-training and instruction fine-tuning.
- Data stored in your S3 bucket, encrypted with your KMS key.
- Fine-tuned models are private to your account.

**RAG (Knowledge Bases):**
- Bedrock Knowledge Bases: Fully managed RAG — connect foundation models to your data sources.
- Automatic document ingestion, chunking, embedding, and index storage.
- Supports S3, Salesforce, Confluence, SharePoint, and web crawling.
- Choose vector store: Amazon OpenSearch Serverless, Aurora PostgreSQL, Pinecone, Redis Enterprise Cloud.
- Query decomposition and multi-step retrieval for complex questions.

**Agents:**
- Bedrock Agents: Build AI agents that reason, plan, and execute multi-step tasks.
- Define actions via Lambda functions or API schemas (OpenAPI 3.0).
- Agents can call multiple APIs, process results, and continue the conversation.
- Support for human-in-the-loop approval workflows.

**Guardrails:**
- Content filtering (hate, insults, harassment, sexual, violence, misconduct).
- Denied topics — block specific subjects from being discussed.
- PII redaction — automatically detect and redact personally identifiable information.
- Contextual grounding checks — verify model responses are grounded in source material.
- Word filters — block specific words or phrases.

**Model Evaluation:**
- Automated model evaluation with curated datasets.
- Human evaluation workflows for quality assessment.
- Comparison reports across models.

### 2.3 Pricing Models

Bedrock offers three pricing models:

**1. On-Demand:**
- Pay per token processed (input and output separately).
- No upfront commitment, no time-based commitment.
- Ideal for variable workloads, prototyping, low-volume production.
- Example pricing (approximate, 2026):
  - Claude 4 Opus: $15.00/1M input tokens, $75.00/1M output tokens.
  - Claude 4 Sonnet: $3.00/1M input tokens, $15.00/1M output tokens.
  - Claude 4 Haiku: $0.25/1M input tokens, $1.25/1M output tokens.
  - Llama 4 8B: $0.10/1M input tokens, $0.40/1M output tokens.
  - Titan Text Lite: $0.08/1M input tokens, $0.30/1M output tokens.

**2. Provisioned Throughput (PT):**
- Reserve model units for a 1-month or 6-month term.
- Each model unit provides a specific throughput (e.g., 1000 tokens/minute for Claude 4 Opus).
- 1-month commitment offers ~20% discount vs on-demand.
- 6-month commitment offers ~30–40% discount.
- Ideal for predictable, high-volume workloads.

**3. Batch Inference:**
- 50% discount vs on-demand for asynchronous batch processing.
- Results returned in S3.
- Slower turnaround (minutes to hours depending on volume).

**SageMaker Pricing (for custom models):**
- Training: Pay per instance-hour (compute, GPU, or Trainium instances).
- Hosting: Pay per instance-hour for real-time endpoints; per MB for serverless inference.
- Pipelines: Pay for the compute used in each step.
- Studio: No additional cost (pay for underlying compute).

### 2.4 Security & VPC Integration

Bedrock is designed for enterprise security requirements:

**Network Security:**
- **AWS PrivateLink:** Access Bedrock endpoints privately within your VPC — no traffic traverses the public internet.
- **VPC Endpoints:** Create interface endpoints for Bedrock in your VPC.
- **Proxy via Lambda:** For complete network isolation, route Bedrock calls through a Lambda function inside your VPC.

**Data Protection:**
- **Encryption at rest:** All data in Bedrock is encrypted with AWS KMS. Bring your own KMS keys (CMK).
- **Encryption in transit:** TLS 1.2+ for all API communications.
- **No training on your data:** AWS does not use your prompts or responses for model training (confirmed in customer agreement).
- **Data deletion:** Fine-tuned models and associated data are deleted when you delete the model.

**Access Control:**
- **IAM policies:** Granular control over which models users can access, what actions they can perform, and which resources they can use.
- **Service control policies (SCPs):** Organization-wide controls to restrict model access.
- **IAM roles:** Use cross-account roles for multi-account Bedrock access.
- **CloudTrail:** Full audit trail of all Bedrock API calls.

**Compliance Certifications:**
- SOC 1/2/3
- PCI DSS
- HIPAA (with BAA)
- FedRAMP High (in select regions)
- GDPR
- ISO 27001/27017/27018

### 2.5 Bedrock Agents & Knowledge Bases

**Knowledge Bases** enable RAG without managing infrastructure:

1. Set up a data source (S3 bucket, web crawler, or SaaS connector).
2. Bedrock automatically chunks documents, generates embeddings (using Titan Embeddings), and stores them in a vector database.
3. At query time, the knowledge base retrieves relevant chunks and passes them to the foundation model as context.
4. Results include citations and confidence scores.

**Agents** extend knowledge bases with action execution:

- Agents can break down complex requests into multi-step plans.
- Each step may involve: calling a knowledge base, invoking an API (via Lambda), or asking the user for clarification.
- Return of results — the agent presents a final response with supporting reasoning.

**Example: Customer Support Agent**

```
User: "I need to return my order #12345 and find a replacement product."

Agent Plan:
1. Query order system (Lambda → DynamoDB) to get order details.
2. Initiate return in order management system.
3. Retrieve product catalog (Knowledge Base from S3) to find similar products.
4. Present replacement options with links.
5. Ask user to confirm selection.
```

### 2.6 Bedrock Guardrails

Bedrock Guardrails provide safety controls that can be applied across all models:

**Content Filters:**
- Configure thresholds for each harmful category (hate, insults, harassment, sexual, violence, misconduct).
- Adjustable strength (low, medium, high) — higher thresholds block more but may reduce helpfulness.
- Separate configurations for user prompts and model responses.

**Denied Topics:**
- Define up to 50 topics that the model should never discuss.
- Example: "Investment advice" — block all model responses related to stock tips or financial recommendations.

**Sensitive Information Filters:**
- Detect and redact PII (SSN, credit card numbers, email addresses, etc.).
- Block or mask detected PII in model inputs and/or outputs.
- Custom regex patterns for domain-specific sensitive data.

**Contextual Grounding:**
- Verify that model responses are factually supported by the provided reference sources.
- Detect and flag hallucinations in RAG responses.
- Block responses that lack sufficient grounding in source material.

**Word Filters:**
- Deny specific words or phrases from appearing in model inputs or outputs.
- Useful for complying with brand guidelines or regulatory requirements.

---

## 3. Amazon SageMaker — End-to-End ML Platform

Amazon SageMaker is the most comprehensive ML platform in the cloud, covering the entire machine learning lifecycle.

### 3.1 SageMaker Studio & Canvas

**SageMaker Studio:**
- Web-based IDE for ML development.
- Integrated Jupyter notebooks with SageMaker kernels.
- Visual pipeline builder for drag-and-drop ML workflows.
- Built-in experimentation tracking and model lineage.
- Collaborative features for team-based ML development.

**SageMaker Canvas:**
- No-code ML for business analysts.
- Prepare data, build models, and generate predictions without writing code.
- Built-in data visualization and feature importance analysis.
- Ready-to-use models for common use cases (demand forecasting, churn prediction, etc.).

### 3.2 Training: Managed, Distributed, Spot

**Managed Training:**
- SageMaker automatically provisions and manages training infrastructure.
- Supports CPU, GPU (A100, H100, L40S), and AWS Trainium instances.
- Automatic model parallelism for large models.
- Checkpointing: Automatic saving and resuming of training jobs.

**Distributed Training:**
- SageMaker Distributed Data Parallel (DDP): For large datasets across multiple GPUs.
- SageMaker Distributed Model Parallel: For models too large to fit on a single GPU.
- Integration with PyTorch FSDP, DeepSpeed, and Hugging Face Accelerate.

**Managed Spot Training:**
- Use EC2 Spot instances for training at 60–70% discount.
- SageMaker automatically manages checkpointing and resumption.
- Best effort — training may be interrupted but will complete eventually.
- Ideal for hyperparameter tuning and non-critical training.

**Training Compiler:**
- SageMaker Training Compiler optimizes deep learning models for faster training.
- Can reduce training time by up to 50% for Transformer models.
- Supports PyTorch and TensorFlow.

**Hyperparameter Tuning (HPO):**
- Automatic hyperparameter optimization.
- Supports Bayesian, random, and grid search strategies.
- Early stopping to avoid wasting compute on poor trials.

### 3.3 SageMaker Pipelines

SageMaker Pipelines is a purpose-built CI/CD service for ML:

**Pipeline Steps:**
- Data processing (SageMaker Processing Jobs)
- Training (SageMaker Training Jobs)
- Evaluation (Processing Jobs or Lambda)
- Model registration (Model Registry)
- Deployment (SageMaker Endpoints or Batch Transform)

**Features:**
- **DAG-based workflow:** Define pipeline as a directed acyclic graph of steps.
- **Parameterization:** Parametrize pipelines for reuse across environments.
- **Caching:** Cache step outputs to avoid recomputing unchanged steps.
- **Conditional execution:** Branch pipeline execution based on model quality metrics.
- **Integration with CodePipeline/CodeCommit:** Trigger pipelines on code changes.

**Example Pipeline YAML:**

```yaml
Pipeline:
  Name: bert-text-classification
  Steps:
    - Processing:
        Name: PreprocessData
        Instance: ml.m5.xlarge
        Image: 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py38
    - Training:
        Name: TrainBertModel
        Instance: ml.p4d.24xlarge
        HyperParameters:
          learning_rate: 2e-5
          epochs: 3
        DependsOn: PreprocessData
    - Evaluation:
        Name: EvaluateModel
        Instance: ml.m5.xlarge
        DependsOn: TrainBertModel
    - Condition:
        Name: ConditionAccuracy
        Condition: accuracy > 0.85
        IfTrue: [RegisterModel, DeployModel]
        IfFalse: [NotifyFailure]
```

### 3.4 Model Hosting & Inference

SageMaker provides multiple options for model deployment:

**Real-Time Inference:**
- Persistent endpoints with auto-scaling.
- Support for A100, H100, Inferentia, Trainium, and GPU instances.
- Automatic scaling based on request latency or request count.
- Multi-model endpoints: Host multiple models on the same endpoint to reduce costs.
- Multi-container endpoints: Chain containers (e.g., preprocessor + model + postprocessor).

**Serverless Inference:**
- No infrastructure to manage — pay per request and compute duration.
- Automatic scaling from zero to thousands of requests.
- Ideal for intermittent workloads, prototyping, and variable traffic.
- Cold start latency (1–10 seconds) limits use for latency-sensitive applications.
- Maximum concurrency and memory configurable.

**Async Inference:**
- For large payloads and longer processing times.
- Requests are queued, processed asynchronously, and results stored in S3.
- Built-in auto-scaling and retry logic.
- Up to 1 GB payloads, 1 hour processing time.

**Batch Transform:**
- Process large batches of data offline.
- Distributed inference across multiple instances.
- Results written to S3.
- 50% lower cost than real-time inference.

**Inference Recommender:**
- Automatically recommends optimal instance type and configuration.
- Runs load tests against your model to find best cost/performance tradeoff.
- Integrates with SageMaker Model Registry.

### 3.5 Model Registry & Governance

SageMaker Model Registry provides model cataloging, versioning, and approval workflows:

**Model Groups:**
- Logical groupings for related model versions (e.g., "Customer Churn Model").
- Each version tracks: model artifacts, training metrics, lineage, and metadata.

**Approval Status:**
- Draft (in development).
- PendingReview (submitted for evaluation).
- Approved (passed review, ready for production).
- Rejected (failed evaluation).

**Model Lineage:**
- Automatic tracking of data sources, processing jobs, training jobs, and endpoints.
- Full audit trail for compliance and reproducibility.
- Visual lineage graph in SageMaker Studio.

**Model Cards:**
- Standardized documentation for each model version.
- Includes intended use, evaluation results, training data, ethical considerations.
- Exportable to PDF for regulatory submissions.

### 3.6 Clarify, Debugger, and Monitor

**SageMaker Clarify (Bias & Explainability):**
- Detect bias in training data and model predictions.
- SHAP-based feature importance for model explanations.
- Pre-training and post-training bias metrics.
- Integration with Model Registry for compliance reporting.

**SageMaker Debugger:**
- Real-time monitoring of training jobs.
- Capture gradients, weights, and activations during training.
- Built-in rules for detecting common training issues (overfitting, vanishing gradients, etc.).
- Custom rules for domain-specific monitoring.

**SageMaker Model Monitor:**
- Continuous monitoring of deployed models.
- Detect data drift, feature drift, and model quality degradation.
- Automatically trigger retraining pipelines when drift is detected.
- Scheduled monitoring with configurable cadence.

---

## 4. Amazon Q — AI Assistant

Amazon Q is AWS's AI-powered assistant, available in two editions:

**Amazon Q Developer:**
- AI coding assistant integrated in IDE (VS Code, JetBrains, AWS Cloud9).
- Code generation, completion, debugging, and refactoring.
- Understands AWS services and best practices.
- Code transformation (e.g., Java 8 → Java 17) with automatic testing.

**Amazon Q Business:**
- Enterprise AI assistant connected to business data.
- Connect to 40+ data sources (S3, SharePoint, Salesforce, Confluence, etc.).
- Answer questions, generate summaries, create content.
- Role-based access — users only see data they have permission to.
- Citation with source links in every response.

**Amazon Q in QuickSight:**
- Natural language querying of business data.
- Generate dashboards and visualizations from natural language.
- Automatic chart selection based on query intent.

---

## 5. Specialized AI Services

### 5.1 Amazon Lex — Conversational AI

Amazon Lex builds conversational interfaces using automatic speech recognition (ASR) and natural language understanding (NLU):

**Capabilities:**
- Chatbots for web, mobile, and messaging platforms.
- IVR systems for contact centers (Amazon Connect integration).
- Multi-language support (EN, ES, FR, DE, IT, JP, PT, etc.).
- Intent classification and slot filling.
- Lambda integration for business logic.
- Integration with Amazon Kendra for knowledge base Q&A.

**Pricing:**
- Pay per text request or per speech request.
- $0.004 per text request; $0.0065 per speech request (standard tiers).
- Free tier: 10,000 text requests and 5,000 speech requests per month.

### 5.2 Amazon Polly — Text-to-Speech

Amazon Polly converts text into realistic speech:

**Capabilities:**
- 100+ voices across 30+ languages.
- Neural TTS (NTTS) for natural-sounding speech.
- Generative voices (newscaster, conversational, expressive styles).
- SSML support for fine-grained control (emphasis, breathing, pacing).
- Speech Marks for word-level timing (useful for lip-sync).

**Pricing:**
- $4.00 per 1M characters for standard voices.
- $16.00 per 1M characters for neural voices.
- Generative voices: $100.00 per 1M characters (premium tier).

### 5.3 Amazon Rekognition — Image & Video Analysis

**Image Analysis:**
- Object and scene detection (1000+ labels).
- Facial analysis (age range, emotions, glasses, etc.).
- Content moderation (explicit and suggestive content).
- Text detection in images.
- Celebrity recognition.
- Image property detection (quality, dominant colors).

**Video Analysis:**
- Real-time video streaming analysis (Kinesis Video Streams).
- Stored video analysis (segmentation, label detection, activity recognition).
- People tracking across video frames.
- Technical features: Custom labels, consistency checks, adapters.

**Pricing:**
- Image: $0.0010 per image (first 1M/month); $0.0008 per image thereafter.
- Video: $0.05 per minute of video processed.
- Content moderation: $0.0010 per image (first 1M); $0.0008 thereafter.

### 5.4 Amazon Textract — Document Processing

Amazon Textract extracts text, handwriting, and data from documents:

**Capabilities:**
- OCR (printed and handwritten text).
- Form extraction (key-value pairs).
- Table extraction with cell-level details.
- Document analysis: Layout, signatures, selection elements.
- Expense analysis (invoices, receipts).
- Lending document analysis (loan applications).
- ID document extraction (driver's licenses, passports).

**Integration:**
- Asynchronous processing for batch documents (S3 input and output).
- Synchronous processing for single pages (API).
- Comprehend integration for NLP on extracted text.
- A2I (Augmented AI) for human review of low-confidence extractions.

**Pricing:**
- $0.0015 per page (first 1M pages/month) for text detection.
- $0.0025 per page for form/table extraction.
- $0.0035 per expense document.
- Volume discounts available through Reserved Capacity.

### 5.5 Amazon Comprehend — NLP

Amazon Comprehend extracts insights from text:

**Capabilities:**
- Entity recognition (people, places, dates, organizations).
- Key phrase extraction.
- Sentiment analysis (positive, negative, neutral, mixed).
- Language detection.
- Custom entity recognition (train on domain-specific entities).
- Document classification (custom or pre-built categories).
- Targeted sentiment (sentiment toward specific entities).

**Amazon Comprehend Medical:**
- HIPAA-eligible NLP for medical text.
- Extract medical conditions, medications, dosages, test results.
- ICD-10-CM and RxNorm ontology mapping.
- Protected health information (PHI) detection.

**Pricing:**
- $0.0001 per unit (100 characters) for entities, key phrases, language detection.
- $0.0002 per unit for syntax analysis.
- Custom model training: $1.00 per hour (per instance).
- Custom model inference: $0.0005 per unit.

### 5.6 Amazon Translate & Transcribe

**Amazon Translate:**
- Neural machine translation between 75+ languages.
- Real-time API and batch translation.
- Terminology customization (glossary for domain-specific terms).
- Active custom translation (parallel data training).
- Formality settings for different use cases.
- Pricing: $15.00 per 1M characters (standard); $60.00 per 1M (active custom translation).

**Amazon Transcribe:**
- Automatic speech recognition (speech-to-text).
- Real-time streaming and batch transcription.
- Speaker diarization (who spoke when).
- Custom vocabulary and language models.
- Content filtering (redact or mask sensitive content).
- Multi-language transcription.
- Medical transcription (Transcribe Medical, HIPAA-eligible).
- Pricing: $0.024 per minute (real-time); $0.015 per minute (batch).

---

## 6. Architecture Patterns on AWS

### 6.1 Bedrock + Lambda + S3 RAG Pattern

```
User → API Gateway → Lambda (query processing)
                         │
                    ┌────▼─────┐
                    │ Bedrock  │
                    │ (LLM +   │
                    │  KB)     │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │ S3 +     │
                    │ OpenSearch│
                    │ (Vector  │
                    │  Store)  │
                    └──────────┘
```

**Implementation steps:**
1. Store documents in S3 with appropriate lifecycle policies.
2. Create a Bedrock Knowledge Base pointing to the S3 bucket.
3. Use Titan Embeddings for chunking and vectorization.
4. Store vectors in Amazon OpenSearch Serverless.
5. Build a Lambda function that receives user queries, invokes Bedrock with KB retrieval, and returns responses.
6. Expose via API Gateway with IAM auth or Cognito user pools.
7. Add Bedrock Guardrails for content filtering.

### 6.2 SageMaker + Bedrock Hybrid Pipeline

```
Raw Data → SageMaker Processing → Feature Store
                                         │
                              SageMaker Training (custom model)
                                         │
                              Model Registry
                                         │
                         ┌───────────────┴───────────────┐
                         │                               │
               SageMaker Endpoint                 Bedrock (FM)
               (custom classifier)              (LLM for reasoning)
                         │                               │
                         └───────────────┬───────────────┘
                                         │
                                   Application
```

This architecture is common for production ML systems where:
- A custom classification/regression model (trained in SageMaker) handles structured prediction.
- A foundation model (via Bedrock) handles text generation, summarization, or reasoning.
- The two models are orchestrated by an application layer.

### 6.3 Real-Time Inference with Multi-Model Endpoints

```
API Gateway → Application Load Balancer → SageMaker Multi-Model Endpoint
                                          │
                                   ├── Model A (classification)
                                   ├── Model B (NER)
                                   ├── Model C (summarization)
                                   └── Model D (language detection)
```

Multi-Model Endpoints (MME) reduce hosting costs by sharing infrastructure across models:
- Models are stored in S3 and loaded on demand.
- Auto-scaling based on aggregate endpoint load.
- Up to 50% cost savings vs. separate endpoints per model.
- Best for models with spiky or unpredictable usage patterns.

---

## 7. Pricing Comparisons & Optimization

### Bedrock vs. Equivalent Services

| Model | AWS Bedrock (On-Demand) | Azure OpenAI | GCP Vertex AI |
|---|---|---|---|
| GPT-4o class | Titan Premier: $2.50/$10.00 per 1M tokens | $2.50/$10.00 per 1M tokens | Gemini 1.5 Pro: $1.25/$5.00 |
| Claude 4 Opus | $15.00/$75.00 | N/A (only via AWS) | $15.00/$75.00 |
| Llama 4 8B | $0.10/$0.40 | N/A | $0.10/$0.40 |
| Embeddings | Titan: $0.02/1K | text-embedding-3: $0.13/1M | text-embedding-004: $0.05/1M |

### Cost Optimization Strategies for AWS AI

1. **Use Provisioned Throughput** for predictable workloads → 20–40% discount.
2. **Leverage Spot Instances** for SageMaker training → 60–70% discount.
3. **Batch inference** for bulk processing → 50% discount vs. real-time.
4. **Multi-Model Endpoints** for hosting many small models.
5. **Inference Recommender** to select optimal instance types.
6. **SageMaker Compiler** to reduce training time.
7. **SageMaker Pipelines caching** to avoid redundant computations.
8. **Lifecycle policies** on S3 training data to reduce storage costs.
9. **SageMaker Savings Plans** for consistent compute usage (up to 44% savings).
10. **Bedrock model tiering** — use smaller models for simple queries.

---

## 8. Security & Compliance Best Practices

1. **Use AWS PrivateLink** for all Bedrock and SageMaker API calls — never traverse the public internet.
2. **Encrypt everything** with KMS CMK — training data, model artifacts, and inference results.
3. **Implement least-privilege IAM policies** — restrict model access, data access, and API actions.
4. **Use VPC endpoints** for S3 and DynamoDB to keep traffic within AWS network.
5. **Enable CloudTrail** for audit logging of all AI service API calls.
6. **Apply Bedrock Guardrails** to all production model endpoints.
7. **Configure SCPs** at the AWS Organization level to enforce model access policies.
8. **Use SageMaker Model Cards** for compliance documentation.
9. **Regularly rotate IAM keys** and use IAM roles instead of long-term credentials.
10. **Implement resource tagging** for cost allocation and security group management.
11. **Enable VPC Flow Logs** for network traffic analysis.
12. **Use AWS Config rules** to monitor AI service configuration compliance.
13. **Set up AWS Budgets and alerts** for AI cost monitoring.
14. **Conduct regular IAM Access Analyzer reviews** for AI service permissions.

---

## 9. Integration with AWS Ecosystem

Bedrock and SageMaker integrate deeply with the broader AWS ecosystem:

| Service | Integration |
|---|---|
| **Lambda** | Invoke Bedrock from Lambda for serverless inference |
| **S3** | Store training data, model artifacts, KB documents, batch results |
| **API Gateway** | Expose Bedrock/SageMaker endpoints as REST or WebSocket APIs |
| **CloudWatch** | Logs, metrics, and alarms for inference and training jobs |
| **KMS** | Encryption keys for data, models, and fine-tuning |
| **VPC** | Private networking for all AI services |
| **IAM** | Fine-grained access control |
| **CloudTrail** | Audit logging of all API calls |
| **Step Functions** | Orchestrate multi-step AI workflows |
| **EventBridge** | Event-driven ML pipelines (e.g., trigger training on new data) |
| **Cognito** | User authentication for AI applications |
| **OpenSearch** | Vector store for Bedrock Knowledge Bases |
| **DynamoDB** | Session state, chat history, metadata storage |
| **ECS/EKS** | Run custom model containers (SageMaker or DIY) |
| **CodePipeline** | CI/CD for ML pipelines |
| **Systems Manager** | Parameter store for model configurations |

---

## 10. Comparison with Azure and GCP

| Dimension | AWS | Azure | GCP |
|---|---|---|---|
| **Model Breadth** | Widest (7+ providers) | Strong (OpenAI focus) | Strong (150+ in Model Garden) |
| **Custom Training** | Best-in-class (SageMaker) | Good (AI Foundry) | Good (Vertex AI Training) |
| **OpenAI Access** | No direct | Yes (Azure OpenAI Service) | Yes (via Vertex AI) |
| **TPU Access** | No | No | Yes |
| **Enterprise Integration** | AWS ecosystem | Microsoft 365, Dynamics | Google Workspace, BigQuery |
| **RAG (Managed)** | Bedrock Knowledge Bases | Azure AI Search | Vertex AI Search |
| **Model Evaluation** | Bedrock automated eval | AI Foundry evaluation | Vertex AI evaluation |
| **Security Features** | VPC + PrivateLink + KMS | VNet + Private Endpoint + MI | VPC + Private Service Connect |
| **Cost for GPT-4o class** | ~$2.50/$10.00 per 1M tokens | ~$2.50/$10.00 per 1M tokens | ~$1.25/$5.00 per 1M tokens (Gemini) |
| **Serverless Inference** | SageMaker Serverless | Mosaic AI (via Databricks) | Vertex AI endpoints |

---

## 11. References

1. AWS. (2026). *Amazon Bedrock User Guide*. https://docs.aws.amazon.com/bedrock/latest/userguide/
2. AWS. (2026). *Amazon SageMaker Developer Guide*. https://docs.aws.amazon.com/sagemaker/latest/dg/
3. AWS. (2026). *Amazon Bedrock Pricing*. https://aws.amazon.com/bedrock/pricing/
4. AWS. (2026). *Amazon SageMaker Pricing*. https://aws.amazon.com/sagemaker/pricing/
5. AWS. (2026). *AWS Well-Architected Framework — Machine Learning Lens*.
6. AWS. (2026). *Security Best Practices for Amazon Bedrock*.
7. AWS. (2026). *Amazon Q Developer Guide*. https://docs.aws.amazon.com/amazonq/latest/

---

> **Next:** [03 — Azure AI Services Deep Dive](03-Azure-AI-Services-Deep-Dive.md)

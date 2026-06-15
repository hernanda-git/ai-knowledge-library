# 02 — odysseus AI Workspace

## Overview

**odysseus** is a self-hosted AI workspace that has become the defining platform of the local AI revolution. With over 71,000 GitHub stars as of June 2026, it is the fastest-growing open-source AI project in the world — a comprehensive platform that combines model serving, agent orchestration, RAG pipelines, tool use, collaboration, and extensibility into a single coherent system.

Unlike single-purpose tools (Ollama for model serving, LangChain for agents, ChromaDB for vectors), odysseus integrates all of these capabilities into a unified workspace with a consistent API and user interface. It is designed to be the operating system for local AI — the layer on which all AI-powered workflows run.

---

## What Is odysseus? A Detailed Definition

At its core, odysseus is a **self-hosted AI orchestration platform**. It provides:

1. **A model serving layer** that can run multiple models simultaneously across different backends (llama.cpp, vLLM, exllama, transformers)
2. **An agent execution engine** that manages autonomous AI agents with tool-use capabilities
3. **A RAG pipeline** that indexes local documents and provides retrieval-augmented generation
4. **A plugin system** that allows extending functionality with community-contributed modules
5. **A multi-user workspace** with authentication, permissions, and collaborative features
6. **An observability stack** with logging, metrics, tracing, and debugging tools
7. **A configuration management system** for managing models, agents, and workflows declaratively

### What odysseus Is Not

- It is not a model format — it consumes models from HuggingFace, Ollama, and local files
- It is not a training framework — fine-tuning is handled by integrations with Axolotl/Unsloth
- It is not a vector database — it integrates with ChromaDB, LanceDB, Qdrant, and others
- It is not a chat application — though it includes chat interfaces, it is far more comprehensive

---

## Why odysseus Matters: The 71K★ Story

### The Gap odysseus Filled

Before odysseus, running local AI required assembling a stack of disconnected tools:

- One tool to download and run models (Ollama or llama.cpp)
- Another to create a chat interface (Open WebUI, SillyTavern)
- Another for agents (LangChain, AutoGen)
- Another for RAG (ChromaDB, LangChain)
- Another for tool use (MCP servers, function calling wrappers)
- Custom integration code to wire everything together

This stack was fragile, hard to maintain, and required significant expertise to set up and operate. odysseus replaced this fragmented ecosystem with a single, integrated platform.

### Growth Trajectory

| Date | GitHub Stars | Event |
|---|---|---|
| Jan 2025 | 5,000 | Initial public release |
| Mar 2025 | 10,000 | v1.0 release with basic agent support |
| May 2025 | 20,000 | RAG pipeline integration |
| Aug 2025 | 35,000 | Plugin system and MCP support |
| Nov 2025 | 50,000 | Multi-user workspace, enterprise features |
| Jan 2026 | 60,000 | v2.0 with agentic workflows |
| Jun 2026 | 71,000 | Mainstream adoption |

### Key Factors in Its Success

1. **Integrated experience**: Everything works together out of the box
2. **Active development**: Multiple releases per week with new features
3. **Strong community**: 100K+ Discord members, 2,000+ contributors
4. **Excellent documentation**: Comprehensive guides, examples, and API reference
5. **Plugin ecosystem**: 500+ community plugins for every use case
6. **Enterprise readiness**: Authentication, permissions, audit logging, SSO

---

## Architecture

odysseus follows a modular, service-oriented architecture. The system is composed of several interconnected components, each of which can be scaled independently.

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    odysseus Workspace                         │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Model    │  │ Agent    │  │ RAG      │  │ Plugin   │   │
│  │ Serving  │  │ Engine   │  │ Pipeline │  │ System   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Workflow │  │ Auth/    │  │ API      │  │ UI       │   │
│  │ Engine   │  │ RBAC     │  │ Gateway  │  │ Server   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
├─────────────────────────────────────────────────────────────┤
│                    Configuration (YAML)                       │
└─────────────────────────────────────────────────────────────┘
```

#### 1. Model Serving Layer

The model serving layer is responsible for loading, running, and serving inference requests for language models. It supports multiple backends:

- **llama.cpp backend**: For GGUF models. Supports CPU, CUDA, Metal, Vulkan, and ROCm.
- **vLLM backend**: For high-throughput serving of multiple concurrent requests. Supports PagedAttention for efficient memory management.
- **ExLlamaV2 backend**: For GPTQ and EXL2 quantized models. Optimized for NVIDIA GPUs.
- **Transformers backend**: For HuggingFace models in PyTorch format. Supports all model architectures.
- **Ollama backend**: Delegates model management to Ollama, using its API for inference.

Models can be loaded from:
- HuggingFace Hub (automatic download and caching)
- Local GGUF files
- Ollama library
- Custom model repositories
- S3-compatible object storage

The model serving layer handles:
- Model loading and unloading (with LRU caching)
- Request queuing and batching
- Context window management
- Token streaming via SSE
- Quantization configuration
- GPU memory management

#### 2. Agent Execution Engine

The agent engine manages autonomous AI agents — LLM-powered systems that can plan, use tools, and execute multi-step tasks.

**Agent capabilities:**
- **Planning**: Chain-of-thought, ReAct, tree-of-thought, and custom planning strategies
- **Tool use**: Function calling with automatic tool selection and execution
- **Memory**: Episodic, semantic, and procedural memory for persistent context
- **Multi-step execution**: Decomposing complex tasks into sub-tasks
- **Error recovery**: Retry, fallback, and graceful degradation on failure

**Supported agent frameworks:**
- **odysseus native agents**: Built-in agent system with declarative YAML configuration
- **LangChain agents**: Execute LangChain agent definitions within odysseus
- **AutoGen agents**: Multi-agent conversations powered by Microsoft's AutoGen
- **CrewAI agents**: Role-based agent teams for complex workflows
- **Custom agents**: Python-based agent implementations using the odysseus SDK

**Example agent configuration (YAML):**

```yaml
name: research-agent
model: qwen2.5-32b-q4_k_m
system_prompt: >
  You are a research assistant. You search the web,
  retrieve relevant documents, and synthesize findings.
tools:
  - web_search
  - document_retrieval
  - code_execution
  - file_write
memory:
  type: episodic
  limit: 50
max_steps: 20
error_behavior: retry
```

#### 3. RAG Pipeline

The RAG pipeline provides retrieval-augmented generation using local documents. It handles the full lifecycle:

**Document ingestion:**
- File format support: PDF, DOCX, TXT, MD, HTML, JSON, CSV, EPUB, and more
- Automatic text extraction and normalization
- Document chunking with configurable strategies (fixed size, recursive, semantic)
- Metadata extraction (document title, author, date, source)
- OCR for scanned documents (via Tesseract integration)

**Embedding:**
- Support for all major local embedding models (BGE, E5, Jina, Nomic, Instructor)
- Embedding caching to avoid recomputation
- Batch processing for efficiency
- GPU acceleration for embedding generation

**Vector storage:**
- ChromaDB integration (default, persists to local disk)
- LanceDB integration (columnar storage for large datasets)
- Qdrant integration (local mode for production deployments)
- Milvus integration for scale

**Retrieval:**
- Dense retrieval (semantic similarity)
- Sparse retrieval (BM25 keyword matching)
- Hybrid retrieval (combined dense + sparse)
- Re-ranking using cross-encoder models
- Contextual compression and extraction

#### 4. Plugin System

The plugin system allows extending odysseus with custom functionality. Plugins can add:

- New tool types for agents
- New model backends
- New embedding providers
- New vector store integrations
- UI components
- Authentication providers
- Webhook triggers
- Custom workflows

**Plugin architecture:**
- Plugins are Python packages installed via pip
- They register themselves using odysseus's plugin API
- Configuration is managed through the odysseus config system
- Plugins can expose their own API endpoints
- Community plugin repository with 500+ available plugins

**Example plugin categories:**

| Category | Example Plugins |
|---|---|
| Tools | web_search, code_executor, file_operations, database_query, email_send |
| Models | exllama_backend, mlx_backend, onnx_backend |
| RAG | ocr_processor, image_embedding, pdf_extractor |
| Auth | ldap_auth, oauth_provider, saml_auth |
| UI | dashboard_widgets, custom_theme, workflow_editor |
| Monitoring | prometheus_metrics, datadog_integration, sentry_logging |

#### 5. Auth / RBAC

Multi-user support with role-based access control:

**Roles:**
- **Admin**: Full system access, user management, configuration
- **Operator**: Manage models, agents, and workflows
- **Developer**: Create and modify agents, access logs
- **Viewer**: Read-only access to dashboards and interfaces
- **Custom roles**: Define arbitrary role hierarchies

**Authentication methods:**
- Local username/password with bcrypt hashing
- OAuth 2.0 / OpenID Connect (Google, GitHub, Microsoft, etc.)
- LDAP / Active Directory
- SAML 2.0
- API keys for programmatic access

**Permissions:**
- Per-model access control
- Per-agent execution permissions
- Per-document collection access
- API rate limiting per user
- Audit logging of all actions

#### 6. API Gateway

odysseus exposes a comprehensive REST API and WebSocket interface:

- **OpenAI-compatible API**: Drop-in replacement for OpenAI client libraries (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`)
- **odysseus API**: Native endpoints for workspace management
- **Agent API**: Create, execute, and monitor agents
- **RAG API**: Ingest documents, query collections, manage indices
- **Admin API**: System management, monitoring, configuration
- **WebSocket API**: Real-time streaming of agent execution, model inference, and system events

#### 7. UI Server

odysseus includes a web-based user interface built with React/TypeScript:

- **Chat interface**: Multi-model chat with streaming, conversation management, and document upload
- **Agent dashboard**: Create, monitor, and debug agents with real-time execution traces
- **Model manager**: Browse, download, and manage models
- **RAG console**: Upload documents, manage collections, test retrieval
- **Plugin marketplace**: Browse, install, and configure plugins
- **Admin panel**: User management, system monitoring, configuration
- **Workflow editor**: Visual editor for creating agent workflows

#### 8. Workflow Engine

The workflow engine enables declarative definition and execution of complex AI workflows:

**Workflow components:**
- **Nodes**: Individual steps (model calls, tool executions, data transformations)
- **Edges**: Data flow between nodes with conditional routing
- **State**: Shared state across workflow execution
- **Triggers**: Schedule-based, event-based, or manual execution

Workflows are defined in YAML:

```yaml
name: document-summary-pipeline
triggers:
  - type: webhook
    path: /webhook/new-document
steps:
  - id: extract
    type: document_processor
    input: $trigger.document
    params:
      extract_text: true
  
  - id: summarize
    type: model_inference
    model: qwen2.5-72b-q4_k_m
    system_prompt: "Summarize the following document:"
    input: $steps.extract.text
    
  - id: store
    type: vector_store
    collection: summaries
    document: $steps.summarize.output
```

---

## Installation

### System Requirements

**Minimum (personal use, 7B models):**
- 8GB RAM (16GB recommended)
- 4 CPU cores
- NVIDIA GPU with 8GB VRAM (or Apple Silicon with 16GB unified memory)
- 50GB free disk space
- Docker or bare-metal Linux

**Recommended (production, multiple models):**
- 32GB RAM
- 8 CPU cores
- NVIDIA GPU with 24GB VRAM (or Apple Silicon with 64GB+ unified memory)
- 200GB+ SSD
- Docker Compose or Kubernetes

**Enterprise (large-scale deployment):**
- 64GB+ RAM
- 16+ CPU cores
- Multiple GPUs (2× RTX 4090 / 4× RTX 6000 / Apple M3 Ultra)
- 1TB+ NVMe storage
- Kubernetes cluster

### Installation Methods

#### Method 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/odysseus/odysseus.git
cd odysseus

# Create configuration (interactive)
cp config.example.yaml config.yaml
# Edit config.yaml with your preferences

# Start with Docker Compose
docker compose up -d

# Access the UI at http://localhost:8080
```

#### Method 2: Bare Metal (Linux)

```bash
# Prerequisites
curl -fsSL https://ollama.com/install.sh | sh  # Optional, for Ollama backend

# Install odysseus
python3 -m venv odysseus-env
source odysseus-env/bin/activate
pip install odysseus

# Initialize configuration
odysseus init
# Edit ~/.config/odysseus/config.yaml

# Start the server
odysseus start
```

#### Method 3: Docker Compose (Full Stack)

```yaml
version: '3.8'
services:
  odysseus:
    image: odysseus/odysseus:latest
    ports:
      - "8080:8080"
    volumes:
      - ./config:/etc/odysseus
      - ./data:/var/lib/odysseus
      - ./models:/models
    environment:
      - OLLAMA_HOST=http://ollama:11434
    depends_on:
      - chromadb
      - ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  chromadb:
    image: chromadb/chroma:latest
    volumes:
      - ./chroma:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ./ollama:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

#### Method 4: Kubernetes (Helm)

```bash
helm repo add odysseus https://helm.odysseus.ai
helm install my-odysseus odysseus/odysseus \
  --set gpu.enabled=true \
  --set models.include={qwen2.5-7b,llama3.2-8b} \
  --set persistence.size=500Gi
```

### Post-Installation Setup

1. **Access the UI:** Navigate to `http://localhost:8080`
2. **Create an admin account:** First-run wizard guides you through account creation
3. **Download a model:** Use the model manager UI or CLI
4. **Configure RAG:** Set up document collections in the RAG console
5. **Create an agent:** Use the agent dashboard or YAML configuration
6. **Install plugins:** Browse the plugin marketplace

---

## Configuration

odysseus uses a YAML-based configuration system. The main configuration file (`config.yaml`) controls all aspects of the system.

### Basic Configuration

```yaml
server:
  host: 0.0.0.0
  port: 8080
  workers: 4
  log_level: info

models:
  backends:
    - type: llama
      path: /models
      gpu_layers: 35
  
  auto_download: true
  cache_size: 50GB

agents:
  default_model: qwen2.5-32b-q4_k_m
  max_concurrent: 10
  max_steps: 50

rag:
  default_collection: documents
  embedding_model: BAAI/bge-large-en-v1.5
  chunk_size: 512
  chunk_overlap: 64
  vector_store: chroma

auth:
  enabled: true
  session_ttl: 24h
  providers:
    - type: local
    - type: oauth
      provider: github

plugins:
  auto_install: false
  allowlist: []
```

### Advanced Configuration

```yaml
models:
  backends:
    - type: vllm
      models:
        - name: qwen2.5-72b
          quantization: awq
          max_model_len: 32768
          gpu_memory_utilization: 0.9
          tensor_parallel_size: 2
  
  scheduling:
    strategy: least_loaded
    preemption: true
    model_loading: on_demand

agents:
  frameworks:
    - native
    - langchain
    - autogen
  
  sandboxing:
    enabled: true
    type: docker
    image: odysseus/agent-sandbox:latest

rag:
  embedding:
    model: nomic-embed-text-v1.5
    batch_size: 32
    normalize: true
  
  chunking:
    strategy: recursive
    separators: ["\n\n", "\n", ".", " "]
  
  retrieval:
    top_k: 10
    score_threshold: 0.5
    reranker: BAAI/bge-reranker-v2-m3

monitoring:
  metrics:
    - type: prometheus
      port: 9090
  logging:
    - type: file
      path: /var/log/odysseus
      retention: 30d
    - type: elasticsearch
      host: elasticsearch:9200
  tracing:
    enabled: true
    exporter: otlp
```

---

## Popular Use Cases

### 1. Personal AI Assistant

Run a local AI assistant that knows your documents, email, and calendar — with zero data leaving your machine.

```yaml
name: personal-assistant
model: llama3.2-8b-q4_k_m
tools:
  - file_search
  - document_qa
  - web_search
  - calendar
  - email
memory:
  type: episodic
  limit: 200
workspace:
  rag_collections:
    - personal_docs
    - emails
```

### 2. Code Development Assistant

Power an AI coding assistant with local models for completions, explanations, code review, and refactoring.

```yaml
name: coding-assistant
model: deepseek-coder-33b-q4_k_m
tools:
  - code_completion
  - code_explain
  - code_review
  - code_refactor
  - git_operations
rag_collections:
  - codebase
  - documentation
```

### 3. Document Research and Analysis

Index a library of documents and perform sophisticated analysis, summarization, and question-answering.

```yaml
name: research-assistant
model: qwen2.5-72b-q3_k_m
tools:
  - document_qa
  - summarize
  - cross_reference
  - extract_entities
  - generate_report
rag_collections:
  - research_library
  - papers
  - notes
planning: tree_of_thought
```

### 4. Customer Support Bot

Deploy a self-hosted customer support agent that knows your product documentation and can handle common inquiries.

```yaml
name: support-bot
model: qwen2.5-14b-q4_k_m
system_prompt: >
  You are a helpful customer support agent for [Company].
  Use the documentation to answer questions accurately.
  Escalate to human agents when you cannot resolve an issue.
tools:
  - document_qa
  - ticket_create
  - ticket_update
rag_collections:
  - product_docs
  - faq
  - known_issues
max_steps: 10
```

### 5. Data Pipeline Agent

Automate data processing tasks: extract, transform, analyze, and report.

```yaml
name: data-pipeline
model: qwen2.5-32b-q4_k_m
tools:
  - csv_reader
  - json_processor
  - sql_query
  - data_visualize
  - report_generate
  - file_upload
triggers:
  - type: schedule
    cron: "0 6 * * *"
  - type: webhook
    path: /webhook/new-data
```

---

## Comparison with Other Local AI Platforms

| Feature | odysseus | Open WebUI | LocalAI | LM Studio |
|---|---|---|---|---|
| **GitHub Stars** | 71K★ | 40K★ | 25K★ | — (proprietary) |
| **Model Backends** | llama, vLLM, ExLlama, Transformers, Ollama | Ollama | llama, vLLM, transformers | llama.cpp |
| **Multi-Model Serving** | ✅ Yes | ✅ Yes (via Ollama) | ✅ Yes | ❌ Single model |
| **Agent Engine** | ✅ Built-in | ❌ No | ❌ No | ❌ No |
| **RAG Pipeline** | ✅ Built-in | ✅ Via plugins | ✅ Basic | ❌ No |
| **Plugin System** | ✅ 500+ plugins | ✅ 100+ plugins | ✅ Limited | ❌ No |
| **Multi-User** | ✅ Full RBAC | ✅ Basic auth | ❌ Single user | ❌ Single user |
| **Workflows** | ✅ Declarative YAML | ❌ No | ❌ No | ❌ No |
| **API Compatibility** | ✅ OpenAI full | ✅ Chat only | ✅ OpenAI full | ✅ Chat only |
| **GPU Support** | ✅ CUDA/Metal/ROCm | ✅ Via Ollama | ✅ CUDA | ✅ CUDA/Metal |
| **Easy to Deploy** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Extensibility** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Production Ready** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

---

## Extending odysseus

### Creating a Plugin

Plugins are Python packages that implement the odysseus plugin interface:

```python
# my_plugin.py
from odysseus.plugin import BasePlugin, tool

class MyCustomTool(BasePlugin):
    name = "my_custom_tool"
    version = "1.0.0"
    
    @tool
    def analyze_text(self, text: str) -> dict:
        """Analyze text and return statistics."""
        return {
            "word_count": len(text.split()),
            "char_count": len(text),
            "sentences": len(text.split(".")),
        }
```

### Custom Model Backend

```python
from odysseus.model import BaseBackend

class MyBackend(BaseBackend):
    name = "my_backend"
    
    def load(self, model_path: str, **kwargs):
        # Load model from path
        pass
    
    def generate(self, prompt: str, **kwargs):
        # Generate text
        pass
    
    async def generate_stream(self, prompt: str, **kwargs):
        # Streaming generation
        yield "token"
```

### Custom Workflow Nodes

```python
from odysseus.workflow import BaseNode

class DataTransformNode(BaseNode):
    name = "data_transform"
    
    async def execute(self, input_data, **params):
        # Transform data
        return transformed_data
```

---

## Performance and Scalability

### Single-User Performance

On a single RTX 4090 (24GB):
- 7B model (Q4_K_M): 80–120 tokens/second
- 13B model (Q4_K_M): 45–65 tokens/second
- 34B model (Q4_K_M): 20–30 tokens/second
- 70B model (Q4_K_M): 10–15 tokens/second

### Production Scaling

odysseus can scale horizontally by adding more model servers:

- **Single GPU**: 1–5 concurrent users (7B model)
- **Multi-GPU (2× RTX 4090)**: 5–20 concurrent users
- **Workstation (4× RTX 6000)**: 20–50 concurrent users
- **Small cluster**: 50–200 concurrent users

For high-throughput scenarios, vLLM backend with PagedAttention and continuous batching provides the best performance.

---

## Security Considerations

- **API authentication**: Enable auth in production; use API keys for automated access
- **Network isolation**: Run behind a reverse proxy; bind to localhost for single-user setups
- **Plugin security**: Audit plugins before installation; use the allowlist feature
- **Sandboxing**: Agent tool execution can be sandboxed in Docker containers
- **Secrets management**: Use environment variables or a vault for API keys and credentials
- **Regular updates**: Keep odysseus updated for security patches

---

## Troubleshooting Common Issues

| Issue | Solution |
|---|---|
| Model won't load | Check VRAM capacity; try a lower quantization level |
| GPU not detected | Verify NVIDIA drivers and CUDA toolkit installation |
| Slow inference | Reduce context length; use smaller model or higher quantization |
| Plugin installation fails | Check Python version compatibility; install dependencies |
| RAG queries return no results | Verify collection is populated; check embedding model |
| Authentication errors | Reset admin password via CLI: `odysseus reset-admin` |
| WebSocket disconnects | Check reverse proxy configuration for WebSocket support |

---

## Community and Resources

- **GitHub**: github.com/odysseus (71K★, 2K+ contributors)
- **Documentation**: docs.odysseus.ai (comprehensive, with examples)
- **Discord**: discord.gg/odysseus (100K+ members, active support)
- **Reddit**: r/odysseus_ai (discussions and showcases)
- **Plugin Registry**: plugins.odysseus.ai (500+ plugins)
- **Template Gallery**: templates.odysseus.ai (pre-built agent and workflow templates)

---

## Roadmap

odysseus's planned development trajectory:

| Feature | Target | Status |
|---|---|---|
| Fine-tuning integration (QLoRA) | Q3 2026 | In development |
| Multi-modal support (vision, audio) | Q4 2026 | In development |
| Distributed inference across multiple machines | Q4 2026 | In design |
| Model marketplace with one-click deploy | Q1 2027 | Planned |
| Mobile companion app | Q1 2027 | Planned |
| Native Windows installer | Q2 2027 | Planned |

---

## Conclusion

odysseus represents the culmination of the local AI movement — a comprehensive, integrated platform that makes self-hosted AI practical for individuals, teams, and enterprises. Its 71K GitHub stars reflect not just popularity but genuine utility: thousands of organizations have replaced complex multi-tool stacks with a single odysseus deployment.

Whether you are a solo developer looking for a personal AI assistant, a startup building an AI-powered product, or an enterprise deploying AI across departments, odysseus provides the infrastructure you need — running entirely on your own hardware, under your control.

The platform's modular architecture, extensive plugin ecosystem, and commitment to open-source development ensure that it will continue to evolve with the rapidly changing AI landscape. As models improve, hardware becomes more capable, and use cases expand, odysseus provides a stable foundation that adapts and grows.

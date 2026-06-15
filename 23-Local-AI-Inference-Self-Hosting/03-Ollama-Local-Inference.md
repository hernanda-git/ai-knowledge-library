# 03 — Ollama Local Inference

## Overview

Ollama has become the most popular tool for running large language models locally. Often described as "Docker for LLMs," it provides a streamlined workflow for downloading, managing, and serving models with minimal configuration. As of June 2026, Ollama supports over 300,000 models (including variants) across the Ollama library and HuggingFace, making it the go-to choice for both beginners and experienced practitioners.

This document provides a comprehensive guide to the Ollama ecosystem — its architecture, model management capabilities, API, GPU acceleration, quantization support, custom model files, performance optimization, and integration patterns.

---

## What Is Ollama?

Ollama is an open-source model runner that:

- **Downloads models** from the Ollama library or HuggingFace with a single command
- **Runs inference** with hardware acceleration (CUDA, Metal, ROCm, Vulkan, CPU)
- **Serves an API** that is compatible with the OpenAI chat completions format
- **Manages model versions** with tags and automatic updates
- **Supports custom models** through Modelfiles — a Dockerfile-like format for configuring models
- **Provides multi-model serving** — run multiple models simultaneously and switch between them

### Key Statistics

- **GitHub stars**: 100K+ (Ollama repository)
- **Model library**: 300,000+ model variants
- **Downloads**: 50M+ container pulls
- **Backend engines**: llama.cpp (primary), with experimental backends
- **Supported platforms**: macOS, Linux, Windows (native and WSL2)

---

## Architecture

Ollama follows a client-server architecture with a Go-based server and a C++ inference backend.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Ollama Server                           │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  HTTP API   │  │  Model      │  │  Request Queue  │ │
│  │  Server     │  │  Manager    │  │  + Scheduler    │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              llama.cpp Backend                        │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │ │
│  │  │ GGUF     │  │ Context  │  │  GPU/CPU         │   │ │
│  │  │ Loader   │  │ Manager  │  │  Scheduler       │   │ │
│  │  └──────────┘  └──────────┘  └──────────────────┘   │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Model Store │  │ Config      │  │  Plugin/        │ │
│  │ (Local FS)  │  │ Manager     │  │  Extension API  │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Server Components

1. **HTTP API Server**: Listens on port 11434 by default. Exposes REST endpoints for model management and inference. Supports both JSON and streaming responses.

2. **Model Manager**: Handles model downloading, caching, versioning, and loading into memory. Uses a content-addressable storage system to deduplicate model files.

3. **Request Queue and Scheduler**: Manages concurrent inference requests. Queues requests when models are loading or when GPU memory is constrained. Supports batching for efficiency.

4. **llama.cpp Backend**: The core inference engine. Compiled with hardware-specific optimizations (CUDA kernels, Metal Performance Shaders, ROCm blas, Vulkan compute shaders).

5. **Context Manager**: Manages the KV cache for ongoing conversations. Supports context window expansion and cache reuse between requests.

6. **GPU/CPU Scheduler**: Routes computation to available hardware. Can split layers between GPU and CPU for models that exceed VRAM capacity.

7. **Model Store**: Stores downloaded models in `~/.ollama/models/`. Uses hard links and reference counting to minimize disk usage when multiple model variants share the same base weights.

8. **Plugin System**: Experimental support for custom backends and model loaders.

### Request Flow

```
Client → API Request → [Authentication] → [Router] → [Model Loader]
→ [Context Manager] → [llama.cpp Inference] → [Token Streaming] → Client
```

1. Client sends a POST request to `/api/chat` or `/api/generate`
2. Server validates the request and authenticates (if configured)
3. Router determines which model to use based on the `model` parameter
4. Model loader checks if the model is already loaded in memory; if not, loads it from disk
5. Context manager initializes or extends the KV cache for the conversation
6. llama.cpp performs inference, generating tokens one at a time
7. For streaming requests, tokens are sent via server-sent events (SSE)
8. For non-streaming requests, tokens are buffered and returned as a single response

---

## Installation

### macOS

```bash
# Official installer (recommended)
curl -fsSL https://ollama.com/install.sh | sh

# Or via Homebrew
brew install ollama
```

### Linux

```bash
# Official installer (supports CUDA, ROCm, and CPU variants)
curl -fsSL https://ollama.com/install.sh | sh

# The installer detects your GPU and installs the appropriate variant
# For NVIDIA GPUs: CUDA version is installed
# For AMD GPUs: ROCm version is installed
# For CPU-only: CPU version is installed

# Start the service
systemctl start ollama
systemctl enable ollama
```

### Windows

```bash
# Download the installer from https://ollama.com/download/windows
# Run the installer (includes WSL2 backend automatically)

# Or install via WSL2 manually:
# 1. Install WSL2 and a Linux distribution
# 2. Follow Linux installation instructions inside WSL2
# 3. Expose the Ollama server to Windows:
#    export OLLAMA_HOST=0.0.0.0
```

### Docker

```bash
# CPU-only
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# NVIDIA GPU
docker run -d --gpus all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# AMD GPU
docker run -d --device /dev/kfd --device /dev/dri \
  -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:rocm
```

### Verify Installation

```bash
# Check version
ollama --version

# Run a test model
ollama run llama3.2:8b

# Check server status
curl http://localhost:11434/api/tags
```

---

## Model Management

### Pulling Models

```bash
# Pull a model from the library
ollama pull llama3.2:8b

# Pull a specific variant
ollama pull qwen2.5:7b-q4_K_M
ollama pull deepseek-coder:33b-q3_K_L

# Pull from HuggingFace (GGUF files)
ollama pull huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF:Q4_K_M
```

### Listing and Removing Models

```bash
# List all local models
ollama list

# Remove a model
ollama rm llama3.2:8b

# Copy a model with a new tag
ollama cp llama3.2:8b my-custom-llama

# Show model details
ollama show llama3.2:8b
```

### Model Storage

Models are stored in `~/.ollama/models/`. The structure is:

```
~/.ollama/models/
  ├── blobs/          # Content-addressable model file storage
  │   ├── sha256-abc123...  # Model weight shard
  │   ├── sha256-def456...  # Config file
  │   └── ...
  ├── manifests/      # Model manifest files (JSON)
  │   └── registry.ollama.ai/
  │       └── library/
  │           └── llama3.2-8b/
  │               └── latest
  └── tmp/            # Temporary download directory
```

### Supported Model Families

Ollama's model library includes all major open-source model families:

| Family | Example Models | Description |
|---|---|---|
| Llama | llama3.2 (8B, 70B), llama3.1 (8B, 70B, 405B) | Meta's flagship models |
| Mistral | mistral (7B), mixtral (8x7B, 8x22B) | Efficient models from Mistral AI |
| Qwen | qwen2.5 (0.5B–110B), codeqwen, qwq | Alibaba's comprehensive model family |
| DeepSeek | deepseek-v2, deepseek-coder, deepseek-r1 | Strong open-source models |
| Phi | phi-3 (mini, small, medium), phi-4 | Microsoft's small efficient models |
| Gemma | gemma-2 (2B, 9B, 27B) | Google's open models |
| Command | command-r, command-r-plus | Cohere's enterprise models |
| Yi | yi (6B, 9B, 34B) | 01.AI's models |
| CodeLlama | codellama (7B, 13B, 34B) | Meta's code-specialized models |
| Neural | neural-chat, solar | Various community favorites |
| StableLM | stablelm-2 (1.6B, 12B) | Stability AI's models |

---

## Running Models

### Interactive Mode

```bash
# Run a model interactively
ollama run llama3.2:8b

# With system prompt
ollama run llama3.2:8b --system "You are a helpful coding assistant."

# With temperature and other parameters
ollama run llama3.2:8b --temperature 0.7 --top_p 0.9
```

### One-Shot Inference

```bash
# Single prompt, no streaming
ollama run llama3.2:8b "What is the capital of France?"

# With verbose output showing token timing
ollama run llama3.2:8b "Explain quantum computing" --verbose
```

### Using the API

#### Chat Completions (OpenAI-Compatible)

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:32b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is machine learning?"}
    ],
    "temperature": 0.7,
    "max_tokens": 500,
    "stream": true
  }'
```

#### Ollama Native API

```bash
# Generate completion
curl http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:8b",
    "prompt": "What is the meaning of life?",
    "stream": false
  }'

# Chat completion (with conversation history)
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:8b",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'

# Generate embeddings
curl http://localhost:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text",
    "prompt": "The sky is blue"
  }'
```

---

## GPU Acceleration

### NVIDIA CUDA

Ollama automatically detects NVIDIA GPUs and loads the CUDA backend. You can verify GPU usage:

```bash
# Check if GPU is being used
ollama run llama3.2:8b --verbose

# Look for lines like:
# "gpu_layers": 35
# "gpu_used": true
# "load_time": 2.3s (loading 35 layers on GPU)

# Control GPU layers
ollama run llama3.2:70b --gpu-layers 40  # Only offload 40 layers to GPU
```

**GPU support matrix (NVIDIA):**

| GPU | VRAM | Max Model Size (Q4_K_M) | Performance |
|---|---|---|---|
| RTX 3060 | 12GB | 13B | 30–50 tok/s (7B) |
| RTX 4060 Ti | 16GB | 20B | 40–60 tok/s (7B) |
| RTX 4070 | 12GB | 13B | 50–70 tok/s (7B) |
| RTX 4080 | 16GB | 20B | 60–80 tok/s (7B) |
| RTX 4090 | 24GB | 34B | 80–120 tok/s (7B) |
| RTX 5090 | 32GB | 70B | 100–140 tok/s (7B) |
| RTX 6000 Ada | 48GB | 70B | 90–130 tok/s (7B) |
| A100 80GB | 80GB | 120B+ | 120–160 tok/s (7B) |

### Apple Metal

On Apple Silicon Macs, Ollama uses Metal Performance Shaders automatically:

```bash
# Metal is used by default; no special configuration needed
ollama run llama3.2:8b

# Check Metal usage with verbose flag
ollama run llama3.2:8b --verbose
```

**Apple Silicon performance:**

| Chip | Unified Memory | Max Model (Q4_K_M) | Performance (7B) |
|---|---|---|---|
| M1 | 8–16GB | 7B | 15–25 tok/s |
| M1 Pro/Max | 16–64GB | 34B | 25–40 tok/s |
| M2 | 8–24GB | 13B | 20–30 tok/s |
| M2 Pro/Max | 16–96GB | 70B | 30–50 tok/s |
| M2 Ultra | 64–192GB | 120B+ | 35–55 tok/s |
| M3 Pro/Max | 18–128GB | 70B | 35–55 tok/s |
| M3 Ultra | 64–192GB | 120B+ | 40–60 tok/s |
| M4 Max | 36–128GB | 70B | 45–65 tok/s |

### AMD ROCm

Ollama supports AMD GPUs through the ROCm backend:

```bash
# Ensure ROCm is installed
# On Ubuntu: sudo apt install rocm-dev

# Run with ROCm
HSA_OVERRIDE_GFX_VERSION=11.0.0 ollama run llama3.2:8b

# Or set environment variables in the service
sudo systemctl edit ollama
# Add: Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
```

**AMD GPU performance:**

| GPU | VRAM | Max Model (Q4_K_M) | Relative Performance |
|---|---|---|---|
| RX 6800 XT | 16GB | 20B | ~65% of RTX 4090 |
| RX 6900 XT | 16GB | 20B | ~70% of RTX 4090 |
| RX 7900 GRE | 16GB | 20B | ~75% of RTX 4090 |
| RX 7900 XT | 20GB | 30B | ~80% of RTX 4090 |
| RX 7900 XTX | 24GB | 34B | ~85% of RTX 4090 |
| Pro W7900 | 48GB | 70B | ~80% of RTX 6000 |

### Vulkan

For GPUs that don't support CUDA, Metal, or ROCm, Ollama can use Vulkan:

```bash
# Install Vulkan-enabled Ollama
curl -fsSL https://ollama.com/install.sh | OLLAMA_VULKAN=1 sh

# Or set environment variable
export OLLAMA_VULKAN=1
ollama serve
```

### CPU-Only Mode

```bash
# CPU-only mode (for systems without a compatible GPU)
export OLLAMA_NO_GPU=1
ollama serve

# Or run a specific model on CPU
OLLAMA_NO_GPU=1 ollama run llama3.2:8b
```

**CPU performance (7B model, Q4_K_M):**

| CPU | RAM | Tokens/Second |
|---|---|---|
| Intel Core i5-13400 | DDR5-4800 | 5–8 |
| Intel Core i7-13700K | DDR5-5600 | 8–12 |
| Intel Core i9-13900K | DDR5-6000 | 10–15 |
| AMD Ryzen 5 7600 | DDR5-5200 | 6–9 |
| AMD Ryzen 7 7800X3D | DDR5-5600 | 9–13 |
| AMD Ryzen 9 7950X | DDR5-6000 | 11–16 |
| Apple M3 Max | Unified | 35–55 |
| Intel Xeon Platinum | DDR5-4800 (8-channel) | 15–25 |

---

## Modelfiles

Modelfiles are Ollama's equivalent of Dockerfiles — they allow you to create custom models by configuring parameters, system prompts, and model templates.

### Modelfile Syntax

```dockerfile
# Import a base model
FROM llama3.2:8b

# Set the system prompt
SYSTEM """
You are a helpful coding assistant specialized in Python.
You provide concise, well-commented code examples.
"""

# Set model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 8192
PARAMETER repeat_penalty 1.1

# Set a custom template (advanced)
TEMPLATE """
{{ if .System }}<|system|>
{{ .System }}
{{ end }}<|user|>
{{ .Prompt }}
{{ end }}<|assistant|>
"""

# License information (optional)
LICENSE MIT
```

### Creating a Custom Model

```bash
# Create a Modelfile
cat > Modelfile << 'EOF'
FROM llama3.2:8b

SYSTEM """
You are an expert Python developer. You write clean, efficient,
well-documented code following PEP 8 standards.
Always include type hints and docstrings.
"""

PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER num_ctx 16384
EOF

# Create the model
ollama create my-python-assistant -f Modelfile

# Use the custom model
ollama run my-python-assistant
```

### Advanced Modelfile Examples

#### Coding Assistant with Code-Specific Parameters

```dockerfile
FROM deepseek-coder:33b

PARAMETER temperature 0.1
PARAMETER top_p 0.85
PARAMETER num_ctx 32768
PARAMETER stop "```"
PARAMETER stop "<|file_sep|>"

SYSTEM """
You are an expert software engineer. Follow these rules:
1. Write complete, runnable code
2. Include tests with your code
3. Explain your design decisions
4. Use the requested programming language
5. Handle edge cases and errors
"""
```

#### Creative Writing Assistant

```dockerfile
FROM qwen2.5:72b

PARAMETER temperature 0.9
PARAMETER top_p 0.95
PARAMETER top_k 60
PARAMETER num_ctx 16384

SYSTEM """
You are a creative writing assistant. You help with:
- Story generation and plot development
- Character creation and dialogue
- World-building and setting description
- Poetry and prose editing
- Genre-specific writing advice

Be imaginative and evocative in your language.
Use literary devices appropriately.
Adapt your style to match the requested genre.
"""
```

#### RAG-Enhanced Question Answering

```dockerfile
FROM llama3.2:70b

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 32768

SYSTEM """
You are a precise question-answering system.
Use the provided context to answer questions accurately.
If the context doesn't contain enough information, say so.
Cite relevant parts of the context in your answers.
Do not make up information not present in the context.
"""
```

### Modelfile Parameters Reference

| Parameter | Default | Description |
|---|---|---|
| `temperature` | 0.8 | Controls randomness (0.0 = deterministic, 2.0 = very random) |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `top_k` | 40 | Top-K sampling threshold |
| `num_ctx` | 2048 | Context window size in tokens |
| `repeat_penalty` | 1.1 | Penalty for repeating tokens |
| `stop` | — | Stop sequences (can be specified multiple times) |
| `seed` | random | Random seed for reproducibility |
| `num_predict` | 128 | Max tokens to generate |
| `num_batch` | 512 | Batch size for prompt processing |
| `tfs_z` | 1.0 | Tail free sampling |
| `typical_p` | 1.0 | Locally typical sampling threshold |
| `mirostat` | 0 | Mirostat sampling mode (0 = disabled) |
| `mirostat_tau` | 5.0 | Mirostat target entropy |
| `mirostat_eta` | 0.1 | Mirostat learning rate |
| `num_gpu` | — | Number of GPU layers to offload (0 = CPU only) |
| `main_gpu` | 0 | Main GPU index for multi-GPU setups |
| `low_vram` | false | Enable low VRAM mode |

---

## Performance Optimization

### Environment Variables

```bash
# GPU and memory settings
export OLLAMA_NUM_GPU=1          # Number of GPUs to use
export OLLAMA_GPU_OVERLAP=true   # Overlap GPU compute with data transfer
export OLLAMA_KEEP_ALIVE=5m      # Keep model loaded between requests
export OLLAMA_MAX_LOADED_MODELS=3 # Max models loaded simultaneously

# Server settings
export OLLAMA_HOST=0.0.0.0       # Bind address
export OLLAMA_PORT=11434          # Port
export OLLAMA_NUM_PARALLEL=4     # Max concurrent requests
export OLLAMA_MAX_QUEUE=512      # Max queued requests

# Debug and logging
export OLLAMA_DEBUG=1            # Enable debug logging
export OLLAMA_LOGFILE=/var/log/ollama.log

# Memory management
export OLLAMA_SCHED_SPREAD=1     # Spread models across GPUs
export OLLAMA_MMAP=1             # Enable memory mapping
export OLLAMA_FLASH_ATTN=1       # Enable flash attention (faster, uses less VRAM)
```

### Flash Attention

Flash attention dramatically improves both speed and memory usage for long contexts:

```bash
# Enable flash attention globally
export OLLAMA_FLASH_ATTN=1

# Or per-request (via API)
curl http://localhost:11434/api/generate \
  -d '{
    "model": "llama3.2:8b",
    "prompt": "Long document...",
    "options": {"flash_attn": true}
  }'
```

With flash attention, context processing time is reduced by 2–5× for long sequences, and memory usage scales linearly instead of quadratically with context length.

### Context Window Management

```bash
# Set context window size in Modelfile
PARAMETER num_ctx 32768

# Or via API
curl http://localhost:11434/api/generate \
  -d '{
    "model": "llama3.2:8b",
    "prompt": "...",
    "options": {"num_ctx": 32768}
  }'
```

**Context window size and VRAM usage (Q4_K_M):**

| Context | 7B Model | 13B Model | 34B Model | 70B Model |
|---|---|---|---|---|
| 2,048 | 6GB | 9GB | 19GB | 38GB |
| 4,096 | 6.5GB | 10GB | 20GB | 40GB |
| 8,192 | 7.5GB | 12GB | 22GB | 44GB |
| 16,384 | 9.5GB | 16GB | 26GB | 52GB |
| 32,768 | 13.5GB | 24GB | 34GB | 68GB |
| 65,536 | 21.5GB | 40GB | 50GB | 100GB |
| 128,000 | 37.5GB | 72GB | 82GB | 164GB |

Note: Flash attention reduces KV cache memory by approximately 50% for long contexts.

### Multi-GPU Configuration

```bash
# Use two GPUs
export OLLAMA_NUM_GPU=2

# Specific GPU selection
export CUDA_VISIBLE_DEVICES=0,1  # For NVIDIA GPUs

# In Docker
docker run -d --gpus '"device=0,1"' \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  ollama/ollama
```

### Concurrent Request Handling

```bash
# Allow up to 4 parallel requests
export OLLAMA_NUM_PARALLEL=4

# Queue limit
export OLLAMA_MAX_QUEUE=512

# Keep models warm between requests
export OLLAMA_KEEP_ALIVE=10m
```

---

## API Reference

### Complete API Endpoints

| Endpoint | Method | Description | OpenAI Compatible |
|---|---|---|---|
| `/api/generate` | POST | Generate a completion | No |
| `/api/chat` | POST | Generate a chat completion | No |
| `/api/embeddings` | POST | Generate embeddings | No |
| `/api/pull` | POST | Download a model | No |
| `/api/push` | POST | Upload a model | No |
| `/api/create` | POST | Create a model from Modelfile | No |
| `/api/tags` | GET | List local models | No |
| `/api/show` | POST | Show model details | No |
| `/api/copy` | POST | Copy a model | No |
| `/api/delete` | DELETE | Delete a model | No |
| `/v1/chat/completions` | POST | Chat completions (OpenAI format) | Yes |
| `/v1/completions` | POST | Text completions (OpenAI format) | Yes |
| `/v1/embeddings` | POST | Embeddings (OpenAI format) | Yes |
| `/v1/models` | GET | List models (OpenAI format) | Yes |

### Using with OpenAI Client Libraries

#### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Not used but required by the SDK
)

response = client.chat.completions.create(
    model="llama3.2:8b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=500,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

#### JavaScript

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:11434/v1',
  apiKey: 'ollama'
});

const response = await client.chat.completions.create({
  model: 'llama3.2:8b',
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'What is the capital of France?' }
  ],
  stream: true
});

for await (const chunk of response) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '');
}
```

#### cURL (Streaming)

```bash
curl -N http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:8b",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

---

## Integration Patterns

### Python Client (Ollama Library)

```bash
pip install ollama
```

```python
import ollama

# Basic chat
response = ollama.chat(
    model='llama3.2:8b',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}]
)
print(response['message']['content'])

# Streaming
stream = ollama.chat(
    model='llama3.2:8b',
    messages=[{'role': 'user', 'content': 'Tell me a story'}],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)

# Embeddings
embeddings = ollama.embeddings(
    model='nomic-embed-text',
    prompt='The sky is blue because of Rayleigh scattering'
)
print(embeddings['embedding'])

# List models
models = ollama.list()
for model in models['models']:
    print(model['name'], model['modified_at'])

# Create custom model
modelfile = """
FROM llama3.2:8b
SYSTEM "You are a pirate chatbot! Arrr!"
PARAMETER temperature 1.0
"""
ollama.create(model='pirate-llama', modelfile=modelfile)
```

### Integration with LangChain

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Chat model
llm = ChatOllama(
    model="qwen2.5:32b",
    temperature=0.7,
    num_predict=1024,
    num_ctx=32768
)

# Embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# RAG with LangChain
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64
)
chunks = text_splitter.split_documents(documents)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Create RAG chain
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

response = qa_chain.invoke({"query": "What is the main topic of these documents?"})
```

### Integration with Open WebUI

```bash
# Run Open WebUI connected to Ollama
docker run -d \
  -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

### Integration with Continue (VS Code)

```json
// .continuerc.json
{
  "models": [
    {
      "title": "Ollama Qwen 2.5",
      "provider": "ollama",
      "model": "qwen2.5:32b",
      "apiBase": "http://localhost:11434"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Ollama CodeQwen",
    "provider": "ollama",
    "model": "codeqwen:7b"
  }
}
```

---

## Security Configuration

### Basic Authentication

Ollama supports basic HTTP authentication:

```bash
# Start server with authentication
ollama serve --auth
# You will be prompted to set a username and password

# Or via environment variables
export OLLAMA_AUTH=username:password
ollama serve
```

### TLS/SSL

```bash
# Using a reverse proxy (nginx)
server {
    listen 443 ssl;
    server_name ollama.example.com;

    ssl_certificate /etc/ssl/certs/ollama.crt;
    ssl_certificate_key /etc/ssl/private/ollama.key;

    location / {
        proxy_pass http://localhost:11434;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_buffering off;
    }
}
```

### Firewall Configuration

```bash
# Bind to localhost only (default for security)
export OLLAMA_HOST=127.0.0.1

# For LAN access, bind to specific interface
export OLLAMA_HOST=192.168.1.100:11434

# Or use firewall rules
sudo ufw allow from 192.168.1.0/24 to any port 11434
```

---

## Monitoring and Logging

### Built-in Metrics

Ollama exposes Prometheus metrics at `/api/metrics`:

- `ollama_model_load_seconds`: Time to load model
- `ollama_generate_seconds`: Time to generate response
- `ollama_generate_tokens`: Number of tokens generated
- `ollama_gpu_memory_bytes`: GPU memory usage
- `ollama_requests_active`: Current active requests
- `ollama_requests_queue`: Queue depth

### Integration with Prometheus

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ollama'
    static_configs:
      - targets: ['localhost:11434']
    metrics_path: '/api/metrics'
```

### Logging

```bash
# Enable verbose logging
export OLLAMA_DEBUG=1

# Log to file
export OLLAMA_LOGFILE=/var/log/ollama/ollama.log

# Access logs are available through the server
# Every API request is logged with method, path, status, and duration
```

---

## Troubleshooting

### Common Issues

| Problem | Likely Cause | Solution |
|---|---|---|
| Model fails to load | VRAM exhaustion | Use smaller model or higher quantization; reduce num_ctx |
| CUDA errors | NVIDIA driver issues | Reinstall CUDA toolkit; check nvidia-smi |
| Slow response | CPU inference or memory swap | Check if GPU is being used; reduce concurrent requests |
| Out of memory | Context window too large | Reduce num_ctx; enable flash attention |
| Connection refused | Server not running | Check `ollama serve`; verify port 11434 |
| Model download fails | Network issue or disk space | Check internet; clear model cache |
| API returns 404 | Wrong model name | `ollama list` to verify model name |
| Metal errors (macOS) | Outdated macOS | Update to latest macOS version |

### Debug Commands

```bash
# Check server logs
journalctl -u ollama -f

# Test GPU connectivity
ollama run llama3.2:8b --verbose 2>&1 | grep -i gpu

# Check model details
ollama show llama3.2:8b

# Reset everything (preserves models)
rm -rf ~/.ollama/models/manifests
ollama serve

# Full reset (removes all models)
rm -rf ~/.ollama
ollama serve
```

### Performance Profiling

```bash
# Measure token generation speed
ollama run llama3.2:8b --verbose "Generate 100 words about AI"

# The verbose output shows:
# - Load time (model loading)
# - Prompt evaluation time
# - Generation time per token
# - Total tokens/second
```

---

## Best Practices

### Model Selection

- **7B models** (Q4_K_M): Best for most tasks on consumer GPUs. Llama 3.2 8B and Qwen 2.5 7B are excellent general-purpose choices.
- **13B models** (Q4_K_M): Better reasoning and knowledge, requires 12GB+ VRAM. Qwen 2.5 14B is a strong choice.
- **34B models** (Q3_K_M or Q4_K_M): Near-GPT-3.5 quality, requires 24GB VRAM. Yi-34B and Qwen 2.5 32B are top picks.
- **70B models** (Q3_K_M or Q4_K_M): Approaches GPT-4 quality, requires 48GB VRAM. Llama 3 70B and Qwen 2.5 72B are the standards.

### Performance Tuning

1. **Reduce context window** to the minimum needed for your use case
2. **Enable flash attention** for long contexts
3. **Use appropriate quantization** — Q4_K_M is the sweet spot for most use cases
4. **Keep models loaded** with a long keep-alive setting if you make frequent requests
5. **Limit concurrent requests** based on your GPU memory
6. **Use GPU layers** to offload as many layers as possible to GPU
7. **Monitor swap usage** — if the system is swapping, reduce model size

### Production Deployment

- Run Ollama behind a reverse proxy (nginx, Caddy) for TLS and load balancing
- Use Docker with resource limits for predictable performance
- Set `OLLAMA_KEEP_ALIVE` to 5–10 minutes for active deployments
- Configure `OLLAMA_MAX_LOADED_MODELS` to prevent memory exhaustion
- Enable authentication for multi-user access
- Monitor GPU memory usage with `nvidia-smi` or Prometheus

---

## Comparison with Other Model Runners

| Feature | Ollama | llama.cpp (CLI) | LM Studio | vLLM |
|---|---|---|---|---|
| **Installation** | ⭐⭐⭐⭐⭐ One command | ⭐⭐ Build from source | ⭐⭐⭐⭐⭐ GUI installer | ⭐⭐⭐ pip install |
| **Model Management** | ⭐⭐⭐⭐⭐ Built-in | ⭐⭐ Manual | ⭐⭐⭐⭐⭐ Built-in GUI | ⭐⭐ Manual |
| **API Server** | ⭐⭐⭐⭐⭐ OpenAI-compatible | ⭐⭐ Basic HTTP | ⭐⭐ Local only | ⭐⭐⭐⭐⭐ Full OpenAI |
| **GPU Support** | ⭐⭐⭐⭐ CUDA/Metal/ROCm | ⭐⭐⭐⭐⭐ All backends | ⭐⭐⭐⭐ CUDA/Metal | ⭐⭐⭐⭐ CUDA only |
| **Performance** | ⭐⭐⭐⭐ Good | ⭐⭐⭐ Good | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐⭐ Excellent |
| **Concurrent Users** | ⭐⭐⭐⭐ 1–10 | ⭐⭐ 1 | ⭐ 1 | ⭐⭐⭐⭐⭐ 100+ |
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Custom Models** | ⭐⭐⭐⭐⭐ Modelfiles | ⭐⭐⭐⭐⭐ Any GGUF | ⭐⭐⭐⭐ GUI import | ⭐⭐ Pipelines |
| **Multi-GPU** | ⭐⭐⭐⭐ Supported | ⭐⭐⭐⭐⭐ Supported | ⭐⭐ Limited | ⭐⭐⭐⭐⭐ Native |

---

## Future Directions

Ollama's development roadmap includes:

- **Multi-modal support**: Vision models, audio, and image generation
- **Distributed inference**: Running models across multiple machines
- **Plugin system**: Extensions for custom backends and preprocessing
- **Fine-tuning**: Native fine-tuning capabilities within Ollama
- **Enhanced monitoring**: Better observability and debugging tools
- **Model marketplace**: Community model sharing and discovery

---

## Conclusion

Ollama has become the de facto standard for running local language models, and for good reason. Its combination of ease of use, broad model support, GPU acceleration, and OpenAI-compatible API makes it accessible to beginners while remaining powerful enough for production deployments.

Whether you are running a single 7B model on a laptop for personal use or serving dozens of models across a GPU cluster for enterprise applications, Ollama provides the foundation for reliable, performant local inference. Its integration with the broader ecosystem — Open WebUI, Continue.dev, LangChain, and others — ensures that it fits naturally into any AI workflow.

The local AI revolution runs on models, and for most people, Ollama is how they run them.

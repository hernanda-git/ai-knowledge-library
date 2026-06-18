# Small Language Models — Efficiency, Edge Deployment & On-Device AI

> June 2026

The race to build ever-larger models has an equally important counter-current: **small language models (SLMs)** that deliver frontier-level capabilities in a fraction of the size, cost, and energy. This document covers the state of the art in model compression, distillation, and on-device deployment.

---

## 1. The Small Model Revolution

### 1.1 Why Small Models Matter

| Factor | Large Model (70B+) | Small Model (<8B) |
|--------|-------------------|-------------------|
| **Cost per 1M tokens** | $0.50–$15.00 | $0.01–$0.30 |
| **Latency (first token)** | 500ms–5s | 20ms–200ms |
| **Memory (GPU)** | 140GB+ (FP16) | 2–16GB |
| **Energy per query** | 10–100 Wh | 0.1–2 Wh |
| **On-device capable** | ❌ | ✅ (phones, laptops, edge) |
| **Privacy** | Cloud-only | Local execution possible |

### 1.2 Key Milestones

- **2023**: Phi-1 (1.3B) shows small models can code
- **2024**: Phi-3 (3.8B) matches Llama-2-70B on benchmarks; Gemma 2B/7B released
- **2024–2025**: Llama-3.2-1B/3B — efficient, multilingual; Qwen2.5-Coder-1.5B
- **2025**: Phi-4 (14B) — small but mighty; Gemma 2 (2B/9B/27B); SmolLM2 (135M–1.7B)
- **2026**: Llama 4 Scout (17B MoE), Qwen3 (1.7B–235B), DeepSeek-V3-Lite

---

## 2. Architectures for Efficiency

### 2.1 Mixture of Experts (MoE) for Small Models

MoE allows small active parameters with large total capacity:

```
Total Parameters: 17B | Active Parameters per Token: 3B

Router → Expert 1  (specialized: math)       
       → Expert 2  (specialized: code)        → Output
       → Expert 7  (specialized: creative)
       (Only top-2 experts activated per token)
```

**Llama 4 Scout** (17B MoE, ~3B active):
- 10M context window (largest of any open model)
- Competitive with Llama 3.1-70B on many benchmarks
- Runs on a single GPU (24GB VRAM)

### 2.2 Shared-Weight Architectures

Weight-tying across layers reduces parameters:
- **ALBERT**: Cross-layer parameter sharing
- **Universal Transformers**: Recurrent depth with shared weights
- **T5-style encoder-decoder**: Shared embedding + output layers

### 2.3 Efficient Attention Mechanisms

| Mechanism | Description | Memory Saving | Models |
|-----------|-------------|---------------|--------|
| **Grouped Query Attention (GQA)** | Multiple queries share key-value heads | 2–4× | Llama 3/4, Gemma 2 |
| **Sliding Window Attention** | Local window + global tokens | 2–8× | Mistral, Phi-3 |
| **Multi-Query Attention (MQA)** | All queries share one KV head | 4–8× | PaLM, Falcon |
| **Flash Attention** | Tiled, fused kernel for attention | Memory efficient | All modern models |
| **Attention Sinks** | Preserve initial tokens in KV cache | 16× for long context | Streaming LLM |

---

## 3. Model Landscape (June 2026)

### 3.1 Small Language Model Comparison

| Model | Params | Context | Key Strength | Open? | License |
|-------|--------|---------|-------------|-------|--------|
| **Phi-4** | 14B | 128K | Reasoning, math, code | ✅ | MIT |
| **Phi-3.5-mini** | 3.8B | 128K | Multilingual, reasoning | ✅ | MIT |
| **Gemma 2 2B** | 2.6B | 8K | General, efficient | ✅ | Gemma |
| **Gemma 2 9B** | 9.2B | 8K | Strong general model | ✅ | Gemma |
| **Llama 3.2 1B** | 1.2B | 128K | Multilingual, efficient | ✅ | Llama 3.2 |
| **Llama 3.2 3B** | 3.2B | 128K | Best in class <4B | ✅ | Llama 3.2 |
| **Llama 4 Scout** | 17B MoE (3B active) | 10M | MoE, extreme context | ✅ | Llama 4 |
| **Qwen3 1.7B** | 1.7B | 32K | Strong reasoning in size | ✅ | Apache 2.0 |
| **Qwen3 4B** | 3.8B | 32K | Excellent multilingual | ✅ | Apache 2.0 |
| **Qwen3 8B** | 7.6B | 32K | Solid all-rounder | ✅ | Apache 2.0 |
| **SmolLM2 1.7B** | 1.7B | 8K | Best for mobile | ✅ | Apache 2.0 |
| **DeepSeek-V3-Lite** | 16B MoE | 128K | MoE efficiency | ✅ | MIT |
| **H2O-Danube3** | 0.5B–4B | 8K | Ultra-efficient, long context | ✅ | Apache 2.0 |

### 3.2 Benchmarks at Each Size Tier

| Size Tier | Best Model | MMLU | MATH | HumanEval |
|-----------|------------|------|------|-----------|
| **<3B** | Llama 3.2 3B | 70% | 42% | 57% |
| **3–7B** | Phi-3.5-mini 3.8B | 73% | 48% | 62% |
| **7–15B** | Phi-4 14B | 85% | 81% | 82% |
| **~20B MoE** | Llama 4 Scout | 82% | 74% | 80% |

---

## 4. Quantization & Compression

### 4.1 Quantization Levels

| Precision | Bits/Weight | Size (7B model) | Quality Loss | Hardware |
|-----------|-------------|-----------------|-------------|----------|
| **FP32** | 32 | 28 GB | None | Full precision GPUs |
| **FP16/BF16** | 16 | 14 GB | None | Most GPUs |
| **INT8** | 8 | 7 GB | Minimal | GPUs + some NPUs |
| **INT4** | 4 | 3.5 GB | Small (recoverable) | Modern NPUs, Metal |
| **INT3 / FP4** | 3–4 | 2.6 GB | Moderate | Research (NVIDIA B200+) |
| **INT2 / NF2** | 2 | 1.75 GB | Significant | Experimental |

### 4.2 Quantization Methods

| Method | Description | Quality Recovery |
|--------|-------------|-----------------|
| **GPTQ** | Post-training quant, one-shot calibration | Good (4-bit) |
| **AWQ** | Activation-aware weight quantization | Excellent (4-bit) |
| **GGUF / llama.cpp** | CPU-optimized quant (2–8 bit) | Good |
| **QLoRA** | Quantized LoRA fine-tuning | Good (4-bit) |
| **QuIP#** | Lattice quantization + incoherence processing | Best (2-bit) |
| **AQLM** | Additive quantization for LLMs | Best at 2-bit |
| **BitDelta** | Delta compression between models | Lossless |

### 4.3 Quantization Impact on Benchmarks (Phi-4)

| Precision | Size | MMLU | GSM8K | Reasoning Quality |
|-----------|------|------|-------|-------------------|
| BF16 | 28 GB | 85.1% | 92% | Full |
| INT4 (AWQ) | 7.8 GB | 84.5% | 91% | Nearly identical |
| INT4 (GPTQ) | 7.8 GB | 83.9% | 90% | Slight degradation |
| GGUF Q4_K_M | 4.8 GB | 84.2% | 91% | Nearly identical |
| GGUF Q3_K_M | 3.8 GB | 82.1% | 88% | Moderate |
| GGUF Q2_K | 2.9 GB | 76.3% | 81% | Significant |

---

## 5. On-Device Deployment

### 5.1 Hardware Targets

| Platform | Typical RAM | Supported Formats | Typical Model |
|----------|-------------|------------------|---------------|
| **iPhone 15/16 Pro** | 8 GB | Core ML, MLX | 3B–7B INT4 |
| **Android flagship** | 12–16 GB | TFLite, NNAPI, GPU | 3B–8B INT4 |
| **MacBook M3/M4** | 8–36 GB | MLX, Core ML, gguf | 7B–30B INT4 |
| **Windows Laptop** | 8–32 GB | DirectML, ONNX, gguf | 3B–13B INT4 |
| **Raspberry Pi 5** | 8 GB | gguf (CPU only) | 1B–3B INT4 (slow) |
| **Edge servers** | 32–128 GB | TensorRT, ONNX | 8B–70B INT4 |

### 5.2 Inference Engines

| Engine | Platform | Speed | Notes |
|--------|----------|-------|-------|
| **llama.cpp** | CPU, GPU (CUDA/Metal/Vulkan) | Fast | Gold standard, gguf format |
| **MLX** | Apple Silicon | Very fast | Apple-optimized, easy API |
| **ExecuTorch** | Mobile, embedded | Good | PyTorch-native mobile |
| **ONNX Runtime** | Cross-platform | Good | Enterprise, many optimizations |
| **TensorRT-LLM** | NVIDIA GPU | Very fast | Production inference server |
| **MediaPipe** | Mobile | Good | Google's mobile framework |
| **Transformers.js** | Browser (WebGPU) | Moderate | In-browser inference |
| **Ollama** | Desktop, server | Good | User-friendly, model management |

---

## 6. Applications & Use Cases

### 6.1 On-Device Privacy

```
❌ Cloud solution:
User data → upload to API → process → return → (data stored on server)

✅ On-device solution:
User data → local LLM → result → (data never leaves device)
```

**Applications**:
- Medical chat (HIPAA-compliant, no data sent)
- Personal assistant (calendar, email, contacts — all on device)
- Secure document analysis (legal, financial)

### 6.2 Real-Time Applications

| Application | Ideal Model | Max Latency |
|-------------|-------------|-------------|
| **Voice assistant** | 1B–3B INT4 | <100ms TTT |
| **Code autocomplete** | 1B–3B INT4 | <50ms |
| **Keyboard autocomplete** | 100M–1B INT4 | <10ms |
| **Live translation** | 1B–3B INT4 | <200ms |
| **Camera OCR + analysis** | 3B–8B INT4 | <500ms |

### 6.3 Edge Computing

- **IoT devices**: Sensor data analysis with tiny LLMs (<500M)
- **Manufacturing**: Real-time defect detection + reasoning
- **Retail**: Local inventory management, customer interaction
- **Agriculture**: Offline field analysis in remote areas

---

## 7. Training Small Models

### 7.1 Data-Centric Approaches

Small models need high-quality, deduplicated data:

| Technique | Impact | Example |
|-----------|--------|---------|
| **Data deduplication** | 3× effective data size | Phi series |
| **Code + textbook quality** | Strong reasoning | Phi-1 trained on "textbook quality" data |
| **Multi-turn synthetic data** | Better instruction following | Orca, phi-3 |
| **Knowledge distillation** | Transfers reasoning from large models | Distilled R1 models |

### 7.2 Phi-4 Training Recipe

```
Data: 4T tokens of "textbook quality"
  ├── Filtered web data (3× dedup)
  ├── Synthetic textbooks (DeepSeek-generated)
  └── Code repositories with solutions

Architecture: Dense decoder, 14B params
  ├── GQA (Grouped Query Attention)
  ├── RoPE (Rotary Position Embedding)
  └── 48 layers, 40 heads

Training:
  ├── 30K steps on 8× H100 nodes
  ├── Context: 4K → 128K (long context curriculum)
  └── Final SFT + DPO alignment

Result: Matches Llama-3-70B on math benchmarks
```

---

## 8. Open Challenges & Research Directions

| Challenge | Status | Research Directions |
|-----------|--------|-------------------|
| **Context length** | Small models struggle beyond 32K | Hierarchical memory, RAG-native SLMs |
| **Multimodal SLMs** | Emerging (Phi-4-vision) | Joint vision-language compression |
| **Agentic SLMs** | Limited tool use | Tool-formatted training data, function calling distilled |
| **Reasoning in small models** | Distillation helps (R1-distill) | Better RL for small models |
| **Hardware co-design** | Early (Apple, Qualcomm) | Custom NPUs for transformer inference |
| **Battery impact** | Active research | Sparsity, event-driven inference |

---

*This document is part of the AI Knowledge Library — 30-Small-Language-Models directory.*
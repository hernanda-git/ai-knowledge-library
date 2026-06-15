# 05 — GGUF Quants and Model Optimization

## Overview

GGUF (GPT-Generated Unified Format) is the dominant file format for running large language models locally. It was introduced in August 2023 as a replacement for the earlier GGML format and has since become the standard for quantized models in the local AI ecosystem.

This document provides a comprehensive guide to GGUF quantization: what it is, how the different quantization levels work, how to choose the right quantization for your hardware, performance benchmarks, and practical guidance for working with quantized models.

---

## What Is Quantization?

### The Core Idea

Large language models are composed of billions of parameters (weights). These weights are typically stored as 16-bit floating-point numbers (FP16/BF16), requiring 2 bytes per parameter. A 70B-parameter model thus requires 70,000,000,000 × 2 bytes = **140 GB** of memory — far more than any consumer GPU can handle.

**Quantization** reduces the precision of these weights, storing them in fewer bits:

| Precision | Bits per Weight | 7B Model Size | 13B Model Size | 34B Model Size | 70B Model Size |
|---|---|---|---|---|---|
| FP32 (32-bit float) | 32 | 28 GB | 52 GB | 136 GB | 280 GB |
| FP16/BF16 | 16 | 14 GB | 26 GB | 68 GB | 140 GB |
| INT8 (Q8_0) | 8 | 7 GB | 13 GB | 34 GB | 70 GB |
| INT4 (Q4_K_M) | 4.5 (avg) | 3.9 GB | 7.3 GB | 19 GB | 39 GB |
| INT2 (Q2_K) | 2.6 (avg) | 2.3 GB | 4.3 GB | 11 GB | 23 GB |

### Why Quantization Works

The surprising fact about LLM quantization is that it works remarkably well with minimal quality loss. This is because:

1. **Neural network weights are highly redundant.** Many weights can be represented with lower precision without significantly affecting the model's output distribution.

2. **Quantization-aware training (QAT)** and **post-training quantization (PTQ)** have been refined to minimize information loss. Modern techniques like importance matrices, per-group quantization, and activation-aware scaling preserve accuracy far better than simple rounding.

3. **The model's output is probabilistic.** Even at FP16 precision, repeated forward passes with the same input produce slightly different outputs (due to sampling randomness). The additional noise from quantization is often comparable to this inherent randomness.

4. **Larger models are more quantization-tolerant.** A 70B model at Q4_K_M often performs better than a 7B model at FP16, because the sheer number of parameters compensates for reduced per-parameter precision.

### The Quantization Trade-off

```
Memory Usage:     Q2_K < Q3_K < Q4_K < Q5_K < Q6_K < Q8_0 < FP16
Performance:      Q2_K < Q3_K < Q4_K < Q5_K < Q6_K < Q8_0 < FP16
Quality:          Q2_K < Q3_K < Q4_K < Q5_K < Q6_K < Q8_0 ≈ FP16
```

The key insight: **Q4_K_M is the sweet spot.** It offers approximately 4× compression with only 1–3% quality degradation on most benchmarks. Going to Q3 or Q2 saves more memory but quality degradation accelerates.

---

## The GGUF Format

### History

| Date | Format | Significance |
|---|---|---|
| Mar 2023 | Original GGML | Early format for llama.cpp |
| Nov 2023 | GGUF v1 | Standardized format, extensible metadata |
| Jan 2024 | GGUF v2 | Improved tensor encoding, padding |
| Jun 2024 | GGUF v3 | Support for MoE models, QMoE |
| 2025 | GGUF v4+ | Ongoing improvements in quantization kernels |

### GGUF File Structure

A GGUF file contains:

1. **Header**: Magic number (GGUF), version, tensor count, metadata key-value pairs
2. **Metadata**: Model architecture, tokenizer config, hyperparameters, quantization info
3. **Tokenizer**: Vocabulary, merges (for BPE), or scores (for unigram)
4. **Tensors**: The actual weight data, split into tensors (layers, attention, MLP, etc.)

```
┌─────────────────────┐
│ Header (GGUF magic)  │
├─────────────────────┤
│ Metadata KV pairs     │
├─────────────────────┤
│ Tokenizer data        │
├─────────────────────┤
│ Tensor 1: (name, dims, type, offset) │
│ Tensor 1 data         │
├─────────────────────┤
│ Tensor 2: (...)       │
│ Tensor 2 data         │
├─────────────────────┤
│ ...                   │
└─────────────────────┘
```

### Advantages Over Other Formats

| Feature | GGUF | GPTQ | AWQ | bitsandbytes |
|---|---|---|---|---|
| **Ecosystem support** | Ollama, llama.cpp, LM Studio, odysseus | ExLlama, AutoGPTQ | vLLM, TGI | HuggingFace Transformers |
| **CPU inference** | ✅ Native | ❌ GPU only | ❌ GPU only | ❌ GPU only |
| **GPU offloading** | ✅ Partial | ✅ Full | ✅ Full | ✅ Full |
| **MoE support** | ✅ Native | ⚠️ Limited | ⚠️ Limited | ❌ |
| **Flash attention** | ✅ | ✅ | ✅ | ❌ |
| **Quantization types** | 15+ (Q2–Q8) | 3 (3, 4, 8 bit) | 2 (4 bit) | 2 (4, 8 bit) |
| **File size efficiency** | Good | Excellent | Excellent | Poor |
| **Loading speed** | Fast (mmap) | Moderate | Moderate | Slow |

---

## Quantization Levels: A Complete Guide

### The K-Quant Family (GGUF)

The "K" in Q4_K_M stands for "K-quant" — a sophisticated quantization scheme developed for llama.cpp. K-quants use different bit widths for different parts of each tensor, based on importance. This is far more effective than uniform quantization.

#### Quantization Type Table

| Type | Bits/Weight | Description | Quality | Use Case |
|---|---|---|---|---|
| **Q2_K** | 2.56–2.66 | 2-bit K-quant with 4-bit for important weights | Poor | Absolute minimum memory; lowest acceptable quality |
| **Q3_K_S** | 3.21–3.33 | 3-bit K-quant, small | Below average | Very tight memory budgets |
| **Q3_K_M** | 3.33–3.50 | 3-bit K-quant, medium hybrid | Below average | Memory-constrained with slight quality preference |
| **Q3_K_L** | 3.50–3.67 | 3-bit K-quant, large (more 4-bit) | Average | Better than Q3_M, still tight |
| **Q4_0** | 4.00 | 4-bit plain quantization | Average | Legacy; use Q4_K variants instead |
| **Q4_K_S** | 4.00–4.17 | 4-bit K-quant, small | Good | Fast 4-bit, slightly less quality than Q4_K_M |
| **Q4_K_M** | 4.25–4.50 | 4-bit K-quant, medium | **Very Good** | **RECOMMENDED — best quality/size trade-off** |
| **Q4_K_L** | 4.50–4.75 | 4-bit K-quant, large (more 6-bit) | Very Good | Slightly better quality than Q4_K_M |
| **Q5_0** | 5.00 | 5-bit plain quantization | Good | Legacy; use Q5_K variants instead |
| **Q5_K_S** | 5.00–5.17 | 5-bit K-quant, small | Excellent | Near-FP16 quality with ~3× compression |
| **Q5_K_M** | 5.25–5.50 | 5-bit K-quant, medium | Excellent | Best quality for size-conscious users |
| **Q6_K** | 6.00–6.25 | 6-bit K-quant | Near-FP16 | Excellent quality, ~2.5× compression |
| **Q8_0** | 8.00 | 8-bit plain quantization | FP16-level | Virtually no quality loss, ~2× compression |
| **FP16** | 16.00 | Half-precision float | Reference | Original precision, no quantization loss |
| **F32** | 32.00 | Full-precision float | Ground truth | Rarely used; only for research baselines |

### How to Read Quantization Names

The naming convention is:
- **Q** = Quantized
- **Number** = Base bit width (2, 3, 4, 5, 6, 8)
- **K** = K-quant variant (improved over plain quantization)
- **S/M/L** = Size/quality trade-off (Small = smaller/slightly worse, Medium = balanced, Large = larger/better)

So **Q4_K_M** means: 4-bit base, K-quant variant, Medium size/quality balance.

### What About F16?

FP16 is not quantization — it's the standard format used in PyTorch and by most cloud providers. A model in FP16:

- Takes up the most memory (14GB for a 7B model)
- Has no quantization artifacts
- Is the reference for quality comparisons
- Is generally not used in local inference due to large size

Most "Q8_0" quantizations achieve quality virtually indistinguishable from FP16 while using half the memory.

---

## Quality vs. Size: Empirical Data

### Perplexity Degradation

Perplexity is a standard measure of model quality. Lower is better. The following table shows the perplexity increase (on WikiText-2) relative to FP16 for a Llama 3 8B model:

| Quantization | Size (GB) | Perplexity | Degradation |
|---|---|---|---|
| FP16 (reference) | 16.0 | 5.82 | 0.00% |
| Q8_0 | 8.0 | 5.83 | +0.17% |
| Q6_K | 6.0 | 5.85 | +0.52% |
| Q5_K_M | 5.0 | 5.88 | +1.03% |
| Q5_K_S | 4.5 | 5.91 | +1.55% |
| **Q4_K_M** | **3.9** | **5.97** | **+2.58%** |
| Q4_K_S | 3.5 | 6.02 | +3.44% |
| Q3_K_M | 3.0 | 6.21 | +6.70% |
| Q3_K_S | 2.7 | 6.35 | +9.11% |
| Q2_K | 2.3 | 6.82 | +17.2% |

### Perplexity on Larger Models

Larger models are more quantization-tolerant (the degradation percentage is smaller):

| Quantization | Llama 3 8B | Llama 3 70B | Qwen 2.5 72B |
|---|---|---|---|
| FP16 | 5.82 | 4.31 | 4.12 |
| Q8_0 | 5.83 (+0.2%) | 4.32 (+0.2%) | 4.13 (+0.2%) |
| Q4_K_M | 5.97 (+2.6%) | 4.38 (+1.6%) | 4.19 (+1.7%) |
| Q3_K_M | 6.21 (+6.7%) | 4.48 (+3.9%) | 4.28 (+3.9%) |
| Q2_K | 6.82 (+17.2%) | 4.71 (+9.3%) | 4.49 (+9.0%) |

Key insight: **A 70B model at Q4_K_M (39GB) has better perplexity than an 8B model at FP16 (16GB)** — 4.38 vs 5.82. This is why local inference practitioners prefer larger quantized models over smaller full-precision ones.

### Downstream Task Performance

Perplexity doesn't tell the whole story. On downstream tasks (MMLU, HellaSwag, ARC, GSM8K), the degradation is often smaller than perplexity suggests:

| Quantization | MMLU (Accuracy) | GSM8K (Math) | HumanEval (Code) |
|---|---|---|---|
| FP16 | 68.4% | 56.8% | 42.1% |
| Q8_0 | 68.3% (-0.1) | 56.5% (-0.3) | 41.8% (-0.3) |
| Q4_K_M | 67.5% (-0.9) | 55.2% (-1.6) | 40.5% (-1.6) |
| Q4_K_S | 67.1% (-1.3) | 54.5% (-2.3) | 39.8% (-2.3) |
| Q3_K_M | 65.8% (-2.6) | 52.1% (-4.7) | 37.2% (-4.9) |
| Q2_K | 63.2% (-5.2) | 48.3% (-8.5) | 33.5% (-8.6) |

---

## Hardware Requirements by Quantization

### VRAM Requirements

The following table shows approximate VRAM needed to run each model size at each quantization level. Values include model weights plus ~2GB overhead for KV cache (at 4K context).

| Model Size | Q2_K | Q3_K_M | Q4_K_M | Q5_K_M | Q6_K | Q8_0 | FP16 |
|---|---|---|---|---|---|---|---|
| 1B | 0.5 GB | 0.6 GB | 0.8 GB | 0.9 GB | 1.1 GB | 1.3 GB | 2.0 GB |
| 3B | 1.2 GB | 1.5 GB | 1.9 GB | 2.3 GB | 2.7 GB | 3.5 GB | 6.0 GB |
| 7B | 2.3 GB | 3.0 GB | 3.9 GB | 4.8 GB | 5.8 GB | 7.5 GB | 14 GB |
| 8B (Llama 3) | 2.6 GB | 3.4 GB | 4.4 GB | 5.4 GB | 6.5 GB | 8.5 GB | 16 GB |
| 13B | 4.0 GB | 5.1 GB | 6.8 GB | 8.4 GB | 10.1 GB | 13.5 GB | 26 GB |
| 20B | 5.8 GB | 7.5 GB | 9.8 GB | 12.0 GB | 14.5 GB | 19.0 GB | 38 GB |
| 34B | 9.5 GB | 12.5 GB | 16.0 GB | 19.5 GB | 23.5 GB | 31.0 GB | 62 GB |
| 70B | 19 GB | 25 GB | 33 GB | 40 GB | 48 GB | 64 GB | 128 GB |
| 110B | 30 GB | 39 GB | 51 GB | 63 GB | 76 GB | 100 GB | 200 GB |
| 180B (MoE) | 30 GB | 39 GB | 51 GB | 63 GB | 76 GB | 100 GB | 200 GB |

### Hardware Matching Guide

| Hardware | VRAM | Max Q4_K_M | Max Q3_K_M | Recommended Model |
|---|---|---|---|---|
| RTX 3060 12GB | 12 GB | 20B | 34B | Qwen 2.5 14B (Q4_K_M) |
| RTX 4060 Ti 16GB | 16 GB | 34B | 34B | Qwen 2.5 32B (Q3_K_M) |
| RTX 4070 12GB | 12 GB | 20B | 34B | Qwen 2.5 14B (Q4_K_M) |
| RTX 4080 16GB | 16 GB | 34B | 34B | Qwen 2.5 32B (Q3_K_M) |
| RTX 4090 24GB | 24 GB | 34B | 70B | Qwen 2.5 72B (Q3_K_M) or 34B (Q4_K_M) |
| RTX 5090 32GB | 32 GB | 70B | 70B | Llama 3 70B (Q4_K_M) |
| RTX 6000 Ada 48GB | 48 GB | 70B | 110B | Qwen 2.5 72B (Q5_K_M) |
| A100 80GB | 80 GB | 110B | 180B | Qwen 2.5 110B (Q4_K_M) |
| M1 Mac (16GB) | 16 GB (unified) | 13B | 20B | Mistral 7B + Qwen 2.5 14B |
| M2 Mac (24GB) | 24 GB (unified) | 20B | 34B | Qwen 2.5 32B (Q3_K_M) |
| M3 Max (64GB) | 64 GB (unified) | 70B | 110B | Llama 3 70B (Q4_K_M) |
| M3 Ultra (192GB) | 192 GB (unified) | 180B | 180B | Qwen 2.5 110B (Q5_K_M) |
| 2× RTX 4090 | 48 GB (total) | 70B | 110B | 70B (Q4_K_M) with tensor parallelism |
| 4× RTX 4090 | 96 GB (total) | 110B | 180B | Qwen 2.5 110B (Q4_K_M) |

---

## Performance Benchmarks

### Tokens per Second by Hardware and Quantization

#### NVIDIA RTX 4090 (24GB VRAM, CUDA)

| Model Size | Q2_K | Q3_K_M | Q4_K_M | Q5_K_M | Q6_K | Q8_0 |
|---|---|---|---|---|---|---|
| 7B | 140 t/s | 130 t/s | 110 t/s | 95 t/s | 80 t/s | 65 t/s |
| 13B | 80 t/s | 70 t/s | 60 t/s | 50 t/s | 42 t/s | 35 t/s |
| 34B | 35 t/s | 30 t/s | 25 t/s | 20 t/s | 17 t/s | 14 t/s |
| 70B | 15 t/s | 13 t/s | 11 t/s | — | — | — |

#### Apple M2 Ultra (192GB Unified Memory, Metal)

| Model Size | Q2_K | Q3_K_M | Q4_K_M | Q5_K_M | Q6_K | Q8_0 |
|---|---|---|---|---|---|---|
| 7B | 50 t/s | 45 t/s | 40 t/s | 35 t/s | 30 t/s | 25 t/s |
| 13B | 30 t/s | 28 t/s | 25 t/s | 22 t/s | 18 t/s | 15 t/s |
| 34B | 14 t/s | 13 t/s | 11 t/s | 10 t/s | 8 t/s | 7 t/s |
| 70B | 7 t/s | 6 t/s | 5 t/s | — | — | — |

#### AMD RX 7900 XTX (24GB VRAM, ROCm)

| Model Size | Q2_K | Q3_K_M | Q4_K_M | Q5_K_M | Q6_K | Q8_0 |
|---|---|---|---|---|---|---|
| 7B | 105 t/s | 98 t/s | 85 t/s | 72 t/s | 60 t/s | 50 t/s |
| 13B | 60 t/s | 55 t/s | 48 t/s | 40 t/s | 33 t/s | 28 t/s |
| 34B | 27 t/s | 24 t/s | 20 t/s | 16 t/s | — | — |

#### CPU Only (AMD Ryzen 9 7950X, DDR5-6000)

| Model Size | Q2_K | Q3_K_M | Q4_K_M | Q5_K_M | Q6_K | Q8_0 |
|---|---|---|---|---|---|---|
| 7B | 18 t/s | 15 t/s | 12 t/s | 10 t/s | 8 t/s | 6 t/s |
| 13B | 9 t/s | 8 t/s | 6 t/s | 5 t/s | 4 t/s | 3 t/s |
| 34B | 4 t/s | 3 t/s | 2.5 t/s | — | — | — |

### Key Performance Observations

1. **Lower quantization = faster inference.** Smaller weights mean less data to transfer from VRAM to compute units. Q2_K is typically 2× faster than FP16 on the same hardware.

2. **GPU memory bandwidth is the bottleneck.** Inference speed is limited by how fast weights can be moved from VRAM to compute units. Faster memory (HBM3 > GDDR6X > GDDR6 > DDR5) directly translates to higher tokens/second.

3. **Apple Silicon is bandwidth-constrained.** Apple's unified memory has lower bandwidth than dedicated GPU VRAM (e.g., M2 Ultra: 800 GB/s vs. RTX 4090: 1,008 GB/s). This gap is larger for memory-intensive operations and results in proportionally lower tokens/second for large models.

4. **CPU inference is practical for small models.** A 7B model at Q4_K_M on a modern CPU yields 10–15 tokens/second — usable for many applications, especially with streaming.

5. **Batch processing is much faster.** Ollama and llama.cpp can process prompts in batches, achieving much higher throughput for batch jobs compared to single-stream inference.

---

## Obtaining GGUF Models

### From HuggingFace

HuggingFace hosts hundreds of thousands of GGUF model variants. The most popular quantizers are:

- **TheBloke**: The original large-scale quantizer (now run by community maintainers)
- **Bartowski**: Current leading quantizer with regular updates
- **MaziyarPanahi**: Community quantizer with broad model support
- **CompendiumLabs**: Specialized in embedding model GGUFs

```bash
# Download via HuggingFace CLI
pip install huggingface-hub

# Download a specific GGUF file
huggingface-cli download bartowski/Qwen2.5-32B-Instruct-GGUF \
  --include "Q4_K_M/*" \
  --local-dir ./models/

# Or via wget
wget https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Q4_K_M/Qwen2.5-32B-Instruct-Q4_K_M.gguf
```

### From Ollama

```bash
# Ollama pulls the appropriate GGUF automatically
ollama pull qwen2.5:7b-q4_K_M

# Pre-quantized models are available in the library
ollama pull llama3.2:8b-q4_K_M
ollama pull deepseek-coder:33b-q3_K_L

# List available quantizations for a model
ollama show qwen2.5:7b --versions
```

### From the Ollama Library

Ollama's model library includes pre-configured quantizations. The naming convention is:

- `model:tag` — e.g., `llama3.2:8b` (default quantization, usually Q4_K_M)
- `model:tag-quant` — e.g., `llama3.2:8b-q8_0` (specific quantization)

Common tags:
- `:latest` — Default quantization (usually Q4_K_M)
- `:q4_k_m` — Recommended balance of quality and size
- `:q3_k_m` — Smaller, good for limited VRAM
- `:q2_k` — Smallest, for very constrained hardware
- `:q8_0` — Highest quality, largest size

---

## Custom Quantization

### Quantizing Your Own Models

If you have a model that isn't available in GGUF format, you can quantize it yourself using llama.cpp.

#### Step 1: Clone and Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with CUDA support
make LLAMA_CUDA=1 -j

# Or for Metal (macOS)
make LLAMA_METAL=1 -j

# Or for ROCm
make LLAMA_HIPBLAS=1 -j
```

#### Step 2: Convert Model to GGUF

```bash
# For HuggingFace models (FP16)
python convert_hf_to_gguf.py /path/to/model --outfile model-fp16.gguf

# For PyTorch checkpoints
python convert.py /path/to/model \
  --outfile model-fp16.gguf \
  --model-name mymodel
```

#### Step 3: Quantize

```bash
# Single quantization
./quantize model-fp16.gguf model-q4_k_m.gguf q4_k_m

# Create multiple quantizations at once
for q in q2_k q3_k_m q4_k_m q5_k_m q6_k q8_0; do
  ./quantize model-fp16.gguf "model-${q}.gguf" "$q"
done
```

#### Step 4: Verify

```bash
# Test the quantized model
./main -m model-q4_k_m.gguf -p "Hello, how are you?" -n 50

# Run perplexity test
./perplexity -m model-q4_k_m.gguf -f test.txt
```

### Important Quantization Settings

```bash
# Pure (non-K) quantization types
./quantize model.gguf model-q4_0.gguf q4_0
./quantize model.gguf model-q5_0.gguf q5_0
./quantize model.gguf model-q8_0.gguf q8_0

# K-quant types (recommended)
./quantize model.gguf model-q2_k.gguf q2_k
./quantize model.gguf model-q3_k_m.gguf q3_k_m
./quantize model.gguf model-q4_k_m.gguf q4_k_m  # Recommended
./quantize model.gguf model-q5_k_m.gguf q5_k_m
./quantize model.gguf model-q6_k.gguf q6_k
```

### Using Importance Matrices

For better quality at very low bit widths (Q2, Q3), you can provide an importance matrix:

```bash
# Generate importance matrix from calibration data
./imatrix -m model-fp16.gguf -f calibration.txt -o imatrix.dat

# Quantize with importance matrix (better quality)
./quantize model-fp16.gguf model-q2_k.gguf q2_k --imatrix imatrix.dat
./quantize model-fp16.gguf model-q3_k_m.gguf q3_k_m --imatrix imatrix.dat
```

This produces significantly better quality for aggressive quantizations, often reducing the quality gap by 30–50%.

### Quantizing MoE Models

Mixture-of-Experts models (Mixtral, Qwen2.5-MoE, DeepSeek-MoE) require special handling:

```bash
# MoE quantization is supported in recent versions
./quantize model-fp16.gguf model-q4_k_m.gguf q4_k_m --moe

# Or explicitly
./quantize model-fp16.gguf model-q4_k_m.gguf q4_k_m --allow-moe
```

MoE models benefit from selective quantization: experts can be quantized more aggressively than shared parameters:

```python
# This requires manual configuration
# typical: experts at Q3_K_M, shared layers at Q4_K_M or Q5_K_M
```

---

## Running GGUF Models with llama.cpp

### Basic Usage

```bash
# Simple generation
./main -m model-q4_k_m.gguf \
  -p "What is the capital of France?" \
  -n 256 \
  -t 8

# With GPU offloading (use GPU for layers 0-35)
./main -m model-q4_k_m.gguf \
  -p "Explain quantum computing" \
  -n 512 \
  -ngl 35 \
  -t 8

# Interactive chat mode
./main -m model-q4_k_m.gguf \
  -i \
  --chatml \
  -ngl 35
```

### Advanced Options

```bash
# Performance tuning
./main -m model.gguf \
  -p "Your prompt" \
  -n 512 \           # Max tokens to generate
  -t 8 \             # Thread count
  -ngl 35 \          # GPU layers (0 = CPU only)
  -c 4096 \          # Context size
  -b 512 \           # Batch size for prompt processing
  --mlock \          # Lock model in RAM (prevents swapping)
  --no-mmap \        # Don't use memory mapping
  --flash-attn \     # Use flash attention (saves memory)
  --cont-batching \  # Enable continuous batching

# Quality control
  --temp 0.7 \       # Temperature
  --top-k 40 \       # Top-K sampling
  --top-p 0.9 \      # Top-P sampling
  --repeat-penalty 1.1 \  # Repeat penalty
```

### Server Mode

```bash
# Start the HTTP server
./server -m model-q4_k_m.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 35 \
  --embeddings

# Then use any OpenAI-compatible client
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Using GGUF Models with Ollama

```bash
# Option 1: Import a GGUF file directly
ollama create mymodel --model /path/to/model.gguf

# Option 2: Create a Modelfile
cat > Modelfile << 'EOF'
FROM /path/to/model-q4_k_m.gguf

PARAMETER temperature 0.7
PARAMETER num_ctx 4096

SYSTEM "You are a helpful assistant."
EOF

ollama create mymodel -f Modelfile

# Use the model
ollama run mymodel
```

---

## Choosing the Right Quantization

### Decision Framework

```
Q1: What hardware do you have?
  ↓
Q2: How much VRAM / unified memory?
  ↓
Q3: What model size do you need?
  ↓
Q4: How important is response speed?
  ↓
Q5: How important is output quality?
  ↓
→ Choose quantization
```

### Practical Guidelines

#### "I want the best quality possible"

Use the highest quantization that fits in your VRAM with room for context:

```
VRAM Available → Max Quantization
 8-12GB    → Q4_K_M (up to 13B) or Q3_K_M (up to 20B)
 16-24GB   → Q4_K_M (up to 34B) or Q3_K_M (up to 70B)
 32-48GB   → Q5_K_M (up to 34B) or Q4_K_M (up to 70B)
 64-96GB   → Q5_K_M (up to 70B) or Q4_K_M (up to 110B)
 128GB+    → Q6_K or Q8_0 (any model)
```

#### "I need the fastest possible responses"

Use the lowest quantization that meets your quality threshold:

```
Speed Priority → Best Quantization
 Maximum speed  → Q2_K or Q3_K_M (smallest weights = fastest)
 Good speed     → Q4_K_M (4× compression, 2× speed vs FP16)
```

#### "I have limited VRAM and want the largest model possible"

```
Available VRAM → Max Model (at Q4_K_M)
 8GB  → 7B (comfortable) or 13B (tight)
12GB  → 13B (comfortable) or 20B (tight)
16GB  → 20B (comfortable) or 34B (tight)
24GB  → 34B (comfortable) or 70B (tight, Q3_K_M)
32GB  → 34B (very comfortable) or 70B (Q4_K_M)
48GB  → 70B (comfortable)
80GB  → 70B (very comfortable) or 110B (tight)
```

#### "I want to use CPU inference only"

```
CPU + RAM → Best Model
 16GB RAM → 3B or 7B (Q4_K_M)
 32GB RAM → 7B (Q4_K_M) or 13B (Q2_K)
 64GB RAM → 13B (Q4_K_M) or 20B (Q3_K_M)
128GB RAM → 34B (Q4_K_M) or 70B (Q2_K)
```

### General Recommendations

| Use Case | Recommended Quantization | Why |
|---|---|---|
| General chat, Q&A | **Q4_K_M** | Best quality/size trade-off |
| Creative writing, storytelling | **Q5_K_M** or Q6_K | Higher quality for nuanced outputs |
| Coding, math, reasoning | **Q4_K_M** or Q5_K_M | Precision matters for code/math |
| Classification, simple tasks | **Q3_K_M** or Q4_K_S | Quality less critical |
| Batch processing, automation | **Q4_K_M** | Good speed and quality |
| Mobile/low-power devices | **Q3_K_M** or Q2_K | Minimize memory and compute |
| Research, benchmarking | **Q8_0** or FP16 | No quantization artifacts |
| Maximum throughput | **Q2_K** | Fastest generation, smallest size |

---

## The Future of GGUF and Quantization

### Current Trends

1. **More quantization types**: The K-quant family continues to evolve. New variants offer better quality at very low bit widths.

2. **Quantization-aware training (QAT)**: Models trained with quantization in mind from the start produce significantly better quantized versions. This is becoming standard practice for new model releases.

3. **Dynamic quantization**: Per-layer, per-token, and even per-weight quantization that adapts to input content. This promises better quality without increasing model size.

4. **NVFP4 on Blackwell**: NVIDIA's Blackwell architecture introduces 4-bit floating point support in hardware, which will further improve quantized inference performance.

5. **Extreme quantization (1.5–2 bit)**: Research into ternary and binary weight networks continues. While not yet practical for general use, 2-bit models are approaching usability.

### Emerging Formats

- **GGUF IQ (Importance-aware Quantization)**: Newer variant that uses importance matrices for better quality at ultra-low bit widths
- **EXL2**: ExLlamaV2's native quantization format, competitive with GGUF for GPU-only inference
- **AQLM**: Additive quantization for LLMs, promising better quality at 2–3 bits
- **QuIP#**: Lattice-based quantization with state-of-the-art quality at low bit widths

### Legacy Formats

| Format | Status | Notes |
|---|---|---|
| GGML | Deprecated (2023) | Replaced by GGUF |
| GPTQ | Stable | Still used, primarily with ExLlamaV2 |
| AWQ | Stable | Growing adoption, integrated with vLLM |
| bitsandbytes | Stable | Primarily for fine-tuning (QLoRA) |

---

## Troubleshooting Quantized Models

| Problem | Likely Cause | Solution |
|---|---|---|
| Model loads but produces gibberish | Wrong tokenizer or compatibility | Check model family compatibility; try different GGUF source |
| CUDA out of memory | VRAM insufficient | Use higher quantization (lower memory); reduce context size; enable flash attention |
| Slow prompt processing | Large batch size | Reduce batch size (`-b 256` or `-b 128`) |
| Slow generation | Low GPU offloading | Increase `-ngl`; check GPU utilization |
| Model fails to load | Corrupted file | Re-download; check SHA256 checksum |
| Metal errors on macOS | Outdated llama.cpp | Update llama.cpp; check macOS version |
| ROCm not detected | Driver issues | Verify `rocminfo`; install ROCm properly |
| Output differs between quantizations | Expected behavior | Lower quantizations introduce more randomness; this is normal |
| Sudden quality drop after update | Model or quantization version change | Use same quantization source; check for benchmark regressions |

---

## Best Practices Summary

1. **Start with Q4_K_M** for virtually all use cases. It's the sweet spot.
2. **Go higher (Q5_K_M, Q6_K) for creative tasks** where output quality is paramount.
3. **Go lower (Q3_K_M, Q2_K) only when VRAM-constrained** and you need a larger model.
4. **Larger model at lower quantization > smaller model at higher quantization**. Always prefer a 70B at Q3_K_M over a 34B at Q8_0 if they fit in your VRAM.
5. **Leave 2–4GB VRAM headroom** for the KV cache (more for long context windows).
6. **Enable flash attention** to reduce KV cache memory usage by ~50% for long contexts.
7. **Test before committing** to a quantization level for your specific use case.
8. **Use importance matrices** when quantizing your own models for better quality at low bit widths.
9. **Keep llama.cpp updated** for the latest quantization kernel optimizations.
10. **Download from trusted sources** (bartowski, TheBloke, official Ollama library) and verify checksums.

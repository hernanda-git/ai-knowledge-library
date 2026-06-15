# 06 — Hardware for Local Inference

## Overview

Running large language models locally requires carefully matched hardware. Unlike cloud inference, where you rent someone else's GPUs, local inference means you need to buy, build, and maintain the silicon yourself. This guide covers every major hardware option available in 2026 — from consumer GPUs to Apple Silicon, NPUs, and CPU-only inference — and helps you match hardware to model sizes, use cases, and budgets.

The landscape has shifted dramatically since the first local LLM experiments in 2023. Today we have dedicated AI accelerators in phones and laptops, unified memory architectures that can hold 192 GB+ in a single machine, and inference engines that can run 70B-parameter models on a single consumer GPU via aggressive quantization.

---

## 1. GPU Hardware for Local Inference

### 1.1 NVIDIA GeForce RTX 4090 (Ada Lovelace)

The RTX 4090 has been the gold standard for local inference since its 2022 launch. Even in 2026, it remains one of the most popular choices for running 7B-70B parameter models.

| Specification | RTX 4090 |
|---|---|
| VRAM | 24 GB GDDR6X |
| Memory Bandwidth | 1,008 GB/s |
| CUDA Cores | 16,384 |
| Tensor Cores (4th Gen) | 512 |
| FP16 TFLOPS | 82.6 |
| TDP | 450W |
| Price (new, 2026) | ~$1,600–$1,800 |
| Price (used) | ~$1,200–$1,400 |

**Typical workloads:**
- 7B models at Q4_K_M: 90+ tokens/s
- 13B models at Q4_K_M: 45–55 tokens/s
- 34B models at Q4_K_M: 18–22 tokens/s
- 70B models at Q4_K_M: 8–12 tokens/s (fits with ~18 GB VRAM usage)

**Limitations:** The 24 GB VRAM ceiling means 70B models at 4-bit barely fit, and 120B+ models are impossible without offloading to system RAM. Multi-GPU setups are needed for larger models.

---

### 1.2 NVIDIA GeForce RTX 5090 (Blackwell)

Released in late 2025, the RTX 5090 is the new flagship for local inference. Its 32 GB VRAM is a massive upgrade over the 4090.

| Specification | RTX 5090 |
|---|---|
| VRAM | 32 GB GDDR7 |
| Memory Bandwidth | 1,792 GB/s |
| CUDA Cores | 21,760 |
| Tensor Cores (5th Gen) | 680 |
| FP16 TFLOPS | 104.9 |
| TDP | 575W |
| Price (new) | ~$2,200–$2,500 |

**Typical workloads:**
- 7B models at Q4_K_M: 110+ tokens/s
- 13B models at Q4_K_M: 60–70 tokens/s
- 34B models at Q4_K_M: 28–32 tokens/s
- 70B models at Q4_K_M: 15–18 tokens/s (comfortably fits)
- 120B models at Q3_K_M: 8–10 tokens/s (fits with careful quant selection)

**Advantage over 4090:** 33% more VRAM plus higher bandwidth means larger models run entirely on GPU without CPU offloading. The 32 GB is enough for quantized 70B models with context windows up to 32K tokens.

---

### 1.3 NVIDIA RTX 6000 Ada / A6000 (Professional)

For users who need more than 32 GB, professional cards offer 48 GB VRAM at much higher prices.

| Specification | RTX 6000 Ada | A6000 |
|---|---|---|
| VRAM | 48 GB GDDR6 | 48 GB GDDR6 |
| Memory Bandwidth | 960 GB/s | 768 GB/s |
| TDP | 300W | 300W |
| Price | ~$6,800 | ~$4,500 (used: ~$3,000) |

**Typical workloads:**
- 70B models at Q4_K_M: 14–16 tokens/s (comfortable fit with large context)
- 120B models at Q4_K_M: 7–9 tokens/s
- 180B models at Q3_K_M: 5–7 tokens/s

These cards are overpriced for inference-only workloads compared to consumer options, but the 48 GB VRAM is indispensable for running 120B+ models on a single GPU.

---

### 1.4 Multi-GPU Configurations (NVIDIA)

For the largest models, multiple GPUs are required. Key configuration options:

**Dual RTX 4090 (48 GB total):**
- Cost: ~$3,200–$3,600
- Power: 900W system draw
- Supports: 70B Q4_K_M comfortably, 120B Q3_K_M, 180B Q2_K with CPU offload
- Performance: 70B at ~18–22 tokens/s, 120B at ~8–10 tokens/s

**Dual RTX 5090 (64 GB total):**
- Cost: ~$4,800–$5,500
- Power: 1,150W system draw
- Supports: 120B Q4_K_M, 180B Q3_K_M, 405B Q2_K with some CPU offload
- Performance: 70B at ~28–32 tokens/s, 120B at ~14–16 tokens/s

**Four RTX 3090/4090s (96 GB total):**
- Cost: ~$4,000–$7,000
- Supports: 180B Q4_K_M, 405B Q2_K
- Performance: 405B at 3–5 tokens/s

**NVLink considerations:** NVLink is not supported on RTX 4090/5090 (requires RTX 6000 or A-series). For inference through llama.cpp or vLLM, the model is split across GPUs via tensor parallelism, which works over PCIe without NVLink. The performance penalty is roughly 5–10% versus NVLink for inference workloads.

---

### 1.5 AMD GPUs for Local Inference

AMD has become a viable option for local inference, particularly with ROCm 6.x and the RX 7000 series.

| Specification | RX 7900 XTX | RX 9070 XT (RDNA 4) |
|---|---|---|
| VRAM | 24 GB GDDR6 | 16 GB GDDR6 |
| Memory Bandwidth | 960 GB/s | 800 GB/s |
| FP16 TFLOPS | 61 | 55 |
| TDP | 355W | ~300W |
| Price | ~$900–$1,000 | ~$600–$700 |
| ROCm Support | Full (ROCm 6.2+) | Partial (RDNA 4 support maturing) |

**Performance comparison vs NVIDIA:**
- ROCm backend in llama.cpp: ~75–85% of equivalent NVIDIA performance
- Vulkan backend: ~65–75% of NVIDIA
- vLLM with ROCm: ~80–90% of CUDA performance for batch inference

**Limitations on AMD:**
- Flash attention support still catching up
- Some quantization types not optimized (IQ4_NL, etc.)
- Mixtral MOE models can have unexpected behavior on ROCm
- RDNA 4 (RX 9070 XT) support is ongoing — check per-engine compatibility

**AMD Instinct (Data Center):**
- MI50 (16 GB): good budget option at ~$300 used, ~40 tokens/s on 7B
- MI100 (32 GB HBM2): ~$600 used, solid for 70B models
- MI250 (128 GB, dual-die): ~$3,000 used, excellent for large models but complex setup
- MI300X (192 GB HBM3): $10,000+, enterprise-grade, probably overkill for local inference

**Bottom line on AMD:** The RX 7900 XTX offers the best VRAM-per-dollar of any consumer GPU. At $900–$1,000 for 24 GB, it's significantly cheaper than a 4090 for the same VRAM capacity. Performance lag is real but shrinking with each ROCm release. For budget-conscious builders who need 24 GB, AMD is now a legitimate option.

---

### 1.6 Apple Silicon for Local Inference

Apple Silicon has become a powerhouse for local AI inference thanks to its unified memory architecture (UMA), where CPU, GPU, and NPU share the same memory pool.

**The Unified Memory Advantage:**
With Apple Silicon, VRAM is system RAM. Every GB you install is available as GPU memory. This means:
- A Mac Studio with 192 GB of unified memory can hold a 180B Q4_K_M model entirely in GPU-accessible memory
- No PCIe bottleneck between CPU and GPU
- Lower power consumption (80–150W for an entire system)

**Current Apple Silicon lineup (2026):**

| Chip | Max Memory | GPU Cores | Neural Engine | Price Range |
|---|---|---|---|---|
| M4 | 32 GB | 10-core | 16-core | $1,400–$2,000 (Mac Mini/Pro/MacBook Pro) |
| M4 Pro | 48 GB | 20-core | 16-core | $2,000–$3,000 |
| M4 Max | 128 GB | 40-core | 16-core | $3,000–$5,000 |
| M4 Ultra | 192 GB | 80-core | 32-core | $5,000–$8,000 |
| M3 Ultra | 192 GB | 80-core | 32-core | $4,000–$7,000 (previous gen, discounted) |

**Performance benchmarks (llama.cpp, Metal backend):**

| Model | M4 Max (128 GB) | M4 Ultra (192 GB) | M3 Ultra (192 GB) |
|---|---|---|---|
| 7B Q4_K_M | 80–90 t/s | 110–120 t/s | 90–100 t/s |
| 13B Q4_K_M | 40–50 t/s | 60–70 t/s | 50–60 t/s |
| 34B Q4_K_M | 18–22 t/s | 28–32 t/s | 22–26 t/s |
| 70B Q4_K_M | 8–10 t/s | 14–16 t/s | 12–14 t/s |
| 120B Q4_K_M | N/A (out of memory) | 8–10 t/s | 7–9 t/s |
| 180B Q4_K_M | N/A | 5–6 t/s | 4–5 t/s |

**Key observations:**
- Apple Silicon excels at holding large models thanks to high memory capacity
- Token throughput is 30–50% lower than a 4090 for the same model size, due to lower memory bandwidth
- M4 Ultra's bandwidth (~1,200 GB/s) is competitive with RTX 4090 (~1,008 GB/s)
- M3 Ultra bandwidth (~800 GB/s) is the bottleneck for large models
- Neural Engine is NOT used by llama.cpp or most inference engines — models run on the GPU via Metal

**Apple Neural Engine (ANE):**
The 16-core or 32-core Neural Engine is designed for on-device ML inference but is largely inaccessible to third-party LLM inference. CoreML can use it, and Apple's internal models leverage it, but open-source inference frameworks rarely target ANE. This is a missed opportunity — ANE could potentially offer very efficient inference at much lower power than the GPU.

**Best use cases for Apple Silicon:**
- Single-system deployments needing 70B+ models
- Low-power always-on inference servers
- Development and experimentation with large models
- Users who also need a workstation for other tasks

---

### 1.7 Intel NPU and Arc GPUs

Intel entered the discrete GPU market with Arc Alchemist and continues with Battlemage (2025+) and Celestial (2026). Intel also has built-in NPUs on Meteor Lake, Arrow Lake, and Lunar Lake processors.

**Intel Arc GPUs:**

| Specification | Arc A770 16 GB | Arc B580 12 GB | Arc B770 (2026) |
|---|---|---|---|
| VRAM | 16 GB GDDR6 | 12 GB GDDR6 | 16–20 GB GDDR6X |
| Memory Bandwidth | 560 GB/s | 456 GB/s | ~700 GB/s |
| Price | ~$250 used | ~$250 new | ~$400–$500 (estimated) |
| XPX Support | Limited | Improved | Full |
| Inference Quality | Varies by model | Good | Expected best-in-class for Intel |

**Intel NPU (Neural Processing Unit):**

| Generation | TOPS | Available In | Use Case |
|---|---|---|---|
| Meteor Lake NPU | ~10 TOPS (INT8) | Core Ultra 1st gen | Very small models, always-on voice |
| Arrow Lake NPU | ~13 TOPS (INT8) | Core Ultra 200S | Small 1–3B models |
| Lunar Lake NPU | ~48 TOPS (INT8) | Core Ultra 200V | 3–7B models at low speed |
| Panther Lake NPU | ~100+ TOPS (INT8) | Core Ultra 300 (2026) | 7B+ models possible |

**Real-world NPU performance:**
- Lunar Lake NPU: Phi-3-mini (3.8B) at ~15–20 tokens/s — usable but not fast
- Models must be converted to OpenVINO format
- Very power-efficient: 1–3W for small model inference
- Cannot run large models (>7B) due to DRAM and compute constraints

**Intel inference summary:**
Intel hardware is viable for small models (<7B) and entry-level inference. Arc GPUs work through the SYCL backend in llama.cpp but lag NVIDIA significantly in performance and model compatibility. The NPU is interesting for always-on, low-power edge scenarios but cannot replace a GPU for serious LLM work.

---

### 1.8 Qualcomm AI Engine (Snapdragon X Elite / Gen 3)

Qualcomm's Snapdragon X Elite and Snapdragon 8 Gen 3 include Hexagon NPUs designed for on-device AI.

| Specification | Snapdragon X Elite | Snapdragon 8 Gen 3 |
|---|---|---|
| NPU TOPS | 45 TOPS (INT8) | 26 TOPS (INT8) |
| GPU | Adreno (4.6 TFLOPS FP16) | Adreno 750 (3.0 TFLOPS) |
| Target Platform | Windows on ARM Laptops | Android Phones |
| LLM Support | ONNX Runtime, Qualcomm AI Hub | Qualcomm AI Hub, MediaPipe |
| Max Model | 7B Q4 (fits in 8 GB) | 3–7B Q4 (fits in LPDDR5) |

**Performance:**
- Snapdragon X Elite: 7B Q4 model at ~20–30 tokens/s (NPU + GPU hybrid)
- Snapdragon 8 Gen 3: 7B Q4 at ~10–15 tokens/s
- Very power-efficient (<5W for NPU inference)

**Considerations:**
- Extremely limited ecosystem — few frameworks support Qualcomm NPU directly
- llama.cpp works on Snapdragon via the Vulkan backend but doesn't use the NPU
- Qualcomm AI Hub provides SDKs but requires model conversion
- Best suited for mobile/edge deployment, not server-grade inference

---

## 2. CPU-Only Inference

Inference on CPU is slower than GPU but eliminates GPU cost entirely. Modern CPUs with AVX-512 and VNNI/AMX instructions have narrowed the gap significantly.

### 2.1 CPU Performance by Architecture

| CPU | Cores | Memory | 7B Q4_K_M | 13B Q4_K_M | 34B Q4_K_M | 70B Q4_K_M |
|---|---|---|---|---|---|---|
| AMD Ryzen 7950X | 16C/32T | DDR5-6000 | 8–10 t/s | 4–6 t/s | 2–3 t/s | 0.8–1.2 t/s |
| AMD Ryzen 9950X (2025) | 16C/32T | DDR5-6400 | 10–12 t/s | 6–8 t/s | 3–4 t/s | 1.2–1.8 t/s |
| Intel Core Ultra 9 285K | 24C/24T | DDR5-6400 | 9–11 t/s | 5–7 t/s | 2–3 t/s | 1.0–1.5 t/s |
| AMD Threadripper 7980X | 64C/128T | DDR5-5200 (8-channel) | 20–25 t/s | 12–15 t/s | 6–8 t/s | 2–3 t/s |
| Intel Xeon w9-3495X | 56C/112T | DDR5-4800 (4-channel) | 15–20 t/s | 9–12 t/s | 4–6 t/s | 1.5–2.5 t/s |
| Apple M4 Ultra | 32 CPU cores | Unified (800 GB/s) | 40–50 t/s | 22–28 t/s | 10–14 t/s | 4–6 t/s |
| AWS Graviton4 | 96 cores | DDR5 | 12–16 t/s | 7–10 t/s | 3–5 t/s | 1.2–2 t/s |

### 2.2 Key CPU Inference Facts

- **Memory bandwidth is the bottleneck.** CPU inference is almost always bandwidth-bound. Each token requires reading the entire model from RAM. DDR5-6000 provides ~96 GB/s per channel (dual-channel = 192 GB/s). Compare to RTX 4090's 1,008 GB/s.
- **AVX-512 matters.** CPUs with AVX-512 (Intel Ice Lake and newer, AMD Zen 4 and newer) see 20–40% performance improvement over AVX2-only CPUs for inference.
- **AMX (Advanced Matrix Extensions)** on Intel Sapphire Rapids and newer provides matrix-multiply acceleration, reducing CPU inference latency by another 30–50%.
- **Memory channels matter more than core count.** Threadripper's 8-channel memory gives it a massive advantage over desktop CPUs with 2-channel memory.
- **Quantization is critical for CPU inference.** Q4_K_M or Q3_K_M quantization reduces memory bandwidth requirements by 75% compared to FP16.

### 2.3 When to Choose CPU Over GPU for Inference

- **Batch inference / async workloads:** If you don't need real-time responses (e.g., batch document processing), CPU inference at 2–5 t/s may be adequate.
- **Already have high-end workstation:** Adding inference to an existing Threadripper or Xeon workstation costs $0 extra.
- **Need very large context windows:** CPU can handle 128K+ context easier since system RAM is abundant.
- **Multitenant low-throughput serving:** One CPU can serve many concurrent users at 1–2 t/s each, which is fine for retrieval-augmented generation (RAG) where the LLM just summarizes retrieved context.

---

## 3. RAM Requirements Per Model Size

The single most important hardware specification for local inference is memory capacity — both system RAM and (more importantly) GPU VRAM.

### 3.1 VRAM Requirements by Quantization Level

| Model Size | Q2_K (2-bit) | Q3_K_M (3-bit) | Q4_K_M (4-bit) | Q5_K_M (5-bit) | Q6_K (6-bit) | Q8_0 (8-bit) | FP16 (16-bit) |
|---|---|---|---|---|---|---|---|
| 1B | 0.5 GB | 0.7 GB | 0.9 GB | 1.1 GB | 1.3 GB | 1.5 GB | 2.5 GB |
| 3B | 1.3 GB | 1.8 GB | 2.3 GB | 2.8 GB | 3.3 GB | 4.0 GB | 6.5 GB |
| 7B | 2.8 GB | 3.8 GB | 4.8 GB | 5.8 GB | 6.8 GB | 8.0 GB | 14.0 GB |
| 8B | 3.2 GB | 4.3 GB | 5.5 GB | 6.6 GB | 7.7 GB | 9.0 GB | 16.0 GB |
| 13B | 5.0 GB | 6.8 GB | 8.5 GB | 10.3 GB | 12.0 GB | 14.0 GB | 25.0 GB |
| 20B | 7.5 GB | 10.0 GB | 12.5 GB | 15.0 GB | 17.5 GB | 20.0 GB | 38.0 GB |
| 34B | 12.5 GB | 17.0 GB | 21.0 GB | 25.5 GB | 29.5 GB | 34.0 GB | 65.0 GB |
| 70B | 25.0 GB | 34.0 GB | 42.0 GB | 51.0 GB | 59.0 GB | 68.0 GB | 130.0 GB |
| 120B | 45.0 GB | 60.0 GB | 75.0 GB | 90.0 GB | 105.0 GB | 120.0 GB | 230.0 GB |
| 180B | 67.0 GB | 90.0 GB | 112.0 GB | 135.0 GB | 157.0 GB | 180.0 GB | 340.0 GB |
| 405B | 150.0 GB | 200.0 GB | 250.0 GB | 300.0 GB | 350.0 GB | 400.0 GB | 760.0 GB |

**Notes:**
- Figures include ~0.5–1 GB overhead for KV cache at 2K context
- Each additional 2K tokens of context adds roughly: 0.1–0.5 GB for 7B, 0.3–1.2 GB for 34B, 0.8–2.5 GB for 70B
- At 128K context, add 6–40 GB extra VRAM depending on model size
- MOE models (Mixtral, DBRX, DeepSeek) use less VRAM per parameter because only part of the model activates per token

### 3.2 MOE Model VRAM Requirements

| Model | Active Params | Total Params | Q4_K_M VRAM |
|---|---|---|---|
| Mixtral 8x7B | 12.9B | 46.7B | ~26 GB |
| Mixtral 8x22B | 39B | 141B | ~78 GB |
| DBRX | 36B | 132B | ~73 GB |
| DeepSeek V2 | 21B | 236B | ~65 GB (Q4_K_M, plus 15 GB for KV cache) |
| Qwen1.5-MoE | 2.7B | 14.3B | ~8 GB |

### 3.3 System RAM Recommendations

| Model Size | Minimum RAM | Recommended RAM | With GPU Offloading |
|---|---|---|---|
| Up to 7B | 16 GB | 32 GB | 16 GB |
| 13B–20B | 32 GB | 64 GB | 32 GB |
| 34B–70B | 64 GB | 128 GB | 64 GB |
| 120B–180B | 128 GB | 256 GB | 128 GB |
| 405B | 256 GB | 512 GB | 256 GB |

**Why system RAM matters even with a GPU:**
- Models that don't fully fit in VRAM are partially offloaded to system RAM (llama.cpp `-ngl` flag)
- Tokenization, prompt processing, and KV cache may use system RAM
- System RAM speed (DDR5 vs DDR4, dual-channel vs quad-channel) affects offloaded inference speed
- Large context windows require significant system memory for the KV cache

---

## 4. Storage Considerations

### 4.1 Model Storage Space

| Model Size | Q4_K_M Size | FP16 Size |
|---|---|---|
| 7B | 4.2 GB | 13.5 GB |
| 13B | 7.7 GB | 24.5 GB |
| 34B | 19.5 GB | 62.0 GB |
| 70B | 39.5 GB | 126.0 GB |
| 120B | 68.0 GB | 220.0 GB |
| 180B | 105.0 GB | 330.0 GB |
| 405B | 235.0 GB | 740.0 GB |

### 4.2 Storage Recommendations

- **Minimum:** 500 GB NVMe SSD — holds 5–10 quantized models plus OS and tools
- **Recommended:** 2 TB NVMe SSD — comfortable space for experimenting with multiple models
- **Enthusiast:** 4 TB NVMe SSD — enough for the top 5–10 models at multiple quantization levels
- **Collector:** 8 TB+ SSD or NAS — for full model collections and datasets

**Key tip:** Use a fast NVMe drive (PCIe 4.0 or 5.0) for model storage. While models are loaded into RAM/VRAM for inference, loading time from a slow drive adds 30–90 seconds for large models. A Gen5 NVMe can load a 70B Q4 model in under 5 seconds.

---

## 5. Cost Analysis: Buy vs. Rent

One of the most important decisions in local AI is whether to buy hardware or rent cloud GPU instances.

### 5.1 Seven-Year Total Cost of Ownership (Buying)

| Build | Initial Cost | Annual Electricity | 7-Year Total | vs. Cloud Equivalent |
|---|---|---|---|---|
| Budget CPU (7B max) | $1,000 | $150 | $2,050 | ~2 months of A100-80GB |
| Mid GPU (4090, 24 GB) | $3,500 | $300 | $5,600 | ~6 months of A10G-24GB |
| High-End (Dual 4090) | $5,500 | $450 | $8,650 | ~8 months of A100-80GB |
| Enthusiast (Dual 5090) | $8,000 | $600 | $12,200 | ~12 months of A100-80GB |
| Max (4x 4090 / Threadripper) | $15,000 | $800 | $20,600 | ~18 months of H100-80GB |
| Mac Pro (M4 Ultra 192 GB) | $7,500 | $200 | $8,900 | ~9 months of A100-80GB |
| Apple Max (M4 Ultra 192 GB) | $8,000 | $250 | $9,750 | ~10 months of A100-80GB |

### 5.2 Cloud GPU Rental Costs (2026)

| Instance | GPU | VRAM | Hourly | Monthly (730h) | Per-Token Cost |
|---|---|---|---|---|---|
| RunPod Secure Cloud | RTX 4090 | 24 GB | $0.39 | $285 | ~$0.000005 |
| RunPod Community | RTX 4090 | 24 GB | $0.18 | $131 | ~$0.000002 |
| RunPod | A100-80GB | 80 GB | $1.59 | $1,161 | ~$0.000003 |
| RunPod | H100-80GB | 80 GB | $2.49 | $1,818 | ~$0.000002 |
| Lambda Labs | A100-80GB | 80 GB | $1.10 | $803 | ~$0.000002 |
| Vast.ai (cheap) | RTX 3090 | 24 GB | $0.12 | $88 | ~$0.000003 |
| Vast.ai (cheap) | A100-80GB | 80 GB | $0.80 | $584 | ~$0.0000015 |
| Azure NCas_H100_v5 | H100-80GB | 80 GB | $5.20 | $3,796 | ~$0.000001 |

### 5.3 Decision Framework

**Buy hardware if:**
- You use inference more than 6–10 hours per day, every day
- You need guaranteed availability (no "all GPUs in use")
- You handle sensitive data (medical, legal, financial)
- You want a fixed, predictable cost
- You have the upfront capital
- You need low latency for real-time applications

**Rent cloud GPUs if:**
- You use inference less than 4–6 hours per day average
- You need access to H100s or A100s with 80 GB+ VRAM
- You want to try many different hardware configurations
- You need burst capacity (scale up for batch jobs, scale down to zero)
- You don't have adequate cooling or power at your location
- You want to avoid hardware maintenance and upgrades

**Hybrid approach (recommended for most):**
- Own a local machine for development, experimentation, and private inference
- Rent cloud capacity for large batch jobs, model training, or serving users
- This gives you the best of both worlds: privacy + scalability

---

## 6. Build Recommendations by Budget

### 6.1 $500–$800 Budget (Entry Level)

Goal: Run 7B models comfortably, 13B models with CPU offloading.

**Option A: Used Workstation + GPU**
- Used Dell Precision T5810 or HP Z440: $200–$300
- Used RTX 3060 12 GB: $150–$200
- 64 GB DDR4: $60–$80
- 1 TB NVMe: $50
- Total: ~$500–$630

**Option B: New Budget Build**
- Ryzen 5 7600: $180
- B650 motherboard: $120
- 32 GB DDR5-6000: $80
- 1 TB NVMe SSD: $50
- 550W PSU: $60
- Used RTX 3060 12 GB: $150–$200
- Total: ~$640–$690

**Expected performance:** 7B Q4_K_M: 50–60 t/s, 13B Q4_K_M: 20–30 t/s (with some CPU offload)

### 6.2 $1,500–$2,500 Budget (Good)

Goal: Run 7B–34B models at full GPU speed, 70B with CPU offloading.

**Option A: RTX 4090 Build**
- Ryzen 7 7700: $280
- B650 motherboard: $130
- 64 GB DDR5-6000: $160
- 2 TB NVMe SSD: $120
- 1000W PSU: $150
- Used RTX 4090: $1,400
- Total: ~$2,240

**Option B: Dual 3090 Build**
- Ryzen 7 7700: $280
- B650 motherboard (needs two x16 slots): $180
- 64 GB DDR5-6000: $160
- 2 TB NVMe SSD: $120
- 1200W PSU: $180
- Two used RTX 3090 24 GB: $1,000 ($500 each)
- Total: ~$1,920

**Expected performance:** 7B: 80–90 t/s, 34B Q4_K_M: 20–22 t/s, 70B Q4_K_M: 12–16 t/s

### 6.3 $3,000–$5,000 Budget (High-End)

Goal: Run 70B models at full GPU speed, 120B+ with CPU offloading.

**Option A: Dual 4090 Build (Best performance)**
- Ryzen 9 7950X: $550
- X670E motherboard: $350
- 128 GB DDR5-6000: $300
- 2 TB NVMe SSD: $120
- 1600W PSU: $250
- Two used RTX 4090: $2,800 ($1,400 each)
- Total: ~$4,370

**Option B: Mac Studio M4 Max (128 GB)**
- Mac Studio M4 Max (128 GB unified memory): $4,500
- 2 TB SSD: +$400
- Total: ~$4,900

**Expected performance (Dual 4090):** 7B: 90+ t/s, 70B Q4_K_M: 18–22 t/s, 120B Q3_K_M: 8–10 t/s
**Expected performance (M4 Max):** 7B: 80–90 t/s, 70B Q4_K_M: 8–10 t/s, without the complexity of dual GPU setup

### 6.4 $5,000–$10,000 Budget (Enthusiast)

Goal: Run 120B+ models, serve multiple users, fine-tuning and inference.

**Option A: Dual RTX 5090 Build (Highest throughput)**
- AMD Threadripper 7960X (24-core): $1,500
- TRX50 motherboard: $800
- 128 GB DDR5-6400: $400
- 4 TB NVMe SSD: $250
- 2000W PSU: $400
- Two RTX 5090: $5,000 ($2,500 each)
- Total: ~$8,350

**Option B: Mac Studio M4 Ultra (192 GB)**
- Mac Studio M4 Ultra (192 GB): $7,500
- 4 TB SSD: +$600
- Total: ~$8,100

**Option C: Quad RTX 3090 Server**
- EPYC 9124 (16-core): $600
- SP5 motherboard: $1,000
- 256 GB DDR5-4800: $600
- 4 TB NVMe SSD: $250
- 2000W PSU: $400
- Server chassis: $300
- GPU risers and cables: $200
- Four used RTX 3090 24 GB: $2,000 ($500 each)
- Total: ~$5,350

**Expected performance (Dual 5090):** 7B: 110+ t/s, 70B Q4_K_M: 25–30 t/s, 120B Q4_K_M: 14–16 t/s
**Expected performance (M4 Ultra):** 7B: 100–110 t/s, 70B Q4_K_M: 12–15 t/s, 180B Q4_K_M: 5–6 t/s
**Expected performance (Quad 3090):** 70B Q4_K_M: 15–18 t/s, 120B Q4_K_M: 8–10 t/s, 180B Q3_K_M: 6–8 t/s

### 6.5 Over $10,000 (Data Center)

- Dual or quad RTX 6000 Ada (48 GB each): $15,000–$30,000
- AMD Instinct MI300X (192 GB): $12,000–$15,000
- NVIDIA H100 (80 GB): $25,000–$35,000
- NVIDIA H200 (141 GB): $35,000–$50,000

At this tier, you are running production inference serving for an organization. Cloud rental is usually more cost-effective unless utilization is near 100%.

---

## 7. Power and Cooling

### 7.1 Power Requirements

| Configuration | Idle Power | Load Power | Annual Electricity Cost ($0.12/kWh, 8h/day) |
|---|---|---|---|
| CPU-only (7950X) | 50W | 200W | $70 |
| Single RTX 4090 | 150W | 600W | $210 |
| Dual RTX 4090 | 250W | 1,050W | $368 |
| Dual RTX 5090 | 300W | 1,250W | $438 |
| Quad RTX 3090 | 350W | 1,500W | $526 |
| Mac Studio M4 Ultra | 30W | 140W | $49 |
| Threadripper + Dual 5090 | 200W | 1,500W | $526 |

### 7.2 Cooling Considerations

- **Single GPU:** Standard tower cooling (3–4 case fans, CPU tower cooler) is sufficient
- **Dual GPU:** Requires excellent case airflow. Consider 360mm AIO for CPU, high-static-pressure fans
- **Quad GPU:** Requires open-air mining-style frame or server chassis with high-CFM fans. GPUs will run at 70–80°C
- **Mac Studio:** Passive cooling, silent, runs cool — a major advantage
- **GPU undervolting** can reduce power by 15–25% with <5% performance loss. Highly recommended for multi-GPU builds.

---

## 8. Form Factor and Noise

### 8.1 Desktop vs. Server vs. Mac

| Factor | Desktop PC | Server / Rack | Mac Studio / Mini |
|---|---|---|---|
| Noise | Moderate (fans under load) | Loud (high-RPM fans) | Silent (passive or very quiet) |
| Size | Mid-tower to full-tower | 2U–4U rack chassis | Small (7.7" square) |
| Upgradeability | High | High | None (soldered) |
| GPU Options | Any consumer card | Requires blower-style or active cooling | None (fixed) |
| Power Efficiency | Moderate | Low | Very High |
| Initial Cost | Low to High | High | Moderate to High |
| Best For | General purpose | Multi-GPU / production | Low-noise environments |

---

## 9. Benchmark Methodologies and Real-World Numbers

### 9.1 Inference Performance Metrics

**Tokens per second (t/s):** The standard measure of inference speed. For interactive use:
- 1–5 t/s: Painfully slow, only usable for batch processing
- 5–15 t/s: Usable for reading, noticeable lag
- 15–30 t/s: Comfortable for most use cases
- 30–60 t/s: Fast, minimal perceived latency
- 60+ t/s: Very fast, text appears instantly

**Time to First Token (TTFT):** Latency before the model starts responding. Influenced by prompt processing speed. Target: <500ms for interactive use.

**Prompt processing speed (t/s):** How fast the model can ingest your prompt + context. Usually 2–10x faster than generation speed. Important for RAG applications with large context.

### 9.2 Multi-GPU Scaling Efficiency

| Configuration | Single GPU | Dual GPU (tensor parallel) | Quad GPU (tensor parallel) |
|---|---|---|---|
| RTX 4090 (24 GB) | 1.0x | 1.75–1.85x | 3.2–3.5x |
| RTX 5090 (32 GB) | 1.0x | 1.8–1.9x | 3.4–3.6x |
| RTX 3090 (24 GB) | 1.0x | 1.7–1.8x | 3.0–3.3x |

Scaling efficiency decreases with more GPUs due to inter-GPU communication overhead. PCIe 4.0 x16 is recommended for each GPU.

### 9.3 Quantization Performance Impact

| Quant | 7B t/s (4090) | 70B t/s (4090) | Quality (vs FP16) |
|---|---|---|---|
| Q2_K (2-bit) | 95 | 14 | Significant degradation |
| Q3_K_M (3-bit) | 90 | 12 | Some degradation |
| Q4_K_M (4-bit) | 85 | 10 | Minimal, often indistinguishable |
| Q5_K_M (5-bit) | 78 | 8 | Very close to FP16 |
| Q6_K (6-bit) | 72 | 7 | Essentially identical |
| Q8_0 (8-bit) | 60 | 5 | Nearly identical |
| FP16 | 35 | 2.5 | Reference |

Q4_K_M is the sweet spot for most use cases — ~85% of FP16 quality with ~4x less VRAM and ~2x speed.

---

## 10. Future Hardware Trends (2026–2027)

- **NVIDIA RTX 6090 (Blackwell Ultra, late 2026):** Rumored 48 GB GDDR7 on a single card. Would be a game-changer for local inference.
- **AMD RDNA 5:** Expected 32 GB+ VRAM, improved ROCm support, targeting inference workloads specifically.
- **Intel Celestial (Arc 3rd gen):** Targeting 16–24 GB with full AI acceleration in hardware.
- **Apple M5 Ultra (2027):** Expected 256 GB unified memory, ~1,500 GB/s bandwidth — potentially the ultimate local inference machine.
- **CXL Memory Expansion:** PCIe-connected memory pools could let GPUs access 256 GB+ of poolable memory, though latency will be higher.
- **NPU Competition:** By 2027, it's expected that all desktop/laptop CPUs will have 40+ TOPS NPUs, making basic local inference standard on every device.

---

## 11. Quick Decision Trees

### "I want to run a single model of size X"

| Model Size | Best Single GPU | Budget Option | Best Overall System |
|---|---|---|---|
| Up to 7B | RTX 3060 12 GB ($200 used) | CPU only (any modern CPU) | Any system with 8 GB+ VRAM |
| 13B–20B | RTX 4090 24 GB ($1,400 used) | RX 7900 XTX ($900) | RTX 4090 |
| 34B–70B | RTX 5090 32 GB ($2,500) | Dual RTX 3090 ($1,000 used) | Dual RTX 4090 or M4 Ultra 192 GB |
| 120B–180B | RTX 6000 Ada 48 GB ($6,800) | Dual RTX 4090 ($2,800 used) | M4 Ultra 192 GB |
| 405B | H100 80 GB ($25,000+) | Quad RTX 3090 ($2,000) / 4x3090 | Cloud rental |

### "I want the most VRAM for my money"

| Ranking | Configuration | VRAM | Cost | Cost per GB |
|---|---|---|---|---|
| 1 | Quad used RTX 3090 | 96 GB | $2,000 | $20.83/GB |
| 2 | Dual used RTX 3090 | 48 GB | $1,000 | $20.83/GB |
| 3 | Dual used RTX 4090 | 48 GB | $2,800 | $58.33/GB |
| 4 | Mac Studio M4 Ultra | 192 GB | $7,500 | $39.06/GB |
| 5 | Single used RTX 3090 | 24 GB | $500 | $20.83/GB |
| 6 | Used RX 7900 XTX | 24 GB | $900 | $37.50/GB |

Quad RTX 3090 is the absolute best VRAM-per-dollar, but requires significant power, cooling, and motherboard bandwidth. Mac Studio M4 Ultra offers the most VRAM in a single simple system.

### "Budget is my main constraint"

$500 → Single 3060 12 GB + used workstation: runs 7B models well
$1,000 → Single 3090 24 GB: runs 7B to 34B, 70B with offloading
$2,000 → Dual 3090 48 GB: runs 70B Q4_K_M at full speed
$3,500 → Single 5090 32 GB: runs 70B comfortably, 120B with offloading
$5,000 → M4 Ultra 192 GB: runs almost everything, silent, low power
$8,000 → Dual 5090 64 GB: highest throughput consumer build

---

## 12. Summary

| Factor | Recommendation |
|---|---|
| Best single GPU (value) | RTX 4090 (used, ~$1,400) — 24 GB, fast, widely supported |
| Best single GPU (performance) | RTX 5090 (~$2,500) — 32 GB, fastest consumer card |
| Best VRAM per dollar | Quad RTX 3090 (~$2,000 for 96 GB) |
| Best all-in-one system | Mac Studio M4 Ultra (192 GB, silent, efficient) |
| Best budget GPU | RTX 3060 12 GB (used, ~$150–$200) |
| Best for 7B models | Any GPU with 8–12 GB VRAM |
| Best for 70B models | 24–32 GB GPU or dual GPU setup |
| Best for 120B+ models | Mac Ultra 192 GB or multi-GPU server |
| Best CPU-only | AMD Threadripper with 8-channel DDR5 |
| Most energy efficient | Apple Silicon (M4 Ultra) — 140W vs 1,000W+ for GPU build |

The landscape of local inference hardware in 2026 is more accessible than ever. You can run capable 7B–13B models on a $500 used workstation, 70B models on a $2,000 build, and 180B models on a $5,000 Mac Studio. The era of needing $50,000+ data center hardware for capable LLMs is over — local inference is now a realistic option for individuals, startups, and enterprises alike.

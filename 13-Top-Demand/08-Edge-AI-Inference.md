# Edge AI & On-Device Inference

> **Last Updated:** June 2026  
> **Category:** Top Demand — Current Market Snapshot  
> **Cross-References:** 02-AI-Agent-Development.md, 04-Multimodal-AI.md, 07-Fine-Tuning-Custom-Models.md, 10-Real-Time-AI-Systems.md

---

## Table of Contents

1. [Market Context & Demand](#1-market-context--demand)
2. [On-Device Inference Frameworks](#2-on-device-inference-frameworks)
   - 2.1 MLX (Apple)
   - 2.2 CoreML (Apple)
   - 2.3 TFLite / TFLite GPU
   - 2.4 ONNX Runtime
   - 2.5 ExecuTorch (PyTorch)
   - 2.6 MediaPipe
3. [Model Quantization](#3-model-quantization)
   - 3.1 GPTQ (Post-Training Quantization)
   - 3.2 AWQ (Activation-Aware Weight Quantization)
   - 3.3 GGUF / GGML
   - 3.4 BitsAndBytes (NF4, FP4)
   - 3.5 FP8 Training and Inference
   - 3.6 Comparison
4. [Edge Hardware Platforms](#4-edge-hardware-platforms)
   - 4.1 Apple Neural Engine (ANE)
   - 4.2 Qualcomm AI Engine (Hexagon NPU)
   - 4.3 Google Edge TPU / Coral
   - 4.4 NVIDIA Jetson (Orin, Thor)
   - 4.5 Raspberry Pi (RP2040, Pi 5)
   - 4.6 Intel/AMD NPU
   - 4.7 Samsung Exynos NPU
5. [Use Cases & Applications](#5-use-cases--applications)
   - 5.1 On-Device LLM Chat
   - 5.2 Edge Vision (Camera-based)
   - 5.3 Voice Assistants
   - 5.4 Industrial IoT
   - 5.5 Healthcare
6. [Performance Benchmarks](#6-performance-benchmarks)
7. [Optimization Techniques](#7-optimization-techniques)
8. [Deployment Strategies](#8-deployment-strategies)
9. [Future Directions](#9-future-directions)

---

## 1. Market Context & Demand

Edge AI — running AI models on local devices rather than in the cloud — has become a mainstream requirement in 2026. The convergence of powerful on-device hardware, efficient model architectures, and privacy regulations is driving massive adoption.

**Market dynamics:**
- Edge AI market: $38B (2026), projected $78B by 2030 (Grand View Research)
- 75% of smartphone users have on-device AI features active
- Apple Intelligence ships on all M-series Macs and iPhone 17+ (2025)
- Google AI Core runs on Samsung Galaxy S30+, Pixel 12+
- On-device LLMs: 7B parameter models run on high-end phones at 20+ tokens/sec

**Why edge AI now:**
- **Privacy** — EU AI Act, GDPR, and user demand for local processing
- **Latency** — Zero network dependency, sub-100ms inference
- **Offline capability** — No connectivity required
- **Cost** — No API calls saves $0.01-0.10 per query at scale
- **Hardware maturation** — Neural engines, NPUs, and efficient architectures are finally production-ready

---

## 2. On-Device Inference Frameworks

### 2.1 MLX (Apple)

Apple's MLX framework (2024) is optimized for Apple Silicon (M-series, A17+):

**Key features:**
- Unified memory architecture (CPU/GPU/NPU share memory)
- NumPy-compatible API
- Lazy computation for graph optimization
- 16-bit and 8-bit inference
- LoRA fine-tuning on device

**Performance (M4 Ultra, 7B model):**
- Llama 4 7B: 45 tokens/sec (8-bit)
- Qwen 2.5 7B: 50 tokens/sec (8-bit)
- Whisper v4: 5x real-time factor

**Example:**
```python
import mlx.core as mx
import mlx.nn as nn

# Load MLX-compatible model
model, tokenizer = mlx.load("mlx-community/Llama-4-7B-4bit")

# Generate
prompt = "Explain edge AI in simple terms"
tokens = tokenizer.encode(prompt)
output = model.generate(mx.array(tokens), max_tokens=200)

print(tokenizer.decode(output))
```

### 2.2 CoreML (Apple)

CoreML is Apple's production machine learning framework for iOS/macOS:

**Key features:**
- Automatic GPU/ANE delegation
- Model compression (quantization, pruning)
- On-device training support
- Vision and Natural Language frameworks
- Xcode integration

**Conversion pipeline:**
```
PyTorch Model → ONNX → CoreML (.mlpackage) → Xcode Project
              ↗
HuggingFace Optimum (exporters.coreml)
```

**ANE utilization (Apple Neural Engine):**
- iPhone 17 Pro: 45 TOPS NPU
- M4 Ultra: 64 TOPS ANE
- Dedicated matrix multiply units
- Best for compute-bound operations (attention, MLP)

### 2.3 TFLite / TFLite GPU

Google's TensorFlow Lite is the most widely deployed mobile ML framework:

```python
import tensorflow as tf

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

# Delegate to GPU
from tensorflow.lite.python import interpreter
interpreter = tf.lite.Interpreter(
    model_content=tflite_model,
    experimental_delegates=[tf.lite.experimental.GpuDelegate()]
)
interpreter.allocate_tensors()
```

**TFLite features (2026):**
- XNNPACK delegate (CPU optimization)
- GPU delegate (OpenGL/ES, Vulkan, Metal)
- NNAPI delegate (Android NPU access)
- Hexagon delegate (Qualcomm DSP)
- CoreML delegate (iOS ANE access)

### 2.4 ONNX Runtime

ONNX Runtime provides cross-platform inference with hardware-specific execution providers:

```yaml
onnx_runtime:
  execution_providers:
    - CPU: OpenMP, MKL-DNN
    - CUDA: NVIDIA GPU
    - TensorRT: NVIDIA optimizations
    - CoreML: Apple ANE
    - DirectML: Windows GPU
    - OpenVINO: Intel CPU/GPU/NPU
    - QNN: Qualcomm NPU
    - ACL: ARM CPU
    
  quantization:
    - dynamic (weight-only)
    - static (weight + activation)
    - QDQ (Quantize-Dequantize) for QAT
```

### 2.5 ExecuTorch (PyTorch)

Meta's ExecuTorch (2024) brings PyTorch models to mobile and edge devices:

**Key features:**
- Minimal runtime (~100KB)
- Arm, Qualcomm, Apple Silicon, x86 support
- Delegation to backends (CoreML, QNN, XNNPACK)
- Program memory management
- On-devise training (limited)

```python
# Export to ExecuTorch
import torch
import executorch

# Program (aot): export for edge
executorch.exporter.export(
    model,
    example_inputs,
    edge_backend="apple_coreml"  # or "qualcomm_qnn"
)
```

### 2.6 MediaPipe

Google's MediaPipe provides ready-to-use on-device ML pipelines:

**Solutions (2026):**
- **MediaPipe Face Landmarker** — 478 landmarks, 30+ FPS on phone
- **MediaPipe Hand Landmarker** — 21 landmarks per hand
- **MediaPipe Object Detector** — EfficientDet-Lite models
- **MediaPipe LLM Inference** — On-device LLM support (Gemini Nano)
- **MediaPipe GenAI** — Diffusion model inference on device

---

## 3. Model Quantization

### 3.1 GPTQ (Post-Training Quantization)

GPTQ (Frantar et al., 2023) performs one-shot weight quantization using approximate second-order information:

**Process:**
```
1. Compute Hessian of loss w.r.t. weights (using calibration data)
2. Quantize weights one-by-one, adjusting remaining weights to compensate
3. Result: 4-bit or 3-bit weights with minimal accuracy loss
```

**Characteristics:**
- Requires calibration data (~128 samples)
- GPU-intensive quantization process (1-4 hours for 7B model)
- Excellent weight-only quantization (4-bit W4A16)
- 16-bit activations, 4-bit weights
- 3-4x memory reduction vs. FP16

**Usage:**
```python
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "model-name",
    quantize_config=BaseQuantizeConfig(bits=4, group_size=128),
    device="cuda:0"
)
```

### 3.2 AWQ (Activation-Aware Weight Quantization)

AWQ (2024) observes that a small fraction of "salient" weight channels are critical for quality and protects them during quantization.

**Key insight:**
~1% of weight channels are "salient" (handle large activations). AWQ scales these channels before quantization to preserve quality.

**Advantages over GPTQ:**
- Faster quantization (no Hessian computation)
- Better quality at extreme quantization (3-bit, 2-bit)
- Hardware-friendly (doesn't need group size optimizations)
- 10-20% faster inference due to hardware-friendly kernels

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("model-name")
model.quantize(tokenizer, quant_config={"bits": 4, "group_size": 128})
model.save_quantized("awq-model")
```

### 3.3 GGUF / GGML

GGUF (GPT-Generated Unified Format) is the standard for CPU-friendly inference via llama.cpp:

**Format features:**
- **GGUF** — Single-file format (model + tokenizer + config)
- **Multiple quantization levels** — Q2_K through Q8_0
- **K-quant** — Mixed precision (more important layers get higher bit width)
- **IMatrix** — Importance matrix for better quantization

**Quantization levels comparison:**

| Format | Bits/Weight | Quality | Speed (CPU) | File Size (7B) |
|--------|-------------|---------|-------------|-----------------|
| Q2_K | 2.56 | Low | 30 t/s | 2.7 GB |
| Q3_K_M | 3.35 | Medium | 25 t/s | 3.3 GB |
| Q4_K_M | 4.35 | Good | 22 t/s | 4.1 GB |
| Q5_K_M | 5.35 | Very Good | 18 t/s | 4.8 GB |
| Q6_K | 6.35 | Excellent | 15 t/s | 5.5 GB |
| Q8_0 | 8.35 | Near-Original | 12 t/s | 6.9 GB |
| FP16 | 16 | Original | 6 t/s | 13.0 GB |

### 3.4 BitsAndBytes (NF4, FP4)

BitsAndBytes (HuggingFace integration) enables 4-bit models in transformers:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "model",
    quantization_config=bnb_config
)
```

- **NF4 (NormalFloat4):** Optimal 4-bit information-theoretic quantization
- **FP4:** Simpler 4-bit floating point, less accurate but easier to compute
- **Double quantization:** Quantize the quantization constants (saves ~0.5 bits/weight)

### 3.5 FP8 Training and Inference

FP8 (8-bit floating point) has become the standard for H100/B200 training:

- **E4M3** (4 exponent, 3 mantissa) — Used for weights and activations
- **E5M2** (5 exponent, 2 mantissa) — Used for gradients (wider dynamic range)
- **H100/B200:** Native FP8 tensor cores (2x throughput vs. FP16)

### 3.6 Quantization Comparison

| Method | Bits | Quality (vs FP16) | GPU Inference | CPU Inference | Compress Time |
|--------|------|-------------------|---------------|---------------|---------------|
| FP16 | 16 | 100% | ✅ Fast | ❌ | — |
| INT8 | 8 | 99.5% | ✅ Fast | ✅ | Minutes |
| FP8 | 8 | 99.7% | ✅ Fastest (H100) | ❌ | — |
| GPTQ 4-bit | 4 | 98-99% | ✅ Fast | ❌ | Hours |
| AWQ 4-bit | 4 | 98-99% | ✅ Fast | ❌ | Minutes |
| GGUF Q4_K_M | 4.35 | 98-99% | ❌ | ✅ Fast | Hours |
| NF4 (bnb) | 4 | 97-99% | ✅ | ❌ | Instant (load) |

---

## 4. Edge Hardware Platforms

### 4.1 Apple Neural Engine (ANE)

| Chip | TOPS | Devices | RAM | Notes |
|------|------|---------|-----|-------|
| A17 Pro | 35 | iPhone 15 Pro | 8 GB | First 3nm chip |
| A18/A19 | 45 | iPhone 17 | 12 GB | Improved ANE |
| M4 | 38 | Mac, iPad Pro | 18-48 GB | Unified memory |
| M4 Ultra | 64 | Mac Studio | 128 GB | 2x M4 Max |

**ANE architecture:**
- 16-core design (M4)
- Dedicated matrix multiplication engine
- Shared memory with CPU/GPU (no copy overhead)
- Optimized for transformer operations in iOS 20+

### 4.2 Qualcomm AI Engine (Hexagon NPU)

| Chip | TOPS | Devices | Notes |
|------|------|---------|-------|
| Snapdragon 8 Gen 3 | 34 | Android flagship | First to run 7B LLM on-device |
| Snapdragon 8 Gen 4 | 50 | 2025 flagships | Improved transformer engine |
| Snapdragon X Elite | 45 | Windows on ARM | Laptop NPU |
| Snapdragon X Gen 2 | 75 | 2026 laptops | Next-gen |

**Qualcomm AI Stack:**
- **Qualcomm AI Hub** — Model zoo for on-device models
- **QNN (Qualcomm Neural Network)** SDK — Low-level access
- **Qualcomm AI Engine Direct** — Unified API for NPU/GPU/DSP
- **Execution Provider** — ONNX Runtime and TensorFlow Lite delegates

### 4.3 Google Edge TPU / Coral

| Platform | TOPS | Use Case | Notes |
|----------|------|----------|-------|
| Coral USB Accelerator | 4 | Prototyping | USB-attached |
| Coral M.2 Accelerator | 4 | Embedded | M.2 form factor |
| Coral Dev Board | 4 | Development | Integrated TPU + SoC |
| Pixel Neural Core | 15 | Pixel phones | Gemini Nano |

**Gemini Nano:**
- Google's on-device LLM (1.8B-3.8B parameters)
- Runs on Pixel 12+, Samsung Galaxy S30+
- Features: Smart Reply, on-device summarization, Gboard
- 30+ tokens/sec on Pixel 12 Pro

### 4.4 NVIDIA Jetson (Orin, Thor)

| Platform | GPU | AI TOPS | RAM | Use Case |
|----------|-----|---------|-----|----------|
| Jetson Orin Nano | 1024-core Ampere | 40 | 8 GB | Entry-level robotics |
| Jetson Orin NX | 1024-core Ampere | 70 | 16 GB | Mid-range |
| Jetson AGX Orin | 2048-core Ampere | 275 | 64 GB | High-end |
| Jetson Thor (2025+) | Blackwell | 1000+ | 128 GB | Autonomous vehicles |

**Jetson software stack:**
- **JetPack SDK** — CUDA, cuDNN, TensorRT for edge
- **Isaac ROS** — Robotics AI stack
- **TAO Toolkit** — Transfer learning and model optimization
- **NVIDIA AI Enterprise** — Production edge AI management

### 4.5 Raspberry Pi

| Model | CPU | RAM | AI Capability | Notes |
|-------|-----|-----|---------------|-------|
| Pi 5 | BCM2712 quad-core | 8 GB | Via Hailo-8L | Add-on AI accelerator |
| Pi 5 + AI Kit | + Hailo-8L | 8 GB | 13 TOPS | USB or M.2 HAT |

**Raspberry Pi AI use cases:**
- Object detection (YOLO-NAS, MobileNet)
- Speech recognition (Whisper tiny)
- Local LLM inference (Phi-3-mini, Gemma-2B at 3-5 t/s)
- Sensor processing with ML

### 4.6 Intel/AMD NPU

| Platform | NPU TOPS | Devices | Notes |
|----------|----------|---------|-------|
| Intel Core Ultra (Meteor Lake) | 11 | Laptops | First-gen NPU |
| Intel Core Ultra (Lunar Lake) | 45 | 2025 laptops | Improved |
| Intel Core Ultra (Panther Lake) | 60+ | 2026 laptops | Next-gen |
| AMD Ryzen AI 300 | 50 | 2025 laptops | XDNA 2 architecture |

**OpenVINO Toolkit:**
- Model optimizer
- Runtime inference engine
- NPU, GPU, CPU execution
- ONNX, TensorFlow, PyTorch model support

### 4.7 Samsung Exynos NPU

| Chip | NPU TOPS | Devices | Notes |
|------|----------|---------|-------|
| Exynos 2400 | 32 | Galaxy S24 (select) | |
| Exynos 2500 | 45 | Galaxy S25 | Samsung Gauss on-device |
| Exynos 2600 | 60 | Galaxy S26 (2026) | |

---

## 5. Use Cases & Applications

### 5.1 On-Device LLM Chat

**Current capabilities (2026):**
- 7B models run on high-end phones (30+ GB storage, 12+ GB RAM)
- 3-7B models run on laptops/tablets
- 1-3B models run on mid-range phones
- Context windows: 4K-8K tokens

**Implementations:**
- **Apple Intelligence** — On-device Apple LLM for writing tools, Siri
- **Google Gemini Nano** — On-device in Pixel/Samsung
- **Mistral 7B on device** — Via llama.cpp on high-end phones
- **Microsoft Phi-3** — Small model optimized for edge

### 5.2 Edge Vision

**Common deployments:**
- **Smart cameras** — Person/vehicle/object detection
- **QR code/document scanning** — On-device text extraction
- **Augmented reality** — Real-time object tracking
- **Quality inspection** — Manufacturing defect detection
- **Retail analytics** — Shelf monitoring, customer counting

**Typical pipeline:**
```
Camera → Frame Capture → Preprocessing → Model Inference → Post-processing → Action
         (30fps)       (resize, normalize)  (YOLO-NAS, EfficientNet)  (NMS, tracking)
```

### 5.3 Voice Assistants

Fully on-device voice pipeline (Apple Siri, Google Assistant):

```
Wake Word → VAD → ASR (Whisper tiny) → NLU (Gemini Nano) → TTS (Edge TTS)
(>99% detection  (100ms)   (<500ms)         (<500ms)           (<300ms)
 at -20dB SNR)
```

### 5.4 Industrial IoT

- **Predictive maintenance** — Vibration analysis on sensor data
- **Anomaly detection** — Real-time sensor reading monitoring
- **Worker safety** — PPE detection via camera, proximity alerts
- **Energy optimization** — HVAC control with ML models

### 5.5 Healthcare

- **Diabetic retinopathy screening** — On-device fundus image analysis
- **ECG analysis** — Arrhythmia detection on wearable
- **Skin condition classification** — On-device dermatology AI
- **Medication adherence** — Pill recognition via camera

---

## 6. Performance Benchmarks

### LLM Inference on Edge (June 2026)

| Model | Device | Quantization | Tokens/sec | Memory |
|-------|--------|-------------|------------|--------|
| Llama 4 7B | iPhone 17 Pro | 4-bit (GGML) | 28 t/s | 4.5 GB |
| Llama 4 7B | MacBook Pro M4 | 4-bit | 55 t/s | 4.5 GB |
| Llama 4 7B | Snapdragon X Elite | 4-bit | 22 t/s | 4.5 GB |
| Llama 4 7B | Raspberry Pi 5 + Hailo | 4-bit | 4 t/s | 4.5 GB |
| Phi-3.5 3.8B | iPhone 17 Pro | 4-bit | 60 t/s | 2.5 GB |
| Gemini Nano | Pixel 12 Pro | 8-bit | 32 t/s | 2 GB |
| Qwen 2.5 0.5B | Raspberry Pi 5 | 4-bit | 25 t/s | 0.5 GB |

### Vision Model Performance

| Model | Device | Resolution | FPS | Use Case |
|-------|--------|-----------|-----|----------|
| YOLO-NAS-S | iPhone 17 Pro | 640×640 | 120 | Object detection |
| YOLO-NAS-M | Jetson Orin NX | 640×640 | 200 | Robotics |
| EfficientNet-Lite | Raspberry Pi 5 | 224×224 | 60 | Classification |
| MobileNetV4 | Qualcomm 8 Gen 4 | 224×224 | 150 | Classification |
| DeepLabV3 | Jetson AGX Orin | 512×512 | 90 | Segmentation |

### Latency Breakdown (On-Device LLM)

```
Prompt processing (prefill):  100ms for 512 tokens (7B, 4-bit, M4)
Token generation:              35ms/token (~28 t/s)
Total for 200 tokens:         100ms + 200×35ms = 7.1s
```

---

## 7. Optimization Techniques

### Beyond Quantization

**1. Speculative decoding on device:**
Use a tiny draft model (0.5B) to generate tokens, verify with full model (7B):
- 1.5-2.5x speedup on CPU/GPU

**2. KV-cache optimization:**
- **KV cache quantization** — Quantize KV cache to FP8 or INT8
- **KV cache pruning** — Remove less important tokens from cache
- **Sliding window attention** — Only keep recent N tokens

**3. Model architecture optimization:**
- **Mamba / State-space models** — Linear-time attention, 3-5x faster on long sequences
- **Multi-query attention** (MQA) — Shared KV heads reduce memory
- **Grouped-query attention** (GQA) — Balance between MHA and MQA

**4. Compilation and kernel optimization:**
- **CoreML model optimization** — ANE ops fusion
- **Android NNAPI** — Hardware-specific graph optimization
- **Kotlin/Swift native bindings** — Avoid Python overhead

**5. Pruning:**
- Layer removal (remove 5-10% of layers with minimal quality loss)
- Head pruning (remove less important attention heads)
- Unstructured pruning (zero out small weights) + N:M sparsity (2:4 for 2x speedup)

---

## 8. Deployment Strategies

### Update & Distribution

```yaml
edge_deployment:
  model_delivery:
    - Initial install: Bundled with app (100-500 MB for 4-bit 7B)
    - Over-the-air updates: Component-wise (only changed modules)
    - Delta updates: Only download changed weights
  
  model_registry:
    - Version management with rollback capability
    - A/B testing of model versions on device
    - Gradual rollout (canary → 10% → 50% → 100%)
  
  fallback_strategy:
    - Tier 1: On-device (primary, <100ms, no cost)
    - Tier 2: On-device + cloud fallback (for complex queries)
    - Tier 3: Full cloud (for very complex queries)
    - Automatic fallback on low confidence
```

### Platform-Specific Packaging

| Platform | Package Format | App Store | Update Mechanism |
|----------|---------------|-----------|-----------------|
| iOS/macOS | CoreML .mlpackage | App Store | App update + on-demand download |
| Android | TFLite .tflite | Play Store | App update + Google Play Delivery |
| Windows | ONNX .ort | Windows Store | App update |
| Linux | GGUF | Package manager | OTA updates |

---

## 9. Future Directions

### Near-term (H2 2026)
- **7B+ models on phone** — iPhone 18 with 16GB RAM can run 13B models
- **On-device LoRA fine-tuning** — Personalize models locally with user data
- **Multi-model pipelines** — Vision + LLM + TTS all running on device
- **NPU performance parity** — Edge NPU matches cloud GPU for small models

### Medium-term (2027-2028)
- **3nm/2nm chips** — 100+ TOPS NPUs become standard
- **On-device multi-modal** — Text + image + audio all local
- **Edge-to-cloud collaboration** — Seamless task splitting between device and cloud
- **Federated on-device learning** — Privacy-preserving model improvement

### Challenges
- **Memory** — LLMs need 4-8GB just for model weights; phones have 8-16GB
- **Battery** — Continuous LLM inference drains even modern batteries in 2-3 hours
- **Fragmentation** — Android devices have wildly different NPU capabilities
- **Model updates** — 4GB+ model downloads over cellular are bandwidth-prohibitive
- **Heat** — Sustained NPU usage causes thermal throttling on phones

---

> **Related KB documents:**
> - [02-AI-Agent-Development.md](02-AI-Agent-Development.md) — Agents on edge devices  
> - [04-Multimodal-AI.md](04-Multimodal-AI.md) — Multimodal edge inference  
> - [07-Fine-Tuning-Custom-Models.md](07-Fine-Tuning-Custom-Models.md) — Quantization-aware fine-tuning  
> - [10-Real-Time-AI-Systems.md](10-Real-Time-AI-Systems.md) — Real-time edge streaming

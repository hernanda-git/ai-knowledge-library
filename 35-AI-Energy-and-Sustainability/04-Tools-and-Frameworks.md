# AI Energy and Sustainability: Tools and Frameworks

> **Category:** 35 — AI Energy and Sustainability  
> **Last Updated:** July 2026  
> **Cross-references:** [01-Overview.md](./01-Overview.md), [02-Core-Topics.md](./02-Core-Topics.md), [63-GPU-Kernel-and-Inference-Performance-Engineering/](../63-GPU-Kernel-and-Inference-Performance-Engineering/), [41-AI-Cost-Optimization-and-Enterprise-ROI/](../41-AI-Cost-Optimization-and-Enterprise-ROI/)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Measurement & Carbon Tracking Tools](#2-measurement--carbon-tracking-tools)
3. [Efficient Inference Servers](#3-efficient-inference-servers)
4. [Quantization & Compression Toolkits](#4-quantization--compression-toolkits)
5. [Efficient Training Frameworks](#5-efficient-training-frameworks)
6. [Carbon-Aware Orchestration](#6-carbon-aware-orchestration)
7. [Clean Power Procurement Platforms](#7-clean-power-procurement-platforms)
8. [Cloud Provider Sustainability Consoles](#8-cloud-provider-sustainability-consoles)
9. [Hardware & Cooling Vendors](#9-hardware--cooling-vendors)
10. [Open Datasets & Intensity Feeds](#10-open-datasets--intensity-feeds)
11. [Tool Selection Matrix](#11-tool-selection-matrix)
12. [Reference Stack (Recommended)](#12-reference-stack-recommended)

---

## 1. Executive Summary

You do not have to build sustainability tooling from scratch. A mature ecosystem exists across measurement, efficient serving, compression, and clean-power procurement. This guide maps the landscape so you can assemble a working stack.

---

## 2. Measurement & Carbon Tracking Tools

| Tool | Type | Best for |
|------|------|----------|
| **CodeCarbon** | Python lib | per-run emissions estimate |
| **CarbonTracker** | Python lib | online tracking during training |
| **Energy Monitor (pyRAPL)** | Intel RAPL | CPU/SoC power on Intel |
| **nvidia-smi / DCGM** | CLI/agent | GPU power telemetry |
| **Scaphandre** | Linux agent | process-level energy (any HW) |
| **Kepler (CNCF)** | K8s exporter | cluster-level energy per pod |

### CodeCarbon example

```python
from codecarbon import EmissionsTracker
tracker = EmissionsTracker(
    project_name="rag-pipeline",
    measure_power_secs=10,
    tracking_mode="process",   # attribute to this process
)
tracker.start()
run_pipeline()
tracker.stop()   # prints kg CO₂e + kWh
```

---

## 3. Efficient Inference Servers

| Server | Strength | Energy note |
|--------|----------|-------------|
| **vLLM** | continuous batching, PagedAttention | high GPU util → less waste |
| **TensorRT-LLM** | NVIDIA-optimized kernels | best perf/W on NVIDIA |
| **LMDeploy** | Turbomind engine | INT4/INT8 efficient |
| **llama.cpp** | CPU/GPU, GGUF quantization | edge/low-power friendly |
| **Ollama** | easy local SLMs | on-device energy savings |
| **SGLang** | fast serving + RadixAttention | KV reuse → less memory |

```bash
# vLLM: high utilization = energy-efficient
python -m vLLM.entrypoints.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --quantization awq --dtype half \
  --max-num-seqs 256        # batch big → amortize watts
```

---

## 4. Quantization & Compression Toolkits

| Toolkit | Capabilities |
|---------|--------------|
| **bitsandbytes** | 8-bit/4-bit LLM loading (LLM.int8, NF4) |
| **GPTQModel / AutoGPTQ** | GPTQ INT4 post-training quant |
| **AWQ** | Activation-aware weight quant |
| **llama.cpp (GGUF)** | 2–8 bit quant for edge |
| **torch.ao** | PyTorch native quant (PTQ, QAT) |
| **Optimum (HF)** | ONNX/quant export pipeline |

```python
# Load a 70B model in 4-bit — fits one GPU, ~4× less energy than FP16
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B", quantization_config=cfg)
```

---

## 5. Efficient Training Frameworks

| Framework | Efficiency feature |
|-----------|--------------------|
| **DeepSpeed** | ZeRO offload, CPU offload |
| **FSDP (PyTorch)** | sharded states, lower peak mem |
| **PEFT / LoRA** | train 0.1–1% params |
| **torchao** | sparse + quant training |
| **Mosaic / Composer** | efficiency recipes |

```python
# LoRA: fine-tune a tiny adapter, not the whole model → less energy
from peft import LoraConfig, get_peft_model
cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"])
model = get_peft_model(base_model, cfg)   # trains ~0.5% of weights
```

---

## 6. Carbon-Aware Orchestration

| Tool | Role |
|------|------|
| **Carbon-Aware SDK (Green Software Foundation)** | APIs for CFE data |
| **Kubernetes descheduler + CFE webhook** | move flexible jobs to clean nodes |
| **Federation-friendly schedulers** | region-shift by carbon |
| **ElectricityMaps / WattTime API** | real-time intensity feeds |

```yaml
# Conceptual K8s annotation: only schedule when grid is clean
apiVersion: batch/v1
kind: Job
metadata:
  annotations:
    carbon-policy: "run-if-cfe>=0.8"
spec:
  template:
    spec:
      containers:
        - name: trainer
          image: mytrain:latest
```

---

## 7. Clean Power Procurement Platforms

| Mechanism | Description |
|-----------|-------------|
| **PPA (Power Purchase Agreement)** | long-term clean energy contract |
| **RECs / GOs** | unbundled certificates (market-based) |
| **24/7 CFE matching** | hourly clean matching (Google model) |
| **Behind-the-meter SMR** | on-site nuclear for data center |
| **Green tariffs** | utility clean-energy programs |

> Procurement reduces *carbon*, not *watts*. Pair with efficiency (watts) for full impact.

---

## 8. Cloud Provider Sustainability Consoles

| Provider | Tool |
|----------|------|
| AWS | Customer Carbon Footprint Tool |
| Azure | Emissions Impact Dashboard |
| GCP | Carbon Sense / Active Assist |
| OCI | Carbon Rating |
| All | region picker with carbon intensity labels |

**Tip:** Choose cloud region by carbon intensity, not just latency. `us-west (Oregon)` and `europe-west9 (Paris)` are typically cleaner than `us-east-1`.

---

## 9. Hardware & Cooling Vendors

| Category | Examples |
|----------|----------|
| GPUs | NVIDIA, AMD, Intel |
| AI accelerators | Google TPU, AWS Trainium, Groq, Cerebras, Tenstorrent, d-Matrix |
| Liquid cooling | Iceotope, Submer (immersion), Asetek (direct-to-chip) |
| DCIM/BMS | Schneider, Vertiv, Siemens |

---

## 10. Open Datasets & Intensity Feeds

| Source | What |
|--------|------|
| ElectricityMaps | global real-time carbon intensity |
| WattTime | marginal emissions API |
| Ember / EEA | open grid data |
| MLPerf Power | measured perf/W benchmarks |
| PCAF | financed-emissions methodology |

---

## 11. Tool Selection Matrix

| Goal | Start with |
|------|------------|
| "How much CO₂ did this run emit?" | CodeCarbon |
| "Serve cheaply at scale" | vLLM + INT4 quant |
| "Run on my laptop/phone" | Ollama / llama.cpp (GGUF) |
| "Fine-tune without huge cluster" | PEFT/LoRA on 1 GPU |
| "Track cluster energy in K8s" | Kepler + Prometheus |
| "Buy clean power" | PPA / 24-7 CFE program |
| "Pick green region" | ElectricityMaps + cloud console |

---

## 12. Reference Stack (Recommended)

```
┌──────────────────────────────────────────────────────────┐
│ App layer   │ Ollama / vLLM (INT4/INT8)  ← efficient serve│
├──────────────────────────────────────────────────────────┤
│ Model layer │ LoRA fine-tune, distilled/SLM models        │
├──────────────────────────────────────────────────────────┤
│ Measure     │ CodeCarbon + Kepler + DCGM  → dashboard     │
├──────────────────────────────────────────────────────────┤
│ Schedule    │ carbon-aware K8s (flexible jobs gated)      │
├──────────────────────────────────────────────────────────┤
│ Power       │ clean region + PPA/24-7 CFE (market-based)  │
└──────────────────────────────────────────────────────────┘
```

This stack attacks energy at every layer: watts (efficient serve + models), waste (scheduling), and carbon (clean power + measurement).

---

*See also: [01-Overview.md](./01-Overview.md) · [02-Core-Topics.md](./02-Core-Topics.md) · [03-Technical-Deep-Dive.md](./03-Technical-Deep-Dive.md)*

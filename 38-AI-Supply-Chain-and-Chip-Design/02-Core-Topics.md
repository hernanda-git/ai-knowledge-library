# 02 — Core Topics in AI Chip Design and Supply Chain

> **Why this document exists.** The previous document (01-Overview.md) mapped the strategic landscape. This document dives into the core technical topics that every AI practitioner, investor, and policymaker needs to understand: chip architecture families, the economics of fabrication, packaging innovation, memory technology, and the software-hardware co-design revolution. These are the technical foundations that determine who wins, who loses, and who gets left behind in the AI hardware race.

---

## Table of Contents

1. [Chip Architecture Families for AI](#1-chip-architecture-families-for-ai)
2. [The Economics of Chip Fabrication](#2-the-economics-of-chip-fabrication)
3. [Advanced Packaging: The New Frontier](#3-advanced-packaging-the-new-frontier)
4. [Memory Technology for AI: HBM and Beyond](#4-memory-technology-for-ai-hbm-and-beyond)
5. [Power Delivery and Thermal Management](#5-power-delivery-and-thermal-management)
6. [Software-Hardware Co-Design](#6-software-hardware-co-design)
7. [Chiplet Architectures and Modular Design](#7-chiplet-architectures-and-modular-design)
8. [Yield Optimization and Defect Management](#8-yield-optimization-and-defect-management)
9. [EDA Tools and Design Automation](#9-eda-tools-and-design-automation)
10. [Testing and Validation](#10-testing-and-validation)
11. [Cross-References](#11-cross-references)

---

## 1. Chip Architecture Families for AI

### 1.1 The Four Architectural Paradigms

AI chips in 2026 fall into four architectural families, each optimized for different workload characteristics:

**Family 1: GPU (SIMT — Single Instruction, Multiple Threads)**

| Company | Chip | Architecture | Compute (TFLOPS) | Memory | Power |
|---------|------|-------------|-----------------|--------|-------|
| NVIDIA | B200 | Blackwell | 2,500 (FP8) | 192 GB HBM3e | 1,000W |
| NVIDIA | GB200 | Blackwell+Grace | 5,000 (FP8) | 384 GB HBM3e | 1,200W |
| AMD | MI400 | CDNA 4 | 2,000 (FP8) | 288 GB HBM3e | 750W |
| Intel | Gaudi 3 | — | 1,837 (FP8) | 128 GB HBM2e | 900W |

GPU architecture excels at:
- Large-scale distributed training (data parallelism, tensor parallelism)
- Mixed-precision computation (FP16/BF16/FP8/FP4)
- Flexible programming model (CUDA, ROCm, Triton)
- High memory bandwidth (8–16 TB/s with HBM)

GPU architecture limitations:
- High power consumption (750–1,200W per chip)
- Memory wall: compute grows 3x/year, memory bandwidth grows 1.5x/year
- Large die sizes increase defect rates and cost

**Family 2: Systolic Array (Google TPU style)**

| Company | Chip | Architecture | Compute (TFLOPS) | Memory | Power |
|---------|------|-------------|-----------------|--------|-------|
| Google | TPU v6 | Systolic | 1,400 (BF16) | 144 GB HBM | 400W |
| Google | TPU v7 | Systolic | 2,000 (BF16) | 256 GB HBM3e | 500W |

Systolic arrays excel at:
- Matrix multiplication (the core of neural network inference)
- High energy efficiency (TFLOPS/watt)
- Deterministic performance (no memory contention)

Systolic array limitations:
- Less flexible than GPUs for non-matrix workloads
- Requires JAX/XLA or TensorFlow (CUDA ecosystem not available)
- Limited external availability (Google Cloud only for TPU)

**Family 3: Dataflow (Cerebras/SambaNova style)**

| Company | Chip | Architecture | Compute (TFLOPS) | Memory | Power |
|---------|------|-------------|-----------------|--------|-------|
| Cerebras | WSE-3 | Wafer-Scale | 125,000 (FP16) | 44 GB SRAM (on-chip) | 20,000W (system) |
| SambaNova | RDU v3 | Reconfigurable Dataflow | 2,500 (FP8) | 128 GB HBM3e | 600W |

Dataflow architectures excel at:
- Massive on-chip memory (Cerebras: 44 GB SRAM, 33 TB/s bandwidth)
- Eliminating the memory wall for certain workloads
- Deterministic latency (no cache misses)

Dataflow architecture limitations:
- Power-hungry (Cerebras system: 20kW)
- Limited software ecosystem
- Niche workloads where on-chip memory is decisive

**Family 4: Custom ASIC (Cloud Provider In-House)**

| Company | Chip | Focus | Compute (TFLOPS) | Memory | Power |
|---------|------|-------|-----------------|--------|-------|
| Microsoft | Maia 2 | Azure inference | 1,400 (FP8) | 144 GB HBM3e | 500W |
| Meta | MTIA v3 | Meta inference | 1,200 (INT8) | 128 GB HBM3e | 400W |
| Amazon | Trainium 3 | AWS training+inference | 1,600 (FP8) | 144 GB HBM3e | 500W |

Custom ASICs excel at:
- Optimized for specific model architectures
- Tight integration with cloud provider's software stack
- Lower cost at scale (no NVIDIA margin)

Custom ASIC limitations:
- Single-vendor availability
- Limited to the provider's cloud
- Requires massive R&D investment ($1B+ per chip generation)

### 1.2 Architecture Selection Matrix

| Workload | Best Architecture | Why |
|----------|------------------|-----|
| Large-scale training (100B+ params) | GPU (NVIDIA B200/GB200) | Mature distributed training stack, NVLink interconnect |
| Large-scale training (cost-optimized) | GPU (AMD MI400) or TPU v6 | 30–50% lower cost, sufficient ecosystem |
| High-throughput inference | Systolic (TPU v6) or Custom ASIC | Best TFLOPS/watt, lowest $/token |
| Low-latency inference | Dataflow (Cerebras WSE-3) | On-chip memory eliminates memory wall |
| Edge inference | Custom ASIC (Qualcomm, Apple) | Power efficiency, on-device constraints |
| Scientific computing (HPC) | GPU (NVIDIA) or TPU | Flexibility + scale |

### 1.3 The Architecture Convergence Trend

Despite these four families, architectures are converging:

- GPUs are adding matrix engines (Tensor Cores) — becoming more systolic
- Systolic arrays are adding programmable elements — becoming more GPU-like
- Custom ASICs are adopting chiplet designs — borrowing from GPU modularity
- All families are adopting similar precision formats (FP8, FP4, MXFP)

By 2028, the distinction between GPU and systolic may blur, with "AI accelerators" as a unified category.

---

## 2. The Economics of Chip Fabrication

### 2.1 The Cost Stack of a Single AI Chip

The total cost of producing a single advanced AI chip (e.g., NVIDIA B200):

| Cost Component | Amount | % of Total |
|---------------|--------|-----------|
| **Wafer fabrication** (TSMC 4N) | $8,000–$12,000 | 35–40% |
| **Design amortization** (NVIDIA R&D / units) | $500–$1,000 | 3–5% |
| **Advanced packaging** (CoWoS) | $3,000–$5,000 | 15–20% |
| **HBM memory** (8 stacks) | $2,000–$3,000 | 12–15% |
| **Testing and validation** | $200–$500 | 1–2% |
| **Assembly and thermal solution** | $300–$500 | 2–3% |
| **Yield loss** (30–50% defect rate at 4N) | $4,000–$6,000 | 20–25% |
| **Total die cost** | $18,000–$28,000 | 100% |
| **NVIDIA margin** (~65%) | $33,000–$53,000 | — |
| **Selling price** | $50,000–$80,000 | — |

### 2.2 The Yield Problem

Yield (percentage of functional chips per wafer) is the #1 cost driver:

| Node | Typical Yield (complex GPU) | Defect Density | Cost Impact |
|------|---------------------------|----------------|-------------|
| TSMC 5N | 70–80% | 0.09 defects/cm² | Baseline |
| TSMC 4N | 55–65% | 0.12 defects/cm² | +20–30% |
| TSMC 3N | 45–55% | 0.15 defects/cm² | +40–60% |
| TSMC 2N (projected) | 35–45% | 0.18 defects/cm² | +80–100% |

**The yield paradox**: As AI chips get larger (more transistors = more capability), yield drops exponentially. A 2x die size can mean 40–60% lower yield.

**Mitigation strategies:**
1. **Chiplet design**: Split a large die into smaller, higher-yield chiplets
2. **Redundancy**: Add extra compute units that can be disabled if defective
3. **Yield learning**: Use machine learning to predict and prevent defects
4. **Wafer-level packaging**: Test at wafer level before dicing

### 2.3 The Capital Expenditure Arms Race

Building a cutting-edge chip fabrication facility:

| Component | Cost | Timeline |
|-----------|------|----------|
| Land and infrastructure | $1–2B | 1 year |
| Cleanroom construction | $3–5B | 1–2 years |
| EUV lithography tools (10–20 machines) | $3–8B | 18 months lead time |
| Other process tools | $5–10B | 12–18 months |
| Installation and qualification | $1–2B | 6–12 months |
| **Total fab cost** | **$15–28B** | **3–4 years** |

Annual operating costs:
| Item | Cost per Year |
|------|--------------|
| Utilities (power, water, gases) | $500M–$1B |
| Maintenance and consumables | $300–$500M |
| Workforce (1,000–2,000 engineers) | $200–$400M |
| **Total operating cost** | **$1–2B/year** |

### 2.4 The R&D Cost of a New Chip

Designing a new AI chip from scratch:

| Phase | Cost | Duration |
|-------|------|----------|
| Architecture definition | $50–100M | 6–12 months |
| RTL design and verification | $200–400M | 12–18 months |
| Physical design (layout, place-and-route) | $100–200M | 6–12 months |
| Tape-out and first silicon | $50–100M | 3–6 months |
| Validation and bug-fixing | $100–200M | 6–12 months |
| Software stack (compilers, drivers) | $100–300M | 12–24 months |
| **Total R&D** | **$600M–$1.3B** | **2–4 years** |

This is why only the largest companies (NVIDIA, AMD, Google, Microsoft, Meta, Apple) can afford to design custom AI chips.

---

## 3. Advanced Packaging: The New Frontier

### 3.1 Why Packaging Matters More Than Ever

The "More than Moore" era has shifted the bottleneck from transistor scaling to packaging:

- Transistor density is approaching physical limits (2nm and below)
- But system performance requires integrating multiple dies, memory, and interconnects
- Advanced packaging enables "heterogeneous integration" — combining different chip types in one package

### 3.2 Packaging Technology Comparison

| Technology | Developer | Interconnect Density | Bandwidth | Cost | Readiness |
|-----------|----------|---------------------|-----------|------|-----------|
| **2.5D (CoWoS)** | TSMC | High (400+ bumps/mm²) | 5–10 TB/s | High | Production |
| **3D (Foveros)** | Intel | Very High (10,000+ bumps/mm²) | 10+ TB/s | Very High | Limited production |
| **Fan-Out (InFO)** | TSMC | Medium | 1–3 TB/s | Medium | Production |
| **Embedded (EMIB)** | Intel | Medium-High | 3–5 TB/s | Medium | Production |
| **Hybrid Bonding** | TSMC/Intel | Extremely High (100K+ bumps/mm²) | 100+ TB/s | Very High | R&D/limited |
| **Silicon Bridge (UCIe)** | Industry consortium | High | 5–10 TB/s | Medium | Production |

### 3.3 CoWoS Deep Dive

CoWoS (Chip-on-Wafer-on-Substrate) is the packaging technology that makes modern AI GPUs possible:

**How CoWoS works:**
1. Create a silicon interposer (thin silicon wafer with metal routing)
2. Mount GPU die and HBM stacks on the interposer using micro-bumps
3. Attach the interposer to an organic substrate
4. The substrate connects to the PCB via BGA balls

**CoWoS variants:**

| Variant | Interposer | Interconnect | Use Case |
|---------|-----------|-------------|----------|
| CoWoS-S | Silicon | Micro-bumps (40μm pitch) | NVIDIA B200, AMD MI400 |
| CoWoS-R | Organic redistribution layer | RDL traces | Lower cost, lower density |
| CoWoS-L | Local silicon interconnect | Combines S and R | Next-gen cost optimization |

**CoWoS capacity planning:**

```python
# Simple CoWoS capacity model
def cowos_capacity_model(year, target_tpm):
    """
    year: target year (2024-2028)
    target_tpm: target GPU shipments per month
    """
    cowos_per_gpu = 1  # each GPU needs 1 CoWoS package
    
    # TSMC CoWoS capacity ramp (wafers/month)
    capacity = {
        2024: 35000,
        2025: 45000,
        2026: 60000,
        2027: 85000,
        2028: 110000,
    }
    
    # Good dies per CoWoS wafer (varies by die size)
    dies_per_wafer = {
        'B200_size': 30,  # ~800mm² die
        'H100_size': 40,  # ~600mm² die
        'TPU_v6': 50,     # smaller die
    }
    
    available_capacity = capacity.get(year, 0)
    max_gpus_per_month = available_capacity * dies_per_wafer['B200_size']
    
    shortage = max(0, target_tpm - max_gpus_per_month)
    utilization = min(100, (target_tpm / max_gpus_per_month) * 100)
    
    return {
        'capacity_wpm': available_capacity,
        'max_gpus': max_gpus_per_month,
        'shortage': shortage,
        'utilization_pct': round(utilization, 1),
        'bottleneck': shortage > 0
    }

# Example: 2026 demand vs. supply
result = cowos_capacity_model(2026, 1500000)  # 1.5M GPUs/month target
print(f"CoWoS capacity: {result['capacity_wpm']:,} wafers/month")
print(f"Max GPU output: {result['max_gpus']:,} GPUs/month")
print(f"Shortage: {result['shortage']:,} GPUs/month")
print(f"Utilization: {result['utilization_pct']}%")
print(f"Bottleneck: {result['bottleneck']}")
```

### 3.4 The Chiplet Revolution

Chiplet architecture is the industry's answer to the yield problem:

**Monolithic vs. Chiplet:**

| Aspect | Monolithic | Chiplet |
|--------|-----------|---------|
| Die size | Single large die | Multiple small dies |
| Yield | Low (large die = more defects) | High (small dies = fewer defects) |
| Flexibility | Fixed configuration | Mix-and-match |
| Cost | High (yield loss) | Lower (higher yield + reuse) |
| Interconnect | On-die (fast) | Die-to-die (slower, but improving) |

**UCIe (Universal Chiplet Interconnect Express):**

UCIe is an industry standard for die-to-die interconnect:
- Bandwidth: 32 GT/s per pin (UCIe 2.0)
- Latency: <2ns
- Power efficiency: <0.5 pJ/bit
- Supported by: Intel, AMD, ARM, TSMC, Samsung, Qualcomm

**Chiplet design pattern for AI:**

```
┌─────────────────────────────────────────┐
│              AI Chip Package             │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │  GPU Die  │  │  GPU Die  │  │ Cache  │ │
│  │ (Compute) │  │ (Compute) │  │  Die   │ │
│  └────┬─────┘  └────┬─────┘  └───┬────┘ │
│       │              │            │       │
│  ─────┴──────────────┴────────────┴─────  │
│           Silicon Interposer (CoWoS)      │
│  ─────┬──────────────┬────────────┬─────  │
│       │              │            │       │
│  ┌────┴────┐  ┌──────┴──┐  ┌─────┴────┐ │
│  │ HBM3e  │  │  HBM3e  │  │  HBM3e   │ │
│  │ Stack  │  │  Stack  │  │  Stack   │ │
│  └────────┘  └─────────┘  └──────────┘ │
└─────────────────────────────────────────┘
```

---

## 4. Memory Technology for AI: HBM and Beyond

### 4.1 The Memory Wall Problem

AI workloads are memory-bound, not compute-bound:

- A single B200 GPU has 2,500 TFLOPS of compute but only 8 TB/s of memory bandwidth
- To keep the compute units busy, you need to feed them data at 8 TB/s
- For a 70B parameter model, the entire model must be loaded from memory for each inference
- Memory bandwidth is the #1 limiter of AI inference performance

### 4.2 HBM Evolution

| Generation | Year | Bandwidth | Capacity | Power | Price/Stack |
|-----------|------|-----------|----------|-------|-------------|
| HBM2 | 2018 | 256 GB/s | 8 GB | 5W | $200 |
| HBM2E | 2020 | 410 GB/s | 16 GB | 6W | $300 |
| HBM3 | 2022 | 819 GB/s | 24 GB | 7W | $500 |
| HBM3E | 2024 | 1,180 GB/s | 36 GB | 8W | $700 |
| HBM3E+ | 2025 | 1,500 GB/s | 48 GB | 9W | $900 |
| HBM4 | 2027 | 3,000+ GB/s | 64+ GB | 10W | $1,200+ |

### 4.3 HBM Market Dynamics

| Company | HBM Market Share (2026) | Technology | Key Advantage |
|---------|------------------------|-----------|---------------|
| **SK Hynix** | ~50% | HBM3E, HBM4 lead | First to HBM4; NVIDIA primary supplier |
| **Samsung** | ~35% | HBM3E (yield issues) | Scale; diversified customer base |
| **Micron** | ~15% | HBM3E | US-based; CHIPS Act beneficiary |

**The HBM supply chain:**

```
DRAM Wafer → TSV Fabrication → Stack Assembly → Test → HBM Module → CoWoS Integration → GPU Package
   (SK Hynix/Samsung/Micron)                                                    (TSMC)
```

### 4.4 Beyond HBM: Emerging Memory Technologies

| Technology | Developer | Potential | Timeline | Challenge |
|-----------|----------|-----------|----------|-----------|
| **CXL Memory** | Industry consortium | Disaggregated memory pools | 2026–2027 | Latency overhead |
| **LPDDR5X/6** | Qualcomm, MediaTek | Mobile AI inference | 2026–2027 | Lower bandwidth than HBM |
| **Compute Express Link (CXL)** | Industry | Memory expansion | 2026+ | Ecosystem maturity |
| **Photonic Memory** | Research labs | Ultra-high bandwidth | 2030+ | Distance from chip |
| **MRAM/ReRAM** | Multiple | Non-volatile, fast | 2028+ | Density limitations |

### 4.5 Memory System Design for AI

```python
# Memory hierarchy optimization for AI inference
class AIMemoryHierarchy:
    """Optimize memory placement for a given model architecture."""
    
    def __init__(self, model_params_b, batch_size, sequence_length):
        self.model_params = model_params_b * 1e9
        self.batch_size = batch_size
        self.seq_length = sequence_length
        
    def compute_memory_requirements(self):
        """Calculate memory needs at each level."""
        
        # Model weights (BF16 = 2 bytes per param)
        weights_bytes = self.model_params * 2
        
        # KV cache (grows with sequence length)
        kv_cache_bytes = (
            self.model_params * 0.1  # ~10% of model size
            * self.batch_size
            * self.seq_length
            * 2  # BF16
        )
        
        # Activations (proportional to batch size)
        activation_bytes = (
            self.model_params * 0.05  # ~5% of model size
            * self.batch_size
        )
        
        total = weights_bytes + kv_cache_bytes + activation_bytes
        
        return {
            'weights_gb': weights_bytes / 1e9,
            'kv_cache_gb': kv_cache_bytes / 1e9,
            'activations_gb': activation_bytes / 1e9,
            'total_gb': total / 1e9,
            'hbm_stacks_needed': int(total / (36e9)) + 1,  # 36GB per HBM3E stack
        }
    
    def recommend_deployment(self):
        """Recommend deployment strategy based on memory needs."""
        reqs = self.compute_memory_requirements()
        
        if reqs['total_gb'] <= 192:
            return "Single GPU (B200 with 192GB HBM3e)"
        elif reqs['total_gb'] <= 384:
            return "Dual GPU with tensor parallelism (GB200 NVL2)"
        elif reqs['total_gb'] <= 1500:
            return "Multi-GPU with pipeline parallelism (8x B200)"
        else:
            return "Distributed training across multiple nodes"
        
# Example: 70B model, batch=64, seq=4096
hierarchy = AIMemoryHierarchy(70, 64, 4096)
reqs = hierarchy.compute_memory_requirements()
print(f"Weights: {reqs['weights_gb']:.1f} GB")
print(f"KV Cache: {reqs['kv_cache_gb']:.1f} GB")
print(f"Activations: {reqs['activations_gb']:.1f} GB")
print(f"Total: {reqs['total_gb']:.1f} GB")
print(f"HBM stacks needed: {reqs['hbm_stacks_needed']}")
print(f"Recommendation: {hierarchy.recommend_deployment()}")
```

---

## 5. Power Delivery and Thermal Management

### 5.1 The Power Challenge

AI chips consume enormous amounts of power:

| Chip | TDP | Power Density | Annual Energy Cost |
|------|-----|--------------|-------------------|
| NVIDIA B200 | 1,000W | 1.25 W/mm² | $8,760 |
| NVIDIA GB200 | 1,200W | 1.5 W/mm² | $10,512 |
| Cerebras WSE-3 | 20,000W (system) | 0.4 W/mm² (spread) | $175,200 |
| Google TPU v6 | 400W | 0.3 W/mm² | $3,504 |

### 5.2 Power Delivery Architecture

Modern AI chips require sophisticated power delivery:

```
Utility Grid (480V/33kV)
    → Transformer (MV to LV)
        → UPS (uninterruptible power supply)
            → PDU (power distribution unit)
                → VRM (voltage regulator module, on-board)
                    → Chip (0.6V–1.0V core voltage)
```

Key challenges:
- Voltage drop: from 480V to 0.6V requires 800:1 step-down
- Transient response: AI workloads spike power 30–50% in microseconds
- Current delivery: a 1,000W chip at 0.7V needs 1,400+ amps
- Power delivery network (PDN) impedance must be <1 milliohm

### 5.3 Cooling Solutions

| Cooling Method | Capacity | Cost | Efficiency | Use Case |
|---------------|----------|------|-----------|----------|
| Air cooling | 300–500W/chip | Low | Low | Consumer GPUs |
| Liquid cooling (cold plate) | 500–1,500W/chip | Medium | Medium | Data center GPUs |
| Direct liquid immersion | 1,500–5,000W/chip | High | High | AI training clusters |
| Two-phase immersion | 5,000–20,000W/chip | Very High | Very High | Wafer-scale (Cerebras) |

### 5.4 The Data Center Power Constraint

AI data centers face a power constraint:

| Metric | Value |
|--------|-------|
| Average power per AI rack (2026) | 80–120 kW |
| Traditional server rack | 5–15 kW |
| Data center total capacity | 50–200 MW |
| Power required for 100K GPU cluster | 100–150 MW |
| Annual electricity cost (100K GPUs) | $80–120M |
| Time to secure power (new data center) | 2–4 years |

---

## 6. Software-Hardware Co-Design

### 6.1 The Co-Design Revolution

In 2026, the most successful AI chips are designed with software in mind from day one:

| Company | Chip | Software Stack | Co-Design Feature |
|---------|------|---------------|-------------------|
| NVIDIA | B200 | CUDA, TensorRT, Triton | Tensor Core ISA optimized for transformer attention |
| Google | TPU v6 | JAX/XLA | Compiler-driven placement and optimization |
| Cerebras | WSE-3 | Cerebras Software Platform | Compiler maps model to wafer-scale topology |
| Microsoft | Maia 2 | ONNX Runtime, Triton | Inference-specific kernel fusion |

### 6.2 Compiler-Driven Optimization

Modern AI compilers optimize for specific hardware:

```python
# Example: XLA compilation for TPU
import jax
import jax.numpy as jnp

@jax.jit
def transformer_attention(Q, K, V):
    """Attention mechanism — XLA will optimize for TPU systolic array."""
    # XLA automatically:
    # 1. Tiles matrix multiply for systolic array dimensions
    # 2. Pipelines memory loads to hide latency
    # 3. Fuses operations to minimize memory traffic
    # 4. Selects optimal precision (BF16/FP8) per operation
    
    scores = jnp.matmul(Q, K.transpose(-2, -1))
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights, V)

# On TPU v6, this compiles to a sequence of:
# - TPU matrix multiply units (MXU)
# - Vector multiply units (VPU)
# - Scalar units
# All scheduled to maximize utilization
```

### 6.3 The Kernel Writing Revolution

In 2026, custom kernel writing is becoming a competitive advantage:

| Tool | Purpose | Target Hardware |
|------|---------|----------------|
| **Triton** | Python-based kernel writing | GPUs (NVIDIA, AMD) |
| **CUDA** | Low-level GPU programming | NVIDIA only |
| **ROCm/HIP** | AMD GPU programming | AMD only |
| **JAX/XLA** | Compiler-driven optimization | TPU, GPU |
| **Modular MAX** | Multi-hardware compilation | Any hardware |
| **Cerebras SDK** | Wafer-scale programming | Cerebras only |

---

## 7. Chiplet Architectures and Modular Design

### 7.1 The Chiplet Design Philosophy

Chiplet design breaks a single large die into multiple smaller dies:

**Benefits:**
- Higher yield (small dies have fewer defects)
- Mixed process nodes (compute on 3nm, I/O on 7nm)
- Reuse across product lines
- Faster time-to-market

**Challenges:**
- Die-to-die interconnect overhead
- Increased package complexity
- Thermal management across dies
- Testing complexity

### 7.2 Chiplet Topology Patterns

| Pattern | Description | Example |
|---------|-------------|---------|
| **Compute + Memory** | GPU die + HBM stacks on interposer | NVIDIA B200 |
| **Compute + Cache** | Multiple compute chiplets + shared cache die | AMD MI400 |
| **Heterogeneous** | CPU chiplet + GPU chiplet + I/O chiplet | Intel Ponte Vecchio |
| **Uniform** | Identical compute chiplets in array | Cerebras WSE-3 (conceptually) |

### 7.3 The UCIe Standard

UCIe (Universal Chiplet Interconnect Express) is the emerging standard:

| Version | Bandwidth per Pin | Total Bandwidth | Latency | Power |
|---------|------------------|----------------|---------|-------|
| UCIe 1.0 | 8 GT/s | 32 GB/s | <2ns | <0.5 pJ/bit |
| UCIe 1.1 | 16 GT/s | 64 GB/s | <2ns | <0.5 pJ/bit |
| UCIe 2.0 | 32 GT/s | 128 GB/s | <1.5ns | <0.3 pJ/bit |

---

## 8. Yield Optimization and Defect Management

### 8.1 The Yield Challenge

Yield = (functional chips / total chips produced) × 100%

| Factor | Impact on Yield |
|--------|----------------|
| Die size | Larger dies = exponentially lower yield |
| Defect density | Higher defect density = lower yield |
| Process maturity | New processes have lower yield |
| Design complexity | More transistors = more failure points |

### 8.2 Yield Improvement Strategies

| Strategy | Description | Yield Improvement |
|----------|-------------|------------------|
| **Design for Manufacturing (DFM)** | Optimize layout for manufacturing constraints | +5–15% |
| **Redundancy** | Add spare rows/columns of SRAM; disable defective ones | +10–20% |
| **In-line monitoring** | Real-time defect detection during fabrication | +5–10% |
| **Machine learning yield prediction** | Predict defects before they occur | +5–15% |
| **Lot dispositioning** | Route wafers based on predicted yield | +3–8% |

### 8.3 The Cost of Yield Loss

For a $20B fab producing B200-class chips:

| Yield | Good Dies/Wafer | Cost/Die | Annual Loss |
|-------|----------------|----------|-------------|
| 70% | 21 | $952K | — |
| 60% | 18 | $1.11M | $280M/year |
| 50% | 15 | $1.33M | $560M/year |
| 40% | 12 | $1.67M | $840M/year |

---

## 9. EDA Tools and Design Automation

### 9.1 The EDA Oligopoly

Electronic Design Automation (EDA) tools are essential for chip design:

| Company | Revenue (2025) | Market Share | Key Products |
|---------|---------------|-------------|-------------|
| Synopsys | $6.1B | ~30% | Design Compiler, ICC2, PrimeTime |
| Cadence | $4.6B | ~23% | Innovus, Tempus, Voltus |
| Siemens EDA | $2.1B | ~10% | Calibre, Catapult |
| Others | ~$7B | ~35% | Ansys, Altium, etc. |

### 9.2 AI in EDA

AI is increasingly used to design AI chips:

| Application | Tool/Company | Impact |
|------------|-------------|--------|
| **Chip floorplanning** | Google (AlphaChip), Synopsys DSO.ai | 30–50% faster design cycles |
| **Timing closure** | Cadence Cerebrus | 2–3x faster convergence |
| **Power optimization** | Synopsys AI-driven P&R | 15–25% power reduction |
| **Verification** | Synopsys VC Formal | 5–10x faster formal verification |

---

## 10. Testing and Validation

### 10.1 AI Chip Testing Pipeline

| Stage | What's Tested | Method | Duration |
|-------|--------------|--------|----------|
| **Wafer sort** | Each die on wafer | Probe station, functional test | Minutes per wafer |
| **Packaging** | — | CoWoS assembly, wire bonding | Hours |
| **Final test** | Complete package | ATE (Automated Test Equipment) | Minutes per chip |
| **Burn-in** | Reliability under stress | High temperature, voltage | Hours to days |
| **System validation** | Real workload testing | Model inference, training | Hours to weeks |

### 10.2 The Testing Cost Challenge

| Test Stage | Cost per Chip | Time |
|-----------|--------------|------|
| Wafer sort | $10–$50 | 1–5 minutes |
| Final test (ATE) | $20–$100 | 2–10 minutes |
| Burn-in | $50–$200 | 4–24 hours |
| System validation | $500–$2,000 | 1–7 days |
| **Total testing** | **$580–$2,350** | **1–8 days** |

---

## 11. Cross-References

| Document | Category | Relevance |
|----------|----------|-----------|
| `38-AI-Supply-Chain-and-Chip-Design/01-Overview.md` | Supply Chain | Strategic overview and geopolitical context |
| `02-LLMs/08-Custom-Silicon-and-AI-Hardware-2026.md` | LLMs | Detailed chip architecture comparison and cost tables |
| `30-Small-Language-Models/` | SLMs | Efficiency and compression techniques |
| `35-AI-Energy-and-Sustainability/` | Energy | Power consumption and sustainability |
| `17-Research-Frontiers-2026/` | Research | Emerging chip technologies |
| `25-Multi-Cloud-AI-Strategy/` | Multi-Cloud | Cloud procurement and deployment |

---

*Last updated: June 29, 2026*

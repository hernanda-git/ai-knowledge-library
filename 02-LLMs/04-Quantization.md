# Quantization — Comprehensive Reference

> This document provides a deep technical reference on model quantization for Large Language Models: precision formats, quantization theory, post-training methods, inference optimization, and practical deployment guidance. Written for AI engineers and ML practitioners.

---

## Table of Contents

1. [What is Quantization?](#what-is-quantization)
2. [Precision Formats](#precision-formats)
3. [Quantization Theory](#quantization-theory)
4. [Post-Training Quantization (PTQ) Methods](#post-training-quantization-ptq-methods)
5. [GPTQ — Optimal Brain Quantizer](#gptq--optimal-brain-quantizer)
6. [AWQ — Activation-Aware Weight Quantization](#awq--activation-aware-weight-quantization)
7. [GGUF/GGML Format and k-Quants](#ggufggml-format-and-k-quants)
8. [bitsandbytes](#bitsandbytes)
9. [AQLM — Additive Quantization of Language Models](#aqlm--additive-quantization-of-language-models)
10. [QuIP# — Lattice Codebook Quantization](#quip--lattice-codebook-quantization)
11. [SpQR — Sparse-Quantized Representation](#spqr--sparse-quantized-representation)
12. [HQQ — Half-Quadratic Quantization](#hqq--half-quadratic-quantization)
13. [Comparison of Quantization Methods](#comparison-of-quantization-methods)
14. [Accuracy vs Speed Tradeoffs](#accuracy-vs-speed-tradeoffs)
15. [Calibration Datasets](#calibration-datasets)
16. [Quantization for Inference vs Training](#quantization-for-inference-vs-training)
17. [Practical Guide](#practical-guide)
18. [Advanced Topics](#advanced-topics)

---

## What is Quantization?

Quantization is the process of reducing the numerical precision of model weights and/or activations from high-precision floating-point formats (e.g., FP32, BF16) to lower-precision formats (e.g., INT8, INT4, FP8). This reduces memory footprint, bandwidth requirements, and computational cost, often with minimal accuracy degradation.

**Why quantize:**

| Benefit | Explanation |
|---------|-------------|
| **Reduced memory** | 4-bit weights use 8× less memory than FP32 |
| **Faster inference** | Lower precision arithmetic is faster on modern GPUs |
| **Higher throughput** | More model fits in GPU memory = larger batch sizes |
| **Lower power** | Less data movement = less energy consumption |
| **Edge deployment** | Models fit on phones, laptops, embedded devices |
| **Larger models on same hardware** | 70B model fits on single GPU at 4-bit |

**The quantization challenge:**
- Weights are learned in high precision (FP32/BF16)
- Reducing precision introduces error
- The goal: minimize accuracy loss while maximizing compression
- Different methods strike different balances between compression, speed, and accuracy

**Key concepts:**
- **Weight quantization:** Quantizing the model parameters (static, done once)
- **Activation quantization:** Quantizing intermediate layer outputs (dynamic, per-input)
- **Quantization-aware training (QAT):** Training with simulated quantization to recover accuracy
- **Post-training quantization (PTQ):** Quantizing a pre-trained model without training

---

## Precision Formats

### FP32 (32-bit Floating Point)

**Format:** 1 sign bit + 8 exponent bits + 23 mantissa bits
**Range:** ±1.4×10⁻⁴⁵ to ±3.4×10³⁸
**Precision:** ~7 decimal digits

```
S | EEEEEEEE | MMMMMMMMMMMMMMMMMMMMMMM
0 | 01111111 | 00000000000000000000000
  (124-127 bias)
```

- **Standard precision** for model training traditionally
- **Pros:** Maximum precision, no special hardware needed
- **Cons:** 4 bytes per value, memory-intensive
- **Use:** Baseline for comparison, mixed-precision master weights

### FP16 (16-bit Float, IEEE)

**Format:** 1 sign bit + 5 exponent bits + 10 mantissa bits
**Range:** ±6.5×10⁻⁸ to ±6.5×10⁴
**Precision:** ~3.3 decimal digits

```
S | EEEEE | MMMMMMMMMM
0 | 01111 | 0000000000
  (15 bias)
```

- **Pros:** 2× memory reduction vs FP32
- **Cons:** Limited range (max 65504), precision issues with small gradients
- **Use:** Mixed-precision training (FP16 compute, FP32 master weights)
- **Limitations:** Prone to underflow for very small values, overflow for large activations
- **Not recommended for LLM training** (BF16 is preferred)

### BF16 (Brain Floating Point 16)

**Format:** 1 sign bit + 8 exponent bits + 7 mantissa bits
**Range:** Same as FP32 (±3.4×10³⁸)
**Precision:** ~2.3 decimal digits

```
S | EEEEEEEE | MMMMMMM
0 | 01111111 | 0000000
```

- **Developed by Google** for TPU training
- **Same range as FP32** but lower precision
- **Ideal for deep learning:** range matters more than precision for gradients
- **Pros:** No overflow/underflow issues during training, widely supported (A100+, H100, TPU)
- **Cons:** Slightly less precise than FP16 for very small values
- **Use:** Standard for modern LLM training (default for Llama 3, Mistral, DeepSeek)

### INT8 (8-bit Integer)

**Range:** -128 to 127 (symmetric), 0 to 255 (asymmetric)
**Format:** 8-bit two's complement integer

- **Pros:** 4× memory reduction vs FP32, 2× vs FP16, fast integer compute
- **Cons:** Very limited range, requires careful scaling
- **Use:** Weight quantization, activation quantization for inference
- **Hardware support:** Tensor Cores (Turing+, T4+), CPU (VNNI instructions)
- **Accuracy:** Typically <1% degradation for weights-only quantization

### INT4 (4-bit Integer)

**Range:** -8 to 7 (symmetric), 0 to 15 (asymmetric)

- **Pros:** 8× memory reduction vs FP32
- **Cons:** Very coarse quantization, higher accuracy loss
- **Use:** Extreme compression for large models, on-device deployment
- **Hardware support:** Limited (some GPU tensor cores support INT4, CPU via lookup tables)
- **Accuracy:** Variable, requires advanced methods (GPTQ, AWQ, GGUF k-quants)

### INT2 / 2-bit (Binary/Ternary)

**Range:** Typically {-1, 0, 1} (ternary) or {-1, 1} (binary)

- **Extreme compression** (16× vs FP32)
- **High accuracy loss** — mostly experimental
- **Use:** Research, extreme edge compression

### FP8 (8-bit Float)

Two variants defined by the OCP (Open Compute Project):

**FP8 E4M3 (4 exponent + 3 mantissa):**
```
S | EEEE | MMM
Range: ±448
Precision: ~1.1 decimal digits
```
- **Pros:** Wider range than INT8, better for activations
- **Use:** H100/H200 native FP8 tensor cores, training and inference

**FP8 E5M2 (5 exponent + 2 mantissa):**
```
S | EEEEE | MM
Range: ±57,344
Precision: ~0.6 decimal digits
```
- **Pros:** Very wide range, good for gradients
- **Use:** Gradient accumulation in FP8 training

**FP8 for LLMs:**
- H100 supports native FP8 matrix multiplication (2× throughput vs FP16)
- Models can be trained entirely in FP8 (DeepSeek-V3 was partially FP8-trained)
- Weight-only FP8 gives near-lossless quantization
- FP8 activation quantization requires careful handling of outliers

### NF4 (Normal Float 4)

**Format:** 4-bit non-uniform quantization, specifically designed for normally distributed weights.

**Developed by QLoRA (Dettmers et al., 2023):**
- Standard INT4 assumes uniform distribution — bad for normally distributed weights
- NF4 maps values to a distribution that **matches the expected normal distribution of neural network weights**
- The quantization levels are non-uniform: more levels near zero (where weights cluster) and fewer in the tails

**NF4 quantization levels (16 values in range [-1, 1]):**
```
Levels: -1.0, -0.84, -0.68, -0.52, -0.36, -0.2, -0.04, 0.0,
         0.04, 0.2, 0.36, 0.52, 0.68, 0.84, 1.0
```
(Actual levels differ slightly — this is approximate.)

**Advantages:**
- Better accuracy than uniform INT4 for normally distributed weights
- Specifically designed for LLM weight quantization
- Used in bitsandbytes 4-bit quantization (QLoRA)

### FP4 (4-bit Float)

**Format:** Not standardized, typically E2M1 (2 exponent + 1 mantissa)

- **Pros:** Better range than INT4 for the same bit width
- **Cons:** Limited hardware support
- **Use:** Experimental, research-focused
- **Comparison with NF4:** NF4 is tailored for bell-shaped distributions, FP4 for exponential ones

### MXFP4 (Microscaling FP4)

**Part of the OCP Microscaling (MX) format specification:**
- Uses a **shared exponent** across a block of values (e.g., 32 elements)
- E1M2 format: 1 exponent bit + 2 mantissa bits per value
- Block-level scaling: 8-bit shared scale factor per 32 elements
- **Effective bit-width:** ~4.25 bits per value (4 + 8/32)

**Advantages:**
- Hardware-friendly (shared exponent reduces compute)
- Good accuracy for block-wise patterns
- Being standardized for future hardware (AMD, Intel, Qualcomm)

---

## Quantization Theory

### Uniform Quantization

The simplest form: divide the value range into equal-sized intervals.

**Affine quantization (asymmetric):**
```
q = round((x - min) / scale)
where scale = (max - min) / (2^n - 1)
```

**Scale quantization (symmetric):**
```
q = round(x / scale)
where scale = max(|x|) / (2^(n-1) - 1)
```

**Dequantization:**
```
x̂ = q × scale + zero_point
```

**Error analysis:**
- For uniform quantization, the rounding error is bounded by scale/2
- Absolute error: |x - x̂| ≤ scale/2
- Relative error: |x - x̂| / |x| can be very large for small x
- This is why uniform quantization struggles with outliers

**Quantization grid:**
```
INT4 (symmetric):
  Levels: -8, -7, ..., -1, 0, 1, ..., 7 (range centered at 0)
  Gap between levels: scale (uniform)
  
INT4 (asymmetric):
  Levels: 0, 1, 2, ..., 15 (starting from 0)
  Gap between levels: scale (uniform)
```

### Non-Uniform Quantization

Levels are not equally spaced — more levels in important regions, fewer in less important regions.

**Logarithmic quantization:**
```
q = round(log(|x|) / scale)
```
- More levels near zero (where small values matter)
- Fewer levels at large magnitudes
- Good for values with exponential distribution

**NF4 (Normal Float):**
- Levels adapted to normal distribution
- More levels near zero, fewer in tails
- Specifically designed for LLM weights

**Lattice quantization:**
- Uses structured codebooks (e.g., E₈ lattice from QuIP#)
- Points are arranged in high-dimensional space rather than per-element
- Better packing efficiency than scalar quantization

**Advantages of non-uniform:**
- Better accuracy for non-uniform value distributions
- Can achieve lower bit-width for the same accuracy
- But more complex to compute (lookup tables needed)

### Symmetric vs Asymmetric

**Symmetric:**
- Range is symmetric around zero: [-max, max]
- Zero-point = 0 (simpler, faster)
- Better for weights that are naturally symmetric (e.g., after LayerNorm)
- Used by: INT8 symmetric, NF4

**Asymmetric:**
- Range adapts to actual min/max values
- Has zero-point offset
- Better for activations (which may be non-negative after ReLU)
- Used by: INT8 affine quantization, most activation quantization

**Comparison:**
```
Symmetric INT4: range [-8, 7], levels at -8*scale, -7*scale, ..., 7*scale
Asymmetric INT4: range [0, 15], levels at 0, 1*scale, ..., 15*scale

For weights centered at 0: symmetric is better
For ReLU activations (all positive): asymmetric is better
```

### Calibration

The process of determining the optimal quantization parameters (scale, zero_point) from data.

**Steps:**
1. Select a **calibration dataset** (representative of inference inputs)
2. Run the model on this dataset
3. Collect statistics of weights and activations (min, max, percentiles, distributions)
4. Compute optimal scale and zero_point from these statistics

**Calibration methods:**

1. **Min-max:** scale = (max - min) / levels
   - Simple but sensitive to outliers
   - If one value is 100× larger than others, the rest are poorly quantized

2. **Percentile:** Use specified percentile (e.g., 99.9%) instead of max
   - More robust to outliers
   - 99.9% calibration clips the top 0.1% of values (introduces clipping error)

3. **KL-divergence (entropy):** Minimize KL divergence between original and quantized distributions
   - Finds the optimal clipping threshold
   - Used by NVIDIA's TensorRT for INT8 calibration
   - Computes: argmin_threshold KL(original_dist || quantized_dist(threshold))

4. **MSE minimization:** Minimize mean squared error
   - Find threshold that minimizes Σ(x - clamp(x, -t, t))²
   - Closed-form for Gaussian distributions

5. **Cross-entropy minimization:** Minimize task loss
   - The most accurate but most expensive
   - Requires evaluating the full model on calibration data

### Quantization Error

Total quantization error = **rounding error** + **clipping error**

**Rounding error:**
- Caused by approximating a continuous value to the nearest discrete level
- Magnitude: bounded by scale/2
- Distribution: roughly uniform in [-scale/2, scale/2]
- Can be reduced by: increasing bit-width, using non-uniform quantization

**Clipping error:**
- Caused by values outside the quantization range being clipped to min/max
- Magnitude: can be very large (outliers)
- More harmful than rounding error for accuracy
- Can be reduced by: widening quantization range (but this increases rounding error)

**Trade-off:**
```
Wide range: small clipping error, large rounding error
Narrow range: small rounding error, large clipping error
Optimal: balance the two based on value distribution
```

### Outlier-Aware Quantization

LLMs are known to have **activation outliers** — a small fraction (0.1-1%) of activation values that are 10-100× larger than the rest.

**Outlier characteristics:**
- Emerge at specific feature dimensions across sequences
- More prominent in larger models
- Grow with model size and depth
- Found in specific attention heads and hidden dimensions
- Cause severe accuracy loss if not handled properly

**Strategies for handling outliers:**

1. **Per-channel/per-token scaling:**
   - Each channel or token has its own quantization scale
   - Outlier channels get larger ranges
   - Used by: AWQ, GPTQ with per-channel quantization

2. **Mixed-precision decomposition:**
   - Keep outlier dimensions in FP16 while quantizing the rest
   - Outlier detection: dimensions where |value| > 3σ from mean
   - Typically 0.1-1% of dimensions kept in high precision
   - Used by: SpQR, Outlier Suppression+

3. **SmoothQuant approach:**
   - Transfer quantization difficulty from activations to weights
   - Multiply activations by a smoothing factor s
   - Divide weights by the same factor s
   - This "smooths" activation outliers
   - s chosen to balance activation and weight quantization errors

4. **Overflow-aware clipping:**
   - Pre-compute clipping thresholds that minimize squared error
   - Consider both quantization noise AND outlier overflow
   - Used by: OmniQuant, AdaRound

---

## Post-Training Quantization (PTQ) Methods

PTQ is the most practical approach — quantize a pre-trained model without needing to retrain it.

### Overview of PTQ Methods

| Method | Bit-width | Algorithm | Calibration Data | Accuracy Preservation |
|--------|-----------|-----------|-----------------|---------------------|
| GPTQ | 2-8 bit | Hessian-based weight quantization | 128 samples | Excellent (4-bit) |
| AWQ | 2-8 bit | Activation-aware scaling | 128 samples | Excellent (4-bit) |
| GGUF (k-quants) | 2-8 bit | Importance-aware quantization | Optional (imatrix) | Very good |
| bitsandbytes | 4-8 bit | Block-wise NF4/FP4 | None | Good |
| AQLM | 2-4 bit | Additive quantization codes | 256 samples | Excellent (2-bit) |
| QuIP# | 2-4 bit | Incoherence + lattice codebook | 256 samples | Excellent (2-bit) |
| SpQR | 3-4 bit | Sparse + quantized decomposition | ~512 samples | Very good |
| HQQ | 1-8 bit | Half-quadratic optimization | None | Good |

### Common PTQ Workflow

```
1. Load pre-trained model (FP16/BF16)
2. Select calibration dataset (128-1024 samples)
3. For each layer:
   a. Collect weight statistics
   b. Optionally collect activation statistics by running calibration data
   c. Compute quantization parameters (scale, zero_point)
   d. Apply quantization (round weights to nearest quantized value)
   e. Optionally adjust weights to minimize layer output error
4. Save quantized model
5. Evaluate accuracy on benchmark tasks
```

---

## GPTQ — Optimal Brain Quantizer

GPTQ (Frantar et al., 2023 — GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers) is one of the most widely-used weight quantization methods.

### Core Idea: Optimal Brain Quantization

GPTQ is adapted from **Optimal Brain Compression (OBC)** , which itself is based on **Optimal Brain Surgeon (OBS)** — a classical pruning method.

**OBS principle:**
- Instead of independently rounding each weight, consider the **Hessian** (second-derivative) of the loss with respect to weights
- The Hessian tells us how important each weight is
- Quantize the least important weights first, adjust remaining weights to compensate for the error

**Key insight:**
- Quantization introduces error ε into a weight vector w
- The impact on the loss function is approximately: ΔL ≈ ½ εᵀ H ε
- Where H is the Hessian matrix (∂²L/∂w²)
- We want to choose which weights to quantize and how to adjust remaining weights to minimize ΔL

### Mathematical Formulation

**Step 1: Layer-wise quantization**

GPTQ works **layer by layer** (or sometimes block by block):
- For each linear layer W (of shape d_row × d_col):
  - Find quantization q for W
  - Minimize: ||Wx - qx||²₂ for calibration inputs x
  - Equivalently: minimize ||W - q||²_H where ||A||²_H = A·H·Aᵀ

**Step 2: Hessian computation**

The Hessian for a linear layer's weights:
```
H = 2 · X·Xᵀ
```
where X is the calibration data's activation matrix for this layer.

- **Dampening:** H is regularized by adding a small diagonal: H ← H + λ·I
  - λ is typically set to 0.01×mean(diag(H))
  - Prevents numerical instability

**Step 3: Greedy quantization**

For each column of W:
1. Compute the inverse Hessian H⁻¹ (using Cholesky decomposition)
2. For each weight in the column (in order of saliency):
   a. Quantize the weight to its nearest quantized value
   b. Compute the error: δ = q - w
   c. Update remaining (un-quantized) weights in the same column to compensate: w_remaining ← w_remaining - δ · H⁻¹[remaining, i] / H⁻¹[i, i]
   d. Remove the quantized dimension from H⁻¹ (using the "OBS update")

**Step 4: Group-wise quantization**

For better accuracy with small bit-widths:
- Weights are quantized in groups (e.g., group_size = 128)
- Each group has its own scale and zero_point
- This allows finer granularity: different parts of the weight matrix use different quantization ranges

### GPTQ Algorithm Pseudocode

```
def gptq_quantize(W, H, bits, group_size):
    """W: weight matrix (out_dim × in_dim)
       H: Hessian (in_dim × in_dim)
       bits: target bit-width (e.g., 4)
       group_size: size of quantization groups (e.g., 128)
    """
    # Dampen Hessian
    damp = 0.01 * torch.mean(torch.diag(H))
    H = H + damp * torch.eye(H.shape[0], device=H.device)
    
    # Cholesky decomposition of H⁻¹
    H_inv = torch.inverse(H)
    
    # Initialize quantized weights
    Q = W.clone()
    scale = W.abs().mean() / (2^(bits-1) - 1)  # approximate scale
    
    # Process columns in groups
    for start in range(0, W.shape[1], group_size):
        end = min(start + group_size, W.shape[1])
        W_g = W[:, start:end]
        H_inv_g = H_inv[start:end, start:end]
        
        # Quantize each column
        for i in range(W_g.shape[1]):
            w = W_g[:, i]  # current column
            h_inv_diag = H_inv_g[i, i]
            
            # Quantize
            q = torch.round(w / scale) * scale
            q = torch.clamp(q, -2^(bits-1), 2^(bits-1) - 1) * scale
            
            # Compute error
            err = q - w
            
            # Update remaining weights in this group
            if i < W_g.shape[1] - 1:
                update = err / h_inv_diag
                W_g[:, i+1:] -= torch.outer(update, H_inv_g[i, i+1:])
            
            Q[:, start + i] = q
    
    return Q, scale
```

### GPTQ Configurations

**Common settings:**
- `bits`: 2, 3, 4, 8 (4 is most common)
- `group_size`: 32, 64, 128, -1 (per-channel)
- `sym`: True (symmetric) or False (asymmetric)
- `desc_act`: True (descending activation ordering) or False

**Group size impact:**
```
group_size = 128:   Good accuracy, reasonable overhead
group_size = 32:    Better accuracy, more metadata storage
group_size = -1:    Per-channel quantization, minimal overhead
```

**Descending activation ordering (desc_act=True):**
- Sort weight columns by their activation magnitude
- Quantize the most important columns first
- This gives them the best possible quantization
- Improves accuracy for outlier-heavy layers

### GPTQ Performance

**Accuracy at 4-bit (Llama-2-7B):**
```
FP16 baseline:      Wikitext2 PPL 5.12
GPTQ 4-bit (g128):  Wikitext2 PPL 5.25  (+0.13, ~0.3% degradation)
GPTQ 3-bit (g128):  Wikitext2 PPL 5.80  (+0.68, ~1.2% degradation)
GPTQ 2-bit (g128):  Wikitext2 PPL 8.52  (+3.40, ~6% degradation)
```

**Inference speedup:**
- Without GPU kernel optimization: minimal speedup (still loads as FP16 into CUDA cores)
- With GPTQ kernels (ExLlama v2, AutoGPTQ): 2-3× speedup
- On CPU: 4× speedup (INT4 is much faster than FP32 on CPU)

---

## AWQ — Activation-Aware Weight Quantization

AWQ (Lin et al., 2023 — AWQ: Activation-aware Weight Quantization for On-Device LLM Compression) is a simple but effective method that uses **activation statistics** to guide weight quantization.

### Core Insight

Not all weights are equally important. A small fraction of weights (~1%) are **salient** — they handle the large activation values. These weights should be preserved with higher precision.

**Key observation:**
- 0.1-1% of weight channels correspond to 99% of activation magnitudes
- Quantizing these "salient" channels causes most of the accuracy loss
- Protecting just these channels dramatically improves quantization quality

### AWQ Algorithm

**Step 1: Compute activation statistics**

Run calibration data through the model, collecting per-channel activation magnitudes.

```
activation_scale[channel] = mean(|activation[channel]|) over all calibration tokens
```

**Step 2: Identify salient channels**

Channels with activation_scale > threshold_quantile (e.g., top 1%) are "salient".

**Step 3: Apply per-channel scaling**

Instead of keeping salient channels in high precision (complex hardware support), AWQ uses a clever trick:

- Multiply weights of salient channels by a scaling factor s > 1
- This effectively **increases the quantization resolution** for those channels
- All channels remain quantized to the same bit-width

```
W'[channel] = W[channel] × s[channel]
```

The scaling factor s is chosen per-channel to minimize:
```
min_s || W·x - Q(W·s) · (x / s) ||²₂
```
where Q is the quantization function.

**Step 4: Balance between rounding and clipping**

AWQ finds the optimal scaling factor s that balances:
- **Rounding error** (more resolution for salient weights)
- **Clipping error** (non-salient weights may clip after scaling)

The optimal s is found via hyperparameter search (usually 0-2 with step 0.05).

**Scaling integration:**
```
Original: y = W·x
Scaled:   y = (W·s) · (x / s)
         = Q(W·s) · (x / s)   (after quantization)
```

The scaling is "folded" into the preceding LayerNorm or linear layer during inference, so there's no extra computation.

### AWQ vs GPTQ

| Aspect | AWQ | GPTQ |
|--------|-----|------|
| **Complexity** | Simple (scaling only) | Complex (Hessian-based) |
| **Calibration data** | 128 samples | 128 samples |
| **Quantization time** | Minutes | Hours |
| **Accuracy (4-bit)** | Comparable to GPTQ | Slightly better |
| **Accuracy (3-bit)** | Worse than GPTQ | Better |
| **Implementation** | Easy | Complex (Cholesky, inverses) |
| **Hardware support** | Excellent | Good (special kernels needed) |

### AWQ Configurations

```
bits: 4 (default), 3, 2, 8
group_size: 128 (default), 64, 32, -1
zero_point: True (default), False
q_group_size: 128 (weight grouping)
w_bit: 4
version: GEMM (default), GEMV (for batch=1)
```

**AWQ + GEMM kernel:**
- Special CUDA kernel for INT4 matrix multiplication
- 3-4× speedup vs FP16
- Supports batch inference (multiple sequences simultaneously)

**AWQ + GEMV kernel:**
- Optimized for batch=1 (single sequence generation)
- Memory-bandwidth bound scenario
- 2-3× speedup

---

## GGUF/GGML Format and k-Quants

GGUF (GPT-Generated Unified Format) is a file format for quantized models, developed for the llama.cpp project. It is the most widely-used format for **CPU-based LLM inference**.

### GGML/GGUF File Format

**GGML (old format):**
- Tensor-level storage with serialization
- Supported multiple quantization types
- Limited metadata
- Superseded by GGUF

**GGUF (current format, v3):**
- Rich metadata at the file level
- Key-value pairs for model metadata (architecture, tokenizer, chat template)
- Backward compatibility (forward-compatible reading)
- Supports all llama.cpp features

**GGUF file structure:**
```
[GGUF Header]
  - Magic: "GGUF" 
  - Version (v3)
  - Tensor count
  - Metadata KV count
  
[Metadata KV Pairs]
  - general.architecture: "llama"
  - general.name: "Llama-3-8B"
  - llama.context_length: 8192
  - tokenizer.ggml.model: "gpt2"
  - tokenizer.ggml.tokens: [...]
  
[Padding]
  
[Token Data]
  
[Padding]
  
[Tensor Data]
  - Tensor name, dimensions, quantization type
  - Raw quantized weight data
```

### k-Quant Types

The "k-quants" are a family of quantization types designed by the llama.cpp team. The 'k' stands for "k-quant" (named after the developer, Kerfuffle/Lord-Deathray).

**Design philosophy:**
- Importance-aware quantization
- Blocks of weights are assigned different quantization levels based on their importance
- More important blocks get higher precision, less important blocks get lower precision
- This achieves better quality than uniform quantization at the same bit-width

### Available k-Quant Types

**Q2_K (2.5625 bpw — bits per weight):**
```
- Super-block: 4096 elements, FP16 scale
- Sub-blocks: 16 elements each
- 6-bit scale per sub-block
- Quantization: 2-bit for most weights
- Effective: ~2.56 bpw
- Use: Extreme compression, highest quality loss
```

**Q3_K_S (3.4 bpw — Small):**
```
- Single quantization level across all sub-blocks
- 3-bit quantization
- No importance splitting
- Effective: ~3.4 bpw
- Use: Smallest 3-bit size, moderate quality
```

**Q3_K_M (3.5 bpw — Medium):**
```
- 16 sub-blocks of 16 weights each
- Some sub-blocks (higher importance) use 4-bit
- Others use 3-bit
- Effective: ~3.5 bpw
- Use: Good quality/size balance at 3-bit
```

**Q3_K_L (3.6 bpw — Large):**
```
- More sub-blocks receive 4-bit quantization
- Fewer at 3-bit
- Effective: ~3.6 bpw
- Use: Highest quality 3-bit option
```

**Q4_K_S (4.3 bpw — Small):**
```
- All sub-blocks use 4-bit quantization
- Single scale per super-block
- Effective: ~4.3 bpw
- Use: Standard 4-bit, good quality
```

**Q4_K_M (4.5 bpw — Medium):**
```
- Sub-blocks split into high-importance and low-importance groups
- High-importance: 5-bit quantization
- Low-importance: 4-bit quantization
- Effective: ~4.5 bpw
- Use: Best 4-bit quality, most common choice
```

**Q4_K_L (4.6 bpw — Large):**
```
- Even more sub-blocks at 5-bit
- Effective: ~4.6 bpw
- Use: Highest quality 4-bit option
```

**Q5_K_S (5.3 bpw — Small):**
```
- All sub-blocks at 5-bit
- Effective: ~5.3 bpw
- Use: Near-lossless 5-bit
```

**Q5_K_M (5.4 bpw — Medium):**
```
- Some sub-blocks at 6-bit
- Some at 5-bit
- Effective: ~5.4 bpw
- Use: Best 5-bit quality
```

**Q6_K (6.5 bpw):**
```
- All weights at 6-bit
- Effective: ~6.5 bpw
- Use: Very high quality, near-lossless
```

**Q8_K (8.5 bpw):**
```
- 8-bit quantization
- Effective: ~8.5 bpw
- Use: Virtually lossless
```

### Legacy Quantization Types (Deprecated)

**q4_0 (4.5 bpw — legacy):**
- 32-weight blocks
- 5-bit scale (half precision)
- No importance weighting
- 4-bit symmetric quantization
- Effective bit-width: 4.5 bpw
- Lower quality than Q4_K_M

**q4_1 (4.5 bpw — legacy):**
- 32-weight blocks
- Scale + min (asymmetric)
- 4-bit asymmetric quantization
- Similar quality to q4_0

**q5_0 / q5_1:**
- Similar to q4_0/q4_1 but at 5-bit
- Superseded by Q5_K variants

**q2_K / q3_K / q4_K / q5_K / q6_K / q8_K:**
- These are the "K" variants, though the naming scheme was refined
- The current naming (Q2_K, Q3_K_S, etc.) supersedes these

### iMatrix (Importance Matrix)

An **importance matrix** (imatrix) is a calibration-derived matrix that tells llama.cpp which weights are more important.

**How it works:**
1. Run a calibration dataset (e.g., 512 text samples) through the model
2. For each layer, compute the Hessian approximation (similar to GPTQ)
3. The Hessian diagonal is the importance matrix
4. Store this matrix alongside the quantized model
5. During quantization, use the imatrix to decide which sub-blocks get higher precision

**iMatrix-aware quantization:**
- llama.cpp's k-quant system uses imatrix information
- Without imatrix: importance is estimated heuristically (based on weight magnitude)
- With imatrix: importance is based on actual activation statistics
- This improves quality by 0.1-0.3 perplexity points

**Using imatrix:**
```
# Generate importance matrix
./llama-imatrix -m model.gguf -f calibration.txt -o model.imatrix

# Quantize with imatrix
./llama-quantize --imatrix model.imatrix model.gguf Q4_K_M
```

---

## bitsandbytes

bitsandbytes is a library for efficient LLM quantization, developed by Tim Dettmers. It provides 8-bit and 4-bit quantization optimized for GPU inference and training.

### 8-bit Quantization (LLM.int8())

**Core algorithm — Vector-wise quantization:**

Instead of per-tensor or per-channel quantization, bitsandbytes uses **per-token** and **per-column** quantization:

```
For each token's hidden states:
  - Token dimension (row): quantized independently
For each column of the weight matrix:
  - Column dimension: quantized independently
```

This gives fine-grained quantization scales.

**Mixed-precision decomposition (LLM.int8()):**

The key insight: LLMs have **emergent outliers** at scale — certain hidden dimensions have values 10-100× larger than the rest.

1. Separate weights into "regular" and "outlier" groups
2. Outlier detection: columns where the activation value exceeds a threshold (typically 6.0)
3. Regular values (99.9% of weights) → INT8 quantization
4. Outlier values (0.1% of weights) → FP16 (kept in high precision)

```
y = INT8_matmul(W_regular, x_regular)  +  FP16_matmul(W_outlier, x_outlier)
```

**Memory savings:**
- Weights: 2× less memory than FP16 (8-bit vs 16-bit)
- Activations: 2× less memory
- Overall: ~50% reduction for inference

**Performance:**
- 7B model: ~1.0 perplexity point degradation (small)
- 13B model: ~0.8 PPL degradation
- 70B model: ~0.4 PPL degradation (larger models are more robust)
- Outlier dimension count: 15-50 per layer (increases with model size)

### 4-bit Quantization (QLoRA)

QLoRA (Dettmers et al., 2023 — QLoRA: Efficient Finetuning of Quantized Language Models) introduced 4-bit quantization for both inference and fine-tuning.

**NF4 (Normal Float 4-bit):**

The key innovation: weights in neural networks follow a **normal distribution** (bell-shaped, centered at 0). Standard uniform INT4 quantization wastes levels in the tails and has coarse resolution near zero (where most weights are).

NF4 uses the **quantile function** of the normal distribution:
- Compute the CDF of the normal distribution
- Choose 16 levels (4-bit) that are equally spaced in terms of CDF probability
- This means: levels are closer together near zero (high probability region) and farther apart in the tails (low probability region)

**NF4 level computation:**
```
# 16 levels for 4-bit quantization
quantiles = torch.tensor([0, 1/15, 2/15, ..., 15/15])
# Map to normal distribution quantiles
levels = torch.distributions.Normal(0, 1).icdf(quantiles)
# Normalize to [-1, 1]
levels = levels / levels.abs().max()
```

**NF4 vs FP4:**
```
NF4:  Designed for normal-distributed weights. Higher accuracy near zero.
FP4:  Designed for general use. Better dynamic range.
Empirical: NF4 is 1-3 perplexity points better than FP4 for LLM weights.
```

**Double Quantization:**

bitsandbytes also uses **double quantization** for the quantization scales themselves.

**Why double quantization:**
- Each group of weights has a quantization scale (FP32)
- With small group sizes (64-256), scales add significant overhead
- Example: group_size=64, 4096-dim layer → 64 scales per row × number of rows
- Each scale is 4 bytes (FP32) → overhead of 256 bytes per 64×16 weights = 25% overhead

**Double quantization:**
1. Quantize weights → compute per-group scales in FP32
2. Quantize the FP32 scales into INT8 → store as INT8
3. Store the 2nd-level scale in FP32 (one per 64 groups)

```
Group scales: FP32 → INT8 (4× reduction)
2nd-level scale: one FP32 per 64 groups (negligible overhead)
Total scale overhead: ~0.5% of weight size
```

**Block-wise quantization:**
- Weights within each group are quantized together
- Group size: 64 (default) or 32
- Each group has its own scale and zero_point
- Enables fine-grained adaptation to weight distribution

### bitsandbytes for Inference

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # "nf4" or "fp4"
    bnb_4bit_use_double_quant=True,   # Double quantization for scales
    bnb_4bit_compute_dtype=torch.bfloat16,  # Computation precision
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
```

**Compute dtype impact:**
- `bnb_4bit_compute_dtype=torch.bfloat16`: Matrix multiplies in BF16 (best quality)
- `bnb_4bit_compute_dtype=torch.float16`: Slightly faster, slightly worse quality
- `bnb_4bit_compute_dtype=torch.float32`: Slow but accurate

**Memory requirements (4-bit with double quantization):**
```
7B model: ~4 GB VRAM (vs ~14 GB FP16)
13B model: ~7 GB VRAM (vs ~26 GB FP16)
70B model: ~35 GB VRAM (vs ~140 GB FP16)
```

### bitsandbytes for Training (QLoRA)

```python
from peft import LoraConfig, get_peft_model

# LoRA configuration
lora_config = LoraConfig(
    r=16,          # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to quantized model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Only ~0.1% of parameters are trainable
```

**QLoRA training characteristics:**
- Base weights: 4-bit (NF4 or FP4), frozen
- LoRA adapters: BF16/FP16, trainable
- Gradients flow through the quantized weights (high precision gradients update LoRA weights)
- Memory: 4GB (7B model) → allows fine-tuning on consumer GPUs
- Quality: approaches full fine-tuning quality with rank 64+

---

## AQLM — Additive Quantization of Language Models

AQLM (Egiazarian et al., 2024 — Additive Quantization for Extremely Compressed LLMs) achieves state-of-the-art results at very low bit-widths (2-bit).

### Core Idea: Additive Quantization

Instead of quantizing each weight independently, AQLM represents groups of weights as **sums of codebook vectors**.

**Standard quantization:**
- Each weight → nearest quantized value
- Example: 4-bit → 16 possible values per weight

**Additive quantization:**
- A group of weights (e.g., 8 weights) → represented as sum of M codebook vectors
- Each codebook vector comes from a learned codebook matrix
- The codebook indices are quantized (not the weights directly)

```
W_group ≈ Σ_m C_m[idx_m]
where:
  W_group: group of weights (e.g., 8 values)
  C_m: codebook m (e.g., 256 entries × 8 dimensions)
  idx_m: code index for codebook m
```

**Why additive quantization works:**
- Exponentially more representable states per group
- 2 codebooks of 256 entries each → 256² = 65,536 representable states for a group
- Equivalent to log₂(65536)/8 = 2 bits per weight, but with much better quality
- The additive structure captures interdependencies between weights

### AQLM Architecture

**Codebooks:**
- Typically 2 codebooks per layer (M=2)
- Each codebook: 256 entries (8-bit index) × group_dim (e.g., 8)
- Total: M × 256 × group_dim values per layer
- Codebooks are layer-specific (not shared)

**Residual quantization:**
1. First codebook captures the coarse structure of the weights
2. Second codebook captures the residual (error from first quantization)
3. More codebooks can be added for higher quality

**Outlier-aware training:**
- AQLM identifies outlier weights in each layer
- Outliers are quantized with higher precision (more codebooks or separate treatment)
- Improves quality at very low bit-widths

### Training

AQLM requires an **optimization process** to learn both:
1. The codebooks (C₁, C₂, ..., C_M)
2. The indices (idx_m for each group)

**Training loop:**
```
1. Initialize codebooks (random or k-means on weights)
2. For each iteration:
   a. For each group of weights:
      - Find optimal codebook indices (via beam search or iterative refinement)
   b. Update codebook entries to minimize reconstruction error
3. Repeat until convergence (typically 10-20 iterations)
```

**Beam search for index assignment:**
- For each group, try multiple combinations of codebook indices
- Keep the combination with lowest reconstruction error
- Beam width: 4-8 (number of candidate combinations to track)

### AQLM Performance

**Accuracy at 2-bit (Llama-2-7B):**
```
FP16:              PPL 5.12
AQLM-2bit (2×256): PPL 5.70 (+0.58)  — best 2-bit result
GPTQ-2bit:         PPL 8.52 (+3.40)  — much worse
QuIP#-2bit:        PPL 5.80 (+0.68)  — comparable
```

**Key advantages:**
- Best accuracy at 2-bit quantization
- Good 3-4 bit performance (comparable to GPTQ)
- **Limitations:** Slower inference (need to reconstruct weights from codebooks), larger calibration dataset needed

**Inference optimization:**
- Codebook lookups can be pre-computed
- For small batch sizes: reconstruct weights on-the-fly (overhead)
- For large batch sizes: reconstruct once, reuse

---

## QuIP# — Incoherence Processing + Lattice Codebooks

QuIP# (Tseng et al., 2024 — QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks) achieves excellent 2-bit results through two key innovations.

### Incoherence Processing

**The problem:**
- Weight matrices have **structure** (correlations between rows/columns)
- This structure makes quantization harder — errors compound
- Random matrices (no structure) are easier to quantize

**The solution:**
- Apply a random orthogonal transformation (Hadamard matrix) to the weights
- This "whitens" the weights — makes them look more like random noise
- Quantization performs better on unstructured representations
- After quantization, apply the inverse transformation

```
W_processed = H · W · H^T
Q = quantize(W_processed)
W_quantized = H^T · Q · H
```

**Hadamard matrix properties:**
- H · H^T = I (orthogonal)
- H · (random vector) → random vector with i.i.d. components (decorrelation)
- Fast multiplication: O(n log n) via Walsh-Hadamard transform
- No multiplications needed (entries are ±1)

**Benefits:**
- Reduces outlier structure in weights
- Makes quantization error more uniformly distributed
- Improves 2-4 bit quality by 0.2-0.5 perplexity points

### Lattice Codebooks (E₈)

**Standard codebook limitation:**
- Codebook entries are independent points in space
- Packing efficiency is limited by the "sphere-packing" problem
- For 8-dimensional space, optimal packing uses E₈ lattice

**E₈ (E8) Lattice:**
- The densest sphere packing in 8 dimensions
- 240 points at the minimum distance, billions of points total
- Optimal for representing 8-dimensional vectors (like weight groups)
- Can achieve higher effective resolution per bit

**E₈ quantization:**
1. Take a group of 8 weights
2. Find the nearest E₈ lattice point (fast algorithm exists)
3. Store the quantized index (variable bit rate)
4. Decode: table lookup of E₈ point coordinates

**Codebook size:**
- QuIP# typically uses 2^16 (65,536) entries per codebook
- Each entry is an 8-dimensional vector
- Total: 2^16 × 8 = 524,288 values per codebook
- Codebooks are fine-tuned on the incoherence-processed weights

### QuIP# Algorithm

```
1. Compute incoherence transform H (random Hadamard matrix)
2. Apply H to weights: W̃ = H · W · H^T
3. For each 8-weight group in W̃:
   a. Search E₈ lattice for nearest point
   b. Store lattice index
4. Fine-tune codebook entries via k-means
5. Apply inverse transform
```

**Performance at 2-bit:**
```
QuIP#-2bit (Llama-2-7B): PPL 5.80  (+0.68)
vs GPTQ-2bit:             PPL 8.52  (+3.40)  
vs AQLM-2bit:             PPL 5.70  (+0.58)
```

---

## SpQR — Sparse-Quantized Representation

SpQR (Dettmers et al., 2023 — SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression) combines **sparsity** with quantization.

### Core Idea

A small fraction of weight values are **outliers** that contribute disproportionately to model quality. SpQR keeps these outliers in high precision (sparse storage) while quantizing the remaining weights.

**SpQR pipeline:**
```
1. Identify outlier weights (typically 1-5% of weights)
2. Outlier criteria: weight × activation_magnitude > threshold
3. Store outliers in sparse format: (index, value) pairs in FP16
4. Quantize remaining (non-outlier) weights with standard method
5. During inference: reconstruct weights as sparse + quantized
```

**Outlier selection criteria:**
```
saliency(w_ij) = |w_ij| × mean(|activation_j|) 
```

Where activation_j is the mean activation of weight column j over a calibration dataset.

**Sparse format:**
- Outlier indices: variable-length encoding (typically 2 bytes per index)
- Outlier values: FP16 (2 bytes per value)
- Overhead: ~5% × (2+2) = 0.2 bytes per weight
- For a 7B model: ~1.4B × 0.2 = ~280MB for sparse outliers

**Quantization of non-outliers:**
- 3-4 bit quantization (can use GPTQ-like approach)
- Smaller range (outliers removed) → better quantization quality
- Group size: 32-128

### SpQR Performance

**Llama-2-7B:**
```
FP16:             PPL 5.12
SpQR-4bit:        PPL 5.14 (+0.02)  — nearly lossless
SpQR-3bit:        PPL 5.28 (+0.16)  — very good
GPTQ-4bit:        PPL 5.25 (+0.13)  — SpQR is better
```

**Trade-offs:**
- Sparse storage adds irregular memory access
- Mixed-precision (sparse FP16 + INT4) is harder to optimize
- Best for scenarios where near-lossless 4-bit is needed
- Less commonly used than GPTQ/AWQ due to implementation complexity

---

## HQQ — Half-Quadratic Quantization

HQQ (Badri & Shaji, 2024 — Half-Quadratic Quantization of Large Machine Learning Models) is a **fast, calibration-free** quantization method.

### Core Idea

HQQ uses **half-quadratic optimization** to find optimal quantization parameters without requiring calibration data.

**The optimization problem:**
```
min_{scale, zero_point, Q} || W - Q ||²₂ + λ·||Q - q(W; scale, zero_point)||²₂
```

Where:
- W: original weights
- Q: quantized weights
- q(W; scale, zero_point): quantization operation
- λ: regularization parameter

**Half-quadratic algorithm:**
1. Fix (scale, zero_point), optimize Q:
   - Q = solution to quadratic form (closed form)
   - Combines the original weights with the quantization constraint
2. Fix Q, optimize (scale, zero_point):
   - Update scale and zero_point to best match Q
3. Repeat until convergence (typically 3-5 iterations)

**Why no calibration data:**
- HQQ only uses the weight matrix itself
- No calibration data or forward passes needed
- This makes it extremely fast (seconds to quantize a 7B model)
- Trade-off: slightly worse quality than calibration-based methods

### HQQ Performance

**Llama-2-7B:**
```
HQQ-4bit (g128): PPL 5.32 (+0.20)
vs GPTQ-4bit:    PPL 5.25 (+0.13)
vs AWQ-4bit:     PPL 5.28 (+0.16)
```

**Advantages:**
- No calibration data needed
- Extremely fast quantization (30 seconds per billion parameters)
- Simple implementation
- Works well for most models

**Disadvantages:**
- Slightly lower quality than GPTQ/AWQ
- No activation awareness (cannot handle activation outliers)
- Limited to weight-only quantization

---

## Comparison of Quantization Methods

### Accuracy Comparison (Wikitext2 Perplexity, Llama-2-7B)

| Method | 2-bit | 3-bit | 4-bit | 8-bit |
|--------|-------|-------|-------|-------|
| FP16 baseline | — | — | — | 5.12 |
| GPTQ (g128) | 8.52 | 5.80 | 5.25 | 5.14 |
| AWQ (g128) | — | 5.85 | 5.28 | 5.14 |
| GGUF Q4_K_M | — | — | 5.32 | — |
| GGUF Q8_K | — | — | — | 5.16 |
| bitsandbytes NF4 | — | — | 5.40 | 5.20 |
| bitsandbytes FP4 | — | — | 5.55 | — |
| AQLM (2×256) | 5.70 | 5.30 | 5.20 | — |
| QuIP# | 5.80 | 5.35 | 5.22 | — |
| SpQR (1% sparse) | — | 5.28 | 5.14 | — |
| HQQ (g128) | 8.80 | 5.95 | 5.32 | 5.15 |

### Speed Benchmarks (Tokens/s, Llama-2-7B)

| Method | GPU (A100) | GPU (RTX 4090) | CPU (Apple M2) | CPU (AMD 7950X) |
|--------|-----------|---------------|---------------|----------------|
| FP16 | 85 t/s | 65 t/s | — | — |
| GPTQ-4bit (ExLlama) | 210 t/s | 160 t/s | — | — |
| AWQ-4bit | 200 t/s | 150 t/s | — | — |
| GGUF Q4_K_M | — | — | 35 t/s | 25 t/s |
| GGUF Q5_K_M | — | — | 30 t/s | 22 t/s |
| GGUF Q8_K | — | — | 20 t/s | 15 t/s |
| bitsandbytes-4bit | 70 t/s | 55 t/s | — | — |
| bitsandbytes-8bit | 80 t/s | 60 t/s | — | — |

### Memory Savings Comparison

| Method | 7B Model | 13B Model | 70B Model |
|--------|---------|----------|----------|
| FP32 | 28 GB | 52 GB | 280 GB |
| FP16/BF16 | 14 GB | 26 GB | 140 GB |
| 8-bit | 7 GB | 13 GB | 70 GB |
| GPTQ-4bit | 3.5 GB | 6.5 GB | 35 GB |
| AWQ-4bit | 3.5 GB | 6.5 GB | 35 GB |
| GGUF Q4_K_M | 4.3 GB | 8.5 GB | 42 GB |
| GGUF Q8_K | 7.8 GB | 15 GB | 78 GB |
| bitsandbytes-4bit | 4.0 GB | 7.5 GB | 38 GB |
| AQLM-2bit | 2.2 GB | 4.0 GB | 22 GB |

### GPU Compatibility

| Method | NVIDIA (CUDA) | AMD (ROCm) | Apple MPS | Intel (x86 CPU) | ARM CPU |
|--------|-------------|-----------|-----------|----------------|---------|
| GPTQ (AutoGPTQ) | ✅ Excellent | ⚠️ Partial | ❌ | ⚠️ Slow | ❌ |
| AWQ | ✅ Excellent | ⚠️ Partial | ❌ | ⚠️ Slow | ❌ |
| GGUF (llama.cpp) | ✅ Good | ✅ Good | ✅ Good | ✅ Excellent | ✅ Good |
| bitsandbytes | ✅ Excellent | ⚠️ Partial | ❌ | ❌ | ❌ |
| AQLM | ✅ Good | ⚠️ Partial | ❌ | ❌ | ❌ |
| QuIP# | ✅ Good | ⚠️ Partial | ❌ | ❌ | ❌ |
| HQQ | ✅ Good | ✅ Good | ⚠️ Partial | ⚠️ Partial | ❌ |

### CPU Compatibility

| Method | x86 (AVX2) | x86 (AVX512) | ARM NEON | Apple Silicon |
|--------|-----------|-------------|---------|---------------|
| GGUF Q4_K_M | ✅ Excellent | ✅ Excellent | ✅ Good | ✅ Excellent |
| GGUF Q5_K_M | ✅ Good | ✅ Good | ✅ Good | ✅ Excellent |
| GGUF Q8_K | ✅ Excellent | ✅ Excellent | ✅ Good | ✅ Good |
| GPTQ (CPU) | ⚠️ Partial | ⚠️ Partial | ❌ | ❌ |
| AWQ (CPU) | ❌ | ❌ | ❌ | ❌ |

---

## Accuracy vs Speed Tradeoffs

### Perplexity Degradation

Perplexity (PPL) is the most common metric for quantization quality. Lower is better.

**Expected PPL increase at different bit-widths:**

| Bit-width | Expected PPL increase | Quality Assessment |
|-----------|---------------------|-------------------|
| 8-bit | +0.01 to +0.05 | Lossless (imperceptible) |
| 6-bit | +0.02 to +0.10 | Lossless |
| 5-bit | +0.05 to +0.20 | Nearly lossless |
| 4-bit (good method) | +0.10 to +0.40 | Very good |
| 4-bit (basic method) | +0.30 to +0.80 | Good |
| 3-bit (good method) | +0.40 to +1.00 | Moderate |
| 3-bit (basic method) | +1.00 to +3.00 | Noticeable degradation |
| 2-bit (AQLM/QuIP#) | +0.50 to +1.50 | Moderate degradation |
| 2-bit (basic) | +3.00 to +8.00 | Severe degradation |

**Task-specific accuracy loss:**
- **MMLU:** Typically drops 0.1-1% at 4-bit, 1-5% at 3-bit
- **HumanEval (coding):** More sensitive — 1-3% drop at 4-bit, 3-8% at 3-bit
- **GSM-8K (math):** Most sensitive — 2-5% drop even at 4-bit
- **Reasoning tasks (ARC, HellaSwag):** Less sensitive — <1% drop at 4-bit

### Speed Benchmarks

**GPU speedup factors (relative to FP16):**

| Operation | FP16 | INT8 | INT4 |
|-----------|------|------|------|
| GEMM (compute-bound) | 1.0× | 1.5-2.0× | 2.0-3.0× |
| GEMV (memory-bound) | 1.0× | 1.2-1.5× | 1.5-2.0× |
| Batch=1 (token gen) | 1.0× | 1.1-1.3× | 1.3-1.8× |
| High batch (prefill) | 1.0× | 1.8-2.5× | 2.5-4.0× |

**CPU speedup (relative to FP32):**

| Operation | Q8_K | Q4_K_M | Q2_K |
|-----------|------|--------|------|
| Token generation | 2-3× | 3-5× | 4-6× |
| Prompt processing | 1.5-2× | 2-3× | 2.5-4× |

### Throughput Estimation

**Formula for estimating throughput:**
```
Total memory = model_size + KV_cache + activations
Tokens/s = memory_bandwidth / (bytes_per_token)

Where:
- bytes_per_token ≈ 2 × param_count × bytes_per_param / context_length (approximate)
- For LLM inference: typically bandwidth-bound
```

**Practical throughput (Llama-2-7B, single GPU):**

| Hardware | FP16 | GPTQ-4bit | GGUF Q4_K_M |
|----------|------|-----------|-------------|
| RTX 4090 (24GB, 1008 GB/s) | 65 t/s | 160 t/s | — |
| RTX 3090 (24GB, 936 GB/s) | 50 t/s | 130 t/s | — |
| RTX 4080 (16GB, 716 GB/s) | 40 t/s | 100 t/s | — |
| RTX 3060 (12GB, 360 GB/s) | 20 t/s | 55 t/s | — |
| Apple M2 Max (64GB) | — | — | 35 t/s |
| Apple M1 Pro (16GB) | — | — | 15 t/s |
| AMD 7950X (128GB DDR5) | — | — | 25 t/s |
| Intel i9-13900K (64GB) | — | — | 20 t/s |

---

## Calibration Datasets

### How Calibration Datasets Work

Calibration datasets are used by methods like GPTQ, AWQ, and AQLM to:
1. Compute activation statistics (mean, variance, percentiles)
2. Compute Hessian information (GPTQ)
3. Determine optimal quantization parameters

**Process:**
```
1. Select 64-1024 text samples (typically 128-256)
2. Tokenize each sample (max length 2048 or 4096)
3. For each layer:
   a. Run calibration data through the model up to this layer
   b. Collect input activations
   c. Compute statistics needed by the quantization method
4. Apply quantization using these statistics
```

### How to Choose a Calibration Dataset

**Key considerations:**

1. **Domain matching:**
   - The calibration set should be representative of the model's expected use cases
   - For general-purpose models: Wikipedia + code + books (general-purpose text)
   - For code models: code snippets (GitHub, Stack Overflow)
   - For multilingual: mix of languages matching deployment needs

2. **Diversity:**
   - Include different domains, styles, and formats
   - This ensures robust quantization across all inputs
   - A homogeneous calibration set can cause poor performance on out-of-distribution inputs

3. **Quality:**
   - Use clean, well-formed text
   - Avoid noisy or anomalous inputs (they distort statistics)
   - Deduplicate to prevent over-representation of specific patterns

### Calibration Dataset Size

**Effect of dataset size on PPL (GPTQ-4bit, Llama-2-7B):**

| Samples | Wikitext2 PPL | Notes |
|---------|--------------|-------|
| 0 (no calib) | 5.45 | Poor — no activation awareness |
| 16 | 5.30 | Reasonable |
| 64 | 5.26 | Good |
| 128 | 5.25 | Standard choice |
| 256 | 5.24 | Diminishing returns |
| 1024 | 5.23 | Minimal additional benefit |

**Recommended minimum sizes:**
- GPTQ: 128 samples
- AWQ: 128 samples
- AQLM: 256 samples
- QuIP#: 256 samples
- GGUF (imatrix): 512 samples
- SpQR: 512 samples

### Common Calibration Datasets

**Open-source options:**

| Dataset | Size | Domain | Language | Used By |
|---------|------|--------|----------|---------|
| WikiText-2 | ~2MB | Wikipedia | English | GPTQ, AWQ, etc. |
| C4 (subset) | 100MB+ | Web text | English | GPTQ, SpQR |
| Pile (subset) | 100MB+ | Diverse | English | General |
| Alpaca | 50K examples | Instructions | English | AWQ |
| RedPajama | Variable | Web text | Multilingual | General |
| Custom domain | User-defined | Target domain | User | Production |

**Standard practice:**
- For most quantization work: 128 random samples from C4 validation set
- For production: domain-specific calibration set matching deployment use
- For multilingual: balanced samples across target languages

### Distribution Matching

The calibration dataset should **match the distribution** the model will see in deployment.

**Why distribution matching matters:**
- Quantization parameters are optimized for the calibration distribution
- If deployment data has different statistics, quantization quality degrades
- The model's activations determine which weights are "important"
- Mismatch → wrong weights protected → accuracy loss

**Example of mismatch impact:**
```
Calibration: Wikipedia articles (formal prose)
Deployment: Medical dialogue (conversational, technical terms)

Result: 
- Medical terms have unusual tokenization
- Activation statistics differ
- Quantization performs worse than expected
```

**Mitigation:**
- Use deployment-like data for calibration
- If deployment use case varies, use a diverse calibration set
- Consider multi-dataset calibration (mix of domains)

---

## Quantization for Inference vs Training

### Mixed Precision Training

Modern LLM training uses **mixed precision** — high precision for critical values, lower precision for others.

**Standard mixed precision setup:**

| Component | Precision | Notes |
|-----------|-----------|-------|
| Master weights | FP32 | Accumulated gradient updates |
| Forward pass | BF16/FP16 | Matrix multiplications |
| Backward pass | BF16/FP16 | Gradient computation |
| Gradients | BF16/FP16 | After gradient scaling |
| Optimizer states (Adam) | FP32 | Momentum and variance |
| Loss scaling | FP32 | Prevents gradient underflow |

**FP16/BF16 master weights:**
- Master weights stored in FP32 to preserve gradient update precision
- During forward pass: master_weights → cast to BF16 → compute
- After backward: gradients accumulated in FP32 → update master weights
- This prevents precision loss from repeated quantization

**FP8 Training (H100/H200):**
```
Forward: FP8 matrix multiply (E4M3 format)
Backward: FP8 matrix multiply for activations (E4M3), 
          BF16/FP8 for gradients (E5M2)
Master weights: BF16/FP32
```

FP8 training can achieve 2× throughput vs BF16 with similar quality.

### Gradient Checkpointing

**Problem:** During training, all intermediate activations must be stored for the backward pass. This uses large amounts of memory.

**Gradient checkpointing (activation checkpointing):**
- During forward pass: only store a subset of activations (checkpoints)
- During backward pass: recompute discarded activations from the nearest checkpoint
- Memory savings: up to 70% (fewer stored activations)
- Compute overhead: ~15-30% (recomputation cost)

**Checkpointing granularity:**
- Every N layers (e.g., checkpoint every 4 layers)
- Typically 0.5-1 checkpoint per transformer block
- More checkpoints = less recomputation but more memory

### Optimizer State Quantization

The Adam optimizer maintains two states per parameter: momentum (m) and variance (v). These double the memory requirement.

**8-bit Adam (bitsandbytes):**
- Quantize momentum and variance to 8-bit
- Dynamic quantization (per-tensor)
- Memory: 2 bytes per parameter (m,v) instead of 8 bytes (FP32)
- Saves 6 GB for a 7B model

**Optimizer state quantization algorithm:**
```
For each tensor:
  1. Compute min/max of optimizer state
  2. Quantize to 8-bit: q = round((state - min) / (max - min) × 255)
  3. Store q (1 byte) + min/max (2 floats = 8 bytes total per tensor)

During optimization step:
  1. Dequantize: state = q × (max - min) / 255 + min
  2. Apply optimizer update
  3. Re-quantize updated state
```

**Effectiveness:**
- Negligible impact on training quality
- ~50% reduction in optimizer memory
- Common in QLoRA and full-parameter fine-tuning with limited VRAM

---

## Practical Guide

### Which Quantization for Which Use Case

**Consumer GPU deployment (4GB-24GB VRAM):**

| Scenario | Recommended Method | Reason |
|----------|-------------------|--------|
| 7B model on 8GB GPU | GPTQ-4bit or AWQ-4bit | Best quality, fast inference |
| 13B model on 8-12GB GPU | GPTQ-4bit or AWQ-4bit | Fits in VRAM, good speed |
| 30B model on 16-24GB GPU | GPTQ-4bit or AWQ-4bit | Largest model on consumer GPU |
| 70B model on 24GB GPU | GGUF Q3_K_M or Q4_K_M | Must use CPU offloading or smaller quants |
| Mac (Apple Silicon) | GGUF Q4_K_M or Q5_K_M | Only GGUF works well on Metal |
| CPU-only inference | GGUF Q4_K_M or Q5_K_M | Only GGUF supports CPU well |

**Production API deployment:**

| Scenario | Recommended Method | Reason |
|----------|-------------------|--------|
| High-throughput API | FP16/BF16 or 8-bit | Quality priority, multiple GPUs |
| Cost-optimized API | GPTQ-4bit or AWQ-4bit | 2× throughput, minimal quality loss |
| Batch processing | FP16 or INT8 | Higher batch sizes benefit from full precision |
| Real-time (<100ms) | GPTQ-4bit (ExLlama) | Fastest single-batch inference |

**Edge/On-device deployment:**

| Scenario | Recommended Method | Reason |
|----------|-------------------|--------|
| Phone (iOS/Android) | GGUF Q4_K_M | Universal, optimized for ARM |
| Laptop (no GPU) | GGUF Q4_K_M or Q5_K_M | Best CPU performance |
| Embedded Linux | GGUF Q3_K_M | Limited RAM, need compression |
| Web (WASM) | GGUF Q4_K_M | Only format supported by WebLLM |

**Fine-tuning:**

| Scenario | Recommended Method | Reason |
|----------|-------------------|--------|
| QLoRA fine-tuning | bitsandbytes NF4 | Full training support, good quality |
| Full fine-tuning (limited VRAM) | FP8 (H100) or BF16 | Standard training precision |
| Parameter-efficient FT | bitsandbytes NF4 + LoRA | Most memory-efficient training |

### GPU Memory Calculator

**Estimating memory for inference:**

```
Model memory = param_count × bytes_per_param × 1.05 (5% overhead)

Examples:
  7B FP16:   7B × 2 = 14 GB
  7B INT4:   7B × 0.5 = 3.5 GB
  13B FP16:  13B × 2 = 26 GB
  13B INT4:  13B × 0.5 = 6.5 GB
  70B FP16:  70B × 2 = 140 GB
  70B INT4:  70B × 0.5 = 35 GB
  70B Q4_K_M: ~42 GB (GGUF has overhead from k-quant metadata)
```

**Plus KV cache memory:**
```
KV_cache = 2 × num_layers × num_heads × head_dim × seq_len × bytes_per_value

Examples (batch=1, 4096 context, FP16):
  Llama-2-7B (32 layers, 32 heads, 128 dim): 2 × 32 × 32 × 128 × 4096 × 2 = 2.1 GB
  Llama-2-13B (40 layers, 40 heads, 128 dim): 2 × 40 × 40 × 128 × 4096 × 2 = 3.3 GB
  Llama-2-70B (80 layers, 64 heads, 128 dim): 2 × 80 × 64 × 128 × 4096 × 2 = 10.5 GB

With GQA (8 KV heads):
  Llama-3-70B: 2 × 80 × 8 × 128 × 4096 × 2 = 1.3 GB
```

**Plus activations (during prefill):**
```
Activations = batch_size × seq_len × hidden_dim × num_layers × bytes_per_value

Llama-2-7B (batch=1, seq=2048): 1 × 2048 × 4096 × 32 × 2 = 0.5 GB
Llama-2-70B (batch=1, seq=4096): 1 × 4096 × 8192 × 80 × 2 = 5.2 GB
```

**Total memory (inference):**
```
Total = model_weights + KV_cache + activations + overhead

Example: Llama-2-7B, GPTQ-4bit, 4096 context
  Model: 3.5 GB
  KV cache: 2.1 GB
  Activations: 0.5 GB
  Overhead: 0.5 GB
  Total: ~6.6 GB (fits in 8GB GPU)
```

### Throughput Estimation

**Formula:**
```
Throughput (tokens/s) ≈ memory_bandwidth / (bytes_per_token_generated)

For batch=1 (autoregressive generation):
  bytes_per_token ≈ model_weights_bytes + KV_cache_bytes_per_token

For batch=B (prefill/batch inference):
  bytes_per_token ≈ model_weights_bytes/B + KV_cache_bytes_per_token + activation_memory/B
```

**Practical estimation:**

| Model | Quant | GPU | Peak Memory | Estimated t/s |
|-------|-------|-----|------------|---------------|
| 7B | FP16 | RTX 4090 (24GB) | 16 GB | 65 t/s |
| 7B | GPTQ-4bit | RTX 4090 | 6 GB | 160 t/s |
| 13B | FP16 | RTX 4090 | 28 GB | ❌ Doesn't fit |
| 13B | GPTQ-4bit | RTX 4090 | ~10 GB | 80 t/s |
| 70B | GGUF Q4_K_M | 2× RTX 4090 | ~50 GB | 15 t/s |
| 70B | GPTQ-4bit | A100-80GB | 40 GB | 50 t/s |

### Step-by-Step Guide

**Quantizing with AutoGPTQ:**
```python
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer
import datasets

model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoGPTQForCausalLM.from_pretrained(model_id)

# Prepare calibration data
dataset = datasets.load_dataset("c4", "en", split="train", streaming=True)
calibration_samples = []
for i, sample in enumerate(dataset):
    if i >= 128:
        break
    calibration_samples.append(sample["text"])

# Quantize
model.quantize(
    calibration_samples,
    bits=4,
    group_size=128,
    damp_percent=0.01,
    desc_act=True,
    static_groups=False,
    sym=True,
)

# Save quantized model
model.save_quantized("llama-2-7b-gptq-4bit")
tokenizer.save_pretrained("llama-2-7b-gptq-4bit")
```

**Quantizing with AWQ:**
```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_id = "meta-llama/Llama-2-7b-hf"
quant_path = "llama-2-7b-awq-4bit"

model = AutoAWQForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Quantize
model.quantize(
    tokenizer,
    quant_config={
        "bits": 4,
        "group_size": 128,
        "zero_point": True,
        "version": "GEMM",
    },
    calib_data="wikitext",
    split="train",
    max_calib_samples=128,
)

# Save
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

**Quantizing with llama.cpp:**
```bash
# Convert HuggingFace model to GGUF FP16
python convert.py ./llama-2-7b --outfile llama-2-7b-fp16.gguf

# Quantize to Q4_K_M
./quantize llama-2-7b-fp16.gguf llama-2-7b-q4_k_m.gguf Q4_K_M

# For better quality, generate importance matrix first
./imatrix -m llama-2-7b-fp16.gguf -f calibration.txt -o llama-2-7b.imatrix
./quantize --imatrix llama-2-7b.imatrix llama-2-7b-fp16.gguf llama-2-7b-q4_k_m.gguf Q4_K_M
```

**Quantizing with bitsandbytes (inference):**
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    device_map="auto",
)
```

---

## Advanced Topics

### Quantization-Aware Training (QAT)

QAT simulates quantization effects during training, allowing the model to adapt to reduced precision.

**Process:**
1. Start with a pre-trained FP16 model
2. Insert "fake quantization" nodes (quantize → dequantize) before/after operations
3. Continue training with standard forward/backward pass
4. The Straight-Through Estimator (STE) passes gradients through fake quantization
5. After training: replace fake quantization with actual quantization

**Fake quantization:**
```python
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, bits):
        # Quantize and dequantize (simulate quantization)
        q = torch.round(x / scale + zero_point)
        q = torch.clamp(q, 0, 2**bits - 1)
        x_q = (q - zero_point) * scale
        return x_q
    
    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradients through unchanged
        return grad_output, None, None, None
```

**Benefits of QAT:**
- Can recover 50-80% of accuracy loss from quantization
- Essential for very low bit-widths (2-bit, binary)
- Reduces outlier issues (model learns to avoid them)

**Costs:**
- Requires training infrastructure (GPUs, time)
- Need representative training data
- May cause overfitting to quantization grid

### KV Cache Quantization

The KV cache becomes the dominant memory cost for long contexts. Quantizing it can dramatically reduce memory.

**Current approaches:**

1. **KIVI:** 
   - Non-uniform quantization for key cache
   - Per-channel scales for value cache
   - 2-bit KV cache with ~0.1 PPL degradation
   - 4× memory reduction for long context

2. **KVQuant:**
   - Per-channel + per-token quantization
   - Outlier-aware handling
   - 2-4 bit KV cache with minimal accuracy loss
   
3. **WKVQuant:**
   - Multi-level quantization (different precisions for different KV layers)
   - Earlier layers: lower precision, later layers: higher precision
   - Optimizes memory-quality tradeoff

**Memory savings:**
```
Original FP16 KV cache (128K context, 70B model): ~150 GB
KVQuant 4-bit: ~38 GB (4× savings)
KVQuant 2-bit: ~19 GB (8× savings)
```

### Activation Quantization

While weight quantization is more common, activation quantization can further reduce memory and accelerate compute.

**Challenges:**
- Activations vary per input (dynamic range)
- Outliers cause more problems in activations than weights
- LLM activations can have values from 0.001 to 1000+

**SmoothQuant approach:**
- Transfer quantization difficulty from activations to weights
- Multiply activations by smoothing factor s: X' = X · s
- Divide weights by s: W' = W / s
- The product remains unchanged: Y = X · W = X' · W'
- Choose s so that activation quantization becomes easier
- Typically: 0.5-0.9 smoothing for layer input activations

**FP8 activation quantization:**
- H100 native support for FP8
- E4M3 for forward pass, E5M2 for gradients
- ~2× throughput improvement
- Minimal accuracy loss when combined with proper scaling

### Model-Specific Quantization Considerations

**Different architectures handle quantization differently:**

1. **MoE models:**
   - Routers are sensitive to quantization (routing decisions change)
   - Experts can be quantized more aggressively (their outputs are combined)
   - Need calibration data that tests routing
   - QwenMoE, Mixtral, DeepSeek-V3 all work well at 4-bit

2. **Attention mechanisms (GQA, MQA, MLA):**
   - KV head projection weights more sensitive (affects many heads)
   - GQA: keep KV projections in higher precision
   - MLA (DeepSeek): latent projections are critical
   - Recommend 4-bit minimum for attention projections

3. **Embedding layers:**
   - Large vocabulary embeddings → significant memory at FP16
   - Can be quantized to 4-bit with minimal quality loss
   - But: embedding lookup is memory-bound, not compute-bound
   - Quantization may not speed up embedding lookup

4. **Norm layers:**
   - RMSNorm / LayerNorm: very few parameters, very important
   - Best kept in FP16/BF16 (negligible memory impact)
   - Quantizing norms causes disproportionate accuracy loss

---

*This document provides comprehensive coverage of LLM quantization — from precision formats and theory through practical deployment. Understanding quantization is essential for deploying LLMs efficiently, reducing costs, and enabling on-device AI applications.*

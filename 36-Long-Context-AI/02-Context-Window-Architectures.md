# Context Window Architectures: How Long-Context Models Actually Work

> **Deep technical dive into the architectures, algorithms, and engineering that make million-token context windows practical.** This document covers the evolution from O(n²) attention to the sparse, distributed, and hardware-aware techniques that power today's longest-context models.

---

## Table of Contents

1. [The Fundamental Problem: Quadratic Attention](#1-the-fundamental-problem-quadratic-attention)
2. [FlashAttention: IO-Aware Exact Attention](#2-flashattention-io-aware-exact-attention)
3. [Sparse Attention Patterns](#3-sparse-attention-patterns)
4. [Ring Attention: Distributed Long-Context](#4-ring-attention-distributed-long-context)
5. [MiniMax Sparse Attention (MSA)](#5-minimax-sparse-attention-msa)
6. [KV Cache Management at Scale](#6-kv-cache-management-at-scale)
7. [Multi-Query and Grouped-Query Attention](#7-multi-query-and-grouped-query-attention)
8. [Sliding Window and Hierarchical Attention](#8-sliding-window-and-hierarchical-attention)
9. [Linear Attention and State-Space Models](#9-linear-attention-and-state-space-models)
10. [Hardware Optimizations](#10-hardware-optimizations)
11. [The Training Pipeline for Long Context](#11-the-training-pipeline-for-long-context)
12. [Evaluation Methodologies](#12-evaluation-methodologies)
13. [Code Examples](#13-code-examples)

---

## 1. The Fundamental Problem: Quadratic Attention

### Self-Attention Recap

The core operation in Transformers is scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- Q (Query): [batch, seq_len, d_model]
- K (Key): [batch, seq_len, d_model]
- V (Value): [batch, seq_len, d_model]
- d_model: model dimension (e.g., 4096)

### The Quadratic Bottleneck

The computation of `QK^T` creates a [seq_len, seq_len] matrix:

| Sequence Length | QK^T Matrix Size | Memory (FP16) | FLOPs (approx) |
|----------------|-------------------|---------------|-----------------|
| 4,096 | 16M elements | 32 MB | 134 GFLOPs |
| 32,768 | 1.07B elements | 2 GB | 8.6 TFLOPs |
| 128,000 | 16.4B elements | 32 GB | 131 TFLOPs |
| 1,000,000 | 1T elements | 2 TB | 8 PFLOPs |

At 1M tokens, standard attention requires **2 TB of memory** just for the attention matrix — impossible on any single device.

### Why This Matters

This isn't just a compute problem — it's a memory bandwidth problem. Even if you could compute the attention matrix, moving 2 TB between GPU memory and compute units would take minutes, destroying inference speed.

---

## 2. FlashAttention: IO-Aware Exact Attention

### Core Insight

FlashAttention (Dao et al., 2022-2023) recognizes that attention is **memory-bandwidth bound**, not compute-bound. The key innovation is restructuring the computation to minimize HBM (High Bandwidth Memory) reads/writes.

### How FlashAttention Works

**Traditional Attention**:
1. Compute QK^T → store full [n, n] matrix in HBM
2. Apply softmax → read/write full matrix
3. Multiply by V → read/write full matrix
4. Total HBM accesses: O(n²)

**FlashAttention**:
1. Tile the Q, K, V matrices into blocks that fit in SRAM (on-chip memory)
2. For each block of Q, iterate through all blocks of K, V
3. Accumulate results in SRAM, never materializing the full matrix
4. Total HBM accesses: O(n² × d / M) where M is SRAM size (much smaller constant)

### Pseudocode

```python
def flash_attention(Q, K, V, block_size=256):
    """Simplified FlashAttention algorithm."""
    n, d = Q.shape
    O = torch.zeros_like(Q)          # Output
    L = torch.zeros(n)                # Log-sum-exp for softmax
    M = torch.full((n,), -float('inf'))  # Running max
    
    # Iterate over K, V blocks
    for j in range(0, n, block_size):
        K_block = K[j:j+block_size]   # Load to SRAM
        V_block = V[j:j+block_size]   # Load to SRAM
        
        for i in range(0, n, block_size):
            Q_block = Q[i:i+block_size]  # Load to SRAM
            
            # Compute attention scores for this block
            S_block = Q_block @ K_block.T / sqrt(d)
            
            # Online softmax (stable)
            M_new = torch.maximum(M[i:i+block_size], S_block.max(dim=-1).values)
            
            # Exponential and normalization
            P_block = torch.exp(S_block - M_new.unsqueeze(-1))
            
            # Update running statistics
            L_new = torch.exp(M[i:i+block_size] - M_new) * L[i:i+block_size] + P_block.sum(-1)
            
            # Update output
            O[i:i+block_size] = (
                torch.exp(M[i:i+block_size] - M_new).unsqueeze(-1) * O[i:i+block_size] 
                + P_block @ V_block
            )
            
            L[i:i+block_size] = L_new
            M[i:i+block_size] = M_new
    
    return O / L.unsqueeze(-1)
```

### FlashAttention-2 Improvements

- Better work partitioning across GPU threads
- Reduced non-matmul FLOPs
- ~2x speedup over FlashAttention-1
- Memory usage: O(n) instead of O(n²)

### FlashAttention-3 (2024)

- **Asynchronous softmax**: Overlaps softmax computation with matrix multiply
- **Warp-specialization**: Different GPU warps handle different operations
- **FP8 support**: 1.5-2x speedup on H100 GPUs
- **Hardware-level optimizations**: Exploits H100's Tensor Memory Accelerator

### Impact on Long Context

| Version | Max Practical Context (A100) | Max Practical Context (H100) |
|---------|------------------------------|------------------------------|
| FlashAttention-1 | 128K tokens | 256K tokens |
| FlashAttention-2 | 256K tokens | 512K tokens |
| FlashAttention-3 | 512K tokens | 1M+ tokens |

---

## 3. Sparse Attention Patterns

### The Core Idea

Instead of every token attending to every other token, **sparsify** the attention pattern so each token attends to only a subset.

### Pattern Types

#### Local (Sliding Window) Attention
Each token attends only to nearby tokens within a window of size W:

```
Token positions:  0  1  2  3  4  5  6  7  8  9  10
Window (W=3):    [0  1  2] [1  2  3] [2  3  4] ...
```

**Advantages**: O(n×W) complexity, excellent for local patterns
**Disadvantages**: Cannot attend to distant tokens without multiple layers

#### Global Attention
A small set of "global" tokens attend to everything:

```
Global tokens: 0, 100, 200, 300, ...
Every other token attends to global tokens
Global tokens attend to everything
```

**Advantages**: Provides global information flow
**Disadvantages**: Global tokens become bottlenecks

#### Dilated Attention
Like local attention but with gaps:

```
Token:    0  1  2  3  4  5  6  7  8  9  10
Dilated:  X     X     X     X     X     X  (dilation=2)
```

**Advantages**: Larger effective receptive field with fewer connections
**Disadvantages**: May miss important nearby tokens

#### Random Attention
Each token randomly attends to a fixed number of distant tokens:

```
Token 5 attends to: [3, 7, 42, 128, 512, ...]  (random positions)
```

**Advantages**: Provides long-range connections with O(n) complexity
**Disadvantages**: Non-deterministic, may miss important connections

### Combined Patterns

Most production models use **hybrid patterns**:

```
Layer 0-15:  Local attention (window=8K)
Layer 16-31: Local + global (8K local, 256 global tokens)
Layer 32-47: Local + dilated + global
Layer 48-63: Full attention (for short sequences only)
```

### BigBird (Google, 2020)

Combines three patterns:
1. **Local window** (W=256)
2. **Global tokens** (G=64)
3. **Random connections** (R=64)

Total attention per token: W + G + R = 384 (instead of full n)

**Theoretical guarantee**: BigBird with these patterns is a universal approximator — it can represent any function that full attention can.

### Longformer (Allen AI, 2020)

Similar hybrid approach:
- Local sliding window attention
- Dilated sliding window for larger receptive field
- Task-specific global attention on [CLS] and question tokens

---

## 4. Ring Attention: Distributed Long-Context

### The Problem

Even with FlashAttention, a 1M-token sequence requires ~16 GB of KV cache — more than most GPUs have. Ring Attention distributes the context across multiple GPUs.

### How Ring Attention Works

**Setup**: N GPUs, each holding 1/N of the sequence

**Algorithm**:
1. Each GPU holds a block of Q, K, V
2. GPU 0 starts with its K, V block and computes attention for its Q block
3. K, V blocks are passed to the next GPU (ring topology)
4. Each GPU accumulates partial attention results
5. After N-1 steps, every GPU has attended to every K, V block
6. Final normalization combines results

**Complexity**: 
- Communication: O(n × d × N) total, O(n × d) per GPU
- Computation: O(n² / N) per GPU (perfectly parallel)
- Memory: O(n² / N²) per GPU (quadratic reduction!)

### Multi-Head Ring Attention

With M attention heads and N GPUs:
- Each GPU holds M/N heads
- K, V blocks include all heads
- Ring communication passes K, V for all heads

### Example: 1M Tokens Across 8 GPUs

```
Per GPU:
- Sequence length: 128K tokens (1M / 8)
- Q block: 128K × d_model
- K, V blocks: 128K × d_model
- Attention matrix: 128K × 128K = 16 GB per step
- 8 ring steps → 8 × 16 GB = 128 GB total computation
- But each step only needs 16 GB in memory!

Total: 1M tokens on 8 × 80GB A100s = feasible
```

### Challenges

1. **Load balancing**: Uneven sequence lengths cause stragglers
2. **Communication overhead**: Inter-GPU bandwidth becomes bottleneck
3. **Fault tolerance**: If one GPU fails, the ring breaks
4. **Heterogeneous hardware**: Different GPU types complicate scheduling

---

## 5. MiniMax Sparse Attention (MSA)

### The Breakthrough (June 2026)

MiniMax M3's MSA architecture achieves the most significant efficiency gain in long-context processing to date:

- **1/20th the per-token compute** compared to dense attention
- **9x faster prefilling** for 1M tokens
- **15x faster decoding** for 1M tokens

### How MSA Works

MSA is a learned sparse attention pattern that adaptively selects which tokens to attend to:

#### Step 1: Attention Budget Allocation
Instead of allocating equal compute to every token pair, MSA learns a "budget" for each token:

```python
# Pseudocode for MSA budget allocation
class MSABudget(nn.Module):
    def __init__(self, d_model, max_tokens):
        super().__init__()
        self.budget_predictor = nn.Linear(d_model, max_tokens)
    
    def forward(self, query_states):
        # Predict how many tokens each query should attend to
        budgets = torch.sigmoid(self.budget_predictor(query_states))
        # budgets: [batch, seq_len, 1] — values between 0 and 1
        return budgets
```

#### Step 2: Top-K Token Selection
For each query token, select the K most important key tokens:

```python
def msa_topk_selection(Q, K, budgets, k_per_token=64):
    """Select top-k tokens for each query."""
    # Compute full attention scores (in flash, this is tiled)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_model)
    
    # Determine per-token k based on budget
    k_values = (budgets * k_per_token).long().clamp(min=1)
    
    # For each query, select top-k keys
    topk_scores, topk_indices = torch.topk(scores, k_per_token, dim=-1)
    
    # Mask out tokens beyond each query's budget
    mask = torch.arange(k_per_token).unsqueeze(0) < k_values
    topk_scores = topk_scores.masked_fill(~mask, -float('inf'))
    
    return topk_scores, topk_indices
```

#### Step 3: Sparse Attention Computation
Compute attention only on selected tokens:

```python
def msa_attention(Q, K, V, topk_indices, topk_scores):
    """Compute sparse attention using selected tokens."""
    # Gather selected K, V
    K_selected = torch.gather(K, 2, topk_indices)
    V_selected = torch.gather(V, 2, topk_indices)
    
    # Compute attention on sparse selection
    attn_weights = torch.softmax(topk_scores, dim=-1)
    output = torch.matmul(attn_weights, V_selected)
    
    return output
```

### Why MSA Is Different

Previous sparse attention patterns (BigBird, Longformer) used **fixed** patterns. MSA uses **learned, input-dependent** patterns:

| Approach | Pattern Type | Adaptivity | Quality |
|----------|-------------|------------|---------|
| BigBird | Fixed random+local+global | None | Good |
| Longformer | Fixed local+dilated | None | Good |
| MSA (MiniMax) | Learned, input-dependent | Full | Excellent |

### Efficiency Gains

| Sequence Length | Dense Attention FLOPs | MSA FLOPs | Speedup |
|----------------|----------------------|-----------|---------|
| 128K | 131 TFLOPs | 13 TFLOPs | 10x |
| 512K | 2.1 PFLOPs | 105 TFLOPs | 20x |
| 1M | 8.1 PFLOPs | 405 TFLOPs | 20x |
| 5M | 203 PFLOPs | 10 PFLOPs | 20x |

---

## 6. KV Cache Management at Scale

### The KV Cache Problem

During autoregressive generation, the key-value pairs from previous tokens must be stored for future attention computation. This is the **KV cache**:

```
Memory for KV cache:
- Per layer: 2 × seq_len × d_model × num_heads × sizeof(dtype)
- For a 70B model with 80 layers, 64 heads, d=8192:
  - 128K tokens: ~16 GB
  - 1M tokens: ~128 GB
  - 10M tokens: ~1.28 TB
```

### KV Cache Compression Techniques

#### 1. KV Cache Quantization

Reduce precision of cached K, V values:

```python
def quantize_kv_cache(K_cache, V_cache, bits=4):
    """Quantize KV cache to lower precision."""
    # Group-wise quantization
    group_size = 128
    
    K_quantized = []
    V_quantized = []
    
    for k, v in zip(K_cache, V_cache):
        # Quantize K
        k_grouped = k.reshape(-1, group_size)
        k_max = k_grouped.abs().max(dim=-1, keepdim=True).values
        k_scale = k_max / (2 ** (bits - 1) - 1)
        k_int = (k_grouped / k_scale).round().to(torch.int8)
        
        # Quantize V similarly
        v_grouped = v.reshape(-1, group_size)
        v_max = v_grouped.abs().max(dim=-1, keepdim=True).values
        v_scale = v_max / (2 ** (bits - 1) - 1)
        v_int = (v_grouped / v_scale).round().to(torch.int8)
        
        K_quantized.append((k_int, k_scale))
        V_quantized.append((v_int, v_scale))
    
    return K_quantized, V_quantized
```

**Impact**: 4-bit KV cache reduces memory by 4x with <1% quality loss.

#### 2. KV Cache Eviction

For very long sequences, evict less important KV pairs:

```python
def kv_cache_eviction(K_cache, V_cache, importance_scores, keep_ratio=0.5):
    """Evict least important KV pairs."""
    n_tokens = K_cache.shape[1]
    n_keep = int(n_tokens * keep_ratio)
    
    # Select top-k most important
    _, topk_indices = torch.topk(importance_scores, n_keep, dim=-1)
    topk_indices = topk_indices.sort(dim=-1).values
    
    # Gather kept KV pairs
    K_evicted = torch.gather(K_cache, 1, topk_indices.unsqueeze(-1).expand_as(K_cache))
    V_evicted = torch.gather(V_cache, 1, topk_indices.unsqueeze(-1).expand_as(V_cache))
    
    return K_evicted, V_evicted
```

**Importance metrics**:
- Attention weight magnitude
- Position-based (recent = more important)
- Task-specific (entities, keywords = more important)

#### 3. PagedAttention (vLLM)

Dynamic memory allocation for KV cache, similar to OS virtual memory:

- Divide KV cache into fixed-size "pages"
- Allocate pages on demand, not upfront
- Share pages across requests when possible (prefix caching)
- **Impact**: 2-4x better GPU memory utilization

#### 4. KV Cache Offloading

Store less-recent KV pairs on CPU or disk:

```python
class KVCacheOffloader:
    def __init__(self, gpu_memory_gb=80, cpu_offload_threshold=0.8):
        self.gpu_memory_gb = gpu_memory_gb
        self.threshold = cpu_offload_threshold
        self.cpu_cache = {}
    
    def manage_cache(self, K_cache, V_cache, current_usage_gb):
        """Offload old KV pairs to CPU when GPU is near capacity."""
        if current_usage_gb > self.gpu_memory_gb * self.threshold:
            # Calculate how much to offload
            target_usage = self.gpu_memory_gb * 0.7
            bytes_to_offload = (current_usage_gb - target_usage) * 1e9
            
            # Offload oldest tokens (simplest strategy)
            tokens_to_offload = int(bytes_to_offload / (K_cache.element_size() * K_cache.shape[-1] * 2))
            
            # Move to CPU
            self.cpu_cache['K'] = K_cache[:, :tokens_to_offload].cpu()
            self.cpu_cache['V'] = V_cache[:, :tokens_to_offload].cpu()
            self.cpu_cache['offset'] = tokens_to_offload
            
            # Return GPU-resident portion
            return K_cache[:, tokens_to_offload:], V_cache[:, tokens_to_offload:]
        
        return K_cache, V_cache
```

---

## 7. Multi-Query and Grouped-Query Attention

### Multi-Query Attention (MQA)

Share a single key-value head across all query heads:

```python
# Standard Multi-Head Attention
# Q: [batch, seq, num_heads, d_head]
# K: [batch, seq, num_heads, d_head]  ← num_heads KV pairs
# V: [batch, seq, num_heads, d_head]

# Multi-Query Attention
# Q: [batch, seq, num_heads, d_head]
# K: [batch, seq, 1, d_head]          ← 1 KV pair shared
# V: [batch, seq, 1, d_head]          ← 1 KV pair shared
```

**Impact**: KV cache reduced by num_heads factor (e.g., 32x for 32-head model)
**Trade-off**: Slight quality degradation

### Grouped-Query Attention (GQA)

Compromise between MHA and MQA — group query heads, share KV within groups:

```python
# Grouped-Query Attention
num_groups = 8  # Instead of 32 heads or 1 head
# Q: [batch, seq, 32, d_head]     ← 32 query heads
# K: [batch, seq, 8, d_head]      ← 8 KV pairs (4 heads per group)
# V: [batch, seq, 8, d_head]      ← 8 KV pairs
```

**Impact**: KV cache reduced by (num_heads / num_groups) factor
**Quality**: Nearly identical to full MHA

### KV Cache Savings

| Attention Type | KV Cache Size (70B model, 128K) | Quality |
|---------------|--------------------------------|---------|
| MHA (32 heads) | 16 GB | Baseline |
| GQA (8 groups) | 4 GB | ~99% of baseline |
| MQA (1 head) | 0.5 GB | ~97% of baseline |

---

## 8. Sliding Window and Hierarchical Attention

### Sliding Window Attention

Each token attends only to W previous tokens:

```python
def sliding_window_attention(Q, K, V, window_size=4096):
    """Sliding window attention."""
    seq_len = Q.shape[1]
    output = torch.zeros_like(Q)
    
    for i in range(seq_len):
        # Define window
        start = max(0, i - window_size)
        end = i + 1
        
        # Compute attention within window
        q_i = Q[:, i:i+1]  # [batch, 1, d]
        k_window = K[:, start:end]  # [batch, window, d]
        v_window = V[:, start:end]
        
        scores = torch.matmul(q_i, k_window.transpose(-2, -1)) / sqrt(d)
        weights = torch.softmax(scores, dim=-1)
        output[:, i] = torch.matmul(weights, v_window).squeeze(1)
    
    return output
```

### Hierarchical Attention

Process context at multiple resolutions:

```
Level 0 (Fine):   Token-level attention within local blocks
Level 1 (Medium): Block-level attention across nearby blocks
Level 2 (Coarse): Summary-level attention across distant blocks
```

**Example implementation**:
```python
class HierarchicalAttention:
    def __init__(self, block_size=1024, num_levels=3):
        self.block_size = block_size
        self.num_levels = num_levels
    
    def forward(self, x):
        # Level 0: Local attention within blocks
        for block_start in range(0, seq_len, self.block_size):
            block = x[:, block_start:block_start+self.block_size]
            block_out = self.local_attention(block)
            x[:, block_start:block_start+self.block_size] = block_out
        
        # Level 1: Attend across block summaries
        block_summaries = self.pool_blocks(x)
        summary_out = self.attend_across_blocks(block_summaries)
        
        # Level 2: Global attention on compressed representation
        compressed = self.compress(summary_out)
        global_out = self.global_attention(compressed)
        
        return self.upsample_and_combine(x, global_out)
```

---

## 9. Linear Attention and State-Space Models

### Linear Attention

Replace softmax attention with kernel approximations:

```
Standard: softmax(QK^T/√d)V     → O(n²)
Linear:   φ(Q)(φ(K)^T V)        → O(n)
```

Where φ is a kernel feature map.

### RWKV (Receptance Weighted Key Value)

Combines linear attention with recurrence:

```python
def rwkv_forward(r, k, v, w):
    """RWKV forward pass — linear complexity."""
    # r: receptance, k: key, v: value, w: time-decay
    
    output = []
    state = torch.zeros_like(v[0])
    
    for i in range(seq_len):
        # Exponential moving average with decay
        state = state * w + k[i] * v[i]
        
        # Output is receptance weighted by current state
        out = r[i] * state
        output.append(out)
    
    return torch.stack(output)
```

**Complexity**: O(n × d) — truly linear
**Trade-off**: Cannot perform arbitrary retrieval (limited by recurrent state size)

### Mamba (State-Space Model)

A different approach — selective state spaces:

```python
def mamba_selective_scan(x, A, B, C, delta):
    """Selective state-space model scan."""
    # x: input [batch, seq, d_model]
    # A: state matrix [d_model, d_state]
    # B: input matrix [batch, seq, d_state]
    # C: output matrix [batch, seq, d_state]
    # delta: time step [batch, seq, d_model]
    
    batch, seq_len, d_model = x.shape
    d_state = A.shape[1]
    
    # Discretize
    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # [batch, seq, d_model, d_state]
    deltaB_x = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)
    
    # Selective scan (sequential)
    h = torch.zeros(batch, d_model, d_state)
    outputs = []
    
    for i in range(seq_len):
        h = deltaA[:, i] * h + deltaB_x[:, i]
        y = (h * C[:, i].unsqueeze(1)).sum(-1)
        outputs.append(y)
    
    return torch.stack(outputs, dim=1)
```

### Hybrid Architectures

Most production models combine approaches:

```
Model: Hybrid Attention (typical 2026 architecture)
├── Layers 0-15:  Full dense attention (local window)
├── Layers 16-31: Sliding window + global tokens
├── Layers 32-47: MSA (learned sparse)
└── Layers 48-63: Linear attention (for very long contexts)
```

---

## 10. Hardware Optimizations

### GPU Memory Hierarchy

```
┌─────────────────────────────────────────┐
│  HBM (High Bandwidth Memory)            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ GPU 0   │  │ GPU 1   │  │ GPU 2   │ │
│  │ 80 GB   │  │ 80 GB   │  │ 80 GB   │ │
│  │ 3.35TB/s│  │ 3.35TB/s│  │ 3.35TB/s│ │
│  └─────────┘  └─────────┘  └─────────┘ │
├─────────────────────────────────────────┤
│  SRAM (On-chip) — ~20 MB per SM        │
│  ~19 TB/s bandwidth                     │
│  Used by FlashAttention tiling           │
├─────────────────────────────────────────┤
│  NVLink — 900 GB/s inter-GPU           │
│  Used by Ring Attention                  │
└─────────────────────────────────────────┘
```

### NVIDIA H100 Optimizations

- **Tensor Memory Accelerator (TMA)**: Hardware unit for loading data from HBM to SRAM
- **FP8 Tensor Cores**: 2x throughput for matrix operations
- **Transformer Engine**: Automatic mixed precision for attention
- **DPX Instructions**: Dynamic programming instructions for attention patterns

### Custom Silicon for Long Context

| Chip | Key Feature | Context Optimization |
|------|-------------|---------------------|
| NVIDIA Cosmos 3 | Omnimodel support | Hardware-level sparse attention |
| Intel Gaudi 3 | Cost-effective inference | Optimized for long sequences |
| Groq LPU | Deterministic latency | Deterministic memory access |
| Cerebras WSE-3 | On-chip memory | Eliminates HBM bottleneck |
| AWS Trainium2 | Custom attention kernels | Optimized for long-context training |

---

## 11. The Training Pipeline for Long Context

### Phase 1: Short-Context Pre-training

Train on standard sequences (4K-8K tokens) to learn language fundamentals.

### Phase 2: Context Extension

Extend context window through:

1. **Position Interpolation**: Scale position embeddings to longer sequences
2. **NTK-Aware Scaling**: Non-uniform scaling that preserves local attention
3. **YaRN**: Yet another RoPE extension — combines NTK and linear scaling
4. **LongRoPE**: Learned scaling factors per position

```python
def yarn_extend_rope(original_rope, target_length):
    """Extend RoPE to longer sequences using YaRN."""
    # Original RoPE supports up to original_length
    original_length = original_rope.max_length
    
    # Compute scaling factor
    scale = target_length / original_length
    
    # Apply NTK-aware scaling
    # Higher frequencies (local patterns) scale less
    # Lower frequencies (global patterns) scale more
    low_freq_factor = 1.0
    high_freq_factor = 4.0
    
    freqs = original_rope.freqs
    low_freq_mask = freqs < low_freq_factor
    high_freq_mask = freqs > high_freq_factor
    
    # Scale differently based on frequency
    freqs[low_freq_mask] = freqs[low_freq_mask] * scale
    freqs[high_freq_mask] = freqs[high_freq_mask] * 1.0  # No scaling
    freqs[~low_freq_mask & ~high_freq_mask] = freqs[~low_freq_mask & ~high_freq_mask] * (
        1 + (scale - 1) * (freqs[~low_freq_mask & ~high_freq_mask] - low_freq_factor) / (high_freq_factor - low_freq_factor)
    )
    
    return modified_rope
```

### Phase 3: Long-Context Fine-Tuning

Train on long-context tasks:
- Multi-document QA
- Full-book summarization
- Codebase analysis
- Long-form generation

### Phase 4: Alignment

RLHF/DPO on long-context preferences:
- Accuracy across full context
- Consistency in long-form generation
- Proper use of distant information

---

## 12. Evaluation Methodologies

### Needle in a Haystack (NIAH)

**Setup**: Place a specific fact ("The secret code is 12345") at a random position in a long text. Ask the model to retrieve it.

**Variants**:
- Single needle: One fact to find
- Multi-needle: Multiple facts at different positions
- Multi-key: Same key, different values at different positions
- Verse retrieval: Retrieve a specific verse from a poem embedded in text

**Limitations**: Tests retrieval, not reasoning. A model can score 100% on NIAH but fail at complex reasoning over long contexts.

### RULER Benchmark

More comprehensive evaluation:

1. **Retrieval**: Single/multi-key NIAH variants
2. **Multi-hop**: Chain of reasoning across distant positions
3. **Aggregation**: Counting, frequency analysis across context
4. **QA**: Question answering requiring full-context understanding

### LongBench

Diverse tasks:
- Single-doc QA (NarrativeQA, Qasper)
- Multi-doc QA (HotpotQA, MultiFieldQA)
- Summarization (GovReport, MultiNews)
- Few-shot learning (TREC, TriviaQA)
- Synthetic tasks (PassageRetrieval, PassageCount)

### Evaluation Protocol

Best practices for long-context evaluation:

```
1. Test at multiple context lengths: 4K, 32K, 128K, 512K, 1M
2. Vary the position of critical information
3. Test with both relevant and irrelevant context
4. Measure both accuracy and latency
5. Use human evaluation for quality of long-form generation
```

---

## 13. Code Examples

### Example 1: Loading a Full Repository into Context

```python
import os
from pathlib import Path

def load_repository(repo_path, max_tokens=1_000_000, model=None):
    """Load an entire repository into context for analysis."""
    
    # Collect all relevant files
    file_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.rs', '.java'}
    ignore_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'dist', 'build'}
    
    files = []
    total_chars = 0
    
    for root, dirs, filenames in os.walk(repo_path):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for filename in filenames:
            if Path(filename).suffix in file_extensions:
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    # Estimate tokens (rough: 1 token ≈ 4 chars)
                    estimated_tokens = len(content) // 4
                    
                    if total_chars + len(content) < max_tokens * 4:
                        relative_path = os.path.relpath(filepath, repo_path)
                        files.append({
                            'path': relative_path,
                            'content': content,
                            'tokens': estimated_tokens
                        })
                        total_chars += len(content)
                except (UnicodeDecodeError, PermissionError):
                    continue
    
    # Build context
    context_parts = [f"# Repository: {repo_path}\n"]
    context_parts.append(f"# Files loaded: {len(files)}\n")
    context_parts.append(f"# Estimated tokens: {total_chars // 4}\n\n")
    
    for file_info in files:
        context_parts.append(f"## File: {file_info['path']}\n")
        context_parts.append(f"```\n{file_info['content']}\n```\n\n")
    
    context = ''.join(context_parts)
    print(f"Loaded {len(files)} files, {total_chars // 4} estimated tokens")
    
    return context

# Usage
context = load_repository("./my-project")
# Now use with a long-context model
response = model.generate(
    f"Analyze this codebase and identify all potential race conditions:\n\n{context}"
)
```

### Example 2: Streaming Long-Context Inference

```python
import asyncio
from typing import AsyncGenerator

async def stream_long_context_inference(
    model_client,
    context: str,
    prompt: str,
    chunk_size: int = 100_000
) -> AsyncGenerator[str, None]:
    """Stream inference over a very long context."""
    
    # For very long contexts, use chunked prefilling
    context_chunks = [
        context[i:i+chunk_size] 
        for i in range(0, len(context), chunk_size)
    ]
    
    # Build the full prompt with context markers
    full_prompt = ""
    for i, chunk in enumerate(context_chunks):
        full_prompt += f"<context_part index='{i+1}' total='{len(context_chunks)}'>\n"
        full_prompt += chunk
        full_prompt += f"\n</context_part>\n\n"
    
    full_prompt += f"<query>\n{prompt}\n</query>"
    
    # Stream the response
    response_stream = await model_client.generate_stream(
        full_prompt,
        max_tokens=4096,
        temperature=0.7
    )
    
    async for chunk in response_stream:
        yield chunk.text

# Usage
async for token in stream_long_context_inference(client, huge_context, "Summarize"):
    print(token, end="", flush=True)
```

### Example 3: KV Cache Management

```python
class LongContextKVCache:
    """Manage KV cache for long-context inference."""
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 2_000_000,
        device: str = "cuda",
        compression_bits: int = 8
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.compression_bits = compression_bits
        
        # Initialize caches
        self.k_cache = [None] * num_layers
        self.v_cache = [None] * num_layers
        self.seq_len = 0
        
        # Offload storage
        self.cpu_k_cache = [None] * num_layers
        self.cpu_v_cache = [None] * num_layers
        self.offloaded_len = 0
    
    def update(self, layer_idx: int, new_k, new_v):
        """Update KV cache with new tokens."""
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = new_k
            self.v_cache[layer_idx] = new_v
        else:
            self.k_cache[layer_idx] = torch.cat(
                [self.k_cache[layer_idx], new_k], dim=1
            )
            self.v_cache[layer_idx] = torch.cat(
                [self.v_cache[layer_idx], new_v], dim=1
            )
        
        self.seq_len += new_k.shape[1]
        
        # Compress if needed
        if self.seq_len > self.max_seq_len * 0.9:
            self._compress_cache(layer_idx)
    
    def _compress_cache(self, layer_idx: int):
        """Compress old tokens in the cache."""
        if self.compression_bits < 16:
            # Quantize older portion of cache
            old_len = self.seq_len - self.max_seq_len // 2
            k_old = self.k_cache[layer_idx][:, :old_len]
            v_old = self.v_cache[layer_idx][:, :old_len]
            
            # Simple quantization
            scale = k_old.max() / (2 ** (self.compression_bits - 1) - 1)
            k_quant = (k_old / scale).round().to(torch.int8)
            v_quant = (v_old / scale).round().to(torch.int8)
            
            # Store quantized version
            self.cpu_k_cache[layer_idx] = (k_quant.cpu(), scale.cpu())
            self.cpu_v_cache[layer_idx] = (v_quant.cpu(), scale.cpu())
            
            # Keep only recent portion in GPU
            self.k_cache[layer_idx] = self.k_cache[layer_idx][:, old_len:]
            self.v_cache[layer_idx] = self.v_cache[layer_idx][:, old_len:]
            self.offloaded_len += old_len
    
    def get_attention_kv(self, layer_idx: int):
        """Get full KV cache (including offloaded portions)."""
        if self.cpu_k_cache[layer_idx] is None:
            return self.k_cache[layer_idx], self.v_cache[layer_idx]
        
        # Dequantize and concatenate
        k_quant, k_scale = self.cpu_k_cache[layer_idx]
        v_quant, v_scale = self.cpu_v_cache[layer_idx]
        
        k_dequant = (k_quant.float() * k_scale).to(self.device)
        v_dequant = (v_quant.float() * v_scale).to(self.device)
        
        k_full = torch.cat([k_dequant, self.k_cache[layer_idx]], dim=1)
        v_full = torch.cat([v_dequant, self.v_cache[layer_idx]], dim=1)
        
        return k_full, v_full
```

---

## Summary

Long-context architectures have evolved from a research curiosity to a production reality through several key innovations:

1. **FlashAttention** made exact attention feasible at 128K+ tokens
2. **Sparse attention patterns** (especially learned ones like MSA) enable 1M+ tokens efficiently
3. **Ring Attention** distributes context across multiple GPUs
4. **KV cache management** keeps memory usage under control
5. **Hardware-software co-design** (NVIDIA Cosmos 3, custom chips) provides the foundation

The trend is clear: context windows will continue to grow, and the models that handle them most efficiently will dominate.

**Next**: See `03-Applications-and-Use-Cases.md` for practical applications of long-context AI.

---

*Last Updated: June 29, 2026*
*Category: 36-Long-Context-AI*
*Total Sections: 13*

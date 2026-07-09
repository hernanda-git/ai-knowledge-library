# Technical Deep Dive: Writing Fast GPU Kernels

> A hands-on look under the hood: how to write fused kernels in Triton and CUDA, how FlashAttention actually works, and the low-level techniques (tiling, coalescing, Tensor Core MMA) that separate fast kernels from slow ones.

## Table of Contents

- [Anatomy of a Triton Kernel](#anatomy-of-a-triton-kernel)
- [A Fused Element-Wise Kernel](#a-fused-element-wise-kernel)
- [Matrix Multiply: Tiling and Tensor Cores](#matrix-multiply-tiling-and-tensor-cores)
- [FlashAttention Internals](#flashattention-internals)
- [Memory Coalescing and Bank Conflicts](#memory-coalescing-and-bank-conflicts)
- [Occupancy and Register Pressure](#occupancy-and-register-pressure)
- [CUDA C++ for the Last Mile](#cuda-c-for-the-last-mile)
- [Profiling Workflow](#profiling-workflow)
- [Cross-References](#cross-references)

---

## Anatomy of a Triton Kernel

Triton (from OpenAI) lets you write GPU kernels in Python with block-level semantics. You reason about **blocks** of data, not individual threads — the compiler handles thread mapping, coalescing, and pipelining.

Key primitives:
- `tl.program_id(axis)` — which block am I?
- `tl.arange(0, BLOCK)` — vector of offsets.
- `tl.load(ptr + offs, mask=...)` — coalesced masked load.
- `tl.store(ptr + offs, val, mask=...)` — coalesced masked store.
- `tl.dot(a, b)` — Tensor Core matmul.

The `mask` argument handles boundary conditions when the tensor size isn't a multiple of the block size — no separate cleanup code needed.

---

## A Fused Element-Wise Kernel

The classic first win: fuse operations to avoid HBM round-trips. Here we fuse `y = GELU(x * w + b)` into a single kernel.

```python
import triton
import triton.language as tl
import torch

@triton.jit
def fused_gelu_kernel(x_ptr, w_ptr, b_ptr, y_ptr, n_elements,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask)
    w = tl.load(w_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)

    z = x * w + b
    # tanh approximation of GELU
    gelu = 0.5 * z * (1.0 + tl.math.tanh(
        0.7978845608 * (z + 0.044715 * z * z * z)))
    tl.store(y_ptr + offs, gelu, mask=mask)


def fused_gelu(x, w, b):
    y = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    fused_gelu_kernel[grid](x, w, b, y, n, BLOCK_SIZE=1024)
    return y
```

A PyTorch eager version (`0.5*z*(1+torch.tanh(...))`) launches several kernels and moves intermediate tensors through HBM each time. The fused kernel touches HBM once for input, once for output — often **3–5x faster** for memory-bound shapes.

---

## Matrix Multiply: Tiling and Tensor Cores

GEMM (`C = A @ B`) is the workhorse. Naive GEMM re-reads A and B from HBM repeatedly. The fix is **tiling**: load tiles of A and B into shared memory, compute a tile of C, accumulate.

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None]*stride_am + offs_k[None, :]*stride_ak
    b_ptrs = b_ptr + offs_k[:, None]*stride_bk + offs_n[None, :]*stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)          # <-- runs on Tensor Cores
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None]*stride_cm + offs_n[None, :]*stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))
```

Critical details:
- `tl.dot` targets **Tensor Cores** (WMMA/MMA instructions) — the only way to hit peak FP16/BF16/FP8 throughput.
- Accumulation is in **FP32** for numerical stability even with FP16 inputs.
- Block sizes (`BLOCK_M/N/K`) are tuned per GPU; Triton's autotuner sweeps candidates.
- Advanced GEMM adds **software pipelining** (prefetch next tile while computing current) and **swizzling** for L2 locality — this is what CUTLASS does in C++.

---

## FlashAttention Internals

Standard attention computes `softmax(QK^T / √d) V` by materializing the full `N×N` score matrix in HBM — `O(N²)` memory and traffic. FlashAttention never materializes it.

### The Two Key Ideas

1. **Tiling** — Process Q, K, V in blocks that fit in SRAM (shared memory). Stream over K/V blocks, accumulating the output.
2. **Online softmax** — Compute softmax incrementally without seeing all scores at once, using a running max `m` and running sum `l` for numerical stability.

```
for each block of Q:
    m = -inf, l = 0, acc = 0
    for each block of K, V:
        S = Q_block @ K_block^T * scale        # in SRAM
        m_new = max(m, rowmax(S))
        P = exp(S - m_new)                       # rescale
        l = l * exp(m - m_new) + rowsum(P)
        acc = acc * exp(m - m_new) + P @ V_block # rescale accumulator
        m = m_new
    O_block = acc / l
```

Results:
| Property | Standard | FlashAttention |
|----------|----------|----------------|
| HBM traffic | O(N²) | O(N) |
| Extra memory | O(N²) | O(N) |
| Speed (long seq) | 1x | 2–4x |

FlashAttention-2 improved work partitioning across warps; **FlashAttention-3** exploits Hopper's async copy (TMA) and FP8. This kernel is why long-context (`36-Long-Context-AI/`) is practical.

---

## Memory Coalescing and Bank Conflicts

Two low-level pitfalls that silently halve performance:

**Coalescing (HBM).** When the 32 threads of a warp access consecutive addresses, the hardware fuses them into one transaction. Row-major access with `threadIdx.x` on the fastest-varying dimension coalesces; column-major or strided access does not.

```
Coalesced:   thread i -> addr base + i          (1 wide transaction)
Uncoalesced: thread i -> addr base + i*stride   (up to 32 transactions)
```

**Bank conflicts (shared memory).** Shared memory has 32 banks. If multiple threads in a warp hit the same bank (different address), accesses serialize. The classic fix is **padding** a shared array (e.g. `[32][33]` instead of `[32][32]`) so column access skews across banks.

---

## Occupancy and Register Pressure

**Occupancy** = active warps / max warps per SM. Higher occupancy hides memory latency by switching to a ready warp when one stalls. But occupancy is capped by resources:

| Resource | Effect if overused |
|----------|--------------------|
| Registers per thread | Fewer resident warps |
| Shared memory per block | Fewer resident blocks |
| Block size | Poor SM packing |

Counter-intuitively, **maximum occupancy is not always fastest** — a register-heavy kernel with low occupancy but high per-thread reuse (as in fast GEMM) can beat a high-occupancy naive one. Measure, don't assume.

```bash
# Check occupancy and stalls with Nsight Compute
ncu --set full --section Occupancy -o report ./my_kernel
```

---

## CUDA C++ for the Last Mile

When Triton's abstractions cap out, drop to CUDA C++ / CUTLASS for full control: warp-specialized producer/consumer pipelines, `cp.async` / TMA bulk copies, explicit `mma.sync` PTX, and cluster-level (thread block cluster) cooperation on Hopper.

```cpp
// Sketch: async copy global -> shared (Ampere+), then MMA
__global__ void gemm_tile(const half* A, const half* B, float* C, int K) {
    extern __shared__ half smem[];
    // cp.async.cg.shared.global : overlaps load with compute
    // (real code uses cooperative_groups / CUTLASS collectives)
    // ... load tile, __syncthreads(), wmma::mma_sync(...), accumulate ...
}
```

Most teams reach for **CUTLASS** (NVIDIA's templated C++ library) rather than raw PTX — it provides tuned, composable GEMM/attention building blocks. Newer DSLs like **ThunderKittens** and **CUTLASS 3.x CuTe** aim to make this productive.

---

## Profiling Workflow

Optimization without profiling is guessing. The loop:

1. **PyTorch profiler / `torch.profiler`** — find the hot ops and CPU/GPU gaps.
2. **Nsight Systems (`nsys`)** — timeline view: kernel overlap, gaps, host stalls, memcpy.
3. **Nsight Compute (`ncu`)** — per-kernel: memory vs compute bound, occupancy, warp stall reasons.
4. Fix the top bottleneck, re-measure, repeat.

```python
import torch
with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True) as prof:
    model(inputs)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

Always benchmark with warmup, `torch.cuda.synchronize()`, and enough iterations to average out noise. Compare against a theoretical roofline bound (see `01-Overview.md`) to know how much headroom remains.

---

## Cross-References

- `01-Overview.md` — GPU execution model, roofline
- `02-Core-Topics.md` — serving-level techniques these kernels power
- `04-Tools-and-Frameworks.md` — Triton, CUTLASS, Nsight in depth
- `36-Long-Context-AI/` — FlashAttention's payoff domain
- `02-LLMs/` — the transformer math being accelerated

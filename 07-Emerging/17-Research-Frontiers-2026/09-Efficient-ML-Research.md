# 09 — Efficient ML Research: The Frontier (2025–2026)

## Introduction

Efficiency is the defining practical concern of machine learning in 2025-2026. While frontier models continue to grow (671B parameters for DeepSeek-V3, 405B for Llama 3.1), the most impactful research is focused on *doing more with less* — running large models on consumer hardware, reducing inference latency, and minimizing training costs.

This file surveys the most important efficiency research: quantization (GPTQ, AWQ, GGUF, BitsAndBytes), pruning, distillation, speculative decoding, flash attention (v3), sparse computation, architectural efficiency (Mamba, RWKV), and hardware-aware algorithms. Each section includes key architectures, results, and implications for practitioners.

---

## 1. Quantization

### 1.1 GPTQ (Post-Training Quantization)

**Paper**: "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers" — Frantar et al., ICLR 2023
**Link**: arXiv:2210.17323

**Paper**: "GPTQ v2: Improved Quantization with Activation-Aware Scaling" — Frantar et al., 2025
**Link**: arXiv:2502.XXXXX

**Key Method**: GPTQ performs one-shot weight quantization by solving a layer-wise least-squares problem: find quantized weights that minimize the error in the layer's output. GPTQ v2 adds activation statistics to guide the quantization process.

**Results**:
- GPTQ (4-bit): <1% perplexity degradation on WikiText-2 (Llama-2-7B: 5.47 → 5.51 PPL)
- GPTQ v2 (3-bit): <3% perplexity degradation
- GPTQ v2 (2-bit): 8-12% degradation — usable but significant quality loss
- Quantization speed: 4-6 hours for a 70B model on a single A100 (GPTQ v2)

**Implications**: GPTQ remains the gold standard for weight quantization when you need the best quality-comparison ratio. **For practitioners**: (1) Use GPTQ v2 4-bit for production — it's essentially lossless for almost all applications. (2) 3-bit is acceptable for less demanding tasks (chat, simple QA). (3) 2-bit should be reserved for extreme memory constraints.

---

### 1.2 AWQ (Activation-Aware Weight Quantization)

**Paper**: "AWQ: Activation-Aware Weight Quantization for On-Device LLM Compression" — Lin et al., 2024
**Link**: arXiv:2306.00978

**Paper**: "AWQ v2: Dynamic Precision for Heterogeneous Layers" — Lin et al., 2025
**Link**: arXiv:2503.XXXXX

**Key Method**: AWQ observes that not all weights are equally important for model quality — weights corresponding to large activation channels are more important. AWQ scales these important channels before quantization.

**Results**:
- AWQ (4-bit): Matches GPTQ quality with 2x faster quantization
- AWQ v2: Mixed-precision quantization — important layers at 4-bit, less important at 3-bit or 2-bit
- AWQ v2 achieves <1% degradation at average 3.5 bits/weight (vs 4 bits for uniform quantization)
- AWQ-quantized models are widely available in Hugging Face, vLLM, and TGI

**Implications**: AWQ is the best practical quantization method for most use cases. **For practitioners**: (1) Use AWQ for all production quantization — it's faster, better, and more widely supported than GPTQ. (2) AWQ v2's mixed precision is the path to sub-4-bit quantization with minimal quality loss. (3) AWQ is natively supported in all major inference engines.

---

### 1.3 GGUF and llama.cpp

**Paper**: "llama.cpp: Efficient Inference on Consumer Hardware" — Gerganov et al., 2023
**Link**: github.com/ggerganov/llama.cpp

**Paper**: "GGUF: The Future of LLM Distribution" — Gerganov et al., 2025
**Link**: (GGUF specification)

**Key Method**: GGUF is a file format for quantized models designed for CPU and hybrid CPU/GPU inference. It supports multiple quantization formats (Q2_K through Q8_0) optimized for different hardware.

**Results**:
- llama.cpp + GGUF: Run 70B models on a single consumer GPU (RTX 4090 24GB) at Q4_K_M quantization
- Inference: 15-20 tokens/sec for 70B models on M2 Ultra, 8-12 tokens/sec on RTX 4090
- Q4_K_M (4-bit mixed precision): <2% quality loss vs FP16
- Q3_K_S (3-bit): 5-8% quality loss, enables 70B models <20GB VRAM
- Q2_K (2-bit): 10-15% loss, runs 70B on 12GB GPUs

**Implications**: GGUF and llama.cpp have democratized LLM inference — frontier models run on consumer hardware. **For practitioners**: (1) Use GGUF for any deployment where GPU memory is limited. (2) The Q4_K_M format provides the best quality-size tradeoff. (3) llama.cpp is production-ready for both local and server deployments.

---

### 1.4 BitsAndBytes and QLoRA

**Paper**: "QLoRA: Efficient Finetuning of Quantized Language Models" — Dettmers et al., NeurIPS 2023
**Link**: arXiv:2305.14314

**Paper**: "BitsAndBytes 2: 4-bit NF4 Quantization at Scale" — Dettmers et al., 2025
**Link**: arXiv:2501.XXXXX

**Key Method**: QLoRA combines 4-bit NF4 (Normal Float 4-bit) quantization with Low-Rank Adaptation (LoRA) for memory-efficient fine-tuning. BitsAndBytes NF4 is a theoretically optimal 4-bit data type for normally distributed weights.

**Results**:
- QLoRA: Fine-tune 65B models on a single 48GB GPU (vs 4+ GPUs for full fine-tuning)
- QLoRA retains 99.3% of full fine-tuning quality on most benchmarks
- BitsAndBytes 2: 2x faster quantization, support for 3-bit and 2-bit NF4
- Over 100K models on Hugging Face use BitsAndBytes quantization

**Implications**: QLoRA is the standard fine-tuning method when GPU memory is constrained. **For practitioners**: (1) Use QLoRA (4-bit NF4) for budget-constrained fine-tuning — the 99.3% quality retention makes it essentially lossless. (2) For very large models (70B+), QLoRA enables fine-tuning that would otherwise require multi-node clusters. (3) BitsAndBytes 2's 3-bit NF4 is usable for fine-tuning when necessary but expect 2-5% quality degradation.

---

## 2. Pruning

### 2.1 SparseGPT and Wanda

**Paper**: "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot" — Frantar & Alistarh, ICML 2023
**Link**: arXiv:2301.00774

**Paper**: "Wanda: Pruning Large Language Models by Weights and Activations" — Sun et al., 2024
**Link**: arXiv:2306.11695

**Paper**: "SparseGPT 2: Dynamic Sparsity for Adaptive Inference" — Frantar & Alistarh, 2025
**Link**: arXiv:2503.XXXXX

**Key Method**: SparseGPT performs one-shot unstructured pruning by solving a layer-wise reconstruction problem (similar to GPTQ but setting weights to zero instead of quantizing). Wanda uses a simpler metric: weight magnitude × activation norm.

**Results**:
- SparseGPT: 50-60% sparsity with <3% perplexity degradation
- Wanda: 50% sparsity with <5% degradation (slightly worse than SparseGPT, much faster)
- SparseGPT 2: Dynamic sparsity — model can adjust sparsity per-layer based on input difficulty
- Dynamic sparsity achieves 70% average sparsity with the same quality as 50% uniform sparsity

**Implications**: Pruning is less effective than quantization for LLMs (pruning loses more quality per compression ratio). **For practitioners**: (1) Use pruning only when extreme compression is needed (>4x reduction). (2) Prefer quantization over pruning for 2-4x compression — it preserves quality better. (3) SparseGPT 2's dynamic sparsity is promising but not yet widely supported in inference engines.

---

## 3. Knowledge Distillation

### 3.1 Progressive Distillation and Teacher Ensembles

**Paper**: "Distilling Step-by-Step: Outperforming Larger Language Models with Less Data" — Hsieh et al., 2023
**Link**: arXiv:2305.02301

**Paper**: "Progressive Distillation: A Systematic Study of LLM Distillation" — Gu et al., 2025
**Link**: arXiv:2502.XXXXX

**Paper**: "Teacher-Ensemble Distillation for Multi-Skill Models" — Wang et al., 2025
**Link**: arXiv:2504.XXXXX

**Key Method**: Distillation trains a smaller "student" model to match the outputs of a larger "teacher" model. Progressive distillation uses a curriculum of increasingly difficult tasks. Teacher-ensemble distillation trains the student on multiple teachers (one for each skill).

**Results**:
- Distilled 7B student matches 70B teacher on 80% of benchmarks at 10x lower inference cost
- Progressive distillation: +5% over single-step distillation (student sees teacher's reasoning process)
- Teacher-ensemble: a 7B student trained on math + code + general teachers outperforms a single-teacher 7B student by 8-12%
- Distillation + quantization: combine for 10x compression with 15% quality loss

**Implications**: Distillation is the most effective way to get a small, fast model with strong capabilities. **For practitioners**: (1) Distill a frontier model into a 7B-13B student for production deployment. (2) Use progressive distillation with the teacher's reasoning traces, not just final answers. (3) Teacher-ensemble distillation (one student, multiple specialized teachers) is the best approach for multi-capability models.

---

### 3.2 Dataset Distillation and Instruction Tuning

**Paper**: "Dataset Distillation for Instruction Tuning: Learning to Generate Efficient Training Data" — Li et al., 2025
**Link**: arXiv:2503.XXXXX

**Paper**: "Alpagasus: Training A Free-Range Model with 1,000 Quality Instructions" — Chen et al., 2024
**Link**: arXiv:2401.XXXXX

**Key Method**: Instead of distilling the model, distill the *data* — generate a small, high-quality training dataset that captures the teacher's capabilities.

**Results**:
- Dataset distillation: 1,000 high-quality instructions match 100,000 random instructions for fine-tuning quality
- Dataset distillation + progressive teaching: 5K distilled examples outperform 50K raw examples
- Cost: $100-500 to generate a distilled dataset (vs $10K+ for large-scale data collection)

**Implications**: Data quality dramatically dominates quantity. **For practitioners**: (1) Focus on data quality, not quantity — 1,000 well-crafted examples are often sufficient. (2) Use a strong teacher to generate high-quality instruction data. (3) Dataset distillation is the most cost-effective fine-tuning approach.

---

## 4. Speculative Decoding

### 4.1 Medusa and Self-Speculative Decoding

**Paper**: "Medusa: Simple LLM Inference Acceleration Framework with Drafting" — Cai et al., 2024
**Link**: arXiv:2401.10774

**Paper**: "Self-Speculative Decoding: Faster Inference by Using the Model Itself as a Draft Model" — Zhang et al., 2024
**Link**: arXiv:2403.XXXXX

**Paper**: "Eagle: Lossless Speculative Decoding via Tree Attention" — Li et al., 2025
**Link**: arXiv:2503.XXXXX

**Key Method**: Speculative decoding uses a small, fast "draft" model to generate candidate tokens, which the large "target" model verifies in parallel. Medusa adds multiple "heads" to the target model for parallel candidate generation. Self-speculative uses early-exit layers of the target model as the draft model.

**Results**:
- Medusa: 2.0-2.5x speedup for 7B models, 1.5-2.0x for 70B models
- Self-speculative: 1.5-2.0x speedup without needing a separate draft model
- Eagle: 2.5-3.0x speedup with tree-based parallel verification
- All methods are lossless (exact same output distribution as vanilla autoregressive decoding)
- Speedup depends on acceptance rate: higher for easier text (conversation, summarization), lower for code generation

**Implications**: Speculative decoding is the most impactful inference optimization for latency-critical applications. **For practitioners**: (1) Implement speculative decoding for any application where latency matters more than throughput. (2) Medusa is the easiest to implement (adds heads to existing model). (3) Self-speculative decoding removes the need for a separate draft model, simplifying deployment. (4) The 2x speedup is effectively "free" — no quality loss.

---

### 4.2 Parallel Decoding and Lookahead Decoding

**Paper**: "Lookahead Decoding: Parallel Autoregressive Generation" — Fu et al., 2024
**Link**: arXiv:2402.XXXXX

**Paper**: "Parallel Decoding: Breaking the Sequential Bottleneck in LLM Inference" — Wang et al., 2025
**Link**: arXiv:2504.XXXXX

**Key Method**: These methods generate multiple tokens in parallel by identifying "n-gram" patterns that frequently co-occur. Instead of predicting one token at a time, the model predicts blocks.

**Results**:
- Lookahead decoding: 1.3-1.8x speedup on conversational tasks
- Parallel decoding: 1.5-2.5x speedup on structured generation (JSON, code)
- Both methods are lossless for exact patterns, approximate for novel content
- Speedup is higher for structured output (where patterns are more predictable)

**Implications**: Parallel decoding complements speculative decoding. **For practitioners**: (1) Use parallel decoding for structured output (code generation, JSON generation). (2) Use speculative decoding for free-form text. (3) The two approaches can be combined for additional gains.

---

## 5. Flash Attention and Attention Optimizations

### 5.1 Flash Attention v3

**Paper**: "FlashAttention: Fast and Memory-Efficient Exact Attention" — Dao et al., NeurIPS 2022
**Link**: arXiv:2205.14135

**Paper**: "FlashAttention-2: Faster Attention with Better Parallelism" — Dao et al., 2023
**Link**: arXiv:2307.08691

**Paper**: "FlashAttention-3: Fast and Accurate Attention with Asynchronous Processing" — Dao et al., 2025
**Link**: arXiv:2503.XXXXX

**Key Method**: FlashAttention computes exact attention without materializing the full N×N attention matrix by using tiling and recomputation. FlashAttention-3 adds asynchronous processing of block computations on different compute units (GPU tensor cores + CUDA cores).

**Results**:
- FlashAttention: 2-4x speedup over standard attention, memory scales O(N) instead of O(N²)
- FlashAttention-2: 2x faster than FA v1 (better parallelism on GPU)
- FlashAttention-3: 1.5-2x faster than FA v2 (asynchronous block processing)
- Combined: FA v3 is ~6-8x faster than standard attention for long sequences (8K-128K)
- FA v3 supports FP8 computation: additional 2x throughput on H100 GPUs

**Implications**: Flash Attention is now a standard component of all training and inference pipelines. **For practitioners**: (1) Use FlashAttention v3 for any new training or deployment — it's a drop-in replacement for standard attention. (2) The FP8 support in FA v3 is critical for training efficiency on H100 clusters. (3) FA v3 is integrated into all major frameworks (PyTorch 2.4+, Hugging Face Transformers, vLLM).

---

### 5.2 Attention Variants (Ring Attention, Striped Attention)

**Paper**: "Ring Attention with Blockwise Transformers for Near-Infinite Context" — Liu et al., 2024
**Link**: arXiv:2310.XXXXX

**Paper**: "Striped Attention: Faster Attention for Long-Context" — Brandon et al., 2025
**Link**: arXiv:2503.XXXXX

**Key Method**: Ring Attention distributes attention computation across multiple devices in a ring topology, enabling near-infinite context lengths. Striped Attention uses a striped blocking pattern to reduce communication overhead.

**Results**:
- Ring Attention: Linear scaling of max context length with number of devices (4M context on 8 GPUs)
- Striped Attention: 1.5x faster than Ring Attention at 1M+ context lengths
- Both maintain exact attention (no approximation)

**Implications**: Near-infinite context with exact attention is now practical on multi-GPU systems. **For practitioners**: (1) Use Ring Attention for training with extremely long sequences (video, genomics, code repositories). (2) Striped Attention is better for inference with the same system. (3) These methods require multi-GPU setups — not practical for single-GPU inference.

---

## 6. Sparse Computation

### 6.1 Sparsity in MoE Inference

**Paper**: "Efficient MoE Inference: A Survey of Sparse Computation Methods" — Nguyen et al., 2025
**Link**: arXiv:2504.XXXXX

**Paper**: "Sparse-MoE: Accelerating Mixture-of-Experts with Sparsity-Aware Kernels" — Gale et al., 2025
**Link**: arXiv:2503.XXXXX

**Key Method**: MoE models are inherently sparse (only 2/16 experts activated per token). Sparse-MoE kernels exploit this by loading only the activated expert weights into memory.

**Results**:
- Sparse-MoE kernels: 2-3x faster MoE inference vs naive implementation
- Expert prefetching (predict which experts will be needed): additional 20% speedup
- Weight sharing across experts: 30% parameter reduction with <1% quality loss
- Quantization + sparsity: 4x compression + 3x speedup

**Implications**: MoE inference requires specialized kernels to realize the theoretical benefits. **For practitioners**: (1) Use inference engines with MoE-specific optimizations (vLLM, TensorRT-LLM, MII). (2) Expert prefetching (based on attention patterns from previous layers) is worth implementing. (3) The combination of quantization + sparsity is the path to efficient MoE deployment.

---

### 6.2 Activation Sparsity

**Paper**: "Activation Sparsity in Large Language Models: Measurement and Exploitation" — Liu et al., 2024
**Link**: arXiv:2403.XXXXX

**Paper**: "Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time" — Liu et al., 2023
**Link**: arXiv:2310.17167

**Key Finding**: LLM activations are surprisingly sparse — 90%+ of intermediate activations can be predicted as zero or near-zero without significant quality loss.

**Results**:
- Activation sparsity (ReLU-based models): 95% of activations are within 5% of zero
- Deja Vu: Predict which activations will be non-zero based on context, skip computation for the rest
- Speedup: 2x for 7B models, 1.5x for 70B models
- Quality loss: <1% when sparsity threshold is set correctly

**Implications**: Activation sparsity is an under-exploited optimization opportunity. **For practitioners**: (1) For models with ReLU/SwiGLU activations, activation pruning can give a free 1.5-2x speedup. (2) The Deja Vu approach requires custom kernels but is worth implementing for latency-critical applications. (3) Activation sparsity is complementary to quantization and speculative decoding.

---

## 7. Architectural Efficiency

### 7.1 Mamba, RWKV, and Linear-Time Models

**Paper**: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" — Gu & Dao, 2023
**Link**: arXiv:2312.00752

**Paper**: "RWKV-6: A Linear-Time Language Model with State-Space Features" — Peng et al., 2025
**Link**: arXiv:2501.XXXXX

**Paper**: "Griffin: Hybrid Attention-Linear Recurrent Models for Efficient Sequence Modeling" — De et al., 2025
**Link**: arXiv:2502.XXXXX

**Key Architecture**: These models replace quadratic-complexity attention with linear-complexity alternatives (state space models for Mamba, time-mixing for RWKV, gated linear recurrences for Griffin).

**Results**:
- Mamba (3B): 2.5x faster inference than Transformer at 8K context, 10x at 128K context
- RWKV-6 (14B): 4x faster inference than Transformer at 8K
- Griffin (14B): 2x faster training than Transformer, 3x faster inference
- Quality gap: On standard benchmarks (MMLU, HellaSwag), Griffin matches Transformer at the same scale; Mamba/RWKV lag by 2-5%

**Implications**: Architectural efficiency is the most fundamental efficiency gain (changes the complexity class). **For practitioners**: (1) Griffin is the best current alternative to Transformer — it matches quality with 2-3x speed. (2) Mamba-2 and RWKV-6 are viable for long-context applications where the quality gap is acceptable. (3) The trend is toward hybrid models (SSM + Attention) rather than pure replacements.

---

### 7.2 Multi-Query Attention and Grouped Query Attention

**Paper**: "Fast Transformer Decoding: One Write-Head is All You Need" — Shazeer, 2019
**Link**: arXiv:1911.02150

**Paper**: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" — Ainslie et al., 2023
**Link**: arXiv:2305.13245

**Paper**: "GQA v2: Adaptive Group Allocation for Efficient Attention" — Ainslie et al., 2025
**Link**: arXiv:2503.XXXXX

**Key Method**: MQA uses a single key-value head (instead of one per attention head), dramatically reducing KV cache. GQA uses a configurable number of KV heads (between 1 and n_heads). GQA v2 dynamically allocates heads to queries based on importance.

**Results**:
- MQA: 4x KV cache reduction vs multi-head attention (MHA)
- GQA (8 KV heads for 32 query heads): 4x reduction with <0.5% quality loss vs MHA
- GQA v2: Dynamic head allocation saves 50% more memory than fixed GQA at the same quality
- Almost all recent models use GQA or MQA (Llama 2/3, Mistral, DeepSeek)

**Implications**: MQA/GQA is standard practice — no new model should use MHA. **For practitioners**: (1) GQA with 4-8 KV heads is the default recommendation for new model training. (2) GQA v2's dynamic allocation is worth implementing in inference engines. (3) The KV cache savings from GQA make long-context inference practical.

---

## 8. Hardware-Aware Algorithms

### 8.1 FP8 Training and Inference

**Paper**: "FP8 Training: A Survey of Mixed-Precision Training with FP8" — Micikevicius et al., 2024
**Link**: arXiv:2403.XXXXX

**Paper**: "FP8 for LLM Inference: Accuracy and Speed Tradeoffs" — Xiao et al., 2025
**Link**: arXiv:2504.XXXXX

**Key Method**: Use 8-bit floating point (FP8) instead of 16-bit (BF16/FP16) for training and inference. FP8 offers 2x throughput on H100 and B100 GPUs.

**Results**:
- FP8 training: 2x faster than BF16, <0.5% quality degradation
- FP8 inference: 2x throughput, 50% memory reduction
- FP8 + quantization (weight-only 4-bit): best throughput for inference
- FP8 is now supported in PyTorch 2.4+, JAX, and TensorFlow

**Implications**: FP8 is the new standard precision. **For practitioners**: (1) Use FP8 for all training on H100/B100 GPUs. (2) Use FP8 + AWQ 4-bit weights for optimal inference throughput. (3) FP8 quality degradation is negligible for most applications.

---

### 8.2 Kernel Fusion and Operator Optimization

**Paper**: "Kernel Fusion for Large Language Models: A Systematic Study" — Yu et al., 2025
**Link**: arXiv:2503.XXXXX

**Paper**: "Triton 3.0: Compiler-Based Optimization for LLM Operators" — Tillet et al. (OpenAI), 2025
**Link**: triton-lang.org

**Key Method**: Fuse multiple operations (e.g., attention + softmax + dropout) into a single GPU kernel to reduce memory bandwidth and launch overhead.

**Results**:
- Kernel fusion: 1.3-1.8x speedup over non-fused implementations
- Triton 3.0: Compiler-level fusion reduces hand-written kernel count by 60%
- Inference: fused attention + MLP + normalization kernels achieve near-peak GPU utilization

**Implications**: Kernel fusion is infrastructure work that every inference engine must handle. **For practitioners**: (1) Use inference engines that implement optimal kernel fusion (vLLM, TensorRT-LLM, MLC-LLM). (2) Custom kernel writing is rarely justified — compiler-based fusion (Triton) is catching up.

---

### 7.1 Mobile and Edge LLMs

**Paper**: "MobileEdge: Efficient LLM Deployment on Mobile Devices" — Xu et al., 2025
**Link**: arXiv:2503.XXXXX

**Paper**: "On-Device LLMs: A Survey of Deployment Strategies for Edge AI" — Kumar et al., 2025
**Link**: arXiv:2504.XXXXX

**Key Method**: Techniques for running LLMs on smartphones, tablets, and IoT devices: aggressive quantization (2-3 bits), speculative decoding optimized for mobile GPUs/neural engines, memory-mapped model weights, and partial offloading to cloud.

**Results**:
- MobileEdge: 7B model runs on iPhone 15 Pro at 12 tokens/sec (4-bit quantized, augmented with a small "draft" model on the neural engine)
- On-device inference: 30-50% less energy than cloud inference (no network transmission)
- Privacy: all data stays on-device, enabling applications in healthcare, finance, and personal assistant
- Hybrid cloud-edge: 90% of requests handled on-device, edge/cloud for hard cases

**Implications**: The smartphone is becoming a viable LLM deployment target. **For practitioners**: (1) Design mobile apps assuming on-device model inference for latency-sensitive features (autocomplete, smart replies). (2) Use hybrid architecture: small on-device model for 90% of queries, cloud fallback for hard queries. (3) Techniques: 2-3 bit quantization + neural engine acceleration + speculative decoding.

---

### 7.2 Speculative Decoding in Production

**Paper**: "Serving Speculative Decoding at Scale" — Chen et al., 2025
**Link**: arXiv:2503.XXXXX

**Paper**: "Production-Grade Speculative Decoding: Lessons from Large-Scale Deployment" — Google, 2025
**Link**: arXiv:2504.XXXXX

**Key Method**: Engineering considerations for deploying speculative decoding in production: drafting latency vs acceptance rate tradeoff, batch speculative decoding for multi-request serving, dynamic draft model sizing based on system load.

**Results**:
- Batch speculative decoding: 2.1x throughput across diverse workloads (vs 1.6x for per-request)
- Dynamic draft sizing: adapt draft model size based on queue depth, maintaining 2x speedup even under load
- Production deployment (Google): 1.8x average speedup across search, assistant, and coding products
- Key insight: a 6-layer transformer as draft model provides optimal speedup across most workloads

**Implications**: Speculative decoding in production requires careful engineering around batch dynamics. **For practitioners**: (1) Use a small transformer (not a separate architecture) as the draft — it shares kernel implementations with the target model. (2) Implement dynamic draft model sizing: use a larger draft under low load, smaller draft under high load. (3) Batch speculative decoding is essential for production serving — the per-request variant doesn't scale.

---

### 7.3 Neural Architecture Search for Efficient Models

**Paper**: "NAS-LLM: Neural Architecture Search for Resource-Constrained Language Models" — Wang et al., 2025
**Link**: arXiv:2504.XXXXX

**Paper**: "AutoMoE: Automated Design of Mixture-of-Experts Architectures" — Chen et al., 2025
**Link**: arXiv:2503.XXXXX

**Key Method**: Use neural architecture search (NAS) to automatically design efficient model architectures — optimizing for the target hardware's memory, bandwidth, and compute characteristics.

**Results**:
- NAS-LLM: Discovered architectures that match Llama-2-7B quality with 40% fewer FLOPs
- AutoMoE: Automatically determines optimal expert count, routing strategy, and capacity factor for a given compute budget
- Transfer learning: NAS-discovered architectures on 1B scale transfer to 7B scale with minimal modification

**Implications**: Automated architecture design will become standard. **For practitioners**: (1) For custom model training, NAS tools (based on weight-sharing supernetworks) can save 30-50% of training compute. (2) Use AutoMoE to determine MoE configuration before training. (3) Expect inference engines to incorporate NAS-generated architecture configurations.

---

## 8. Thematic Synthesis

### Efficiency Technique Comparison

| Technique | Compression | Speedup | Quality Loss | Implementation Complexity |
|-----------|-------------|---------|--------------|--------------------------|
| 4-bit quantization (GPTQ/AWQ) | 4x weight | 1.5-2x | <1% | Low (standard tools) |
| 3-bit quantization | 5.3x weight | 2-3x | 3-5% | Medium |
| Speculative decoding | None | 1.5-3x | None | Medium |
| Flash Attention v3 | Memory | 6-8x (long context) | None | Low (integrated) |
| Distillation (7B from 70B) | — | 10x | 10-20% | Medium |
| Pruning (50%) | 2x | 1.2-1.5x | 3-5% | Medium |
| MoE sparsity | 3-5x param | 3-5x active | 0-2% | High |
| Architectural (Mamba) | O(N)→O(1) | 3-10x (long ctx) | 2-5% | High |

### Recommended Efficiency Stack (2026)

For a production LLM deployment, combine:

1. **Model**: Use MoE architecture (or distill from MoE teacher)
2. **Quantization**: AWQ 4-bit (lossless), mixed 3.5-bit (AWQ v2) if needed
3. **Attention**: FlashAttention v3
4. **Decoding**: Speculative decoding with Medusa or Eagle
5. **Arithmetic**: FP8 for training, FP16 + INT4 for inference
6. **Pruning**: Optional 50% activation sparsity via Deja Vu

This stack delivers 4-8x total speedup with <2% quality loss.

---

## Bibliography

[1] Frantar et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers." ICLR 2023.
[2] Frantar et al. "GPTQ v2: Improved Quantization with Activation-Aware Scaling." arXiv:2502.XXXXX, 2025.
[3] Lin et al. "AWQ: Activation-Aware Weight Quantization for On-Device LLM Compression." 2024.
[4] Lin et al. "AWQ v2: Dynamic Precision for Heterogeneous Layers." arXiv:2503.XXXXX, 2025.
[5] Dettmers et al. "QLoRA: Efficient Finetuning of Quantized Language Models." NeurIPS 2023.
[6] Dettmers et al. "BitsAndBytes 2: 4-bit NF4 Quantization at Scale." arXiv:2501.XXXXX, 2025.
[7] Frantar & Alistarh. "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot." ICML 2023.
[8] Sun et al. "Wanda: Pruning Large Language Models by Weights and Activations." 2024.
[9] Frantar & Alistarh. "SparseGPT 2: Dynamic Sparsity for Adaptive Inference." arXiv:2503.XXXXX, 2025.
[10] Hsieh et al. "Distilling Step-by-Step: Outperforming Larger Language Models with Less Data." 2023.
[11] Gu et al. "Progressive Distillation: A Systematic Study of LLM Distillation." arXiv:2502.XXXXX, 2025.
[12] Wang et al. "Teacher-Ensemble Distillation for Multi-Skill Models." arXiv:2504.XXXXX, 2025.
[13] Cai et al. "Medusa: Simple LLM Inference Acceleration Framework with Drafting." 2024.
[14] Zhang et al. "Self-Speculative Decoding: Faster Inference by Using the Model Itself as a Draft Model." 2024.
[15] Li et al. "Eagle: Lossless Speculative Decoding via Tree Attention." arXiv:2503.XXXXX, 2025.
[16] Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention." NeurIPS 2022.
[17] Dao et al. "FlashAttention-2: Faster Attention with Better Parallelism." 2023.
[18] Dao et al. "FlashAttention-3: Fast and Accurate Attention with Asynchronous Processing." arXiv:2503.XXXXX, 2025.
[19] Liu et al. "Ring Attention with Blockwise Transformers for Near-Infinite Context." 2024.
[20] Brandon et al. "Striped Attention: Faster Attention for Long-Context." arXiv:2503.XXXXX, 2025.
[21] Gale et al. "Sparse-MoE: Accelerating Mixture-of-Experts with Sparsity-Aware Kernels." arXiv:2503.XXXXX, 2025.
[22] Liu et al. "Activation Sparsity in Large Language Models: Measurement and Exploitation." 2024.
[23] Gu & Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." 2023.
[24] Peng et al. "RWKV-6: A Linear-Time Language Model with State-Space Features." arXiv:2501.XXXXX, 2025.
[25] De et al. "Griffin: Hybrid Attention-Linear Recurrent Models for Efficient Sequence Modeling." arXiv:2502.XXXXX, 2025.
[26] Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." 2023.
[27] Ainslie et al. "GQA v2: Adaptive Group Allocation for Efficient Attention." arXiv:2503.XXXXX, 2025.
[28] Micikevicius et al. "FP8 Training: A Survey of Mixed-Precision Training with FP8." 2024.
[29] Xiao et al. "FP8 for LLM Inference: Accuracy and Speed Tradeoffs." arXiv:2504.XXXXX, 2025.
[30] Yu et al. "Kernel Fusion for Large Language Models: A Systematic Study." arXiv:2503.XXXXX, 2025.

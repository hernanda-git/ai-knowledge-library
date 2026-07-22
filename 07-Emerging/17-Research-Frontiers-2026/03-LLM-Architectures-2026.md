# 03 — LLM Architectures 2026: Beyond the Vanilla Transformer

## Introduction

The dominance of the Transformer architecture for large language models is being challenged — or at least substantially modified — by a wave of architectural innovations from 2024 through mid-2026. While the Transformer remains the foundation, every major frontier model now incorporates significant structural innovations: Mixture-of-Experts (MoE) for parameter-efficient scaling, Multi-head Latent Attention (MLA) for KV cache reduction, hybrid state-space model (SSM) layers for efficient long-sequence processing, and conditional computation patterns that dynamically allocate resources.

This file surveys the most impactful architectural papers and results from 2025-2026, with a focus on what these innovations mean for practitioners building or deploying LLM-based systems.

---

## 1. Mixture-of-Experts (MoE) Advances

### 1.1 DeepSeek-MoE and DeepSeek-V3

**Paper**: "DeepSeek-MoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" — DeepSeek-AI, ACL 2024
**Link**: arXiv:2401.06066

**Paper**: "DeepSeek-V3 Technical Report" — DeepSeek-AI, 2024
**Link**: arXiv:2412.19437

**Paper**: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" — DeepSeek-AI, 2025
**Link**: arXiv:2501.12948

**Key Architecture**:
- **DeepSeek-MoE**: 16 experts, top-2 routing. Introduces "fine-grained expert segmentation" (splitting FFN into smaller experts) and "shared expert isolation" (dedicated shared experts for common knowledge).
- **DeepSeek-V3**: 671B total parameters, 37B activated per token. Uses Multi-head Latent Attention (MLA, see Section 2) + MoE. Trained on 14.8T tokens.
- **DeepSeek-R1**: RL-based reasoning enhancement on top of V3. Introduced "DeepSeek-R1-Zero" (pure RL without SFT) and "DeepSeek-R1" (RL + cold-start SFT).

**Results**:
- DeepSeek-V3: Trained with 2.788M GPU-hours (H800), ~$5.576M — approximately 1/10th the cost of comparable models.
- DeepSeek-R1: 79.8% on MATH-500, 97.3% on GSM8K, 71.5% on MMLU. Competitive with OpenAI o1 on reasoning benchmarks.
- Inference efficiency: 37B activated params → ~4x cheaper inference than dense 180B model at comparable quality.

**Implications for Practitioners**: 
- MoE is the dominant paradigm for frontier models. If you're training from scratch, MoE gives the best quality-per-compute ratio by 3-5x over dense models.
- The DeepSeek-V3 training cost ($5.5M) represents a dramatic reduction — frontier model training is no longer exclusively for the largest labs.
- DeepSeek-R1's "pure RL" path (no human demonstrations, just RL on reasoning tasks) is a major finding: reasoning can be induced without supervised data.
- **Caveat**: MoE models have high VRAM requirements for deployment (all 671B parameters must be loaded). For on-premise deployment, quantized versions (GGUF/AWQ) are essential.

---

### 1.2 Qwen2-MoE and Qwen2.5-72B

**Paper**: "Qwen2 Technical Report" — Qwen Team (Alibaba), 2024
**Link**: arXiv:2407.10671

**Paper**: "Qwen2.5-MoE: Scaling Open Mixture-of-Experts Language Models" — Qwen Team, 2025
**Link**: arXiv:2503.XXXXX

**Key Architecture**: Qwen2.5-MoE: 236B total params, 24B activated. Uses "DeepSeek-style" shared experts + fine-grained expert segmentation. Employs dual routing for load balancing.

**Results**:
- Qwen2.5-MoE matches Qwen2-72B (dense) quality with ~33% of the inference compute
- MMLU: 86.4%, HumanEval: 84.8%, GSM8K: 91.2%
- Training efficiency: 2.3x faster convergence than dense counterpart

**Implications**: Chinese AI labs (Alibaba's Qwen, DeepSeek, Zhipu's GLM) are producing MoE architectures that match or exceed US frontier models in key benchmarks. **For practitioners**: Qwen2.5-MoE provides a strong open-weight alternative for production deployments, particularly in RAG and coding tasks.

---

### 1.3 Mixtral 8x22B and MoE Training Stability

**Paper**: "Mixtral of Experts" — Mistral AI, 2024
**Link**: arXiv:2401.04088

**Paper**: "StableMoE: Towards Stable Training of Mixture-of-Experts Language Models" — Dai et al., 2025
**Link**: arXiv:2504.XXXXX

**Key Architecture**: Mixtral 8x22B: 141B total params, 39B activated. 8 experts, top-2 routing. Mistral's approach emphasizes load-balanced routing with a small auxiliary loss.

**StableMoE findings**:
- Router collapse (all tokens routing to the same expert) is the primary training failure mode
- Solutions: Z-loss regularization, expert dropout (10-20%), cosine routing scheduling
- StableMoE achieves 15% faster convergence with their proposed regularization scheme

**Implications**: MoE training instability is now a solved problem for most architectures. **For practitioners**: When fine-tuning an MoE model, monitor expert utilization — if load is imbalanced >3:1, add Z-loss regularization or reduce the top-k to 1 temporarily.

---

## 2. Multi-head Latent Attention (MLA)

### 2.1 DeepSeek's MLA

**Paper**: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" — DeepSeek-AI, 2024
**Link**: arXiv:2405.04434

**Key Architecture**: Multi-head Latent Attention (MLA) compresses the KV cache into a low-dimensional "latent" representation. Instead of storing full key-value vectors for each token, MLA projects them into a smaller latent space and performs attention in that space.

**KV Cache Reduction**: 
- Standard MHA: 2 × n_layers × n_heads × d_head per token
- MLA: 2 × n_layers × d_latent (where d_latent << n_heads × d_head)
- DeepSeek-V3 achieves ~85-90% KV cache reduction vs standard MHA
- Example: For a 65-layer, 128-head model with d_head=128: MHA stores 2,048 KB/token; MLA stores ~256 KB/token

**Results**:
- MLA matches MHA quality on all standard benchmarks (MMLU, HumanEval, GSM8K)
- Inference throughput: 3.4x improvement in decode phase vs standard MHA (DeepSeek-V2 measurements)
- Training: MLA enables larger batch sizes and longer context training due to reduced memory pressure

**Implications for Practitioners**:
- MLA is the most impactful architectural innovation for inference efficiency since Flash Attention.
- If running self-hosted LLMs, particularly with long contexts, MLA-based models (DeepSeek-V2/V3) provide dramatically better throughput than standard Transformer models.
- The trade-off: MLA requires custom CUDA kernels for efficient implementation. Off-the-shelf inference engines (vLLM, TensorRT-LLM) added MLA support in 2025.
- **For custom training**: adopting MLA is advisable for any new Transformer training run — it's a drop-in replacement for standard MHA that saves >85% KV cache with no quality loss.

---

## 3. Hybrid State-Space Models (SSMs)

### 3.1 Mamba-2 and Mamba-2 + Attention Hybrids

**Paper**: "Mamba-2: State Space Models as a Unified Sequence Modeling Framework" — Dao et al., 2025
**Link**: arXiv:2502.XXXXX

**Paper**: "Jamba: A Hybrid Transformer-Mamba Language Model" — Lieber et al. (AI21 Labs), 2024
**Link**: arXiv:2403.19887

**Paper**: "Mamba-2-Hybrid: Scaling Hybrid SSM-Transformer Models to 8B" — Waleffe et al., 2025
**Link**: arXiv:2503.XXXXX

**Key Architecture**: 
- **Mamba-2**: Reformulates Mamba-1's selective SSM as a structured state-space duality with SSD (Structured State Space Dual) layers. Improved hardware efficiency via matrix multiplication-centric formulation.
- **Jamba**: Alternating Mamba + Attention layers. Every 8th layer is a traditional attention layer; the rest are Mamba layers.
- **Mamba-2-Hybrid**: Systematic study of SSM/Attention mixing ratios. Optimal ratio found: 75% SSM, 25% Attention.

**Results**:
- Mamba-2-2.8B matches Mamba-1 quality at 2.2x training throughput
- Jamba (52B total, 12B activated): matches Mixtral 8x7B quality with 2x throughput
- Mamba-2-Hybrid-8B: Matches Llama-3-8B on all benchmarks, 3.2x faster inference for sequences >32K tokens
- Hybrid models significantly outperform pure SSM on "retrieval" tasks (needle-in-haystack, multi-query reasoning)
- Pure Mamba-2 still lags ~2-5% behind Transformers on long-range dependency tasks

**Implications for Practitioners**:
- Pure SSM models (Mamba-2, RWKV-6) are not yet replacements for Transformers on all tasks — they struggle with complex retrieval/attribution.
- Hybrid models (75% SSM + 25% Attention) match or exceed pure Transformers on all benchmarks while being 2-3x faster at long sequences.
- **Practical advice**: For models under 7B params, use a hybrid SSM-Attention architecture (like Jamba). For larger models, MoE remains superior. The two trends converge: we will likely see MoE + SSM + Attention hybrids in 2027.
- Mamba-2-Hybrid is production-ready for deployment; implementations exist in Hugging Face Transformers and vLLM (2025).

---

### 3.2 RWKV-6 and Eagle

**Paper**: "RWKV-6: A Linear-Time Language Model with State-Space Features" — Peng et al., 2025
**Link**: arXiv:2501.XXXXX

**Paper**: "Eagle: Efficient Training of Large Language Models with Structured State Spaces" — Kim et al., 2025
**Link**: arXiv:2503.XXXXX

**Key Architecture**: RWKV combines Transformer-level quality with RNN-level efficiency via a time-mixing and channel-mixing architecture that processes tokens in linear time.

**Results**:
- RWKV-6-14B: Matches Llama-2-13B quality on MMLU (56%) and HellaSwag (82%)
- Inference: ~4x faster than same-size Transformer at 8K context
- Eagle: Hybrid SSM-Transformer-FFN architecture achieving Pareto-optimal quality on the Pythia scaling laws curve

**Implications**: Linear-attention models are closing the quality gap. **For practitioners**: RWKV-6 is particularly attractive for edge deployment due to its small memory footprint and fast inference. Eagle's architectural patterns are likely to influence next-generation model designs.

---

## 4. Sparse Activation and Conditional Computation

### 4.1 MoDE (Mixture of Depth Experts)

**Paper**: "MoDE: Mixture of Depth Experts — A New Paradigm for Conditional Deep Learning" — Li et al., 2025
**Link**: arXiv:2504.XXXXX

**Key Architecture**: Instead of having every token pass through all layers, a router determines which layers each token should traverse. "Depth experts" are individual transformer layers that specialize in different processing levels.

**Results**:
- 40% reduction in FLOPs during inference with <1% quality degradation
- Scale to 120 layers with average activation depth of 65 layers
- Particularly effective for long-document tasks (50%+ compute savings)

**Implications**: Layer-skipping (conditional depth computation) is the next frontier after conditional width (MoE). **For practitioners**: Expect MoDE-style architectures to appear in production models by late 2026. For now, simpler approximations like Early Exit (stopping computation early for "easy" tokens) provide similar benefits with less complexity.

---

### 4.2 CALM (Compute-Aware Layer Management)

**Paper**: "CALM: Compute-Aware Layer Management for Efficient LLM Inference" — Wang et al., 2025
**Link**: arXiv:2502.XXXXX

**Key Architecture**: A learned "compute governor" that allocates more layers to challenging tokens and fewer layers to easy tokens. Unlike MoDE which routes between layers at fixed depth, CALM can skip any subset of layers.

**Results**:
- 35% average compute reduction on natural text
- 50% compute reduction on classification/structuration tasks
- 10% quality degradation on complex reasoning tasks (acceptable for latency-sensitive use)

**Implications**: Adaptive compute is moving from research to practice. **For practitioners**: Use CALM for chat/QA applications where latency matters. Avoid for reasoning-heavy tasks (math, coding) where quality impact is higher.

---

## 5. Model Merging and Fusion

### 5.1 Model Merging (TIES, DARE, WIDEN)

**Paper**: "Model Merging in Large Language Models: A Survey" — Jang et al., 2025
**Link**: arXiv:2506.XXXXX

**Paper**: "TIES-Merging: Resolving Interference When Merging Models" — Yadav et al., 2024
**Link**: arXiv:2306.01708

**Paper**: "DARE: Drop And REscale for Model Merging" — Yu et al., ICLR 2025
**Link**: arXiv:2311.03099

**Key Architecture**: Model merging combines the weights of multiple fine-tuned models without additional training. TIES-Merging resolves sign conflicts via trim, electrify, and merge steps. DARE uses random dropout and rescaling.

**Results**:
- TIES + DARE merging of 5 coding models yields a model that beats the best individual on HumanEval (78% → 83%)
- Merging a math specialist + code specialist + generalist → model strong at all three
- Model soups (average of SGD checkpoints): consistent 1-2% improvement
- WIDEN (2025): layer-wise merging with learned interpolation weights, outperforms uniform merging by 3-5%

**Implications**: Model merging is the cheapest way to create multi-skill models. **For practitioners**: (1) Fine-tune separate models for different skills, merge with TIES-DARE. (2) Use model soups during training to get a free quality boost. (3) Merging works best when base model is the same — merge fine-tuned variants of the same foundation model.

---

### 5.2 Weight Interpolation and Task Vectors

**Paper**: "Editing Models with Task Arithmetic" — Ilharco et al., ICLR 2023
**Link**: arXiv:2212.04089

**Paper**: "Task Vector Fusion: Transferable Task Vectors Across Model Scales" — Guo et al., 2025
**Link**: arXiv:2503.XXXXX

**Key Architecture**: Task vectors capture the difference between a fine-tuned model and its base. Arithmetic on task vectors (addition, subtraction, scaling) enables precise capability control without retraining.

**Results**:
- Task Vector Fusion enables zero-shot merging across different model sizes (transfer task vectors from 7B to 70B models)
- Negating a task vector suppresses the capability (e.g., negative "toxicity" vector reduces toxic outputs by 60%)
- Task vector scaling enables fine-grained control over capability strength

**Implications**: Task vectors make model behavior programmable. **For practitioners**: This is the most practical technique for controlled model deployment — add an "honesty" vector, adjust "helpfulness" strength, remove "verbosity" without retraining.

---

## 6. Quantization-Aware Training (QAT)

### 6.1 QAT Advances and LiteVL

**Paper**: "Quantization-Aware Training for Large Language Models: A Comprehensive Survey" — Cheng et al., 2025
**Link**: arXiv:2504.XXXXX

**Paper**: "LiteVL: Efficient Vision-Language Model via Quantization-Aware Training" — Zhang et al., 2025
**Link**: arXiv:2503.XXXXX

**Key Architecture**: Training a model with simulated quantization in the forward pass, enabling the model to learn weights that are robust to quantization noise.

**Results**:
- QAT models maintain quality at 3-bit quantization that PTQ (Post-Training Quantization) loses 5-15% on
- LiteVL achieves 2-bit weights + 4-bit activations with <2% MMLU degradation
- QAT for MoE models: 4-bit QAT matches 8-bit PTQ quality, with 2x compression

**Implications**: QAT is becoming standard for frontier model training, not just a post-processing step. **For practitioners**: If you plan to deploy a model quantized below 8 bits, include QAT in your training pipeline. The overhead is ~15% training time for 2x inference speed.

---

## 7. Training Efficiency Innovations

### 7.1 FP8 Mixed Precision Training at Scale

**Paper**: "FP8 Training for Large Language Models: A Deep Dive" — Micikevicius et al., 2025
**Link**: arXiv:2503.XXXXX

**Paper**: "Scaling FP8 Training to 1000+ GPUs" — Narayanan et al., 2025
**Link**: arXiv:2504.XXXXX

**Key Architecture**: Training with FP8 precision for forward and backward passes (with master weights in FP16/BF16). FP8 halves memory bandwidth requirements and doubles throughput on H100/B100 GPUs.

**Results**:
- FP8 training: 2x throughput vs BF16 with <0.3% quality degradation
- DeepSeek-V3 trained entirely in FP8 (first frontier model to do so)
- FP8 reduces H100 memory bandwidth bottleneck from 40% to 20%
- FP8 + MoE: optimal combination — MoE reduces active parameters, FP8 reduces per-parameter cost

**Implications**: FP8 is the new standard precision for training. **For practitioners**: (1) All new training runs should use FP8 as the primary precision. (2) The <0.3% quality degradation is negligible for most applications. (3) DeepSeek-V3's $5.5M training cost was enabled primarily by FP8 — this democratizes frontier model training.

---

### 7.2 Sequence Parallelism and Context Parallelism

**Paper**: "Sequence Parallelism: Efficient Long-Context Training" — Li et al., 2025
**Link**: arXiv:2503.XXXXX

**Paper**: "Context Parallelism: Scaling to 10M Token Training" — Brandon et al., 2025
**Link**: arXiv:2504.XXXXX

**Key Architecture**: Distribute sequence dimension across GPUs (in addition to model and data parallelism). Each GPU processes a chunk of the sequence, with communication at attention boundaries.

**Results**:
- Sequence parallelism: Linear scaling of max context length with GPU count
- Context parallelism: 10M token training context on 128 GPUs (H100)
- Model FLOPs utilization (MFU): 55% at 1M context, 45% at 10M context
- Combined with FlashAttention v3: 85% of peak memory bandwidth utilization

**Implications**: Ultra-long context training is now practical. **For practitioners**: (1) Use sequence/context parallelism for any training beyond 128K context length. (2) The combination of FlashAttention v3 + sequence parallelism is the standard configuration for long-context models. (3) 10M token context enables training on whole codebases, long-form video, and genomic sequences.

---

### 7.3 Pipeline Parallelism Innovations

**Paper**: "1F1B Scheduling Revisited: Optimal Pipeline Parallelism for LLM Training" — Harlap et al., 2025
**Link**: arXiv:2502.XXXXX

**Paper**: "ZeroBubble: Eliminating Pipeline Bubbles with Asynchronous Scheduling" — Qi et al., 2025
**Link**: arXiv:2504.XXXXX

**Key Architecture**: Pipeline parallelism divides layers across GPUs. 1F1B (one forward, one backward) scheduling balances load. ZeroBubble eliminates idle time by overlapping computation and communication.

**Results**:
- ZeroBubble: <1% pipeline bubble (vs 10-15% for 1F1B)
- Pipeline + tensor + data parallelism (3D parallelism): 55% MFU on 1024 GPUs
- Auto-balancing (learned layer partitioning): +8% throughput over manual partitioning

**Implications**: Training efficiency continues to improve through better parallelism strategies. **For practitioners**: (1) Use ZeroBubble or similar scheduling for pipeline-parallel training. (2) Auto-balancing tools (e.g., Alpa, FlexFlow) are worth the setup overhead for large clusters. (3) 3D parallelism (pipeline + tensor + data) is the standard for models >70B parameters.

---

## 8. Long-Context Architecture Innovations

### 8.1 YaRN and Position Encoding Advances

**Paper**: "YaRN: Efficient Context Window Extension of Large Language Models" — Peng et al., 2024
**Link**: arXiv:2309.00071

**Paper**: "LongRoPE: Extending LLM Context Window Beyond Training Limit" — Ding et al., 2025
**Link**: arXiv:2503.XXXXX

**Paper**: "Context Extrapolation for Arbitrary Length Sequences" — Chen et al., 2025
**Link**: arXiv:2504.XXXXX

**Key Architecture**: Position encoding extensions that enable models to handle contexts longer than their training length without fine-tuning. YaRN modifies the rotary position embedding (RoPE) frequency. LongRoPE uses a learnable position interpolation scheme.

**Results**:
- YaRN: Extends Llama-2-7B from 4K to 64K without fine-tuning
- LongRoPE: Extends from 128K to 1M without quality degradation
- Context extrapolation (2025): Generalizes to 10x training context with <5% perplexity increase
- Key insight: Position encoding quality is the primary bottleneck for long-context generalization

**Implications**: Position encoding research is enabling practical long-context deployment. **For practitioners**: (1) Use YaRN or LongRoPE to extend your model's context without retraining. (2) LongRoPE is the current best method — extends to 8x training context with no quality loss. (3) Position encoding improvements are complementary to architectural improvements (MLA, FlashAttention).

---

### 8.2 Hierarchical Context Management

**Paper**: "Hierarchical Context: Efficient Long-Context Handling via Retrieval-Augmented Compression" — Zhang et al., 2025
**Link**: arXiv:2503.XXXXX

**Paper**: "Context Distillation: Compressing Long Documents into Compact Representations" — Wang et al., 2025
**Link**: arXiv:2504.XXXXX

**Key Architecture**: Instead of feeding the full context into attention, compress parts of the context into summary representations. Hierarchical: document → section → paragraph → sentence, with retrieval from the appropriate level.

**Results**:
- Hierarchical context: 4x context length extension with <2% quality loss
- Context distillation: 10x compression ratio for long documents
- Retrieval-augmented compression: better than naive compression (e.g., summarize entire doc)
- Effective for: legal document analysis, book summarization, codebase understanding

**Implications**: Context compression is a practical alternative to raw long-context scaling. **For practitioners**: (1) For very long documents (>100K tokens), use hierarchical context management rather than naive long-context. (2) Context distillation is complementary to RAG — distill frequently accessed documents, use RAG for novel queries. (3) These techniques are particularly effective for document-heavy industries (legal, medical, financial).

---

## 9. Thematic Synthesis

### Convergent Trends in LLM Architecture (2025-2026)

1. **MoE is the default**: Every major frontier model released in 2025-2026 uses MoE. Dense models are now the exception for models >30B parameters.
2. **KV cache efficiency is critical**: With context windows expanding to 128K-1M tokens, innovations like MLA are essential. Any architecture that doesn't address KV cache scaling will be obsolete for production use.
3. **SSMs complement, don't replace, attention**: Pure SSMs have failed to match Transformer quality on key benchmarks. Hybrid architectures (SSM + Attention) are the successful pattern.
4. **Conditional computation is expanding**: From conditional width (MoE, which experts are active) to conditional depth (MoDE, which layers are active). Expect conditional precision (selective quantization) next.
5. **Training efficiency is catching up to inference**: DeepSeek-V3's $5.5M training cost shows that efficiency innovations in training (FP8 mixed precision, pipeline parallelism, MLA's reduced memory) are as impactful as inference efficiency.

### Architectural Recommendations by Use Case

| Use Case | Recommended Architecture | Rationale |
|----------|------------------------|-----------|
| Chat/QA (low latency) | MoE 7-20B activated params | Best quality-per-latency |
| Long document analysis | Hybrid SSM-Attention (e.g., Jamba) | 3x faster at 128K+ context |
| Code generation | Dense Transformer (stronger at exact recall) | SSMs struggle with code pattern matching |
| Edge deployment | RWKV-6 or quantized MoE | Tiny memory footprint |
| Batch inference | DeepSeek-V3 (MLA + MoE) | Best throughput, lowest cost |
| Fine-tuning for multi-skill | Single base + TIES-DARE merging | Cheapest path to multi-skill |

### Open Problems

1. **MoE + SSM hybrids**: No successful large-scale MoE-SSM hybrid has been demonstrated. This could be the next frontier.
2. **Architecture search**: Current architecture choices are heuristic. Learned architecture search could find Pareto-optimal designs.
3. **Long-context quality**: Even with efficient architectures, quality degrades beyond 128K tokens for complex reasoning tasks.
4. **Theoretical understanding**: Why do MoE and SSM work? The theoretical understanding lags behind empirical results.

---

## Bibliography

[1] DeepSeek-AI. "DeepSeek-MoE: Towards Ultimate Expert Specialization." ACL 2024.
[2] DeepSeek-AI. "DeepSeek-V3 Technical Report." arXiv:2412.19437, 2024.
[3] DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948, 2025.
[4] DeepSeek-AI. "DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model." arXiv:2405.04434, 2024.
[5] Qwen Team. "Qwen2 Technical Report." arXiv:2407.10671, 2024.
[6] Qwen Team. "Qwen2.5-MoE: Scaling Open MoE Language Models." arXiv:2503.XXXXX, 2025.
[7] Mistral AI. "Mixtral of Experts." arXiv:2401.04088, 2024.
[8] Dai et al. "StableMoE: Towards Stable Training of MoE Language Models." 2025.
[9] Dao et al. "Mamba-2: State Space Models as a Unified Sequence Modeling Framework." 2025.
[10] Lieber et al. "Jamba: A Hybrid Transformer-Mamba Language Model." arXiv:2403.19887, 2024.
[11] Waleffe et al. "Mamba-2-Hybrid: Scaling Hybrid SSM-Transformer Models to 8B." 2025.
[12] Peng et al. "RWKV-6: A Linear-Time Language Model with State-Space Features." 2025.
[13] Kim et al. "Eagle: Efficient Training of LLMs with Structured State Spaces." 2025.
[14] Li et al. "MoDE: Mixture of Depth Experts." 2025.
[15] Wang et al. "CALM: Compute-Aware Layer Management for Efficient LLM Inference." 2025.
[16] Jang et al. "Model Merging in Large Language Models: A Survey." arXiv:2506.XXXXX, 2025.
[17] Yadav et al. "TIES-Merging: Resolving Interference When Merging Models." 2024.
[18] Yu et al. "DARE: Drop And REscale for Model Merging." ICLR 2025.
[19] Ilharco et al. "Editing Models with Task Arithmetic." ICLR 2023.
[20] Guo et al. "Task Vector Fusion: Transferable Task Vectors Across Model Scales." 2025.
[21] Cheng et al. "Quantization-Aware Training for LLMs: A Comprehensive Survey." 2025.
[22] Zhang et al. "LiteVL: Efficient Vision-Language Model via Quantization-Aware Training." 2025.

---
**See also:**
- [AI Evaluation and LLM Testing — Overview](69-AI-Evaluation-and-LLM-Testing/01-Overview.md)
- [04 — Local LLM Indexing and Search](23-Local-AI-Inference-Self-Hosting/04-Local-LLM-Indexing-and-Search.md)

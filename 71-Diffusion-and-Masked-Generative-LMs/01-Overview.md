# Diffusion & Masked Generative Language Models

> A comprehensive guide to non-autoregressive text generation: diffusion language models (DLMs), masked generative models (MGMs / masked diffusion), and parallel decoding transformers. Covers the architectural shift away from left-to-right autoregressive (AR) decoding, the leading 2025–2026 systems (LLaDA, Mercury, Gemini Diffusion, SEDD, MDLM), latency/throughput trade-offs, and production guidance.

> **Library cross-references:** Foundational transformer and AR text generation is covered in `02-LLMs/01-Transformer-Architecture.md` and `02-LLMs/02-Model-Families.md`. Inference cost and quantization are in `02-LLMs/04-Quantization.md` and `63-GPU-Kernel-and-Inference-Performance-Engineering/`. Reasoning/time-scaling is in `29-Reasoning-and-Inference-Scaling/`. Speech/audio diffusion (closely related) is in `19-Voice-AI-and-Agents/03-Text-to-Speech-Advances.md`.

---

## 1. Why this matters in 2026

For four years the dominant paradigm for generating text with LLMs has been **autoregressive decoding**: the model emits one token at a time, each conditioned on all previous tokens, left-to-right. This is simple and high-quality but suffers from a structural bottleneck — **latency grows linearly with output length**, and the GPU is starved because each step depends on the previous one (low arithmetic intensity, poor hardware utilization).

In 2025–2026 a competing family matured to production quality:

- **Inception Labs "Mercury"** (Feb 2025): a commercial diffusion-LM API claiming 5–10x faster generation than comparable AR models at similar quality, positioned for code and structured output.
- **LLaDA (Large Language Diffusion with mAsking)** — a Llama-style masked diffusion model from Peking University / StepFun, open-weights, demonstrating competitive capability with reversibility and editability properties.
- **Gemini Diffusion** (Google, 2025): a diffusion-based model integrated into the Gemini family, emphasizing speed for code and long-form generation.
- **SEDD (Score Entropy Discrete Diffusion)** and **MDLM (Masked Diffusion Language Model)**: the leading academic discrete-diffusion formulations that most production systems are built on.

The driving forces:

| Force | Effect |
|-------|--------|
| Latency-sensitive apps (agents, autocomplete, code gen, voice) | AR linear latency is unacceptable |
| Inference cost pressure | Parallel decoding improves GPU utilization (see `41-AI-Cost-Optimization-and-Enterprise-ROI/`) |
| Agentic workloads (`03-Agents/`) | Many short generations → throughput dominates |
| Editability / controllability demand | Diffusion naturally supports in-fill and refinement |

---

## 2. The core idea: generate in parallel

Autoregressive generation:

```
x_1 ~ p(x_1 | ∅)
x_2 ~ p(x_2 | x_1)
x_3 ~ p(x_3 | x_1, x_2)
...
x_T ~ p(x_T | x_1..x_{T-1})
```

Each token is sampled **sequentially**. The joint is factorized as a chain.

Diffusion / masked generation reverses a **corruption process**. Start from a fully masked sequence and iteratively *unmask* tokens until a clean sentence remains:

```
Init:    [M][M][M][M][M][M]        (all masked)
Step 1:  [M] the [M] [M] [M] [M]
Step 2:  [M] the cat [M] on [M]
Step 3:  The the cat sat on [M]
Final:   The cat sat on the mat
```

At every step the model **sees the whole sequence** (masked) and predicts a denoising distribution over *all* currently-masked positions simultaneously. Updates can be applied in parallel → sub-linear wall-clock time.

---

## 3. Two main formulations

### 3.1 Masked Generative Models (MGM / masked diffusion)

Also called **Masked Diffusion Language Models (MDLM)**. A fixed forward noising process randomly replaces real tokens with a `[MASK]` token. Training teaches the model to recover the originals. Sampling is an **iterative unmasking** loop (typically 8–64 steps). This is the formulation behind LLaDA and "Mercury."

### 3.2 Continuous / score-based Diffusion Language Models

Tokens are embedded into a continuous space and a **score model** learns to reverse a Gaussian noising process (like DDPM/DDIM in images). The discrete text is recovered by rounding/argmax at the end. **SEDD** is the canonical discrete-score version; **Diffusion-LM** (Gao et al., 2022, Stanford) is the original continuous formulation enabling controllable generation via the latent.

| Property | AR Transformer | Masked Diffusion (MGM) | Continuous Diffusion (SEDD) |
|----------|----------------|------------------------|-----------------------------|
| Generation order | Strictly left→right | Parallel, unordered | Parallel, unordered |
| Latency vs length | Linear | Sub-linear (steps independent of T) | Sub-linear |
| Editability (in-fill) | Poor (must regenerate) | Native | Native |
| Reversibility | No | Yes (can re-noise) | Yes |
| Sampling steps | 1 per token (T steps) | ~8–64 steps | ~8–100 steps |
| Maturity (2026) | Dominant | Production (LLaDA, Mercury) | Research → early prod |

---

## 4. Terminology map

| Term | Meaning |
|------|---------|
| **AR** | Autoregressive — one token at a time, left-to-right |
| **NAR / non-autoregressive** | Any model that emits tokens in parallel |
| **DLM** | Diffusion Language Model |
| **MGM / masked diffusion** | Diffusion via random masking + iterative unmasking |
| **MDLM** | Masked Diffusion Language Model (paper name) |
| **SEDD** | Score Entropy Discrete Diffusion |
| **Remasking** | Re-masking an already-unmasked token if confidence is low (used in LLaDA) |
| **Semantic reward / guidance** | Classifier-free or reward guidance applied during sampling (à la classifier-free guidance in images) |
| **Self-distillation / CONCAT** | Training trick to align few-step sampling with many-step teacher |
| **Block-wise / block diffusion** | Diffuse and denoise contiguous blocks, often AR across blocks but parallel within — a hybrid |

---

## 5. Capability status (mid-2026)

| Capability | AR (GPT-class) | Diffusion/Masked LM | Notes |
|-----------|----------------|---------------------|-------|
| Short-form chat | Excellent | Excellent | Parity reached |
| Code generation | Excellent | Strong (Mercury targets this) | Parallel helpful for boilerplate |
| Long-form coherence | Excellent | Good–Strong | More steps needed for very long |
| Math/reasoning | Excellent (+inference scaling, `29-Reasoning-and-Inference-Scaling/`) | Emerging | AR still leads on hard reasoning |
| Instruction following | Excellent | Strong | Comparable at 7–70B scale |
| Controllability / constraints | Prompt-only | Strong (latent guidance, in-fill) | Diffusion advantage |
| Reversibility / editing | None | Native | Unique diffusion property |

**Bottom line:** Diffusion/masked LMs are **production-ready for latency-sensitive, high-throughput, and controllable generation** (code, agents, autocomplete, structured output, editing). AR remains preferred for the hardest multi-step reasoning where chain-of-thought scaling is essential.

---

## 6. Architecture at a glance (masked diffusion)

```
            Forward (training, random mask)
x_clean ──random [MASK] with prob t──► x_corrupted(t)

            Reverse (sampling, iterative unmask)
x_corrupted(t) ──Transformer(θ)──► p(unmask | x_corrupted)
                         │
                         ▼
            unmask top-k most confident positions
                         │
                         ▼
                  x_corrupted(t-1)  ──►  ...  ──►  x_clean
```

The "denoiser" is a standard Transformer encoder-style (bidirectional attention) backbone — **no causal mask**. Bidirectional context is exactly why in-fill and editing work naturally.

---

## 7. When to reach for a diffusion/masked LM

Use diffusion/masked LMs when **any** of these hold:

1. **p95 latency budget < 200 ms** for outputs of 50–500 tokens (autocomplete, agent tool-call generation, voice fillers — see `19-Voice-AI-and-Agents/`).
2. **Very high QPS** with many short generations (cost per token dominates — see `41-AI-Cost-Optimization-and-Enterprise-ROI/`).
3. **Editable / controllable text** is required (templates, constrained formats, iterative refinement).
4. **Structured output** (JSON, SQL, code) where parallel decoding is efficient.

Stick with AR when:

1. Hard multi-step reasoning / planning (need CoT scaling — `29-Reasoning-and-Inference-Scaling/`).
2. Maximum capability ceiling on benchmarks is the priority.
3. You depend on a mature ecosystem (tooling, fine-tuning recipes — `64-Model-Fine-Tuning-and-Post-Training/`).

---

## 8. Relationship to adjacent fields

- **Diffusion in images/audio/video** (`06-Advanced/02-Diffusion-Models.md`, `50-Multimodal-AI/28-AI-Video-Audio-Generation/`): same mathematical family; text uses *discrete* diffusion (masking or score-entropy) because the data is categorical.
- **Non-autoregressive machine translation (NAT)**: the historical precursor (Gu et al., 2018). Modern DLMs solve NAT's multi-modality problem via iterative refinement.
- **Speculative decoding** (`02-LLMs/`, draft-model acceleration): also parallelizes, but still anchored to an AR verifier. Diffusion removes the AR backbone entirely.
- **Parallel decoding / block diffusion**: hybrids that are AR across blocks but parallel within.

---

## 9. Ecosystem snapshot (mid-2026)

| System | Org | Type | Status | License/open |
|--------|-----|------|--------|--------------|
| Mercury | Inception Labs | Masked diffusion | Commercial API + code model | API, some open |
| LLaDA 8B | Peking Univ / StepFun | Masked diffusion | Open weights | Apache-2.0 |
| LLaDA-1.5 / variants | community | Masked diffusion | Fine-tunes | Open |
| Gemini Diffusion | Google | Diffusion | API (Gemini family) | Hosted |
| SEDD | academic (Stanford/UT) | Score-entropy discrete | Research code | Open |
| MDLM | academic (UT Austin) | Masked diffusion | Research code | Open |
| Dream / Diffusion-LM | Stanford | Continuous diffusion | Research | Open |

See `04-Tools-and-Frameworks.md` for repos, APIs, and runnable examples.

---

## 10. Roadmap of this category

- `01-Overview.md` (this file): concepts, terminology, when to use.
- `02-Core-Topics.md`: training objectives, sampling algorithms, guidance, the math of discrete diffusion.
- `03-Technical-Deep-Dive.md`: implementations, remasking, block diffusion, distillation, evaluation, failures.
- `04-Tools-and-Frameworks.md`: repos, APIs, code samples, deployment.
- `05-Future-Outlook.md`: trends, hybrid AR+diffusion, research frontiers.

---

## 11. Quick FAQ

**Q: Are diffusion LMs "real" language models or just a trick?**
A: They are fully-trained generative models with their own training objective (denoising a corrupted sequence). They model the data distribution; they are not post-processing on top of an AR model.

**Q: Do they support streaming output?**
A: Yes — final tokens stabilize early; systems stream as confidence crosses a threshold. True left-to-right streaming is less natural than AR but achievable via "confidence-ordered" emission.

**Q: Can I fine-tune them like LLaMA?**
A: Increasingly yes for masked models (LLaDA fine-tunes resemble SFT; `64-Model-Fine-Tuning-and-Post-Training/`). The objective differs, so standard LoRA recipes need small adaptation (usually just the loss).

**Q: Why not just use speculative decoding?**
A: Speculative decoding still needs an AR draft+verify loop and a target model; it accelerates but does not change the paradigm. Diffusion removes sequential dependency, giving better scaling to very high batch throughput.

---

## 12. References (canonical)

- Austin, Johnson, Ho, Tarlow, van den Berg — *Structured Denoising Diffusion Models in Discrete State-Spaces* (D3PM), 2021.
- Gao, Bai, Xiong, et al. — *Diffusion-LM: Controllable Text Generation with Diffusion Models*, 2022.
- Lou, Meng, Ermon — *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution* (SEDD), 2023.
- Sahoo, Arriola, Schiff, et al. — *Simple and Effective Masked Diffusion Language Models* (MDLM), 2024.
- Nie, Zhu, Kong, et al. — *Large Language Diffusion Models* (LLaDA), 2024/2025.
- Inception Labs — *Mercury: The First Commercial-scale Diffusion LLM*, 2025.
- Google DeepMind — *Gemini Diffusion*, 2025.

---

## 13. Historical context: from NAT to diffusion

The dream of parallel text generation is old:

| Era | Approach | Outcome |
|-----|----------|---------|
| 2018 | Non-autoregressive MT (NAT) | Fast but multi-modality collapse |
| 2020 | Insertion/deletion models | Better but complex |
| 2021 | D3PM (discrete diffusion theory) | Unified framework |
| 2022 | Diffusion-LM (continuous) | Controllable gen proven |
| 2023 | SEDD / MDLM | Tractable discrete reverse |
| 2024 | LLaDA (8B open) | Quality reaches AR range |
| 2025 | Mercury (commercial), Gemini Diffusion | Production reality |

The crucial insight that unlocked quality was **iterative refinement** — NAT failed because it committed in one pass; diffusion succeeds because it drafts, then fixes.

---

## 14. Detailed comparison with speculative decoding

Speculative decoding (a common AR accelerator) also parallelizes, so why prefer diffusion?

| Axis | Speculative decoding | Diffusion / masked LM |
|------|----------------------|------------------------|
| Core model | Still AR (draft + verify) | No AR backbone |
| Parallelism | Draft `k` tokens, verify all | Unmask many positions |
| Latency scaling | ~`k`x at best | Decoupled from length |
| Best case | AR quality preserved | AR-comparable quality |
| Worst case | Verify rejects → fallback | Too few steps → repetition |
| Complexity | Needs draft model + verifier | Single denoiser |
| Reversibility | None | Native (re-noise) |

Use speculative decoding when you must keep an exact AR model and just want a speed bump. Use diffusion when you can adopt a new architecture for fundamentally better latency/throughput and editability.

---

## 15. Adoption decision tree

```
Need to generate text?
├─ Latency budget < 200 ms for 50–500 tokens?
│   └─ YES → consider diffusion/masked LM (category 71)
├─ Very high QPS, many short outputs?
│   └─ YES → diffusion wins on throughput (41-)
├─ Need editable / in-fill text?
│   └─ YES → diffusion is native
├─ Need valid JSON/SQL/code by construction?
│   └─ YES → constrained unmasking (04-Tools)
├─ Hard multi-step reasoning / planning?
│   └─ YES → keep AR, or hybrid block diffusion (29-)
└─ Max capability ceiling is the priority?
    └─ YES → AR (02-LLMs/02-Model-Families.md)
```

---

## 16. Common misconceptions

1. **"Diffusion LMs are just AR with a trick."** — False; they have a distinct training objective and no causal mask.
2. **"They're lower quality."** — False for most tasks at 16–64 steps; true only for the hardest reasoning.
3. **"They can't stream."** — False; confidence-ordered streaming works.
4. **"They need special hardware."** — False; any transformer accelerator works, often better (compute-bound).
5. **"You can't fine-tune them."** — False; LLaDA SFT/LoRA recipes exist (`64-Model-Fine-Tuning-and-Post-Training/`).

---

## 17. Representative benchmark posture (mid-2026)

| Model class | HumanEval (code) | MMLU | GSM8K | Relative latency |
|-------------|------------------|------|-------|------------------|
| AR 7–8B | ~60–70 | ~65–70 | ~70–80 | 1.0x (baseline) |
| Masked LM 8B (32 steps) | ~55–65 | ~62–68 | ~60–72 | 1.5–3x faster |
| Mercury (code) | competitive | n/a (code-focused) | n/a | 5–10x faster |
| AR 70B | ~80+ | ~78+ | ~90+ | slower |

Exact numbers shift monthly; the *pattern* is stable: diffusion trades a few quality points for large speed gains, and is at parity on many tasks.

---

## 18. Summary

Diffusion and masked generative language models are the most important **architectural** alternative to autoregressive decoding to reach production in 2025–2026. Their value is not "smarter text" but **faster, cheaper, editable text** — which is exactly what latency-bound, high-volume, agentic deployments need. This category documents the math, the systems, the tooling, and the deployment playbook so a practitioner can decide when diffusion beats AR and how to ship it.

*Next: `02-Core-Topics.md` for training objectives and sampling algorithms.*

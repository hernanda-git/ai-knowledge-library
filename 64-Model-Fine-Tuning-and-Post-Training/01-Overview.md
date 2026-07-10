# Model Fine-Tuning and Post-Training

> Fine-tuning and post-training are the disciplines of adapting a pretrained base model to specific behaviors, domains, and preferences. In 2026, as open-weight models close the gap with frontier APIs, the ability to customize models through supervised fine-tuning (SFT), parameter-efficient methods (LoRA/QLoRA), and preference optimization (RLHF/DPO/GRPO) has become one of the highest-leverage and most in-demand AI engineering skills.

## Table of Contents

- [Why Post-Training Matters in 2026](#why-post-training-matters-in-2026)
- [The Post-Training Pipeline](#the-post-training-pipeline)
- [Pretraining vs Post-Training](#pretraining-vs-post-training)
- [The Fine-Tuning Decision Tree](#the-fine-tuning-decision-tree)
- [Categories of Adaptation](#categories-of-adaptation)
- [When NOT to Fine-Tune](#when-not-to-fine-tune)
- [Data: The Real Bottleneck](#data-the-real-bottleneck)
- [Cost and Compute Reality](#cost-and-compute-reality)
- [Career and Demand Signal](#career-and-demand-signal)
- [Cross-References](#cross-references)

---

## Why Post-Training Matters in 2026

A pretrained base model is a raw next-token predictor. It has absorbed vast knowledge but does not, on its own, follow instructions, refuse harmful requests, adopt a house style, or reason in a chain-of-thought format. Everything that makes a model *useful* as a product happens in **post-training** — the phase after pretraining where the model is aligned, instruction-tuned, and specialized.

Three forces make post-training central in 2026:

1. **Open-weight parity.** High-quality open base models (Llama, Qwen, Mistral, DeepSeek, Gemma families) are now strong enough that a well-executed fine-tune on proprietary data can beat a generic frontier API on a narrow task — at a fraction of the inference cost.
2. **Data moats over model moats.** The competitive advantage has shifted from "who has the biggest model" to "who has the best proprietary training data and the pipeline to use it." Post-training is how that data becomes a model behavior.
3. **Reasoning as a trainable skill.** The 2025–2026 wave of reasoning models (see `29-Reasoning-and-Inference-Scaling/`) showed that reinforcement learning on verifiable rewards (RLVR) can *teach* chain-of-thought. Post-training is no longer just alignment — it is capability creation.

| Era | Dominant lever | Key skill |
|-----|----------------|-----------|
| 2020–2022 | Scale pretraining | Distributed training |
| 2023–2024 | Prompt the API | Prompt engineering, RAG |
| 2025–2026 | **Customize open weights** | **SFT, LoRA/QLoRA, DPO, RLVR** |

This category complements `02-LLMs/` (base model architecture), `29-Reasoning-and-Inference-Scaling/` (RL for reasoning), `51-Synthetic-Data-Generation/` (training data creation), and `04-RAG/` (the main alternative to fine-tuning for knowledge injection).

---

## The Post-Training Pipeline

Modern post-training is a multi-stage pipeline. A frontier-quality chat/reasoning model typically goes through:

```
Base Model (pretrained next-token predictor)
        │
        ▼
┌───────────────────────────┐
│ 1. Supervised Fine-Tuning │  Instruction/response pairs
│    (SFT / Instruction     │  Teaches format & task-following
│     tuning)               │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ 2. Preference Optimization│  Chosen vs rejected pairs
│    (RLHF / DPO / GRPO)    │  Aligns to human/AI preferences
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ 3. Reasoning RL (optional)│  Verifiable rewards (math/code)
│    (RLVR / GRPO)          │  Teaches long chain-of-thought
└───────────────────────────┘
        │
        ▼
   Aligned, specialized model
```

Not every project needs all three stages. A domain-specialization fine-tune may be pure SFT. A safety/tone alignment may add DPO. A reasoning model requires the RL stage. See `03-Technical-Deep-Dive.md` for each stage in depth.

---

## Pretraining vs Post-Training

| Dimension | Pretraining | Post-Training |
|-----------|-------------|---------------|
| Data volume | Trillions of tokens | Thousands–millions of examples |
| Data type | Raw web/code/books | Curated instruction/preference data |
| Compute | Thousands of GPU-months | Hours–days on 1–8 GPUs (with LoRA) |
| Goal | General knowledge | Behavior, format, alignment, specialization |
| Who does it | Frontier labs | **Almost any team** |
| Cost | $10M–$1B+ | $10–$10,000 |

The democratization is stark: pretraining remains the domain of a handful of labs, but **post-training is accessible to individual engineers** thanks to parameter-efficient methods and consumer/cloud GPUs.

---

## The Fine-Tuning Decision Tree

Before fine-tuning, always ask whether a cheaper approach solves the problem:

```
Is the problem KNOWLEDGE (facts the model lacks)?
   → Use RAG (04-RAG/), not fine-tuning.
     Fine-tuning is a poor way to inject facts and causes forgetting.

Is the problem FORMAT / STYLE / BEHAVIOR?
   → Fine-tuning is ideal. (e.g., always output JSON, adopt brand voice)

Is the problem a NARROW REPEATED TASK at scale?
   → Fine-tune a small model to replace an expensive API call.
     Big cost win at high volume.

Is the problem REASONING on verifiable tasks?
   → RL post-training (GRPO/RLVR).

Is it a ONE-OFF or LOW-VOLUME task?
   → Just prompt-engineer a frontier model (06-Advanced/).
```

This decision tree is the single most important mental model in the category. Teams waste enormous effort fine-tuning when RAG or prompting would have been faster, cheaper, and more maintainable.

---

## Categories of Adaptation

| Technique | Params updated | Memory | Use case |
|-----------|----------------|--------|----------|
| Full fine-tuning | 100% | Very high | Max quality, large budget |
| LoRA | ~0.1–1% | Low | Most common; near-full quality |
| QLoRA | ~0.1–1% (4-bit base) | Very low | Consumer GPU, single card |
| Prefix/Prompt tuning | <0.1% | Minimal | Lightweight task switching |
| DPO | LoRA or full | Low–High | Preference alignment |
| GRPO/RLVR | LoRA or full | High | Reasoning capability |

Details and code for each are in `03-Technical-Deep-Dive.md` and tooling in `04-Tools-and-Frameworks.md`.

---

## Data: The Real Bottleneck

In post-training, **data quality dominates data quantity**. Landmark results (e.g., the "LIMA: Less Is More for Alignment" line of work) showed that ~1,000 carefully curated examples can outperform tens of thousands of noisy ones for SFT.

Principles:

- **Diversity over volume** — cover the distribution of real user inputs.
- **Consistency** — inconsistent labels teach the model to be inconsistent.
- **Format fidelity** — every example should model the exact output you want.
- **Deduplication and decontamination** — remove near-duplicates and any eval-set leakage.
- **Synthetic augmentation** — use a strong teacher model to generate/refine data (see `51-Synthetic-Data-Generation/`), then human-verify.

A typical SFT dataset row (chat template):

```json
{
  "messages": [
    {"role": "system", "content": "You are a support agent for Acme Corp."},
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "Go to Settings → Security → Reset Password. You'll receive an email link valid for 15 minutes."}
  ]
}
```

A DPO preference row:

```json
{
  "prompt": "Explain quantum entanglement to a 10-year-old.",
  "chosen": "Imagine two magic coins that always land the same way...",
  "rejected": "Quantum entanglement is a phenomenon in which the quantum states of two particles..."
}
```

---

## Cost and Compute Reality

Approximate 2026 costs for fine-tuning a 7–8B parameter model:

| Method | Hardware | Time | Est. cost |
|--------|----------|------|-----------|
| QLoRA | 1× RTX 4090 (24GB) | 2–6 hrs | ~$5–15 (or free locally) |
| LoRA | 1× A100 (80GB) | 1–3 hrs | ~$5–20 |
| Full FT (7B) | 4–8× A100 | 4–12 hrs | ~$100–500 |
| DPO (LoRA) | 1× A100 | 2–4 hrs | ~$10–30 |

The upshot: **a single engineer with one cloud GPU can produce a production-grade specialized model in an afternoon.** This is the core reason post-training skills are in such high demand — the barrier is now expertise, not compute.

---

## Career and Demand Signal

Fine-tuning / post-training engineer roles are among the fastest-growing AI job categories in 2026 (see `34-AI-Workforce-Transformation/`). Employers want engineers who can:

- Curate and clean instruction/preference datasets.
- Run LoRA/QLoRA fine-tunes and evaluate them rigorously (`52-AI-Hallucination-Detection-and-Mitigation/`).
- Apply DPO/GRPO for alignment and reasoning.
- Decide correctly between fine-tuning, RAG, and prompting.
- Deploy and serve fine-tuned adapters efficiently (`63-GPU-Kernel-and-Inference-Performance-Engineering/`).

The premium goes to those who combine data judgment with training-loop competence — not just running a script, but knowing *why* a fine-tune regressed and how to fix it.

---

## Cross-References

- `02-LLMs/` — base model architectures being fine-tuned
- `29-Reasoning-and-Inference-Scaling/` — RL for reasoning (GRPO/RLVR)
- `51-Synthetic-Data-Generation/` — creating training data
- `04-RAG/` — the primary alternative for knowledge injection
- `06-Advanced/` — prompt engineering as the no-training baseline
- `30-Small-Language-Models/` — fine-tuning small models to replace big APIs
- `63-GPU-Kernel-and-Inference-Performance-Engineering/` — serving fine-tuned models
- `52-AI-Hallucination-Detection-and-Mitigation/` — evaluating fine-tune quality

Continue to `02-Core-Topics.md` for the conceptual toolkit, then `03-Technical-Deep-Dive.md` for algorithms and code.

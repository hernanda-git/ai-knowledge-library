# Core Topics in Fine-Tuning and Post-Training

> This document surveys the conceptual toolkit of post-training: the training objectives, the parameter-efficient methods, the preference-optimization family, and the evaluation practices that determine whether a fine-tune succeeds or silently regresses.

## Table of Contents

- [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
- [Parameter-Efficient Fine-Tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
- [LoRA and QLoRA](#lora-and-qlora)
- [The Preference Optimization Family](#the-preference-optimization-family)
- [RLHF vs DPO vs GRPO](#rlhf-vs-dpo-vs-grpo)
- [Catastrophic Forgetting](#catastrophic-forgetting)
- [Chat Templates and Tokenization](#chat-templates-and-tokenization)
- [Hyperparameters That Matter](#hyperparameters-that-matter)
- [Evaluation of Fine-Tunes](#evaluation-of-fine-tunes)
- [Common Failure Modes](#common-failure-modes)
- [Cross-References](#cross-references)

---

## Supervised Fine-Tuning (SFT)

SFT is the foundational post-training step. You take instruction/response pairs and train the model with the standard **causal language modeling loss** — but crucially, you usually **mask the loss on the prompt tokens** so the model only learns to predict the *response*.

Objective (next-token cross-entropy over response tokens):

```
L_SFT = - Σ_t  log P(y_t | y_<t, x)     for t in response span only
```

Key concepts:

- **Completion-only loss masking** — do not train the model to generate the user's question; only the assistant's answer.
- **Packing** — concatenate multiple short examples into one sequence to use context efficiently (with attention boundaries to prevent cross-contamination).
- **Epochs** — SFT usually needs only 1–3 epochs; more causes overfitting and forgetting.

SFT teaches *format and task-following*. It cannot, by itself, teach the model to prefer good answers over merely plausible ones — that is what preference optimization adds.

---

## Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning updates all weights and requires memory for weights + gradients + optimizer states (roughly 4× the model size in memory for Adam). For a 7B model in fp16 that is ~56GB+ just for optimizer states — infeasible on a single consumer card.

**PEFT** methods freeze the base model and train a tiny set of new parameters:

| Method | Idea |
|--------|------|
| LoRA | Inject low-rank update matrices into linear layers |
| QLoRA | LoRA on top of a 4-bit quantized frozen base |
| Adapters | Insert small bottleneck MLP layers |
| Prefix tuning | Learn virtual tokens prepended to the key/value cache |
| (IA)³ | Learn per-channel scaling vectors |

LoRA/QLoRA dominate in practice because they retain ~95–100% of full-FT quality at ~0.1–1% of the trainable parameters.

---

## LoRA and QLoRA

**LoRA (Low-Rank Adaptation)** is built on the hypothesis that the weight *update* during fine-tuning has low intrinsic rank. Instead of learning a full update ΔW (d×d), you learn two small matrices A (d×r) and B (r×d) with rank r ≪ d:

```
W' = W + ΔW = W + B·A          (W frozen; only A, B trained)
scaled: W' = W + (α / r) · B·A
```

- `r` (rank): typically 8–64. Higher = more capacity, more params.
- `α` (alpha): scaling factor; common heuristic α = 2r.
- `target_modules`: which layers get adapters (q_proj, k_proj, v_proj, o_proj, and often the MLP projections).

**QLoRA** adds three tricks to run LoRA on a single GPU:

1. **4-bit NormalFloat (NF4)** quantization of the frozen base weights.
2. **Double quantization** — quantize the quantization constants too.
3. **Paged optimizers** — offload optimizer state to CPU to survive memory spikes.

Result: a 7B model fine-tunable on a 24GB consumer GPU; a 70B model on a single 48–80GB card.

At inference, LoRA adapters can be **merged** back into the base weights (zero overhead) or **served dynamically** so one base model hosts many adapters (multi-tenant serving — see `04-Tools-and-Frameworks.md`).

---

## The Preference Optimization Family

SFT produces a model that follows instructions; preference optimization produces a model that follows instructions *the way humans prefer*. The training signal is **comparative**: given a prompt, a "chosen" (preferred) response and a "rejected" (dispreferred) one.

The family, roughly in order of complexity:

| Method | Reward model? | On-policy? | Notes |
|--------|---------------|-----------|-------|
| RLHF (PPO) | Yes (trained) | Yes | Original ChatGPT recipe; complex |
| DPO | No | No | Closed-form; stable; popular |
| IPO / KTO | No | No | DPO variants, robustness tweaks |
| GRPO | Reward fn/verifier | Yes | Group-relative; reasoning RL |
| RLVR | Verifier (rules) | Yes | Verifiable rewards (math/code) |

---

## RLHF vs DPO vs GRPO

**RLHF (Reinforcement Learning from Human Feedback)** — the classic three-step recipe:
1. SFT the base model.
2. Train a **reward model** on human preference comparisons.
3. Optimize the policy with **PPO** against the reward model, with a KL penalty to stay close to the SFT model.

Powerful but operationally heavy: four models in memory (policy, reference, reward, value), reward hacking risk, and tricky RL stability.

**DPO (Direct Preference Optimization)** — the 2023 breakthrough that skips the reward model and RL loop entirely. It reframes preference learning as a simple classification loss directly on the policy:

```
L_DPO = -log σ( β [ log π(y_w|x)/π_ref(y_w|x) − log π(y_l|x)/π_ref(y_l|x) ] )
```
where y_w is chosen, y_l is rejected, π_ref is the frozen SFT reference, and β controls deviation. DPO is stable, needs only two models, and is the default preference method for most teams in 2026.

**GRPO (Group Relative Policy Optimization)** — popularized by DeepSeek for reasoning. It removes the value/critic model of PPO by sampling a *group* of responses per prompt and computing advantages *relative to the group mean*:

```
A_i = (r_i − mean(r_group)) / std(r_group)
```

Combined with **verifiable rewards** (RLVR) — e.g., "did the code pass tests?", "is the math answer correct?" — GRPO can teach long chain-of-thought without a learned reward model. See `29-Reasoning-and-Inference-Scaling/02-RL-Training-Methodology.md` for the reasoning-specific treatment.

**Choosing:** DPO for tone/safety/style alignment with preference pairs; GRPO/RLVR when you have an automatic verifier and want to grow a *capability* (reasoning, coding); classic PPO-RLHF only if you specifically need a learned reward model and have the infra.

---

## Catastrophic Forgetting

The central hazard of fine-tuning: teaching the model your task while it **forgets** general abilities. A model over-tuned on JSON support tickets may lose its ability to write prose or do arithmetic.

Mitigations:

- **Low learning rate** and **few epochs**.
- **LoRA instead of full FT** — smaller footprint of change reduces forgetting.
- **Replay / data mixing** — blend a small fraction of general instruction data into your task data.
- **KL regularization** (built into DPO/PPO) — penalize drift from the reference model.
- **Evaluate on held-out general benchmarks**, not just your task metric.

---

## Chat Templates and Tokenization

A subtle but frequent source of broken fine-tunes: **using the wrong chat template**. Each model family (Llama, Qwen, Mistral, ChatML) has a specific format with special tokens delimiting roles. If your training data's template differs from what inference uses, quality collapses.

Rules:
- Always apply the model's official `chat_template` (via the tokenizer) during data prep.
- Ensure special/EOS tokens are present so the model learns to *stop*.
- Verify BOS/EOS aren't double-added.
- Match the exact system-prompt formatting used at serving time.

---

## Hyperparameters That Matter

| Hyperparameter | Typical range | Effect |
|----------------|---------------|--------|
| Learning rate | 1e-5 – 2e-4 (LoRA higher) | Too high → forgetting/instability |
| Epochs | 1–3 | Too many → overfit |
| LoRA rank `r` | 8–64 | Capacity of the adapter |
| LoRA `alpha` | 16–64 | Effective update scale |
| Batch size (effective) | 16–128 | Gradient stability |
| Warmup ratio | 0.03–0.1 | Stabilizes early steps |
| Max seq length | 1k–8k | Must cover your data |
| DPO `beta` | 0.1–0.5 | Lower = more deviation allowed |

---

## Evaluation of Fine-Tunes

You cannot improve what you cannot measure. A rigorous eval harness is non-negotiable:

- **Held-out task test set** — the metric you actually care about.
- **General capability regression suite** — MMLU-style, coding, instruction-following to catch forgetting.
- **LLM-as-judge** — a strong model scores outputs on rubrics (pairwise vs baseline).
- **Human spot-checks** — always eyeball a sample; metrics lie.
- **Safety/refusal evals** — ensure alignment wasn't degraded.

See `52-AI-Hallucination-Detection-and-Mitigation/` for evaluation depth and `05-Enterprise/` for production QA gates.

---

## Common Failure Modes

| Symptom | Likely cause |
|---------|--------------|
| Model never stops generating | Missing EOS in training data |
| Great on task, dumb elsewhere | Catastrophic forgetting (LR too high / too many epochs) |
| Ignores system prompt | Template mismatch |
| Overconfident wrong answers | Trained on knowledge it should get via RAG |
| DPO makes it worse | β too low, or noisy/contradictory preference pairs |
| Loss goes to ~0 fast | Overfitting / data leakage / too small dataset |

---

## Cross-References

- `03-Technical-Deep-Dive.md` — algorithms and code for each method
- `04-Tools-and-Frameworks.md` — libraries and platforms
- `29-Reasoning-and-Inference-Scaling/` — GRPO/RLVR for reasoning
- `51-Synthetic-Data-Generation/` — building preference/SFT data
- `52-AI-Hallucination-Detection-and-Mitigation/` — evaluation
- `30-Small-Language-Models/` — the ideal fine-tuning targets

Continue to `03-Technical-Deep-Dive.md`.

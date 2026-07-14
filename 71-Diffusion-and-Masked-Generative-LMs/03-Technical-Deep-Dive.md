# Diffusion & Masked Generative LMs — Technical Deep Dive

> Implementation reality: runnable code for masked diffusion sampling, remasking, distillation, block diffusion, serving under load, evaluation methodology, and known failure analysis. Companion to `02-Core-Topics.md`.

---

## 1. Minimal masked-diffusion sampler (PyTorch)

A self-contained confidence-based unmasking loop over a bidirectional transformer `model` that returns logits for the *clean* tokens at every position.

```python
import torch, torch.nn.functional as F

MASK_ID   = 126336          # example [MASK] token id (LLaDA vocab)
EOS_ID    = 126336 - 1      # example EOS/TYPE id
PROMPT_LEN = 0              # unmasking happens on the "response" span

@torch.no_grad()
def diffusion_sample(model, prompt_ids, resp_len, steps=32, remask=True,
                     cfg_scale=0.0, gen_only=True):
    """
    model:  bidirectional transformer,.forward(input_ids, t) -> logits over vocab
    prompt_ids: (1, P) instruction tokens
    resp_len:   number of response positions to generate
    """
    device = prompt_ids.device
    mask = torch.full((1, resp_len), MASK_ID, dtype=torch.long, device=device)
    # pad-style concat so the model sees prompt + masked response
    x = torch.cat([prompt_ids, mask], dim=1) if gen_only else mask
    P = prompt_ids.shape[1] if gen_only else 0

    t_schedule = torch.linspace(1.0, 0.0, steps + 1)[:-1]  # 1 -> 0

    for step_idx, t in enumerate(t_schedule):
        is_last = (step_idx == steps - 1)
        # timestep as a scalar embedding (broadcast)
        logits = model(x, t.expand(1)).logits if hasattr(model(x, t), 'logits') \
                 else model(x, t)

        # restrict to response span
        resp_logits = logits[:, P:] if gen_only else logits
        probs = F.softmax(resp_logits.float(), dim=-1)
        pred = probs.argmax(-1)                       # predicted clean token
        conf = probs.max(-1).values                   # confidence per position

        # only consider currently-masked positions
        cur_mask = (x[:, P:] == MASK_ID) if gen_only else (x == MASK_ID)
        if cur_mask.sum() == 0:
            break

        # how many to unmask this step (spread remaining over remaining steps)
        k = max(1, int(torch.ceil(cur_mask.sum().float() /
                                  (steps - step_idx)).item()))
        conf_masked = conf.masked_fill(~cur_mask, -1e9)
        topk = conf_masked.topk(min(k, int(cur_mask.sum())), dim=-1)
        out = x.clone()
        new_resp = out[:, P:] if gen_only else out
        new_resp[0, topk.indices] = pred[0, topk.indices]
        out[:, P:] = new_resp

        # remasking: re-mask tokens whose confidence is low (not last step)
        if remask and not is_last:
            # recompute confidence of newly placed tokens
            new_conf = conf[0, topk.indices]
            # re-mask the lowest-fraction (e.g., 5%) of placed tokens
            re_n = max(0, int(0.05 * topk.indices.numel()))
            if re_n > 0:
                order = new_conf.argsort()[:re_n]
                remask_pos = topk.indices[order]
                new_resp[0, remask_pos] = MASK_ID
                out[:, P:] = new_resp

        x = out
    return x[:, P:] if gen_only else x
```

Notes:
- `model(x, t)` must accept the scalar timestep `t` (added as an embedding, or as a prepended special token like LLaDA's `<task>`/timestep token).
- The backbone must be **bidirectional** (no causal mask) or quality collapses.

---

## 2. Training loop sketch (MDLM-style)

```python
def mdlm_loss(model, x0, mask_id, t_sampler, vocab_size):
    # x0: (B, T) clean token ids
    B, T = x0.shape
    t = t_sampler(B)                              # uniform in (0,1]
    # mask each token independently with prob t
    mask_prob = t.unsqueeze(1)                    # (B,1)
    do_mask = torch.rand(B, T) < mask_prob
    xt = x0.masked_fill(do_mask, mask_id)
    logits = model(xt, t)                         # predict clean x0 everywhere
    # loss only on masked positions, weighted 1/t
    loss = F.cross_entropy(logits.view(-1, vocab_size),
                           x0.view(-1), reduction='none')
    loss = loss.view(B, T)
    w = (1.0 / t.unsqueeze(1).clamp_min(1e-3))
    loss = loss * do_mask.float() * w
    return loss.sum() / do_mask.float().sum().clamp_min(1)
```

Key details:
- Sample `t ~ U(0,1)` (or `Beta`). Avoid `t=0` (no masking → trivial).
- `1/t` weighting: early (noisy) steps are hard and need signal.
- Only masked positions contribute to the loss (we don't ask the model to "predict" tokens it can already see verbatim, though some recipes do).
- Add remasking augmentation: with small prob, re-mask a model-predicted token during training so the model learns correction.

---

## 3. Distillation: few-step from many-step teacher

Production needs 16–32 steps, not 1024. Train a **student** to match the teacher's output distribution at low step count.

```
L_distill = E[ CE( student_S(x),  x_0 ) ]  +  λ · KL( student_S(x) || teacher_S'(x) )
```

- `student_S`: student run with `S` steps.
- `teacher_S'`: teacher run with many steps (frozen), used as a soft target.
- **CONCAT** objective (LLaDA): concatenate a few-step and many-step trajectory and train jointly.

Result: LLaDA-8B reaches AR-comparable quality at 32 steps; Mercury reports 5–10x speed at 1–2 steps-equivalent throughput.

---

## 4. Block diffusion (hybrid AR + parallel)

Block diffusion splits the sequence into blocks `B_1..B_K`. Within a block, tokens are diffused/unmasked **in parallel**; across blocks, generation is **autoregressive** (block `i` conditions on completed blocks `< i`). This:
- Keeps long-range causality for reasoning (`29-Reasoning-and-Inference-Scaling/`),
- Parallelizes the expensive local generation,
- Reduces global step count.

```python
for i in range(num_blocks):
    block_len = block_sizes[i]
    # diffuse this block for S_i steps, conditioning on prior blocks
    block = sample_block(model, prior_blocks, block_len, steps=S_i)
    prior_blocks.append(block)
```

Best of both worlds; the dominant production pattern for long outputs in 2026.

---

## 5. Serving under load

Diffusion LMs change the serving math (see `41-AI-Cost-Optimization-and-Enterprise-ROI/`, `63-GPU-Kernel-and-Inference-Performance-Engineering/`):

| Dimension | AR serving | Diffusion serving |
|-----------|-----------|-------------------|
| Batching | continuous batching of token streams | **batch by step** (all positions per step) |
| KV-cache | essential, grows with seq | none (re-encode small fixed seq) |
| Throughput knob | tokens/sec/model | steps × batch positions |
| Latency knob | token count | **step budget** (decoupled from length) |
| Memory | KV-cache memory | activation memory for full-seq attention |

**Scheduling tip:** Because a single request does `S` full-sequence passes, prefer **large micro-batches of positions** and moderate `S` (16–32). Use a step-budget SLA: `S = f(latency_budget)`.

---

## 6. Constrained / structured generation

Diffusion makes **constrained decoding** natural via masking-aware guidance:

```python
def constrained_unmask(probs, masked_pos, schema_valid_fn):
    # bias unmasking toward tokens that keep the partial output schema-valid
    for pos in masked_pos:
        for tok in candidate(pos):
            if not schema_valid_fn(partial_output_with(tok)):
                probs[pos, tok] = -1e9     # forbid invalid token
    return probs
```

This yields **JSON / SQL / code that is valid by construction** without an AR grammar parser (no need for a separate grammar engine like `jsonformer`/`outlines` for AR). Excellent for agent tool-calls (`03-Agents/05-Tool-Implementations.md`).

---

## 7. Evaluation methodology

Diffusion output is **sampler-stochastic**; evaluate like a generative model, not a deterministic one.

```python
def evaluate(model, dataset, n_samples=8, steps=32):
    rows = []
    for ex in dataset:
        lat = []; quals = []
        for _ in range(n_samples):
            t0 = time.time()
            out = diffusion_sample(model, ex.prompt, resp_len=ex.len, steps=steps)
            lat.append(time.time() - t0)
            quals.append(quality_metric(out, ex.ref))   # accuracy / pass@1
        rows.append({
            "speedup_vs_AR": ar_latency(ex) / mean(lat),
            "quality_mean": mean(quals),
            "quality_std": std(quals),      # diffusion variance matters
            "p95_latency": quantile(lat, .95),
        })
    return aggregate(rows)
```

Report **both** quality and speed; a 2% quality drop at 6x speed is often a win for agents. Benchmarks: HumanEval/MBPP (code), MMLU/GSM8K (knowledge/reason), plus a **latency/throughput** column. See harness in `69-AI-Evaluation-and-LLM-Testing/`.

---

## 8. Failure analysis (deep)

### 8.1 Repetition
**Symptom:** "the the cat cat". **Cause:** multi-modality not resolved in too-few steps, or remasking disabled. **Fix:** enable remasking, raise `S` to ≥16, add CFG.

### 8.2 Length drift
**Symptom:** output too short/long. **Cause:** no explicit length signal. **Fix:** prepend a length token; train the model to emit an `<eos>`/padding pattern; for block diffusion, fix block sizes.

### 8.3 Long-form degradation
**Symptom:** coherence drops past ~512 tokens. **Cause:** `O(T²)` attention + low `S`. **Fix:** block diffusion; sparse/linear attention; raise `S` for long spans; chunk-and-stitch with overlap.

### 8.4 Reasoning gap
**Symptom:** hard math/logic worse than AR+CoT. **Cause:** diffusion does not natively "think step by step" in a causal chain. **Fix:** hybrid — AR reasoning prefix + diffusion draft, or block diffusion with AR blocks; see `29-Reasoning-and-Inference-Scaling/`.

### 8.5 Format/locale drift
**Symptom:** mixed languages or wrong casing. **Cause:** guidance too weak. **Fix:** CFG toward instruction embedding; constrained unmasking.

---

## 9. Quantization compatibility

Diffusion LMs quantize like any transformer (`02-LLMs/04-Quantization.md`): INT8/INT4 (AWQ, GPTQ) work; the bidirectional attention and `t` embedding are unaffected. Because serving is compute-bound (not memory-bound like AR), **weights-only quantization** still yields speedups via faster matmuls. KV-cache quantization is irrelevant (no KV-cache).

---

## 10. Fine-tuning a masked LM (practical)

```python
# LoRA on a masked LM: swap the loss, add timestep embedding, keep attention bidirectional.
from peft import LoraConfig, get_peft_model
model = get_peft_model(base_bidirectional_lm, LoraConfig(r=16, lora_alpha=32))
# training_step = mdlm_loss(model, x0, MASK_ID, t_sampler, V)   # see §2
# optimizer = AdamW(...)  -- standard
```

- Do **not** add a causal mask.
- Keep `t` conditioning.
- Instruction SFT: build `(prompt, masked_response)` pairs; the model learns to denoise the response given the prompt.
- DPO: define preference pairs on final `x_0`, backprop through a short sampled trajectory (REINFORCE or straight-through estimator).

---

## 11. Hybrid AR + diffusion patterns (2026 production)

| Pattern | When | Benefit |
|---------|------|---------|
| AR reasoning → diffusion draft | agents, code | fast bulk, correct logic |
| Block diffusion (AR blocks) | long-form | causality + speed |
| Diffusion for structured, AR for prose | docs | valid JSON + fluent text |
| Speculative w/ diffusion draft |过渡 | reuse AR verifier |

The field is converging on **hybrids**, not a winner-take-all AR→diffusion switch.

---

## 12. Cost model worked example

Assume A100, `T=256` tokens, AR ~ 60 tok/s memory-bound, diffusion 32 steps × full-seq matmul.

```
AR time        ≈ 256 / 60      ≈ 4.3 s  (p50)
Diffusion time ≈ 32 × 90 ms    ≈ 2.9 s  (with batching, high util)
Speedup        ≈ 1.5x
For T=1024:
AR time        ≈ 17 s
Diffusion time ≈ 32 × 180 ms   ≈ 5.8 s  (attention grows)
Speedup        ≈ 3x
```

At higher batch the diffusion gap widens because compute utilization stays high. See `41-AI-Cost-Optimization-and-Enterprise-ROI/`.

---

## 13. Observability & debugging

Treat the sampler as a pipeline (`20-Agent-Infrastructure-and-Observability/`):
- Log **steps used**, **remask count**, **final confidence distribution**.
- Alert on **high remask ratio** (model unsure → quality risk) and **length mismatch rate**.
- Track **per-step confidence** to detect drift / distribution shift.
- Keep a **golden set** of constrained outputs (JSON/SQL) to regression-test after any model update.

---

## 14. Security considerations

- **Prompt injection** (`18-Agent-Security-and-Trust/`): diffusion agents are as vulnerable as AR; constrain unmasking and validate outputs against a schema before acting.
- **Jailbreak via partial mask**: an attacker could seed the mask with malicious tokens; sanitize the *initial* masked sequence, not just the prompt.
- **Extraction**: bidirectional models may leak training data into unmasked tokens; apply membership/extraction probes (see `52-AI-Hallucination-Detection-and-Mitigation/`).

---

## 15. Summary

In production, diffusion/masked LMs are served **step-batched** (no KV-cache), fine-tuned with a denoising loss + LoRA, distilled to 16–32 steps, and often **hybridized** with AR for reasoning and long-form. The dominant failure modes (repetition, length drift, long-form coherence, reasoning gap) all have known mitigations. With constraints/guidance they produce valid structured output natively — a strong fit for agents (`03-Agents/`) and high-throughput services.

*Next: `04-Tools-and-Frameworks.md` for concrete repos, APIs, and deployment snippets.*

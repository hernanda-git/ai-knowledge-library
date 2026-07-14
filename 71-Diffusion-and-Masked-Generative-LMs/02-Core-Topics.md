# Diffusion & Masked Generative LMs — Core Topics

> Mathematical and algorithmic foundations: the discrete diffusion framework (D3PM), masked diffusion (MDLM) training objective, score-entropy discrete diffusion (SEDD), iterative sampling/unmasking algorithms, classifier-free and reward guidance, and the multi-modality problem. Companion to `01-Overview.md`.

---

## 1. The discrete diffusion framework (D3PM)

The unifying theory for **discrete-state** diffusion (Austin et al., 2021) generalizes Gaussian diffusion from continuous to categorical data. A forward **markov chain** gradually corrupts a clean token sequence `x_0` into noise `x_T` over `T` steps using a transition matrix `Q_t`:

```
q(x_t | x_{t-1}) = Cat(x_t ; Q_t x_{t-1})
```

where each column of `Q_t` is a categorical distribution over the next state given the current token. For text we need `K = vocabulary size + 1` (the +1 is `[MASK]`).

Common corruption choices:
- **Absorbing state (mask)**: `Q_t` replaces a token with `[MASK]` with probability `1 - α_t`; otherwise keeps it. This is the **masked diffusion** instantiation.
- **Uniform**: blend toward a uniform categorical over the vocabulary (used in some SEDD variants and continuous-flow discrete diffusion).
- **Token-level random replace**: replace with a random vocab token (no separate mask token).

The backward (reverse) process is learned:

```
p_θ(x_{t-1} | x_t) = Cat(x_{t-1} ; ... )
```

trained with a **variational bound** (ELBO) on `log p_θ(x_0)`.

---

## 2. Masked Diffusion Language Models (MDLM) objective

MDLM (Sahoo et al., 2024) uses the **absorbing-mask** forward process. At noise level `t ∈ [0,1]`, each clean token is independently replaced by `[MASK]` with probability `t` (so `α_t = 1 - t` for the keep probability; more precisely `α_t = 1 - σ_t` under a schedule `σ`).

The training loss (simplified, the "σ-ELBO" / rate-matching form) is:

```
L_simple = E_{t ~ U(0,1), x_0, x_t}
            [  (1/t) * Σ_i  [ x_{t,i} == MASK ]
               * CE( model(x_t, t)_i , x_{0,i} ) ]
```

Intuition: at a random timestep `t`, mask some tokens, ask the bidirectional Transformer to predict the **original** (clean) token at every masked position, weight by `1/t` (harder/earlier steps matter more), and average over masked positions. The model input is the **masked sequence + the scalar timestep `t`** (often added as an embedding or as a special token).

Key properties:
- The backbone is a **non-causal (bidirectional) Transformer** — no causal attention mask.
- The model conditions on `t` so it knows *how noisy* the input is.
- Sampling never needs the full ELBO; a simple heuristic unmasking rule works well.

---

## 3. Score-Entropy Discrete Diffusion (SEDD)

SEDD (Lou, Meng, Ermon, 2023) takes a **score-matching** view instead of an ELBO. It learns the *ratios* of the data distribution (scores) and defines a reverse process via a variant of **probability-flow ODE / SDE** on the simplex of categorical distributions.

The score used is:

```
s_θ(x_t, t)_i = ∇_{e_i} log p_t(x_t)
```

and the loss is a **score-entropy / cross-entropy** matching objective that avoids computing intractable normalizers:

```
L_SEDD = E [ Σ_i CE_target( p_t(x_{t-1}|x_t) ,  model softmax ) ]
```

where the "target" distribution `p_t(x_{t-1}|x_t)` is the true one-step reverse posterior, computable in closed form under the discrete forward process (this is the advantage of the discrete case — the reverse posterior is tractable, unlike continuous diffusion where we approximate it).

SEDD typically uses a **uniform** corruption (blend to uniform) rather than an absorbing mask, and samples via an **ODE solver** (e.g., Euler/Heun) over `t` from 1 → 0, with a final argmax to recover discrete tokens.

| | MDLM | SEDD |
|---|------|------|
| Forward corruption | Absorbing `[MASK]` | Uniform blend |
| Sampling | Iterative unmask (heuristic) | ODE over t (1→0) |
| Loss | σ-ELBO rate matching | Score entropy / CE |
| Reverse posterior | Approx (heuristic remask) | Exact (tractable) |

---

## 4. The multi-modality problem (and how diffusion solves it)

**Non-autoregressive MT** (Gu et al., 2018) failed at scale because of **multi-modality**: a sentence has many valid translations; a single forward pass collapses the distribution, producing duplicated or dropped words (e.g., "the the cat cat").

Diffusion fixes this via **iterative refinement**: the model does not commit in one shot. Early steps produce a fuzzy/multi-modal draft; later steps disambiguate. Each denoising step re-reads the *whole* sequence and corrects correlations. This is the core reason modern DLMs beat classical NAT.

```
NAT:           x_0 ──single parallel pass──► x_out   (multi-modality collapse)
Diffusion:     x_T ──► x_{T-1} ──► ... ──► x_0        (refinement resolves modes)
```

---

## 5. Sampling / unmasking algorithms

### 5.1 Confidence-based unmasking (LLaDA-style)

At each step `t → t - Δ`:
1. Run the model on the current masked sequence, get `p(unmask_i)` for each masked position.
2. Unmask the **top-`k` most confident** positions (where `k` scales with the number of steps remaining, e.g., `~ (remaining masks)/steps_left`).
3. **Remasking**: after unmasking, optionally re-mask tokens whose confidence dropped (enables correction).
4. Repeat until no `[MASK]` remains.

Pseudocode:

```
def sample(model, mask_id, steps=32):
    x = full_mask(seq_len)
    for t in linspace(1, 0, steps):          # from noisy to clean
        logits = model(x, t)                 # predict clean tokens everywhere
        probs  = softmax(logits, dim=-1)
        # which positions are still masked?
        masked = (x == mask_id)
        # confidence = prob of predicted token
        conf = probs.argmax(-1).gather(probs)  # or use max prob
        # how many to unmask this step
        k = max(1, round(masked.sum() / steps))
        # pick top-k confident masked positions
        to_unmask = topk(conf[masked], k).indices
        x[masked][to_unmask] = probs[masked][to_unmask].argmax(-1)
    return x
```

### 5.2 ODE sampling (SEDD-style)

Treat `t ∈ [0,1]` as continuous; solve:

```
dx/dt = v_θ(x_t, t)        # velocity field / score-derived drift
```

with an off-the-shelf ODE solver. Convert final continuous simplex to discrete via argmax.

### 5.3 Block diffusion (hybrid)

Diffuse **contiguous blocks** in parallel, but move **left-to-right across blocks** (autoregressive at the block level). This keeps long-range causal structure (good for reasoning) while parallelizing within blocks. Block-diffusion transformers (Arriola et al., 2025) show strong quality with fewer global steps.

---

## 6. Guidance in discrete diffusion

Analogous to **classifier-free guidance (CFG)** in image diffusion, we can steer text diffusion:

```
ε̂ = ε_uncond + w * (ε_cond - ε_uncond)
```

Here `ε` is the predicted clean/score from the model, conditioned vs unconditioned (e.g., with/without a class or instruction embedding). Weight `w` controls strength. Used for:
- **Format adherence** (force JSON/SQL structure),
- **Style / tone** control,
- **Reward guidance**: add a differentiable or RL-derived reward gradient (e.g., from a verifier) into the sampling update — relevant to `29-Reasoning-and-Inference-Scaling/` and `69-AI-Evaluation-and-LLM-Testing/`.

For masking models, a practical guidance is **confidence re-weighting**: bias the unmasking toward tokens that also satisfy a constraint (e.g., valid syntax).

---

## 7. Schedules and step budgets

| Step budget | Quality | Latency | Use case |
|-------------|---------|---------|----------|
| 4–8 steps | Fair | Lowest | Autocomplete, high-QPS |
| 16–32 steps | Good | Low | Agents, code, chat |
| 64+ steps | Strong | Moderate | Long-form, hard tasks |

Unlike AR (always `T` steps for `T` tokens), diffusion **latency is set by the step budget, not token count** — this is the whole point. A 500-token answer in 16 steps can be faster than a 50-token AR answer.

---

## 8. Training recipes that work

- **Timestep sampling**: uniform `t ~ U(0,1)` for MDLM; cosine/sqrt schedules also common.
- **Loss weighting**: `1/t` reweight (or `1/(t+ε)`) stabilizes early-step learning.
- **Self-distillation / CONCAT** (LLaDA): train a few-step student to mimic a many-step teacher so production uses 16–32 steps without quality loss.
- **Length prediction**: discrete diffusion needs to know output length; either predict a length token, or start from a fixed-length mask and let the model learn an "end" / padding pattern.
- **Remasking during training**: randomly re-mask some model-predicted tokens so the model learns to *correct* (critical for final quality).

---

## 9. How it differs from AR fine-tuning

Standard SFT / LoRA (`64-Model-Fine-Tuning-and-Post-Training/`) minimizes next-token CE. For masked LMs:
- The loss is **denoising CE over masked positions** with timestep conditioning.
- A LoRA adapter can be trained largely the same way if you swap the loss and add the `t` embedding.
- Instruction data works: format `(instruction, masked response)`, train to recover the response.
- **RLHF / DPO** also applies: treat the diffusion sampler as the policy and reward final outputs (preference pairs on `x_0`); gradients flow back through the sampling trajectory (straight-through / REINFORCE variants exist).

---

## 10. The (lack of) KV-cache and attention

AR decoding streams tokens through a **KV-cache** to avoid recompute. Diffusion has **no KV-cache** in the same sense — every step re-encodes the (small, fixed-length) masked sequence with full bidirectional attention. The cost per step is `O(T²)` attention over `T` tokens, done `S` times. Total work `≈ S · O(T²)`. For short–medium `T` and small `S`, this beats AR's `T · O(T)` (with KV) on wall-clock because it parallelizes and uses the GPU better. For very long `T`, attention cost grows — motivating block diffusion and sparse attention.

---

## 11. Comparison: generation cost model

Let `T` = tokens, `S` = diffusion steps, `C_attn = O(T²)` per full pass, `C_ar_step = O(T)` with KV-cache.

```
AR wall-clock        ≈ T · step_time(step_time dominated by memory-bound token fetch)
Diffusion wall-clock ≈ S · step_time(step_time dominated by compute-bound matmul, batched)
```

Because diffusion batches all positions, **compute utilization is high** → on modern GPUs the per-step matmul is fast and the *batch* of positions amortizes overhead. AR is **memory-bandwidth bound** (one token at a time). This is why diffusion wins on throughput and latency for moderate `T`.

---

## 12. Failure modes to know

| Failure | Cause | Mitigation |
|---------|-------|-----------|
| Repeated tokens | insufficient steps / weak remasking | more steps, enable remasking |
| Dropped concepts | multi-modality not resolved | more steps, CFG |
| Wrong length | length not predicted | length token / padding pattern |
| Degraded long-form | attention O(T²) + few steps | block diffusion, more steps |
| Format errors | no constraint guidance | CFG / constrained unmasking |

---

## 13. Relationship to evaluation

Because outputs come from a stochastic sampler, **evaluation must be averaged over multiple samples** and measure both quality and **speed/cost** (not just accuracy). Benchmarks:
- **Quality**: standard LLM evals (MMLU, GSM8K, HumanEval) — see `69-AI-Evaluation-and-LLM-Testing/`.
- **Speed**: tokens/sec, p50/p95 latency vs length.
- **Editability**: in-fill success rate, partial-regeneration fidelity.
- **Controllability**: constraint-satisfaction rate.

---

## 14. Glossary (deep)

- **Absorbing state**: a special token (`[MASK]`) that, once entered, stays (forward process only adds masks).
- **Rate matching**: training the model to match the per-step transition rate (MDLM loss form).
- **Reverse posterior**: `q(x_{t-1}|x_t, x_0)` — tractable for discrete diffusion.
- **Self-distillation**: student mimics teacher's few-step behaviour.
- **CONCAT**: concatenative distillation objective for low-step sampling.
- **Velocity field**: the ODE drift `v_θ` in score/flow formulations.
- **Remasking**: re-corrupting low-confidence predictions during sampling.

---

## 15. Cross-references

- Transformer backbone & attention: `02-LLMs/01-Transformer-Architecture.md`
- Quantization to shrink the model: `02-LLMs/04-Quantization.md`
- GPU kernels / matmul efficiency: `63-GPU-Kernel-and-Inference-Performance-Engineering/`
- Inference cost math: `41-AI-Cost-Optimization-and-Enterprise-ROI/`
- Reasoning scaling (where AR still leads): `29-Reasoning-and-Inference-Scaling/`
- Evaluation harness: `69-AI-Evaluation-and-LLM-Testing/`
- Audio diffusion (sibling technique): `19-Voice-AI-and-Agents/03-Text-to-Speech-Advances.md`

---

## 17. The ELBO derivation (intuition)

For the discrete diffusion model, maximize `log p_θ(x_0)`. Using the standard VAE-style bound with the forward (fixed) process `q`:

```
log p_θ(x_0) ≥ E_q [ log p_θ(x_0 | x_1) ]
                − KL( q(x_1|x_0) || p_θ(x_1) )
                − Σ_t KL( q(x_t | x_{t+1}, x_0) || p_θ(x_t | x_{t+1}) )
```

The third term is the **denoising matching** term: at each `t`, predict `x_0` (or the reverse posterior) given the noised `x_t`. MDLM simplifies this to predicting `x_0` at masked positions. SEDD rewrites the same bound via scores. Both reduce to "train the model to denoise."

---

## 18. Timestep conditioning implementations

The model must know *how noisy* the input is. Common schemes:

1. **Scalar embedding**: `t` → learned embedding added to token embeddings (like diffusion-time embeddings in images).
2. **Special token**: prepend a `[t]`-like token (LLaDA uses a timestep/type token).
3. **Continuous signal**: concatenate a Fourier-featured `t` to every position.
4. **Noising-level as mask ratio**: since mask ratio = `t`, the model can infer `t` from how many `[MASK]` it sees — some impls omit explicit `t` and rely on this.

Scheme 1+2 are most common; scheme 4 is elegant but less precise at low noise.

---

## 19. Length modeling in detail

Discrete diffusion needs to know the target length `L` before/while generating. Options:

| Method | How | Pros | Cons |
|--------|-----|------|------|
| Length token | prepend a length id | simple, exact | needs length in training data |
| Fixed span | always generate `L` masks | trivial | wastes tokens on short answers |
| EOS diffusion | let model emit `<eos>` among masks | flexible | eos sometimes misplaced |
| AR length head | small AR head predicts `L` first | accurate | adds AR component |
| Block diffusion | block sizes fixed/growing | natural for long | less flexible |

LLaDA-style models typically use a fixed response span + an `<eos>`/type token; production systems add a length predictor.

---

## 20. Reward / verifier-guided sampling (advanced)

Beyond CFG, you can inject a differentiable reward `R(x_0)` into sampling (relevant to `29-Reasoning-and-Inference-Scaling/` and `69-AI-Evaluation-and-LLM-Testing/`):

```
x_{t-1} = sampler(x_t) + η · ∇_{x_t} R( decode(x_t) )
```

- `R` can be a verifier (code compiles? unit tests pass?), a classifier, or an LLM-judge score.
- The gradient steers the trajectory toward higher-reward outputs **during** generation, not just after.
- This is the diffusion analog of **best-of-N / RLHF-on-policy** and is an active 2026 research area.

Caveat: the gradient flows through a discrete decode; use a continuous relaxation (Gumbel-softmax on the logits) for stable gradients.

---

## 21. Sampling temperature and stochasticity

Like AR, diffusion sampling has a temperature-like knob:

- **Greedy unmasking**: always take top-1 confident token → deterministic, but repeats more.
- **Stochastic unmasking**: sample from `p` at masked positions → diverse, needs more steps.
- **Top-k/p unmasking**: restrict the candidate set.
- **Step count** acts like a "patience" knob: more steps → more refinement → higher quality but slower.

A useful production pattern: **low temperature + 24–32 steps** for quality, **greedy + 8 steps** for autocomplete.

---

## 22. Multi-modal discrete diffusion

The same machinery extends to **images/audio tokenized into discrete codes** (VQ-VAE / SoundStream / EnCodec). Then a single masked-diffusion model can denoise *token sequences* of any modality (`50-Multimodal-AI/`). This is the bridge to **unified multi-modal diffusion** discussed in `05-Future-Outlook.md`. The only change is the tokenizer feeding the discrete diffusion backbone.

---

## 23. Relationship to flow matching

Flow matching (Lipman et al.) learns a velocity field `v_θ` to transport noise → data via an ODE. For **continuous** text latents it's a smooth alternative to DDPM-style diffusion; for **discrete** text, masked diffusion is the practical choice. Expect 2026–2027 work unifying flow-matching objectives with discrete masking (smoother sampling, fewer steps).

---

## 24. Worked numeric example

Suppose vocab `V=32000`, sequence `T=128`, steps `S=32`.

- Per step: one full bidirectional forward over 128 positions → `O(128²)=16,384` attention ops + FFN.
- Total: `32 × 16,384 ≈ 524,288` attention-ops-equivalent × FFN.
- AR equivalent: `128` sequential steps, each `O(128)` with KV = `128 × 128 = 16,384` but **memory-bound** (one token fetched at a time) → wall-clock dominated by memory bandwidth, not compute.

On a compute-rich GPU, the diffusion matmuls batch well and finish faster despite more total FLOPs, because utilization is high. On a memory-bound edge chip, AR can win for short `T`. This is why **diffusion shines on GPUs in the cloud** and AR still matters on tiny edge devices (`62-Edge-AI-and-On-Device-Inference/`).

---

## 25. Checklist: is your problem a good diffusion fit?

- [ ] Output length moderate (≤ ~1–2k tokens) OR you use block diffusion
- [ ] Latency/throughput matters more than squeezing the last 1% benchmark point
- [ ] Output is structured or editable
- [ ] You can serve step-batched (no KV-cache reliance)
- [ ] You have a verifier/reward to guide sampling (bonus)

If most are true → diffusion/masked LM is likely the better architecture.

---

## 26. Summary

The core topics are: (1) a **discrete diffusion** forward process (mask or uniform), (2) a **denoising objective** (ELBO for MDLM, score-entropy for SEDD), (3) an **iterative unmasking / ODE sampler** that resolves multi-modality via refinement, and (4) **guidance** for control. The defining advantage is that **latency is decoupled from output length** and generation is natively editable. `03-Technical-Deep-Dive.md` turns this into implementations and deployment reality.

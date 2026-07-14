# World Models — Technical Deep Dive

> July 2026

This file goes under the hood: concrete architectures, training objectives with derivations, the sim-to-real gap, rollout mechanics, and runnable reference code. Pair with [02-Core-Topics.md](./02-Core-Topics.md) and the tooling list in [04-Tools-and-Frameworks.md](./04-Tools-and-Frameworks.md).

Cross-references: [63-GPU-Kernel-and-Inference-Performance-Engineering](../63-GPU-Kernel-and-Inference-Performance-Engineering/), [64-Model-Fine-Tuning-and-Post-Training](../64-Model-Fine-Tuning-and-Post-Training/), [60-Physical-AI-and-Embodied-Intelligence](../60-Physical-AI-and-Embodied-Intelligence/).

---

## 1. Architecture Stack (Reference Implementation)

A production-grade latent world model typically has five learned modules:

```
Observation o ──▶ Encoder Eφ ──▶ embed e
                                  │
                h ──(GRU)──▶ h'   ▼
Action a ──────────────────▶ [concat] ──▶ Prior p(z'|h')
                e' ─────────▶ [concat h'] ──▶ Posterior q(z'|h',e')
                                  │
                (h', z') ──▶ Decoder Dψ ──▶ o' (recon)
                (h', z') ──▶ Reward  Rρ ──▶ r'
                (h', z') ──▶ Discount Γ ──▶ γ'
                (h', z') ──▶ Continue C  ──▶ p(term)
```

Everything is trained jointly with the ELBO in §2.

---

## 2. The Full ELBO (with all heads)

```
L(θ) = E_q [ Σ_t  log p_ψ(o_t | h_t, z_t)
               + log p_ρ(r_t | h_t, z_t)
               + log p_γ(γ_t | h_t, z_t)
               - β · KL( q_θ(z_t|h_t,e_t) ‖ p_θ(z_t|h_t) ) ]
```

- `β ≈ 1.0` (free bits often applied: `max(KL, free_nats)`).
- Reward/value heads make the latent a **control-ready** representation.

### 2.1 Value Learning Inside the Dream

Actor-critic runs on **imagined** trajectories:

```
imagine H steps:
  for k in 1..H:
     a_k ~ π_ψ(s_k)                      # policy
     s_{k+1} ~ p_θ(s_{k+1}|s_k, a_k)     # dynamics
     v_k = V_ξ(s_k)                       # value
  λ-return:  G_k = r_k + γ_k( (1-λ)v_{k+1} + λG_{k+1} )
loss_actor = -Σ log π(a_k|s_k) · (G_k - V_ξ(s_k))
loss_value = Σ (V_ξ(s_k) - G_k)²
```

All gradients flow back into `θ` (the world model) too — so **better imagination improves the model itself**.

---

## 3. DreamerV3 Tricks That Made It Work

DreamerV3 (2023) was the first single agent to learn across 150+ tasks including Minecraft diamond collection. Key innovations:

| Trick | Why it matters |
|-------|----------------|
| **Symlog / two-hot targets** | Handles rewards/states spanning many orders of magnitude (e.g., -1 to +10⁶) |
| **Stochastic discount γ ~ Beta** | Models episode termination natively |
| **Ignore-stochasticity (KL) clipping** | Stabilizes training when posterior collapses |
| **LayerNorm + no batch norm** | Robust across domains |
| **Scaled hyperparams by model size** | Scales like transformers |

```python
# two-hot encoding of a scalar target for regression stability
def two_hot(x, bins=255, lo=-20, hi=20):
    t = torch.linspace(lo, hi, bins)
    w = torch.softmax(-((x - t)**2) / 0.1, -1)   # soft one-hot
    return w
```

---

## 4. Sim-to-Real: The Hardest Problem

A world model trained in simulation fails in reality due to **dynamics mismatch**. Mitigations:

1. **Domain randomization** — randomize friction, mass, lighting in sim so the model learns invariant features.
2. **System identification** — fit the model's parameters to real rollouts online.
3. **Real-world fine-tuning** — a few hundred real steps correct the prior each deployment.
4. **Uncertainty-gated acting** — if ensemble disagreement is high, act conservatively or ask for real data.

| Strategy | Cost | Effectiveness |
|----------|------|---------------|
| Pure sim | Free | Low (domain gap) |
| Domain rand | Free | Medium |
| Online sys-ID | Low | Medium-High |
| Real fine-tune | High | High |

---

## 5. Video World Models — Training Details

Action-conditional video prediction objective (diffusion variant):

```
L_video = E [ ‖ ε - ε_θ( frame_{t+1}, frame_{≤t}, a_t, t ) ‖² ]
```

where `ε ~ N(0,I)` is the noise and `ε_θ` is the denoising network. Conditioning on `a_t` is what makes it a *simulator* rather than a forecaster.

### 5.1 Genie's Latent Action Model

Genie (2024) solves the "no action labels" problem:

```
Stage 1: VQ-VAE tokenizer  → discrete frame tokens
Stage 2: Latent action model Lφ(video) → inferred action a_t (no labels!)
Stage 3: Dynamics model  p(frame_{t+1} | frame_t, a_t)  (video transformer)
```

The latent action model **invents** an action space from video alone, enabling playable worlds from a single image.

---

## 6. Rollout Mechanics & Batching

Imagined rollouts are done in big parallel tensors for GPU efficiency:

```python
def dream_trajectory(model, state, policy, horizon=15, batch=512):
    states, rewards, values = [], [], []
    s = state
    for _ in range(horizon):
        a = policy(s).sample()                 # (batch, A)
        s = model.dynamics(s, a)               # (batch, S)
        r = model.reward(s)                    # (batch,)
        v = model.value(s)                     # (batch,)
        states.append(s); rewards.append(r); values.append(v)
    return torch.stack(states), torch.stack(rewards), torch.stack(values)
```

Batching 512 environments' worth of imagination on one GPU is what makes training fast.

---

## 7. Memory & Compute Budget

| Component | Typical size | Note |
|-----------|--------------|------|
| Encoder | 4–10 M params | CNN or ViT patch |
| RSSM GRU | 1–4 M | hidden 256–512 |
| Decoder | 4–10 M | transpose-CNN |
| Actor/Critic | 1–2 M each | MLPs |
| Total | 10–30 M | Fits on a single consumer GPU |

Video world models are far heavier (hundreds of M to B params) and need A100/H100-class hardware — see [63-GPU-Kernel-and-Inference-Performance-Engineering](../63-GPU-Kernel-and-Inference-Performance-Engineering/).

---

## 8. Debugging a World Model

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Low real return but high dream return | Model exploitation | Ensemble penalty, shorter H |
| Blurry reconstructions | Under-capacity decoder | Bigger decoder / less KL β |
| Collapsed latent (z constant) | KL too strong | Free bits, KL clipping |
| Diverging training | LR too high / no LN | LayerNorm, lower LR |
| Good sim, bad real | Domain gap | Domain rand + real FT |

---

## 9. Reproducing the Loop End-to-End (Sketch)

```bash
# Conceptual pipeline
python train_encoder.py      # reconstruct observations
python train_dynamics.py     # RSSM prior/posterior
python train_imagination.py  # actor-critic in dream
python evaluate_real_env.py  # zero-shot transfer to real env
```

Frameworks that already do this: **Dreamer / dreamerv3** (reference), **Stable-Baselines3** (model-free baseline), **Diffusion Policy** (action-conditional video for robotics), **MineDojo + STEVE** (Minecraft), **NVIDIA Isaac Sim** (physical sim for sys-ID).

## 10. Scaling Laws for World Models

Recent 2025–2026 results suggest world-model quality scales with:

| Axis | Effect |
|------|--------|
| Model params | Better long-horizon coherence |
| Training interactions | Lower sim-to-real gap |
| Imagination horizon H | Better plans, but more drift |
| Ensemble size | Better uncertainty, less exploitation |

Unlike LLMs, world models have a **hard ceiling** at the sim-to-real gap — scaling compute alone does not close it; real correction does.

## 11. Distillation: Student World Models on Edge

For [62-Edge-AI-and-On-Device-Inference](../62-Edge-AI-and-On-Device-Inference/), a large world model is distilled to a tiny RSSM that runs on-device for fast look-ahead:

```python
# distillation: small model mimics large model's imagined returns
loss_distill = F.mse(small.imagine(s), large.imagine(s).detach())
```

This enables robots/agents to plan locally without cloud round-trips.

## 12. World Models as Evaluators

A trained world model can *self-evaluate* candidate policies by simulating them:

```python
def evaluate_policy(model, policy, n_rollouts=32):
    returns = []
    for _ in range(n_rollouts):
        s = model.reset(); ret = 0.0
        for _ in range(model.horizon):
            a = policy(s); s = model.dynamics(s, a).sample()
            ret += model.reward(s)
        returns.append(ret)
    return torch.tensor(returns).mean()   # cheap proxy for real eval
```

This is far cheaper than real-environment evaluation (see [69-AI-Evaluation-and-LLM-Testing](../69-AI-Evaluation-and-LLM-Testing/)).

## 13. Common Implementation Bugs

| Bug | Symptom | Fix |
|-----|---------|-----|
| Forgetting to detach posterior in prior loss | Training collapse | Separate prior/posterior graphs |
| Reusing same RNN state across batch | Cross-contamination | Per-sample hidden init |
| Too-large KL β | Latent collapse | Free bits / KL annealing |
| No gradient through imagination | Model doesn't improve | `retain_graph` / proper BPTT |

## 14. Reference Hyperparameters (DreamerV3-style)

| Param | Small | Medium | Large |
|-------|-------|--------|-------|
| hidden | 256 | 512 | 1024 |
| stoch | 32 | 32 | 32 |
| batch | 128 | 256 | 512 |
| horizon H | 15 | 15 | 15 |
| model_lr | 3e-4 | 2e-4 | 1e-4 |
| β (KL) | 1.0 | 1.0 | 1.0 |

## 15. Summary

The deep-dive shows world models are engineering-complete but data/hardware-hungry at the video end. The workhorse is the RSSM + imagination actor-critic; the frontier is action-conditional video and sim-to-real, with distillation enabling edge deployment and self-evaluation lowering the cost of benchmarking. The next file maps these to actual libraries and benchmarks you can run today.

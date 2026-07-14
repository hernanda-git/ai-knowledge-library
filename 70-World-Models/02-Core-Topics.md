# World Models — Core Topics

> July 2026

This file covers the conceptual core of world models: latent dynamics, the Recurrent State-Space Model (RSSM), model-based reinforcement learning, imagination, self-supervised predictive learning (JEPA), and generative video world models. It pairs with [01-Overview.md](./01-Overview.md) and the deeper math in [03-Technical-Deep-Dive.md](./03-Technical-Deep-Dive.md).

Cross-references: [29-Reasoning-and-Inference-Scaling](../29-Reasoning-and-Inference-Scaling/), [60-Physical-AI-and-Embodied-Intelligence](../60-Physical-AI-and-Embodied-Intelligence/), [03-Agents](../03-Agents/).

---

## 1. Latent Dynamics Models

The central idea: instead of predicting raw pixels, predict a **compressed latent state**. This is both more sample-efficient and more stable.

### 1.1 The RSSM (Recurrent State-Space Model)

Dreamer's RSSM factorizes the world into:

- A **deterministic** path: an RNN hidden state `h_t` carrying long-term memory.
- A **stochastic** path: a latent `z_t ~ p(z_t | h_t, o_t)` that captures uncertain, partially-observed factors.

The generative model:

```
h_t = f_θ(h_{t-1}, z_{t-1}, a_{t-1})        # GRU recurrence
z_t ~ p_θ(z_t | h_t)                          # prior
o_t ~ p_θ(o_t | h_t, z_t)                     # decoder
r_t ~ p_θ(r_t | h_t, z_t)                     # reward
```

And a posterior `q_θ(z_t | h_t, o_t)` used only during training (when real observations exist). The agent plans in the **prior** during inference.

| Component | Input | Output | Trained by |
|-----------|-------|--------|------------|
| Encoder | observation `o` | embed `e` | reconstruction loss |
| GRU | `e, z, a` | hidden `h` | BPTT through time |
| Prior | `h` | `z` distribution | KL to posterior |
| Posterior | `h, e` | `z` distribution | KL to prior + recon |
| Decoder | `h, z` | `o` reconstruction | ELBO |
| Reward | `h, z` | scalar `r` | regression |

---

## 2. Model-Based Reinforcement Learning

### 2.1 Why Model-Based?

| Property | Model-Free (PPO/SAC) | Model-Based (Dreamer) |
|----------|----------------------|------------------------|
| Env steps needed | 10⁶–10⁹ | 10³–10⁵ |
| Training location | Real env | Imagined rollouts |
| Sample cost | High | Low |
| Bias risk | Low | Model error compounds |

The trade-off is **model bias**: if the world model is wrong, the agent optimizes a fantasy. Mitigations: ensemble disagreement (see §2.3), shorter imagination horizons, and periodic real-world correction.

### 2.2 The Training Objective (ELBO)

```
L = E_q [ Σ_t  log p(o_t | h_t, z_t)
              + log p(r_t | h_t, z_t)
              - β · KL(q(z_t | h_t, o_t) ‖ p(z_t | h_t)) ]
```

The KL term keeps the prior honest (a free-energy / information bottleneck). Reward prediction turns the latent into a value estimator.

### 2.3 Epistemic Uncertainty & Ensembles

To avoid over-confident imagination, modern systems use:

- **Probabilistic ensembles** (e.g., 5 models); disagreement = uncertainty.
- **Reward penalty**: `r_penalized = r_mean − λ·std(r)`.
- **Trajectory truncation** when uncertainty exceeds a threshold.

```python
# Model-ensemble disagreement penalty (conceptual)
mean_r = torch.stack([m(h, z) for m in models]).mean(0)
std_r  = torch.stack([m(h, z) for m in models]).std(0)
r_eff  = mean_r - 0.5 * std_r     # conservative imagined reward
```

---

## 3. Imagination and Latent Planning

Once `p_θ(s', o | s, a)` is learned, planning is **gradient-based** inside the model:

1. Start from the current latent `s_0`.
2. For `k = 1..H`: propose actions `a_k`, roll out `s_k ~ p_θ(·|s_{k-1}, a_k)`.
3. Accumulate imagined return `Σ γ^k r(s_k, a_k)`.
4. Backprop through the *entire imagined trajectory* to update an action/value network.

Because everything is differentiable, you can use straight `torch.optim` — no separate planner search required (though CEM/MPC are also used).

| Planner | Pros | Cons |
|---------|------|------|
| Gradient (actor-critic in dream) | End-to-end, fast | Local optima |
| CEM (cross-entropy method) | Robust, gradient-free | Sample heavy |
| MPC (model predictive control) | Replans each step | Needs fast model |
| Tree search (MuZero) | Global, strong | Expensive |

---

## 4. MuZero: Planning Without an Explicit Dynamics Model

MuZero learns **three** functions from pixels alone:

- `f_θ`: representation (obs → latent)
- `g_θ`: dynamics (latent, action → next latent) — the *model*
- `h_θ`: prediction (latent → policy, value)

It never reconstructs observations — it only predicts **value and reward**, searched with MCTS. This is "model-based at the level of value," not pixels, and is why MuZero masters Go, chess, shogi, and 57 Atari games without knowing the rules.

```
MCTS loop (MuZero):
  select  → expand(g_θ) → evaluate(h_θ) → backup
  action  = argmax over visit counts
```

---

## 5. Self-Supervised Predictive Learning: JEPA

Yann LeCun's **Joint Embedding Predictive Architecture** family avoids reconstructing pixels entirely:

- **I-JEPA** (images): predict masked embeddings from visible ones.
- **V-JEPA** (video): predict future embeddings; learns a *representational* world model good for planning/control.

Why it matters: reconstruction (pixel/VAE) wastes capacity on irrelevant detail (e.g., every leaf on a tree). JEPA predicts **abstract embeddings**, which is closer to how a planner wants the world modeled.

| Approach | Predicts | Good for | Wastes capacity on |
|----------|----------|----------|--------------------|
| VAE/Diffusion recon | Pixels | Fidelity, content | Irrelevant texture |
| JEPA | Embeddings | Control, planning | Nothing (abstract) |

---

## 6. Generative Video World Models

A diffusion or autoregressive video model trained on egocentric/interactive footage becomes a **pixel-space simulator**:

- **Genie** (2024): a latent action model infers *actions from video alone* — you can turn any image into a playable world.
- **GameNGen** (2024): a diffusion model reproduces Doom at ~20 fps from a single prompt-following agent.
- **World-model video** (2025+): robotics labs train video models conditioned on robot actions, then use them for sim-to-real and data augmentation.

The key addition over plain video generation is the **action-conditional** term:

```
frame_{t+1} ~ p_θ(· | frame_{≤t}, action_t)
```

Without `action_t` the model just predicts "likely future"; with it, it becomes a controllable simulator.

---

## 7. World Models for Language and Agents

LLM-based agents can also run *implicit* world models:

- **Theory-of-mind / user simulation:** an agent models "what the user will say next."
- **Tool-state simulation:** predicts "after I call `transfer_funds`, the balance field changes."
- **Code-world simulation:** predicts "running this test will fail because of import error."

These are soft, learned, and uncalibrated — which is why [68-Context-Engineering](../68-Context-Engineering/) and [69-AI-Evaluation-and-LLM-Testing](../69-AI-Evaluation-and-LLM-Testing/) matter for grounding them.

---

## 8. Benchmarks and What "Good" Means

| Benchmark | Measures | Used for |
|-----------|----------|----------|
| Atari-100k / 200k | Sample-efficient control | Latent MBRL |
| DMControl | Continuous control | RSSM |
| Minecraft (MineDojo) | Open-ended, long horizon | Genie/Dreamer |
| Habitat / RLBench | Embodied, 3D | Robotics world models |
| Video prediction (KITTI, BAIR) | Frame fidelity + action-cond | Video world models |

A world model is "good" not when frames look pretty but when **plans derived from it transfer to the real environment** (sim-to-real success rate).

---

## 9. Failure Modes

1. **Model exploitation** — the agent finds loopholes in the imagined reward the real world doesn't have.
2. **Compounding error** — small prediction drift accumulates over long rollouts.
3. **Distribution shift** — real states drift outside the model's training manifold.
4. **Hallucinated physics** — video models generate plausible-but-impossible dynamics (relevant to [52-AI-Hallucination-Detection-and-Mitigation](../52-AI-Hallucination-Detection-and-Mitigation/)).

---

## 10. Contrast With Model-Free Methods (Worked Example)

Consider learning to walk in DMControl. Model-free PPO needs ~10⁷ environment steps;
DreamerV3 needs ~10⁵. The difference is *where* learning happens:

| Step | Model-free (PPO) | Model-based (Dreamer) |
|------|------------------|------------------------|
| Interact | 10⁷ real steps | 10⁵ real steps |
| Learn policy | On-policy, online | In imagined rollouts |
| Replay | Limited | Unlimited (free dreams) |
| Wall-clock | Hours–days | Minutes–hours |

## 11. Language-Agent World Models (Deeper)

LLM agents maintain an *implicit* forward model of tool/environment state:

- **State:** "DB has tables A, B; user is authenticated."
- **Action:** `INSERT INTO A ...`
- **Predicted next state:** "Row added; A has N+1 rows; trigger fired."
- **Reward:** task progress / test pass.

This is soft and uncalibrated, so it must be grounded with real tool calls and
validated by [69-AI-Evaluation-and-LLM-Testing](../69-AI-Evaluation-and-LLM-Testing/).

## 12. Uncertainty-Aware Planning (Detail)

```python
import torch
def plan_uncertain(model, s0, horizon=10, n_candidates=8):
    best_a, best_score = None, -1e9
    for _ in range(n_candidates):
        a_seq = sample_action_sequence(horizon)
        s, score, unc = s0, 0.0, 0.0
        for a in a_seq:
            dist = model.dynamics(s, a)          # stochastic
            s = dist.sample()
            score += model.reward(s)
            unc   += dist.entropy().mean()        # higher = less certain
        score -= 0.1 * unc                        # penalize uncertainty
        if score > best_score:
            best_a, best_score = a_seq, score
    return best_a[0]                              # execute first action
```

## 13. Video World Models vs Latent: When to Use Which

| Need | Use | Reason |
|------|-----|--------|
| Control, robotics | Latent RSSM | Cheap, stable, differentiable |
| Data augmentation | Video model | High-fidelity frames |
| Human-inspectable sim | Video model | Pixels are legible |
| Long-horizon planning | Latent | No pixel cost per step |
| Sim-to-real robotics | Both (latent plan + video verify) | Combine strengths |

## 14. Research Frontier: Causal World Models

Standard world models learn *correlations*. Causal variants add interventions:

```
do(a): force action, observe counterfactual state
```

This makes them robust to distribution shift and lets them answer "what if" cleanly —
a hot 2026 direction tied to [29-Reasoning-and-Inference-Scaling](../29-Reasoning-and-Inference-Scaling/).

## 15. Summary

Core topics reduce to one loop: **learn a latent or generative forward model, then plan by imagination**. RSSM gives sample-efficient control; MuZero/JEPA give rule-free, abstract planning; video models give controllable, high-fidelity simulation. Language agents carry a soft version of the same idea. The next file turns these concepts into trainable architectures and runnable code.

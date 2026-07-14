# World Models — Learning, Simulating, and Predicting the World Inside a Model

> July 2026

**World models** are internal, learnable representations that an AI system builds of how an environment works, enabling it to simulate, predict, and plan over future states. Rather than mapping a perception directly to an action (the classic reactive policy), a world model asks the system to *imagine*: given my current state and a candidate action, what will the world look like one step — or many steps — from now?

This document introduces the field, why it has surged to the center of AI research and product roadmaps in 2025–2026, and how it connects to the rest of this library (see cross-references to [60-Physical-AI-and-Embodied-Intelligence](../60-Physical-AI-and-Embodied-Intelligence/), [29-Reasoning-and-Inference-Scaling](../29-Reasoning-and-Inference-Scaling/), [03-Agents](../03-Agents/), [39-Digital-Twins](../39-Digital-Twins/), [50-Multimodal-AI](../50-Multimodal-AI/), and [61-AI-for-Gaming](../61-AI-for-Gaming/)).

---

## 1. What Is a World Model?

A world model is **a trainable internal simulator of an environment**. Formally, it learns a transition function:

```
s_{t+1} ~ p_θ(· | s_t, a_t)
o_t      ~ p_φ(· | s_t)
```

where `s_t` is a (often latent) state, `a_t` is an action, and `o_t` is an observation. The agent can then *roll out* imagined trajectories inside the model without touching the real environment — this is the essence of **model-based reinforcement learning** (MBRL) and **imagination-based planning**.

### 1.1 Two Lenses

| Lens | Question it answers | Example systems |
|------|---------------------|-----------------|
| **Predictive / latent** | "What latent state comes next?" | DreamerV3, PlaNet, RSSM |
| **Generative / video** | "What pixels / frames come next?" | Genie, GameNGen, Sora-class video models |

The field straddles both: latent world models power control and robotics; generative world models power simulation, content, and video prediction.

### 1.2 Why "World" and not "Simulator"?

A hand-coded simulator (e.g., a physics engine) *is* a world model in the broadest sense, but the defining trait of the modern research area is that **the model is learned from data**, not programmed. This lets it capture dynamics that are hard to specify analytically — human behavior, deformable objects, traffic flow, language-game interplay.

---

## 2. Why World Models Exploded in 2025–2026

Five forces converged:

1. **Sample efficiency ceiling in model-free RL.** Model-free methods (PPO, SAC) need millions of environment steps. World models let an agent train *inside its own imagination*, cutting real-world interaction by 10–100×.
2. **The embodied AI push** (see [60-Physical-AI-and-Embodied-Intelligence](../60-Physical-AI-and-Embodied-Intelligence/)). Robots cannot safely or cheaply try every failure in the real world; a learned simulator is the only scalable training ground.
3. **Generative video as a byproduct.** Diffusion and transformer video models turned out to be *de facto* pixel-space world models, usable for simulation, data augmentation, and planning.
4. **Scaling laws extended to simulation.** Just as scaling data/compute improved LLMs, scaling world-model capacity improved long-horizon prediction — making big world models tractable.
5. **Agent planning needs look-ahead** (see [03-Agents](../03-Agents/)). An agent that can simulate outcomes before acting is dramatically more reliable than a reactive one.

---

## 3. The Core Loop: Dream, Plan, Act

```
        ┌─────────────────────────────────────────┐
        │  Learned World Model  p_θ(s', o | s, a)  │
        └─────────────────────────────────────────┘
                         ▲                   │ (imagined)
        (real obs)       │                   ▼
   Real env ──▶ Encoder ──▶ Latent state s ──▶ Planner ──▶ action a ──▶ Real env
                         ▲                                      │
                         └──────── loss backprop ──────────────┘
```

- **Encode:** map observations to latent states.
- **Dream:** roll out `s_{t+k}` inside the model for k steps.
- **Plan:** pick the action sequence that maximizes imagined return.
- **Act:** execute the first action, observe, repeat.

This is the **Dreamer** family's signature workflow.

---

## 4. Taxonomy of World Models

| Type | State space | Typical use | Strengths | Weaknesses |
|------|-------------|-------------|-----------|------------|
| **Latent / RSSM** | Compressed latent | RL, control, robotics | Sample efficient, stable | Abstract, not inspectable |
| **Diffusion video** | Pixel space | Simulation, content, data | High fidelity, general | Expensive, not causal-control-aware |
| **Autoregressive token** | Discrete tokens | Language-world, games | Composable, trainable on GPUs | Error accumulation |
| **NeRF / Gaussian-splat** | 3D scene | Navigation, rendering | Geometric accuracy | Static, needs many views |
| **JEPA-style predictive** | Embedding | Self-supervised, planning | Efficient, robust | Less generative |

---

## 5. Relationship to Other Library Topics

- **[29-Reasoning-and-Inference-Scaling](../29-Reasoning-and-Inference-Scaling/)** — world models add *environmental* structure to reasoning; chain-of-thought reasons over logic, world models reason over physics/states.
- **[60-Physical-AI-and-Embodied-Intelligence](../60-Physical-AI-and-Embodied-Intelligence/)** — the primary deployment surface for latent world models.
- **[39-Digital-Twins](../39-Digital-Twins/)** — digital twins are *engineered* simulators; world models are *learned* ones. They increasingly fuse.
- **[50-Multimodal-AI](../50-Multimodal-AI/)** — video/frame prediction is a multimodal generation problem.
- **[03-Agents](../03-Agents/)** — model-based agents plan in imagination before acting.
- **[68-Context-Engineering](../68-Context-Engineering/)** — the imagined trajectory is part of the agent's working context.
- **[69-AI-Evaluation-and-LLM-Testing](../69-AI-Evaluation-and-LLM-Testing/)** — evaluating "is this simulation faithful?" is an open eval problem.

---

## 6. Key Milestones (Selected)

| Year | System | Significance |
|------|--------|--------------|
| 2018 | World Models (Ha & Schmidhuber) | VAE + MDN-RNN agent "dreams" to train |
| 2019 | PlaNet / Dreamer | RSSM latent dynamics, latent planning |
| 2021 | MuZero (model-based planning w/ value) | Masters Go/Chess/Atari without rules |
| 2023 | DreamerV3 | First to train a single agent across 150+ tasks, including Minecraft diamond |
| 2024 | Genie | Generate playable worlds from a single image |
| 2024 | GameNGen | Diffuse-based Doom simulator |
| 2024 | JEPA / I-JEPA | Self-supervised predictive architectures (LeCun) |
| 2025 | Video world models for robotics | Pixel-space sim-to-real for manipulation |

---

## 7. What This Category Covers

- **01-Overview.md** (this file) — definitions, taxonomy, the dream-plan-act loop.
- **02-Core-Topics.md** — latent dynamics, RSSM, model-based RL, imagination, JEPA, video world models.
- **03-Technical-Deep-Dive.md** — architectures, training objectives, sim-to-real, rollout, code.
- **04-Tools-and-Frameworks.md** — Dreamer, Diffusion Policy, MineDojo, Isaac Sim, video models, benchmarks.
- **05-Future-Outlook.md** — trends, risks, productization, open problems.

---

## 8. A First Intuition in Code

A minimal latent world model step (conceptual, PyTorch-like):

```python
import torch, torch.nn as nn

class RSSM(nn.Module):
    """Recurrent State-Space Model: combines deterministic RNN state
    with stochastic latent z to model (s', o | s, a)."""
    def __init__(self, action_dim, hidden=256, stoch=32, embed=256):
        super().__init__()
        self.rnn = nn.GRU(embed + action_dim, hidden)
        self.z_prior = nn.Linear(hidden, stoch * 2)   # predict next z
        self.z_post  = nn.Linear(hidden + embed, stoch * 2)  # infer z from obs
        self.decoder = nn.Linear(stoch + hidden, embed) # reconstruct obs

    def observe(self, embed_t, action_t, h):
        _, h = self.rnn(torch.cat([embed_t, action_t], -1).unsqueeze(0), h)
        z_post = self.sample(self.z_post(torch.cat([h[-1], embed_t], -1)))
        return z_post, h

    def imagine(self, z, action_t, h):
        _, h = self.rnn(torch.cat([self.decoder(z), action_t], -1).unsqueeze(0), h)
        z_next = self.sample(self.z_prior(h[-1]))
        return z_next, h
```

This single module is the heart of every Dreamer-style agent. Expand on it in [03-Technical-Deep-Dive.md](./03-Technical-Deep-Dive.md).

---

## 9. Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| "A world model is just a video generator" | Video models are one *flavor*; latent RSSM models are another, and they don't render pixels at all. |
| "World models replace policies" | They *support* policies by providing a simulator to plan with; the policy still chooses actions. |
| "More data always helps" | Past a point, model capacity and sim-to-real gap dominate; real correction matters more than raw data. |
| "They are only for games/robots" | Language agents, trading, and scientific surrogates all use world-model-style imagination. |

## 10. Glossary

- **Latent state `s`** — compressed internal representation of the world.
- **Observation `o`** — raw sensor/ pixel / token input.
- **RSSM** — Recurrent State-Space Model (deterministic RNN + stochastic latent).
- **Imagination / Dream** — rolling out trajectories inside the learned model.
- **Sim-to-real** — transferring a policy trained in simulation to the real world.
- **Model-based RL** — RL that learns and uses a forward model of the environment.
- **Action-conditional** — the model's prediction depends on a chosen action (vs. passive forecast).
- **JEPA** — Joint Embedding Predictive Architecture; predicts embeddings, not pixels.
- **Foundation world model** — a large pretrained simulator fine-tuned per domain/task.

## 11. A Tiny End-to-End Example (Conceptual Pipeline)

```bash
# 1. Collect interactions (real or sim)
python collect.py --env dmcontrol:cheetah --steps 100000

# 2. Train the world model (ELBO on latent dynamics)
python train_world_model.py --model rssm --batch 512 --gpu 0

# 3. Train actor-critic INSIDE the dream (no real env needed)
python train_imagination.py --horizon 15 --epochs 200

# 4. Deploy: act using only the learned model + a few real corrections
python deploy.py --real-finetune-steps 500
```

This four-stage loop is the template used by Dreamer-style systems and by embodied
robotics pipelines (replacing `dmcontrol:cheetah` with a robot task in Isaac Sim).

## 12. How This Fits the Library's Map

```
Foundations (01) ──▶ LLMs (02) ──▶ Agents (03) ──▶ Reasoning (29)
                                            │
                                            ▼
                                      World Models (70)  ◄── Physical AI (60)
                                            │                Embodied (60)
                                            ▼
                                      Digital Twins (39) ── Simulation / Surrogates
```

World models sit at the intersection of agents, reasoning, and embodied AI: they are the
mechanism that lets an agent *deliberate* about consequences rather than react.

## 13. Reading Order

1. This overview → mental model and taxonomy.
2. [02-Core-Topics.md](./02-Core-Topics.md) → RSSM, MBRL, JEPA, video models.
3. [03-Technical-Deep-Dive.md](./03-Technical-Deep-Dive.md) → architectures, code, sim-to-real.
4. [04-Tools-and-Frameworks.md](./04-Tools-and-Frameworks.md) → what to actually install.
5. [05-Future-Outlook.md](./05-Future-Outlook.md) → where the field is going.

## 14. Summary

World models are the bridge between **perception** and **planning**: they let an AI system *think before it acts* by simulating the future in a learned internal space. They are simultaneously a classic RL idea (model-based control), a frontier of generative video, and the substrate on which embodied and agentic AI will scale. The remaining files in this category go deep on the math, the tooling, and the roadmap.

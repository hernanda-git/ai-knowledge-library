# World Models — Tools and Frameworks

> July 2026

A practical catalog of open-source and commercial tooling for building, training, and deploying world models — from latent MBRL libraries to generative video simulators and embodied benchmarks. Pair with [03-Technical-Deep-Dive.md](./03-Technical-Deep-Dive.md).

Cross-references: [60-Physical-AI-and-Embodied-Intelligence](../60-Physical-AI-and-Embodied-Intelligence/), [61-AI-for-Gaming](../61-AI-for-Gaming/), [39-Digital-Twins](../39-Digital-Twins/), [50-Multimodal-AI](../50-Multimodal-AI/).

---

## 1. Latent World-Model Libraries (Model-Based RL)

| Tool | Lang | Highlights | Maturity |
|------|------|-----------|----------|
| **dreamerv3** (original repo) | Jax | Reference RSSM + imagination actor-critic | High |
| **torchrl** (PyTorch) | Python | Modular MBRL, Dreamer, RSSM, PlaNet | High |
| **Stable-Baselines3** | Python | Model-free baselines for comparison | High |
| **mbpo** (Model-Based Policy Opt.) | Python | Ensemble MBRL, uncertainty penalty | Medium |
| **SimPLe** | Python | World models for Atari | Research |
| **Dreamer.jl / JAX** | Multi | Re-implementations | Medium |

### 1.1 torchrl Quick Start

```python
from torchrl.env import GymEnv
from torchrl.modules import DreamerActor, RSSM
from torchrl.collectors import SyncDataCollector

env = GymEnv("CartPole-v1", device="cuda")
rssm = RSSM(observation_dim=4, action_dim=2, hidden=256, stoch=32)
actor = DreamerActor(rssm, hidden=256, action_dim=2)

collector = SyncDataCollector(env, policy=actor, frames_per_batch=2048)
for batch in collector:
    loss = rssm.loss(batch)         # ELBO + imagination
    loss.backward(); opt.step()
```

---

## 2. Generative / Video World Models

| Tool | Type | Use | Notes |
|------|------|-----|-------|
| **Genie** (DeepMind) | Latent-action video | Playable worlds from images | Research release |
| **GameNGen** | Diffusion Doom | Real-time game simulation | Demo |
| **Sora-class / video diffusion** | Text/video→video | General world sim | Closed/API |
| **WorldDreamer / GAIA-1** | Action-cond video | Driving world model | Research |
| **Diffusion Policy** (Chi et al.) | Diffusion + BC | Robot manipulation sim→real | Open (GitHub) |
| **Open-Sora / CogVideo** | Open video gen | Fine-tunable world sim base | Open |

### 2.1 Diffusion Policy Snippet (robotics)

```python
# Action-conditional: predict robot action chunk from obs window
import torch, torch.nn as nn
class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, T=16):
        super().__init__()
        self.net = nn.Transformer(d_model=obs_dim, num_layers=4)
        self.head = nn.Linear(obs_dim, act_dim * T)
        self.noise_sched = "ddpm"
    def forward(self, obs_window, noise):
        h = self.net(obs_window)
        return self.head(h) + noise   # denoise toward action chunk
```

---

## 3. Embodied / Robotics Simulators (for Sys-ID & Real Transfer)

| Sim | Strength | World-model link |
|-----|----------|------------------|
| **NVIDIA Isaac Sim / Orbit** | Physically accurate, GPU | Generate real-like data; sys-ID |
| **MuJoCo** | Fast, contact-rich | Classic MBRL benchmark |
| **Habitat / Habitat 2.0** | Photo-real indoor | Embodied world-model evals |
| **RLBench** | Task-oriented manipulation | Per-task world models |
| **AI2-THOR** | Interactive 3D scenes | Agent simulation |
| **MineDojo / Minecraft** | Open-ended, long horizon | Genie / STEVE training |

---

## 4. Benchmarks & Datasets

| Benchmark | Domain | What to measure |
|-----------|--------|-----------------|
| **Atari-100k** | Arcade | Sample-efficient control |
| **DeepMind Control (DMControl)** | Continuous control | RSSM quality |
| **MineDojo** | Open world | Long-horizon, language-cond |
| **BAIR / KITTI / Cityscapes** | Video pred | Frame fidelity, action-cond |
| **Meta-World** | Multi-task | Transfer / generalization |
| **WAYMO / nuScenes** | Driving | Driving world models (GAIA-1) |

---

## 5. Self-Supervised Predictive (JEPA) Tooling

| Tool | Repo | Notes |
|------|------|-------|
| **I-JEPA** | facebookresearch | Image embedding prediction |
| **V-JEPA** | facebookresearch | Video embedding prediction, control-ready |
| **MC-JEPA / MC-NEAT** | research | Multimodal predictive |

```python
# V-JEPA conceptual: predict future embedding from masked past
from je pa import VJEPA  # illustrative
model = VJEPA(encoder="vit-h", predictor="small")
loss = model.forward(video_clip)   # predicts masked future tokens
```

---

## 6. Cloud & Managed Offerings (2026)

| Provider | Offering | Relevance |
|----------|----------|-----------|
| **NVIDIA** | Isaac Sim + Cosmos world model | Robotics + video world model |
| **Google DeepMind** | Genie, Veo (video) | Research + API |
| **Microsoft** | Muse / world-model research | Gaming |
| **OpenAI** | Sora (video sim) | Content + sim |
| **AWS / Azure** | GPU sim farms | Training infra |

---

## 7. Evaluation Tooling

| Tool | Checks |
|------|--------|
| **torchrl metrics** | Return, model ELBO, sim-to-real gap |
| **Video metrics (FVD, PSNR, SSIM)** | Frame fidelity |
| **Custom sim-to-real harness** | Real success rate vs dream success |

```python
def sim_to_real_gap(dream_return, real_return):
    return (dream_return - real_return) / dream_return   # 0 = perfect transfer
```

---

## 8. How to Choose a Stack

| Goal | Recommended stack |
|------|-------------------|
| Sample-efficient control, fast iteration | torchrl + RSSM on DMControl |
| Robotics sim-to-real | Diffusion Policy + Isaac Sim |
| Playable world from image | Genie (or Open-Sora fine-tune) |
| Driving simulation | GAIA-1 / action-cond video |
| Abstract planning, no pixels | V-JEPA + downstream policy |

---

## 9. Integration with Agent Frameworks

World models plug into [03-Agents](../03-Agents/) loops as a **look-ahead module**:

```python
# Agent uses world model to pre-screen actions
def choose_action(agent_state, world_model, candidate_actions):
    best, best_score = None, -inf
    for a in candidate_actions:
        score = world_model.imagined_return(agent_state, a, horizon=10)
        if score > best_score:
            best, best_score = a, score
    return best
```

This is "model-based deliberation" and pairs with [68-Context-Engineering](../68-Context-Engineering/) (the imagined trajectory is context) and [69-AI-Evaluation-and-LLM-Testing](../69-AI-Evaluation-and-LLM-Testing/) (verify the simulation).

## 10. Cost & Hardware Planning

| Workload | GPU | Approx cost | Time |
|----------|-----|-------------|------|
| RSSM on DMControl | 1× RTX 4090 | Low | Minutes–hours |
| MineDojo / Genie | 8× A100 | Medium-High | Days |
| Video world model (diffusion) | 8× H100 | High | Weeks |
| Foundation world model | Cluster | Very High | Months |

For budgeting details see [41-AI-Cost-Optimization-and-Enterprise-ROI](../41-AI-Cost-Optimization-and-Enterprise-ROI/).

## 11. Open vs Closed Tooling (2026)

| Open | Closed / API |
|------|--------------|
| torchrl, dreamerv3, Diffusion Policy, Open-Sora, V-JEPA | Sora, Veo, Genie 2, Cosmos (partial), Muse |

Prefer open stacks when you need to introspect the dynamics; use closed APIs for fast content/sim prototypes.

## 12. MLOps for World Models

World-model training needs the same rigor as any ML system — see [31-AI-Workflow-Orchestration-and-Durable-Execution](../31-AI-Workflow-Orchestration-and-Durable-Execution/) and [20-Agent-Infrastructure-and-Observability](../20-Agent-Infrastructure-and-Observability/):

- Version the **environment** (not just the model) — sim params are part of the artifact.
- Track **sim-to-real gap** as a first-class metric.
- Reproduce with fixed seeds + logged domain-randomization ranges.

## 13. A Minimal "Hello World" (torchrl)

```python
from torchrl.env import GymEnv
from torchrl.modules import DreamerActor, RSSM
env = GymEnv("CartPole-v1")
rssm = RSSM(observation_dim=4, action_dim=2)
actor = DreamerActor(rssm, action_dim=2)
# train loop as in §1.1 ...
```

## 14. Checklist: Picking Your Stack

- [ ] Is the task continuous control? → RSSM / torchrl.
- [ ] Is it robotics with real hardware? → Diffusion Policy + Isaac Sim.
- [ ] Do you need renderable simulation? → action-conditional video.
- [ ] No action labels available? → Genie-style latent action model.
- [ ] Need abstract planning, no pixels? → V-JEPA + downstream policy.

## 15. Summary

You do not need to build a world model from scratch: Dreamer/torchrl cover latent MBRL, Diffusion Policy + Isaac Sim cover robotics, and open video models cover generative simulation. Pick by goal (control vs robotics vs content), then measure rigorously with sim-to-real gap. Pair the stack with proper MLOps and cost tracking, and integrate as a look-ahead module in agent loops. The final file looks at where the field is heading.

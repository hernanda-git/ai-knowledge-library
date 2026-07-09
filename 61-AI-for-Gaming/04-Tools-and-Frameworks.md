# AI for Gaming — Tools and Frameworks

> **Category 61 — AI Knowledge Library** | A practical catalog of engines, ML libraries, generative pipelines, and services used to build AI-driven games in 2026, with decision guidance on when to adopt each. Match tooling to the pillar you are building.

---

## Table of Contents

1. [Game Engines](#game-engines)
2. [RL and ML Libraries](#rl-and-ml-libraries)
3. [Simulation at Scale](#simulation-at-scale)
4. [Procedural Generation Toolkits](#procedural-generation-toolkits)
5. [LLM / Dialogue Tooling](#llm--dialogue-tooling)
6. [Generative Art and Asset Pipelines](#generative-art-and-asset-pipelines)
7. [Automated QA and Playtesting](#automated-qa-and-playtesting)
8. [Cloud and Inference Services](#cloud-and-inference-services)
9. [Observability and MLOps](#observability-and-mlops)
10. [Open-Source Starter Stack](#open-source-starter-stack)
11. [Build vs Buy](#build-vs-buy)
12. [Selection Guide](#selection-guide)

---

## Game Engines

| Engine | AI Strengths | Notes |
|--------|--------------|-------|
| Unity | ML-Agents, C# BTs, huge asset store | Best RL-in-editor story |
| Unreal | Behavior Trees, Mass AI, Blueprints | AAA standard; Mass for crowds |
| Godot | Lightweight, open source | Growing ML community |
| Bevy / Custom | Full control | Used by ML research envs |

### Unity ML-Agents (example)

```python
# Train a Unity agent via mlagents CLI
# mlagents-learn config/ppo.yaml --run-id=dungeon_v1
from mlagents_envs.environment import UnityEnvironment
env = UnityEnvironment(file_name="build/Dungeon")
```

### Unreal Mass AI

For thousands of agents (battles, crowds), Unreal's Mass Entity framework distributes simulation across threads — pair with utility/BT logic.

### Godot

Excellent for prototypes and small teams; C# and GDScript both work, and Python bridges are common for ML experiments.

---

## RL and ML Libraries

| Library | Use | Why |
|---------|-----|-----|
| Stable-Baselines3 | PPO/SAC/A2C | Battle-tested, simple API |
| RLlib | Distributed RL | Scale to many agents |
| CleanRL | Single-file reference | Learning + reproducibility |
| PyTorch / JAX | Custom nets | Full control |
| Isaac Gym / Isaac Sim | GPU sim | Massively parallel physics |

```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", "GameEnv-v0", verbose=1)
model.learn(total_timesteps=1_000_000)
```

**Parallel simulators** are the force multiplier: train at 10^5 steps/sec by running thousands of environments in GPU-backed sims.

---

## Simulation at Scale

For self-play and population training you need throughput:

- **Isaac Gym / Sim**: GPU-accelerated thousands of envs.
- **Env pools**: launch N processes, batch rollouts.
- **League servers**: persist opponent snapshots.

```python
# Conceptual env pool
pool = [make_env() for _ in range(4096)]
rollouts = [env.step(policy.act(env.obs)) for env in pool]
```

---

## Procedural Generation Toolkits

- **WFC implementations**: multiple open-source ports (tile + overlap).
- **PCGML repos**: Level GAN, MarioGAN-style generators.
- **Houdini / SideFX**: node-based procedural art for AAA.
- **Blender + Python**: scripted asset generation.
- **L-systems libraries**: plant/city generation.

```python
# Conceptual: generate a level with a trained GAN generator
z = torch.randn(1, latent_dim)
level = generator(z).argmax(-1).reshape(32, 32)
assert playability_solver(level), "reject, resample"
```

---

## LLM / Dialogue Tooling

| Tool | Role |
|------|------|
| OpenAI / Anthropic / local LLMs | NPC dialogue |
| LangChain / LlamaIndex | Orchestration + RAG |
| Vector DBs (see [04-RAG/03](../04-RAG/03-Vector-Databases.md)) | Lore retrieval |
| Managed NPC-dialogue services | Hosted conversational characters |
| Guardrail libs | Safety filtering |

Connect dialogue NPCs to [Agent Memory (32)](../32-Agent-Memory-Systems/01-Overview.md) for persistent character recall across sessions. Use [RAG (04)](../04-RAG/01-RAG-Architectures.md) to ground responses in canonical lore.

### Latency Patterns

- **Pre-generate + cache** common lines.
- **Stream** long monologues.
- **Fallback** to templated lines when the model is slow.

---

## Generative Art and Asset Pipelines

| Modality | Tooling |
|----------|---------|
| 2D concept | Diffusion (text-to-image) |
| 3D models | Text/Image-to-3D, NeRF, Gaussian Splatting |
| Animation | Motion diffusion, rigging auto-tools |
| Audio/Music | Neural synthesis, adaptive music engines |
| Voices | TTS with emotion control |

> Pipeline pattern: `prompt → diffusion draft → human art-direction pass → engine import`. AI drafts; humans direct — the dominant 2026 workflow. See [Multimodal AI (50)](50-Multimodal-AI/01-Overview.md) for the underlying models.

---

## Automated QA and Playtesting

- **Game-specific bot frameworks** (engine test harnesses).
- **Computer-use agents** ([26/46](../26-Browser-Based-AI/46-Agentic-Browser-Automation-Computer-Use/03-Browser-Agent-Architectures.md)) for UI testing.
- **Fuzzing tools** for input stress.
- **Telemetry/observability** ([20/03](../20-Agent-Infrastructure-and-Observability/03-Agent-Tracing-and-Observability.md)).

### CI Integration

```
build → launch 200 bots × 50 episodes → aggregate telemetry → gate release
```

---

## Cloud and Inference Services

- **Model hosting** for LLM NPCs (server-side) with caching.
- **Edge deployment** ([62](../62-Edge-AI-and-On-Device-Inference/01-Overview.md)) for on-device SLM NPCs.
- **Synthetic data generation** ([51](../51-Synthetic-Data-Generation/01-Overview.md)) to bootstrap training environments.
- **Cost monitoring** ([41](../41-AI-Cost-Optimization-and-Enterprise-ROI/01-Overview.md)) — token spend per active player is a real P&L line.

---

## Observability and MLOps

Treat game AI like any production ML system:

- Log every NPC decision (see [Agent Tracing (20/03)](../20-Agent-Infrastructure-and-Observability/03-Agent-Tracing-and-Observability.md)).
- Track drift in player models.
- Version policies and prompt templates ([20/08](../20-Agent-Infrastructure-and-Observability/08-Agent-Registry-and-Versioning.md)).
- Alert on crash-rate regressions from playtests.
- Monitor token/FLOP cost per session ([41](../41-AI-Cost-Optimization-and-Enterprise-ROI/01-Overview.md)).

---

## Open-Source Starter Stack

| Need | Pick |
|------|------|
| Engine | Godot (free) |
| RL | Stable-Baselines3 |
| PCG | WFC port + L-systems |
| Dialogue | Local LLM + Chroma vector DB |
| QA | Custom bot harness |
| Edge | ONNX-quantized SLM |

---

## Build vs Buy

- **Buy/managed**: when you lack ML staff; NPC-dialogue services accelerate time-to-market.
- **Build**: when differentiation is the AI itself (competitive titles, unique emergent behavior).
- **Hybrid (common)**: managed LLM dialogue + in-house RL behavior + in-house PCG.

---

## Selection Guide

| If you need… | Start with… |
|--------------|-------------|
| Train NPC in editor | Unity ML-Agents + SB3 |
| Massively parallel sim | Isaac Gym + RLlib |
| Generate levels | WFC, then PCGML if data exists |
| Conversational NPC | LLM + RAG + guardrails |
| Ship art fast | Diffusion + art-direction pass |
| Auto QA | Bot harness in CI |
| On-device inference | Quantized SLM + Edge AI |

---

*Tooling covered. Finish with [05-Future-Outlook.md](05-Future-Outlook.md).*

# Core Topics in Physical AI: VLA Models, Sim-to-Real, Data Flywheels, and Safety

> **Description:** The technical and organizational core of Physical AI in 2026. Covers the Vision-Language-Action (VLA) model architecture, the sim-to-real transfer stack, the data flywheel that powers commercial Physical AI companies, the safety and alignment problem for embodied agents, and the safety/liability/regulation landscape.

---

## Table of Contents

1. [The Physical AI Stack in 2026](#1-the-physical-ai-stack-in-2026)
2. [Vision-Language-Action (VLA) Models](#2-vision-language-action-vla-models)
3. [Action Generation: From Tokens to Trajectories](#3-action-generation-from-tokens-to-trajectories)
4. [Sim-to-Real: The World Model as Training Substrate](#4-sim-to-real-the-world-model-as-training-substrate)
5. [The Data Flywheel: Demonstrations, Teleop, and Self-Play](#5-the-data-flywheel-demonstrations-teleop-and-self-play)
6. [Safety for Embodied Agents](#6-safety-for-embodied-agents)
7. [Alignment, Jailbreaks, and the Physical Dimension](#7-alignment-jailbreaks-and-the-physical-dimension)
8. [The Hardware Bridge: Sensors, Actuators, and Compute](#8-the-hardware-bridge-sensors-actuators-and-compute)
9. [The Deployment Stack](#9-the-deployment-stack)
10. [Comparison Tables of VLA Foundations](#10-comparison-tables-of-vla-foundations)
11. [Open Research Questions](#11-open-research-questions)

---

## 1. The Physical AI Stack in 2026

A working Physical AI deployment in 2026 looks like this:

```
┌──────────────────────────────────────────────────────────────┐
│                      HIGH-LEVEL TASK                          │
│           "Sort the laundry by color into three bins"         │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼ LLM planner (GPT-5, Claude 4.5, Qwen-3)
┌──────────────────────────────────────────────────────────────┐
│                  TASK DECOMPOSITION                           │
│   Step 1: Walk to basket. Step 2: Pick shirt. ... Step 17.   │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼ VLA policy (π₀, Helix, OpenVLA)
┌──────────────────────────────────────────────────────────────┐
│                ACTION CHUNKS (50–100 Hz)                      │
│   (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_open) × H        │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼ Whole-body controller (WBC)
┌──────────────────────────────────────────────────────────────┐
│                   JOINT COMMANDS (1 kHz)                      │
│   q_des[28] = inverse_dynamics(action, current_state)        │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼ Low-level motor controllers
┌──────────────────────────────────────────────────────────────┐
│                  ACTUATOR TORQUES (10 kHz)                    │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │   PHYSICAL    │
                       │     WORLD     │
                       └──────────────┘
                              │
                              ▼ Sensors (vision, force, IMU, audio)
                              │
                              ▼ VLA perception (encoder)
                              │
                              ▼ Closed loop
```

The same stack appears in every 2026 deployment, with minor variations. The "magic" is in the VLA box in the middle.

---

## 2. Vision-Language-Action (VLA) Models

A **VLA model** is a transformer that maps (image, language instruction, proprioceptive state) → (action chunk). The canonical 2026 architecture is:

```python
import torch
import torch.nn as nn

class VLA(nn.Module):
    """A 2026-style Vision-Language-Action model."""

    def __init__(self, vision_encoder, language_encoder, fusion, action_head):
        super().__init__()
        self.vision_encoder = vision_encoder      # SigLIP-400M or DINOv2-ViT-L
        self.language_encoder = language_encoder  # Qwen-2.5-7B or Llama-3-8B
        self.fusion = fusion                      # Cross-attention or Q-Former
        self.action_head = action_head            # Diffusion or flow-matching head

    def forward(self, images, instructions, proprio):
        # images: (B, T_v, 3, 224, 224) — last 4 egocentric frames
        # instructions: (B, T_l) token ids
        # proprio: (B, action_dim) joint positions

        # Step 1: encode each modality
        v_tokens = self.vision_encoder(images)         # (B, T_v, N_v, D)
        l_tokens = self.language_encoder(instructions) # (B, T_l, D)
        p_tokens = proprio.unsqueeze(1)                # (B, 1, D)

        # Step 2: cross-modal fusion
        fused = self.fusion(v_tokens, l_tokens, p_tokens)  # (B, N_total, D)

        # Step 3: decode to action chunk via diffusion
        action_chunk = self.action_head(fused)         # (B, H, action_dim)

        return action_chunk
```

### 2.1 Why VLAs work

VLAs work for the same reason LLMs work: **scale + data + next-token (or next-action) prediction**. The empirical evidence (2023–2026) is that a single VLA trained on 100+ manipulation tasks across 1000+ objects can:

- Generalize to **novel objects** at ~80% success rate.
- Follow **free-form natural-language instructions** without scripted templates.
- **Compose skills** ("pick up the X and put it on the Y" works even if that exact pairing is rare in training).
- Improve predictably with more data (scaling laws for actions look like scaling laws for text, but with shallower slopes).

### 2.2 The 2026 VLA roster

| Model | Lab | Backbone | Action head | Open? | Notable |
|-------|-----|----------|-------------|-------|---------|
| **π₀** | Physical Intelligence | PaliGemma-3B | Flow matching | Weights | First "general-purpose" VLA |
| **π₀-FAST** | Physical Intelligence | PaliGemma-3B | FAST tokenizer | Weights | 10× faster inference |
| **OpenVLA** | Stanford/Berkeley | Llama-2-7B + SigLIP | Continuous | ✅ Full | The "Llama of VLAs" |
| **OpenVLA-OFT** | Stanford | OpenVLA + optimizations | Continuous | ✅ Full | 25× faster than OpenVLA |
| **RT-2** | Google DeepMind | PaLI-X-55B | Discrete tokens | ❌ | First VLA |
| **RT-2-X** | Google DeepMind | PaLM-E-12B | Discrete | ❌ | Smaller variant |
| **Helix** | Figure AI | Custom 7B | Continuous | ❌ | First commercial humanoid VLA |
| **GR00T N1** | Nvidia | Cosmos-based | Diffusion | ✅ Weights | Cross-embodiment foundation |
| **Qwen-VLA** | Alibaba | Qwen-2.5-VL | Flow matching | ✅ Weights | Best multilingual VLA |
| **NEO VLA** | 1X Technologies | Custom 1.5B | Diffusion | ❌ | Optimized for home |
| **Optimus VLA** | Tesla | Custom (FSD-derived) | Diffusion | ❌ | In-house |
| **Skild Brain** | Skild AI | Custom | Continuous | ❌ | Cross-embodiment |
| **HPT (Heterogeneous Pre-trained Transformers)** | Stanford | Custom | Continuous | ✅ | One model, many embodiments |

> **Cross-reference:** `25-World-Models/01-Overview.md` §3 for the related LLM → World Model paradigm shift.

---

## 3. Action Generation: From Tokens to Trajectories

The hardest design choice in a VLA is **how to represent the action output**. There are three approaches in 2026:

### 3.1 Direct regression

Output a continuous vector at each timestep.

```python
action_head = nn.Linear(D, action_dim * chunk_horizon)
```

- **Pros:** Fast, simple, easy to train with MSE.
- **Cons:** Mode collapse (averages over multi-modal action distributions), poor for tasks with discrete choices ("left vs. right").

### 3.2 Diffusion policy

Generate the action chunk by denoising Gaussian noise, conditioned on the fused representation.

```python
class DiffusionActionHead(nn.Module):
    def __init__(self, dim, action_dim, chunk_horizon, num_diffusion_steps=100):
        super().__init__()
        self.dim = dim
        self.noise_predictor = nn.Sequential(
            nn.Linear(dim + action_dim * chunk_horizon + 1, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_dim * chunk_horizon),
        )
        self.num_diffusion_steps = num_diffusion_steps

    def forward(self, fused_tokens, noisy_actions, t):
        # Concatenate condition + noisy actions + timestep
        B = fused_tokens.shape[0]
        cond = fused_tokens.mean(dim=1, keepdim=True).expand(-1, noisy_actions.shape[1], -1)
        x = torch.cat([cond, noisy_actions, t.view(B, 1, 1).expand(-1, noisy_actions.shape[1], -1)], dim=-1)
        return self.noise_predictor(x)
```

- **Pros:** Handles multi-modal actions, stable training, state of the art on most manipulation benchmarks.
- **Cons:** Slow inference (10–100 denoising steps), memory-intensive.

### 3.3 Flow matching

A faster alternative to diffusion that learns a vector field between noise and data, often converging in 1–4 steps.

```python
class FlowMatchingActionHead(nn.Module):
    """Optimal-transport flow matching (Lipman et al. 2023) for actions."""

    def forward(self, fused_tokens, noise):
        # noise: (B, chunk_horizon, action_dim), standard Gaussian
        # The head learns the velocity field v_t = action - noise
        t = torch.rand(noise.shape[0], 1, 1, device=noise.device)
        # Linear interpolation
        x_t = (1 - t) * noise + t * self.target_action
        # Predicted velocity
        v_pred = self.velocity_net(x_t, t, fused_tokens)
        return v_pred
```

- **Pros:** 10–50× faster than diffusion, comparable or better quality.
- **Cons:** Newer, less mature tooling.

> **2026 consensus:** Flow matching (e.g. π₀) and discrete tokenization (FAST, π₀-FAST) are winning for new deployments, but diffusion remains the most reliable for safety-critical tasks.

---

## 4. Sim-to-Real: The World Model as Training Substrate

The **sim-to-real gap** is the central engineering challenge of Physical AI. The 2026 stack uses **three layers of simulation**:

### 4.1 Rigid-body physics (MuJoCo, Isaac, Genesis)

For manipulation and locomotion RL, you need fast, accurate contact dynamics. The 2026 leaders are:

| Simulator | Speed (relative) | Differentiability | License | Specialty |
|-----------|------------------|------------------|---------|-----------|
| **MuJoCo XLA / MJX** | 10,000× real-time | ❌ | Open source | GPU batched |
| **Nvidia Isaac Lab 2.0** | 50,000× real-time (RTX 5090) | ✅ (through Warp) | Free / commercial | Photorealism + RL |
| **Genesis** | 100,000× real-time | ✅ Fully | Open source | Differentiable physics |
| **ManiSkill 3** | 20,000× real-time | Partial | Open source | Manipulation benchmark |
| **Isaac Gym** | 30,000× real-time | ❌ | Free | GPU RL |
| **Bullet / PyBullet** | 200× real-time | ❌ | Open source | Legacy |

### 4.2 Neural world models (Cosmos, GAIA, Dreamer)

For long-horizon planning and rare-event simulation, you need a **learned** world model. See `25-World-Models/01-Overview.md` for the full taxonomy. In 2026:

- **Nvidia Cosmos** — the dominant industrial choice. Generates photorealistic video of plausible robot scenes, used for synthetic data and planning. Released 2025, open weights as of January 2026.
- **Dreamer V3** — the academic benchmark. Reaches human-level on Minecraft, Atari, and manipulation from pixels alone.
- **GAIA-1 / GAIA-2** (Wayve) — driving-specific world model, now applied to mobile manipulation.
- **Sora 2** (OpenAI) — general video model, can be prompted to simulate robot scenes with good prompt-engineering.

### 4.3 The "neural simulator" hybrid: Cosmos + Isaac Lab

The 2026 best practice (Nvidia, Tesla, Figure) is to use **rigid-body sim for control training** and **neural world model for data augmentation and rare-event rollouts**:

```python
# Pseudocode for a 2026 Physical AI training pipeline
import cosmos, isaaclab, torch

# 1. Train base policy in rigid-body sim
policy = train_ppo(
    env=isaaclab.make("FrankaPickPlace-v1", num_envs=4096, gpu=True),
    total_timesteps=10_000_000_000,  # 10B
)

# 2. Augment training with neural sim rollouts (rare events, edge cases)
for rare_event in ["cup_breaks", "human_walks_by", "lighting_changes"]:
    for _ in range(1000):
        initial_state = sample_from_cosmos(rare_event)  # photorealistic seed
        rollout = policy.rollout(env=initial_state, horizon=200)
        # Re-label with sim, add to replay buffer
        replay.add(rollout)

# 3. Fine-tune the policy on the augmented dataset
policy.finetune(replay, epochs=5)

# 4. Deploy on real robot
policy.deploy(robot="Franka", control_freq=50)
```

This is the **first time in robotics** that the dominant training paradigm is **data-driven, not hand-engineered**. See `17-Research-Frontiers-2026/03-LLM-Architectures-2026.md` for the broader LLM analog.

---

## 5. The Data Flywheel: Demonstrations, Teleop, and Self-Play

The Physical AI flywheel of 2026 has three data sources, in order of cost:

### 5.1 Human demonstrations (expensive, gold-standard)

Humans teleoperate robots to collect expert trajectories. The 2026 state of the art:

- **VR teleoperation:** Meta Quest 3, Apple Vision Pro, Manus gloves — 1 expert can collect 50–200 demos/hour.
- **Kinesthetic teaching:** physically guiding the robot arm — slower but higher quality.
- **Shadowing:** a human performs the task while the robot watches with cameras — 100s of demos/day, but noisier.

Industry consensus: **$5–$50 per usable demonstration** after filtering, and the best VLAs need **10,000–1,000,000 demonstrations** for a single task category.

### 5.2 Synthetic data (cheap, growing)

Using sim and neural world models to generate labeled trajectories. The 2026 ecosystem:

| Method | Cost per demo | Realism | Best for |
|--------|---------------|---------|----------|
| Isaac Lab scripted | $0.001 | Medium | Pre-training |
| Cosmos-generated | $0.05 | High | Edge-case augmentation |
| MimicGen-style from real seeds | $0.10 | Very high | Fine-tuning |
| Video-to-action (Sora → VLA) | $0.50 | Medium | New tasks |
| Diffusion-synthesized actions | $0.02 | Variable | Long-tail |

**The 2026 ratio:** Top Physical AI labs train on a mix of **~70% real, ~30% synthetic** data. The 30% synthetic is concentrated in long-tail events (fragile object handling, human presence, novel lighting).

### 5.3 Self-play and autonomous improvement (cheapest, hardest)

Robots practicing on themselves. The 2026 frontier:

- **Reinforcement learning from human feedback (RLHF) on robot trajectories:** humans rate videos of robot behavior; the VLA is fine-tuned to maximize preference score.
- **Self-distillation:** a V0 VLA generates a dataset, which trains a V1 VLA, etc. Convergence is not yet proven.
- **Cross-robot learning:** one robot's successes are uploaded to a shared model that all robots download. (Tesla's Optimus fleet, Figure's BMW pilot, 1X's home robots all do this.)

> **Cross-reference:** `17-Research-Frontiers-2026/09-Efficient-ML-Research.md` for the data-efficiency research front.

---

## 6. Safety for Embodied Agents

A digital agent that fails produces bad text. A physical agent that fails produces **broken bodies, broken objects, and broken buildings**. The 2026 safety stack has four layers.

### 6.1 Hardware safety (the last line of defense)

| Layer | Implementation | Failure mode caught |
|-------|----------------|---------------------|
| **E-stops** | Hardware kill switches, ISO 13849 | All software failures |
| **Force/torque limits** | Per-joint current limiters | Collision |
| **Compliance control** | Impedance/admittance control | Unexpected contact |
| **Speed limits** | ISO/TS 15066 (collaborative robots) | Pinch, crush |
| **Geofencing** | Workspace barriers (physical + LiDAR) | Workspace violation |
| **Human detection** | 360° cameras + person detection | Human-in-zone |

### 6.2 Software safety layer

```python
class SafetyShield:
    """A 2026 safety monitor that wraps a VLA policy."""

    def __init__(self, base_policy, world_model, human_detector, e_stop):
        self.policy = base_policy
        self.world_model = world_model
        self.detector = human_detector
        self.e_stop = e_stop

    def safe_act(self, obs, instruction):
        # 1. Check emergency stop
        if self.e_stop.triggered:
            return self.safe_hold_position(obs)

        # 2. Check human presence in safety zone (1.5m)
        humans = self.detector.detect(obs.image)
        if any(h.distance < 1.5 for h in humans):
            # Switch to a slower, more conservative policy
            action = self.policy.act(obs, instruction, speed_scale=0.25)
        else:
            # Normal operation
            action = self.policy.act(obs, instruction)

        # 3. Predict 1-second-ahead collision using the world model
        predicted_traj = self.world_model.rollout(obs, action, horizon=10)
        if self.detector.collision_imminent(predicted_traj, obs):
            return self.safe_hold_position(obs)

        # 4. Enforce joint limits, velocity limits, torque limits
        action = self.enforce_limits(action, obs.joint_state)

        return action
```

This pattern — **safety shield wrapping a learned policy** — is the 2026 industry default. See `18-Agent-Security-and-Trust/02-...` for the digital-agent analog.

### 6.3 Formal verification (emerging)

- **Reachability analysis:** pre-compute the set of states a robot can reach from the current state, and verify it stays within a safe set.
- **Lyapunov stability proofs:** for locomotion policies.
- **Shielded reinforcement learning:** mathematically guarantee the policy never enters an unsafe state.

Tools: **VerifAI** (Berkeley), **Hamilton-Jacobi reachability** (Stanford ASL), **Differential Dynamic Programming** for safety, **JuliaReach**, **AROC**.

### 6.4 Organizational safety (the missing layer)

The hardest safety problem is **organizational**. See `24-AI-Agent-Autonomy-Accountability/` for the full framework. Key 2026 concepts:

- **Operator's license** (proposed in EU AI Act amendment Q1 2026)
- **Reasonable operator duty of care** (8 components, see `24/01-Overview.md`)
- **Blast radius containment** (autonomy budget, kill switch)
- **Safety case documentation** (FAA/aviation analog, growing in robotics)
- **Pre-deployment red-teaming** (adversarial testing of the VLA)

---

## 7. Alignment, Jailbreaks, and the Physical Dimension

A 2026 concern specific to Physical AI: **the alignment failure modes are physically dangerous**.

| Attack | Mechanism | Impact | Real-world example |
|--------|-----------|--------|---------------------|
| **Adversarial patch** | Speckled sticker confuses vision encoder | Robot picks up wrong object | Berkeley 2024 |
| **Audio prompt injection** | Hidden voice command in ambient noise | Robot performs unintended action | CMU 2024 |
| **Pose spoofing** | Fake depth map tricks manipulation | Robot grasps air | — |
| **Sensor jamming** | Strong IR pulse blinds depth camera | Robot stops, or runs blind | Tesla 2023 |
| **Reward hacking** | RL agent finds shortcut that scores high but is dangerous | Robot learns to "fake" task | Multiple 2024–2025 |
| **Specification gaming** | Goal mis-specified, agent optimizes wrong thing | Famous: CoastRunners boat |
| **Physical jailbreak** | Painted markings on floor tell robot to bypass safety | Robot ignores geofence | (Research) 2025 |

> **2026 industry response:** Most safety-critical VLAs are now trained with **adversarial robustness** objectives and deployed with **multi-modal sensor fusion** (vision + LiDAR + tactile + force) so that no single sensor attack can compromise the system.

---

## 8. The Hardware Bridge: Sensors, Actuators, and Compute

### 8.1 Sensors

| Sensor | Type | Use case | 2026 leader |
|--------|------|----------|-------------|
| RGB camera | Passive vision | Object recognition | Sony IMX585 |
| Stereo camera | Depth | Manipulation | Intel RealSense D555 |
| Event camera | High-speed | Fast motion | Prophesee EVK4 |
| LiDAR | 3D | Navigation | Hesai XT32 |
| IMU | Inertial | Balance | Bosch BMI088 |
| Force/torque | Contact | Manipulation | ATI Mini45 |
| Tactile skin | Pressure | Grasping | SynTouch, Xela |
| Microphone | Audio | Voice commands | Respeaker |
| Thermal | Heat | Safety | FLIR Boson |

### 8.2 Actuators

- **Electric motors:** the dominant choice in 2026. Unitree, Figure, 1X, Tesla all use high-torque-density BLDC motors with quasi-direct-drive (QDD) or harmonic-drive reducers.
- **Hydraulic:** Boston Dynamics Atlas (electric now, since 2024), some legacy industrial arms.
- **Series elastic actuators (SEA):** older approach, still used in some rehab robots.
- **Pneumatic:** not competitive in 2026.

### 8.3 Edge compute

The dominant 2026 edge compute platform is **Nvidia Jetson Thor** (released 2025, 2,000 FP8 TFLOPS, 75W). Alternatives:

- **Qualcomm RB5** (lower power, less compute)
- **Apple Silicon** (M3 Ultra in research robots)
- **Intel Core Ultra** (some ManipHypothesis platforms)
- **Custom ASICs** (Tesla's in-house chip, rumored for Optimus V4)

See `23-Local-AI-Inference-Self-Hosting/06-Hardware-for-Local-Inference.md` for the broader local-AI hardware landscape.

---

## 9. The Deployment Stack

A 2026 humanoid deployment (Figure's BMW pilot is the canonical case) looks like:

```
┌────────────────────────────────────────────────────────┐
│                  CLOUD / FLEET MGMT                     │
│   - Over-the-air policy updates                        │
│   - Fleet telemetry aggregation                        │
│   - Cross-robot learning aggregation                   │
│   - Incident reporting and review                      │
└────────────────────────────────────────────────────────┘
                          ↕ 5G / Wi-Fi 6E
┌────────────────────────────────────────────────────────┐
│                EDGE COMPUTE (Jetson Thor)               │
│   - VLA inference (50–100 Hz)                          │
│   - World model rollouts (1 Hz lookahead)              │
│   - Safety shield                                      │
│   - Local memory + RAG                                 │
│   - Multi-modal sensor fusion                          │
└────────────────────────────────────────────────────────┘
                          ↕ EtherCAT / CAN
┌────────────────────────────────────────────────────────┐
│                REAL-TIME CONTROLLER (MCU)              │
│   - 1 kHz joint control                                │
│   - E-stop arbitration                                 │
│   - Motor current limiting                             │
│   - Battery management                                 │
└────────────────────────────────────────────────────────┘
                          ↕ SPI / CAN-FD
┌────────────────────────────────────────────────────────┐
│                ACTUATORS + SENSORS                     │
└────────────────────────────────────────────────────────┘
```

This is the **canonical Physical AI edge stack** of 2026. See `20-Agent-Infrastructure-and-Observability/` for the cloud/observability analog for digital agents.

---

## 10. Comparison Tables of VLA Foundations

| Property | π₀ | Helix | OpenVLA-OFT | GR00T N1 | Qwen-VLA | NEO VLA |
|----------|----|-------|-------------|----------|----------|---------|
| Backbone | PaliGemma-3B | Custom 7B | Llama-2-7B | Cosmos | Qwen-2.5-VL-7B | Custom 1.5B |
| Action head | Flow matching | Diffusion | Continuous | Diffusion | Flow matching | Diffusion |
| Inference latency (ms) | 25 | 35 | 12 | 40 | 28 | 18 |
| Generalization (held-out) | 78% | 63% | 71% | 75% | 74% | 68% |
| Open weights | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| Best for | General | Industrial | Research | Cross-embodiment | Multilingual | Home |
| Released | 2024-Q4 | 2025-Q1 | 2025-Q2 | 2025-Q4 | 2026-Q1 | 2025-Q3 |

> **Reading:** π₀ is the most general-purpose open VLA. Helix is the most production-deployed. OpenVLA-OFT is the most reproducible for research.

---

## 11. Open Research Questions

The 2026 open problems in Physical AI:

1. **Long-horizon autonomy.** A VLA that can perform a 30-minute task without a human taking over. Current best: ~5 minutes for warehouse picking.
2. **Cross-embodiment generalization.** A single model that works on Franka arms, Unitree humanoids, and mobile bases. GR00T and HPT are the leaders.
3. **Sample-efficient manipulation.** Learning a new skill from 10 demonstrations, not 10,000.
4. **Multimodal safety.** Proving that no combination of sensor attacks can defeat the safety shield.
5. **Self-improvement.** A robot that improves its own policy from its own experience, without human labeling.
6. **Verifiable autonomy.** A formal proof that the policy will not enter an unsafe state, even under adversarial conditions.
7. **Embodied common sense.** A VLA that knows you don't put a live cat in the oven, even if prompted.
8. **Energy efficiency.** 8-hour battery life on a humanoid doing useful work. Current best: ~3 hours for Optimus V2, ~4 hours for Figure 02.
9. **Cost per task.** Below $0.50 per pick in a warehouse (vs. ~$0.10 for a human in low-wage countries).
10. **Trust calibration.** When should the robot *ask* a human vs. proceed? The 2026 answer is "ask if confidence < 0.7", but this is heuristic.

---

*Cross-references: `25-World-Models/01-Overview.md` for the simulation foundation; `17-Research-Frontiers-2026/02-AI-Agents-Research.md` for VLA research; `24-AI-Agent-Autonomy-Accountability/05-Governance-Auditing-and-Regulatory-Frameworks.md` for governance; `23-Local-AI-Inference-Self-Hosting/06-Hardware-for-Local-Inference.md` for edge compute.*

*Next: `03-Technical-Deep-Dive.md` — a code-level walkthrough of training and deploying a VLA on a real robot.*

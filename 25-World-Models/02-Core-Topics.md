# World Models — Core Topics: Architectures, Training, Evaluation

> **Description:** A structured reference to the core technical topics in world models as of mid-2026. Covers the architectural taxonomy (JEPA family, generative-video family, latent-dynamics family, hybrid), training data sources, evaluation benchmarks, safety/alignment, and the convergence points. Designed for engineers, PMs, and researchers who need a precise map of the field.

---

## Table of Contents

1. [The Architectural Taxonomy of 2026](#1-the-architectural-taxonomy-of-2026)
2. [The JEPA Family (Non-Generative Latent)](#2-the-jepa-family-non-generative-latent)
3. [The Generative-Video Family (Sora / Genie / Veo)](#3-the-generative-video-family-sora--genie--veo)
4. [The Latent-Dynamics Family (Dreamer / IRIS / PlaNet)](#4-the-latent-dynamics-family-dreamer--iris--planet)
5. [The Driving-First Family (GAIA, Wayve, Waabi)](#5-the-driving-first-family-gaia-wayve-waabi)
6. [The Foundation-Model Family (Cosmos, Genie 3)](#6-the-foundation-model-family-cosmos-genie-3)
7. [The Hybrid Family](#7-the-hybrid-family)
8. [Training Data: What World Models Learn From](#8-training-data-what-world-models-learn-from)
9. [Evaluation and Benchmarks](#9-evaluation-and-benchmarks)
10. [Safety, Alignment, and Watermarking](#10-safety-alignment-and-watermarking)
11. [The Action Representation Problem](#11-the-action-representation-problem)
12. [The Reward Function Problem](#12-the-reward-function-problem)
13. [The Memory and Context Problem](#13-the-memory-and-context-problem)
14. [The Multi-Agent World Model Problem](#14-the-multi-agent-world-model-problem)
15. [Open Problems](#15-open-problems)

---

## 1. The Architectural Taxonomy of 2026

World models in 2026 are not a single architecture. They are a **family of architectures** that trade off different points in the design space. The taxonomy is best organized by what the model **outputs** and what the model **predicts**.

### 1.1 The Output / Prediction Matrix

| | Predict Latents (non-generative) | Predict Pixels (generative) | Predict Both (hybrid) |
|---|---|---|---|
| **Action-conditioned** | JEPA-2, V-JEPA 2, IRIS, Dreamer V3, GAIA-2 (latent head) | Sora 2, Genie 3, Veo 3, Cosmos, GAIA-1 | Cosmos 2.0, Sora 2 + planner |
| **Unconditioned (generation)** | (n/a — latents need action) | Sora 2 image2vid, Genie 2/3 image2world, Veo 3 | Hybrid foundation models |
| **Multi-agent conditioned** | Multi-agent JEPA (research) | Game-NPC WMs (Inworld) | Research frontier |

The dominant 2026 production architectures are:
- **JEPA-2** (action-conditioned, latent) — LeCun, Meta FAIR
- **Dreamer V3** (action-conditioned, latent) — DeepMind, open
- **Sora 2 / Veo 3** (action-conditioned, generative) — OpenAI, Google
- **Genie 3** (unconditioned image-to-world) — DeepMind
- **Cosmos 1 / 2** (foundation, hybrid) — NVIDIA

### 1.2 The Four Design Axes

Every world model is positioned along four axes:

1. **Generative vs. latent.** Does the model predict pixels (generative) or embeddings (latent)? Generative models are more grounded but more expensive. Latent models are more efficient but harder to interpret.
2. **Continuous vs. discrete actions.** Robotic joint torques are continuous. Tool calls are discrete. Most WMs support both via discretization.
3. **Single-agent vs. multi-agent.** Most WMs are single-agent. Multi-agent WMs must represent other agents' policies.
4. **Online vs. offline learning.** A WM can be trained once on a fixed corpus (offline) or continuously updated from a deployed agent's experience (online). Online WMs are powerful but unstable.

### 1.3 Architectural Common Ground

Despite the diversity, every world model in 2026 shares:

- **A state encoder** (E_o, E_a) that maps observations and actions to a shared latent space.
- **A dynamics module** (f) that predicts the next state from the current state and action.
- **A decoder / predictor** (D) that maps the predicted state to a usable output (pixels, embeddings, rewards).
- **A training objective** that compares predicted states to actual states.

The differences are in the **specific instantiations** of these components, the **scale**, and the **data**.

---

## 2. The JEPA Family (Non-Generative Latent)

JEPA stands for **Joint Embedding Predictive Architecture**. It was proposed by Yann LeCun in 2022 and has matured into a major research line by 2026, with V-JEPA 2 (Meta, 2025) and JEPA 2 (LeCun's newco, 2026) as the flagship.

### 2.1 Core Idea

Predict the **embedding** of a future observation, not the future observation itself. This sidesteps the cost and difficulty of pixel-level prediction while preserving the structure of the world.

```python
# Pseudocode for a single JEPA training step
def jepa_step(encoder, predictor, batch):
    # Two views of the world at time t and t+k
    z_t   = encoder(batch.context_frames)    # context embedding
    z_tk  = encoder(batch.target_frames).detach()  # target embedding, stop-grad

    # Predict the target embedding from the context embedding + action
    z_hat = predictor(z_t, batch.action)

    # Loss: make the predicted embedding close to the actual target embedding
    loss = smooth_l1_loss(z_hat, z_tk)
    return loss
```

The key design choices:

1. **Stop-gradient on the target encoder.** Prevents representational collapse (a known failure mode where the model learns to predict the same constant for everything).
2. **Latent-space regression loss.** Smooth L1 or VICReg-style. Crucially **not** a contrastive loss — JEPAs are trained to predict specific future latents, not just "future ≠ past."
3. **Action conditioning.** The predictor takes both the current embedding and the action; this forces it to learn action-conditioned dynamics.

### 2.2 Why Non-Generative?

LeCun's central argument: **prediction in pixel space is wasteful**. The number of bits in a 1080p video frame is ~6 Mb, but the number of bits of *useful* information about the world is far smaller. JEPA forces the model to compress observations into a small latent space and to predict the small latent of the next observation, not the full pixel array.

Trade-offs:

| Pro | Con |
|-----|-----|
| Computationally cheaper per training step | Harder to evaluate (no pixel-level signal) |
| More robust to irrelevant visual detail | Harder to interpret |
| Better for downstream planning (latents are already compact) | Requires careful design to avoid collapse |
| Aligns with human cognition (we don't see pixels) | Less impressive for demos / media |

### 2.3 V-JEPA 2 Architecture (Meta FAIR, 2025)

The 2025 release is the largest non-generative WM to date:

- **Backbone:** ViT-Large / ViT-Huge for vision, with a transformer predictor.
- **Training data:** ~1B+ video clips from public datasets (HowTo100M, Kinetics, Ego4D, plus internal).
- **Training compute:** thousands of GPU-days.
- **Downstream tasks:** object permanence, physical reasoning, action anticipation, video question answering.
- **Open source:** weights and inference code are public on Hugging Face.

### 2.4 When to Use JEPA

Use JEPA when:
- You need **sample efficiency** (training on small data).
- The downstream task is **planning** rather than generation.
- The evaluation is **semantic** (did the model understand the scene?) rather than **visual** (does the video look real?).
- You need to **decompress** the world, not re-render it.

Do NOT use JEPA when:
- You need to show the world to a human (use a generative model).
- You are building a content-creation tool (use Sora 2 or Veo 3).
- The user is paying for a media artifact, not a reasoning engine.

---

## 3. The Generative-Video Family (Sora / Genie / Veo)

The 2024-2026 wave of large video models (Sora 1/2, Veo 3, Genie 2/3, MovieGen) established generative video as a viable world model representation. The key idea: if a model can produce a coherent minute of video from a text prompt, it must have learned something about how the world works.

### 3.1 Core Architecture (2026 standard)

All frontier generative video models in 2026 share a common backbone:

```python
# Pseudocode for a diffusion-transformer video model
class VideoWorldModel(nn.Module):
    def __init__(self, ...):
        self.text_encoder = ...        # frozen LLM or CLIP-style
        self.diffusion_transformer = ...  # DiT-style
        self.video_vae = ...           # compresses video to latent

    def forward(self, text_prompt, video_frames=None, action=None):
        # Encode text and (optional) action as conditioning
        text_emb = self.text_encoder(text_prompt)
        cond = concat(text_emb, action_emb)

        # Diffusion on video latents
        if self.training:
            # Standard diffusion loss
            x0 = self.video_vae.encode(video_frames)
            t = sample_timestep()
            xt = add_noise(x0, t)
            eps_pred = self.diffusion_transformer(xt, t, cond)
            loss = mse(eps_pred, noise)
            return loss
        else:
            # Inference: denoise from pure noise
            x_T = sample_noise(...)
            for t in tqdm(scheduled_timesteps):
                eps = self.diffusion_transformer(x_t, t, cond)
                x_t = denoise_step(x_t, eps, t)
            return self.video_vae.decode(x_T)
```

Key components:
- **Diffusion Transformer (DiT).** Replaces U-Net with a transformer that denoises latent patches. Standardized by Sora 1 (2024).
- **3D video VAE.** Compresses a 60-frame video clip to a compact latent that the transformer can process.
- **Action adapter.** A small module that conditions the denoising on actions or control signals. This is what turns a "video generator" into a "world model."
- **Spacetime patches.** The transformer operates on spacetime patches (3D voxels), not just spatial patches. This is critical for temporal coherence.

### 3.2 Sora 2 (OpenAI, 2025-2026)

Sora 2 is the most-deployed generative world model in 2026. Key properties:

- **Resolution:** 1080p, up to 60 seconds, 24 fps.
- **Inputs:** text, image, video clip (for continuation).
- **Action conditioning:** via "storyboard" interface — discrete actions per clip.
- **Coherence:** maintains identity and physics for ~60 seconds.
- **Safety:** C2PA metadata + invisible watermarks on all outputs.
- **Scale:** estimated 10-50B parameters (OpenAI has not disclosed).

### 3.3 Genie 3 (DeepMind, 2026)

Genie 3 is interactive: the user can move through the world. Key properties:

- **Inputs:** single image, or text.
- **Outputs:** navigable 3D scene at 24+ fps.
- **Action space:** discrete 8-direction movement + jump/look.
- **Latency:** ~100-200 ms per frame on H100.
- **Differentiator:** maintains 3D consistency over long exploration.

### 3.4 Veo 3 (Google, 2026)

Veo 3 is the most photorealistic of the three. Key properties:

- **Resolution:** 4K, up to 2 minutes, 30 fps.
- **Inputs:** text, image, multi-image.
- **Outputs:** photorealistic video with native audio.
- **Integration:** native to YouTube, Google Photos, Vertex AI.
- **Differentiator:** native audio track generation (ambient sound, speech, music).

### 3.5 When to Use Generative-Video WMs

Use them when:
- The output is consumed by **humans** (entertainment, marketing, education).
- You need **demonstrations** of policies (synthetic expert data).
- The training corpus is **video-first** (driving, robotics, sports).
- The downstream task is **content creation** more than planning.

Do NOT use them when:
- Latency matters and you cannot afford 100ms+/step (use latent models).
- You need interpretable internal states (JEPA-style is better).
- The action space is high-dimensional and continuous (Dreamer is better).
- Compute is severely constrained.

---

## 4. The Latent-Dynamics Family (Dreamer / IRIS / PlaNet)

The Dreamer line of work (Hafner et al.) has been the workhorse of latent-dynamics world models since 2019. By 2026, Dreamer V3 is the most generally capable open-source world model in existence.

### 4.1 Core Idea

Learn a **recurrent state-space model (RSSM)** of the world, then use it to **imagine** rollouts that train a policy entirely in latent space.

```python
# Pseudocode for Dreamer-style training
class RSSM(nn.Module):
    """Recurrent state-space model — the core of Dreamer."""
    def __init__(self, ...):
        self.encoder = ...      # observation -> latent
        self.dynamics = ...     # (prev_state, action) -> next_state
        self.decoder = ...      # latent -> observation
        self.reward_predictor = ...

    def step(self, prev_state, action, observation=None):
        # Predict the prior over the next state (from dynamics)
        prior = self.dynamics(prev_state, action)

        if observation is not None:
            # Update the posterior using the actual observation
            embedding = self.encoder(observation)
            posterior = combine(prior, embedding)
        else:
            posterior = prior  # imagination mode

        return prior, posterior


def dreamer_train_step(world_model, actor, critic, batch):
    # 1. Update the world model from real data
    states = world_model.unroll(batch.observations, batch.actions)
    recon_loss = reconstruction_loss(world_model, batch.observations)
    reward_loss = reward_prediction_loss(world_model, batch.rewards)
    dynamics_loss = kl_divergence(posterior, prior)
    wm_loss = recon_loss + reward_loss + 0.1 * dynamics_loss
    wm_loss.backward()

    # 2. Imagine rollouts in latent space
    imagined_states = world_model.rollout(initial=states[-1], policy=actor, horizon=50)

    # 3. Train actor and critic on imagined rollouts
    actor_loss = -critic(imagined_states).mean()
    critic_loss = td_lambda_loss(imagined_states, world_model.reward_predictor)
    actor_loss.backward()
    critic_loss.backward()
```

### 4.2 Dreamer V3 (2024, DeepMind)

The version that achieved **diamond-level Minecraft from pixels** — a long-standing AI grand challenge. Key innovations:

- **Free bits.** A modification of the KL term that allows the latent to encode as much information as needed, preventing posterior collapse.
- **Symlog predictions.** Predicting `sign(x) * log(1 + |x|)` of rewards/values, which handles the dynamic range of Minecraft rewards.
- **Discrete latents.** Using categorical latents (like VQ-VAE) instead of Gaussian, which improves stability.
- **Robust scaling.** Same hyperparameters work across 150+ tasks with only the reward scale tuned.

### 4.3 IRIS (Micheli et al., 2022-2023)

IRIS is the "transformer all the way down" world model:

- A discrete autoencoder for observations.
- A transformer for dynamics over discrete latents.
- A transformer for the policy.
- Achieves state-of-the-art on Atari 100k and Minecraft.

### 4.4 PlaNet and the Latent Imagination Lineage

PlaNet (Hafner 2019) introduced latent imagination. The lineage:

```text
PlaNet (2019) — continuous RSSM, CEM planning
Dreamer V1 (2020) — actor-critic in latent space
Dreamer V2 (2021) — discrete latents, Atari
IRIS (2022) — discrete autoencoder + transformer
Dreamer V3 (2024) — robust scaling, Minecraft
```

### 4.5 When to Use Latent-Dynamics WMs

Use them when:
- The downstream task is **control** (robotics, game-playing, optimization).
- You have a **reward function** or can define one.
- The action space is **continuous** (joint torques, steering).
- You need **sample efficiency** (real-world interaction is expensive).
- You want **interpretable** internal states.

Do NOT use them when:
- The output needs to be shown to a human (use generative).
- The action space is purely discrete and very small (RL without a WM may suffice).
- You need to generate from a single image (use Genie 3).

---

## 5. The Driving-First Family (GAIA, Wayve, Waabi)

Driving is the most heavily funded application of world models, with ~$1.5B deployed in H1 2026 alone. The driving domain has produced several specialized WM architectures.

### 5.1 Why Driving is the Killer App

- **Safety-critical.** You cannot learn driving by trial-and-error in the real world; a single mistake kills someone. WMs let you crash a million virtual cars.
- **Sensor-rich.** Cameras, lidar, radar, GPS, IMU, HD maps. All of this can be conditioned on.
- **Structured action space.** Steering, throttle, brake. Continuous but bounded.
- **Rich training data.** Millions of hours of driving video.
- **Clear evaluation.** Closed-course testing, simulation benchmarks, public road miles.

### 5.2 GAIA-1 and GAIA-2 (Wayve)

GAIA was the first publicly-known "driving world model" at scale. Architecture:

- **Inputs:** text description of the scene + multi-camera video + action.
- **Outputs:** predicted next-second multi-camera video.
- **Training data:** Wayve's fleet of UK test vehicles, ~5% of all UK driving.
- **Use case:** synthetic data for downstream driving policy training.
- **GAIA-2** (2023) added multi-task conditioning (weather, time of day, traffic density).

### 5.3 Wayve AV2.0 (2026)

Wayve's full-stack 2026 product is built around their world model. The architecture:

```python
# Conceptual AV2.0 stack
class AV2Stack:
    def __init__(self, ...):
        self.world_model = GAIA2()  # foundation WM
        self.planner = DreamerStylePolicy(world_model)
        self.controller = ModelPredictiveController(planner)

    def step(self, sensor_data, hd_map):
        # 1. Encode current state from sensors
        state = self.world_model.encode(sensor_data, hd_map)

        # 2. Imagine candidate trajectories
        candidates = self.planner.imagine(state, num_candidates=64, horizon=8)

        # 3. Score candidates with the world model
        scores = self.world_model.score(candidates)

        # 4. Execute the highest-scored action
        action = candidates[scores.argmax()][0]
        return action
```

The key innovation is that **the entire driving policy is trained inside the world model**. There is no hand-coded behavior tree, no HD-map-only routing, no hand-engineered perception. The world model does everything.

### 5.4 Waabi's "GenAI Driver" (2026)

Waabi takes a similar approach but with stronger LLM integration. Their system:

- Uses an LLM to parse high-level driving instructions ("merge onto the M1").
- Uses a world model to imagine candidate trajectories.
- Uses a separate critic WM to score safety.
- Achieves human-expert level on closed courses and is in public road pilots.

### 5.5 Tesla FSD V13 (2026)

Tesla is the largest-scale deployer of a driving world model in 2026. FSD V13:

- Trained on ~10B miles of fleet data.
- End-to-end neural network with world-model-style architecture (Tesla has not disclosed details).
- "Supervised FSD" deployed to ~5M vehicles.
- The most-deployed WM-based system in history.

---

## 6. The Foundation-Model Family (Cosmos, Genie 3)

A 2025-2026 trend is the emergence of **foundation world models** — large, general-purpose, pre-trained world models that can be fine-tuned for downstream tasks (similar to how GPT is a foundation language model).

### 6.1 NVIDIA Cosmos

Cosmos is the first foundation world model suite. Released in 2024, updated to 2.0 in 2026.

- **Sizes:** 1B, 4B, 14B parameters.
- **Modalities:** video, multi-camera, action, text, lidar.
- **Pre-training data:** ~100M hours of video from public and licensed sources.
- **License:** open weights for research, commercial license for production.
- **Use cases:** synthetic data generation for robotics, autonomous driving, training RL policies, content creation.

Architecture:

```text
Cosmos = video VAE + diffusion transformer + action adapter + text adapter
        ↓
  Pre-trained on 100M hours of video
        ↓
  Fine-tuned for specific domains (driving, manipulation, navigation)
        ↓
  Generates synthetic data for downstream policy training
```

### 6.2 DeepMind Genie 3

Genie 3 is a foundation model for **interactive 3D worlds**. Key features:

- **Inputs:** single image, text, or video clip.
- **Outputs:** navigable 3D scene, ~24 fps.
- **Action space:** discrete game-style (move, look, jump).
- **Scale:** estimated 10B+ parameters.
- **Use cases:** game development, simulation, training embodied agents, content creation.

### 6.3 Sora 2 as Foundation Model

Sora 2 is increasingly used as a foundation model for downstream video tasks. The "Sora-as-foundation" ecosystem is growing: editing, re-styling, extension, interpolation.

### 6.4 Why Foundation WMs Matter

Foundation WMs change the economics:

- **Compute amortized across many downstream tasks.** A single Cosmos 14B can serve 1000 different robotics customers.
- **Data efficiency for downstream.** Fine-tuning a foundation WM on 100 hours of robot-specific data often beats training from scratch on 1000 hours.
- **Lower barrier to entry.** A small team can build a domain-specific world model by fine-tuning Cosmos, instead of training one from scratch.

---

## 7. The Hybrid Family

The most active research frontier in 2026 is the **hybrid model** — combining generative and latent objectives, or combining WM with LLM.

### 7.1 Why Hybrid?

- Generative models are good for grounding and demos.
- Latent models are good for planning and efficiency.
- LLMs are good for language, intent, tool use.
- A hybrid gets the best of all three.

### 7.2 Cosmos 2.0 Architecture (Hybrid)

Cosmos 2.0 is the first major hybrid release:

```python
class Cosmos2(nn.Module):
    def __init__(self, ...):
        # Generative head: predicts pixels
        self.video_vae = VideoVAE()
        self.diffusion_transformer = DiT()

        # Latent head: predicts abstract state for planning
        self.latent_encoder = ...
        self.latent_predictor = ...

        # LLM head: predicts language
        self.text_encoder = ...
        self.text_decoder = ...

    def forward(self, observation, action, text_query):
        # 1. Encode to shared latent
        z = self.latent_encoder(observation)

        # 2. Predict next latent
        z_next = self.latent_predictor(z, action)

        # 3. Decode to pixels (generative head)
        pixels = self.diffusion_transformer(z_next)

        # 4. Decode to language (LLM head)
        if text_query:
            text = self.text_decoder(z_next, text_query)

        return {"latent": z_next, "pixels": pixels, "text": text}
```

### 7.3 LLM + WM Hybrids (2026 frontier)

The "LLM plans, WM verifies" pattern is becoming standard:

```python
def llm_wm_hybrid(task_description, llm, world_model, environment):
    # LLM proposes a plan
    plan = llm.generate_plan(task_description)

    # WM simulates the plan
    for action in plan:
        predicted_next_state = world_model.predict(current_state, action)
        if world_model.is_unsafe(predicted_next_state):
            plan = llm.regenerate_plan(task_description, feedback="unsafe state predicted")
            break
        current_state = predicted_next_state

    # Execute the (now verified) plan
    return execute(plan, environment)
```

This pattern appears in:
- **Anthropic's** agent stack (rumored).
- **OpenAI's** Operator-style agent (uses a world model internally).
- **Google's** Gemini Robotics (LLM + WM).

See [Category 24](../24-AI-Agent-Autonomy-Accountability/01-Overview.md) for the safety implications of "verified plans."

---

## 8. Training Data: What World Models Learn From

The data is the moat. The 2026 frontier WMs are trained on datasets that no single lab could assemble five years ago.

### 8.1 The Major Data Sources

| Source | Type | Size | Notable |
|--------|------|------|---------|
| Ego4D | First-person video | 3,600 hours | Academic |
| HowTo100M | Instructional video | 100M clips | Academic |
| Kinetics-700 | Action-labeled video | ~650K clips | Academic |
| YouTube-8M | General video | 8M videos | Academic |
| Wayve fleet | Driving video | 50M+ hours | Commercial |
| Tesla fleet | Driving video | 10B+ miles | Commercial |
| Figure fleet | Robot teleop | 1M+ hours | Commercial |
| Meta Ego-Exo4D | Multi-view video | 1,400 hours | Academic |
| Something-Something | Action video | 220K clips | Academic |
| Cosmos training set | Mixed | 100M hours | Commercial |
| Open X-Embodiment | Robot manipulation | 60+ datasets, 22 embodiments | Academic/industrial consortium |
| DROID | Robot manipulation | 76K trajectories | Academic |
| Bridge Data | Robot manipulation | 7K trajectories | Academic |
| Internal gaming data | Gameplay | varies | Commercial |

### 8.2 The Three Data Properties That Matter

1. **Action labels.** For action-conditioned WMs, you need paired (state, action, next_state) tuples. Most video is unlabeled; the action must be inferred or measured.
2. **Diversity of environments.** A WM trained only on one type of kitchen won't generalize to other kitchens. The big labs win on data diversity.
3. **Embodiment coverage.** For robotics, you need data from many embodiments (different arms, grippers, mobile bases). The Open X-Embodiment consortium is the largest effort.

### 8.3 The Synthetic Data Flywheel

A 2026 production pattern is:
- Train a world model.
- Use it to generate synthetic data.
- Train downstream policies on the synthetic data.
- Use the downstream policies to collect more diverse real data.
- Retrain the world model on the larger, more diverse real data.
- Repeat.

This flywheel is the secret behind Figure 02's rapid improvement in 2026 (15% improvement per month) and Wayve's UK fleet expansion.

### 8.4 Licensing and Provenance

The major unresolved data question in 2026 is **licensing**. Most frontier WMs are trained on web video, much of which is copyrighted. Lawsuits are pending:

- **Getty v. Stability AI** (image, ongoing) — precedent for WMs.
- **NYT v. OpenAI** (text, 2023-2026) — may extend to video.
- **Disney / Universal v. Sora 2** (video, rumored 2026) — over training data and over generated output resembling copyrighted characters.

This is the most material legal risk for frontier WM companies. See [Category 21: AI Regulation & Antitrust](../21-AI-Regulation-Antitrust/01-Overview.md).

---

## 9. Evaluation and Benchmarks

How do you know if a world model is good? The 2026 evaluation landscape is still immature but is consolidating around a small set of benchmarks.

### 9.1 The Evaluation Pyramid

```text
                          ┌──────────────────────┐
                          │   Real-world deploy  │  ← ultimate test
                          └──────────┬───────────┘
                                     │
                          ┌──────────┴───────────┐
                          │  Task transfer bench │  ← Atari 100k, MineDojo
                          └──────────┬───────────┘
                                     │
                          ┌──────────┴───────────┐
                          │  Counterfactual      │  ← Physical-IQ, IntPhys
                          │  reasoning tests     │
                          └──────────┬───────────┘
                                     │
                          ┌──────────┴───────────┐
                          │  Image / video       │  ← FVD, IS, LPIPS
                          │  quality metrics     │
                          └──────────┬───────────┘
                                     │
                          ┌──────────┴───────────┐
                          │  Latent-space        │  ← Shape, topology
                          │  diagnostics         │
                          └──────────────────────┘
```

### 9.2 Video Quality Metrics

- **FVD (Fréchet Video Distance):** extension of FID to video. Standard.
- **IS (Inception Score):** older, still used.
- **LPIPS:** perceptual similarity for pairs.
- **SSIM, PSNR:** pixel-level, not great for semantic content.
- **VBench (2024):** decomposes video quality into 16 dimensions (subject consistency, background consistency, temporal flicker, motion smoothness, etc.). Standard for Sora / Veo evaluations.

### 9.3 Physical Reasoning Benchmarks

- **Physical-IQ (2024):** tests understanding of object permanence, solidity, continuity, immutability, gravity, inertia, etc.
- **IntPhys 2 (2024):** extension with more complex scenes.
- **CATER (2019):** object tracking and reasoning.
- **CLEVRER (2020):** counterfactual reasoning in physical scenes.
- **Physion (2020):** physical prediction in naturalistic scenes.

### 9.4 Task Transfer Benchmarks

- **Atari 100k (Dreamer V3 winning):** sample efficiency on Atari.
- **MineDojo / Minecraft Diamond:** long-horizon task in Minecraft.
- **DMControl:** continuous control from pixels.
- **OpenX-Embodiment:** cross-embodiment robotic manipulation.
- **Waymax / nuScenes:** driving simulation.
- **Procgen:** generalization in procedural environments.

### 9.5 Counterfactual and "What-If" Evaluation

This is the new frontier in 2026. The "ImagineBench" (2026) tests:

- Can the model predict what would happen if an action were changed?
- Can it generate consistent counterfactual trajectories?
- Can it answer "why did X happen?" with a simulation.

### 9.6 Safety Evaluation

- **WM-SafetyBench (2026):** does the world model generate unsafe content when prompted?
- **DeepfakeDetect 2026:** how easily can a generated video fool a deepfake detector?
- **CausalPathBench (2026):** does the model encode correct causal structure?

---

## 10. Safety, Alignment, and Watermarking

World models introduce novel safety concerns beyond what LLMs face. The 2026 landscape is fragmenting into a small set of approaches.

### 10.1 The Unique Safety Risks

| Risk | Description | Example |
|------|-------------|---------|
| **Deepfake at scale** | Photorealistic video of any person, any event, on demand | Political disinformation |
| **Unsafe action synthesis** | WM generates videos that depict illegal or dangerous acts | Bomb-making tutorials |
| **Imitative misbehavior** | WM trained on bad data learns to simulate bad behavior | Racist driving policy |
| **Reward hacking in imagination** | WM-learned policy finds a flaw in its own simulator | Robot learns to "satisfy" a fake reward |
| **Privacy violation** | WM memorizes and regenerates training data | Faces of private individuals |
| **Adversarial perturbation** | Small input changes cause large simulation errors | Self-driving car crashes when a single sticker is added |
| **Sim-to-real gap exploitation** | Policy is good in sim but bad in real | Robot that works in Cosmos but not in your kitchen |

See [Category 18: Agent Security & Trust](../18-Agent-Security-and-Trust/01-Overview.md) and [Category 24: AI Agent Autonomy & Accountability](../24-AI-Agent-Autonomy-Accountability/01-Overview.md) for the broader treatment.

### 10.2 Watermarking

All frontier generative WMs in 2026 ship with at least one of:

- **C2PA metadata** (cryptographically signed provenance).
- **Invisible pixel-level watermarks** (e.g., Google SynthID).
- **Latent-space signatures** (only the model owner can verify).

The EU AI Act (Article 50) requires disclosure of AI-generated content. Watermarking is the de facto compliance mechanism.

### 10.3 Red-Teaming

Frontier labs run internal red-teams for WMs. Common attack vectors:

- **Prompt injection** — text that causes the model to generate unsafe content.
- **Action injection** — actions that trigger unsafe state predictions.
- **Imitative misuse** — training the model on illegal content.
- **Reward hacking** — gaming the reward predictor.

External red-teams (e.g., the Coalition for Content Provenance and Authenticity, C2PA) publish annual reports.

### 10.4 Alignment

A 2026 open problem: **how do you align a world model**? An aligned LLM is one that follows human intent in language. An aligned WM is one that... what? Some proposals:

- **Trajectories should be physically plausible** (no flying cars unless physics says so).
- **Imitation of unsafe behavior should be penalized.**
- **Generated content should be honestly disclosed.**
- **The model should be "curious" rather than "deceptive"** (e.g., should not learn to "trick" a downstream policy).

There is no consensus in 2026 on what "alignment" means for a world model. The field is still negotiating the definition.

---

## 11. The Action Representation Problem

How an action is represented is one of the most consequential design choices in a world model. The 2026 spectrum:

### 11.1 Discrete Actions

```python
action_space = Discrete(n=1000)  # e.g., a 1000-token vocabulary of "actions"
```

Used in: game-playing, navigation, tool use, Genie 3 (8-direction movement).

**Pros:** tractable, easy to use with transformers, well-defined.

**Cons:** lossy for continuous control (joint torques, steering).

### 11.2 Continuous Actions

```python
action_space = Box(low=-1, high=1, shape=(7,))  # 7-DOF arm
```

Used in: robot manipulation, autonomous driving, drone control.

**Pros:** preserves full information.

**Cons:** harder to model, requires Gaussian head or normalizing flows.

### 11.3 Tokenized Actions

The LLM-style approach: represent any action as a sequence of discrete tokens.

```python
action_tokens = [0.3, 0.1, -0.5, 0.7]  # continuous
tokenized = action_tokenizer.encode(action_tokens)  # [342, 187, 991, 567]
```

Used in: hybrid LLM + WM systems, Sora 2 (storyboard actions).

**Pros:** unifies language and action in one token space; one model can do both.

**Cons:** quantization error; tokens don't have natural semantics.

### 11.4 Action-Free World Models

Some 2026 systems learn a world model without explicit action labels — they predict the next state from the previous state alone. Actions are inferred as "what changed between t and t+1."

Used in: video prediction without labels (Sora 1 in image-to-video mode).

**Pros:** no labeled data needed.

**Cons:** cannot be used for planning (no way to specify "what if I did X?").

---

## 12. The Reward Function Problem

For RL-style WMs, the reward function is the source of the objective. In 2026 there are roughly four approaches.

### 12.1 Hand-Coded Rewards

```python
def reward(state, action):
    # For a driving task
    return -collision_penalty(state) - 0.1 * jerk(state) - 0.01 * distance_to_goal(state)
```

**Pros:** precise, interpretable.

**Cons:** brittle, misspecified, expensive to design.

### 12.2 Learned Rewards (Inverse RL)

```python
# Learn reward from expert demonstrations
learned_reward = inverse_rl(expert_demos, environment_dynamics)
```

**Pros:** captures what humans actually value.

**Cons:** reward hacking, distribution shift.

### 12.3 Preference-Based Rewards (RLHF)

```python
# Learn a reward model from human preferences
def learn_reward_model(human_comparisons):
    # A is better than B, B is better than C, ...
    reward_model = train_on_preferences(human_comparisons)
    return reward_model
```

**Pros:** scales with human feedback.

**Cons:** human bias, sycophancy, gaming.

### 12.4 Foundation Reward Models (2026)

A 2026 trend: a single reward model pre-trained on many tasks, transferred to new tasks with minimal fine-tuning. Example: NVIDIA's "RM-1" foundation reward model.

**Pros:** amortizes reward-design cost.

**Cons:** still in research.

---

## 13. The Memory and Context Problem

World models need memory. But what kind?

### 13.1 Implicit Memory (in the Latent State)

The simplest: the entire history is compressed into the current state vector.

```python
state = encoder(history)  # one vector
```

**Pros:** simple.

**Cons:** limited by state dimensionality. Cannot handle long-term dependencies (e.g., "the password I told you 50 turns ago").

### 13.2 Episodic Memory (External Database)

A separate vector store of past states, retrievable by similarity.

```python
memory = VectorDB()  # Chroma, Weaviate, etc.
state = encoder(current_obs)
retrieved = memory.query(state, k=10)  # 10 most similar past states
world_model_input = concat(state, retrieved)
```

**Pros:** scales to long histories.

**Cons:** retrieval errors, not differentiable.

### 13.3 Structured Memory (Graphs, Trees)

Knowledge graph of entities, events, relations.

**Pros:** interpretable, queryable.

**Cons:** hard to integrate with neural nets.

### 13.4 Compression-Based Memory (HiPPO, S4)

Recent research: neural memory with provably good long-range recall (S4, Mamba, Hyena, RWKV).

**Pros:** efficient, O(L) in sequence length.

**Cons:** still research-grade for WMs.

---

## 14. The Multi-Agent World Model Problem

Most WMs are single-agent. But many real environments are multi-agent (driving, games, social simulation, markets). 2026 progress is uneven.

### 14.1 The Two Levels of Multi-Agent WMs

| Level | Description | Example |
|-------|-------------|---------|
| **Fixed-policy multi-agent** | WM models other agents' actions but treats their policies as fixed | Driving, where other drivers' policies are learned but not changed |
| **Adaptive-policy multi-agent** | WM models other agents and can predict how they will adapt to changes | Economic simulation, negotiation, game theory |

Most production systems are fixed-policy. Adaptive-policy is a research frontier.

### 14.2 The Joint Action Space

For n agents with discrete action space of size k, the joint action space is k^n. This is exponential. Practical approaches:

- **Mean-field:** approximate other agents as a single "average" agent.
- **Attention-based:** attend to other agents with weights.
- **Communication:** model explicit messages between agents.

### 14.3 The Recursive World Model Problem

If agent A's WM models agent B, and B's WM models A, you have a recursive simulation. This is computationally intractable in general but works in practice for short horizons.

---

## 15. Open Problems

The 2026 frontier. These are the active research questions.

### 15.1 Open: Long-Horizon Coherence

Current WMs are good for ~60 seconds (Sora 2) or ~30 minutes (Dreamer on Minecraft). What does it take to get to days? Years?

### 15.2 Open: Common Sense Physics

WMs learn the *statistics* of physics from data. They miss edge cases (a glass breaking in a specific way, a rope wrapping around an obstacle). Injecting symbolic physics priors is a 2026 research push.

### 15.3 Open: Sim-to-Real Transfer for Foundation WMs

A policy trained in Cosmos does not transfer perfectly to the real world. The sim-to-real gap is the biggest obstacle to embodied AI deployment. 2026 is seeing rapid progress but the problem is unsolved.

### 15.4 Open: World Models for Language

Can a world model help with reasoning about text? Some 2026 work (StoryGen, World-of-Thought) suggests yes, but the result is mixed.

### 15.5 Open: The Compute Footprint

Frontier WM training requires 10-50K H100-equivalent GPUs. This is ~1% of the world's AI compute. Can it be done with 1% of that? Open problem.

### 15.6 Open: Safety and Alignment

As above. No consensus in 2026.

### 15.7 Open: The Latent Space Should Be Interpretable

Current WMs are black boxes. The latent state is not human-readable. A 2026 research push is to make WMs "inspectable" — you can ask "what do you think is happening?" and get a useful answer.

### 15.8 Open: The Memory Scaling Problem

Current WMs have fixed context. Real-world deployment needs unbounded memory. The interaction between long-context transformers and WMs is still being figured out.

---

## Appendix: Cross-References to Other Library Categories

- **Foundations (01)**: [01-Foundations/06-Reinforcement-Learning.md](../01-Foundations/06-Reinforcement-Learning.md) — RL is the natural consumer of WMs.
- **LLMs (02)**: [02-LLMs/01-Transformer-Architecture.md](../02-LLMs/01-Transformer-Architecture.md) — LLMs and WMs share the transformer backbone.
- **Agents (03)**: [03-Agents/01-Agent-Architectures.md](../03-Agents/01-Agent-Architectures.md) — WMs are the planner layer for agents.
- **Advanced (06)**: [06-Advanced/01-Multimodal-AI.md](../06-Advanced/01-Multimodal-AI.md), [06-Advanced/02-Diffusion-Models.md](../06-Advanced/02-Diffusion-Models.md) — Multimodal and diffusion are sub-components of generative WMs.
- **Robotics (10)**: [10-Industry/03-AI-for-Robotics.md](../10-Industry/03-AI-for-Robotics.md) — Robotics is the dominant WM application.
- **Research Frontiers (17)**: [17-Research-Frontiers-2026/01-Overview.md](../17-Research-Frontiers-2026/01-Overview.md) — WMs are a major share of frontiers.
- **Agent Security (18)**: [18-Agent-Security-and-Trust/01-Overview.md](../18-Agent-Security-and-Trust/01-Overview.md) — WMs create new attack surfaces.
- **Agent Infrastructure (20)**: [20-Agent-Infrastructure-and-Observability/01-Overview.md](../20-Agent-Infrastructure-and-Observability/01-Overview.md) — WM-as-a-service is a new infra category.
- **Local AI (23)**: [23-Local-AI-Inference-Self-Hosting/01-Overview.md](../23-Local-AI-Inference-Self-Hosting/01-Overview.md) — Running small WMs locally is now possible.
- **Agent Autonomy (24)**: [24-AI-Agent-Autonomy-Accountability/01-Overview.md](../24-AI-Agent-Autonomy-Accountability/01-Overview.md) — A WM-equipped agent is more capable and more accountable.

---

*Next: [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md) — the math, loss functions, and training recipes.*

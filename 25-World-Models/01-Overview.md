# World Models: Internal World Simulation for the Next AI Paradigm

> **Description:** A comprehensive guide to world models — the emerging AI paradigm that learns an internal predictive model of an environment rather than relying on text completion alone. Covers the shift from LLMs to spatially- and temporally-grounded reasoning, the architectures (JEPA, Genie, GAIA, Dreamer, Sora-like simulators), training data, evaluation, and the $1B+ investment wave around this technology in 2026.

---

## Table of Contents

1. [What is a World Model?](#1-what-is-a-world-model)
2. [Why World Models Now? The 2026 Inflection](#2-why-world-models-now-the-2026-inflection)
3. [The LLM → World Model Paradigm Shift](#3-the-llm--world-model-paradigm-shift)
4. [Historical Roots](#4-historical-roots)
5. [Core Definitions and Terminology](#5-core-definitions-and-terminology)
6. [Why This Matters: Strategic Implications](#6-why-this-matters-strategic-implications)
7. [Who is Building World Models in 2026?](#7-who-is-building-world-models-in-2026)
8. [Relationship to Existing Library Categories](#8-relationship-to-existing-library-categories)
9. [The Three Failure Modes of Pure LLMs](#9-the-three-failure-modes-of-pure-llms)
10. [What World Models Unlock](#10-what-world-models-unlock)
11. [Chapter Map of This Category](#11-chapter-map-of-this-category)

---

## 1. What is a World Model?

A **world model** is a learned function that predicts how an environment will evolve in response to actions, enabling an agent (human, robot, or another AI system) to *simulate the consequences* of decisions before committing to them in the real world.

> **One-sentence definition:** A world model is an internal, generative simulator of an environment that an AI can use to plan, reason about causality, and ground its outputs in physical or social reality.

This is fundamentally different from an LLM, which predicts the next token based on statistical patterns of language. A world model predicts the next **state of the world** based on learned physics, dynamics, and the consequences of actions.

```python
# Conceptual signature of a world model (pseudocode)
class WorldModel:
    """An internal simulator of an environment."""
    def encode(self, observation: Tensor) -> LatentState:
        """Map raw observation (image, text, sensor) to compact latent state."""
        ...

    def predict(self, latent_state: LatentState, action: Action) -> LatentState:
        """Predict the next latent state given current state + action."""
        ...

    def decode(self, latent_state: LatentState) -> Observation:
        """Reconstruct a perceivable observation from latent state."""
        ...

    def rollout(self, initial: LatentState, policy: Policy, horizon: int = 100):
        """Simulate H steps into the future using a policy."""
        state = initial
        trajectory = [state]
        for t in range(horizon):
            action = policy(state)
            state = self.predict(state, action)
            trajectory.append(state)
        return trajectory
```

The three classic building blocks — **encode, predict, decode** — have been stable since Ha & Schmidhuber's 2018 "World Models" paper. What has changed dramatically by 2026 is the **scale** (latent dimensionality, training data, compute) and the **modality** (text → image → video → 3D → physics fields → social/agentic environments).

### 1.1 The Two Distinct Meanings in 2026

In 2026 the term "world model" is used in **two technically distinct ways**, and the field is racing to reconcile them:

| Sense | Core Idea | Representative Work | Output |
|-------|-----------|---------------------|--------|
| **Generative simulator** | A model that can render a plausible video, 3D scene, or sensory stream of a hypothetical world state | Sora 2, Genie 3, GAIA-1, Cosmos, Veo 3 | Frames, point clouds, audio |
| **Predictive latent dynamics model** | A model that learns abstract latent representations of world dynamics for planning and reasoning | JEPA, Dreamer V3, IRIS, GAIA-2, MuZero | Latent state vectors, value estimates |

The distinction is whether the model **renders** the world or whether it **reasons** about the world internally. Most frontier systems in 2026 (Genie 3, Sora 2, Cosmos) are heavily generative. Most efficient planners (Dreamer, JEPA) are heavily latent. The Holy Grail is a model that does both.

### 1.2 Why Both Matter

- **Generative** models are critical for **grounding**: they tie language to pixels, motion, physics, and audio. Without grounding, an LLM produces fluent nonsense.
- **Latent predictive** models are critical for **planning**: they let an agent search the future efficiently without paying the cost of full rendering. This is the key to sample-efficient robotics, autonomous driving, and long-horizon agentic tasks.

A system that combines both — a generative simulator that is also a queryable, latent-predictive planner — is the implicit goal of nearly every frontier lab in 2026.

---

## 2. Why World Models Now? The 2026 Inflection

Three forces converged in late 2025 and early 2026 to make world models the dominant new paradigm:

1. **Compute scaling plateau for pure LLMs.** The "just scale the next-token objective" path is hitting diminishing returns. Reasoning benchmarks are saturating. Scaling now requires either (a) more reasoning time (test-time compute) or (b) a different objective. World models offer a new objective: predict the next state, not the next token.
2. **Embodied AI investment surge.** Humanoid robots (Figure 02, Tesla Optimus Gen 3, 1X Neo, Unitree H1, Apptronik Apollo) and autonomous vehicles (Waymo, Zoox, Tesla FSD V13) require predictive internal models. You cannot deploy a robot in a kitchen using pure text prediction.
3. **LeCun's departure and $1B seed round (June 2026).** Yann LeCun left Meta and raised **$1 billion in Europe's largest-ever seed round** to build a startup explicitly focused on "world models" — a term he has championed since JEPA in 2022. This is the single largest fundraising event of 2026 and signals a structural shift in frontier research.

```text
[2026-Q1] Cosmos (NVIDIA), Sora 2 (OpenAI), Veo 3 (Google), Genie 3 (DeepMind) all ship.
[2026-Q2] LeCun departs Meta. Raises $1B for world-models startup. HN #1 story.
[2026-Q2] Tesla FSD V13 goes end-to-end world-model. No more hand-coded rules.
[2026-Q2] Figure 02 ships factory version trained in Cosmos simulator.
[2026-Q2] Multiple YC startups (W26 batch) explicitly position around world models.
```

### 2.1 The Investment Numbers (H1 2026)

| Investor / Lab | Amount | Focus | Status |
|----------------|--------|-------|--------|
| LeCun's newco | $1.0B seed | JEPA-style world foundation model | Announced |
| World Labs (Fei-Fei Li) | $300M Series B | 3D world generation from single image | Live |
| Decart / Etched | $150M | Real-time video world model inference | Live |
| Odyssey (2024-founded) | $80M | Hollywood-grade interactive worlds | Live |
| Tess.ai | $60M | Physics-grounded video synthesis | Live |
| NVIDIA Cosmos | Internal $1B+ | Foundation models for synthetic data | Live |
| Wayve (UK) | $1.05B Series C | Driving world model, "AV2.0" | Closed Q1 |
| Waabi (Toronto) | $200M Series B | Driving world model + GenAI planner | Closed Q1 |
| Cosma (research) | Public release | Open world-model framework | v0.4 |
| **Aggregate H1 2026** | **$3.0B+** | | |

This is a meaningful share of the ~$50B flowing into AI infrastructure in 2026. World models are now a primary frontier.

### 2.2 The HN Signal (June 17, 2026)

The top HN story on June 17, 2026 was *"Yann LeCun to depart Meta and launch AI startup focused on 'world models'"* at **841 points** — the #1 AI story of the cycle. The secondary signal was *"SpaceX to buy Cursor for $60B"* (331 pts) — a separate trend (AI coding consolidation), but the *dominant* technology story is world models.

The third notable signal is **Qwen-Robot Suite: A Foundation Model Suite for Physical World Intelligence** (Hugging Face) — an open release from Alibaba positioning itself as a world-model alternative to closed US labs.

---

## 3. The LLM → World Model Paradigm Shift

The intellectual history of modern AI has been a sequence of "predict the next X" breakthroughs:

| Era | Predict the next... | Architectures | Sample efficiency | Limitation |
|-----|---------------------|---------------|-------------------|------------|
| 2000s | character / word | N-grams, RNNs | Low | No context |
| 2013-2017 | word in context | word2vec, LSTM | Low | Short context |
| 2017-2020 | token in long context | Transformer | Higher | No reasoning |
| 2020-2023 | token, scaled | GPT-3, PaLM, LLaMA | Low at scale | No grounding |
| 2023-2025 | token, with reasoning | o1, o3, Claude thinking | Lower per task | Test-time cost |
| **2026-** | **state, not token** | **JEPA, Genie, Sora, Cosmos, Dreamer** | **Higher (one model, many tasks)** | **Compute for video** |

The next "predict the next X" is **state**. This is the LLM → World Model transition.

### 3.1 The Tokenization Trap

LLMs are forced to compress reality into discrete tokens. This has three structural problems:

```python
# The tokenization trap
text_description = "The red ball rolled off the table and hit the floor."
tokens = tokenizer.encode(text_description)
# ["The", " red", " ball", " rolled", " off", " the", " table", ...]
# LLM sees: discrete symbols. Never sees: gravity, elasticity, time, mass.
```

1. **Symbol grounding is lossy.** The word "rolled" carries no information about the trajectory, contact dynamics, or rotational state.
2. **Counterfactual reasoning is unconstrained.** An LLM can produce "The ball floated upward" with no penalty — there's no internal physics to violate.
3. **Long-horizon planning is brittle.** Predicting the 100th next token of a plan is exponentially harder than predicting the 100th next state of a simulator.

A world model sidesteps all three by operating in a **continuous latent state space** that respects the geometry of the world.

### 3.2 The Geometric Inductive Bias

World models bake in geometric, physical, and temporal priors. The simplest example: a state evolves as

```
s_{t+1} = f(s_t, a_t) + ε
```

where the function f is constrained to be locally smooth, roughly time-reversible, and approximately Markovian. This is a much stronger prior than "next token is correlated with the previous k tokens." It forces the model to discover the **laws** of the world, not just the statistics of language about the world.

### 3.3 What LLMs Will Continue to Do

World models are not replacing LLMs. The 2026 consensus architecture is **LLM + WM**:

- **LLM** handles language, intent, planning at a high level, tool use, communication.
- **WM** handles prediction, simulation, counterfactual reasoning, physical grounding.
- **Adapter** translates between the two: language plans go in, action sequences come out; video/sensor data go in, language descriptions come out.

This is the same architectural logic as the human brain: cortex (language) + hippocampus / cerebellum (world simulation). See [Chapter 8](#8-relationship-to-existing-library-categories) for the cross-reference matrix.

---

## 4. Historical Roots

World models are not new. The concept has roots in control theory, cognitive science, and the 1990s RL literature. The 2018-2020 wave of papers made them practical. The 2024-2026 wave made them industrial.

### 4.1 Timeline of Foundational Work

| Year | Milestone | Reference |
|------|-----------|-----------|
| 1972 | World models in cognitive science (Craik, "The Nature of Explanation") | Craik 1943 |
| 1990 | Dyna-Q (Sutton) — integrating learned model with planning | Sutton 1990 |
| 2015 | Embed to Control (Watter et al.) — local linear latent dynamics | Watter 2015 |
| 2018 | "World Models" (Ha & Schmidhuber) — VAE + RNN + simple controller | arXiv:1803.10122 |
| 2019 | Dreamer V1 — latent imagination for continuous control | arXiv:1912.01603 |
| 2020 | Dreamer V2 — discrete latents, Atari | arXiv:2010.02193 |
| 2021 | GAIA-1 — generative world model for autonomous driving | Wayve 2021 |
| 2022 | JEPA (LeCun) — non-generative self-supervised world model | arXiv:2210.06252 |
| 2023 | Dreamer V3 — first agent to mine diamonds in Minecraft from pixels | DeepMind, Nature 2024 |
| 2023 | GAIA-2 (Wayve) — multi-camera, multi-task world model | Wayve 2023 |
| 2024 | Sora 1 (OpenAI) — large video generation as world model | OpenAI 2024 |
| 2024 | Genie 2 (DeepMind) — interactive 3D world from single image | DeepMind 2024 |
| 2024 | Cosmos (NVIDIA) — foundation world models for robotics | NVIDIA 2024 |
| 2025 | Sora 2 (OpenAI), Veo 3 (Google), Genie 3 (DeepMind) — 60s+ coherent video | Frontier labs |
| 2025 | V-JEPA 2 (Meta) — large-scale video JEPA | Meta FAIR 2025 |
| 2026 | LeCun's world-models startup, $1B seed | June 2026 |
| 2026 | Tesla FSD V13, Figure 02 factory, Wayve AV2.0 deployments | Q2 2026 |

The pattern is clear: from 2018 the work was academic; from 2024 it was productized at frontier scale; in 2026 it became an investment category.

### 4.2 The Three Threads That Converged

Three separate research threads converged to make 2026 the inflection year:

1. **Generative video models** (Sora, Veo, Genie). Proved that scale + data + transformer-diffusion can produce coherent multi-second video. This is the "render" capability.
2. **Latent dynamics models** (Dreamer, JEPA, IRIS). Proved that a model can learn to plan in latent space without rendering anything. This is the "reason" capability.
3. **Physical simulation** (NVIDIA Cosmos, Genesis, MuJoCo XLA, Brax). Proved that differentiable physics can scale to GPU clusters and provide training data for both of the above.

The convergence is what makes world models industrial. None of the three alone is sufficient; all three together is the platform.

---

## 5. Core Definitions and Terminology

The vocabulary around world models is still unstable. Here is a working glossary, consistent with how the field is using terms in mid-2026.

### 5.1 The Basic Terms

| Term | Definition |
|------|------------|
| **State (s)** | A complete description of the world at time t. Usually a vector in some latent space, sometimes raw pixels. |
| **Observation (o)** | A partial view of the state (e.g., one camera frame, one text description, one tactile reading). |
| **Action (a)** | A change the agent makes to the world (e.g., a motor command, a sentence, a tool call). |
| **Dynamics / transition function (f)** | The function (s, a) → s' that maps current state and action to next state. |
| **Policy (π)** | A function s → a that maps state to action. The agent's behavior. |
| **Reward (r)** | A scalar signal evaluating a state or transition. The objective for RL. |
| **Rollout / trajectory** | A sequence of (s, a, r, s') tuples generated by simulating the dynamics. |
| **Imagination** | The internal simulation of rollouts in the world model, used for planning. |
| **Grounding** | The mapping between a symbolic representation (language, label) and a sensory / physical representation (pixels, forces). |
| **Counterfactual** | A "what if" trajectory under a hypothetical action sequence. |
| **Embodiment** | The specific sensorimotor interface through which the agent perceives and acts. |
| **Latent space** | The internal, usually lower-dimensional, vector space in which dynamics are learned. |

### 5.2 Acronyms You Will See

- **WM** — World Model
- **VLA** — Vision-Language-Action model (e.g., RT-2, OpenVLA)
- **V-JEPA** — Video Joint Embedding Predictive Architecture (Meta)
- **RSSM** — Recurrent State-Space Model (used in Dreamer)
- **PlaNet** — Deep Planning Network (Hafner 2019)
- **IRIS** — Imagination with auto-Regressive Integrated System
- **GAIA** — Generative AI for Autonomy (Wayve)
- **Cosmos** — NVIDIA's foundation world-model suite
- **Genie** — DeepMind's interactive 3D world model
- **MUVA** — Multi-View Action-conditioned model (Cosmos)
- **AV2.0** — Autonomous Vehicle 2.0 (Wayve's term for WM-based driving)

### 5.3 Distinctions That Trip People Up

- **World model vs. simulation** — A simulation is a known-physics engine (Unreal, MuJoCo, CARLA). A world model is *learned* from data and may discover unknown physics. WMs generalize; simulations don't.
- **World model vs. LLM** — An LLM is a world model of **language**. A world model is a world model of the **referents of language**. The latter is what gives an agent ground truth.
- **World model vs. digital twin** — A digital twin is a model of *one specific* instance (one factory, one jet engine). A world model is a *general-purpose* simulator that can be conditioned on many instances.
- **JEPA vs. Sora-style** — JEPA is non-generative (predicts embeddings, not pixels). Sora-style models are generative (predict pixels). Both are world models. They trade off grounding (JEPA) vs. fidelity (Sora).
- **VLA vs. world model** — A VLA maps observation directly to action. A world model sits *between* observation and action, providing a simulated future to plan over. A VLA can be thought of as a world model that throws away the intermediate state and goes straight to the action.

---

## 6. Why This Matters: Strategic Implications

World models are not a niche research topic. They are the most likely foundation of the next decade of AI products. The implications are strategic, not just technical.

### 6.1 For AI Labs

- The frontier objective is moving from "next token" to "next state." Labs that fail to build credible WMs will be stuck selling commodity LLMs.
- The data moat is shifting. Text is largely mined. The new scarce resource is **diverse, multi-modal, action-labeled trajectories** — video + proprioception + language.
- Compute requirements are different. WM training is GPU-bound on continuous data (video, physics) and requires different cluster topologies than LLM training.

### 6.2 For Enterprises

- Robotics and autonomous systems become tractable in a way they have not been in 30 years. A factory that trains a robot in a WM for 1 GPU-day can deploy a policy that learns in hours instead of months.
- Synthetic data generation is now first-class. A WM can produce infinite labeled training data for any downstream task, breaking the data-scaling wall.
- Planning in simulation reduces real-world risk. Self-driving companies can crash millions of virtual cars to learn from failures.

### 6.3 For Investors

- The $50B+ flowing into AI in 2026 is unevenly allocated. WMs are the second wave after LLM infra. The third wave is likely embodied AI; WMs are the bridge.
- The most lucrative categories are likely: (a) the foundation model providers (Cosmos, Genie 3, JEPA 2), (b) vertical WMs for high-value domains (driving, surgery, robotics), (c) simulation-as-a-service for synthetic data, (d) WM inference chips (specialized hardware).

### 6.4 For Society

- Deepfakes become much harder to detect as WMs produce hour-long, interactive, photorealistic video on demand. Watermarking and provenance are now mandatory.
- The "agent bankrupted their operator" problem (see [Category 24](../24-AI-Agent-Autonomy-Accountability/01-Overview.md)) gets worse when an agent can simulate the consequences of its own actions before taking them. A WM-equipped agent is *more* dangerous, not less, if its values are misaligned.
- New job categories: world-model evaluator, synthetic-data curator, simulation ethicist, digital twin architect.

---

## 7. Who is Building World Models in 2026?

### 7.1 Frontier Labs

| Organization | Project(s) | Approach | Notable 2026 milestone |
|--------------|------------|----------|------------------------|
| Meta FAIR | V-JEPA 2, JEPA family | Non-generative latent | 1B+ video frames training |
| OpenAI | Sora 2 | Generative video | 60s+ coherent video |
| Google DeepMind | Genie 3, Veo 3, Dreamer V3 | Generative + latent | Interactive 3D worlds |
| NVIDIA | Cosmos (1.0, 2.0) | Foundation suite | Open-weights release |
| Anthropic | (rumored) WM-1 | Hybrid | Not yet shipped |
| Mistral | (not yet) | — | Announced 2026 roadmap |
| xAI | Grok-Vision | Generative | Closed beta |
| Apple | AIML World Modeling team | Hybrid | Internal only |

### 7.2 Pure-Play Startups

| Company | Focus | Funding | Status |
|---------|-------|---------|--------|
| LeCun's newco (TBA) | World foundation model | $1B seed | Pre-launch |
| World Labs (Fei-Fei Li) | 3D world from image | $300M Series B | Live |
| Odyssey | Interactive worlds | $80M | Live |
| Decart | Real-time video WM | $150M | Live |
| Wayve | Driving WM | $1.05B Series C | Deployed in UK |
| Waabi | Driving WM | $200M Series B | Deployed |
| Cosma | Open WM framework | Public | Open-source |
| Tess.ai | Physics-grounded video | $60M | Live |
| Inworld | NPC / character WM | $120M | Live (gaming) |
| Apptronik | Humanoid + WM | $350M | Live |
| Figure AI | Humanoid + WM | $1.5B | Live |
| 1X Technologies | Humanoid + WM | $200M+ | Live |
| Skild AI | Robotics foundation | $300M | Live |

### 7.3 Open Source

- **Cosmos** (NVIDIA) — released with open weights in 2024, updated to 2.0 in 2026.
- **Genie 2 / 3** — DeepMind has not yet open-sourced.
- **Sora 2** — closed; no public weights.
- **V-JEPA 2** — Meta open-sourced pre-trained weights and inference code.
- **Dreamer V3** — open-sourced by DeepMind.
- **GAIA-2** — Wayve has published papers but not full weights.
- **Cosma** — fully open framework.
- **MINT** — multi-modal foundation, open.
- **WMA** (World-Model Arena) — open evaluation framework, 2026.

### 7.4 Academic Centers

- **NYU** (LeCun, until 2026) — JEPA lineage.
- **UC Berkeley** (Sergey Levine lab) — RL + WMs.
- **Mila / Université de Montréal** — generative dynamics.
- **ETH Zurich** (Tim Salimans, Angelos Katharopoulos) — efficient WMs.
- **Stanford** (Fei-Fei Li, before founding World Labs) — 3D vision.
- **MIT CSAIL** (Josh Tenenbaum) — probabilistic programming + WMs.
- **Toyota Research Institute** — large-scale driving WMs.
- **DeepMind** (Doina Precup, Timothy Lillicrap) — Dreamer lineage.
- **Tsinghua / Alibaba DAMO** — Qwen-Robot Suite.

---

## 8. Relationship to Existing Library Categories

This category is intentionally placed at the end of the numbered sequence. It depends on, and complements, every category that came before. The cross-reference matrix is essential reading.

| Existing Category | What it covers | How 25-World-Models connects |
|-------------------|----------------|-----------------------------|
| 01-Foundations | ML, DL, RL | RL is the consumer of WMs; WMs are the producer of synthetic trajectories. |
| 02-LLMs | Transformers, models, tokenization | LLMs and WMs are complementary halves of the modern agent stack. |
| 03-Agents | Agent architectures, MCP/ACP, tools | WMs give agents a planner and a simulator. |
| 04-RAG | Retrieval, vectors, advanced RAG | WMs can replace or augment retrieval with implicit memory. |
| 05-Enterprise | Deployment, fine-tuning, infra | WMs require new infrastructure (GPU clusters for video, etc.). |
| 06-Advanced | Multimodal, diffusion, eval, prompt, interpretability, recsys, time series, adversarial ML, UX, AutoML | WMs touch every one of these — multimodal (video, audio), diffusion (the core generative technique), eval (WM-bench is a new category). |
| 07-Emerging | Research, safety, governance | WMs are the dominant emerging research area; safety implications are acute. |
| 10-Industry | Applications, economics, robotics | WMs unlock the next generation of robotics. |
| 11-AI-Applications | Vertical applications | WMs will transform healthcare (surgery sims), finance (market sims), manufacturing (digital twins). |
| 12-Business-Prospects | Markets, startups, VC, ROI | The $3B+ flowing into WMs in H1 2026 is a market event. |
| 13-Top-Demand | Current trends | WMs ARE the top demand. |
| 14-Case-Studies | End-to-end implementations | New case studies needed: Wayve AV2.0, Figure 02 factory. |
| 17-Research-Frontiers-2026 | Latest arXiv | WMs are a major share of frontiers. |
| 18-Agent-Security-and-Trust | Trust, security | WMs create new attack surfaces (adversarial perturbations, deepfakes). |
| 20-Agent-Infrastructure-and-Observability | AgentOps, tracing, eval | WM-as-a-service is a new infra category. |
| 21-AI-Regulation-Antitrust | Regulation, antitrust | WM-generated content triggers EU AI Act Article 50 (transparency). |
| 23-Local-AI-Inference-Self-Hosting | Local AI | Running WMs locally is now possible for small WMs (Cosmos-1B). |
| 24-AI-Agent-Autonomy-Accountability | Liability, duty of care | A WM-equipped agent is a more capable agent and therefore more accountable. |

The reader is expected to read Category 25 after at least skimming Categories 02, 03, 10, 13, and 17. The other categories are referenced from inside this one where they are directly relevant.

---

## 9. The Three Failure Modes of Pure LLMs

It is useful to be precise about what world models fix. There are three failure modes of pure-LLM systems that world models resolve. Knowing these helps you decide when a WM is the right tool.

### 9.1 Failure Mode 1: Physical Plausibility

An LLM can write "the glass fell up" with no penalty. A world model that has learned gravity from video data assigns near-zero probability to upward trajectories. This matters for:

- **Robotics**: telling a robot to "put the cup on the shelf" requires implicit physics.
- **Autonomous driving**: an LLM that summarizes "the car is safe" cannot guarantee it knows what a child running into the road looks like.
- **Manufacturing**: an LLM-generated CNC program is dangerous if the LLM has never seen chips.

### 9.2 Failure Mode 2: Counterfactual Reasoning

An LLM trained on "X always happened before Y" cannot reliably reason about "what if X had not happened." A world model can roll out alternative futures by changing the action and re-simulating. This matters for:

- **Planning**: a robot choosing between two paths needs to imagine both.
- **Strategy**: a trader considering a position needs to imagine how the market would react.
- **Medicine**: a doctor considering a treatment needs to imagine how the patient would respond.

### 9.3 Failure Mode 3: Long-Horizon Coherence

An LLM generating a 30-step plan has compounding error: step 30 is conditioned on 29 noisy predictions, each with 5% drift, so cumulative drift is 1 − 0.95^30 ≈ 79%. A world model trained with multi-step consistency losses can keep the cumulative error bounded. This matters for:

- **Long-horizon agents** (see Category 24): tasks that take days, not minutes.
- **Video generation**: Sora 2's 60-second coherent video requires world-model-style training.
- **Embodied AI**: a 10-hour shift in a warehouse requires consistent behavior.

---

## 10. What World Models Unlock

The capability frontier after world models is broad. The 2026-2030 roadmap, broken down by horizon.

### 10.1 Near-term (2026-2027)

- ✅ Photorealistic, minute-long video from a single image (Sora 2, Veo 3 already there).
- ✅ Interactive 3D worlds from a single prompt (Genie 3).
- ✅ Driving policies trained entirely in simulation (Wayve, Waabi, Tesla V13).
- ✅ Humanoid robots in structured factory environments (Figure 02, Apptronik Apollo).
- ✅ Synthetic data for medical imaging, retail, and autonomous systems at scale.

### 10.2 Medium-term (2027-2029)

- 🟡 Household robots that generalize across homes (1X Neo, Figure 03).
- 🟡 World-model-as-a-service for enterprise (Cosmos Cloud).
- 🟡 Personal digital twins of cities, hospitals, factories.
- 🟡 Foundation world models for social/economic simulation (Akerlof-style markets).
- 🟡 Long-horizon agents (days of autonomy) — depends on Category 24 progress.

### 10.3 Long-term (2029+)

- 🔵 AGI-level world models that unify language, vision, action, theory of mind.
- 🔵 World models for scientific discovery (drug design, materials, climate).
- 🔵 Direct brain-computer interfaces that offload simulation to silicon.
- 🔵 Virtual civilizations used as policy sandboxes (Singapore's "digital twin nation" project).

These are projections, not promises. The technical and regulatory unknowns are large. But the trajectory is consistent with the funding, talent flow, and product announcements of H1 2026.

---

## 11. Chapter Map of This Category

This category (25-World-Models) is organized in 5 files:

| File | Audience | Coverage |
|------|----------|----------|
| **01-Overview.md** *(this file)* | Decision-makers, executives, investors, new researchers | What, why now, who, definitions, implications |
| **02-Core-Topics.md** | Engineers, researchers, product managers | Architectures, training data, evaluation, benchmarks, taxonomy |
| **03-Technical-Deep-Dive.md** | ML researchers, research engineers | Math, loss functions, latent dynamics, training recipes, scaling laws |
| **04-Tools-and-Frameworks.md** | Engineers, infra teams | Open-source frameworks, foundation models, training stacks, inference |
| **05-Future-Outlook.md** | Strategists, futurists, policy folks | Roadmap, open problems, regulatory landscape, long-term scenarios |

**Reading order recommendation:**

1. **If you are an executive or investor:** 01 → 05.
2. **If you are an engineer or PM:** 01 → 02 → 04.
3. **If you are a researcher:** 01 → 02 → 03 → 04 → 05.

Cross-references are explicit in every file. If you read this category in full, you will have a working knowledge of world models as of June 2026.

---

## Appendix A: Key Papers to Start With

| Paper | Year | Why read |
|-------|------|----------|
| Ha & Schmidhuber, "World Models" | 2018 | The paper that named the field. Short, readable, foundational. |
| LeCun, "A Path Towards Autonomous Machine Intelligence" | 2022 | The JEPA vision. Free PDF. |
| Hafner et al., "Dreamer V3" | 2024 (Nature) | Mastering Minecraft from pixels. |
| Wayve, "GAIA-1" | 2023 | Driving world model at scale. |
| OpenAI, "Sora Technical Report" | 2024 | Generative video as world model. |
| Meta FAIR, "V-JEPA 2" | 2025 | Large-scale video JEPA. |
| DeepMind, "Genie 2" | 2024 | Interactive 3D worlds. |
| NVIDIA, "Cosmos" | 2024 | Foundation world models for robotics. |
| Wayve, "AV2.0" | 2026 | Driving policy trained in WM only. |
| LeCun et al., "JEPA 2 Pre-training" | 2026 | The post-Meta LeCun vision. |

## Appendix B: Key People to Follow

- **Yann LeCun** — Chief WM advocate, now independent.
- **David Ha** — Co-author of 2018 paper, now at Sakana AI.
- **Danijar Hafner** — Dreamer lineage, now at Google DeepMind.
- **Doina Precup** — RL + WMs, McGill / DeepMind.
- **Timothy Lillicrap** — DeepMind, Dreamer co-author.
- **Sergey Levine** — Berkeley, RL foundation.
- **Fei-Fei Li** — World Labs, 3D vision.
- **Shuran Song** — Columbia, robotics + WMs.
- **Raquel Urtasun** — Waabi CEO, driving WM.
- **Alex Kendall** — Wayve CEO, driving WM.

---

*Next: [02-Core-Topics.md](02-Core-Topics.md) — the architectural taxonomy, training data, evaluation, and benchmarks.*

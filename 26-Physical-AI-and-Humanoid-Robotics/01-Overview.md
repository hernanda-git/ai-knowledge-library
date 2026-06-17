# Physical AI & Humanoid Robotics: The Embodied Frontier of 2026

> **Description:** A comprehensive guide to Physical AI and humanoid robotics — the convergence of foundation models, world models, and mechatronics that is producing the first commercial wave of general-purpose robots. Covers the Vision-Language-Action (VLA) paradigm, the major platforms (Tesla Optimus, Figure 03, 1X NEO, Unitree G1, Apptronik Apollo, Agility Digit, Sanctuary Phoenix), the simulation-to-real (sim-to-real) stack (Nvidia Isaac, Cosmos, MuJoCo, Genesis, Isaac Lab), safety/liability for embodied agents, and the emerging physical-AI software stack as of June 2026.

---

## Table of Contents

1. [What is Physical AI?](#1-what-is-physical-ai)
2. [Why Physical AI Now? The 2026 Inflection](#2-why-physical-ai-now-the-2026-inflection)
3. [From Digital Agents to Embodied Agents](#3-from-digital-agents-to-embodied-agents)
4. [Historical Roots: From Cybernetics to Humanoids](#4-historical-roots-from-cybernetics-to-humanoids)
5. [Core Definitions and Terminology](#5-core-definitions-and-terminology)
6. [Why This Matters: Strategic Implications](#6-why-this-matters-strategic-implications)
7. [Who is Building Physical AI in 2026?](#7-who-is-building-physical-ai-in-2026)
8. [Relationship to Existing Library Categories](#8-relationship-to-existing-library-categories)
9. [The Three Failure Modes of Pure Language Models in the Physical World](#9-the-three-failure-modes-of-pure-language-models-in-the-physical-world)
10. [What Physical AI Unlocks](#10-what-physical-ai-unlocks)
11. [Chapter Map of This Category](#11-chapter-map-of-this-category)

---

## 1. What is Physical AI?

**Physical AI** is AI that perceives, reasons about, and acts upon the **real, three-dimensional, time-bound physical world** through sensors and actuators. It is the union of three previously-separate stacks:

1. **Foundation models** — the large pretrained backbones (LLMs, VLMs, video models) that provide general reasoning and perception.
2. **World models** — the predictive latent dynamics models (see `25-World-Models/`) that let an agent *imagine* the consequences of an action before committing to it.
3. **Embodiment** — the mechanical, electrical, and software stack that turns model outputs into motor torques, gripper forces, and locomotion commands.

> **One-sentence definition:** Physical AI is foundation-model intelligence that has been grounded to a body — a robot, a vehicle, a drone, or any sensorimotor system that interacts with the physical world.

The single most consequential shift in 2026 is that the foundation models of the digital world (GPT, Claude, Gemini, Qwen-VL, Llama) are being **extended with action heads** to produce Vision-Language-**Action** (VLA) models that emit robot trajectories directly.

```python
# Conceptual signature of a Physical AI / VLA policy
class PhysicalAIPolicy:
    """A foundation-model policy that maps (image, instruction, state) to action."""

    def __init__(self, vision_encoder, language_encoder, action_decoder):
        self.vision_encoder = vision_encoder      # e.g. SigLIP, DINOv2
        self.language_encoder = language_encoder  # e.g. Qwen-2.5, Llama-3
        self.action_decoder = action_decoder      # e.g. diffusion action head

    def perceive(self, image: np.ndarray, instruction: str, proprio: np.ndarray) -> "ActionChunk":
        """Map raw sensor + language + proprioception to a chunk of motor commands."""
        visual_tokens = self.vision_encoder(image)
        text_tokens = self.language_encoder(instruction)
        # Cross-modal fusion in latent space, then decode to action sequence
        action_chunk = self.action_decoder(
            visual_tokens, text_tokens, proprio
        )
        return action_chunk  # shape: (chunk_horizon, action_dim)

    def act(self, image, instruction, proprio):
        chunk = self.perceive(image, instruction, proprio)
        return chunk[0]  # execute the first action; replan every step
```

This is the new abstraction layer for robotics. Where classical robotics wrote **PID controllers → motion planners → task scripts** by hand, Physical AI writes **data → foundation model → policy**.

### 1.1 The Three Subfields of Physical AI in 2026

| Subfield | Domain | Representative Models | Body |
|----------|--------|------------------------|------|
| **Manipulation** | Grasping, tool use, assembly | OpenVLA, RT-2, π₀, Helix | Stationary or mobile arms |
| **Locomotion** | Walking, balancing, running | Dreamer V3, ASIML, G1 policies | Quadrupeds, humanoids |
| **Mobile autonomy** | Driving, flying, navigating | Waymo's EMMA, Tesla FSD V13, Cosmos-Drive | Cars, drones, AMRs |

A full Physical AI stack usually combines all three: a humanoid that walks to a shelf (locomotion), picks a part (manipulation), and navigates the warehouse (mobile autonomy).

### 1.2 Why Physical AI is Different from Embodied AI

The terms are often used interchangeably, but in 2026 they have a useful distinction:

- **Embodied AI** is the broader academic discipline — any AI with a body (including simulated agents, RL in MuJoCo, Habitat).
- **Physical AI** in 2026 is the **commercial / industrial wave** — AI products that are sold, deployed, and operated in real human environments at scale. The term was popularized by Nvidia's Jensen Huang and has become the industry-standard label.

This category covers both, but emphasizes the commercial/industrial frontier.

---

## 2. Why Physical AI Now? The 2026 Inflection

Four conditions converged in 2024–2026 to make Physical AI a deployable product, not a research demo:

### 2.1 Foundation models scaled past the perception bottleneck

Until 2023, robot perception was bespoke: object detectors, pose estimators, depth networks, all hand-tuned. By 2024, vision-language models like **SigLIP, DINOv2, and SAM-2** had generalized so well that a single pretrained backbone could segment, classify, and reason about novel objects in zero-shot fashion. By 2025, **Vision-Language-Action (VLA)** models — RT-2, OpenVLA, π₀, Helix — demonstrated that the same transformer architecture could map pixels + instructions to robot actions without task-specific code.

### 2.2 World models made planning sample-efficient

Classic RL needs millions of real-world rollouts. With **world models** (see `25-World-Models/`), a robot can simulate 10,000 hours of manipulation in 1 hour of GPU time. Nvidia's Cosmos (released 2025) and Genesis (open-source, 2024) made this practical. The result: a single policy can be trained in simulation in days, then transferred to a real robot via sim-to-real.

### 2.3 Hardware reached a price-performance inflection

The **Unitree G1** launched at $16,000 in 2024; by mid-2026 comparable humanoids (Figure 03, 1X NEO, Apptronik Apollo) are priced at $30K–$50K for early production units, with roadmaps to $20K by 2027. **Tesla Optimus** is targeted at under $20K. At these prices, the **payback period for replacing a $40K/year warehouse worker is <18 months** in the US.

### 2.4 LLMs became reliable task planners

A 2024 insight: the same LLM that writes code can also **decompose a high-level instruction into a sequence of robot skills**. "Sort the laundry by color" → walk to basket → pick shirt → place in red pile. This is now standard in Physical AI stacks (SayCan, PaLM-E, Code-as-Policies, and the 2026 generation of VLA planners).

> **Industry quote (June 2026):** "We are about three years from the ChatGPT moment for physical AI agents." — *Rev Lebaredian, VP of Omniverse & Simulation Technology, Nvidia*

---

## 3. From Digital Agents to Embodied Agents

The library already covers digital agents in `03-Agents/`, `20-Agent-Infrastructure-and-Observability/`, and `24-AI-Agent-Autonomy-Accountability/`. The 2026 step is taking these digital-agent patterns and **adding a body**.

| Pattern | Digital Agent (2024) | Embodied Agent (2026) |
|---------|----------------------|------------------------|
| Tool use | `web_search()`, `code_exec()` | `move_to(x,y)`, `grasp(object)`, `pour(cup)` |
| Memory | Conversation history, vector store | Conversation history + **egocentric video memory** + **topological map** |
| Planning | ReAct, Plan-and-Execute, Tree-of-Thoughts | Same, plus **model-predictive control** over continuous actions |
| Failure modes | Hallucination, infinite loops | Hallucination + **physical destruction** |
| Liability | Operator liability (see `24`) | Operator liability + **product liability** + **negligence** + **workers' comp** |
| Evaluation | Pass@k on a benchmark | Pass@k + **safety rate** + **collision rate** + **human-intervention rate** |

The last three rows are why Physical AI is fundamentally more complex than digital agents: **the agent can now break the world and the people in it**.

---

## 4. Historical Roots: From Cybernetics to Humanoids

Physical AI is not new. What is new is the **scale and generalization** of the underlying models.

| Era | Paradigm | Bottleneck |
|-----|----------|------------|
| 1940s–1960s | Cybernetics (Wiener, Grey Walter) | Analog control, no learning |
| 1970s–1980s | Shakey the Robot, Stanford Cart | Hand-written rules |
| 1990s | Behavior-based robotics (Brooks) | Subsumption, no unified model |
| 2000s | SLAM + modular pipelines (ROS) | Brittle, task-specific |
| 2010s | Deep RL for control (DQN, PPO, SAC) | Sample-inefficient, sim-to-real gap |
| 2018 | **World Models** (Ha & Schmidhuber) | First learned dynamics models |
| 2022 | RT-1, BC-Z (Google) | First end-to-end visuomotor policies |
| 2023 | **RT-2, PaLM-E** | First foundation-model policies |
| 2024 | **OpenVLA, π₀, Helix, GR00T** | Open VLA ecosystem emerges |
| 2025 | **Nvidia Cosmos, Genesis, Isaac Lab 2.0** | World-model sim-to-real |
| 2026 | **Commercial deployment** | Figure 03, Optimus V3, 1X NEO in factories |

The library's existing category `25-World-Models/` covers 2018–2025 in detail. This category focuses on **2026's commercial deployment** of these models on real bodies.

---

## 5. Core Definitions and Terminology

| Term | Definition | Library Reference |
|------|------------|-------------------|
| **VLA (Vision-Language-Action)** | A foundation model that maps (image, text) → robot action | `02-LLMs/`, `25-World-Models/` |
| **Sim-to-real** | Training a policy in simulation, then deploying on a real robot | `17-Research-Frontiers-2026/` |
| **Imitation learning (IL)** | Learning a policy from expert demonstrations | `03-Agents/` |
| **Behavior cloning (BC)** | Supervised learning of actions from observations | `03-Agents/` |
| **DAgger** | Interactive imitation learning (Ross et al., 2011) | — |
| **Diffusion policy** | Representing the action distribution as a diffusion denoising process | `17-Research-Frontiers-2026/` |
| **Flow matching** | A faster alternative to diffusion for action generation | `17-Research-Frontiers-2026/` |
| **Ego-centric video** | First-person video from the robot's onboard cameras | `04-RAG/` (multimodal) |
| **Proprioception** | The robot's internal sense of its own joint positions, velocities, and forces | — |
| **Force/torque sensing** | Per-joint or end-effector force measurement | — |
| **MPC (Model Predictive Control)** | Receding-horizon control using a learned dynamics model | `25-World-Models/` |
| **Whole-body control (WBC)** | Coordinated control of all joints (esp. humanoids) | — |
| **Zero-shot generalization** | Performing a task never seen in training data | `02-LLMs/` |
| **Foundation policy** | A large pretrained policy (the VLA analog of a foundation model) | `25-World-Models/` |
| **Teleoperation** | A human remotely controlling the robot to collect demos | — |
| **AVR (Active Volume Reduction)** | Industry term for "operator takes over" in humanoid fleets | — |
| **Embodied agent** | An agent with a body (the academic term) | `03-Agents/` |
| **Physical AI** | The commercial/industrial term (Nvidia-coined) | This category |
| **Humanoid** | A robot with a bipedal, human-like form factor | This category |
| **Dexterous hand** | A multi-fingered, high-DoF end-effector | This category |
| **VLA-0 / π₀** | "General-purpose" VLA foundation models (2025) | `17-Research-Frontiers-2026/` |

---

## 6. Why This Matters: Strategic Implications

### 6.1 The labor substitution question

Humanoid robots are the first embodied technology designed to **substitute for human labor at the task level**, not the motion level. A 6-DoF industrial arm can weld a car chassis; a humanoid can do whatever a human worker can do in any factory that hasn't been retooled for robots.

- **Total addressable market:** Goldman Sachs (2026) estimates the humanoid TAM at **$38 billion by 2035**.
- **Deployment projections:** Figure, Tesla, Agility, Apptronik, 1X combined have **~150,000 units on pre-order** as of mid-2026.
- **Productivity impact:** A 2026 McKinsey study estimates 20–30% productivity gains in early adopter warehouses (Amazon, BMW, Figure's own facility).

### 6.2 The geopolitical dimension

The race for Physical AI is increasingly viewed as a **third industrial revolution** with national-security implications. See `21-AI-Regulation-Antitrust/` for related analysis.

- **US:** Nvidia's GPUs, Tesla's Optimus, Figure's US manufacturing.
- **China:** Unitree, Fourier Intelligence, Xpeng's IronX, Galbot, the **Qwen-Robot Suite** foundation model (released April 2026, 117 HN points).
- **Japan:** Toyota Research Institute, Honda, Sony.
- **EU:** PAL Robotics, Karlsruhe, German AI initiative.

> **Quote (Eric Schmidt, June 2026):** "China could dominate the Physical AI future if the US doesn't treat this as a strategic priority."

### 6.3 The software stack is the new moat

Building a humanoid is now a solved-mechatronics problem (Unitree, Fourier have commoditized the platform). The **defensible value is in the model** — the VLA, the world model, the data flywheel, and the safety layer. This is the same dynamic that turned smartphone hardware into a commodity and moved value to iOS/Android.

The library's existing categories `02-LLMs/`, `17-Research-Frontiers-2026/`, and `25-World-Models/` cover the algorithmic foundations. This category covers the **deployment stack** that turns them into commercial products.

---

## 7. Who is Building Physical AI in 2026?

### 7.1 Foundation model providers

| Lab | Model | Type | Open? |
|-----|-------|------|-------|
| Google DeepMind | RT-2, RT-H, OpenVLA-OFT | VLA | Partial |
| Physical Intelligence | π₀, π₀-FAST | VLA | Partial |
| Stanford / Berkeley | OpenVLA, OpenVLA-OFT | VLA | ✅ Open |
| Tesla | Optimus VLA, FSD-V13 transfer | VLA | ❌ Closed |
| Figure AI | Helix (VLA) | VLA | ❌ Closed |
| Nvidia | GR00T N1, Cosmos | Foundation + World | ✅ Open |
| Apple | MLLM (rumored) | VLA | ❌ Closed |
| **Alibaba** | **Qwen-Robot Suite** | VLA + World | ✅ Open |
| Meta | V-JEPA 2, LeCun's new startup | World | Partial |
| Toyota Research Institute | Large Behavior Model | VLA | Partial |
| 1X Technologies | NEO VLA | VLA | ❌ Closed |
| Skild AI | Skild Brain | Foundation | ❌ Closed |
| Covariant | Brain-1V | Foundation | ❌ Closed |
| General Trajectory (YC W25) | Dexterous manipulation FM | VLA | ❌ Closed |

### 7.2 Hardware platforms (humanoid form factor)

| Robot | Manufacturer | Height | DoF | Price (2026) | Status |
|-------|--------------|--------|-----|---------------|--------|
| **Tesla Optimus V3** | Tesla | 5'8" | 28 | $20K–$30K (target) | Internal deployment, 2026 |
| **Figure 03** | Figure AI | 5'6" | 41 | $30K–$50K | BMW, 2nd fleet customer |
| **1X NEO** | 1X Technologies | 5'5" | 26 | $20K–$30K | Home pilots |
| **Apptronik Apollo** | Apptronik | 5'8" | 32 | $50K+ | Mercedes-Benz, Google |
| **Agility Digit** | Agility Robotics | 5'9" | 16 | $250K+ (Gen 1) | Amazon GXO, 2nd gen in development |
| **Unitree G1** | Unitree | 4'2" | 43 | **$16,000** | Mass market, EDU/research |
| **Unitree H1** | Unitree | 5'11" | 19 | $90K | Industrial pilots |
| **Fourier GR-1** | Fourier Intelligence | 5'5" | 40 | TBD | China market |
| **XPeng IronX** | XPeng | 5'8" | 60+ | TBD | Factory pilots |
| **Galbot G1** | Galbot | 5'3" | 40+ | TBD | China retail pilots |
| **Sanctuary Phoenix** | Sanctuary AI | 5'7" | 20 | TBD | General-purpose manipulation |
| **Boston Dynamics Atlas (electric)** | Boston Dynamics | 5'9" | 28 | TBD | Hyundai factories |

### 7.3 Sim-to-real platforms

| Platform | Vendor | License | Specialty |
|----------|--------|---------|-----------|
| **Nvidia Isaac Lab 2.0** | Nvidia | Free / commercial | GPU-accelerated RL |
| **Nvidia Cosmos** | Nvidia | Open weights | World foundation model |
| **MuJoCo XLA (DeepMind)** | Google DeepMind | Open source | Fast physics, MJX on GPU |
| **Genesis** | Genesis team (CMU et al.) | Open source | Ultra-fast differentiable sim |
| **Isaac Sim** | Nvidia | Commercial | Photorealistic rendering |
| **ManiSkill 3** | UCSD | Open source | Manipulation benchmark |
| **RoboCasa** | Stanford | Open source | Household tasks |
| **Habitat 3.0** | Meta | Open source | Mobile manipulation |
| **SAPIEN** | USC | Open source | Articulated objects |
| **RLBench / RLBench2** | Imperial | Open source | 100 manipulation tasks |
| **Behavior-1K / B1K** | Stanford | Open source | Long-horizon household |
| **MimicGen** | NVIDIA | Open source | Synthetic demo generation |

### 7.4 Component / subsystem providers

- **Actuators:** Unitree, Maxon, Harmonic Drive, Schaeffler
- **Dexterous hands:** Robot Era, Shadow Robot, Psyonic, Clone Robotics
- **Sensors (vision):** Intel RealSense, Stereolabs ZED, Lucid, Prophesee event cameras
- **Sensors (tactile):** SynTouch, Contactile, Xela Robotics
- **Compute:** Nvidia Jetson Thor, Qualcomm RB5, Apple Silicon edge
- **Batteries:** CATL, Samsung SDI (high-cycle fast-charge)

---

## 8. Relationship to Existing Library Categories

| Existing Category | Relationship to Physical AI |
|-------------------|----------------------------|
| `01-Foundations/` | Provides the ML, RL, and optimization background. |
| `02-LLMs/` | LLMs are the language backbone of every VLA. |
| `03-Agents/` | Digital-agent patterns (ReAct, memory, planning) are reused in VLAs. |
| `04-RAG/` | Egocentric video RAG is a 2026 research frontier (see `04-RAG/03-...`). |
| `05-Enterprise/` | Physical AI is the next enterprise-AI category after RAG/agents. |
| `06-Advanced/` | Fine-tuning, alignment, evaluation methods apply directly. |
| `07-Emerging/` | Physical AI is the canonical "emerging" topic of 2026. |
| `11-AI-Applications/` | Manufacturing AI is the first Physical AI application. |
| `14-Case-Studies-Real-World-Projects/` | BMW, Amazon, Figure's own factory are canonical cases. |
| `17-Research-Frontiers-2026/` | Covers VLA, world-model, diffusion-policy research. |
| `18-Agent-Security-and-Trust/` | Physical-agent security (e.g. adversarial patches, jailbreaks). |
| `20-Agent-Infrastructure-and-Observability/` | Physical-agent telemetry is a new infra layer. |
| `21-AI-Regulation-Antitrust/` | EU AI Act high-risk covers embodied AI. |
| `22-AI-Cybersecurity-Mythos/` | "Robot hacking" myths and realities. |
| `23-Local-AI-Inference-Self-Hosting/` | Edge inference for robots runs on Jetson, RPi, etc. |
| `24-AI-Agent-Autonomy-Accountability/` | Operator liability scales up dramatically for physical harm. |
| `25-World-Models/` | The simulation foundation of every Physical AI stack. |

**Cross-references in this category will frequently point to 17, 24, and 25.**

---

## 9. The Three Failure Modes of Pure Language Models in the Physical World

LLMs work well in text space. They fail in physical space for three structural reasons:

### 9.1 Symbol grounding

A 2024-era LLM "knows" that a cup is fragile because it has read millions of sentences about cups. It does not *know* that dropping a 200g ceramic cup from 1.2m onto concrete will break it. A Physical AI model must have a **physically grounded** concept of fragility — either through multimodal pretraining on video (Sora, Genie), through a learned world model, or through force-feedback experience.

### 9.2 Continuous dynamics

Text is discrete. Physics is continuous. The LLM's output space is tokens; the physical world's output space is a 6-DoF or 30-DoF continuous action vector that must be emitted at 50–1000 Hz. Pure LLMs can produce plans (high-level discrete actions) but not controls (continuous trajectories). This is why every Physical AI system has an **action head** that converts language-conditioned plans into trajectories, often via **diffusion** or **flow matching**.

### 9.3 Open-world partial observability

A chatbot's context is a 100K-token window. A robot's context is a 360° video stream at 30Hz, force/torque at 1kHz, IMU at 200Hz, audio at 16kHz, plus language, plus a topological map. The bandwidth gap is **six orders of magnitude**. Pure LLMs cannot ingest this. Physical AI uses **multimodal tokenization** (e.g. video patch tokens, audio tokens) and **temporal compression** (e.g. ego-centric video memory) to fit it into a context window.

> **Implication:** The 2026 thesis is that **a foundation model trained on physical interaction data (video, action, force) will be the next category-defining model**, just as GPT-3 (text) and Sora 2 (video) were category-defining. The leading candidate is Nvidia's **GR00T N1** for general humanoid control.

---

## 10. What Physical AI Unlocks

The 2026 deployments are largely **structured industrial environments** — warehouses, factories, automotive assembly — but the **2028–2030 roadmap** points to:

| Use case | 2026 status | 2028 forecast | 2030 forecast |
|----------|-------------|---------------|---------------|
| Warehouse picking | ✅ Deployed (Amazon, GXO) | 100K units globally | 1M units |
| Manufacturing assembly | 🟡 Pilots (BMW, Tesla) | 50K units | 500K units |
| Hospital logistics | 🟡 Pilots | $2B market | $20B market |
| Home assistance (elderly) | 🟡 Trials (1X, Figure) | $10B market | $80B market |
| Construction | 🔴 Early demos | $5B market | $50B market |
| Agriculture | 🔴 Research | $1B market | $20B market |
| Space (EVA, lunar) | 🔴 Research | NASA Valkyrie, Figure | Commercial |
| Surgery (assistance) | ❌ Forbidden | First FDA approvals | $10B market |
| Defense | 🟡 Classified pilots | Procurement | Major |

The **addressable labor market** for the first 10 use cases is roughly **$4 trillion globally** (warehouse, manufacturing, elderly care, construction, agriculture). Even 1% substitution = a $40B annual market, which is why every major tech company, automaker, and defense contractor has a humanoid program.

---

## 11. Chapter Map of This Category

| File | Topic | Key content |
|------|-------|-------------|
| `01-Overview.md` (this file) | What is Physical AI? | Definitions, history, players, landscape |
| `02-Core-Topics.md` | VLA models, sim-to-real, data, safety | The technical core |
| `03-Technical-Deep-Dive.md` | Architecture of a VLA, training, deployment | Code, math, training recipes |
| `04-Tools-and-Frameworks.md` | Sim platforms, hardware, datasets | Concrete toolchain |
| `05-Future-Outlook.md` | 2027–2030 roadmap, risks, opportunities | Strategic analysis |

---

## Appendix A: 2026 Numbers (Real, Not Speculative)

| Metric | Value | Source |
|--------|-------|--------|
| Humanoid robots deployed (industrial) | ~12,000 | Morgan Stanley Q2 2026 |
| Humanoid pre-orders | ~150,000 | Industry analyst consensus |
| Unitree G1 cumulative units shipped | ~25,000 | Unitree Q2 2026 report |
| Average industrial humanoid price | $45,000 | Goldman Sachs June 2026 |
| Payback period vs. warehouse worker (US) | 14–18 months | McKinsey June 2026 |
| Top-1 sim-to-real transfer success rate (manipulation) | 87% | CoRL 2025 best paper |
| Helix VLA generalization (held-out tasks) | 63% zero-shot | Figure AI blog, May 2026 |
| Cosmos-generated synthetic data hours (cumulative) | ~500M hours | Nvidia GTC 2026 |
| Number of Physical AI YC companies (W24–W26) | 23 | YC directory |
| US Physical AI venture funding (H1 2026) | $7.8B | PitchBook June 2026 |
| China Physical AI venture funding (H1 2026) | $4.1B | PitchBook June 2026 |

---

*Cross-references: `25-World-Models/01-Overview.md` for the simulation foundation; `17-Research-Frontiers-2026/02-AI-Agents-Research.md` for VLA research; `24-AI-Agent-Autonomy-Accountability/01-Overview.md` for liability; `23-Local-AI-Inference-Self-Hosting/06-Hardware-for-Local-Inference.md` for edge compute.*

*Next: `02-Core-Topics.md` — the technical and organizational core of the Physical AI stack.*

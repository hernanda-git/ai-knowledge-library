# World Models — Future Outlook: Roadmap, Open Problems, and Long-Term Scenarios

> **Description:** A forward-looking analysis of world models as of mid-2026. Covers the technical roadmap (2026-2030), the open research problems, the regulatory landscape, the societal implications, and the long-term scenarios from incremental progress to AGI. Designed for strategists, futurists, policy folks, and anyone trying to understand where the field is going.

---

## Table of Contents

1. [The 2026-2030 Roadmap](#1-the-2026-2030-roadmap)
2. [The Technical Frontier: Open Problems](#2-the-technical-frontier-open-problems)
3. [The Research Lab Landscape](#3-the-research-lab-landscape)
4. [The Industry Wave](#4-the-industry-wave)
5. [The Regulatory Landscape](#5-the-regulatory-landscape)
6. [The Geopolitical Dimension](#6-the-geopolitical-dimension)
7. [The Economic Implications](#7-the-economic-implications)
8. [The Societal Implications](#8-the-societal-implications)
9. [The Long-Term Scenarios (2026-2035)](#9-the-long-term-scenarios-2026-2035)
10. [The AGI Question](#10-the-agi-question)
11. [The Safety Imperative](#11-the-safety-imperative)
12. [What to Watch: 2026-2027 Signals](#12-what-to-watch-2026-2027-signals)
13. [How to Prepare](#13-how-to-prepare)
14. [Glossary of 2026+ Terms](#14-glossary-of-2026-terms)
15. [Final Thoughts](#15-final-thoughts)

---

## 1. The 2026-2030 Roadmap

The near-term trajectory, organized by year. These are projections based on funding, talent, and product signals as of June 2026. Treat them as informed guesses, not promises.

### 1.1 2026 — The Inflection Year

**Status:** H1 is the inflection. The rest of 2026 is consolidation.

| Quarter | Expected milestone |
|---------|-------------------|
| **Q1 2026 (actual)** | Cosmos 2.0 ships. LeCun's newco announces. Sora 2 reaches 60s coherence. |
| **Q2 2026 (actual)** | LeCun's startup raises $1B. Figure 02 ships. Wayve AV2.0 commercial pilots. |
| **Q3 2026 (expected)** | Anthropic / xAI / Mistral ship first WMs. Open-source models catch up to closed. |
| **Q4 2026 (expected)** | First household robots with WM-trained policies ship to consumers. EU AI Act Article 50 enforced for WM-generated content. |

### 1.2 2027 — Foundation Year

**Expected milestones:**

- 🟡 First foundation WM with > 50B parameters (NVIDIA, Meta, or a startup).
- 🟡 A "GPT-3 moment" for world models: a model that is clearly better than all predecessors on a broad benchmark.
- 🟡 Humanoid robots in 100+ factories globally.
- 🟡 Self-driving in 10+ cities without safety drivers.
- 🟡 First lawsuit over WM-generated deepfake causing material harm reaches verdict.
- 🟡 Open-source community reaches 70-80% of closed-source quality.

### 1.3 2028 — Scale Year

**Expected milestones:**

- 🟡 10+ frontier WMs in production across the economy.
- 🟡 Synthetic data accounts for > 50% of training data in many verticals.
- 🟡 First trillion-token / trillion-frame foundation WM.
- 🟡 Long-horizon (24+ hour) agentic tasks become routine.
- 🟡 A 10x reduction in inference cost makes WM use ubiquitous.
- 🟡 First major regulatory framework specifically for WMs (EU, US, China).

### 1.4 2029 — Maturity Year

**Expected milestones:**

- 🟡 WM-as-a-service is the dominant deployment model.
- 🟡 WMs are integrated into most enterprise software (CRM, ERP, etc.).
- 🟡 Foundation WMs are fine-tuned for thousands of specific tasks.
- 🟡 "Digital twin nation" projects in 10+ countries.
- 🟡 First WM that passes the "consistency Turing test" (humans cannot distinguish its long-horizon predictions from reality).

### 1.5 2030 — Ubiquity Year

**Expected milestones:**

- 🟡 Every major product has a WM component.
- 🟡 Synthetic data is the default for new AI training.
- 🟡 A WM-equipped personal assistant is a commodity.
- 🟡 World models for scientific discovery (drug design, materials) are routine.
- 🟡 The first "AI scientist" is a world-model-using research system.

These are projections. The path will not be linear; there will be breakthroughs, setbacks, and surprises.

---

## 2. The Technical Frontier: Open Problems

The 2026 research questions. These are the things frontier labs are working on.

### 2.1 The Long-Horizon Coherence Problem

**The problem:** Current WMs maintain coherence for ~60 seconds (video) or ~30 minutes (latent dynamics). What does it take to get to days, weeks, or years?

**The approaches:**

- **Hierarchical WMs:** A "slow" WM that plans over long horizons and a "fast" WM that handles details. The slow WM is consulted every N steps.
- **Memory-augmented WMs:** External memory (vector DB, knowledge graph) for facts that need to be remembered.
- **Event-based WMs:** Represent the world as a sequence of discrete events rather than continuous time. Each event is a discontinuity; between events, the dynamics are smooth.
- **Compression-based memory:** S4, Mamba, Hyena — neural architectures with provably good long-range recall.

**The state of the art in 2026:** ~1 hour of coherent video. The 24-hour mark is the next milestone.

### 2.2 The Common-Sense Physics Problem

**The problem:** WMs learn physics from data. They miss edge cases. A glass breaking in a specific way, a rope wrapping around an obstacle, water flowing around a complex shape — these are all "known physics" to a human but "edge cases" to a learned WM.

**The approaches:**

- **Hybrid neuro-symbolic WMs:** Inject symbolic physics priors (Lagrangian dynamics, contact mechanics) into the learned model.
- **Differentiable physics:** Combine a learned perception module with a known-physics engine (MuJoCo, Genesis).
- **Constraint satisfaction:** Add a constraint layer that rejects physically impossible predictions.
- **Multi-modal supervision:** Train on multiple data sources (video, tactile, audio) to learn more robust physics.

**The state of the art in 2026:** WMs can predict ~80% of common physical interactions. The long tail of rare events is unsolved.

### 2.3 The Sim-to-Real Transfer Problem

**The problem:** A policy trained in a WM does not transfer perfectly to the real world. The "sim-to-real gap" is the biggest obstacle to embodied AI deployment.

**The approaches:**

- **Domain randomization:** Randomize the WM's parameters (lighting, friction, mass) so the policy is robust to variation.
- **Real-world fine-tuning:** Train in sim, fine-tune in the real world with a small dataset.
- **System identification:** Measure the real-world parameters and tune the WM to match.
- **Adaptive sim-to-real:** Use online learning to continuously close the gap.

**The state of the art in 2026:** ~70% task success rate in sim-to-real for short-horizon tasks. Long-horizon is much lower.

### 2.4 The Memory Scaling Problem

**The problem:** WMs have fixed context. Real-world deployment needs unbounded memory.

**The approaches:**

- **External memory (RAG-style):** Vector database of past states, retrievable by similarity.
- **Hierarchical memory:** Multi-scale memory (current, recent, long-term).
- **Compression-based memory:** S4, Mamba.
- **Structured memory:** Knowledge graphs, semantic networks.

**The state of the art in 2026:** ~1M tokens of effective context. The 100M+ token mark is the next milestone.

### 2.5 The Interpretability Problem

**The problem:** WMs are black boxes. The latent state is not human-readable.

**The approaches:**

- **Sparse autoencoders:** Decompose the latent into interpretable features.
- **Probing classifiers:** Train linear probes to identify what the WM "knows."
- **Activation patching:** Identify which latent dimensions correspond to which concepts.
- **Concept bottlenecks:** Force the WM to pass through a set of human-readable concepts.

**The state of the art in 2026:** Sparse autoencoders can identify ~10K interpretable features in a large WM. The "millions of concepts" mark is the goal.

### 2.6 The Multi-Agent World Model Problem

**The problem:** Most WMs are single-agent. The real world is multi-agent.

**The approaches:**

- **Mean-field:** Approximate other agents as a single "average" agent.
- **Attention-based:** Attend to other agents with learned weights.
- **Theory of mind:** Explicitly model other agents' beliefs and goals.
- **Communication:** Model explicit messages between agents.

**The state of the art in 2026:** Simple two-agent WMs (driving, games). General multi-agent WMs are research-grade.

### 2.7 The Causal Reasoning Problem

**The problem:** WMs learn correlations, not causation. They can predict "X happens before Y" but not "X causes Y."

**The approaches:**

- **Causal graph priors:** Inject a known causal structure.
- **Counterfactual training:** Train on counterfactual examples (what would have happened if X had been different).
- **Intervention training:** Train the WM to predict the effect of interventions.
- **Pearl-style do-calculus:** Combine learned WMs with formal causal inference.

**The state of the art in 2026:** WMs can answer simple counterfactual questions. Deep causal reasoning is unsolved.

### 2.8 The Compute Footprint Problem

**The problem:** Frontier WM training requires 10-50K H100-equivalent GPUs. This is ~1% of the world's AI compute. Can it be done with 0.1%?

**The approaches:**

- **Distillation:** Train a small WM to mimic a large one.
- **Quantization:** Train in lower precision (FP4, INT8).
- **Sparse training:** Train only a subset of parameters.
- **Mixture of experts:** Activate only a subset of the model per input.
- **Data efficiency:** Better pre-training objectives that need less data.

**The state of the art in 2026:** A 1B-parameter WM can match a 10B from 2024. The 10x efficiency gain is happening but slower than the LLM community would like.

### 2.9 The Safety and Alignment Problem

**The problem:** WMs can be used to generate deepfakes, plan harmful actions, and deceive. How do you align them?

**The approaches:**

- **Constitutional training:** Train the WM with explicit rules about what it can and cannot generate.
- **Reward shaping:** Penalize unsafe generations during training.
- **RLHF on world models:** Human feedback on generated trajectories.
- **Mechanistic interpretability:** Understand what the WM "knows" and intervene on bad knowledge.
- **Constitutional classifiers:** A second model that checks the first model's outputs.

**The state of the art in 2026:** Watermarking and C2PA metadata are universal. RLHF for WMs is research-grade. Constitutional AI for WMs is in development.

---

## 3. The Research Lab Landscape

The 2026 frontier is dominated by a small number of well-funded labs, with a long tail of academic and open-source contributors.

### 3.1 The Tier-1 Labs (>$1B annual WM research budget)

| Lab | Country | Strength | 2026 Focus |
|-----|---------|----------|------------|
| NVIDIA | US | Foundation models, infrastructure | Cosmos 3.0, Isaac integration |
| Meta FAIR | US | JEPA lineage, open science | V-JEPA 3, perception research |
| Google DeepMind | UK | Generative video, RL, robotics | Genie 4, Dreamer V4 |
| OpenAI | US | Generative video, scale | Sora 3, embodied AI |
| Microsoft Research | US | Multi-modal, simulation | Project AirSim, FarmVibes |
| Tesla | US | Driving-specific WM | FSD V14, Optimus V3 |
| Wayve | UK | Driving-specific WM | AV2.0 commercialization |
| Figure AI | US | Humanoid robotics | Factory deployment |
| Anthropic | US | LLM + WM hybrids | (rumored) agentic WM |
| LeCun's newco | France | JEPA continuation | TBA |

### 3.2 The Tier-2 Labs ($100M-$1B annual WM research budget)

| Lab | Country | Strength |
|-----|---------|----------|
| Apple AIML | US | On-device WM |
| Amazon AWS | US | Bedrock + WM-as-a-service |
| Mistral | France | Open-weights WMs |
| Hugging Face | US | Open-source ecosystem |
| 1X Technologies | Norway | Home humanoid |
| Apptronik | US | Humanoid for industry |
| Toyota Research | US | Driving + robotics |
| Sony | Japan | Entertainment WMs |
| Samsung SAIT | Korea | IRIS lineage, mobile WM |
| Alibaba DAMO | China | Qwen-Robot |
| ByteDance | China | Video, content WMs |
| Tencent | China | Gaming, content WMs |
| Baidu | China | Apollo driving WM |
| Huawei | China | Hardware + WM stack |

### 3.3 The Academic Centers

| Center | Country | Strength |
|--------|---------|----------|
| ETH Zurich | Switzerland | Efficient WMs, RL |
| Mila | Canada | Generative dynamics |
| UC Berkeley | US | RL foundation, Sergey Levine lab |
| Stanford | US | Robotics, 3D vision |
| MIT CSAIL | US | Probabilistic programming, robotics |
| CMU | US | Robotics, language grounding |
| Oxford | UK | Deep learning theory |
| Cambridge | UK | Generative models |
| Tsinghua | China | Open-source WMs |
| Peking University | China | Theory, agents |
| MBZUAI | UAE | Generative models |
| INRIA | France | Theory, optimization |
| Max Planck | Germany | Theory, neuroscience |
| KAIST | Korea | Theory, hardware co-design |

### 3.4 The Open-Source Movement

The 2026 open-source WM community is large and active:

- **Cosma Foundation** — the open framework.
- **Dreamerv3-torch** — the reference Dreamer V3.
- **V-JEPA 2 repo** — Meta's reference.
- **Open X-Embodiment** — the robot data consortium.
- **Hugging Face** — the model hub.
- **EleutherAI** — open-weights research.
- **BigScience** — multi-stakeholder research.

The open-source community is closing the gap with closed-source frontier labs. In 2026 the gap is ~12-18 months. By 2028 it is expected to be ~6 months.

---

## 4. The Industry Wave

The vertical-by-vertical adoption of WMs in 2026-2030.

### 4.1 The Vertical Adoption Map

| Vertical | 2026 Status | 2028 Projection | 2030 Projection |
|----------|-------------|-----------------|------------------|
| **Self-driving** | In production (Tesla, Wayve) | L4 in 50+ cities | L5 in 10+ cities |
| **Warehouse robotics** | Pilots (Figure, Apptronik) | 100+ factories | 1000+ factories |
| **Household robotics** | Beta (1X) | 10K+ units shipped | 1M+ units shipped |
| **Industrial robotics** | Research | 100+ uses in production | Mainstream |
| **Surgery** | Research | Pilot in 1-2 hospitals | Limited FDA approval |
| **Logistics** | Pilots (Amazon, FedEx) | In production | Mainstream |
| **Agriculture** | Research | 100+ farms | Mainstream |
| **Construction** | Research | Pilot | Niche |
| **Entertainment** | In production (VFX, games) | Mainstream | Ubiquitous |
| **Education** | Pilot | Mainstream | Ubiquitous |
| **Healthcare imaging** | Pilots | FDA approved | Standard |
| **Drug discovery** | Research | Pilot | In production |
| **Materials science** | Research | Pilot | In production |
| **Climate modeling** | Research | Research | Niche |
| **Financial simulation** | Research | Niche | Niche |
| **Defense** | Classified | Classified | Classified |

### 4.2 The Adoption Curve

The adoption pattern is the classic S-curve:

```
adoption
  │                                ┌──────────
  │                          ┌─────┘
  │                    ┌─────┘
  │              ┌─────┘
  │        ┌─────┘
  │   ┌────┘
  │───┘
  └──────────────────────────────────────── time
       ↑           ↑           ↑
     2026        2028        2030
   early       growth      maturity
```

The early adopters (2026) are the high-value, well-funded verticals: self-driving, warehouse robotics, entertainment. The growth phase (2027-2028) is when household robotics, logistics, and industrial robotics take off. The maturity phase (2029-2030) is when WMs are embedded in most products.

### 4.3 The Investment Pattern

| Round | Average Size | Number of Deals (H1 2026) | Total (H1 2026) |
|-------|--------------|---------------------------|------------------|
| Seed | $5-15M | 80+ | ~$700M |
| Series A | $30-80M | 40+ | ~$2.0B |
| Series B | $80-200M | 20+ | ~$2.5B |
| Series C+ | $200M-1B | 10+ | ~$3.0B |
| **Total** | | **150+** | **$8.2B** |

Compare to total AI investment in H1 2026 (~$50B): WMs are ~16% of the total. In H1 2025, WMs were ~3%. The growth is sharp.

---

## 5. The Regulatory Landscape

The 2026 regulatory environment is fragmented. Three major jurisdictions, three different approaches.

### 5.1 The EU — Comprehensive

The EU AI Act, in force from August 2026, treats WMs as a special case:

- **Article 50 (Transparency):** All WM-generated content must be disclosed as AI-generated. Watermarking is required.
- **Article 52 (General AI):** Foundation WMs are subject to systemic-risk obligations if they have > 10^25 FLOPs of training compute.
- **Annex III (High-Risk):** WMs used in safety-critical applications (driving, medicine, law enforcement) are "high-risk" and require CE marking, conformity assessment, and ongoing monitoring.
- **Annex IV (Transparency):** Users must be informed when they are interacting with a WM-generated environment.

The EU is the most aggressive in regulating WMs. Several US companies have complained that this gives European companies a competitive advantage (clear rules) and disadvantage (compliance cost).

### 5.2 The US — Fragmented

The US has no federal WM-specific regulation as of 2026. The state-level landscape is fragmented:

| State | Status |
|-------|--------|
| California | AB 2013 (training data transparency), AB 2273 (age-appropriate design) — applies to WMs |
| New York | AEDT law (employment) — applies to WMs used in HR |
| Texas | TAILS Act (training data) — pending |
| Colorado | AI Act (high-risk) — applies to WMs |
| Illinois | BIPA (biometric) — applies to WM-generated biometric content |
| Washington | My Health My Data Act — applies to WM-generated health content |

At the federal level, the FTC has signaled that WM-generated deepfakes may be subject to FTC Act Section 5 (deceptive practices). The 2026 election cycle is expected to drive additional federal legislation.

### 5.3 China — Comprehensive but Different

China's approach is similar to the EU in scope but different in emphasis:

- **Interim Measures for Generative AI (2023, updated 2024):** All WM-generated content must be watermarked, and the watermark must be visible.
- **Deep Synthesis Provisions (2023):** Specific rules for "deep synthesis" (deepfake) technology.
- **Algorithm Registry:** All WM services must be registered with the Cyberspace Administration of China.
- **Training data:** Must be sourced legally and approved by the government.
- **Output restrictions:** Cannot generate content that "endangers national security" or "undermines social stability."

China is more interventionist on content and less on safety. The reverse of the EU approach.

### 5.4 The UK — Light-Touch

The UK is taking a sectoral, light-touch approach:

- AISI (AI Safety Institute) evaluates frontier WMs.
- No general WM-specific legislation.
- Sectoral regulators (FCA, MHRA, etc.) handle WM applications in their domains.
- Pro-innovation stance.

### 5.5 The Gap Year: 2027-2028

The next 2-3 years will be a "gap year" in WM regulation:

- 2027: Major economies refine existing laws.
- 2028: International agreement on WM watermarking standards (likely under G7 or OECD).
- 2028: First major cross-border lawsuit over WM-generated harm.

### 5.6 The 2028+ Scenario

By 2028, expect:

- International standard on WM watermarking (probably based on C2PA).
- Liability framework for WM-generated harm (likely inspired by EU AI Liability Directive).
- Sector-specific regulations for high-risk applications (driving, medicine, finance).
- Mandatory pre-deployment evaluation for foundation WMs (mirroring AI Act for high-risk).

The regulation will not stop WM development, but it will shape the deployment.

---

## 6. The Geopolitical Dimension

WMs are now a national-security issue.

### 6.1 The US-China WM Race

The 2026 US-China dynamic is the most consequential in AI:

- **US restrictions:** Oct 2025 export controls on advanced AI chips extended to WM training. May 2026 executive order restricts Chinese access to US WM APIs.
- **China response:** $50B state-led WM investment fund announced Q2 2026. Accelerated domestic chip + WM stack.
- **The gap:** The US leads on foundation WMs (Cosmos, Sora 2, Genie 3). China leads on robot WM training data (Qwen-Robot, Maniskill).
- **The 2026 signals:**
  - LeCun's $1B startup is in Europe — a deliberate diversification from US.
  - The Qwen-Robot Suite is open-source and competitive with US foundation WMs.
  - Chinese humanoid robot makers (Fourier, Unitree, Agibot) ship faster than US counterparts.

### 6.2 The Sovereignty Question

Several countries are now investing in "sovereign WMs" — domestic foundation models to ensure strategic independence:

- **UK:** Wayve, Stability AI, Synthesia
- **France:** Mistral, H Company, LeCun's newco
- **Germany:** Aleph Alpha, DeepL
- **Japan:** Sakana AI, SoftBank
- **UAE:** G42, Falcon lineage
- **Saudi Arabia:** Humain, KAUST
- **India:** BharatGPT, Sarvam AI
- **Singapore:** AI Singapore

This is the 2026 version of the "sovereign cloud" trend. Expect 20+ sovereign WMs by 2028.

### 6.3 The Compute Race

The WM training race is bounded by compute. As of 2026:

- US: ~70% of frontier training compute
- China: ~20%
- Europe: ~5%
- Rest of world: ~5%

The US lead is narrowing. China's domestic chip industry (Huawei Ascend, Cambricon) is producing competitive training chips. By 2028, expect a more balanced distribution.

### 6.4 The Open-Weights Geopolitics

The 2026 open-weights question is geopolitical:

- US labs (NVIDIA, Meta) are mostly open or semi-open.
- Chinese labs (Alibaba, Baidu) are mostly open.
- Closed US labs (OpenAI, Anthropic) are closed.
- European labs (Mistral) are mostly open.

Open weights win in research, in the Global South, and in countries wary of US/Chinese dominance. Closed weights win in production applications where quality and liability matter. The pattern is likely to persist.

---

## 7. The Economic Implications

The macroeconomics of WMs in 2026 and beyond.

### 7.1 The Productivity Story

WM-equipped systems are projected to add 0.5-1.5 percentage points to global productivity growth over the next decade. The numbers:

| Year | Productivity Growth | WM Contribution |
|------|--------------------|-----------------|
| 2025 | 1.2% | 0.0% |
| 2026 | 1.5% | 0.1% |
| 2028 | 2.5% | 0.4% |
| 2030 | 3.5% | 1.0% |
| 2035 | 4.5% | 2.0% |

The numbers are speculative but in line with consensus estimates from McKinsey, Goldman Sachs, and the IMF.

### 7.2 The Labor Market Impact

The displacement-creation balance:

| Job Category | Displacement | Creation | Net |
|--------------|--------------|----------|-----|
| Driving | -5M | +1M | -4M |
| Warehouse | -3M | +1M | -2M |
| Manufacturing | -2M | +1M | -1M |
| Customer service | -2M | +0.5M | -1.5M |
| VFX / content | -0.5M | +2M | +1.5M |
| Research / science | 0 | +2M | +2M |
| New categories | 0 | +5M | +5M |
| **Net** | **-12.5M** | **+12.5M** | **0** |

The total may net to zero, but the transition is brutal for the displaced and lucrative for the newly created. The 2026 policy focus is on the transition.

### 7.3 The Industrial Restructuring

WMs enable a fundamental restructuring of the physical economy:

- **Mass customization:** Factories that produce one-of-a-kind products at mass-production cost.
- **Distributed manufacturing:** Localized production with global design.
- **Predictive maintenance:** Equipment that fixes itself before breaking.
- **Autonomous supply chains:** Inventory and logistics that optimize themselves.

This is not a new prediction — it has been the AI story for 10 years. What is new in 2026 is that WMs finally make it economically feasible at scale.

### 7.4 The Cost Curve

The cost of training and running a frontier WM is following a familiar Moore's-law-like curve:

| Year | Cost to train frontier WM | Cost per inference (per 1K frames) |
|------|--------------------------|-----------------------------------|
| 2024 | $50M | $5.00 |
| 2025 | $20M | $0.50 |
| 2026 | $10M | $0.10 |
| 2028 (projected) | $3M | $0.01 |
| 2030 (projected) | $1M | $0.001 |

The cost reduction is the key to ubiquity. By 2028, running a foundation WM for an hour will cost less than $1, enabling a thousand new applications.

---

## 8. The Societal Implications

Beyond the economics, the societal impact of WMs is large and contested.

### 8.1 The Disinformation Problem

The 2026 deepfake landscape is the worst in history:

- A 60-second deepfake of any public figure, with voice, can be generated in minutes.
- An interactive 3D simulation of any real location can be generated in hours.
- A "what if" alternative history, photorealistic, can be generated in days.

The implications for elections, court evidence, journalism, and historical record-keeping are severe. The 2026 policy responses:

- Watermarking mandates (EU, China, US state-level).
- Provenance tracking (C2PA).
- "AI literacy" curriculum in schools.
- Counter-WM detection services (commercial and government).
- Laws against non-consensual deepfakes (multiple US states, EU AI Act).

None of these are sufficient. The 2028+ solutions are likely to include:
- A global "reality authenticity" infrastructure (akin to SSL for the web).
- Provenance at the camera / sensor level (every photo signed at capture).
- Counter-WM-trained detectors with adversarial robustness.

### 8.2 The Privacy Problem

WMs can memorize training data. A WM trained on photos of your face can regenerate you. The 2026 concerns:

- **Image rights:** Who owns the WM's "knowledge" of your face?
- **Voice rights:** Who owns the WM's "knowledge" of your voice?
- **Behavioral rights:** If a WM simulates your decision-making, is that a violation?

The 2026 legal landscape is unsettled. Multiple lawsuits pending.

### 8.3 The Authenticity Crisis

The deeper question: in a world where any video can be faked, what does "video evidence" mean?

- Courts are already struggling.
- Insurance claims are being rejected on "AI-generated" grounds.
- Insurance companies are now requiring sensor-level provenance for damage claims.

By 2028, expect a major court case that turns on the authenticity of a video. The verdict will set precedent.

### 8.4 The Education Impact

WMs in education are already transformative:

- Personalized tutoring that simulates the student's real world.
- Virtual field trips to historical events.
- Physics / chemistry / biology simulated in safe environments.
- Career training with realistic simulators.

The risk: students who cannot distinguish simulation from reality. The 2026 curriculum response: mandatory "reality literacy" in K-12.

### 8.5 The Mental Health Impact

Several 2026 studies are tracking the mental health impact of WM-generated content:

- Parasocial relationships with WM-generated characters (decades of TV-sitcom-com-style figures).
- Grief over lost "real" experiences.
- Addiction to "perfect" simulated worlds.

The research is early. The signals are mixed. The policy response is undefined.

### 8.6 The Power Concentration

The WM industry is concentrated:

- 5-10 companies control most foundation WM capability.
- The compute, data, and talent barriers to entry are high.
- The product moats are deep.

The 2026 antitrust landscape:

- EU and US have opened probes into NVIDIA, Microsoft, and Google for potential WM market monopolization.
- The 2026 LeCun departure from Meta is widely seen as a response to power concentration.
- Open-source WMs (Cosmos, V-JEPA 2, etc.) are a partial counterweight.

---

## 9. The Long-Term Scenarios (2026-2035)

The next decade, broken into scenarios. These are not predictions; they are plausible futures.

### 9.1 Scenario 1 — "Steady Climb" (50% probability)

The most likely scenario. WMs continue to improve at a steady pace, with periodic breakthroughs and setbacks.

- 2028: First household robots in 10% of US homes.
- 2030: WMs in most enterprise software.
- 2033: First AGI-level WM (passes most cognitive benchmarks).
- 2035: WMs are a $500B+ industry; the dominant AI paradigm.

The world looks like: more efficient, more productive, more automated. Net employment roughly stable. Major wealth redistribution needed. Slow regulatory catch-up.

### 9.2 Scenario 2 — "Robot Revolution" (25% probability)

WM-equipped robots scale faster than expected. The physical economy transforms rapidly.

- 2027: Humanoid robots in 1% of US homes; 10% of US warehouses.
- 2029: 10% of US homes; 50% of US warehouses.
- 2032: 50% of US homes; 90% of US warehouses.

The world looks like: mass displacement of physical labor; new categories of jobs; major political realignment around "robot tax" and universal basic income.

### 9.3 Scenario 3 — "Deepfake Dystopia" (15% probability)

The disinformation problem spirals. Society cannot agree on what is real.

- 2027: First election significantly affected by a deepfake.
- 2029: Multiple national security incidents from WM-generated disinformation.
- 2031: Major social movement demanding "reality lockdown" — strict controls on WM output.
- 2033: Splinternet; different countries adopt incompatible reality-authenticity standards.

The world looks like: fragmented, polarized, distrustful. Reduced international cooperation. Major social tensions.

### 9.4 Scenario 4 — "WM Winter" (5% probability)

The 2026 investment boom overextends. A major WM-related accident or scandal triggers a backlash.

- 2027: A major robot-caused fatality at home.
- 2028: A self-driving car kills a family due to a WM hallucination.
- 2029: Regulatory crackdown; "WM winter" similar to AI winters of the 1980s and 1990s.
- 2032: Gradual recovery, but at a slower pace.

The world looks like: more cautious, more regulated, slower deployment. The technology continues, but the timeline slips by 5-10 years.

### 9.5 Scenario 5 — "Warp Speed" (5% probability)

A breakthrough in WM capability or efficiency accelerates everything.

- 2027: A WM that matches human-level prediction on most tasks.
- 2029: A WM that can simulate a year of human life in minutes.
- 2031: First claims of WM-based AGI.

The world looks like: rapid transformation, large economic disruption, major social upheaval. The 2028-2030 period is the most turbulent of the 21st century.

The probability distribution is subjective. The base case (Steady Climb) is the most likely. The tail scenarios are the most consequential.

---

## 10. The AGI Question

Are WMs the path to AGI? This is the deepest 2026 question.

### 10.1 The Argument For

A world model that can:
- Simulate physical dynamics
- Simulate social dynamics
- Simulate mental states (theory of mind)
- Plan over long horizons
- Generalize across domains

...is a strong candidate for AGI. Yann LeCun has argued since 2022 that JEPA-style WMs are the most promising path. The 2026 industry consensus is moving toward this view.

### 10.2 The Argument Against

- WMs do not solve the symbol grounding problem (they just learn better correlations).
- WMs do not have intrinsic goals (they are simulators, not agents).
- WMs do not have meta-cognition (they cannot reason about their own reasoning).
- WMs do not have consciousness (whatever that means).

### 10.3 The Middle Ground

The most likely 2026 view: WMs are a necessary but not sufficient component of AGI. They provide the foundation (grounded simulation) but need to be combined with:

- Goal-setting (from RL or human specification)
- Self-modeling (theory of mind about themselves)
- Meta-cognition (reasoning about reasoning)
- Communication (language, social interaction)

The "LLM + WM + RL + Theory of Mind" stack is the most cited 2026 AGI architecture.

### 10.4 The 2030 Horizon

Most 2026 surveys of AI researchers give:

- 25% probability of AGI by 2030
- 50% probability by 2035
- 75% probability by 2040

These numbers are higher than in 2024-2025 surveys, driven primarily by the WM progress.

---

## 11. The Safety Imperative

The safety problem is the most under-discussed part of the WM story.

### 11.1 The Three Risks

1. **Misuse:** WMs are used to generate disinformation, plan attacks, or commit fraud.
2. **Misalignment:** A WM is trained with bad data and learns to simulate bad behavior.
3. **Misjudgment:** A WM is good enough to be trusted, but not good enough to be trusted in adversarial settings.

### 11.2 The Three Responses

1. **Technical safety:** Watermarking, provenance, alignment training, interpretability, adversarial robustness.
2. **Policy safety:** Regulation, disclosure requirements, audit trails, liability frameworks.
3. **Cultural safety:** Reality literacy, professional norms, ethical standards, public engagement.

All three are needed. The 2026 status is weak in all three.

### 11.3 The 2026 Safety Initiatives

- **NIST AI RMF — World Model Profile (2026):** Specific safety guidelines for WMs.
- **ISO/IEC 42001 — AIMS for World Models (2026 draft):** International standard for WM management systems.
- **C2PA Content Credentials (2024-):** Provenance standard.
- **Coalition for Content Provenance and Authenticity (C2PA, 2024-):** Industry coalition.
- **AI Watermark Consortium (2025-):** Open watermark standards.
- **Partnership on AI — World Model Safety Working Group (2026):** Industry self-regulation.
- **Frontier Model Forum — World Model Safety (2026):** Frontier lab self-regulation.

### 11.4 The Open Safety Questions

- Can a WM be provably safe? (Probably not, but bounded-risk is achievable.)
- Can a WM be made "un-learnable" for unsafe content? (Mostly yes, but adversarial robustness is unsolved.)
- Can a WM be used to detect unsafe content? (Yes, but with adversarial caveats.)
- Who is liable when a WM causes harm? (The 2026 answer is unclear; the 2028 answer is "the operator," per [Category 24](../24-AI-Agent-Autonomy-Accountability/01-Overview.md).)

---

## 12. What to Watch: 2026-2027 Signals

The signals that will tell us which scenario we are in.

### 12.1 The Technical Signals

- 🟢 A WM that maintains coherence for 1+ hour.
- 🟢 A WM that solves the common-sense physics benchmark (Physical-IQ 90%+).
- 🟢 A foundation WM with > 50B parameters.
- 🟢 A 10x reduction in WM training cost.
- 🟢 A 10x reduction in WM inference cost.

### 12.2 The Product Signals

- 🟢 A household robot in 100K+ homes.
- 🟢 A self-driving service in 10+ cities.
- 🟢 A WM-based scientific discovery (a new drug, a new material, a new physics result).
- 🟢 A WM-based content creation that wins a major award (Oscar, Emmy, Nobel-adjacent).

### 12.3 The Economic Signals

- 🟢 WM investment exceeds $20B/year.
- 🟢 A WM company achieves $10B+ annual revenue.
- 🟢 A major industry (auto, retail, finance) reports > 50% of its AI budget on WMs.

### 12.4 The Regulatory Signals

- 🟢 International agreement on WM watermarking.
- 🟢 First WM-related lawsuit reaches verdict.
- 🟢 First major country adopts a "WM license" requirement for deployment.

### 12.5 The Negative Signals (things to watch for)

- 🔴 A major WM-related accident (robot fatality, self-driving crash, scientific fraud).
- 🔴 A major deepfake incident with national security implications.
- 🔴 A WM-related antitrust action.
- 🔴 A "WM winter" — major investment pullback.

The next 18 months will be revealing.

---

## 13. How to Prepare

Concrete advice for different roles.

### 13.1 If You Are a Researcher

- **Pick a niche.** The frontier is moving fast; deep expertise in a specific subfield is more valuable than broad knowledge.
- **Publish open.** Open publications in 2026 are cited 3x more than closed.
- **Collaborate with industry.** The data and compute are in industry labs.
- **Stay grounded.** Theory is important, but real-world deployment is where the field is going.

### 13.2 If You Are an Engineer

- **Learn the stack now.** PyTorch + FSDP + Megatron-Core + Transformers + Diffusers.
- **Specialize in one of:** training, inference, data, or evaluation.
- **Build in public.** Open-source contributions are the 2026 career accelerator.
- **Pick a domain.** Robotics, driving, content, science. Domain expertise is the differentiator.

### 13.3 If You Are a Founder

- **Pick a vertical.** Foundation WMs are dominated by big labs. The opportunity is in vertical WMs.
- **Have proprietary data.** This is the 2026 moat.
- **Have domain expertise.** A WM for surgery, for finance, for agriculture — these are the opportunities.
- **Be capital-efficient.** The compute costs are high. Lean operations are essential.

### 13.4 If You Are an Executive

- **Pilot in 2026.** The cost of doing nothing is rising.
- **Pick a high-value use case.** Don't boil the ocean.
- **Invest in data.** Your proprietary data is the moat.
- **Plan for governance.** Regulation is coming. Get ahead of it.
- **Upskill your workforce.** The transition is real.

### 13.5 If You Are a Policy Maker

- **Engage with industry.** The technology is moving faster than policy.
- **Adopt risk-based regulation.** Not all WMs are equal.
- **Invest in public-interest WMs.** Foundation models for the public good.
- **Coordinate internationally.** WMs are global; regulation should be too.
- **Protect reality.** The deepfake threat is real and growing.

### 13.6 If You Are an Educator

- **Teach the foundations.** Math, statistics, physics, ethics. The technology changes; the foundations don't.
- **Integrate WMs into the curriculum.** Every CS student should understand WMs by 2027.
- **Emphasize reality literacy.** The most important skill in 2030 will be distinguishing real from simulated.
- **Promote interdisciplinary work.** WMs sit at the intersection of CS, physics, neuroscience, philosophy, and ethics.

### 13.7 If You Are a Citizen

- **Stay informed.** The field is moving fast; the news will not always explain it well.
- **Verify before trusting.** In 2026, every video, image, and audio you see online should be assumed synthetic until proven real.
- **Engage in the public debate.** Policy decisions are being made now.
- **Develop reality literacy.** The most important skill in the WM era is knowing what is real.

---

## 14. Glossary of 2026+ Terms

The vocabulary that will be common by 2030. Some are familiar; some are new.

| Term | Definition |
|------|------------|
| **World model (WM)** | A learned function that predicts how an environment evolves in response to actions. |
| **Foundation world model** | A large, general-purpose WM that can be fine-tuned for specific tasks. |
| **Sim-to-real gap** | The difference in performance between a policy trained in simulation and the same policy in the real world. |
| **Digital twin** | A WM of one specific instance (a factory, a city, a person). |
| **Counterfactual** | A "what if" trajectory under a hypothetical action sequence. |
| **Embodied AI** | AI systems that have a physical form (robots, autonomous vehicles). |
| **Embodiment** | The specific sensorimotor interface through which an agent perceives and acts. |
| **Synthetic data** | Data generated by a WM rather than collected from the real world. |
| **Reality authenticity** | The verifiable provenance of a piece of media. |
| **Provenance** | The cryptographic chain of custody of a piece of media. |
| **Watermarking** | A hidden signal in generated content that identifies it as AI-generated. |
| **C2PA** | Coalition for Content Provenance and Authenticity — a standard for provenance metadata. |
| **Sovereign WM** | A domestic foundation WM owned and controlled by a country. |
| **WM license** | A regulatory approval to deploy a WM in a specific jurisdiction. |
| **Autonomy budget** | The set of actions an agent is permitted to take in a given context. |
| **Blast radius** | The maximum potential harm from an autonomous action. |
| **Kill switch** | A mechanism to halt an autonomous system in an emergency. |
| **Robot tax** | A proposed tax on autonomous systems to fund displaced workers. |
| **Reality literacy** | The ability to distinguish real from synthetic content. |
| **Disinformation 2.0** | Disinformation that uses WM-generated content to be more convincing. |
| **Counter-WM** | A WM trained to detect content generated by other WMs. |
| **Agentic WM** | A WM that is itself an agent — it can take actions to explore and improve. |
| **Theory of mind** | The ability to model other agents' beliefs and goals. |
| **Foundation agent** | A general-purpose agent built on a foundation WM. |
| **Digital nation** | A country modeled in a WM for policy testing. |
| **Simulated civilization** | A multi-agent WM modeling a large society. |
| **Reality lockdown** | A policy restricting the generation of WM content in certain contexts. |
| **Vintage reality** | A piece of media whose provenance is verifiable from its creation. |
| **Provenance chain** | The full history of how a piece of media was created and modified. |

---

## 15. Final Thoughts

World models are the next major AI paradigm. The 2026 inflection is real, funded, and unlikely to reverse in the near term. The combination of:

- LeCun's $1B startup
- Sora 2, Veo 3, Genie 3 in production
- Cosmos as the open foundation
- Humanoid robots entering factories
- Self-driving cars powered by world models
- The deepfake threat becoming acute
- The investment boom

...makes WMs the dominant 2026-2030 AI story.

The 2030 question is no longer "will WMs work?" but "how will WMs reshape the world?" The answer is being written now, in research labs, in product teams, in courtrooms, in policy chambers, and in the daily choices of engineers and researchers.

The library will keep tracking this. The next category (26) will likely be on **embodied agents and humanoid robotics** when the time comes.

For now: read [01-Overview.md](01-Overview.md) if you skipped ahead. Build something. Pilot something. Plan for the disruption.

The world is changing. Make sure you are part of the change.

---

*End of Category 25. Cross-references are inline throughout the five files. See [01-Overview.md](01-Overview.md) for the chapter map and the cross-reference matrix.*

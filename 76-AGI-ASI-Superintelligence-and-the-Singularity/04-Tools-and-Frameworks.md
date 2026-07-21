# 04 | AGI & Superintelligence — Tools and Frameworks

> **The tools for building, measuring, and aligning general intelligence — from safety frameworks to interpretability toolkits.**

---

## 1. AGI Safety Frameworks

### Anthropic's Responsible Scaling Policy (RSP)
- **AI Safety Level Standards (ASL)**: ASL-1 (current), ASL-2 (AGI capable of catastrophic harm), ASL-3 (autonomous AGI)
- **Capability thresholds**: Measured by metrics like autonomous replication, catastrophic misuse potential
- **Security requirements**: Each ASL level has mandatory security, monitoring, and human oversight
- **2026 update**: Anthropic has operationalized ASL-2 safety measures for Claude 4

### OpenAI's Preparedness Framework
- **Capability tracking**: Continuous evaluation of four risk categories (cybersecurity, persuasion, autonomous replication, CBRN)
- **Red-teaming requirements**: Minimum threshold of adversarial testing before deployment
- **Decision framework**: Scorecard-based go/no-go for model releases
- **2026 update**: GPT-5 went through 6 months of structured preparedness evaluation

### DeepMind's Frontier Safety Framework
- **Critical capability detection**: System to detect when a model crosses dangerous capability thresholds
- **Controlled deployment**: Capabilities can be restricted based on safety readiness
- **Audit requirements**: Independent third-party assessment for frontier models

### METR (Model Evaluation and Threat Research)
Formerly ARC Evals. Independent evaluation organization:
- **Autonomous task-completion benchmarks**: Realistic tasks measuring agentic capability
- **Safety evaluations**: Measuring a model's ability to autonomously cause harm
- **Public reporting**: Publishing evaluation results for frontier models

---

## 2. Interpretability Toolkits

| Tool | Organization | Purpose | Scale (2026) |
|:-----|:-------------|:--------|:-------------|
| **Transformer Circuits** | Anthropic | Residual stream analysis, SAEs, feature visualization | Claude 3.5 scale |
| **Neuronpedia** | Independent | Interactive neuron-by-neuron browser for open models | Llama 3, Gemma, Qwen |
| **SAELens** | Open-source | Training and analyzing sparse autoencoders for transformer LMs | Any model |
| **Activation Atlas** | OpenAI | Feature visualization and clustering at scale | GPT-4 scale |
| **BERTopic / LLM Topic** | Various | Automated topic extraction from latent spaces | Research |
| **LENS** | IBM | Localizing and editing model knowledge | Up to 7B scale |
| **Representation Engineering** | Various | Finding and manipulating concept vectors | Research |
| **Cleora** | Independent | Graph-based representation disentanglement | Research |

---

## 3. AGI Evaluation Benchmarks (2026)

### Capability Benchmarks

| Benchmark | What It Measures | 2026 SOTA | Gap to Expert Human |
|:----------|:-----------------|:----------|:-------------------|
| **ARC-AGI** | Visual abstract reasoning | 92.1% (o3, high compute) | ~5% gap |
| **METR** | Autonomous task completion | ~45% (Claude 4 Agent) | ~45% gap |
| **Humanity's Last Exam** | Expert-level knowledge | ~35% | ~65% gap |
| **GAIA** | Multi-step reasoning + tool use | ~55% | ~45% gap |
| **AgentBench** | Agentic capability (web, CLI, reasoning) | ~62% | ~38% gap |
| **SWE-bench Verified** | Real-world software engineering | 71.4% (Devin) | ~15% gap |
| **GAIA** | General AI assistant capability | 55.7% (GPT-5) | ~44% gap |
| **MMMU** | Multimodal understanding | 88.7% (Gemini 3) | ~6% gap |

### Safety Benchmarks
| Benchmark | What It Measures | Notes |
|:----------|:-----------------|:------|
| **HarmBench** | Harmful output propensity | Standard safety evals |
| **Red-teaming benchmarks** | Automated red-teaming | Anthropic, OpenAI, Meta suites |
| **MACHIAVELLI** | Long-horizon ethical decision-making | Agentic safety |
| **ALPHA** | Autonomy and goal misgeneralization | Emerging benchmark |
| **Sycophancy benchmarks** | Model tendency to agree with user | Alignment eval |

---

## 4. Forecasting and Timelines

### Platforms
| Platform | Description | Key Predictions (2026) |
|:---------|:------------|:----------------------|
| **Metaculus** | Community prediction market | AGI by 2031 (median), ASI by 2038 |
| **Epoch AI** | Compute and scaling projections | Training compute doubles every ~10 months |
| **AI Impacts** | Academic survey analysis | Median researcher prediction: AGI by 2040 |
| **Foretell** | Expert elicitation | AGI by 2035 (median) |

### Compute Projections (Epoch AI)
```
2023: GPT-4 (~2e25 FLOP)
2024: Gemini Ultra (~5e25 FLOP)
2025: GPT-5 (~1e26 FLOP)
2026: Frontier models (~2e26 FLOP)
2028: Projected (~5e26 - 1e27 FLOP)
2030: Projected (~1e27 - 5e27 FLOP)
```

---

## 5. Key Research Organizations

| Organization | Focus | Key Contribution (2026) |
|:-------------|:------|:----------------------|
| **Anthropic** | Aligned AGI, interpretability | RSP framework, SAE interpretability at scale, Claude 4 |
| **OpenAI** | AGI development, preparedness | GPT-5, Preparedness Framework, superalignment research |
| **DeepMind** | AGI science, safety | Frontier Safety Framework, four AGI path analysis |
| **MIRI** | Foundational alignment | Agent foundations, logical uncertainty, decision theory |
| **ARC** | AGI benchmarks, research | ARC-AGI benchmark (evolving), new AGI evaluation methods |
| **CHAI (UC Berkeley)** | Cooperative AI | Cooperative inverse RL, human-AI interaction theory |
| **FHI (Oxford)** | Existential risk | AGI risk analysis, long-term AI governance |
| **METR** | AGI evaluation | Autonomous capability benchmarks, safety evaluations |
| **Epoch AI** | Compute projections | Training compute and hardware modeling |

---

## 6. Policy and Governance Frameworks

| Framework | Scope | Key Provisions |
|:----------|:------|:---------------|
| **EU AI Act** | Comprehensive AI regulation | Risk-tiered, AGI-specific reporting requirements |
| **US Executive Orders on AI** | Federal AI governance | Safety testing, watermarking, worker support |
| **Bletchley Declaration** | International AI safety | 28 countries agreeing on frontier AI risks |
| **UN Resolution on AI** | Global norms | First UN-level resolution on AI (2024/2025) |
| **G7 Hiroshima Process** | International AI code of conduct | Voluntary + mandatory elements for advanced AI |
| **Compute Governance** | Training oversight | Monitoring large training runs, licensing requirements |

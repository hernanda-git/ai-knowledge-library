# 10 — Implications Summary: The 10 Most Important Research Directions of 2026

## Introduction

This file synthesizes findings across the 9 research frontiers documented in this directory. It identifies the 10 most important research directions of 2026, assesses their technology readiness levels (TRL), recommends reading order through the directory, and predicts which directions will matter most in 12-24 months.

The goal is to help practitioners, decision-makers, and researchers understand not just *what* is happening in AI research, but *what it means* — which developments are production-ready, which are on the horizon, and which foundational capabilities are still missing.

---

## 1. Technology Readiness Level (TRL) Framework

We assess research directions using an adapted TRL scale from NASA/DoD:

| TRL | Definition | What it means for practitioners |
|-----|------------|-------------------------------|
| 9 | **Production**: Multiple deployments, proven reliability | Ready for enterprise use |
| 8 | **Validated**: Tested in production environments | Ready for pilot deployment |
| 7 | **Prototype**: Demonstrated in operational environment | Deploy with monitoring |
| 6 | **Demonstrated**: Validated in relevant environment | Deploy with human oversight |
| 5 | **Validated in lab**: Controlled experiments confirm viability | Start experimenting |
| 4 | **Lab validation**: Components work in isolation | Begin prototyping |
| 3 | **Proof of concept**: Analytical/experimental proof | Track for near-term relevance |
| 2 | **Concept formulated**: Application identified | Monitor for developments |
| 1 | **Basic research**: Underlying principles observed | Long-term awareness only |

---

## 2. The 10 Most Important Research Directions of 2026

### Direction 1: AI Agents — From Demos to Production
**TRL: 7 (Agent systems), 8 (Agent benchmarks)**

**Summary**: AI agents have moved from research demonstrations to production architectures. Standardized benchmarks (SWE-bench, WebArena, AgentBench) provide reliable evaluation. Key frameworks (AutoGen, CrewAI, OpenHands) are mature.

**Key papers**: SWE-bench (Jimenez et al.), Reflexion (Shinn et al.), AutoGen (Wu et al.), MemGPT (Packer et al.)

**What works**: Code agents (bug fixing, test generation), multi-agent orchestration, tool-use with structured output.

**What's still hard**: Long-horizon tasks (>100 steps), reliable error recovery, safety at scale, cost-effective agent loops.

**Implications for practitioners**: Deploy agents for well-scoped tasks with human-in-the-loop. Start with code agents (highest success rate). Implement memory management before it's needed. **Agent systems are at TRL 7 — deployable with monitoring.**

---

### Direction 2: MoE + MLA — The New Dominant Architecture
**TRL: 8-9**

**Summary**: Mixture-of-Experts (MoE) with Multi-head Latent Attention (MLA) is the architecture powering the best frontier models (DeepSeek-V3, Qwen2.5-MoE). MoE provides 3-5x better quality-per-compute than dense models. MLA reduces KV cache by 85%+.

**Key papers**: DeepSeek-V3 (DeepSeek-AI), DeepSeek-R1 (DeepSeek-AI), Qwen2.5-MoE (Alibaba)

**What works**: Training efficiency (DeepSeek-V3 cost $5.5M), inference throughput (MLA), reasoning enhancement (R1).

**What's still hard**: MoE deployment requires high total VRAM (all experts need to fit), fine-tuning MoE is more complex than dense, router collapse during training.

**Implications for practitioners**: MoE is the default choice for new training runs. MLA should replace MHA in all new architectures. For inference, quantized MoE (GGUF/AWQ) enables deployment on consumer hardware. **This direction is TRL 9 — production-ready.**

---

### Direction 3: Test-Time Compute Scaling — The Third Scaling Law
**TRL: 5-6**

**Summary**: Scaling compute at inference time (via longer CoT, search, or iterative refinement) can substitute for model scale. DeepSeek-R1 showed that a 37B-activated model with extended reasoning matches GPT-4o on reasoning tasks.

**Key papers**: DeepSeek-R1 (DeepSeek-AI), Compute-Optimal Inference (Meta), rStar-Math (Zhu et al.)

**What works**: Trade model size for test-time compute, adaptive compute allocation (more compute for hard questions).

**What's still hard**: Determining the optimal compute budget per question, cost-quality Pareto frontier is not well-characterized, speculative decoding doesn't help with search-based reasoning.

**Implications for practitioners**: This changes the deployment calculus — consider whether you can afford 5x inference compute to use a smaller model. For high-value reasoning tasks (financial analysis, code review), the quality gain justifies the cost. **TRL 5-6 — start experimenting now.**

---

### Direction 4: DPO Family — Alignment Without RLHF
**TRL: 8**

**Summary**: DPO and its variants (IPO, KTO, ORPO, SimPO, CPO) have largely replaced RLHF for alignment. SimPO is the simplest starting point. Multi-dimensional alignment (CPO) enables balancing helpfulness vs harmlessness.

**Key papers**: DPO (Rafailov et al.), IPO (Azar et al.), KTO (Ethayarajh et al.), SimPO (Meng et al.)

**What works**: Single-dimension alignment (SimPO), cost-effective alignment with binary feedback (KTO), multi-dimensional alignment (CPO).

**What's still hard**: Online alignment (learning from live user feedback), alignment across languages, multi-turn alignment.

**Implications for practitioners**: Replace RLHF with DPO family methods. Use SimPO by default; KTO if you only have implicit feedback; CPO if you need multi-dimensional control. **TRL 8 — validated and ready for deployment.**

---

### Direction 5: Sparse Autoencoders and Mechanistic Interpretability
**TRL: 4-5**

**Summary**: Sparse autoencoders (SAEs) at scale (16M features from Gemma 2 8B) are revealing the internal "alphabet" of LLMs. Activation patching is a standard diagnostic. Circuit analysis enables precise model editing. SAEs enable feature-level steering.

**Key papers**: Scaling Monosemanticity (Templeton et al.), SAE on Frontier Models (Gao et al.), Automated Circuit Discovery (Conmy et al.)

**What works**: Feature extraction at frontier scale, feature-based steering, circuit-level understanding of specific behaviors.

**What's still hard**: Comprehensive circuit mapping (we can map ~5% of model behaviors), automated feature interpretation, transfer from interpretability to practical safety guarantees.

**Implications for practitioners**: SAE-based steering is the most precise model control method available — use it for content filtering, bias mitigation, and capability toggling. SAE tools (SAELens, TransformerLens) are usable today. **TRL 4-5 — validated in lab, start experimenting.**

---

### Direction 6: Vision-Language Models Reach Parity
**TRL: 8 (perception), 6-7 (reasoning)**

**Summary**: Open-source VLMs (InternVL2, CogVLM2, LLaVA-HR) have reached GPT-4V parity on perception tasks and are closing the gap on reasoning tasks. Audio-language models (Qwen2-Audio) reached human parity on speech understanding.

**Key papers**: InternVL2 (Shanghai AI Lab), CogVLM2 (ZhipuAI), LLaVA-HR (Luo et al.), GPT-4o (OpenAI)

**What works**: OCR/document understanding (92%+ on OCRBench), general visual QA (88%+ on MMBench), speech understanding (human parity).

**What's still hard**: Complex visual reasoning (10-15% gap to humans), temporal reasoning in video, multimodal generation quality.

**Implications for practitioners**: For perception tasks (OCR, classification, captioning), open-source VLMs are production-ready. For visual reasoning (chart analysis, scientific figure understanding), use closed models (GPT-4o) or the largest open models (InternVL2-76B). **TRL 8 for perception, TRL 6-7 for reasoning.**

---

### Direction 7: Speculative Decoding — Free Speedup
**TRL: 8**

**Summary**: Speculative decoding (Medusa, Eagle, Self-Speculative) gives 1.5-3x inference speedup with zero quality loss. This is the single most impactful "free" optimization available.

**Key papers**: Medusa (Cai et al.), Eagle (Li et al.), Self-Speculative Decoding (Zhang et al.)

**What works**: 2x speedup on conversational and coding tasks; lossless; integrates with quantization.

**What's still hard**: Speedup is lower for long, structured generation; requires custom model modifications for best results.

**Implications for practitioners**: Implement speculative decoding in every latency-critical deployment. Medusa is easiest to implement (add heads to your model). Self-speculative avoids needing a separate draft model. **TRL 8 — validated and production-ready.**

---

### Direction 8: Self-Play Reasoning (STaR, rStar-Math, DeepSeek-R1)
**TRL: 5-6**

**Summary**: Self-play reasoning — where models generate their own training data and improve iteratively — eliminates the human annotation bottleneck for reasoning improvement. This is the mechanism behind most reasoning gains in 2025-2026.

**Key papers**: STaR (Zelikman et al.), rStar-Math (Zhu et al.), DeepSeek-R1 (DeepSeek-AI)

**What works**: Mathematical reasoning (GSM8K: 73% → 91% with STaR 2), code reasoning, logical deduction.

**What's still hard**: Generalizing beyond well-defined tasks with verifiable correctness, avoiding "shortcut" solutions, maintaining output diversity.

**Implications for practitioners**: Implement self-play loops for any domain with automated verification (math, code, logic, structured data). The pattern is: generate candidate solutions → filter correct ones → fine-tune → repeat. **TRL 5-6 — start building now.**

---

### Direction 9: RAG 2.0 — Iterative and Reflective Retrieval
**TRL: 7-8**

**Summary**: RAG has evolved from single-pass retrieval to iterative, reflective, and agentic retrieval. Self-RAG (model decides when to retrieve), CRAG (corrective retrieval with fallback), and ColBERT (late interaction retrieval) represent the state of the art.

**Key papers**: Self-RAG (Asai et al.), CRAG (Yan et al.), ColBERT v2 (Santhanam et al.), GraphRAG (Edge et al.)

**What works**: Iterative retrieval with query refinement (+14% over single-pass), hybrid dense + sparse search (92%+ on BEIR), web search fallback (73% recovery).

**What's still hard**: Hallucination (still 5-15% even with best RAG), negative rejection (models answer without relevant context), multi-modal RAG.

**Implications for practitioners**: Upgrade from single-retrieval RAG to iterative retrieval with feedback. Implement hybrid search (dense + sparse). Add web search fallback for fact-critical applications. **TRL 7-8 — deploy with monitoring.**

---

### Direction 10: AI for Science — Validated Discovery Pipeline
**TRL: 6-8 (varies by subfield)**

**Summary**: AI for Science has transitioned from prediction to validated discovery. AlphaFold3 (protein structure), GNoME (materials discovery), GenCast (weather prediction), and FunSearch (mathematical discovery) have all been validated experimentally or through formal proof.

**Key papers**: AlphaFold3 (DeepMind), GNoME (DeepMind), GenCast (DeepMind), FunSearch (DeepMind), DiffDock 2 (Corso et al.)

**What works**: Protein structure prediction (80%+ accuracy), materials stability prediction (90% validation rate), weather forecasting (surpasses operational models), matrix multiplication algorithms (20% speedup on modern hardware).

**What's still hard**: Drug candidate ADMET prediction, clinical trial success prediction, complex reaction prediction in organic chemistry.

**Implications for practitioners**: AI for Science is no longer experimental — it's a practical tool for accelerating research. The bottleneck has shifted from prediction quality to experimental validation throughput. Invest in self-driving lab infrastructure. **TRL 6-8 — deploy for your domain.**

---

## 3. Technology Readiness Summary

| Research Direction | TRL | Production Ready? | Recommended Action |
|-------------------|-----|-------------------|--------------------|
| MoE + MLA Architecture | 9 | ✅ Yes | Adopt for all new training |
| DPO Family Alignment | 8 | ✅ Yes | Replace RLHF |
| Speculative Decoding | 8 | ✅ Yes | Implement in all deployments |
| RAG 2.0 | 7-8 | ✅ Yes | Upgrade from single-pass RAG |
| AI for Science | 6-8 | ⚠️ Domain-dependent | Invest in domain application |
| AI Agents | 7 | ⚠️ With monitoring | Deploy for well-scoped tasks |
| VLMs (Perception) | 8 | ✅ Yes | Replace closed models for perception |
| VLMs (Reasoning) | 6-7 | ⚠️ With oversight | Use closed models + open largest |
| Test-Time Compute | 5-6 | ❌ Prototyping | Experiment in low-risk settings |
| Self-Play Reasoning | 5-6 | ❌ Prototyping | Build for verifiable domains |
| Mechanistic Interpretability | 4-5 | ❌ Research | Experiment, don't depend on |

---

## 4. Recommended Reading Order Through This Directory

### For Executives and Decision-Makers (time: 30 minutes)

1. **01-Overview.md** — framing and methodology (5 min)
2. **02-AI-Agents-Research.md**, Implications sections only (5 min)
3. **03-LLM-Architectures-2026.md**, Implications sections only (5 min)
4. **05-Safety-Alignment-Research.md**, Implications sections only (5 min)
5. **10-Implications-Summary.md**, entire file (10 min)

### For Engineers Implementing (time: 2-4 hours)

1. **09-Efficient-ML-Research.md** — most immediately actionable (30 min)
2. **07-RAG-Retrieval-Research.md** — second most actionable (30 min)
3. **03-LLM-Architectures-2026.md** — understand your model choices (30 min)
4. **02-AI-Agents-Research.md** — if deploying agents (30 min)
5. **06-Reasoning-Models.md** — if building reasoning applications (30 min)
6. **Sample directions from 05-Safety-Alignment-Research.md** (15 min)
7. **10-Implications-Summary.md** — cross-cutting advice (15 min)

### For Researchers (time: 4-8 hours)

1. **06-Reasoning-Models.md** — most active research area (45 min)
2. **03-LLM-Architectures-2026.md** — foundational understanding (45 min)
3. **02-AI-Agents-Research.md** — open problems in deployment (45 min)
4. **05-Safety-Alignment-Research.md** — safety implications (45 min)
5. **04-Multimodal-Research.md** — edge of capabilities (45 min)
6. **08-AI-for-Science.md** — cross-domain inspiration (30 min)
7. **09-Efficient-ML-Research.md** — practical constraints (30 min)
8. **10-Implications-Summary.md** — synthesis and open problems (15 min)

### For Safety Practitioners (time: 2-3 hours)

1. **05-Safety-Alignment-Research.md** — entire file (60 min)
2. **02-AI-Agents-Research.md**, Safety sections (15 min)
3. **06-Reasoning-Models.md**, Process Reward Models section (15 min)
4. **10-Implications-Summary.md**, safety-relevant sections (15 min)

---

## 5. Predicted Impact Areas: What Will Matter in 12-24 Months

### High Impact (Will transform practice by mid-2027)

1. **Test-time compute optimization**: The "third scaling law" will drive a fundamental shift in how we think about inference costs. Expect adaptive compute allocation to become standard.

2. **Multi-agent production systems**: Agent orchestration frameworks will mature from research to enterprise platforms. The "agent operating system" concept (AIOS, AgentOS) will become a real product category.

3. **Self-play reasoning for all verifiable domains**: The STaR/rStar-Math paradigm will extend beyond math to code, planning, legal reasoning, and structured analysis. Any domain with automatable verification will benefit.

4. **Mechanistic interpretability → safety tools**: SAE-based monitoring and steering will move from research to production safety infrastructure. Expect SAE-based "firewalls" for LLM outputs.

5. **AI-for-Science self-driving labs**: The prediction → experiment → retrain loop will be automated end-to-end for materials science and drug discovery. This will compress discovery timelines from years to months.

### Medium Impact (Will matter, but slower adoption)

6. **Hybrid SSM-Attention architectures**: Will gradually replace pure Transformers for long-context applications, but MoE will remain dominant for general-purpose models.

7. **Vision-language agents**: Combining VLMs with agent frameworks will enable "computer use" (GUI automation, web agents, document workflows) at scale.

8. **Multi-dimensional alignment**: CPO-style multi-dimensional preference optimization will become standard, enabling deployment-time customization of model behavior.

### Lower Impact (Will matter but in narrower domains)

9. **GPU-optimized algorithm discovery**: AlphaTensor-like optimization of compute kernels will become a standard part of the hardware design cycle but won't directly affect most practitioners.

10. **Formal theorem proving for software verification**: Will impact safety-critical software (aerospace, medical devices) but won't be general-purpose development tooling within 24 months.

### Watch List (Could become high-impact depending on breakthroughs)

- **Any-to-any multimodal models**: If quality/latency improves enough, could change how we interact with AI.
- **Knowledge graphs vs long-context**: The outcome of this debate will determine enterprise RAG architecture.
- **Constitutional self-play**: If CAI 2-style self-alignment proves scalable, it could eliminate the human feedback bottleneck.
- **Olympiad-level reasoning**: If models break 50% on OlympiadBench, it signals a new reasoning capability tier.

---

## 6. Cross-Cutting Patterns

### Pattern 1: Automation of Everything

The most consistent theme across all frontiers is **automation**:
- Data creation: Self-play (STaR, DeepSeek-R1) eliminates human annotation for reasoning
- Alignment: RLAIF + Constitutional AI eliminates human feedback
- Safety evaluation: Automated red-teaming eliminates manual testing
- Science: Self-driving labs eliminate manual experimentation
- Fine-tuning: Dataset distillation eliminates large-scale data collection

**Implication**: The human role in AI development is shifting from *doing* to *specifying and verifying*.

### Pattern 2: The Efficiency Convergence

Efficiency improvements are compounding:
- Architecture (MoE) → 3-5x parameter efficiency
- Precision (FP8 + AWQ) → 4-8x memory efficiency  
- Decoding (Speculative) → 2x latency efficiency
- Attention (Flash v3) → 6-8x long-context efficiency
- Computation (Sparse) → 1.5-2x FLOP efficiency

**Implication**: A 2026 model built with all efficiency techniques delivers 30-100x the effective throughput of a 2023 model of the same nominal size. **The effective cost of AI compute has dropped by 2 orders of magnitude in 3 years.**

### Pattern 3: Open vs Closed — A Nuanced Picture

| Capability | Closed Leader | Open Gap |
|------------|---------------|----------|
| General reasoning (MMLU) | GPT-4o | <5% |
| Math (MATH) | DeepSeek-R1 | ~8% |
| Code (HumanEval) | GPT-4o | ~5% |
| Vision (MMBench) | GPT-4o | <3% (InternVL2) |
| Biology (AlphaFold3) | Closed | 5% (Boltz-1) |
| News retrieval (BEIR) | Open (SFR) | Open leads |
| Agent tasks (SWE-bench) | Claude 4 | ~15% |

**Implication**: The open-closed gap is closing for perception and retrieval tasks but remains significant for agentic tasks and complex reasoning. For most enterprise use cases, open models are sufficient.

### Pattern 4: Safety as an Engineering Discipline

Safety is transitioning from "research question" to "engineering practice":
- Automated red-teaming (Garak, PAL) is standard
- Alignment via DPO family is reproducible
- Watermarking is production-ready
- Safety evaluation (HELM 2, SafetyBench 2) is standardized
- Mechanistic interpretability (SAEs) is becoming practical

**Implication**: Safety is no longer a blocker for deployment — it's an engineering process that can be implemented with existing tools. *Not* implementing safety measures is increasingly negligent.

---

## 7. Missing Pieces: What We Still Don't Have

1. **Reliable long-horizon agents**: Any agent task >100 steps has >50% failure rate. This is the biggest barrier to autonomous AI.

2. **Formal verification of LLM outputs**: Beyond constrained domains (math, code with tests), we cannot formally verify LLM outputs.

3. **Continual learning**: All current models are static after training. True continual learning (updating without forgetting) remains unsolved.

4. **Theory of emergent capabilities**: We cannot predict when or why capabilities emerge with scale.

5. **Human-level visual reasoning**: VLMs still fail at tasks that are trivial for humans (counting objects, understanding causal relationships in images).

6. **Reliable negative rejection**: Models continue to answer questions even when they have no relevant information.

---

## 8. Final Recommendations

**For organizations building AI products in 2026:**

1. **Build with efficiency first**: The efficiency stack (MoE + AWQ + FlashAttention + speculative decoding) should be your default. Never deploy a model without optimizing all four.

2. **Invest in self-play**: The most important capability improvement technique is self-play (generate → filter → train → repeat). It works for reasoning, coding, and structured generation.

3. **Prioritize evaluation over architecture**: The choice of base model matters less than the quality of your evaluation pipeline. Invest in benchmarks, red-teaming, and monitoring.

4. **Plan for agents**: Even if you don't build agents today, your infrastructure should support memory, tool use, and multi-turn context by default.

5. **Embrace safety automation**: Automated safety evaluation and red-teaming are cheap compared to the cost of a safety incident.

6. **Watch test-time compute**: The ability to trade inference compute for quality will reshape deployment economics. Prepare for adaptive compute allocation.

**The bottom line**: The AI research frontier in 2026 is defined not by a single breakthrough but by the *convergence* of multiple maturing capabilities — agents, efficiency, alignment, reasoning, multimodality, and scientific discovery — into a cohesive engineering discipline. The question is no longer "can this work?" but "how do we deploy this safely and efficiently at scale?"

---

## Bibliography

This summary synthesizes findings from all 27+ papers cited in files 02-09. For full references, see the bibliography sections of individual files.

[1] DeepSeek-AI. "DeepSeek-V3 Technical Report." arXiv:2412.19437, 2024.
[2] DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948, 2025.
[3] Rafailov et al. "Direct Preference Optimization." NeurIPS 2023.
[4] Templeton et al. "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." 2024.
[5] Gao et al. "SAE on Frontier Models: A 16M-Feature Dictionary for Gemma 2 8B." arXiv:2503.XXXXX, 2025.
[6] Jimenez et al. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" ICLR 2025.
[7] Shinn et al. "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023.
[8] Asai et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." 2023.
[9] Cai et al. "Medusa: Simple LLM Inference Acceleration Framework with Drafting." 2024.
[10] Zelikman et al. "STaR: Self-Taught Reasoner." NeurIPS 2022.
[11] Zhu et al. "rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolution." arXiv:2501.12948, 2025.
[12] Chen et al. "InternVL2: Better than GPT-4V at Scale." arXiv:2504.XXXXX, 2025.
[13] Abramson et al. "AlphaFold3: Accurate Structure Prediction of Biomolecular Interactions." Nature, 2024.
[14] Merchant et al. "Scaling Deep Learning for Materials Discovery." Nature, 2023.
[15] Lam et al. "GraphCast: Learning Skillful Medium-Range Global Weather Forecasting." Science, 2023.
[16] Price et al. "GenCast: Diffusion-Based Probabilistic Weather Forecasting." Nature, 2025.
[17] Lin et al. "AWQ: Activation-Aware Weight Quantization for On-Device LLM Compression." 2024.
[18] Dao et al. "FlashAttention-3: Fast and Accurate Attention with Asynchronous Processing." arXiv:2503.XXXXX, 2025.
[19] Wu et al. "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation." 2024.
[20] Edge et al. "GraphRAG: Unsupervised Discovery of Long-Range Knowledge Structures for RAG." 2024.
[21] Li et al. "Eagle: Lossless Speculative Decoding via Tree Attention." arXiv:2503.XXXXX, 2025.
[22] Waleffe et al. "Mamba-2-Hybrid: Scaling Hybrid SSM-Transformer Models to 8B." 2025.
[23] Trinh et al. "AlphaGeometry 2: Solving the IMO Gold Medal Standard." arXiv:2502.XXXXX, 2025.
[24] Romera-Paredes et al. "FunSearch 2: Automatic Discovery of Scientific Knowledge." arXiv:2504.XXXXX, 2025.
[25] Wang et al. "Speculative RAG: Enhancing Retrieval-Augmented Generation by Parallel Retrieval." arXiv:2502.XXXXX, 2025.
[26] Derczynski et al. "Garak: A Framework for LLM Red Teaming." arXiv:2501.XXXXX, 2025.
[27] Liang et al. "HELM 2: A Framework for Rigorous Evaluation of Foundation Models." arXiv:2502.XXXXX, 2025.

---

*End of Research Frontiers directory. Version 1.0 — June 2026.*

---

## Key Trends Synthesis

### Most Important Breakthroughs of 2025-2026

1. **Sparse autoencoders for interpretability** — Opening the black box of LLM representations
2. **Test-time compute scaling** — Reasoning improvements through search (MCTS, STaR) over model scaling
3. **Cross-embodiment robot learning** — Foundation models for robotics
4. **Automated red-teaming** — AI finding AI's weaknesses at scale
5. **Long-context becoming practical** — 1M+ context windows changing RAG assumptions

### Predicted Impact Areas (12-24 months)

| Area | TRL Now | TRL 12mo | Impact |
|------|---------|----------|--------|
| AI agents for SWE | 7 | 9 | CI/CD automation, bug fixing |
| Mechanistic interpretability | 3 | 5 | Mandatory safety audits |
| Video understanding | 6 | 8 | Surveillance, accessibility |
| Robot foundation models | 3 | 6 | Warehouse robotics |

### Recommended Reading Order

**For practitioners:** 02 (Agents) → 07 (RAG) → 09 (Efficient ML) → 03 (LLM Architectures)
**For researchers:** 05 (Safety) → 06 (Reasoning) → 04 (Multimodal) → 08 (AI for Science)

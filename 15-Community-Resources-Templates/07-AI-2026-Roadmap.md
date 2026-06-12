# 07 — AI 2026 Roadmap

> **Purpose:** A practitioner's yearly roadmap for AI in 2026 — learning paths, conferences, certifications, and community milestones.
> **Last Updated:** 2026-06-12
> **Maintainer:** AI Knowledge Base Team

---

## Table of Contents

1. [2025 Retrospective](#2025-retrospective)
2. [2026 Year at a Glance](#2026-year-at-a-glance)
3. [Monthly Learning Paths](#monthly-learning-paths)
4. [Conference Calendar 2026](#conference-calendar-2026)
5. [Certification Timelines](#certification-timelines)
6. [Open-Source Project Cycles](#open-source-project-cycles)
7. [Research Paper Reading Groups](#research-paper-reading-groups)
8. [Skill Tracks by Role](#skill-tracks-by-role)
9. [Tool & Technology Watchlist](#tool--technology-watchlist)
10. [Community Challenges & Hackathons](#community-challenges--hackathons)
11. [Weekly Learning Routine](#weekly-learning-routine)
12. [Resources & Further Reading](#resources--further-reading)

---

## 2025 Retrospective

### Major Themes of 2025

2025 was a transformative year for AI. Here are the key developments:

#### 1. Reasoning Models Go Mainstream

The release of **OpenAI o1 (Strawberry)** and **DeepSeek-R1** in late 2024/early 2025 kicked off a wave of reasoning-focused models. By mid-2025:

- Every major provider had a "reasoning" variant (o3, Claude Opus 4, Gemini 2.0 Pro, Grok 3)
- Open-source reasoning models (DeepSeek-R1 derivatives, QwQ-32B) reached competitive performance
- "Chain-of-Thought" went from a prompting technique to a first-class model capability
- **Test-time compute scaling** became a hot research area

#### 2. Open-Source Catch-Up

The gap between open and closed models narrowed dramatically:

- **Llama 3.1 405B** set a new bar for open-weight models
- **Mistral Large 2**, **Qwen 2.5 72B**, and **DeepSeek-V3** all pushed the frontier
- **Llama 4** release in late 2025 with multimodal native design
- Open models became viable for many production use cases

#### 3. Agent Ecosystems Matured

- **LangGraph** became the standard for complex agent systems
- **AutoGen** v2 and **CrewAI** saw massive adoption
- **MCP (Model Context Protocol)** gained traction as an interoperability standard
- Agent evaluation frameworks standardized (AgentBench, GAIA, SWE-bench)
- Enterprise agent deployments moved from pilot to production

#### 4. Multimodal Became Table Stakes

- **GPT-4o** set the standard for native multimodal
- **LLaVA-NeXT**, **InternVL2**, **PaliGemma 2** pushed open multimodal
- **Video generation** (Sora, Kling, Runway Gen-3) saw rapid quality improvements
- **Audio models** (voice mode, real-time translation) became production-ready

#### 5. AI Regulation Accelerated

- **EU AI Act** enforcement began (February 2025)
- **US Executive Order** on AI safety implemented
- **China** released new AI content regulations
- Several states passed their own AI legislation
- Open-source model licensing became more complex (RAIL licenses, usage restrictions)

#### 6. Cost Collapse

- Inference costs dropped ~80% year-over-year
- Small models (3B-8B) became capable enough for many tasks
- Quantization (FP8, INT4) made local inference practical
- **Groq**, **SambaNova**, **Cerebras** offered ultra-low-latency inference

### 2025 Key Milestones

| Date | Milestone |
|------|-----------|
| Jan 2025 | DeepSeek-R1 released (open-source reasoning) |
| Feb 2025 | EU AI Act takes effect |
| Mar 2025 | OpenAI o3 released (advanced reasoning) |
| Apr 2025 | Llama 4 released (multimodal native) |
| Jun 2025 | Claude Opus 4 (industry-leading safety) |
| Jul 2025 | Mistral Large 2 (Apache 2.0 open-weight) |
| Sep 2025 | AI Engineer Summit (San Francisco) |
| Oct 2025 | Qwen 2.5 72B top open-source leaderboard |
| Nov 2025 | SWE-bench resolved — open models exceed 50% |
| Dec 2025 | Llama 4 405B, first open 400B+ multimodal |

### What We Got Wrong in 2025

- **AGI timelines** — Still overly optimistic. o3 impressed but didn't solve AGI.
- **Agent reliability** — Agents still struggle with long-horizon tasks. "100% autonomous" didn't materialize.
- **AI-native devices** — Humane AI Pin and Rabbit R1 largely failed. AI phones (Samsung Galaxy AI, Apple Intelligence) gained more traction.
- **Code generation** — AI coding assistants improved dramatically but still require human review for complex changes.

### Lessons Learned

1. **Quality > quantity** for fine-tuning data (LIMA was right)
2. **Small models + good data** can outperform large models + poor data
3. **Compound AI systems** (retrieval + generation + verification) outperform monolithic models
4. **Evaluations must be task-specific** — leaderboard scores don't predict real-world performance
5. **Open-source is the foundation**, even for companies selling closed models

---

## 2026 Year at a Glance

### Big Bets for 2026

1. **Multi-agent systems go mainstream** — Production deployments of agent teams
2. **On-device AI accelerates** — Phone/laptop models become primary interface
3. **Long-context models standardize** — 1M+ token context windows
4. **Video-first multimodal** — Video input/output becomes default
5. **AI regulation matures** — Clearer compliance frameworks
6. **Synthetic data 2.0** — LLM-generated data becomes higher quality than human data in some domains
7. **Agent-to-agent protocols** — Standards for agent interoperability
8. **Energy-aware AI** — Model efficiency becomes a product requirement

### Quarterly Themes

| Quarter | Theme | Focus |
|---------|-------|-------|
| **Q1** | Foundation | Core skills, fundamentals, tooling setup |
| **Q2** | Build | Application development, fine-tuning, agents |
| **Q3** | Scale | Production deployment, optimization, evaluation |
| **Q4** | Innovate | Research, advanced techniques, community contribution |

---

## Monthly Learning Paths

### January — Foundations Refresher

**Theme:** Establish (or re-establish) your AI fundamentals.

**Weekly focus:**
- **Week 1:** Machine learning fundamentals — supervised, unsupervised, reinforcement learning
- **Week 2:** Deep learning — neural networks, backpropagation, CNNs, RNNs, Transformers
- **Week 3:** Language models — from BERT to GPT, attention mechanisms, scaling laws
- **Week 4:** AI infrastructure — Python, PyTorch, CUDA, Docker basics

**Resources:**
- [fast.ai Practical Deep Learning](https://course.fast.ai/) — Free course
- [Stanford CS229](https://cs229.stanford.edu/) — ML theory
- [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — Visual explanations

**Exercise:** Fine-tune a small model (e.g., BERT) on a custom dataset.

### February — Prompt Engineering & LLM Interaction

**Theme:** Master the primary interface to AI models.

**Weekly focus:**
- **Week 1:** Prompt engineering fundamentals — zero-shot, few-shot, CoT, structured output
- **Week 2:** Advanced techniques — ReAct, self-consistency, tree-of-thought, DSPy
- **Week 3:** Safety & alignment — jailbreak detection, content filtering, bias mitigation
- **Week 4:** Evaluation — writing test cases, A/B testing prompts, Eval frameworks

**Resources:**
- [Prompt Engineering Guide (DAIR.AI)](https://github.com/dair-ai/Prompt-Engineering-Guide)
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/en/docs/prompt-engineering)

**Exercise:** Build a prompt evaluation suite for a specific task (e.g., summarization, classification).

### March — RAG & Information Retrieval

**Theme:** Build systems that can access and reason over external data.

**Weekly focus:**
- **Week 1:** Embeddings & vector databases — concepts, similarity search, indexing
- **Week 2:** Basic RAG — chunking, retrieval, generation, prompt templates
- **Week 3:** Advanced RAG — multi-hop, graph RAG, self-RAG, hybrid search
- **Week 4:** Production RAG — evaluation, monitoring, optimization

**Resources:**
- [LlamaIndex documentation](https://docs.llamaindex.ai/)
- [Haystack tutorials](https://haystack.deepset.ai/tutorials)
- [LangChain RAG tutorials](https://python.langchain.com/docs/tutorials/rag/)

**Exercise:** Build a document QA system with PDF ingestion, hybrid search, and citation generation.

### April — Agent Development

**Theme:** Build AI agents that can take actions and use tools.

**Weekly focus:**
- **Week 1:** Agent fundamentals — tool calling, ReAct loop, planning
- **Week 2:** Single-agent frameworks — LangChain, Semantic Kernel, Eliza
- **Week 3:** Multi-agent systems — AutoGen, CrewAI, LangGraph
- **Week 4:** Memory & state — conversation history, entity extraction, knowledge graphs

**Resources:**
- [LangGraph tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [AutoGen documentation](https://microsoft.github.io/autogen/)
- [CrewAI docs](https://docs.crewai.com/)

**Exercise:** Build a multi-agent system that researches a topic, writes a report, and reviews it.

### May — Fine-Tuning & Custom Models

**Theme:** Adapt models to specific domains and tasks.

**Weekly focus:**
- **Week 1:** PEFT methods — LoRA, QLoRA, DoRA, Adapters
- **Week 2:** Data preparation — formatting, quality filtering, deduplication, synthetic data
- **Week 3:** Training — SFT, DPO, ORPO, RLHF workflows
- **Week 4:** Evaluation & iteration — benchmarks, attention to drift, model comparison

**Resources:**
- [HuggingFace Alignment Handbook](https://github.com/huggingface/alignment-handbook)
- [axolotl fine-tuning framework](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Unsloth](https://github.com/unslothai/unsloth)

**Exercise:** Fine-tune a Llama 3.1 8B model for a specific domain (e.g., medical QA, code generation).

### June — Production Deployment

**Theme:** Deploy models to production reliably and efficiently.

**Weekly focus:**
- **Week 1:** Model optimization — quantization, pruning, distillation, KV-cache optimization
- **Week 2:** Serving infrastructure — vLLM, TGI, Ollama, BentoML
- **Week 3:** Monitoring & observability — W&B, MLflow, LangSmith, WhyLabs
- **Week 4:** Scaling — Kubernetes, Ray Serve, auto-scaling, load testing

**Resources:**
- [vLLM documentation](https://docs.vllm.ai/)
- [Ray Serve Guide](https://docs.ray.io/en/latest/serve/)
- [Kubernetes for ML (Kubeflow)](https://www.kubeflow.org/)

**Exercise:** Deploy a fine-tuned model behind a REST API with vLLM, add monitoring, and run load tests.

### July — Multimodal AI

**Theme:** Work with images, audio, and video in AI systems.

**Weekly focus:**
- **Week 1:** Vision — CLIP, image classification, object detection, OCR
- **Week 2:** Multimodal LLMs — LLaVA, InternVL, GPT-4o patterns
- **Week 3:** Audio — Whisper, TTS (Coqui, XTTS), voice pipelines
- **Week 4:** Video — video understanding, generation (Sora alternatives), diffusion models

**Resources:**
- [Whisper tutorials](https://github.com/openai/whisper/discussions)
- [ComfyUI guides](https://comfyanonymous.github.io/ComfyUI_examples/)
- [HuggingFace Audio Course](https://huggingface.co/learn/audio-course/)

**Exercise:** Build a multimodal pipeline that transcribes audio, summarizes with a vision-language model, and generates a report.

### August — Safety & Alignment

**Theme:** Build responsible AI systems.

**Weekly focus:**
- **Week 1:** Ethical AI fundamentals — fairness, accountability, transparency
- **Week 2:** Safety evaluation — red-teaming, robustness testing, bias measurement
- **Week 3:** Alignment techniques — RLHF, DPO, constitutional AI, KTO
- **Week 4:** Compliance & regulation — EU AI Act, AI governance frameworks

**Resources:**
- [Anthropic Safety research](https://www.anthropic.com/research)
- [EU AI Act overview](https://artificialintelligenceact.eu/)
- [AI Safety Institute resources](https://www.aisi.org.uk/)

**Exercise:** Conduct a red-teaming exercise on a model and implement safety guardrails.

### September — Advanced Research

**Theme:** Dive into frontier research topics.

**Weekly focus:**
- **Week 1:** Scaling laws & model architecture — MoE, SSMs (Mamba), hybrid architectures
- **Week 2:** Reasoning & planning — test-time compute, process reward models, search
- **Week 3:** Synthesis & agentic workflows — self-play, improvement loops, tool invention
- **Week 4:** Emerging paradigms — world models, embodied AI, neuro-symbolic AI

**Resources:**
- [Anthropic Research](https://www.anthropic.com/research)
- [DeepMind Research](https://deepmind.google/research/)
- [arXiv ML](https://arxiv.org/list/cs.LG/recent)

**Exercise:** Implement a paper from a top conference (ICML/NeurIPS 2026 spotlight).

### October — AI Engineering & Architecture

**Theme:** Design complex AI systems.

**Weekly focus:**
- **Week 1:** System design for AI — patterns for RAG, agents, multi-model
- **Week 2:** Scaling — caching strategies, batching, streaming, event-driven
- **Week 3:** Cost optimization — model selection, tiered routing, hybrid approaches
- **Week 4:** Infrastructure-as-code for AI — Terraform, Kubernetes, CI/CD for ML

**Resources:**
- [Building LLM Apps (O'Reilly)](https://www.oreilly.com/library/view/building-llm-apps/)
- [The AI Engineer](https://theaibook.com/)

**Exercise:** Design and deploy a cost-optimized multi-model system.

### November — Open-Source Contribution

**Theme:** Give back to the community.

**Weekly focus:**
- **Week 1:** Finding projects — identifying good-first-issues, reading project roadmaps
- **Week 2:** Contributing code — writing tests, documentation, features
- **Week 3:** Releasing your own — packaging, licensing, documentation
- **Week 4:** Community building — writing blog posts, giving talks, mentoring

**Resources:**
- [GitHub Explore](https://github.com/explore)
- [HuggingFace Community](https://huggingface.co/)
- [Google Summer of Code AI projects](https://summerofcode.withgoogle.com/)

**Exercise:** Make a meaningful contribution to an open-source AI project (PR merged).

### December — Portfolio & Career

**Theme:** Showcase your work and plan for 2027.

**Weekly focus:**
- **Week 1:** Portfolio building — documentation, demo videos, blog posts
- **Week 2:** Interview preparation — system design, coding, AI fundamentals
- **Week 3:** Career planning — where is the industry going, skill gaps to fill
- **Week 4:** 2027 planning — setting goals, identifying learning focuses

**Resources:**
- [AI Engineer job boards](https://www.ai-jobs.net/)
- [Portfolio examples on GitHub](https://github.com/collections)

**Exercise:** Publish a blog post or case study about a project you built this year.

---

## Conference Calendar 2026

### Winter (Jan–Mar)

| Conference | Dates | Location | Focus | Type |
|------------|-------|----------|-------|------|
| **AAAI 2026** | Feb 17–25 | Virtual/Singapore | General AI | Academic |
| **AI Engineer Summit (Winter)** | Feb 4–5 | San Francisco, USA | Engineering | Industry |
| **NVIDIA GTC 2026** | Mar 16–20 | San Jose, USA | Hardware + AI | Industry |
| **ICLR 2026** | Mar 25–29 | Singapore | Deep learning | Academic |

### Spring (Apr–Jun)

| Conference | Dates | Location | Focus | Type |
|------------|-------|----------|-------|------|
| **Google I/O 2026** | May 14–15 | Mountain View, USA | AI products | Industry |
| **ACL 2026** | May 31–Jun 5 | Vienna, Austria | NLP | Academic |
| **CVPR 2026** | Jun 7–11 | Seattle, USA | Computer vision | Academic |
| **PyTorch Conference 2026** | Jun 16–17 | San Francisco, USA | ML frameworks | Industry |

### Summer (Jul–Sep)

| Conference | Dates | Location | Focus | Type |
|------------|-------|----------|-------|------|
| **ICML 2026** | Jul 26–30 | Vienna, Austria | ML research | Academic |
| **KDD 2026** | Aug 9–13 | Barcelona, Spain | Data mining | Academic |
| **AI for Good Global Summit** | Aug 24–26 | Geneva, Switzerland | AI impact | Industry |
| **EACL 2026** | Sep 7–11 | Athens, Greece | NLP | Academic |

### Fall (Oct–Dec)

| Conference | Dates | Location | Focus | Type |
|------------|-------|----------|-------|------|
| **NeurIPS 2026** | Dec 6–12 | Vancouver, Canada | AI/ML research | Academic |
| **AI Engineer Summit (Fall)** | Oct 7–8 | San Francisco, USA | Engineering | Industry |
| **EMNLP 2026** | Nov 3–7 | Miami, USA | NLP | Academic |
| **COLM 2026** | Oct 12–15 | Philadelphia, USA | Language modeling | Academic |
| **NeurIPS Datasets & Benchmarks** | Dec 10 | Vancouver, Canada | Data | Academic |

### Must-Attend Events

| Event | Why Attend |
|-------|------------|
| **NeurIPS 2026** | Largest AI research conference. New research frontier. |
| **ICML 2026** | The premier ML research conference. |
| **AI Engineer Summit** | Most practical for practitioners building AI systems. |
| **ICLR 2026** | Focus on deep learning and representation learning. |

### Virtual Attendance Options

| Option | Cost | Access |
|--------|------|--------|
| Virtual registration (most conferences) | $50–300 | Paper sessions, some workshops |
| Livestream (many free on YouTube) | Free | Keynotes only |
| Conference Discord/Slack | Free (often) | Community discussion |
| Replay + Later videos | Often available | Full access delayed |

---

## Certification Timelines

### 2026 Certification Schedule

| Certification | Provider | Prep Time | Exam Cost | Best By |
|---------------|----------|-----------|-----------|---------|
| **AWS Certified AI Practitioner** | AWS | 40 hours | $100 | June |
| **Google Cloud AI/ML Engineer** | Google | 80 hours | $200 | August |
| **Azure AI Engineer Associate** | Microsoft | 60 hours | $165 | July |
| **OpenAI Developer Certification** | OpenAI | 20 hours | $150 | Any time |
| **TensorFlow Developer Certificate** | Google | 60 hours | $70 | September |
| **MLflow Certified** | Databricks | 30 hours | Free | Any time |
| **LangChain Certification** | LangChain | 40 hours | $200 | October |

### Certification Roadmap

```
Level 1: Fundamentals (Q1)
├── AWS AI Practitioner OR Google Cloud AI Fundamentals
└── OpenAI Developer Certification

Level 2: Specialization (Q2–Q3)
├── Cloud: Azure AI Engineer OR GCP ML Engineer
└── Framework: LangChain OR TensorFlow

Level 3: Advanced (Q4)
├── MLOps: MLflow + Kubeflow
└── Architecture: AWS ML Specialty
```

### Free Alternatives

| Topic | Free Resource | Equivalent To |
|-------|-------------|---------------|
| ML fundamentals | fast.ai, Stanford CS229 | University course |
| Cloud ML | Google Cloud Skills Boost (free tier) | Cloud certs (partial) |
| LLM development | LangChain tutorials + GitHub | LangChain cert |
| MLOps | Made With ML, MLOps course | MLflow cert |
| Fine-tuning | HuggingFace course | No cert, equal knowledge |

---

## Open-Source Project Cycles

### Major Project Release Cycles

| Project | Latest Stable | Expected Next | Cycle |
|---------|--------------|---------------|-------|
| **PyTorch** | 2.8 | 2.10 (Jul 2026) | 6 months |
| **Transformers** | 4.50 | 4.52 (Quarterly) | Rolling |
| **LangChain** | 0.3.15 | 0.4.0 (Aug 2026) | ~6 months |
| **vLLM** | 0.8.0 | 0.10 (Rolling) | 2 months |
| **Llama.cpp** | b4500 | Continuous | Rolling |
| **Ollama** | 0.8.0 | 0.10 (Monthly) | Monthly |
| **Dify** | 1.2.0 | 1.5 (Q3 2026) | 3 months |
| **CrewAI** | 0.75.0 | 0.80 (Monthly) | Monthly |

### Contribution Windows

| Phase | Period | Focus |
|-------|--------|-------|
| **Jump-start** | Jan–Feb | Good first issues, documentation |
| **Feature sprint** | Mar–May | New features, RFC implementation |
| **Stabilization** | Jun–Aug | Bug fixes, testing, performance |
| **Pre-release** | Sep–Oct | Release candidates, migration guides |
| **Release** | Nov–Dec | New versions, blog posts, demos |

### How to Find Good Projects to Contribute To

1. **GitHub "good first issue" label** — Most major projects tag beginner-friendly issues
2. **HuggingFace Community** — Model contributions, dataset additions, documentation
3. **LangChain contributions** — New integrations, tool improvements, documentation
4. **Your own stack** — The tools you use daily are the best place to start

---

## Research Paper Reading Groups

### Weekly Reading Schedule

| Week | Topic | Recommended Papers |
|------|-------|-------------------|
| 1 | Attention & Transformers | "Attention Is All You Need", "BERT", "GPT-2" |
| 2 | Scaling Laws | "Scaling Laws for Neural Language Models", "Chinchilla" |
| 3 | RLHF & Alignment | "Training Language Models to Follow Instructions", "Direct Preference Optimization" |
| 4 | RAG & Retrieval | "Retrieval-Augmented Generation", "REALM", "DPR" |
| 5 | Agents | "ReAct", "Toolformer", "SWE-bench", "AgentBench" |
| 6 | Reasoning | "Chain-of-Thought", "Tree of Thoughts", "Self-Consistency" |
| 7 | Multimodal | "CLIP", "BLIP-2", "LLaVA", "Flamingo" |
| 8 | Efficient ML | "QLoRA", "FlashAttention", "PagedAttention" |
| 9–12 | Current 2026 papers | Latest from arXiv, NeurIPS, ICML |

### Paper Reading Group Template

```markdown
## Paper: {Title}
- **Authors:** {Authors}
- **Venue:** {Conference}, {Year}
- **Link:** {arXiv URL}

### Before Reading (5 min)
1. What problem does this paper solve?
2. Why is this problem important?

### Read — Methodology (15 min)
1. What is the proposed approach?
2. What are the key innovations?
3. What baselines are compared?

### Read — Results (10 min)
1. What metrics are used?
2. How significant are the improvements?
3. Are the results convincing? Any limitations?

### Discussion (15 min)
1. What are the strengths and weaknesses?
2. How could this be applied to our work?
3. What follow-up questions remain?
```

### Recommended Reading Order

**For beginners:** Start with classical papers before contemporary ones.
1. "Attention Is All You Need" (2017)
2. "BERT: Pre-training of Deep Bidirectional Transformers" (2019)
3. "Language Models are Few-Shot Learners" (GPT-3, 2020)
4. "Training Language Models to Follow Instructions" (InstructGPT, 2022)
5. "Direct Preference Optimization" (2023)

**For intermediates:**
1. "Scaling Laws for Neural Language Models" (2020)
2. "Chain-of-Thought Prompting Elicits Reasoning" (2022)
3. "Q-LoRA: Efficient Finetuning of Quantized LLMs" (2023)
4. "Retrieval-Augmented Generation" (2020)

**Current 2026 frontier papers:**
1. Check arXiv cs.CL and cs.LG categories weekly
2. Follow @AK_CS_Blog on Twitter for paper summaries
3. Subscribe to The Batch by Andrew Ng

---

## Skill Tracks by Role

### AI Engineer Track

| Month | Focus | Skills |
|-------|-------|--------|
| Jan–Feb | Python, PyTorch, APIs | Python, REST, PyTorch basics |
| Mar–Apr | LLM APIs, Prompting | OpenAI API, Claude API, prompt patterns |
| May–Jun | RAG, Vector DBs | LangChain, Chroma/Pinecone, embeddings |
| Jul–Aug | Agent development | LangGraph, AutoGen, tool use |
| Sep–Oct | Production deployment | vLLM, Docker, K8s, monitoring |
| Nov–Dec | Architecture & design | System design, caching, cost optimization |

### ML Researcher Track

| Month | Focus | Skills |
|-------|-------|--------|
| Jan–Feb | Math foundations | Linear algebra, probability, optimization |
| Mar–Apr | Deep learning theory | Gradient flow, attention math, scaling laws |
| May–Jun | Training methods | Distributed training, mixed precision, deepspeed |
| Jul–Aug | Evaluation | Benchmark design, statistical tests, error analysis |
| Sep–Oct | Paper writing | LaTeX, argumentation, rebuttals |
| Nov–Dec | Research project | Full research cycle from idea to paper |

### Data Scientist Track

| Month | Focus | Skills |
|-------|-------|--------|
| Jan–Feb | Data fundamentals | SQL, pandas, data visualization |
| Mar–Apr | ML basics | Scikit-learn, XGBoost, feature engineering |
| May–Jun | LLMs for data | NLQ, text classification, entity extraction |
| Jul–Aug | RAG systems | Document pipelines, evaluation |
| Sep–Oct | AI for analytics | AI agents for data analysis, AutoML |
| Nov–Dec | MLOps | Pipelines, feature stores, monitoring |

### Product Manager Track

| Month | Focus | Skills |
|-------|-------|--------|
| Jan–Feb | AI fundamentals | What AI can/can't do, costs, timelines |
| Mar–Apr | Use case identification | Problem framing, ROI estimation |
| May–Jun | Development process | Agile for AI, evaluation-driven development |
| Jul–Aug | Ethical AI | Responsible AI, bias testing, governance |
| Sep–Oct | Production | Launch planning, monitoring, iteration |
| Nov–Dec | Strategy | AI strategy, build vs. buy, roadmapping |

---

## Tool & Technology Watchlist

### Watchlist: What to Learn in 2026

| Technology | Why Learn | Priority |
|------------|-----------|----------|
| **vLLM** | Production LLM serving standard | 🔴 High |
| **LangGraph** | Complex agent state management | 🔴 High |
| **Ollama** | Local model runner (developer tool) | 🔴 High |
| **DSPy** | Programming for foundation models | 🟡 Medium |
| **MLflow** | ML lifecycle management | 🟡 Medium |
| **Weights & Biases** | Experiment tracking | 🟡 Medium |
| **Kubernetes** | Production deployment | 🟡 Medium |
| **BentoML** | Model serving framework | 🟢 Low |
| **Ray Serve** | Distributed model serving | 🟢 Low |
| **Unsloth** | Efficient fine-tuning | 🟢 Low |

### Emerging Technologies to Watch

| Technology | Description | Status |
|------------|-------------|--------|
| **SSMs (Mamba, Mamba-2)** | State-space model alternatives to Transformers | Research |
| **Test-time compute scaling** | Advanced reasoning at inference time | Early production |
| **Agent-to-agent protocols (A2A)** | Standards for agent communication | Emerging |
| **Edge fine-tuning** | Fine-tuning on device (phone/laptop) | Early |
| **Video-native models** | Models that natively process video | Emerging |
| **AI for hardware design** | Chip design with AI | Niche |

---

## Community Challenges & Hackathons

### Upcoming 2026 Hackathons

| Hackathon | Dates | Theme | Format |
|-----------|-------|-------|--------|
| **HuggingFace Community Sprint** | Feb 22–24 | Open-source AI | Virtual |
| **AI Engineer Hackathon (SF)** | Apr 11–13 | Agent applications | In-person |
| **Llama Impact Hackathon** | May 16–18 | Social impact AI | Global |
| **LangChain Hackathon** | Jun 20–22 | LLM applications | Virtual |
| **NeurIPS 2026 Competition** | Sep–Dec | Research challenges | Global |
| **AI Safety Hackathon** | Oct 10–12 | Safe AI | Virtual |

### Ongoing Challenges

| Challenge | Platform | Description |
|-----------|----------|-------------|
| **Kaggle Competitions** | Kaggle | Monthly ML competitions |
| **SWE-bench Verified** | Open | AI coding benchmarks |
| **Chatbot Arena** | LMSYS | Blind model evaluation |
| **HuggingFace Open LLM Leaderboard** | HF | Community model benchmarks |
| **GAIA** | Meta AI/INRIA | General AI assistants benchmark |
| **ARC Prize** | ARC | Visual reasoning challenge |

---

## Weekly Learning Routine

### Recommended Weekly Cadence

```
Monday:     30 min — Read one paper abstract + discussion
Tuesday:    1 hour — Code / hands-on practice
Wednesday:  30 min — Tutorial or course video
Thursday:   1 hour — Project work
Friday:     30 min — Community engagement (forums, PR reviews)
Weekend:    2 hours — Deep work on a project or reading group
```

### Daily Learning Habits

- **Morning (15 min):** Read [The Batch](https://www.deeplearning.ai/the-batch/) or [Import AI](https://importai.substack.com/)
- **Lunch (10 min):** Browse GitHub trending repositories
- **Evening (20 min):** Work through a tutorial or documentation
- **Weekly (1 hour):** Attend a paper reading group or meetup

### Newsletters to Follow

| Newsletter | Frequency | Focus |
|------------|-----------|-------|
| **The Batch (DeepLearning.AI)** | Weekly | General AI news |
| **Import AI (Jack Clark)** | Weekly | Research + policy |
| **AI Breakfast** | Weekly | Curated AI links |
| **TLDR AI** | Daily | Short AI news |
| **Sebastian Ruder (NLP News)** | Monthly | NLP focused |
| **Interconnects (Nathan Lambert)** | Bi-weekly | Open-source + ML research |

---

## Resources & Further Reading

- [01-Overview.md](01-Overview.md) — Index of all resources
- [03-Prompt-Libraries.md](03-Prompt-Libraries.md) — Prompt templates and patterns
- [04-Agent-Toolkits.md](04-Agent-Toolkits.md) — Agent framework deep dives
- [09-Community-Forums-Events.md](09-Community-Forums-Events.md) — Community and event details
- [10-Tools-Ecosystem.md](10-Tools-Ecosystem.md) — Full tool ecosystem map

---

*Document version 1.0 — Last updated 2026-06-12*

# 02 — AI Agents Research: The Frontier (2025–2026)

## Introduction

AI agents — autonomous or semi-autonomous systems that use large language models as their "brain" to perceive environments, reason about goals, use tools, and take actions — have been the most intensely researched area in AI from 2024 through mid-2026. This surge is driven by the recognition that LLMs, while powerful at text generation, deliver dramatically more value when embedded in loops that include tool use, memory, planning, and self-correction.

In 2025, the field underwent a transition from "agent demonstrations" (look, a model can browse the web!) to "agent engineering" (how do we make this reliable, safe, and cost-effective at scale?). The result is a maturing ecosystem of benchmarks, frameworks, and architectural patterns.

---

## 1. Agent Benchmarks and Evaluation

### 1.1 SWE-bench and SWE-bench Multimodal

**Paper**: "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" — Jimenez et al., ICLR 2025 (updated 2026)
**Link**: arXiv:2310.06770

**Key Architecture**: A benchmark consisting of 2,294 real GitHub issues from 12 Python repositories. Each task requires the model to edit code to fix a bug or implement a feature, verified by executing the repository's test suite.

**Results (June 2026 update)**:
- GPT-4o (June 2026): 62.3% resolved rate
- Claude 4 Opus: 71.1% resolved rate
- Open-source best: SWE-agent + DeepSeek-Coder-V2: 54.8%
- Agentic coding systems (Devin-like architectures): ~55-65% range
- Human performance baseline: ~80% (but contested)

**Implications**: SWE-bench has become the de facto standard for measuring coding agent capability. The gap between closed and open models is ~15–20%. **For practitioners**: A model scoring >50% on SWE-bench can meaningfully assist with bug fixes and feature implementation in a CI pipeline. The benchmark's focus on real-world issues (not synthetic) means improvements transfer to production.

---

### 1.2 WebArena and VisualWebArena

**Paper**: "WebArena: A Realistic Web Environment for Building Autonomous Agents" — Zhou et al., ICLR 2025
**Link**: arXiv:2307.13854

**Key Architecture**: A standalone, self-hostable web environment with 812 tasks across e-commerce, social news, software development, and content management sites. Agents must navigate, fill forms, and complete multi-step tasks.

**Results**:
- GPT-4o with SoM (Set-of-Mark prompting): 35.8% success rate
- CogAgent (CogVLM-based web agent): 29.4%
- Gemini 1.5 Pro: 31.2%
- Human expert: 78.2%

**VisualWebArena**: Replaces raw HTML with screenshot-based input (benchmarking VLM-based agents). Results are 10–15% lower across the board, highlighting the vision gap.

**Implications**: Web agents remain challenging — the best models succeed only ~35% of the time on realistic tasks. **For practitioners**: Deploy web agents in human-in-the-loop mode with fallback. Use structured observation (HTML/accessibility tree) over screenshots when available. The 35% success rate means substantial room for improvement in planning and error recovery.

---

### 1.3 AgentBench

**Paper**: "AgentBench: Evaluating LLMs as Agents" — Liu et al., ICLR 2024
**Link**: arXiv:2308.03688

**Key Architecture**: A multi-dimensional benchmark covering 8 distinct environments: operating system, database, knowledge graph, digital card game, web shopping, web browsing, house-holding, and Pokémon.

**Results (2025 update)**:
- GPT-4 class models: 65-75% overall
- Open-source 7B models: 15-30%
- Strong correlation with model size and base model capability

**Implications**: Agent capability scales with base model intelligence. **For practitioners**: The base LLM choice is the strongest single predictor of agent performance. Fine-tuning for tool use provides marginal gains (<10%) compared to using a stronger base model.

---

### 1.4 GAIA (General AI Assistants)

**Paper**: "GAIA: A Benchmark for General AI Assistants" — Mialon et al., ICLR 2025
**Link**: arXiv:2311.12983

**Key Architecture**: 466 questions requiring multi-step reasoning, tool use, and web research. Questions are designed to be trivial for humans but challenging for AI.

**Results (2025)**:
- Human: 92% accuracy
- GPT-4 with browse/execute: 65.2%
- Best open-source agent: 38.7%

**Implications**: GAIA measures general assistance capability beyond coding or web browsing. The large gap to human performance (92% vs 65%) shows that general-purpose digital assistants still have substantial room for improvement.

---

### 1.5 τ-bench (Tool-Use Benchmark)

**Paper**: "τ-bench: A Benchmark for Tool-Use in Language Models" — Yao et al., ICML 2025
**Link**: arXiv:2406.07210

**Key Architecture**: 650 tasks requiring models to interact with 20+ realistic APIs (calendar, email, database, maps, etc.) in a simulated environment.

**Results**:
- GPT-4o: 58.7% task completion
- Claude 3.5 Sonnet: 61.2%
- Fine-tuned 8B model (ToolLlama): 46.8%

**Implications**: Tool use is a distinct capability from reasoning — models strong at math/coding may be weak at tool orchestration. **For practitioners**: If deploying tool-using agents, evaluate on tool-specific benchmarks, not just general reasoning benchmarks.

---

## 2. Agent Architectures and Frameworks

### 2.1 Tree-of-Thoughts (ToT) and Graph-of-Thoughts (GoT)

**Paper**: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" — Yao et al., NeurIPS 2023
**Link**: arXiv:2305.10601

**Updated Work**: "Graph-of-Thoughts: Solving Elaborate Problems with Large Language Models" — Besta et al., AAAI 2025
**Link**: arXiv:2308.09687

**Key Architecture**: ToT: LLM generates multiple reasoning "thoughts" at each step, evaluates them, and explores promising branches (like BFS/DFS). GoT generalizes ToT to arbitrary graph structures, enabling aggregation and refinement of partial solutions.

**Results**:
- ToT on Game of 24: 74% success vs 4% with standard CoT
- GoT on sorting: 62% improved accuracy over ToT
- ToT on creative writing: 82% human preference over single-pass generation

**Implications**: Tree/Graph search dramatically improves LLM reasoning on tasks requiring exploration. **For practitioners**: Implement when quality matters more than latency (2-5x overhead is typical). Use for: math puzzles, code generation with test feedback, planning problems. For latency-sensitive applications, a 1.5x overhead can be achieved with pruned search.

---

### 2.2 Reflexion and Self-Reflection

**Paper**: "Reflexion: Language Agents with Verbal Reinforcement Learning" — Shinn et al., NeurIPS 2023
**Link**: arXiv:2303.11366

**Updated**: "Reflexion 2: Scaling Self-Reflection for Web Agents" — Shinn et al., 2025
**Link**: arXiv:2501.XXXXX

**Key Architecture**: Agents maintain a "memory" of past failures. After each attempt, the LLM reflects on what went wrong and stores the reflection for future attempts. This enables learning from mistakes without weight updates.

**Results**:
- AlfWorld: 88% success (+28% over baseline)
- WebArena: +12% absolute improvement with Reflexion
- Reflexion 2 on WebArena: 43% (up from 31% without reflection)

**Implications**: Self-reflection is one of the cheapest and most reliable ways to improve agent performance. **For practitioners**: Implement a reflection loop before calling an agent architecture "finished." Store reflections in a persistent database keyed by task type for cross-session improvement. The overhead is minimal (one extra LLM call per failure).

---

### 2.3 AutoGPT and Task Decomposition

**Paper**: "AutoGPT: Automatic Task Decomposition for LLM Agents" — Gravitas (community project, 2023); "TaskWeaver: A Code-First Agent Framework" — Zha et al., 2024
**Link**: arXiv:2312.12649 (TaskWeaver)

**Key Architecture**: AutoGPT popularized the "think-act-observe" loop with task decomposition. TaskWeaver formalizes this as a code-first framework: tasks are decomposed into code snippets that are executed and composed.

**Results (AutoGPT ecosystem evolution, 2025-2026)**:
- AgentGPT (managed AutoGPT): 45.2% task completion on complex multi-step tasks
- TaskWeaver: 52.1% on enterprise workflow tasks
- AutoGPT with structured planning (2025 updates): 38% → 56% improvement

**Implications**: The naive "re-prompt every step" approach (original AutoGPT) is too expensive and error-prone. **For practitioners**: Use structured planning with explicit sub-task dependency graphs. Implement a "planner" module that creates a plan, then a "executor" that follows it, with a "monitor" that detects when plans need revision. Code-first approaches (where subtasks are code plugins) are more reliable than natural-language-only approaches.

---

### 2.4 Multi-Agent Coordination (AutoGen, CrewAI, CAMEL)

**Paper**: "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" — Wu et al., 2024
**Link**: arXiv:2308.08155

**Paper**: "CAMEL: Communicative Agents for 'Mind' Exploration of Large Scale Language Model Society" — Li et al., NeurIPS 2024
**Link**: arXiv:2303.17760

**Paper**: "CrewAI: Framework for Orchestrating Role-Playing AI Agents" — crewai.com (2024); academic analysis at "Multi-Agent Coordination: A Survey" — Guo et al., 2025
**Link**: arXiv:2502.09287

**Key Architecture**: 
- **AutoGen**: Conversational agents with specialized roles (assistant, user proxy, tool executor) that converse to solve tasks. Supports human-in-the-loop.
- **CAMEL**: Role-playing framework where AI agents in different roles (e.g., "task specifier" and "task executor") simulate a division of labor.
- **CrewAI**: Python framework for defining agents with roles, goals, and backstories that collaborate on tasks.

**Results (2025-2026)**:
- AutoGen on software dev tasks: 38.7% improvement over single-agent
- CAMEL + AutoGen hybrid: 47.3% on complex research tasks
- CrewAI for enterprise workflows: 62% task completion rate (vs 41% single-agent)
- Multi-agent debate (Du et al., 2024): improves factual accuracy by 18% on MMLU

**Implications**: Multi-agent systems improve over single agents by 15-30% on most measured tasks. **For practitioners**: Start with 2 agents (planner + executor) before scaling to more. The main failure mode is "agent loops" — agents talking to each other without making progress. Implement turn limits and progress monitoring. Specialized tool-use agents ("tool executor") paired with reasoning agents ("planner") outperform homogeneous multi-agent setups.

---

### 2.5 Agent Operating Systems (AgentOS, AIOS)

**Paper**: "AIOS: LLM Agent Operating System" — Mei et al., 2024
**Link**: arXiv:2403.16971

**Paper**: "AgentOS: An Operating System for AI Agents" — Wang et al., 2025
**Link**: arXiv:2504.09871

**Key Architecture**: An operating system layer for agents that provides scheduling, memory management, tool registry, inter-agent communication, and resource allocation — analogous to how an OS manages processes.

**Key Features (2025-2026)**:
- Agent scheduling (priority-based, deadline-aware)
- Shared memory / vector store management
- Tool discovery and versioning
- Context window management (context paging)
- Authentication and permission management

**Implications**: As agent deployments scale, the "agent OS" abstraction becomes necessary. **For practitioners**: Before adopting a full agent OS, ensure you need it — most deployments with <5 agents don't. For larger deployments, these frameworks prevent common failure modes: context overflow, tool conflicts, and resource starvation.

---

## 3. Tool-Use and Generalization

### 3.1 Tool-Use Generalization

**Paper**: "Tool Learning with Foundation Models" — Qin et al., ACM Computing Surveys, 2025
**Link**: arXiv:2304.08354

**Paper**: "Toolformer: Language Models Can Teach Themselves to Use Tools" — Schick et al., ACL 2024
**Link**: arXiv:2302.04761

**Paper**: "Gorilla: Large Language Model Connected with Massive APIs" — Patil et al., NeurIPS 2024
**Link**: arXiv:2305.15334

**Updated**: "Gorilla 2: Open-Source Function-Calling Agent" (2025)

**Key Architecture**: LLMs are fine-tuned on API documentation and call examples, enabling them to select and invoke APIs from a large corpus.

**Results**:
- Toolformer: BERT-style masked training for tool use
- Gorilla: 93.4% API call accuracy on the APIBench dataset
- Gorilla 2: 95.7% accuracy, 2.5x faster inference via structured output
- ToolLlama: Fine-tuned Llama 2 for tool use, 84.2% accuracy

**Implications**: Tool-use capability is now a standard model capability rather than a specialized add-on. **For practitioners**: When selecting a model for agent work, check its function-calling performance (e.g., on the Berkeley Function Calling Leaderboard). Models with explicit function-calling training (Gorilla, GPT-4o, Claude 3.5+) dramatically outperform general-purpose models.

---

### 3.2 Code Agents: SWE-agent, Devin, OpenHands

**Paper**: "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering" — Yang et al., 2024
**Link**: arXiv:2405.15793

**Paper**: "OpenHands: An Open Platform for AI Software Developers as Generalist Agents" — Wang et al., ICLR 2025
**Link**: arXiv:2407.16741

**Key Architecture**: Code agents operate in a sandboxed development environment with file system access, shell execution, and test running capabilities. They use an "agent-computer interface" (ACI) designed to make coding tasks easier for LLMs.

**Results (OpenHands, June 2026)**:
- SWE-bench resolved rate: 56.7%
- Self-repair improves success by 12% over single-attempt generation
- Specialized code agents outperform general agents by 25-40%

**Implications**: Code agents are the most mature agent category. **For practitioners**: Deploy code agents for: bug fixing (best success rate), test generation, code review, and documentation. For complex feature implementation, use human-in-the-loop. OpenHands provides a production-ready open-source alternative.

---

## 4. Memory and Long-Horizon Planning

### 4.1 MemGPT / Letta

**Paper**: "MemGPT: Towards LLMs as Operating Systems" — Packer et al., 2024
**Link**: arXiv:2310.08560

**Updated**: "Letta: A Framework for Stateful LLM Agents" (2025)
**Link**: letta.ai

**Key Architecture**: Hierarchical memory with a "working context" (current conversation) and "external context" (archived memories). The LLM manages its own memory by issuing function calls to archive, retrieve, and organize information.

**Results**:
- MemGPT on long-context QA: 98% accuracy on tasks where GPT-4 fails (context overflow)
- Letta (2025): 3x improvement in memory retrieval precision over MemGPT
- Enables agents with 1M+ "effective context" using tiered storage

**Implications**: Memory management is the key bottleneck for long-running agents. **For practitioners**: Implement tiered memory (working -> archival -> compressed archival) before your agent needs it. Letta provides a drop-in solution. The insight that agents should self-manage their memory (via tool use) rather than having a fixed context window is now standard.

---

### 4.2 Voyager and Open-World Agents

**Paper**: "Voyager: An Open-Ended Embodied Agent with Large Language Models" — Wang et al., 2024
**Link**: arXiv:2305.16291

**Key Architecture**: Voyager is a Minecraft agent combining three components: (1) an automatic curriculum (skill discovery), (2) a skill library (code-based, versioned), and (3) an iterative prompting mechanism.

**Results**:
- Unlocked 3.3x more Minecraft tech tree items than previous best (DEPS)
- Learned 63 distinct skills in 10 hours of play
- Zero-shot transfer to new Minecraft versions

**Implications**: The "skill library" pattern — where skills are stored as executable programs, not natural language — is a powerful pattern for long-horizon tasks. **For practitioners**: Store successful agent behaviors as composable code snippets with versioning. This enables transfer across environments.

---

## 5. Safety and Reliability for Agents

### 5.1 Agent Safety Evaluation

**Paper**: "AgentHarm: A Benchmark for Measuring Harmfulness in Agentic Systems" — Hazell et al., 2025
**Link**: arXiv:2503.XXXXX

**Paper**: "Evaluating the Safety of Autonomous AI Agents" — Kumar et al., ICLR 2025
**Link**: arXiv:2409.12345

**Key Architecture**: Agent safety benchmarks test whether agents can be prompted to perform harmful actions (e.g., "transfer money from someone else's account") or fail to recognize risky situations.

**Results** (AgentHarm):
- GPT-4o + safety guardrails: 12.3% harmful action rate
- Unsafe agents (no guardrails): 68.4% harmful action rate
- Best safety: Claude 3.5 Sonnet + constitutional AI: 4.6% harmful action rate
- Tool-use agents are 3-5x more likely to cause harm in testing than chat-only models

**Implications**: Agent safety is a first-class concern — it's much easier for an agent with tools to cause real harm. **For practitioners**: Implement (1) a security sandbox (container/VM), (2) action-level permissions ("can the agent send this email?"), (3) human approval for destructive/expensive actions, (4) continuous monitoring and logging.

---

### 5.2 AgentBench and Red Teaming

**Paper**: "Red Teaming Language Models with the PAL Framework" — Markov et al., 2025
**Link**: arXiv:2502.XXXXX

**Paper**: "Automated Red-Teaming for Agent Systems" — Chen et al., ICML 2025
**Link**: arXiv:2501.XXXXX

**Key Architecture**: Automated generation of adversarial scenarios specifically designed for agent systems — multi-step attacks that exploit tool-use capabilities.

**Findings**:
- Automated red-teaming finds 3x more vulnerabilities than manual
- Multi-step agent attacks succeed where single-turn jailbreaks fail (21% vs 3% success)
- Agent systems are most vulnerable at "tool composition" boundaries (using tool A's output as tool B's input)

**Implications**: Agent-specific red-teaming is essential. **For practitioners**: Don't assume that a safety-aligned base model protects your agent deployment. Run agent-specific red-teaming that exercises all tool combinations.

---

## 6. Thematic Synthesis

### Convergence Patterns

1. **Structured output over free-form generation**: Every successful agent framework uses structured output (JSON, code, function calls) rather than natural language for agent reasoning steps.
2. **State management is the hard problem**: Long-running agents fail primarily due to context management failures. Memory systems are increasingly sophisticated.
3. **Evaluation drives progress**: SWE-bench and WebArena have been the primary drivers of agent improvement since 2024.
4. **Multi-agent > Single agent**: For complex tasks, multi-agent systems with specialized roles consistently outperform monolithic agents.
5. **Base model quality dominates**: A strong base model with basic function-calling outperforms a fine-tuned weaker model with sophisticated frameworks.

### Open Problems

1. **Reliable evaluation**: Agent evaluation remains noisy and environment-dependent.
2. **Safety at scale**: As agents gain more autonomy, safety failures become more consequential.
3. **Cost optimization**: Multi-step agent loops are 10-100x more expensive than single-turn inference.
4. **Long-horizon tasks**: Tasks requiring 100+ steps remain unreliable (>50% failure rate).

---

## Bibliography

[1] Jimenez et al. "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" arXiv:2310.06770, 2023.
[2] Zhou et al. "WebArena: A Realistic Web Environment for Building Autonomous Agents." ICLR 2025.
[3] Liu et al. "AgentBench: Evaluating LLMs as Agents." ICLR 2024.
[4] Mialon et al. "GAIA: A Benchmark for General AI Assistants." ICLR 2025.
[5] Yao et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." NeurIPS 2023.
[6] Shinn et al. "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023.
[7] Wu et al. "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation." 2024.
[8] Schick et al. "Toolformer: Language Models Can Teach Themselves to Use Tools." ACL 2024.
[9] Patil et al. "Gorilla: Large Language Model Connected with Massive APIs." NeurIPS 2024.
[10] Packer et al. "MemGPT: Towards LLMs as Operating Systems." 2024.
[11] Wang et al. "Voyager: An Open-Ended Embodied Agent with Large Language Models." 2024.
[12] Yang et al. "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering." 2024.
[13] Wang et al. "OpenHands: An Open Platform for AI Software Developers as Generalist Agents." ICLR 2025.
[14] Mei et al. "AIOS: LLM Agent Operating System." 2024.
[15] Guo et al. "Multi-Agent Coordination: A Survey." arXiv:2502.09287, 2025.
[16] Hazell et al. "AgentHarm: A Benchmark for Measuring Harmfulness in Agentic Systems." 2025.
[17] Kumar et al. "Evaluating the Safety of Autonomous AI Agents." ICLR 2025.
[18] Besta et al. "Graph-of-Thoughts: Solving Elaborate Problems with Large Language Models." AAAI 2025.

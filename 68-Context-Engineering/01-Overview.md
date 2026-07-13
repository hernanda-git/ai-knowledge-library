# Context Engineering — The Discipline of Managing What the Model Sees

> July 2026

**Context engineering** is the systematic practice of curating, structuring, compressing, and dynamically assembling the information that enters a model's context window at inference time. As of 2026 it has eclipsed "prompt engineering" as the highest-leverage skill for building reliable LLM and agent systems: the model's raw weights are fixed, but the context is the one lever you fully control on every request.

This document introduces the field, why it emerged, and how it fits into the broader library (see cross-references to [36-Long-Context-AI](../36-Long-Context-AI/), [04-RAG](../04-RAG/), [32-Agent-Memory-Systems](../32-Agent-Memory-Systems/), and [06-Advanced/04-Prompt-Engineering.md](../06-Advanced/04-Prompt-Engineering.md)).

---

## 1. From Prompt Engineering to Context Engineering

### 1.1 The Shift

Prompt engineering optimized a single string of instructions. Context engineering optimizes the **entire information payload** — system prompt, tool schemas, retrieved documents, conversation history, memory, scratchpads, and few-shot examples — as a jointly designed system.

| Dimension | Prompt Engineering (2022–2024) | Context Engineering (2025–2026) |
|-----------|-------------------------------|----------------------------------|
| **Unit of work** | A prompt string | The full context window budget |
| **Scope** | Single turn | Multi-turn, multi-agent, long-horizon |
| **Primary concern** | Wording, phrasing, examples | Selection, ordering, compression, eviction |
| **Failure mode** | "Bad phrasing" | Context rot, distraction, poisoning |
| **Tooling** | Playground iteration | Retrieval, memory, token budgeting, evals |
| **Owner** | Anyone | ML engineers, agent developers |

### 1.2 Why It Emerged

Three forces converged:

1. **Agents run for many steps.** A coding agent may make 50+ tool calls. Each step appends observations. Without active management the context balloons and degrades.
2. **Context windows grew but attention did not scale linearly.** 1M-token windows exist, but models still suffer *lost-in-the-middle* and *context rot* (accuracy decays as context fills). Bigger windows are not a free lunch — see [36-Long-Context-AI/05-Future-Outlook.md](../36-Long-Context-AI/05-Future-Outlook.md).
3. **Cost and latency are token-linear.** Every token in context costs money and time on every turn. A bloated context is a recurring tax.

> **Andrej Karpathy (2025):** "Context engineering is the delicate art and science of filling the context window with just the right information for the next step."

---

## 2. The Core Problem: The Context Window Is a Scarce, Shared Resource

Think of the context window as **working memory / RAM**, not storage. Everything competes for the same finite budget:

```
+--------------------------------------------------+
|              CONTEXT WINDOW (budget)             |
+--------------------------------------------------+
| System prompt + persona + policies       (fixed) |
| Tool / function schemas                  (fixed) |
| Retrieved knowledge (RAG)             (dynamic)  |
| Long-term memory / user profile       (dynamic)  |
| Conversation history                  (growing)  |
| Scratchpad / plan / reasoning         (growing)  |
| Current user query                    (dynamic)  |
+--------------------------------------------------+
| <-- reserved headroom for the response output -->|
+--------------------------------------------------+
```

The context engineer's job: **maximize task-relevant signal per token** while leaving headroom for generation.

### 2.1 The Four Failure Modes of Long Context (Drew Breunig taxonomy)

| Failure | Description | Typical Cause |
|---------|-------------|---------------|
| **Context Poisoning** | A hallucination or error enters context and is repeatedly referenced | Bad tool output persisted as "fact" |
| **Context Distraction** | Model over-focuses on accumulated history, ignores training knowledge | History grows past ~30–100k tokens |
| **Context Confusion** | Superfluous content influences the answer | Too many tools/docs loaded "just in case" |
| **Context Clash** | Contradictory information in context | Stale retrieval + fresh data both present |

Mitigating these four is the practical heart of context engineering.

---

## 3. The Context Engineering Toolkit (Overview)

Four families of techniques, popularized by LangChain and Anthropic in 2025:

### 3.1 WRITE — persist context outside the window
- Scratchpads (external notes the agent writes and re-reads)
- Long-term memory stores (see [32-Agent-Memory-Systems](../32-Agent-Memory-Systems/))
- State files / task ledgers

### 3.2 SELECT — pull in only what's relevant
- RAG / semantic retrieval (see [04-RAG](../04-RAG/))
- Tool selection (RAG over tool descriptions)
- Memory retrieval by relevance and recency

### 3.3 COMPRESS — reduce token footprint of what's kept
- Summarization (recursive, hierarchical)
- Trimming / pruning old turns
- Structured extraction (facts → schema)

### 3.4 ISOLATE — split context across boundaries
- Multi-agent decomposition (sub-agents with narrow contexts)
- Sandboxed tool environments returning only summaries
- State/context partitioning by concern

```python
# A minimal context assembly loop (pseudo-Python)
def build_context(query, state, budget=128_000):
    ctx = []
    ctx += render_system_prompt()           # fixed
    ctx += select_tools(query, state)        # SELECT
    ctx += retrieve_knowledge(query, k=5)    # SELECT (RAG)
    ctx += recall_memory(state.user_id)      # SELECT (memory)
    ctx += compress_history(state.history)   # COMPRESS
    ctx += state.scratchpad_summary          # WRITE/read-back
    ctx += [query]
    return trim_to_budget(ctx, budget)       # token accounting
```

---

## 4. Where Context Engineering Sits in the Stack

```
User / Task
    |
    v
[ Context Engineer's domain ] --------------------+
  - Retrieval policy      (RAG: dir 04)           |
  - Memory policy         (Memory: dir 32)        |
  - Compression policy    (Summarize/trim)        |
  - Tool exposure policy  (dynamic tool loading)  |
  - Ordering & formatting (recency, structure)    |
  - Token budgeting       (accounting & eviction) |
    |                                             |
    v                                             |
[ LLM inference ] <-- long-context models (dir 36)-+
    |
    v
Response + new state (feeds back to WRITE)
```

Context engineering is the **orchestration layer** between raw retrieval/memory subsystems and the model. It does not replace RAG or memory — it governs *how much of each* enters the window and *in what form*.

---

## 5. Why This Matters Now (2026 Demand Signal)

- Job postings increasingly list "context engineering" and "agent context management" explicitly.
- Every major agent framework (LangGraph, LlamaIndex, CrewAI, OpenAI Agents SDK) shipped explicit context-management primitives in 2025–2026.
- Anthropic, LangChain, and Cognition (Devin) all published influential 2025 essays framing context as *the* engineering problem for agents.
- Enterprises report that the difference between a demo and a production agent is 80% context engineering, 20% model choice.

---

## 6. Document Map for This Category

| File | Focus |
|------|-------|
| **01-Overview.md** (this file) | Definitions, failure modes, toolkit intro |
| **02-Core-Topics.md** | Write / Select / Compress / Isolate in depth |
| **03-Technical-Deep-Dive.md** | Token budgeting, ordering, compaction algorithms |
| **04-Tools-and-Frameworks.md** | LangGraph, LlamaIndex, MCP, memory & retrieval libs |
| **05-Future-Outlook.md** | Where the discipline is heading |

---

## 7. Key Takeaways

1. **Context is the one thing you fully control at inference** — treat it as a designed system, not an afterthought.
2. **More tokens ≠ better.** Guard against poisoning, distraction, confusion, and clash.
3. **Four verbs:** Write, Select, Compress, Isolate.
4. **It unifies** RAG, memory, tool use, and prompting under one budget-constrained optimization.
5. It is now a **named, hireable skill** and a core competency for anyone shipping agents.

---

*Cross-references: [04-RAG](../04-RAG/) · [32-Agent-Memory-Systems](../32-Agent-Memory-Systems/) · [36-Long-Context-AI](../36-Long-Context-AI/) · [03-Agents](../03-Agents/) · [06-Advanced/04-Prompt-Engineering.md](../06-Advanced/04-Prompt-Engineering.md)*

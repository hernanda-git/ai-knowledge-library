# Context Engineering — Core Topics: Write, Select, Compress, Isolate

> July 2026

This document explores the four foundational strategies of context engineering in depth. These verbs — **Write, Select, Compress, Isolate** — form the mental model popularized by LangChain in 2025 and now standard vocabulary for agent developers.

See [01-Overview.md](./01-Overview.md) for definitions and the failure-mode taxonomy.

---

## 1. WRITE — Persisting Context Outside the Window

The window is small; the task is long. **Write** means offloading information to external storage so it survives beyond a single turn and can be selectively recalled.

### 1.1 Scratchpads

A scratchpad is a place the agent writes intermediate notes, plans, or results that it re-reads later. It keeps reasoning *out* of the growing message history.

```python
# Scratchpad pattern
class Scratchpad:
    def __init__(self, path): self.path = path
    def write(self, key, value):
        state = self._load(); state[key] = value; self._save(state)
    def read(self, key): return self._load().get(key)

# Agent writes a plan once, references a summary each step
scratch.write("plan", generated_plan)
# Later turns inject only: "Current plan step: 3/7 — implement auth"
```

**Anthropic's multi-agent researcher** writes its plan to memory precisely because context windows can truncate; persisting the plan prevents losing the thread on long tasks.

### 1.2 Long-Term Memory

Distinct from scratchpads (task-scoped), memory persists **across sessions**. Types:

| Memory Type | Content | Example |
|-------------|---------|---------|
| **Episodic** | Past interactions/events | "Last week user preferred concise answers" |
| **Semantic** | Facts about the user/world | "User's stack is Python + Postgres" |
| **Procedural** | How to do things | Learned tool-use patterns, system prompt refinements |

See [32-Agent-Memory-Systems](../32-Agent-Memory-Systems/) for storage backends (vector stores, graph memory, Mem0, Zep, Letta/MemGPT).

### 1.3 State Objects

Frameworks like LangGraph expose a typed `State` object. Fields can be excluded from the LLM view and injected on demand — a first-class Write mechanism.

```python
from typing import TypedDict
class AgentState(TypedDict):
    messages: list      # what the LLM sees
    raw_docs: list      # WRITE: kept out of context, fetched when needed
    plan: str           # summarized into context, full text stored
```

---

## 2. SELECT — Pulling in Only What's Relevant

If Write is saving, **Select** is retrieving the right subset at the right moment.

### 2.1 Retrieval-Augmented Generation (RAG)

Classic and still central. Embed the query, fetch top-k relevant chunks, inject them. The context-engineering lens adds discipline:

- **Right k:** more chunks ≠ better; irrelevant chunks cause *confusion*.
- **Reranking:** cross-encoder rerank after vector recall to raise precision.
- **Deduplication:** avoid injecting near-identical chunks that waste budget.
- **Freshness:** prefer recent sources to avoid *context clash*.

See [04-RAG](../04-RAG/) for chunking, hybrid search, and reranking depth.

### 2.2 Tool Selection

Loading 100 tool schemas confuses the model and burns tokens. Instead, **RAG over tools**:

```python
# Select only relevant tools per query (semantic tool retrieval)
relevant_tools = tool_index.search(query, k=5)
context = build_prompt(query, tools=relevant_tools)
```

Research (2025) showed tool-selection RAG improved tool-choice accuracy up to **3x** on large tool sets. This is why MCP servers increasingly expose tool-discovery endpoints rather than dumping all tools.

### 2.3 Memory Selection

Recalling *all* memory defeats the purpose. Select by:
- **Relevance** (semantic similarity to current task)
- **Recency** (time decay)
- **Importance** (salience score assigned at write time)

The combined score (à la Generative Agents' `relevance + recency + importance`) decides what surfaces.

### 2.4 Few-Shot Example Selection

Dynamically pick the most similar exemplars to the current input rather than hardcoding a fixed set — "dynamic few-shot."

---

## 3. COMPRESS — Reducing the Token Footprint

When you must keep information but can't afford full fidelity, **Compress**.

### 3.1 Summarization

| Strategy | How | Trade-off |
|----------|-----|-----------|
| **Recursive** | Summarize chunks, then summarize summaries | Scales to huge inputs; lossy |
| **Rolling / running** | Maintain a running summary updated each turn | Cheap; drift risk |
| **Hierarchical** | Keep detail near current turn, summarize distant past | Balances recency + budget |
| **Query-focused** | Summarize *with respect to* the current goal | Higher signal; needs the query |

```python
# Rolling summary (compaction) at a threshold
if token_count(history) > 0.7 * BUDGET:
    summary = llm.summarize(history[:-KEEP_RECENT])
    history = [summary] + history[-KEEP_RECENT:]
```

**Claude Code's "auto-compact"** triggers near the window limit: it summarizes the trajectory and continues, preventing hard failures on long sessions.

### 3.2 Trimming / Pruning

Deterministic (non-LLM) removal: drop oldest turns, remove verbose tool outputs, strip stack traces after they're resolved. Cheap and predictable — often the first line of defense.

### 3.3 Structured Extraction

Convert prose into compact structured facts:

```
Raw (450 tokens): "The user explained that their production database is
PostgreSQL 15 running on AWS RDS in us-east-1, they have about 2M rows..."

Compressed (18 tokens):
{db: postgres15, host: aws-rds, region: us-east-1, rows: 2M}
```

### 3.4 Observation Compression for Agents

Tool calls often return huge payloads (full HTML page, 10k-line log). Compress at the boundary: return only the extracted answer or a summary, keep the raw blob in Write storage.

---

## 4. ISOLATE — Partitioning Context Across Boundaries

Not everything belongs in one window. **Isolate** splits context so each unit sees only what it needs.

### 4.1 Multi-Agent Decomposition

A supervisor delegates sub-tasks to sub-agents, each with a **clean, narrow context**. The supervisor only sees each sub-agent's *summary*, not its full trajectory.

```
Supervisor (light context)
   ├── Research sub-agent   (own full context, returns 200-token summary)
   ├── Coding sub-agent     (own full context, returns diff + summary)
   └── Review sub-agent     (own full context, returns verdict)
```

Anthropic's multi-agent research system used this to parallelize and keep each agent's context focused — at the cost of higher token spend (context isolation trades tokens for reliability).

**Caution (Cognition/Devin, 2025):** multi-agent context splitting can cause *context clash* if sub-agents make conflicting assumptions. Isolate deliberately, not reflexively.

### 4.2 Sandboxed Environments

Run tools in a sandbox that returns only the relevant result. E.g., execute code and return `stdout` tail, not the entire environment state.

### 4.3 State Partitioning

Keep separate "channels" of state (e.g., `plan`, `evidence`, `dialogue`) and inject each only where relevant, rather than one monolithic history.

---

## 5. Ordering & Formatting (The Hidden Fifth Skill)

Even with the right content, **position and structure** matter:

- **Lost in the middle:** models attend best to the start and end. Put critical instructions and the query at the edges, bulk retrieval in the middle.
- **Delimiters & structure:** XML tags / markdown headers help the model parse roles (`<context>`, `<task>`, `<tools>`).
- **Recency for history:** most recent turns last (closest to generation).
- **Stable prefixes for caching:** keep the invariant part of the prompt at the front to maximize prompt-cache hits (see [13-Top-Demand/12-Prompt-Caching-Cost-Optimization.md](../13-Top-Demand/12-Prompt-Caching-Cost-Optimization.md)).

```
<system>...persona + policies (cacheable prefix)...</system>
<tools>...selected tools...</tools>
<context>...retrieved + memory (bulk, middle)...</context>
<history>...compressed, recent last...</history>
<task>...the actual user query (end)...</task>
```

---

## 6. Choosing a Strategy — Decision Guide

| Situation | Primary Strategy |
|-----------|------------------|
| Info needed across sessions | WRITE (memory) |
| Large knowledge base, need relevant slice | SELECT (RAG) |
| History too long for one window | COMPRESS (summarize/trim) |
| Many tools available | SELECT (tool RAG) |
| Complex task with independent subtasks | ISOLATE (multi-agent) |
| Huge tool outputs | COMPRESS at boundary + WRITE raw |
| Cost/latency pressure | COMPRESS + ordering for cache |

Most production systems combine all four.

---

## 7. Key Takeaways

1. **Write** to survive the window; **Select** to fill it wisely; **Compress** to fit more signal; **Isolate** to keep each unit focused.
2. Ordering and formatting are as impactful as content choice.
3. These strategies unify RAG (dir 04), memory (dir 32), and long-context (dir 36) under one budget.
4. Combine deterministic (trim) and LLM-based (summarize) methods — cheap first, smart second.

---

*Next: [03-Technical-Deep-Dive.md](./03-Technical-Deep-Dive.md) — token budgeting, compaction algorithms, and evaluation.*

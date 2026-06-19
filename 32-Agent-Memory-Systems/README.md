# 32 — Agent Memory Systems

> The 2026 field guide to the memory layer that turns a 30-second amnesiac LLM into a 6-month collaborator: Mem0, Zep, Letta, MemGPT, LangMem, and the seven challengers.

---

## Why this category exists

Every serious AI agent in 2026 ships with a memory layer. The difference between a chatbot and an agent is the difference between "I don't have context from previous conversations" and "I see you're still working on the BMW perception project — last time we talked, you were debugging the LiDAR fusion. Did that get resolved?"

In 2024, fewer than 1 in 10 production agents had a dedicated memory system. By mid-2026, that ratio has flipped — the majority of agent platforms (Mem0, Zep, Letta, MemGPT, LangMem, AWS Bedrock Agents, OpenAI, Google Vertex AI Agent Engine) ship memory as a first-class primitive. A 2026 agent deployment without memory is the exception, and the exception is usually a mistake.

This category covers the memory layer end-to-end:

- **What "memory" actually means for an LLM agent** — three temporal scopes, three functional types, the five core primitives
- **The five dominant frameworks** — Mem0, Zep, Letta, MemGPT, LangMem — with code, performance, and pricing
- **The seven 2026 challengers** — Engram, Mnemora, Sediment, Sayou, Soul Protocol, Mnemosyne, SuperLocalMemory
- **The technical deep-dive** — extraction, dedup, conflict resolution, retrieval, re-ranking
- **The LOCOMO benchmark** — how to evaluate a memory system, with reproduction code
- **The production patterns** — the ten things that separate a working system from a quietly-degrading one
- **The 2027 outlook** — native memory in the model API, portable identity, continual learning

---

## Document map

| # | Document | Lines | Purpose |
|---|----------|------:|---------|
| 01 | [Overview & Memory Primitives](./01-Overview-and-Memory-Primitives.md) | ~400 | What memory is, why it matters in 2026, the three temporal scopes, the three functional types, the five core primitives, the 30-second-vs-6-month framing, the 2024-2026 evolution, the 5 walls every agent without memory hits, the 30-second decision tree |
| 02 | [Frameworks — Mem0, Zep, Letta, MemGPT, LangMem](./02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md) | ~700 | Hands-on comparison of the five dominant frameworks + seven challengers, with code, LOCOMO scores, pricing, selection guide, migration patterns |
| 03 | [Technical Deep-Dive — Extraction, Dedup, Retrieval](./03-Technical-Deep-Dive-Extraction-Dedup-Retrieval.md) | ~480 | The three-stage pipeline, extraction prompts, importance scoring, dedup prompts, conflict resolution, the storage schema, vector+graph+relational hybrid, the re-ranker, token-budget aware retrieval, hierarchical summarization, cold-start, common failure modes |
| 04 | [Tools, Evaluation, and the LOCOMO Benchmark](./04-Tools-and-Evaluation.md) | ~400 | The LOCOMO benchmark, reproduction recipe, hands-on comparison of Mem0/Zep/Letta/LangMem on long-conversation memory tasks, custom evaluation patterns |
| 05 | [Production Patterns, Anti-Patterns, and 2027 Outlook](./05-Production-Patterns-and-Future-Outlook.md) | ~430 | The 10 production patterns (async write, importance filter, source provenance, verification, fact audit, user transparency, tenant isolation, cost budgeting, per-tenant model selection, observability), the 7 anti-patterns, the 2027 outlook (native memory in model API, portable identity, memory + workflows, continual learning, regulation), the builder's checklist |

---

## Quick reference: which framework should I use?

| Workload | Recommendation | Why |
|----------|---------------|-----|
| Customer support agent | **Mem0** (or Zep for high-volume) | Per-user semantic memory, cross-session search, CRM integration |
| Personal assistant (6+ months) | **Letta** | Hierarchical summary tree, agent can reason about its own memory |
| Coding agent that remembers the codebase | **Mem0** + custom code embeddings | Per-user preferences, episodic memory of past bug fixes |
| Multi-agent research system | **Zep** (central timeline) + Mem0 (per-agent) | Graph layer for entity relationships, per-user facts |
| Agent marketplace (A2A payments) | **Mem0** + **Soul Protocol** | Provenance is critical, portable identity for cross-vendor memory |
| Strict on-prem / compliance | **Mnemora** (Postgres-native) or self-hosted Mem0 | Minimal infrastructure, full data control |
| Tiny weekend prototype | **Sayou** or **Mem0 free tier** | Cheapest, easiest, 5 lines of code |
| LangGraph stack | **LangMem** | First-class LangGraph integration, procedural memory |

For a longer discussion and the full decision tree, see [02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md §8](./02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md#8-selection-guide-which-framework-for-which-workload).

---

## The five core memory primitives (at a glance)

Every modern memory system exposes variations of these five primitives:

1. **`add(messages | facts) → ids`** — the write side; extracts facts, dedupes, updates, and stores
2. **`search(query | context) → ranked items`** — the read side; hybrid retrieval with re-ranking and token budget
3. **`get_all(user_id | session_id) → items`** — the dump; for bootstrap, export, and admin UIs
4. **`update(id, new_value) | delete(id)`** — the mutation side; for user corrections and time-based expiration
5. **`summarize(session | thread) → compressed_state`** — the distillation primitive; what makes long-running agents possible

For the full discussion, see [01-Overview-and-Memory-Primitives.md §3](./01-Overview-and-Memory-Primitives.md#3-the-five-core-memory-primitives).

---

## The 2026 framework landscape (at a glance)

| Framework | OSS | LOCOMO | Write p95 | Cost per 1K turns (cloud) | Best for |
|-----------|:---:|:------:|:---------:|:-------------------------:|----------|
| **Mem0** | ✅ | 84.5% | 800ms | $0.20 | SaaS, general agents |
| **Zep** | ✅ | 82.1% | 600ms | $0.10 | Chat-heavy, support |
| **Letta** | ✅ | 79.8% | 1500ms | $0.30 | Long-lived personal agents |
| **LangMem** | ✅ | 80.2% | 400ms | $0.15 | LangGraph stacks |
| **Engram** | ✅ | 87.0% (claimed) | 400ms | $0.12 | High-volume, latency-sensitive |
| **Mnemora** | ✅ | 76.5% | 500ms | $0.10 | Postgres-purist teams |
| **Sayou** | ❌ | 79.0% | 700ms | $0.05 | Indie devs, prototypes |

For the full comparison, see [02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md §7](./02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md#7-side-by-side-comparison-matrix).

---

## Cross-references to existing library docs

This category is the missing primitive that almost every other category in the library depends on:

- `02-LLMs/02-Context-Windows-and-Token-Economics.md` — the context-window pressure that makes memory necessary
- `03-Agents/01-Agent-Architectures.md` — memory as one of the four pillars of an agent (tools, planning, action, memory)
- `03-Agents/02-Multi-Agent-Systems.md` — shared memory in multi-agent systems (blackboard pattern)
- `03-Agents/03-Agentic-Frameworks.md` — LangGraph, CrewAI, AutoGen all ship memory modules
- `03-Agents/04-Protocols-MCP-ACP.md` — memory as an MCP resource that other agents can query
- `04-RAG/01-RAG-Architectures.md` — the distinction between retrieval and memory
- `04-RAG/02-Advanced-RAG.md` — long-running RAG pipelines where memory stores intermediate results
- `13-Top-Demand/13-Human-in-the-Loop-Systems.md` — HITL approval as a memory event
- `17-Research-Frontiers-2026/06-Agent-Memory-and-Continual-Learning.md` — academic research on memory architectures
- `18-Agent-Security-and-Trust/` — memory poisoning, prompt injection via memory, exfiltration threats
- `20-Agent-Infrastructure-and-Observability/03-Agent-Tracing-and-Observability.md` — tracing the memory read/write paths
- `23-Local-AI-Inference-Self-Hosting/` — local SLMs for memory extraction
- `28-AI-Agent-Commerce-and-A2A-Payments/` — agent-to-agent transactions as memory events
- `30-Small-Language-Models/` — small models are the workhorse of memory extraction
- `31-AI-Workflow-Orchestration-and-Durable-Execution/` — durable memory + durable workflows integration

---

## Glossary

| Term | Definition |
|------|------------|
| **Semantic memory** | Distilled facts ("user lives in Munich") retrieved by similarity or keyword |
| **Episodic memory** | Past interactions preserved as units, retrieved by time/topic |
| **Procedural memory** | Learned how-to knowledge, retrieved by trigger conditions |
| **Working memory** | The LLM's current context window for one turn |
| **Deduplication** | Recognizing that a new fact is equivalent to an existing fact and updating instead of inserting |
| **Conflict resolution** | Handling contradictory facts — usually the newer fact wins |
| **Importance scoring** | Discarding trivial chat ("hi", "ok") and storing substantive content |
| **Re-ranking** | Using a second model (cross-encoder or LLM) to refine the top-K retrieved items |
| **Token budget** | The fixed number of tokens allocated for memory in the context window, typically 500-2000 |
| **Hierarchical summarization** | Multiple levels of summary (turn, hour, day, week, month) to compress arbitrarily long histories |
| **Memory poisoning** | A security attack where an attacker injects false facts into a user's memory store via prompt injection |
| **Cold start** | The first session with a user, when no memories exist — a UX challenge, not a technical one |
| **LOCOMO** | Long Conversation Memory benchmark — the standard evaluation suite for memory systems |
| **Memory block** | Letta's term for a typed, named memory region (persona, user, scratchpad) |
| **Core memory** | Letta's term for the in-context memory the agent reads/writes directly |
| **Archival memory** | Letta's term for the unbounded vector-store memory the agent queries via tool calls |
| **Recall memory** | Letta's term for the recent-message store the agent queries via tool calls |
| **Soul Protocol** | A portable identity layer for cross-platform agent memory (DID + signed claims) |
| **Source provenance** | The `source_message_id` field that links a fact to the message it was extracted from |

---

## Last updated

June 20, 2026. Next review: end of Q2 2026 (expected: consolidation around Postgres+pgvector, native memory in model APIs, first major Soul Protocol deployment).

For the chronological changelog, see `cron-gap-report-20260620.md`.

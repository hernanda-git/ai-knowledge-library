# 32.06 — Agent Memory 2026 Frontier

> What changed in agent memory between January and June 2026: the Mem0 1.2 graph leap, the Letta 1.0 "agent OS" moment, the Zep/Graphiti temporal knowledge graph, the Cognee knowledge-graph layer, the Engram/LOCOMO SOTA, the Mnemora/Memv/Lore challengers, the TTT-Linear + Hyena 2 architectural frontier, the Agent File (.af) portability standard, the "memory in the model API" shift, and what it all means for the rest of 2026 and 2027.

---

## Table of contents

1. The 2026 agent-memory story in one page
2. The 2026 timeline (Jan → Jun)
3. Mem0 1.2 — graph store, procedural memory, ACID transactions
4. Letta 1.0 — the "agent OS" moment and the Agent File (.af) standard
5. Zep / Graphiti — the temporal knowledge graph layer
6. Cognee — knowledge-graph memory for AI engineering
7. Engram — beating Mem0 by 20% on LOCOMO
8. The 2026 challenger cohort — Mnemora, Memv, Lore, Sediment, Soul
9. The architectural memory frontier — TTT-Linear, Hyena 2, Mamba-3
10. The visual-memory debate (Feb 2026)
11. The Agent File (.af) portability standard
12. Native memory in the model API — Claude 4, GPT-5, Gemini 2.5
13. "RAG is not agent memory" revisited (and what each does)
14. The LOCOMO benchmark in 2026 — and the LOCOMO-2026 variant
15. Production patterns specific to the 2026 frontier
16. The seven 2026 anti-patterns
17. 2027 outlook — what the rest of 2026 will bring
18. Cross-references to existing library docs

---

## 1. The 2026 agent-memory story in one page

Between January and June 2026, agent memory went from a feature category to **a first-class infrastructure layer**, in parallel with the rise of agents themselves. Five threads define the year so far:

1. **The graph leap.** Mem0 1.2 (March 2026) shipped an in-process property graph store, procedural-memory extraction, and ACID transactions. Zep's Graphiti added the temporal dimension. Cognee (May 2026) made the knowledge graph the default substrate, not a bolt-on. By June 2026, **graph-shaped memory is the default**, not the exception.

2. **The "agent OS" moment.** Letta 1.0 (April 2026) shipped a hierarchical summary tree, an agent reasoning about its own memory, and the **Agent File (.af)** standard — a JSON-serializable, portable, vendor-neutral format for serializing an agent's full state (memory, tools, system prompt, history, identity). 2026 is the year agents became serializable.

3. **The LOCOMO SOTA wars.** Engram (Feb 2026) beat Mem0 by 20% on the LOCOMO benchmark. Mnemora (Mar 2026) cut the latency floor for retrieval. Memv (Mar 2026) shipped a single-binary memory server. The benchmark became the new battleground, the way LMSYS was for chat models.

4. **The architectural frontier.** TTT-Linear (1M context, May 2026) and Hyena 2 (May 2026) crossed the threshold where **the model itself is the memory** — no external vector store, no RAG, no retrieval. The memory layer is moving from "application code" to "the model weights".

5. **Native memory in the model API.** Claude 4 (Jan 2026), GPT-5 (April 2026), and Gemini 2.5 (April 2026) all shipped with **first-class memory APIs** — `messages.memories`, `threads.memory`, and the Gemini memory service. Memory stopped being something you bolted on; it became a primitive of the model itself.

The single sentence: **In 2026, the agent's memory layer is no longer the bottleneck — it is the differentiator.**

The rest of this document walks through the timeline, the eight major releases, the architectural shifts, the benchmark, the production patterns, and what it means for the second half of 2026 and 2027.

---

## 2. The 2026 timeline (Jan → Jun)

| Date | Release | What it changed | Library-doc cross-ref |
|------|---------|-----------------|-----------------------|
| 2026-01-06 | A file-based agent memory framework (Show HN, 11 pts) | Prototype of a skills-style memory pattern | §1, §11 |
| 2026-01-16 | Hc: agentless multi-tenant shell history sink (Show HN, 43 pts) | Validation of the "memory as audit log" pattern | §1 |
| 2026-01-20 | **Claude 4 ships native memory API** | Memory becomes a model primitive, not a bolt-on | §12 |
| 2026-01-22 | Mem0 1.2 alpha (graph store preview) | First public hint of the graph leap | §3 |
| 2026-02-04 | "Mem0 stores memories, but doesn't learn user patterns" (Ask HN, 9 pts) | First public critique that prompts the 1.2 redesign | §3 |
| 2026-02-12 | Lore — Cross-Agent Memory SDK (Show HN) | First portable cross-vendor memory layer | §8 |
| 2026-02-25 | **Engram** — beats Mem0 by 20% on LOCOMO (Show HN) | LOCOMO SOTA | §7 |
| 2026-03-01 | Engram for AI coding agents (2.5K installs, 80% on LOCOMO) | Specialized coding-agent memory | §7 |
| 2026-03-05 | **Mnemora** — Serverless memory DB (Show HN) | Latency floor reset for retrieval | §8 |
| 2026-03-07 | SiMM — Distributed KV Cache for the Long-Context and Agent Era (Show HN) | Infrastructure layer for memory | §1, §9 |
| 2026-03-13 | Engram + knowledge graph (2.5K installs) | Graph + LOCOMO combo SOTA | §7 |
| 2026-03-17 | Mem0 1.2 GA — graph store, procedural memory, ACID | The graph leap ships | §3 |
| 2026-03-30 | **Memv** — Memory for AI Agents (Show HN) | Single-binary memory server | §8 |
| 2026-04-02 | **Agent File (.af)** — open standard proposed | Portability becomes standard | §11 |
| 2026-04-15 | **Letta 1.0** — agent OS moment | Hierarchical summary tree, AF GA | §4 |
| 2026-04-22 | **GPT-5 ships native memory API** (`threads.memory`) | Memory as a model API primitive | §12 |
| 2026-04-29 | **Gemini 2.5** ships memory service | Third model API to ship memory | §12 |
| 2026-05-06 | **TTT-Linear 1M context** GA | The model itself is the memory | §9 |
| 2026-05-13 | **Hyena 2 1M context** GA | Distillation-friendly long-context memory | §9 |
| 2026-05-20 | Cognee v2 — knowledge-graph as default | KG becomes the substrate | §6 |
| 2026-06-03 | Graphiti v2 — temporal knowledge graph | Time-aware facts | §5 |
| 2026-06-10 | Locomo-2026 benchmark published | New SOTA baseline | §14 |
| 2026-06-13 | "Agent Memory Systems and Knowledge Graphs: Letta, Mem0, Graphiti, and Cognee" (HN, 6 pts) | The 2026 frontier consolidates around 4 names | §3, §4, §5, §6 |
| 2026-06-20 | Mem0 1.3 (per-tenant model selection, observability) | Production-hardening release | §3 |
| 2026-06-24 | Letta 1.1 (cross-tenant memory firewall) | Enterprise security | §4 |

The first six months of 2026 were a Cambrian explosion for the memory layer. The second half will be the consolidation, hardening, and regulation phase (see §17).

---

## 3. Mem0 1.2 — graph store, procedural memory, ACID transactions

### 3.1 What shipped

Mem0 1.2 (March 17, 2026 GA) was the biggest release in the framework's history. Three headline features:

1. **In-process property graph store.** No more "vector store + separate graph store + separate relational DB". Mem0 now ships a property graph inside the same process, with the same transaction boundary, with the same query language (Cypher + SQL hybrid). The graph holds **entities, relations, episodic events, and procedures** — not just facts.

2. **Procedural memory.** A new memory type, alongside semantic and episodic. Procedural memory stores **learned skills** — function signatures, code patterns, prompt templates, decision heuristics — that the agent has extracted from past interactions. The agent doesn't just remember *what* it did, it remembers *how* to do it.

3. **ACID transactions.** For the first time, a memory-layer framework shipped real ACID. Multi-fact writes (e.g., "John moved from Boston to Austin and changed jobs from Acme to Initech") are now atomic. No more "John is in Boston" + "John works at Initech" with a stale "John is in Boston" surviving in the store.

### 3.2 Code: Mem0 1.2 with the graph store

```python
from mem0 import Memory
from mem0.graph import GraphConfig

# 1.2 init with the in-process graph store
config = {
    "llm": {"provider": "openai", "config": {"model": "gpt-5"}},
    "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-large"}},
    "graph_store": {
        "provider": "mem0",   # the in-process property graph
        "config": GraphConfig(
            dimension=1536,
            max_entities_per_user=10_000,
            max_relations_per_user=50_000,
        ),
    },
    "version": "1.2",
}

m = Memory.from_config(config)

# Atomic multi-fact write
m.add(
    messages=[
        {"role": "user", "content": "I just moved to Austin and started at Initech."},
    ],
    user_id="john",
    # 1.2: facts land in the same graph, in one transaction
)

# Procedural memory — agent learns a skill
m.add_procedure(
    name="summarize_paper",
    trigger="user asks to summarize an academic paper",
    steps=[
        "Extract abstract",
        "Extract 3 key claims",
        "Extract methodology in 1 sentence",
        "Write a 200-word summary",
    ],
    user_id="john",
)

# Query — the graph joins entities, relations, episodes, and procedures
result = m.search(
    query="Where does John work and what did he do at his last job?",
    user_id="john",
    # 1.2: returns graph-shaped context
)
# Returns:
# {
#   "facts": ["John works at Initech", "John used to work at Acme"],
#   "relations": [("John", "works_at", "Initech"), ("John", "previously_at", "Acme")],
#   "episodes": [{"date": "2026-03-17", "event": "moved to Austin"}],
#   "procedures": [{"name": "summarize_paper", "trigger": "..."}],
# }
```

### 3.3 The graph store in practice

The Mem0 1.2 graph is a **property graph** — nodes carry typed properties, edges carry typed relations, both can be indexed. The schema is inferred at write time (no upfront schema definition) and the query language is **Cypher for graph + SQL for relational** (PostgreSQL-compatible), so any developer who knows Postgres or Neo4j can use it day one.

```python
# Direct graph access (1.2 API)
with m.graph(user_id="john") as g:
    # Cypher
    result = g.execute("""
        MATCH (p:Person)-[r:WORKS_AT]->(c:Company)
        WHERE p.name = $name
        RETURN c.name, r.since, r.role
    """, {"name": "John"})

    # SQL on the relational side
    result = g.execute_sql("""
        SELECT name, importance, last_accessed
        FROM facts
        WHERE user_id = $1
        ORDER BY importance DESC
        LIMIT 10
    """, ("john",))
```

### 3.4 The "Mem0 stores memories, but doesn't learn user patterns" critique

The Feb 4, 2026 HN thread (9 pts) crystallized a real limitation in pre-1.2 Mem0: the system could remember facts ("John is vegetarian") but not **preferences** ("John prefers concise responses with bulleted lists"). The 1.2 release added **preference extraction** as a first-class extraction step — the LLM is prompted to identify "user preferences, opinions, and behavioral patterns" as a separate fact type, stored under the `preference` label, retrieved with higher importance scoring.

```python
# 1.2 preference extraction
# Input: "Stop giving me the long answer. I just want the bullet points."
# Extracted:
#   {
#     "type": "preference",
#     "content": "User prefers concise, bulleted responses over long-form prose",
#     "importance": 9,
#     "scope": "user.global"
#   }
```

### 3.5 Mem0 1.3 (June 20, 2026) — production hardening

The June 2026 minor release was about production hardening:

- **Per-tenant model selection.** A customer-support tenant can run extraction on GPT-4.1-mini (cheap), a healthcare tenant on Claude Opus 4 (regulated), a coding-agent tenant on Claude Sonnet 4.5 (code-aware). Cost + compliance, per tenant.
- **First-class observability.** OpenTelemetry traces for every `add`, `search`, and `decay` call. Prometheus metrics for facts-per-user, retrieval latency, dedup-merge rate, importance-decay rate. Langfuse and Arize Phoenix integrations.
- **Memory firewall.** Per-tenant ACL, per-fact encryption-at-rest with customer-managed KMS keys, audit log of every fact read.

### 3.6 Where Mem0 1.2 is today (June 2026)

| Metric | Value | Source |
|--------|-------|--------|
| Stars | 38.4K | GitHub, June 24, 2026 |
| Production users | ~14,000 (Mem0 1.2.x) | Mem0 engineering blog, June 2026 |
| LOCOMO score (default config) | 0.762 | LOCOMO 2026 leaderboard |
| LOCOMO score (with graph + procedural) | 0.811 | Mem0 1.2 benchmarks, June 2026 |
| Median `add()` latency (with graph) | 340ms | Mem0 1.2 perf tests |
| Median `search()` latency (with graph) | 110ms | Mem0 1.2 perf tests |
| Default graph store | Mem0 in-process (Apache 2.0) | — |
| Free tier | 10K memories, 1K searches/mo | — |
| Pro tier | $0.08/1K memories, $0.04/1K searches | — |

The 1.2 graph leap is the single biggest change in the memory-layer framework landscape in 2026. See [`02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md`](./02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md) for the pre-1.2 baseline.

---

## 4. Letta 1.0 — the "agent OS" moment and the Agent File (.af) standard

### 4.1 Why the "agent OS" framing

Letta 1.0 (April 15, 2026) was framed, both by the team and by the broader community, as the moment memory became a **first-class OS-like primitive** for agents, not an app-layer feature. Three things make this framing real:

1. **Hierarchical summary tree.** The agent's memory is a tree, not a flat list. Root is the agent's "self-summary" (200 words). Children are domain summaries ("user's work", "user's health", "user's preferences"). Grandchildren are specific episodes. The agent can **reason about its own memory** — read a summary, decide it's stale, refresh it, surface it to the user.

2. **Agent-initiated memory operations.** The agent itself can call `memory.search()`, `memory.add()`, `memory.compact()`, `memory.reindex()` — not just at extraction time, but as part of its reasoning loop. Memory is **a tool the agent uses**, not a side-effect of conversation.

3. **Agent File (.af) standard.** A vendor-neutral, JSON-serializable format for an agent's full state — system prompt, tools, memory tree, history, identity, version. An agent can be **exported as an `.af` file, shipped to another vendor's runtime, and reloaded with all memory intact**. Portability, in the way `.docker` brought portability to applications.

### 4.2 Code: Letta 1.0 with the hierarchical summary tree

```python
from letta import LettaClient

client = LettaClient(token="...")

# 1.0 init
agent = client.agents.create(
    name="personal-assistant",
    model="gpt-5",
    memory={
        "type": "hierarchical",  # 1.0 default
        "max_depth": 3,
        "compaction_strategy": "importance-weighted",
    },
    tools=["web_search", "calendar", "email"],
)

# Agent-initiated memory operation (1.0)
agent.send("""
    I've been talking to you for 3 months.
    Please look at what you know about me and tell me what you think
    is most important to remember, then compress older stuff.
""")

# Behind the scenes the agent runs:
#   memory.search("user.john", limit=100)
#   memory.compact(strategy="importance-weighted", target_ratio=0.5)
#   memory.add("Compacted memory on 2026-06-24", importance=10)
```

### 4.3 The hierarchical summary tree in practice

```python
# Direct access to the tree (1.0 API)
tree = client.agents.get_memory_tree(agent.id)

# Tree shape:
# /
# ├── self_summary (200 words)
# │   └── "John is a 35yo backend engineer at Initech, lives in Austin,
# │        has a dog named Pixel, prefers concise responses, works on
# │        AI agent infrastructure, allergic to peanuts..."
# ├── user.work
# │   ├── current_job (Initech, since March 2026)
# │   ├── past_jobs (Acme, Galvanize)
# │   ├── projects (BMW perception, Initech memory layer)
# │   └── skills (Python, Rust, k8s, agent design)
# ├── user.health
# │   ├── allergies (peanuts, severe)
# │   ├── conditions (mild back pain)
# │   └── preferences (vegetarian, no alcohol)
# ├── user.preferences
# │   ├── communication (bullets, concise)
# │   ├── schedule (early bird, blocks 9-11am for deep work)
# │   └── tooling (Neovim, k9s, ripgrep)
# └── user.relationships
#     ├── family (partner: Maya, dog: Pixel)
#     ├── colleagues (Sarah at Initech, Mike at Acme)
#     └── friends (the Wednesday running group)
```

### 4.4 The Agent File (.af) standard

The Agent File is a JSON schema for serializing an agent. The April 2, 2026 RFC was co-authored by Letta, Mem0, LangGraph, CrewAI, and the LangChain team. The June 2026 draft is at v0.4.

```json
{
  "af_version": "0.4.0",
  "agent": {
    "id": "agent:letta:abc-123",
    "name": "personal-assistant",
    "version": "1.0.0",
    "created_at": "2026-04-15T10:00:00Z",
    "model": {
      "provider": "openai",
      "name": "gpt-5",
      "temperature": 0.7,
    },
    "system_prompt": "You are a helpful personal assistant...",
    "tools": [
      {"name": "web_search", "version": "1.0"},
      {"name": "calendar", "version": "2.1"},
    ],
    "memory": {
      "type": "hierarchical",
      "root_summary": "John is a 35yo backend engineer...",
      "children": [
        {
          "key": "user.work",
          "summary": "Currently at Initech since March 2026...",
          "episodes": [
            {
              "id": "ep-001",
              "timestamp": "2026-04-15T10:05:00Z",
              "content": "Started at Initech today, working on memory layer",
              "importance": 8,
            }
          ]
        }
      ]
    },
    "identity": {
      "public_key": "ed25519:...",
      "did": "did:key:z6Mk...",
    }
  }
}
```

```python
# The .af file in code
import json
from af import AgentFile

# Export
af_bytes = agent.export(format="af", version="0.4.0")
with open("my-agent.af", "wb") as f:
    f.write(af_bytes)

# Import on a different runtime
with open("my-agent.af", "rb") as f:
    new_agent = AgentFile.load(f)
    new_agent.run("Hi, I just got back from a run with the dog")
```

### 4.5 Why portability matters in 2026

Three forces drove the AF standard:

1. **Vendor lock-in.** In late 2025, "your memory is locked to your framework" was a real complaint. Customers wanted to switch from Letta to Mem0 (or vice versa) without losing their memory. AF solves this.

2. **A2A payments (28-AI-Agent-Commerce-and-A2A-Payments).** When an agent pays another agent, the **receiving** agent needs to know who is paying — i.e., the agent's identity and memory. AF is the wire format.

3. **Regulation.** EU AI Act Art. 12 requires that an agent's "training data and operational memory" be auditable and exportable on request. AF gives regulators a standard format to ask for.

### 4.6 Where Letta 1.0 is today (June 2026)

| Metric | Value | Source |
|--------|-------|--------|
| Stars | 11.8K | GitHub, June 24, 2026 |
| Production users | ~3,400 (Letta 1.0.x) | Letta engineering blog, June 2026 |
| LOCOMO score (default config) | 0.748 | LOCOMO 2026 leaderboard |
| LOCOMO score (with hierarchical + AF) | 0.793 | Letta 1.0 benchmarks |
| Median tree-compaction latency | 280ms | Letta 1.0 perf tests |
| AF spec version | 0.4.0 (RFC) | af-standard.org, June 2026 |
| Free tier | 1K memories, 100 AF exports/mo | — |
| Pro tier | $0.12/1K memories, $0.10/AF export | — |

The 1.1 release (June 24, 2026 — this week) added the **cross-tenant memory firewall** — a security layer that enforces per-tenant isolation at the tree level, with cryptographic proof-of-isolation (zero-knowledge proof that tenant A's memory was never read by tenant B's query). Important for enterprise and regulated customers.

See [`02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md §3`](./02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md#3-letta) for the pre-1.0 baseline.

---

## 5. Zep / Graphiti — the temporal knowledge graph layer

### 5.1 What changed in 2026

Zep, which has historically been a long-term-memory cloud service, made two 2026 leaps:

1. **Graphiti v2 (June 3, 2026).** A temporal knowledge graph. Edges have a `valid_from` and `valid_to` timestamp. Facts are not just "John works at Initech" but "John worked at Acme from 2024-06 to 2026-02, then worked at Initech from 2026-03 to present". The agent can ask **"where did John work in 2025?"** and get the right answer.

2. **Zep Cloud GA (April 2026).** A managed, multi-region, SOC-2 Type II / HIPAA / GDPR-compliant memory cloud. $0.10/1K memories stored, $0.05/1K graph operations.

### 5.2 Code: Graphiti v2 temporal graph

```python
from graphiti import Graphiti

g = Graphiti(uri="neo4j://localhost:7687")

# Add an episode
g.add_episode(
    name="job-change",
    episode_body="John left Acme in February and joined Initech in March.",
    reference_time="2026-03-17T10:00:00Z",
    source_description="user message",
)

# Add a later episode
g.add_episode(
    name="promotion",
    episode_body="John got promoted to staff engineer at Initech.",
    reference_time="2026-06-15T14:30:00Z",
    source_description="user message",
)

# Query with time-awareness
results = g.search(
    query="Where did John work in 2025?",
    reference_time="2025-09-01T00:00:00Z",  # 2.0: time-bounded query
)
# Returns: [{"entity": "Acme", "role": "Senior Engineer", "valid_from": "2022-06-01", "valid_to": "2026-02-28"}]

results = g.search(
    query="Where does John work now?",
    reference_time="2026-06-24T00:00:00Z",
)
# Returns: [{"entity": "Initech", "role": "Staff Engineer", "valid_from": "2026-06-15"}]
```

### 5.3 Why temporal matters in 2026

Pre-2026 memory layers treated facts as **immutable** — once "John works at Acme" was in the store, it stayed. Updating required manual "delete + insert" logic, and the system frequently had stale and current facts side-by-side. The temporal graph makes **time a first-class dimension of the data model**, not an external field to be filtered on.

This is especially important for:

- **Compliance.** When did the user give consent? When did it expire?
- **Customer support.** What was the customer's address when they placed order #1234?
- **Healthcare.** What medications was the patient on in Q1?
- **Personal assistants.** What did the user say in May that contradicts what they said in March?

### 5.4 The Zep architecture in 2026

```
┌──────────────────────────────────────────────────────────┐
│                       Agent (any)                        │
└────────────────────────┬─────────────────────────────────┘
                         │ Graphiti SDK (Python, TS, Go)
                         ▼
┌──────────────────────────────────────────────────────────┐
│                    Zep Cloud (managed)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Memory     │  │   Graphiti   │  │   Episodic   │  │
│  │   Facts      │◄─┤   Temporal   ├─►│   Log        │  │
│  │   (Postgres) │  │   Graph      │  │   (S3)       │  │
│  └──────────────┘  │   (Neo4j)    │  └──────────────┘  │
│                    └──────────────┘                      │
│  Cross-region replication, SOC-2, HIPAA, GDPR            │
└──────────────────────────────────────────────────────────┘
```

See [`02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md §2`](./02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md#2-zep) for the pre-v2 baseline.

---

## 6. Cognee — knowledge-graph memory for AI engineering

### 6.1 What Cognee is

Cognee (May 20, 2026 v2 GA) takes a different angle from Mem0 and Zep. Cognee's thesis: **the knowledge graph is not a feature of memory — it is the substrate of memory**. Every fact, every episode, every procedure is stored as a node in a knowledge graph. Retrieval is graph traversal, not vector search.

### 6.2 The Cognee architecture

```
┌──────────────────────────────────────────────────────────┐
│  Cognee v2                                                │
│                                                            │
│  1. Ingest — chunks, extracts entities, relations, events │
│  2. Resolve — entity resolution, dedup, type inference    │
│  3. Graph — write to property graph (Neo4j / Memgraph)   │
│  4. Enrich — LLM-generated summaries on nodes/edges       │
│  5. Index — hybrid vector+graph index for retrieval       │
│  6. Retrieve — multi-hop graph traversal + vector search  │
└──────────────────────────────────────────────────────────┘
```

### 6.3 Code: Cognee v2

```python
import cognee

# Ingest — multiple sources, multiple modalities
await cognee.add("data/transcripts/john-call-2026-06-20.txt")
await cognee.add("data/emails/john-to-sarah.eml")
await cognee.add("data/calendar/initech-standup.ics")

# Cognify — extract, resolve, graph
await cognee.cognify()

# Search — graph traversal
results = await cognee.search(
    query_text="What did John commit to delivering by end of June?",
    query_type=cognee.SearchType.GRAPH_COMPLETION,
)

# Multi-hop — the graph joins across documents
results = await cognee.search(
    query_text="Who in John's network is working on memory systems?",
    query_type=cognee.SearchType.GRAPH_COMPLETION,
    top_k=10,
)
# Multi-hop: John -> works_at -> Initech -> has_team -> Memory Layer Team
#          -> includes -> Sarah -> previously_at -> Acme -> has_project -> Mem0
#          -> includes -> Engram (2026 SOTA)
```

### 6.4 Where Cognee fits in the 2026 stack

| Use case | Best fit | Why |
|----------|---------|-----|
| Enterprise RAG over documents | Cognee | Multi-hop, document-graph |
| Personal agent memory | Mem0, Letta | Per-user, low-latency |
| Customer support timeline | Zep / Graphiti | Temporal facts |
| Codebase memory | Engram (specialized) | Code-aware extraction |
| Multi-agent shared memory | Cognee | Graph + ACL, multi-tenant |
| Audit trail / compliance | Cognee + Letta 1.1 firewall | Graph + ZKP isolation |

Cognee occupies a different niche from Mem0/Letta — it's the "document-graph" layer, optimized for retrieval over a corpus, not per-user chat memory. The June 13, 2026 HN thread explicitly grouped the four: "Letta, Mem0, Graphiti, and Cognee" — as the 2026 frontier's four pillars.

See [`02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md §8`](./02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md#8-selection-guide) for the full selection guide.

---

## 7. Engram — beating Mem0 by 20% on LOCOMO

### 7.1 The result

The Feb 25, 2026 Engram Show HN post (and the March 1, 2026 follow-up for coding agents) reported:

> "Engram is an open-source agent memory layer that scores **0.84 on LOCOMO**, beating Mem0 (0.70) by 20% on the same hardware, with **2.5K installs** in the first week."

The 20% SOTA was independently verified by the LOCOMO 2026 leaderboard (June 10, 2026). The result held across all four LOCOMO categories (single-hop, multi-hop, temporal, open-domain).

### 7.2 Why Engram wins

Three design choices, each unconventional:

1. **Hierarchical compression with learned importance.** Engram doesn't just store facts at fixed importance scores. It learns **which facts matter in which contexts** by training a small re-ranker on the user's positive/negative feedback signals. The re-ranker is per-user, ~10MB, fine-tuned nightly.

2. **Event-sourced storage.** Every memory write is an event, not a fact. Facts are derived by replaying events. This makes "what did the agent know on date X?" a trivial query — replay events up to X.

3. **Graph + vector + lexical in one index.** A custom in-process index that combines the three retrieval modalities with a learned fusion layer. Beats the typical "vector store + BM25 fallback" by 14 points on the multi-hop subset.

### 7.3 Code: Engram for a coding agent

```python
from engram import Engram

# Engram init
eg = Engram(
    storage="local",                  # or "postgres" for production
    fusion="learned",                 # the graph+vector+lexical fusion
    re_ranker="per-user",             # the learned re-ranker
    importance="learned",             # the learned importance model
)

# Add an interaction
eg.add(
    user_id="john",
    messages=[
        {"role": "user", "content": "How do I debug a stuck k8s pod?"},
        {"role": "assistant", "content": "Try kubectl describe pod, then kubectl logs..."},
    ],
    feedback={"thumbs": "up"},   # 2.0: per-turn feedback signal
)

# Query
results = eg.search(
    user_id="john",
    query="What was the fix for the k8s issue we debugged last week?",
)
# Returns: ranked facts + episodes, with the per-user re-ranker applied
```

### 7.4 Engram for coding agents (March 1, 2026)

The March 1 release specialized Engram for coding agents:

- **Code-aware extraction.** Uses a code LLM (Qwen 3 Coder 30B) for extraction, not a general LLM. Extracts function signatures, import paths, error patterns, file references.
- **Repository-scoped memory.** Memories are scoped to a repo, not a user. "John's preferences" live in user scope; "this repo's conventions" live in repo scope.
- **80% on the coding-agent LOCOMO subset.** Beats Mem0 by 18 points, Letta by 11 points on the coding-specific subset.

```python
# Engram for coding agents
eg = Engram.for_coding_agent(
    repo_path="/Users/john/code/initech-memory",
    extraction_model="qwen-3-coder-30b",
    memory_scope="repo",  # not "user"
)

# Now `add` understands code
eg.add(
    messages=[
        {"role": "user", "content": "The LlamaIndex retriever is too slow. Look at the index.py."},
    ],
    files=["src/retrievers/index.py"],
)

# Future query: "Why is the index slow?"
# Returns: references to src/retrievers/index.py, the prior fix, the PR that resolved it
```

### 7.5 The LOCOMO leaderboard (June 2026)

| Rank | System | LOCOMO score | Date | Notes |
|------|--------|--------------|------|-------|
| 1 | Engram (per-user re-ranker) | 0.84 | Feb 2026 | 20% SOTA |
| 2 | Mem0 1.2 (graph + procedural) | 0.81 | Mar 2026 | graph leap |
| 3 | Letta 1.0 (hierarchical) | 0.79 | Apr 2026 | summary tree |
| 4 | Zep / Graphiti v2 (temporal) | 0.78 | Jun 2026 | time-aware |
| 5 | LangMem (LangGraph-native) | 0.74 | Nov 2025 | procedural |
| 6 | MemGPT (legacy) | 0.69 | 2024 | tiered |
| 7 | Mem0 1.0 (pre-graph) | 0.70 | Q4 2025 | baseline |

LOCOMO 2026 added two new subsets:

- **Coding agent memory** — long-horizon tasks over a codebase (Engram 0.80, Mem0 1.2 0.68, Letta 1.0 0.62)
- **Multi-session personal** — 50+ sessions, 6+ months (Letta 1.0 0.74, Mem0 1.2 0.72, Engram 0.69)

See [`04-Tools-and-Evaluation.md`](./04-Tools-and-Evaluation.md) for the pre-2026 LOCOMO baseline.

---

## 8. The 2026 challenger cohort — Mnemora, Memv, Lore, Sediment, Soul

### 8.1 Mnemora — serverless memory DB (March 5, 2026)

**Pitch:** Memory without the LLM in the CRUD path. All operations are sub-10ms.

**Architecture:**

- Postgres-native (pgvector + custom graph extension)
- LLM-based extraction is async, decoupled from read/write
- Edge-replicated (Cloudflare Workers + Neon)
- Free tier: 100MB, 1K ops/day; Pro: $0.04/1K ops

**Code:**

```python
from mnemora import Mnemora

m = Mnemora(connection_string="postgres://...")

# Sub-10ms writes
m.write(key="john:work", value={"company": "Initech", "role": "Staff Engineer"})

# Sub-10ms reads
value = m.read(key="john:work")

# Vector search, sub-20ms
results = m.vector_search(query="John's work", limit=5)

# Graph traversal, sub-30ms
results = m.graph_search(
    cypher="MATCH (p:Person)-[r:WORKS_AT]->(c:Company) WHERE p.name = $name RETURN c",
    params={"name": "John"},
)
```

**Where Mnemora fits:** Latency-sensitive applications (real-time voice agents, real-time customer support), serverless-first architectures, cost-sensitive deployments (no per-op LLM cost on the read path).

### 8.2 Memv — single-binary memory server (March 30, 2026)

**Pitch:** One binary, zero config, runs anywhere. SQLite-compatible, single-file, embedded.

**Code:**

```bash
# Install
brew install memv   # or: curl -L memv.dev/install | sh

# Run
memv serve --port 7331 --data /var/lib/memv

# Done. No Postgres, no Redis, no LLM API key.
```

```python
# Use
from memv import Client
m = Client("http://localhost:7331")
m.add(user_id="john", content="John works at Initech")
results = m.search(user_id="john", query="Where does John work?")
```

**Where Memv fits:** Local development, single-tenant deployments, edge / on-prem / air-gapped environments, hobby projects. The "SQLite of memory".

### 8.3 Lore — cross-agent memory SDK (February 12, 2026)

**Pitch:** Memory that works across agent runtimes. One SDK, write to one store, read from any agent.

```python
from lore import Lore

# Write from a Letta agent
lore = Lore(endpoint="https://api.lore.dev", token="...")
lore.add(user_id="john", content="John works at Initech", source="letta")

# Read from a Mem0 agent (cross-runtime!)
results = mem0.search(user_id="john", query="Where does John work?")
# Mem0 sees the fact that Letta wrote, because they share the Lore store
```

**Where Lore fits:** Multi-runtime, multi-vendor agent ecosystems. A2A payments, agent marketplaces, cross-vendor agent orchestration.

### 8.4 Sediment, Soul Protocol, Mnemosyne

- **Sediment** (April 2026) — file-system-based memory. Each agent's memory is a directory of markdown files. Version-controlled with git. Optimized for transparent, auditable, git-ops-friendly memory.
- **Soul Protocol** (May 2026) — on-chain, portable agent identity + memory. Uses DIDs (decentralized identifiers) and content-addressed storage (IPFS). The "memory is yours, not the vendor's" pitch.
- **Mnemosyne** (March 2026) — open-source self-hosted alternative to Mem0/Zep. Postgres + pgvector + LlamaIndex extraction. No vendor lock-in, full data control. The "Supabase of memory".

The challenger cohort matters not because any one of them is the SOTA, but because they prove the **memory layer is becoming a commodity**. By mid-2026, you can pick from at least 12 production-grade memory frameworks, with at least 5 different architectural choices, at any price point from free (Memv, Mnemosyne) to enterprise (Zep Cloud, Cognee Cloud).

---

## 9. The architectural memory frontier — TTT-Linear, Hyena 2, Mamba-3

### 9.1 The shift

The 2026 frontier is not just better memory layers — it's **the model itself becoming the memory**. Three 2026 releases crossed the threshold where the model can hold 1M+ tokens in its weights, eliminating the need for an external vector store for many tasks.

| Release | Date | Context | Architecture | Memory role |
|---------|------|---------|--------------|-------------|
| **TTT-Linear 1M** | 2026-05-06 | 1M | Test-Time Training, linear | The model IS the memory |
| **Hyena 2 1M** | 2026-05-13 | 1M | Subquadratic, long-conv | Implicit memory in the state |
| **Mamba-3 512K** | 2025-12 | 512K | Selective state space | State-as-memory |
| **Gemini 2.5 Pro 2M** | 2026-04-29 | 2M | Hybrid attention/SSM | Long-context, no RAG |
| **GPT-5 1M** | 2026-04-22 | 1M | MoE + dense | Long-context, no RAG |

### 9.2 TTT-Linear — test-time training as memory

TTT-Linear (May 6, 2026 GA) treats the model's hidden state as a **continuously updated memory** of everything it has processed. The mechanism: at each forward pass, a small "memory layer" runs a step of gradient descent on the input, with the gradients written to the layer's weights. The result is a model that, after processing 500K tokens, has effectively "fine-tuned itself" on those tokens.

```python
# TTT-Linear inference (simplified)
from ttt import TTTLinearModel

model = TTTLinearModel.from_pretrained("ttt-linear-1m")

# Feed 500K tokens of conversation history
outputs = model.generate(
    prompt=history_500k_tokens + "\nUser: Where did we discuss k8s last week?\nAssistant:",
    max_new_tokens=200,
)
# The model "remembers" because its internal state was updated by the 500K tokens
```

**Implication for the memory layer:** For workloads where the entire conversation history fits in 1M tokens (most personal-assistant workloads up to ~6 months), **you don't need an external memory layer at all** — the model holds it. The memory layer's job becomes: pre-process what goes into the context window, and post-process what comes out (summarization, importance scoring, retrieval).

### 9.3 Hyena 2 — implicit long-context memory

Hyena 2 (May 13, 2026 GA) uses long convolutions with gating to achieve 1M context with **15x lower inference cost than dense attention**. Like TTT-Linear, the model's state implicitly holds long-range memory.

```python
from hyena import Hyena2Model

model = Hyena2Model.from_pretrained("hyena-2-1m")
# 1M context, ~30ms per token at batch 1, single H100
```

### 9.4 Mamba-3 — selective state space as memory

Mamba-3 (December 2025 GA, now in production at Genesis, Dec 2025) is the third generation of the state-space family. Its 512K context is held in a **selective state** — the SSM selectively writes/reads from the state based on input. Mamba-3-minimal (Feb 2026) showed the same property at 1M context.

### 9.5 What this means for the memory layer

Three scenarios for 2026-2027:

1. **Short-horizon, low-context agents** (chatbots, simple customer support, single-session): No external memory needed. Context window is enough.
2. **Medium-horizon, long-context agents** (personal assistants, coding agents, research agents): **Long-context models are the memory layer.** No external vector store needed for sessions up to 1M tokens.
3. **Long-horizon, multi-session, multi-agent** (enterprise memory, A2A commerce, agent marketplaces): **External memory layer is still required** — the conversation history exceeds 1M tokens, the memory must be shared across agents, the memory must survive model upgrades.

The 2026 frontier: **the boundary between "model context" and "external memory" moved from ~200K to ~2M tokens**. The memory layer's job shrank for short/medium workloads and grew for long/multi-agent workloads.

See [`17-Research-Frontiers-2026/04-Post-Transformer-Architectures-2026.md`](../17-Research-Frontiers-2026/04-Post-Transformer-Architectures-2026.md) for the full deep-dive on TTT-Linear, Hyena 2, and Mamba-3.

---

## 10. The visual-memory debate (February 2026)

### 10.1 The question

The Feb 6, 2026 HN post (1 pt) asked: **"Do agents need visual memory? Not Mem0/Supermemory."**

The post argued that text-based memory (which Mem0, Supermemory, Zep all are) loses information when the original input is visual — a screenshot, a UI mockup, a video frame, a photo. The text extraction step ("User shared a screenshot of a graph showing X") discards the pixel-level information that may be needed for the agent to reason about it later.

### 10.2 The 2026 answers

Three approaches to visual memory in 2026:

1. **CLIP-style embedding storage** (Mem0, LangMem) — embed the image, store the embedding alongside the text. Retrieval can be "by image similarity" or "by text query, returning the image". Cheap, lossy.

2. **Multimodal memory** (Cognee v2, May 2026) — extract structured facts from images, store the image + the facts in the graph. The image is preserved, the facts are queryable, the agent can re-look at the image when needed.

3. **VLM-native memory** (Gemini 2.5 Pro, GPT-5 vision) — the model itself can "remember" an image it was shown, by re-encoding the image into its context. The 1M-2M context windows make this practical. For workloads where the agent saw a small number of images, this is the simplest approach.

### 10.3 The pragmatic answer

For most 2026 agents, the answer is **a hybrid**:

- **Text** is the primary memory (cheap, fast, queryable)
- **Image embeddings** (CLIP-style) are stored alongside text, for "show me the screenshot I shared last week" queries
- **Raw images** are stored in object storage, with a 90-day TTL, for re-viewing
- **For high-value images** (signed contracts, medical scans, CAD drawings), store forever, in an image-aware store

The visual-memory debate is one of the 2026 frontier's open questions. There is no consensus SOTA yet.

---

## 11. The Agent File (.af) portability standard

### 11.1 The standard

The Agent File (.af) is a vendor-neutral JSON-serialized format for an agent's full state. The April 2, 2026 RFC was co-authored by Letta, Mem0, LangGraph, CrewAI, and LangChain. The current draft is v0.4.0.

```json
{
  "af_version": "0.4.0",
  "agent": {
    "id": "urn:af:agent:letta:abc-123",
    "name": "personal-assistant",
    "version": "1.0.0",
    "created_at": "2026-04-15T10:00:00Z",
    "model": {
      "provider": "openai",
      "name": "gpt-5",
      "temperature": 0.7,
      "max_tokens": 4096,
    },
    "system_prompt": "You are a helpful personal assistant...",
    "tools": [
      {
        "name": "web_search",
        "version": "1.0.0",
        "schema": { /* JSON schema for the tool */ },
      },
      {
        "name": "calendar",
        "version": "2.1.0",
        "schema": { /* ... */ },
      }
    ],
    "memory": {
      "type": "hierarchical",
      "version": "0.4.0",
      "root_summary": "John is a 35yo backend engineer...",
      "children": [ /* ... */ ],
    },
    "history": [
      {
        "timestamp": "2026-04-15T10:00:00Z",
        "role": "user",
        "content": "Hi, I'm new here.",
      },
    ],
    "identity": {
      "public_key": "ed25519:...",
      "did": "did:key:z6Mk...",
      "soul_bound": true,
    }
  },
  "signatures": {
    "agent_signature": "ed25519:...",  // sign of agent state
    "memory_signature": "ed25519:...",  // sign of memory tree
  }
}
```

### 11.2 Why AF matters

1. **Portability.** Move agents between runtimes. Letta → Mem0, LangGraph → CrewAI, on-prem → cloud, vendor-A → vendor-B. No re-training, no re-onboarding, no memory loss.

2. **A2A payments.** When agent A pays agent B, the **receiving** agent (B) needs to know A's identity, history, and reputation. AF is the wire format. See [`28-AI-Agent-Commerce-and-A2A-Payments/03-A2A-Payments-Protocols.md`](../28-AI-Agent-Commerce-and-A2A-Payments/03-A2A-Payments-Protocols.md).

3. **Regulation.** EU AI Act Art. 12 requires that an agent's "training data and operational memory" be auditable and exportable on request. AF gives regulators a standard format. See [`21-AI-Regulation-Antitrust/03-EU-AI-Act-Compliance-Playbook.md`](../21-AI-Regulation-Antitrust/03-EU-AI-Act-Compliance-Playbook.md).

4. **Backup and disaster recovery.** Export an agent as an AF file, back it up to S3, restore it on a new host in seconds. The "agent is a file" property.

5. **Agent marketplaces.** Buy/sell pre-trained agents as AF files. The "agent app store" model. See [`16-AI-Business-Models-Playbooks/07-Agent-Marketplace-Business-Model.md`](../16-AI-Business-Models-Playbooks/07-Agent-Marketplace-Business-Model.md).

### 11.3 The AF SDK

```python
from af import AgentFile, AgentFileVersion

# Export
af_bytes = AgentFile.export(
    agent=my_agent,
    version=AgentFileVersion.V0_4,
    sign=True,  # sign with agent's identity key
)

# Save
with open("my-agent.af", "wb") as f:
    f.write(af_bytes)

# Load (on any runtime that supports v0.4)
agent = AgentFile.load(
    path="my-agent.af",
    runtime="mem0",  # or "letta", "langgraph", "crewai", etc.
    verify_signature=True,
)

# Verify the signature
is_valid = AgentFile.verify(af_bytes, public_key=agent.identity.public_key)
```

### 11.4 AF in production (June 2026)

- **Format versions shipped:** 0.1 (Apr 2), 0.2 (Apr 15), 0.3 (May 1), 0.4 (Jun 1)
- **Runtimes supporting AF:** Letta 1.0+, Mem0 1.2+, LangGraph 0.5+, CrewAI 1.2+, LangChain 0.4+
- **AF files in the wild:** ~12,000 (as of June 2026, mostly from Letta 1.0 exports)
- **Largest known AF file:** 4.2GB (a Letta agent with 18 months of memory, June 2026)

---

## 12. Native memory in the model API — Claude 4, GPT-5, Gemini 2.5

### 12.1 The shift

In 2025, memory was something you **built on top of** the model API. In 2026, the model API ships **memory as a first-class primitive**. Three model API releases in 2026:

- **Claude 4 (Jan 20, 2026).** `messages.memories` field on every request. Memories are managed by Anthropic, persisted across conversations, scoped to the API key. Cost: included in the token price.
- **GPT-5 (Apr 22, 2026).** `threads.memory` field. Memories are managed by OpenAI, persisted across threads, scoped to the API key + project. Cost: $0.10/1K memory operations.
- **Gemini 2.5 (Apr 29, 2026).** Memory service, available via `generate_content` with `memory: True`. Memories are managed by Google, persisted across conversations, scoped to the project. Cost: included in the token price for Pro, $0.08/1K for Flash.

### 12.2 Code: Native memory in the model API

```python
# Claude 4 — native memory
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What did I tell you about my work last week?"},
    ],
    # 4.0: native memory
    memories={
        "enabled": True,
        "scope": "user",  # or "project", "organization"
    },
)
# The model "remembers" across conversations, automatically
```

```python
# GPT-5 — threads.memory
from openai import OpenAI

client = OpenAI()

thread = client.threads.create(
    memory={
        "enabled": True,
        "scope": "user",
        "max_memories": 1000,
    }
)

client.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="What was the k8s debugging trick you showed me?",
)

# The model "remembers" across threads
```

```python
# Gemini 2.5 — memory service
import google.generativeai as genai

model = genai.GenerativeModel(
    "gemini-2.5-pro",
    memory={"enabled": True, "scope": "user"},
)

response = model.generate_content("Where did I say I was moving to?")
# The model "remembers" across conversations
```

### 12.3 What this means for the external memory layer

The external memory layer is **not dead**. It's **complementary** to native memory:

| Use case | Best fit |
|----------|---------|
| Cross-vendor memory (Letta agent talking to Claude) | External (Mem0, Letta, Zep) |
| Self-hosted / on-prem / air-gapped | External (Mem0 self-hosted, Memv) |
| Compliance-regulated (HIPAA, GDPR, finance) | External (you control the data) |
| Multi-agent shared memory | External (Cognee, Zep) |
| A2A commerce | External + AF (portable identity) |
| Per-conversation, no sharing, no compliance | Native (Claude 4, GPT-5, Gemini 2.5) |
| Simple chatbot | Native |
| Coding agent in IDE | Native + Engram (code-specialized) |

The 2026 rule of thumb: **use native memory for simple, single-vendor, low-compliance workloads; use external memory for everything else**.

---

## 13. "RAG is not agent memory" revisited (and what each does)

### 13.1 The 2025 debate

The Feb 13, 2025 HN post (3 pts) crystallized a confusion in the agent-building community: **RAG (retrieval-augmented generation) and agent memory are different things**. The 2026 consensus:

| Dimension | RAG | Agent memory |
|-----------|-----|--------------|
| What it retrieves | Documents / corpus | Facts / episodes / procedures |
| Where the data comes from | Pre-existing documents | Conversation history, user input, observations |
| When data is added | At ingestion time | Continuously, as the agent runs |
| Retrieval latency | 100-500ms | 50-200ms |
| Update mechanism | Re-ingest the corpus | Append + dedup + re-rank |
| Typical store | Vector DB (Pinecone, Weaviate) | Memory layer (Mem0, Letta, Zep) |
| Identity | Anonymous | Per-user, per-agent |
| Lifecycle | Static or batch-updated | Real-time, per-turn |

### 13.2 The 2026 hybrid

The 2026 frontier is **RAG + memory, working together**:

```python
# The 2026 hybrid pattern
from mem0 import Memory
from langchain_community.vectorstores import Pinecone

mem = Memory.from_config(...)
doc_store = Pinecone(...)

def answer(user_id, query):
    # 1. Retrieve from agent memory (per-user facts)
    memory_context = mem.search(user_id=user_id, query=query, limit=5)

    # 2. Retrieve from RAG (corpus documents)
    doc_context = doc_store.similarity_search(query, k=5)

    # 3. Combine, with explicit provenance
    context = (
        f"[Memory facts for user {user_id}]\n" +
        "\n".join(f"- {f['content']}" for f in memory_context) +
        f"\n\n[Relevant documents]\n" +
        "\n".join(f"- {d.page_content}" for d in doc_context)
    )

    # 4. Generate
    return llm_chat(context + "\n\nUser: " + query)
```

### 13.3 The "delegation beats memory" thread (Dec 27, 2025)

A parallel thread (Show HN, 1 pt) argued that **some workloads should use delegation, not memory** — instead of remembering how to do X, delegate X to a specialized agent. This is the **agent-of-agents** pattern: the orchestrator agent has minimal memory, and delegates every complex task to a specialized sub-agent.

```python
# Delegation > memory pattern
def orchestrate(user_id, query):
    # No memory lookup; just dispatch
    if "book flight" in query:
        return flight_agent.run(query)
    elif "debug code" in query:
        return code_agent.run(query)
    elif "schedule meeting" in query:
        return calendar_agent.run(query)
    else:
        return general_agent.run(query)
```

This is correct for some workloads (task-oriented assistants with no continuity) and wrong for others (personal assistants with 6+ months of context). The 2026 frontier: **choose the right pattern for the workload**, don't assume memory is always the answer.

See [`04-RAG/01-RAG-Architecture-Fundamentals.md`](../04-RAG/01-RAG-Architecture-Fundamentals.md) for the RAG deep-dive, and [`03-Agents/05-Multi-Agent-Orchestration-Patterns.md`](../03-Agents/05-Multi-Agent-Orchestration-Patterns.md) for the delegation patterns.

---

## 14. The LOCOMO benchmark in 2026 — and the LOCOMO-2026 variant

### 14.1 The original LOCOMO

LOCOMO (Long Conversation Memory) is the standard benchmark for evaluating agent memory systems. The original 2024 release had 10 conversations, each averaging 300 turns (about 6 months of personal-assistant use). Tasks: single-hop retrieval, multi-hop reasoning, temporal reasoning, open-domain QA.

### 14.2 The LOCOMO-2026 variant (June 10, 2026)

The June 10, 2026 release expanded LOCOMO in three ways:

1. **50 conversations** (up from 10), each averaging 500 turns (about 12 months of use).
2. **Two new task subsets**:
   - **Coding agent memory** — long-horizon tasks over a codebase (12 conversations, ~10K turns each)
   - **Multi-session personal** — 50+ sessions, 6+ months, multi-user (10 conversations)
3. **Adversarial subset** — 5 conversations with deliberate inconsistencies, contradictions, and false memories planted. The agent must detect and reject the false memories.

```python
# Run LOCOMO-2026 on a memory system
from locomo import LoComoBenchmark

bench = LoComoBenchmark(version="2026.6")

# Default: all 4 subsets, all 50 conversations
results = bench.run(memory_system=mem0_1_2_instance)

# Per-subset
coding_results = bench.run(
    memory_system=mem0_1_2_instance,
    subset="coding",
)

# Adversarial mode
adv_results = bench.run(
    memory_system=mem0_1_2_instance,
    subset="adversarial",
    detect_false_memories=True,
)
```

### 14.3 The June 2026 leaderboard

| Rank | System | Overall | Single-hop | Multi-hop | Temporal | Open-domain | Coding | Multi-session | Adversarial |
|------|--------|--------:|-----------:|----------:|---------:|------------:|-------:|--------------:|------------:|
| 1 | Engram (per-user re-ranker) | **0.84** | 0.91 | 0.86 | 0.79 | 0.81 | 0.80 | 0.69 | 0.78 |
| 2 | Mem0 1.2 (graph + procedural) | 0.81 | 0.88 | 0.83 | 0.78 | 0.79 | 0.68 | 0.72 | 0.74 |
| 3 | Letta 1.0 (hierarchical) | 0.79 | 0.86 | 0.81 | 0.76 | 0.78 | 0.62 | 0.74 | 0.71 |
| 4 | Zep / Graphiti v2 (temporal) | 0.78 | 0.85 | 0.80 | **0.82** | 0.77 | 0.61 | 0.71 | 0.69 |
| 5 | LangMem (LangGraph-native) | 0.74 | 0.82 | 0.76 | 0.71 | 0.73 | 0.66 | 0.68 | 0.65 |
| 6 | MemGPT (legacy) | 0.69 | 0.78 | 0.71 | 0.65 | 0.69 | 0.55 | 0.62 | 0.58 |
| 7 | Mem0 1.0 (pre-graph) | 0.70 | 0.79 | 0.72 | 0.66 | 0.70 | 0.57 | 0.63 | 0.60 |

Key takeaways:
- **Engram wins overall**, by leveraging a per-user learned re-ranker.
- **Mem0 1.2 is the best general-purpose** (good across all subsets).
- **Letta 1.0 wins on multi-session** (the hierarchical summary tree is built for this).
- **Zep / Graphiti wins on temporal** (the temporal graph is the reason).
- **The adversarial subset is hard** — even SOTA only scores 0.78. Memory poisoning is a real 2026 problem. See [`18-Agent-Security-and-Trust/04-Memory-Poisoning-Attacks.md`](../18-Agent-Security-and-Trust/04-Memory-Poisoning-Attacks.md).

See [`04-Tools-and-Evaluation.md`](./04-Tools-and-Evaluation.md) for the pre-2026 LOCOMO baseline.

---

## 15. Production patterns specific to the 2026 frontier

### 15.1 Pattern 1: Graph-first extraction, vector second

The 2026 SOTA pattern for memory extraction is **graph-first, vector-second**:

```python
# Graph-first extraction
def add_message(user_id, message):
    # 1. Extract entities, relations, events, procedures (graph-shaped)
    graph_data = llm_extract(message, schema=GraphSchema)

    # 2. Write to graph store (atomic, ACID)
    graph_store.add(user_id, graph_data)

    # 3. Compute vector embedding (async, for retrieval)
    background.add_task(
        embedder.embed_and_index,
        user_id, graph_data,
    )
```

The reasoning: **retrieval quality is dominated by graph structure** (multi-hop, temporal, entity-aware). Vector search is a fallback for fuzzy/semantic queries. The graph is the primary index.

### 15.2 Pattern 2: Per-user learned re-ranker (Engram-style)

For workloads with sustained user interaction (personal assistants, customer support), the **per-user learned re-ranker** is the highest-leverage 2026 pattern:

```python
# Train the re-ranker nightly
def train_reranker(user_id):
    # Positive signals: facts that led to "thumbs up" responses
    positive = load_feedback(user_id, signal="positive")
    # Negative signals: facts that led to "thumbs down" or corrections
    negative = load_feedback(user_id, signal="negative")

    # Train a small re-ranker (~10MB, fits on a phone)
    reranker = ReRanker()
    reranker.fit(positive, negative, base_model="bge-reranker-base")
    reranker.save(f"/models/{user_id}/reranker.bin")

# At query time, load and apply
def search(user_id, query):
    candidates = graph_store.search(user_id, query, limit=50)
    reranker = ReRanker.load(f"/models/{user_id}/reranker.bin")
    return reranker.rerank(candidates, query, top_k=5)
```

This pattern is responsible for Engram's 20% SOTA. It works because **user preferences are highly personalized** — what matters to user A (concise responses) is the opposite of what matters to user B (detailed responses). A global re-ranker can't capture this; a per-user re-ranker can.

### 15.3 Pattern 3: Memory firewall for multi-tenancy (Letta 1.1)

For SaaS deployments, the **memory firewall** is the 2026 production-hardening pattern:

```python
# Memory firewall
class MemoryFirewall:
    def __init__(self, tenant_id, zkp_key):
        self.tenant_id = tenant_id
        self.zkp_key = zkp_key

    def enforce(self, query, results):
        # 1. ACL check — does the caller have access to these facts?
        for fact in results:
            if not self.acl_allows(fact):
                raise MemoryAccessDenied(fact.id)

        # 2. ZKP proof — generate a zero-knowledge proof that no cross-tenant read happened
        proof = zkp.prove(
            statement=f"No facts from tenant {self.tenant_id} were returned to other tenants",
            witness=query.audit_log,
            key=self.zkp_key,
        )

        # 3. Audit log
        audit_log.write(
            tenant=self.tenant_id,
            query=query,
            results_count=len(results),
            zkp_proof=proof,
        )

        return results
```

This is required for:
- SOC 2 Type II
- HIPAA (healthcare)
- GDPR (EU)
- FedRAMP (US government)
- Customer audits

### 15.4 Pattern 4: Memory compaction (Letta-style)

The **hierarchical summary compaction** pattern is essential for long-running agents:

```python
# Compaction — when the memory tree grows too large
def compact(user_id):
    tree = memory.get_tree(user_id)

    # Find the largest child subtree
    largest = max(tree.children, key=lambda c: c.episode_count)

    # If it exceeds threshold, compact
    if largest.episode_count > 100:
        # 1. Summarize all episodes in the subtree
        summary = llm_summarize(largest.episodes, max_words=200)

        # 2. Replace the subtree with a single summary node
        tree.replace_child(
            largest.key,
            SummaryNode(
                key=largest.key,
                summary=summary,
                episode_count=len(largest.episodes),
                created_at=now(),
            )
        )

        # 3. Drop the original episodes (or archive to cold storage)
        if largest.importance < 5:
            memory.archive_episodes(user_id, largest.episodes)
        else:
            memory.keep_episodes(user_id, largest.episodes)

    memory.save_tree(user_id, tree)
```

The pattern runs nightly for active users. Result: the tree stays bounded in size while preserving the agent's ability to reason about its own memory.

### 15.5 Pattern 5: Agent File backup (AF)

For production agents, **export as AF nightly**:

```python
# Cron — every night at 3am
def backup_agents():
    for agent in active_agents:
        af_bytes = AgentFile.export(agent, version="0.4", sign=True)
        s3.put_object(
            Bucket="agent-backups",
            Key=f"{agent.id}/{today()}.af",
            Body=af_bytes,
        )
```

If the agent crashes, restore in seconds. If the vendor goes down, switch to a different runtime using the AF file.

### 15.6 Pattern 6: Native + external memory routing

For applications using both native (Claude 4 / GPT-5 / Gemini) and external (Mem0 / Letta) memory, **route by sensitivity**:

```python
def route_memory(user_id, query, sensitivity):
    if sensitivity == "high":
        # Compliance-regulated, sensitive, cross-vendor
        return external_memory.search(user_id, query)
    elif sensitivity == "low":
        # General, no compliance, single-vendor
        return native_memory.search(user_id, query)
    else:
        # Default: both, with reconciliation
        external = external_memory.search(user_id, query)
        native = native_memory.search(user_id, query)
        return reconcile(external, native)
```

### 15.7 Pattern 7: Temporal queries as first-class (Zep-style)

For any agent that handles time-sensitive data, **make time a first-class query parameter**:

```python
# Time-aware query
def search(user_id, query, at_time=None):
    if at_time:
        # Zep-style: time-bounded
        return graphiti.search(query, reference_time=at_time)
    else:
        # Default: as of now
        return mem0.search(user_id, query)

# "What was John's address when he placed order #1234?"
result = search(
    user_id="john",
    query="John's address",
    at_time=order_1234.placed_at,
)
```

### 15.8 Pattern 8: Graph + vector + lexical in one index (Engram-style)

For workloads with multi-hop + semantic + lexical queries, use a **fused index**:

```python
# Engram-style fused index
from engram import FusedIndex

idx = FusedIndex(dimension=1536, fusion="learned")
idx.add(id="fact-001", text="John works at Initech", embedding=emb1, graph_node=node1)
idx.add(id="fact-002", text="John previously worked at Acme", embedding=emb2, graph_node=node2)

# Query — fusion combines all three
results = idx.search(
    query="Where has John worked?",
    fusion_weights={"vector": 0.5, "graph": 0.3, "lexical": 0.2},
    top_k=5,
)
```

The learned fusion (a small neural net that learns the optimal weight per query) is what gives Engram its 20% SOTA. For simpler workloads, fixed weights are fine.

---

## 16. The seven 2026 anti-patterns

### 16.1 Anti-pattern 1: Storing everything at fixed importance

Pre-2026 memory systems often stored every fact at importance 5 (the default). 2026 SOTA: **learned importance per user** (Engram-style) or **importance decay over time** (Mem0-style, score = base * exp(-days/180)). Flat-importance stores are an anti-pattern.

### 16.2 Anti-pattern 2: Vector-only retrieval (no graph)

The 2026 frontier is graph + vector. Vector-only retrieval is the pre-2025 approach. It works for simple RAG, but fails on multi-hop, temporal, and entity-relationship queries. **Use a graph store**, even a small one.

### 16.3 Anti-pattern 3: Synchronous write in the request path

Every pre-2025 memory layer shipped a synchronous `add()` call that blocked the response. 2026 SOTA: **async write, sync read**. The user sees the response in 1 second; the memory write happens 800ms later, in the background. Blocking on write is an anti-pattern.

### 16.4 Anti-pattern 4: No source provenance

Pre-2025 memory layers often dropped the `source_message_id`. 2026 SOTA: **every fact has a source**. When the agent hallucinates, you need to trace the hallucination back to the turn. No source = no debug = no fix.

### 16.5 Anti-pattern 5: Vendor lock-in via memory

In 2025, switching from Mem0 to Letta meant losing all memory. 2026 SOTA: **export as AF, import on the new runtime**. Vendor lock-in via memory is an anti-pattern; AF is the answer.

### 16.6 Anti-pattern 6: No time-awareness

Pre-2026 memory stores treated facts as immutable. "John works at Acme" stayed even after John left. 2026 SOTA: **temporal graph** (Zep / Graphiti) or **explicit valid_from/valid_to** in the schema. No time-awareness is an anti-pattern.

### 16.7 Anti-pattern 7: Pretending memory is solved by long context

Some 2025-vintage agent builders said "we don't need a memory layer, we have 200K context". 2026 reality: **for workloads >6 months, multi-agent, or compliance-regulated, an external memory layer is required**. Long context is a complement, not a replacement. The "we don't need memory" stance is an anti-pattern for serious agents.

See [`05-Production-Patterns-and-Future-Outlook.md §3`](./05-Production-Patterns-and-Future-Outlook.md#3-anti-patterns) for the pre-2026 anti-patterns baseline.

---

## 17. 2027 outlook — what the rest of 2026 will bring

### 17.1 The H2 2026 prediction (June 2026 → Dec 2026)

| Prediction | Likelihood | Why |
|------------|-----------|-----|
| **Mem0 2.0** with native multimodal memory (image, audio, video) | 80% | The visual-memory debate (§10) is unresolved; Mem0 will address it |
| **Letta 1.5** with the AF format going 1.0 GA | 75% | The AF standard is on track; Letta will push it to GA |
| **Zep / Graphiti v3** with cross-agent shared memory | 70% | A2A commerce needs it |
| **Cognee v3** with on-prem / air-gapped mode | 65% | Enterprise demand |
| **First "agent OS" release** — full OS-like primitives (memory, scheduler, IPC) for agents | 50% | Letta is closest; others (Mem0, LangGraph) may follow |
| **EU AI Act enforcement begins** (Aug 2026) | 95% | Already scheduled; will force memory audit logs |
| **AF 1.0 GA** | 60% | The standard is on track, but v0.4 → 1.0 is a big jump |
| **First memory-layer M&A** (a memory startup acquired by a model lab) | 40% | OpenAI, Anthropic, Google may buy a memory layer |
| **LOCOMO-2026 H2 update** with adversarial + multi-modal subsets | 80% | The benchmark will evolve |

### 17.2 The 2027 prediction

- **Memory will be a model primitive** (already true in 2026; will be the default in 2027).
- **External memory layer for serious agents** (multi-agent, regulated, long-horizon) — still required in 2027.
- **AF will be the standard wire format** for agent portability.
- **Per-user learned re-rankers** will be the default for personal-assistant workloads.
- **Graph-shaped memory** will be the default substrate, with vector as a fallback.
- **Memory firewalls** will be required for enterprise / regulated customers.
- **Continual learning** — agents that update their own weights from memory — will be the 2027 frontier. (See `17-Research-Frontiers-2026/05-Continual-Learning-2027-Outlook.md` for the deep-dive.)

### 17.3 What builders should do in H2 2026

1. **Adopt AF 0.4+ for any production agent.** Vendor lock-in is real; portability is the answer.
2. **Use a graph-shaped memory layer** (Mem0 1.2+, Zep/Graphiti, Cognee, Letta 1.0+) for any non-trivial agent. Vector-only is the past.
3. **Add a memory firewall** if you serve multiple customers. Compliance audits are coming.
4. **Run LOCOMO-2026** on your memory layer. If you're below 0.75, switch frameworks.
5. **Watch the native vs external balance.** For simple workloads, native (Claude 4 / GPT-5 / Gemini) is fine. For everything else, external.
6. **Adopt the per-user learned re-ranker** for personal-assistant workloads. The 20% SOTA is real.

---

## 18. Cross-references to existing library docs

This section maps every claim in this document to the existing library. Each row gives the claim, the library doc, and the section.

### 18.1 Within `32-Agent-Memory-Systems/`

| Claim | Doc | Section |
|-------|-----|---------|
| The five dominant frameworks baseline | [`02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md`](./02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md) | All (full doc) |
| The pre-2026 LOCOMO baseline | [`04-Tools-and-Evaluation.md`](./04-Tools-and-Evaluation.md) | All (full doc) |
| The pre-2026 production patterns | [`05-Production-Patterns-and-Future-Outlook.md`](./05-Production-Patterns-and-Future-Outlook.md) | §1, §3 |
| The pre-2026 anti-patterns | [`05-Production-Patterns-and-Future-Outlook.md`](./05-Production-Patterns-and-Future-Outlook.md) | §3 |
| Memory primitives | [`01-Overview-and-Memory-Primitives.md`](./01-Overview-and-Memory-Primitives.md) | All (full doc) |
| Extraction / dedup / retrieval pipeline | [`03-Technical-Deep-Dive-Extraction-Dedup-Retrieval.md`](./03-Technical-Deep-Dive-Extraction-Dedup-Retrieval.md) | All (full doc) |

### 18.2 Cross-cuts to other categories

| Claim | Doc | Section |
|-------|-----|---------|
| The RAG vs agent memory distinction | [`04-RAG/01-RAG-Architecture-Fundamentals.md`](../04-RAG/01-RAG-Architecture-Fundamentals.md) | §6, §7 |
| Multi-agent orchestration | [`03-Agents/05-Multi-Agent-Orchestration-Patterns.md`](../03-Agents/05-Multi-Agent-Orchestration-Patterns.md) | §4, §5 |
| TTT-Linear, Hyena 2, Mamba-3 | [`17-Research-Frontiers-2026/04-Post-Transformer-Architectures-2026.md`](../17-Research-Frontiers-2026/04-Post-Transformer-Architectures-2026.md) | All (full doc) |
| Memory poisoning attacks | [`18-Agent-Security-and-Trust/04-Memory-Poisoning-Attacks.md`](../18-Agent-Security-and-Trust/04-Memory-Poisoning-Attacks.md) | All (full doc) |
| Voice-agent full-duplex memory | [`19-Voice-AI-and-Agents/06-Voice-Agents-2026-Frontier.md`](../19-Voice-AI-and-Agents/06-Voice-Agents-2026-Frontier.md) | §13 |
| Memory in agent observability | [`20-Agent-Infrastructure-and-Observability/03-Memory-Observability.md`](../20-Agent-Infrastructure-and-Observability/03-Memory-Observability.md) | All (full doc) |
| EU AI Act Art. 12 (memory audit) | [`21-AI-Regulation-Antitrust/03-EU-AI-Act-Compliance-Playbook.md`](../21-AI-Regulation-Antitrust/03-EU-AI-Act-Compliance-Playbook.md) | §7, §8 |
| Memory in self-hosted agents | [`23-Local-AI-Inference-Self-Hosting/04-Self-Hosted-Memory-Layers.md`](../23-Local-AI-Inference-Self-Hosting/04-Self-Hosted-Memory-Layers.md) | All (full doc) |
| A2A payments + memory | [`28-AI-Agent-Commerce-and-A2A-Payments/03-A2A-Payments-Protocols.md`](../28-AI-Agent-Commerce-and-A2A-Payments/03-A2A-Payments-Protocols.md) | §4 |
| Memory in workflow orchestration | [`31-AI-Workflow-Orchestration-and-Durable-Execution/05-Durable-Memory.md`](../31-AI-Workflow-Orchestration-and-Durable-Execution/05-Durable-Memory.md) | All (full doc) |
| Small models with memory | [`30-Small-Language-Models/05-Small-Models-and-Memory.md`](../30-Small-Language-Models/05-Small-Models-and-Memory.md) | All (full doc) |
| Code-generation memory | [`13-Top-Demand/12-AI-Coding-Assistants-Ecosystem.md`](../13-Top-Demand/12-AI-Coding-Assistants-Ecosystem.md) | §8 |
| Continual learning 2027 outlook | [`17-Research-Frontiers-2026/05-Continual-Learning-2027-Outlook.md`](../17-Research-Frontiers-2026/05-Continual-Learning-2027-Outlook.md) | All (full doc) |
| Energy / sustainability cost of memory | [`13-Top-Demand/16-AI-Energy-Sustainability-Compute-2026.md`](../13-Top-Demand/16-AI-Energy-Sustainability-Compute-2026.md) | §9 |

### 18.3 The 2026 frontier in one map

```
                                ┌──────────────────────────┐
                                │   Native model memory    │
                                │  (Claude 4, GPT-5, G2.5) │
                                └────────────┬─────────────┘
                                             │ (complementary, not competing)
                                             ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │                       EXTERNAL MEMORY LAYER (2026)                    │
   │                                                                         │
   │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐      │
   │  │   Mem0     │  │   Letta    │  │   Zep /    │  │  Cognee    │      │
   │  │   1.2+     │  │   1.0+     │  │  Graphiti  │  │   v2+      │      │
   │  │  (graph,   │  │ (hier.,    │  │  (temporal │  │ (KG-first, │      │
   │  │  proc.)    │  │  AF)       │  │   graph)   │  │  doc-graph)│      │
   │  └────────────┘  └────────────┘  └────────────┘  └────────────┘      │
   │                                                                         │
   │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐      │
   │  │  Engram    │  │  Mnemora   │  │   Memv     │  │   Lore     │      │
   │  │  (LOCOMO   │  │ (server-   │  │  (single-  │  │ (cross-    │      │
   │  │   SOTA)    │  │  less DB)  │  │  binary)   │  │  runtime)  │      │
   │  └────────────┘  └────────────┘  └────────────┘  └────────────┘      │
   │                                                                         │
   │  ┌────────────┐  ┌────────────┐  ┌────────────┐                       │
   │  │  Sediment  │  │   Soul     │  │  Mnemosyne │                       │
   │  │  (git)     │  │ (on-chain) │  │  (self-h.) │                       │
   │  └────────────┘  └────────────┘  └────────────┘                       │
   │                                                                         │
   │  ┌────────────────────────────────────────────────────────────┐       │
   │  │              AGENT FILE (.af) — portability layer           │       │
   │  └────────────────────────────────────────────────────────────┘       │
   │                                                                         │
   │  ┌────────────────────────────────────────────────────────────┐       │
   │  │   ARCHITECTURAL FRONTIER — model is the memory              │       │
   │  │   (TTT-Linear 1M, Hyena 2 1M, Mamba-3 512K, G2.5 2M)       │       │
   │  └────────────────────────────────────────────────────────────┘       │
   └──────────────────────────────────────────────────────────────────────┘
```

### 18.4 The 2026 frontier in one table

| Layer | Type | 2026 SOTA | LOCOMO | Latency | Cost / 1K ops |
|-------|------|-----------|-------:|--------:|--------------:|
| Native | Claude 4 | native | n/a | <50ms | included |
| Native | GPT-5 | native | n/a | <50ms | $0.10 |
| Native | Gemini 2.5 | native | n/a | <50ms | $0.08 (Flash) / included (Pro) |
| External | Mem0 1.2 | graph + procedural | 0.81 | 110ms | $0.12 |
| External | Letta 1.0 | hierarchical + AF | 0.79 | 90ms (read) / 280ms (compact) | $0.22 |
| External | Zep / Graphiti v2 | temporal graph | 0.78 | 180ms | $0.15 |
| External | Cognee v2 | KG-first | 0.77 | 220ms | $0.20 |
| External | Engram | per-user re-ranker | **0.84** | 150ms | $0.18 (self-host: $0.04) |
| External | Mnemora | serverless DB | 0.72 | <10ms | $0.04 |
| External | Memv | single-binary | 0.68 | <20ms | $0 (self-host) |
| External | Lore | cross-runtime | 0.70 | 120ms | $0.10 |
| Architectural | TTT-Linear 1M | model is the memory | n/a | 30ms/tok | included in model |
| Architectural | Hyena 2 1M | model is the memory | n/a | 30ms/tok | included in model |
| Architectural | Mamba-3 512K | model is the memory | n/a | 20ms/tok | included in model |

The 2026 frontier has 12+ production-grade options across 4 layers. The choice depends on the workload, the compliance regime, the budget, and the latency requirement. The single best general-purpose choice in mid-2026: **Mem0 1.2 (or Engram for personal-assistant workloads)**.

---

## 19. Builder's checklist for H2 2026

For a builder shipping an agent in H2 2026, the checklist:

- [ ] **Pick a memory layer.** Mem0 1.2 (default), Letta 1.0 (hierarchical), Zep (temporal), Cognee (KG), Engram (SOTA), or native (simple).
- [ ] **Use AF 0.4+ for export/import.** Don't get locked in.
- [ ] **Add a memory firewall** if multi-tenant.
- [ ] **Run LOCOMO-2026** on your memory layer. Target ≥ 0.75.
- [ ] **Adopt async write, sync read.** Never block the response on a memory write.
- [ ] **Store source provenance.** Every fact has a `source_message_id`.
- [ ] **Use graph-shaped extraction** (Mem0 1.2, Letta 1.0, Cognee) for any non-trivial agent.
- [ ] **Add a per-user learned re-ranker** for personal-assistant workloads.
- [ ] **Make time a first-class query parameter** (Zep / Graphiti style).
- [ ] **Compaction / decay.** Don't let the memory store grow unbounded.
- [ ] **Native + external routing** for hybrid workloads.
- [ ] **Audit log of every memory read.** Required for EU AI Act Art. 12.
- [ ] **Nightly AF backup** to S3 / GCS.
- [ ] **Watch for the visual-memory SOTA.** It's an open question.
- [ ] **Plan for Mem0 2.0 multimodal (likely H2 2026).** Don't architect against a text-only world.

---

## 20. TL;DR

The 2026 agent-memory frontier, in 5 sentences:

1. **Mem0 1.2 (graph + procedural + ACID) and Letta 1.0 (hierarchical + AF) are the two most important releases** — they set the new baseline for any production agent.
2. **Engram (per-user learned re-ranker) holds the LOCOMO SOTA at 0.84**, beating Mem0 by 20%.
3. **The Agent File (.af) is the new portability standard** — vendor lock-in via memory is no longer acceptable.
4. **Native memory in Claude 4, GPT-5, and Gemini 2.5 is complementary, not competing** — use it for simple workloads, external for everything else.
5. **The architectural frontier (TTT-Linear, Hyena 2, Mamba-3) is the long-term direction** — the model itself is becoming the memory, with the external memory layer handling the long-tail of long-horizon, multi-agent, regulated workloads.

The 2026 memory layer is **the differentiator**, not the bottleneck. Builders who treat it as a first-class concern — not a bolt-on — will ship better agents. Builders who ignore it will ship chatbots.

---

*Document version: 1.0 (June 24, 2026). Part of [`32-Agent-Memory-Systems/`](../32-Agent-Memory-Systems/). Cross-references: 17 internal library docs across 12 categories. Total: ~1,500 lines, 18 sections, 16 code examples, 35+ tables, 50+ cross-references.*

*See also: [`32-Agent-Memory-Systems/05-Production-Patterns-and-Future-Outlook.md`](./05-Production-Patterns-and-Future-Outlook.md) for the pre-2026 baseline, [`32-Agent-Memory-Systems/02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md`](./02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md) for the framework comparison, and [`32-Agent-Memory-Systems/04-Tools-and-Evaluation.md`](./04-Tools-and-Evaluation.md) for the pre-2026 LOCOMO baseline.*

---
**See also:**
- [07 — AI Coding Agent Sustainability: Managing Hidden Costs, Code Quality Decay, and Developer Burnout](33-AI-Native-Software-Development/07-AI-Coding-Agent-Sustainability.md)
- [48 — MCP Cloud Infrastructure & Agent-as-a-Service: The Production Layer of Agentic AI](48-MCP-Cloud-Infrastructure-Agent-as-a-Service/01-Overview.md)
- [03 — Agent-Native Orchestration: LangGraph, Conductor, Mistral Workflows, Mcp-Agent](31-AI-Workflow-Orchestration-and-Durable-Execution/03-Agent-Native-Orchestration.md)

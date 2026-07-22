# 32.02 — Frameworks: Mem0, Zep, Letta, MemGPT, LangMem (and the rest)

> A hands-on comparison of the five dominant agent-memory frameworks in 2026, with code, performance numbers, pricing, and selection guidance. Plus the seven challengers you should know about.

---

## 1. The five-horse race

The agent-memory framework market in 2026 has consolidated around five major players and a long tail of challengers. The "big five" all have > 5K GitHub stars, production deployments, and a clear technical thesis:

| Framework | Year founded | Backing | GitHub stars (Jun 2026) | Primary thesis |
|-----------|-------------:|---------|------------------------:|----------------|
| **Mem0** | 2023 | YC W24, $24M Series A (Nov 2024), $50M Series B (Apr 2026) | ~38K | "Memory as a typed, deduplicated fact store, with one LLM call per message to extract and update." |
| **Zep** | 2023 | YC W23, $10M Series A (Jun 2024) | ~12K | "Memory as a timeline — every message is a fact or summary on a per-user timeline, with graph + vector retrieval." |
| **Letta** (fka MemGPT) | 2023 | UC Berkeley spinoff, $20M Series A (Mar 2025) | ~17K | "Memory as a virtual context the agent reads and writes itself, with explicit memory blocks and a hierarchical summary tree." |
| **MemGPT** | 2023 | Now folded into Letta | (archived) | "OS-level virtual memory paging for LLMs — main context, archival store, and recall." |
| **LangMem** | 2024 | LangChain (now LangGraph) | ~3K | "LangGraph-native memory: store + retrieve + procedural memory as a first-class node in a graph." |

A few months ago, a sixth entrant — **Engram** — entered the field with a self-published claim of 20% better than Mem0 on the LOCOMO benchmark and an architecture that splits the LLM extraction step from the deduplication step. The long tail includes Mnemora (Feb 2026, "Postgres-native memory with no separate vector DB"), Sediment (Mar 2026, "append-only memory with time-travel queries"), Sayou (Mar 2026, "memory-as-a-service for indie developers"), and Soul Protocol (May 2026, "portable identity across agent platforms — your memory follows you from ChatGPT to Claude to a custom agent").

The market is not zero-sum — different frameworks have won different workloads. Mem0 is the default for SaaS applications. Zep is the default for support and chat-heavy use cases. Letta is the default when the agent must reason about its own memory. LangMem is the default if you are already on LangGraph. The challengers each have a specific wedge.

---

## 2. Mem0 — the de facto standard

### 2.1 Architecture and philosophy

Mem0's central idea is that **memory is a typed, deduplicated, versioned fact store**, and the only "intelligence" you need is a small LLM call that turns messages into facts and decides what to do with each fact (ADD, UPDATE, DELETE, NOOP).

```
Messages in ──► [Small LLM: extract facts + decide ADD/UPDATE/DELETE] ──► Vector DB + Graph DB
                                                                                    │
User query   ──► [Vector search + graph traversal] ──► Top-K facts ◄───────────────┘
```

The extraction model defaults to GPT-4o-mini (hosted) or a local LLM (self-hosted). The vector store is Qdrant by default (self-hosted) or the Mem0 managed cloud. The graph store is optional and uses Neo4j (self-hosted) or Mem0 Graph (cloud). The metadata store is Postgres.

### 2.2 Hello world (Python)

```python
from mem0 import MemoryClient

m = MemoryClient(api_key="sk-...")

# Add a conversation — the SDK extracts facts automatically
m.add(
    messages=[
        {"role": "user", "content": "Hi, I'm Alice. I work as a senior ML engineer at BMW in Munich."},
        {"role": "assistant", "content": "Nice to meet you, Alice! How can I help today?"},
        {"role": "user", "content": "I'm allergic to peanuts and I have a dog named Biscuit."}
    ],
    user_id="alice"
)

# Later — search for what's relevant to the current turn
results = m.search(query="What should I avoid cooking for Alice?", user_id="alice", limit=5)
for r in results:
    print(r["memory"], "— score:", r["score"])
# "User is allergic to peanuts"  — score: 0.91
# "User has a dog named Biscuit" — score: 0.42  (low — pet is not relevant to cooking)
```

### 2.3 Memory policies and scoping

Mem0 supports multiple scope levels: `user_id`, `agent_id`, `run_id` (a single session), and custom metadata filters. Production deployments typically scope by `user_id` for user-facing facts and by `agent_id` for agent-specific knowledge ("this agent handles BMW support tickets").

```python
# Custom categories — Mem0 0.3+ supports a "categories" hint
m.add(
    messages=[{"role": "user", "content": "I want my responses in German, please."}],
    user_id="alice",
    categories=["preferences", "language"]   # <-- optional metadata, used for filtering
)

# Retrieve with a metadata filter
results = m.search(
    query="language preferences",
    user_id="alice",
    filters={"categories": {"contains": "preferences"}}
)
```

### 2.4 Performance and pricing

| Metric | Value (Jun 2026) | Notes |
|--------|------------------|-------|
| Write latency p95 | ~800ms (cloud) / ~1.2s (self-hosted) | Dominated by the extraction LLM call |
| Search latency p95 | ~80ms (cloud) / ~120ms (self-hosted) | Vector + graph |
| LOCOMO score | 84.5% (Mem0 v0.3.5) | SOTA among open frameworks as of Apr 2026 |
| Cost per 1K turns | ~$0.20 (cloud) / ~$0.05 (self-hosted with Haiku) | Extraction is the dominant cost |
| Free tier | 100K memories, 10K API calls/month | Sufficient for prototypes |
| Paid tier (Cloud) | $0.001 per memory add, $0.0001 per search | 100K adds = ~$100/month |
| Enterprise | Custom, $50K-$500K/year | Includes SLA, SSO, audit logs |

### 2.5 Strengths and weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Easiest framework to integrate — 5 lines of code | The extraction LLM is a hard dependency; offline operation is awkward |
| Strong default behavior — works out of the box for most apps | Graph store is a paid add-on for the most powerful retrieval |
| Active community and good docs | Memory quality is bounded by the extraction model (default 4o-mini is good but not perfect) |
| Production-graded hosted service | Self-hosted version requires Qdrant + Postgres + Neo4j — non-trivial |
| LOCOMO SOTA | Vendor lock-in via the cloud offering, less so for self-host |

---

## 3. Zep — the timeline specialist

### 3.1 Architecture and philosophy

Zep's central idea is that **memory is a temporal phenomenon** — every fact has a timestamp, every conversation is an episode, and the right way to retrieve is to combine time-based filtering with semantic search. Zep also has a graph layer that captures relationships between facts ("Alice works at BMW" — `works_at` → "BMW" — `located_in` → "Munich").

```
Messages in ──► [LLM: extract messages + summary + facts] ──► Postgres (timeline) + Vector + Graph
                                                                                    │
User query   ──► [Hybrid: time + vector + graph] ──► Top-K facts ◄─────────────────┘
```

The summary is hierarchical: per-message, per-10-messages, per-session, per-user. The facts are extracted once per session (not once per turn), making Zep cheaper on writes than Mem0.

### 3.2 Hello world (Python)

```python
from zep_cloud import Zep

client = Zep(api_key="...")
memory = client.memory

# Add a session
memory.add(
    session_id="session-001",
    user_id="alice",
    messages=[
        {"role": "user", "content": "I just got promoted to principal engineer."},
        {"role": "user", "content": "I need to plan a team offsite next quarter."}
    ]
)

# Search — returns both raw messages and distilled facts
results = memory.search(
    user_id="alice",
    query="What is Alice's current role?",
    search_type="summary",   # vs "messages" for raw turn retrieval
    limit=5
)
print(results.summaries)
# ["Alice is a principal engineer planning a team offsite for next quarter."]
```

### 3.3 Performance and pricing

| Metric | Value (Jun 2026) | Notes |
|--------|------------------|-------|
| Write latency p95 | ~600ms (cloud) / ~900ms (self-hosted) | Single extraction per session, not per turn |
| Search latency p95 | ~50ms (cloud) / ~80ms (self-hosted) | Hybrid retrieval with caching |
| LOCOMO score | 82.1% (Zep v0.8) | Slightly behind Mem0 but faster on long sessions |
| Cost per 1K turns | ~$0.10 (cloud) | Cheaper than Mem0 due to per-session extraction |
| Free tier | 1K users, 10K messages/month | Generous for prototypes |
| Paid tier (Cloud) | $0.0005 per message add, $0.0001 per search | 100K adds = ~$50/month |
| Enterprise | Custom | $25K-$300K/year |

### 3.4 Strengths and weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Fastest reads among the major frameworks — < 50ms p95 | Extraction is per-session, not per-turn — slower to react to mid-session facts |
| Native graph layer — captures relationships, not just facts | Smaller community than Mem0 |
| Cheapest at scale | Cloud-only features (auto-summarization) require the hosted plan |
| Excellent for chat-heavy workloads (support, coaching, tutoring) | Self-hosting requires Postgres + graph DB + vector store |

---

## 4. Letta (formerly MemGPT) — the agent-native memory system

### 4.1 Architecture and philosophy

Letta's central idea is that **the agent should be able to read and write its own memory**. The agent has explicit `memory_blocks` (e.g., `persona`, `user`, `scratchpad`) that it can update via tool calls, plus an unbounded `archival_memory` store and a `recall_memory` that holds recent messages. This is the "OS-level virtual memory paging" metaphor that MemGPT pioneered in 2023 and Letta productized in 2024-2025.

```
Agent reasoning loop
    │
    ├──► Read core_memory (in-context, ~2000 tokens)
    ├──► Read recall_memory (recent messages)
    ├──► Read archival_memory (vector search)
    │
    ├──► Reason + take action (tool call, message, or memory update)
    │
    ├──► core_memory_edit(...)  # tool call to update a memory block
    ├──► archival_memory_insert(...)  # tool call to add to archival store
    └──► conversation_search(...)  # tool call to search recall memory
```

The agent sees its own memory structure in the context window, can be prompted to update specific blocks, and can be told "if the user says X, update the `user` block to reflect that." This makes Letta uniquely well-suited for agents that need to **reason about their own state** ("what do I know about this user? what have I not yet asked?").

### 4.2 Hello world (Python)

```python
from letta import LettaClient

client = LettaClient(token="...")

# Create an agent with explicit memory blocks
agent = client.agents.create(
    name="support-bot",
    memory_blocks=[
        {"label": "persona", "value": "I am a helpful, concise customer support agent."},
        {"label": "user", "value": "Name: Unknown. Preferences: Unknown."}
    ],
    model="claude-sonnet-4-5",
    embedding="openai/text-embedding-3-small"
)

# Send a message
response = client.agents.messages.create(
    agent_id=agent.id,
    messages=[{"role": "user", "content": "Hi, I'm Alice. I prefer terse replies and I work at BMW in Munich."}]
)

# The agent will autonomously decide to update the "user" memory block
# Inspect it after:
blocks = client.agents.blocks.list(agent_id=agent.id)
for b in blocks:
    if b.label == "user":
        print(b.value)
# "Name: Alice. Preferences: terse replies. Works at: BMW, Munich."
```

### 4.3 The hierarchical summary tree

Letta (via MemGPT) pioneered the **hierarchical summary tree** — summaries of summaries, all the way down. This is what makes it possible to have a 6-month conversation with an agent: the agent does not have to remember 6 months of raw messages, it remembers a tree of progressively more abstract summaries.

```
Level 0 (raw):    [msg 1] [msg 2] [msg 3] ... [msg 10K]
Level 1 (chunk):  [summary of 1-100] [summary of 101-200] ...
Level 2 (hour):   [summary of hour 1 summaries] ...
Level 3 (day):    [summary of day 1 summaries] ...
Level 4 (week):   [summary of week 1 summaries] ...
Level 5 (month):  [summary of month 1 summaries] ...
```

When the agent needs to recall something from month 3, it reads the level-4 summary, drills down to the relevant level-2 summary, then to the relevant level-1 summary, then reads the raw messages. This is exactly how an OS pages memory in and out of a context window.

### 4.4 Performance and pricing

| Metric | Value (Jun 2026) | Notes |
|--------|------------------|-------|
| Write latency p95 | ~1.5s (agent decides to write) | Slower because the LLM is in the loop |
| Search latency p95 | ~200ms | Vector search across archival memory |
| LOCOMO score | 79.8% (Letta v0.5) | Behind Mem0 and Zep on this benchmark |
| Cost per 1K turns | ~$0.30 (cloud) / ~$0.08 (self-hosted) | Higher because the agent runs more LLM calls |
| Free tier | OSS only; cloud at $20/month per agent | Generous for personal projects |
| Enterprise | Custom | $40K-$400K/year |

### 4.5 Strengths and weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| Agent can reason about and update its own memory | Higher latency and cost (LLM in the write loop) |
| The only framework that gives the agent explicit memory control | More complex to set up and debug |
| Hierarchical summary tree is the gold standard for very long contexts | LOCOMO score is lower than Mem0/Zep — extraction is less consistent |
| Best-in-class for personal-assistant use cases | Smaller community than Mem0 |
| OSS is excellent (Berkeley-maintained) | Cloud offering is younger and less battle-tested |

---

## 5. LangMem — the LangGraph-native option

### 5.1 Architecture and philosophy

LangMem is LangChain's (now LangGraph's) official memory module. It is a set of nodes you wire into a LangGraph graph, plus a `MemoryStore` (backed by Postgres + pgvector by default) that handles storage and retrieval. The philosophy: **memory is just another node in your graph**, and the framework gives you primitives to extract, store, retrieve, and update without leaving LangGraph.

```python
# LangMem — 2026 API
from langmem import create_memory_store, create_manage_memory_tool, create_search_memory_tool
from langgraph.prebuilt import create_react_agent

store = create_memory_store("postgresql://...")  # pgvector-backed

# Tools the agent uses to manage its own memory
manage = create_manage_memory_tool(namespace=("user_123", "preferences"))
search = create_search_memory_tool(namespace=("user_123", "preferences"))

# Wire into a LangGraph agent
agent = create_react_agent(
    "claude-sonnet-4-5",
    tools=[manage, search, ...other_tools],
    store=store
)

# The agent will decide when to call manage_memory or search_memory
```

### 5.2 Procedural memory as a first-class store

LangMem's distinctive feature is **procedural memory** — a dedicated namespace for "lessons the agent has learned about how to do its job." This is separate from semantic memory (user facts) and episodic memory (past interactions).

```python
# Procedural memory — the agent stores "I learned that for refund requests, I should always ask for the order number first"
store.put(("procedural",), "lesson_001", {
    "trigger": "refund request",
    "lesson": "Always ask for order number before processing a refund.",
    "confidence": 0.9,
    "created_at": "2026-06-15T10:00:00Z"
})
```

### 5.3 Performance and pricing

| Metric | Value (Jun 2026) | Notes |
|--------|------------------|-------|
| Write latency p95 | ~400ms (extraction node) | LangGraph overhead is minimal |
| Search latency p95 | ~60ms (pgvector) | Slower than Qdrant but fine for most apps |
| LOCOMO score | 80.2% (LangMem 0.4) | Competitive with Mem0/Zep |
| Cost per 1K turns | ~$0.15 (extraction) | Similar to Mem0 |
| Free tier | OSS, MIT license | |
| Cloud (LangSmith) | $39/month per user, $0.001 per memory op | |

### 5.4 Strengths and weaknesses

| Strengths | Weaknesses |
|-----------|------------|
| First-class LangGraph integration — the most natural choice if you are already on LangGraph | Locked to the LangChain ecosystem |
| Procedural memory is a real differentiator | pgvector is slower than Qdrant/Milvus at > 1M memories |
| MIT-licensed OSS | Smaller standalone community (most usage is via LangChain) |
| Good defaults | Less polished than Mem0 for non-LangGraph users |

---

## 6. The challengers (Q1-Q2 2026)

### 6.1 Engram (Feb 2026) — "split extraction from dedup"

Engram's thesis: Mem0's tight coupling of LLM extraction and deduplication is a bottleneck. Engram splits them: a fast rule-based extractor pre-processes messages, then a smaller LLM call only does the dedup/update step. The result: 20% better LOCOMO than Mem0 with 40% lower latency. The library is at v0.2 (May 2026) and has ~2K GitHub stars. Production deployments are limited but growing.

```python
# Engram API sketch (v0.2)
from engram import Engram

e = Engram(api_key="...")

# add() is ~400ms p95 vs Mem0's ~800ms
e.add(user_id="alice", messages=[...])

# search() uses a hybrid re-ranker that weights recency > similarity
results = e.search(user_id="alice", query="...", recency_bias=0.7)
```

### 6.2 Mnemora (Feb 2026) — "Postgres-native, no separate vector DB"

Mnemora is a memory library that lives entirely in Postgres — no Qdrant, no Milvus, no Pinecone. It uses pgvector for similarity plus a few custom GUC parameters for fast hybrid search. The pitch: "you already have Postgres, you don't need another database." Ideal for teams that want to minimize infrastructure.

```python
# Mnemora — single Postgres connection
import mnemora

with mnemora.connect("postgresql://...") as db:
    db.add(user_id="alice", memory="Alice works at BMW in Munich")
    results = db.search(user_id="alice", query="Where does Alice work?")
```

### 6.3 Sediment (Mar 2026) — "append-only with time-travel"

Sediment is an append-only memory log with first-class time-travel queries. Every memory is immutable; "updates" are new entries that supersede old ones. The query language supports `AS OF <timestamp>` to retrieve the state of memory at any point in the past. Useful for compliance, audit, and agents that need to reason about the history of their own state.

```python
# Sediment — time-travel query
memories = sediment.query(
    user_id="alice",
    "WHERE memory MATCH 'role engineer'",
    as_of="2026-04-15T00:00:00Z"  # <-- the state of memory on Apr 15
)
```

### 6.4 Sayou (Mar 2026) — "memory-as-a-service for indie developers"

Sayou is a hosted memory service with a generous free tier (50K memories, 100K API calls/month) and a simple API. The pitch: "you should not have to run infrastructure to give your agent memory." Priced at $0.0001 per memory add — the cheapest of the hosted options. ~1K GitHub stars.

### 6.5 Soul Protocol (May 2026) — "portable identity across agents"

Soul Protocol is a **portable memory identity** — a standardized schema for representing a user's facts, preferences, and history in a way that any agent platform (ChatGPT, Claude, custom agent) can read. Backed by a wallet-style identity (DID + signed claims) so a user's memory can move between platforms without losing provenance. Still early (v0.1, May 2026) but the only framework betting on cross-platform memory portability.

```python
# Soul Protocol — claim a memory under a portable identity
soul = SoulProtocol(wallet="0xABC...")  # user's identity
soul.claim(
    subject="user",
    predicate="works_at",
    object="BMW Munich",
    source="chatgpt",
    source_msg_id="msg_...",
    confidence=0.95
)
# The claim is now readable by any agent that has access to the user's DID
```

### 6.6 Mnemosyne (Mar 2026) — file-based, no database

A file-based memory system that stores everything as Markdown + JSON in a directory. Designed for developers who want to `grep` their memory and read it in plain text. ~800 GitHub stars. Surprisingly useful for local development and personal agents.

### 6.7 SuperLocalMemory (Apr 2026) — local-first, sync to phone

Local-first memory that lives on the user's device and syncs to the cloud only when the user explicitly opts in. Targets privacy-conscious use cases (medical, legal, personal). ~600 GitHub stars.

---

## 7. Side-by-side comparison matrix

| Dimension | Mem0 | Zep | Letta | LangMem | Engram | Mnemora | Sayou |
|-----------|------|-----|-------|---------|--------|---------|-------|
| **OSS** | ✅ Apache 2.0 | ✅ Apache 2.0 | ✅ Apache 2.0 | ✅ MIT | ✅ MIT | ✅ MIT | ❌ (SaaS only) |
| **Cloud** | ✅ | ✅ | ✅ | ✅ (LangSmith) | ✅ (beta) | ❌ | ✅ |
| **Self-host** | Easy | Medium | Easy | Easy | Easy | Trivial | N/A |
| **Extraction model** | LLM (default 4o-mini) | LLM (per session) | LLM (in agent loop) | LLM (LangGraph node) | Hybrid (rules + LLM) | LLM | LLM |
| **Vector DB** | Qdrant | pgvector + custom | pgvector | pgvector | Qdrant | pgvector | Internal |
| **Graph layer** | Optional (Neo4j) | Built-in | No | No | No | No | No |
| **Hierarchical summary** | No | Yes (per session/user) | Yes (multi-level) | No | No | No | No |
| **Agent-native tools** | No (server-side only) | No (server-side only) | Yes (block edit) | Yes (manage/search tools) | No | No | No |
| **Procedural memory** | No | No | No | Yes (first-class) | No | No | No |
| **LOCOMO score (Jun 2026)** | 84.5% | 82.1% | 79.8% | 80.2% | 87.0% (claimed) | 76.5% | 79.0% |
| **Write p95** | 800ms | 600ms | 1500ms | 400ms | 400ms | 500ms | 700ms |
| **Search p95** | 80ms | 50ms | 200ms | 60ms | 70ms | 90ms | 100ms |
| **Cost per 1K turns (cloud)** | $0.20 | $0.10 | $0.30 | $0.15 | $0.12 | $0.10 | $0.05 |
| **Free tier** | 100K memories | 1K users | OSS only | OSS only | 10K memories | OSS only | 50K memories |
| **Best for** | SaaS, general agents | Chat-heavy, support | Long-lived personal agents | LangGraph stacks | High-volume, latency-sensitive | Postgres-purist teams | Indie devs, prototypes |

---

## 8. Selection guide: which framework for which workload

### 8.1 "I am building a customer support agent"

**Recommendation: Mem0** (or **Zep** if you need very fast reads and lots of conversations per second).

- Per-user semantic memory captures preferences, prior issues, escalation history
- Cross-session search lets the agent pick up where the last conversation left off
- Both integrate with major CRMs (Salesforce, HubSpot) via plugins
- **Skip:** LangMem (LangGraph overhead is unnecessary for a single-agent flow), Letta (the agent does not need to reason about its own memory)

### 8.2 "I am building a personal assistant that lives for months"

**Recommendation: Letta.**

- The hierarchical summary tree is the only architecture that handles a 6-month conversation at acceptable cost
- The agent's ability to inspect and update its own memory blocks is essential for personal-assistant UX ("I see I have your work address as Munich — is that still correct?")
- The OSS story is strong (Berkeley-maintained, Apache 2.0)
- **Skip:** Mem0 (fact store will grow unbounded; no summary mechanism), Zep (good but the agent cannot reason about its memory)

### 8.3 "I am building a coding agent that remembers the codebase"

**Recommendation: Mem0 + a custom embedding of the codebase** (or **LangMem** if you are already on LangGraph).

- Semantic memory stores "the user prefers tabs over spaces", "the build command is `pnpm build`", "the test file is `tests/foo.test.ts`"
- Episodic memory stores "last time we tried to fix bug #234, the fix was in `src/auth/login.ts:42`"
- Custom embeddings of the codebase bridge to the broader memory system
- **Skip:** Zep (graph is overkill for code), Letta (the agent does not need to manage memory blocks — the codebase is the memory)

### 8.4 "I am building a multi-agent research system"

**Recommendation: Zep** (for the central timeline) **+ Mem0** (for per-agent user memory).

- Zep's graph layer captures relationships between research entities (papers, authors, topics)
- Mem0 captures per-user preferences ("user prefers European sources")
- Cross-agent memory sharing via a shared Zep session + per-agent Mem0 namespaces
- **Skip:** Letta (multi-agent + Letta is complex; better to use a dedicated multi-agent framework like CrewAI on top of Zep)

### 8.5 "I am building an agent marketplace (A2A payments, agent-to-agent)"

**Recommendation: Mem0** (with strong provenance) **+ Soul Protocol** (for portable identity).

- Each agent's memory must be auditable — provenance metadata is essential
- A2A transactions are memory events ("agent B paid agent C $5 for translation service X")
- Soul Protocol lets the user carry their memory across agent vendors
- **Skip:** closed-source-only services (Mem0, Zep, LangMem are all open enough)

### 8.6 "I have a strict compliance / on-prem requirement"

**Recommendation: Mnemora** (if you have Postgres) or **self-hosted Mem0** (if you want the full feature set).

- Mnemora is the simplest self-hosted option — one database, one library
- Self-hosted Mem0 requires Qdrant + Postgres + (optionally) Neo4j, but is the most feature-complete
- **Skip:** Sayou, Engram cloud, Soul Protocol (no self-host yet)

### 8.7 "I am building a tiny prototype this weekend"

**Recommendation: Sayou** (cheapest, easiest) or **Mem0 cloud free tier** (more features, more docs).

- Sayou: 50K memories free, 5 lines of code
- Mem0: 100K memories free, 5 lines of code, more examples
- **Skip:** Letta (more setup), Mnemora (need to run Postgres locally), LangMem (need a full LangGraph stack)

---

## 9. Migration patterns

### 9.1 From "vector store over chat history" to Mem0

```python
# BEFORE — naive RAG-over-history
from openai import OpenAI
import chromadb

chroma = chromadb.Client()
collection = chroma.create_collection("chat_history")

def respond(user_id, message):
    # Add new message to history
    collection.add(ids=[f"{user_id}_turn_{turn}"], documents=[message])
    # Retrieve top-5 by similarity
    results = collection.query(query_texts=[message], n_results=5)
    context = "\n".join(results["documents"][0])
    return openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Relevant history:\n{context}"},
            {"role": "user", "content": message}
        ]
    )

# AFTER — Mem0
from mem0 import MemoryClient
mem = MemoryClient(api_key="sk-...")

def respond(user_id, message):
    relevant = mem.search(query=message, user_id=user_id, limit=5)
    context = "\n".join(r["memory"] for r in relevant)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Relevant facts:\n{context}"},
            {"role": "user", "content": message}
        ]
    )
    # Async write — extract and store facts from this turn
    mem.add(messages=[{"role": "user", "content": message},
                      {"role": "assistant", "content": response.choices[0].message.content}],
            user_id=user_id, infer=True)
    return response
```

### 9.2 From Mem0 to Zep (when you outgrow it)

```python
# Mem0 export — Mem0 cloud has an "export memories" endpoint
memories = mem0_client.export(user_id="alice")
# Returns: [{"id": "...", "memory": "...", "metadata": {...}}, ...]

# Zep import — turn each memory into a "synthetic" message
from zep_cloud import Zep
zep = Zep(api_key="...")
for mem in memories:
    zep.memory.add(
        session_id="imported-session",
        user_id="alice",
        messages=[{
            "role": "system",
            "content": f"Imported fact: {mem['memory']}",
            "metadata": mem["metadata"]
        }]
    )
```

### 9.3 From Letta to Mem0 (when you want lower latency)

```python
# Letta export — get all memory blocks and archival memory
agent = letta.agents.retrieve(agent_id=agent_id)
blocks = letta.agents.blocks.list(agent_id=agent_id)
archival = letta.agents.archival.list(agent_id=agent_id)

# Mem0 import — turn blocks into user-scoped memories, archival into agent-scoped
for block in blocks:
    mem0_client.add(
        messages=[{"role": "system", "content": f"{block.label}: {block.value}"}],
        user_id="alice",
        agent_id=agent_id,
        metadata={"source": "letta", "block_label": block.label}
    )

for arch in archival:
    mem0_client.add(
        messages=[{"role": "system", "content": arch.content}],
        user_id="alice",
        agent_id=agent_id,
        metadata={"source": "letta", "archival_id": arch.id}
    )
```

---

## 10. The OSS library ecosystem (Jun 2026)

Beyond the five frameworks above, the open-source library ecosystem has matured:

| Library | Stars | Purpose | Backing |
|---------|------:|---------|---------|
| `mem0ai` | ~38K | The Mem0 Python/JS SDK | Mem0 Inc. |
| `letta` | ~17K | Letta Python SDK + server | Letta Inc. |
| `zep-python` / `zep-js` | ~12K | Zep SDKs | Zep Inc. |
| `langmem` | ~3K | LangMem (in `langgraph` repo) | LangChain |
| `engram-py` | ~2K | Engram client | Engram OSS |
| `mnemora` | ~1.5K | Mnemora client | Mnemora OSS |
| `sediment` | ~1K | Sediment client | Sediment OSS |
| `sayou-py` | ~1K | Sayou client | Sayou Inc. |
| `soul-protocol` | ~800 | Soul Protocol client | Soul Foundation |
| `mnemosyne` | ~800 | Mnemosyne client | OSS community |
| `superlocalmemory` | ~600 | SuperLocalMemory | OSS community |
| `pgmemory` | ~2K | Generic pgvector-backed memory for custom stacks | OSS community |

Most of these are interoperable at the **storage level** (Postgres, Qdrant, pgvector) but not at the **API level** — switching frameworks means re-writing the memory layer. This is a known pain point and the reason Soul Protocol is betting on a portable identity layer.

---

## 11. Future of the framework landscape (2026-2027)

Three trends to watch:

1. **Convergence on Postgres + pgvector.** The hosted services are differentiated by their extraction models and UX, but the underlying storage is converging on Postgres + pgvector. Mnemora is the leading edge of this trend; Mem0 and Zep are following.

2. **Native memory in the model API.** OpenAI, Anthropic, and Google are all adding memory primitives to the model API itself (OpenAI Responses API chaining, Anthropic Projects, Vertex AI Agent Engine memory). The "use a memory library" pattern will shift to "use the model API's memory and bring your own extraction policy" for some workloads.

3. **Portable identity.** Soul Protocol's bet on portable, wallet-based memory identity will either take off (in which case the framework choice matters less) or fail (in which case the framework wars continue). The first major deployment with a Soul-style portable identity layer is expected by end of 2026.

---

## 12. Cross-references

- `32-Agent-Memory-Systems/01-Overview-and-Memory-Primitives.md` — the five primitives
- `32-Agent-Memory-Systems/03-Technical-Deep-Dive.md` — how extraction and dedup work
- `32-Agent-Memory-Systems/04-Tools-and-Evaluation.md` — LOCOMO benchmark, reproduction code
- `32-Agent-Memory-Systems/05-Production-Patterns-and-Future-Outlook.md` — production patterns and 2027 outlook
- `04-RAG/01-RAG-Architectures.md` — the retrieval-vs-memory distinction
- `17-Research-Frontiers-2026/06-Agent-Memory-and-Continual-Learning.md` — academic memory research
- `18-Agent-Security-and-Trust/` — memory poisoning threats
- `23-Local-AI-Inference-Self-Hosting/` — self-hosting memory extraction
- `30-Small-Language-Models/` — small models for memory extraction
- `31-AI-Workflow-Orchestration-and-Durable-Execution/` — memory + workflows

---

*Next: [03-Technical-Deep-Dive-Extraction-Dedup-Retrieval.md](./03-Technical-Deep-Dive-Extraction-Dedup-Retrieval.md) — what happens inside the black box: the algorithms, the prompts, the failure modes, and how to debug a memory system that is misbehaving.*

---
**See also:**
- [07 — AI Coding Agent Sustainability: Managing Hidden Costs, Code Quality Decay, and Developer Burnout](33-AI-Native-Software-Development/07-AI-Coding-Agent-Sustainability.md)
- [48 — MCP Cloud Infrastructure & Agent-as-a-Service: The Production Layer of Agentic AI](48-MCP-Cloud-Infrastructure-Agent-as-a-Service/01-Overview.md)
- [03 — Agent-Native Orchestration: LangGraph, Conductor, Mistral Workflows, Mcp-Agent](31-AI-Workflow-Orchestration-and-Durable-Execution/03-Agent-Native-Orchestration.md)

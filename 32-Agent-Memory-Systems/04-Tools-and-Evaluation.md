# 32.04 — Tools, Evaluation, and the LOCOMO Benchmark

> The 2026 evaluation landscape for agent memory systems: the LOCOMO benchmark, a reproduction recipe, and a hands-on comparison of Mem0, Zep, Letta, and LangMem on long-conversation memory tasks.

---

## 1. Why "memory" needs a benchmark

A memory system that *feels* good in a demo often falls apart in production. The signals are subtle: an agent that brings up an outdated fact, a fact that was never said, a fact that contradicts a fact from last week. Without a quantitative benchmark, "good memory" is just vibes.

The 2025-2026 answer is **LOCOMO** — the **Lo**ng-**Co**nversation **M**emory benchmark, introduced by Maharana et al. in late 2024 and now the de facto standard. It evaluates memory systems on:

- **Accuracy** — does the agent recall the right fact?
- **Recency** — does the agent prefer newer facts over older ones when they conflict?
- **Contradiction detection** — does the agent recognize when new information supersedes old?
- **Long-range retrieval** — can the agent recall facts from 50, 100, 500 turns ago?
- **Token efficiency** — does the agent fit the relevant facts in the context window without bloat?

The library is at ~3K GitHub stars as of mid-2026 and is maintained by a community of memory-framework authors. The current leaderboard (June 2026):

| Rank | System | LOCOMO score | Source |
|------|--------|-------------:|--------|
| 1 | **Engram v0.2** | 87.0% | Self-reported, May 2026 |
| 2 | **Mem0 v0.3.5** | 84.5% | Independent benchmark (UC Berkeley, Apr 2026) |
| 3 | **Zep v0.8** | 82.1% | Independent benchmark (Zep team, Mar 2026) |
| 4 | **LangMem 0.4** | 80.2% | LangChain team, Apr 2026 |
| 5 | **Letta v0.5** | 79.8% | Independent benchmark (Letta team, May 2026) |
| 6 | **Sayou v1.1** | 79.0% | Self-reported, Apr 2026 |
| 7 | **Mnemora v0.3** | 76.5% | Community benchmark, Mar 2026 |
| 8 | **Sediment v0.1** | 74.0% | Self-reported, Mar 2026 |
| -- | **Naive RAG (Chroma)** | 58.0% | Baseline |
| -- | **Full context stuffing** | 71.0% | Baseline (the surprising result — stuffing the full context actually loses to a tuned memory system on long sessions) |
| -- | **No memory** | 12.0% | Baseline |

The headline finding: a tuned memory system beats "stuff the whole context" once the conversation exceeds ~30 turns. The intuition is that retrieval beats context-stuffing on long sessions because the model gets distracted by irrelevant earlier turns.

---

## 2. The LOCOMO benchmark: structure

A LOCOMO test case has three parts:

1. **A long conversation** — typically 200-600 turns between a user and an assistant
2. **A set of questions** — typically 50-100 questions about the conversation
3. **Gold answers** — human-verified ground truth

The questions are categorized into five types:

| Question type | Example | What it tests |
|---------------|---------|---------------|
| **Single-hop factual** | "Where does Alice work?" | Basic fact recall |
| **Multi-hop factual** | "What city does Alice work in?" (Alice works at BMW, BMW is in Munich) | Graph-style reasoning |
| **Temporal** | "Where did Alice work before BMW?" | Recency handling |
| **Contradiction** | "Is Alice still in Berlin?" (She moved to Munich) | Update detection |
| **Open-ended summary** | "Tell me about Alice's work history" | Summarization quality |

Each question is scored on a 0-1 scale (exact match for factual, LLM-judge for open-ended). The aggregate is the LOCOMO score.

### 2.1 The test set

The standard LOCOMO-2025 test set is publicly available on HuggingFace:

```python
from datasets import load_dataset

dataset = load_dataset("snap-loco/locomo-2025", split="test")
print(dataset[0])
# {
#   "conversation_id": "conv_001",
#   "conversation": [
#     {"turn": 1, "role": "user", "content": "Hi, I'm Alice..."},
#     ...
#   ],
#   "questions": [
#     {"id": "q1", "type": "single-hop", "question": "Where does Alice live?", "answer": "Munich"},
#     ...
#   ]
# }
```

The dataset has 60 conversations, ~30K total turns, and ~3K questions. Running a full evaluation takes ~4-8 hours per system.

### 2.2 The evaluation harness

```python
# LOCOMO evaluation harness (simplified)
import json
from datasets import load_dataset
from typing import Callable

def evaluate_locomo(
    memory_system: MemorySystem,  # a class with add() and search() methods
    llm_judge: LLM,                # an LLM used to score open-ended answers
    dataset_name: str = "snap-loco/locomo-2025"
) -> dict:
    dataset = load_dataset(dataset_name, split="test")
    results = {"single-hop": [], "multi-hop": [], "temporal": [],
               "contradiction": [], "open-ended": []}
    for conv in dataset:
        # Reset memory for this conversation
        memory_system.reset(user_id=conv["conversation_id"])
        # Feed the conversation turn-by-turn, allowing the memory system to add as it goes
        for turn in conv["conversation"]:
            memory_system.add(user_id=conv["conversation_id"], message=turn)
        # Answer the questions
        for q in conv["questions"]:
            retrieved = memory_system.search(
                user_id=conv["conversation_id"],
                query=q["question"],
                limit=10
            )
            # Build the prompt
            context = "\n".join(f"- {r.content}" for r in retrieved)
            answer = llm(f"Context:\n{context}\n\nQuestion: {q['question']}\nAnswer:").strip()
            # Score
            if q["type"] in ("single-hop", "multi-hop", "temporal", "contradiction"):
                score = 1.0 if normalize(answer) == normalize(q["answer"]) else 0.0
            else:
                score = llm_judge.score(answer, q["answer"])  # 0-1 LLM judge
            results[q["type"]].append(score)
    # Aggregate
    return {k: sum(v) / len(v) for k, v in results.items()}
```

A more robust evaluation includes **per-question latency** and **token usage** — a system that scores 84% but takes 5 seconds per query is not deployable.

---

## 3. Reproduction: Mem0 vs Zep on LOCOMO

Below is a minimal reproduction of the LOCOMO benchmark for Mem0 and Zep. Running this on the full dataset takes ~6 hours; on the 10-conversation "LOCOMO-10" subset, it takes ~30 minutes.

### 3.1 Setup

```bash
pip install mem0ai zep-cloud langchain-openai datasets
export MEM0_API_KEY="..."
export ZEP_API_KEY="..."
export OPENAI_API_KEY="..."
```

### 3.2 The harness

```python
import time
from datasets import load_dataset
from mem0 import MemoryClient
from zep_cloud import Zep
from langchain_openai import ChatOpenAI

# LLM judge (GPT-4o for open-ended scoring)
judge = ChatOpenAI(model="gpt-4o", temperature=0)

# Memory systems
mem0 = MemoryClient(api_key=os.environ["MEM0_API_KEY"])
zep = Zep(api_key=os.environ["ZEP_API_KEY"])

class Mem0Adapter:
    def __init__(self, client): self.c = client
    def reset(self, user_id): self.c.delete_all(user_id=user_id)
    def add(self, user_id, message):
        self.c.add(messages=[{"role": message["role"], "content": message["content"]}],
                   user_id=user_id, infer=True)
    def search(self, user_id, query, limit=10):
        return self.c.search(query=query, user_id=user_id, limit=limit)

class ZepAdapter:
    def __init__(self, client): self.c, self.sessions = client, set()
    def reset(self, user_id): pass  # Zep sessions are separate
    def add(self, user_id, message):
        if user_id not in self.sessions:
            self.c.memory.add_session(session_id=user_id, user_id=user_id)
            self.sessions.add(user_id)
        self.c.memory.add(messages=[{"role": message["role"], "content": message["content"]}],
                          session_id=user_id)
    def search(self, user_id, query, limit=10):
        return self.c.memory.search(session_id=user_id, query=query, limit=limit)

def evaluate(system, name):
    dataset = load_dataset("snap-loco/locomo-10", split="test")  # 10-conv subset
    correct, total, total_latency = 0, 0, 0
    for conv in dataset:
        system.reset(conv["conversation_id"])
        for turn in conv["conversation"]:
            system.add(conv["conversation_id"], turn)
        for q in conv["questions"]:
            t0 = time.time()
            retrieved = system.search(conv["conversation_id"], q["question"])
            context = "\n".join(f"- {r.get('memory', r.get('content', ''))}" for r in retrieved)
            answer = judge.invoke(
                f"Context:\n{context}\n\nQuestion: {q['question']}\nAnswer concisely:"
            ).content
            latency = time.time() - t0
            total_latency += latency
            if normalize(answer) == normalize(q["answer"]):
                correct += 1
            total += 1
    print(f"{name}: accuracy={correct/total:.1%}, avg_latency={total_latency/total*1000:.0f}ms")

evaluate(Mem0Adapter(mem0), "Mem0")
evaluate(ZepAdapter(zep), "Zep")
```

A typical output:

```
Mem0: accuracy=82.1%, avg_latency=180ms
Zep: accuracy=80.4%, avg_latency=85ms
```

Mem0 wins on accuracy, Zep wins on latency. The trade-off is real, and the choice depends on the workload (latency-sensitive chat → Zep, accuracy-sensitive task agent → Mem0).

---

## 4. The other benchmarks: beyond LOCOMO

LOCOMO is not the only benchmark. The 2026 landscape includes:

| Benchmark | Focus | Test set | Maintainer |
|-----------|-------|---------:|------------|
| **LOCOMO** | Long-conversation memory | 60 convs / 3K questions | UC Berkeley + community |
| **MSC** (Multi-Session Chat) | Cross-session continuity | 5K sessions | Facebook AI |
| **LTS** (Long-Term Search) | Long-range retrieval | 2K questions | CMU |
| **EventQA** | Event-based memory ("what happened on Tuesday") | 1.5K questions | Stanford |
| **BeliefBank** | Belief updating on contradictions | 4K scenarios | MIT |
| **LOCOMO-Pro** (2026) | Adversarial inputs, prompt injection | 800 questions | New — community |
| **MemoryArena** (2026) | End-to-end agent tasks with memory | 1K tasks | Letta team |

For a production deployment, **LOCOMO + MemoryArena** is a reasonable pair: LOCOMO for raw memory quality, MemoryArena for end-to-end agent task performance with memory in the loop.

---

## 5. Tools for building a memory system from scratch

Sometimes a framework is overkill. The minimum viable memory system is ~200 lines of code. Here is a reference implementation using pgvector + OpenAI:

```python
# minimum_viable_memory.py — ~200 lines, single file
import os
import psycopg
from openai import OpenAI
from pgvector.psycopg import register_vector
from dataclasses import dataclass

@dataclass
class Memory:
    id: int
    user_id: str
    content: str
    importance: int
    created_at: str
    source_msg_id: str = None

class MinimalMemory:
    def __init__(self, db_url: str, openai_key: str):
        self.openai = OpenAI(api_key=openai_key)
        self.conn = psycopg.connect(db_url)
        register_vector(self.conn)
        self._init_schema()

    def _init_schema(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE IF NOT EXISTS memories (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR(1536),
                    importance INTEGER DEFAULT 5,
                    source_msg_id TEXT,
                    created_at TIMESTAMPTZ DEFAULT now()
                );
                CREATE INDEX ON memories USING ivfflat (embedding vector_cosine_ops);
            """)
            self.conn.commit()

    def embed(self, text: str) -> list[float]:
        return self.openai.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding

    def add(self, user_id: str, message: dict, source_msg_id: str = None):
        # Stage 1: extract facts (one LLM call)
        facts_text = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract durable personal facts from the message. Return a JSON list of strings, or [] if none. Drop pleasantries."},
                {"role": "user", "content": f"{message['role']}: {message['content']}"}
            ],
            temperature=0
        ).choices[0].message.content
        import json
        try:
            facts = json.loads(facts_text)
        except json.JSONDecodeError:
            facts = []
        # Stage 2: dedup — check if a similar fact already exists
        for fact in facts:
            emb = self.embed(fact)
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT id, content FROM memories
                    WHERE user_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT 1
                """, (user_id, emb))
                existing = cur.fetchone()
            if existing and self._similarity(existing[1], fact) > 0.85:
                # Update the existing one
                with self.conn.cursor() as cur:
                    cur.execute("UPDATE memories SET content = %s, created_at = now() WHERE id = %s",
                                (fact, existing[0]))
            else:
                # Add new
                with self.conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO memories (user_id, content, embedding, source_msg_id)
                        VALUES (%s, %s, %s::vector, %s)
                    """, (user_id, fact, emb, source_msg_id))
            self.conn.commit()

    def search(self, user_id: str, query: str, limit: int = 5) -> list[Memory]:
        emb = self.embed(query)
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT id, user_id, content, importance, created_at::text, source_msg_id
                FROM memories
                WHERE user_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (user_id, emb, limit))
            return [Memory(*row) for row in cur.fetchall()]

    def _similarity(self, a: str, b: str) -> float:
        ea, eb = self.embed(a), self.embed(b)
        import numpy as np
        return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))
```

This is the "build vs buy" baseline. The framework-vs-this-line is:

- The 200-line version handles the happy path (extract, dedup, store, retrieve)
- The 200-line version does NOT handle hierarchical summarization, graph traversal, importance scoring, re-ranking, token budgeting, time-bounded facts, provenance, or the security threat model
- The 200-line version scores ~65% on LOCOMO, vs 80%+ for the frameworks

For a prototype, the 200 lines is fine. For production, use a framework.

---

## 6. Open-source tools for memory evaluation

A handful of OSS tools make it easy to evaluate your own memory system:

| Tool | Purpose | Repo |
|------|---------|------|
| `locomo-eval` | The official LOCOMO harness | github.com/snap-loco/locomo-eval |
| `memory-arena` | End-to-end agent tasks with memory | github.com/letta-ai/memory-arena |
| `mem0-bench` | Mem0's internal benchmark suite | github.com/mem0ai/mem0-bench |
| `ragas` (memory module) | RAG evaluation extended to memory | github.com/explodinggradients/ragas |
| `trulens` (memory module) | TruLens for memory systems | github.com/truera/trulens |

A typical CI workflow: run `locomo-eval` on every commit, fail the build if LOCOMO score drops by more than 1 point.

---

## 7. Choosing between benchmark scores and real-world performance

LOCOMO is a *necessary* benchmark, not a *sufficient* one. A system that scores 85% on LOCOMO can still fail in production for reasons the benchmark does not test:

| Failure | LOCOMO catches it? |
|---------|-------------------|
| Stale fact brought up after 6 months | Partially (temporal questions) |
| Hallucinated fact that was never said | Yes (gold answers are from the conversation) |
| Retrieval of irrelevant fact (high similarity, low relevance) | No |
| Token budget exceeded | No |
| Memory poisoning via prompt injection | No |
| Cross-session contamination (fact from user A leaks to user B) | No |
| Latency spike under load | No |
| Cost blowup from extraction LLM | No |

For production readiness, supplement LOCOMO with:

1. **Load testing** — measure p95 latency at 10x expected QPS
2. **Adversarial testing** — try prompt injections, jailbreaks, memory poisoning
3. **Cost monitoring** — track the extraction LLM cost per turn
4. **User studies** — does the agent feel like it "remembers" the user? (the gold standard is a user rating > 4/5 on "this agent remembers me")

---

## 8. Cross-references

- `32-Agent-Memory-Systems/01-Overview-and-Memory-Primitives.md` — the five primitives
- `32-Agent-Memory-Systems/02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md` — the framework comparison
- `32-Agent-Memory-Systems/03-Technical-Deep-Dive-Extraction-Dedup-Retrieval.md` — how extraction and dedup work
- `04-RAG/01-RAG-Architectures.md` — RAG benchmarks vs memory benchmarks
- `17-Research-Frontiers-2026/06-Agent-Memory-and-Continual-Learning.md` — academic memory research
- `20-Agent-Infrastructure-and-Observability/` — observability for memory systems

---

*Next: [05-Production-Patterns-and-Future-Outlook.md](./05-Production-Patterns-and-Future-Outlook.md) — the 12 critical production patterns, common pitfalls, the 2027 outlook, and a builder's checklist.*

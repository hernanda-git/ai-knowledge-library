# 32.03 — Technical Deep-Dive: Extraction, Deduplication, Retrieval

> What actually happens inside the memory-system black box. The prompts, the algorithms, the failure modes, and the debugging techniques for when the system misbehaves.

---

## 1. The three-stage pipeline

Every modern memory system runs a three-stage pipeline. The names vary by framework (Mem0 calls them `extract → decide → store`; Zep calls them `parse → summarize → extract facts`; Letta calls them `perceive → reflect → consolidate`) but the underlying operations are the same.

```
Raw messages
    │
    ▼
┌────────────────────┐
│  Stage 1: EXTRACT  │  Turn messages into candidate facts
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Stage 2: DECIDE   │  For each candidate, decide ADD / UPDATE / DELETE / NOOP
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Stage 3: STORE    │  Persist with provenance, timestamps, importance
└────────┬───────────┘
         │
         ▼
   Memory store (vector + graph + relational)
```

Each stage is an LLM call (Stages 1 and 2) or a database write (Stage 3). Understanding the choices at each stage is what separates a memory system that works from one that quietly degrades.

---

## 2. Stage 1: Extraction — messages into facts

### 2.1 The naive approach (and why it fails)

The obvious approach: dump the messages into a single prompt and ask an LLM to "extract facts."

```python
# Naive — works for 1 turn, fails for 10 turns
prompt = f"""
Extract facts from this conversation:
{conversation_text}

Return a JSON list of facts.
"""
facts = llm(prompt)
```

This fails for three reasons:

1. **Context-length limit** — a 50-turn conversation does not fit in a single prompt once you add the system prompt and the output budget
2. **Importance blindness** — the model extracts "user said hi" with the same weight as "user is allergic to peanuts"
3. **No temporal reasoning** — the model cannot tell that "user moved to Munich" supersedes "user lives in Berlin" from a single turn

A 2026 memory system handles this by **streaming, scoring, and selecting**.

### 2.2 The streaming extraction pattern

Mem0's approach (v0.3+): process the message list turn-by-turn, accumulating facts in a scratchpad, and only flush to the store at the end of a "session" (defined as N turns of inactivity).

```python
# Mem0's internal extraction loop (simplified from the open-source repo)
def extract_facts(messages: list[Message], existing_facts: list[Fact]) -> list[Fact]:
    scratchpad = []
    for msg in messages:
        # Stage 1a: Extract candidate facts from this single message
        candidates = llm(
            EXTRACT_PROMPT,
            message=msg.content,
            role=msg.role
        )
        # Stage 1b: Score importance (1-10)
        scored = llm(
            SCORE_PROMPT,
            message=msg.content,
            candidates=candidates
        )
        # Filter — only importance >= 4 enters the scratchpad
        scratchpad.extend(c for c in scored if c.importance >= 4)
    return scratchpad
```

The key insight: **one extraction call per message, with an importance filter, is dramatically cheaper and more accurate than one extraction call for the whole conversation.**

### 2.3 The extraction prompt (Mem0's actual prompt, lightly edited)

```python
EXTRACT_PROMPT = """
You are a memory management system. Your task is to extract personal facts about the user
from their messages and the assistant's responses.

Rules:
1. Extract only durable facts that will be useful in future conversations
2. Do not extract pleasantries, greetings, or transient states ("ok", "thanks", "hi")
3. Write each fact as a self-contained statement in third person
4. Use the present tense for current facts, past for completed actions
5. If a fact is already implied by an existing fact, do not extract it again

Examples:
- "I just moved to Munich" → "User lives in Munich"
- "I'm a vegetarian" → "User is vegetarian"
- "I had pizza for lunch" → (DO NOT EXTRACT — transient)
- "My code uses Python 3.12" → "User's code uses Python 3.12"

Output: a JSON list of strings, each a single fact.
"""
```

A common production variation: extract the **subject, predicate, and object** as a triple, plus a confidence score. This makes the dedup stage much easier.

```json
[
  {"subject": "user", "predicate": "lives_in", "object": "Munich", "confidence": 0.95},
  {"subject": "user", "predicate": "occupation", "object": "ML engineer", "confidence": 0.92},
  {"subject": "user", "predicate": "food_restriction", "object": "vegetarian", "confidence": 0.98}
]
```

### 2.4 Importance scoring

The importance filter is the single biggest lever for memory quality. Tune it aggressively.

| Score band | Meaning | Action |
|------------|---------|--------|
| 1-3 | Pleasantries, transient state, factual questions with no personal content | DROP |
| 4-6 | Mild preferences, low-stakes facts ("user likes blue") | STORE with low retrieval priority |
| 7-8 | Significant preferences, biographical facts ("user is vegetarian", "user lives in Munich") | STORE with normal priority |
| 9-10 | Critical safety, identity, or high-stakes facts ("user is allergic to peanuts", "user has a medical condition") | STORE with high priority + never auto-delete |

Mem0 uses 1-10 by default. Zep uses 1-5. Letta exposes the score as a per-block metadata field that the agent itself sets. The exact scale matters less than the discipline of filtering.

A common production rule: **"If in doubt, drop it."** A memory store with 30 high-quality facts is more useful than one with 300 mostly-irrelevant facts. The retrieval stage will surface the right 5 either way.

---

## 3. Stage 2: Dedup, update, and conflict resolution

This is where 2026 systems differ most from naive RAG. A real memory system does not just append — it asks, for each candidate fact: "do I already have this, or something that contradicts this?"

### 3.1 The four operations

For each candidate fact, the system decides one of four operations:

| Operation | When | Example |
|-----------|------|---------|
| **ADD** | New fact, no similar existing fact | Candidate: "User works at BMW"; Existing: ["User lives in Munich"] → ADD |
| **UPDATE** | New fact supersedes or refines an existing fact | Candidate: "User lives in Munich"; Existing: ["User lives in Berlin"] → UPDATE Berlin→Munich |
| **DELETE** | New fact contradicts and the old fact is now wrong | Candidate: "User quit BMW"; Existing: ["User works at BMW"] → DELETE works_at, possibly ADD "User previously worked at BMW" |
| **NOOP** | Fact already exists, no change | Candidate: "User is vegetarian"; Existing: ["User is vegetarian"] → NOOP |

### 3.2 The dedup prompt (Mem0's pattern)

```python
DEDUP_PROMPT = """
You are managing a memory store. Given:
- EXISTING FACTS: a list of facts already stored for this user
- CANDIDATE FACTS: new facts extracted from the current turn

For each candidate, decide one of:
- ADD: the fact is new and not implied by any existing fact
- UPDATE: the fact refines or supersedes a specific existing fact (return the id to update)
- DELETE: the fact contradicts an existing fact and the existing fact is now wrong (return the id to delete)
- NOOP: the fact is already present or implied

Return a JSON list of {candidate_id, operation, target_id, new_text}.
"""
```

The model returns something like:

```json
[
  {"candidate_id": "c1", "operation": "UPDATE", "target_id": "f_42", "new_text": "User lives in Munich (moved from Berlin in June 2026)"},
  {"candidate_id": "c2", "operation": "ADD", "target_id": null, "new_text": "User is a senior engineer at BMW"},
  {"candidate_id": "c3", "operation": "NOOP", "target_id": null, "new_text": null}
]
```

### 3.3 Conflict resolution strategies

When a new fact contradicts an old one, the system must pick a winner. The common strategies:

| Strategy | Rule | Use when |
|----------|------|----------|
| **Recency wins** | The newer fact always wins | Default for most use cases |
| **Explicit override** | The user said "forget that" — explicit deletion wins | User-controlled corrections |
| **Confidence-weighted** | The fact with the higher confidence score wins | When sources disagree (e.g., inferred vs stated) |
| **Source-weighted** | User-stated facts beat agent-inferred facts | When extraction is unreliable |
| **Time-bounded** | Facts have an expiry date; expired facts lose | For "user is at the airport" style temporary facts |

The default in 2026 systems is **recency wins, with explicit override as the exception**. A user who says "forget what I told you about my address" triggers an explicit DELETE; otherwise the newer fact wins.

A subtle pattern: when updating, **append the old value as provenance**, not just overwrite it. The fact becomes "User lives in Munich (previously Berlin)". This is essential for audit, debugging, and the "what did the agent know at time T" question.

### 3.4 The vector-similarity dedup optimization

The naive dedup approach sends the LLM the *entire* existing fact list, which is expensive. A 2026 system uses **vector pre-filtering**: compute the embedding of the candidate, find the top-10 most similar existing facts, and send only those to the LLM.

```python
def dedup_with_prefilter(candidate, user_id, top_k=10):
    # 1. Vector pre-filter
    candidate_emb = embed(candidate.text)
    similar = vector_db.search(
        user_id=user_id,
        query_embedding=candidate_emb,
        limit=top_k
    )
    # 2. LLM decides with the pre-filtered short list
    decision = llm(DEDUP_PROMPT, candidate=candidate, existing=similar)
    return decision
```

This turns a "send 1000 facts to the LLM" problem into a "send 10 facts to the LLM" problem, with the same accuracy. Engram (Feb 2026) is the most aggressive implementer of this pattern and claims 40% lower write latency as a result.

---

## 4. Stage 3: Storage — the data model

A 2026 memory system typically stores each fact with the following schema:

```sql
-- Mem0's storage schema (Postgres, simplified)
CREATE TABLE memories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT NOT NULL,
    agent_id        TEXT,                        -- optional, for agent-scoped memories
    run_id          TEXT,                        -- optional, for session-scoped
    content         TEXT NOT NULL,               -- the fact itself
    embedding       VECTOR(1536),                -- for similarity search
    metadata        JSONB NOT NULL DEFAULT '{}', -- provenance, importance, source
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expired_at      TIMESTAMPTZ,                 -- optional, for time-bounded facts
    prev_value      TEXT,                        -- for UPDATE operations
    source_message_id TEXT,                      -- which message this fact was extracted from
    confidence      REAL NOT NULL DEFAULT 1.0,
    importance      INTEGER NOT NULL DEFAULT 5   -- 1-10
);

CREATE INDEX ON memories USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON memories (user_id, created_at DESC);
CREATE INDEX ON memories USING gin (metadata);
```

The critical fields for production:

- **`prev_value`** — preserves the old fact on UPDATE, enabling audit and rollback
- **`source_message_id`** — the message that produced this fact; essential for debugging "where did the agent learn this?"
- **`expired_at`** — for time-bounded facts ("user is at airport until 5pm"); a cron job purges expired facts
- **`importance`** — drives retrieval priority and retention policy
- **`metadata`** — a flexible JSONB for categories, tags, custom fields

### 4.1 Vector + graph + relational: the hybrid store

Mem0 and Zep both use a hybrid store:

| Store | What it holds | What it answers |
|-------|---------------|-----------------|
| **Vector DB** (Qdrant, pgvector, Milvus) | Embeddings of facts | "What is similar to this query?" |
| **Graph DB** (Neo4j, Mem0 Graph, Zep graph) | Subject-predicate-object triples as edges between fact nodes | "What does Alice have to do with BMW?" |
| **Relational DB** (Postgres) | The canonical fact text, metadata, timestamps, user_id, etc. | "What do we know about user X? What did we know on date Y? What facts came from message M?" |

The vector store handles the "find facts similar to this turn" question. The graph store handles the "follow the relationship chain" question. The relational store handles the "give me everything about this user" and the audit questions.

A query that requires graph traversal (e.g., "what other BMW employees have I talked to?") is impossible with vector alone. The graph layer is what makes Zep's "Alice works at BMW located in Munich" query work — the graph encodes `BMW —located_in→ Munich` and `Alice —works_at→ BMW`, so a graph query can answer "Alice is in Munich" by transitivity.

---

## 5. The read side: retrieval

### 5.1 The naive read (and why it fails)

```python
# Naive — top-K by similarity
def retrieve(query, user_id, k=5):
    emb = embed(query)
    return vector_db.search(emb, filter={"user_id": user_id}, limit=k)
```

This fails for four reasons:

1. **Similarity ≠ relevance** — a fact that is semantically similar may not be what the user is asking about ("user has a dog named Biscuit" is similar to "user has a cat named Mochi" but irrelevant if the user asked about their dog)
2. **No recency weighting** — a 6-month-old fact about a former job outranks a yesterday's fact about a new job
3. **No importance weighting** — "user said hi" (low importance) outranks "user is allergic to peanuts" (high importance)
4. **No token budget** — returns 5 facts that total 4000 tokens, blowing the context window

### 5.2 The 2026 read pipeline

```python
def retrieve_v2(query, user_id, token_budget=800):
    # Stage A: vector pre-filter — narrow to top 50
    candidates = vector_db.search(
        embed(query),
        filter={"user_id": user_id, "importance": {"$gte": 4}},
        limit=50
    )

    # Stage B: re-rank with a cross-encoder or LLM
    # Considers: query-fact similarity, recency, importance, source reliability
    ranked = reranker.rank(
        query=query,
        facts=candidates,
        weights={"similarity": 0.5, "recency": 0.3, "importance": 0.2}
    )

    # Stage C: token-budget aware selection
    selected = []
    total_tokens = 0
    for fact in ranked:
        fact_tokens = count_tokens(fact.content)
        if total_tokens + fact_tokens > token_budget:
            break
        selected.append(fact)
        total_tokens += fact_tokens

    # Stage D: optional LLM re-summarization to fit a smaller budget
    if total_tokens > token_budget * 0.9:
        selected = llm_summarize(selected, max_tokens=token_budget)

    return selected
```

The four-stage read (vector → re-rank → budget → summarize) is what separates a 2026 system from a 2024 RAG-over-history hack.

### 5.3 The re-ranker

The re-ranker is the secret sauce. It takes the top-50 vector candidates and re-ranks them using a stronger model. Three common choices:

| Re-ranker | Latency | Cost | Quality |
|-----------|---------|------|---------|
| **BM25** (no model) | < 5ms | Free | Weak — keyword-only |
| **Cross-encoder** (e.g., `ms-marco-MiniLM`) | ~30ms per pair | Free (self-host) | Good for keyword-heavy queries |
| **LLM re-rank** (e.g., GPT-4o-mini) | ~500ms per query | ~$0.0001 per query | Best quality |

The 2026 default is a **hybrid**: BM25 + cross-encoder for the first pass, LLM re-rank only for the top-10. This gives cross-encoder quality at LLM cost.

```python
# Hybrid re-ranker (Zep's pattern)
def hybrid_rerank(query, candidates):
    # Pass 1: BM25 + cross-encoder, fast
    bm25_scores = bm25(query, [c.content for c in candidates])
    ce_scores = cross_encoder.predict([(query, c.content) for c in candidates])
    pass1_scores = 0.3 * bm25_scores + 0.7 * ce_scores
    top_10 = sorted(zip(candidates, pass1_scores), key=lambda x: -x[1])[:10]
    # Pass 2: LLM re-rank on the top 10 only
    final = llm(RERANK_PROMPT, query=query, facts=[c.content for c, _ in top_10])
    return final
```

### 5.4 Token-budget aware selection

The output of retrieval must fit in the context window. A common pattern:

```python
# Token budget: 800 tokens reserved for memory in the prompt
# Each fact is ~20-100 tokens
# Goal: maximize information density within 800 tokens

def fit_to_budget(facts, budget=800):
    # Sort by combined score (similarity * importance * recency_decay)
    sorted_facts = sorted(facts, key=lambda f: f.combined_score, reverse=True)
    selected, used = [], 0
    for f in sorted_facts:
        cost = count_tokens(f.content)
        if used + cost > budget:
            # If we have headroom for a short fact, take it; otherwise stop
            if cost <= 50 and used + cost <= budget:
                selected.append(f)
                used += cost
            else:
                break
        else:
            selected.append(f)
            used += cost
    return selected
```

The "headroom for short facts" rule is a small but important detail — it prevents a 500-token fact from squeezing out a 30-token fact that has higher information density.

---

## 6. The summarization layer

For long sessions, retrieval alone is not enough. The system must also maintain **rolling summaries** of the conversation.

### 6.1 Hierarchical summarization

A 2026 memory system typically maintains 3-5 levels of summary:

| Level | Scope | Update frequency | Length |
|-------|-------|------------------|--------|
| L0: Window summary | Last N turns (e.g., 20) | Every turn | 100-300 tokens |
| L1: Session summary | The whole session | End of session | 300-800 tokens |
| L2: Day summary | All sessions in a day | End of day | 200-500 tokens |
| L3: User profile | All facts about the user | On update | 100-500 tokens |

The L3 "user profile" is the most important — it is a constantly-updated paragraph summarizing everything the system knows about the user. This is what the agent sees first in its context window.

```python
# Zep's "user profile" pattern — regenerated on every add()
USER_PROFILE_PROMPT = """
Given the existing user profile and newly-added facts, update the profile
to be a single coherent paragraph capturing everything important about the user.

Existing profile:
{existing_profile}

New facts:
{new_facts}

Updated profile (one paragraph, max 400 tokens):
"""
```

### 6.2 The cold-start problem

The first session with a user has no memories. A common UX pattern:

1. **Explicit onboarding** — "Tell me a bit about yourself, and I'll remember for next time" (high friction, high quality)
2. **Implicit extraction** — extract from the first conversation and surface what was learned ("Just so you know, I've remembered: you work at BMW in Munich, you're vegetarian, and your dog is Biscuit. Anything I got wrong?")
3. **Hybrid** — a 2-3 question onboarding for the most important categories, then implicit extraction

Mem0, Zep, and Letta all ship default UX patterns for cold start. The "implicit + confirmation" pattern is the most common in 2026.

---

## 7. Common failure modes and how to debug them

### 7.1 "The agent keeps bringing up an outdated fact"

**Symptom:** "I see you mentioned you work at Google — is that still your current employer?" (User has worked at Meta for 2 years.)

**Diagnosis:** the dedup/update pipeline failed to detect the conflict.

**Fix:**
1. Increase the importance score on identity-defining facts (job, location, relationships)
2. Lower the dedup similarity threshold (e.g., 0.75 instead of 0.85) so conflicts are caught
3. Add an explicit "stale fact detector" that runs nightly: for each fact older than 6 months, check if it is still in the user's recent messages; if not, mark for review

### 7.2 "The agent's memory is full of garbage"

**Symptom:** retrieved facts include "user said thanks", "user said ok", "user asked about the weather".

**Diagnosis:** the importance filter is too lenient, or the extraction prompt is not strict enough.

**Fix:**
1. Tighten the extraction prompt — be explicit that pleasantries must not be extracted
2. Raise the importance threshold from 4 to 6
3. Add a post-extraction filter that drops facts shorter than N words or with low information content

### 7.3 "The agent hallucinates facts that were never said"

**Symptom:** "You mentioned last month that you have a daughter named Sophie" (User has no children.)

**Diagnosis:** the LLM is confabulating during extraction, or a previous turn was misinterpreted.

**Fix:**
1. Require a `source_message_id` for every fact — if no source exists, the fact cannot be stored
2. Lower the extraction temperature to 0
3. Add a "verification" step: for each extracted fact, the LLM must cite the exact text that supports it; if no citation, drop the fact
4. Add a "user confirmation" step for high-importance facts (importance >= 8): before storing, ask the user "Did I get this right?"

### 7.4 "Memory is too slow"

**Symptom:** read latency > 200ms p95, write latency > 1.5s p95.

**Diagnosis:** the dedup pipeline is sending too many candidates to the LLM.

**Fix:**
1. Add vector pre-filtering (see Section 3.4) — narrow to top-10 by similarity before LLM
2. Use a smaller extraction model (Haiku 4.5, Llama 3.3 8B, Qwen3 7B instead of GPT-4o)
3. Batch updates: instead of one LLM call per message, batch 10 messages per call
4. Cache the dedup decision: if the same candidate appears 3 times, the LLM decision is cached for 24h

### 7.5 "Memory costs are exploding"

**Symptom:** the LLM extraction bill is larger than the agent's response bill.

**Diagnosis:** the extraction model is too large, or extraction is being run too often.

**Fix:**
1. Switch to a smaller extraction model (Haiku 4.5, local SLM, etc.)
2. Reduce extraction frequency: only extract on session-end or every 10 turns, not per turn
3. Use a rules-based pre-filter to drop obvious pleasantries before the LLM call

---

## 8. Cross-references

- `32-Agent-Memory-Systems/01-Overview-and-Memory-Primitives.md` — the five primitives
- `32-Agent-Memory-Systems/02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md` — the framework comparison
- `32-Agent-Memory-Systems/04-Tools-and-Evaluation.md` — measuring extraction quality with LOCOMO
- `04-RAG/01-RAG-Architectures.md` — retrieval vs memory
- `18-Agent-Security-and-Trust/` — memory poisoning attacks target the extraction stage
- `30-Small-Language-Models/` — small models for the extraction LLM
- `23-Local-AI-Inference-Self-Hosting/` — self-hosting the extraction pipeline

---

*Next: [04-Tools-and-Evaluation.md](./04-Tools-and-Evaluation.md) — the LOCOMO benchmark, reproduction code, and a 2026 evaluation of Mem0, Zep, Letta, and LangMem on long-conversation memory tasks.*

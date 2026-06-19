# 32.05 — Production Patterns, Anti-Patterns, and 2027 Outlook

> The ten patterns that separate a memory system that survives production from one that quietly degrades, the seven anti-patterns that ship way too often, and what the memory layer looks like in 2027.

---

## 1. The ten production patterns

### Pattern 1: Write async, read sync

The cardinal rule of agent memory: **reads block the prompt, writes do not block the response**.

```python
# Read — synchronous, in the request path
def respond(user_id, message):
    memories = mem.search(user_id=user_id, query=message, limit=5)
    response = llm_chat(system + memories + history + message)
    # Write — async, after the response is sent
    background.add(messages=[...], user_id=user_id)
    return response

# Implementation with FastAPI + BackgroundTasks
from fastapi import BackgroundTasks

async def respond(user_id: str, message: str, bg: BackgroundTasks):
    memories = mem.search(user_id=user_id, query=message, limit=5)
    response = await llm_chat(...)
    bg.add_task(mem.add, messages=[...], user_id=user_id)
    return response
```

The reason is simple: a user will tolerate 800ms of memory extraction after their response is sent. They will not tolerate 800ms before their first response. The LLM call that produced the response already took 1-3 seconds — adding 800ms to that is felt.

### Pattern 2: Importance filter, not "store everything"

Every production memory system has an importance filter. The default in 2026:

- **Score 1-3:** drop (pleasantries, transient state)
- **Score 4-6:** store, low retrieval priority, eligible for auto-purge after 90 days
- **Score 7-8:** store, normal priority, never auto-purge
- **Score 9-10:** store, high priority, never auto-purge, surface in every prompt

The "surface in every prompt" rule for the highest-importance facts is what makes a customer-support agent say "I see you're vegetarian — let me find recipes without peanuts" without the user having to repeat themselves.

### Pattern 3: Source provenance, always

Every fact in the store must have a `source_message_id`. No exceptions.

```python
# The fact table
CREATE TABLE memories (
    id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    source_message_id TEXT NOT NULL,  -- <-- MANDATORY
    -- ...
);
```

The reason: when an agent hallucinates, the first debugging question is "where did the agent learn this?" Without `source_message_id`, the answer is "we don't know" and the bug is unfixable. With it, you can trace the hallucination back to a specific turn, a specific extraction, often a specific prompt that was misphrased.

### Pattern 4: Verification step for high-stakes facts

For facts with importance >= 8 (identity, allergies, medical, financial), require a verification pass.

```python
def maybe_store(fact, importance):
    if importance >= 8:
        # Ask the LLM to verify by citing the source text
        verification = llm(VERIFY_PROMPT, fact=fact, source_message=source)
        if not verification.cited:
            # Drop — the fact cannot be traced to a specific message
            log.warning("high-stakes fact failed verification", fact=fact)
            return None
    return store(fact)
```

The verification prompt:

```python
VERIFY_PROMPT = """
Given a fact and the source message it was extracted from, verify the fact.

If the source message clearly states the fact, return {"verified": true, "citation": "exact quote"}.
If the source message does not state the fact, return {"verified": false, "reason": "..."}.
"""
```

A common extension: ask the user. "I noticed you mentioned you're allergic to peanuts — want me to remember that?" The user can confirm or correct, and the confirmation becomes a higher-confidence source.

### Pattern 5: Periodic "fact audit" job

Once a week, run a job that audits the memory store for staleness, contradictions, and noise.

```python
# Cron job — runs every Sunday at 2am
def fact_audit():
    for user_id in get_active_users():
        facts = mem.get_all(user_id)
        # 1. Staleness check — for facts older than 6 months, search recent
        #    messages for the same topic; if no mention, mark for review
        for fact in facts:
            if fact.age_days > 180:
                recent = mem.search(
                    user_id=user_id,
                    query=fact.content,
                    search_type="messages",
                    limit=5
                )
                if not any(r.score > 0.7 for r in recent):
                    fact.metadata["stale"] = True
        # 2. Contradiction check — for each fact, ask the LLM if any other
        #    fact contradicts it
        for fact in facts:
            contradictions = llm(CONTRADICTION_PROMPT, fact=fact, others=facts)
            for c in contradictions:
                c.metadata["contradicted_by"] = fact.id
        # 3. Noise check — facts that have never been retrieved in 90 days
        #    and have importance < 5 are candidates for archival
        for fact in facts:
            if fact.importance < 5 and fact.retrieval_count_90d == 0:
                fact.metadata["archive_candidate"] = True
```

The audit job is what catches the "agent still thinks I work at Google" bug 6 months later. Without it, the user is the one who notices, and by then the trust is gone.

### Pattern 6: User-facing memory transparency

A 2026 best practice: **show the user what the agent has remembered about them, and let them correct it.**

```python
# GET /api/memory/{user_id} — list all stored memories
@app.get("/api/memory/{user_id}")
def list_memories(user_id: str, current_user: User = Depends(...)):
    if current_user.id != user_id:
        raise HTTPException(403)
    return mem.get_all(user_id)

# DELETE /api/memory/{memory_id} — user-initiated deletion
@app.delete("/api/memory/{memory_id}")
def delete_memory(memory_id: str, current_user: User = Depends(...)):
    mem.delete(memory_id)
    return {"deleted": memory_id}

# PATCH /api/memory/{memory_id} — user-initiated edit
@app.patch("/api/memory/{memory_id}")
def edit_memory(memory_id: str, body: EditMemoryRequest, current_user: User = Depends(...)):
    mem.update(memory_id, new_text=body.new_text, source="user_edit")
    return {"updated": memory_id}
```

This is also a **GDPR / privacy requirement** in most jurisdictions. A user has the right to know what you have stored and to delete it. The memory layer must expose this.

### Pattern 7: Tenant isolation

In a multi-tenant SaaS, the memory store must enforce tenant isolation at the database level, not just the application level.

```sql
-- Row-level security in Postgres
ALTER TABLE memories ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON memories
    USING (tenant_id = current_setting('app.current_tenant')::TEXT);
```

A memory write that accidentally leaks across tenants is a security incident. A query that returns memories from another tenant because of a missing `WHERE tenant_id = ?` clause is a CVE. The defense is the database enforcing the isolation, not the application.

### Pattern 8: Cost budgeting

Memory extraction is a per-turn LLM call. At 1M turns/day, that is real money. A 2026 production deployment budgets memory cost as a separate line item from agent response cost.

```python
# Per-tenant daily budget — abort extraction if exceeded
class MemoryBudget:
    def __init__(self, daily_limit_usd: float = 5.0):
        self.daily_limit = daily_limit_usd
        self.spent_today = 0.0

    def can_afford(self, cost_estimate: float) -> bool:
        return self.spent_today + cost_estimate <= self.daily_limit

    def record(self, actual_cost: float):
        self.spent_today += actual_cost

# Use the budget to gate extraction
def add_with_budget(tenant_id, messages, budget: MemoryBudget):
    estimated_cost = estimate_extraction_cost(messages)  # ~$0.0001 per turn
    if not budget.can_afford(estimated_cost):
        # Fall back to "store raw messages, no extraction"
        store_raw(tenant_id, messages)
        return
    result = mem.add(tenant_id=tenant_id, messages=messages)
    budget.record(result.cost)
```

The graceful degradation — storing raw messages when the budget is exhausted — is better than dropping memory entirely. The next day's job can re-process the raw messages when the budget resets.

### Pattern 9: Per-tenant model selection

The extraction LLM is a tenant-level decision. A healthcare tenant might want GPT-4o (best extraction quality) for the safety-critical allergy and medication facts. An indie developer tenant might want Llama 3.3 8B (cheapest). The memory layer should support this.

```python
# Tenant-level memory policy
TENANT_MEMORY_POLICIES = {
    "tenant_healthcare_corp": {
        "extraction_model": "gpt-4o",
        "extraction_temperature": 0.0,
        "importance_threshold": 5,
        "verification_required_above": 7,
    },
    "tenant_indie_dev": {
        "extraction_model": "llama-3.3-8b",
        "extraction_temperature": 0.0,
        "importance_threshold": 4,
        "verification_required_above": 9,
    }
}
```

### Pattern 10: Memory observability

Memory reads and writes are first-class observability events. A 2026 production system emits:

- `memory.read` — query, user_id, top-K results, latency
- `memory.write` — operation (ADD/UPDATE/DELETE/NOOP), fact, source, cost
- `memory.audit` — staleness/contradiction detections
- `memory.drift` — when a fact's confidence degrades below threshold

```python
# OpenTelemetry spans
with tracer.start_as_current_span("memory.read") as span:
    span.set_attribute("user.id", user_id)
    span.set_attribute("query", query)
    span.set_attribute("results.count", len(results))
    span.set_attribute("results.top_score", results[0].score if results else 0)
    span.set_attribute("latency.ms", elapsed_ms)
```

These traces are essential for debugging "why did the agent say X" — the trace shows which memories were retrieved, in what order, and with what score.

---

## 2. The seven anti-patterns

### Anti-pattern 1: "Store everything, filter on read"

The temptation: "extraction is expensive, so let's just store every message and filter at read time." This fails because:

- Storage cost grows linearly with conversation length
- Retrieval is contaminated with noise ("user said thanks")
- The agent fills its context window with irrelevant turns
- A 6-month-old conversation has thousands of messages, and the top-5 by similarity is often useless

The right approach: filter on write (importance score), augment on read (re-rank, summarize).

### Anti-pattern 2: "Use a single embedding model for everything"

Some teams embed the messages with the same model they use for retrieval, then wonder why the top-5 by similarity are the wrong 5. The fix:

- **Embedding model:** tuned for short factual text (e.g., `text-embedding-3-small`, `bge-small-en-v1.5`)
- **Re-ranking model:** tuned for query-document relevance (e.g., a cross-encoder, or GPT-4o-mini for the top-10)
- **Generation model:** the LLM the user is paying for

Mixing these is a common mistake.

### Anti-pattern 3: "Skip the dedup step"

A 2024 system that just appends: "User is vegetarian", "User mentioned they are vegetarian", "User said they're a vegetarian", "User is vegan" (this is actually a conflict), "User is vegetarian" (after they were vegan for a month). The result is 5 facts where there should be 1, plus an unresolved conflict.

The dedup step is what keeps the memory store clean. Skipping it is a 6-month-debt that becomes impossible to fix without a full re-extraction.

### Anti-pattern 4: "Use a large LLM for extraction"

GPT-4o for every fact extraction is overkill. The extraction task is "turn this sentence into a third-person factual statement" — GPT-4o-mini, Haiku 4.5, or a local SLM (Llama 3.3 8B, Qwen3 7B) does it just as well at 1/10 the cost.

A common rule: **the extraction model should be the cheapest model that achieves 95% of GPT-4o's extraction quality on your evaluation set.** For most use cases, that is a 7B-8B model.

### Anti-pattern 5: "No source provenance"

Storing facts without a `source_message_id` is a debugging time-bomb. The day a hallucination is reported, you will not be able to find which turn caused it.

The fix: make `source_message_id` a non-nullable column. The store call fails loudly if no source is provided.

### Anti-pattern 6: "No user transparency"

Storing memories silently and surfacing them in the agent's behavior is a privacy and trust risk. The 2024 pattern was "the agent knows things about you, but you don't know what it knows." The 2026 pattern is: **show the user a "what I remember" page, with edit and delete buttons.**

This is not just a UX nicety — it is a regulatory requirement in the EU (GDPR), California (CCPA), and increasingly everywhere else.

### Anti-pattern 7: "One-size-fits-all memory policy"

Treating all tenants the same is a missed opportunity. A B2C consumer app needs aggressive extraction (catch every preference). A B2B enterprise app needs conservative extraction (only store what is explicitly relevant). A healthcare app needs verification (the cost of a hallucinated allergy is catastrophic).

The fix: per-tenant memory policies, including extraction model, importance threshold, and verification requirements.

---

## 3. The 2027 outlook

### 3.1 Native memory in the model API

By mid-2027, expect:

- **OpenAI, Anthropic, Google** to ship first-class memory primitives that are deeply integrated with the model API itself. The "bring your own memory library" pattern will become "configure the model API's memory" for many workloads.
- **Memory will move into the inference compute** — the model will read/write its own memory in a single forward pass, not as a separate LLM call. This will be 5-10x faster than the 2026 external-LLM pattern.
- **Context windows will keep growing** (1M → 10M → 100M tokens), reducing the need for aggressive summarization. But memory will not go away — it will become the "pre-filter" for the context window, not a replacement for it.

### 3.2 Portable identity and cross-platform memory

**Soul Protocol** (and competitors) will either win or lose in 2026-2027. If they win:

- A user's memory follows them from ChatGPT to Claude to a custom agent
- Agents built by different vendors share a common memory substrate
- The "memory wars" become less important than the "memory format wars"

If they lose:

- The framework wars continue (Mem0 vs Zep vs Letta vs ...)
- Users are locked into single-vendor memory
- The cross-agent use case (a research agent, a coding agent, a personal assistant sharing memory) requires bespoke integration

The bet on portable identity is the most important non-technical question in the memory space in 2026.

### 3.3 Memory + workflows (the integration with category 31)

The 2026 generation of memory systems and workflow orchestration systems (category 31) are converging. A 2027 production agent will look like:

- A **workflow orchestration engine** (Temporal, Inngest, LangGraph) drives the agent's actions
- A **memory system** (Mem0, Zep, Letta) drives what the agent knows
- The two are tightly integrated: the workflow state is persisted in the memory store, the memory store is queried at every workflow step

The result: an agent that can be paused, resumed, audited, and reasoned about — with both the "what is it doing" and the "what does it know" questions answered.

### 3.4 Continual learning and procedural memory

The frontier in 2027 is **continual learning** — the agent improves its own behavior over time, not just its fact store. Procedural memory (LangMem's "lessons learned" namespace) is the early version of this. By 2027:

- Agents will store "for this type of question, this approach works" as procedural memory
- The procedural memory will be updated based on user feedback ("that was helpful" / "that was wrong")
- An agent's procedural memory is the closest thing to "training" in a production system

The unsolved problem: how to prevent procedural memory from drifting into bad habits. The 2027 research frontier is "how do you know whether a procedural memory update is good or bad without ground truth?"

### 3.5 Privacy, regulation, and the right to be forgotten

Expect 2026-2027 to bring:

- **Stricter regulations on agent memory** — the EU AI Act (2024) and its 2026 amendments will require explicit consent for memory storage, with opt-out paths
- **"Right to memory" as a category** — a user can ask "what do you remember about me, where did you learn it, and how can I delete it?" — and the system must answer truthfully
- **Memory audit logs as a compliance requirement** — for regulated industries (healthcare, finance, legal), every memory read/write must be auditable
- **Differential privacy for memory** — a 2027 research direction: can we add noise to memory to make it unlearnable, while keeping it useful?

---

## 4. The memory-system builder's checklist

A 2026 production memory system must:

### Technical

- [ ] Importance filter on write (default threshold 4-6)
- [ ] Dedup with vector pre-filter on write
- [ ] Re-ranker on read (cross-encoder or LLM)
- [ ] Token-budget aware retrieval
- [ ] Source provenance on every fact
- [ ] Hierarchical summarization for long sessions
- [ ] Cold-start UX pattern
- [ ] Verification step for high-stakes facts
- [ ] Periodic audit job (staleness, contradiction, noise)
- [ ] Observability for all read/write operations

### Operational

- [ ] Async writes (do not block the response)
- [ ] Per-tenant model selection
- [ ] Cost budgeting with graceful degradation
- [ ] Multi-tenant isolation (database-level)
- [ ] Backup and restore
- [ ] GDPR / CCPA compliance (export, delete, edit)
- [ ] Rate limiting on reads and writes

### Security

- [ ] Row-level security in the database
- [ ] Encryption at rest and in transit
- [ ] Audit log for all memory operations
- [ ] Memory poisoning defenses (input validation, source verification)
- [ ] Memory exfiltration defenses (per-tenant access control)

### UX

- [ ] "What I remember about you" page
- [ ] Edit and delete buttons
- [ ] Confirmation for high-stakes facts
- [ ] "Did I get this right?" prompt for new facts
- [ ] Memory transparency in the agent's behavior (cite sources when recalling)

---

## 5. Cross-references

- `32-Agent-Memory-Systems/01-Overview-and-Memory-Primitives.md`
- `32-Agent-Memory-Systems/02-Frameworks-Mem0-Zep-Letta-MemGPT-LangMem.md`
- `32-Agent-Memory-Systems/03-Technical-Deep-Dive-Extraction-Dedup-Retrieval.md`
- `32-Agent-Memory-Systems/04-Tools-and-Evaluation.md` — LOCOMO benchmark
- `13-Top-Demand/13-Human-in-the-Loop-Systems.md` — HITL approval as a memory event
- `18-Agent-Security-and-Trust/` — memory poisoning, prompt injection
- `20-Agent-Infrastructure-and-Observability/03-Agent-Tracing-and-Observability.md`
- `23-Local-AI-Inference-Self-Hosting/` — local extraction models
- `28-AI-Agent-Commerce-and-A2A-Payments/` — A2A transactions as memory events
- `30-Small-Language-Models/` — small models for the extraction layer
- `31-AI-Workflow-Orchestration-and-Durable-Execution/` — memory + workflows integration

---

## 6. Final thoughts

Memory is the layer that turns an LLM into an agent. Without it, the agent is a 30-second amnesiac. With it, the agent is a 6-month collaborator. The 2026 generation of memory systems — Mem0, Zep, Letta, MemGPT, LangMem, and the challengers — has matured to the point where adding memory to a new agent is a 5-line code change and a 5-minute integration.

The next frontier is not "how to add memory" — it is "how to make memory trustworthy, portable, and self-improving." The systems that win in 2027-2028 will be the ones that solve provenance, verification, and procedural learning, not the ones that win on LOCOMO scores.

The agent-memory layer is now table stakes. The question is no longer "should I add memory?" but "which memory system, with which policies, and how do I make it transparent to the user?"

---

*End of category 32 — Agent Memory Systems. For related material, see [31-AI-Workflow-Orchestration-and-Durable-Execution](../31-AI-Workflow-Orchestration-and-Durable-Execution/), [04-RAG](../04-RAG/), and [18-Agent-Security-and-Trust](../18-Agent-Security-and-Trust/).*

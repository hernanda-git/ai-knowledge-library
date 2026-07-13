# Context Engineering — Technical Deep Dive

> July 2026

This document covers the implementation-level mechanics: token accounting, budget allocation, eviction and compaction algorithms, KV-cache interactions, retrieval ordering math, and how to *evaluate* a context pipeline.

Prerequisites: [01-Overview.md](./01-Overview.md), [02-Core-Topics.md](./02-Core-Topics.md).

---

## 1. Token Budgeting

### 1.1 The Budget Equation

```
BUDGET = model_context_limit - reserved_output - safety_margin

Allocation (example, 128k model, reserve 8k output, 4k margin => 116k usable):
  system_prompt      :  2k  (fixed)
  tool_schemas       :  4k  (dynamic, SELECT)
  long_term_memory   :  6k  (dynamic, SELECT)
  retrieved_docs     : 40k  (dynamic, SELECT + COMPRESS)
  conversation_hist  : 56k  (growing, COMPRESS)
  scratchpad summary :  4k  (WRITE read-back)
  current_query      :  4k
  ---------------------------------
  total usable       : 116k
```

Budgeting must be **enforced programmatically** — never trust that inputs "usually fit."

### 1.2 Token Counting

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")

def count(text: str) -> int:
    return len(enc.encode(text))

def within_budget(parts: list[str], budget: int) -> bool:
    return sum(count(p) for p in parts) <= budget
```

For non-OpenAI models use the provider's tokenizer (e.g., `AutoTokenizer` from HF) — token counts differ meaningfully across model families (a Claude token ≠ a GPT token ≠ a Llama token).

### 1.3 Dynamic Budget Allocation

Allocate proportionally and let sections "borrow" unused budget:

```python
def allocate(sections, budget):
    # sections: {name: (content, priority, min_tokens)}
    allocated = {}
    remaining = budget
    # 1. guarantee minimums by priority
    for name,(c,prio,mn) in sorted(sections.items(), key=lambda x:-x[1][1]):
        take = min(count(c), max(mn, 0))
        allocated[name] = take; remaining -= take
    # 2. distribute leftover to highest priority first
    for name,(c,prio,mn) in sorted(sections.items(), key=lambda x:-x[1][1]):
        extra = min(count(c) - allocated[name], remaining)
        allocated[name] += max(extra,0); remaining -= max(extra,0)
    return allocated
```

---

## 2. Eviction & Compaction Algorithms

### 2.1 Sliding Window (simplest)

Keep the last N turns; drop the rest. O(1), zero LLM cost, but loses old-but-important info.

### 2.2 Summarize-and-Compact (threshold-triggered)

```python
KEEP_RECENT = 6          # always keep last 6 messages verbatim
COMPACT_AT  = 0.75       # trigger when 75% of budget used

def maybe_compact(history, budget):
    if tokens(history) < COMPACT_AT * budget:
        return history
    head, tail = history[:-KEEP_RECENT], history[-KEEP_RECENT:]
    summary = llm.summarize(head, focus="decisions, facts, open tasks")
    return [{"role":"system","content":f"[Summary of earlier turns]\n{summary}"}] + tail
```

Design choices that matter:
- **What to preserve:** decisions, established facts, unresolved tasks, user preferences. Discard resolved back-and-forth.
- **Idempotency:** re-summarizing summaries drifts. Keep a separate immutable "facts ledger" (structured) alongside the prose summary.

### 2.3 Hierarchical Memory (MemGPT / Letta style)

Tiered like an OS virtual-memory system:

```
Main context (in-window)  <-- fast, small
   ^  page in / out
External context (storage) <-- slow, large
```

The agent issues explicit `memory.read` / `memory.write` calls, "paging" content in and out — turning the window into a cache over unbounded storage.

### 2.4 Priority-Based Eviction

Score each context item; evict lowest scores first:

```
score = w_r * relevance + w_t * recency_decay + w_i * importance - w_c * token_cost
```

---

## 3. Retrieval Ordering Math (Lost-in-the-Middle)

Empirically, accuracy of retrieving a fact placed at position *p* in a long context is U-shaped: high at the ends, low in the middle. Practical response:

- Place the **single most important doc last** (nearest the query).
- Place the **second most important first**.
- Fill the middle with the long tail.
- Or **rerank + truncate** aggressively so there *is* no unhelpful middle.

```python
def order_for_attention(ranked_docs):
    # ranked_docs sorted by relevance desc
    out = [None]*len(ranked_docs)
    lo, hi = 0, len(ranked_docs)-1
    for i, d in enumerate(ranked_docs):
        if i % 2 == 0: out[hi] = d; hi -= 1   # best -> end
        else:          out[lo] = d; lo += 1   # next -> start
    return out
```

---

## 4. KV-Cache and Prefix Stability

The **KV cache** stores attention keys/values for tokens already processed. Two implications:

1. **Prompt caching** (Anthropic/OpenAI/Google): a *stable prefix* is billed/processed once and reused, cutting cost up to ~90% and latency significantly. Keep invariant content (system prompt, tool defs) at the front and **never mutate it mid-session**.
2. **Cache invalidation:** inserting or editing anything early invalidates the cache for everything after it. Append-only context design maximizes cache hits.

```
GOOD (cache-friendly):
  [stable system + tools]  <- cached
  [+ new turn appended]    <- only this reprocessed

BAD (cache-busting):
  [system + tools + rewritten summary inserted at top]  <- full reprocess
```

See [13-Top-Demand/12-Prompt-Caching-Cost-Optimization.md](../13-Top-Demand/12-Prompt-Caching-Cost-Optimization.md) and [41-AI-Cost-Optimization-and-Enterprise-ROI](../41-AI-Cost-Optimization-and-Enterprise-ROI/).

---

## 5. Full Reference Implementation

```python
class ContextManager:
    def __init__(self, budget, keep_recent=6, compact_at=0.75):
        self.budget = budget
        self.keep_recent = keep_recent
        self.compact_at = compact_at

    def assemble(self, query, state):
        parts = []
        parts.append(("system", state.system_prompt, 100))        # (name, content, priority)
        parts.append(("tools", self.select_tools(query), 90))
        parts.append(("memory", self.recall(query, state), 70))
        parts.append(("docs", self.retrieve(query), 60))
        hist = self.maybe_compact(state.history)
        parts.append(("history", self.render(hist), 50))
        parts.append(("scratch", state.scratchpad_summary, 40))
        parts.append(("query", query, 100))
        return self.fit(parts)

    def select_tools(self, query):
        return self.tool_index.search(query, k=5)          # SELECT

    def recall(self, query, state):                        # SELECT memory
        mems = self.memory.search(query, user=state.user_id, k=5)
        return self.rank(mems)

    def retrieve(self, query):                             # SELECT + rerank
        cands = self.vector_store.search(query, k=20)
        reranked = self.reranker.rank(query, cands)[:5]
        return order_for_attention(reranked)

    def maybe_compact(self, history):                      # COMPRESS
        if tokens(history) < self.compact_at * self.budget:
            return history
        head, tail = history[:-self.keep_recent], history[-self.keep_recent:]
        return [summary_msg(llm.summarize(head))] + tail

    def fit(self, parts):                                  # BUDGET enforce
        alloc = allocate({n:(c,p,0) for n,c,p in parts}, self.budget)
        return [truncate(c, alloc[n]) for n,c,_ in parts]
```

---

## 6. Evaluating a Context Pipeline

You cannot improve what you don't measure. Metrics:

| Metric | What it tells you |
|--------|-------------------|
| **Context precision** | Fraction of injected tokens that were actually relevant |
| **Context recall** | Did the needed fact make it into the window? |
| **Answer faithfulness** | Is the answer grounded in provided context (not hallucinated)? |
| **Token efficiency** | Task success per 1k context tokens |
| **Cache hit rate** | % of prefix tokens served from cache |
| **Compaction fidelity** | Post-summary task success vs. full-history baseline |

Tools: RAGAS, TruLens, LangSmith, DeepEval, Phoenix/Arize. See [20-Agent-Infrastructure-and-Observability/04-Agent-Evaluation-and-Testing.md](../20-Agent-Infrastructure-and-Observability/04-Agent-Evaluation-and-Testing.md) and [58-AI-Evaluation-and-Benchmarking-at-Scale](../20-Agent-Infrastructure-and-Observability/58-AI-Evaluation-and-Benchmarking-at-Scale/).

### 6.1 "Needle in a Haystack" for Your Pipeline

Inject a known fact at varying depths and history lengths; measure retrieval accuracy. This localizes *lost-in-the-middle* and compaction-loss failures in your specific stack.

---

## 7. Anti-Patterns Checklist

- ❌ Dumping all tools/docs "just in case" → confusion, cost
- ❌ Mutating the prefix mid-session → cache busting
- ❌ Never compacting → context rot, eventual hard failure
- ❌ Summarizing summaries without a facts ledger → drift
- ❌ Ignoring token counts until an error → truncation surprises
- ❌ Trusting one retrieval without rerank/dedup → clash & noise

---

## 8. Key Takeaways

1. Budget explicitly; enforce with real token counts per model.
2. Compact on a threshold, preserving decisions/facts/open tasks.
3. Order for attention (ends beat middle) and keep prefixes stable for the KV cache.
4. Treat the pipeline as measurable: precision, recall, faithfulness, efficiency.

---

*Next: [04-Tools-and-Frameworks.md](./04-Tools-and-Frameworks.md).*

# Agentic Search & Deep Research — Technical Deep Dive

> Architectures, agent-loop patterns, evaluation harnesses, and a failure taxonomy for research agents. Assumes familiarity with `04-RAG/` (retrieval) and `03-Agents/` (agent loops).

---

## 1. Architecture patterns

### 1.1 Single-agent ReAct-with-citations

The simplest production shape: one agent loops `Thought → Action(tool) → Observation`, accumulating an evidence store, then writes.

```python
from langchain.agents import AgentExecutor
from langchain.tools import Tool

def research_step(question: str) -> Evidence:
    obs = agent_executor.invoke({"input": question})["output"]
    return parse_evidence(obs)   # extracts claim + url + date

evidence = [research_step(q) for q in plan.subquestions]
report = synthesizer.invoke({"evidence": evidence, "plan": plan})
```

**Pros:** easy. **Cons:** no separation of concerns; planner and writer share one context window.

### 1.2 Multi-agent (planner / researcher / writer / critic)

The dominant pattern in open deep-research scaffolds (gpt-researcher, Open Deep Research):

```
                 ┌─────────────┐
  user query ──►│   Planner   │──plan──┐
                 └─────────────┘         │
                                          ▼
                                   ┌─────────────┐
                                   │  Researcher  │◄── loop
                                   │ (web/code/  │    (tool calls)
                                   │  doc tools)  │
                                   └──────┬──────┘
                                          │ evidence_store
                                          ▼
                                   ┌─────────────┐
                                   │   Writer    │──draft──┐
                                   └─────────────┘         │
                                                            ▼
                                                   ┌─────────────┐
                                                   │   Critic    │──issues─► Researcher
                                                   └─────────────┘
```

This isolates prompt contexts: the planner never sees raw HTML; the writer only sees the evidence store; the critic only sees draft + store.

### 1.3 Workflow / state-machine (LangGraph, LlamaIndex Workflows)

For durability (long runs, resumability) model the research as an explicit graph with checkpointing (`31-AI-Workflow-Orchestration-and-Durable-Execution/`):

```python
from langgraph.graph import StateGraph, END

sg = StateGraph(ResearchState)
sg.add_node("plan", planner)
sg.add_node("research", researcher)
sg.add_node("write", writer)
sg.add_node("critic", critic)
sg.add_edge("plan", "research")
sg.add_edge("research", "write")
sg.add_conditional_edges("write",
    lambda s: "research" if s["critic_issues"] else END,
    {"research": "research", END: END})
```

Checkpointing lets a 20-minute run survive crashes and resume.

---

## 2. Prompt & context engineering

### 2.1 Context budget discipline

Deep research eats context. Strategy:

| Technique | What | Ref |
|-----------|------|-----|
| Per-step truncation | Store summaries, not raw pages | `36-Long-Context-AI/` |
| Evidence compression | One line per evidence triple | `32-Agent-Memory-Systems/` |
| Separate contexts | Planner/writer/critic isolated | §1.2 |
| Retrieved-text cache | Avoid re-reading same URL | `04-RAG/05-Caching.md` |

### 2.2 The "cite or abstain" contract

Inject this into the writer system prompt to hard-enforce fidelity:

```text
CITATION CONTRACT
- You have an `evidence_store` of verified (claim, source_url, date) triples.
- Every factual sentence MUST end with [eN] referencing a store id.
- If you cannot find a supporting triple, write "(unverified)" — do NOT guess a source.
- Quoted numbers must match the stored source exactly.
```

### 2.3 Tool-result sanitization

Retrieved pages may contain prompt injection (`18-Agent-Security-and-Trust/`). Treat all fetched content as untrusted:

```python
def sanitize(html: str) -> str:
    text = strip_tags(html)
    text = redact_emails_phones(text)
    text = truncate_middle(text, max_chars=8000)
    return text  # never let raw HTML reach the model as "instructions"
```

---

## 3. Evaluation harnesses

"Good research" is hard to grade, but several proxies are standard (`69-AI-Evaluation-and-LLM-Testing/`).

### 3.1 Metrics

| Metric | Definition | How measured |
|--------|-----------|--------------|
| **Citation accuracy** | % of claims with a valid, matching source | Gold citations or LLM-judge |
| **Source validity** | % of cited URLs that resolve & support claim | Automated fetch + NLI |
| **Coverage** | % of plan sub-questions answered | Plan vs report diff |
| **Faithfulness** | No unsupported claims (hallucination rate) | NLI vs evidence store |
| **Recency compliance** | % of time-sensitive claims using in-window sources | Date check |
| **Answer usefulness** | Human/LLM rating of report quality | Rubric judge |

### 3.2 A minimal eval loop

```python
def eval_run(report, evidence_store, plan):
    claims = extract_claims(report)
    unsupported = [c for c in claims if not has_evidence(c, evidence_store)]
    dead_links = [u for u in cited_urls(report) if not resolves(u)]
    coverage = answered_subquestions(plan, report) / len(plan.subquestions)
    return {
        "faithfulness": 1 - len(unsupported)/len(claims),
        "source_validity": 1 - len(dead_links)/len(cited_urls(report)),
        "coverage": coverage,
    }
```

### 3.3 Public benchmarks (as of 2025–2026)

| Benchmark | Targets |
|-----------|---------|
| **SimpleQA / BrowseComp** (OpenAI) | Search-enabled factuality over the live web |
| **FRAMES** | Retrieval-and-reasoning over many docs |
| **Humanity's Last Exam** | Hard multi-domain reasoning w/ search |
| **GAIA** | General AI assistants solving real tasks w/ tools |
| In-house citation-fidelity sets | Enterprise-specific source graphs |

BrowseComp specifically rewards agents that **persist** through hard-to-find answers — a direct measure of research depth.

---

## 4. Failure taxonomy

| # | Failure | Root cause | Fix |
|---|---------|-----------|-----|
| F1 | Fabricated citations | Writer invents URLs | Evidence-store-only contract |
| F2 | Premature convergence | Stops after first answer | Critic + disconfirmation loop |
| F3 | Stale facts | No recency gate | Date filter + authority weighting |
| F4 | SEO-slop reliance | No authority scoring | Source credibility model (§3.1 of 02-) |
| F5 | Context overflow | Raw pages in context | Summarize before store |
| F6 | Injection hijack | Untrusted HTML as instructions | Sanitize + isolate (§2.3) |
| F7 | Cost blowup | Unbounded loops | Budget cap + parallelism cap |
| F8 | Contradiction shipped | Conflicts unresolved | Explicit conflict section |
| F9 | Private leak | Secret in public query | Redaction before egress |

---

## 5. Advanced: reranking & hybrid retrieval

For the retrieval step itself, the same machinery as production RAG applies (`04-RAG/`):

- **Hybrid**: BM25 + dense vector, fused (RRF).
- **Reranking**: cross-encoder or LLM rerank of top-50 → top-5.
- **Query rewriting**: expand "EU AI law" → "EU AI Act GPAI obligations 2026 guidance".
- **Snippet vs full page**: fetch full page only for top-k after rerank.

```python
from rank_bm25 import BM25Okapi
# hybrid fusion (Reciprocal Rank Fusion)
def rrf(rankings: list[list[str]], k=60) -> list[str]:
    scores = {}
    for r in rankings:
        for i, doc in enumerate(r):
            scores[doc] = scores.get(doc, 0) + 1/(k + i + 1)
    return sorted(scores, key=scores.get, reverse=True)
```

---

## 6. Production reference implementation (sketch)

```python
class DeepResearchAgent:
    def __init__(self, planner, researcher, writer, critic, search, store):
        self.p = planner; self.r = researcher; self.w = writer
        self.c = critic; self.search = search; self.store = store

    def run(self, query: str, budget: int = 80):
        plan = self.p.plan(query)
        for q in plan.subquestions:
            for _ in range(budget):
                res = self.search(q.text)
                ev = self.r.extract(res)        # claim+url+date
                if self.store.add(ev): break
        draft = self.w.write(self.store, plan)
        issues = self.c.review(draft, self.store)
        while issues and budget > 0:
            self.store += self.r.research(issues)   # repair loop
            draft = self.w.write(self.store, plan)
            issues = self.c.review(draft, self.store)
            budget -= 1
        return Report(draft, self.store.bibliography())
```

This is intentionally small; real systems add parallelism, caching, human-in-the-loop checkpoints, and streaming UI.

---

## 7. Comparison: build vs buy

| Axis | Build (LangGraph/gpt-researcher) | Buy (OpenAI/Perplexity/Gemini) |
|------|----------------------------------|----------------------------------|
| Private-data grounding | ✅ full control | ⚠ limited / enterprise tiers |
| Custom tools (SQL, internal APIs) | ✅ | ❌ mostly web |
| Time to value | Weeks–months | Minutes |
| Cost predictability | You control | Per-run / subscription |
| Citation quality | Your responsibility | Vendor-guaranteed-ish |
| Compliance / data residency | ✅ self-host | ⚠ depends on vendor |

Most enterprises **buy for general web research, build for private/custom research** — a hybrid common in `05-Enterprise/`.

---

## 8. Key takeaways

1. The architecture that scales is **multi-agent with an isolated evidence store**.
2. **Citation fidelity is a system property**, enforced by contract + eval, not prompt hope.
3. **Evaluation must be built in from day one** — faithfulness + source validity + coverage.
4. **Cost and injection are the two production killers**; budget caps and sanitization are mandatory.

Next: `04-Tools-and-Frameworks.md` (full landscape + build tutorial).

---
**See also:**
- [AI in Education 2026 Frontier — The AI-Tutor Wave, the Skepticism, and the Agentic Pivot](11-AI-Applications/16-AI-Education-2026-Frontier.md)
- [08 — Agentic Services Pricing: The New Category for AI Agent Monetization](16-AI-Business-Models-Playbooks/08-Agentic-Services-Pricing.md)
- [Agentic Browser Automation & Computer Use: A 2026 Overview](26-Browser-Based-AI/46-Agentic-Browser-Automation-Computer-Use/01-Overview.md)

# Agentic Search & Deep Research — Core Topics

> The composable building blocks of a research agent: query decomposition, retrieval tooling, source credibility scoring, recency control, citation fidelity, reflection, and the evidence store. Builds directly on `04-RAG/` (retrieval) and `03-Agents/` (loops).

---

## 1. Query decomposition & planning

A research agent's first job is to turn an ambiguous ask into a **plan**: an ordered set of sub-questions, each with an intended tool and success criterion.

### 1.1 Plan representations

| Representation | Pros | Cons | Best for |
|----------------|------|------|----------|
| Flat bullet list of sub-questions | Simple, debuggable | No dependencies | Independent facts |
| DAG of dependencies | Captures "answer B needs A" | More prompt engineering | Comparative / causal |
| Tree (expand-and-prune) | Explores then focuses | Token-heavy | Open-ended exploration |

### 1.2 A planner prompt (sketch)

```text
You are a research planner. Given the user's question, output a JSON plan:
{
  "subquestions": [
    {"id": "q1", "question": "...", "tool": "web", "depends_on": []},
    {"id": "q2", "question": "...", "tool": "code", "depends_on": ["q1"]}
  ],
  "success_criteria": "..."
}
Rules:
- Each subquestion must be independently answerable.
- Prefer authoritative primary sources (gov registers, papers, filings).
- Mark dependencies so execution can parallelize.
```

### 1.3 Plan quality checklist

- [ ] Every sub-question is specific and falsifiable.
- [ ] Tools are matched to question type (web vs SQL vs code vs doc-parse).
- [ ] No two sub-questions are redundant.
- [ ] Success criteria are observable (not "be thorough").

Cross-reference: planning loops are a special case of `03-Agents/02-Planning-and-Task-Decomposition.md`.

---

## 2. Retrieval tooling

The agent must choose the **right retrieval tool per sub-question**. Tool diversity is the core differentiator vs single-shot RAG.

### 2.1 Tool taxonomy

| Tool | Use when | Example |
|------|----------|---------|
| Web search API (Tavily, Exa, SerpAPI, Bing) | Broad, current facts | "latest EU AI Act guidance 2026" |
| News/search with date filter | Recency-sensitive | "this week model releases" |
| Deep web / Scholar | Academic claims | "diffusion LM benchmark paper" |
| Internal vector DB | Private corpus (`04-RAG/`) | "our deployment runbook" |
| SQL / warehouse | Structured internal data | "support tickets last 90 days" |
| Browser automation | JS-heavy / login / paginated | "SEC filings table" |
| Code execution | Compute, charts, stats | "regress cost on quality" |
| Document parse (PDF/CSV) | Primary-source attachments | "FDA guidance PDF" |
| APIs (CRMs, product) | Live operational data | "current pricing tiers" |

### 2.2 Retrieval must produce **citable** results

Unlike chat RAG, research agents need structured search returns:

```json
{
  "query": "EU AI Act foundation model obligations 2026",
  "results": [
    {
      "title": "Commission Guidelines on GPAI obligations",
      "url": "https://digital-strategy.ec.europa.eu/...",
      "published": "2026-03-12",
      "snippet": "...providers of general-purpose AI models must...",
      "authority": 0.92
    }
  ]
}
```

The `url` and `published` fields are load-bearing: they become citations.

Cross-reference: embedding/reranking internals in `04-RAG/03-Hybrid-Search-and-Reranking.md`.

---

## 3. Source credibility & recency scoring

A core research-agent competency is **not treating all hits equally**.

### 3.1 Authority heuristics

| Signal | Higher credibility |
|--------|-------------------|
| Domain type | `.gov`, `.edu`, official registers > news > blog > forum |
| Citation count (scholar) | High for academic claims |
| Author identity | Named expert / org vs anonymous |
| Cross-confirmation | Claim repeated by ≥2 independent authoritative sources |
| Date | Recent for fast-moving topics; primary for legal/financial |

### 3.2 Recency gate (code sketch)

```python
from datetime import datetime, timedelta

def recency_ok(published: str, max_age_days: int) -> bool:
    if not published:
        return False
    pub = datetime.fromisoformat(published)
    return (datetime.utcnow() - pub) <= timedelta(days=max_age_days)

# For "this month" queries, force max_age_days=30 and down-weight older hits.
```

### 3.3 Contradiction handling

When two authoritative sources disagree, the agent should:
1. Record both with their sources.
2. Search for a meta-source resolving the conflict (e.g., a later update).
3. In the report, present the conflict explicitly rather than silently picking one.

This is the **anti-premature-convergence** behavior that separates good research agents from bad ones.

---

## 4. Citation fidelity

Citation fidelity = every substantive claim in the final report maps to a real source the agent actually read, with the correct attribution.

### 4.1 Common citation failures

| Failure | Example | Mitigation |
|---------|---------|------------|
| Fabricated URL | "Source: https://example.com/xyz" that 404s | Validate URL exists before citing |
| Misattribution | Claim from blog cited as "EU Commission" | Store (claim→source) pairs, never separate |
| Stale source | 2022 paper cited as current law | Recency gate (§3.2) |
| Hallucinated statistic | "73% of enterprises…" not in source | Quote/paraphrase only retrieved text |

### 4.2 Enforcing fidelity in the prompt

```text
You MUST only state claims that are directly supported by the
`evidence_store` entries you collected. For each claim, attach
[source_id]. If no entry supports a claim, either retrieve more
or explicitly mark it as "unverified". Never invent URLs.
```

The `evidence_store` is a structured log:

```json
{"id":"e12","claim":"EU AI Act applies to GPAI providers","source":"https://...","date":"2026-03-12","confidence":0.95}
```

Cross-reference: grounding & NLI checks in `52-AI-Hallucination-Detection-and-Mitigation/`.

---

## 5. Reflection & self-critique loops

Research quality comes from **iteration with critique**, not one pass.

### 5.1 The critic pass

After a draft, a separate "critic" agent grades it:

```text
Critique this research draft against the evidence_store.
Report:
- Claims with NO supporting evidence_store entry (list them).
- Sources cited but never retrieved (possible fabrication).
- Contradictions between sections.
- Sub-questions from the plan left unanswered.
```

If issues exceed a threshold, the agent **re-enters the research loop** (repair) rather than shipping.

### 5.2 Stopping criteria

| Criterion | Trigger |
|-----------|---------|
| Coverage | All sub-questions answered with ≥1 authoritative source |
| Confidence | Mean evidence confidence ≥ threshold (e.g., 0.8) |
| Budget | Tool-call budget or wall-clock limit reached |
| Redundancy | Last k iterations added no new distinct sources |

---

## 6. The evidence store (working memory)

During a run the agent accumulates a structured **evidence store** — the single source of truth for the final report.

```python
class Evidence:
    id: str
    claim: str
    source_url: str
    published: str | None
    authority: float
    used_in_sections: list[str]

evidence_store: list[Evidence] = []
```

At write time, the synthesizer may **only** reference `evidence_store` entries. This trivially enforces citation fidelity and makes the report auditable.

Cross-reference: memory architectures in `32-Agent-Memory-Systems/02-Working-Memory-vs-Long-Term.md`.

---

## 7. Cost & latency control (core concern)

Deep research is expensive. Core levers:

| Lever | Mechanism | Trade-off |
|-------|-----------|-----------|
| Model routing | Cheap model for planning/search, strong model for synthesis | Slight quality loss |
| Result truncation | Summarize pages before storage | Possible detail loss |
| Budget cap | Max N tool calls | May under-research |
| Cache search | Reuse prior searches per session | Staleness |
| Parallel sub-questions | Fan out independent branches | Higher peak cost |

Cross-reference: `41-AI-Cost-Optimization-and-Enterprise-ROI/` and `30-Small-Language-Models/`.

---

## 8. Security & trust

Research agents that browse the live web inherit web risks:
- **Prompt injection** from malicious pages (`18-Agent-Security-and-Trust/`).
- **Data exfiltration** if private context leaks into a public query.
- **Poisoned sources** that steer conclusions.

Mitigations: isolate the browse context, sanitize retrieved HTML, treat retrieved text as untrusted, and require human review for high-stakes reports.

---

## 9. Core-topics checklist for builders

- [ ] Planner emits a structured, dependency-aware plan.
- [ ] Tool router matches sub-question → appropriate tool.
- [ ] Search returns **citable** structured results (url + date + authority).
- [ ] Recency gate enforced for time-sensitive queries.
- [ ] Contradictions surfaced, not silently resolved.
- [ ] Evidence store is the only citation source.
- [ ] Critic pass runs before finalization.
- [ ] Cost budget enforced.
- [ ] Retrieved content treated as untrusted (injection-safe).

Next: `03-Technical-Deep-Dive.md` for architectures and evaluation harnesses.

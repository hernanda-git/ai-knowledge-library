# Agentic Search & Deep Research

> A comprehensive guide to agentic search and "deep research" systems — AI agents that plan, retrieve, synthesize, and cite across the open web, private corpora, and tools to produce analyst-grade reports. Covers the architectural shift from single-shot retrieval-augmented generation (RAG) to multi-step, tool-using research loops, the leading 2025–2026 products (OpenAI Deep Research, Perplexity, Gemini Deep Research, Genspark, Manus, You.com), and production guidance for building your own.

> **Library cross-references:** Single-pass retrieval is covered in `04-RAG/`. Agent loop and orchestration primitives live in `03-Agents/` and `31-AI-Workflow-Orchestration-and-Durable-Execution/`. Long-context handling is in `36-Long-Context-AI/`, memory in `32-Agent-Memory-Systems/`, evaluation in `69-AI-Evaluation-and-LLM-Testing/`, and hallucination control in `52-AI-Hallucination-Detection-and-Mitigation/`. Cost/RoI trade-offs are in `41-AI-Cost-Optimization-and-Enterprise-ROI/`.

---

## 1. Why this matters in 2026

For two years the dominant pattern for grounding an LLM in external knowledge was **single-shot RAG**: embed a query, fetch the top-k chunks, stuff them into a prompt, generate once. It works for question answering over a fixed corpus, but it breaks on the tasks knowledge workers actually have:

- "What changed in EU AI liability law between January and June 2026, and how does it affect our deployment roadmap?"
- "Benchmark every managed LLM API on cost-per-million-tokens AND coding-benchmark quality, then recommend one."
- "Summarize the competitive landscape for AI voice agents and identify the three most credible startups."

These require **planning a research path, issuing many heterogeneous searches, reading dozens of sources, reconciling contradictions, and citing evidence** — a loop, not a lookup.

In 2025–2026 a distinct product category matured: the **deep research agent**. These are not chatbots with a search bar; they run autonomous multi-minute research loops that emit structured reports with inline citations. They are now a primary differentiator for frontier labs and a fast-growing enterprise workload.

| Force | Effect |
|-------|--------|
| Knowledge work is bottlenecked on synthesis | One analyst can't read 200 sources; an agent can in minutes |
| Web + private data is fragmented | Agents can fan out across many tools/endpoints |
| Citation & provenance demands | Regulated and enterprise users require traceable claims (see `21-AI-Regulation-Antitrust/`, `40-AI-Data-Sovereignty-and-Privacy/`) |
| Frontier labs competing on "research" UX | OpenAI, Google, Perplexity, xAI shipping dedicated products |
| Cost of long agentic runs dropping | Smaller models + better planning make deep research affordable (see `30-Small-Language-Models/`, `41-`) |

---

## 2. Definition: what "agentic search" means

**Agentic search** is retrieval conducted by an autonomous loop that:

1. **Decomposes** a complex query into sub-questions (a plan).
2. **Selects tools** per sub-question (web search, SQL, PDF parse, internal wiki, code execution, APIs).
3. **Iterates**: observe results, decide what's missing, re-query, follow links.
4. **Synthesizes** a final answer with inline citations and an explicit confidence / uncertainty.

**Deep research** is the flagship instantiation: a long-horizon version of agentic search that produces a multi-section written report (often 1,000–5,000 words) with a source list, typically taking 5–30 minutes and consuming many tool calls.

| Dimension | Single-shot RAG | Agentic Search | Deep Research |
|-----------|----------------|----------------|---------------|
| Queries issued | 1 (embed) | Many (adaptive) | Dozens–hundreds |
| Planning | None | Light (sub-questions) | Explicit multi-step plan |
| Tool diversity | Vector DB only | Web + tools | Web + tools + code + docs |
| Latency | Seconds | Seconds–minutes | Minutes–30 min |
| Output | Short answer | Answer + sources | Full report + citations |
| Failure mode | Missing context | Premature stop | Hallucinated synthesis |

---

## 3. The market landscape (2025–2026)

### 3.1 Flagship products

| Product | Owner | Distinguishing trait | Notes |
|---------|-------|---------------------|-------|
| **Deep Research** | OpenAI | Operator-style research agent; long multi-minute runs; strong citations; tied to ChatGPT Pro/Plus | One of the most-cited "wow" demos of 2025 |
| **Gemini Deep Research** | Google | Native to Gemini; integrates with Workspace data; often faster | Leverages Google's web index |
| **Perplexity** | Perplexity AI | Conversational answer engine with "Pro Search" / "Deep Research" modes; real-time web | Pioneered cited, sourced answers at scale |
| **Grok / xAI** | xAI | Real-time X (Twitter) data access | Differentiated by live social signal |
| **Genspark** | Genspark (ex-Baidu) | "Agent engine" with Sparkpages synthesizing multiple sources | Multi-agent synthesis approach |
| **Manus** | Monica | General autonomous agent; can browse, code, produce deliverables | Generalist agent, research is one mode |
| **You.com (YouPro)** | You.com | AI search with research modes and app ecosystem | |
| **Claude (Research / artifacts)** | Anthropic | Web search + Analysis tool + artifacts for structured output | |
| **Microsoft Copilot / Deep Research** | Microsoft | Embedded in M365; enterprise data grounding | |

### 3.2 Open-source & self-hostable

| Project | What it is |
|---------|-----------|
| **LangChain / LangGraph** | Graph-based agent loops; `create_react_agent`, `ResearchAssistant` patterns |
| **LlamaIndex** | `Workflow` events; `DeepResearchAgent`, `AgenticSearch` recipes; strong RAG heritage (`04-RAG/`) |
| **Open Deep Research (Hugging Face)** | Reference implementation of a research agent (plan → search → write) |
| **smolagents (Hugging Face)** | Minimal CodeAgent; good for custom research loops |
| **gpt-researcher** | Popular open deep-research scaffold; multi-agent (planner + researcher + writer) |
| **browser-use / Playwright agents** | Real browser navigation for hard-to-index sources |
| **Tavily / Exa / SerpAPI** | Search APIs purpose-built for LLM agents (clean, cited results) |

---

## 4. Anatomy of a deep research run

A typical deep-research trajectory:

```
[User] "Compare EU vs US AI regulation trajectories for foundation models, 2024-2026"
   │
   ▼
[Planner] breaks into sub-questions:
   - What does the EU AI Act require of FMs? (transposition status 2026)
   - What is the US federal posture (executive orders, state laws)?
   - Where do they conflict / converge?
   - What do enterprises actually do to comply?
   │
   ▼
[Researcher loop] (repeats ~N times):
   - Web search sub-question → read top results
   - Identify a primary source (official register) → fetch & parse
   - Find a contradiction → issue a targeted clarifying search
   - Record (claim, source_url, date, confidence) into working memory
   │
   ▼
[Synthesizer] writes sections, attaching citations from the evidence store
   │
   ▼
[Critic] checks: every major claim has a citation? any stale/conflicting sources?
   │
   ▼
[Final] 2,400-word report + 38-source bibliography
```

Average runs consume 30–120 tool calls and read 20–80 documents.

---

## 5. Why it's harder than it looks

- **Source quality**: the web is full of SEO slop, recycled press releases, and dated pages. Agents must rank by authority and recency.
- **Recency / freshness**: models are pre-trained on stale data; live retrieval is mandatory for "this month" queries (see `09-Papers/` for freshness-eval work).
- **Citation fidelity**: the generator must not invent a URL or misattribute a claim. This is a measurable failure mode (`52-`).
- **Premature convergence**: agents stop after the first plausible answer instead of seeking disconfirming evidence.
- **Cost / latency**: long loops are expensive; naive implementations blow the budget (`41-`).
- **Eval difficulty**: "good research" is subjective; automated graders are weak proxies (`69-`).

---

## 6. Where this category sits in the library

Agentic search is the **convergence point** of several existing categories:

- `04-RAG/` — the retrieval substrate (embeddings, reranking, hybrid search).
- `03-Agents/` — planning, tool-use, reflection loops.
- `31-AI-Workflow-Orchestration-and-Durable-Execution/` — long-running, resumable research jobs.
- `36-Long-Context-AI/` — fitting dozens of sources into context; context compression.
- `32-Agent-Memory-Systems/` — persisting the evidence store across a run.
- `69-AI-Evaluation-and-LLM-Testing/` — grading report quality and citation accuracy.
- `52-AI-Hallucination-Detection-and-Mitigation/` — grounding claims in retrieved evidence.
- `28-AI-Agent-Commerce-and-A2A-Payments/` — future: agents that pay for premium data sources.

This category does **not** re-derive those fundamentals; it specializes them for the research-agent workload.

---

## 7. What you'll find in this category

- `01-Overview.md` (this file) — definitions, market, anatomy.
- `02-Core-Topics.md` — planning, retrieval tooling, citation, reflection, recency control.
- `03-Technical-Deep-Dive.md` — architectures, prompt/agent patterns, evaluation harnesses, failure taxonomies.
- `04-Tools-and-Frameworks.md` — product and OSS landscape, build-vs-buy, reference implementation.
- `05-Future-Outlook.md` — trajectories, risks, and the road to agent-to-agent research.

---

## 8. Representative use cases

| Use case | Why agentic (not RAG) | Typical depth |
|----------|------------------------|---------------|
| **Market / competitive intel** | Requires triangulating many sources + disconfirmation | 20–60 sources |
| **Regulatory compliance mapping** | Needs primary sources + recency + citation | 30–80 sources |
| **Investment / DD research** | Conflicting filings, must reconcile | 40–100 sources |
| **Literature review** | Hundreds of papers, dedupe + synthesize | 50–200 sources |
| **Incident / threat post-mortem** | Multi-source timeline reconstruction | 10–40 sources |
| **"State of the art" brief** | Fast-moving; recency gate mandatory | 30–90 sources |

Each of these fails under single-shot RAG because the answer isn't in any one chunk — it emerges from **reconciling** many.

---

## 9. Architecture at a glance

```
            ┌─────────────────────────────────────────┐
 query ───►│  Planner (decompose → dependency DAG)    │
            └───────────────────┬─────────────────────┘
                                │ sub-questions
                                ▼
            ┌─────────────────────────────────────────┐
            │  Researcher loop (tool router)           │
            │   web · sql · code · docs · browser     │
            │        │                                  │
            │        ▼  append                         │
            │   Evidence Store (claim,url,date,conf)  │
            └───────────────────┬─────────────────────┘
                                │
                                ▼
            ┌─────────────────────────────────────────┐
            │  Writer (cited draft) → Critic (verify)  │
            │        │                │                  │
            │        │   issues       └──► re-research  │
            │        ▼                                  │
            │   Final report + bibliography             │
            └─────────────────────────────────────────┘
```

The **evidence store** is the linchpin: it decouples discovery (messy, many tools) from synthesis (clean, cited), and makes the report auditable. Detailed patterns in `03-Technical-Deep-Dive.md`.

---

## 10. Agentic search vs adjacent concepts

| Concept | Relationship to agentic search |
|---------|-------------------------------|
| **RAG (`04-`)** | The retrieval substrate; agentic search adds planning + iteration on top |
| **Agents (`03-`)** | The loop machinery; research agents are a domain of agents |
| **Workflow orchestration (`31-`)** | Makes long research runs durable/resumable |
| **Long-context (`36-`)** | Lets more sources fit; compression still needed at scale |
| **Memory (`32-`)** | The evidence store is a form of working memory |
| **Evaluation (`69-`)** | Grades report quality + citation accuracy |
| **Hallucination mitigation (`52-`)** | Grounding claims in the evidence store |

---

## 11. Adoption readiness checklist

- [ ] Identify the highest-value research task (start narrow).
- [ ] Decide build vs buy (private data → build; general web → buy).
- [ ] Stand up a citation-ready search API (Tavily/Exa).
- [ ] Implement the evidence store as the only citation source.
- [ ] Add a critic/repair loop before any production use.
- [ ] Define eval metrics (faithfulness, source validity, coverage).
- [ ] Set cost + injection safeguards.

---

## 12. Quick glossary

| Term | Meaning |
|------|---------|
| **Deep Research** | Long-horizon research agent producing a cited report |
| **Agentic Search** | Any search conducted by an adaptive multi-step agent |
| **Source graph** | The set of (claim, source, date, confidence) triples an agent accumulates |
| **Recency gate** | A filter requiring sources newer than a date |
| **Citation fidelity** | Whether every claim maps to a real, correct source |
| **Premature convergence** | Stopping research before disconfirming evidence is sought |
| **Synthesis critic** | A verification pass that checks citations and contradictions |

---
**See also:**
- [AI in Education 2026 Frontier — The AI-Tutor Wave, the Skepticism, and the Agentic Pivot](11-AI-Applications/16-AI-Education-2026-Frontier.md)
- [08 — Agentic Services Pricing: The New Category for AI Agent Monetization](16-AI-Business-Models-Playbooks/08-Agentic-Services-Pricing.md)
- [Agentic Browser Automation & Computer Use: A 2026 Overview](26-Browser-Based-AI/46-Agentic-Browser-Automation-Computer-Use/01-Overview.md)

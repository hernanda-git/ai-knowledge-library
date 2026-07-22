# Agentic Search & Deep Research — Future Outlook

> Where research agents are heading through 2026 and beyond: from web research to agent-to-agent investigation, vertical specialization, and the open problems that will define the category. Companion to `01-Overview.md` and `03-Technical-Deep-Dive.md`.

---

## 1. Trajectory (2026 → 2027)

| Phase | Capability | Status in 2026 |
|-------|-----------|-----------------|
| 1. Web Q&A with citations | Cited answers over public web | Mature (Perplexity, Gemini) |
| 2. Deep reports | Multi-section autonomous reports | Shipping (OpenAI, Gemini Deep Research) |
| 3. Tool-native research | SQL + code + internal docs + web | Emerging (enterprise builds) |
| 4. Agent-to-agent research | Agents commission other agents for sub-investigations | Early (`28-AI-Agent-Commerce-and-A2A-Payments/`) |
| 5. Continuous research | Standing agents that re-research when the world changes | Research/beta |

The center of gravity is moving **from "answer my question" to "monitor my world and tell me when it changes."**

---

## 2. Verticalization

Generic research agents will be commoditized; value moves to **domain-specialized** agents with curated source graphs:

- **Legal research** — statutes, case law, dockets (`49-AI-for-Legal-and-LegalTech/`).
- **Competitive/market intel** — filings, news, pricing (`24-AI-Sales-and-Marketing/`).
- **Scientific literature** — papers, datasets, reproducibility (`42-AI-for-Science-and-Drug-Discovery/`).
- **Security threat research** — CVEs, advisories (`65-AI-for-Cybersecurity/`).
- **Financial research** — filings, earnings, macro (`11-AI-Applications/67-AI-in-Finance-and-Financial-Services/`).

Each vertical needs a **trusted source whitelist** and domain evaluators — the generic web-credibility heuristic is insufficient.

---

## 3. Agent-to-agent research & the data marketplace

As agents gain wallets (`28-`), research will fragment:

```
Enterprise agent ──pays──► Specialist "patent-research agent"
                        └──pays──► "regulatory-monitoring agent"
```

Agents will **commission** other agents for sub-investigations and **pay** for premium/contracted data sources. This turns research from a single loop into a **market of specialized research services**. Open questions: trust attestation of agent output, dispute resolution, and provenance of paid sources.

---

## 4. Continuous / standing research

The next UX shift: instead of one-shot queries, users define **research mandates** ("track EU AI Act enforcement actions") and a durable agent (`31-AI-Workflow-Orchestration-and-Durable-Execution/`) polls sources on a schedule, diffs against prior state, and pushes alerts only on meaningful change. This collapses the gap between "search" and "monitoring."

---

## 5. Open problems (where the research is)

| Problem | Why hard | Promising directions |
|---------|----------|---------------------|
| **Faithful synthesis at scale** | Long reports amplify any single hallucination | Evidence-store contracts; NLI verifiers (`52-`) |
| **Disconfirmation bias** | Agents confirm, rarely seek refutation | Explicit "devil's advocate" critic node |
| **Source authority in real time** | Authority is contextual & shifting | Learned credibility models + provenance graphs |
| **Eval that correlates with humans** | LLM-judges drift | Rubric-based, multi-judge, adversarial sets (`69-`) |
| **Cost-quality Pareto** | Deep research is expensive | Smaller planner models (`30-`), result summarization |
| **Injection resistance** | Web is adversarial (`18-`) | Strict sanitization, isolated browse contexts |
| **Multimodal research** | Reports need charts, figures, video (`50-Multimodal-AI/`) | Vision-grounded evidence, figure citation |

---

## 6. Risks & governance

- **Misinformation amplification**: a confident, well-cited *wrong* report is more dangerous than an unhelpful one. Cite-accuracy ≠ truth.
- **Source manipulation**: actors can poison high-authority-looking pages to steer agent conclusions. Provenance/attestation (`40-`) matters.
- **Bias via source selection**: default web indexes carry commercial/geographic bias; regulated use needs auditable source policy.
- **Over-reliance**: humans stop checking cited reports. Keep human-in-the-loop for high-stakes outputs.

---

## 7. What to watch (signals)

- Frontier labs shipping **longer-horizon, tool-native** research modes.
- Emergence of **research-agent benchmarks** beyond BrowseComp/GAIA.
- **Standards** for citation/attestation of agent output (provenance frameworks).
- **Pricing** shift from per-query to per-research-credit or subscription tiers.
- Convergence of research agents with **agentic coding** (`33-AI-Native-Software-Development/`) — agents that research *and* implement.

---

## 8. Strategic recommendations

1. **For enterprises**: start with bought web research for open questions; build only for private/custom-source needs.
2. **For builders**: invest in the **evidence store + critic loop** — that's the durable moat, not the LLM.
3. **For evaluators**: ship citation-fidelity + source-validity checks from day one.
4. **For researchers**: disconfirmation, multimodal evidence, and injection-resistant retrieval are the open frontiers.

---

## 9. Deployment patterns emerging in 2026

| Pattern | Description | Maturity |
|---------|-------------|----------|
| **On-demand research** | User prompts, agent returns a report | Mainstream |
| **Scheduled mandates** | Standing agent re-runs on a cadence, alerts on change | Beta |
| **Event-triggered research** | Fires when a watched source updates (webhook/RSS) | Early |
| **Research-as-a-service** | Internal agent others call via API | Early |
| **Agent-to-agent subcontracting** | Agent pays specialist agents (`28-`) | Experimental |

The trend line is clear: from **reactive** (answer my question now) to **proactive + composable** (watch my world; delegate sub-investigations).

---

## 10. The economics

Deep research is a **token- and tool-call-intensive** workload. As models get cheaper (`30-Small-Language-Models/`) and planning improves, the cost-per-report curve drops sharply through 2026, unlocking:

- SMB access to analyst-grade research previously affordable only to enterprises.
- "Research-on-every-query" as a default rather than a premium mode.
- Continuous monitoring that was previously too expensive to run broadly.

Cost levers remain central (`41-AI-Cost-Optimization-and-Enterprise-ROI/`): model routing, result summarization, budget caps, and caching of repeated searches.

---

## 11. Standardization & interoperability

As the category matures, expect:
- **Citation/attestation formats** so reports are machine-verifiable (provenance frameworks, `40-`).
- **Source-graph exchange** so one agent's evidence store can be audited by another.
- **Benchmark suites** beyond BrowseComp/GAIA for vertical research quality.
- **Agent-to-agent protocols** for commissioning and paying for research (`28-`).

---

## 12. Closing

Agentic search and deep research are the **synthesis layer** of the AI stack — the point where retrieval (`04-`), agents (`03-`), memory (`32-`), evaluation (`69-`), and long-context (`36-`) converge into something a knowledge worker can trust. The category will mature from "smart search" into "standing, paying, specialized research collaborators." This library's `72-` category will track that arc.

> See also: `03-Agents/`, `04-RAG/`, `31-AI-Workflow-Orchestration-and-Durable-Execution/`, `69-AI-Evaluation-and-LLM-Testing/`, `28-AI-Agent-Commerce-and-A2A-Payments/`.

---
**See also:**
- [AI in Education 2026 Frontier — The AI-Tutor Wave, the Skepticism, and the Agentic Pivot](11-AI-Applications/16-AI-Education-2026-Frontier.md)
- [08 — Agentic Services Pricing: The New Category for AI Agent Monetization](16-AI-Business-Models-Playbooks/08-Agentic-Services-Pricing.md)
- [Agentic Browser Automation & Computer Use: A 2026 Overview](26-Browser-Based-AI/46-Agentic-Browser-Automation-Computer-Use/01-Overview.md)

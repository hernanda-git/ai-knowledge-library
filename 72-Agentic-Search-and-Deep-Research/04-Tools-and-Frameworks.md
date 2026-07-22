# Agentic Search & Deep Research — Tools & Frameworks

> The practical landscape: flagship products, open-source scaffolds, search/retrieval APIs, and a step-by-step tutorial to build a minimal deep-research agent. Companion to `03-Technical-Deep-Dive.md`.

---

## 1. Flagship commercial products (2025–2026)

| Product | Owner | Best at | Access |
|---------|-------|---------|--------|
| **Deep Research** | OpenAI | Deep multi-section reports, strong citations | ChatGPT Pro/Plus/Team |
| **Gemini Deep Research** | Google | Fast research, Workspace integration | Gemini Advanced |
| **Perplexity (Pro Search / Deep Research)** | Perplexity AI | Conversational cited answers, real-time | Web / Pro |
| **Grok** | xAI | Live X/social signal, real-time | xAI |
| **Genspark** | Genspark | Multi-agent "Sparkpage" synthesis | Web |
| **Manus** | Monica | General autonomous agent (research is one mode) | Waitlist |
| **You.com (YouPro)** | You.com | AI search w/ research modes + apps | Web / Pro |
| **Copilot Deep Research** | Microsoft | Enterprise M365 data grounding | M365 Copilot |
| **Claude** (web search + Analysis) | Anthropic | Structured output via artifacts + analysis | Claude |

Selection heuristic:
- Need **private/enterprise data** → Copilot, or build.
- Need **fast cited facts** → Perplexity.
- Need **long-form deep report** → OpenAI/Gemini Deep Research.
- Need **live social sentiment** → Grok.

---

## 2. Open-source scaffolds

| Project | Stack | Notes |
|---------|-------|-------|
| **gpt-researcher** | Python, LangChain | Multi-agent (planner/researcher/writer); very popular; good starting point |
| **Open Deep Research** (HF) | Python | Reference impl; plan→search→write; supports many LLMs |
| **smolagents** | Hugging Face | Minimal CodeAgent; easy custom loops |
| **LangGraph** | LangChain | Graph orchestration + checkpointing (`31-`) |
| **LlamaIndex Workflows** | LlamaIndex | Event-driven agent workflows; RAG-native (`04-`) |
| **browser-use** | Python | Browser automation for hard sources |
| **CrewAI** | Python | Role-based multi-agent crews |

---

## 3. Search & retrieval APIs (the agent's senses)

| API | Strength | Pricing model |
|-----|----------|---------------|
| **Tavily** | LLM-optimized search; clean snippets; citation-ready | Credits |
| **Exa** | Neural / semantic search; good for "find similar" | Credits |
| **SerpAPI** | Google results structured | Monthly |
| **Brave Search API** | Independent index, privacy | Credits |
| **Bing Web Search** | Broad coverage | Azure |
| **Metaphor / Exa** | Link-prediction search | Credits |
| **Arxiv / Semantic Scholar API** | Academic | Free |
| **NewsAPI / GDELT** | Recent news | Freemium |

For research agents, **Tavily and Exa** are the common defaults because they return clean, citation-ready JSON (url + content + score) rather than raw SERP HTML.

---

## 4. Build tutorial: a minimal deep-research agent (Python)

Dependencies: `pip install openai tavily-python langgraph`.

### 4.1 Search tool wrapper (citable output)

```python
from tavily import TavilyClient
tavily = TavilyClient(api_key="TVLY-...")

def web_search(query: str, max_age_days: int | None = None) -> list[dict]:
    resp = tavily.search(query=query, max_results=5,
                         include_raw_content=True)
    out = []
    for r in resp["results"]:
        out.append({
            "title": r["title"],
            "url": r["url"],
            "published": r.get("published_date"),
            "content": r["content"][:4000],
        })
    return out
```

### 4.2 Evidence store

```python
import uuid, datetime

class EvidenceStore:
    def __init__(self): self.items = {}
    def add(self, claim: str, url: str, published: str | None, authority: float):
        eid = f"e{len(self.items)+1}"
        self.items[eid] = {
            "id": eid, "claim": claim, "url": url,
            "published": published, "authority": authority,
            "added": datetime.datetime.utcnow().isoformat()
        }
        return eid
    def bibliography(self): return list(self.items.values())
```

### 4.3 Planner

```python
from openai import OpenAI
client = OpenAI()

def plan(query: str) -> list[str]:
    sys = ("You are a research planner. Break the user's question into "
           "3-6 specific, independent sub-questions. Return one per line.")
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":query}])
    return [l.strip("- ").strip() for l in r.choices[0].message.content.splitlines() if l.strip()]
```

### 4.4 Researcher + Writer (citation contract)

```python
def research(subq: str, store: EvidenceStore):
    for hit in web_search(subq):
        # extract a claim from the page via the LLM
        claim = extract_claim(subq, hit["content"])
        if claim:
            store.add(claim, hit["url"], hit["published"], authority(hit["url"]))

def write_report(query: str, store: EvidenceStore) -> str:
    ev = "\n".join(f"[{e['id']}] {e['claim']} (src: {e['url']})"
                    for e in store.bibliography())
    sys = ("Write a research report. Every factual sentence MUST end with "
           "[eN] referencing an evidence id below. Do not invent sources.")
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":f"QUESTION: {query}\n\nEVIDENCE:\n{ev}"}])
    return r.choices[0].message.content
```

### 4.5 Wire it together

```python
store = EvidenceStore()
subs = plan("Compare EU vs US AI regulation for foundation models in 2026")
for s in subs:
    research(s, store)
report = write_report("Compare EU vs US AI regulation for foundation models in 2026", store)
print(report)
print("\n=== BIBLIOGRAPHY ===")
for e in store.bibliography():
    print(f"{e['id']}: {e['url']} ({e['published']})")
```

This ~80-line script reproduces the core loop: **plan → citeable search → evidence store → cited report**. Production systems add the critic loop, parallelism, budget caps, and sanitization from `03-Technical-Deep-Dive.md`.

---

## 5. Frameworks: when to use what

| Need | Pick |
|------|------|
| Quick prototype | gpt-researcher or smolagents |
| Durable, resumable long runs | LangGraph / LlamaIndex Workflows (`31-`) |
| Heavy private RAG + research | LlamaIndex (RAG-native) (`04-`) |
| Browser-only sources | browser-use + Playwright |
| Enterprise, governed | Build on LangGraph + internal vector DB + HITL |

---

## 6. Tooling checklist

- [ ] Search API returns **citation-ready** JSON (url + date + content).
- [ ] Evidence store is the single citation source.
- [ ] LLM planner emits dependency-aware sub-questions.
- [ ] Critic loop implemented (repair pass).
- [ ] Budget cap + sanitization in place.
- [ ] Eval harness measures faithfulness + source validity (`69-`).

---

## 7. Vendor risk notes

- **Rate limits / cost**: deep research burns many tool calls; set org budgets.
- **Data residency**: sending internal queries to a public research API may violate `40-AI-Data-Sovereignty-and-Privacy/`.
- **Lock-in**: reports may not be exportable in structured form; prefer APIs with JSON/bibliography export.
- **Hallucination**: even flagship products sometimes mis-cite — always spot-check (`52-`).

Next: `05-Future-Outlook.md`.

---
**See also:**
- [AI in Education 2026 Frontier — The AI-Tutor Wave, the Skepticism, and the Agentic Pivot](11-AI-Applications/16-AI-Education-2026-Frontier.md)
- [08 — Agentic Services Pricing: The New Category for AI Agent Monetization](16-AI-Business-Models-Playbooks/08-Agentic-Services-Pricing.md)
- [Agentic Browser Automation & Computer Use: A 2026 Overview](26-Browser-Based-AI/46-Agentic-Browser-Automation-Computer-Use/01-Overview.md)

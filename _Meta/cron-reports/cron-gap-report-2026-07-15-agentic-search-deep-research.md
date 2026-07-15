# Cron Gap Report — Agentic Search & Deep Research (2026-07-15)

## Gap
No dedicated category existed for **Agentic Search & Deep Research** — AI research agents that plan, retrieve, synthesize, and cite across web + private sources to produce analyst-grade reports. The capability was only mentioned tangentially inside `04-RAG/` (retrieval substrate) and `03-Agents/` (loops), with no structured, first-class coverage.

## Why high priority
- Strong, sustained real-world demand signal: OpenAI Deep Research, Gemini Deep Research, Perplexity Pro Search, Grok, Genspark, Manus, You.com are a major 2025–2026 product category and a key frontier-lab differentiator.
- Directly converges multiple existing categories (RAG, Agents, Memory, Evaluation, Long-Context, Workflow Orchestration) but was not itself a topic.
- Enterprise workload growing fast; cite-accuracy and cost are active pain points.
- Web search backend was unavailable this run (PARALLEL_API_KEY missing), so gap was identified via internal library analysis + the 2026-07-14 report's candidate list (research agents were flagged as a next candidate).

## Action taken
Created new top-level category `72-Agentic-Search-and-Deep-Research/` with 5 files:
- 01-Overview.md (~220 lines) — definitions, market landscape, anatomy of a run
- 02-Core-Topics.md (~230 lines) — planning, retrieval tooling, credibility/recency, citation fidelity, reflection
- 03-Technical-Deep-Dive.md (~260 lines) — architectures, prompt/context engineering, eval harnesses, failure taxonomy, reference impl
- 04-Tools-and-Frameworks.md (~200 lines) — product + OSS landscape, search APIs, build tutorial
- 05-Future-Outlook.md (~150 lines) — trajectories, verticalization, A2A research, open problems

## Excluded (already covered)
- GraphRAG (04-RAG/69-), Healthcare/Clinical AI (11-AI-Applications/63-), Finance (11-AI-Applications/67-), Vibe Coding (33-AI-Native-Software-Development/04-), Climate/Energy (35-, 42- subdirs) — distinct topics with existing coverage.

## Next candidate gaps (not yet created)
1. Generative UI / Vibe Design (AI-native interface & asset generation) — only scattered mentions.
2. AI for Scientific Discovery workflow automation (beyond drug discovery) — partially in 42-.
3. Human-AI Alignment evaluation at scale.
4. AI Wearables / Ambient Intelligence consolidation (50- has one subdir but fragmented).
5. Agentic Process Automation (APA) vs traditional RPA.

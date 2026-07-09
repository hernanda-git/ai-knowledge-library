# GraphRAG and Knowledge-Graph Retrieval

> A comprehensive reference on graph-augmented retrieval-augmented generation (GraphRAG): knowledge-graph construction from unstructured text, entity/relationship extraction, graph-based indexing, global and local query algorithms, and production patterns. GraphRAG extends classic vector RAG (see `../04-RAG/`) with structured, relational context that survives multi-hop reasoning and "global" questions over an entire corpus.

---

## Table of Contents

1. [What Is GraphRAG?](#what-is-graphrag)
2. [Why Graphs, Why Now](#why-graphs-why-now)
3. [The Core Problem GraphRAG Solves](#the-core-problem-graphrag-solves)
4. [GraphRAG vs. Classic Vector RAG](#graphrag-vs-classic-vector-rag)
5. [The GraphRAG Pipeline at a Glance](#the-graphrag-pipeline-at-a-glance)
6. [Global vs. Local Queries](#global-vs-local-queries)
7. [Ecosystem Map (2024–2026)](#ecosystem-map-2024-2026)
8. [When NOT to Use GraphRAG](#when-not-to-use-graphrag)
9. [Maturity and Adoption Signals](#maturity-and-adoption-signals)
10. [Relationship to Other Library Categories](#relationship-to-other-library-categories)
11. [References](#references)

---

## What Is GraphRAG?

**GraphRAG** is a retrieval-augmented generation pattern in which an unstructured (or semi-structured) corpus is first distilled into a **knowledge graph** — a set of entities (nodes) and the relationships (edges) between them — and then both the graph and a derived textual index are used to answer questions. The term was popularized by Microsoft Research's open-source `graphrag` library (2024), but the underlying idea (graph-augmented retrieval) predates it and is implemented independently by Neo4j, AWS, and several research projects.

Two retrieval modes are central:

- **Local queries** — answer a question about specific entities using the extracted graph subgraph plus source text.
- **Global queries** — answer a question about *the whole corpus* (e.g. "What are the main themes?") using a two-stage *community summarization* (Leiden graph clustering + hierarchical summarization). This is the capability classic vector RAG fundamentally cannot do well.

---

## Why Graphs, Why Now

Classical RAG (see `../04-RAG/01-RAG-Architectures.md`) indexes documents as dense vector embeddings and retrieves the *k* nearest neighbors of a query. This works for **point lookups** ("What does the contract say about termination?") but degrades on:

1. **Multi-hop reasoning** — answers that require chaining facts across documents ("Who supplies the vendor that acquired the company that makes X?").
2. **Global / summarizing questions** — "Summarize the entire corpus," which vector top-k cannot represent.
3. **Sparse-but-important entities** — a rare but central entity may never be the nearest neighbor yet is pivotal to the answer.

Graphs encode **structure** that vectors discard. A knowledge graph makes relationships explicit, queryable, and navigable, and lets an LLM reason over a small, relevant subgraph rather than a sea of token-heavy chunks.

---

## The Core Problem GraphRAG Solves

Consider 10,000 internal PDFs. A user asks: *"Across all our research, what are the most discussed risks and which teams own them?"*

- **Vector RAG**: retrieves the 20 most similar chunks. Likely misses 90% of the relevant mentions; cannot aggregate.
- **GraphRAG**: builds entities (Risk, Team, Project), edges (owns, mentioned_in), clusters into communities, summarizes each community, then folds community summaries into a global answer. Returns a structured, attributable synthesis.

This is the single most-cited reason teams adopt GraphRAG in 2025–2026: **global sense-making over private corpora** (legal, biomedical, research, enterprise knowledge).

---

## GraphRAG vs. Classic Vector RAG

| Dimension | Vector RAG | GraphRAG |
|---|---|---|
| Index unit | Chunks / passages | Entities + relationships + communities |
| Best at | Point lookups, similarity | Multi-hop, global summary, connection discovery |
| Global query | Poor (top-k can't summarize) | Strong (community summarization) |
| Indexing cost | Low–medium | High (LLM extraction + graph build) |
| Query cost | Low | Medium (local) / High (global) |
| Explainability | Medium (chunk citations) | High (explicit entity/edge trails) |
| Refresh | Re-embed chunks | Incremental graph merge |
| Storage | Vector DB | Graph DB + text store (+ vector optional) |

GraphRAG is **not a replacement** for vector RAG — production systems increasingly **hybridize** both (see `04-Tools-and-Frameworks.md`).

---

## The GraphRAG Pipeline at a Glance

1. **Source ingestion** — text chunks from documents.
2. **Entity & relationship extraction** — an LLM extracts typed entities and relationships, emitting a graph (Microsoft GraphRAG uses a tuned prompt with examples).
3. **Graph construction** — nodes/edges loaded into a graph store; element summaries attached.
4. **Community detection** — the graph is partitioned (Microsoft uses the **Leiden** algorithm) into hierarchically nested communities.
5. **Community summarization** — each community's nodes/edges are summarized into a "community report" (this is the global index).
6. **Index persistence** — graph + community reports + text units stored.
7. **Query** — local (graph traversal + text) or global (map-reduce over community reports).

```
documents --> chunks --> [LLM extract] --> graph
                                    |
                                    v
                            [Leiden cluster] --> community reports (global index)
                                    |
                                    v
            query ---> local (subgraph+text) | global (community fold-in)
```

---

## Global vs. Local Queries

**Local query** walks the graph from entities mentioned in the question, pulls the relevant subgraph + source text units, and generates an answer with citations. Good for "Tell me about entity X and its relationships."

**Global query** uses a *map-reduce* pattern over the **community reports**:
- *Map*: each relevant community report is used to answer the query independently.
- *Reduce*: answers are shuffled (to avoid length bias) and folded into a final composite.

A *drift* reduction step can re-run the reduce over concatenated partial answers to improve coherence. This is what makes "summarize the corpus" feasible.

---

## Ecosystem Map (2024–2026)

| Project / Vendor | Approach | Notes |
|---|---|---|
| **Microsoft GraphRAG** | LLM extraction → graph → Leiden → community reports | Reference OSS implementation; Python + Azure OpenAI |
| **Neo4j + LLM Graphs** | Property graph + Cypher retrieval | Strong query language, mature tooling |
| **LightRAG** | Dual-level (local/global) retrieval over a simple graph | Lightweight, fast indexing, popular 2025 alternative |
| **HippoRAG** | Neurobiologically inspired, "Personalized PageRank" over entities | Strong multi-hop; retrieval as activation spreading |
| **GraphRAG-SDK / AWS Neptune Analytics** | Managed graph + Bedrock | Enterprise managed path |
| **LlamaIndex `PropertyGraphIndex`** | Graph index abstraction | Integrates with vector + keyword |
| **LangChain + Neo4j** | GraphCypherQAChain | Natural-language → Cypher |

(See `04-Tools-and-Frameworks.md` for deeper coverage.)

---

## When NOT to Use GraphRAG

- **Small corpora (< a few hundred docs)** where vector RAG is simpler and cheaper.
- **Pure point-lookup Q&A** with no relational/multi-hop need.
- ** Strict low-latency / low-cost** constraints at indexing time (GraphRAG extraction is expensive).
- **Highly unstructured, low-entity domains** (e.g. poetry) where graph structure adds little.

A pragmatic rule: start with vector RAG (cheap), add a graph layer only when you hit global-query or multi-hop failure modes.

---

## GraphRAG Variants at a Glance

| Variant | Graph builder | Query style | Signature strength |
|---|---|---|---|
| Microsoft GraphRAG | LLM extraction + Leiden | Local + Global (map-reduce) | Global summarization |
| LightRAG | Lightweight entity graph | Dual-level (low/high) | Fast, cheap indexing |
| HippoRAG | Entity graph + PPR | Personalized PageRank | Multi-hop reasoning |
| Neo4j GraphRAG | Property graph | Cypher + vector hybrid | Queryable enterprise graph |
| LlamaIndex PropertyGraph | Abstraction over backends | Unified index query | Ecosystem integration |

All share the same conceptual spine — **extract structure, index structure, retrieve over structure** — differing mainly in graph model, clustering, and query algorithms.

---

## Prompt Engineering for Extraction

The quality of the graph is bounded by extraction prompts. Best practices observed across implementations:

- **Few-shot examples** tailored to your domain (legal, biomedical, financial).
- **Closed type lists** to keep entities queryable.
- **Explicit JSON schema** enforcement with repair on malformed output.
- **Self-reflection / gleanings**: a second pass asks "did I miss any entities?" to lift recall.
- **Chunk sizing**: too small fragments context; too large dilutes extraction precision. 800–1500 tokens is a common sweet spot.

---

## Adoption by Industry

- **Legal & compliance** (`../49-AI-for-Legal-and-LegalTech/`): mapping statutes, cases, and obligations; multi-hop "which law cites X" queries.
- **Biomedical / science** (`../42-AI-for-Science-and-Drug-Discovery/`): protein–disease–drug relationship graphs.
- **Enterprise knowledge**: "what does the org know about X" global summaries across wikis, tickets, docs.
- **Government** (`../68-AI-in-Government-and-Public-Sector/`): policy relationship mapping under audit constraints.
- **Finance** (`../67-AI-in-Finance-and-Financial-Services/`): entity resolution across filings and news.

---

## Common Misconceptions

1. **"GraphRAG replaces vector RAG."** No — it augments it; hybrids win.
2. **"The graph reasons."** The LLM reasons over the graph; the graph supplies structured context.
3. **"It's only for big tech."** Any corpus with relational structure benefits.
4. **"Indexing is free."** Extraction is the dominant cost; budget for it.
5. **"More entities = better."** Entity explosion harms communities; curate types.

---

## Getting-Started Checklist

- [ ] Inventory corpus size and entity density.
- [ ] Decide open vs closed entity typing.
- [ ] Pick a tool (Microsoft GraphRAG for reference; LightRAG for speed).
- [ ] Run a small pilot on ~100 docs; inspect the graph.
- [ ] Benchmark global + multi-hop queries vs your existing vector RAG.
- [ ] Add evaluation gates (see `../58-AI-Evaluation-and-Benchmarking-at-Scale/`).
- [ ] Plan incremental indexing for production.

---

## Maturity and Adoption Signals

- Microsoft's `graphrag` repo reached substantial adoption on GitHub within a year of release; LightRAG and HippoRAG became frequent citations in 2025 RAG surveys.
- Neo4j, AWS, and Databricks all published GraphRAG reference architectures in 2024–2025.
- It is a recurring topic in RAG evaluation literature (see `../58-AI-Evaluation-and-Benchmarking-at-Scale/`) where global-answer quality is measured.
- Listed among top "advanced RAG" techniques in practitioner surveys alongside Agentic RAG and re-ranking.

Even though live web verification was unavailable at enrichment time, the **internal library signal is decisive**: GraphRAG is referenced as a subsection in `../04-RAG/02-Advanced-RAG.md` but has no dedicated, deep treatment — a clear gap for a technique now considered core to modern RAG.

---

## Relationship to Other Library Categories

- `../04-RAG/` — the parent RAG category; GraphRAG is an advanced RAG variant.
- `../32-Agent-Memory-Systems/` — graphs as long-term structured memory.
- `../46-Agentic-Browser-Automation/` & `../03-Agents/` — agents that navigate graphs.
- `../37-AI-Native-Databases/` — graph-native storage backends.
- `../52-AI-Hallucination-Detection-and-Mitigation/` — graphs improve attribution.
- `../58-AI-Evaluation-and-Benchmarking-at-Scale/` — measuring global-answer quality.
- `../43-AI-Data-Provenance-and-Content-Authenticity/` — entity trails as provenance.
- `../64-AI-Model-Explainability-and-XAI/` — explicit relational reasoning aids interpretability.

---

## References

- Microsoft Research, *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* (2024).
- Neo4j, *GraphRAG: Knowledge Graph-Driven RAG* (2024).
- LightRAG: *Simple and Fast GraphRAG* (2025).
- HippoRAG: *Neurobiologically Inspired Long-Term Memory for LLMs* (2024).
- LlamaIndex `PropertyGraphIndex` documentation.
- See also `../04-RAG/02-Advanced-RAG.md` §GraphRAG for a practitioner intro.

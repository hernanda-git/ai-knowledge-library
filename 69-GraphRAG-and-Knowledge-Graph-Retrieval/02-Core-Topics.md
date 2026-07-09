# GraphRAG — Core Topics

> Deep coverage of the conceptual core of graph-augmented retrieval: knowledge extraction, entity typing, relationship modeling, graph construction, community detection (Leiden), hierarchical summarization, and the local/global query algorithms. Companion to `01-Overview.md`.

---

## Table of Contents

1. [Knowledge Extraction](#knowledge-extraction)
2. [Entity and Relationship Typing](#entity-and-relationship-typing)
3. [Text Units and Claims](#text-units-and-claims)
4. [Graph Construction and Storage](#graph-construction-and-storage)
5. [Community Detection: Leiden](#community-detection-leiden)
6. [Hierarchical Community Summarization](#hierarchical-community-summarization)
7. [Local Query Algorithm](#local-query-algorithm)
8. [Global Query Algorithm (Map-Reduce)](#global-query-algorithm-map-reduce)
9. [Entity Disambiguation and Resolution](#entity-disambiguation-and-resolution)
10. [Incremental Indexing](#incremental-indexing)
11. [Evaluation Dimensions](#evaluation-dimensions)

---

## Knowledge Extraction

The first LLM pass turns raw text into a graph. Given a chunk, the extractor emits:

- **Entities**: `{name, type, description}` (e.g. `OpenAI, Organization, "AI lab..."`).
- **Relationships**: `{source, target, type, description}` (e.g. `OpenAI --funded_by--> Microsoft`).
- **Optional claims** for fact-level attribution.

Microsoft GraphRAG uses *few-shot* prompts with domain examples and enforces JSON schema. Extraction is **the dominant cost** and is usually batched over chunks.

```python
# Conceptual extraction prompt structure (Microsoft GraphRAG style)
EXTRACT_PROMPT = """
Given a text document, identify entities and relationships.
Output JSON:
{
  "entities": [{"name": str, "type": str, "description": str}],
  "relationships": [{"source": str, "target": str, "type": str, "description": str}]
}
Document:
{input_text}
"""
```

Key levers: **model choice** (small model = cheaper, more noise), **few-shot examples**, **chunk size**, and **domain dictionaries** (restricting entity types to your schema).

---

## Entity and Relationship Typing

Typed graphs are far more useful than untyped ones. Define a closed or open type system:

| Open typing | Closed typing |
|---|---|
| LLM invents types freely | You predefine `Person, Org, Product, Law, Risk` |
| Flexible, noisy | Precise, queryable, governed |
| Good for exploration | Good for regulated/enterprise |

For enterprise and legal (see `../49-AI-for-Legal-and-LegalTech/` and `../68-AI-in-Government-and-Public-Sector/`), **closed typing** is preferred because downstream queries and compliance depend on consistent categories.

Relationship direction and labels matter: `acquired`, `regulates`, `cites`, `owns`. Bidirectional relationships should be normalized to a canonical direction during construction.

---

## Text Units and Claims

A **text unit** is the atomic piece of source text (often a sentence or small chunk) associated with the entities/relationships it produced. Storing the linkage enables:

- **Attribution**: every graph element traces to source spans.
- **Local query grounding**: answers cite the original text, not just the graph.
- **Provenance** (see `../43-AI-Data-Provenance-and-Content-Authenticity/`): entity trails become auditable evidence.

Claims (subject–predicate–object with a source span) are a finer grain than relationships and support fact-level citation — increasingly used where hallucination risk must be minimized (see `../52-AI-Hallucination-Detection-and-Mitigation/`).

---

## Graph Construction and Storage

After extraction, elements are loaded into a store. Options:

| Store | Pros | Cons |
|---|---|---|
| **NetworkX (in-memory)** | Simple, great for prototyping | Doesn't scale |
| **Neo4j** | Cypher, mature, scalable | Operational overhead |
| **Kuzu / DuckDB-graph** | Embeddable, fast | Smaller ecosystem |
| **Cosmos DB / Neptune** | Managed, cloud-native | Vendor lock-in |
| **GraphML / Parquet files** | Portable, cheap | No live query |

Microsoft GraphRAG persists to Parquet + a graph table; you can then load into Neo4j for Cypher queries.

```python
# Load extracted graph into Neo4j (conceptual)
from neo4j import GraphDatabase
driver = GraphDatabase.driver(URI, auth=(USER, PW))

def load(tx, entities, relationships):
    for e in entities:
        tx.run("MERGE (n:Entity {name:$n}) SET n.type=$t, n.desc=$d",
               n=e["name"], t=e["type"], d=e["description"])
    for r in relationships:
        tx.run("""
        MATCH (a:Entity {name:$s}), (b:Entity {name:$t})
        MERGE (a)-[:REL {type:$rel}]->(b)
        """, s=r["source"], t=r["target"], rel=r["type"])
```

---

## Community Detection: Leiden

Community detection groups densely connected nodes. Microsoft GraphRAG uses the **Leiden** algorithm (an improvement over Louvain) because it guarantees well-connected communities.

- Output: a hierarchical partition — Level 0 (fine) … Level N (coarse).
- Each level yields a set of **community reports** summarizing its members.
- The **top-level** (fewest, largest) communities power global queries; lower levels add granularity.

Why hierarchy matters: global queries can target a specific level for the right granularity (e.g. summarize at mid-level for "by department").

```python
# Using python-louvain / leidenalg conceptually
import leidenalg  # or networkx.community.louvain_communities
partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition)
communities = {i: members for i, members in enumerate(partition)}
```

---

## Hierarchical Community Summarization

For each community, an LLM summarizes its entities and relationships into a **community report** containing:

- A human-readable summary.
- The rating of importance (for global query prioritization).
- Extracted key entities/facts.

These reports are the **global index**. They are far smaller than the corpus yet capture its structure, enabling "summarize everything" queries at tractable token cost.

```python
COMMUNITY_SUMMARY_PROMPT = """
Summarize the following community of entities and relationships.
List key entities, their relationships, and the main themes.
Rate overall importance 1-10 for answering broad questions.
Entities: {entities}
Relationships: {relationships}
"""
```

---

## Local Query Algorithm

1. Identify entities in the user question (NER over the query).
2. Expand a **k-hop subgraph** around those entities.
3. Retrieve associated text units (grounding).
4. Build a context window: subgraph description + relevant text.
5. Generate answer with citations to text units / entity IDs.

Local queries excel at "tell me about X and how it connects to Y." They are cheap (small subgraph) and highly attributable.

```python
def local_query(question, graph, k=2):
    seeds = extract_entities(question)
    subgraph = expand_neighborhood(graph, seeds, hops=k)
    text_units = fetch_text_units(subgraph)
    context = format(subgraph, text_units)
    return llm_answer(question, context)  # with citations
```

---

## Global Query Algorithm (Map-Reduce)

The signature GraphRAG capability. Steps:

1. **Map**: for each top-level community report, an LLM answers the question using only that report → partial answers.
2. **Shuffle**: partial answers are randomly ordered to avoid position/length bias.
3. **Reduce**: an LLM folds partials into a composite answer.
4. **(Optional) Drift reduce**: re-reduce concatenated partials to improve coherence.

```python
def global_query(question, community_reports):
    partials = [llm_answer(question, report) for report in community_reports]
    random.shuffle(partials)                      # reduce bias
    answer = llm_reduce(question, partials)
    answer = llm_reduce(question, [answer])       # drift reduction
    return answer
```

Token cost scales with the number of communities, so coarse levels (fewer communities) are used for broad questions.

---

## Entity Disambiguation and Resolution

LLMs produce variant names ("MS", "Microsoft", "Microsoft Corp"). Resolution strategies:

- **Blocking + embedding similarity**: cluster near-identical names before merge.
- **LLM-based resolution**: ask the model if two entities are the same.
- **Canonicalization rules**: lowercase, strip legal suffixes, map aliases.

Poor resolution fragments the graph (same entity as many nodes), weakening communities. Budget a resolution pass.

---

## Incremental Indexing

Rebuilding the whole graph on each document addition is wasteful. Incremental patterns:

- Extract only new chunks; add nodes/edges.
- Resolve against existing entities (merge, don't duplicate).
- Re-run community detection only on affected regions, or accept slight staleness.
- Append new community reports; invalidate stale global summaries lazily.

This matters for **agent memory** (see `../32-Agent-Memory-Systems/`) and continuously-ingested enterprise data.

---

## Evaluation Dimensions

Measure GraphRAG against vector RAG on:

| Dimension | How to measure |
|---|---|
| Global answer quality | Human/grader rubric on "summarize corpus" tasks |
| Multi-hop accuracy | QA sets requiring ≥2 hops |
| Attribution | % of claims linked to source text |
| Latency / cost | Index build time, query tokens |
| Faithfulness | Hallucination rate vs source |

See `../58-AI-Evaluation-and-Benchmarking-at-Scale/` for harness design. GraphRAG typically wins on global/multi-hop and trails on per-query latency.

---

## Cross-References

- Extraction prompting ↔ `../04-RAG/02-Advanced-RAG.md`
- Graph storage ↔ `../37-AI-Native-Databases/`
- Attribution ↔ `../43-AI-Data-Provenance-and-Content-Authenticity/` , `../52-AI-Hallucination-Detection-and-Mitigation/`
- Incremental memory ↔ `../32-Agent-Memory-Systems/`
- Evaluation ↔ `../58-AI-Evaluation-and-Benchmarking-at-Scale/`

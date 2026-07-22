# GraphRAG — Technical Deep Dive

> Implementation-level reference: building a GraphRAG index end-to-end with the Microsoft `graphrag` library, a Neo4j-native variant, HippoRAG's Personalized PageRank, LightRAG's dual retrieval, hybrid graph+vector retrieval, cost modeling, and production pitfalls. Companion to `01-Overview.md` and `02-Core-Topics.md`.

---

## Table of Contents

1. [End-to-End with Microsoft GraphRAG](#end-to-end-with-microsoft-graphrag)
2. [Configuration Deep Dive](#configuration-deep-dive)
3. [Indexing Internals](#indexing-internals)
4. [Query Internals](#query-internals)
5. [Neo4j-Native GraphRAG](#neo4j-native-graphrag)
6. [HippoRAG: Personalized PageRank](#hipporag-personalized-pagerank)
7. [LightRAG: Dual-Level Retrieval](#lightrag-dual-level-retrieval)
8. [Hybrid Graph + Vector Retrieval](#hybrid-graph--vector-retrieval)
9. [Cost Model](#cost-model)
10. [Scaling and Performance](#scaling-and-performance)
11. [Production Pitfalls](#production-pitfalls)
12. [Security and Governance](#security-and-governance)

---

## End-to-End with Microsoft GraphRAG

Install and init:

```bash
pip install graphrag
mkdir -p ./input
cp my_docs/*.txt ./input
graphrag init --root ./ragtest
graphrag index --root ./ragtest
graphrag query --root ./ragtest --method local "Who is mentioned alongside X?"
graphrag query --root ./ragtest --method global "What are the main themes?"
```

The `init` creates `settings.yaml` and `prompts/`. Indexing writes Parquet outputs under `./ragtest/output/<run_id>/`:
- `entities.parquet`, `relationships.parquet`, `text_units.parquet`
- `communities.parquet`, `community_reports.parquet`
- `covariates.parquet` (if claims enabled)

---

## Configuration Deep Dive

`settings.yaml` key levers:

```yaml
llm:
  model: azure_openai.gpt-4o-mini   # extraction model
  api_base: ${AZURE_OPENAI_ENDPOINT}
  api_version: 2024-06-01
embeddings:
  model: azure_openai. text-embedding-3-small
chunks:
  size: 1200
  overlap: 100
input:
  file_type: text
extract_graph:
  model: azure_openai.gpt-4o        # stronger for extraction
  max_gleanings: 1
report:
  model: azure_openai.gpt-4o        # community summaries
  max_length: 1500
cluster:
  max_cluster_size: 10
```

Tips:
- Use a **cheaper model** for extraction if budget-bound, but expect noisier graphs.
- `max_gleanings` lets the LLM re-scan a chunk for missed entities (improves recall).
- `max_cluster_size` controls community granularity → global query cost.

---

## Indexing Internals

The index pipeline stages:

1. `create_base_text_units` — chunk documents.
2. `extract_graph` — LLM entity/relationship extraction (+ optional gleanings).
3. `finalize_graph` — build networkx graph, compute degrees.
4. `cluster_graph` — Leiden, multiple levels.
5. `extract_community_reports` — summarize each community.
6. `extract_graph_embeddings` — embed entities for vector-assisted lookup.
7. `generate_leverages` / `create_final_*_tables` — assemble outputs.

Each stage is resumable; re-running skips completed stages by default.

```python
from graphrag.api import build_index
# Programmatic indexing for pipelines
await build_index(config=settings, root="./ragtest")
```

---

## Query Internals

**Local** (`local_search`):
- Embed the query; retrieve similar text units + entities.
- Expand entity neighborhoods (graph traversal).
- Concatenate: community context + entity context + text units + conversation history.
- Generate with citations.

**Global** (`global_search`):
- Load community reports at a chosen level.
- Map → partial answers; shuffle; reduce; optional drift reduce.
- Return answer + the underlying report IDs used.

```python
from graphrag.query.api import local_search, global_search
from graphrag.query.factory import get_default_text_embedder

context = await local_search(
    query="What risks does project Athena face?",
    search_engine=engine,   # built from the index
    runtime=None,
)
print(context.response)
```

---

## Neo4j-Native GraphRAG

Instead of Parquet, persist to Neo4j and query with Cypher — powerful for structured follow-ups.

```python
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_qa.cypher import GraphCypherQAChain

graph = Neo4jGraph(URI, USER, PW)
chain = GraphCypherQAChain.from_llm(
    llm, graph=graph, verbose=True,
    return_intermediate_steps=True,
)
print(chain.invoke("Which organizations regulate X and what laws cite them?"))
```

Neo4j's own GraphRAG package adds **vector + graph hybrid** retrievers and a "kg builder" for ingestion.

---

## HippoRAG: Personalized PageRank

HippoRAG mimics human long-term memory: entities form a graph; retrieval is **activation spreading** via Personalized PageRank (PPR) from query-seeded nodes.

```python
# Conceptual HippoRAG retrieval
def hipporag_retrieve(query, graph, llm, alpha=0.85):
    seed_entities = llm.extract_entities(query)
    personalization = {e: 1.0 for e in seed_entities}
    scores = personalized_pagerank(graph, personalization, alpha)
    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return retrieve_text_for(top)
```

PPR naturally handles **multi-hop**: importance propagates through edges, surfacing distant but connected evidence. Strong on multi-hop QA benchmarks.

---

## LightRAG: Dual-Level Retrieval

LightRAG uses a simpler graph (entity + relation) and a **dual-level** retrieval:

- **Low-level**: entity-specific details.
- **High-level**: relationships and themes.

It combines keyword + graph retrieval and is prized for **fast indexing** and low resource use versus Microsoft's heavier pipeline.

```python
import lightrag
rag = lightrag.LightRAG(working_dir="./lightrag_data")
rag.insert_documents([open("doc.txt").read()])
print(rag.query("Summarize the key conflicts.", mode="global"))
```

---

## Hybrid Graph + Vector Retrieval

The emerging best practice (2025–2026): **retrieve candidates by vector, re-rank/filter by graph, and ground by text.**

```python
def hybrid_retrieve(query, vector_store, graph, k=20):
    # 1. vector candidate passages
    candidates = vector_store.similarity_search(query, k=k*3)
    # 2. extract entities, pull graph context
    ents = extract_entities(query)
    graph_ctx = expand_neighborhood(graph, ents, hops=1)
    # 3. fuse: passages whose entities appear in graph_ctx rank higher
    scored = [(c, score(c, graph_ctx)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:k]]
```

This pairs the strengths of `../04-RAG/03-Vector-Databases.md` (cheap similarity) with graph structure (relational reasoning).

---

## Cost Model

| Stage | Cost driver | Mitigation |
|---|---|---|
| Extraction | #chunks × LLM tokens | smaller model, bigger chunks |
| Community reports | #communities × summary tokens | coarser clustering |
| Global query | #communities in query | query higher level only |
| Local query | subgraph + text units | limit hops/k |

Rule of thumb: **indexing is the big one-time bill; global queries are the recurring one.** Cache community reports; they change only when the corpus changes.

---

## Scaling and Performance

- **Millions of entities**: move off NetworkX to Neo4j/Kuzu; parallelize extraction across chunks (map stage).
- **Indexing time**: linearly proportional to tokens; use batch + concurrency limits to respect rate limits.
- **Global query latency**: dominated by #communities; choose the coarsest level that answers the question.
- **Memory**: in-memory graphs blow up past ~100k nodes; use disk-backed stores.

---

## Debugging a GraphRAG Index

When answers are weak, diagnose in order:

1. **Inspect extraction**: sample `entities.parquet` / `relationships.parquet`. Are key entities missing? → strengthen extraction prompt, raise `max_gleanings`, use a stronger model.
2. **Inspect communities**: are communities coherent? Fragmented communities → resolution issue or `max_cluster_size` too small.
3. **Inspect community reports**: do they capture themes? Thin reports → larger `report.max_length`, better extraction.
4. **Inspect query**: for global, is the right community level queried? For local, is the subgraph too small (raise hops)?

A quick health check:

```python
import pandas as pd
ents = pd.read_parquet("output/<run>/entities.parquet")
rels = pd.read_parquet("output/<run>/relationships.parquet")
print("entities:", len(ents), "relationships:", len(rels))
print(ents["type"].value_counts().head())
print("avg degree:", 2*len(rels)/max(len(ents),1))
```

Low average degree (<2) signals a disconnected, weak graph.

---

## Embedding Entity Descriptions

Microsoft GraphRAG embeds **entity descriptions**, enabling vector-style entity lookup during local queries (find entities semantically near the question). This hybrid (graph + embedding) is why local search retrieves relevant entities even with paraphrased questions.

```python
from graphrag.query.embeddings import OpenAIEmbeddingClient
embedder = OpenAIEmbeddingClient(model="text-embedding-3-small")
q_emb = embedder.embed(query)
similar_entities = vector_search(entity_embeddings, q_emb, top_k=10)
```

---

## Handling Very Large Corpora

Past ~1M tokens, single-machine indexing strains. Patterns:

- **Shard by source**: index sub-corpora, then merge graphs (resolve entities across shards).
- **Parallel extraction**: map chunks to worker LLM calls; respect rate limits.
- **Disk-backed graph**: Kuzu/Neo4j instead of NetworkX.
- **Two-tier**: vector RAG for hot/point queries, GraphRAG for periodic global summaries.

---

## Production Pitfalls

1. **Noisy extraction** → fragmented graph. Fix: stronger model, few-shot, resolution pass.
2. **Entity explosion** → unmanageable graph. Fix: closed type schema, filtering.
3. **Stale global index** → answers lag corpus. Fix: incremental indexing (see `02-Core-Topics.md`).
4. **Token blowup on global queries** → cap community level, cache.
5. **Attribution gaps** → answers not traceable. Fix: keep text-unit links; verify with `../52-AI-Hallucination-Detection-and-Mitigation/`.
6. **Over-engineering** → GraphRAG where vector RAG suffices. Fix: benchmark first.

---

## Security and Governance

- **PII in graphs**: extracted entities may include personal data — apply `../40-AI-Data-Sovereignty-and-Privacy/` controls; mask before extraction where required.
- **Poisoning**: malicious documents can inject false edges. Use `../61-AI-Red-Teaming-for-LLMs/` and provenance checks (`../43-AI-Data-Provenance-and-Content-Authenticity/`).
- **Access control**: graph nodes may span confidentiality tiers — enforce row/entity-level ACLs at query time.
- **Regulated domains** (legal `../49-AI-for-Legal-and-LegalTech/`, gov `../68-AI-in-Government-and-Public-Sector/`): prefer closed typing and full attribution for auditability.

---

## Cross-References

- Vector baseline: `../04-RAG/01-RAG-Architectures.md`, `../04-RAG/03-Vector-Databases.md`
- Memory: `../32-Agent-Memory-Systems/`
- Graph DBs: `../37-AI-Native-Databases/`
- Eval: `../58-AI-Evaluation-and-Benchmarking-at-Scale/`
- Hallucination: `../52-AI-Hallucination-Detection-and-Mitigation/`
- Provenance: `../43-AI-Data-Provenance-and-Content-Authenticity/`
- Privacy: `../40-AI-Data-Sovereignty-and-Privacy/`
- Red teaming: `../61-AI-Red-Teaming-for-LLMs/`

---
**See also:**
- [07 — RAG and Retrieval Research: The Frontier (2025–2026)](07-Emerging/17-Research-Frontiers-2026/07-RAG-Retrieval-Research.md)
- [AI-Powered Search: Beyond Traditional RAG](06-Advanced/11-AI-Powered-Search.md)
- [06 — Enterprise RAG Search System](14-Case-Studies-Real-World-Projects/06-RAG-Search-System.md)

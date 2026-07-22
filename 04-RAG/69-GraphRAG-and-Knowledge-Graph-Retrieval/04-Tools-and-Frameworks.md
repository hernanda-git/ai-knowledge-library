# GraphRAG — Tools and Frameworks

> A practical map of the GraphRAG tooling landscape (2024–2026): Microsoft GraphRAG, LightRAG, HippoRAG, Neo4j, LlamaIndex PropertyGraphIndex, LangChain, AWS/Neptune, and managed services. Includes selection guidance, quickstart snippets, and integration patterns. Companion to `03-Technical-Deep-Dive.md`.

---

## Table of Contents

1. [Tool Landscape Overview](#tool-landscape-overview)
2. [Microsoft GraphRAG](#microsoft-graphrag)
3. [LightRAG](#lightrag)
4. [HippoRAG](#hipporag)
5. [Neo4j GraphRAG](#neo4j-graphrag)
6. [LlamaIndex PropertyGraphIndex](#llamaindex-propertygraphindex)
7. [LangChain Graph Integrations](#langchain-graph-integrations)
8. [AWS Neptune Analytics + Bedrock](#aws-neptune-analytics--bedrock)
9. [Managed / Cloud Services](#managed--cloud-services)
10. [Vector Stores That Add Graph](#vector-stores-that-add-graph)
11. [Selection Decision Tree](#selection-decision-tree)
12. [Integration with the Rest of the Stack](#integration-with-the-rest-of-the-stack)

---

## Tool Landscape Overview

| Tool | Style | Best for | License / Hosting |
|---|---|---|---|
| Microsoft GraphRAG | LLM→graph→Leiden→reports | Global summarization, reference impl | OSS (MIT), Azure-friendly |
| LightRAG | Simple graph + dual retrieval | Fast, lightweight indexing | OSS |
| HippoRAG | PPR over entity graph | Multi-hop QA | OSS (research) |
| Neo4j GraphRAG | Property graph + Cypher | Queryable enterprise graphs | OSS/Enterprise |
| LlamaIndex | Graph index abstraction | Multi-index RAG apps | OSS |
| LangChain | GraphCypherQAChain | NL→Cypher apps | OSS |
| AWS Neptune + Bedrock | Managed graph + LLM | AWS-native enterprises | Managed cloud |

---

## Microsoft GraphRAG

The reference implementation. Strongest for **global queries** via community reports.

```bash
pip install graphrag
graphrag init --root ./ragtest
graphrag index --root ./ragtest
```

```python
from graphrag.query.cli import run_global_search
# or via API as shown in 03-Technical-Deep-Dive.md
```

Pros: battle-tested, great docs, Azure OpenAI integration. Cons: heavier indexing, Parquet-centric (load to Neo4j for live queries).

---

## LightRAG

Lightweight alternative emphasizing speed and a simple dual-level (local/global) retrieval.

```python
import lightrag
rag = lightrag.LightRAG(working_dir="./data")
rag.insert_documents([text])
rag.query("What are the main conflicts?", mode="global")
```

Pros: fast indexing, low resource; good for prototypes and smaller corpora. Cons: less mature ecosystem than Microsoft/Neo4j.

---

## HippoRAG

Research-grade, biologically-inspired; excels at **multi-hop** via Personalized PageRank.

```python
from hipporag import HippoRAG
hippo = HippoRAG()
hippo.index(docs)
results = hippo.retrieve("Who funded the lab that published X?")
```

Pros: strong multi-hop benchmark results. Cons: research-focused, less turnkey for production.

---

## Neo4j GraphRAG

Production-grade graph database with first-class GraphRAG support (`neo4j-graphrag` package): vector + graph retrievers, KG builder.

```python
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever, HybridRetriever
# KG builder ingests docs -> graph automatically
from neo4j_graphrag.llm import OpenAILLM
```

Pros: Cypher power, scalability, enterprise features (security, ACLs). Cons: operational overhead of running Neo4j.

---

## LlamaIndex PropertyGraphIndex

Unifies graph, vector, and keyword retrieval behind one index.

```python
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

docs = SimpleDirectoryReader("./data").load_data()
index = PropertyGraphIndex.from_documents(
    docs, llm=OpenAI(), embed_model=OpenAIEmbedding(),
    show_progress=True,
)
query_engine = index.as_query_engine()
print(query_engine.query("Summarize relationships between the main entities."))
```

Pros: integrates with the broad LlamaIndex ecosystem; flexible backends (Neo4j, Kuzu, Memgraph). Cons: abstraction can hide tuning.

---

## LangChain Graph Integrations

`GraphCypherQAChain` turns natural language into Cypher over a Neo4j graph.

```python
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_qa.cypher import GraphCypherQAChain

graph = Neo4jGraph(URI, USER, PW)
chain = GraphCypherQAChain.from_llm(llm, graph=graph, verbose=True)
chain.invoke("Which regulations cite the AI Act?")
```

Pros: familiar LangChain patterns; good for NL→graph apps. Cons: Cypher generation is itself an LLM task needing guardrails.

---

## AWS Neptune Analytics + Bedrock

Managed path for AWS shops: Neptune Analytics for graph storage/query, Bedrock for LLM extraction.

```python
# Conceptual: Bedrock extracts -> Neptune loads -> query
import boto3
bedrock = boto3.client("bedrock-runtime")
# extract entities via Bedrock, load via neptune analytics client
```

Pros: zero graph-ops, scales, IAM-integrated. Cons: AWS lock-in.

---

## Managed / Cloud Services

- **Azure AI Search** now offers graph-aware retrieval features alongside vector.
- **Databricks** GraphFrames + LLM extraction for lakehouse-native graphs.
- **Google** Vertex + BigQuery graph patterns.
- **Neo4j AuraDB** — hosted Neo4j.

These reduce ops burden at the cost of portability.

---

## Vector Stores That Add Graph

Some vector DBs now ship graph/relational features bridging to GraphRAG:
- **Weaviate** — cross-references / GraphQL.
- **Milvus** — graph-adjacent metadata filtering.
- **Qdrant** — payload graph traversal helpers.

See `../04-RAG/03-Vector-Databases.md` for the vector baseline these extend.

---

## Selection Decision Tree

```
Need global "summarize corpus"? ──yes──> Microsoft GraphRAG or LightRAG (global)
       │no
Need strong multi-hop? ──yes──> HippoRAG or Neo4j + PPR
       │no
Need live Cypher queries / enterprise ACLs? ──yes──> Neo4j GraphRAG
       │no
Already on LlamaIndex/LangChain? ──yes──> PropertyGraphIndex / GraphCypherQAChain
       │no
Small corpus / prototype? ──yes──> LightRAG
       │no
Start with vector RAG (../04-RAG/) and add graph only if needed.
```

---

## Integration with the Rest of the Stack

- **Agents** (`../03-Agents/`, `../32-Agent-Memory-Systems/`): graphs as structured long-term memory the agent queries.
- **Orchestration** (`../31-AI-Workflow-Orchestration-and-Durable-Execution/`): schedule incremental indexing as durable jobs.
- **Eval** (`../58-AI-Evaluation-and-Benchmarking-at-Scale/`): gate graph quality with automated checks.
- **Observability** (`../20-Agent-Infrastructure-and-Observability/`): trace extraction/query token cost.
- **Cost control** (`../59-AI-Agent-Financial-Governance-and-Cost-Control/`): cap indexing and global-query spend.
- **Security** (`../18-Agent-Security-and-Trust/`, `../61-AI-Red-Teaming-for-LLMs/`): defend against graph poisoning.

---

## Quick Reference: Which to Learn First

1. **Microsoft GraphRAG** — to understand the canonical pipeline.
2. **LightRAG** — to see a leaner alternative fast.
3. **Neo4j GraphRAG** — to productionize with real queries.
4. **LlamaIndex PropertyGraphIndex** — to embed in a larger RAG app.

---

## Cross-References

- Parent RAG: `../04-RAG/`
- Vector baseline: `../04-RAG/03-Vector-Databases.md`
- Graph DBs: `../37-AI-Native-Databases/`
- Agents/memory: `../03-Agents/`, `../32-Agent-Memory-Systems/`
- Workflows: `../31-AI-Workflow-Orchestration-and-Durable-Execution/`
- Eval: `../58-AI-Evaluation-and-Benchmarking-at-Scale/`
- Cost: `../59-AI-Agent-Financial-Governance-and-Cost-Control/`
- Security: `../18-Agent-Security-and-Trust/`, `../61-AI-Red-Teaming-for-LLMs/`

---
**See also:**
- [07 — RAG and Retrieval Research: The Frontier (2025–2026)](07-Emerging/17-Research-Frontiers-2026/07-RAG-Retrieval-Research.md)
- [AI-Powered Search: Beyond Traditional RAG](06-Advanced/11-AI-Powered-Search.md)
- [06 — Enterprise RAG Search System](14-Case-Studies-Real-World-Projects/06-RAG-Search-System.md)

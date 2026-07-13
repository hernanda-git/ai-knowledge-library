# Context Engineering — Tools & Frameworks

> July 2026

A practical survey of the libraries, frameworks, and infrastructure used to implement context engineering in production, with code snippets and selection guidance.

Prerequisites: [01-Overview.md](./01-Overview.md), [02-Core-Topics.md](./02-Core-Topics.md), [03-Technical-Deep-Dive.md](./03-Technical-Deep-Dive.md).

---

## 1. Landscape Overview

| Layer | Purpose | Representative Tools |
|-------|---------|----------------------|
| **Orchestration / State** | Manage agent state & context flow | LangGraph, LlamaIndex Workflows, CrewAI, OpenAI Agents SDK, PydanticAI |
| **Retrieval (SELECT)** | Semantic search over knowledge/tools | LlamaIndex, Haystack, vector DBs, rerankers |
| **Memory (WRITE/SELECT)** | Cross-session persistence | Mem0, Zep, Letta (MemGPT), LangMem |
| **Compression (COMPRESS)** | Summarize/prune context | Framework summarizers, LLMLingua |
| **Tool/Context protocol** | Standardize context sources | MCP (Model Context Protocol) |
| **Observability / Eval** | Measure context quality | LangSmith, Phoenix/Arize, RAGAS, DeepEval |

---

## 2. Orchestration Frameworks

### 2.1 LangGraph

Graph-based agent runtime with an explicit, typed `State`. Context engineering is first-class: you decide exactly what state fields render into the model call.

```python
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # LLM-visible
    raw_docs: list                           # WRITE: hidden from LLM
    summary: str                             # COMPRESS: injected, full text stored

def retrieve(state):
    docs = vector_store.search(state["messages"][-1].content, k=5)
    return {"raw_docs": docs}

def compact(state):
    if len(state["messages"]) > 20:
        s = llm.invoke(f"Summarize: {state['messages'][:-6]}")
        return {"summary": s.content, "messages": state["messages"][-6:]}
    return {}

g = StateGraph(State)
g.add_node("retrieve", retrieve); g.add_node("compact", compact)
```

LangGraph + **LangMem** provides memory primitives (extract, consolidate, recall) designed for the Write/Select loop.

### 2.2 LlamaIndex

Strong on the SELECT side: ingestion, chunking, indexing, retrieval, reranking, plus `Workflows` for orchestration and a `Memory` module with token-limited buffers.

```python
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3000)  # auto-trims to budget
index = VectorStoreIndex.from_documents(docs)
engine = index.as_chat_engine(memory=memory, similarity_top_k=5)
```

### 2.3 CrewAI / OpenAI Agents SDK / PydanticAI

- **CrewAI:** multi-agent role decomposition → natural ISOLATE. Each agent has its own scoped context.
- **OpenAI Agents SDK:** handoffs + sessions; sessions auto-manage conversation context.
- **PydanticAI:** typed dependencies injected into the prompt; clean control over what enters context.

---

## 3. Memory Systems (WRITE + SELECT)

| Tool | Model | Notable feature |
|------|-------|-----------------|
| **Mem0** | Extract facts → store → recall | Auto fact extraction, dedup, decay; drop-in memory layer |
| **Zep** | Temporal knowledge graph | Time-aware facts, handles context clash via recency |
| **Letta (MemGPT)** | OS-style tiered memory | Self-editing memory, paging in/out |
| **LangMem** | LangGraph-native | Semantic/episodic/procedural memory utilities |

```python
# Mem0 example
from mem0 import Memory
m = Memory()
m.add("User prefers Python and dislikes verbose answers", user_id="u1")  # WRITE
relevant = m.search("how should I format the code sample?", user_id="u1")  # SELECT
```

See [32-Agent-Memory-Systems](../32-Agent-Memory-Systems/) for a deeper comparison.

---

## 4. Retrieval & Reranking (SELECT)

- **Vector DBs:** Pinecone, Weaviate, Qdrant, Milvus, pgvector, Chroma.
- **Rerankers:** Cohere Rerank, BGE-reranker, Voyage rerank, cross-encoders.
- **Hybrid search:** BM25 + dense (Weaviate, Qdrant, Elastic) for precision.

```python
# Retrieve wide, rerank narrow (precision for the budget)
cands = qdrant.search(query, limit=25)
top   = cohere.rerank(query=query, documents=[c.text for c in cands], top_n=5)
context = [cands[r.index].text for r in top.results]
```

See [04-RAG](../04-RAG/) for chunking strategies and hybrid search details.

---

## 5. Compression Tooling (COMPRESS)

### 5.1 LLMLingua (Microsoft)

Prompt compression that drops low-information tokens using a small model — can compress prompts 2–20x with minimal quality loss.

```python
from llmlingua import PromptCompressor
c = PromptCompressor()
compressed = c.compress_prompt(long_context, rate=0.5, force_tokens=['\n','?'])
# compressed['compressed_prompt'] -> feed to the large model
```

### 5.2 Framework Summarizers

LangChain's `ConversationSummaryMemory`, LlamaIndex `SummaryIndex`, and LangGraph compaction nodes provide rolling/hierarchical summarization out of the box.

---

## 6. Model Context Protocol (MCP)

**MCP** (open standard, Anthropic 2024, now broadly adopted) standardizes how applications feed context — resources, tools, prompts — to models. It's the "USB-C for context sources."

- **Resources:** files, DB rows, API data exposed to the model.
- **Tools:** callable functions with schemas (enables tool-selection RAG).
- **Prompts:** reusable templated context.

```jsonc
// MCP server exposes a resource the client can SELECT into context
{
  "resources": [
    {"uri": "db://customers/schema", "name": "Customer DB schema",
     "mimeType": "text/plain"}
  ]
}
```

Because MCP servers can expose *many* tools/resources, context engineering (selective loading, discovery endpoints) is essential to avoid confusion. See [48-MCP-Cloud-Infrastructure-Agent-as-a-Service](../48-MCP-Cloud-Infrastructure-Agent-as-a-Service/).

---

## 7. Observability & Evaluation

| Tool | Strength |
|------|----------|
| **LangSmith** | Trace exact context sent per call; debug bloat |
| **Phoenix / Arize** | Open-source tracing, RAG eval |
| **RAGAS** | Context precision/recall, faithfulness metrics |
| **DeepEval** | Unit-test style LLM evals (pytest integration) |
| **TruLens** | Feedback functions for groundedness |

```python
# RAGAS: measure whether retrieved context was on-point
from ragas.metrics import context_precision, context_recall, faithfulness
scores = evaluate(dataset, metrics=[context_precision, context_recall, faithfulness])
```

Tracing the **actual assembled context** per request is the single most useful debugging habit — most agent bugs are context bugs. See [20-Agent-Infrastructure-and-Observability/03-Agent-Tracing-and-Observability.md](../20-Agent-Infrastructure-and-Observability/03-Agent-Tracing-and-Observability.md).

---

## 8. Reference Stack Recommendations

| Use case | Suggested stack |
|----------|-----------------|
| **RAG chatbot** | LlamaIndex + Qdrant + Cohere Rerank + RAGAS |
| **Long-horizon agent** | LangGraph + LangMem + LangSmith + auto-compaction |
| **Multi-agent system** | CrewAI or LangGraph supervisor + isolated sub-agent contexts |
| **Cross-session assistant** | Mem0 or Zep + prompt caching + hybrid retrieval |
| **Tool-heavy agent** | MCP + tool-selection RAG + schema compression |
| **Cost-critical** | Prompt caching + LLMLingua + aggressive trim |

---

## 9. Selection Checklist

1. Do you need cross-session memory? → add Mem0/Zep/Letta.
2. Is your knowledge base large? → LlamaIndex/Haystack + reranker.
3. Long sessions? → LangGraph with compaction, or LlamaIndex token-limited memory.
4. Many tools/data sources? → MCP + selective loading.
5. Can't tell why answers are wrong? → LangSmith/Phoenix tracing first.

---

## 10. Key Takeaways

1. LangGraph/LlamaIndex give you explicit control over what enters context — the core requirement.
2. Memory (Mem0/Zep/Letta) implements Write+Select across sessions.
3. LLMLingua and framework summarizers implement Compress.
4. MCP standardizes context sources; pair it with selective loading.
5. Always trace the assembled context and measure precision/recall/faithfulness.

---

*Next: [05-Future-Outlook.md](./05-Future-Outlook.md).*

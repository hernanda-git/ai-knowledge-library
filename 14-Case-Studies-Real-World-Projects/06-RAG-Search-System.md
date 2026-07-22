# 06 — Enterprise RAG Search System

## Case Study: Hybrid Search + Multi-Hop Retrieval for Enterprise Knowledge

| Metadata | Value |
|----------|-------|
| **Industry** | Enterprise / Knowledge Management |
| **Domain** | Information retrieval, RAG, semantic search |
| **Difficulty** | Intermediate |
| **Est. Timeline** | 6-10 weeks |
| **Team Size** | 4-5 engineers (2 ML/search, 1 backend, 1 data engineer) |

---

## 🎯 Problem Statement

### Business Context

**Company:** CorpLegal Inc. (500+ lawyers, 3M+ legal documents, 15TB of knowledge)
**Current Search:** Elasticsearch full-text search (BM25), 48% satisfaction rate
**Search Volume:** 12,000+ searches/day across 5 practice areas

### Pain Points

1. **Poor Recall** — Keyword search misses relevant documents due to vocabulary mismatch (e.g., search "termination clause" misses "end of agreement")
2. **No Semantic Understanding** — "Statute of limitations for breach of contract in California" returns documents about "limitations" unrelated to time
3. **Multi-hop Questions** — "Who is the plaintiff in the case where the defendant was Acme Corp in 2022?" requires multiple retrieval hops
4. **Freshness Requirements** — New documents must be searchable within 5 minutes of upload
5. **Hybrid Corpus** — 70% structured (contracts, clauses) and 30% unstructured (memos, emails, notes)
6. **Per-user Relevance** — Each lawyer has different specialties; search results should be personalized

### Success Criteria

| Metric | Target | Baseline (BM25) |
|--------|--------|----------------|
| **NDCG@10** | > 0.85 | 0.62 |
| **Recall@20** | > 0.90 | 0.65 |
| **Mean Reciprocal Rank (MRR)** | > 0.80 | 0.55 |
| **P99 Latency** | < 500ms | 200ms (already fast) |
| **User Satisfaction** | > 85% | 48% |
| **Index Freshness** | < 5 min | 60 min |
| **Zero-Result Rate** | < 1% | 12% |

---

## 🏗️ Solution Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         DOCUMENT INGESTION PIPELINE                                 │
│                                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │  PDF/    │  │  Text    │  │  Chunk   │  │  Embed   │  │  Index   │             │
│  │  Docx    │──▶│  Extract │──▶│  500 tok │──▶│  ada-002 │──▶│  Qdrant  │             │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘             │
│       │              │              │              │              │                │
│       │              ▼              │              │              │                │
│       │  ┌──────────────────────┐   │              │              │                │
│       │  │  Metadata Extract    │   │              │              │                │
│       │  │  (NER, dates,       │   │              │              │                │
│       │  │   parties, citations)│──┘              │              │                │
│       │  └──────────────────────┘                  │              │                │
│       │                                            ▼              │                │
│       │                    ┌──────────────────────────────┐       │                │
│       │                    │  Sparse Index (BM25)         │       │                │
│       │                    │  Elasticsearch               │◀──────┘                │
│       │                    └──────────────────────────────┘                        │
│       │                                                                           │
│       └───────────────────────────────────────────────────────────────────────────┘
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          QUERY PROCESSING PIPELINE                                   │
│                                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │  User    │  │  Query   │  │  Query   │  │  Hybrid  │  │  Re-     │             │
│  │  Query   │──▶│  Rewrite │──▶│  Enrich  │──▶│  Search  │──▶│  Ranker  │             │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └────┬─────┘             │
│                                                                │                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │                   │
│  │  Multi-  │  │  Final   │  │  Response│                     │                   │
│  │  Hop     │◀─│  Rerank  │◀─│  Generate│◀────────────────────┘                   │
│  │  Iterate │  └──────────┘  │  (LLM)   │                                         │
│  └──────────┘                └──────────┘                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Hybrid Search Detail

```
                         HYBRID SEARCH FUSION
                         
┌─────────────────────────────┐       ┌─────────────────────────────┐
│   DENSE RETRIEVAL (Qdrant)  │       │  SPARSE RETRIEVAL (ES BM25) │
│                             │       │                             │
│   Query Embedding ─────►    │       │  Tokenized Query ─────►    │
│   vector search (cosine)   │       │  inverted index search      │
│                             │       │                             │
│   Top-100 by similarity     │       │  Top-100 by TF-IDF score    │
└──────────────┬──────────────┘       └──────────────┬──────────────┘
               │                                     │
               └─────────────────┬───────────────────┘
                                 │
                                 ▼
               ┌─────────────────────────────────────┐
               │         RRF Fusion (Reciprocal       │
               │         Rank Fusion)                 │
               │                                     │
               │  score = Σ 1/(k + rank_dense)       │
               │         + 1/(k + rank_sparse)       │
               │                                     │
               │  Top-30 documents (k=60)            │
               └──────────────────┬──────────────────┘
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │       COHERE RERANK                  │
               │       (cross-encoder)                │
               │                                     │
               │  Score each doc against query        │
               │  Top-5 final results                 │
               └──────────────────────────────────────┘
```

### Multi-Hop Retrieval Flow

```
Query: "What was the settlement amount in the patent case 
        where the judge was Judge Chen?"

   Hop 1:          ┌─────────────────────────┐
  "patent case     │  Retrieve cases involving│
   Judge Chen"     │  Judge Chen → find case  │
                   │  IDs                     │
                   └────────────┬─────────────┘
                                │
                                ▼
   Hop 2:          ┌─────────────────────────┐
  "settlement      │  From case IDs, retrieve│
   amount"         │  settlement amounts     │
                   └────────────┬─────────────┘
                                │
                                ▼
   Final:          ┌─────────────────────────┐
                   │  Synthesize answer      │
                   │  "The settlement in     │
                   │  Acme v. Beta (2022)    │
                   │  was $12.5M"           │
                   └─────────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| **Vector Database** | Qdrant | 1.9 | Fast HNSW, filtering, on-prem |
| **Sparse Search** | Elasticsearch | 8.14 | BM25, mature, already in infra |
| **Embeddings** | text-embedding-3-large | OpenAI | 3072-d, SOTA quality |
| **Re-ranker** | Cohere Rerank v3 English | — | Cross-encoder precision |
| **LLM** | GPT-4-turbo / Claude 3.5 | — | Answer generation |
| **Document Parsing** | Unstructured + PDFPlumber + python-docx | 0.14 / 0.11 / 1.1 | Multi-format support |
| **Chunking** | LangChain text splitter + custom NLP splitter | 0.2 | Semantic chunk boundaries |
| **Orchestration** | LangChain + LangGraph | 0.2 / 0.1 | Retrieval chains, multi-hop |
| **Pipeline** | Airflow (batch), Kafka (real-time) | 2.8 / 3.6 | Document ingestion |
| **Monitoring** | LangSmith + Prometheus + Grafana | — | Trace retrieval quality |
| **Cache** | Redis | 7.x | Query caching |
| **Backend** | FastAPI + Celery | 0.111 / 5.4 | Async API |

### Installation

```bash
pip install qdrant-client==1.9.1 elasticsearch==8.14.0
pip install langchain==0.2.12 langchain-openai==0.1.21
pip install unstructured[pdf]==0.14.10 pdfplumber==0.11.1 python-docx==1.1.2
pip install cohere==5.6.4 tiktoken==0.7.0
pip install fastapi==0.111.1 uvicorn[standard]==0.30.1
pip install celery[redis]==5.4.0 apache-airflow==2.8.4
```

---

## ⚙️ Implementation Details

### 1. Document Ingestion Pipeline

```python
# src/ingestion/document_processor.py
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
import hashlib
from typing import List, Dict

class DocumentIngestionPipeline:
    """Ingest documents → extract text → chunk → embed → index."""

    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150

    def __init__(self, qdrant_client, es_client, embedding_model):
        self.qdrant = qdrant_client
        self.es = es_client
        self.embeddings = embedding_model

    def process_document(self, filepath: str, metadata: dict) -> List[str]:
        """Full ingestion: parse → chunk → embed → index both stores."""
        # Step 1: Parse document
        text, doc_metadata = self._parse(filepath)
        metadata.update(doc_metadata)

        # Step 2: Chunk intelligently
        chunks = self._semantic_chunk(text)
        chunk_metadata = [
            {**metadata, "chunk_id": i, "chunk_total": len(chunks)}
            for i in range(len(chunks))
        ]

        # Step 3: Generate embeddings
        embeddings = self.embeddings.embed_documents(chunks)

        # Step 4: Index to Qdrant (dense)
        qdrant_ids = self._index_to_qdrant(chunks, embeddings, chunk_metadata)

        # Step 5: Index to Elasticsearch (sparse)
        es_ids = self._index_to_elasticsearch(chunks, chunk_metadata)

        return {"qdrant_ids": qdrant_ids, "es_ids": es_ids}

    def _parse(self, filepath: str) -> tuple[str, dict]:
        """Parse document based on file type."""
        import os
        ext = os.path.splitext(filepath)[1].lower()
        metadata = {"source": filepath, "filetype": ext}

        if ext == ".pdf":
            elements = partition_pdf(filepath)
            text = "\n\n".join([str(e) for e in elements])
            metadata["pages"] = len([e for e in elements if "PageNumber" in str(e.metadata)])
        elif ext in (".docx", ".doc"):
            elements = partition_docx(filepath)
            text = "\n\n".join([str(e) for e in elements])
        else:
            with open(filepath, "r") as f:
                text = f.read()

        return text, metadata

    def _semantic_chunk(self, text: str) -> List[str]:
        """Chunk at section boundaries when possible, fall back to token count."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=lambda t: len(t.split()),
        )
        return splitter.split_text(text)

    def _index_to_qdrant(
        self, chunks: List[str], embeddings: List, metadata: List[Dict]
    ) -> List[str]:
        """Index chunks with embeddings to Qdrant vector store."""
        from qdrant_client.http.models import PointStruct

        ids = [hashlib.md5(chunk.encode()).hexdigest() for chunk in chunks]
        points = [
            PointStruct(
                id=id,
                vector=emb,
                payload={"text": chunk, **meta}
            )
            for id, chunk, emb, meta in zip(ids, chunks, embeddings, metadata)
        ]
        self.qdrant.upsert(
            collection_name="enterprise_docs",
            points=points,
        )
        return ids

    def _index_to_elasticsearch(
        self, chunks: List[str], metadata: List[Dict]
    ) -> List[str]:
        """Index chunks to Elasticsearch for BM25 search."""
        from elasticsearch.helpers import bulk

        actions = [
            {
                "_index": "enterprise_docs",
                "_id": hashlib.md5(chunk.encode()).hexdigest(),
                "_source": {"text": chunk, **meta},
            }
            for chunk, meta in zip(chunks, metadata)
        ]
        success, errors = bulk(self.es, actions)
        return [a["_id"] for a in actions]
```

### 2. Hybrid Search with RRF Fusion

```python
# src/search/hybrid_search.py
import numpy as np
from typing import List, Dict, Optional

class HybridSearchEngine:
    """Combines dense (vector) and sparse (BM25) retrieval with RRF fusion."""

    def __init__(self, qdrant_client, es_client, embedding_model, reranker=None):
        self.qdrant = qdrant_client
        self.es = es_client
        self.embeddings = embedding_model
        self.reranker = reranker  # Cohere Rerank or similar
        self.RRF_K = 60  # Reciprocal Rank Fusion constant

    async def search(
        self,
        query: str,
        top_k_dense: int = 100,
        top_k_sparse: int = 100,
        top_k_final: int = 10,
        filters: Optional[dict] = None,
    ) -> List[Dict]:
        """Execute hybrid search with RRF fusion and optional re-ranking."""

        # Step 1: Dense search (vector similarity)
        query_embedding = self.embeddings.embed_query(query)
        dense_results = self._dense_search(
            query_embedding, top_k_dense, filters
        )

        # Step 2: Sparse search (BM25)
        sparse_results = self._sparse_search(query, top_k_sparse, filters)

        # Step 3: RRF Fusion
        fused_results = self._reciprocal_rank_fusion(
            dense_results, sparse_results, top_k=top_k_final * 3
        )

        # Step 4: Re-ranking (optional, adds ~100ms)
        if self.reranker and len(fused_results) > top_k_final:
            fused_results = await self._rerank(query, fused_results, top_k_final)

        return fused_results[:top_k_final]

    def _dense_search(
        self, query_vec: List[float], top_k: int, filters: Optional[dict]
    ) -> List[Dict]:
        """Query Qdrant vector store."""
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue

        query_filter = None
        if filters:
            query_filter = Filter(
                must=[
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filters.items()
                ]
            )

        results = self.qdrant.search(
            collection_name="enterprise_docs",
            query_vector=query_vec,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            {
                "id": r.id,
                "text": r.payload["text"],
                "score": r.score,
                "metadata": r.payload,
            }
            for r in results
        ]

    def _sparse_search(
        self, query: str, top_k: int, filters: Optional[dict]
    ) -> List[Dict]:
        """Query Elasticsearch with BM25."""
        must_clause = [{"match": {"text": query}}]
        if filters:
            for k, v in filters.items():
                must_clause.append({"term": {k: v}})

        response = self.es.search(
            index="enterprise_docs",
            query={"bool": {"must": must_clause}},
            size=top_k,
        )

        return [
            {
                "id": hit["_id"],
                "text": hit["_source"]["text"],
                "score": hit["_score"],
                "metadata": hit["_source"],
            }
            for hit in response["hits"]["hits"]
        ]

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        top_k: int = 30,
    ) -> List[Dict]:
        """Fuse ranked lists using Reciprocal Rank Fusion."""
        scores = {}  # doc_id -> combined score

        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.RRF_K + rank + 1)
            scores[doc_id + "_text"] = doc["text"]
            scores[doc_id + "_meta"] = doc["metadata"]

        for rank, doc in enumerate(sparse_results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.RRF_K + rank + 1)
            if doc_id + "_text" not in scores:
                scores[doc_id + "_text"] = doc["text"]
                scores[doc_id + "_meta"] = doc["metadata"]

        # Sort by combined score
        doc_ids = [k for k in scores.keys() if not k.endswith(("_text", "_meta"))]
        doc_ids.sort(key=lambda x: scores[x], reverse=True)

        return [
            {
                "id": doc_id,
                "text": scores.get(doc_id + "_text", ""),
                "score": scores[doc_id],
                "metadata": scores.get(doc_id + "_meta", {}),
            }
            for doc_id in doc_ids[:top_k]
        ]

    async def _rerank(
        self, query: str, documents: List[Dict], top_k: int = 5
    ) -> List[Dict]:
        """Re-rank using Cohere cross-encoder."""
        reranked = self.reranker.rerank(
            query=query,
            documents=[d["text"] for d in documents],
            model="rerank-english-v3.0",
            top_n=min(top_k, len(documents)),
        )
        return [
            {**documents[r.index], "rerank_score": r.relevance_score}
            for r in reranked.results
        ]
```

### 3. Multi-Hop Retrieval with LangGraph

```python
# src/search/multi_hop.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

class RetrievalState(TypedDict):
    query: str
    sub_queries: List[str]
    current_hop: int
    max_hops: int
    retrieved_docs: List[dict]
    answers: List[str]
    final_answer: Optional[str]

class MultiHopRetriever:
    """Breaks complex queries into sub-questions and retrieves iteratively."""

    def __init__(self, llm, search_engine):
        self.llm = llm
        self.search = search_engine
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RetrievalState)

        workflow.add_node("decompose_query", self._decompose_query)
        workflow.add_node("retrieve_hop", self._retrieve_for_hop)
        workflow.add_node("extract_answer", self._extract_answer)
        workflow.add_node("check_complete", self._check_complete)
        workflow.add_node("synthesize_final", self._synthesize_final)

        workflow.set_entry_point("decompose_query")
        workflow.add_edge("decompose_query", "retrieve_hop")
        workflow.add_edge("retrieve_hop", "extract_answer")
        workflow.add_edge("extract_answer", "check_complete")
        workflow.add_conditional_edges(
            "check_complete",
            self._decide_next,
            {
                "next_hop": "retrieve_hop",
                "finalize": "synthesize_final",
            }
        )
        workflow.add_edge("synthesize_final", END)

        return workflow.compile()

    async def retrieve(self, query: str, max_hops: int = 3) -> dict:
        initial_state = RetrievalState(
            query=query,
            sub_queries=[],
            current_hop=0,
            max_hops=max_hops,
            retrieved_docs=[],
            answers=[],
            final_answer=None,
        )
        result = await self.graph.ainvoke(initial_state)
        return result

    async def _decompose_query(self, state: RetrievalState):
        """Decompose complex query into retrieval sub-questions."""
        prompt = f"""Decompose the following question into {state['max_hops']} 
        sub-questions that need to be answered in sequence. Each sub-question 
        should build on the previous answer.

        Question: {state['query']}

        Return as a numbered list of sub-questions:"""
        response = await self.llm.ainvoke(prompt)
        sub_queries = [
            line.strip().split(". ", 1)[1]
            for line in response.content.strip().split("\n")
            if ". " in line
        ]
        state["sub_queries"] = sub_queries
        return state

    async def _retrieve_for_hop(self, state: RetrievalState):
        """Retrieve documents for current hop's sub-query."""
        hop_idx = state["current_hop"]
        query = state["sub_queries"][hop_idx] if hop_idx < len(state["sub_queries"]) else state["query"]

        # Use previous answers as context
        context = "\n".join(state["answers"]) if state["answers"] else ""
        enriched_query = f"{query}\nContext from previous answers: {context}" if context else query

        docs = await self.search.search(enriched_query, top_k=5)
        state["retrieved_docs"].extend(docs)
        return state

    async def _extract_answer(self, state: RetrievalState):
        """Extract answer for current hop from retrieved docs."""
        context = "\n\n".join([d["text"] for d in state["retrieved_docs"][-5:]])
        prompt = f"""Based on the following context, answer the sub-question.

        Context: {context}

        Sub-Question: {state['sub_queries'][state['current_hop']]}
        Short Answer:"""
        response = await self.llm.ainvoke(prompt)
        state["answers"].append(response.content.strip())
        return state

    async def _check_complete(self, state: RetrievalState):
        return state

    def _decide_next(self, state: RetrievalState) -> str:
        if state["current_hop"] < len(state["sub_queries"]) - 1:
            state["current_hop"] += 1
            return "next_hop"
        return "finalize"

    async def _synthesize_final(self, state: RetrievalState):
        context = "\n".join(state["answers"])
        prompt = f"""Synthesize a comprehensive answer from the following 
        intermediate answers.

        Original Question: {state['query']}
        Intermediate Answers: {context}

        Final Answer:"""
        response = await self.llm.ainvoke(prompt)
        state["final_answer"] = response.content.strip()
        return state
```

### 4. Query Rewriting & Expansion

```python
# src/search/query_processor.py
from typing import List, Optional

class QueryProcessor:
    """Rewrite, expand, and enrich search queries."""

    def __init__(self, llm):
        self.llm = llm

    async def rewrite(self, query: str, user_history: Optional[List[str]] = None) -> str:
        """Rewrite raw user query into optimal search query."""
        history_context = ""
        if user_history:
            history_context = f"User's recent queries: {'; '.join(user_history[-3:])}"

        prompt = f"""Rewrite the following search query to improve retrieval. 
        Expand abbreviations, correct typos, add domain-specific synonyms.

        {history_context}
        Original query: {query}

        Rewritten query (only output the query text):"""
        response = await self.llm.ainvoke(prompt)
        return response.content.strip()

    async def generate_hypothetical_answer(
        self, query: str, num_docs: int = 3
    ) -> str:
        """Generate a hypothetical answer (HyDE) for better retrieval."""
        prompt = f"""Given the search query, write a short hypothetical document 
        that would perfectly answer this query.

        Query: {query}

        Hypothetical document:"""
        response = await self.llm.ainvoke(prompt)
        return response.content.strip()

    def extract_entities(self, query: str) -> dict:
        """Extract known entities for structured filters."""
        import re
        entities = {
            "dates": re.findall(r'\b(19|20)\d{2}\b', query),
            "monetary": re.findall(r'\$\d+(?:\.\d{2})?[MKMB]?', query),
            "case_numbers": re.findall(r'\b\d{2,4}-[A-Z]{2,5}-\d{1,5}\b', query),
        }
        return {k: v for k, v in entities.items() if v}
```

### 5. API Server

```python
# src/serving/api.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict

app = FastAPI(title="Enterprise RAG Search API", version="1.0.0")

# Initialize components (in production: dependency injection)
search_engine = None  # Initialized at startup
query_processor = None
multi_hop = None
llm = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    filters: Optional[Dict[str, str]] = None
    multi_hop: bool = False
    generate_answer: bool = False

class SearchResponse(BaseModel):
    query: str
    rewritten_query: Optional[str]
    results: List[Dict]
    answer: Optional[str]
    latency_ms: float

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    import time
    start = time.time()

    # Rewrite query
    rewritten = await query_processor.rewrite(request.query)

    # Multi-hop or direct search
    if request.multi_hop:
        retrieval_result = await multi_hop.retrieve(request.query)
        docs = retrieval_result["retrieved_docs"]
        answer = retrieval_result["final_answer"]
    else:
        docs = await search_engine.search(
            rewritten,
            top_k=request.top_k,
            filters=request.filters,
        )
        # Optional LLM answer generation
        if request.generate_answer:
            context = "\n\n".join([d["text"] for d in docs[:5]])
            response = await llm.ainvoke(
                f"Context: {context}\n\nQuestion: {request.query}\n\nAnswer:"
            )
            answer = response.content
        else:
            answer = None

    latency = (time.time() - start) * 1000

    return SearchResponse(
        query=request.query,
        rewritten_query=rewritten,
        results=docs,
        answer=answer,
        latency_ms=latency,
    )

@app.get("/search/stream")
async def search_stream(query: str = Query(...)):
    """Server-sent events for streaming search results."""
    from fastapi.responses import StreamingResponse
    import json
    import asyncio

    async def event_stream():
        # Stream retrieval + generation progress
        rewritten = await query_processor.rewrite(query)
        yield f"data: {json.dumps({'event': 'rewrite', 'query': rewritten})}\n\n"
        await asyncio.sleep(0.05)

        docs = await search_engine.search(rewritten, top_k=5)
        yield f"data: {json.dumps({'event': 'retrieval', 'results': docs})}\n\n"

        # Generate answer with streaming tokens
        context = "\n\n".join([d["text"] for d in docs])
        response = await llm.ainvoke(
            f"Context: {context}\n\nQuestion: {query}\n\nAnswer:",
            stream=True
        )
        async for token in response:
            yield f"data: {json.dumps({'event': 'token', 'token': token})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

---

## 📊 Metrics & Results

### Retrieval Performance (Offline Evaluation)

| Metric | BM25 Only | Dense Only | Hybrid (RRF) | Hybrid + Rerank |
|--------|-----------|------------|--------------|-----------------|
| **NDCG@10** | 0.62 | 0.74 | 0.82 | **0.89** |
| **Recall@20** | 0.65 | 0.78 | 0.88 | **0.92** |
| **MRR** | 0.55 | 0.68 | 0.76 | **0.83** |
| **Precision@5** | 0.52 | 0.63 | 0.71 | **0.79** |

### End-to-End System Performance

| Metric | Baseline (ES Only) | New System | Delta |
|--------|-------------------|------------|-------|
| **NDCG@10** | 0.62 | 0.89 | +0.27 |
| **Recall@20** | 0.65 | 0.92 | +0.27 |
| **Zero-Result Rate** | 12% | 0.3% | -11.7 pp |
| **P50 Latency** | 85ms | 240ms | +155ms |
| **P95 Latency** | 180ms | 430ms | +250ms |
| **P99 Latency** | 350ms | 680ms* | +330ms |
| **User Satisfaction** | 48% | 92% | +44 pp |
| **Time Saved per Lawyer/day** | — | 45 min | — |

*\*P99 with re-ranker enabled. Without re-ranker: 450ms.*

### Business Impact

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| **Search Success Rate** | 48% | 92% | +44 pp |
| **Avg. Time per Search** | 8 min | 2 min | -75% |
| **Billable Hours Recovered** | — | 1,875 hrs/month | — |
| **Revenue from Recovered Time** | $11.2M/yr | $14.5M/yr | +$3.3M |
| **IT Support Tickets (search)** | 450/month | 45/month | -90% |
| **Document Re-find Rate** | 35% (lost) | 95% | +60 pp |

---

## 💡 Lessons Learned

### ✅ What Went Well

1. **Hybrid retrieval is mandatory** — Dense + sparse together beat either alone by 15-20% NDCG. RRF fusion with K=60 was optimal.

2. **Re-ranking matters most at the very top** — Re-ranker improved NDCG@10 by 0.07 but took 150ms. Worth it for the first page of results.

3. **Multi-hop for complex questions** — 15% of queries required multi-hop. The LangGraph-based decomposer correctly split 89% of complex queries.

4. **User feedback loop** — Implicit signals (click-through, dwell time, copy-paste) improved ranking by 0.03 NDCG/month via online learning.

### ❌ What Went Wrong

1. **Chunk size was too big initially** — Started with 1500-token chunks; recall was poor for specific questions (quotes, clauses). Settled at 500 tokens with 100 overlap.

2. **Embedding dimension mismatch** — Initially used ada-002 (1536-d) but Qdrant HNSW performed better with 768-d. Used dimensionality reduction.

3. **Cold cache on new documents** — Frequently accessed docs were slow on first query after indexing. Implemented proactive warm-up for top-1000 documents.

4. **Excessive multi-hop for simple queries** — Multi-hop mode was triggered too often. Added a "query complexity classifier" — only decompose if > 3 entities present.

### ⚠️ Critical Warnings

```
! WARNING: Indexing pipeline must handle PDFs with scanned text (OCR fallback).
! WARNING: Monitor query latency budget — re-ranker adds 100-200ms.
! WARNING: RRF K parameter significantly affects results — tune on your data.
! WARNING: Embeddings drift as language evolves — schedule quarterly re-index.
```

### Best Practices

```
Chunk Size:     500-800 tokens (domain-dependent)
Chunk Overlap:  100-200 tokens
Embedding:      text-embedding-3-large (3072-d, reduce to 768-d for speed)
RRF K:          60 (tune between 1-100)
Top-K Dense:    100
Top-K Sparse:   100
Re-ranker:      Only for final top-30 (not all candidates)
```

---

## 📁 Reusable Project Template

### Directory Structure

```
TEMPLATE-RAG-SEARCH/
├── README.md
├── Makefile
├── requirements.txt
├── docker-compose.yml
├── .env.example
│
├── configs/
│   ├── config.yaml
│   ├── search_config.yaml
│   ├── chunking.yaml
│   ├── embedding.yaml
│   └── collections.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   ├── chunker.py
│   │   ├── metadata_extractor.py
│   │   └── indexer.py
│   │
│   ├── search/
│   │   ├── __init__.py
│   │   ├── hybrid_search.py
│   │   ├── dense_search.py
│   │   ├── sparse_search.py
│   │   ├── multi_hop.py
│   │   ├── reranker.py
│   │   └── query_processor.py
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── schemas.py
│   │   ├── middleware.py
│   │   └── streaming.py
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── quality_monitor.py
│   │   └── alerts.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── cache.py
│       └── evaluator.py
│
├── tests/
│   ├── unit/
│   │   ├── test_search.py
│   │   ├── test_chunker.py
│   │   └── test_reranker.py
│   ├── integration/
│   │   ├── test_qdrant.py
│   │   ├── test_elasticsearch.py
│   │   └── test_end_to_end.py
│   └── fixtures/
│       ├── sample_docs/
│       └── queries.json
│
├── notebooks/
│   ├── 01-search-quality-eval.ipynb
│   ├── 02-hybrid-vs-dense-vs-sparse.ipynb
│   └── 03-multi-hop-evaluation.ipynb
│
├── scripts/
│   ├── ingest_documents.py
│   ├── evaluate_search.py
│   ├── warm_cache.sh
│   └── reindex_all.sh
│
├── k8s/
│   ├── namespace.yaml
│   ├── deployment-api.yaml
│   ├── deployment-ingestion-worker.yaml
│   ├── service.yaml
│   └── hpa.yaml
│
└── docs/
    ├── architecture.md
    ├── search_config_guide.md
    └── evaluation_methodology.md
```

### Getting Started

```bash
# 1. Copy template
cp -r TEMPLATE-RAG-SEARCH ~/my-rag-search
cd ~/my-rag-search

# 2. Start infrastructure
docker-compose up -d qdrant elasticsearch redis

# 3. Install dependencies
make install

# 4. Ingest sample documents
python scripts/ingest_documents.py --dir ./tests/fixtures/sample_docs/

# 5. Run evaluation
python scripts/evaluate_search.py

# 6. Start API
uvicorn src.serving.api:app --reload

# 7. Test search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "termination clause for breach of contract", "top_k": 5}'
```

---

## 📚 References & Further Reading

### Academic Papers
- Karpukhin et al. (2020) — "Dense Passage Retrieval for Open-Domain Question Answering" — [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
- Robertson & Zaragoza (2009) — "The Probabilistic Relevance Framework: BM25 and Beyond" — [Foundations and Trends in IR](https://doi.org/10.1561/1500000019)
- Cormack et al. (2009) — "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" — [SIGIR 2009](https://doi.org/10.1145/1571941.1572114)
- Gao et al. (2023) — "Retrieval-Augmented Generation for Large Language Models: A Survey" — [arXiv:2312.10997](https://arxiv.org/abs/2312.10997)

### Tools & Documentation
- Qdrant: https://qdrant.tech/documentation/
- Elasticsearch: https://www.elastic.co/guide/en/elasticsearch/reference/current
- Cohere Rerank: https://docs.cohere.com/docs/rerank
- LangChain Retrieval: https://python.langchain.com/docs/modules/data_connection/retrieval/

---

> **Next**: [07-AI-Code-Assistant.md](07-AI-Code-Assistant.md) — Internal AI code assistant for dev teams with fine-tuned CodeLlama and RAG.

---
**See also:**
- [07 — RAG and Retrieval Research: The Frontier (2025–2026)](07-Emerging/17-Research-Frontiers-2026/07-RAG-Retrieval-Research.md)
- [Advanced RAG (Retrieval-Augmented Generation)](04-RAG/02-Advanced-RAG.md)
- [AI-Powered Search: Beyond Traditional RAG](06-Advanced/11-AI-Powered-Search.md)

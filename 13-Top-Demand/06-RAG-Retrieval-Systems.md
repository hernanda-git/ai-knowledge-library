# RAG & Retrieval Systems

> **Last Updated:** June 2026  
> **Category:** Top Demand — Current Market Snapshot  
> **Cross-References:** 02-AI-Agent-Development.md, 04-Multimodal-AI.md, 07-Fine-Tuning-Custom-Models.md, 10-Real-Time-AI-Systems.md

---

## Table of Contents

1. [Market Context & Demand](#1-market-context--demand)
2. [Indexing Strategies](#2-indexing-strategies)
   - 2.1 Hierarchical Indexing
   - 2.2 Sliding Window Indexing
   - 2.3 Semantic Chunking
   - 2.4 ColBERT-style Late Interaction Indexing
   - 2.5 Multimodal Indexing
3. [Retrieval Methods](#3-retrieval-methods)
   - 3.1 Dense Retrieval (Embedding-Based)
   - 3.2 Sparse Retrieval (BM25, SPLADE)
   - 3.3 Hybrid Retrieval
   - 3.4 Late Interaction / Token-Level Retrieval
4. [Re-Ranking](#4-re-ranking)
   - 4.1 Cross-Encoders
   - 4.2 Cohere Rerank
   - 4.3 ColBERT Re-Rank
   - 4.4 LLM-as-Judge Re-Ranking
5. [Query Transformations](#5-query-transformations)
   - 5.1 Query Rewriting
   - 5.2 Query Decomposition
   - 5.3 Hypothetical Document Embeddings (HyDE)
   - 5.4 Multi-Query Retrieval
6. [Agentic RAG](#6-agentic-rag)
   - 6.1 Self-RAG
   - 6.2 Corrective RAG (CRAG)
   - 6.3 Adaptive RAG
   - 6.4 Tool-Augmented Retrieval
7. [Evaluation Metrics & Frameworks](#7-evaluation-metrics--frameworks)
   - 7.1 RAGAS
   - 7.2 TruLens
   - 7.3 ARES
   - 7.4 Custom Evaluation Pipelines
8. [Vector Database Comparison](#8-vector-database-comparison)
9. [Production RAG Architecture](#9-production-rag-architecture)
10. [Future Directions](#10-future-directions)

---

## 1. Market Context & Demand

RAG (Retrieval-Augmented Generation) has evolved from a simple "retrieve + generate" pattern into a sophisticated ecosystem of retrieval strategies, indexing approaches, and agent-driven pipelines.

**Market dynamics (June 2026):**
- RAG is the most deployed LLM architecture — 85%+ of production systems use some form of retrieval
- RAG platform market: $8.3B (growing 45% YoY)
- Average production RAG system: 3.2 retrieval stages, 2 re-rank passes
- Vector database market: $4.5B (Pinecone, Weaviate, Chroma, Qdrant)

**Why RAG dominates:**
- **Hallucination reduction** — RAG grounds LLM outputs in retrieved context (hallucination: 15-20% → 3-5%)
- **Freshness** — No retraining needed for new information
- **Control** — Document-level access control, audit trails
- **Cost** — Much cheaper than fine-tuning for knowledge-intensive tasks

---

## 2. Indexing Strategies

### 2.1 Hierarchical Indexing

Documents are chunked at multiple granularities for efficient retrieval:

```yaml
hierarchical_index:
  levels:
    - level: document
      chunk_size: full_document
      description: Whole document summary embedding
      metadata: [title, author, date, category]
    
    - level: section
      chunk_size: 2000-5000 tokens
      description: Section-level chunks with structural metadata
      stride: 0 (non-overlapping)
      metadata: [section_title, heading_level, page_range]
    
    - level: passage
      chunk_size: 256-512 tokens
      description: Fine-grained retrievable units
      stride: 128 (50% overlap)
      metadata: [parent_section, document_id]

  retrieval_logic:
    - First retrieve top-k documents via coarse embeddings
    - Then retrieve relevant passages within those documents
    - Optionally, re-retrieve at section level for parent context
```

**Benefits:**
- Efficient retrieval across large corpora
- Preserves document structure
- Enables "zooming in" from document to passage level
- 30-50% better recall than single-level chunking

### 2.2 Sliding Window Indexing

Overlapping chunks preserve context at boundaries:

```python
def sliding_window_chunks(text, chunk_size=512, overlap=128):
    """Create overlapping chunks with metadata."""
    tokens = tokenize(text)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = tokens[start:end]
        
        # Track overlap with previous chunk
        overlap_start = max(0, start - overlap)
        context_window = tokens[overlap_start:end]
        
        chunks.append({
            "text": detokenize(chunk),
            "context_text": detokenize(context_window),
            "start_idx": start,
            "end_idx": end,
            "chunk_id": len(chunks)
        })
        
        start += chunk_size - overlap
    
    return chunks
```

**Optimal parameters (2026 guidance):**
- **Chunk size:** 256-1024 tokens (balance between specificity and context)
- **Overlap:** 10-25% of chunk size
- **Chunk count:** Libraries with 10M+ chunks need sharded indices
- **For code:** Smaller chunks (128-256 tokens), language-aware boundaries

### 2.3 Semantic Chunking

Rather than fixed token/character boundaries, semantic chunking splits at natural topic boundaries:

```python
# Semantic chunking using embedding similarity
def semantic_chunk(text, min_chunk=256, max_chunk=1024):
    sentences = split_sentences(text)
    sentence_embeddings = embedder.encode(sentences)
    
    chunks = []
    current_chunk = []
    current_len = 0
    
    for i, (sent, emb) in enumerate(zip(sentences, sentence_embeddings)):
        current_chunk.append(sent)
        current_len += len(tokenize(sent))
        
        # Check if this sentence starts a new topic
        if current_len >= min_chunk and i < len(sentences) - 1:
            similarity = cosine_similarity(emb, sentence_embeddings[i+1])
            if similarity < 0.6 or current_len >= max_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
```

**Tools for semantic chunking (2026):**
- **Semantic Chunkers** (LangChain) — Embedding-based boundary detection
- **ChunkViz** — Visualization tool for chunk boundary analysis
- **Unstructured** — Document-aware chunking (PDFs, HTML, markdown)
- **Jina Segmenter** — AI-powered document segmentation

### 2.4 ColBERT-style Late Interaction Indexing

ColBERT's late interaction model encodes queries and documents independently, then computes fine-grained scores via MaxSim:

```
Query: [q1, q2, ..., qm]    Doc: [d1, d2, ..., dn]
       ↓                           ↓
Query Encoder                  Doc Encoder
       ↓                           ↓
Query Vectors: [q1',...,qm']  Doc Vectors: [d1',...,dn']
              ↘ ↙
         MaxSim: Σ max(q_i · d_j)
```

**Indexing requirements:**
- Each document token stored as a vector (contextualized word embedding)
- Compressed via PLAID (Pre-computed Lookups for late Interaction Disambiguation)
- Index size: ~50GB per million documents (versus ~2GB for dense single-vector)

### 2.5 Multimodal Indexing

For documents with images, tables, and text:

```yaml
multimodal_index:
  modalities:
    text: 
      encoder: text-embedding-3-large
      index_type: dense_hnsw
    image:
      encoder: clip-vit-large
      index_type: dense_hnsw    
    table:
      encoder: table-transformer
      index_type: hybrid (dense + structured)
  
  storage:
    primary: vector_db (Pinecone/Weaviate)
    metadata_db: PostgreSQL
    blob_storage: S3-compatible (for full images/documents)
```

See 04-Multimodal-AI.md for more on multimodal retrieval.

---

## 3. Retrieval Methods

### 3.1 Dense Retrieval (Embedding-Based)

Dense retrieval encodes queries and documents into fixed-size vectors and searches via approximate nearest neighbor (ANN).

**Popular embedding models (June 2026):**

| Model | Dimensions | Max Tokens | Languages | MTEB Score | Provider |
|-------|-----------|------------|-----------|------------|----------|
| text-embedding-3-large | 3072 | 8191 | 100+ | 64.6 | OpenAI |
| Cohere Embed v3 | 4096 | 8192 | 100+ | 64.2 | Cohere |
| BGE-M3 | 1024 | 8192 | 100+ | 63.8 | BAAI |
| E5-Mistral-7B | 4096 | 32768 | 50+ | 65.1 | Microsoft |
| instructor-xl | 768 | 512 | 10 | 61.4 | HKU |
| voyage-large-2 | 1536 | 32000 | 30+ | 63.5 | Voyage |

**ANN algorithms:**

| Algorithm | Index Build | Query Latency | Recall@10 | Memory Usage |
|-----------|-------------|---------------|-----------|--------------|
| HNSW | Medium | <5ms | 0.95 | High (2x vector size) |
| IVF-PQ | Fast | <10ms | 0.90 | Low (0.3x vector size) |
| DiskANN | Very Fast | <20ms | 0.92 | Disk-based |
| ScaNN | Slow (optimized) | <3ms | 0.97 | High (2x vector size) |

### 3.2 Sparse Retrieval (BM25, SPLADE)

Sparse retrieval uses keyword matching — still essential for many use cases:

**BM25 (Okapi BM25):**
- Term frequency × inverse document frequency
- No training required
- Excellent for exact match scenarios
- Weak on semantic similarity

**SPLADE (Sparse Lexical and Expansion):**
- Neural extension of BM25
- Encodes queries and documents into sparse vectors (vocabulary-sized)
- Combines semantic expansion with exact matching
- 15-25% better recall than BM25

**When to use sparse:**
- High-precision requirements (legal, medical)
- Rare or domain-specific terminology
- When data distribution differs from embedding model training data
- As a complement to dense retrieval (hybrid)

### 3.3 Hybrid Retrieval

Combining dense and sparse retrieval gives the best of both worlds:

```python
class HybridRetriever:
    def __init__(self, dense_weight=0.7, sparse_weight=0.3):
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
    
    def search(self, query, k=10):
        # Parallel retrieval
        dense_results = self.dense.search(query, k=k*2)
        sparse_results = self.sparse.search(query, k=k*2)
        
        # Reciprocal Rank Fusion (RFF)
        scores = {}
        for rank, (doc, _) in enumerate(dense_results):
            scores[doc.id] = self.dense_weight * (1 / (rank + 60))
        for rank, (doc, _) in enumerate(sparse_results):
            scores[doc.id] = scores.get(doc.id, 0) + \
                             self.sparse_weight * (1 / (rank + 60))
        
        # Sort and return top-k
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [doc_id for doc_id, _ in ranked]
```

**Hybrid weighting strategies:**
- **Static weights** — Dense 0.7 + Sparse 0.3 (general purpose)
- **Query-dependent** — Classify query type, adjust weights
- **Adaptive** — Learn weights from user feedback

### 3.4 Late Interaction / Token-Level Retrieval

ColBERT-style late interaction enables fine-grained matching:

**Advantages over dense:**
- Token-level scoring preserves fine-grained relevance signals
- Better out-of-domain generalization
- Handles multi-vector queries naturally

**Disadvantages:**
- Higher storage cost (50GB vs 2GB per million docs)
- Slower query time (needs MaxSim computation)
- More complex infrastructure

---

## 4. Re-Ranking

Re-ranking is the second pass that refines initial retrieval results with more expensive, higher-quality models.

### 4.1 Cross-Encoders

Cross-encoders jointly encode query + document for relevance scoring:

```python
from sentence_transformers import CrossEncoder

# Load a cross-encoder model
model = CrossEncoder("cross-encoder/ms-marco-electra-base")

# Re-rank initial results
query = "What is RAG?"
initial_results = retriever.search(query, k=100)

# Score all pairs
pairs = [(query, doc.text) for doc in initial_results]
scores = model.predict(pairs)

# Re-rank by cross-encoder score
top_results = [doc for _, doc in sorted(
    zip(scores, initial_results), key=lambda x: -x[0]
)][:10]
```

**Cross-encoder models (2026):**

| Model | Parameters | Speed (pairs/s) | MRR@10 | Use Case |
|-------|-----------|-----------------|--------|----------|
| ms-marco-MiniLM-L6-v2 | 22M | 5000 | 39.0 | General |
| BGE-reranker-v2 | 560M | 500 | 44.5 | High-accuracy |
| Cohere Rerank v3 | - (API) | 2000 | 44.0 | Managed |
| Jina-reranker-v2 | 560M | 600 | 43.2 | Multilingual |

### 4.2 Cohere Rerank

Cohere's managed reranking API is widely used:

```python
import cohere
co = cohere.Client("api_key")

results = co.rerank(
    model="rerank-v3.5",
    query="What is retrieval augmented generation?",
    documents=["RAG combines retrieval with generation...", ...],
    top_n=10,
    relevance_threshold=0.3  # Filter low-relevance results
)
```

### 4.3 ColBERT Re-Rank

ColBERT can be used for re-ranking by computing MaxSim scores for query-document pairs:

```python
# ColBERT re-rank (simplified)
def colbert_rerank(query, documents, colbert_model):
    Q = colbert_model.encode_query(query)  # [num_q_tokens, dim]
    
    scores = []
    for doc in documents:
        D = colbert_model.encode_doc(doc)  # [num_d_tokens, dim]
        # MaxSim: for each query token, find max similarity
        maxsim = torch.max(Q @ D.T, dim=-1).values
        score = maxsim.sum().item()
        scores.append(score)
    
    return sorted(zip(documents, scores), key=lambda x: -x[1])
```

### 4.4 LLM-as-Judge Re-Ranking

Using an LLM to evaluate relevance:

```python
def llm_rerank(query, documents, llm):
    scored_docs = []
    for doc in documents:
        prompt = f"""On a scale of 1-5, rate how relevant this document is to the query.
        
Query: {query}
Document: {doc.text[:500]}...

Relevance score (1-5):"""
        
        score = extract_score(llm.generate(prompt))
        scored_docs.append((doc, score))
    
    return sorted(scored_docs, key=lambda x: -x[1])[:5]
```

---

## 5. Query Transformations

### 5.1 Query Rewriting

Transform the user's raw query to improve retrieval:

```python
def rewrite_query(user_query, llm):
    prompt = f"""Rewrite the following search query to be more effective for retrieval. 
Add relevant keywords, clarify ambiguous terms, and expand acronyms.
Keep it concise (under 50 words).

Original query: {user_query}
Rewritten query:"""
    
    rewritten = llm.generate(prompt)
    return rewritten
```

**When to rewrite:**
- User queries are short or ambiguous ("it", "that", "the thing")
- Queries lack domain-specific terminology
- Language mismatch between query and documents

### 5.2 Query Decomposition

Break complex queries into simpler sub-queries:

```python
def decompose_query(complex_query, llm):
    prompt = f"""Break this complex question into simpler sub-questions that can be answered independently:

Question: {complex_query}

Sub-questions:
1."""
    
    sub_questions = llm.generate(prompt)
    return parse_list(sub_questions)

# Then retrieve answers for each sub-question and synthesize
```

### 5.3 Hypothetical Document Embeddings (HyDE)

HyDE generates a hypothetical ideal document from the query, then uses that document's embedding for retrieval:

```python
def hyde_search(query, llm, embedder, retriever):
    # Generate hypothetical document
    prompt = f"Write a paragraph that would perfectly answer: {query}"
    hypothetical_doc = llm.generate(prompt)
    
    # Embed the hypothetical document
    hyde_embedding = embedder.encode(hypothetical_doc)
    
    # Search with the HyDE embedding
    results = retriever.search_by_embedding(hyde_embedding, k=10)
    return results
```

### 5.4 Multi-Query Retrieval

Generate multiple query variants and merge results:

```python
def multi_query_retrieval(original_query, llm, retriever, n_queries=5):
    queries = [original_query]
    
    # Generate query variants
    prompt = f"Generate {n_queries-1} alternative phrasings of this query: {original_query}"
    variants = llm.generate(prompt)
    queries.extend(parse_variants(variants))
    
    # Retrieve for each query
    all_results = []
    for q in queries:
        results = retriever.search(q, k=10)
        all_results.extend(results)
    
    # Deduplicate and re-rank
    return deduplicate_and_rerank(all_results)
```

---

## 6. Agentic RAG

Agentic RAG gives the retrieval system agency — deciding when, how, and in what order to retrieve.

### 6.1 Self-RAG

Self-RAG adds self-reflection: the model decides whether retrieval is needed and evaluates retrieved passages:

```python
class SelfRAG:
    def __call__(self, query):
        # Decide if retrieval is needed
        if self.should_retrieve(query):
            passages = self.retrieve(query)
            
            # Evaluate each passage
            relevant_passages = []
            for p in passages:
                relevance = self.evaluate_relevance(query, p)
                if relevance > 0.5:
                    relevant_passages.append(p)
                    self.check_support(p)  # "is relevant"
                else:
                    self.check_no_support(p)  # "not relevant"
            
            # Generate with relevant passages
            response = self.generate(query, relevant_passages)
            
            # Evaluate response utility
            usefulness = self.evaluate_usefulness(response)
            return response, {
                "retrieved": len(relevant_passages),
                "usefulness": usefulness
            }
        else:
            return self.generate(query)
```

### 6.2 Corrective RAG (CRAG)

CRAG adds a correctness verification step:

```
Retrieve → Verify (retrieval evaluator) → [Correct → Generate]
                                          [Incorrect → Web Search → Generate]
                                          [Ambiguous → Both + weighted]
```

### 6.3 Adaptive RAG

Adaptive RAG routes queries to different retrieval strategies based on query complexity:

```python
class AdaptiveRAG:
    def route_query(self, query):
        complexity = self.assess_complexity(query)
        
        if complexity == "simple":
            return "direct_generate"  # No retrieval needed
        elif complexity == "medium":
            return "single_hop_retrieval"
        elif complexity == "complex":
            return "multi_hop_retrieval"
        elif complexity == "reasoning":
            return "multi_hop_retrieval + reasoning"
```

### 6.4 Tool-Augmented Retrieval

Agents use tools to retrieve information dynamically:

```python
# Tool-augmented RAG agent
tools = [
    RetrievalTool("vector_search", "Search internal knowledge base"),
    RetrievalTool("web_search", "Search the web"),
    RetrievalTool("sql_query", "Query structured databases"),
    RetrievalTool("code_search", "Search code repositories")
]

agent = create_rag_agent(
    llm=llm,
    tools=tools,
    strategy="plan_then_retrieve"  # Plan retrieval strategy, then execute
)
```

See 02-AI-Agent-Development.md for more on agent architectures.

---

## 7. Evaluation Metrics & Frameworks

### 7.1 RAGAS

RAGAS (RAG Assessment) is the most widely used evaluation framework:

**Metrics:**
| Metric | What It Measures | Calculation |
|--------|-----------------|-------------|
| **Faithfulness** | Is the answer factually grounded in retrieved context? | NLI-based, % of claims supported |
| **Answer Relevance** | How well does answer address the question? | Cosine similarity, answer→question |
| **Context Precision** | Are retrieved chunks all relevant? | % of relevant chunks in top-k |
| **Context Recall** | Are all relevant chunks retrieved? | Ground truth relevance ranking |
| **Answer Correctness** | Is the answer factually correct? | LLM-based comparison with ground truth |

**Usage:**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

dataset = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}

results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
print(results)
```

### 7.2 TruLens

TruLens provides feedback functions for RAG evaluation:

```python
from trulens_eval import Feedback, TruLlama
from trulens_eval.feedback import Groundedness

groundedness = Groundedness(groundedness_provider=llm)
f_groundedness = Feedback(groundedness.groundedness_measure_on_input).on(
    Select.RecordCalls.retrieve.rets[:].collect()
).on_output()

# Evaluate
tru_recorder = TruLlama(app, feedbacks=[f_groundedness])
with tru_recorder as recording:
    response = app.query(user_input)
```

### 7.3 ARES

ARES (Automated RAG Evaluation) uses LLMs to generate synthetic evaluation data:

**Pipeline:**
```
Train LLM → Generate synthetic Q&A pairs with relevance judgments
→ Train lightweight classifiers → Evaluate RAG system
```

**Advantages:**
- No human annotation needed
- Domain-specific evaluation
- Scalable to large document collections

### 7.4 Custom Evaluation Pipelines

Production RAG systems need custom evaluation:

```yaml
evaluation_pipeline:
  offline:
    - retrieval: [recall@k, mrr@k, ndcg@k, precision@k]
    - generation: [faithfulness, relevance, correctness, completeness]
    - end_to_end: [user_satisfaction_score, task_completion_rate]
  
  online:
    - A/B testing between retrieval strategies
    - User feedback collection (thumbs up/down)
    - Implicit signals (dwell time, follow-up rate)
    - Cost tracking ($ per query)
  
  monitoring:
    - Embedding drift detection
    - Retrieval quality degradation alerts
    - Latency P50/P95/P99 tracking
    - Embedding cache hit rate
```

---

## 8. Vector Database Comparison

| Feature | Pinecone | Weaviate | Chroma | Qdrant | pgvector |
|---------|----------|----------|--------|--------|----------|
| **Type** | Managed | Managed/OSS | Embedded | Managed/OSS | Extension |
| **Open Source** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **ANN Algorithm** | HNSW | HNSW | HNSW | HNSW | IVFFlat/HNSW |
| **Hybrid Search** | ✅ (2025+) | ✅ | ❌ | ✅ | ✅ (BM25 ext.) |
| **Multi-tenancy** | ✅ | ✅ | ❌ | ✅ | Via schemas |
| **Filtering** | ✅ | ✅ | ⚠️ Basic | ✅ | ✅ Full SQL |
| **Persistence** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Cloud-native** | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Pricing** | Usage-based | Self-host/cloud | Free | Self-host/cloud | Free |
| **Latency (P50)** | 5-10ms | 10-20ms | 1-5ms | 5-15ms | 10-30ms |
| **Market Share** | 35% | 20% | 15% | 15% | 10% |

---

## 9. Production RAG Architecture

### End-to-End Pipeline

```yaml
rag_pipeline:
  ingestion:
    - document_parsing: [PDF, DOCX, HTML, Markdown, Images]
    - chunking: [semantic_chunking, sliding_window]
    - embedding: [dense: text-embedding-3-large, sparse: SPLADE-v3]
    - indexing: [vector_db: Pinecone, sparse_index: Elasticsearch]
    
  retrieval:
    - step_1: auto_query_rewriting (LLM-based)
    - step_2: hybrid_search (dense_weight=0.6, sparse_weight=0.4)
    - step_3: initial_k=100 (HNSW, ef_search=200)
    - step_4: rerank (BGE-reranker-v2, top_k=10)
    - step_5: fusion_with_previous_context (discount duplicate info)
    
  generation:
    - prompt_builder: [system_prompt, context_prompt, query_prompt]
    - model: gpt-5-turbo (cost-optimized)
    - max_context_tokens: 8000
    - citation_generation: [Markdown footnotes, numbered citations]
    - hallucination_check: [second LLM evaluation, confidence scoring]
```

### Latency Budget (target: < 2s end-to-end)

| Component | Budget | Optimization |
|-----------|--------|-------------|
| Query rewriting | 150ms | Use small LLM (8B) |
| Embedding | 100ms | Pre-warm GPU, batch if multiple queries |
| ANN search | 50ms | Optimized HNSW parameters |
| Sparse search | 50ms | Elasticsearch caching |
| Re-rank | 200ms | GPU-accelerated |
| LLM generation | 1000ms | Speculative decoding, 4-bit quantization |
| Post-processing | 50ms | Citation insertion, formatting |

### Caching Strategy

```yaml
caching:
  query_cache:
    type: Redis
    ttl: 3600s (1 hour)
    invalidate_on: document_update
    hit_rate_target: > 30%
    
  embedding_cache:
    type: Local GPU memory
    ttl: session_duration
    capacity: 100K embeddings
    
  context_cache:
    type: Redis + Disk
    ttl: 86400s (24 hours)
    for: frequently accessed documents
```

---

## 10. Future Directions

### Emerging Trends (H2 2026)
- **Agentic RAG matures** — 50% of new RAG deployments use agent-driven retrieval
- **Multimodal RAG standard** — Text + image + table retrieval is table stakes
- **Long-context RAG** — 2M token models reduce need for complex chunking
- **Graph RAG** — Knowledge graphs augment vector search (Microsoft GraphRAG v2)
- **Streaming RAG** — Real-time retrieval for live data feeds

### Challenges
- **Evaluation** — Still no single metric that captures RAG quality
- **Latency** — Multi-stage retrieval + generation struggles with real-time requirements
- **Maintenance** — Document updates require re-indexing; drift in embedding models
- **Cost** — Hybrid retrieval + re-ranking + LLM generation is expensive at scale

### Recommended Reading
- "RAG vs Long Context: The Great Debate" (Lewis et al., 2025)
- "GraphRAG: Unsupervised Discovery of Community Structures" (Microsoft, 2024)
- "CRAG: Corrective RAG for Reliable QA" (Yan et al., 2024)
- "RAGAS: Automated Evaluation of RAG Systems" (Es et al., 2024)

---

> **Related KB documents:**
> - [02-AI-Agent-Development.md](02-AI-Agent-Development.md) — Agentic RAG integration  
> - [04-Multimodal-AI.md](04-Multimodal-AI.md) — Multimodal retrieval  
> - [07-Fine-Tuning-Custom-Models.md](07-Fine-Tuning-Custom-Models.md) — Embedding model fine-tuning  
> - [10-Real-Time-AI-Systems.md](10-Real-Time-AI-Systems.md) — Real-time RAG systems

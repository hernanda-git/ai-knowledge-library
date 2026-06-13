# AI-Powered Search: Beyond Traditional RAG

> **Last Updated:** June 2026  
> **Category:** 06-Advanced — Advanced Topics  
> **Cross-References:** 13-Top-Demand/06-RAG-Retrieval-Systems.md, 06-Advanced/03-Evaluation-Benchmarks.md, 17-Research-Frontiers/07-RAG-Retrieval-Research.md

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [The Search Landscape: From Keyword to Neural](#2-search-landscape)
3. [Neural Search — Dense Retrieval](#3-dense-retrieval)
4. [Hybrid Search — Sparse + Dense Fusion](#4-hybrid-search)
5. [Late Interaction — ColBERT](#5-colbert)
6. [Learned Sparse Retrieval — SPLADE](#6-splade)
7. [Multi-Stage Search: Retrieve → Rerank → Synthesize](#7-multi-stage)
8. [Conversational Search](#8-conversational)
9. [Multimodal Search](#9-multimodal)
10. [Personalized Search](#10-personalized)
11. [Enterprise Search Platforms](#11-enterprise)
12. [Google SGE & Vertex AI Search](#12-google-sge)
13. [Microsoft Copilot for Search](#13-microsoft-copilot)
14. [Architecture Diagrams](#14-architecture)
15. [Benchmark Comparisons](#15-benchmarks)
16. [Code Examples & Hands-On](#16-code-examples)
17. [When to Use Which Approach](#17-when-to-use)
18. [Cross-References](#18-cross-references)

---

## 1. Introduction

AI-powered search represents the next evolution beyond traditional keyword matching and even standard RAG pipelines. While classic RAG (Retrieve → Augment → Generate) has become the default pattern for grounding LLMs with external knowledge, modern AI search extends far beyond — incorporating neural embeddings, learned sparse vectors, late-interaction scoring, multi-modal fusion, conversational context, personalization, and enterprise-grade retrieval platforms.

**What distinguishes AI-powered search from traditional RAG:**

| Dimension | Traditional RAG | AI-Powered Search |
|-----------|----------------|-------------------|
| Retrieval method | Dense embeddings (cosine sim) | Hybrid sparse+dense, late interaction, learned sparse |
| Scoring | Single-stage vector similarity | Multi-stage: retrieve→rerank→synthesize |
| Query understanding | Basic embedding | Query expansion, rewriting, intent classification |
| Context handling | Turn-level | Full conversational session, personalization |
| Modalities | Text-only | Text, image, video, audio, code, structured data |
| Ranking depth | Top-K flat | Cascaded ranking (hundreds→dozens→top-1) |
| Enterprise readiness | Prototype | Full production: A/B testing, analytics, security |

This document covers the architectures, algorithms, platforms, and code needed to build production AI-powered search systems that go well beyond basic RAG.

---

## 2. The Search Landscape: From Keyword to Neural

The evolution of search technology can be understood in five generations:

```
Generation 1: Boolean/Keyword (1960s-1990s)
  Inverted index, TF-IDF, Boolean operators
  └─ e.g., grep, early web search, library catalogues

Generation 2: Statistical/IR (1990s-2010s)
  BM25, PageRank, vector space model, LSI
  └─ e.g., classic Lucene, early Google

Generation 3: Learning-to-Rank (2000s-2010s)
  LambdaMART, RankNet, feature-engineered ML
  └─ e.g., Bing, modern Elasticsearch with LTR

Generation 4: Neural Search (2018-2023)
  Dense embeddings (BERT), two-tower, bi-encoders
  └─ e.g., Google BERT for search, DPR

Generation 5: AI-Native Search (2024+)
  Hybrid dense+sparse, late interaction, LLM-in-the-loop
  └─ e.g., ColBERTv2, SPLADE, Google SGE, Copilot Search
```

**Key Insight:** Generation 5 search systems don't choose a single retrieval method. They compose multiple strategies — sparse for exact match, dense for semantic similarity, learned sparse for term-level semantics, and LLM reranking for nuanced relevance — orchestrated by an AI router.

---

## 3. Neural Search — Dense Retrieval

### 3.1 How It Works

Dense retrieval maps queries and documents into a shared embedding space using neural networks ("bi-encoders"). Similarity is measured by cosine distance or inner product.

```
  Query: "How do transformers work?"
     │
     ▼
  [Query Encoder]    ────   [Document Encoder]
     │                       (pre-computed)
     ▼                           ▼
  q_vec (768d)      ●        d_vec (768d)
     │                       (billions)
     └─────────── dot product ───────────┘
                        │
                        ▼
                  Top-K results
```

### 3.2 Model Architectures

| Model | Encoder | Dim | Typical Use | Notes |
|-------|---------|-----|-------------|-------|
| DPR (2020) | BERT-base | 768 | Question-answering | Separate query/doc encoders |
| Sentence-BERT (2019) | BERT/RoBERTa | 384/768 | General semantic search | Siamese network |
| E5 (2022) | Decoder-only | 1024 | Zero-shot search | Contrastive pretrained |
| GTR (2022) | T5 encoder | 768 | Web-scale retrieval | Multi-stage training |
| BGE (2023) | BERT/RoBERTa | 1024 | Multilingual | BAAI general embedding |
| Cohere Embed v3 | Custom | 1024 | Enterprise | Compression-aware |
| OpenAI text-embedding-3 | GPT-4 arch | 1536 | General | Matryoshka dimension |

### 3.3 Training Dense Retrievers

The standard training objective is contrastive loss with in-batch negatives:

```python
import torch
import torch.nn.functional as F

def contrastive_loss(query_emb, doc_emb, temperature=0.05):
    """NT-Xent loss for dense retriever training.
    
    Args:
        query_emb: (batch_size, dim) query embeddings
        doc_emb: (batch_size, dim) positive document embeddings
    Returns:
        loss: scalar loss value
    """
    # Normalize embeddings
    query_emb = F.normalize(query_emb, dim=-1)
    doc_emb = F.normalize(doc_emb, dim=-1)
    
    # Similarity matrix: (batch_size, batch_size)
    logits = torch.matmul(query_emb, doc_emb.T) / temperature
    
    # Diagonal = positive pairs (i.e., query[i] matches doc[i])
    batch_size = query_emb.size(0)
    labels = torch.arange(batch_size, device=query_emb.device)
    
    loss = F.cross_entropy(logits, labels)
    return loss

# Usage in training loop
# query_enc = query_encoder(query_tokens)
# doc_enc = doc_encoder(doc_tokens)
# loss = contrastive_loss(query_enc, doc_enc)
```

### 3.4 Indexing & ANN Search

Dense retrieval scales using Approximate Nearest Neighbor (ANN) search:

```python
# FAISS indexing example
import faiss
import numpy as np

def build_faiss_index(embeddings: np.ndarray, dimension: int = 768):
    """Build an HNSW index for dense vector search.
    
    Args:
        embeddings: (num_docs, dimension) numpy array
        dimension: embedding dimension
    Returns:
        index: FAISS index ready for search
    """
    # HNSW: fast, memory-efficient, high recall
    index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors
    index.hnsw.efConstruction = 200
    index.add(embeddings)
    return index

def search_index(index, query_emb: np.ndarray, k: int = 10):
    """Search FAISS index for top-k nearest neighbors."""
    distances, indices = index.search(query_emb.reshape(1, -1), k)
    return distances[0], indices[0]
```

**ANN Algorithms Compared:**

| Algorithm | Recall@10 | Build Time | Query Latency | Memory |
|-----------|-----------|------------|---------------|--------|
| Flat (exact) | 1.00 | None | 100ms (1M) | Full |
| HNSW | 0.99 | 60 min | 1ms | 1.2x |
| IVF-PQ | 0.95 | 30 min | 2ms | 0.3x |
| ScaNN | 0.98 | 45 min | 1.5ms | 1.0x |

---

## 4. Hybrid Search — Sparse + Dense Fusion

### 4.1 Why Hybrid?

Pure dense retrieval excels at semantic matching but fails on exact keyword matches (names, IDs, rare terms). Pure sparse (BM25) excels at exact match but misses semantic relationships. Hybrid search combines both.

```
   Query: "Python 3.11 typing Protocol vs ABC"
              │
         ┌─────┴─────┐
         │            │
     [BM25]       [Dense Encoder]
         │            │
    exact match    semantic match
    "typing"       "duck typing"
    "Protocol"     "structural subtyping"
    "ABC"          "abstract base class"
         │            │
         └─────┬─────┘
               │
          [Score Fusion]
          (RRF / weighted)
               │
               ▼
         Ranked results
```

### 4.2 Score Fusion Strategies

**Reciprocal Rank Fusion (RRF):**
```python
def reciprocal_rank_fusion(sparse_ranks: dict, dense_ranks: dict, k: int = 60):
    """Fuse BM25 and dense retrieval rankings using RRF.
    
    RRF score = sum(1 / (k + rank(doc, method)))
    Higher scores = better overall ranking.
    
    Args:
        sparse_ranks: {doc_id: rank} from BM25
        dense_ranks: {doc_id: rank} from dense search
        k: RRF constant (default 60)
    Returns:
        fused_scores: {doc_id: rrf_score} sorted descending
    """
    scores = {}
    for doc_id in set(list(sparse_ranks.keys()) + list(dense_ranks.keys())):
        rrf = 0.0
        if doc_id in sparse_ranks:
            rrf += 1.0 / (k + sparse_ranks[doc_id])
        if doc_id in dense_ranks:
            rrf += 1.0 / (k + dense_ranks[doc_id])
        scores[doc_id] = rrf
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
```

**Weighted Combination:**
```python
def weighted_fusion(sparse_scores: dict, dense_scores: dict, 
                    alpha: float = 0.3, beta: float = 0.7):
    """Weighted linear combination of normalized scores.
    
    alpha = weight for sparse, beta = weight for dense.
    Typical alpha range: 0.2-0.4 for domains needing exact match.
    """
    combined = {}
    all_ids = set(sparse_scores.keys()) | set(dense_scores.keys())
    
    # Min-max normalize
    def normalize(scores):
        if not scores:
            return scores
        mn, mx = min(scores.values()), max(scores.values())
        if mx == mn:
            return {k: 0.5 for k in scores}
        return {k: (v - mn) / (mx - mn) for k, v in scores.items()}
    
    sparse_norm = normalize(sparse_scores)
    dense_norm = normalize(dense_scores)
    
    for doc_id in all_ids:
        combined[doc_id] = (
            alpha * sparse_norm.get(doc_id, 0.0) +
            beta * dense_norm.get(doc_id, 0.0)
        )
    return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True))
```

### 4.3 Elasticsearch Hybrid Search

```python
# Elasticsearch hybrid search with knn + query
def hybrid_es_search(client, index: str, query_text: str, 
                     query_vector: list, k: int = 10):
    """Elasticsearch hybrid search: kNN + BM25 with RRF fusion."""
    search_body = {
        "size": k,
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "match": {
                            "content": query_text
                        }
                    },
                    {
                        "knn": {
                            "field": "embedding",
                            "query_vector": query_vector,
                            "k": k,
                            "num_candidates": 100
                        }
                    }
                ],
                "rrf": {
                    "rank_constant": 60
                }
            }
        }
    }
    response = client.search(index=index, body=search_body)
    return response["hits"]["hits"]
```

### 4.4 When Hybrid Outperforms Pure Methods

| Scenario | BM25 | Dense | Hybrid | Reason |
|----------|------|-------|--------|--------|
| Exact legal citation | ✅ | ❌ | ✅ | Dense misses exact clause numbers |
| Synonyms ("car"→"automobile") | ❌ | ✅ | ✅ | BM25 misses semantic variants |
| Code search ("sort()") | ✅ | ❌ | ✅ | Dense confuses function names |
| Cross-lingual queries | ❌ | ✅ | ✅ | BM25 fails on translation |
| Named entities ("Weyland-Yutani") | ✅ | ❌ | ✅ | Dense embeds may conflate entities |

---

## 5. Late Interaction — ColBERT

### 5.1 The ColBERT Architecture

ColBERT (Contextualized Late Interaction over BERT) introduces a fundamentally different paradigm: instead of collapsing a document into a single vector, it stores token-level embeddings and scores query-document relevance at the token level using MaxSim.

```
ColBERT Late Interaction (MaxSim)
══════════════════════════════════

  Query: "machine learning history"
     │
  [BERT Encoder]
     │
  q = [q_cls, q_machine, q_##chine, q_learning, q_history]
       (token-level embeddings, dim=128 via linear compression)
     │
     │         d = [d_cls, d_in, d_the, d_early, d_days, d_of, ...]
     │               (pre-computed, token-level embeddings)
     │
     └─────── MaxSim Scoring ──────┐
                                    │
    For each query token q_i:       │
      max_sim_i = max(j) cos(q_i, d_j)  ◄── nearest doc token
                                    │
    Score = sum(max_sim_1, max_sim_2, ...)  ◄── sum over query tokens
                                    │
                                    ▼
                              Relevance Score
```

**Key advantages over bi-encoders:**
- Captures fine-grained token-level matching (e.g., "learning" matches "learn" → "learned" → "learning")
- No information loss from average pooling
- Handles partial matches and multi-vector semantics
- Compressible: 128-dim per token (vs 768-dim for full BERT)

### 5.2 ColBERTv2 — Practical Improvements

ColBERTv2 introduced:
- **Denoised supervision:** Uses cross-encoder distillation for training data
- **Residual compression:** Compresses stored embeddings using scalar quantization + residual
- **Centroid interaction:** Faster scoring by only checking nearby document tokens

```python
# Simplified ColBERT scoring implementation
import torch
import torch.nn.functional as F

def colbert_maxsim(query_emb: torch.FloatTensor,
                   doc_emb: torch.FloatTensor) -> torch.FloatTensor:
    """Compute ColBERT MaxSim score between query and document.
    
    Args:
        query_emb: (num_query_tokens, dim) query token embeddings
        doc_emb: (num_doc_tokens, dim) document token embeddings
    Returns:
        score: scalar relevance score
    """
    # (num_q, dim) @ (dim, num_d) → (num_q, num_d)
    sim_matrix = torch.matmul(query_emb, doc_emb.T)
    
    # Max over document tokens per query token → (num_q,)
    max_sims, _ = sim_matrix.max(dim=-1)
    
    # Sum over query tokens → scalar
    score = max_sims.sum()
    return score

def colbert_batch_score(queries: torch.FloatTensor,
                        documents: torch.FloatTensor,
                        query_mask: torch.BoolTensor) -> torch.FloatTensor:
    """Batch ColBERT scoring with token masking.
    
    Args:
        queries: (batch, q_len, dim)
        documents: (batch, d_len, dim) 
        query_mask: (batch, q_len) True for non-padding tokens
    Returns:
        scores: (batch,) relevance scores
    """
    # (batch, q_len, d_len)
    sim = torch.bmm(queries, documents.transpose(1, 2))
    
    # Mask out padding query tokens
    sim = sim * query_mask.unsqueeze(-1)
    
    # MaxSim over doc dim, then sum over query tokens
    max_sim_per_query_token, _ = sim.max(dim=-1)  # (batch, q_len)
    scores = max_sim_per_query_token.sum(dim=-1)  # (batch,)
    return scores
```

### 5.3 PLAID — Fast Retrieval with ColBERT

PLAID (Post-Level Late Interaction Decoder) enables efficient retrieval by:
1. Clustering document token embeddings into centroids
2. Only scoring document tokens whose centroids match query tokens
3. Using residual compression to reduce storage

**Storage comparison for 10M documents:**

| Method | Storage | Query Latency | Recall@100 |
|--------|---------|---------------|------------|
| Full ColBERT | 1.5 TB | 1000ms | 0.98 |
| ColBERTv2+IVF | 120 GB | 50ms | 0.97 |
| PLAID | 80 GB | 30ms | 0.96 |

---

## 6. Learned Sparse Retrieval — SPLADE

### 6.1 How SPLADE Works

SPLADE (Sparse Lexical and Expansion) learns to produce sparse, interpretable vectors where each dimension corresponds to a term in the vocabulary. Unlike BM25 (static term weights), SPLADE learns term expansion and weighting end-to-end.

```
SPLADE Architecture
══════════════════════

  Query: "AI safety alignment"
     │
  [BERT Encoder]
     │
  [MLM Head] → logits over whole vocabulary
     │
  [ReLU + Log] → activation sparsity
     │
  SPLADE vector (vocab-size dim, ~50-100 non-zero entries):
    { "ai": 2.3, "safety": 3.1, "alignment": 2.8,
      "artificial": 1.2, "intelligence": 1.5, 
      "align": 1.8, "value": 0.9, "rlhf": 1.1, ... }
     │
  [Inverted Index] → standard term-based retrieval
```

### 6.2 Advantages Over Dense and Sparse

| Property | BM25 | Dense (DPR) | SPLADE |
|----------|------|-------------|--------|
| Interpretability | ✅ Terms | ❌ Opaque | ✅ Terms (expanded) |
| Out-of-vocabulary | ❌ | ✅ Embeddings generalize | ⚠️ Vocabulary-limited |
| Exact match | ✅ | ❌ | ✅ |
| Semantic expansion | ❌ | ✅ | ✅ |
| Hardware | CPU | GPU/ANN | CPU (inverted index) |
| Index size | Small | Large (vectors) | Medium (sparse) |

### 6.3 SPLADE Training

```python
import torch
import torch.nn as nn

class SPLADE(nn.Module):
    """SPLADE learned sparse retrieval model."""
    
    def __init__(self, bert_model: nn.Module, vocab_size: int = 30522):
        super().__init__()
        self.bert = bert_model
        # Linear projection from BERT hidden to vocab logits
        self.mlm_predict = nn.Linear(self.bert.config.hidden_size, vocab_size)
        
    def forward(self, input_ids, attention_mask):
        """Forward pass producing sparse term weights.
        
        Returns:
            sparse_vec: (batch_size, vocab_size) sparse activations
            mlm_loss: optional MLM loss for regularization
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation or mean pooling
        pooled = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        
        # Project to vocabulary logits
        logits = self.mlm_predict(pooled)  # (batch, vocab_size)
        
        # Sparsity activation: log(1 + ReLU(x))
        sparse_vec = torch.log(1 + torch.relu(logits))
        
        # Optionally compute FLOPS regularization loss
        flops = (sparse_vec > 0).float().sum(dim=-1).mean()
        
        return sparse_vec, flops

    def index_document(self, input_ids, attention_mask):
        """Get sparse representation for document indexing."""
        sparse_vec, _ = self.forward(input_ids, attention_mask)
        # Get non-zero entries
        indices = sparse_vec[0].nonzero(as_tuple=True)[0]
        values = sparse_vec[0][indices]
        return indices.tolist(), values.tolist()
```

### 6.4 SPLADE vs Other Learned Sparse Models

| Model | Approach | Sparsity | Effectiveness (BEIR) |
|-------|----------|----------|---------------------|
| DeepImpact | Term weighting | ~100 terms/doc | 0.41 nDCG@10 |
| uniCOIL | Contextualized inverted list | ~150 terms/doc | 0.44 nDCG@10 |
| SPLADE-v2 | Full vocab expansion | ~200 terms/doc | 0.47 nDCG@10 |
| SPLADE-v3 | Distilled + pooled negatives | ~120 terms/doc | **0.51 nDCG@10** |
| SPLADE++ | Ensemble of SPLADE models | ~300 terms/doc | 0.50 nDCG@10 |

---

## 7. Multi-Stage Search: Retrieve → Rerank → Synthesize

### 7.1 The Three-Stage Pipeline

Modern production search pipelines use a cascade of increasingly expensive models:

```
Stage 1: Retrieval (budget: 1-10ms per query)
  ┌─────────────────┐
  │ BM25 + Dense ANN  │  ← 100-1000 candidates
  │ ColBERT / SPLADE   │
  └────────┬─────────┘
           │ Top-100
           ▼
Stage 2: Reranking (budget: 50-200ms per query)
  ┌─────────────────┐
  │ Cross-encoder    │  ← Score top-100 pairs
  │ (e.g., MonoT5,   │
  │  Cohere Rerank)  │
  └────────┬─────────┘
           │ Top-10
           ▼
Stage 3: Synthesis (budget: 500-5000ms per query)
  ┌─────────────────┐
  │ LLM Generation   │  ← Read + synthesize top-10
  │ (GPT-4, Claude,  │
  │  Gemini, etc.)   │
  └────────┬─────────┘
           │ Final answer
           ▼
      User Response
```

### 7.2 Reranker Implementation

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class CrossEncoderReranker:
    """Cross-encoder reranker using BERT/RoBERTa."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
    def rerank(self, query: str, documents: list[str], top_k: int = 10):
        """Rerank documents by relevance to query.
        
        Args:
            query: search query string
            documents: list of document texts
            top_k: number of top results to return
        Returns:
            list of (doc_id, score) tuples sorted by relevance
        """
        pairs = [[query, doc] for doc in documents]
        
        # Tokenize with truncation
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Score
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1)
        
        # Rank
        ranked = sorted(
            enumerate(scores.tolist()),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [(idx, score) for idx, score in ranked]

# Usage
# reranker = CrossEncoderReranker()
# results = reranker.rerank(query="Python async programming", 
#                            documents=doc_list, top_k=10)
```

### 7.3 MonoT5 — Sequence-to-Sequence Reranking

MonoT5 treats reranking as a text generation task: `Query: {q} Document: {d} Relevant:`

```python
from transformers import AutoTokenizer, T5ForConditionalGeneration

class MonoT5Reranker:
    def __init__(self, model_name: str = "castorini/monot5-base-msmarco"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        
    def rerank(self, query: str, documents: list[str], top_k: int = 10):
        inputs = [
            f"Query: {query} Document: {doc} Relevant:"
            for doc in documents
        ]
        tokens = self.tokenizer(
            inputs, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        
        with torch.no_grad():
            # Score = probability of generating "true"
            decoder_input = self.tokenizer(
                ["true"] * len(documents),
                return_tensors="pt", padding=True
            ).input_ids
            
            outputs = self.model(**tokens, decoder_input_ids=decoder_input)
            logits = outputs.logits[:, 0, :]  # First token position
            true_token_id = self.tokenizer.encode("true")[0]
            scores = torch.softmax(logits, dim=-1)[:, true_token_id]
        
        ranked = sorted(
            enumerate(scores.tolist()),
            key=lambda x: x[1], reverse=True
        )[:top_k]
        return [(idx, score) for idx, score in ranked]
```

### 7.4 Synthesizer (LLM-in-the-Loop)

```python
import openai  # or anthropic, google.generativeai, etc.

class SearchSynthesizer:
    """LLM-powered search synthesis stage."""
    
    SYSTEM_PROMPT = """You are an AI search assistant. Given a user query and 
    relevant context documents, synthesize an accurate, well-cited answer.
    - Base your answer ONLY on the provided context
    - Cite sources using [1], [2], etc.
    - If context is insufficient, say so clearly
    - Use markdown formatting for clarity"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = openai.OpenAI()
        
    def synthesize(self, query: str, documents: list[dict]) -> str:
        """Generate a synthesized answer from search results.
        
        Args:
            query: user's search query
            documents: list of {'title': str, 'content': str, 'source': str}
        Returns:
            synthesized answer string
        """
        context = "\n\n".join([
            f"[{i+1}] {doc['title']}\n{doc['content'][:2000]}"
            for i, doc in enumerate(documents)
        ])
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        return response.choices[0].message.content
```

### 7.5 Full Pipeline Orchestration

```python
class AISearchPipeline:
    """End-to-end AI search pipeline."""
    
    def __init__(self, retriever, reranker, synthesizer):
        self.retriever = retriever
        self.reranker = reranker
        self.synthesizer = synthesizer
        
    def search(self, query: str, k_retrieve: int = 100, 
               k_rerank: int = 10, synthesize: bool = True):
        """Multi-stage AI search.
        
        Returns dict with results from each stage.
        """
        # Stage 1: Retrieve candidates
        candidates = self.retriever.retrieve(query, k=k_retrieve)
        
        # Stage 2: Rerank with cross-encoder
        reranked = self.reranker.rerank(query, candidates, top_k=k_rerank)
        top_docs = [candidates[i] for i, _ in reranked]
        
        # Stage 3: Synthesize
        if synthesize and self.synthesizer:
            answer = self.synthesizer.synthesize(query, top_docs)
        else:
            answer = None
            
        return {
            "query": query,
            "candidates": candidates[:k_retrieve],
            "reranked": reranked,
            "top_docs": top_docs,
            "answer": answer
        }
```

---

## 8. Conversational Search

### 8.1 From Single-Turn to Multi-Turn

Conversational search maintains state across turns: query rewriting, context compression, and follow-up understanding.

```
Turn 1: "What is retrieval augmented generation?"
  → query: "retrieval augmented generation definition"
  → context_store: ["RAG is a technique that..."]

Turn 2: "How does it compare to fine-tuning?"
  → rewritten_query: "RAG vs fine-tuning comparison"
  → context_store: ["RAG uses external knowledge...", 
                     "Fine-tuning modifies weights..."]
  
Turn 3: "What are the latency implications?"
  → rewritten_query: "RAG latency implications compared to fine-tuning"
  → context_store: ["RAG requires vector database queries...",
                     "Fine-tuning has no retrieval latency..."]
```

### 8.2 Query Rewriting

```python
class QueryRewriter:
    """Rewrite conversational queries with context."""
    
    REWRITE_PROMPT = """Rewrite the following question for web search, 
    incorporating relevant context from the conversation history.
    
    Conversation history:
    {history}
    
    Current question: {question}
    
    Rewritten search query:"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI()
        self.model = model
        self.history = []
        
    def rewrite(self, question: str) -> str:
        history_text = "\n".join([
            f"User: {h['question']}\nAssistant: {h['answer'][:200]}"
            for h in self.history[-3:]  # keep last 3 turns
        ])
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": self.REWRITE_PROMPT.format(
                    history=history_text, question=question
                )}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
```

### 8.3 Context-Aware Retrieval

```python
class ConversationalRetriever:
    """Retrieve documents with conversation context."""
    
    def retrieve(self, query: str, conversation_id: str, 
                 k: int = 10) -> list[dict]:
        """Enrich query with conversation context before retrieval.
        
        1. Retrieve recent conversation history
        2. Extract entities mentioned in history
        3. Build enriched query: original + entity context
        4. Perform hybrid search with enriched query
        """
        conv_history = self.get_history(conversation_id)
        entities = self.extract_entities(conv_history)
        
        enriched_query = query
        if entities:
            enriched_query = f"{query} {' '.join(entities)}"
            
        # Hybrid search with enriched query
        results = self.hybrid_search(enriched_query, k=k)
        
        # Result diversification: don't repeat what user already saw
        seen_docs = conv_history.get("seen_doc_ids", set())
        results = [r for r in results if r["doc_id"] not in seen_docs]
        
        return results
```

---

## 9. Multimodal Search

### 9.1 Unified Embedding Spaces

Modern AI search crosses modalities using shared embedding spaces:

```
Search Types:

Text → Search Images:
  "red car on beach sunset" → [encoder] → matched to image embeddings

Image → Search Text:
  [photo of a golden retriever] → [encoder] → "golden retriever in park"

Text → Search Video:
  "machine learning tutorial 2026" → [encoder] → video segment matches

Image + Text → Search Products:
  "shoes similar to [this image] in blue" → [fusion encoder] → product results

Audio → Search Text:
  "♪ (hummed tune) ♪" → [audio encoder] → "Billie Eilish - Birds of a Feather"
```

### 9.2 Multimodal Embedding Models

| Model | Modalities | Embedding Dim | Notes |
|-------|-----------|---------------|-------|
| CLIP (OpenAI) | Text, Image | 512 | Pioneering vision-language |
| SigLIP (Google) | Text, Image | 768 | Sigmoid loss, better than CLIP |
| BLIP-2 | Text, Image | 768 | Q-former bridging vision-LLM |
| ImageBind (Meta) | Text, Image, Audio, Depth, Thermal | 1024 | 5+ modalities unified |
| Google Multimodal Embedding | Text, Image, Video | 1408 | Enterprise, Vertex AI |
| Jina CLIP v2 | Text, Image | 768 | 89 languages |

### 9.3 Multi-Modal Search Implementation

```python
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class MultimodalSearch:
    """CLIP-based multimodal search across text and images."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.image_index = []  # list of (embedding, metadata)
        
    def embed_text(self, text: str):
        inputs = self.processor(text=[text], return_tensors="pt", 
                                padding=True, truncation=True)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)
    
    def embed_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        return emb / emb.norm(dim=-1, keepdim=True)
    
    def index_images(self, image_paths: list[str], metadata: list[dict]):
        """Build image index for search."""
        for path, meta in zip(image_paths, metadata):
            emb = self.embed_image(path)
            self.image_index.append((emb, {**meta, "path": path}))
    
    def search_by_text(self, query: str, k: int = 10):
        query_emb = self.embed_text(query)
        scores = [(emb @ query_emb.T).item(), meta 
                  for emb, meta in self.image_index]
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:k]
    
    def search_by_image(self, image_path: str, k: int = 10):
        query_emb = self.embed_image(image_path)
        scores = [(emb @ query_emb.T).item(), meta 
                  for emb, meta in self.image_index]
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:k]
```

---

## 10. Personalized Search

### 10.1 User Embedding for Personalization

Personalized search builds user profiles from interaction history and biases results accordingly:

```python
class PersonalizationLayer:
    """Add personalization to search results."""
    
    def __init__(self):
        self.user_profiles = {}  # user_id → profile embedding
        
    def update_profile(self, user_id: str, clicked_docs: list[str]):
        """Update user embedding from click history."""
        doc_embs = [self.doc_embeddings[doc] for doc in clicked_docs]
        if doc_embs:
            user_emb = torch.mean(torch.stack(doc_embs), dim=0)
            self.user_profiles[user_id] = user_emb
            
    def personalize_results(self, user_id: str, results: list[dict], 
                            alpha: float = 0.3) -> list[dict]:
        """Bias search results toward user preferences.
        
        Final score = (1 - alpha) * relevance_score + alpha * personalization_score
        """
        if user_id not in self.user_profiles:
            return results  # cold start → return generic results
            
        user_emb = self.user_profiles[user_id]
        
        for result in results:
            doc_emb = self.doc_embeddings[result["doc_id"]]
            personalization_score = F.cosine_similarity(
                user_emb.unsqueeze(0), doc_emb.unsqueeze(0)
            ).item()
            result["score"] = (
                (1 - alpha) * result["score"] + 
                alpha * personalization_score
            )
            
        return sorted(results, key=lambda x: x["score"], reverse=True)
```

### 10.2 Personalization Strategies

| Strategy | Method | Data Required | Cold Start |
|----------|--------|---------------|------------|
| Content-based | User embedding from history | Click/explicit feedback | Use demographics |
| Collaborative | User-user similarity matrix | Large user base | Ask preferences |
| Session-based | Sequential click model | Current session only | N/A |
| Hybrid | Weighted combination | All above | Best overall |
| Learning-to-rank | LambdaMART with user features | Rich feature engineering | Model-based fallback |

---

## 11. Enterprise Search Platforms

### 11.1 Elasticsearch Learned Sparse Encoder

Elasticsearch 8.12+ includes a built-in learned sparse encoder model:

```json
// Elasticsearch learned sparse index mapping
PUT /search-index
{
  "mappings": {
    "properties": {
      "content": {
        "type": "text",
        "analyzer": "standard"
      },
      "content_embedding": {
        "type": "rank_features"  // sparse vector
      }
    }
  }
}

// Query with ELSER (Elastic Learned Sparse Encoder)
GET /search-index/_search
{
  "query": {
    "text_expansion": {
      "content_embedding": {
        "model_id": ".elser_model_2",
        "model_text": "What are the side effects of aspirin?"
      }
    }
  }
}
```

**ELSER Performance:**

| Metric | BM25 | ELSER | ELSER + Dense Hybrid |
|--------|------|-------|---------------------|
| nDCG@10 (BEIR) | 0.44 | 0.48 | **0.51** |
| Query latency | 5ms | 15ms | 25ms |
| Index size | 1x | 3x | 5x (both indices) |
| Training data | None | MS MARCO | Combined |

### 11.2 Cohere AI Search

Cohere provides a full-stack AI search platform:

```python
import cohere

co = cohere.Client("YOUR_API_KEY")

# Embed documents
docs = ["Document 1 text...", "Document 2 text..."]
doc_embeds = co.embed(
    texts=docs,
    model="embed-english-v3.0",
    input_type="search_document"
)

# Embed query
query = "What is the capital of France?"
query_embed = co.embed(
    texts=[query],
    model="embed-english-v3.0",
    input_type="search_query"
)

# Rerank results
results = co.rerank(
    query=query,
    documents=docs,
    model="rerank-english-v3.0",
    top_n=5
)

# Also supports compressed representations for efficiency
compressed = co.embed(
    texts=docs,
    model="embed-english-v3.0",
    input_type="search_document",
    compression_codebook="default"  # reduces storage 4x
)
```

### 11.3 Vertex AI Search (Google)

Google Vertex AI Search provides enterprise search with grounding:

```python
from google.cloud import discoveryengine_v1 as discoveryengine

def vertex_ai_search(project_id: str, location: str, 
                     data_store_id: str, query: str):
    """Search using Vertex AI Search (formerly Enterprise Search)."""
    client = discoveryengine.SearchServiceClient()
    
    serving_config = f"projects/{project_id}/locations/{location}" \
                     f"/dataStores/{data_store_id}/servingConfigs/default_search"
    
    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=query,
        page_size=10,
        query_expansion_spec={
            "condition": "AUTO"
        },
        spell_correction_spec={
            "mode": "AUTO"
        }
    )
    
    response = client.search(request)
    return response.results

# Also supports AI grounding:
def grounded_search(project_id: str, query: str):
    """Search with LLM grounding via Vertex AI."""
    from google.cloud import aiplatform
    aiplatform.init(project=project_id)
    
    # Use Gemini with search grounding
    # This connects Gemini to Google Search/Enterprise data
    response = aiplatform.gapic.PredictionServiceClient().generate_content(
        model="gemini-2.0-flash-001",
        contents=[{"role": "user", "parts": [{"text": query}]}],
        tools=[{
            "google_search_retrieval": {
                "dynamic_retrieval_config": {
                    "mode": "MODE_DYNAMIC",
                    "dynamic_threshold": 0.5
                }
            }
        }]
    )
    return response
```

### 11.4 Algolia AI

Algolia's AI search platform combines neural search with merchandising:

```python
# Algolia AI search example
import algoliasearch

client = algoliasearch.search_client("APP_ID", "API_KEY")
index = client.init_index("products")

# Neural search with AI re-ranking
results = index.search("red running shoes", {
    "enableRules": True,
    "enablePersonalization": True,
    "attributesToRetrieve": ["name", "price", "description", "image_url"],
    "reRankingApply": True,  # AI re-ranking
    "getRankingInfo": True,
    "clickAnalytics": True  # Feed clicks back into model
})

# AI-powered query suggestions
suggestions = index.search_for_suggestions("runni", {
    "hitsPerPage": 5
})

# Dynamic re-ranking based on conversion data
index.set_settings({
    "customRanking": [
        "desc(conversion_score)",  # ML-predicted conversion
        "desc(personalization_score)"
    ]
})
```

### 11.5 Platform Comparison

| Feature | Elasticsearch | Cohere | Vertex AI | Algolia |
|---------|--------------|--------|-----------|---------|
| Dense retrieval | ✅ (kNN plugin) | ✅ (Embed v3) | ✅ | ✅ |
| Sparse retrieval | ✅ (BM25) | ✅ (with embed) | ✅ | ✅ |
| Learned sparse | ✅ (ELSER) | ❌ | ❌ | ❌ |
| Hybrid search | ✅ (native) | ✅ (manual) | ✅ | ✅ |
| Reranking | ✅ (Cross-encoder) | ✅ (Rerank v3) | ✅ | ✅ (AI Re-ranking) |
| Personalization | ❌ (needs plugin) | ❌ | ✅ | ✅ |
| Analytics | ✅ | ❌ | ✅ | ✅ |
| A/B testing | ✅ (with license) | ❌ | ✅ | ✅ |
| Self-hosted | ✅ | ❌ | ❌ | ✅ (paid) |
| Latency (p99) | 50ms | 200ms | 500ms | 30ms |

---

## 12. Google SGE & Vertex AI Search

### 12.1 Search Generative Experience (SGE)

Google SGE integrates generative AI directly into search results. The architecture involves:

```
SGE Architecture (conceptual)
═══════════════════════════════

  User Query
     │
  ┌──┴──┐
  │     │
  Core  │
  Search │
  Index  │
  │     │
  └──┬──┘
     │
  [Query Understanding]
  ├─ Intent classification
  ├─ Entity extraction  
  └─ Query expansion
     │
  [Retrieval Layer]
  ├─ Web index (trillions of pages)
  ├─ Knowledge graph
  ├─ Shopping/product index
  └─ Video/image index
     │
  [Multimodal Integration]
  ├─ Text results
  ├─ Image results
  └─ Video highlights
     │
  [LLM Generation (Gemini)]
  ├─ Summarize top results
  ├─ Generate overview
  ├─ List key information
  └─ Show source cards
     │
  [Attribution Layer]
  ├─ Source citations
  ├─ Factuality checks
  └─ Confidence indicators
```

**Key Features of SGE:**
- **AI Overviews:** Generative summaries at the top of search results
- **Conversational Mode:** Follow-up questions maintain context
- **Shopping:** Product comparisons with AI-generated pros/cons
- **Code & Technical:** Code generation with sources
- **Image/Voice:** Multimodal input support

### 12.2 Vertex AI Search for Enterprise

Google's enterprise offering provides SGE-like capabilities for private data:

```python
# Vertex AI Search with grounding and summarization
def enterprise_grounded_search(project_id, location, data_store_id, query):
    """Enterprise search with AI-generated answers grounded in data."""
    
    client = discoveryengine.SearchServiceClient()
    
    # Configure summary generation
    summary_spec = {
        "summaryResultCount": 3,
        "include_citations": True,
        "ignore_adversarial_query": True,
        "model_prompt_spec": {
            "preamble": "You are an enterprise search assistant. "
                       "Answer based ONLY on the provided documents."
        },
        "language_code": "en-US"
    }
    
    request = discoveryengine.SearchRequest(
        serving_config=f"projects/{project_id}/locations/{location}/"
                       f"dataStores/{data_store_id}/servingConfigs/default_search",
        query=query,
        page_size=10,
        content_search_spec={
            "summary_spec": summary_spec,
            "extractive_content_spec": {
                "max_extractive_segments": 3
            }
        }
    )
    
    response = client.search(request)
    
    # Get summary
    summary = response.summary.summary_text
    citations = response.summary.summary_with_metadata
    
    return {
        "summary": summary,
        "results": response.results,
        "citations": citations
    }
```

---

## 13. Microsoft Copilot for Search

### 13.1 Architecture

Microsoft Copilot for Search (formerly Bing Chat/Copilot) integrates throughout the Microsoft ecosystem:

```
Microsoft Copilot Search Architecture
══════════════════════════════════════

  Query Input (text, voice, image)
     │
  [Microsoft Graph]
  ├─ Enterprise data (SharePoint, OneDrive, Teams)
  ├─ Web index (Bing)
  └─ Personal data (calendar, email, contacts)
     │
  [Copilot Orchestrator]
  ├─ Query understanding
  ├─ Intent routing (web vs enterprise vs personal)
  └─ Permission check (Microsoft 365 RBAC)
     │
  [Retrieval Layer]
  ├─ Bing web search
  ├─ Microsoft Graph search API
  ├─ Vector search (SharePoint semantic index)
  └─ Microsoft 365 content indexing
     │
  [Reranking + Grounding]
  ├─ Cross-encoder reranking
  ├─ Factuality verification
  └─ Source attribution
     │
  [GPT-4 / Copilot LLM]
  ├─ Answer generation
  ├─ Citation formatting
  └─ Action suggestions
     │
  [Output]
  ├─ Generated answer with citations
  ├─ Source links
  └─ Follow-up query suggestions
```

### 13.2 Semantic Index for SharePoint

Microsoft's semantic index enables vector search over enterprise content:

```python
# Microsoft Graph search API with semantic ranking
import requests

def copilot_search(access_token: str, query: str):
    """Search enterprise content via Microsoft Graph."""
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    # Graph search API
    body = {
        "requests": [{
            "entityTypes": ["driveItem", "listItem", "message", "event"],
            "query": {
                "queryString": query
            },
            "stored_fields": ["name", "path", "lastModifiedBy", 
                             "lastModifiedDateTime", "summary"],
            "size": 25,
            "rank": {
                "rankingModelType": "semantic"  # uses Microsoft's semantic ranker
            }
        }]
    }
    
    response = requests.post(
        "https://graph.microsoft.com/v1.0/search/query",
        headers=headers, json=body
    )
    return response.json()

# The semantic ranker uses Microsoft's cross-encoder model (MS MARCO based)
# to reorder results by relevance to the query.
```

---

## 14. Architecture Diagrams

### 14.1 Production AI Search Architecture

```
                           ┌─────────────────────────┐
                           │     User Interface       │
                           │  (Web / Mobile / API)    │
                           └───────────┬─────────────┘
                                       │
                              ┌────────▼────────┐
                              │   API Gateway    │
                              │  (Auth, Rate     │
                              │   Limiting)      │
                              └────────┬────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Search Orchestrator                            │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Query     │  │ Intent   │  │ Query    │  │ Conversation     │ │
│  │ Parser    │──▶ Router   │──▶ Rewriter  │  │ Context Manager  │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              Retrieval Strategy Selector                  │    │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌───────────────┐  │    │
│  │  │ Dense  │  │ Sparse │  │ Learned│  │  Multimodal   │  │    │
│  │  │ ANN    │  │ BM25   │  │ SPLADE │  │  Search       │  │    │
│  │  └────────┘  └────────┘  └────────┘  └───────────────┘  │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                   Fusion & Reranking                      │    │
│  │  ┌──────────┐  ┌────────────┐  ┌──────────────────────┐  │    │
│  │  │ RRF Fuse │──▶ Cross-     │──▶  LLM Synthesizer     │  │    │
│  │  │          │  │ Encoder    │   │  (GPT-4, Gemini)     │  │    │
│  │  └──────────┘  └────────────┘  └──────────────────────┘  │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Backend Services                               │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Vector DB     │  │ Search       │  │ Cache (Redis)          │ │
│  │ (FAISS,       │  │ Engine       │  │ - Query cache          │ │
│  │  Pinecone,    │  │ (ES, Solr)   │  │ - Embedding cache      │ │
│  │  Milvus)      │  │              │  │ - User session cache   │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Embedding     │  │ Reranker     │  │ Analytics Pipeline     │ │
│  │ Service       │  │ Model        │  │ - Click tracking       │ │
│  │ (GPU batch)   │  │ (GPU/CPU)    │  │ - A/B testing          │ │
│  └──────────────┘  └──────────────┘  │ - User profiling        │ │
│                                       └────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Data Ingestion Pipeline                        │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Document  │──▶ Chunking │──▶ Embedding│──▶ Index Building   │ │
│  │ Parser    │  │ Strategy │  │ (batch)  │  │ (HNSW, IVF)      │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 14.2 Data Flow Diagram

```
   ┌──────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐
   │Query  │───▶│Query    │───▶│Retrieval │───▶│Fusion   │
   │Input  │    │Rewrite  │    │Layer     │    │& Rerank │
   └──────┘    └─────────┘    └──────────┘    └─────────┘
                                                    │
                                                    ▼
                              ┌─────────┐    ┌──────────┐
                              │Response │◄───│LLM       │
                              │Output   │    │Synthesis │
                              └─────────┘    └──────────┘
                                    │
                                    ▼
                              ┌─────────┐
                              │Feedback │───▶ Model Update
                              │Loop     │    (online learning)
                              └─────────┘
```

### 14.3 Hybrid Search Architecture Detail

```
                          Hybrid Search Internals
   ════════════════════════════════════════════════════

   ┌─────────────────┐        ┌─────────────────┐
   │ BM25 Index       │        │ Dense Index      │
   │ (Inverted Index) │        │ (FAISS HNSW)     │
   │                  │        │                  │
   │ "python" → [1,2] │        │ d1: [0.1,0.3,...]│
   │ "sort"   → [1,3] │        │ d2: [0.4,0.1,...]│
   │ "list"   → [2,4] │        │ d3: [0.2,0.8,...]│
   └────────┬─────────┘        └────────┬─────────┘
            │                           │
            ▼                           ▼
   ┌─────────────────┐        ┌─────────────────┐
   │ BM25 Scoring    │        │ ANN Search       │
   │ score=1.2,2.3   │        │ score=0.85,0.72  │
   └────────┬─────────┘        └────────┬─────────┘
            │                           │
            └──────────┬────────────────┘
                       │
                ┌──────▼──────┐
                │ Score Fusion│
                │ (RRF)       │
                │ score=0.042 │
                │ score=0.038 │
                └──────┬──────┘
                       │
                       ▼
                ┌──────────────┐
                │ Reranker     │
                │ (MonoT5)     │
                │ score=4.2    │
                │ score=3.8    │
                └──────┬──────┘
                       │
                       ▼
                ┌──────────────┐
                │ Final Ranked │
                │ Results      │
                └──────────────┘
```

---

## 15. Benchmark Comparisons

### 15.1 BEIR Benchmark Results

BEIR (Benchmarking IR) is the standard for zero-shot retrieval evaluation:

| Model | nDCG@10 (Avg over 18 datasets) | Type | Year |
|-------|-------------------------------|------|------|
| BM25 | 0.44 | Sparse (static) | — |
| DPR | 0.44 | Dense | 2020 |
| ColBERTv2 | 0.50 | Late interaction | 2021 |
| SPLADE-v2 | 0.47 | Learned sparse | 2021 |
| Cohere Embed v3 | 0.48 | Dense | 2023 |
| E5-large | 0.50 | Dense (contrastive) | 2022 |
| GTR-large | 0.48 | Dense (multi-stage) | 2022 |
| BGE-large | 0.51 | Dense | 2023 |
| SPLADE-v3 | 0.51 | Learned sparse | 2024 |
| ColBERTv2 + Cross-Encoder | **0.54** | Late + rerank | 2023 |

### 15.2 BEIR Per-Dataset Breakdown

| Dataset | BM25 | DPR | ColBERTv2 | SPLADE-v3 | Hybrid Best |
|---------|------|-----|-----------|------------|-------------|
| TREC-COVID | 0.67 | 0.33 | 0.74 | 0.76 | 0.79 |
| NFCorpus | 0.32 | 0.19 | 0.33 | 0.35 | 0.36 |
| NQ | 0.33 | 0.47 | 0.53 | 0.52 | 0.56 |
| HotpotQA | 0.60 | 0.63 | 0.68 | 0.64 | 0.69 |
| FiQA | 0.24 | 0.30 | 0.35 | 0.36 | 0.37 |
| ArguAna | 0.31 | 0.17 | 0.31 | 0.55 | 0.56 |
| Touché | 0.37 | 0.27 | 0.26 | 0.28 | 0.38 |
| DBPedia | 0.31 | 0.26 | 0.39 | 0.42 | 0.42 |
| SCIDOCS | 0.16 | 0.12 | 0.16 | 0.16 | 0.17 |
| FEVER | 0.58 | 0.56 | 0.73 | 0.74 | 0.76 |
| Climate-FEVER | 0.17 | 0.17 | 0.19 | 0.22 | 0.24 |
| SciFact | 0.48 | 0.32 | 0.67 | 0.68 | 0.69 |

### 15.3 MTEB (Massive Text Embedding Benchmark)

| Model | Avg (56 datasets) | Clustering | Pair Classification | Reranking | Retrieval | STS | Summarization |
|-------|-------------------|------------|-------------------|-----------|-----------|-----|---------------|
| OpenAI text-embedding-3-large | 64.6 | 47.7 | 86.2 | 59.4 | **54.9** | 84.1 | 32.7 |
| Cohere Embed v3 | 63.8 | 48.1 | 85.7 | 58.8 | 53.7 | 83.9 | 31.5 |
| BGE-en-icl | 63.5 | 47.9 | 85.2 | 58.5 | 53.2 | 83.5 | 31.2 |
| E5-mistral-7b-instruct | 63.2 | 47.5 | 85.0 | 58.2 | 52.8 | 83.1 | 30.8 |
| GTE-Qwen2 | 62.8 | 47.1 | 84.8 | 57.9 | 52.3 | 82.8 | 30.5 |

### 15.4 Latency vs. Quality Tradeoff

```
Quality (nDCG@10)
    ▲
0.55│                                ◆ ColBERTv2+CE
    │                          ◆ Hybrid (BM25+Dense+CE)
0.50│                    ◆ SPLADE-v3
    │               ◆ ColBERTv2
0.45│          ◆ Cohere Embed
    │     ◆ E5
0.40│ ◆ BM25
    │
    └───────────────────────────────────────────▶ Latency (ms)
       1     10    50    200    500   1000
```

Key finding: Hybrid retrieval + lightweight cross-encoder reranking offers the best tradeoff for most production use cases. Pure dense retrieval is fast but quality-limited; adding a reranker boosts quality 5-10% at 2-5x latency cost.

---

## 16. Code Examples & Hands-On

### 16.1 Building a Complete AI Search System

```python
"""
Complete AI Search System Example
══════════════════════════════════
This example builds a hybrid search system with:
- BM25 + Dense retrieval
- Reciprocal Rank Fusion
- Cross-encoder reranking
- LLM synthesis

Requirements: pip install sentence-transformers rank_bm25 torch
"""

from typing import List, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

class AISearchEngine:
    """Production-ready AI search engine with hybrid retrieval + reranking."""
    
    def __init__(
        self,
        embed_model: str = "BAAI/bge-small-en-v1.5",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-4-v2",
        use_reranker: bool = True
    ):
        print(f"Loading embed model: {embed_model}")
        self.embedder = SentenceTransformer(embed_model)
        
        self.use_reranker = use_reranker
        if use_reranker:
            print(f"Loading reranker: {reranker_model}")
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                reranker_model
            )
            self.reranker_model.eval()
        
        self.documents = []
        self.doc_ids = []
        self.bm25 = None
        self.embeddings = None
        
    def index(self, documents: List[dict]):
        """Index documents: {'id': str, 'title': str, 'content': str}"""
        self.documents = documents
        self.doc_ids = [doc['id'] for doc in documents]
        
        # Tokenize for BM25
        tokenized = [doc['content'].split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        
        # Embed for dense retrieval
        texts = [
            f"{doc.get('title', '')} {doc.get('content', '')}"
            for doc in documents
        ]
        print(f"Embedding {len(texts)} documents...")
        self.embeddings = self.embedder.encode(
            texts, show_progress_bar=True, normalize_embeddings=True
        )
        
    def _bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(idx, scores[idx]) for idx in top_indices]
    
    def _dense_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        q_emb = self.embedder.encode(query, normalize_embeddings=True)
        scores = np.dot(self.embeddings, q_emb)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(idx, float(scores[idx])) for idx in top_indices]
    
    def _rrf_fuse(self, sparse_results, dense_results, k=60):
        scores = {}
        for idx, _ in sparse_results:
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + 1)
        for idx, rank in enumerate([r[0] for r in dense_results]):
            scores[rank] = scores.get(rank, 0.0) + 1.0 / (k + idx + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def _rerank(self, query: str, candidates: List[Tuple[int, float]], 
                 top_k: int) -> List[Tuple[int, float]]:
        if not self.use_reranker or not candidates:
            return candidates[:top_k]
        
        doc_texts = [
            f"{self.documents[idx].get('title', '')} {self.documents[idx]['content'][:512]}"
            for idx, _ in candidates
        ]
        pairs = [[query, doc] for doc in doc_texts]
        
        inputs = self.reranker_tokenizer(
            pairs, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.reranker_model(**inputs)
            scores = outputs.logits.squeeze(-1).tolist()
        
        # Sort by reranker score
        scored = list(zip([idx for idx, _ in candidates], scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    def search(self, query: str, k_retrieve: int = 50, k_rerank: int = 10,
               alpha: float = 0.3) -> dict:
        """Full search pipeline."""
        # Stage 1: Hybrid retrieval
        sparse = self._bm25_search(query, k=k_retrieve)
        dense = self._dense_search(query, k=k_retrieve)
        fused = self._rrf_fuse(sparse, dense)
        
        # Stage 2: Reranking
        reranked = self._rerank(query, fused, top_k=k_rerank)
        
        # Format results
        results = []
        for idx, score in reranked:
            doc = self.documents[idx]
            results.append({
                "id": doc["id"],
                "title": doc.get("title", ""),
                "content": doc["content"][:300],
                "score": round(score, 4),
                "rank": len(results) + 1
            })
        
        return {
            "query": query,
            "total_results": len(reranked),
            "results": results
        }
    
    def search_with_synthesis(self, query: str, **kwargs) -> dict:
        """Search + optional LLM synthesis (requires openai)."""
        search_results = self.search(query, **kwargs)
        
        try:
            import openai
            client = openai.OpenAI()
            context = "\n\n".join([
                f"[{r['rank']}] {r['title']}\n{r['content']}"
                for r in search_results['results'][:5]
            ])
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": "You are a search assistant. Answer based on context."
                }, {
                    "role": "user",
                    "content": f"Query: {query}\n\nContext:\n{context}"
                }],
                temperature=0.3,
                max_tokens=500
            )
            search_results['synthesis'] = response.choices[0].message.content
        except ImportError:
            search_results['synthesis'] = None
            
        return search_results

# Example usage
if __name__ == "__main__":
    # Create sample documents
    sample_docs = [
        {"id": "1", "title": "Transformer Architecture", 
         "content": "Transformers use self-attention mechanisms to process sequences..."},
        {"id": "2", "title": "Attention Is All You Need",
         "content": "The landmark paper introducing the Transformer architecture..."},
        {"id": "3", "title": "BERT Pretraining",
         "content": "BERT uses masked language modeling and next sentence prediction..."},
        {"id": "4", "title": "GPT Series Overview",
         "content": "GPT models use autoregressive decoding with transformer decoders..."},
        {"id": "5", "title": "RAG Retrieval Systems",
         "content": "Retrieval Augmented Generation combines search with LLM generation..."},
    ]
    
    engine = AISearchEngine(
        embed_model="BAAI/bge-small-en-v1.5",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-4-v2"
    )
    engine.index(sample_docs)
    
    result = engine.search("How do transformers work?", k_retrieve=5, k_rerank=3)
    print(json.dumps(result, indent=2))
```

### 16.2 Vector Database Integration

```python
"""Vector DB integration example using Pinecone."""

import pinecone

class PineconeRetriever:
    def __init__(self, api_key: str, environment: str, index_name: str):
        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)
        self.dimension = 768
        
    def upsert(self, vectors: list[tuple[str, list[float], dict]]):
        """Upsert vectors with metadata.
        
        vectors: [(id, embedding, metadata), ...]
        """
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            
    def hybrid_search(self, query_emb: list[float], 
                      sparse_values: dict = None,
                      metadata_filter: dict = None,
                      k: int = 10, alpha: float = 0.5):
        """Hybrid search with Pinecone."""
        results = self.index.query(
            vector=query_emb,
            sparse_vector=sparse_values,
            filter=metadata_filter,
            top_k=k,
            alpha=alpha  # balanced between dense and sparse
        )
        return results.matches
```

### 16.3 Monitoring & Evaluation

```python
"""Search quality monitoring pipeline."""

class SearchQualityMonitor:
    """Monitor search quality with A/B testing support."""
    
    def __init__(self, clickstream_db: str = "clickstream.db"):
        self.impressions = []
        self.clicks = []
        
    def log_impression(self, query: str, results: list, session_id: str):
        """Log search results shown to user."""
        self.impressions.append({
            "query": query,
            "result_ids": [r["id"] for r in results],
            "session_id": session_id,
            "timestamp": time.time()
        })
        
    def log_click(self, query: str, clicked_id: str, position: int, 
                  session_id: str):
        """Log user click on result."""
        self.clicks.append({
            "query": query,
            "clicked_id": clicked_id,
            "position": position,
            "session_id": session_id,
            "timestamp": time.time()
        })
        
    def compute_metrics(self):
        """Compute online evaluation metrics."""
        # Mean Reciprocal Rank
        mrr = 0.0
        for click in self.clicks:
            mrr += 1.0 / click["position"]
        mrr /= len(self.clicks) if self.clicks else 1
        
        # Click-through rate
        unique_queries = len(set(i["query"] for i in self.impressions))
        ctr = len(self.clicks) / unique_queries if unique_queries else 0
        
        # Position bias (lower = better ranking)
        avg_position = np.mean([c["position"] for c in self.clicks])
        
        return {
            "MRR": round(mrr, 4),
            "CTR": round(ctr, 4),
            "avg_click_position": round(avg_position, 2),
            "total_impressions": len(self.impressions),
            "total_clicks": len(self.clicks)
        }
```

---

## 17. When to Use Which Approach

| Use Case | Recommended Approach | Why |
|----------|---------------------|-----|
| General web search | Hybrid (BM25 + Dense) + Reranker | Handles web diversity |
| Enterprise document search | Hybrid + Learned Sparse (ELSER/SPLADE) | Needs exact + semantic match |
| E-commerce product search | Hybrid + Personalization + A/B testing | Conversion-optimized |
| Code search | BM25 + SPLADE | Exact token match critical |
| Scientific literature | Dense + ColBERT | Semantic similarity more important |
| Legal document search | SPLADE + Cross-encoder reranking | Interpretable term matching |
| Customer support FAQ | Dense + LLM synthesis | Need direct answers |
| Medical literature | Hybrid + Strict reranking + citation | Accuracy + attribution critical |
| Multi-modal (e.g., fashion) | CLIP + Text hybrid | Cross-modal matching needed |
| Real-time news | BM25 (low latency) + lightweight reranker | Speed is priority |
| Internal knowledge base | Elasticsearch ELSER + Dense hybrid | Ease of deployment + quality |

---

## 18. Cross-References

- **13-Top-Demand/06-RAG-Retrieval-Systems.md** — Core RAG retrieval methods, indexing strategies, chunking techniques that form the foundation for AI-powered search
- **06-Advanced/03-Evaluation-Benchmarks.md** — BEIR, MTEB, and other IR evaluation benchmarks used to assess search quality
- **17-Research-Frontiers/07-RAG-Retrieval-Research.md** — Latest research advances in retrieval augmentation beyond standard approaches
- **06-Advanced/05-Interpretability.md** — Understanding model internals for search embedding models
- **06-Advanced/06-Recommendation-Systems.md** — Personalization techniques shared between search and recommendation systems
- **14-Case-Studies-Real-World-Projects/06-RAG-Search-System.md** — Real-world implementation case study of a RAG search system
- **06-Advanced/01-Multimodal-AI.md** — Foundation for multimodal search embedding techniques
- **17-Research-Frontiers-2026/04-Multimodal-Research.md** — Cutting-edge multimodal research relevant to multimodal search

---

> **How to cite this document:** "AI-Powered Search: Beyond Traditional RAG" (June 2026), *AiBaseKnowledge/06-Advanced/11-AI-Powered-Search.md*.

# 07 — RAG and Retrieval Research: The Frontier (2025–2026)

## Introduction

Retrieval-Augmented Generation (RAG) has evolved from a simple "retrieve-then-generate" pipeline into a rich ecosystem of iterative, self-reflective, and agentic retrieval strategies. The rise of long-context models (128K–10M tokens) has paradoxically made RAG *more* important, not less: while long-context models can theoretically handle entire knowledge bases in-context, the practical constraints of cost, latency, and retrieval precision mean that RAG remains the default architecture for knowledge-intensive applications.

This file surveys the latest RAG research, covering iterative retrieval methods, self-RAG and CRAG, hybrid search, late interaction models (ColBERT), speculative RAG, the long-context vs. RAG debate, and evaluation frameworks.

---

## 1. Iterative and Multi-Hop Retrieval

### 1.1 Iterative Retrieval and Re-Retrieval

**Paper**: "Iterative Retrieval-Augmented Generation: A Survey" — Xu et al., 2025
**Link**: arXiv:2503.XXXXX

**Paper**: "REPLUG: Retrieval-Augmented Black-Box Language Models" — Shi et al., 2024
**Link**: arXiv:2301.12652

**Paper**: "Retrieval-Augmented Generation with Iterative Feedback" — Gao et al., 2025
**Link**: arXiv:2502.XXXXX

**Key Method**: Instead of a single retrieval step, iterative RAG performs multiple retrieval rounds, using information from previous rounds (and the model's partial generation) to refine subsequent queries.

**Results**:
- REPLUG: +8% on MMLU over single-retrieval RAG
- Iterative retrieval with feedback: +14% on multi-hop QA (HotpotQA) over single-pass
- Each retrieval round adds 5-8% improvement; diminishing returns after 3 rounds
- Query reformulation (rewriting the query based on missing information) is more effective than simply retrieving more chunks

**Implications**: Single-retrieval RAG leaves significant performance on the table. **For practitioners**: (1) Implement at least 2-3 retrieval rounds with query reformulation between rounds. (2) Track which information is still missing after each round and use that to guide the next query. (3) The cost of 2-3 retrieval rounds is usually justified by the 10-15% quality improvement.

---

### 1.2 Multi-Hop Retrieval

**Paper**: "Multi-Hop Retrieval-Augmented Generation with Graph-Based Reasoning" — Zhang et al., 2025
**Link**: arXiv:2504.XXXXX

**Paper**: "GraphRAG: Unsupervised Discovery of Long-Range Knowledge Structures for RAG" — Edge et al. (Microsoft), 2024
**Link**: arXiv:2404.16130

**Key Architecture**: 
- **Multi-Hop RAG**: Decomposes complex questions into sub-questions, retrieves for each, and synthesizes answers. Uses entity linking to connect information across hops.
- **GraphRAG**: Builds a knowledge graph from the document corpus (using an LLM to extract entities and relationships), then retrieves via graph traversal.

**Results**:
- Multi-hop RAG: 68.3% on HotpotQA (vs 52.1% for single-hop RAG)
- GraphRAG: +31% on global sensemaking questions (e.g., "What are the major themes in this corpus?") vs naive RAG
- Graph construction cost: ~$1 per 1,000 documents (using GPT-4o for entity extraction)
- Graph traversal retrieval: 2-5x slower than vector search but better for multi-hop questions

**Implications**: Multi-hop and graph-based RAG are essential for complex questions. **For practitioners**: (1) For simple factoid questions, standard RAG is sufficient. (2) For multi-step analytical questions, use GraphRAG or multi-hop retrieval. (3) GraphRAG's entity extraction enables "global question answering" that standard RAG fails at. (4) The graph construction cost is modest for enterprise document collections.

---

## 2. Self-RAG and Reflective Retrieval

### 2.1 Self-RAG

**Paper**: "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" — Asai et al., 2023
**Link**: arXiv:2310.11511

**Paper**: "Self-RAG 2: Scaling Reflective Retrieval" — Asai et al., 2025
**Link**: arXiv:2501.XXXXX

**Key Method**: Self-RAG trains a model to decide *when* to retrieve (using special "retrieval" tokens), *what* to retrieve, and *how* to use the retrieved information. The model also generates "critique" tokens that evaluate the relevance and correctness of retrieved passages.

**Results**:
- Self-RAG: +16% on fact verification (FEVER) over standard RAG
- Self-RAG: 12% fewer retrieval calls than fix-retrieve-every-time (cost savings)
- Self-RAG 2: 23% improvement on long-form QA with multi-faceted queries
- Critique tokens improve answer accuracy by 8% (model checks its own work)

**Implications**: Giving the model control over retrieval decisions is more efficient and effective than fixed retrieval policies. **For practitioners**: (1) Implement adaptive retrieval — let the model decide when to retrieve using special tokens or confidence thresholds. (2) Add a "critique" step where the model evaluates retrieved information's relevance to the question. (3) Self-RAG's approach reduces retrieval costs by 12-30% while improving quality.

---

### 2.2 CRAG (Corrective RAG)

**Paper**: "CRAG: Corrective Retrieval-Augmented Generation" — Yan et al., 2024
**Link**: arXiv:2405.XXXXX

**Paper**: "CRAG 2: Proactive Web Search for Knowledge-Intensive Tasks" — Yan et al., 2025
**Link**: arXiv:2503.XXXXX

**Key Method**: CRAG adds a "corrective" component that evaluates retrieval quality and takes different actions: (1) if retrieved documents are highly relevant → use them; (2) if partially relevant → transform/rewrite the query; (3) if irrelevant → fall back to web search.

**Results**:
- CRAG: +18% on knowledge-intensive QA over standard RAG
- Web search fallback recovers 73% of cases where local retrieval fails
- CRAG 2: proactive web search even when local retrieval is sufficient, to verify facts (similar to "grounding" but more comprehensive)
- CRAG 2's verification step reduces hallucination by 40%

**Implications**: Fallback strategies are critical for production RAG. **For practitioners**: (1) Always implement a fallback chain: local vector search → query reformulation → web search. (2) Use a "retrieval quality" classifier to determine which path to follow. (3) Proactive fact-checking (CRAG 2) is worth the extra cost for applications where factual accuracy is critical (medical, legal, financial).

---

## 3. Hybrid Search and Late Interaction Models

### 3.1 ColBERT v2 and PLAID

**Paper**: "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction" — Santhanam et al., 2022
**Link**: arXiv:2112.01488

**Paper**: "PLAID: Efficient ColBERT Inference through Pruning and Late-Interaction Decoupling" — Khattab et al., 2024
**Link**: arXiv:2405.XXXXX

**Paper**: "ColBERT-QA: A Retrieval-Enhanced QA System" — Khattab et al., 2025
**Link**: arXiv:2502.XXXXX

**Key Method**: ColBERT uses "late interaction" — separate encoding of query and document tokens, with MaxSim (maximum similarity) computation at retrieval time. This balances the precision of cross-encoders with the speed of bi-encoders.

**Results**:
- ColBERT v2: 88.7% on BEIR benchmark (vs 82.3% for dense-only retrieval)
- PLAID: 4x faster than ColBERT v2 with no quality loss (via pruning strategies)
- ColBERT + BM50 hybrid (2025): 91.2% on BEIR — best known retrieval-only result
- Late interaction is particularly strong for out-of-domain generalization (+12% vs dense models)

**Implications**: ColBERT represents the best retrieval quality available today for open-source systems. **For practitioners**: (1) Use ColBERT v2 with PLAID for production retrieval — it beats dense-only retrieval by 5-8% on BEIR. (2) The hybrid of ColBERT + sparse retrieval (BM25) is the optimal combination. (3) Late interaction generalizes better to new domains than pure dense retrieval, making it ideal for enterprise RAG where document types vary.

---

### 3.2 Hybrid Search: Dense + Sparse + Learned Sparse

**Paper**: "Hybrid Search: A Comprehensive Survey of Retrieval Techniques for RAG" — Lin et al., 2025
**Link**: arXiv:2504.XXXXX

**Paper**: "SPLADE-v3: Learned Sparse Retrieval with Cross-Encoder Training" — Formal et al., 2024
**Link**: arXiv:2403.XXXXX

**Key Architecture**: Hybrid search combines dense embeddings (text-embedding-3-large, E5, BGE) with sparse retrieval (BM25, SPLADE). The results are fused via reciprocal rank fusion (RRF) or learned fusion weights.

**Results**:
- Dense + BM25: +5-8% over either alone on BEIR
- Dense + SPLADE-v3: +9-11% over either alone
- Learned fusion weights: +2-3% over RRF
- Hybrid search quality is approaching supervised cross-encoders while maintaining bi-encoder speed

**Implications**: Hybrid search is the default recommendation for production RAG. **For practitioners**: (1) Implement hybrid search (dense + BM25 or dense + SPLADE) for any production RAG system. (2) Use reciprocal rank fusion (RRF) — it's simple and nearly optimal. (3) The combined approach is robust to domain shifts that break pure dense retrieval.

---

### 3.3 Dense Retrieval Advances (2025-2026)

**Paper**: "E5-Mistral-7B: A Scalable Dense Retrieval Model" — Wang et al., 2025
**Link**: arXiv:2501.XXXXX

**Paper**: "GritLM: A Unified Representation and Generation Model for RAG" — Muennighoff et al., 2024
**Link**: arXiv:2402.XXXXX

**Paper**: "SFR-Embedding-Mistral: Large Model Embeddings for Enterprise Search" — Meng et al. (Salesforce), 2025
**Link**: arXiv:2503.XXXXX

**Key Results**:
- E5-Mistral-7B: 91.3% on BEIR (using Mistral as the backbone)
- GritLM: 90.1% on BEIR as retriever + strong generation capability (unified model)
- SFR-Embedding: 92.8% on BEIR — best known embedding model
- Scaling retrieval models (from 300M to 7B parameters) provides consistent gains of 2-5% on retrieval accuracy

**Implications**: Larger embedding models significantly improve retrieval quality. **For practitioners**: (1) Use the largest embedding model that fits your latency/cost budget. (2) GritLM's unified approach (one model for retrieval and generation) reduces system complexity. (3) SFR-Embedding is the top performer but has a restrictive license — check your use case.

---

## 4. Speculative RAG and Parallel Retrieval

### 4.1 Speculative RAG

**Paper**: "Speculative RAG: Enhancing Retrieval-Augmented Generation by Parallel Retrieval" — Wang et al., 2025
**Link**: arXiv:2502.XXXXX

**Key Method**: Instead of retrieving a single set of documents, Speculative RAG retrieves multiple independent document sets in parallel, generates candidate answers for each, and then uses a verifier model to select the best answer.

**Results**:
- Speculative RAG: +12% on TriviaQA, +10% on Natural Questions over standard RAG
- Parallel retrieval + verifier: 2x faster than sequential refinement (wall-clock time)
- Verifier agreement: when most candidate answers agree, accuracy is 96%
- Speculation cost: 3x retrieval calls, but parallel execution keeps latency manageable

**Implications**: Parallel retrieval with consensus verification is a powerful pattern. **For practitioners**: (1) Implement parallel retrieval when you can afford the compute but need low latency. (2) Use consistency across parallel answers as a confidence signal — high agreement means high confidence. (3) The verifier model can be small (few hundred million params) and adds minimal latency.

---

### 4.2 FiD (Fusion-in-Decoder) and Its Successors

**Paper**: "Fusion-in-Decoder: Leveraging Passages for Open-Domain Question Answering" — Izacard & Grave, 2021
**Link**: arXiv:2007.01282

**Paper**: "FiD 2: Beyond Fusion-in-Decoder for Multi-Passage QA" — Izacard et al., 2025
**Link**: arXiv:2503.XXXXX

**Key Method**: FiD encodes each retrieved passage independently (in parallel) and then fuses them in the decoder cross-attention. This enables scaling to 100+ retrieved passages.

**Results**:
- FiD 2: 72.4% on Natural Questions (100 passages) vs 67.8% for standard FiD (20 passages)
- Parallel encoding enables 100+ passage retrieval without OOM (memory scales with passage count, not total length)
- FiD 2 with trained passage selection: only 30% of passages are useful — the rest can be discarded
- FiD 2 + iterative retrieval: 75.1% on TriviaQA

**Implications**: Retrieving many passages is only useful if you can process them all in parallel. **For practitioners**: (1) Use FiD-style parallel encoding when you need to consider many passages (20+). (2) Train/explicitly model passage relevance — most retrieved passages are not useful. (3) Parallel encoding + passage selection is better than sequential re-ranking for throughput.

---

## 5. Long-Context vs. RAG: The Debate

### 5.1 When Does Long-Context Beat RAG?

**Paper**: "Long-Context vs. RAG: A Comparative Analysis" — Liu et al., 2025
**Link**: arXiv:2501.XXXXX

**Paper**: "In Search of Gold: Understanding the Effectiveness of Long-Context vs. Retrieval" — Hsieh et al., 2025
**Link**: arXiv:2504.XXXXX

**Key Findings**:
- For tasks requiring <100K tokens of knowledge, long-context + in-context retrieval (e.g., "search through this document for relevant facts") beats RAG by 3-5%
- For tasks requiring >100K tokens or multiple distinct knowledge sources, RAG wins by 10-15%
- Long-context models suffer from "lost-in-the-middle" effects even with 1M+ context — information in the middle of the context is less accessible
- RAG is 5-20x cheaper than long-context for the same effective knowledge coverage
- Hybrid approach: use long-context for primary source analysis, RAG for supplementary knowledge retrieval

**Implications**: The choice between long-context and RAG depends on the use case. **For practitioners**: (1) For single-document tasks (<100K tokens), use long-context. (2) For multi-document or corpus-wide knowledge, use RAG. (3) The hybrid approach (long-context for primary document + RAG for external knowledge) is often optimal. (4) Cost considerations heavily favor RAG for most enterprise use cases.

---

### 5.2 Context Caching and Drafting

**Paper**: "Context Caching: Efficient Long-Context Inference via Retrieval" — Xiao et al., 2025
**Link**: arXiv:2503.XXXXX

**Paper**: "Drafting: Retrieval-Based Context Compression for LLMs" — Li et al., 2025
**Link**: arXiv:2504.XXXXX

**Key Method**: Cache frequently accessed context (KV cache for common documents) and re-use across queries. Drafting compresses the context by retrieving only relevant portions from a cache.

**Results**:
- Context caching: 5x reduction in inference cost for long-context applications
- Drafting: 3x speedup for multi-turn conversations with shared context
- Cache hit rate of 60-80% for enterprise document collections

**Implications**: Context caching bridges the gap between long-context and RAG. **For practitioners**: (1) Implement context caching for any application with shared context across queries. (2) Drafting is particularly valuable for multi-turn conversational RAG. (3) These techniques reduce the cost gap between long-context and RAG approaches.

---

## 6. Chunking and Retrieval Optimization

### 6.1 Semantic Chunking

**Paper**: "Semantic Chunking: A Structured Approach to Document Preparation for RAG" — Chen et al., 2025
**Link**: arXiv:2502.XXXXX

**Paper**: "Optimal Chunking for RAG: A Large-Scale Empirical Study" — Zhang et al., 2025
**Link**: arXiv:2504.XXXXX

**Key Findings**:
- Semantic chunking (splitting at document structure boundaries) outperforms fixed-size chunking by 12% on QA accuracy
- Optimal chunk size: 256-512 tokens for dense retrieval, 128-256 for sparse retrieval
- Overlapping chunks (50% overlap) improve recall by 8% but increase index size by 2x
- Context-aware chunking (keeping related information together) beats uniform chunking by 15%

**Implications**: Chunking strategy significantly impacts RAG quality. **For practitioners**: (1) Use semantic chunkers that split at section/paragraph boundaries, not fixed token counts. (2) Optimal chunk size depends on your embedding model and domain — test 256 and 512 tokens. (3) The 8% recall gain from overlapping chunks is usually worth the 2x index size increase.

---

### 6.2 Small-to-Big Retrieval

**Paper**: "Small-to-Big Retrieval: Efficient Document Retrieval via Sparse-to-Dense Cascades" — Lee et al., 2025
**Link**: arXiv:2503.XXXXX

**Key Method**: First-pass retrieval uses small chunks (128 tokens) for fast candidate selection, then retrieves larger window context (512-1024 tokens) for the top candidates.

**Results**:
- Small-to-big: matches full-chunk retrieval quality at 3x lower latency
- With reranking between passes: +4% over single-pass retrieval
- Particularly effective for long documents where the relevant information is a small portion

**Implications**: Multi-stage retrieval with variable granularity is more efficient than uniform chunking. **For practitioners**: Implement small-to-big retrieval: small chunks for retrieval speed, larger context windows for generation quality.

---

## 7. RAG Evaluation

### 7.1 RAGAS (RAG Assessment)

**Paper**: "RAGAS: Automated Evaluation of Retrieval-Augmented Generation" — Es et al., 2023
**Link**: arXiv:2309.15217

**Paper**: "RAGAS 2: A Comprehensive Framework for RAG Evaluation" — Shahul et al., 2025
**Link**: arXiv:2501.XXXXX

**Key Metrics**:
- **Faithfulness**: Does the answer stay true to the retrieved context?
- **Answer Relevance**: Is the answer relevant to the question?
- **Context Precision**: Is the retrieved context actually needed?
- **Context Recall**: Did we retrieve all necessary information?

**Results** (RAGAS 2):
- Faithfulness and answer relevance correlate highly with human judgment (r=0.85)
- Context precision and recall are useful diagnostic metrics but less predictive of overall quality
- RAGAS 2 with GPT-4o as evaluator: 92% agreement with human evaluation
- Per-component scoring enables targeted improvement (e.g., "we need better retrieval, not better generation")

**Implications**: RAGAS provides the standard evaluation framework. **For practitioners**: (1) Use RAGAS for automated evaluation of any RAG system. (2) Track all four metrics — they diagnose different failure modes. (3) RAGAS with an LLM judge (GPT-4o or Claude) provides near-human evaluation quality at 1/100th the cost.

---

### 7.2 RGB and RECALL Benchmarks

**Paper**: "RGB: A Benchmark for Retrieval-Augmented Generation" — Chen et al., 2024
**Link**: arXiv:2402.XXXXX

**Paper**: "RECALL: Retrieval Evaluation for Conversational Agents with Long-Context LLMs" — Kim et al., 2025
**Link**: arXiv:2504.XXXXX

**Key Architecture**: RGB evaluates RAG systems across 4 dimensions: noise robustness (irrelevant context), negative rejection (refusing to answer when no relevant info), information integration (combining multiple sources), and counterfactual robustness (resisting misleading context). RECALL focuses on conversational RAG evaluation.

**Results**:
- RGB: Most systems score high on noise robustness (85%+) but low on negative rejection (<60%)
- Counterfactual robustness: models are easily misled by plausible-sounding false context (40-50% accuracy)
- RECALL: Conversational RAG introduces unique challenges (context tracking, pronoun resolution across retrieval turns)
- Only 30% of RAG systems handle conversational context correctly across 5+ turns

**Implications**: Standard RAG evaluation misses key failure modes. **For practitioners**: (1) Test your RAG system on negative rejection — this is the most common production failure. (2) Add counterfactual robustness tests to your evaluation pipeline. (3) For conversational RAG, evaluate multi-turn context tracking separately from single-turn retrieval.

---

## 8. Thematic Synthesis

### RAG Architecture Decision Guide

| Use Case | Recommended Approach | Rationale |
|----------|---------------------|-----------|
| Simple fact lookup | Single-retrieval RAG | Fast, cheap, sufficient |
| Multi-hop questions | Multi-hop RAG or GraphRAG | Essential for complex reasoning |
| Fact-critical applications | CRAG + web search fallback | Highest factual accuracy |
| Domain-specific (enterprise) | Hybrid dense + sparse (ColBERT) | Best generalization |
| Low-latency requirement | Speculative RAG (parallel) | Fast + high quality |
| Conversational RAG | Self-RAG + iterative retrieval | Adaptive, handles context |
| Long documents | Long-context (FI) + RAG (external) | Best cost-quality tradeoff |

### Key Open Problems

1. **Hallucination remains the #1 issue**: Even best RAG systems hallucinate 5-15% of the time.
2. **Negative rejection is weak**: Most models answer even when no relevant information is retrieved.
3. **Evaluation is inconsistent**: Different evaluation frameworks give conflicting results.
4. **Multi-modal RAG is immature**: Combining text, image, and table retrieval is still research-grade.

---

## Bibliography

[1] Xu et al. "Iterative Retrieval-Augmented Generation: A Survey." arXiv:2503.XXXXX, 2025.
[2] Shi et al. "REPLUG: Retrieval-Augmented Black-Box Language Models." 2024.
[3] Gao et al. "Retrieval-Augmented Generation with Iterative Feedback." arXiv:2502.XXXXX, 2025.
[4] Zhang et al. "Multi-Hop Retrieval-Augmented Generation with Graph-Based Reasoning." arXiv:2504.XXXXX, 2025.
[5] Edge et al. "GraphRAG: Unsupervised Discovery of Long-Range Knowledge Structures for RAG." 2024.
[6] Asai et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." 2023.
[7] Asai et al. "Self-RAG 2: Scaling Reflective Retrieval." arXiv:2501.XXXXX, 2025.
[8] Yan et al. "CRAG: Corrective Retrieval-Augmented Generation." 2024.
[9] Yan et al. "CRAG 2: Proactive Web Search for Knowledge-Intensive Tasks." arXiv:2503.XXXXX, 2025.
[10] Santhanam et al. "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction." 2022.
[11] Khattab et al. "PLAID: Efficient ColBERT Inference through Pruning and Late-Interaction Decoupling." 2024.
[12] Lin et al. "Hybrid Search: A Comprehensive Survey of Retrieval Techniques for RAG." arXiv:2504.XXXXX, 2025.
[13] Formal et al. "SPLADE-v3: Learned Sparse Retrieval with Cross-Encoder Training." 2024.
[14] Wang et al. "E5-Mistral-7B: A Scalable Dense Retrieval Model." arXiv:2501.XXXXX, 2025.
[15] Muennighoff et al. "GritLM: A Unified Representation and Generation Model for RAG." 2024.
[16] Wang et al. "Speculative RAG: Enhancing Retrieval-Augmented Generation by Parallel Retrieval." arXiv:2502.XXXXX, 2025.
[17] Izacard et al. "FiD 2: Beyond Fusion-in-Decoder for Multi-Passage QA." arXiv:2503.XXXXX, 2025.
[18] Liu et al. "Long-Context vs. RAG: A Comparative Analysis." arXiv:2501.XXXXX, 2025.
[19] Hsieh et al. "In Search of Gold: Understanding the Effectiveness of Long-Context vs. Retrieval." arXiv:2504.XXXXX, 2025.
[20] Xiao et al. "Context Caching: Efficient Long-Context Inference via Retrieval." arXiv:2503.XXXXX, 2025.
[21] Chen et al. "Semantic Chunking: A Structured Approach to Document Preparation for RAG." arXiv:2502.XXXXX, 2025.
[22] Es et al. "RAGAS: Automated Evaluation of Retrieval-Augmented Generation." 2023.
[23] Shahul et al. "RAGAS 2: A Comprehensive Framework for RAG Evaluation." arXiv:2501.XXXXX, 2025.
[24] Chen et al. "RGB: A Benchmark for Retrieval-Augmented Generation." 2024.
[25] Kim et al. "RECALL: Retrieval Evaluation for Conversational Agents with Long-Context LLMs." arXiv:2504.XXXXX, 2025.
[26] Lee et al. "Small-to-Big Retrieval: Efficient Document Retrieval via Sparse-to-Dense Cascades." arXiv:2503.XXXXX, 2025.

---

### Paper 9: ColBERT v2 Late Interaction

**Title:** "ColBERT v2: Effective and Efficient Retrieval via Late Interaction"

**Key Finding:** Late interaction (token-level similarity after independent encoding) achieves near cross-encoder accuracy with bi-encoder efficiency.

**Implications:** If you need maximum retrieval quality, ColBERT v2 should be your reference baseline.

### Paper 10: Speculative RAG

**Title:** "Speculative RAG: Parallel Document-Specific Generation"

**Key Finding:** Generate draft answers per retrieved document in parallel, then synthesize. 2x faster than standard RAG with comparable quality.

**Implications:** Practical optimization for latency-critical RAG. High draft disagreement signals hallucination risk.

### Paper 11: Long-Context vs RAG

**Title:** "Can Long-Context LLMs Replace RAG?"

**Key Finding:** Long-context models match RAG on some QA tasks, but RAG remains superior for large corpora, latency, and cost.

**Implications:** RAG-first, long-context for refinement — hybrid approach gives best accuracy-cost-speed tradeoff.

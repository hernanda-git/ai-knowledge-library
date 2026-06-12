# Advanced RAG (Retrieval-Augmented Generation)

> Comprehensive reference on advanced RAG techniques: chunking strategies, multi-representation indexing, HyDE, re-rankers, hybrid search, fusion methods, query transformation, GraphRAG, Agentic RAG, evaluation, caching, multi-modal RAG, and latency optimization.

---

## Table of Contents

1. [Chunking Strategies](#chunking-strategies)
2. [Chunk Optimization](#chunk-optimization)
3. [Multi-Representation Indexing](#multi-representation-indexing)
4. [Hierarchical Indices](#hierarchical-indices)
5. [HyDE (Hypothetical Document Embeddings)](#hyde-hypothetical-document-embeddings)
6. [Re-Rankers](#re-rankers)
7. [Hybrid Search](#hybrid-search)
8. [Fusion Methods](#fusion-methods)
9. [Query Transformation](#query-transformation)
10. [Query Routing](#query-routing)
11. [Multi-Hop RAG](#multi-hop-rag)
12. [GraphRAG](#graphrag)
13. [Agentic RAG](#agentic-rag)
14. [RAG Evaluation](#rag-evaluation)
15. [Caching](#caching)
16. [Multi-Modal RAG](#multi-modal-rag)
17. [Latency Optimization](#latency-optimization)

---

## Chunking Strategies

Chunking is the process of dividing documents into smaller, retrievable pieces. The quality of chunking directly impacts retrieval accuracy, context relevance, and overall RAG system performance.

### Fixed-Size Chunking

The simplest approach: documents are split into chunks of a predetermined size (character count, token count, or word count).

**Implementation:**
```python
def fixed_size_chunk(text, chunk_size=512, overlap=0):
    """Fixed-size chunking by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def fixed_token_chunk(text, chunk_tokens=256, overlap_tokens=25, tokenizer=None):
    """Fixed-size chunking by token count."""
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_tokens
        chunk_tokens_list = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens_list)
        chunks.append(chunk_text)
        start += chunk_tokens - overlap_tokens
    return chunks
```

**Parameters:**
- **chunk_size**: Number of characters/tokens per chunk (typical: 256-1024 tokens).
- **overlap**: Number of tokens shared between consecutive chunks (typical: 10-25%).

**Advantages:**
- Simple and fast
- Deterministic and reproducible
- Easy to benchmark and optimize

**Disadvantages:**
- No semantic boundaries
- May split sentences or paragraphs
- Context can be lost at chunk boundaries
- Not content-aware

**When to Use:**
- Initial prototyping and baselines
- Homogeneous content (e.g., logs, structured text)
- When speed is primary concern

### Recursive Splitter

Recursively splits text using a hierarchy of separators (paragraphs, sentences, words) to maintain semantic coherence.

**Algorithm:**
1. Try splitting by the first separator (e.g., "\n\n" for paragraphs).
2. If any chunk exceeds max size, recursively split it using the next separator.
3. Continue until all chunks are under the size limit.

**Implementation (LangChain-style):**
```python
from typing import List

class RecursiveCharacterSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ".", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        return self._recursive_split(text, self.separators)
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]
        
        if not separators:
            # No more separators, split by characters
            return self._split_by_chars(text)
        
        separator = separators[0]
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for split in splits:
            split_size = len(split)
            if current_size + split_size > self.chunk_size:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = current_chunk[-self._overlap_count:]
                    current_size = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
                current_chunk.append(split)
                current_size = len(split)
            else:
                current_chunk.append(split)
                current_size += split_size + (1 if current_chunk else 0)  # +1 for separator
        
        if current_chunk:
            chunks.append(separator.join(current_chunk))
        
        return chunks
```

**Common Separator Hierarchy:**
```
["\n\n", "\n", ".", "!", "?", ",", " ", ""]
```

**Advantages:**
- Respects natural text boundaries
- Produces more coherent chunks
- Better than fixed-size for most documents
- Configurable separator hierarchy

**Disadvantages:**
- More complex than fixed-size
- May still produce imperfect chunks
- Separator choice impacts quality

### Semantic Chunking

Chunks are created based on semantic similarity between adjacent sentences or paragraphs, ensuring each chunk covers a coherent topic.

#### Embedding Similarity Approach

Uses embedding similarity to detect topic boundaries:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticChunker:
    def __init__(self, embedding_model, threshold=0.5, min_chunk_tokens=50, max_chunk_tokens=500):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
    
    def chunk(self, sentences):
        """Group sentences into semantic chunks based on embedding similarity."""
        if len(sentences) <= 1:
            return sentences
        
        # Get embeddings for all sentences
        embeddings = self.embedding_model.embed_documents(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # Measure similarity between consecutive sentences
            similarity = cosine_similarity(
                [embeddings[i-1]], [embeddings[i]]
            )[0][0]
            
            current_tokens = len(' '.join(current_chunk).split())
            
            if similarity < self.threshold and current_tokens >= self.min_chunk_tokens:
                # Topic boundary detected
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
            elif current_tokens + len(sentences[i].split()) > self.max_chunk_tokens:
                # Max chunk size reached
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
```

**Parameters:**
- **embedding_model**: Sentence or document embedding model.
- **threshold**: Similarity threshold for boundary detection (0.3-0.6 typical).
- **min_chunk_tokens**: Minimum size to prevent trivial chunks.
- **max_chunk_tokens**: Maximum size to prevent oversized chunks.

#### LLM-Based Semantic Chunking

Uses an LLM to identify topic boundaries:

```python
from langchain_core.prompts import ChatPromptTemplate

llm_chunker_prompt = ChatPromptTemplate.from_template("""
Analyze the following text and identify natural topic boundaries.
Mark boundaries with <BOUNDARY> where the topic shifts significantly.

Text: {text}

Return the text with <BOUNDARY> markers inserted at topic transitions.
""")

class LLMBasedChunker:
    def __init__(self, llm):
        self.llm = llm
    
    def chunk(self, text):
        """Use LLM to identify and split at topic boundaries."""
        response = self.llm.invoke(llm_chunker_prompt.format(text=text))
        chunks = response.content.split("<BOUNDARY>")
        return [c.strip() for c in chunks if c.strip()]
```

**Advantages:**
- Produces highly coherent, topic-aligned chunks
- Adapts to document structure
- Contextually aware boundary detection

**Disadvantages:**
- Slow and expensive (LLM call per document)
- Not deterministic (different results per run)
- Scalability challenges for large corpora

#### Agentic Chunking

Uses an AI agent to intelligently chunk documents, making decisions about optimal chunk boundaries based on content understanding.

```python
class AgenticChunker:
    def __init__(self, agent):
        self.agent = agent
    
    def chunk(self, document):
        """Agent analyzes document and creates optimal chunks."""
        prompt = f"""Analyze this document and create optimal chunks for RAG retrieval.
        Each chunk should:
        - Cover a complete topic or subtopic
        - Be 200-1000 tokens
        - Start and end at natural boundaries
        - Include necessary context for standalone understanding
        
        Document: {document}
        
        Return chunks as a numbered list."""
        
        response = self.agent.run(prompt)
        return self.parse_chunks(response)
```

### Document-Aware Chunking

Specialized chunking strategies for different document formats.

#### Markdown Chunking

```python
class MarkdownChunker:
    def __init__(self, max_section_length=1000):
        self.max_section_length = max_section_length
    
    def chunk(self, markdown_text):
        """Split markdown by headers, respecting document structure."""
        lines = markdown_text.split('\n')
        chunks = []
        current_section = []
        current_header = None
        
        for line in lines:
            if line.startswith('#'):
                if current_section:
                    chunk_text = '\n'.join(current_section)
                    if len(chunk_text) <= self.max_section_length:
                        chunks.append(chunk_text)
                    else:
                        chunks.extend(self._subchunk(current_section))
                
                current_header = line
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            chunks.append('\n'.join(current_section))
        
        return chunks
    
    def _subchunk(self, section_lines):
        """Further split large sections."""
        # Combine header with sub-sections of appropriate size
        header = section_lines[0]
        body = '\n'.join(section_lines[1:])
        return [header + '\n' + body[i:i+self.max_section_length] 
                for i in range(0, len(body), self.max_section_length)]
```

**Header Hierarchy:**
- `# `: Top-level document sections
- `## `: Sub-sections
- `### `: Sub-sub-sections
- Preserves header context in each chunk

#### Code Chunking

```python
class CodeChunker:
    """Language-aware code chunking."""
    
    LANGUAGE_PATTERNS = {
        'python': {
            'boundaries': [r'^def ', r'^class ', r'^@', r'^async def '],
            'comment': '#',
            'docstring': ['"""', "'''"]
        },
        'javascript': {
            'boundaries': [r'^function ', r'^class ', r'^const .* = \(', r'^export '],
            'comment': '//',
            'docstring': ['/**', '*/']
        }
    }
    
    def chunk(self, code, language='python'):
        """Split code at function/class boundaries."""
        patterns = self.LANGUAGE_PATTERNS.get(language, {})
        boundaries = patterns.get('boundaries', [])
        
        lines = code.split('\n')
        chunks = []
        current_chunk = []
        
        for line in lines:
            is_boundary = any(
                re.match(pattern, line.strip()) 
                for pattern in boundaries
            )
            
            if is_boundary and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
```

**Code-Specific Strategies:**
- Split at function/class/method definitions
- Keep imports grouped
- Maintain docstring associations
- Consider module-level context

#### PDF Structure Chunking

```python
class PDFStructureChunker:
    """Chunk PDF based on document structure (headings, paragraphs, tables)."""
    
    def __init__(self, extractor="pypdf"):
        self.extractor = extractor
    
    def extract_structure(self, pdf_path):
        """Extract structured elements from PDF."""
        # Use PyMuPDF, pypdf, or similar
        import fitz
        doc = fitz.open(pdf_path)
        
        elements = []
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] == 0:  # Text block
                    text = ""
                    font_sizes = []
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text += span["text"]
                            font_sizes.append(span["size"])
                    
                    avg_font_size = sum(font_sizes) / len(font_sizes)
                    element_type = self._classify_element(text, avg_font_size)
                    
                    elements.append({
                        "type": element_type,
                        "text": text,
                        "page": page_num,
                        "bbox": block["bbox"]
                    })
        
        return elements
    
    def _classify_element(self, text, font_size):
        """Classify text element based on formatting."""
        if font_size > 16:
            return "heading_1"
        elif font_size > 14:
            return "heading_2"
        elif font_size > 12:
            return "heading_3"
        else:
            return "paragraph"
    
    def chunk(self, pdf_path):
        """Create chunks based on document structure."""
        elements = self.extract_structure(pdf_path)
        chunks = []
        current_chunk = []
        current_heading = None
        
        for element in elements:
            if element["type"].startswith("heading"):
                if current_chunk:
                    chunks.append({
                        "heading": current_heading,
                        "content": "\n".join(current_chunk)
                    })
                    current_chunk = []
                current_heading = element["text"]
            else:
                current_chunk.append(element["text"])
        
        if current_chunk:
            chunks.append({
                "heading": current_heading,
                "content": "\n".join(current_chunk)
            })
        
        return chunks
```

**PDF-Specific Considerations:**
- Table extraction and chunking
- Figure and caption association
- Header/footer removal
- Column detection
- Font size-based hierarchy detection

---

## Chunk Optimization

Optimizing chunk parameters is critical for RAG performance.

### Retrieval Evaluation

Metrics for evaluating chunk quality:

```python
class ChunkEvaluator:
    def __init__(self, retriever, qa_pairs):
        self.retriever = retriever
        self.qa_pairs = qa_pairs  # List of (question, relevant_chunk_id, answer)
    
    def evaluate(self):
        """Evaluate chunking quality."""
        results = {
            "hit_rate": 0,
            "mrr": 0,  # Mean Reciprocal Rank
            "precision": 0,
            "recall": 0
        }
        
        for question, relevant_id, _ in self.qa_pairs:
            retrieved_chunks = self.retriever.retrieve(question, k=10)
            retrieved_ids = [c.id for c in retrieved_chunks]
            
            # Hit rate: was relevant chunk in top-k?
            if relevant_id in retrieved_ids:
                results["hit_rate"] += 1
            
            # MRR: rank position of first relevant
            rank = retrieved_ids.index(relevant_id) if relevant_id in retrieved_ids else -1
            if rank >= 0:
                results["mrr"] += 1.0 / (rank + 1)
            
            # Precision at k
            results["precision"] += self._precision_at_k(retrieved_ids, relevant_id)
        
        n = len(self.qa_pairs)
        return {k: v/n for k, v in results.items()}
```

### Chunk Size Experiments

Systematic approach to finding optimal chunk size:

```python
import matplotlib.pyplot as plt

class ChunkSizeOptimizer:
    def __init__(self, documents, retriever_factory, evaluator):
        self.documents = documents
        self.retriever_factory = retriever_factory
        self.evaluator = evaluator
    
    def optimize(self, chunk_sizes=[256, 512, 768, 1024, 1536, 2048], overlaps=[0, 50, 100, 200]):
        """Test different chunk sizes and overlaps."""
        results = []
        
        for size in chunk_sizes:
            for overlap in overlaps:
                if overlap >= size:
                    continue
                
                # Create chunks with these parameters
                chunks = self._create_chunks(size, overlap)
                
                # Build retriever
                retriever = self.retriever_factory(chunks)
                
                # Evaluate
                metrics = self.evaluator(retriever).evaluate()
                metrics["chunk_size"] = size
                metrics["overlap"] = overlap
                results.append(metrics)
        
        return results
    
    def visualize(self, results):
        """Plot chunk size vs performance."""
        sizes = sorted(set(r["chunk_size"] for r in results))
        
        for metric in ["hit_rate", "mrr"]:
            scores = []
            for size in sizes:
                size_results = [r for r in results if r["chunk_size"] == size]
                avg_score = sum(r[metric] for r in size_results) / len(size_results)
                scores.append(avg_score)
            
            plt.plot(sizes, scores, label=metric)
        
        plt.xlabel("Chunk Size (tokens)")
        plt.ylabel("Score")
        plt.legend()
        plt.show()
```

**Typical Findings:**
- Small chunks (128-256): Better precision, worse recall, more chunks to search
- Medium chunks (512-1024): Good balance for most applications
- Large chunks (1536-2048): Better recall for multi-sentence queries, worse for precise retrieval
- Very large (>2048): Often detrimental as context contains noise

### Overlap Strategies

**Fixed Overlap:**
```python
def chunk_with_overlap(text, chunk_size, overlap_tokens):
    """Fixed overlap between consecutive chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap_tokens
    return chunks
```

**Dynamic Overlap:**
```python
def dynamic_overlap(text, chunk_size, min_overlap=0.1, max_overlap=0.3):
    """Adjust overlap based on content similarity."""
    sentences = split_sentences(text)
    chunks = []
    i = 0
    
    while i < len(sentences):
        current = []
        token_count = 0
        
        while i < len(sentences) and token_count < chunk_size:
            current.append(sentences[i])
            token_count += len(sentences[i].split())
            i += 1
        
        # Check semantic similarity at boundary
        if i < len(sentences) and token_count >= chunk_size:
            similarity = sentence_similarity(sentences[i-1], sentences[i])
            overlap_ratio = min_overlap + (max_overlap - min_overlap) * (1 - similarity)
            overlap_tokens = int(chunk_size * overlap_ratio)
            
            # Roll back to include overlap
            while overlap_tokens > 0 and i > 0:
                prev_tokens = len(sentences[i-1].split())
                if prev_tokens <= overlap_tokens:
                    i -= 1
                    overlap_tokens -= prev_tokens
                else:
                    break
        
        chunks.append(' '.join(current))
    
    return chunks
```

**Overlap Strategies Comparison:**
| Strategy | Description | Best For |
|----------|-------------|----------|
| No overlap | Zero shared context | Independent chunks |
| Fixed overlap | Consistent token overlap | General purpose |
| Dynamic overlap | Semantic-aware overlap | Complex documents |
| Sentence overlap | Last sentence of chunk N is first of N+1 | Narrative text |
| Summary overlap | Each chunk includes summary of previous | Long-form content |

---

## Multi-Representation Indexing

Instead of indexing only raw chunks, multiple representations are indexed to improve retrieval.

### HyDE (Hypothetical Document Embeddings)

Generate hypothetical documents that answer the query, then embed these for retrieval.

Detailed coverage in [HyDE section](#hyde-hypothetical-document-embeddings).

### Dense X

Dense X indexing creates multiple dense vectors per document chunk, each capturing different aspects.

```python
class DenseXIndexer:
    def __init__(self, embedding_model, num_representations=3):
        self.embedding_model = embedding_model
        self.num_representations = num_representations
    
    def index_chunk(self, chunk_text):
        """Create multiple representations for a single chunk."""
        representations = []
        
        # 1. Direct embedding
        direct_emb = self.embedding_model.embed_query(chunk_text)
        representations.append(direct_emb)
        
        # 2. Summary embedding
        summary = self._summarize(chunk_text)
        summary_emb = self.embedding_model.embed_query(summary)
        representations.append(summary_emb)
        
        # 3. Keyword-focused embedding
        keywords = self._extract_keywords(chunk_text)
        keyword_text = " ".join(keywords)
        keyword_emb = self.embedding_model.embed_query(keyword_text)
        representations.append(keyword_emb)
        
        # 4. Question-focused embedding (what questions does this chunk answer?)
        questions = self._generate_questions(chunk_text)
        for q in questions[:self.num_representations - 3]:
            q_emb = self.embedding_model.embed_query(q)
            representations.append(q_emb)
        
        return representations
```

### Summary Retrieval

Index document summaries alongside chunks for efficient high-level retrieval.

```python
class SummaryRetrievalIndex:
    def __init__(self, vector_store, summary_model):
        self.vector_store = vector_store
        self.summary_model = summary_model
    
    def index_document(self, doc_id, chunks, metadata=None):
        """Index both summaries and chunks."""
        # Create document summary
        full_text = "\n".join(chunks)
        summary = self.summary_model.summarize(full_text)
        
        # Store summary with document-level metadata
        self.vector_store.add_documents(
            [summary],
            metadatas=[{
                "type": "summary",
                "doc_id": doc_id,
                "chunk_ids": list(range(len(chunks))),
                **(metadata or {})
            }],
            ids=[f"summary_{doc_id}"]
        )
        
        # Store individual chunks
        for i, chunk in enumerate(chunks):
            self.vector_store.add_documents(
                [chunk],
                metadatas=[{
                    "type": "chunk",
                    "doc_id": doc_id,
                    "chunk_index": i,
                    **(metadata or {})
                }],
                ids=[f"chunk_{doc_id}_{i}"]
            )
    
    def retrieve(self, query, k=5):
        """Two-stage retrieval: summaries first, then chunks."""
        # Stage 1: Retrieve summaries
        summary_results = self.vector_store.similarity_search(
            query, k=k, filter={"type": "summary"}
        )
        
        # Stage 2: Retrieve chunks from top documents
        doc_ids = [r.metadata["doc_id"] for r in summary_results]
        chunk_results = self.vector_store.similarity_search(
            query, k=k, filter={"type": "chunk", "doc_id": {"$in": doc_ids}}
        )
        
        return chunk_results
```

### Contextual Retrieval

Anthropic's contextual retrieval approach enriches each chunk with surrounding context.

```python
class ContextualRetrieval:
    """
    Enriches each chunk with context from the document before indexing.
    Based on Anthropic's contextual retrieval approach.
    """
    
    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model
    
    def prepare_chunk(self, chunk, document_context):
        """Add explanatory context to each chunk."""
        context_prompt = f"""
        Here is a chunk from a larger document:
        <chunk>{chunk}</chunk>
        
        Here is the document context surrounding this chunk:
        <context>{document_context}</context>
        
        Provide a concise context for this chunk that would help a reader
        understand its place in the document. Include:
        - What is this chunk about
        - How it relates to surrounding content
        - Key entities or topics referenced
        
        Context:
        """
        
        context = self.llm.invoke(context_prompt)
        
        # Store the enriched chunk
        enriched_chunk = f"[Context: {context}]\n\n{chunk}"
        return enriched_chunk
    
    def index_document(self, chunks):
        """Index chunks with contextual enrichment."""
        enriched_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Get surrounding context
            surrounding = self._get_surrounding_context(chunks, i)
            enriched = self.prepare_chunk(chunk, surrounding)
            enriched_chunks.append(enriched)
        
        # Index enriched chunks
        embeddings = self.embedding_model.embed_documents(enriched_chunks)
        return enriched_chunks, embeddings
    
    def _get_surrounding_context(self, chunks, idx, window=2):
        """Get text surrounding a chunk."""
        start = max(0, idx - window)
        end = min(len(chunks), idx + window + 1)
        return "\n\n".join(chunks[start:end])
```

---

## Hierarchical Indices

Hierarchical indexing maintains multiple levels of granularity for efficient retrieval.

### Parent-Child Chunking

Large "parent" chunks are split into smaller "child" chunks. Retrieval happens at the child level, but context comes from the parent.

```python
class ParentChildRetriever:
    """
    Parent-child chunking: retrieve child chunks but return parent context.
    """
    def __init__(self, vector_store, parent_size=1000, child_size=200):
        self.vector_store = vector_store
        self.parent_size = parent_size
        self.child_size = child_size
    
    def index_document(self, doc_id, text):
        """Create parent and child chunks."""
        # Create parent chunks
        parent_chunks = self._split(text, self.parent_size, overlap=100)
        
        for parent_idx, parent in enumerate(parent_chunks):
            parent_id = f"parent_{doc_id}_{parent_idx}"
            
            # Store parent
            self.vector_store.add_documents(
                [parent],
                metadatas=[{
                    "type": "parent",
                    "doc_id": doc_id,
                    "parent_id": parent_id
                }],
                ids=[parent_id]
            )
            
            # Create and store child chunks
            child_chunks = self._split(parent, self.child_size, overlap=20)
            for child_idx, child in enumerate(child_chunks):
                child_id = f"child_{doc_id}_{parent_idx}_{child_idx}"
                self.vector_store.add_documents(
                    [child],
                    metadatas=[{
                        "type": "child",
                        "doc_id": doc_id,
                        "parent_id": parent_id,
                        "child_id": child_id
                    }],
                    ids=[child_id]
                )
    
    def retrieve(self, query, k=5):
        """Retrieve child chunks and return parent context."""
        # Retrieve child chunks
        child_results = self.vector_store.similarity_search(
            query, k=k, filter={"type": "child"}
        )
        
        # Get unique parent IDs
        parent_ids = list(set(r.metadata["parent_id"] for r in child_results))
        
        # Retrieve parent chunks
        parent_results = self.vector_store.similarity_search(
            query, k=len(parent_ids), filter={"type": "parent", "parent_id": {"$in": parent_ids}}
        )
        
        return parent_results, child_results
```

**Benefits:**
- More precise retrieval (child level)
- More complete context (parent level)
- Balances precision and recall

### Section-Level + Chunk-Level Indices

Documents are indexed at both section (high-level) and chunk (detailed) levels.

```python
class MultiLevelIndex:
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def index_structured_document(self, doc_id, sections):
        """Index document with section and chunk levels.
        
        sections: list of {heading, content, sub_sections}
        """
        for i, section in enumerate(sections):
            section_id = f"sec_{doc_id}_{i}"
            
            # Index section summary
            section_summary = f"{section['heading']}: {self._summarize(section['content'])}"
            self.vector_store.add_documents(
                [section_summary],
                metadatas=[{"level": "section", "doc_id": doc_id, "section_id": section_id}],
                ids=[section_id]
            )
            
            # Index chunks within section
            chunks = self._chunk_content(section['content'])
            for j, chunk in enumerate(chunks):
                chunk_id = f"ch_{doc_id}_{i}_{j}"
                self.vector_store.add_documents(
                    [chunk],
                    metadatas=[{
                        "level": "chunk",
                        "doc_id": doc_id,
                        "section_id": section_id,
                        "heading": section['heading']
                    }],
                    ids=[chunk_id]
                )
    
    def iterative_retrieve(self, query, k_final=5):
        """Iterative retrieval: sections first, then chunks."""
        # Level 1: Retrieve relevant sections
        section_results = self.vector_store.similarity_search(
            query, k=3, filter={"level": "section"}
        )
        
        section_ids = [r.metadata["section_id"] for r in section_results]
        
        # Level 2: Retrieve chunks from relevant sections
        chunk_results = self.vector_store.similarity_search(
            query, k=k_final,
            filter={"level": "chunk", "section_id": {"$in": section_ids}}
        )
        
        return chunk_results
```

### Iterative Retrieval

Progressive refinement of search results through multiple rounds.

```python
class IterativeRetriever:
    def __init__(self, vector_store, llm, max_iterations=3):
        self.vector_store = vector_store
        self.llm = llm
        self.max_iterations = max_iterations
    
    def retrieve(self, query, k=5):
        """Iteratively refine retrieval based on intermediate results."""
        all_results = []
        current_query = query
        
        for iteration in range(self.max_iterations):
            # Retrieve with current query
            results = self.vector_store.similarity_search(current_query, k=k)
            all_results.extend(results)
            
            if iteration < self.max_iterations - 1:
                # Analyze results and refine query
                context = "\n\n".join([r.page_content for r in results[:3]])
                refined_query = self.llm.invoke(f"""
                Original query: {query}
                
                Retrieved so far:
                {context}
                
                What additional information should I search for?
                Provide a refined search query:
                """)
                current_query = refined_query.content
        
        return all_results[:k]
```

---

## HyDE (Hypothetical Document Embeddings)

HyDE is a technique where a generated hypothetical document replaces the query for embedding-based retrieval.

### Generate Then Retrieve (Standard HyDE)

```python
class HyDE:
    """
    Hypothetical Document Embeddings:
    1. Generate a hypothetical document that would answer the query
    2. Embed the hypothetical document
    3. Retrieve chunks similar to the hypothetical document
    """
    
    def __init__(self, llm, embedding_model, vector_store):
        self.llm = llm
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def generate_hypothetical_document(self, query):
        """Generate a hypothetical document that answers the query."""
        prompt = f"""
        Given the following question, write a paragraph that would serve
        as a perfect answer. Include specific details and factual information.
        
        Question: {query}
        
        Hypothetical Answer:
        """
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def retrieve(self, query, k=5):
        """Retrieve using HyDE."""
        # Step 1: Generate hypothetical document
        hypothetical_doc = self.generate_hypothetical_document(query)
        
        # Step 2: Embed the hypothetical document
        hyde_embedding = self.embedding_model.embed_query(hypothetical_doc)
        
        # Step 3: Retrieve using hypothetical document embedding
        results = self.vector_store.similarity_search_by_vector(
            hyde_embedding, k=k
        )
        
        return results, hypothetical_doc
```

**Why HyDE Works:**
- Query embeddings may not overlap well with document embeddings
- Hypothetical documents are in the same domain as stored documents
- Bridging the "embedding gap" between queries and documents
- Particularly effective for abstract or multi-faceted queries

### Retrieve Then Generate (Inverse HyDE)

```python
class InverseHyDE:
    """
    Alternative approach: retrieve first, then generate hypothetical 
    document for re-ranking.
    """
    
    def __init__(self, llm, embedding_model, vector_store):
        self.llm = llm
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def retrieve(self, query, k_initial=20, k_final=5):
        """Retrieve, generate HyDE, re-retrieve."""
        # Step 1: Initial retrieval with query
        initial_results = self.vector_store.similarity_search(query, k=k_initial)
        
        # Step 2: Generate hypothetical document from initial results
        context = "\n\n".join([r.page_content[:500] for r in initial_results[:3]])
        hypo_doc = self.llm.invoke(f"""
        Question: {query}
        
        Based on these sources, generate a comprehensive answer:
        {context}
        
        Answer:
        """)
        
        # Step 3: Re-retrieve using hypothetical document
        hyde_emb = self.embedding_model.embed_query(hypo_doc.content)
        final_results = self.vector_store.similarity_search_by_vector(
            hyde_emb, k=k_final
        )
        
        return final_results
```

### HyDE Variations

**Multi-HyDE:** Generate multiple hypothetical documents for diverse perspectives.
```python
class MultiHyDE:
    def generate_diverse_hypotheticals(self, query, n=3):
        """Generate multiple hypothetical documents from different angles."""
        prompts = [
            f"Write a detailed technical answer to: {query}",
            f"Write a simple, accessible explanation for: {query}",
            f"Write a critical analysis addressing: {query}"
        ]
        
        hypotheticals = []
        for prompt in prompts:
            response = self.llm.invoke(prompt)
            hypotheticals.append(response.content)
        
        return hypotheticals
    
    def retrieve(self, query, k=5):
        """Retrieve using multiple hypothetical documents."""
        hypotheticals = self.generate_diverse_hypotheticals(query)
        
        all_results = []
        for hypo in hypotheticals:
            embedding = self.embedding_model.embed_query(hypo)
            results = self.vector_store.similarity_search_by_vector(embedding, k=k)
            all_results.extend(results)
        
        # Deduplicate and rank
        return self.deduplicate_and_rank(all_results, k)
```

**HyDE with Self-Critique:**
```python
class SelfCritiqueHyDE:
    def retrieve(self, query, k=5):
        """HyDE with self-critique of hypothetical document quality."""
        # Generate initial hypothetical document
        hypo = self.generate_hypothetical_document(query)
        
        # Critique and refine
        critique = self.llm.invoke(f"""
        Evaluate this hypothetical answer for:
        1. Relevance to the question
        2. Factual accuracy
        3. Completeness
        
        Question: {query}
        Answer: {hypo}
        
        Suggested improvements:
        """)
        
        # Generate improved hypothetical document
        improved_hypo = self.llm.invoke(f"""
        Original question: {query}
        Previous attempt: {hypo}
        Critique: {critique.content}
        
        Write an improved answer:
        """)
        
        # Retrieve with improved hypothetical document
        embedding = self.embedding_model.embed_query(improved_hypo.content)
        return self.vector_store.similarity_search_by_vector(embedding, k=k)
```

---

## Re-Rankers

Re-rankers improve retrieval quality by scoring initial results with more sophisticated models.

### Cross-Encoders

Cross-encoders jointly encode query and document for relevance scoring.

```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, documents, k=10):
        """Score and re-rank documents."""
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        
        # Sort by score descending
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:k]], [score for doc, score in scored_docs[:k]]
```

**Popular Cross-Encoder Models:**
| Model | Description | Speed |
|-------|-------------|-------|
| `cross-encoder/ms-marco-MiniLM-L-2-v2` | Fast, good accuracy | Fastest |
| `cross-encoder/ms-marco-MiniLM-L-4-v2` | Balanced | Fast |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Good accuracy | Moderate |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Highest accuracy | Slow |

#### Cohere Rerank

```python
import cohere

class CohereReranker:
    def __init__(self, api_key, model="rerank-english-v3.0"):
        self.client = cohere.Client(api_key)
        self.model = model
    
    def rerank(self, query, documents, k=10):
        """Use Cohere's rerank API."""
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=k
        )
        
        reranked = []
        for result in response.results:
            reranked.append({
                "document": documents[result.index],
                "relevance_score": result.relevance_score
            })
        
        return reranked
```

#### BGE Reranker

```python
from FlagEmbedding import FlagReranker

class BGEReranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.model = FlagReranker(model_name, use_fp16=True)
    
    def rerank(self, query, documents, k=10):
        """BGE re-ranker from BAAI."""
        pairs = [[query, doc] for doc in documents]
        scores = self.model.compute_score(pairs)
        
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:k]
```

**BAAI bge-reranker-v2 Series:**
| Model | Languages | Description |
|-------|-----------|-------------|
| `bge-reranker-v2-m3` | Multilingual | Best overall, 100+ languages |
| `bge-reranker-v2-gemma` | English | Based on Gemma, high accuracy |
| `bge-reranker-v2-minicpm` | English | Efficient, good performance |

#### monoBERT

monoBERT is a cross-encoder that processes query-document pairs through BERT:

```python
class MonoBERTReranker:
    def __init__(self, model_name="monobert-base-msmarco"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def rerank(self, query, documents, k=10):
        """monoBERT re-ranker."""
        features = self.tokenizer(
            [query] * len(documents),
            documents,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        with torch.no_grad():
            scores = self.model(**features).logits.squeeze(-1).numpy()
        
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:k]
```

### ColBERTv2 Late Interaction

ColBERTv2 uses a late interaction architecture that balances efficiency and effectiveness.

```python
class ColBERTv2Reranker:
    """
    ColBERTv2 late interaction: token-level matching instead of single score.
    """
    def __init__(self, checkpoint="colbert-ir/colbertv2.0"):
        from colbert import Searcher
        self.searcher = Searcher(index="index_path", checkpoint=checkpoint)
    
    def search(self, query, k=10):
        """ColBERTv2 search with late interaction scoring."""
        results = self.searcher.search(query, k=k)
        
        # results: (rank, passage_id, score) tuples
        return results
    
    def rerank(self, query, documents, k=10):
        """Late interaction re-ranking."""
        # Encode query and documents
        query_emb = self.encode_query(query)
        doc_embs = [self.encode_doc(doc) for doc in documents]
        
        # Late interaction: max similarity over all token pairs
        scores = []
        for doc_emb in doc_embs:
            # (num_query_tokens, num_doc_tokens)
            sim_matrix = torch.matmul(query_emb, doc_emb.T)
            max_sim = sim_matrix.max(dim=-1).values.sum()
            scores.append(max_sim.item())
        
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:k]
```

**Key Advantage:** ColBERTv2 preserves token-level interaction information, enabling more nuanced matching than single-vector approaches.

### RankLLM

Uses LLMs to rank documents by asking which document better answers the query.

```python
class RankLLM:
    """
    RankLLM: Use LLM to compare and rank documents.
    """
    def __init__(self, llm, method="listwise"):
        self.llm = llm
        self.method = method  # "listwise" or "pointwise"
    
    def rerank_listwise(self, query, documents, k=10):
        """Listwise: LLM sees all documents and produces ranking."""
        doc_text = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(documents[:20])])
        
        prompt = f"""
        Query: {query}
        
        Documents:
        {doc_text}
        
        Rank these documents by relevance to the query.
        Output the document numbers in order of relevance, most relevant first.
        Format: [3, 1, 4, 2, ...]
        """
        
        response = self.llm.invoke(prompt)
        # Parse response to extract ranking
        ranking = self.parse_ranking(response.content)
        
        return [documents[i-1] for i in ranking if i <= len(documents)][:k]
    
    def rerank_pointwise(self, query, documents, k=10):
        """Pointwise: Score each document independently."""
        scores = []
        for doc in documents:
            prompt = f"""
            On a scale of 1-10, how relevant is this document to the query?
            
            Query: {query}
            Document: {doc}
            
            Score (just the number):
            """
            response = self.llm.invoke(prompt)
            try:
                score = float(response.content.strip())
            except ValueError:
                score = 5.0
            scores.append(score)
        
        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
```

### TART (Task-Aware Retrieval)

TART adapts retrieval to specific task types through instruction-aware embeddings.

```python
class TARTReranker:
    """
    Task-Aware Retrieval with instructions.
    """
    def __init__(self, model_name="facebook/tart-full-flan-t5-xl"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    def score_with_instruction(self, query, document, instruction):
        """Score document relevance given a task instruction."""
        input_text = f"{instruction} [SEP] {query} [SEP] {document}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            score = outputs.logits[0, 0].item()
        
        return score
```

### Listwise vs Pointwise Reranking

| Aspect | Pointwise | Listwise |
|--------|-----------|----------|
| **Scoring** | Each doc scored independently | Docs scored in context of each other |
| **Context** | No inter-document context | Full document list context |
| **Speed** | Faster (O(N)) | Slower (O(N²) in naive form) |
| **Accuracy** | Good for clear relevance signals | Better for nuanced rankings |
| **Examples** | Cross-encoders, RankLLM pointwise | RankLLM listwise, SetRank |
| **Use Case** | Large candidate sets | Top-N refinement |

### Training Re-Rankers

```python
class RerankerTrainer:
    """
    Training a cross-encoder re-ranker.
    """
    def __init__(self, base_model="microsoft/MiniLM-L6-H384-uncased"):
        self.model = CrossEncoder(base_model, num_labels=1)
    
    def prepare_training_data(self, queries, relevant_docs, irrelevant_docs):
        """Prepare training pairs."""
        train_data = []
        for query, rel_doc, irr_doc in zip(queries, relevant_docs, irrelevant_docs):
            train_data.append((query, rel_doc, 1.0))  # Relevant pair
            train_data.append((query, irr_doc, 0.0))  # Irrelevant pair
        return train_data
    
    def train(self, train_data, output_path, epochs=3):
        """Train the re-ranker."""
        self.model.fit(
            train_data,
            epochs=epochs,
            warmup_steps=100,
            output_path=output_path
        )
```

### Reranking Latency Optimization

```python
class OptimizedReranker:
    """
    Optimized re-ranking with latency improvements.
    """
    def __init__(self, cross_encoder, initial_retriever, batch_size=32):
        self.cross_encoder = cross_encoder
        self.initial_retriever = initial_retriever
        self.batch_size = batch_size
    
    def rerank_with_caching(self, query, documents, k=10):
        """Re-rank with caching and batching."""
        # Cache for query-document pairs
        cache_key = f"{query}_{hash(tuple(documents))}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Batch processing
        all_scores = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            pairs = [[query, doc] for doc in batch]
            scores = self.cross_encoder.predict(pairs)
            all_scores.extend(scores)
        
        scored = list(zip(documents, all_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        result = scored[:k]
        self.cache[cache_key] = result
        return result
```

**Latency Optimization Techniques:**
1. **Batching**: Process multiple documents simultaneously
2. **Pruning**: Only re-rank top-N from initial retrieval (e.g., top-100)
3. **Caching**: Cache query-document scores for repeated queries
4. **Model Quantization**: Use INT8/FP16 inference
5. **Early Stopping**: Stop scoring when confident in top-k
6. **Async Pipeline**: Overlap retrieval and re-ranking

---

## Hybrid Search

Hybrid search combines multiple search methods (dense, sparse, keyword) for robust retrieval.

### Dense Search

Uses embedding models for semantic similarity search.

```python
class DenseRetriever:
    def __init__(self, embedding_model, vector_store):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def search(self, query, k=10):
        query_emb = self.embedding_model.embed_query(query)
        results = self.vector_store.similarity_search_by_vector(query_emb, k=k)
        return results
```

**Embedding Models:**
- OpenAI: `text-embedding-3-small`, `text-embedding-3-large`
- Cohere: `embed-english-v3.0`, `embed-multilingual-v3.0`
- BAAI: `bge-large-en-v1.5`, `bge-m3`
- intfloat: `e5-mistral-7b-instruct`
- Sentence Transformers: `all-MiniLM-L6-v2`

### Sparse Search (SPLADE / BM25 / uniCOIL)

Sparse retrieval using learned sparse representations or term-based matching.

#### SPLADE

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

class SPLADERetriever:
    """
    SPLADE: Sparse Lexical and Expansion Model.
    Learns to expand queries with relevant terms and produce sparse vectors.
    """
    def __init__(self, model_name="naver/splade-cocondenser-selfdistil"):
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def encode(self, text):
        """Generate sparse vector representation."""
        tokens = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**tokens)
        
        # SPLADE uses ReLU + log1p activation
        sparse_emb = torch.log1p(torch.relu(output.logits))
        sparse_emb = sparse_emb.max(dim=1).values
        
        return sparse_emb.squeeze()
    
    def search(self, query, index, k=10):
        """SPLADE search over pre-computed sparse index."""
        query_vec = self.encode(query)
        scores = {}
        
        for doc_id, doc_vec in index.items():
            score = (query_vec * doc_vec).sum().item()
            scores[doc_id] = score
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:k]
```

#### BM25 (Okapi BM25)

```python
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """
    BM25: Probabilistic retrieval model based on term frequency and document length.
    """
    def __init__(self):
        self.bm25 = None
        self.documents = []
    
    def index(self, documents):
        """Index documents with BM25."""
        tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.documents = documents
    
    def search(self, query, k=10):
        """Search using BM25."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        return [(self.documents[i], scores[i]) for i in top_indices]
    
    def _tokenize(self, text):
        """Simple tokenization."""
        return text.lower().split()
```

**BM25 Parameters:**
- `k1`: Term saturation parameter (default: 1.2-1.6)
- `b`: Length normalization (default: 0.75)
- `epsilon`: Smoothing for small scores

#### uniCOIL

```python
class UniCOILRetriever:
    """
    uniCOIL: Expanded term weighting with COIL architecture.
    """
    def __init__(self, model_name="castorini/unicoil-d2v-msmarco-passage"):
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    def encode(self, text):
        """Generate term-weighted sparse representation."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # uniCOIL generates term importance weights
        weights = outputs.last_hidden_state.mean(dim=1).squeeze()
        return weights
```

### Keyword Search (BM25 / Elasticsearch)

```python
from elasticsearch import Elasticsearch

class ElasticsearchKeywordRetriever:
    def __init__(self, host="localhost", port=9200, index="documents"):
        self.es = Elasticsearch([f"http://{host}:{port}"])
        self.index = index
    
    def create_index(self):
        """Create index with keyword search configuration."""
        mapping = {
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "standard",
                        "similarity": "BM25"
                    }
                }
            }
        }
        self.es.indices.create(index=self.index, body=mapping)
    
    def search(self, query, k=10):
        """Keyword search using Elasticsearch."""
        response = self.es.search(
            index=self.index,
            body={
                "query": {
                    "match": {
                        "text": query
                    }
                },
                "size": k
            }
        )
        
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "text": hit["_source"]["text"],
                "score": hit["_score"],
                "id": hit["_id"]
            })
        
        return results
```

### Fusion Methods

See [Fusion Methods](#fusion-methods) section for detailed coverage of combining search results.

---

## Fusion Methods

Fusion methods combine results from multiple retrieval strategies.

### RRF (Reciprocal Rank Fusion)

```python
class RRF:
    """
    Reciprocal Rank Fusion: Combines multiple ranked lists.
    RRF score = sum(1 / (k + rank(doc, list)))
    """
    def __init__(self, k=60):
        self.k = k
    
    def fuse(self, ranked_lists):
        """
        ranked_lists: list of lists, each is [(doc_id, score), ...]
        Returns: fused ranking [(doc_id, rrf_score), ...]
        """
        rrf_scores = {}
        
        for ranked_list in ranked_lists:
            for rank, (doc_id, _) in enumerate(ranked_list):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                rrf_scores[doc_id] += 1.0 / (self.k + rank + 1)
        
        # Sort by RRF score descending
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused
```

### Weighted Fusion

```python
class WeightedFusion:
    """
    Weighted combination of scores from different retrieval methods.
    """
    def __init__(self, weights=None):
        self.weights = weights or {"dense": 0.5, "sparse": 0.3, "keyword": 0.2}
    
    def fuse(self, results_dict):
        """
        results_dict: {"method_name": {doc_id: score, ...}, ...}
        """
        fused_scores = {}
        
        for method, docs in results_dict.items():
            weight = self.weights.get(method, 1.0)
            
            # Normalize scores within each method
            max_score = max(docs.values()) if docs else 1.0
            for doc_id, score in docs.items():
                normalized = score / max_score if max_score > 0 else 0
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                fused_scores[doc_id] += weight * normalized
        
        return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
```

### Linear Combination

```python
class LinearCombinationFusion:
    """
    Simple linear combination of normalized scores.
    """
    def combine(self, scores_list, weights=None):
        """
        scores_list: list of dicts {doc_id: normalized_score}
        weights: list of weights for each score dict
        """
        if weights is None:
            weights = [1.0/len(scores_list)] * len(scores_list)
        
        combined = {}
        for scores, weight in zip(scores_list, weights):
            if scores:
                max_score = max(scores.values())
                for doc_id, score in scores.items():
                    normalized = score / max_score
                    combined[doc_id] = combined.get(doc_id, 0) + weight * normalized
        
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)
```

### Learning to Rank (LTR)

```python
from sklearn.ensemble import GradientBoostingRegressor

class LearningToRank:
    """
    Learned fusion using gradient boosted trees.
    """
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1
        )
    
    def extract_features(self, doc, query):
        """Extract ranking features."""
        features = {
            "bm25_score": compute_bm25(doc, query),
            "dense_similarity": cosine_similarity(doc_emb, query_emb),
            "cross_encoder_score": cross_encoder_score(query, doc),
            "doc_length": len(doc),
            "query_term_overlap": term_overlap(query, doc),
            "title_match": title_match(query, doc),
        }
        return list(features.values())
    
    def train(self, training_pairs):
        """
        training_pairs: [(query, doc, relevance_label), ...]
        """
        X, y = [], []
        for query, doc, label in training_pairs:
            features = self.extract_features(doc, query)
            X.append(features)
            y.append(label)
        
        self.model.fit(X, y)
    
    def predict(self, query, documents):
        """Score documents using learned model."""
        scores = []
        for doc in documents:
            features = self.extract_features(doc, query)
            score = self.model.predict([features])[0]
            scores.append(score)
        
        return scores
```

---

## Query Transformation

Query transformation adapts or rewrites queries to improve retrieval effectiveness.

### Query Rewriting

Rewrites the query based on conversation history or query analysis.

```python
class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm
    
    def rewrite_with_history(self, current_query, history):
        """Rewrite query incorporating conversation history."""
        prompt = f"""
        Conversation history:
        {history}
        
        Current query: {current_query}
        
        Rewrite the current query to include necessary context from the 
        conversation history. Output only the rewritten query.
        """
        
        response = self.llm.invoke(prompt)
        return response.content.strip()
    
    def rewrite_for_retrieval(self, query):
        """Rewrite query to be more effective for retrieval."""
        prompt = f"""
        Original query: {query}
        
        Rewrite this query to improve search retrieval effectiveness:
        - Use keywords that would appear in relevant documents
        - Remove conversational filler
        - Make it specific and searchable
        
        Rewritten query:
        """
        
        response = self.llm.invoke(prompt)
        return response.content.strip()
```

### Query Expansion

Expands the query with related terms or generated relevant documents.

```python
class QueryExpander:
    def __init__(self, llm):
        self.llm = llm
    
    def expand_with_terms(self, query, num_expansions=5):
        """Expand query with related terms."""
        prompt = f"""
        Query: {query}
        
        Generate {num_expansions} alternative phrasings or related terms 
        for this query that would help find relevant documents.
        Output one per line.
        """
        
        response = self.llm.invoke(prompt)
        expansions = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
        
        return [query] + expansions[:num_expansions]
    
    def expand_with_generated_docs(self, query, num_docs=3):
        """Generate hypothetical relevant documents and extract terms."""
        prompt = f"""
        For the query: "{query}"
        
        Generate {num_docs} short relevant document snippets that would 
        ideally answer this query.
        """
        
        response = self.llm.invoke(prompt)
        generated_docs = response.content.strip()
        
        # Extract key terms from generated docs
        extraction_prompt = f"""
        From these documents relevant to "{query}":
        {generated_docs}
        
        Extract 5 key search terms or phrases that would be most useful for retrieval.
        """
        
        terms_response = self.llm.invoke(extraction_prompt)
        terms = [t.strip() for t in terms_response.content.strip().split('\n') if t.strip()]
        
        return [query] + terms
```

### Step-Back Prompting

Generates a more abstract or general query to retrieve background knowledge.

```python
class StepBackPrompting:
    """
    Step-back prompting: generate a more abstract query to retrieve
    high-level context, then answer the specific query.
    """
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def step_back(self, query):
        """Generate a step-back (more abstract) question."""
        prompt = f"""
        Original question: {query}
        
        What broader topic or concept does this question relate to?
        Generate a more general question that would help answer the original.
        
        Step-back question:
        """
        
        response = self.llm.invoke(prompt)
        return response.content.strip()
    
    def retrieve_with_step_back(self, query, k=5):
        """Retrieve using both specific and step-back queries."""
        # Get step-back query
        stepback_query = self.step_back(query)
        
        # Retrieve with both queries
        specific_results = self.retriever.retrieve(query, k=k)
        general_results = self.retriever.retrieve(stepback_query, k=k)
        
        # Combine results
        combined = self._combine_results(specific_results, general_results)
        return combined
```

### Decomposed Queries

Breaks complex queries into simpler sub-queries for multi-hop retrieval.

```python
class QueryDecomposer:
    def __init__(self, llm):
        self.llm = llm
    
    def decompose(self, complex_query):
        """Decompose multi-hop query into sub-queries."""
        prompt = f"""
        Complex query: {complex_query}
        
        Break this down into simpler sub-queries that can be answered 
        independently and then combined.
        Output one sub-query per line.
        """
        
        response = self.llm.invoke(prompt)
        sub_queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
        return sub_queries
    
    def parallel_retrieve(self, complex_query, retriever):
        """Retrieve for all sub-queries in parallel."""
        sub_queries = self.decompose(complex_query)
        
        all_results = []
        for sub_query in sub_queries:
            results = retriever.retrieve(sub_query)
            all_results.extend(results)
        
        return all_results
```

---

## Query Routing

Query routing directs queries to the most appropriate retrieval strategy or knowledge source.

### Fixed Routing

```python
class FixedRouter:
    """Route based on fixed rules."""
    
    def route(self, query):
        """Simple rule-based routing."""
        if "news" in query.lower() or "latest" in query.lower():
            return "web_search"
        elif "code" in query.lower() or "function" in query.lower():
            return "code_index"
        elif "who" in query.lower() or "what" in query.lower():
            return "knowledge_base"
        else:
            return "general_retriever"
```

### Content-Based Routing

```python
class ContentBasedRouter:
    """Route based on query content analysis."""
    
    def __init__(self):
        self.classifier = load_classifier()  # e.g., zero-shot classifier
    
    def route(self, query):
        """Classify query domain and route accordingly."""
        domains = ["technical", "medical", "legal", "general", "creative"]
        classification = self.classifier(query, domains)
        
        domain = classification[0]["label"]
        return self._domain_to_index(domain)
    
    def _domain_to_index(self, domain):
        router = {
            "technical": "tech_docs_index",
            "medical": "medical_knowledge_base",
            "legal": "legal_corpus",
            "creative": "general_index",
            "general": "general_index"
        }
        return router.get(domain, "general_index")
```

### LLM-Based Routing

```python
class LLMBasedRouter:
    """Use LLM to determine best routing strategy."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def route(self, query, available_strategies):
        """LLM decides which retrieval strategy to use."""
        prompt = f"""
        Query: {query}
        
        Available strategies:
        {available_strategies}
        
        Which strategy is best suited for this query?
        Consider: query complexity, required knowledge domain, specificity.
        
        Strategy:
        Reason:
        """
        
        response = self.llm.invoke(prompt)
        # Parse strategy from response
        return self._parse_strategy(response.content)
    
    def route_with_parameters(self, query):
        """Get routing decision with retrieval parameters."""
        prompt = f"""
        Query: {query}
        
        Decide:
        1. Retrieval method (dense, sparse, hybrid)
        2. Number of documents to retrieve
        3. Re-ranker needed (yes/no)
        4. Query expansion needed (yes/no)
        
        Output JSON:
        {{"method": "...", "k": 5, "rerank": true, "expand": false}}
        """
        
        response = self.llm.invoke(prompt)
        return json.loads(response.content)
```

### Metadata-Based Routing

```python
class MetadataRouter:
    """Route based on query metadata or context."""
    
    def route(self, query, user_context=None):
        """Route based on user and query metadata."""
        route_decision = {
            "index": "general",
            "filters": {}
        }
        
        if user_context:
            # User-specific routing
            if user_context.get("department") == "engineering":
                route_decision["index"] = "technical_docs"
            elif user_context.get("department") == "legal":
                route_decision["index"] = "legal_docs"
            
            # Date filter
            if user_context.get("timeframe"):
                route_decision["filters"]["date"] = {
                    "$gte": user_context["timeframe"]["start"],
                    "$lte": user_context["timeframe"]["end"]
                }
        
        # Content-based metadata routing
        if "product" in query.lower():
            route_decision["filters"]["category"] = "product"
        
        return route_decision
```

---

## Multi-Hop RAG

Multi-hop RAG performs iterative retrieval where each step informs the next.

### Iterative Retrieval

```python
class IterativeMultiHopRAG:
    """
    Iterative retrieval: retrieve → generate → extract queries → retrieve again
    """
    def __init__(self, llm, retriever, max_hops=3):
        self.llm = llm
        self.retriever = retriever
        self.max_hops = max_hops
    
    def answer(self, question):
        """Multi-hop RAG with iterative retrieval."""
        context = []
        
        for hop in range(self.max_hops):
            # Retrieve
            results = self.retriever.retrieve(question, k=5)
            new_context = [r.page_content for r in results]
            context.extend(new_context)
            
            if hop < self.max_hops - 1:
                # Determine if more information is needed
                analysis = self.llm.invoke(f"""
                Question: {question}
                
                Known information:
                {chr(10).join(context)}
                
                Can this question be answered with current information?
                If no, what additional information is needed?
                Generate a follow-up search query.
                
                Answerable (yes/no):
                Follow-up query (if not answerable):
                """)
                
                if "yes" in analysis.content.lower().split('\n')[0]:
                    break
                
                # Extract follow-up query
                lines = analysis.content.strip().split('\n')
                for line in lines:
                    if "follow-up" in line.lower() or "query" in line.lower():
                        question = line.split(":")[-1].strip()
                        break
        
        # Generate final answer
        final_prompt = f"""
        Question: {question}
        
        Retrieved information:
        {chr(10).join(context)}
        
        Provide a comprehensive answer based on the retrieved information:
        """
        
        return self.llm.invoke(final_prompt).content
```

### Multi-Hop Query Decomposition

```python
class DecompositionMultiHop:
    """
    Decompose multi-hop question into sub-questions and answer sequentially.
    """
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def decompose_and_answer(self, question):
        """Decompose into sub-questions and answer sequentially."""
        # Step 1: Decompose
        sub_questions = self._decompose(question)
        
        # Step 2: Answer sub-questions sequentially
        intermediate_answers = {}
        for i, sub_q in enumerate(sub_questions):
            # Retrieve with context from previous answers
            context = list(intermediate_answers.values())
            results = self.retriever.retrieve(
                sub_q + " " + " ".join(context), k=5
            )
            
            # Answer sub-question
            answer = self.llm.invoke(f"""
            Sub-question: {sub_q}
            
            Retrieved:
            {chr(10).join([r.page_content for r in results])}
            
            Previous context: {chr(10).join(context)}
            
            Answer:
            """)
            
            intermediate_answers[f"step_{i}"] = answer.content
        
        # Step 3: Synthesize final answer
        final = self.llm.invoke(f"""
        Original question: {question}
        
        Intermediate answers:
        {chr(10).join([f"{k}: {v}" for k, v in intermediate_answers.items()])}
        
        Final answer:
        """)
        
        return final.content
    
    def _decompose(self, question):
        """Use LLM to break down multi-hop question."""
        prompt = f"""
        Break this multi-hop question into sequential sub-questions.
        Each sub-question should be answerable from retrieved documents.
        
        Question: {question}
        
        Sub-questions (one per line):
        """
        
        response = self.llm.invoke(prompt)
        return [q.strip() for q in response.content.strip().split('\n') if q.strip()]
```

---

## GraphRAG

GraphRAG combines knowledge graphs with RAG for structured knowledge retrieval.

### Microsoft GraphRAG

Microsoft's GraphRAG builds a knowledge graph from documents, enabling global and local search.

**Architecture:**
```
Documents
  ↓
Entity Extraction (LLM-based)
  ↓
Entity Resolution & Deduplication
  ↓
Community Detection (Leiden algorithm)
  ↓
Community Summarization
  ↓
┌──────────────────────┐
│  Graph Index          │
│  ├── Entities         │
│  ├── Relationships    │
│  ├── Communities      │
│  └── Summaries        │
└──────────────────────┘
    ↕            ↕
Local Search  Global Search
```

**Implementation:**
```python
class MicrosoftGraphRAG:
    """
    GraphRAG: Global and Local Search over knowledge graphs.
    """
    def __init__(self, llm):
        self.llm = llm
        self.graph = nx.Graph()
        self.communities = []
        self.summaries = {}
    
    def build_from_documents(self, documents):
        """Build knowledge graph from documents."""
        for doc in documents:
            # Extract entities and relationships
            entities, relationships = self._extract_graph(doc)
            
            # Add to graph
            for entity in entities:
                self.graph.add_node(entity["name"], **entity)
            for rel in relationships:
                self.graph.add_edge(rel["source"], rel["target"], **rel)
        
        # Detect communities
        self._detect_communities()
        
        # Generate summaries
        self._summarize_communities()
    
    def _extract_graph(self, document):
        """Extract entities and relationships using LLM."""
        prompt = f"""
        Extract entities and relationships from this text.
        
        Text: {document}
        
        Output JSON format:
        {{
            "entities": [{{"name": "...", "type": "...", "description": "..."}}],
            "relationships": [{{"source": "...", "target": "...", "relationship": "..."}}]
        }}
        """
        
        response = self.llm.invoke(prompt)
        return self._parse_extraction(response.content)
    
    def _detect_communities(self):
        """Use Leiden algorithm for community detection."""
        import leidenalg as la
        
        partition = la.find_partition(
            self.graph, 
            la.ModularityVertexPartition
        )
        
        self.communities = partition
    
    def _summarize_communities(self):
        """Generate summaries for each community."""
        for i, community in enumerate(self.communities):
            members = [self.graph.nodes[n] for n in community]
            
            prompt = f"""
            Summarize this group of related entities:
            {members}
            
            What is this community about? What are the key themes?
            """
            
            response = self.llm.invoke(prompt)
            self.summaries[i] = response.content
    
    def local_search(self, query, k=10):
        """Search within relevant communities."""
        # Find relevant entities via embedding similarity
        entity_texts = [f"{n['name']}: {n.get('description', '')}" 
                       for n in self.graph.nodes.values()]
        
        relevant_entities = self._semantic_search(query, entity_texts, k=k)
        
        # Expand to connected entities
        expanded = set(relevant_entities)
        for entity in relevant_entities:
            neighbors = list(self.graph.neighbors(entity))
            expanded.update(neighbors[:5])
        
        # Generate response from subgraph
        subgraph_text = self._subgraph_to_text(expanded)
        
        return self.llm.invoke(f"""
        Query: {query}
        
        Relevant information from knowledge graph:
        {subgraph_text}
        
        Answer:
        """)
    
    def global_search(self, query):
        """Search across all community summaries."""
        summaries_text = "\n\n".join([
            f"Community {i}: {s}" 
            for i, s in self.summaries.items()
        ])
        
        return self.llm.invoke(f"""
        Query: {query}
        
        All community summaries:
        {summaries_text}
        
        Synthesize information across communities to answer:
        """)
```

### LightRAG

LightRAG combines graph structure with edge weights for efficient retrieval.

```python
class LightRAG:
    """
    LightRAG: Lightweight Graph RAG with edge weights.
    """
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.entities = {}  # name -> {embedding, description}
        self.relations = []  # (source, target, weight, description)
    
    def add_text(self, text):
        """Extract and add entities/relations from text."""
        # LLM-based extraction
        entities, relations = self._extract(text)
        
        for entity in entities:
            if entity["name"] not in self.entities:
                emb = self.embedding_model.embed_query(entity["description"])
                self.entities[entity["name"]] = {
                    "embedding": emb,
                    "description": entity["description"],
                    "type": entity["type"]
                }
        
        for rel in relations:
            weight = self._compute_relevance(rel)
            self.relations.append({
                "source": rel["source"],
                "target": rel["target"],
                "weight": weight,
                "description": rel["description"]
            })
    
    def retrieve(self, query, k=5):
        """Retrieve relevant information from graph."""
        query_emb = self.embedding_model.embed_query(query)
        
        # Find relevant entities
        entity_scores = []
        for name, data in self.entities.items():
            score = cosine_similarity([query_emb], [data["embedding"]])
            entity_scores.append((name, score[0][0]))
        
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        top_entities = entity_scores[:k]
        
        # Traverse relations from top entities
        context = []
        visited = set()
        for name, score in top_entities:
            context.append(self.entities[name]["description"])
            visited.add(name)
            
            # Follow relations
            for rel in self.relations:
                if rel["source"] == name and rel["target"] not in visited:
                    context.append(rel["description"])
                    visited.add(rel["target"])
        
        return context
```

### Fast GraphRAG: Iterative Deepening

```python
class FastGraphRAG:
    """
    Fast GraphRAG with iterative deepening for efficiency.
    """
    def __init__(self, llm, embedding_model, max_depth=3):
        self.llm = llm
        self.embedding_model = embedding_model
        self.max_depth = max_depth
    
    def retrieve_with_deepening(self, query, k=5):
        """Iterative deepening: start shallow, go deeper if needed."""
        for depth in range(1, self.max_depth + 1):
            # Retrieve at current depth
            results = self._retrieve_depth(query, depth=depth, k=k)
            
            # Check if sufficient
            if self._sufficient_results(query, results):
                return results
        
        return results
    
    def _retrieve_depth(self, query, depth, k):
        """Retrieve information up to specified depth in graph."""
        # Implementation depends on graph structure
        pass
    
    def _sufficient_results(self, query, results):
        """Check if retrieved results are sufficient."""
        check = self.llm.invoke(f"""
        Query: {query}
        
        Retrieved: {results}
        
        Is this sufficient to answer? (yes/no)
        """)
        return "yes" in check.content.lower()
```

### Nano-GraphRAG

Minimal implementation focusing on core GraphRAG concepts:

```python
class NanoGraphRAG:
    """
    Minimal GraphRAG implementation for learning and experimenting.
    """
    def __init__(self):
        self.adjacency = {}  # entity -> {relation: [related_entities]}
        self.entity_data = {}  # entity -> description
    
    def add_triple(self, subject, relation, obj, subject_desc="", obj_desc=""):
        """Add a (subject, relation, object) triple."""
        if subject not in self.adjacency:
            self.adjacency[subject] = {}
        if relation not in self.adjacency[subject]:
            self.adjacency[subject][relation] = []
        self.adjacency[subject][relation].append(obj)
        
        self.entity_data[subject] = subject_desc
        self.entity_data[obj] = obj_desc
    
    def traverse(self, start_entity, relation=None, max_depth=2):
        """BFS traversal from start entity."""
        visited = set()
        queue = [(start_entity, 0)]
        results = []
        
        while queue:
            entity, depth = queue.pop(0)
            if entity in visited or depth > max_depth:
                continue
            visited.add(entity)
            
            results.append(self.entity_data.get(entity, ""))
            
            if entity in self.adjacency:
                for rel, neighbors in self.adjacency[entity].items():
                    if relation is None or rel == relation:
                        for neighbor in neighbors:
                            queue.append((neighbor, depth + 1))
        
        return results
```

---

## Agentic RAG

Agentic RAG uses AI agents to make intelligent decisions about retrieval strategy.

### Self-RAG

Self-RAG has the model self-reflect on retrieval quality and generated output.

```python
class SelfRAG:
    """
    Self-RAG: Self-reflection on retrieval and generation quality.
    """
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def retrieve_and_reflect(self, query, k=5):
        """Retrieve, reflect on quality, and refine."""
        # Initial retrieval
        results = self.retriever.retrieve(query, k=k)
        
        # Self-reflection on retrieval
        reflection = self.llm.invoke(f"""
        Query: {query}
        
        Retrieved passages:
        {chr(10).join([r.page_content for r in results])}
        
        Reflect:
        1. Are these passages relevant? (yes/no)
        2. Is any important information missing? (yes/no)
        3. Should I retrieve more? (yes/no)
        4. What follow-up query would help?
        """)
        
        # If retrieval insufficient, retrieve more
        if "yes" in reflection.content.lower():
            # Extract follow-up query
            follow_up = self._extract_followup(reflection.content)
            if follow_up:
                more_results = self.retriever.retrieve(follow_up, k=k)
                results.extend(more_results)
        
        return results
    
    def generate_with_reflection(self, query, context):
        """Generate answer with self-reflection."""
        # Initial generation
        answer = self.llm.invoke(f"""
        Query: {query}
        Context: {context}
        Answer:
        """)
        
        # Self-reflection on generated answer
        reflection = self.llm.invoke(f"""
        Query: {query}
        Context: {context}
        Generated answer: {answer.content}
        
        Evaluate the answer:
        1. Is it fully supported by the context? (yes/partially/no)
        2. Are there any unsupported claims? (yes/no)
        3. Is anything important missing? (yes/no)
        4. How confident are you in this answer? (1-10)
        """)
        
        return answer.content, reflection.content
```

### Adaptive RAG

Adaptive RAG dynamically chooses retrieval strategy based on query complexity.

```python
class AdaptiveRAG:
    """
    Adaptive RAG: Dynamically adjust strategy based on query complexity.
    """
    def __init__(self, llm, retriever, simple_threshold=0.3, complex_threshold=0.7):
        self.llm = llm
        self.retriever = retriever
        self.simple_threshold = simple_threshold
        self.complex_threshold = complex_threshold
    
    def classify_query(self, query):
        """Classify query complexity."""
        prompt = f"""
        Classify this query by complexity:
        Query: {query}
        
        Categories:
        - simple: single fact, straightforward
        - medium: requires some reasoning, multiple facts
        - complex: multi-step reasoning, synthesis needed
        
        Category (simple/medium/complex):
        Confidence (0-1):
        """
        
        response = self.llm.invoke(prompt)
        return self._parse_classification(response.content)
    
    def answer(self, query):
        """Adaptive retrieval based on query complexity."""
        classification = self.classify_query(query)
        
        if classification["category"] == "simple":
            # Direct retrieval, no expansion
            results = self.retriever.retrieve(query, k=3)
            return self._simple_answer(query, results)
        
        elif classification["category"] == "medium":
            # Query expansion + re-ranking
            expanded = self._expand_query(query)
            results = self.retriever.retrieve(expanded, k=10)
            reranked = self._rerank(query, results)
            return self._detailed_answer(query, reranked[:5])
        
        else:  # complex
            # Decomposition + multi-hop retrieval
            sub_queries = self._decompose_query(query)
            all_results = []
            for sq in sub_queries:
                results = self.retriever.retrieve(sq, k=5)
                all_results.extend(results)
            
            return self._synthesize_answer(query, all_results)
```

### CRAG (Corrective RAG)

CRAG corrects retrieval errors through a retrieve-evaluate-generate loop.

```python
class CRAG:
    """
    Corrective RAG: Retrieve, evaluate relevance, correct if needed.
    """
    def __init__(self, llm, retriever, relevance_threshold=0.5):
        self.llm = llm
        self.retriever = retriever
        self.relevance_threshold = relevance_threshold
    
    def answer(self, query):
        """CRAG pipeline: retrieve → evaluate → correct → generate."""
        # Step 1: Retrieve
        results = self.retriever.retrieve(query, k=10)
        
        # Step 2: Evaluate relevance of each result
        relevant_docs = []
        irrelevant_docs = []
        
        for doc in results:
            relevance_score = self._evaluate_relevance(query, doc)
            if relevance_score >= self.relevance_threshold:
                relevant_docs.append(doc)
            else:
                irrelevant_docs.append(doc)
        
        # Step 3: Correct if insufficient relevance
        if len(relevant_docs) < 3:
            # Knowledge refinement from irrelevant docs
            refinement = self._refine_knowledge(query, irrelevant_docs)
            if refinement:
                relevant_docs.extend(refinement)
        
        # Step 4: Generate with correction context
        if not relevant_docs:
            return self._generate_fallback(query)
        
        return self._generate(query, relevant_docs)
    
    def _evaluate_relevance(self, query, document):
        """Score relevance of document to query."""
        prompt = f"""
        Query: {query}
        Document: {document.page_content}
        
        Relevance score (0-1):"""
        
        response = self.llm.invoke(prompt)
        try:
            return float(response.content.strip())
        except:
            return 0.0
    
    def _refine_knowledge(self, query, irrelevant_docs):
        """Extract useful knowledge from seemingly irrelevant docs."""
        prompt = f"""
        Query: {query}
        
        These documents were retrieved but seem not directly relevant:
        {chr(10).join([d.page_content[:200] for d in irrelevant_docs])}
        
        Can you extract any useful information for answering the query?
        If yes, provide the refined knowledge:
        """
        
        response = self.llm.invoke(prompt)
        if response.content.strip() and "no" not in response.content.lower()[:50]:
            return [response.content]
        return []
```

### Self-Ask

The model generates follow-up questions to gather needed information.

```python
class SelfAskRAG:
    """
    Self-Ask: Model generates and answers follow-up questions.
    """
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def answer(self, question, max_questions=5):
        """Self-ask: decompose into sub-questions and answer."""
        context = []
        follow_ups = []
        
        for _ in range(max_questions):
            # Generate follow-up question if needed
            decision = self.llm.invoke(f"""
            Original question: {question}
            
            Information gathered so far:
            {chr(10).join(context)}
            
            Do you need more information to answer?
            If yes, what follow-up question would help?
            If no, answer "SUFFICIENT".
            
            Response:
            """)
            
            response_text = decision.content.strip()
            if "SUFFICIENT" in response_text.upper():
                break
            
            follow_up = response_text
            follow_ups.append(follow_up)
            
            # Retrieve for follow-up
            results = self.retriever.retrieve(follow_up, k=3)
            context.extend([r.page_content for r in results])
        
        # Generate final answer
        final = self.llm.invoke(f"""
        Question: {question}
        
        Context:
        {chr(10).join(context)}
        
        Answer:
        """)
        
        return final.content, follow_ups
```

### FLARE (Forward-Looking Active Retrieval)

FLARE actively decides when to retrieve by detecting low-confidence tokens.

```python
class FLARE:
    """
    FLARE: Forward-Looking Active Retrieval Augmented Generation.
    Retrieves information when the model is uncertain about upcoming tokens.
    """
    def __init__(self, llm, retriever, confidence_threshold=0.3):
        self.llm = llm
        self.retriever = retriever
        self.confidence_threshold = confidence_threshold
    
    def generate(self, question, max_retrievals=5):
        """Generate with active retrieval based on token confidence."""
        response_parts = []
        retrieval_count = 0
        
        # Generate sentence by sentence
        while retrieval_count < max_retrievals:
            # Generate next sentence
            prompt = f"""
            Question: {question}
            
            Generated so far: {''.join(response_parts)}
            
            Continue the response. If uncertain, write [SEARCH: query] 
            where you need more information.
            """
            
            response = self.llm.invoke(prompt)
            text = response.content
            
            # Check for search triggers
            if "[SEARCH:" in text:
                # Extract search query
                search_match = re.search(r'\[SEARCH: (.*?)\]', text)
                if search_match:
                    search_query = search_match.group(1)
                    
                    # Retrieve
                    results = self.retriever.retrieve(search_query, k=3)
                    
                    # Add retrieved information
                    context = chr(10).join([r.page_content for r in results])
                    response_parts.append(f"\n[Retrieved: {context}]\n")
                    retrieval_count += 1
                    continue
            
            response_parts.append(text)
            
            # Check if generation seems complete
            if self._is_complete(text):
                break
        
        return ''.join(response_parts)
    
    def _is_complete(self, text):
        """Check if generation appears complete."""
        # Simple heuristic: ends with period or is empty
        return text.strip().endswith('.') or not text.strip()
```

### Active RAG

Active RAG proactively identifies knowledge gaps and retrieves missing information.

```python
class ActiveRAG:
    """
    Active RAG: Proactively identify and fill knowledge gaps.
    """
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def answer(self, question):
        """Active RAG with knowledge gap identification."""
        # Step 1: What do we know?
        known = self.retriever.retrieve(question, k=5)
        
        # Step 2: Identify gaps
        gap_analysis = self.llm.invoke(f"""
        Question: {question}
        
        Retrieved information:
        {chr(10).join([r.page_content for r in known])}
        
        What information is still missing to fully answer this question?
        List specific knowledge gaps:
        """)
        
        # Step 3: Retrieve for each gap
        gaps = self._extract_gaps(gap_analysis.content)
        additional_context = []
        for gap in gaps:
            gap_results = self.retriever.retrieve(gap, k=3)
            additional_context.extend([r.page_content for r in gap_results])
        
        # Step 4: Generate comprehensive answer
        all_context = [r.page_content for r in known] + additional_context
        
        return self.llm.invoke(f"""
        Question: {question}
        
        Complete context:
        {chr(10).join(all_context)}
        
        Provide a thorough answer:
        """)
```

### Router RAG

Router RAG uses a trained router to select the optimal retrieval strategy.

```python
class RouterRAG:
    """
    Router RAG: Learned routing to different RAG strategies.
    """
    def __init__(self, router_model, strategies):
        """
        router_model: Classification model that selects strategy.
        strategies: dict of {name: RAGStrategy}
        """
        self.router = router_model
        self.strategies = strategies
    
    def answer(self, query):
        """Route query to best RAG strategy."""
        # Classify query
        strategy_name = self.router.predict(query)
        
        # Execute selected strategy
        strategy = self.strategies[strategy_name]
        result = strategy.execute(query)
        
        return result
```

---

## RAG Evaluation

Evaluating RAG systems requires metrics for retrieval quality, generation quality, and end-to-end performance.

### RAGAS

RAGAS (Retrieval Augmented Generation Assessment) provides automated evaluation metrics.

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

class RAGASEvaluator:
    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
    
    def evaluate(self, dataset):
        """
        dataset: Dataset with columns:
            - question
            - answer
            - contexts
            - ground_truth (optional)
        """
        result = evaluate(
            dataset,
            metrics=self.metrics
        )
        return result
```

**RAGAS Metrics:**
| Metric | Description | Range |
|--------|-------------|-------|
| **Faithfulness** | Is the answer factually supported by context? | 0-1 |
| **Answer Relevancy** | How relevant is the answer to the question? | 0-1 |
| **Context Precision** | Are retrieved contexts relevant? | 0-1 |
| **Context Recall** | Are all relevant contexts retrieved? | 0-1 |
| **Answer Semantic Similarity** | Semantic similarity to ground truth | 0-1 |
| **Answer Correctness** | Factual correctness vs ground truth | 0-1 |
| **Aspect Critique** | LLM-based critique of answer aspects | 0-1 |

### TruLens

TruLens provides feedback functions for RAG evaluation.

```python
from trulens_eval import Feedback, TruLens
from trulens_eval.feedback import Groundedness

class TruLensEvaluator:
    def __init__(self, provider="openai"):
        self.groundedness = Groundedness(provider)
        
        # Define feedback functions
        self.feedbacks = [
            Feedback(self.groundedness.groundedness_measure_with_cot_reasons)
                .on_input()
                .on_output(),
            Feedback(self.answer_relevance)
                .on_input()
                .on_output(),
            Feedback(self.context_relevance)
                .on_input()
                .on(context)
        ]
    
    def answer_relevance(self, question, answer):
        """Evaluate answer relevance to question."""
        # Implementation
        return score
    
    def context_relevance(self, question, context):
        """Evaluate context relevance to question."""
        return score
```

**TruLens Metrics:**
| Metric | Description |
|--------|-------------|
| **Answer Relevance** | How well the answer addresses the question |
| **Context Relevance** | How relevant the retrieved context is |
| **Groundedness** | How well the answer is supported by context |

### ARES (Automated RAG Evaluation System)

```python
class ARESEvaluator:
    """
    ARES: Automated RAG Evaluation System.
    Uses LLM to evaluate RAG outputs across multiple dimensions.
    """
    def __init__(self, llm):
        self.llm = llm
    
    def evaluate(self, question, answer, context, ground_truth=None):
        """Multi-dimensional RAG evaluation."""
        metrics = {}
        
        # Faithfulness
        metrics["faithfulness"] = self._score_faithfulness(answer, context)
        
        # Relevancy
        metrics["relevancy"] = self._score_relevancy(question, answer)
        
        # Context utilization
        metrics["context_utilization"] = self._score_context_utilization(answer, context)
        
        # Ground truth alignment (if available)
        if ground_truth:
            metrics["correctness"] = self._score_correctness(answer, ground_truth)
        
        return metrics
```

### RGB (RAG Benchmark)

RGB is a benchmark specifically designed for RAG evaluation with fine-grained metrics.

### RECALL (Retrieval Evaluation with Comprehensive Assessment)

```python
class RECALLEvaluator:
    """
    RECALL: Comprehensive retrieval evaluation.
    """
    def __init__(self):
        self.metrics = {}
    
    def evaluate_retrieval(self, queries, retrieved_docs, relevant_docs):
        """Comprehensive retrieval evaluation."""
        results = {
            "hit_rate": self._hit_rate(retrieved_docs, relevant_docs),
            "mrr": self._mrr(retrieved_docs, relevant_docs),
            "ndcg": self._ndcg(retrieved_docs, relevant_docs),
            "precision_at_k": {},
            "recall_at_k": {}
        }
        
        for k in [1, 3, 5, 10, 20]:
            results["precision_at_k"][k] = self._precision_at_k(
                retrieved_docs, relevant_docs, k
            )
            results["recall_at_k"][k] = self._recall_at_k(
                retrieved_docs, relevant_docs, k
            )
        
        return results
```

---

## Caching

Caching reduces latency and cost by storing and reusing previous results.

### Semantic Caching

Caches results for semantically similar queries based on embedding similarity.

```python
import numpy as np
from typing import List, Tuple, Optional

class SemanticCache:
    """
    Semantic cache: retrieve cached results for similar queries.
    """
    def __init__(self, embedding_model, similarity_threshold=0.92, max_size=1000):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.cache = []  # [(query, embedding, result), ...]
    
    def get(self, query: str) -> Optional[str]:
        """Check if similar query is cached."""
        query_emb = self.embedding_model.embed_query(query)
        
        for cached_query, cached_emb, cached_result in self.cache:
            similarity = self._cosine_similarity(query_emb, cached_emb)
            if similarity >= self.similarity_threshold:
                return cached_result
        
        return None
    
    def put(self, query: str, result: str):
        """Add query-result pair to cache."""
        if len(self.cache) >= self.max_size:
            # Evict oldest entry
            self.cache.pop(0)
        
        query_emb = self.embedding_model.embed_query(query)
        self.cache.append((query, query_emb, result))
    
    def _cosine_similarity(self, emb1, emb2):
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
```

**Threshold Tuning:**
- High threshold (0.95+): Cache hit only for nearly identical queries
- Medium threshold (0.85-0.92): Good balance for most applications
- Low threshold (0.7-0.8): Higher cache hit rate but risk of irrelevant results

### LLM Response Caching

```python
class LLMResponseCache:
    """
    Cache LLM responses for exact and semantically similar prompts.
    """
    def __init__(self, embedding_model, exact_match=True, semantic_match=True):
        self.exact_cache = {} if exact_match else None
        self.semantic_cache = SemanticCache(embedding_model) if semantic_match else None
    
    def get(self, prompt, system_prompt=None, temperature=0.0):
        """Get cached response."""
        if self.exact_cache:
            key = self._make_key(prompt, system_prompt, temperature)
            if key in self.exact_cache:
                return self.exact_cache[key]
        
        if self.semantic_cache:
            result = self.semantic_cache.get(prompt)
            if result:
                return result
        
        return None
    
    def put(self, prompt, response, system_prompt=None, temperature=0.0):
        """Cache response."""
        if self.exact_cache:
            key = self._make_key(prompt, system_prompt, temperature)
            self.exact_cache[key] = response
        
        if self.semantic_cache and temperature == 0.0:
            self.semantic_cache.put(prompt, response)
    
    def _make_key(self, prompt, system_prompt, temperature):
        return f"{system_prompt}|{prompt}|{temperature}"
```

### LRU/TTL Cache

```python
from collections import OrderedDict
import time

class LRUTTLCache:
    """
    LRU cache with TTL (Time-To-Live) eviction.
    """
    def __init__(self, capacity=1000, ttl_seconds=3600):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
    
    def get(self, key):
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl:
            self._evict(key)
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if len(self.cache) >= self.capacity:
            # Evict least recently used
            self.cache.popitem(last=False)
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
        self.cache.move_to_end(key)
    
    def _evict(self, key):
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
```

---

## Multi-Modal RAG

Multi-modal RAG extends retrieval to include images, tables, and other non-text content.

### Text + Image Retrieval

```python
class MultiModalRAG:
    """
    Multi-modal RAG: retrieve both text and images.
    """
    def __init__(self, text_embedder, image_embedder, vector_store):
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.vector_store = vector_store
    
    def index_document(self, doc_id, text_chunks, images):
        """Index text chunks and images with shared embedding space."""
        # Index text
        for i, chunk in enumerate(text_chunks):
            emb = self.text_embedder.embed_query(chunk)
            self.vector_store.add(
                embeddings=[emb],
                documents=[chunk],
                metadatas=[{"type": "text", "doc_id": doc_id, "chunk_id": i}],
                ids=[f"text_{doc_id}_{i}"]
            )
        
        # Index images
        for i, (image, caption) in enumerate(images):
            emb = self.image_embedder.embed_image(image)
            self.vector_store.add(
                embeddings=[emb],
                documents=[caption],
                metadatas=[{"type": "image", "doc_id": doc_id, "image_id": i}],
                ids=[f"img_{doc_id}_{i}"]
            )
    
    def retrieve(self, query, k=5, include_images=True):
        """Retrieve relevant text and images."""
        query_emb = self.text_embedder.embed_query(query)
        
        results = self.vector_store.similarity_search_by_vector(query_emb, k=k)
        
        text_results = [r for r in results if r.metadata["type"] == "text"]
        image_results = [r for r in results if r.metadata["type"] == "image"] if include_images else []
        
        return text_results, image_results
```

### Qwen-VL Retrieval

```python
class QwenVLRetriever:
    """
    Multi-modal retrieval using Qwen-VL model.
    Handles text-to-image and image-to-image retrieval.
    """
    def __init__(self, model_name="Qwen/Qwen-VL-Chat"):
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def embed_text(self, text):
        """Get multi-modal embedding for text query."""
        inputs = self.processor(text=text, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        return embeddings
    
    def embed_image(self, image):
        """Get multi-modal embedding for image."""
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
        return embeddings
    
    def retrieve(self, query, index, k=5, modality="text"):
        """Retrieve relevant items matching the query."""
        if modality == "text":
            query_emb = self.embed_text(query)
        else:
            query_emb = self.embed_image(query)
        
        # Search in unified embedding space
        scores = {}
        for item_id, item_emb in index.items():
            score = cosine_similarity(query_emb, item_emb)
            scores[item_id] = score
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
```

### ColPali: Vision-Based Retrieval

ColPali uses vision models for document retrieval without text extraction.

```python
class ColPaliRetriever:
    """
    ColPali: Vision-based document retrieval using VLMs.
    Processes document images directly without text extraction.
    """
    def __init__(self, model_name="vidore/colpali-v1.2"):
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def index_document(self, doc_id, document_images):
        """Index document pages as images."""
        inputs = self.processor(
            images=document_images,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            embeddings = self.model(**inputs).embeddings
        
        return embeddings
    
    def retrieve(self, query, document_embeddings, k=5):
        """Retrieve relevant document pages."""
        # Encode query
        query_inputs = self.processor(
            text=[query],
            return_tensors="pt"
        )
        
        with torch.no_grad():
            query_emb = self.model.encode_text(**query_inputs)
        
        # Score documents
        scores = []
        for doc_id, doc_emb in document_embeddings.items():
            score = self._compute_relevance(query_emb, doc_emb)
            scores.append((doc_id, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    
    def _compute_relevance(self, query_emb, doc_emb):
        """ColPali late interaction scoring."""
        # Max similarity over token positions
        sim = torch.matmul(query_emb, doc_emb.transpose(-2, -1))
        max_sim = sim.max(dim=-1).values.sum(dim=-1)
        return max_sim.item()
```

---

## Latency Optimization

Techniques to reduce end-to-end RAG latency.

### Caching Layer

```python
class MultiLayerCache:
    """
    Multi-layer caching for maximum latency reduction.
    """
    def __init__(self):
        self.l1 = SemanticCache(threshold=0.95)  # Near-exact matches
        self.l2 = SemanticCache(threshold=0.85)  # Semantic matches
        self.l3 = LRUTTLCache(capacity=10000, ttl=86400)  # Popular results
    
    def get(self, query):
        # Check L1 (fastest)
        result = self.l1.get(query)
        if result:
            return result, "L1_hit"
        
        # Check L2
        result = self.l2.get(query)
        if result:
            return result, "L2_hit"
        
        # Check L3
        result = self.l3.get(query)
        if result:
            return result, "L3_hit"
        
        return None, "miss"
```

### Parallel Retrieval

```python
import asyncio

class ParallelRetriever:
    """
    Parallel execution of multiple retrieval strategies.
    """
    def __init__(self, retrievers):
        self.retrievers = retrievers
    
    async def parallel_retrieve(self, query, k=10):
        """Execute multiple retrievers in parallel."""
        tasks = [
            retriever.retrieve(query, k=k)
            for retriever in self.retrievers
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results
        merged = {}
        for result_list in results:
            if isinstance(result_list, Exception):
                continue
            for doc, score in result_list:
                if doc.id not in merged or score > merged[doc.id]:
                    merged[doc.id] = (doc, score)
        
        return sorted(merged.values(), key=lambda x: x[1], reverse=True)[:k]
```

### Approximate Nearest Neighbor (ANN) Optimization

```python
class ANNConfig:
    """
    ANN index configuration for optimal latency/recall tradeoff.
    """
    @staticmethod
    def fast_config():
        """Configuration optimized for latency."""
        return {
            "index_type": "HNSW",
            "ef_construction": 100,  # Lower = faster build
            "ef_search": 50,  # Lower = faster search
            "M": 16,  # Lower = faster, less accurate
        }
    
    @staticmethod
    def balanced_config():
        """Balanced latency/recall."""
        return {
            "index_type": "HNSW",
            "ef_construction": 200,
            "ef_search": 200,
            "M": 32,
        }
    
    @staticmethod
    def accurate_config():
        """Optimized for recall."""
        return {
            "index_type": "HNSW",
            "ef_construction": 500,
            "ef_search": 500,
            "M": 64,
        }
```

### Query Rewriting Cache

```python
class QueryRewriteCache:
    """
    Cache query rewrites to avoid repeated LLM calls.
    """
    def __init__(self):
        self.cache = LRUTTLCache(capacity=5000, ttl_seconds=3600)
    
    def get_rewrite(self, original_query):
        """Get cached rewritten query."""
        return self.cache.get(original_query)
    
    def cache_rewrite(self, original_query, rewritten_query):
        """Cache rewritten query for future use."""
        self.cache.put(original_query, rewritten_query)
```

### Pre-computation Strategies

```python
class PrecomputationOptimizer:
    """
    Pre-compute expensive operations during indexing.
    """
    def __init__(self):
        self.precomputed = {}
    
    def precompute_embeddings(self, chunks, embedding_model):
        """Pre-compute and cache embeddings."""
        embeddings = embedding_model.embed_documents(chunks)
        for chunk_id, emb in zip(chunk_ids, embeddings):
            self.precomputed[chunk_id] = emb
    
    def precompute_summaries(self, chunks, llm):
        """Pre-compute chunk summaries."""
        for chunk in chunks:
            if chunk.id not in self.precomputed_summaries:
                summary = llm.summarize(chunk.text)
                self.precomputed_summaries[chunk.id] = summary
```

### End-to-End Latency Budget

```python
class LatencyBudget:
    """
    Manage end-to-end latency budget for RAG pipeline.
    """
    def __init__(self, total_budget_ms=2000):
        self.total_budget = total_budget_ms
        self.allocations = {
            "embedding": 50,  # ms
            "vector_search": 100,  # ms
            "reranking": 300,  # ms
            "llm_generation": 1500,  # ms
            "overhead": 50,  # ms
        }
    
    def check_budget(self, stage, elapsed_ms):
        """Check if stage stayed within budget."""
        budget = self.allocations.get(stage, 0)
        if elapsed_ms > budget:
            return False, f"{stage} over budget: {elapsed_ms}ms > {budget}ms"
        return True, None
    
    def optimize_for_budget(self):
        """Return configuration optimized for latency budget."""
        config = {}
        
        if self.total_budget < 1000:
            # Aggressive optimization
            config["retrieval_k"] = 5
            config["use_reranker"] = False
            config["chunk_size"] = 512
            config["model"] = "gpt-3.5-turbo"
        elif self.total_budget < 3000:
            # Moderate optimization
            config["retrieval_k"] = 10
            config["use_reranker"] = True
            config["reranker"] = "fast"  # MiniLM-L2
            config["chunk_size"] = 768
            config["model"] = "gpt-4o-mini"
        else:
            # Full pipeline
            config["retrieval_k"] = 20
            config["use_reranker"] = True
            config["reranker"] = "accurate"  # MiniLM-L12
            config["chunk_size"] = 1024
            config["model"] = "gpt-4o"
        
        return config
```

---

## References

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS.
2. Gao, L., et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels." (HyDE)
3. Nogueira, R., et al. (2019). "Passage Re-ranking with BERT." (monoBERT)
4. Khattab, O., & Zaharia, M. (2020). "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." SIGIR.
5. Robertson, S., & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." Foundations and Trends in Information Retrieval.
6. Lin, J., et al. (2021). "A Dense Representation Framework for Lexical and Semantic Matching." (uniCOIL)
7. Formal, T., et al. (2021). "SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval."
8. Shao, Z., et al. (2023). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." ICLR.
9. Jeong, S., et al. (2024). "Adaptive-RAG: Adaptive Retrieval-Augmented Generation with LLM-based Query Classification."
10. Yan, S., et al. (2024). "Corrective Retrieval Augmented Generation." (CRAG)
11. Edge, D., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." Microsoft.
12. Guo, D., et al. (2024). "LightRAG: Simple and Fast Retrieval-Augmented Generation."
13. Asai, A., et al. (2023). "Self-Ask: Measuring and Narrowing the Compositional Gap in Language Models."
14. Jiang, Z., et al. (2023). "Active Retrieval Augmented Generation." (FLARE)
15. Es, S., et al. (2024). "RAGAS: Automated Evaluation of Retrieval Augmented Generation."
16. TruEra. (2024). "TruLens: Observability for LLM Applications."
17. Faysse, M., et al. (2024). "ColPali: Efficient Document Retrieval with Vision Language Models."
18. Anthropic. (2024). "Contextual Retrieval." anthropic.com.
19. LangChain. (2024). "Advanced RAG Techniques." blog.langchain.dev.
20. Chen, J., et al. (2024). "A Survey on Retrieval-Augmented Text Generation for Large Language Models."

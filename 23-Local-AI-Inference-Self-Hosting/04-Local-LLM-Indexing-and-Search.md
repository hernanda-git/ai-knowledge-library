# 04 — Local LLM Indexing and Search

## Overview

Retrieval-Augmented Generation (RAG) is one of the most transformative and practical applications of local AI. By combining a local language model with a searchable index of your documents, you can build systems that answer questions based on your data — without sending anything to the cloud.

This document provides a comprehensive guide to local ML indexing and search: embedding models that run locally, vector databases optimized for local deployment, document ingestion pipelines, and complete RAG implementations. We cover everything from the theoretical foundations to production deployment patterns.

---

## Why Local Indexing Matters

### The Cloud Indexing Problem

Cloud-based indexing services (Pinecone, Weaviate Cloud, Milvus Cloud, OpenAI embeddings API) offer convenience but come with significant downsides:

1. **Data egress**: Your documents are sent to third-party servers for embedding and storage
2. **Recurring costs**: Embedding API calls and vector database hosting fees accumulate
3. **Latency**: Network round-trips for every embedding and retrieval operation
4. **Vendor lock-in**: Embedding models and vector databases are tied to specific providers
5. **Privacy**: Sensitive documents (medical, legal, financial) cannot be sent externally
6. **Size limits**: Cloud services often have document size and rate limits

### The Local Indexing Advantage

Local indexing addresses all of these concerns:

1. **Complete data privacy**: Documents never leave your machine
2. **Zero ongoing costs**: No API fees, no per-vector charges
3. **Sub-millisecond queries**: Embedding and retrieval happen on local hardware
4. **Unlimited document sizes**: Your only limit is disk space
5. **Offline operation**: Index and query without internet connectivity
6. **Full control**: Choose any embedding model, vector database, and chunking strategy

### When to Use Local vs. Cloud Indexing

| Factor | Local Indexing | Cloud Indexing |
|---|---|---|
| Document volume | Up to 10M documents | Unlimited (scalable) |
| Document sensitivity | Medical, legal, financial, classified | Public or low-sensitivity |
| Budget | Zero ongoing cost (one-time hardware) | Pay per vector/query |
| Latency requirement | Real-time (<100ms) | Near-real-time (<500ms) |
| Internet dependency | Can work fully offline | Requires internet |
| Team size | 1–100 users | 100+ users |
| Maintenance effort | Moderate (self-managed) | Low (managed service) |

---

## Embedding Models for Local Use

### What Are Embeddings?

Embeddings are dense vector representations of text — numerical arrays (typically 384–1024 dimensions) that capture semantic meaning. They enable:

- **Semantic search**: Find documents by meaning, not just keywords
- **Clustering**: Group similar documents automatically
- **Classification**: Categorize text by topic or sentiment
- **Recommendation**: Find related content

### Local Embedding Model Landscape (2026)

The MTEB (Massive Text Embedding Benchmark) leaderboard has driven rapid improvement in embedding model quality. As of June 2026, the best local embedding models rival or surpass OpenAI's text-embedding-3-* models.

#### Top Local Embedding Models

| Model | Dimensions | Max Tokens | MTEB Score | Size | Strengths |
|---|---|---|---|---|---|
| BGE-large-en-v1.5 | 1024 | 512 | 64.23 | 1.34GB | General purpose, strong retrieval |
| BGE-base-en-v1.5 | 768 | 512 | 63.42 | 0.44GB | Good balance of quality/speed |
| BGE-small-en-v1.5 | 384 | 512 | 61.27 | 0.13GB | Fast, low resource usage |
| E5-mistral-7b-instruct | 4096 | 4096 | 66.63 | 14GB | Best quality, large |
| E5-base-v2 | 768 | 512 | 63.03 | 0.44GB | Strong all-around |
| E5-small-v2 | 384 | 512 | 60.35 | 0.13GB | Lightweight option |
| Jina-embeddings-v3 | 1024 | 8192 | 64.89 | 1.1GB | Long context support |
| Nomic-embed-text-v1.5 | 768 | 8192 | 64.12 | 0.55GB | Good for long documents |
| Nomic-embed-text-v1 | 768 | 8192 | 62.39 | 0.27GB | Lightweight long-context |
| Instructor-XL | 768 | 512 | 63.33 | 1.3GB | Instruction-tuned |
| GTE-large | 1024 | 512 | 63.29 | 1.1GB | Alibaba's strong model |
| GTE-base | 768 | 512 | 62.25 | 0.33GB | Lightweight option |
| Llama-3-embedding | 4096 | 8192 | 65.21 | 8GB | Meta's latest, strong |
| MXBAI-embed-large-v1 | 1024 | 512 | 63.78 | 0.67GB | Mixed quality model |

#### Choosing an Embedding Model

**For maximum quality (when you have GPU resources):**
- Use **E5-mistral-7b-instruct** (14GB, 4096 dims) for best-in-class retrieval quality
- Use **Llama-3-embedding** (8GB, 4096 dims) for strong general performance

**For the best quality/size trade-off:**
- Use **BGE-large-en-v1.5** (1.34GB, 1024 dims) — the most popular choice
- Use **Jina-embeddings-v3** (1.1GB, 1024 dims) if you need long context (8192 tokens)

**For speed and low resource usage:**
- Use **BGE-base-en-v1.5** (0.44GB, 768 dims) for most applications
- Use **Nomic-embed-text-v1.5** (0.55GB, 768 dims) for long documents

**For CPU-only inference:**
- Use **BGE-small-en-v1.5** (0.13GB, 384 dims) or **E5-small-v2** (0.13GB, 384 dims)

### Running Embedding Models Locally

#### Using Sentence-Transformers

```bash
pip install sentence-transformers torch --index-url https://download.pytorch.org/whl/cu121
```

```python
from sentence_transformers import SentenceTransformer

# Load model (downloads first time)
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Single text
embedding = model.encode("The sky is blue because of Rayleigh scattering")
print(embedding.shape)  # (1024,)

# Batch encoding (fastest for multiple texts)
texts = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing enables computers to understand text."
]
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
print(embeddings.shape)  # (3, 1024)

# With normalization (important for cosine similarity)
embeddings = model.encode(texts, normalize_embeddings=True)
```

#### Using Ollama for Embeddings

```bash
# Pull an embedding model
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
```

```python
import ollama

# Generate embedding
response = ollama.embeddings(
    model='nomic-embed-text',
    prompt='The sky is blue because of Rayleigh scattering'
)
embedding = response['embedding']
print(len(embedding))  # 768 (for nomic-embed-text)

# Batch embeddings
texts = ["Text one", "Text two", "Text three"]
embeddings = [
    ollama.embeddings(model='nomic-embed-text', prompt=t)['embedding']
    for t in texts
]
```

#### Using llama.cpp for Embeddings

```bash
# Get a GGUF embedding model
wget https://huggingface.co/CompendiumLabs/bge-base-en-v1.5-gguf/resolve/main/bge-base-en-v1.5-q4_k_m.gguf

# Generate embeddings
./llama-embedding -m bge-base-en-v1.5-q4_k_m.gguf -p "Text to embed"
```

#### Using HuggingFace Transformers

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
model.eval()

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    embeddings = (token_embeddings * mask).sum(1) / mask.sum(1)
    # Normalize
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings[0].numpy()

embedding = embed_text("Your text here")
```

### Embedding Best Practices

1. **Normalize embeddings**: Always normalize embeddings to unit length for cosine similarity
2. **Batch processing**: Encode texts in batches for maximum throughput (GPU utilization)
3. **Use appropriate max_length**: Match the model's maximum token limit; truncate longer texts
4. **Prefix instructions**: Some models (Instructor, E5) benefit from instruction prefixes
5. **Cache embeddings**: Avoid recomputing embeddings for unchanged documents
6. **Mixed precision**: Use FP16 for 2× speed on modern GPUs with minimal quality loss

```python
# Instruction prefix for Instructor models
text = "Represent the Wikipedia question for retrieving supporting documents: What is machine learning?"
embedding = model.encode(text)

# E5 prefix format
text = "query: What is machine learning?"  # For queries
text = "passage: Machine learning is..."   # For documents
```

---

## Vector Databases for Local Use

### Overview

Vector databases store embeddings and enable fast similarity search. For local deployment, the key requirements are:

- **Zero external dependencies**: Must work without cloud services
- **Persistent storage**: Data survives restarts
- **Good performance**: Sub-100ms search on 1M+ vectors
- **Low resource usage**: Runs on consumer hardware
- **Open source**: No licensing restrictions

### Top Local Vector Databases

#### ChromaDB

**Overview**: The most popular local vector database. Simple, easy to use, and designed specifically for local AI applications.

| Feature | Details |
|---|---|
| License | Apache 2.0 |
| Storage | Local disk (SQLite + Parquet) |
| Search | Cosine similarity, L2, IP |
| Filters | Metadata filtering |
| Index type | HNSW (Hierarchical Navigable Small World) |
| Max vectors | 10M+ (practical) |
| Language | Python native |
| API | Native Python + HTTP client |

**Installation and usage:**

```bash
pip install chromadb
```

```python
import chromadb

# Create client (persistent mode)
client = chromadb.PersistentClient(path="./chroma_data")

# Create collection
collection = client.create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"}  # Distance metric
)

# Add documents
collection.add(
    documents=[
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
        "NLP enables text understanding."
    ],
    metadatas=[
        {"source": "textbook", "chapter": 1},
        {"source": "textbook", "chapter": 2},
        {"source": "article", "topic": "nlp"}
    ],
    ids=["doc1", "doc2", "doc3"]
)

# Query
results = collection.query(
    query_texts=["What is AI?"],
    n_results=2
)
print(results['documents'])  # [['Machine learning...', 'Deep learning...']]

# Query with metadata filter
results = collection.query(
    query_texts=["neural networks"],
    n_results=5,
    where={"source": "textbook"}
)

# Delete
collection.delete(ids=["doc1"])
```

#### LanceDB

**Overview**: High-performance columnar vector database built on Lance columnar format. Excellent for large datasets.

| Feature | Details |
|---|---|
| License | Apache 2.0 |
| Storage | Local disk (Lance format) |
| Search | Cosine similarity, L2, dot product |
| Filters | Metadata filtering, full SQL |
| Index type | IVF-PQ, HNSW |
| Max vectors | 100M+ |
| Language | Python, Rust, JS |
| API | Native Python, REST, gRPC |

**Installation and usage:**

```bash
pip install lancedb
```

```python
import lancedb
import numpy as np

# Create database
db = lancedb.connect("./lancedb_data")

# Create table
table = db.create_table(
    "documents",
    data=[
        {
            "vector": np.random.rand(768).astype(np.float32),
            "text": "Machine learning is a subset of AI.",
            "source": "textbook"
        }
    ],
    mode="overwrite"
)

# Add more data
table.add([
    {"vector": np.random.rand(768).astype(np.float32),
     "text": "Deep learning uses neural networks.",
     "source": "textbook"}
])

# Create index (ANN search)
table.create_index(num_partitions=256, num_sub_vectors=96)

# Search
results = table.search(np.random.rand(768).astype(np.float32)) \
    .limit(5) \
    .where("source = 'textbook'") \
    .to_pandas()
print(results)

# Full SQL query
results = table.search().where("source = 'article'").to_pandas()
```

#### Qdrant (Local Mode)

**Overview**: Production-grade vector database that runs locally without a server. Excellent performance and feature set.

| Feature | Details |
|---|---|
| License | Apache 2.0 |
| Storage | Local disk |
| Search | Cosine, L2, IP, Manhattan |
| Filters | Extensive metadata + payload filtering |
| Index type | HNSW |
| Max vectors | 10M+ |
| Language | Python, Rust, Go, JS |
| API | gRPC, REST |

**Installation and usage:**

```bash
pip install qdrant-client
```

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Create local client (no server needed)
client = QdrantClient(path="./qdrant_data")

# Create collection
client.create_collection(
    collection_name="my_documents",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

# Add points
client.upsert(
    collection_name="my_documents",
    points=[
        PointStruct(
            id=1,
            vector=[0.1] * 768,
            payload={"text": "Machine learning...", "source": "textbook"}
        ),
        PointStruct(
            id=2,
            vector=[0.2] * 768,
            payload={"text": "Deep learning...", "source": "textbook"}
        )
    ]
)

# Search with filter
results = client.search(
    collection_name="my_documents",
    query_vector=[0.15] * 768,
    query_filter={
        "must": [
            {"key": "source", "match": {"value": "textbook"}}
        ]
    },
    limit=5
)

for result in results:
    print(result.payload["text"], result.score)
```

#### Comparison Table

| Feature | ChromaDB | LanceDB | Qdrant (Local) |
|---|---|---|---|
| **Ease of Setup** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Performance (1M vectors)** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Disk Efficiency** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Metadata Filtering** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Hybrid Search** | ❌ | ✅ | ✅ (with plugin) |
| **Full SQL Support** | ❌ | ✅ | ❌ |
| **API Ecosystem** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Max Practical Scale** | 10M | 100M+ | 10M+ |
| **Disk Usage (1M, 768d)** | ~4GB | ~2GB | ~3GB |

### Choosing a Vector Database

- **For simplicity and quick prototyping**: ChromaDB — just `pip install` and go
- **For large datasets (10M+ vectors)**: LanceDB — columnar storage is extremely efficient
- **For production deployments**: Qdrant — most features, best ecosystem, gRPC API
- **For hybrid search (BM25 + vectors)**: LanceDB or Qdrant with sparse vectors

---

## Document Ingestion Pipelines

### The Document Processing Pipeline

```
Raw Documents → [Extract] → [Chunk] → [Embed] → [Store in Vector DB]
```

### Step 1: Document Extraction

Supporting multiple file formats is essential for a practical RAG system.

```python
import os
from pathlib import Path

def extract_text(file_path):
    """Extract text from various file formats."""
    ext = Path(file_path).suffix.lower()
    
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    elif ext == '.md':
        import markdown
        with open(file_path, 'r', encoding='utf-8') as f:
            html = markdown.markdown(f.read())
        from bs4 import BeautifulSoup
        return BeautifulSoup(html, 'html.parser').get_text()
    
    elif ext == '.pdf':
        import pypdf
        reader = pypdf.PdfReader(file_path)
        return '\n'.join(page.extract_text() for page in reader.pages)
    
    elif ext == '.docx':
        import docx
        doc = docx.Document(file_path)
        return '\n'.join(p.text for p in doc.paragraphs)
    
    elif ext in ('.html', '.htm'):
        from bs4 import BeautifulSoup
        with open(file_path, 'r', encoding='utf-8') as f:
            return BeautifulSoup(f.read(), 'html.parser').get_text()
    
    elif ext == '.csv':
        import csv
        import io
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            return '\n'.join(' | '.join(row) for row in reader)
    
    elif ext == '.json':
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    
    elif ext == '.epub':
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
        book = epub.read_epub(file_path)
        texts = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                texts.append(BeautifulSoup(item.content, 'html.parser').get_text())
        return '\n'.join(texts)
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")
```

### Step 2: Text Chunking

Chunking is critical — the quality of your RAG system depends heavily on how you split documents.

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter
)

# Strategy 1: Recursive character splitting (recommended default)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,      # Tokens per chunk
    chunk_overlap=64,     # Overlap between chunks
    separators=["\n\n", "\n", ".", " ", ""],  # Try splitting at these points
    length_function=len,  # Character count
)
chunks = text_splitter.split_text(document_text)

# Strategy 2: Token-aware splitting (better for LLM context limits)
token_splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
)
chunks = token_splitter.split_text(document_text)

# Strategy 3: Markdown-aware splitting (preserves document structure)
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
chunks = markdown_splitter.split_text(markdown_document)

# Strategy 4: Semantic chunking (groups related sentences)
# Requires an embedding model to measure similarity between sentences
def semantic_chunk(text, model, max_chunk_size=512, similarity_threshold=0.8):
    """Split text into semantically coherent chunks."""
    import nltk
    nltk.download('punkt')
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_embedding = None
    
    for sentence in sentences:
        sentence_embedding = model.encode(sentence)
        
        if current_chunk and current_embedding is not None:
            similarity = np.dot(current_embedding, sentence_embedding) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(sentence_embedding)
            )
            if similarity < similarity_threshold or len(' '.join(current_chunk + [sentence])) > max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_embedding = sentence_embedding
            else:
                current_chunk.append(sentence)
                # Running average of embeddings
                current_embedding = (current_embedding + sentence_embedding) / 2
        else:
            current_chunk.append(sentence)
            current_embedding = sentence_embedding
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
```

#### Chunking Strategy Comparison

| Strategy | Pros | Cons | Best For |
|---|---|---|---|
| Recursive | Simple, effective, widely used | May break mid-sentence | Most use cases |
| Token-aware | Respects model limits | Can break semantic units | LLM context window management |
| Markdown-aware | Preserves document structure | Requires structured input | Documentation, articles, books |
| Semantic | Coherent chunks | Slow, requires embeddings | Question-answering, analysis |
| Sentence | Clean boundaries | Very small chunks | Factual lookups |
| Fixed size | Maximum simplicity | Poor semantic coherence | Simple indexing tasks |

### Step 3: Complete Ingestion Pipeline

```python
import os
import hashlib
from pathlib import Path
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer

class DocumentIngestionPipeline:
    def __init__(self, persist_dir: str = "./chroma_db"):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        
        # Initialize vector database
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Keep track of processed files
        self.processed_files = set()
    
    def process_file(self, file_path: str) -> Dict:
        """Process a single file and add to vector store."""
        file_path = str(Path(file_path).resolve())
        file_hash = self._hash_file(file_path)
        
        # Check if already processed
        if file_hash in self.processed_files:
            return {"status": "skipped", "file": file_path}
        
        try:
            # Extract text
            text = extract_text(file_path)
            
            # Chunk
            chunks = self._chunk_text(text)
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                chunks, 
                normalize_embeddings=True,
                show_progress_bar=False
            ).tolist()
            
            # Generate IDs and metadata
            ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
            metadatas = [{
                "source": file_path,
                "chunk": i,
                "total_chunks": len(chunks)
            } for i in range(len(chunks))]
            
            # Add to vector store
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            self.processed_files.add(file_hash)
            
            return {
                "status": "processed",
                "file": file_path,
                "chunks": len(chunks)
            }
        
        except Exception as e:
            return {
                "status": "error",
                "file": file_path,
                "error": str(e)
            }
    
    def process_directory(self, directory: str, extensions: List[str] = None):
        """Process all files in a directory."""
        if extensions is None:
            extensions = ['.txt', '.md', '.pdf', '.docx', '.html']
        
        results = []
        for ext in extensions:
            for file_path in Path(directory).rglob(f'*{ext}'):
                if file_path.is_file():
                    result = self.process_file(str(file_path))
                    results.append(result)
        
        return results
    
    def query(self, query_text: str, n_results: int = 5, where: Dict = None):
        """Query the vector store."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        return results
    
    def _hash_file(self, file_path: str) -> str:
        """Generate hash of file content for deduplication."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]
    
    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 64):
        """Split text into chunks with overlap."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        return splitter.split_text(text)
```

---

## Local RAG: Complete Implementation

### Simple Local RAG System

This is a complete, functional RAG system that runs entirely on your local machine.

```python
import os
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

class LocalRAG:
    def __init__(self, 
                 embed_model: str = "BAAI/bge-base-en-v1.5",
                 llm_model: str = "qwen2.5:7b",
                 persist_dir: str = "./chroma_db"):
        
        # Initialize embedding
        self.embed_model = SentenceTransformer(embed_model)
        
        # Initialize vector store
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="rag_docs",
            metadata={"hnsw:space": "cosine"}
        )
        
        # LLM model name
        self.llm_model = llm_model
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """Add documents to the RAG system."""
        if metadatas is None:
            metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
        
        # Generate embeddings
        embeddings = self.embed_model.encode(
            texts, 
            normalize_embeddings=True
        ).tolist()
        
        # Generate IDs
        ids = [f"doc_{hash(t)}_{i}" for i, t in enumerate(texts)]
        
        # Add to vector store
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        return len(texts)
    
    def retrieve(self, query: str, k: int = 4) -> List[str]:
        """Retrieve relevant documents for a query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return results['documents'][0] if results['documents'] else []
    
    def generate(self, query: str, context: List[str]) -> str:
        """Generate answer using retrieved context."""
        # Format context
        context_str = "\n\n".join([
            f"[Document {i+1}]: {doc}" 
            for i, doc in enumerate(context)
        ])
        
        # Create prompt
        prompt = f"""You are a helpful question-answering assistant.
Use the following context to answer the question at the end.
If you don't know the answer, say so instead of making up information.

Context:
{context_str}

Question: {query}

Answer:"""
        
        # Generate with Ollama
        response = ollama.chat(
            model=self.llm_model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1, 'num_predict': 512}
        )
        
        return response['message']['content']
    
    def query(self, question: str, k: int = 4, verbose: bool = False) -> str:
        """Complete RAG query: retrieve + generate."""
        # Retrieve
        context = self.retrieve(question, k=k)
        
        if verbose:
            print(f"\n--- Retrieved {len(context)} documents ---")
            for i, doc in enumerate(context):
                print(f"\n[Doc {i+1}]: {doc[:200]}...")
        
        # Generate
        answer = self.generate(question, context)
        
        return answer

# Usage
rag = LocalRAG(
    embed_model="BAAI/bge-base-en-v1.5",
    llm_model="qwen2.5:7b"  # Must be pulled in Ollama
)

# Add documents
rag.add_documents([
    "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data.",
    "Deep learning is a subset of machine learning that uses neural networks with multiple layers. It has achieved remarkable results in image recognition, natural language processing, and game playing.",
    "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language."
])

# Ask a question
answer = rag.query("What is the relationship between machine learning and deep learning?", verbose=True)
print(f"\nAnswer: {answer}")
```

### Advanced RAG with Re-ranking

```python
from sentence_transformers import CrossEncoder

class AdvancedRAG(LocalRAG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cross-encoder for re-ranking
        self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
    
    def retrieve(self, query: str, k: int = 10, final_k: int = 4) -> List[str]:
        """Retrieve and re-rank documents."""
        # Initial retrieval (more documents)
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        documents = results['documents'][0] if results['documents'] else []
        
        if len(documents) <= final_k:
            return documents
        
        # Re-rank
        pairs = [[query, doc] for doc in documents]
        scores = self.reranker.predict(pairs)
        
        # Sort by score
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        
        # Return top-k after re-ranking
        return [doc for doc, _ in ranked[:final_k]]
```

### Local RAG with Metadata Filtering

```python
# Filter by metadata during retrieval
results = rag.collection.query(
    query_texts=["machine learning basics"],
    n_results=5,
    where={"category": "tutorial"},  # Only from tutorials
    where_document={"$contains": "neural"}  # Documents containing "neural"
)

# More complex filters
results = rag.collection.query(
    query_texts=["transformer architecture"],
    n_results=10,
    where={
        "$and": [
            {"year": {"$gte": 2023}},
            {"category": "research_paper"}
        ]
    }
)
```

---

## Advanced Topics

### Hybrid Search (BM25 + Semantic)

Hybrid search combines keyword matching (BM25) with semantic search (embeddings) for better results.

```python
import chromadb.utils.embedding_functions as embedding_functions
from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearch:
    def __init__(self, persist_dir: str = "./chroma_db"):
        # Semantic search component
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="hybrid_docs",
            metadata={"hnsw:space": "cosine"}
        )
        
        # BM25 component (in-memory index)
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
    
    def add_documents(self, texts: List[str], ids: List[str], metadatas: List[Dict] = None):
        """Add documents to both semantic and BM25 indices."""
        self.documents.extend(texts)
        self.doc_ids.extend(ids)
        
        # Add to semantic index (via ChromaDB)
        self.collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadatas or [{}] * len(texts)
        )
        
        # Rebuild BM25 index
        tokenized = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query: str, k: int = 5, semantic_weight: float = 0.5):
        """Hybrid search combining BM25 and semantic scores."""
        # Semantic search
        semantic_results = self.collection.query(
            query_texts=[query],
            n_results=k * 2  # Get more for re-scoring
        )
        semantic_ids = semantic_results['ids'][0]
        semantic_scores = semantic_results['distances'][0]
        # Convert distances to similarity scores
        semantic_scores = [1 - d for d in semantic_scores]
        
        # BM25 search
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        
        # Combine scores
        combined = {}
        for i, doc_id in enumerate(semantic_ids):
            combined[doc_id] = {
                'semantic': semantic_scores[i],
                'bm25': 0,
                'document': self.documents[self.doc_ids.index(doc_id)]
            }
        
        for i, doc_id in enumerate(self.doc_ids):
            if doc_id not in combined:
                combined[doc_id] = {
                    'semantic': 0,
                    'bm25': bm25_scores[i] if bm25_scores[i] > 0 else 0,
                    'document': self.documents[i]
                }
            else:
                combined[doc_id]['bm25'] = bm25_scores[i] if i < len(bm25_scores) else 0
        
        # Weighted combination
        for doc_id in combined:
            combined[doc_id]['score'] = (
                semantic_weight * combined[doc_id]['semantic'] +
                (1 - semantic_weight) * combined[doc_id]['bm25']
            )
        
        # Sort and return top-k
        sorted_results = sorted(
            combined.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )
        
        return [
            {
                'id': doc_id,
                'score': data['score'],
                'document': data['document']
            }
            for doc_id, data in sorted_results[:k]
        ]
```

### Streaming RAG Responses

```python
async def stream_rag_answer(rag: LocalRAG, question: str):
    """Stream RAG answer token by token."""
    # Retrieve context
    context = rag.retrieve(question)
    context_str = "\n\n".join([f"[Doc {i+1}]: {d}" for i, d in enumerate(context)])
    
    prompt = f"""Context: {context_str}

Question: {question}

Answer:"""
    
    # Stream from Ollama
    stream = ollama.chat(
        model=rag.llm_model,
        messages=[{'role': 'user', 'content': prompt}],
        stream=True
    )
    
    for chunk in stream:
        yield chunk['message']['content']
```

### Multi-Collection RAG

```python
class MultiCollectionRAG:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collections = {}
    
    def add_collection(self, name: str, description: str = ""):
        """Add a named collection (like a knowledge base)."""
        self.collections[name] = self.client.get_or_create_collection(
            name=name,
            metadata={"description": description}
        )
    
    def query_all(self, query: str, k_per_collection: int = 3):
        """Query all collections and aggregate results."""
        all_results = []
        
        for name, collection in self.collections.items():
            results = collection.query(
                query_texts=[query],
                n_results=k_per_collection
            )
            
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    all_results.append({
                        'collection': name,
                        'document': doc,
                        'score': results['distances'][0][i] if results['distances'] else 0
                    })
        
        # Sort by score across all collections
        all_results.sort(key=lambda x: x['score'])
        
        return all_results[:5]  # Top 5 overall
```

---

## Performance Benchmarks

### Embedding Speed (texts/second on RTX 4090)

| Model | Batch Size 1 | Batch Size 32 | Batch Size 256 |
|---|---|---|---|
| BGE-small-en-v1.5 | 120 | 1,800 | 4,200 |
| BGE-base-en-v1.5 | 80 | 1,200 | 2,800 |
| BGE-large-en-v1.5 | 45 | 650 | 1,500 |
| Nomic-embed-text-v1.5 | 90 | 1,400 | 3,200 |
| E5-mistral-7b-instruct | 8 | 90 | 180 |

### Vector Search Latency (ChromaDB, Cosine Similarity)

| Number of Vectors | k=5 | k=10 | k=50 |
|---|---|---|---|
| 10,000 | <1ms | <1ms | 2ms |
| 100,000 | 2ms | 3ms | 8ms |
| 1,000,000 | 15ms | 20ms | 45ms |
| 10,000,000 | 120ms | 150ms | 300ms |

### Vector Search Latency (LanceDB, Cosine Similarity with IVF-PQ Index)

| Number of Vectors | k=5 | k=10 | k=50 |
|---|---|---|---|
| 10,000 | <1ms | <1ms | 1ms |
| 100,000 | 1ms | 2ms | 5ms |
| 1,000,000 | 5ms | 8ms | 20ms |
| 10,000,000 | 15ms | 25ms | 60ms |
| 100,000,000 | 40ms | 60ms | 150ms |

---

## Comparison with Cloud Alternatives

| Aspect | Local RAG | Cloud RAG (OpenAI + Pinecone) |
|---|---|---|
| **Embedding Cost** | $0 (one-time hardware) | ~$0.0001/1K tokens (embedding) |
| **Vector Storage Cost** | $0 (local disk) | ~$70/month per 1M vectors (Pinecone) |
| **Query Cost** | $0 | ~$0.01–0.10 per query |
| **Privacy** | Complete | Data sent to third parties |
| **Latency (embedding)** | 5–50ms | 100–500ms |
| **Latency (search)** | 1–150ms | 10–50ms |
| **Latency (generation)** | 50–500ms | 200–2,000ms |
| **Offline Capability** | ✅ Full offline | ❌ Requires internet |
| **Vector Limit** | Disk space (millions) | Depends on plan |
| **Customization** | Full control | Limited |

---

## Best Practices

### For Document Quality

1. **Clean your documents**: Remove headers, footers, page numbers, and other noise
2. **Normalize whitespace**: Collapse multiple spaces, normalize line endings
3. **Handle Unicode**: Ensure proper encoding (UTF-8) for all text
4. **Preserve structure**: Use markdown-aware chunking for structured documents
5. **Extract metadata**: Track source, date, author, section for filtering

### For Embedding Quality

1. **Use appropriate model size**: BGE-large for quality, BGE-base for speed
2. **Normalize embeddings**: Always normalize to unit length
3. **Batch encode**: Process documents in batches for GPU efficiency
4. **Instruction prefixes**: Some models benefit from query: and passage: prefixes
5. **Mixed precision**: Use FP16 for faster encoding

### For Retrieval Quality

1. **Chunk size matters**: 256–512 tokens is the sweet spot for most use cases
2. **Overlap helps**: 10–15% overlap prevents information loss at chunk boundaries
3. **Retrieve more, then re-rank**: Get 10 results, re-rank to top 3–5
4. **Hybrid search**: Combine BM25 and embeddings for best results
5. **Metadata filtering**: Filter by source, date, category for precision

### For Generation Quality

1. **Low temperature**: Use 0.1–0.3 for factual question-answering
2. **Explicit instructions**: Tell the model to use only the provided context
3. **Source citations**: Ask the model to cite sources from the context
4. **Context window**: Ensure your LLM's context window can fit all retrieved docs
5. **Fallback handling**: Tell the model to say "I don't know" rather than inventing

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| Poor retrieval results | Wrong embedding model | Switch to BGE-large or E5 |
| Empty search results | No documents indexed | Check document count; verify collection name |
| Slow embedding | Large model on CPU | Use BGE-base; enable GPU; batch encode |
| Out of memory | Too many documents in memory | Use incremental indexing; reduce batch size |
| Wrong answers from RAG | Irrelevant context | Increase k; add re-ranker; improve chunking |
| High disk usage | Many large collections | Use LanceDB for better compression; clean old data |
| API connection refused | Server not running | Start Ollama; check ChromaDB persistence path |

---

## Conclusion

Local LLM indexing and search has matured into a practical, powerful capability that can replace cloud-based alternatives for most use cases. With top embedding models now rivaling OpenAI's quality, vector databases that run efficiently on local hardware, and mature RAG frameworks, building a fully local question-answering system over your documents is straightforward.

The key to success is choosing the right combination of embedding model, chunking strategy, vector database, and LLM for your specific use case. Start simple (BGE-base + ChromaDB + recursive chunking + Ollama), benchmark on your data, and iterate.

Local RAG is not just a cost-saving measure — it is a privacy-enabling, sovereignty-preserving approach that puts you in full control of your data and your AI infrastructure.

---
**See also:**
- [03 — LLM Architectures 2026: Beyond the Vanilla Transformer](07-Emerging/17-Research-Frontiers-2026/03-LLM-Architectures-2026.md)
- [AI Evaluation and LLM Testing — Overview](69-AI-Evaluation-and-LLM-Testing/01-Overview.md)

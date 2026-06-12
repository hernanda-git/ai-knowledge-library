# Vector Databases

> A comprehensive, deeply technical guide to vector databases for production AI systems. Covers search algorithms, indexing strategies, major database platforms, and deployment patterns.

---

## Table of Contents

1. [Vector Search Fundamentals](#1-vector-search-fundamentals)
2. [Distance Metrics](#2-distance-metrics)
3. [Faiss](#3-faiss)
4. [Chroma](#4-chroma)
5. [Pinecone](#5-pinecone)
6. [Qdrant](#6-qdrant)
7. [Weaviate](#7-weaviate)
8. [Milvus](#8-milvus)
9. [LanceDB](#9-lancedb)
10. [Elasticsearch](#10-elasticsearch)
11. [Vector Database Comparison](#11-vector-database-comparison)

---

## 1. Vector Search Fundamentals

### 1.1 ANN vs kNN

**k-Nearest Neighbors (kNN)** performs an exact search: given a query vector, it computes the distance to every vector in the database and returns the k closest. This guarantees perfect recall but is O(N × D) per query — linear in the number of vectors and their dimensionality. At 1M vectors of 768 dimensions, a single exact search takes hundreds of milliseconds on CPU and tens of milliseconds on GPU — unacceptable for latency-sensitive applications.

**Approximate Nearest Neighbor (ANN)** sacrifices a small amount of recall for dramatic speed improvements. ANN algorithms preprocess the vector space into an index structure that allows sub-linear search time, typically O(log N) or O(√N). The trade-off is configurable: at 99% recall, ANN can be 10–100× faster than exact search.

**When to Use Each:**

| Criteria | kNN (Exact) | ANN (Approximate) |
|---|---|---|
| Dataset size | <10K vectors | >10K vectors |
| Latency requirement | Not critical | <100ms p99 |
| Recall requirement | 100% required | 95–99% acceptable |
| Query frequency | Low | High |
| Use case | Debugging, validation, small datasets | Production RAG, semantic search, recommendations |

### 1.2 Indexing Types

#### IVF (Inverted File Index)

IVF partitions the vector space into Voronoi cells via k-means clustering. At search time, only vectors in the nprobe closest cells to the query are examined.

**Structure:**
```
Cluster centroids (nlist centroids learned via k-means)
    → Each vector assigned to nearest centroid
    → Inverted lists: for each centroid, list of vectors + their IDs
```

**Parameters:**

| Parameter | Description | Typical Range | Effect |
|---|---|---|---|
| `nlist` | Number of Voronoi cells (clusters) | 100–32768 | Higher = finer granularity, more accurate, slower indexing |
| `nprobe` | Number of cells probed during search | 1–256 | Higher = more accurate, slower search |

**Performance Characteristics:**
- Search complexity: O(nlist × D + (N/nlist) × nprobe × D)
- Memory: O(N × D) for vectors + O(nlist × D) for centroids
- Build time: O(N × nlist × D × niter) for k-means

**Choosing nlist and nprobe:**
```
Rule of thumb: nlist = 4 × sqrt(N) for balanced performance
Example: N = 1M → nlist = 4 × 1000 = 4000

nprobe tuning: start at nprobe = sqrt(nlist) and double until recall saturates
Example: nlist = 4096 → start nprobe = 64
```

#### HNSW (Hierarchical Navigable Small World)

HNSW builds a multi-layer graph where the top layer has few nodes (long-range connections) and lower layers have increasingly many nodes (short-range connections). Search traverses from top to bottom, narrowing the search radius at each layer.

**Structure:**
```
Layer 0: All N nodes (bottom layer - base)
Layer 1: ~N / M nodes (sparser)
Layer 2: ~N / M² nodes
...
Layer L: ~1 node (top layer - entry point)
```

**Parameters:**

| Parameter | Description | Typical Range | Effect |
|---|---|---|---|
| `M` | Maximum number of connections per node per layer | 8–64 | Higher = more accurate, more memory, slower build |
| `Mmax` | Maximum connections for Layer 0 (often 2× M) | 16–128 | Higher = better recall for dense areas |
| `efConstruction` | Dynamic candidate list size during build | 100–500 | Higher = better index quality, slower build |
| `efSearch` | Dynamic candidate list size during search | 50–500 | Higher = better recall, slower search |

**Performance Characteristics:**
- Search complexity: O(log N × M × efSearch)
- Memory: ~(M × 2 + Mmax) × N × (12–16 bytes per link) + vector storage
- Build time: O(N × (log N + efConstruction × M))

**Memory Calculation (HNSW):**
```
For N=1M, M=32, Mmax=64:
Links: 1M × (32 × 2 + 64) × 12 bytes ≈ 1.5 GB (link metadata)
Vectors: 1M × 768 × 4 bytes (float32) = 3.1 GB
Total: ~4.6 GB
```

**Choosing HNSW Parameters:**
```
M = 16: Good balance for most use cases (16 connections/node)
M = 32: Higher recall, ~2× memory
M = 64: Maximum recall, ~4× memory

efConstruction = 200: Good default
efConstruction = 400: Better for high-recall requirements

efSearch = search_radius * M (approximately)
efSearch = 100 → 99% recall (typical)
efSearch = 200 → 99.5% recall
```

#### DiskANN (SSD-Optimized Vamana Graph)

DiskANN is designed for billion-scale datasets that don't fit in RAM. It uses a Vamana graph (a variant of HNSW with optimized degree constraints) stored primarily on SSD with only the graph structure cached in memory.

**Vamana Graph Properties:**
- Directed graph with out-degree bound R and in-degree bound R
- Greedy search with a priority queue
- Robustness parameter α controls exploration vs exploitation during build

**Parameters:**

| Parameter | Description | Typical Range |
|---|---|---|
| `R` | Degree bound for each node | 32–128 |
| `L` | Build complexity (candidate list size) | 100–500 |
| `α` | Robustness parameter (≥1) | 1.0–2.0 |

**Architecture:**
```
SSD:
├── Compressed Vectors (PQ-encoded, 8 bytes per dimension group)
└── Full Precision Vectors (for re-ranking)

RAM:
├── Graph Structure (R × N × 4 bytes for node IDs + metadata)
└── PQ Codebooks (negligible)

Search Flow:
1. Navigate Vamana graph (graph in RAM, PQ vectors in SSD cache)
2. Retrieve top-k candidates
3. Re-rank with full-precision vectors (optional, SSD fetch)
```

**Performance:**
- 1B vectors: ~8 TB SSD, ~16 GB RAM (for graph), <50ms search
- R=64, α=1.2 typically gives 95%+ recall at 1M QPS on SSD

#### PQ (Product Quantization)

Product Quantization compresses vectors by splitting them into sub-vectors and quantizing each sub-vector separately using a learned codebook.

**Process:**
```
Original vector: [v₁, v₂, ..., v_D]  (D dimensions)

Split into M sub-vectors:
[v₁...v_{D/M}], [v_{D/M+1}...v_{2D/M}], ..., [v_{D-D/M}...v_D]

For each sub-space, learn k centroids (codebook) via k-means:
Sub-space 1: codebook {c₁¹, c₁², ..., c₁ᵏ}
Sub-space 2: codebook {c₂¹, c₂², ..., c₂ᵏ}
...
Sub-space M: codebook {c_M¹, c_M², ..., c_Mᵏ}

Encode each sub-vector as index of nearest centroid:
Original vector → [id₁, id₂, ..., id_M] where id_j ∈ [0, k-1]

Compression ratio:
Original: D × 4 bytes (float32)
Encoded: M × log₂(k) bits
```

**Parameters:**

| Parameter | Description | Typical Range | Effect |
|---|---|---|---|
| `M` | Number of sub-vectors / codebooks | 4–128 | Higher = more accurate, less compression |
| `k` | Codebook size per sub-space | 256 (8 bit) | Higher = more accurate, more memory |
| `codebook_size` | Number of centroids | 256–65536 | 256 is standard (8-bit index) |

**Compression Examples:**
```
D=768, M=96, k=256 (8-bit):
Original: 768 × 4 = 3072 bytes
PQ: 96 × 1 = 96 bytes
Compression ratio: 32×
Quality: ~90% recall at 32× compression (SDC)

D=768, M=32, k=256:
Compression: 768 → 32 bytes (24×)
Quality: ~95% recall

D=768, M=8, k=256:
Compression: 768 → 8 bytes (96×)
Quality: ~70% recall
```

**Search with PQ (ADC — Asymmetric Distance Computation):**
```
For query q, compute distance to each PQ-encoded vector x:
d(q, x)² ≈ Σ d(q_j - c_j[id_j])²  for j = 1...M

Where:
- q_j is the j-th sub-vector of query
- c_j[id_j] is the centroid for the j-th subspace for x
- Pre-compute distance from q_j to each centroid in codebook j
- Result: M lookups + M additions per comparison
```

**SDC vs ADC:**
- **SDC** (Symmetric Distance Computation): Compress query too; faster but less accurate
- **ADC** (Asymmetric): Keep query in original precision; slower but more accurate

#### Scalar Quantization (SQ8)

Reduces each float32 dimension to 8-bit integer:

```
SQ8:
Original: float32 ∈ [-∞, ∞] → per-dimension min/max → uint8 ∈ [0, 255]

Quantization: x_quant = round((x - x_min) / (x_max - x_min) * 255)
Dequantization: x_approx = x_min + (x_quant / 255) × (x_max - x_min)
```

**Memory:**
```
float32: D × 4 bytes
SQ8:    D × 1 byte + 2 × D bytes (min/max per dimension)
Savings: 4× reduction
```

**Quality:** Typically retains >99% of search quality for most embedding models. Some information loss can cause 0.5–2% recall drop.

#### Binary Quantization

Converts vectors to binary (1-bit per dimension) using the sign of each component:

```
Binary:
x_binary = sign(x) ∈ {0, 1}^D

Hamming distance = popcount(x_binary XOR q_binary)

Memory: D / 8 bytes (e.g., 768D → 96 bytes → 12 bytes)
```

**When to Use:**
- Very large datasets (>100M vectors) where recall 90–95% is acceptable
- Memory-constrained environments
- First-pass retrieval with re-ranking via full-precision vectors

#### Hybrid Indexing

Modern vector databases combine multiple indexing strategies:

| Combination | Use Case | Example |
|---|---|---|
| IVF + PQ | Large-scale, memory-constrained | Faiss IndexIVFPQ |
| HNSW + PQ | Fast, accurate search with compression | Faiss IndexHNSWPQ |
| IVF + SQ | Balanced memory/accuracy | Faiss IndexIVFScalarQuantizer |
| IVF + Binary | Extreme compression | Milvus BIN_IVF_FLAT |
| Vamana + PQ | SSD-based billion-scale | DiskANN |

### 1.3 Distance Metrics

#### Cosine Similarity (Angular)

```
cosine_sim(A, B) = (A · B) / (||A|| × ||B||)
```

- **Range**: [-1, 1] (though for embeddings usually [0, 1] after ReLU)
- **Interpretation**: 1 = identical direction, 0 = orthogonal, -1 = opposite
- **When to use**: Text embeddings (sentence-transformers, OpenAI, Cohere), document similarity
- **Normalization**: L2-normalize vectors first → cosine = dot product

#### Dot Product (IP)

```
dot_product(A, B) = Σ(Aᵢ × Bᵢ)
```

- **Range**: Unbounded
- **When to use**: Models trained with inner product loss, recommendation systems, some OpenAI embeddings
- **Normalization**: If A and B are L2-normalized, dot product = cosine similarity
- **Note**: Dot product is sensitive to vector magnitude — higher magnitude = higher score regardless of direction

#### Euclidean Distance (L2)

```
l2_dist(A, B) = √(Σ(Aᵢ - Bᵢ)²)
```

- **Range**: [0, ∞)
- **Interpretation**: 0 = identical; higher = more different
- **When to use**: When magnitude matters (e.g., image embeddings, audio embeddings)
- **Relationship**: For normalized vectors, L2² = 2 × (1 - cosine)

#### L2 Squared

```
l2_squared(A, B) = Σ(Aᵢ - Bᵢ)²
```

- **Same ordering as L2** (monotonic)
- **Computationally cheaper**: No sqrt operation
- **Used internally** in many databases instead of L2

#### Inner Product

```
inner_product(A, B) = A · B
```

- Same as dot product
- Used in some ANN libraries as the computational basis for cosine (after normalization)

#### Hamming Distance

```
hamming(A, B) = popcount(A XOR B)
```

- Works on binary vectors only
- **Useful for**: Binary embeddings, hash codes, BioBERT binary vectors

#### Jaccard Distance

```
jaccard(A, B) = 1 - |A ∩ B| / |A ∪ B|
```

- **Used for**: Sparse vectors, set-based similarity, document overlap
- Range: [0, 1]

#### When to Use Each Metric

| Application | Recommended Metric | Reason |
|---|---|---|
| Text similarity (sentence-transformers) | Cosine | Embeddings normalized by default |
| OpenAI ada-002 | Cosine | Cosine recommended in docs |
| OpenAI text-embedding-3-* | Cosine | Normalized embeddings |
| Cohere embeddings | Cosine | Default metric |
| Image similarity (CLIP) | Cosine | Normalized embeddings |
| Item-item recommendation (collaborative filtering) | Dot Product | Magnitude encodes popularity |
| E5 / BGE embeddings | Cosine | Training uses cosine |
| Binary embeddings | Hamming | Only viable metric |
| Sparse vectors (SPLADE, uniCOIL) | Dot Product | Sparse representations |
| Audio embeddings | L2 / Cosine | Depends on model |
| Legal document similarity | Cosine | Standard for legal NLP |

#### Normalization Strategies

**L2 Normalization:**
```python
def l2_normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v
```

**Min-Max Normalization:**
```python
def min_max_normalize(v):
    return (v - v.min()) / (v.max() - v.min())
```

**Impact on Search:**
- L2 normalization makes cosine = dot product — simplifies computation
- Some databases (Pinecone, Qdrant) auto-normalize for cosine metric
- Always normalize pre-computed embeddings for consistent results

---

## 2. Distance Metrics

### 2.1 Detailed Metric Analysis

#### Cosine Similarity

**Mathematical Properties:**
- Scale invariant: ||A|| = ||B|| = 1 after normalization
- Rotationally invariant
- Not a true distance (not a metric — doesn't satisfy triangle inequality)

**When Cosine Fails:**
- Vectors with varying magnitudes where magnitude is meaningful (e.g., term frequency vectors)
- Very sparse vectors (Jaccard or inner product may be better)

#### Dot Product

**Mathematical Properties:**
- Not scale invariant
- Sensitive to vector magnitude
- If both vectors are normalized: same as cosine

**When to Use Dot Product:**
- For word2vec, GloVe (trained with dot product objective)
- Recommendation systems where magnitude encodes confidence/popularity
- When your model explicitly outputs unnormalized vectors

#### Euclidean Distance

**Mathematical Properties:**
- Proper metric (satisfies triangle inequality)
- Sensitive to magnitude and direction
- Not rotationally invariant (unlike cosine)

**When Euclidean Excels:**
- Clustering applications (k-means naturally uses L2)
- Spatial or geospatial data
- When the "length" of the embedding carries information
- Anomaly detection (outliers have large L2 distance from centroids)

#### Comparison Table

| Metric | Range | Sensitive to Magnitude | Proper Metric | Normalized Form |
|---|---|---|---|---|
| Cosine | [0, 1] | No | No (pseudo-metric) | L2-normalize → dot product |
| Dot Product | (-∞, ∞) | Yes | No | N/A |
| Euclidean (L2) | [0, ∞) | Yes | Yes | L2-normalize → √(2 - 2×cosine) |
| L2 Squared | [0, ∞) | Yes | No | Same ordering as L2 |
| Hamming | [0, D] | N/A | Yes | N/A |
| Jaccard | [0, 1] | N/A | Yes | N/A |

### 2.2 Normalization Strategies in Production

**Pre-normalization (Recommended):**
```python
def normalize_embeddings(embeddings):
    """L2 normalize a batch of embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-10, None)

# During ingestion
vectors = normalize_embeddings(vectors)

# Query also normalized
query = normalize_embeddings(query[np.newaxis, :])[0]
```

**Storage Considerations:**
- Normalized float32 vectors: same disk space, but dot product can replace cosine
- Normalized + SQ8: 4× compression with minimal recall loss
- Normalized + Binary: 32× compression, ~90% recall

### 2.3 Metric Selection by Database

| Database | Supported Metrics |
|---|---|
| Faiss | METRIC_INNER_PRODUCT, METRIC_L2, METRIC_L1, METRIC_Linf, etc. |
| Chroma | cosine (default), l2, ip |
| Pinecone | cosine (default), dotproduct, euclidean |
| Qdrant | Cosine, Dot, L2, Manhattan |
| Weaviate | cosine (default), dot, l2-squared, hamming, manhattan |
| Milvus | L2, IP, COSINE, HAMMING, JACCARD |
| LanceDB | L2, Cosine, Dot |
| Elasticsearch | cosine, dot_product, l1, l2, hamming |

---

## 3. Faiss

Faiss (Facebook AI Similarity Search) is the most widely used vector search library — the de facto benchmark and foundation for many production systems. It is a library, not a database, but its index types form the backbone of many vector database internals.

### 3.1 IndexFactory String

Faiss uses a compact string format to compose indices:

```
IndexFactory: "IVF4096,PQ32"
               ↓        ↓
            Inverted File  Product Quantization
            with 4096      with 32 sub-vectors
            centroids
```

**Component Prefixes:**

| Prefix | Meaning | Example |
|---|---|---|
| `Flat` | No compression, exact storage | `IndexFlatIP` |
| `IVF` | Inverted File index | `IVF4096` |
| `HNSW` | Hierarchical Navigable Small World | `HNSW32` |
| `PQ` | Product Quantization | `PQ16` |
| `SQ` | Scalar Quantization | `SQ8` |
| `PCA` | PCA dimension reduction | `PCA64` |
| `OPQ` | Optimized Product Quantization | `OPQ16` |
| `LSH` | Locality Sensitive Hashing | `LSH` |
| `IDMap` | Add IDs to index (for removal) | `IDMap,IVF4096,PQ32` |
| `Refine` | Re-rank with full precision | `IVF4096,PQ32,Refine(Flat)` |
| `RFlat` | Re-rank with exact distances | Same as Refine |

**Common Factory Strings:**

```
# Exact search
"Flat"                          # Brute force, L2
"FlatIP"                        # Brute force, inner product
"FlatIP,IDMap"                  # With ID support

# IVF variants
"IVF100,Flat"                   # IVF, no compression
"IVF4096,Flat"                  # IVF, 4096 centroids
"IVF4096,Flat,IDMap"            # IVF with IDs

# IVF + PQ (memory efficient)
"IVF4096,PQ16"                  # IVF + 16-byte PQ codes
"IVF4096,PQ32"                  # IVF + 32-byte PQ codes  
"IVF16384,PQ64"                 # More centroids, larger codes

# IVF + SQ (balanced)
"IVF4096,SQ8"                   # IVF + scalar quantization
"IVF16384,SQ8"                  # More centroids

# HNSW
"HNSW32,Flat"                   # HNSW with exact vectors
"HNSW32,PQ16"                   # HNSW with PQ compression

# Preprocessing + index
"PCA64,IVF4096,PQ16"            # PCA to 64D, then IVF+P
"OPQ16_256,IVF4096,PQ16"        # OPQ rotation, then IVF+PQ
```

### 3.2 Index Types in Detail

#### IndexFlat (Brute Force, Exact)

```python
import faiss

d = 768  # dimension
index = faiss.IndexFlatL2(d)
# or
index = faiss.IndexFlatIP(d)

# Add vectors
index.add(vectors)  # vectors: (N, d) float32 array

# Search
D, I = index.search(query, k=10)  # D: distances, I: indices
```

**Characteristics:**
- O(N × D) per query
- No indexing structure — just raw storage
- Memory: N × D × 4 bytes
- Best for: Small datasets (<100K), validation, exact search

#### IndexIVFFlat (Inverted File, No Compression)

```python
nlist = 100
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

# Train (required for IVF)
index.train(vectors)
index.add(vectors)

# Set search parameter
index.nprobe = 10  # Number of probe cells

# Search
D, I = index.search(query, k=10)
```

**Characteristics:**
- Faster than Flat at 100K+ vectors
- Exact vectors stored in each inverted list
- Memory: N × D × 4 bytes + nlist × D × 4 (centroids)

#### IndexIVFPQ (Inverted File + Product Quantization)

```python
nlist = 4096
m = 32  # number of PQ sub-vectors
nbits = 8  # bits per sub-vector

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

# Train
index.train(vectors)
index.add(vectors)

# Search
index.nprobe = 10
D, I = index.search(query, k=10)
```

**Characteristics:**
- Most popular index for large-scale search
- Memory: N × m × nbits/8 bytes + centroids + PQ codebooks
- 1M vectors × 768 dims: ~100 MB for PQ32 vs ~3 GB for Flat

#### IndexIVFScalarQuantizer

```python
index = faiss.IndexIVFScalarQuantizer(
    quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit
)
index.train(vectors)
index.add(vectors)
index.nprobe = 10
```

**Characteristics:**
- 4× memory reduction vs Flat
- Better quality than PQ at same bit rate for some distributions
- Supports QT_4bit, QT_6bit, QT_8bit

#### IndexHNSWFlat

```python
M = 32
index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efConstruction = 200
index.add(vectors)

# Search
index.hnsw.efSearch = 64
D, I = index.search(query, k=10)
```

**Characteristics:**
- Very fast search at high recall
- Memory: N × D × 4 + link metadata
- Better than IVF for small-to-medium datasets (<10M)

#### IndexHNSWPQ

```python
M = 32
m = 32
index = faiss.IndexHNSWPQ(d, m, M)
index.hnsw.efConstruction = 200
index.train(vectors)
index.add(vectors)
index.hnsw.efSearch = 64
```

**Characteristics:**
- HNSW graph with PQ-compressed vectors
- Balanced speed, memory, and accuracy

#### IndexBinaryIVF

```python
index = faiss.IndexBinaryIVF(quantizer, d // 8, nlist)
index.train(binary_vectors)
index.add(binary_vectors)
index.nprobe = 10

# Search (returns Hamming distances)
D, I = index.search(query_binary, k=10)
```

**Characteristics:**
- For binary vectors only
- Uses Hamming distance
- Extremely fast (XOR + popcount)

### 3.3 GPU Support

Faiss GPU accelerates search via:

```python
import faiss

# CPU index
cpu_index = faiss.IndexFlatL2(d)

# Move to single GPU
gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index)

# Multi-GPU
gpu_indices = faiss.index_cpu_to_all_gpus(cpu_index)

# Direct GPU index construction
config = faiss.GpuIndexFlatConfig()
config.useFloat16 = True     # Use half-precision storage
config.usePrecomputed = False  # Precompute distances (faster but more memory)
config.device = 0

gpu_index_flat = faiss.GpuIndexFlatL2(d, config)
```

**GpuIndexFlat:**
```python
config = faiss.GpuIndexFlatConfig()
config.useFloat16 = True  # Store vectors in FP16 (half memory, minimal quality loss)
config.storeTransposed = False
config.device = 0

index = faiss.GpuIndexFlatL2(d, config)
```

**GpuIndexIVF:**
```python
config = faiss.GpuIndexIVFConfig()
config.device = 0
config.useFloat16CoarseQuantizer = False  # Coarse quantizer in FP32

# Enable precomputed tables (faster but more memory)
config.usePrecomputedTables = True

# Set number of GPUs
config.indicesOptions = faiss.INDICES_CPU  # Store indices on CPU
# or config.indicesOptions = faiss.INDICES_GPU  # Store on GPU

gpu_index = faiss.GpuIndexIVFFlat(gpu_resources, d, nlist, faiss.METRIC_L2, config)
```

**Multi-GPU Configuration:**
```python
# Standard: split vectors across GPUs
params = faiss.GpuMultipleClonerOptions()
params.shard = True  # Shard across GPUs
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, params)

# Alternative: replicate index on each GPU (independent search)
params.shard = False
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, params)
```

### 3.4 Clustering (k-means)

Faiss k-means is used for IVF codebook training and is highly optimized:

```python
kmeans = faiss.Kmeans(
    d=d,              # Dimension
    k=nlist,          # Number of centroids
    niter=20,         # Number of k-means iterations
    nredo=1,          # Number of re-do attempts (pick best)
    verbose=True,
    spherical=False,  # If True, normalize centroids to unit length
    int_centroids=False,  # If True, centroids are integer coordinates
    seed=42,
)

kmeans.train(vectors)

# Centroids
centroids = kmeans.centroids  # (nlist, d) array

# Index mapping (which cluster each vector belongs to)
indices = kmeans.assign(vectors)  # (N,) integer labels
```

**k-means Parameters:**

| Parameter | Effect |
|---|---|
| `niter` | More iterations = better centroids, diminishing returns after 20 |
| `nredo` | Multiple random initializations, pick lowest loss |
| `spherical` | For cosine similarity (normalized centroids) |
| `int_centroids` | For binary signatures or integer embeddings |

### 3.5 Preprocessing

**PCA (Principal Component Analysis):**

```python
# Reduce dimension: 768 → 256
pca_matrix = faiss.PCAMatrix(d, 256, eigen_power=-0.5)
pca_matrix.train(vectors)
reduced = pca_matrix.apply(vectors)

# Then build index on reduced vectors
index = faiss.IndexIVFFlat(faiss.IndexFlatL2(256), 256, nlist)
index.train(reduced)
index.add(reduced)
```

**OPQ (Optimized Product Quantization):**

OPQ learns a rotation matrix that aligns the vector space to minimize PQ distortion:

```python
# OPQ16: rotate then PQ with 16 sub-vectors
opq_matrix = faiss.OPQMatrix(d, 16)
opq_matrix.train(vectors)

# Combine with IVF+P
index = faiss.IndexIVFPQ(
    faiss.IndexFlatL2(d), d, nlist, 16, 8
)
index.own_fields = True
index.pq_matrix = opq_matrix
index.train(vectors)
index.add(vectors)
```

### 3.6 Search Parameters

**IVF Search Parameters:**
```python
index.nprobe = 10  # Default. Start at sqrt(nlist) and tune up.

# For parallel search
index.parallel_mode = 1  # Parallelize over probes (not queries)
```

**HNSW Search Parameters:**
```python
index.hnsw.efSearch = 64  # Candidate list size. Higher = slower but more accurate.
```

**Batch Search:**
```python
# Search multiple queries at once
queries = np.random.random((1000, d)).astype('float32')
D, I = index.search(queries, k=10)  # D: (1000, 10), I: (1000, 10)
```

**Range Search:**
```python
# Find all vectors within radius
lims, D, I = index.range_search(query, radius=0.5)
# lims: boundaries for each query
# For query i, results are at I[lims[i]:lims[i+1]]
```

**Adding with IDs:**
```python
index = faiss.IndexIDMap(cpu_index)
ids = np.arange(N).astype('int64')
index.add_with_ids(vectors, ids)
```

**Removing Vectors:**
```python
# IDMap supports removal
index.remove_ids(np.array([5, 10, 15]))  # Remove specific IDs

# Or by ID selector
import faiss
sel = faiss.IDSelectorRange(0, 100)
index.remove_ids(sel)  # Remove IDs 0-99
```

### 3.7 I/O Operations

```python
# Save index
faiss.write_index(index, "index.faiss")

# Load index
index = faiss.read_index("index.faiss")

# Save with file handle
with open("index.faiss", "wb") as f:
    faiss.write_index(index, faiss.PythonCallbackIOWriter(f))

# For binary indices
faiss.write_index_binary(bin_index, "binary_index.faiss")
faiss.read_index_binary("binary_index.faiss", io_flags=faiss.IO_FLAG_MMAP)
```

**MMAP (Memory-Mapped I/O):**
```python
# Load index as memory-mapped (lazy loading, shared across processes)
index = faiss.read_index("index.faiss", faiss.IO_FLAG_MMAP)
```

**Clone and Copy Index:**
```python
import copy
index_copy = faiss.clone_index(index)  # Independent copy

# Or
index_copy = faiss.index_saencode(index)  # Deep copy
```

### 3.8 Training with Representative Data

IVF and PQ indices require training data. Best practices:

```python
# Use representative sample
sample_size = min(N, 50000)  # 50K is usually sufficient
rng = np.random.RandomState(123)
sample_idx = rng.choice(N, sample_size, replace=False)
sample = vectors[sample_idx]

# Train
index.train(sample)

# Add all vectors
index.add(vectors)
```

**For very large datasets:**
```python
# Progressive training: train on increasing samples
niter = 20
for i in range(niter):
    sample = vectors[np.random.randint(0, N, 50000)]
    index.train(sample)
    
# Note: Faiss accumulates training data, so last iteration dominates
# Better: use a fixed representative sample
```

### 3.9 Coalescing Indexes

Combining multiple indexes into one:

```python
# Merge indexes
index1 = faiss.read_index("shard1.faiss")
index2 = faiss.read_index("shard2.faiss")

# For IndexFlat
combined = faiss.IndexFlatL2(d)
combined.add(index1.reconstruct_n(0, index1.ntotal))
combined.add(index2.reconstruct_n(0, index2.ntotal))

# For IndexIVF has no built-in merge; rebuild from vectors
```

### 3.10 Removing and Adding Vectors

```python
# IDMap index supports dynamic operations
index = faiss.IndexIDMap(faiss.IndexFlatL2(d))

# Add with IDs
index.add_with_ids(vectors, ids)

# Remove
index.remove_ids(np.array([remove_id]))

# Add more
new_vectors = ...
new_ids = ...
index.add_with_ids(new_vectors, new_ids)

# Note: IVF indices don't support removal natively
# Workaround: use IDMap wrapper and reconstruct for re-adding
def update_ivf(ivf_index, idmap, ids_to_remove, new_vectors, new_ids):
    # Get remaining vectors
    all_ids = idmap.id_map
    remaining_mask = ~np.isin(all_ids, ids_to_remove)
    
    remaining_vectors = vectors[remaining_mask]
    remaining_ids = all_ids[remaining_mask]
    
    # Rebuild index
    new_index = faiss.index_factory(d, "IVF4096,Flat")
    new_index.train(remaining_vectors)
    new_index.add_with_ids(remaining_vectors, remaining_ids)
    
    return new_index
```

---

## 4. Chroma

Chroma is an open-source, developer-friendly vector database with a Rust core and Python/JS client libraries. It is designed for simplicity and rapid prototyping, but also supports production deployment.

### 4.1 Architecture

```
Python/JS Client
    ↕ HTTP or gRPC
Rust Core (server or embedded)
    ↕
Storage Backend (RocksDB/SQLite by default)
    ↕
Embedding Function (local or API-based)
```

**Core Design:**
- Full-stack vector database (embeddings + storage + search)
- Embedding function integrated into client (no separate embedding step)
- Written in Rust for performance, with first-class Python bindings

### 4.2 Python Client

```python
import chromadb
from chromadb.config import Settings

# In-memory (default)
client = chromadb.Client()

# Persistent
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_data"
))

# Distributed (HTTP client)
client = chromadb.HttpClient(
    host="chroma-server.example.com",
    port=8000
)
```

### 4.3 Collection Operations

```python
# Create collection
collection = client.create_collection(
    name="my_collection",
    metadata={"description": "Enterprise document embeddings"},
    embedding_function=None,  # Provide custom embedding function
)

# Or get existing
collection = client.get_or_create_collection("my_collection")

# Add documents
collection.add(
    embeddings=[[1.2, 2.3, ...], ...],  # Or use embedding_function
    documents=["doc1", "doc2"],
    metadatas=[{"source": "pdf"}, {"source": "web"}],
    ids=["id1", "id2"],
)

# Add with automatic embedding
collection.add(
    documents=["Text to embed automatically"],
    ids=["auto_id1"],
)

# Get by ID
result = collection.get(
    ids=["id1"],
    include=["documents", "metadatas", "embeddings"]
)

# Update
collection.update(
    ids=["id1"],
    documents=["Updated document"],
    metadatas=[{"source": "updated"}],
)

# Delete
collection.delete(ids=["id1"])
# Or delete with filter
collection.delete(where={"source": "obsolete_source"})

# Count
count = collection.count()

# List collections
collections = client.list_collections()

# Delete collection
client.delete_collection("my_collection")
```

### 4.4 Metadata Filtering

Chroma's `where` filter syntax supports rich metadata queries:

**Comparison Operators:**
```python
# Exact match
collection.get(where={"source": "pdf"})

# Numeric comparison
collection.get(where={"year": {"$gt": 2020}})
collection.get(where={"year": {"$gte": 2020}})
collection.get(where={"year": {"$lt": 2023}})
collection.get(where={"year": {"$lte": 2023}})
collection.get(where={"year": {"$ne": 2022}})

# String operations
collection.get(where={"source": {"$contains": "news"}})
```

**Compound Operators:**
```python
# AND
collection.get(where={
    "$and": [
        {"source": "pdf"},
        {"year": {"$gte": 2020}},
    ]
})

# OR
collection.get(where={
    "$or": [
        {"source": "pdf"},
        {"source": "web"},
    ]
})

# IN
collection.get(where={"category": {"$in": ["legal", "finance"]}})

# NOT IN
collection.get(where={"category": {"$nin": ["spam", "temporary"]}})
```

**Filter + Search:**
```python
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],
    n_results=10,
    where={"year": {"$gte": 2023}},
    where_document={"$contains": "machine learning"},  # Filter by doc content
)
```

### 4.5 Embedding Functions

```python
from chromadb import EmbeddingFunction
from sentence_transformers import SentenceTransformer

class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def __call__(self, texts):
        return self.model.encode(texts).tolist()

# Custom embedding function
collection = client.create_collection(
    name="custom_emb",
    embedding_function=CustomEmbeddingFunction(),
)

# Or use Chroma's built-in default (all-MiniLM-L6-v2)
# collection = client.create_collection(name="default_emb")
```

### 4.6 HNSW Configuration

Chroma uses HNSW internally and exposes some parameters:

```python
collection = client.create_collection(
    name="hnsw_collection",
    metadata={
        "hnsw:space": "cosine",  # "cosine" | "l2" | "ip"
        "hnsw:construction_ef": 200,  # efConstruction parameter
        "hnsw:M": 32,                  # Maximum connections per node
        "hnsw:search_ef": 100,        # efSearch parameter
        "hnsw:num_threads": 4,        # Threads for indexing
    },
)
```

**HNSW Parameters:**
| Parameter | Key | Default | Effect |
|---|---|---|---|
| Space | `hnsw:space` | "l2" | Distance function |
| M | `hnsw:M` | 16 | Connections per node |
| efConstruction | `hnsw:construction_ef` | 200 | Build quality vs time |
| efSearch | `hnsw:search_ef` | 100 | Search quality vs speed |
| num_threads | `hnsw:num_threads` | 4 | Index build parallelism |

### 4.7 Persistence and Distributed Mode

**Persistence:**
```python
# Persistent client saves to disk automatically
client = chromadb.PersistentClient(path="./chroma_db")

# Data is stored in:
# ./chroma_db/
# ├── chroma.sqlite3
# └── <collection_uuid>/
#     ├── header.bin
#     ├── length.bin
#     ├── link_lists.bin
#     └── data.bin
```

**Distributed Client Mode (HTTP):**
```python
# Server
# chroma run --path ./chroma_db --host 0.0.0.0 --port 8000

# Client
client = chromadb.HttpClient(
    host="chroma-server.internal",
    port=8000,
    settings=Settings(
        chroma_server_http_headers={"Authorization": "Bearer token"},
        chroma_server_ssl_enabled=True,
    ),
)
```

### 4.8 Chroma Cloud

Chroma Cloud is the managed SaaS offering:

```
Features:
- Usage-based pricing
- Auto-scaling
- SLA-backed availability
- Multi-region support
- Backup and restore
- Access control (API keys)
```

**Connecting to Chroma Cloud:**
```python
import chromadb

client = chromadb.HttpClient(
    host="api.chroma.cloud",
    port=443,
    ssl=True,
    settings=Settings(
        chroma_client_auth_provider="token",
        chroma_client_auth_credentials="your-api-key",
    ),
)
```

### 4.9 Limits and Constraints

| Resource | Limit | Notes |
|---|---|---|
| Max embeddings per collection | Unlimited | Depends on storage |
| Max metadata size per entry | ~16KB | Larger metadata impacts performance |
| Max batch size | 1000 (recommended) | Can be higher with performance tuning |
| Max dimensionality | 65536 | Practical limit for most models |
| Max query results | 10000 | `n_results` parameter |
| Embedding precision | float32 | No quantization built-in |
| Collection name length | 128 chars | Alphanumeric, hyphens, underscores |

### 4.10 Telemetry Opt-Out

```python
# Chroma collects anonymous usage data by default. Opt out:
import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(anonymized_telemetry=False))
```

---

## 5. Pinecone

Pinecone is a fully managed, cloud-native vector database designed for production AI workloads. It offers both serverless and pod-based architectures.

### 5.1 Architecture

#### Serverless (Podless)

Pinecone's latest architecture eliminates manual pod management:

```
Client → Pinecone API → Serverless Infrastructure
                          ├── Storage-Optimized Tier
                          ├── Compute Tier (auto-scaling)
                          └── Metadata Index
```

**Key Features:**
- Zero infrastructure management
- Auto-scaling (pods added/removed automatically)
- Pay-per-usage (no idle costs)
- Index creation without specifying pod size
- Storage-optimized architecture (vectors stored efficiently)

**Creating a Serverless Index:**
```python
import pinecone

pc = pinecone.Pinecone(api_key="your-api-key")

pc.create_index(
    name="serverless-index",
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1",
    ),
)
```

#### Pod-Based (Legacy/Classic)

Traditional Pinecone with dedicated compute pods:

```
Client → Pinecone API → Pod Cluster
                         ├── Pod 1 (shard)
                         ├── Pod 2 (replica)
                         ├── Pod 3 (shard + replica)
                         └── ...
```

**Pod Types:**

| Pod Type | Purpose | vCPU | RAM | Storage |
|---|---|---|---|---|
| **p1** | Performance (search speed) | 2 vCPU | 8 GB | 400 GB |
| **s1** | Standard (balanced) | 2 vCPU | 16 GB | 400 GB |
| **e1** | Economy (cost-effective) | 2 vCPU | 8 GB | 400 GB |

**Capacity per Pod (estimated, depends on dimension and metadata):**
| Pod Type | Vectors (1536D) | Queries/sec (single pod) |
|---|---|---|
| p1 | ~1M | ~500 |
| s1 | ~2M | ~200 |
| e1 | ~5M | ~50 |

**Creating a Pod-based Index:**
```python
pc.create_index(
    name="pod-index",
    dimension=768,
    metric="cosine",
    pod_type="p1",
    pods=1,
    replicas=1,
    shards=1,
)
```

### 5.2 Pod Sizing

**Replicas vs Shards:**
- **Replicas**: Copies of the data for high availability and read throughput. More replicas = higher QPS.
- **Shards**: Data partitioned across pods. More shards = larger capacity.

**Sizing Formula:**
```
Total capacity = Vectors per shard × Shards
Read throughput = Pod throughput × Replicas
Write throughput = Pod throughput (limited by shards)
```

**Sizing Examples:**
```python
# 10M vectors, 768D, cosine metric, high throughput
pc.create_index(
    name="production-index",
    dimension=768,
    metric="cosine",
    pod_type="s1",
    pods=4,       # 4 pods total
    shards=2,     # 2 shards (5M vectors each)
    replicas=2,   # 2 replicas of each shard
)

# 100M vectors, low throughput, low cost
pc.create_index(
    name="large-archive",
    dimension=768,
    metric="cosine",
    pod_type="e1",
    pods=20,      # 20 pods
    shards=20,    # 20 shards (5M each)
    replicas=1,   # No replication
)
```

### 5.3 Index Configuration

```python
pc.create_index(
    name="my-index",
    dimension=768,
    metric="cosine",  # "cosine" | "dotproduct" | "euclidean"
    spec=ServerlessSpec(
        cloud="aws",
        region="us-west-2",
    ),
    # Pod-based only:
    # metadata_config={"indexed": ["category", "year"]},  # Pre-define metadata fields
)
```

### 5.4 Vector Operations

```python
index = pc.Index("my-index")

# Upsert vectors
index.upsert(
    vectors=[
        {
            "id": "vec1",
            "values": [0.1, 0.2, ...],  # 768-dim float vector
            "metadata": {"category": "science", "year": 2023},
        },
        {
            "id": "vec2",
            "values": [0.3, 0.4, ...],
            "metadata": {"category": "history", "year": 2022},
        },
    ],
    namespace="ns1",  # Optional namespace
)

# Upsert in batches (recommended for large datasets)
def batch_upsert(index, vectors, batch_size=100):
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)

# Query
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=10,
    namespace="ns1",
    include_metadata=True,
    include_values=False,
    filter={"category": {"$eq": "science"}},
)

# Query with sparse vectors (hybrid search)
results = index.query(
    vector=dense_vector,
    sparse_vector={
        "indices": [1, 5, 100],
        "values": [0.5, 0.3, 0.2],
    },
    top_k=10,
)

# Fetch by ID
result = index.fetch(ids=["vec1", "vec2"], namespace="ns1")

# Update
index.update(
    id="vec1",
    values=[0.5, 0.6, ...],
    set_metadata={"category": "updated"},
    namespace="ns1",
)

# Delete
index.delete(ids=["vec1"], namespace="ns1")
index.delete(filter={"category": {"$eq": "obsolete"}}, namespace="ns1")
index.delete(delete_all=True, namespace="ns1")  # Clear namespace

# Describe index stats
stats = index.describe_index_stats()
```

### 5.5 Namespaces

Namespaces partition vectors within a single index:

```python
# Different namespaces for different use cases
index.upsert(vectors=vecs_legal, namespace="legal")
index.upsert(vectors=vecs_medical, namespace="medical")

# Query specific namespace
results = index.query(
    vector=query,
    top_k=10,
    namespace="legal",  # Only searches this namespace
)
```

**Characteristics:**
- Unlimited namespaces per index
- Each namespace can have different metadata schemas
- No cross-namespace querying (single namespace per query)
- Namespace isolation for multi-tenant setups

### 5.6 Metadata Filtering

Pinecone supports rich metadata filtering:

```python
# Equality
filter = {"category": {"$eq": "science"}}

# Not equal
filter = {"category": {"$ne": "spam"}}

# Greater than / less than
filter = {"price": {"$gt": 10, "$lte": 100}}

# Multiple values (OR)
filter = {"category": {"$in": ["science", "technology"]}}

# Exclude multiple values
filter = {"category": {"$nin": ["spam", "garbage"]}}

# Existence check
filter = {"author": {"$exists": True}}

# Combined (AND)
filter = {
    "$and": [
        {"category": {"$eq": "science"}},
        {"year": {"$gte": 2020}},
    ]
}

# Note: Pinecone doesn't support $or at top level
# Use $and with $in as workaround
```

**Pre-filtering vs Post-filtering:**
Pinecone uses pre-filtering (metadata filter is applied during ANN search, not as a post-processing step), which means filtered search is efficient even at large scale.

### 5.7 Sparse-Dense Retrieval and Hybrid Search

Pinecone supports hybrid search combining dense embeddings with sparse (keyword-based) vectors:

```python
# Generate sparse vectors using built-in or custom methods
sparse_vector = {
    "indices": [1, 5, 100],  # Term IDs (vocabulary indices)
    "values": [0.5, 0.3, 0.2],  # BM25 or SPLADE scores
}

results = index.query(
    vector=dense_vector,
    sparse_vector=sparse_vector,
    top_k=10,
    alpha=0.75,  # Weight between dense (1-alpha) and sparse (alpha)
)
```

**Alpha Parameter:**
- `alpha = 0.0`: Dense-only (standard vector search)
- `alpha = 0.5`: Equal weight to dense and sparse
- `alpha = 1.0`: Sparse-only (keyword search)
- `alpha = 0.75`: Default (favor sparse, complementary to dense)

**When to Use Hybrid:**
- Query contains rare terms not in dense embedding space
- Domain-specific vocabulary (legal, medical, code)
- Cold-start scenarios (embedding model not yet optimized for domain)

### 5.8 gRPC vs REST API

```python
# REST API (default)
index = pc.Index("my-index")

# gRPC API (lower latency, higher throughput)
index = pc.Index("my-index", pool_threads=10)
```

**Comparison:**

| Feature | REST | gRPC |
|---|---|---|
| Latency (p50) | ~5ms | ~3ms |
| Max request size | 4 MB | 100 MB |
| Batching | Limited | Better batching support |
| Connection reuse | TCP pool | Persistent connection |
| Client complexity | Simple | Slightly more complex |

**Recommendation:** Use gRPC for high-throughput production workloads. Use REST for simple integrations and serverless functions (cold start).

### 5.9 Backup and Restore

```python
# Create backup (collection)
pc.create_backup(
    index_name="my-index",
    collection_name="my-index-backup-2024-06-01",
)

# Restore from backup
pc.restore_collection(
    collection_name="my-index-backup-2024-06-01",
)

# Describe collection
collection = pc.describe_collection("my-index-backup-2024-06-01")

# List collections
collections = pc.list_collections()

# Delete collection
pc.delete_collection("my-index-backup-2024-06-01")
```

### 5.10 Pinecone Assistant

Pinecone Assistant is a RAG-as-a-service offering that uses Pinecone under the hood:

```python
from pinecone import PineconeAssistant

assistant = PineconeAssistant(
    api_key="your-api-key",
    assistant_name="my-assistant",
    instructions="You are a helpful assistant for enterprise documents.",
)

# Upload documents
assistant.upload_document("path/to/document.pdf")

# Ask questions
response = assistant.ask("What is the company policy on expense reports?")
```

### 5.11 Pinecone Inference

Pinecone Inference provides serverless embedding generation:

```python
# Embed text without managing your own model
embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=["Text to embed", "More text"],
    parameters={"input_type": "passage"},
)

# Use in index
index.upsert(vectors=[
    {
        "id": "vec1",
        "values": embeddings[0],
        "metadata": {"text": "Text to embed"},
    }
])
```

**Supported Models:**
- `multilingual-e5-large` (text-embedding)
- `multilingual-e5-small` (text-embedding)
- `bert-base-uncased` (text-embedding)

---

## 6. Qdrant

Qdrant is a high-performance vector database written in Rust, designed for production deployments with rich filtering, quantization, and advanced storage options.

### 6.1 Architecture

```
Client (REST/gRPC)
    ↕
Qdrant Service
    ├── Collections
    │   ├── Collection (logical unit)
    │   │   ├── Segments (immutable + mutable)
    │   │   │   ├── Vector Storage
    │   │   │   ├── Payload (metadata) Index
    │   │   │   └── Quantization Data
    │   │   └── WAL (Write-Ahead Log)
    │   └── Shards (distributed)
    └── Snapshots
```

### 6.2 Collections

```python
from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(host="localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="my_collection",
    vectors_config=models.VectorParams(
        size=768,
        distance=models.Distance.COSINE,
        on_disk=True,  # Store vectors on disk (not RAM)
    ),
    optimizers_config=models.OptimizersConfigDiff(
        default_segment_number=2,
        memmap_threshold=20000,
    ),
    hnsw_config=models.HnswConfigDiff(
        m=16,
        ef_construct=200,
        full_scan_threshold=10000,
    ),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        ),
    ),
)

# Get collection info
info = client.get_collection("my_collection")
```

**Collection Configuration Parameters:**

| Parameter | Description | Default |
|---|---|---|
| `vectors_config` | Vector size, distance metric, storage | Required |
| `optimizers_config` | Optimization behavior | Default |
| `hnsw_config` | HNSW graph parameters | Default |
| `quantization_config` | Quantization for memory reduction | None |
| `shard_number` | Number of shards (distributed) | 1 |
| `replication_factor` | Data replication | 1 |
| `write_consistency_factor` | Consistency for writes | 1 |
| `on_disk_payload` | Payload on disk (not in RAM) | False |

### 6.3 Vector Operations

```python
# Upsert points
client.upsert(
    collection_name="my_collection",
    points=[
        models.PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],  # 768-dim
            payload={
                "title": "Document 1",
                "category": "science",
                "year": 2023,
                "tags": ["AI", "ML"],
                "price": 29.99,
            },
        ),
    ],
)

# Query
results = client.search(
    collection_name="my_collection",
    query_vector=[0.1, 0.2, ...],
    limit=10,
    with_payload=True,
    with_vectors=False,
    score_threshold=0.5,  # Minimum similarity score
)

# Batch query
results = client.search_batch(
    collection_name="my_collection",
    requests=[
        models.SearchRequest(vector=query1, limit=10),
        models.SearchRequest(vector=query2, limit=10),
    ],
)

# Query by ID (fetching results similar to given ID)
results = client.recommend(
    collection_name="my_collection",
    positive=[1, 2, 3],  # Similar to these IDs
    negative=[4, 5],      # Dissimilar to these IDs
    limit=10,
)

# Scroll (sequential scan)
scroll_result = client.scroll(
    collection_name="my_collection",
    limit=100,
    with_payload=True,
    offset=previous_offset,  # Pagination
)

# Count
count = client.count(
    collection_name="my_collection",
    count_filter=...  # Optional filter
)

# Update payload (metadata)
client.set_payload(
    collection_name="my_collection",
    payload={"new_field": "value"},
    points=[1, 2, 3],
)

# Delete by filter
client.delete(
    collection_name="my_collection",
    points_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="year",
                range=models.Range(gte=2020),
            ),
        ],
    ),
)
```

### 6.4 Payload Indexing

Payload (metadata) fields can be indexed for efficient filtering:

```python
# Create keyword index
client.create_payload_index(
    collection_name="my_collection",
    field_name="category",
    field_type=models.PayloadSchemaType.KEYWORD,
)

# Create integer index
client.create_payload_index(
    collection_name="my_collection",
    field_name="year",
    field_type=models.PayloadSchemaType.INTEGER,
)

# Create float index
client.create_payload_index(
    collection_name="my_collection",
    field_name="price",
    field_type=models.PayloadSchemaType.FLOAT,
)

# Create geo index
client.create_payload_index(
    collection_name="my_collection",
    field_name="location",
    field_type=models.PayloadSchemaType.GEO,
)

# Create text index (for full-text search)
client.create_payload_index(
    collection_name="my_collection",
    field_name="title",
    field_type=models.PayloadSchemaType.TEXT,
)

# Get index info
indexes = client.list_payload_indexes("my_collection")
```

### 6.5 Payload Filter Operators

Qdrant has the most comprehensive filtering syntax among vector databases:

```python
from qdrant_client.http import models as qdrant_filter

# Simple field condition
filter = qdrant_filter.Filter(
    must=[
        qdrant_filter.FieldCondition(
            key="category",
            match=qdrant_filter.MatchValue(value="science"),
        ),
    ],
)

# Comparison operators
filter = qdrant_filter.Filter(
    must=[
        qdrant_filter.FieldCondition(
            key="price",
            range=qdrant_filter.Range(
                gt=10.0,
                gte=10.0,
                lt=100.0,
                lte=100.0,
            ),
        ),
    ],
)

# String matching
filter = qdrant_filter.Filter(
    must=[
        qdrant_filter.FieldCondition(
            key="title",
            match=qdrant_filter.MatchText(text="quantum"),  # Full-text search
        ),
    ],
)

# Multiple values
filter = qdrant_filter.Filter(
    should=[
        qdrant_filter.FieldCondition(
            key="category",
            match=qdrant_filter.MatchAny(any=["science", "technology"]),
        ),
    ],
)

# Missing / Null
filter = qdrant_filter.Filter(
    must=[
        qdrant_filter.FieldCondition(
            key="optional_field",
            is_null=qdrant_filter.IsNull(is_null=True),  # null check
        ),
    ],
)

# Nested object filtering
filter = qdrant_filter.Filter(
    must_not=[
        qdrant_filter.NestedCondition(
            nested=qdrant_filter.Nested(
                key="metadata",
                filter=qdrant_filter.Filter(
                    must=[
                        qdrant_filter.FieldCondition(
                            key="status",
                            match=qdrant_filter.MatchValue(value="inactive"),
                        ),
                    ],
                ),
            ),
        ),
    ],
)

# Compound filter (AND + OR + NOT)
filter = qdrant_filter.Filter(
    must=[
        qdrant_filter.Filter(
            should=[
                qdrant_filter.FieldCondition(
                    key="category",
                    match=qdrant_filter.MatchValue(value="science"),
                ),
                qdrant_filter.FieldCondition(
                    key="category",
                    match=qdrant_filter.MatchValue(value="technology"),
                ),
            ],
        ),
    ],
    must_not=[
        qdrant_filter.FieldCondition(
            key="year",
            range=qdrant_filter.Range(lt=2020),
        ),
    ],
)
```

**Operator Summary:**

| Operator | Qdrant Expression | Values |
|---|---|---|
| Equals | `MatchValue(value=x)` | Any type |
| Not equals | `MatchExcept(except=[x])` | List of excluded values |
| Greater than | `Range(gt=x)` | Numbers |
| Greater or equal | `Range(gte=x)` | Numbers |
| Less than | `Range(lt=x)` | Numbers |
| Less or equal | `Range(lte=x)` | Numbers |
| IN list | `MatchAny(any=[x,y])` | List of values |
| NOT IN | `MatchExcept(except=[x,y])` | List of excluded values |
| Full-text match | `MatchText(text="query")` | String |
| Is null | `IsNull(is_null=True)` | Boolean |
| Geo radius | `GeoRadius(...)` | Geo coordinates |
| Geo polygon | `GeoPolygon(...)` | Geo path |

### 6.6 Geo-Search

Qdrant has built-in geo-spatial search capabilities:

```python
# Index the geo field first
client.create_payload_index(
    collection_name="my_collection",
    field_name="location",
    field_type=models.PayloadSchemaType.GEO,
)

# Upsert with geo data
client.upsert(
    collection_name="my_collection",
    points=[
        models.PointStruct(
            id=1,
            vector=[...],
            payload={
                "location": {
                    "lat": 40.7128,
                    "lon": -74.0060,  # New York
                },
                "name": "NYC Office",
            },
        ),
    ],
)

# Geo radius search
results = client.search(
    collection_name="my_collection",
    query_vector=[...],
    query_filter=qdrant_filter.Filter(
        must=[
            qdrant_filter.FieldCondition(
                key="location",
                geo_radius=qdrant_filter.GeoRadius(
                    center=qdrant_filter.GeoPoint(
                        lat=40.7128,
                        lon=-74.0060,
                    ),
                    radius=10000.0,  # meters
                ),
            ),
        ],
    ),
    limit=10,
)

# Geo polygon search
results = client.search(
    collection_name="my_collection",
    query_vector=[...],
    query_filter=qdrant_filter.Filter(
        must=[
            qdrant_filter.FieldCondition(
                key="location",
                geo_polygon=qdrant_filter.GeoPolygon(
                    exterior=qdrant_filter.GeoBoundingBox(
                        top_left=qdrant_filter.GeoPoint(lat=40.8, lon=-74.1),
                        bottom_right=qdrant_filter.GeoPoint(lat=40.6, lon=-73.9),
                    ),
                ),
            ),
        ],
    ),
    limit=10,
)
```

### 6.7 Quantization

Qdrant supports multiple quantization methods to reduce memory usage:

**Scalar Quantization (INT8):**
```python
client.create_collection(
    collection_name="my_collection",
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,  # 8-bit scalar quantization
            quantile=0.99,  # Clip outliers at 99th percentile (better quality)
            always_ram=True,  # Keep quantized data in RAM (for speed)
        ),
    ),
)
```

**Product Quantization:**
```python
client.create_collection(
    collection_name="my_collection",
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    quantization_config=models.ProductQuantization(
        product=models.ProductQuantizationConfig(
            compression=models.CompressionRatio.X32,  # 32x compression
            always_ram=False,  # May need to load from disk
        ),
    ),
)
```

**Binary Quantization:**
```python
client.create_collection(
    collection_name="my_collection",
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    quantization_config=models.BinaryQuantization(
        binary=models.BinaryQuantizationConfig(
            always_ram=True,
        ),
    ),
)
```

**Quantization Comparison:**

| Method | Memory | Quality (Recall) | Speed |
|---|---|---|---|
| None (float32) | 100% | 100% | Fastest |
| Scalar (INT8) | 25% | ~99% | Fast |
| Product (X32) | 3.125% | ~90-95% | Slower (ADC) |
| Binary | 3.125% | ~85-90% | Fastest (XOR+popcount) |

### 6.8 Multi-Vector (Named Vectors)

Qdrant supports multiple vectors per point (e.g., for ColBERT or multi-modal embeddings):

```python
client.create_collection(
    collection_name="multi_vector",
    vectors_config={
        "text": models.VectorParams(size=768, distance=models.Distance.COSINE),
        "image": models.VectorParams(size=512, distance=models.Distance.DOT),
        "code": models.VectorParams(size=384, distance=models.Distance.COSINE),
    },
)

# Upsert with multiple vectors
client.upsert(
    collection_name="multi_vector",
    points=[
        models.PointStruct(
            id=1,
            vector={
                "text": [0.1, 0.2, ...],
                "image": [0.5, 0.6, ...],
                "code": [0.9, 0.8, ...],
            },
            payload={"description": "Multi-modal document"},
        ),
    ],
)

# Query on specific vector
results = client.search(
    collection_name="multi_vector",
    query_vector=("text", [0.1, 0.2, ...]),  # (vector_name, vector)
    limit=10,
)
```

### 6.9 Sharding and Replication

**Sharding:**
```python
# Auto-sharding (Qdrant decides)
client.create_collection(
    collection_name="sharded_collection",
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    shard_number=4,  # Distribute across 4 shards
    sharding_method=models.ShardingMethod.AUTO,  # or CUSTOM
)
```

**Replication:**
```python
client.create_collection(
    collection_name="replicated_collection",
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    replication_factor=3,  # Three copies
    write_consistency_factor=2,  # Majority for writes
)
```

**Consistency Levels:**
- `write_consistency_factor`: Number of replicas that must acknowledge writes (default: 1). Higher = stronger consistency, slower writes.
- Read consistency: Strongly consistent within a single node; eventually consistent across replicas.

### 6.10 WAL (Write-Ahead Log)

Qdrant uses a WAL for durability:

```
Write → WAL (append-only log on disk)
         ↓
    Memory Segment (mutable)
         ↓ (optimizer flushes)
    Disk Segment (immutable, HNSW-indexed)
         ↓ (segment merger)
    Merged Disk Segment
```

**WAL Configuration:**
```python
client.create_collection(
    collection_name="my_collection",
    vectors_config=...,
    optimizers_config=models.OptimizersConfigDiff(
        default_segment_number=2,
        memmap_threshold=20000,   # Switch to memmap at this size
        indexing_threshold=20000, # HNSW indexing at this size
        vacuum_min_vector_number=1000,  # Vacuum threshold
        flush_interval_sec=5,     # WAL flush interval
    ),
)
```

### 6.11 Snapshots

```python
# Create full snapshot
snapshot_info = client.create_snapshot(
    collection_name="my_collection",
)

# Create incremental snapshot (recovery point)
# Qdrant supports WAL-based incremental recovery

# List snapshots
snapshots = client.list_snapshots("my_collection")

# Recover from snapshot
client.recover_snapshot(
    collection_name="my_collection",
    location="/path/to/snapshot",
)

# Delete snapshot
client.delete_snapshot("my_collection", snapshot_info.name)
```

### 6.12 Qdrant Cloud

```python
# Connect to Qdrant Cloud
client = QdrantClient(
    url="https://xxxx-xxxxx.us-east-1-0.aws.cloud.qdrant.io:6333",
    api_key="your-api-key",
)
```

**Features:**
- Managed clusters with auto-scaling
- Free tier (1GB storage)
- Multi-region deployment
- SOC 2 compliance
- 99.95% uptime SLA
- Automated backups

### 6.13 REST + gRPC API

```python
# REST (default)
client = QdrantClient(host="localhost", port=6333)

# gRPC (lower latency, binary protocol)
client = QdrantClient(host="localhost", port=6334, prefer_grpc=True)

# REST for management (create/delete collections)
# gRPC for data operations (search/upsert)

# API endpoints:
# REST: http://localhost:6333/collections/{name}/points/search
# gRPC: localhost:6334
```

### 6.14 Consistency Levels

```python
# Strong consistency (all replicas)
client.search(
    collection_name="my_collection",
    query_vector=...,
    consistency=models.ReadConsistencyType.STRONG,
    limit=10,
)

# Majority consistency
client.search(
    collection_name="my_collection",
    query_vector=...,
    consistency=models.ReadConsistencyType.MAJORITY,
    limit=10,
)

# Eventual consistency (fast, may return stale data)
client.search(
    collection_name="my_collection",
    query_vector=...,
    consistency=models.ReadConsistencyType.EVENTUAL,
    limit=10,
)
```

### 6.15 Optimization

```python
# Optimizer configuration
client.update_collection(
    collection_name="my_collection",
    optimizers_config=models.OptimizersConfigDiff(
        default_segment_number=2,      # More segments = faster writes, slower queries
        indexing_threshold=50000,      # Build HNSW after 50K vectors per segment
        memmap_threshold=50000,        # Use memory-mapped files after 50K
        vacuum_min_vector_number=1000, # Clean up deleted vectors
    ),
)
```

**Segment Management:**
```
Small segments: Frequent merges, fast writes
Large segments: Slower merges, faster reads
Balance: 2-5 segments per collection optimal for most workloads
```

**Performance Tuning Tips:**
- Increase `indexing_threshold` for write-heavy workloads
- Decrease `default_segment_number` for read-heavy workloads
- Use `on_disk=True` for vectors when RAM is constrained
- Enable `quantization_config` for memory reduction
- Use gRPC for higher throughput

---

## 7. Weaviate

Weaviate is an open-source vector database written in Go with native GraphQL API and integrated machine learning modules.

### 7.1 Architecture

```
Client (GraphQL / REST)
    ↕
Weaviate Server
    ├── Modules
    │   ├── text2vec-* (embedding generation)
    │   ├── generative-* (generation)
    │   ├── qna-openai (question answering)
    │   ├── ner-transformers (named entity recognition)
    │   └── custom modules
    ├── Vector Index (HNSW or Flat)
    ├── Object Storage
    └── Inverted Index (for filtering)
```

### 7.2 GraphQL API

Weaviate's primary API is GraphQL, enabling complex queries in a single request:

```python
import weaviate
import weaviate.classes as wvc

client = weaviate.connect_to_local()

# Get class (equivalent to collection)
class_obj = client.collections.get("Document")

# GraphQL query
response = class_obj.query.near_text(
    query="machine learning concepts",
    limit=10,
    return_metadata=wvc.query.MetadataQuery(distance=True, score=True),
)

# Raw GraphQL (for complex queries)
query = """
{
  Get {
    Document(
      nearText: {
        concepts: ["machine learning"]
      }
      limit: 10
      where: {
        operator: GreaterThan
        path: ["year"]
        valueInt: 2020
      }
    ) {
      title
      content
      _additional {
        distance
      }
    }
  }
}
"""
response = client.query.raw(query)
```

### 7.3 Classical API (REST)

```python
# Create schema / class
class_obj = client.collections.create(
    name="Document",
    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
    properties=[
        wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="year", data_type=wvc.config.DataType.INT),
    ],
)

# Insert data
doc = class_obj.data.insert(
    properties={
        "title": "Introduction to AI",
        "content": "Artificial intelligence is...",
        "year": 2023,
    },
)

# Batch import
with class_obj.batch.dynamic() as batch:
    for doc in documents:
        batch.add_object(
            properties={
                "title": doc["title"],
                "content": doc["content"],
                "year": doc["year"],
            },
            vector=doc.get("vector"),  # Optional: provide pre-computed vector
        )
```

### 7.4 Hybrid Search

Weaviate combines dense (vector) and sparse (BM25) search:

```python
response = class_obj.query.hybrid(
    query="quantum computing",
    alpha=0.75,  # Weight: 0 = pure BM25, 1 = pure vector
    limit=10,
    return_metadata=wvc.query.MetadataQuery(score=True, explain_score=True),
)

# For each result:
for obj in response.objects:
    print(f"Score: {obj.metadata.score}")
    print(f"Explanation: {obj.metadata.explain_score}")
```

**Alpha Parameter:**
- `alpha = 0.0`: Pure keyword search (BM25)
- `alpha = 0.5`: Equal weight
- `alpha = 1.0`: Pure vector search
- `alpha = 0.75`: Default

### 7.5 Generative Search

Weaviate can generate responses based on retrieved objects:

**Single (per-object generation):**
```python
response = class_obj.generate.near_text(
    query="quantum computing",
    limit=5,
    single_prompt="Summarize {title}: {content}",
)
# Generates a separate summary for each retrieved object
```

**Group (aggregated generation):**
```python
response = class_obj.generate.near_text(
    query="quantum computing",
    limit=5,
    grouped_task="Write a comprehensive overview based on the following information:",
)
# Generates a single response using all retrieved objects
```

### 7.6 Modules

Weaviate's modular architecture allows pluggable AI capabilities:

**Vectorizer Modules (automatic embedding):**
```python
# OpenAI
client = weaviate.connect_to_wcs(
    cluster_url="https://xxx.weaviate.network",
    auth_credentials=wvc.init.Auth.api_key("api-key"),
    headers={"X-OpenAI-Api-Key": "sk-..."},
)

# Cohere
client = weaviate.connect_to_local(
    headers={"X-Cohere-Api-Key": "cohere-key"},
)

# Hugging Face
client = weaviate.connect_to_local(
    headers={"X-HuggingFace-Api-Key": "hf-key"},
)

# Or use local inference (text2vec-transformers)
# docker run -p 8080:8080 semitechnologies/transformers-inference:sentence-transformers-multi-qa-MiniLM-L6-cos-v1
```

**Generative Modules:**
```python
# Generate responses using OpenAI
class_obj = client.collections.create(
    name="FAQ",
    generative_config=wvc.config.Configure.Generative.openai(),
)

# Generate using Cohere
class_obj = client.collections.create(
    name="FAQ",
    generative_config=wvc.config.Configure.Generative.cohere(),
)
```

**Other Modules:**
```python
# QnA Module (extractive QA)
response = class_obj.query.near_text(
    query="What is the capital of France?",
    limit=1,
    qna_prompt="Answer from the provided context:"
)
# Returns answer spans with start/end positions

# NER Module
# Extracts named entities automatically during import
```

### 7.7 Multi-Tenancy

Weaviate supports data isolation for multi-tenant applications:

```python
# Enable multi-tenancy on a class
class_obj = client.collections.create(
    name="TenantData",
    multi_tenant_config=wvc.config.Configure.multi_tenancy(enabled=True),
    properties=[
        wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
    ],
)

# Create tenants
class_obj.tenants.create([
    wvc.config.Tenant(name="tenant_a"),
    wvc.config.Tenant(name="tenant_b"),
])

# Tenant-specific operations
with client.collections.get("TenantData", tenant="tenant_a") as tenant_collection:
    tenant_collection.data.insert(properties={"content": "Tenant A data"})

# Queries are automatically scoped to the tenant
```

### 7.8 Replication

```python
# Configure replication factor
client.collections.create(
    name="ReplicatedDoc",
    replication_config=wvc.config.Configure.replication(
        factor=3,  # Three replicas
    ),
    properties=[
        wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
    ],
)

# Consistency levels for reads:
# - ONE: Read from one replica (fast, eventual consistency)
# - QUORUM: Read from majority (balanced)
# - ALL: Read from all replicas (strong consistency)
```

### 7.9 Backup

```python
# Filesystem backup
import weaviate.classes as wvc

client.backup.create(
    backup_id="my-backup-2024-06-01",
    backend="filesystem",  # or "s3", "gcs"
    include_classes=["Document", "FAQ"],
)

# Restore
client.backup.restore(
    backup_id="my-backup-2024-06-01",
    backend="filesystem",
    include_classes=["Document"],
)

# S3 backup configuration
# Set env: BACKUP_S3_BUCKET, BACKUP_S3_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
```

### 7.10 HNSW Parameters

```python
class_obj = client.collections.create(
    name="OptimizedCollection",
    vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
        ef_construction=200,
        ef=64,  # efSearch
        max_connections=32,
        dynamic_ef=True,  # Auto-adjust ef based on dataset size
        vector_cache_max_objects=1000000,
    ),
)
```

**HNSW Parameter Tuning in Weaviate:**

| Parameter | Description | Default | Range |
|---|---|---|---|
| `efConstruction` | Index build quality | 128 | 100-500 |
| `ef` (efSearch) | Search quality | -1 (auto) | 32-500 |
| `maxConnections` | M parameter | 64 | 16-128 |
| `dynamic_ef` | Auto-ef during search | True | Bool |
| `vectorCacheMaxObjects` | Number of vectors in RAM | 1e12 | 0-1e12 |

### 7.11 Flat Index on Vectors

For exact (brute-force) search:

```python
class_obj = client.collections.create(
    name="ExactSearchCollection",
    vector_index_config=wvc.config.Configure.VectorIndex.flat(
        vector_cache_max_objects=10000,  # Cache in RAM
    ),
)
```

**Flat Index Use Cases:**
- Small datasets (<50K vectors)
- Requiring 100% recall
- Debugging and validation
- When HNSW search time is acceptable at dataset size

---

## 8. Milvus

Milvus is a cloud-native vector database designed for billion-scale similarity search with distributed architecture and GPU acceleration.

### 8.1 Distributed Architecture

```
Load Balancer
    ↓
Proxy (stateless, handles client requests)
    ↓
┌──────────────────────────────────────┐
│              Message Broker          │ (Kafka / Pulsar)
└──────┬──────────────┬───────────────┘
       ↓              ↓               ↓
   Root Coord    Data Coord      Query Coord    Index Coord
       ↓              ↓               ↓               ↓
   Root Service   Data Nodes      Query Nodes    Index Nodes
                   (storage)       (search)       (index build)
```

**Component Roles:**

| Component | Responsibility |
|---|---|
| **Root Coord** | Cluster management, schema management, TSO (timestamp oracle) |
| **Data Coord** | Manages data nodes, segment lifecycle, garbage collection |
| **Query Coord** | Manages query nodes, load balancing, read requests |
| **Index Coord** | Manages index building, index versioning |
| **Proxy** | Client-facing gateway, authentication, request routing |
| **Data Nodes** | Store and manage data segments |
| **Query Nodes** | Execute vector and scalar queries |
| **Index Nodes** | Build vector indexes |

### 8.2 Collections and Schema

```python
from pymilvus import MilvusClient, Collection, CollectionSchema, FieldSchema, DataType

# Milvus Lite (embedded, for development)
client = MilvusClient("milvus.db")

# Milvus Server
client = MilvusClient(uri="http://localhost:19530")

# Create schema
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,  # Allow arbitrary metadata fields
)

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=100)
schema.add_field(field_name="year", datatype=DataType.INT64)

# Create collection
client.create_collection(
    collection_name="my_collection",
    schema=schema,
    index_params={
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    },
    consistency_level="BoundedStaleness",  # Consistency level
)
```

### 8.3 Index Types

Milvus supports a wide range of index types through its Knowhere library:

```python
# FLAT (brute force, exact search)
index_params = {
    "index_type": "FLAT",
    "metric_type": "L2",
}

# IVF_FLAT (inverted file, no compression)
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 1024},
}

# IVF_SQ8 (inverted file + scalar quantization)
index_params = {
    "index_type": "IVF_SQ8",
    "metric_type": "L2",
    "params": {"nlist": 1024},
}

# IVF_PQ (inverted file + product quantization)
index_params = {
    "index_type": "IVF_PQ",
    "metric_type": "L2",
    "params": {"nlist": 1024, "m": 16, "nbits": 8},
}

# HNSW (hierarchical navigable small world)
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200},
}

# DISKANN (SSD-optimized, billion-scale)
index_params = {
    "index_type": "DISKANN",
    "metric_type": "L2",
    "params": {},
}

# Binary indexes
index_params = {
    "index_type": "BIN_FLAT",  # For binary vectors
    "metric_type": "HAMMING",
}

index_params = {
    "index_type": "BIN_IVF_FLAT",
    "metric_type": "HAMMING",
    "params": {"nlist": 1024},
}
```

**Index Type Decision Guide:**

| Index Type | Dataset Size | Recall | Build Speed | Memory |
|---|---|---|---|---|
| FLAT | <100K | 100% | N/A | Highest |
| IVF_FLAT | 100K-10M | 99% | Fast | High |
| IVF_SQ8 | 1M-50M | 98% | Fast | Medium |
| IVF_PQ | 10M-100M | 95% | Medium | Low |
| HNSW | 100K-10M | 99.5% | Slow | High |
| DISKANN | 100M-1B+ | 95% | Slow | Very Low (SSD) |

### 8.4 GPU Acceleration

Milvus supports GPU-accelerated index building and search via Knowhere:

```python
# GPU index types
index_params = {
    "index_type": "GPU_IVF_FLAT",  # GPU version of IVF_FLAT
    "metric_type": "L2",
    "params": {"nlist": 1024},
}

# GPU HNSW
index_params = {
    "index_type": "GPU_HNSW",
    "metric_type": "L2",
    "params": {"M": 16, "efConstruction": 200},
}

# GPU configuration
import pymilvus
pymilvus.config.GPU_ENABLE = True

# Specify GPU devices
# Set env: MILVUS_GPU_DEVICE_COUNT=1
# Or in milvus.yaml:
# gpu:
#   enable: true
#   cache_capacity_gb: 4
#   search_devices: [0, 1]
#   build_index_devices: [0]
```

### 8.5 Consistency Levels

Milvus offers four consistency levels:

```python
client.insert(
    collection_name="my_collection",
    data=[...],
    consistency_level="Strong",  # or "BoundedStaleness", "Session", "Eventually"
)
```

| Level | Description | Use Case |
|---|---|---|
| **Strong** | Read sees all committed writes immediately | Financial, critical data |
| **BoundedStaleness** | Read allows small latency (default) | Most production workloads |
| **Session** | Read-your-writes consistency | User session data |
| **Eventually** | Fastest, may return stale data | High-throughput logging |

### 8.6 Sharding and Partitions

```python
# Channel-based sharding (automatic)
# Milvus automatically shards based on the message broker partitions

# Partitioning (manual data organization)
client.create_partition(
    collection_name="my_collection",
    partition_name="year_2023",
)

client.create_partition(
    collection_name="my_collection",
    partition_name="year_2024",
)

# Insert into specific partition
client.insert(
    collection_name="my_collection",
    data=[...],
    partition_name="year_2023",
)

# Search in specific partition
results = client.search(
    collection_name="my_collection",
    data=[query_vector],
    limit=10,
    partition_names=["year_2023"],
)
```

### 8.7 Replication

```python
# Milvus uses a distributed architecture with built-in replication
# Configure via milvus.yaml:
# dataCoord:
#   segment:
#     replicationFactor: 3
#   channel:
#     replicationFactor: 3

# Query load balancing (read replicas)
# Configure query nodes:
# queryNode:
#   replication:
#     enabled: true
#     replicaNumber: 3
```

### 8.8 Kafka Integration

Milvus uses Kafka (or Pulsar) as its log broker:

```yaml
# milvus.yaml configuration
kafka:
  brokerList: kafka-broker:9092
  saslUsername: ""
  saslPassword: ""
  saslMechanisms: ""
  securityProtocol: ""
  consumer:
    retryBackoffMs: 100
  producer:
    retryBackoffMs: 100

# Or use Pulsar instead:
pulsar:
  address: pulsar://pulsar:6650
  maxMessageSize: 5242880
```

### 8.9 Milvus Cluster (Kubernetes)

```bash
# Install via Helm
helm repo add milvus https://milvus-io.github.io/milvus-helm/
helm repo update
helm install my-milvus milvus/milvus \
  --set cluster.enabled=true \
  --set etcd.replicaCount=3 \
  --set minio.mode=distributed \
  --set pulsar.enabled=true

# Or using Milvus Operator
kubectl apply -f https://raw.githubusercontent.com/zilliztech/milvus-operator/main/deploy/manifests/deployment.yaml

# MilvusCluster CRD
apiVersion: milvus.io/v1beta1
kind: MilvusCluster
metadata:
  name: my-milvus
spec:
  components:
    proxy:
      replicas: 3
    rootCoord:
      replicas: 1
    dataCoord:
      replicas: 1
    dataNode:
      replicas: 3
    queryCoord:
      replicas: 1
    queryNode:
      replicas: 3
    indexCoord:
      replicas: 1
    indexNode:
      replicas: 2
  dependencies:
    etcd:
      replicas: 3
    minio:
      mode: distributed
      replicas: 4
    pulsar:
      replicas: 3
```

### 8.10 Data Operations

```python
# Insert
client.insert(
    collection_name="my_collection",
    data=[
        {"id": 0, "vector": [0.1, 0.2, ...], "category": "science", "year": 2023},
        {"id": 1, "vector": [0.3, 0.4, ...], "category": "history", "year": 2022},
    ],
)

# Flush (ensure data is persisted)
client.flush(collection_name="my_collection")

# Search with filtering
results = client.search(
    collection_name="my_collection",
    data=[query_vector],
    filter="category == 'science' and year >= 2020",
    limit=10,
    output_fields=["id", "category", "year"],
)

# Query by filter (without vector search)
results = client.query(
    collection_name="my_collection",
    filter="year >= 2020",
    output_fields=["id", "category"],
    limit=100,
)

# Get by ID
result = client.get(
    collection_name="my_collection",
    ids=[0, 1, 5],
)

# Delete
client.delete(
    collection_name="my_collection",
    filter="year < 2020",
)

# Compact (clean up deleted data)
client.compact(collection_name="my_collection")
```

### 8.11 Attu GUI

Attu is the official GUI for Milvus management:

```bash
# Deploy Attu
docker run -d -p 8000:3000 \
  -e MILVUS_URL=http://localhost:19530 \
  zilliz/attu:latest
```

**Attu Features:**
- Collection management (create, modify, drop)
- Data browsing and editing
- Vector search testing
- Index management
- Performance monitoring
- User management (RBAC)

### 8.12 CDC (Change Data Capture)

Milvus supports CDC for streaming data synchronization:

```yaml
# Enable CDC in milvus.yaml
cdc:
  enabled: true
  channel: cdc-channel
```

**CDC Use Cases:**
- Real-time data replication to another Milvus instance
- Stream data to data lakes (S3, HDFS)
- Trigger downstream processing pipelines
- Audit logging of all data changes

### 8.13 2.x vs 1.x Migration

Key differences between Milvus 2.x and 1.x:

| Feature | Milvus 1.x | Milvus 2.x |
|---|---|---|
| Architecture | Mishards (sharding proxy) | Cloud-native (coord + worker) |
| Storage | Local SSD | Object storage (S3, MinIO) |
| Log broker | Internal | Kafka / Pulsar |
| SDK | Limited | Python, Java, Go, Node.js, REST |
| GPU support | Limited | Full GPU acceleration |
| Index types | IVF, HNSW | IVF, HNSW, DISKANN, Binary, GPU |
| Consistency | Eventual | Strong, BoundedStaleness, Session, Eventually |
| Schema | Dynamic fields | Strongly typed + dynamic fields |
| Multi-tenancy | By collection | By collection with RBAC |
| Backup | Manual | Built-in backup/restore |

**Migration Tool:**
```bash
# Use milvus-migration tool
docker run -d -p 8080:8080 \
  -e MILVUS_1X_HOST=localhost \
  -e MILVUS_1X_PORT=19530 \
  -e MILVUS_2X_HOST=localhost \
  -e MILVUS_2X_PORT=19530 \
  zilliz/milvus-migration:latest
```

---

## 9. LanceDB

LanceDB is an embedded vector database (like SQLite for vectors) built in Rust with a columnar storage format optimized for ML workflows.

### 9.1 Architecture

```
Client (Python / Rust / JS / Swift)
    ↕
LanceDB Embedded
    ├── Lance Columnar Format
    │   ├── Vector Columns
    │   ├── Scalar Columns
    │   └── Metadata
    ├── ANN Index (IVF + PQ)
    └── GPU-Accelerated Search (optional)
```

### 9.2 Embedded Database

```python
import lancedb

# Open local database (creates if not exists)
db = lancedb.connect("./lancedb_data")

# Create table from data
db.create_table(
    "my_table",
    data=[
        {"vector": [0.1, 0.2, ...], "text": "doc1", "id": 1},
        {"vector": [0.3, 0.4, ...], "text": "doc2", "id": 2},
    ],
)

# Or create table with schema
import pyarrow as pa

schema = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), list_size=768)),
    pa.field("text", pa.string()),
    pa.field("year", pa.int32()),
])

db.create_table("structured_table", schema=schema)

# Open existing table
table = db.open_table("my_table")
```

### 9.3 Columnar Storage

Lance format is columnar (like Parquet) with optimizations for ML:

```python
# Add data efficiently (column-oriented insert)
table.add([
    {"vector": [0.5, 0.6, ...], "text": "doc3", "year": 2023},
    {"vector": [0.7, 0.8, ...], "text": "doc4", "year": 2024},
])

# Query with column selection
results = table.search(query_vector).select(["text", "year"]).to_pandas()

# Streaming reads (for large datasets)
for batch in table.to_batches():
    print(batch.num_rows)
```

**Advantages of Columnar Storage:**
- Only reads required columns (reduces I/O)
- High compression ratios (LZ4, ZSTD)
- Predicate pushdown (filter before loading)
- Zero-copy reads (memory-mapped)
- Efficient partial updates

### 9.4 GPU-Accelerated ANN

```python
# Create IVF_PQ index with GPU support
table.create_index(
    metric="L2",
    num_partitions=256,      # IVF centroids
    num_sub_vectors=96,      # PQ sub-vectors
    index_cache_size=1000,   # Cache size in MB
    accelerator="cuda",      # Use GPU for index building
)

# GPU-accelerated search
results = table.search(query_vector, accelerator="cuda").limit(10).to_pandas()
```

### 9.5 Multi-Modal Support

LanceDB stores any data type supported by Arrow:

```python
# Images (as bytes)
table.add([
    {
        "vector": [0.1, 0.2, ...],
        "image_bytes": open("photo.jpg", "rb").read(),
        "caption": "A photo",
    }
])

# Audio features
table.add([
    {
        "vector": [0.3, 0.4, ...],
        "audio_embedding": audio_emb,
        "transcript": "Hello world",
    }
])
```

### 9.6 Zero-Copy

LanceDB uses memory-mapped files for zero-copy reads:

```python
# Open table with zero-copy (no data loading until accessed)
table = db.open_table("large_table")

# First access triggers page-fault-based loading
results = table.search(query_vector).limit(10).to_pandas()
# Subsequent accesses are cached in OS page cache
```

### 9.7 Disk-Based Index

LanceDB's index is designed for disk-based operation:

```python
# Index stored on disk, loaded lazily
table.create_index(
    metric="L2",
    num_partitions=4096,
    num_sub_vectors=32,
    index_cache_size=2048,  # MB of index to cache in RAM
)

# Memory-efficient for billion-scale datasets
# Typical: 1B vectors × 768D → ~3 TB raw data, ~50 GB index
```

**When to Use LanceDB:**
- Single-process applications (Python scripts, serverless functions)
- Multi-modal data pipelines
- GPU-accelerated workloads on single machine
- Applications needing embedded database (no server dependency)
- Research and experimentation

---

## 10. Elasticsearch

Elasticsearch, traditionally a full-text search engine, now supports dense and sparse vector search. It's often used for hybrid search combining keyword and semantic matching.

### 10.1 Dense Vector Field

```json
PUT /my-index
{
  "mappings": {
    "properties": {
      "title-vector": {
        "type": "dense_vector",
        "dims": 768,
        "index": true,
        "similarity": "cosine",  // "cosine" | "dot_product" | "l1" | "l2" | "hamming"
        "index_options": {
          "type": "hnsw",
          "m": 16,
          "ef_construction": 200
        }
      },
      "content": { "type": "text" },
      "category": { "type": "keyword" },
      "year": { "type": "integer" }
    }
  }
}
```

**Similarity Metrics:**
| Metric | Description | When to Use |
|---|---|---|
| `cosine` | Cosine similarity (1 − cos(θ)) | Text embeddings (default) |
| `dot_product` | 1 − dot product | Normalized vectors |
| `l1` | L1 distance | Sparse or robust embeddings |
| `l2` | Euclidean squared | Standard vector search |
| `hamming` | Hamming distance | Binary vectors |

### 10.2 Vector Search Queries

```python
# kNN search
response = client.search(
    index="my-index",
    knn={
        "field": "title-vector",
        "query_vector": [0.1, 0.2, ...],
        "k": 10,
        "num_candidates": 100,  # How many candidates to consider
    },
)

# Cosine similarity in script_score
response = client.search(
    index="my-index",
    query={
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'title-vector') + 1.0",
                "params": {"query_vector": [0.1, 0.2, ...]},
            },
        }
    },
)
```

### 10.3 Sparse Vector Field

Elasticsearch supports sparse vectors for models like SPLADE:

```json
PUT /my-index
{
  "mappings": {
    "properties": {
      "sparse-vector": {
        "type": "sparse_vector"
      },
      "content": { "type": "text" }
    }
  }
}
```

```python
# Index sparse vector
client.index(
    index="my-index",
    document={
        "sparse-vector": {"1": 0.5, "100": 0.3, "500": 0.2},  # term_id: weight
        "content": "Document text",
    },
)

# Sparse vector query
response = client.search(
    index="my-index",
    query={
        "sparse_vector": {
            "field": "sparse-vector",
            "query_vector": {"1": 0.3, "200": 0.7},
        },
    },
)
```

### 10.4 ELSER (Elastic Learned Sparse EncodeR)

ELSER is Elasticsearch's built-in sparse retrieval model:

```json
PUT /my-index
{
  "mappings": {
    "properties": {
      "content-elser": {
        "type": "sparse_vector"
      },
      "content": { "type": "text" }
    }
  }
}
```

```python
# Deploy ELSER model
client.ml.put_trained_model(
    model_id="elser_model_2",
    input={"field_names": ["text_field"]},
)

# Inference pipeline
client.ingest.put_pipeline(
    id="elser-pipeline",
    processors=[
        {
            "inference": {
                "model_id": ".elser_model_2",
                "target_field": "content-elser",
                "field_map": {"content": "text_field"},
            }
        }
    ],
)

# Reindex with ELSER embeddings
client.reindex(
    source={"index": "my-index-raw"},
    dest={"index": "my-index-elser", "pipeline": "elser-pipeline"},
)
```

### 10.5 Hybrid Query

Elasticsearch's strength is combining vector search with full-text search:

```python
# Hybrid: kNN + BM25
response = client.search(
    index="my-index",
    query={
        "bool": {
            "must": [
                {"match": {"content": "quantum computing"}},  # BM25
            ],
            "filter": [
                {"term": {"category": "science"}},
                {"range": {"year": {"gte": 2020}}},
            ],
        }
    },
    knn={
        "field": "title-vector",
        "query_vector": [0.1, 0.2, ...],
        "k": 10,
        "num_candidates": 100,
        "filter": {
            "term": {"category": "science"},
        },
    },
    # Combine scores: linear combination
    rank={"rrf": {"window_size": 100, "rank_constant": 60}},
)
```

**Reciprocal Rank Fusion (RRF):**
```python
# Combine multiple result sets using RRF
response = client.search(
    index="my-index",
    query={
        "bool": {
            "must": {"match": {"content": "search terms"}},
        },
    },
    knn={
        "field": "title-vector",
        "query_vector": [0.1, 0.2, ...],
        "k": 10,
    },
    rank={
        "rrf": {
            "window_size": 100,      # How many results from each query to consider
            "rank_constant": 60,     # Smoothing factor (higher = less impact of low ranks)
        }
    },
)
```

### 10.6 Inference Pipeline

Elasticsearch supports deploying ML models for inference during indexing and search:

```python
# Deploy a text embedding model
client.ml.put_trained_model(
    model_id="sentence-transformers__all-mpnet-base-v2",
    input={"field_names": ["text_field"]},
)

# Inference pipeline
client.ingest.put_pipeline(
    id="embed-pipeline",
    processors=[
        {
            "inference": {
                "model_id": "sentence-transformers__all-mpnet-base-v2",
                "target_field": "content-embedding",
                "field_map": {"content": "text_field"},
            }
        },
        # Also do text processing
        {"remove": {"field": "text_field", "ignore_missing": True}},
    ],
)

# Use pipeline during indexing
client.index(
    index="my-index",
    document={"content": "Document to embed"},
    pipeline="embed-pipeline",
)
```

### 10.7 Elasticsearch vs Dedicated Vector DBs

| Feature | Elasticsearch | Dedicated Vector DB (Qdrant, Milvus, Pinecone) |
|---|---|
| **Primary strength** | Full-text search + analytics | Vector similarity at scale |
| **Vector search** | Good (kNN + script_score) | Excellent (ANN indexes) |
| **Full-text search** | Excellent (BM25, ELSER) | Limited or via hybrid |
| **Hybrid search** | Excellent (RRF, linear comb.) | Good (alpha-weighted) |
| **Filter performance** | Excellent (inverted index) | Good (payload index) |
| **Analytics** | Aggregations, Kibana | Limited |
| **Scalability (vectors)** | Good up to ~10M | Excellent up to 1B+ |
| **Latency (vector search)** | 5-50ms | 1-10ms |
| **Index build speed** | Slower (HNSW) | Fast (IVF) |
| **Query language** | Query DSL (JSON) | REST + SDK |
| **Ecosystem** | Logstash, Kibana, Beats | Limited |
| **License** | Elastic License (partially open) | Open source (Apache, BSL) |

**When to Use Elasticsearch for Vectors:**
- You already use Elasticsearch for full-text search
- Hybrid search (BM25 + vector) is the primary use case
- You need analytics and aggregation on search results
- Vector dataset is under 10M documents
- Filter-heavy workloads (metadata queries are critical)

**When to Use Dedicated Vector DB:**
- Vector search is the primary or only requirement
- Dataset exceeds 10M vectors
- Sub-5ms latency requirement
- GPU-accelerated search needed
- Billion-scale vector datasets

---

## 11. Vector Database Comparison

### 11.1 Feature Comparison Matrix (20+ Dimensions)

| Feature | Faiss | Chroma | Pinecone | Qdrant | Weaviate | Milvus | LanceDB | Elasticsearch |
|---|---|---|---|---|---|---|---|---|
| **Type** | Library | Embedded/Server | Serverless/Pods | Server | Server | Distributed | Embedded | Server |
| **Language** | C++/Python | Rust | Go | Rust | Go | Go/Java | Rust | Java |
| **License** | MIT | Apache 2.0 | Proprietary | Apache 2.0 | BSD-3 | Apache 2.0 | Apache 2.0 | Elastic |
| **Cloud managed** | No | Chroma Cloud | ✓ | Qdrant Cloud | Weaviate Cloud | Zilliz Cloud | No | Elastic Cloud |
| **Self-hosted** | N/A | ✓ | Pod-based | ✓ | ✓ | ✓ | N/A | ✓ |

| **Performance (1M, 768D)** |
|---|---|---|---|---|---|---|---|
| Index build | Fastest | Fast | Managed | Fast | Medium | Medium | Fast | Slow |
| Query latency (p99) | <5ms | <10ms | <10ms | <5ms | <10ms | <5ms | <5ms | <20ms |
| Recall @10 (HNSW) | 99% | 99% | 99% | 99% | 99% | 99% | 99% | 98% |
| QPS (1M vectors, single) | 1000+ | 500+ | 500+ | 1000+ | 500+ | 2000+ | 500+ | 200+ |

| **Scalability** |
|---|---|---|---|---|---|---|---|
| Max vectors (single node) | 10M | 10M | 5M per pod | 10M+ | 10M+ | 100M+ | 1B+ | 10M |
| Distributed | No | No | ✓ | ✓ | ✓ | ✓ | No | ✓ |
| Sharding | No | No | ✓ | ✓ | Multi-tenant | ✓ | No | ✓ |
| Replication | No | No | ✓ | ✓ | ✓ | ✓ | No | ✓ |

| **Filtering** |
|---|---|---|---|---|---|---|---|
| Metadata filtering | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Geo search | No | No | Limited | ✓ | ✓ | ✓ | No | ✓ |
| Hybrid search | No | No | ✓ | ✓ | ✓ | Limited | No | ✓ |
| Pre-filter optimization | N/A | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| **Vector Features** |
|---|---|---|---|---|---|---|---|
| Multi-vector | No | No | No | ✓ | No | No | No | No |
| Sparse vectors | No | No | ✓ | ✓ | ✓ | No | No | ✓ |
| Binary vectors | ✓ | No | No | ✓ | ✓ | ✓ | No | ✓ |
| Quantization | ✓ | No | No | ✓ | ✓ | ✓ | ✓ | No |
| GPU acceleration | ✓ | No | No | No | No | ✓ | ✓ | No |

| **Operations** |
|---|---|---|---|---|---|---|---|
| Backup/Restore | Manual | No | ✓ | ✓ | ✓ | ✓ | Manual | ✓ |
| Monitoring | No | No | ✓ | ✓ | ✓ | ✓ | No | ✓ |
| RBAC | No | No | ✓ | ✓ | ✓ | ✓ | No | ✓ |
| Audit logging | No | No | ✓ | Limited | ✓ | ✓ | No | ✓ |

| **Ecosystem** |
|---|---|---|---|---|---|---|---|
| Python client | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| JavaScript client | No | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Java client | No | No | ✓ | ✓ | ✓ | ✓ | No | ✓ |
| Go client | No | No | ✓ | ✓ | ✓ | ✓ | No | ✓ |
| REST API | No | ✓ | ✓ | ✓ | ✓ | ✓ | No | ✓ |
| gRPC API | No | No | ✓ | ✓ | No | ✓ | No | No |

### 11.2 Performance Benchmarks

**1M vectors, 768D, Cosine, HNSW (M=16, efConstruction=200):**

| Database | Index Time | Index Size (RAM) | QPS (1 thread) | QPS (8 threads) | p95 Latency | Recall@10 |
|---|---|---|---|---|---|---|
| Faiss (CPU) | 45s | 3.1 GB | 1200 | 8500 | 4ms | 99.2% |
| Chroma | 60s | 3.5 GB | 600 | 4000 | 8ms | 98.5% |
| Qdrant | 50s | 3.2 GB | 1100 | 7800 | 5ms | 99.0% |
| Weaviate | 70s | 3.8 GB | 700 | 5000 | 7ms | 98.8% |
| Milvus | 55s | 3.3 GB | 1500 | 10000 | 3ms | 99.3% |
| LanceDB | 40s | 3.0 GB | 800 | 5500 | 6ms | 99.0% |
| Elasticsearch | 120s | 4.0 GB | 300 | 2000 | 15ms | 98.0% |

**10M vectors, 768D, Cosine, IVF_PQ (nlist=4096, m=32):**

| Database | Index Time | Index Size | QPS | Recall@10 |
|---|---|---|---|---|
| Faiss (CPU) | 8 min | 450 MB | 800 | 95% |
| Qdrant (with SQ) | 10 min | 500 MB | 600 | 94% |
| Milvus | 9 min | 480 MB | 1000 | 95% |
| LanceDB | 7 min | 420 MB | 500 | 94% |

**100M vectors, 768D, Cosine, DISKANN/DiskANN-like:**

| Database | Index Time | Storage Size | QPS | Recall@10 |
|---|---|---|---|---|
| Milvus (DISKANN) | 2 hr | 60 GB (SSD) + 4 GB (RAM) | 300 | 96% |
| Qdrant (on_disk) | 2.5 hr | 65 GB (SSD) + 2 GB (RAM) | 200 | 95% |

### 11.3 Cost Comparison

**Estimated Monthly Costs (100M vectors, 768D, 1000 QPS):**

| Database | Hosting | Monthly Cost | Notes |
|---|---|---|---|
| **Pinecone** | Serverless | $5,000-15,000 | Pay per vector + queries |
| **Pinecone** | Pod-based (s1.x32) | $8,000-12,000 | 32 pods at ~$0.50/hr |
| **Qdrant** | Cloud (dedicated) | $2,000-5,000 | 4 nodes at ~$2/hr |
| **Milvus** | Zilliz Cloud | $3,000-8,000 | 3 query nodes, managed |
| **Weaviate** | Weaviate Cloud | $3,000-6,000 | 2-3 nodes |
| **Elasticsearch** | Elastic Cloud | $4,000-10,000 | Hot + warm tier |
| **Self-hosted (Qdrant)** | 4× bare metal | $2,000-4,000 | Hardware + ops |
| **Self-hosted (Milvus)** | 8-node K8s cluster | $3,000-6,000 | K8s + storage + ops |
| **Faiss + custom API** | 1× large VM | $1,000-2,000 | Manual scaling |

### 11.4 Decision Flowchart

```
Q1: Do you need an embedded database (no server)?
    Yes → LanceDB (GPU) or Chroma (simplicity)
    No → Q2

Q2: Is this a fully managed service or self-hosted?
    Managed → Q3
    Self-hosted → Q4

Q3: What's your scale?
    <10M vectors, hybrid search → Elasticsearch (if already using ES)
    <10M vectors, pure vector → Pinecone or Qdrant Cloud
    >10M vectors → Pinecone serverless or Milvus (Zilliz)

Q4: What's your team's technical expertise?
    High → Milvus (complex but powerful) or Faiss + custom infra
    Medium → Qdrant (easy to operate) or Weaviate (GraphQL-native)
    Low → Chroma (simplicity)

Q5: Specific requirements?
    GPU acceleration → Faiss (GPU) or LanceDB (GPU)
    Billion-scale → Milvus (DISKANN) or Qdrant (on_disk)
    Geo-spatial → Qdrant or Elasticsearch
    Sparse + dense hybrid → Elasticsearch or Weaviate
    Multi-modal / multi-vector → Qdrant
    Multi-tenancy at scale → Qdrant or Weaviate
```

### 11.5 Migration Considerations

**When to Switch Databases:**

| Signal | Current DB | Recommended |
|---|---|---|
| Exceeding 10M vectors | Chroma/Weaviate/Small Milvus | Milvus/Qdrant |
| Need GPU acceleration | CPU-only Faiss/Milvus | Faiss GPU/LanceDB/Milvus GPU |
| Latency too high | Elasticsearch/Pinecone | Qdrant/Milvus |
| Cost too high | Managed cloud | Self-hosted Qdrant/Milvus |
| Need hybrid search | FAISS/Chroma | Elasticsearch/Weaviate/Qdrant |
| Need geo-spatial | Pinecone/Chroma | Qdrant/Weaviate |

**Migration Pattern:**
```
1. Export vectors from old system (with IDs + metadata)
2. Validate exported data (count, checksum)
3. Create collection in new system with same schema
4. Bulk import (batch insert with parallelism)
5. Run comparison queries (verify top-K matches)
6. Gradual traffic migration (10% → 50% → 100%)
7. Monitor latency, recall, error rates
8. Decommission old system after validation period
```

---

## Appendices

### A. Glossary

| Term | Definition |
|---|---|
| **ANN** | Approximate Nearest Neighbor — search that trades accuracy for speed |
| **kNN** | k-Nearest Neighbors — exact search |
| **IVF** | Inverted File Index — partitions space into Voronoi cells |
| **HNSW** | Hierarchical Navigable Small World — multi-layer graph index |
| **PQ** | Product Quantization — compresses vectors via sub-vector quantization |
| **SQ** | Scalar Quantization — reduces precision to 8-bit integers |
| **ADC** | Asymmetric Distance Computation — query full-precision, vectors compressed |
| **Voronoi Cell** | Region of space closest to a centroid |
| **nprobe** | Number of IVF cells to search (higher = more accurate) |
| **efSearch** | HNSW search width (higher = more accurate) |
| **efConstruction** | HNSW build width (higher = better index quality) |
| **M** | HNSW connections per node (higher = more accurate) |

### B. Index Selection Quick Reference

```
Dataset Size    | Recall Needed | Memory Budget | Recommended Index
<100K           | Any           | Any           | FLAT (exact)
100K-1M         | 99%+          | High          | HNSW
100K-1M         | 95-99%        | Medium        | IVF_FLAT
1M-10M          | 95-98%        | Low           | IVF_PQ
1M-10M          | 99%+          | High          | HNSW
10M-100M        | 95-98%        | Low-Medium    | IVF_PQ (with SQ)
10M-100M        | 99%+          | High          | HNSW (if RAM available)
100M-1B         | 90-95%        | Very Low      | DISKANN / ON_DISK
100M-1B         | 95-98%        | Low           | IVF_PQ (SSD-cached)
```

### C. Embedding Dimension Impact

| Dimension | Storage (1M vectors, FP32) | HNSW Memory | IVF_Memory (PQ16) |
|---|---|---|---|
| 384 | 1.5 GB | 3.0 GB | 80 MB |
| 768 | 3.1 GB | 4.6 GB | 160 MB |
| 1024 | 4.0 GB | 5.5 GB | 210 MB |
| 1536 | 6.0 GB | 7.5 GB | 310 MB |
| 2048 | 8.0 GB | 9.5 GB | 410 MB |
| 4096 | 16 GB | 17.5 GB | 820 MB |

### D. Known Limitations

| Database | Limitation | Impact |
|---|---|---|
| **Faiss** | No built-in metadata filtering | Must impl. or use pre-filtering |
| **Faiss** | No persistence (library, not DB) | Must build storage layer |
| **Chroma** | No distributed mode in OSS | Single-node only |
| **Chroma** | Limited metadata filter operators | Simple queries only |
| **Pinecone** | Proprietary (vendor lock-in) | Hard to migrate |
| **Pinecone** | Cost at scale | ~10× self-hosted costs |
| **Qdrant** | Smaller community | Fewer integrations |
| **Weaviate** | Resource-heavy (Go runtime) | Higher baseline memory |
| **Milvus** | Complex deployment | High ops overhead |
| **Milvus** | Kafka dependency | Operational complexity |
| **LanceDB** | Single-process only | Not for distributed apps |
| **Elasticsearch** | Vector search not primary use | Higher latency, lower QPS |

---

> **Last Updated:** May 2026
> **Author:** AI Enterprise Knowledge Base
> **License:** Internal use — Enterprise Knowledge Management

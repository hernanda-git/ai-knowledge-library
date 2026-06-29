# AI-Native Databases: Tools, Frameworks, and Implementations

> **Description:** Comprehensive guide to tools, frameworks, libraries, and platforms for AI-native database systems.

---

## Open-Source Vector Databases

### Milvus
```python
from pymilvus import connections, Collection
connections.connect("default", host="localhost", port="19530")
collection = Collection("documents")
results = collection.search(data=[query_embedding], anns_field="embedding", limit=10)
```

### Qdrant
```python
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
client.upsert(collection_name="documents", points=[PointStruct(id=1, vector=embedding, payload={"text": "Doc"})])
results = client.search(collection_name="documents", query_vector=query_embedding, limit=10)
```

### Weaviate
```python
import weaviate
client = weaviate.connect_to_local()
collection = client.collections.get("Documents")
results = collection.query.hybrid(query="machine learning", alpha=0.7, limit=10)
```

### Chroma
```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("documents")
collection.add(documents=["Doc 1"], ids=["doc1"])
results = collection.query(query_texts=["machine learning"], n_results=5)
```

### LanceDB
```python
import lancedb
db = lancedb.connect("~/.lancedb")
table = db.create_table("documents", data=[...])
results = table.search(query_embedding).limit(10).to_pandas()
```

---

## Managed Cloud Services

### Pinecone
```python
from pinecone import Pinecone
pc = Pinecone(api_key="your-key")
index = pc.Index("documents")
index.upsert(vectors=[{"id": "doc1", "values": embedding}])
results = index.query(vector=query_embedding, top_k=10)
```

### Weaviate Cloud
```python
client = weaviate.connect_to_weaviate_cloud(cluster_url="...", auth_credentials=AuthApiKey("key"))
collection = client.collections.get("Documents")
results = collection.query.near_text(query="machine learning", limit=10)
```

---

## Embedding Frameworks

### Sentence Transformers
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["Document 1", "Document 2"])
```

### OpenAI Embeddings
```python
from openai import OpenAI
client = OpenAI()
response = client.embeddings.create(model="text-embedding-3-large", input=["Doc 1", "Doc 2"])
```

---

## Integration Libraries

### LangChain
```python
from langchain_community.vectorstores import Qdrant
vectorstore = Qdrant.from_documents(chunks, embeddings, url="http://localhost:6333", collection_name="docs")
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})
```

### LlamaIndex
```python
from llama_index.vector_stores.qdrant import QdrantVectorStore
vector_store = QdrantVectorStore(url="http://localhost:6333", collection_name="docs")
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
```

---

## Deployment Patterns

### Local Docker
```bash
docker-compose up -d  # Qdrant + Redis + PostgreSQL
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qdrant
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:v1.8.0
```

### Serverless
```python
from pinecone import Pinecone
pc = Pinecone(api_key="your-key")
index = pc.Index("documents")
```

---

*Last updated: June 29, 2026*

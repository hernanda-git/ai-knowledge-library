# 09 — E-Commerce Recommendation Engine

## Case Study: Two-Tower Neural Network + Real-Time ANN Retrieval for E-Commerce

| Metadata | Value |
|----------|-------|
| **Industry** | E-commerce / Retail |
| **Domain** | Recommendation systems, personalization |
| **Difficulty** | Intermediate |
| **Est. Timeline** | 6-10 weeks |
| **Team Size** | 4-6 engineers (2 ML, 1 backend, 1 data, 1 frontend, 1 PM) |

---

## 🎯 Problem Statement

### Business Context

**Company:** ShopStream (online fashion retailer, 8M monthly active users, 500K SKUs)
**Current System:** Popularity-based + collaborative filtering (ALS), 5-year-old implementation
**Revenue:** $350M/year, 35% from recommendations

### Pain Points

1. **Cold Start Problem** — New products take 2-4 weeks to get recommended (no interaction history)
2. **Popularity Bias** — Tail products (constituting 60% of catalog) get only 15% of recommendation impressions
3. **Stale Recommendations** — ALS model retrained weekly; users see same items for days
4. **Session Context Missing** — Recommendations don't consider current browsing session (what user just looked at)
5. **Cross-sell Failure** — "Complete the look" recommendations (shirt + pants + shoes) poorly handled
6. **Real-Time Scaling** — Must serve 50K requests/second during Black Friday peak

### Success Criteria

| Metric | Target | Baseline |
|--------|--------|----------|
| **CTR (Click-Through Rate)** | +15% | 3.2% |
| **Revenue per Visitor** | +10% | $4.50 |
| **Conversion Rate** | +12% | 2.8% |
| **Diversity (tail coverage)** | +50% | 15% |
| **Cold Start Time** | < 1 hour | 2-4 weeks |
| **P99 Latency** | < 200ms | 350ms (batch) |
| **Training Time** | < 4 hours | 24 hours |

---

## 🏗️ Solution Architecture

### High-Level System

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION LAYER                                         │
│                                                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │  User Events │  │  Product     │  │  Orders      │  │  Reviews     │               │
│  │  (clicks,    │  │  Catalog     │  │  (purchases, │  │  (ratings,   │               │
│  │  views, cart)│  │  (metadata)  │  │  returns)    │  │  text)       │               │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘               │
│         │                 │                  │                 │                       │
│         ▼                 ▼                  ▼                 ▼                       │
│  ┌──────────────────────────────────────────────────────────────────────────────┐     │
│  │  Real-time Stream (Kafka)                    Batch Pipeline (Spark + Airflow)  │     │
│  │  - Click events → session builder           - Daily feature computation        │     │
│  │  - Cart adds → real-time embeddings         - Negative sampling                │     │
│  │  - Purchases → immediate feedback           - Training data generation         │     │
│  └────────────────────────────────┬─────────────────────────────────────────────┘     │
│                                   │                                                   │
└───────────────────────────────────┼───────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼───────────────────────────────────────────────────┐
│                    MODEL TRAINING & FEATURE LAYER                                     │
│                                    │                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────────┐     │
│  │  TWO-TOWER MODEL (TensorFlow)                                                │     │
│  │                                                                               │     │
│  │  ┌──────────────────────┐          ┌──────────────────────┐                   │     │
│  │  │  USER TOWER           │          │  ITEM TOWER           │                  │     │
│  │  │                       │          │                       │                  │     │
│  │  │  User features:       │          │  Item features:       │                  │     │
│  │  │  - User ID embedding  │          │  - Item ID embedding  │                  │     │
│  │  │  - Past categories    │          │  - Category embedding │                  │     │
│  │  │  - Price range pref   │          │  - Price, brand      │                  │     │
│  │  │  - Session context    │          │  - Image embedding    │                  │     │
│  │  │  - Time features      │          │  - Text embedding    │                  │     │
│  │  │                       │          │                       │                  │     │
│  │  │  MLP (256→128→64)     │          │  MLP (256→128→64)     │                  │     │
│  │  └──────────┬────────────┘          └──────────┬────────────┘                  │     │
│  │             │                                  │                               │     │
│  │             ▼                                  ▼                               │     │
│  │         ┌──────────────────────────────────────────────┐                       │     │
│  │         │  DOT PRODUCT → user_vector · item_vector      │                       │     │
│  │         │  Loss: Sampled Softmax + Cross-Entropy       │                       │     │
│  │         └──────────────────────────────────────────────┘                       │     │
│  └────────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────┐     │
│  │  FEATURE STORE (Feast + Redis)                                                │     │
│  │  - User features (online: Redis, offline: PostgreSQL)                          │     │
│  │  - Item features (daily batch + real-time updates)                             │     │
│  │  - Embedding cache (user vectors → Redis for fast retrieval)                   │     │
│  └────────────────────────────────────────────────────────────────────────────────┘     │
└───────────────────────────────────┬──────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼──────────────────────────────────────────────────────┐
│                        SERVING LAYER                                                     │
│                                    ▼                                                     │
│  ┌──────────────────────────────────────────────────────────────────────────────┐       │
│  │                    REAL-TIME RECOMMENDATION SERVICE                          │       │
│  │                                                                              │       │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │       │
│  │  │  User        │  │  ANN         │  │  Ranking     │  │  Filtering   │     │       │
│  │  │  Embedding   │──▶│  Candidate   │──▶│  DeepFM     │──▶│  (bought,    │     │       │
│  │  │  (on-the-fly)│  │  Retrieval   │  │  (re-rank)  │  │   out-of-stk)│     │       │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────┬───────┘     │       │
│  │                                                                 │             │       │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │             │       │
│  │  │  Context     │  │  Diversity   │  │  Business    │          │             │       │
│  │  │  Boosting    │──▶│  Adjustment  │──▶│  Rules       │─────────┘             │       │
│  │  │  (session)   │  │  (MMR algo)  │  │  (promo,    │                        │       │
│  │  │              │  │              │  │  inventory) │                        │       │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                        │       │
│  └──────────────────────────────────┬───────────────────────────────────────────┘       │
│                                     │                                                   │
│                                     ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────┐       │
│  │  ANN INDEX (FAISS) — 500K item embeddings → HNSW index, rebuild daily        │       │
│  │  Top-200 candidates retrieved in < 15ms                                      │       │
│  └──────────────────────────────────────────────────────────────────────────────┘       │
└───────────────────────────────────┬──────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼──────────────────────────────────────────────────────┐
│                        CONSUMER TOUCHPOINTS                                              │
│                                    ▼                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │  Home Page   │  │  Product     │  │  Cart Page   │  │  Search      │                 │
│  │  (personalized)│  │  Detail Page │  │  (cross-sell)  │  (personalized)│                 │
│  │              │  │  ("You may   │  │              │  │  results)    │                 │
│  │              │  │   also like")│  │              │  │              │                 │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘                 │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Cold Start Strategy

```
Cold Start Item Flow:
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│  New Item   │    │  Content-  │    │  Explore   │    │  Collect   │
│  Created    │───▶│  Based     │───▶│  (5% of    │───▶│  Feedback  │
│             │    │  Embedding │    │  traffic)  │    │  (clicks)  │
└────────────┘    └────────────┘    └────────────┘    └────────────┘
                        │                                      │
                        ▼                                      ▼
               ┌────────────────┐                   ┌────────────────┐
               │  Cold-Start    │                   │  Warm Item     │
               │  Recommend via │                   │  (added to     │
               │  similar items │                   │  main index)   │
               └────────────────┘                   └────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| **Deep Learning** | TensorFlow + Keras | 2.15 | TF ecosystem, TF Serving |
| **ANN Index** | FAISS (Facebook AI Similarity Search) | 1.8 | HNSW, GPU-accelerated |
| **Re-ranking** | DeepFM (TensorFlow) | Custom | Feature interactions |
| **Feature Store** | Feast + Redis (online) / PostgreSQL (offline) | 0.37 / 7.x | Low-latency features |
| **Streaming** | Apache Kafka + Flink | 3.6 / 1.18 | Real-time event processing |
| **Orchestration** | Airflow | 2.8 | Training pipeline scheduling |
| **Model Serving** | TF Serving + BentoML | 2.15 / 1.2 | High-throughput, GPU |
| **AB Testing** | Custom framework (statistical) | — | Online experimentation |
| **Backend API** | FastAPI | 0.111 | Async, high performance |
| **Database** | PostgreSQL + Redis + S3 | — | Feature storage |
| **Monitoring** | Prometheus + Grafana + WhyLabs | — | Drift, performance |

### Installation

```bash
pip install tensorflow==2.15.1 tensorflow-serving-api==2.15.0
pip install faiss-gpu==1.8.0  # or faiss-cpu
pip install feast==0.37.1 redis==5.0.8
pip install kafka-python==2.0.2 apache-flink==1.18.1
pip install fastapi==0.111.1 bentoml==1.2.12
pip install scikit-learn==1.5.0 whylabs==0.1.12
```

---

## ⚙️ Implementation Details

### 1. Two-Tower Model

```python
# src/models/two_tower.py
import tensorflow as tf
from tensorflow import keras
from typing import List, Optional

class TwoTowerRecommender(keras.Model):
    """Two-tower neural network for e-commerce recommendation.

    User tower and Item tower independently compute embeddings.
    Dot product gives relevance score.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_categories: int = 100,
        num_brands: int = 2000,
    ):
        super().__init__()

        # Shared embedding dimension
        self.embedding_dim = embedding_dim

        # User Tower
        self.user_id_embedding = keras.layers.Embedding(
            num_users, embedding_dim, name="user_embedding"
        )
        self.user_mlp = keras.Sequential([
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
        ], name="user_mlp")

        # Item Tower
        self.item_id_embedding = keras.layers.Embedding(
            num_items, embedding_dim, name="item_embedding"
        )
        self.category_embedding = keras.layers.Embedding(
            num_categories, 32, name="category_embedding"
        )
        self.brand_embedding = keras.layers.Embedding(
            num_brands, 32, name="brand_embedding"
        )
        self.item_mlp = keras.Sequential([
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
        ], name="item_mlp")

    def call(self, inputs, training=False):
        """Forward pass with dot product scoring."""
        user_inputs, item_inputs = inputs

        # User tower
        user_emb = self.user_id_embedding(user_inputs["user_id"])
        user_features = tf.concat([
            user_emb,
            user_inputs.get("user_side_features", tf.zeros_like(user_emb)),
        ], axis=-1)
        user_vec = self.user_mlp(user_features)

        # Item tower
        item_emb = self.item_id_embedding(item_inputs["item_id"])
        cat_emb = self.category_embedding(item_inputs["category_id"])
        brand_emb = self.brand_embedding(item_inputs["brand_id"])
        price_norm = tf.expand_dims(
            item_inputs["price_normalized"], axis=-1
        )
        item_features = tf.concat([
            item_emb, cat_emb, brand_emb, price_norm
        ], axis=-1)
        item_vec = self.item_mlp(item_features)

        # Dot product score
        scores = tf.reduce_sum(
            user_vec * item_vec, axis=-1, keepdims=True
        )
        return scores

    def get_user_vector(self, user_inputs: dict) -> tf.Tensor:
        """Compute user embedding for retrieval."""
        user_emb = self.user_id_embedding(user_inputs["user_id"])
        user_features = tf.concat([
            user_emb,
            user_inputs.get("user_side_features", tf.zeros_like(user_emb)),
        ], axis=-1)
        return self.user_mlp(user_features)

    def get_item_vector(self, item_inputs: dict) -> tf.Tensor:
        """Compute item embedding for indexing."""
        item_emb = self.item_id_embedding(item_inputs["item_id"])
        cat_emb = self.category_embedding(item_inputs["category_id"])
        brand_emb = self.brand_embedding(item_inputs["brand_id"])
        price_norm = tf.expand_dims(
            item_inputs["price_normalized"], axis=-1
        )
        item_features = tf.concat([item_emb, cat_emb, brand_emb, price_norm], axis=-1)
        return self.item_mlp(item_features)


class SampledSoftmaxLoss(keras.losses.Loss):
    """Sampled softmax loss for efficient training with large item catalogs."""

    def __init__(self, num_items: int, num_samples: int = 500):
        super().__init__()
        self.num_items = num_items
        self.num_samples = num_samples

    def call(self, y_true, y_pred):
        # y_true: positive item indices (batch_size,)
        # y_pred: logits from dot product (batch_size, 1)
        # In practice: use tf.nn.sampled_softmax_loss with candidate sampling
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y_true, logits=y_pred
            )
        )
```

### 2. ANN Candidate Retrieval (FAISS)

```python
# src/retrieval/ann_index.py
import faiss
import numpy as np
from typing import List, Tuple, Optional
import pickle
import os

class ANNIndex:
    """FAISS-based Approximate Nearest Neighbors for item retrieval.

    Builds an HNSW (Hierarchical Navigable Small World) index
    from item embeddings for ultra-fast candidate retrieval.
    """

    def __init__(self, embedding_dim: int = 64, index_path: str = "./data/faiss_index"):
        self.dim = embedding_dim
        self.index_path = index_path
        self.index = None
        self.item_ids = []

    def build_index(self, item_embeddings: np.ndarray, item_ids: List[str]):
        """Build HNSW index from item embeddings."""
        self.item_ids = item_ids
        n_items = len(item_ids)

        # HNSW parameters
        M = 32  # Number of connections per node
        ef_construction = 200  # Build quality vs speed

        self.index = faiss.IndexHNSWFlat(self.dim, M)
        self.index.hnsw.efConstruction = ef_construction

        # Add vectors
        self.index.add(item_embeddings.astype(np.float32))
        print(f"Built HNSW index with {n_items} items (dim={self.dim})")

    def save(self):
        """Save index to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        with open(f"{self.index_path}_ids.pkl", "wb") as f:
            pickle.dump(self.item_ids, f)

    def load(self):
        """Load index from disk."""
        self.index = faiss.read_index(f"{self.index_path}.faiss")
        with open(f"{self.index_path}_ids.pkl", "rb") as f:
            self.item_ids = pickle.load(f)
        print(f"Loaded index with {len(self.item_ids)} items")

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 200,
        ef_search: int = 100,
    ) -> Tuple[List[str], List[float]]:
        """Retrieve top-k nearest neighbors."""
        self.index.hnsw.efSearch = ef_search
        distances, indices = self.index.search(
            query_vector.reshape(1, -1).astype(np.float32), k
        )

        # Map indices to item IDs
        retrieved_ids = [self.item_ids[idx] for idx in indices[0]]
        scores = [1.0 / (1.0 + d) for d in distances[0]]  # Convert to similarity

        return retrieved_ids, scores

    def update_item(self, item_id: str, embedding: np.ndarray):
        """Add or update a single item in the index.
        Note: FAISS HNSW doesn't support deletion efficiently.
        For real-time updates, maintain a secondary index or rebuild periodically.
        """
        idx = len(self.item_ids)
        self.item_ids.append(item_id)
        emb = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(emb)
        return idx
```

### 3. DeepFM Re-ranker

```python
# src/ranking/deepfm.py
import tensorflow as tf
from tensorflow import keras

class DeepFM(keras.Model):
    """Deep Factorization Machine for re-ranking candidates.

    Combines FM (linear + pairwise interactions) with deep MLP.
    """

    def __init__(
        self,
        num_features: int,
        embedding_dim: int = 8,
        deep_layers: List[int] = [64, 32],
    ):
        super().__init__()

        # FM Component
        self.feature_embeddings = keras.layers.Embedding(
            num_features, embedding_dim, name="fm_feature_embedding"
        )
        self.fm_linear = keras.layers.Dense(1, name="fm_linear")

        # Deep Component
        self.deep_network = keras.Sequential([
            keras.layers.Dense(units, activation="relu")
            for units in deep_layers
        ], name="deep_network")

        # Final prediction
        self.final_layer = keras.layers.Dense(
            1, activation="sigmoid", name="final"
        )

    def call(self, inputs, training=False):
        # Inputs: (batch, num_features) — feature index list
        feature_emb = self.feature_embeddings(inputs)  # (B, F, D)

        # FM: pairwise interactions
        sum_squared = tf.square(tf.reduce_sum(feature_emb, axis=1))
        squared_sum = tf.reduce_sum(tf.square(feature_emb), axis=1)
        fm_interactions = 0.5 * tf.reduce_sum(sum_squared - squared_sum, axis=1, keepdims=True)

        # FM: linear term
        fm_linear = self.fm_linear(tf.cast(inputs, tf.float32))

        # Deep component
        flat_emb = tf.reshape(feature_emb, (tf.shape(inputs)[0], -1))
        deep_out = self.deep_network(flat_emb)

        # Combine
        combined = tf.concat([fm_linear, fm_interactions, deep_out], axis=-1)
        return self.final_layer(combined)
```

### 4. Real-Time Serving API

```python
# src/serving/recommendation_service.py
import numpy as np
from typing import List, Dict, Optional
import redis
import json
import time

class RecommendationService:
    """Main recommendation serving class.

    Pipeline: get user embedding → ANN search → re-rank → filter → return.
    """

    def __init__(
        self,
        model: TwoTowerRecommender,
        ann_index: ANNIndex,
        reranker: DeepFM,
        redis_client: redis.Redis,
        feature_store,
    ):
        self.model = model
        self.ann_index = ann_index
        self.reranker = reranker
        self.redis = redis_client
        self.feature_store = feature_store

    def recommend(
        self,
        user_id: str,
        session_id: str,
        context: Optional[Dict] = None,
        top_k: int = 20,
        exclude_item_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Generate personalized recommendations for a user."""

        start_time = time.time()
        timing = {}

        # Step 1: Get or compute user embedding
        user_features = self._get_user_features(user_id, session_id)
        user_vec = self.model.get_user_vector(user_features)
        timing["embedding"] = (time.time() - start_time) * 1000

        # Step 2: ANN candidate retrieval
        candidate_ids, candidate_scores = self.ann_index.search(
            user_vec.numpy().flatten(), k=200
        )
        timing["ann_search"] = (time.time() - start_time) * 1000 - timing["embedding"]

        # Step 3: Filter (exclude viewed/purchased)
        exclude = set(exclude_item_ids or [])
        exclude.update(self._get_user_history(user_id))
        candidates = [
            item_id for item_id in candidate_ids
            if item_id not in exclude
        ][:100]  # Keep top-100 for re-ranking

        # Step 4: Re-rank with DeepFM
        if self.reranker and candidates:
            ranking_features = self._build_ranking_features(
                user_id, candidates, context
            )
            scores = self.reranker.predict(ranking_features)
            scored_candidates = list(zip(candidates, scores.flatten()))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            ranked = scored_candidates[:top_k]
        else:
            # Fallback to ANN scores
            ranked = list(zip(candidates, [1.0] * len(candidates)))[:top_k]

        timing["rerank"] = (time.time() - start_time) * 1000 - sum(v for v in timing.values())

        # Step 5: Build response
        results = []
        for item_id, score in ranked:
            item_info = self._get_item_info(item_id)
            results.append({
                "item_id": item_id,
                "score": float(score),
                "title": item_info.get("title", ""),
                "price": item_info.get("price", 0),
                "image_url": item_info.get("image_url", ""),
                "category": item_info.get("category", ""),
            })

        total_time = (time.time() - start_time) * 1000
        return {
            "user_id": user_id,
            "recommendations": results,
            "total_count": len(results),
            "timing_ms": timing,
            "total_time_ms": total_time,
        }

    def _get_user_features(self, user_id: str, session_id: str) -> dict:
        """Build user feature dictionary for model inference."""
        # Get from Redis (online feature store)
        recent_categories = self.redis.lrange(
            f"user:{user_id}:recent_categories", 0, 4
        )
        price_pref = float(self.redis.get(f"user:{user_id}:avg_price") or 0.5)

        return {
            "user_id": tf.constant([int(user_id)]),
            "user_side_features": tf.constant([
                [float(len(recent_categories)), price_pref]
            ]),
        }

    def _build_ranking_features(
        self, user_id: str, item_ids: List[str], context: Optional[Dict]
    ) -> np.ndarray:
        """Build feature matrix for DeepFM re-ranker."""
        features = []
        for item_id in item_ids:
            feat = self.feature_store.get_ranking_features(user_id, item_id, context)
            features.append(feat)
        return np.array(features)

    def _get_user_history(self, user_id: str) -> List[str]:
        """Get items user has already purchased/viewed."""
        return [
            id.decode() for id in self.redis.smembers(
                f"user:{user_id}:purchased"
            )
        ]

    def _get_item_info(self, item_id: str) -> dict:
        """Get item metadata from cache."""
        data = self.redis.get(f"item:{item_id}:info")
        if data:
            return json.loads(data)
        return {}
```

### 5. A/B Testing Framework

```python
# src/experimentation/ab_test.py
import hashlib
import random
from typing import Dict, List, Optional

class ABTestFramework:
    """Statistical A/B testing for recommendation models."""

    def __init__(self, redis_client):
        self.redis = redis_client

    def get_assignment(
        self, user_id: str, experiment_name: str, variants: List[str]
    ) -> str:
        """Deterministic assignment of user to experiment variant."""
        hash_key = f"{experiment_name}:{user_id}"
        hash_val = int(hashlib.md5(hash_key.encode()).hexdigest(), 16)
        idx = hash_val % len(variants)
        return variants[idx]

    def is_treatment(self, user_id: str, experiment_name: str) -> bool:
        """Check if user is in treatment group."""
        return self.get_assignment(user_id, experiment_name, ["control", "treatment"]) == "treatment"

    def log_event(
        self,
        user_id: str,
        experiment_name: str,
        event_type: str,
        properties: Dict,
    ):
        """Log experiment event for analysis."""
        event = {
            "user_id": user_id,
            "experiment": experiment_name,
            "variant": self.get_assignment(
                user_id, experiment_name, ["control", "treatment"]
            ),
            "event_type": event_type,
            "properties": properties,
        }
        self.redis.xadd("ab_events", event)  # Redis stream for Kafka later

    def get_results(
        self, experiment_name: str
    ) -> Dict:
        """Compute statistical results for experiment."""
        # In production: query from data warehouse
        # This is a simplified example
        return {
            "experiment": experiment_name,
            "control_ctr": 0.032,
            "treatment_ctr": 0.038,
            "uplift": 0.187,
            "p_value": 0.003,
            "significant": True,
            "recommendation": "Launch treatment to 100%",
        }
```

### 6. Training Pipeline

```python
# src/training/train_pipeline.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
from src.models.two_tower import TwoTowerRecommender, SampledSoftmaxLoss

class TrainPipeline:
    """End-to-end training pipeline for two-tower model."""

    def __init__(self, config: dict):
        self.config = config
        self.model = None

    def build_model(self):
        self.model = TwoTowerRecommender(
            num_users=self.config["num_users"],
            num_items=self.config["num_items"],
            embedding_dim=self.config.get("embedding_dim", 64),
        )

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int = 10,
    ):
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=1e-3, decay=1e-5
            ),
            loss=SampledSoftmaxLoss(
                num_items=self.config["num_items"]
            ),
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2
            ),
            keras.callbacks.TensorBoard(log_dir="./logs"),
        ]

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    def generate_item_embeddings(self, item_dataset) -> np.ndarray:
        """Generate all item embeddings for ANN index building."""
        embeddings = []
        for batch in item_dataset:
            item_vecs = self.model.get_item_vector(batch)
            embeddings.append(item_vecs.numpy())
        return np.concatenate(embeddings, axis=0)

    def export_for_serving(self, export_path: str):
        """Export model for TF Serving."""
        self.model.save(export_path)
        print(f"Model exported to {export_path}")
```

---

## 📊 Metrics & Results

### Online A/B Test Results (4-week experiment)

| Metric | Control (ALS) | Treatment (Two-Tower) | Delta | Statistical Significance |
|--------|--------------|----------------------|-------|--------------------------|
| **CTR** | 3.2% | 4.0% | **+25%** | p < 0.001 |
| **Revenue per Visitor** | $4.50 | $5.17 | **+15%** | p < 0.001 |
| **Conversion Rate** | 2.8% | 3.2% | **+14.3%** | p = 0.003 |
| **Revenue per Session** | $12.40 | $14.55 | **+17.3%** | p < 0.001 |
| **Add-to-Cart Rate** | 8.5% | 10.2% | **+20%** | p = 0.002 |

### Offline Metrics

| Metric | ALS | Two-Tower | Two-Tower + DeepFM | Improvement |
|--------|-----|-----------|-------------------|-------------|
| **Recall@20** | 0.34 | 0.52 | 0.61 | +79% |
| **NDCG@20** | 0.28 | 0.44 | 0.53 | +89% |
| **Hit Rate@10** | 0.42 | 0.58 | 0.65 | +55% |
| **Diversity** | 0.15 | 0.32 | 0.30 | +100% |

### System Performance

| Metric | Old System | New System | Delta |
|--------|-----------|------------|-------|
| **P50 Latency** | 120ms | 45ms | -62.5% |
| **P99 Latency** | 350ms | 185ms | -47% |
| **Throughput** | 15K req/s | 80K req/s | +433% |
| **Model Training** | 24 hours | 3.5 hours | -85% |
| **Index Build** | N/A | 8 minutes | N/A |
| **Cold Start (items)** | 2-4 weeks | < 1 hour | -99.7% |

### Business Impact

| Metric | Before | After | Annual Impact |
|--------|--------|-------|--------------|
| **Revenue from Recs** | $122.5M | $152.3M | +$29.8M |
| **Average Order Value** | $48 | $54 | +12.5% |
| **Return Rate** | 12% | 10.5% | -1.5 pp |
| **Customer Lifetime Value** | $420 | $510 | +21.4% |
| **Tail Item Impressions** | 15% | 38% | +23 pp |

---

## 💡 Lessons Learned

### ✅ What Went Well

1. **Two-tower + DeepFM re-ranking** — Candidate retrieval with two-tower (fast, 200 candidates in 15ms) + DeepFM re-ranking (accurate, better feature interactions) was the optimal architecture.

2. **FAISS HNSW is fast enough** — With 500K items, HNSW search is 15ms P99. No need for sharding at this scale.

3. **Session context boosted CTR by 8%** — Simply adding "last clicked category" to user features significantly improved relevance.

4. **Cold start with content embeddings** — Using product image embeddings (ResNet) + text embeddings (BERT) for cold items gave 0.42 recall@20 from hour 1.

### ❌ What Went Wrong

1. **Initial two-tower had poor diversity** — Pure dot-product optimization led to recommending same categories repeatedly. Fixed by adding MMR (Maximum Marginal Relevance) post-processing.

2. **Real-time embedding computation was slow** — Originally recomputed user embedding every request (35ms). Switched to caching user embeddings in Redis with 5-minute TTL, refresh on new interaction.

3. **Negative sampling matters enormously** — Random negative sampling gave poor results. Switched to batch negative sampling (in-batch negatives) + hard negative mining.

4. **A/B test early stopping** — Called the experiment after 1 week (p=0.03). At week 4, p was 0.001 but the effect size had shrunk. Lesson: run for minimum 2 weeks.

### ⚠️ Critical Warnings

```
! WARNING: Diversity metrics are as important as CTR — avoid filter bubbles.
! WARNING: A/B test for minimum 2 weeks to capture weekly seasonality.
! WARNING: Cold start strategy is essential for catalog freshness.
! WARNING: Monitor for popularity bias drift — re-calibrate monthly.
! WARNING: Cache invalidation on user events is critical for freshness.
```

---

## 📁 Reusable Project Template

### Directory Structure

```
TEMPLATE-RECOMMENDATION-ENGINE/
├── README.md
├── Makefile
├── requirements.txt
├── docker-compose.yml
├── .env.example
│
├── configs/
│   ├── config.yaml
│   ├── model_config.yaml
│   ├── features.yaml
│   ├── ab_test.yaml
│   └── retrieval_config.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── negative_sampling.py
│   │   ├── session_builder.py
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── two_tower.py
│   │   ├── deepfm.py
│   │   └── cold_start.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── ann_index.py
│   │   ├── user_embedder.py
│   │   └── candidate_fusion.py
│   │
│   ├── ranking/
│   │   ├── __init__.py
│   │   ├── score_fusion.py
│   │   ├── diversity.py
│   │   └── business_rules.py
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── recommendation_service.py
│   │   ├── api.py
│   │   └── schemas.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_pipeline.py
│   │   ├── evaluate.py
│   │   └── export.py
│   │
│   ├── experimentation/
│   │   ├── __init__.py
│   │   ├── ab_test.py
│   │   ├── assignment.py
│   │   └── analysis.py
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── drift_detector.py
│   │   └── alerts.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── metrics_tracker.py
│
├── tests/
│   ├── unit/
│   │   ├── test_two_tower.py
│   │   ├── test_ann_index.py
│   │   ├── test_reranker.py
│   │   └── test_service.py
│   ├── integration/
│   │   ├── test_training.py
│   │   ├── test_serving.py
│   │   └── test_kafka.py
│   └── fixtures/
│       ├── sample_events.json
│       └── sample_catalog.json
│
├── notebooks/
│   ├── 01-eda-user-behavior.ipynb
│   ├── 02-two-tower-training.ipynb
│   ├── 03-deepfm-ranking.ipynb
│   └── 04-ab-test-analysis.ipynb
│
├── scripts/
│   ├── generate_training_data.py
│   ├── train_model.py
│   ├── build_ann_index.py
│   ├── evaluate_model.py
│   ├── export_tf_serving.py
│   └── simulate_traffic.py
│
├── k8s/
│   ├── namespace.yaml
│   ├── deployment-api.yaml
│   ├── deployment-tf-serving.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   └── configmap.yaml
│
└── docs/
    ├── architecture.md
    ├── feature_descriptions.md
    ├── ab_testing_guide.md
    └── model_card.md
```

### Getting Started

```bash
# 1. Copy template
cp -r TEMPLATE-RECOMMENDATION-ENGINE ~/my-recsys
cd ~/my-recsys

# 2. Install dependencies
make install

# 3. Generate sample training data
python scripts/generate_training_data.py --num-users 10000 --num-items 5000

# 4. Train two-tower model
python scripts/train_model.py --config configs/config.yaml

# 5. Build ANN index
python scripts/build_ann_index.py

# 6. Export for TF Serving
python scripts/export_tf_serving.py

# 7. Start serving
docker-compose up -d redis tf-serving api

# 8. Test recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "42", "session_id": "abc123", "top_k": 10}'
```

---

## 📚 References & Further Reading

### Academic Papers
- Yi et al. (2019) — "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations" — [RecSys 2019](https://dl.acm.org/doi/10.1145/3298689.3346996)
- Guo et al. (2017) — "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction" — [arXiv:1703.04247](https://arxiv.org/abs/1703.04247)
- Covington et al. (2016) — "Deep Neural Networks for YouTube Recommendations" — [RecSys 2016](https://dl.acm.org/doi/10.1145/2959100.2959190)
- Carbonell & Goldstein (1998) — "The Use of MMR, Diversity-Based Reranking for Reordering Documents" — [SIGIR 1998]

### Tools & Documentation
- FAISS: https://github.com/facebookresearch/faiss
- TensorFlow Recommenders: https://www.tensorflow.org/recommenders
- TF Serving: https://www.tensorflow.org/tfx/guide/serving
- Feast Feature Store: https://docs.feast.dev/

---

> **Next**: [10-NLP-Sentiment-Analysis.md](10-NLP-Sentiment-Analysis.md) — Customer feedback sentiment analysis with fine-tuned BERT and multi-language support.

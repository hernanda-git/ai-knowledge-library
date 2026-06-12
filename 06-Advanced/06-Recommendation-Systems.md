# Recommendation Systems: Architecture, Algorithms, and Production

## Table of Contents
1. [Introduction](#1-introduction)
2. [Collaborative Filtering](#2-cf)
3. [Content-Based Filtering](#3-content)
4. [Hybrid Systems](#4-hybrid)
5. [Deep Learning for Recommendations](#5-dl)
6. [Feature Engineering](#6-features)
7. [Candidate Generation](#7-candidate)
8. [Ranking](#8-ranking)
9. [Session-Based Recommendations](#9-session)
10. [Cold Start Solutions](#10-cold-start)
11. [Evaluation](#11-evaluation)
12. [Causal Recommendation Systems](#12-causal)
13. [Multi-Task Learning in RecSys](#13-mtl)
14. [Contextual Bandits for Recommendations](#14-bandits)
15. [Industry Case Studies](#15-case-studies)
16. [Emerging Trends (2024-2026)](#16-trends)
17. [Biases and Pitfalls in Evaluation](#17-biases)
18. [Production Considerations](#18-production)
19. [Cross-References](#19-cross-references)

---

## 1. Introduction

Recommendation systems predict user preferences to suggest relevant items. They power Amazon (35% of revenue from recs), Netflix (80% of watch time), TikTok (algorithm-driven feed), Spotify (Discover Weekly), and YouTube (70% of watch time from recommendations).

### Two-Stage Architecture

All large-scale recsys use a **funnel** approach:
1. **Candidate Generation:** Retrieve hundreds from billions (recall-focused)
2. **Ranking:** Score hundreds to pick top tens (precision-focused)
3. **Re-ranking:** Apply business constraints (diversity, freshness, fairness, business rules)

```python
# Conceptual recsys pipeline
class RecSysPipeline:
    def recommend(self, user_id, user_features, n=20):
        candidates = self.candidate_generator.retrieve(user_id, n_candidates=500)
        scored = self.ranker.score(user_id, candidates, user_features)
        reranked = self.reranker.apply_constraints(scored, user_id)
        return reranked[:n]
```

### Key Challenges
| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| **Sparsity** | Most users interact with <1% of items | Matrix factorization, embeddings |
| **Cold start** | New users/items with no history | Content-based signals, exploration |
| **Scalability** | Billions of users × millions of items | Two-stage funnel, ANN indexing |
| **Diversity** | Over-specialization in recommendations | MMR, DPP-based re-ranking |
| **Bias** | Feedback loops, popularity bias | Counterfactual evaluation, debiasing |
| **Freshness** | New items need exposure | Explore-exploit tradeoff |

---

## 2. Collaborative Filtering (CF)

### 2.1 User-User CF
Find similar users, recommend what they liked.

**Similarity metrics:**
- Pearson correlation: r_uv = corr(r_u, r_v)
- Cosine similarity: sim(u,v) = (r_u · r_v) / (||r_u|| · ||r_v||)
- Adjusted cosine: subtract user mean before cosine

**Limitation:** O(n²) user comparison, sparse → poor coverage for long-tail users.

### 2.2 Item-Item CF
Find similar items based on co-interaction patterns.

Amazon's original algorithm: "Customers who bought X also bought Y"

```python
def item_based_cf(ratings_matrix, target_item, k=10):
    """Find k most similar items to target_item"""
    item_similarities = cosine_similarity(ratings_matrix.T)
    similar_items = item_similarities[target_item].argsort()[::-1][1:k+1]
    return similar_items
```

**Advantage:** More stable than user-user CF (item similarities change slowly). Pre-computed offline for fast lookup.

### 2.3 Matrix Factorization (MF)

R ≈ U · V^T where U ∈ R^{u×k}, V ∈ R^{i×k}

Complete model: R̂_ui = µ + b_u + b_i + U_u · V_i

- µ: global average rating
- b_u: user bias (how lenient/critical is the user)
- b_i: item bias (how popular/well-rated is the item)
- U_u · V_i: latent interaction (learned embedding dot product)

**k** is the latent factor dimension (20-200). Larger k → more capacity but overfitting risk.

### 2.4 Funk SVD (Simon Funk, Netflix Prize 2006)
The algorithm that launched modern recsys. SGD on rating prediction:

```python
def funk_svd_fit(ratings, n_factors=20, lr=0.005, reg=0.02, n_epochs=50):
    users, items = np.unique(ratings[:,0]), np.unique(ratings[:,1])
    U = np.random.normal(0, 0.1, (len(users), n_factors))
    V = np.random.normal(0, 0.1, (len(items), n_factors))
    b_u, b_i = np.zeros(len(users)), np.zeros(len(items))
    mu = ratings[:,2].mean()
    
    for epoch in range(n_epochs):
        np.random.shuffle(ratings)
        for u, i, r in ratings:
            pred = mu + b_u[u] + b_i[i] + U[u] @ V[i]
            e = r - pred
            b_u[u] += lr * (e - reg * b_u[u])
            b_i[i] += lr * (e - reg * b_i[i])
            U[u] += lr * (e * V[i] - reg * U[u])
            V[i] += lr * (e * U[u] - reg * V[i])
    return U, V, b_u, b_i, mu
```

### 2.5 Implicit Feedback Models
Most feedback is **implicit** (clicks, views, purchases, time spent), not explicit (ratings).

| Aspect | Explicit (ratings) | Implicit (clicks) |
|--------|:------------------:|:-----------------:|
| Signal | Direct preference | Indirect signal |
| Absence | Dislike | Negative or unaware |
| Confidence | High per datapoint | Low (noisy) |
| Scale | 1-5 stars | Binary or count |
| Model | RMSE, MAE | BPR, WARP, eALS |

**Bayesian Personalized Ranking (BPR):** Optimize pairwise ranking: prefer observed items over unobserved:

max Σ_{(u,i,j)} log σ(R̂_ui - R̂_uj) - λ||θ||²

Where (u,i,j) = user u preferred item i over unobserved item j.

---

## 3. Content-Based Filtering

Recommend items similar to what the user liked in the past, based on item **features**.

**Item Features:**
| Feature Type | Examples | Technique |
|-------------|----------|-----------|
| Text | Title, description, reviews | TF-IDF, sentence embeddings, LLM embeddings |
| Metadata | Category, price, brand, release date | One-hot, bucketized, learned embeddings |
| Images | Product image, movie poster | CNN/ViT embeddings (ResNet, CLIP) |
| Audio | Song previews, podcast audio | Spectrogram → CNN, MuCoCo embeddings |
| Structured | Color, size, material, weight | Numerical encoding, feature crosses |

**User Profile:** Average (or weighted) of features from items the user liked.

```python
def content_based_profile(user_items, item_features_matrix):
    """Create user profile: mean of liked item features"""
    liked_features = item_features_matrix[user_items]
    return liked_features.mean(axis=0)

def content_recommend(profile, item_features, n=10):
    scores = item_features @ profile  # dot product similarity
    top_items = scores.argsort()[::-1][:n]
    return top_items
```

**Pros:** No cold-start for items, transparent recommendations, handles niche interests.
**Cons:** Limited serendipity (never recommends outside user's interest bubble), user cold-start, over-specialization.

---

## 4. Hybrid Systems

Combine CF + content + context for the best of all worlds.

| Hybrid Method | Approach | Benefit |
|---------------|----------|---------|
| **Weighted** | Linear combination of scores | Simple, interpretable |
| **Feature augmentation** | Add CF features as inputs to content model | Best of both worlds |
| **Cascade** | One method (fast but coarse) → second (precise) | High precision at scale |
| **Switch** | Switch between methods based on confidence | Handles cold-start gracefully |
| **Meta-level** | Model trained on CF embeddings + content features | Most powerful, most complex |

```python
class HybridRecommender:
    def __init__(self):
        self.cf = CollaborativeFiltering()
        self.cb = ContentBasedFiltering()
        self.alpha = 0.7  # weight for CF vs content
        
    def recommend(self, user_id, user_history, n=10):
        cf_scores = self.cf.predict(user_id)
        cb_scores = self.cb.predict(user_history)
        hybrid_scores = self.alpha * cf_scores + (1 - self.alpha) * cb_scores
        return hybrid_scores.argsort()[::-1][:n]
```

---

## 5. Deep Learning for Recommendations

### 5.1 Neural Collaborative Filtering (NCF, He et al., 2017)
Replace dot product with an MLP:

R̂_ui = MLP(U_u || V_i)

```python
def ncf_model(n_users, n_items, n_factors=32):
    """PyTorch-like NCF architecture"""
    user_input = Input(shape=(1,))
    item_input = Input(shape=(1,))
    
    user_embedding = Embedding(n_users, n_factors)(user_input)
    item_embedding = Embedding(n_items, n_factors)(item_input)
    
    concat = Concatenate()([user_embedding, item_embedding])
    x = Dense(128, activation='relu')(concat)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=[user_input, item_input], outputs=output)
```

Generalizes matrix factorization, learns non-linear interactions. NCF > MF > Item-KNN.

### 5.2 Two-Tower Model (Candidate Generation)
Popularized by Google, YouTube, Pinterest.

- **User tower:** user features → user embedding (e_u ∈ R^d)
- **Item tower:** item features → item embedding (e_i ∈ R^d)
- **Score:** dot product s(u,i) = e_u · e_i

**Efficiency:** Pre-compute item embeddings → ANN search for each user query.

```python
class TwoTowerModel(torch.nn.Module):
    def __init__(self, n_users, n_items, n_features_user, n_features_item, dim=64):
        super().__init__()
        self.user_tower = torch.nn.Sequential(
            torch.nn.Linear(n_features_user, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, dim),
        )
        self.item_tower = torch.nn.Sequential(
            torch.nn.Linear(n_features_item, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, dim),
        )
    
    def forward(self, user_features, item_features):
        user_emb = self.user_tower(user_features)  # (batch, dim)
        item_emb = self.item_tower(item_features)  # (batch, dim)
        return (user_emb * item_emb).sum(dim=-1)   # dot product
```

### 5.3 DCN (Deep & Cross Network, Wang et al., 2017)
Cross network learns explicit bounded-degree feature interactions:

x_{l+1} = x₀ · (W_l x_l + b_l) + x_l

Combined with a deep network for non-linear features. Published by Google.

### 5.4 Wide & Deep (Google, 2016)
- **Wide:** Memorization via cross-product feature transformations
- **Deep:** Generalization via embedding layers + MLP

Used by Google Play Store for app recommendations. The wide part handles feature sparsity (memorizing rare patterns like "user installed A and B → installs C"), the deep part generalizes to unseen patterns.

### 5.5 DIN (Deep Interest Network, Alibaba, 2018)
Attention-based user behavior modeling:

v_U = Σ_j a(v_i, v_j) · v_j

Where a is an attention weight computed as: a(v_i, v_j) = MLP(v_i || v_j || v_i - v_j)

Activates the user interests most relevant to **this specific recommendation**. Instead of one fixed user vector, the representation changes per candidate.

### 5.6 Multi-Head Self-Attention (BST, Alibaba)
Billion-scale Sequential Transformer: Apply transformer encoder over user behavior sequence. Captures long-range dependencies (user watched movie A → 10 unrelated movies → now wants genre of A).

### 5.7 Graph-Based Recommendations (PinSage, 2018)
Treat user-item interactions as a bipartite graph. Use Graph Neural Networks (GNN) to propagate information:

- **PinSage** (Pinterest): GCN on 3B node Pinterest graph
- **LightGCN:** Simplified GCN (remove non-linearities, only linear propagation)
- **NGCF:** Neural Graph Collaborative Filtering

### 5.8 LLM-Based Recommendations
| Approach | Method | Example |
|----------|--------|---------|
| **Embedding-based** | Use LLM embeddings as item features | Better semantic understanding |
| **LLM as ranker** | "Given user history [items], rank these candidates" | P5, RecLLM |
| **Conversational** | Chat-based refinement | "Show me more like X but cheaper" |
| **Cold-start** | LLM generates item descriptions from metadata | Zero-shot cold-start |
| **Feature generation** | LLM creates user/item features | Pseudo-labeling long-tail items |

---

## 6. Feature Engineering

### 6.1 Feature Types

| Type | Examples | Processing |
|------|----------|------------|
| Sparse categorical | User ID, item ID, category | Embedding lookup |
| Dense numeric | Price, age, time since last view | Normalization (z-score), log transform |
| Text | Title, description | TF-IDF, sentence-BERT, LLM embeddings |
| Image | Product image, profile picture | CNN/ViT (ResNet, CLIP) embeddings |
| Sequence | Browse history, purchase history | LSTM, transformer, attention pooling |
| Context | Time of day, day of week, location, device | One-hot, bucketization, cyclical encoding |
| Crossed | user_category × item_category | Feature crosses (X1 * X2) |

### 6.2 Contextual Features

| Context | Examples | Encoding |
|---------|----------|----------|
| Temporal | Hour of day, day of week, season | Sine/cosine cyclical encoding |
| Location | Country, city, GPS coordinates | Embeddings, discretization |
| Device | Mobile vs desktop, OS, screen size | One-hot |
| Session | Time on site, pages visited, referrer | Direct numeric, bucketized |

```python
# Cyclical time encoding
def cyclical_time(hour):
    """Encode time as (sin, cos) for continuity (23:00 and 01:00 are close)"""
    return np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
```

### 6.3 Feature Crosses
Combine features to capture interactions that individual features miss:
- user_category × item_category (e.g., "gamer × gaming chair")
- user_age_range × item_price_range (e.g., "young × cheap")
- time_of_day × day_of_week (e.g., "Friday night × food delivery")

### 6.4 Feature Stores
Feast, Tecton, SageMaker Feature Store — centralized, consistent feature computation.
- **Online serving:** Low-latency feature lookup (<10ms p99)
- **Offline training:** Point-in-time correct feature retrieval
- **Feature freshness:** Real-time feature updates

---

## 7. Candidate Generation

### 7.1 ANN (Approximate Nearest Neighbor) Methods

| Method | Scale | Similarity | Best For |
|--------|:----:|:----------:|----------|
| **LSH** (Locality Sensitive Hashing) | 1B+ | Jaccard, cosine | Binary features, text |
| **HNSW** (Hierarchical Navigable Small World) | 10M+ | L2, cosine, IP | Dense embeddings (Faiss default) |
| **SCaNN** (Google) | 1B+ | Cosine, dot product | Maximum inner product search |
| **IVF** (Inverted File Index) | 100M+ | L2 | General-purpose, balanced |
| **IVF-PQ** (product quantization) | 1B+ | L2 asymmetric | Memory-efficient compression |
| **DiskANN** | 1B+ | L2, cosine | SSD-based, >1B vectors |

```python
import faiss

# Build HNSW index for 100K items
dim = 64
index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors in graph
index.hnsw.efConstruction = 200
index.add(item_embeddings)  # item_embeddings: (n_items, dim)

# Query (top 100 candidates)
user_embedding = user_tower(user_features)
scores, indices = index.search(user_embedding.reshape(1, -1), k=100)
```

### 7.2 Retrieval Quality Metrics
- **Recall@k:** Fraction of relevant items in top k
- **Precision@k:** Fraction of top k that are relevant
- **Coverage:** Fraction of items that can be retrieved
- **Personalization:** Diversity of recommendations across users

---

## 8. Ranking

### 8.1 Pointwise Ranking
Predict a score for each (user, item) independently.

- **Regression:** Predict rating (1-5) → MSE loss
- **Binary classification:** Predict click/not-click → cross-entropy loss

AUC is the standard metric (ranking quality independent of threshold).

### 8.2 Pairwise Ranking
Learn to rank pairs: given (item_A, item_B), which is more relevant?

- **BPR (Bayesian Personalized Ranking):** maximize P(r_ui > r_uj) = σ(R̂_ui - R̂_uj)
- **WARP (Weighted Approximate Rank Pairwise):** Weight pairs by rank position
- **LambdaRank:** Gradient-based pair weighting

### 8.3 Listwise Ranking
Optimize the entire ranking list directly.

- **NDCG** (Normalized Discounted Cumulative Gain): position-discounted ranking metric
- **ListNet:** Probability distribution over permutations
- **SoftRank:** Smooth approximation of rank-based metrics

```python
def ndcg_score(y_true, y_pred, k=10):
    """Compute NDCG@k"""
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order][:k]
    
    dcg = y_true_sorted[0] + sum(y_true_sorted[i] / np.log2(i + 2) 
                                  for i in range(1, len(y_true_sorted)))
    ideal = np.sort(y_true)[::-1][:k]
    idcg = ideal[0] + sum(ideal[i] / np.log2(i + 2) 
                           for i in range(1, len(ideal)))
    return dcg / idcg if idcg > 0 else 0
```

---

## 9. Session-Based Recommendations

Handle users without long-term history (anonymous browsing).

**Approaches:**
| Model | Method | Strengths |
|-------|--------|-----------|
| **Item-KNN** | Item co-occurrence in session | Simple, fast |
| **GRU4Rec** (Hidasi et al., 2016) | RNN over click sequences | Sequential patterns |
| **SR-GNN** (Wu et al., 2019) | GNN over session graphs | Complex transition patterns |
| **BERT4Rec** | Masked language modeling on sequences | Bi-directional context |

```python
# Simplified GRU4Rec
class SessionRec(torch.nn.Module):
    def __init__(self, n_items, hidden=100):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_items, hidden)
        self.gru = torch.nn.GRU(hidden, hidden, batch_first=True)
        self.output = torch.nn.Linear(hidden, n_items)
    
    def forward(self, sequences):
        emb = self.embedding(sequences)             # (B, L, H)
        _, hidden = self.gru(emb)                    # (1, B, H)
        scores = self.output(hidden.squeeze(0))      # (B, n_items)
        return scores
```

---

## 10. Cold Start Solutions

### 10.1 User Cold Start (New Users)
| Strategy | Method |
|----------|--------|
| **Popularity** | Recommend trending items |
| **Demographics** | Use age, location, gender for initial predictions |
| **Onboarding survey** | Ask preferences directly |
| **Explore-then-exploit** | Show diverse items, learn fast |
| **Cross-domain** | Use preferences from other services |
| **Contextual bandits** | Thompson sampling for cold users |

### 10.2 Item Cold Start (New Items)
| Strategy | Method |
|----------|--------|
| **Content features** | Use item metadata for similarity matching |
| **LLM-generated features** | Generate relevant tags/descriptions from limited data |
| **Random exploration** | Show to a fraction of users (ε-greedy) |
| **Drop-in embedding** | Predict embedding from content features (DropoutNet) |

---

## 11. Evaluation

### 11.1 Offline Metrics

| Metric | What It Measures |
|--------|-----------------|
| RMSE/MAE | Rating prediction accuracy |
| Precision@k, Recall@k | Ranking accuracy |
| NDCG@k | Ranking quality with position discount |
| Hit Rate@k | Any relevant in top k? |
| Coverage | Fraction of recommendable items surfaced |
| Diversity (ILD) | Intra-list distance between recommended items |
| Novelty (EPC) | How surprising are the recommendations |
| Serendipity | Unexpected but pleasant recommendations |
| Personalization | Pairwise inter-list distance across users |

### 11.2 Online Metrics

| Metric | What It Measures |
|--------|-----------------|
| CTR (click-through rate) | User engagement |
| Conversion rate | Purchase/subscribe after click |
| Engagement time | Session length, item interaction depth |
| Retention rate | D30/D90 return rate |
| User satisfaction | Surveys, NPS scores |
| Revenue per user | Direct business impact |
| Diversity/churn | Are users stuck in filter bubbles? |

### 11.3 A/B Testing

- Randomize users into control (old model) and treatment (new model)
- **Minimum 2-week run** for week-over-week effects
- **Guardrail metrics** to detect unintended harm (revenue, engagement, diversity)
- **Interleaved experiments:** Show users blended results from both models → faster convergence (50% fewer users needed)
- **Holdout:** Measure counterfactual (what would old model have recommended?)

### 11.4 Fairness in Recommendation Systems

Fairness is a growing concern — recommendation systems can amplify bias, create filter bubbles, and disadvantage certain user groups.

**Types of Fairness:**

| Fairness Type | Definition | Metric |
|---------------|------------|--------|
| **Demographic parity** | Same recommendation quality across groups | Precision/Recall gap between groups |
| **Equal opportunity** | Same true positive rate across groups | TPR gap |
| **Exposure fairness** | Items from all providers get proportional exposure | Expected exposure ratio |
| **Calibration** | Recommendations reflect user's true preferences | Calibration error per group |
| **Counterfactual fairness** | Recommendation unchanged if sensitive attribute changed | Counterfactual evaluation |

**Fairness-Aware Methods:**

| Method | Approach | Trade-off |
|--------|----------|-----------|
| **Fairness regularization** | Add fairness penalty to loss | Accuracy ↔ fairness |
| **Adversarial debiasing** | Adversary tries to predict sensitive attribute from embeddings | Removes sensitive info |
| **Post-processing** | Re-rank to satisfy fairness constraints | Simple, non-invasive |
| **Causal fairness** | Model causal graph, remove unfair paths | Requires causal assumptions |
| **DPP with fairness** | Determinantal point process with diversity constraints | Preserves diversity |

**Filter Bubbles & Echo Chambers:**
- Personalization can trap users in narrow content spaces
- **Mitigations:** Inject diversity, serendipity objectives; use multi-objective optimization; allow user control over recommendation goals
- **Measurement:** Track entropy of recommended item categories over time; survey users on recommendation satisfaction

## 12. Causal Recommendation Systems

Traditional recsys find correlations between user behavior and item features. Causal recsys tries to model **cause-effect relationships** to make interventions more effective and debias recommendations.

### 12.1 The Causal View of Recommendations

```
Traditional (correlational): P(y|user, item, context)
Causal: P(y|do(user), do(item), context) — what if we intervene?
```

**Why causality matters:**
1. **Debiasing:** Observed data has selection bias (users choose what to interact with)
2. **Counterfactual reasoning:** What would have happened if we showed a different item?
3. **Interventions:** How will changing the recommendation algorithm change user behavior?
4. **Long-term effects:** Short-term engagement ≠ long-term satisfaction

### 12.2 Causal Debiasing Methods

| Bias | Causal Solution | How |
|------|----------------|-----|
| **Selection bias** | Inverse Propensity Scoring (IPS) | Weight observations by 1/P(observed) |
| **Position bias** | Position-aware models | Model position as treatment variable |
| **Conformity bias** | Causal graph with conformity variable | Separate true preference from conformity |
| **Popularity bias** | Causal backdoor adjustment | Adjust for popularity confounder |

**IPS Estimator:**
```
R_IPS = 1/|U| Σ_u 1/|I| Σ_i O_ui · Y_ui / P(O_ui=1|U,I)
```
Where O_ui is the observation indicator and P(O_ui=1|U,I) is the propensity score.

### 12.3 Counterfactual Reasoning

Given a causal graph of the recommendation process, we can ask counterfactual questions:

- "Would the user have clicked item B if we had shown it instead of item A?"
- "Would the user have purchased if we hadn't recommended anything?"
- "What would the user's satisfaction be if we sacrificed 10% engagement for 30% diversity?"

**Potential Outcomes Framework (Rubin Causal Model):**
```
Individual Treatment Effect = Y_i(treatment) - Y_i(control)
```

Since we can only observe one outcome per user, we estimate the **Average Treatment Effect (ATE)** or **Conditional ATE (CATE)** across users.

### 12.4 Bandits for Causal Recommendation

Contextual bandits provide a natural framework for causal recommendation:

1. **Explore:** Recommend items with some randomness
2. **Observe:** User response (click/purchase/dwell)
3. **Update:** Model which action caused which outcome
4. **Exploit:** Use the causal model to make better recommendations

**Key advantage:** Bandits actively collect data that enables causal inference, avoiding the selection bias inherent in purely observational data.

## 13. Multi-Task Learning in RecSys

Modern recommendation systems often optimize multiple objectives simultaneously:

| Task | Objective | Loss |
|------|-----------|------|
| Click prediction | Will user click? | Binary cross-entropy |
| Conversion prediction | Will user purchase? | Binary cross-entropy |
| Dwell time prediction | How long will user engage? | MSE/MAE |
| Rating prediction | How will user rate? | MSE |
| Engagement diversity | Will user explore? | Entropy bonus |

**MMOE (Multi-gate Mixture of Experts):**
```
Shared bottom → Expert 1 → Gate → Task A tower → Task A output
              → Expert 2 → Gate → Task B tower → Task B output
              → Expert 3
```
Each task learns its own expert weighting (gate), enabling flexible sharing while preventing negative transfer between conflicting objectives.

**PLE (Progressive Layered Extraction, Tencent, 2020):**
Extends MMOE with task-specific and shared expert networks at multiple layers. Used by Tencent for video recommendations — improved both CTR and watch time.

## 14. Contextual Bandits for Recommendations

When the recommender must balance exploration and exploitation in real-time:

| Algorithm | Regret | Model | Use Case |
|-----------|--------|-------|----------|
| **LinUCB** | O(d√T) | Linear | Simple features |
| **Thompson Sampling** | O(√T) | Bayesian | Risk-sensitive (news) |
| **Neural Bandit** | O(√T) (empirical) | Neural network | Complex features |
| **Neural Linear** | O(d√T) | NN encoder + linear head | Best of both worlds |
| **Bootstrap Bandit** | O(√T) | Ensemble of models | Uncertainty estimation |

**LinUCB for News Recommendation (Li et al., 2010):**
Used by Yahoo! News to recommend articles. User features + article features → linear model predicts CTR. UCB exploration bonus: recommend articles with high upper confidence bound. Achieved 12.5% CTR improvement over static control.

## 15. Industry Case Studies

### 15.1 Netflix Prize (2006-2009)

| Aspect | Detail |
|--------|--------|
| **Goal** | Improve rating prediction by 10% |
| **Data** | 100M ratings, 480K users, 18K movies |
| **Winning method** | Ensemble of 107 models (SVD + RBM + kNN + matrix factorization) |
| **Impact** | Launched modern recsys research; proved ensembles beat any single method |
| **Key lesson** | Real improvement was only ~10%, but Netflix found that even small RMSE improvements drove significant engagement gains |

### 15.2 Spotify Discover Weekly

| Aspect | Detail |
|--------|--------|
| **Algorithm** | Collaborative filtering on listening history + audio features + NLP of articles/blog posts |
| **Innovation** | Uses raw audio analysis (CNN embeddings) not just metadata |
| **Result** | 40% of users engage with Discover Weekly within 24 hours |
| **Key insight** | Combining content-based (audio) with collaborative filtering produces superior recommendations |

### 15.3 YouTube Recommendation System

| Aspect | Detail |
|--------|--------|
| **Candidate gen** | Two-tower model (user history → candidate videos) |
| **Ranking** | Deep neural network with watch time weighting |
| **Key challenge** | Scale (billions of users, millions of uploads/day) |
| **Innovation** | Watch time as the optimization target (not clicks) — \"expected watch time per impression\" |
| **Result** | ~70% of watch time from recommendations |

### 15.4 Amazon Product Recommendations

| Aspect | Detail |
|--------|--------|
| **Algorithm** | Item-to-item collaborative filtering (original) + deep learning (modern) |
| **Scale** | 35% of revenue attributed to recommendations |
| **Key feature** | Bought-together, frequently-bought-together, customers-who-bought-this-also-bought |
| **Innovation** | Session-based recommendations for non-logged-in users |
| **Result** | ~300M products recommended daily |

## 16. Emerging Trends (2024-2026)

| Trend | Description | Maturity |
|-------|-------------|:--------:|
| **LLM-powered recommendation** | Using LLMs for semantic understanding, cold-start, conversational recs | Emerging |
| **Generative recommendation** | Diffusion models for recommendation (DiffRec, DreamRec) | Research |
| **Reinforcement learning** | Long-term engagement optimization via policy gradients | Production (notable companies) |
| **Graph neural networks** | LightGCN, PinSage for collaborative filtering | Production |
| **Causal recommendation** | Debiasing, treatment effect estimation | Early production |
| **Federated recommendation** | Privacy-preserving training across user devices | Research/early production |
| **Multimodal recommendation** | Combining text, image, video, audio features | Production (Spotify, TikTok) |
| **Personalized foundation models** | Single model adapted per user | Research |

---

## 17. Biases and Pitfalls in Evaluation

| Bias | Description | Mitigation |
|------|-------------|------------|
| **Popularity bias** | Popular items get recommended → get more popular | Debiasing, long-tail handling |
| **Position bias** | Items at top get more clicks regardless of relevance | Position-aware models, IPW |
| **Selection bias** | Observed ratings not random (self-selection) | Propensity scoring, causal methods |
| **Exposure bias** | Unobserved interactions = negative? Or unaware? | Exposure modeling |
| **Conformity bias** | Users rate in line with existing ratings | Temporal models |
| **Feedback loop** | Recsys promotes certain items → collects data → reinforces | Exploration, randomization |

---

## 18. Production Considerations

### 18.1 Serving Architecture

```
User Request → Feature Pipeline → Candidate Gen → Ranking Model → Re-ranker → Response
                  ↓                    ↓               ↓
            Feature Store         ANN Index       Model Server
```

**Latency budgets:** Feature serving <5ms, ANN search <20ms, Ranking <50ms, Total <200ms p99.

### 18.2 Model Updates

| Update Type | Frequency | Method |
|-------------|-----------|--------|
| **Batch training** | Daily/weekly | Full retrain on all data |
| **Incremental update** | Hourly | Fine-tune on new data |
| **Online learning** | Real-time | Per-batch SGD updates |
| **Bandit adaptation** | Per-request | Thompson sampling exploration |

### 18.3 Model Monitoring

| Metric | Alert | Action |
|--------|-------|--------|
| Prediction drift | KL(P(old)||P(new)) > threshold | Retrain |
| Feature drift | Feature distribution shift | Investigate data pipeline |
| CTR drop | >5% decrease | Rollback model |
| Latency spike | p99 > 2× baseline | Scale servers, optimize inference |
| Coverage drop | Recommended items < 50% of catalog | Investigate retriever |

### 18.4 Scaling Best Practices

- **Pre-compute embeddings:** User embeddings updated asynchronously (not per-request)
- **Caching:** Cache popular recommendations for cold users
- **Sharding:** Partition user/item embeddings across servers
- **Quantization:** INT8 quantization reduces model size 4× with <1% accuracy loss
- **Distillation:** Train smaller student model from large teacher for ranking

---

---

## 18. Multi-Task Learning for Recommendations

Multi-task learning (MTL) jointly optimises multiple objectives from a shared representation, which is critical for real-world recommendation systems that must balance **clicks**, **conversions**, **dwell time**, **shares**, and **long-term satisfaction**. Training separate models for each objective ignores correlated signals and multiplies maintenance cost; MTL shares statistical strength while letting each task maintain task-specific behaviour.

### Shared-Bottom Architecture

The simplest MTL approach: a shared bottom network (e.g. a stack of MLP layers) feeds into task-specific towers.

```
Input features
      ↓
[Shared bottom network]
      ↓
┌─────┼─────┐
│     │     │
Tower A  Tower B  Tower C
(click)  (conv)  (dwell)
  ↓       ↓       ↓
 σ(·)    σ(·)   ReLU(·)
```

**Limitation:** If tasks conflict (e.g. click-bait hurts dwell time), the shared bottom must compromise, leading to *negative transfer*.

### MMoE — Multi-gate Mixture-of-Experts (Ma et al., KDD 2018)

MMoE replaces the single shared bottom with a set of **expert sub-networks** and a **learned gate per task** that weights the experts differently.

| Component | Role |
|-----------|------|
| **Experts** | \(k\) feed-forward networks, each learning a different feature subspace |
| **Gates** | Per-task softmax over experts: \(g^{(t)}(x) = \text{softmax}(W_g^{(t)} x)\) |
| **Task towers** | Take the gated-weighted sum of expert outputs as input |

\[
f^{(t)}(x) = \sum_{i=1}^{k} g_i^{(t)}(x) \cdot e_i(x)
\]

Each task has its own gate, so task A can rely heavily on expert 1 while task B uses expert 2 — avoiding negative transfer while still sharing.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MMoE(nn.Module):
    """Multi-gate Mixture-of-Experts for multi-task recommendation.

    Args:
        input_dim:   Feature dimension after embedding/bottleneck
        num_experts: Number of expert sub-networks (typically 4-16)
        expert_dim:  Hidden dimension inside each expert
        num_tasks:   Number of prediction heads (click, conv, dwell, …)
        task_dims:   Hidden dimensions for each task tower
    """
    def __init__(
        self,
        input_dim: int,
        num_experts: int = 8,
        expert_dim: int = 64,
        num_tasks: int = 3,
        task_dims: list[int] | None = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        if task_dims is None:
            task_dims = [32] * num_tasks

        # Expert networks  (shared across tasks)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, expert_dim),
                nn.ReLU(),
            )
            for _ in range(num_experts)
        ])

        # Task-specific gates  (one softmax per task)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=-1),
            )
            for _ in range(num_tasks)
        ])

        # Task-specific towers
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, task_dims[t]),
                nn.ReLU(),
                nn.Linear(task_dims[t], 1),
            )
            for t in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return list of logits, one per task."""
        # Expert outputs: list of (batch, expert_dim)
        expert_out = [e(x) for e in self.experts]          # k tensors
        expert_stack = torch.stack(expert_out, dim=1)       # (B, k, D)

        outputs = []
        for t in range(self.num_tasks):
            gate_weights = self.gates[t](x)                 # (B, k)
            weighted = (gate_weights.unsqueeze(-1)           # (B, k, 1)
                        * expert_stack).sum(dim=1)          # (B, D)
            logit = self.towers[t](weighted)                 # (B, 1)
            outputs.append(logit)
        return outputs

# -------------------------------------------------------------------
# Training usage
# -------------------------------------------------------------------
def mmoe_loss(task_logits: list[torch.Tensor],
              task_targets: list[torch.Tensor],
              task_weights: list[float] | None = None) -> torch.Tensor:
    """Weighted sum of per-task losses."""
    if task_weights is None:
        task_weights = [1.0] * len(task_logits)
    total = 0.0
    for logits, targets, w in zip(task_logits, task_targets, task_weights):
        # Click/conversion → BCEWithLogits; dwell → MSE after exp or ReLU
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        total += w * loss
    return total

# Example instantiation
model = MMoE(input_dim=128, num_experts=8, expert_dim=64, num_tasks=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in dataloader:
    features = batch["features"]               # (B, 128)
    click_targets = batch["click"].float()     # (B, 1)
    conv_targets  = batch["conversion"].float()
    dwell_targets = batch["dwell_norm"].float()

    logits = model(features)                    # list of 3 tensors
    loss = mmoe_loss(
        logits,
        [click_targets, conv_targets, dwell_targets],
        task_weights=[0.4, 0.4, 0.2],
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Why Multi-Task Matters

| Reason | Explanation |
|--------|-------------|
| **Optimise the real objective** | Users who click but never convert are low value — joint training de-prioritises click-bait |
| **Data efficiency** | Conversion events are rare (1-5% of clicks); sharing representations with the click task improves conversion model quality |
| **Reduced serving cost** | One forward pass replaces three separate model invocations |
| **Cross-task signal** | A user's short dwell time on a clicked item signals disappointment even though the click looks positive |
| **Covariate shift robustness** | When the item catalogue shifts, all tasks adapt together via the shared experts |

### Advanced MTL for Recommendations

| Method | Company | Key Idea |
|--------|---------|----------|
| **MMoE** (Ma et al., 2018) | Google | Multi-gate mixture-of-experts; each task learns its own expert weighting |
| **PLE** (Tang et al., 2020) | Tencent | Progressive Layered Extraction — task-specific + shared experts at multiple layers |
| **SNR** (Sub-Network Routing, 2021) | Alibaba | Learnable binary masks per task over shared weights, enabling fine-grained routing |
| **DSelect-k** (Hazimeh et al., 2021) | Google | Sparse mixture-of-experts with a continuous top-k gate for efficiency |
| **MBLR** (Multi-Block Latent Routing) | ByteDance | Hierarchical routing across blocks of experts for massive multi-task video recommendation |

---

## 18a. Causal Recommendation Systems

While section 12 introduced the causal perspective, this section focuses on practical **estimation** and **debiasing** techniques that can be plugged directly into a recommendation pipeline.

### Why Observational Data Is Biased

Users choose which items to interact with — so the data we observe is a **biased sample**. Popular items are over-represented, niche items are under-represented, and a click on the first position does not mean the same thing as a click on the tenth.

```
True reward landscape     Observed data (biased by exposure)
┌──────────────┐          ┌──────────────┐
│  ○  ○  ○     │          │  ●  ○  ●     │
│    ○  ○  ○   │   ⇒      │    ○  ○  ●   │
│  ○  ●  ○  ○  │          │  ○  ●  ○  ○  │
│  ○     ○  ○  │          │  ○     ○     │
└──────────────┘          └──────────────┘
  ○ = relevant               ● = observed (biased toward popular)
```

### Inverse Propensity Scoring (IPS)

IPS re-weights each observation by the inverse of its probability of being observed (the *propensity score*).

\[
\hat{R}_{\text{IPS}} = \frac{1}{|\mathcal{D}|} \sum_{(u,i) \in \mathcal{D}} \frac{y_{ui} \cdot o_{ui}}{\hat{p}_{ui}} \qquad
o_{ui} = \begin{cases}1 & \text{if } (u,i) \text{ observed} \\ 0 & \text{otherwise}\end{cases}
\]

- **Unbiased** if propensities are correctly specified: \(\mathbb{E}[\hat{R}_{\text{IPS}}] = R^*\)
- **High variance** when propensities are small (rare items get huge weights)

**Estimating propensities:** Use a logistic regression that predicts \(P(\text{observed} \mid u, i)\) from user and item features; include known confounding variables (e.g. position, popularity, recommender score at time of exposure).

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

def estimate_propensities(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    observed: np.ndarray,        # 1 if user-item pair was observed
    user_pop: np.ndarray,        # feature: log(interactions) of user
    item_pop: np.ndarray,        # feature: log(interactions) of item
    position: np.ndarray,        # feature: display position (1, 2, …)
) -> np.ndarray:
    """Train a propensity model and return P(observed) for each pair."""
    X = np.column_stack([user_ids, item_ids, user_pop, item_pop, position])
    clf = LogisticRegression(max_iter=1000).fit(X, observed)
    return clf.predict_proba(X)[:, 1]        # P(o=1 | features)

# IPS-augmented evaluation
def ips_evaluation(
    y_true: np.ndarray,          # ground truth relevance
    y_pred: np.ndarray,          # model scores
    propensities: np.ndarray,    # P(observed)
) -> float:
    """Unbiased estimate of average reward via IPS."""
    weights = 1.0 / np.clip(propensities, 1e-6, None)
    return np.mean(weights * (y_true == y_pred))
```

### Doubly Robust (DR) Estimation

DR combines IPS with an **imputation model** (a learned estimate of the outcome) to reduce variance while preserving unbiasedness.

\[
\hat{R}_{\text{DR}} = \frac{1}{|\mathcal{D}|} \sum_{(u,i) \in \mathcal{D}} \left[ \hat{y}_{ui} + \frac{o_{ui} (y_{ui} - \hat{y}_{ui})}{\hat{p}_{ui}} \right]
\]

| Property | IPS | Imputation (DM) | Doubly Robust |
|----------|:---:|:----------------:|:-------------:|
| Unbiased if | propensities correct | imputation correct | **either** is correct |
| Variance | High (small p) | Low | Low–Medium |
| Robustness | Sensitive to p | Model dependent | **Doubly robust** |
| Practical choice | + | + | **Best of both** |

```python
def doubly_robust(
    y_true: np.ndarray,
    y_imputed: np.ndarray,      # predictions from outcome model
    propensities: np.ndarray,
    observed: np.ndarray,       # boolean mask
    clip: float = 1e-6,
) -> float:
    """Doubly-robust estimate of mean relevance."""
    p = np.clip(propensities, clip, None)
    correction = observed * (y_true - y_imputed) / p
    return np.mean(y_imputed + correction)
```

### Causal Embeddings

Standard embeddings capture correlations; causal embeddings separate **confounded** signals from **causal** ones.

| Approach | Idea | Reference |
|----------|------|-----------|
| **Deconfounded CF** | Learn user/item embeddings from a causal graph that explicitly models the recommender policy as a confounder | Wang et al., WSDM 2020 |
| **Causal Intervention** | Apply back-door adjustment: \(P(y \mid do(u), do(i)) = \sum_c P(y \mid u,i,c) P(c)\) where \(c\) are confounding contexts | Zhang et al., KDD 2021 |
| **IV Recommendation** | Use instrumental variables (e.g. friend's exposure) to isolate causal effect from selection bias | Si et al., AAAI 2023 |
| **Causal Embedding** | Split embedding into a causal part (orthogonal to confounders) and a confounding part via adversarial training | Veitch et al., NeurIPS 2021 |

### Comparison of Debiasing Methods

| Method | Strengths | Weaknesses | When to Use |
|--------|-----------|------------|-------------|
| **IPS** | Simple to implement; unbiased if propensities known | High variance; requires good propensity model | Evaluation & offline policy estimation |
| **Doubly Robust** | Lower variance than IPS; unbiased if **either** model is correct | Two models to maintain; can over-correct | Standard choice for offline evaluation |
| **Self-Normalised IPS (SNIPS)** | Lower variance than IPS; scale-invariant | Biased for small samples | When propensity weights are extreme |
| **Causal Embeddings** | Removes confounding at representation level | Complex training; requires causal graph assumptions | Cold-start and long-term effect modeling |
| **IPS + Clipping** | Cap extreme weights to reduce variance | Introduces bias from clipping | Production pipelines where tail items cause instability |
| **DR with IPS** | Doubly robust evaluation of ranking metrics | Computationally heavier | A/B test surrogate, counterfactual logging |
| **Position-aware model** | Directly models position effect | Requires position data | Any ranking model with position bias |

```python
# Practical recipe: Debiased evaluation with DR
def debiased_ndcg(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    propensities: np.ndarray,
    observed: np.ndarray,
    k: int = 10,
) -> float:
    """Doubly-robust NDCG@k for ranking evaluation."""
    # Rank items by predicted score
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order][:k]
    observed_sorted = observed[order][:k]
    p_sorted = propensities[order][:k]

    # Imputation: use y_true as our outcome model (simple baseline)
    y_imp = 0.0
    correction = observed_sorted * (y_true_sorted - y_imp) / np.clip(p_sorted, 1e-6, None)
    dr_relevance = y_imp + correction

    dcg = dr_relevance[0] + sum(
        dr_relevance[i] / np.log2(i + 2) for i in range(1, len(dr_relevance))
    )
    ideal = np.sort(y_true)[::-1][:k]
    idcg = ideal[0] + sum(
        ideal[i] / np.log2(i + 2) for i in range(1, len(ideal))
    )
    return dcg / idcg if idcg > 0 else 0
```

---

## 18b. LLM-Based Recommendation Pattern

Large Language Models bring **semantic understanding** and **reasoning** to recommendation tasks. This section gives a concrete code pattern and a taxonomy of approaches.

### Pattern: LLM Scoring of Candidates

Given a user's interaction history, the LLM scores each candidate item by reasoning about relevance, recency, diversity, and user intent.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.2-3B-Instruct"  # or any chat model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# -------------------------------------------------------------------
# Build a chat-style prompt encoding user history + a single candidate
# -------------------------------------------------------------------
def build_scoring_prompt(
    user_history: list[str],          # ["item_A title", "item_B title", …]
    candidate_title: str,
    candidate_metadata: str = "",
) -> str:
    """Create a chat-message prompt for scoring one candidate."""
    history_str = "\n".join(f"- {item}" for item in user_history)
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"You are a recommendation expert. Given a user's recent interaction "
        f"history, rate how relevant the candidate item is on a scale of "
        f"0.0 (not relevant at all) to 1.0 (perfect match). "
        f"Respond with ONLY a floating-point number between 0 and 1.\n"
        f"<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"## User History\n{history_str}\n\n"
        f"## Candidate Item\nTitle: {candidate_title}\n"
        f"{candidate_metadata}\n"
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

# -------------------------------------------------------------------
# Score a batch of candidate items
# -------------------------------------------------------------------
def llm_score_candidates(
    user_history: list[str],
    candidates: list[dict],           # [{"title": …, "metadata": …}, …]
    batch_size: int = 4,
    max_new_tokens: int = 8,
) -> list[float]:
    """Return a relevance score in [0, 1] for each candidate."""
    prompts = [
        build_scoring_prompt(user_history, c["title"], c.get("metadata", ""))
        for c in candidates
    ]
    scores = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(
            model.device
        )
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,          # low temperature for stable scoring
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Extract the assistant response after the prompt
        prompt_len = inputs["input_ids"].shape[1]
        replies = tokenizer.batch_decode(
            outputs[:, prompt_len:], skip_special_tokens=True
        )
        for reply in replies:
            try:
                score = float(reply.strip().split()[0])
                scores.append(max(0.0, min(1.0, score)))
            except (ValueError, IndexError):
                scores.append(0.0)  # fallback on parse failure
    return scores

# -------------------------------------------------------------------
# Usage: score 100 candidates, pick top-10
# -------------------------------------------------------------------
user_history = [
    "The Wind-Up Bird Chronicle — Haruki Murakami",
    "Kafka on the Shore — Haruki Murakami",
    "Norwegian Wood — Haruki Murakami",
    "1Q84 — Haruki Murakami",
]
candidates = [
    {"title": "The Catcher in the Rye", "metadata": "Author: J.D. Salinger"},
    {"title": "Hard-Boiled Wonderland", "metadata": "Author: Haruki Murakami"},
    {"title": "1984", "metadata": "Author: George Orwell"},
    # ... typically 100-500 candidates from the retrieval stage
]

scores = llm_score_candidates(user_history, candidates)
top_k = sorted(
    zip(candidates, scores), key=lambda x: x[1], reverse=True
)[:10]

for item, score in top_k:
    print(f"{score:.3f} — {item['title']}")
```

### Taxonomy of LLM Approaches for Recommendation

| Approach | Description | Strengths | Weaknesses | Example Systems |
|----------|-------------|-----------|------------|-----------------|
| **Embedding-based** | Use LLM embeddings (e.g. LLM2Vec, SGPT) as item features in a standard two-tower model | Drop-in replacement for text encoders; low latency at serving | Discards reasoning capability; limited to static semantics | P5 (Huawei), LlamaRec, PALR |
| **LLM as Ranker** | Feed user history + candidates as text; LLM outputs scores or ranks | Rich reasoning; zero-shot cold-start; interpretable | High latency (seconds per batch); expensive per-query | RecLLM (Google), ChatRec, RankGPT |
| **LLM as Retriever** | LLM generates query embeddings or directly retrieves item IDs | End-to-end text-to-item; novel item handling | Hard to scale beyond a few thousand items; hallucination risk | LLMRec, GenRec |
| **Conversational** | Multi-turn dialogue for preference elicitation and refinement | High user engagement; handles complex constraints | UX design challenge; increased interaction cost | ChatGPT + Rec (Kang et al.), RecAgent |
| **Feature Generation** | LLM enriches item/user features (tags, summaries, pseudo-ratings) from sparse metadata | Boosts existing models without changing architecture | Offline batch process; quality depends on prompt design | LLM-FeatureGen, TAG (LLM for cold-start) |
| **LLM as Controller** | LLM decides which sub-system to invoke (CF, content-based, bandit) per query | Adaptive routing; handles multi-domain | Complex pipeline orchestration | RecAgent, CoPAL |

### Production Considerations for LLM-Based Recommendation

| Challenge | Mitigation |
|-----------|------------|
| **Latency** | Use speculative decoding; run LLM asynchronously in a batch-inference server (vLLM, TGI); embed-based retrieval pre-computed offline |
| **Cost** | Rank only the top-50 candidates from a cheaper first-stage retriever; distill LLM scores into a lightweight model |
| **Hallucination** | Constrain output format (regex, JSON schema); use `logits_processors` to force numeric-only generation |
| **Prompt injection** | Sanitise item titles that may contain adversarial text; sandbox LLM via strict system prompt |
| **Evaluation** | Use LLM-as-judge (GPT-4, Claude) to evaluate recommendation LLM outputs; human A/B test for top-line metrics |

---

## 19. Cross-References

| Reference | Description |
|-----------|-------------|
| [01-Foundations/07-Graph-Neural-Networks.md] | GNN for collaborative filtering, PinSage, LightGCN |
| [01-Foundations/04-Data-Engineering.md] | Feature pipelines, data quality, streaming |
| [05-Enterprise/01-Enterprise-AI-Deployment.md] | Production serving, A/B testing, monitoring |
| [10-Industry/01-AI-Industry-Applications.md] | Industry recsys case studies (Netflix, Amazon, Spotify) |
| [06-Advanced/03-Evaluation-Benchmarks.md] | Recommendation benchmarks (ML-20M, Netflix Prize) |
| [06-Advanced/01-Multimodal-AI.md] | Multimodal recommendations (images + text + audio) |
| [08-Reference/01-Glossary.md] | Key recsys terms |

---
*Document version: 2.0 — June 2026 | Tier 2-3: Gap Fill | Expanded with code examples, session-based recs, cold start, biases, and production architecture*

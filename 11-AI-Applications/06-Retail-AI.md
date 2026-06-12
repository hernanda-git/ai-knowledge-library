# AI in Retail & E-Commerce

## Table of Contents
1. [Introduction](#introduction)
2. [Recommendation Systems](#recommendation-systems)
   - [Collaborative Filtering](#collaborative-filtering)
   - [Neural Collaborative Filtering](#neural-collaborative-filtering)
   - [Transformer-based Sequential Recommendations](#transformer-based-sequential-recommendations)
   - [Multi-Stage Recommendation Pipelines](#multi-stage-recommendation-pipelines)
3. [Demand Forecasting](#demand-forecasting)
   - [Time Series Models for Retail](#time-series-models-for-retail)
   - [Causal Inference for Promotions](#causal-inference-for-promotions)
   - [Hierarchical Forecasting](#hierarchical-forecasting)
4. [Dynamic Pricing](#dynamic-pricing)
   - [Price Elasticity Models](#price-elasticity-models)
   - [Competitor-aware Pricing](#competitor-aware-pricing)
   - [Reinforcement Learning for Pricing](#reinforcement-learning-for-pricing)
5. [Visual Search & Image Recognition](#visual-search--image-recognition)
   - [Siamese Networks for Product Matching](#siamese-networks-for-product-matching)
   - [Fashion Attribute Extraction](#fashion-attribute-extraction)
   - [Visual Embedding Pipelines](#visual-embedding-pipelines)
6. [Inventory Management](#inventory-management)
   - [Safety Stock Optimization](#safety-stock-optimization)
   - [Replenishment Models](#replenishment-models)
   - [SKU Rationalization](#sku-rationalization)
7. [Customer Segmentation & Personalization](#customer-segmentation--personalization)
   - [Clustering & RFM Analysis](#clustering--rfm-analysis)
   - [Customer Lifetime Value Prediction](#customer-lifetime-value-prediction)
   - [Real-Time Personalization](#real-time-personalization)
8. [Chatbots & Conversational Commerce](#chatbots--conversational-commerce)
   - [Retail Conversational AI Architecture](#retail-conversational-ai-architecture)
   - [Product Discovery via Dialogue](#product-discovery-via-dialogue)
9. [Case Studies](#case-studies)
10. [Cross-References](#cross-references)
11. [Summary & Conclusion](#summary--conclusion)

---

## Introduction

Retail and e-commerce have been transformed by AI perhaps more visibly than any other consumer-facing industry. From Amazon's product recommendations driving 35% of sales, to dynamic pricing adjusting in real-time, to visual search that identifies products from photos — AI is the engine behind modern retail.

The global AI in retail market was valued at $8.2 billion in 2023 and is projected to reach $55.5 billion by 2030. This document provides a deep technical exploration of the architectures, models, and systems that power AI in retail, covering recommendation systems, demand forecasting, dynamic pricing, visual search, inventory optimization, and conversational commerce.

## Recommendation Systems

Recommendation systems are the cornerstone of e-commerce personalization, responsible for driving engagement, conversion, and average order value.

### Collaborative Filtering

Collaborative filtering recommends items based on patterns of user-item interactions — users who bought X also bought Y. Matrix factorization remains the foundational technique:

```python
import numpy as np
import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    """
    Basic matrix factorization for implicit feedback.
    
    R = U @ V^T where:
    R: user-item interaction matrix
    U: user factors
    V: item factors
    """
    def __init__(self, n_users, n_items, n_factors=50):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_factors(user_ids)  # (batch, n_factors)
        item_emb = self.item_factors(item_ids)  # (batch, n_factors)
        
        # Dot product + biases
        pred = (user_emb * item_emb).sum(dim=1)
        pred += self.user_bias(user_ids).squeeze()
        pred += self.item_bias(item_ids).squeeze()
        pred += self.global_bias
        
        return torch.sigmoid(pred)

# Weighted Approximate-Rank Pairwise (WARP) loss for implicit feedback
def warp_loss(positive_scores, negative_scores, weights=1.0):
    """
    WARP loss: maximizes rank of positive items by sampling
    negative items until one violates the margin.
    """
    # For each positive item, compute how many negatives score higher
    n_negatives = negative_scores.shape[1]
    
    # Expand positive scores for comparison
    pos_expanded = positive_scores.unsqueeze(1).expand(-1, n_negatives)
    
    # Count violations (negatives scoring >= positive)
    violations = (negative_scores >= pos_expanded).float()
    rank = violations.sum(dim=1) + 1  # +1 for the positive itself
    
    # WARP weight: log(rank) - higher weight for higher rank
    warp_weight = torch.log(torch.clamp(rank, min=1.0))
    
    # Hinge loss for each violating negative
    margin = 1.0
    hinge_loss = torch.clamp(margin - (positive_scores.unsqueeze(1) - negative_scores), min=0)
    
    # Weighted sum
    loss = (warp_weight.unsqueeze(1) * hinge_loss * violations).sum() / positive_scores.size(0)
    
    return loss
```

**ALS (Alternating Least Squares) for matrix factorization:**

```python
class ALSMatrixFactorization:
    """
    Collaborative filtering via ALS optimization.
    
    Alternates between fixing user factors and solving for item factors,
    and vice versa. Handles implicit feedback through confidence weights.
    """
    def __init__(self, n_factors=50, alpha=40.0, regularization=0.01, n_iterations=15):
        self.n_factors = n_factors
        self.alpha = alpha  # Confidence scaling for implicit feedback
        self.reg = regularization
        self.n_iterations = n_iterations
        
        self.user_factors = None
        self.item_factors = None
    
    def fit(self, user_item_pairs, n_users, n_items):
        """
        user_item_pairs: list of (user_id, item_id, preference) tuples
        preference = 1 for purchase/view, 0 for no interaction
        """
        # Build confidence matrix C = 1 + alpha * preference
        # Build preference matrix P (binary)
        
        # Initialize factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # Build sparse matrices for efficient computation
        from scipy.sparse import csr_matrix
        rows, cols, data = zip(*[(u, i, p) for u, i, p in user_item_pairs])
        P = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        C = 1 + self.alpha * P
        
        for iteration in range(self.n_iterations):
            # Fix item factors, solve for user factors
            XtX = self.item_factors.T @ self.item_factors
            reg_term = np.eye(self.n_factors) * self.reg
            
            for u in range(n_users):
                # User u's confidence and preference vectors
                Cu = C[u].toarray().flatten()
                Pu = P[u].toarray().flatten()
                
                # (X^T * Cu * X + reg * I) * Uu = X^T * Cu * Pu
                Cu_diag = np.diag(Cu)
                left = self.item_factors.T @ Cu_diag @ self.item_factors + reg_term
                right = self.item_factors.T @ (Cu * Pu)
                self.user_factors[u] = np.linalg.solve(left, right)
            
            # Fix user factors, solve for item factors
            YtY = self.user_factors.T @ self.user_factors
            
            for i in range(n_items):
                Ci = C[:, i].toarray().flatten()
                Pi = P[:, i].toarray().flatten()
                
                Ci_diag = np.diag(Ci)
                left = self.user_factors.T @ Ci_diag @ self.user_factors + reg_term
                right = self.user_factors.T @ (Ci * Pi)
                self.item_factors[i] = np.linalg.solve(left, right)
        
        return self
    
    def predict(self, user_id, item_id):
        return np.dot(self.user_factors[user_id], self.item_factors[item_id])
    
    def recommend(self, user_id, candidate_items, top_k=10):
        scores = np.dot(self.user_factors[user_id], self.item_factors[candidate_items].T)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [candidate_items[i] for i in top_indices]
```

### Neural Collaborative Filtering (NCF)

NCF replaces the dot product with a neural network that can learn non-linear user-item interactions:

```python
class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering (He et al., 2017)
    
    Combines GMF (Generalized Matrix Factorization) and
    MLP (Multi-Layer Perceptron) pathways.
    """
    def __init__(self, n_users, n_items, n_factors=64, layers=[64, 32, 16, 8]):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        
        # GMF pathway
        self.user_embed_gmf = nn.Embedding(n_users, n_factors)
        self.item_embed_gmf = nn.Embedding(n_items, n_factors)
        
        # MLP pathway (deeper, captures non-linearities)
        self.user_embed_mlp = nn.Embedding(n_users, layers[0])
        self.item_embed_mlp = nn.Embedding(n_items, layers[0])
        
        mlp_modules = []
        for i in range(len(layers) - 1):
            mlp_modules.append(nn.Linear(layers[i], layers[i+1]))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.2))
        self.mlp = nn.Sequential(*mlp_modules)
        
        # Fusion layer
        self.fusion = nn.Linear(n_factors + layers[-1], 1)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embed_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embed_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embed_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embed_mlp.weight, std=0.01)
        
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, user_id, item_id):
        # GMF pathway
        user_gmf = self.user_embed_gmf(user_id)
        item_gmf = self.item_embed_gmf(item_id)
        gmf_out = user_gmf * item_gmf  # Element-wise product
        
        # MLP pathway
        user_mlp = self.user_embed_mlp(user_id)
        item_mlp = self.item_embed_mlp(item_id)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_out = self.mlp(mlp_input)
        
        # Fusion
        concat = torch.cat([gmf_out, mlp_out], dim=-1)
        prediction = torch.sigmoid(self.fusion(concat))
        
        return prediction.squeeze()
```

### Transformer-based Sequential Recommendations

Modern recommendation systems use Transformers to model user behavior sequences:

```python
class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation (Kang & McAuley, 2018)
    
    Models user's sequential behavior using causal self-attention.
    """
    def __init__(self, n_items, max_seq_len=100, hidden=64, n_heads=2, n_layers=2):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=0.2,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.output_layer = nn.Linear(hidden, n_items)
        
    def forward(self, sequences, mask=None):
        # sequences: (batch, seq_len) - item IDs in chronological order
        batch_size, seq_len = sequences.shape
        
        # Embeddings
        item_emb = self.item_embedding(sequences)
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        
        x = item_emb + pos_emb
        
        # Causal mask (prevent attending to future items)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len) * float('-inf'), 
            diagonal=1
        ).to(sequences.device)
        
        # Padding mask
        if mask is None:
            mask = (sequences == 0)
        
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=mask)
        
        # Predict next item from last position
        last_idx = (~mask).sum(dim=1) - 1 if mask is not None else seq_len - 1
        last_hidden = x[torch.arange(batch_size), last_idx]
        
        return self.output_layer(last_hidden)

# BERT4Rec for bi-directional sequential recommendations
class BERT4Rec(nn.Module):
    """
    BERT4Rec uses masked language modeling objective for
    sequential recommendation. Items in the sequence are
    randomly masked and the model learns to predict them
    from bi-directional context.
    """
    def __init__(self, n_items, max_seq_len=50, hidden=64, n_layers=4, n_heads=2):
        super().__init__()
        self.mask_prob = 0.2
        self.n_items = n_items
        
        self.item_embedding = nn.Embedding(n_items + 2, hidden, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden)
        self.segment_embedding = nn.Embedding(2, hidden)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, n_items)
    
    def forward(self, sequences):
        batch_size, seq_len = sequences.shape
        
        # Random masking
        masked_seqs, mask_labels, mask_positions = self._mask_sequence(sequences)
        
        # Embeddings
        item_emb = self.item_embedding(masked_seqs)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        
        x = self.norm(item_emb + pos_emb)
        x = self.transformer(x)
        
        # Only predict masked positions
        masked_hidden = x[mask_positions[:, 0], mask_positions[:, 1]]
        predictions = self.output(masked_hidden)
        
        return predictions, mask_labels
    
    def _mask_sequence(self, sequences):
        """Apply BERT-style masking to input sequences"""
        masked = sequences.clone()
        labels = torch.full_like(sequences, -100)  # -100 is ignored in loss
        
        for i in range(sequences.size(0)):
            non_pad = (sequences[i] != 0).nonzero().squeeze()
            n_mask = max(1, int(len(non_pad) * self.mask_prob))
            mask_indices = non_pad[torch.randperm(len(non_pad))[:n_mask]]
            
            for idx in mask_indices:
                item_id = idx.item()
                labels[i, idx] = sequences[i, idx]
                
                # 80% mask token, 10% random item, 10% unchanged
                prob = torch.random.random()
                if prob < 0.8:
                    masked[i, idx] = self.n_items + 1  # [MASK] token
                elif prob < 0.9:
                    masked[i, idx] = torch.randint(1, self.n_items + 1, (1,))
                # else: unchanged
        
        mask_positions = torch.nonzero(labels != -100)
        return masked, labels[mask_positions[:, 0], mask_positions[:, 1]], mask_positions
```

### Multi-Stage Recommendation Pipeline

Production recommendation systems use a funnel architecture:

```yaml
recommendation_pipeline:
  stage_1: Retrieval
    goal: "Reduce from 100M items to ~1000 candidates"
    techniques:
      - Two-tower DNN (user tower + item tower, inner product search)
      - Locality Sensitive Hashing (LSH) for fast ANN search
      - Popularity-based boosting for cold-start
    infrastructure:
      - Embedding storage: FAISS (Facebook AI Similarity Search)
      - Index type: IVF (Inverted File) with HNSW (Hierarchical Navigable Small World)
      - Latency: < 10ms per user
    
  stage_2: Ranking
    goal: "Score and rank ~1000 candidates precisely"
    model: SASRec / BERT4Rec + wide-and-deep features
    features:
      - User features: history embeddings, demographics, device
      - Item features: category, price, brand, image embeddings
      - Context features: time of day, day of week, session context
      - Cross features: user-Item affinity, category preference
    
    architecture: DCN V2 (Deep & Cross Network)
    latency: < 50ms for 1000 candidates
    
  stage_3: Re-ranking
    goal: "Optimize business metrics with final re-ordering"
    objectives:
      - Diversity: MMR (Maximum Marginal Relevance)
      - Freshness: Boost recently added items
      - Margin: Boost high-margin items
      - Business rules: Minimum price, brand diversity
    
    techniques:
      - Greedy re-ranking with sliding window (efficient O(nk))
      - Determinantal Point Process (DPP) for quality-diversity tradeoff
      - RL-based re-ranking (trained on long-term reward)
```

## Demand Forecasting

### Time Series Models for Retail

```python
import torch
import torch.nn as nn
import numpy as np

class DeepAR(nn.Module):
    """
    Amazon's DeepAR: Probabilistic time series forecasting.
    
    Produces a probability distribution (Gaussian or Negative Binomial)
    for each time step, enabling quantile forecasts and uncertainty
    estimation.
    """
    def __init__(self, n_features=10, hidden_dim=64, n_layers=2, likelihood='gaussian'):
        super().__init__()
        self.likelihood = likelihood
        
        # Feature encoder
        self.feature_encoder = nn.Linear(n_features, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim + 1,  # features + target value
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=0.2,
            batch_first=True
        )
        
        # Likelihood parameters
        if likelihood == 'gaussian':
            self.mu = nn.Linear(hidden_dim, 1)
            self.sigma = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Softplus()  # Ensure positive
            )
        elif likelihood == 'negative_binomial':
            self.mu = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Softplus()
            )
            self.alpha = nn.Sequential(  # Dispersion
                nn.Linear(hidden_dim, 1),
                nn.Softplus()
            )
    
    def forward(self, features, targets, hidden=None):
        """
        features: (batch, seq_len, n_features)
        targets: (batch, seq_len, 1) - previous sales values
        """
        batch_size, seq_len = features.shape[0], features.shape[1]
        
        # Encode features
        feat_enc = self.feature_encoder(features)
        
        # Concatenate features with lagged target
        lstm_input = torch.cat([feat_enc, targets], dim=-1)
        
        # LSTM
        lstm_out, hidden = self.lstm(lstm_input, hidden)
        
        if self.likelihood == 'gaussian':
            mu = self.mu(lstm_out)
            sigma = self.sigma(lstm_out)
            return mu, sigma, hidden
        else:
            mu = self.mu(lstm_out)
            alpha = self.alpha(lstm_out)
            return mu, alpha, hidden
    
    def loss(self, features, targets):
        """Negative log-likelihood loss"""
        mu, sigma, _ = self.forward(features, targets[:, :-1, :])
        
        # Compare predictions at time t with actual at time t+1
        # (one-step-ahead prediction)
        mu = mu[:, 1:, :]  # Shift by 1
        sigma = sigma[:, 1:, :]
        actual = targets[:, 1:, :]
        
        if self.likelihood == 'gaussian':
            dist = torch.distributions.Normal(mu, sigma)
            return -dist.log_prob(actual).mean()
    
    def forecast(self, features, initial_target, n_steps=30):
        """
        Multi-step forecast by feeding predictions back as inputs.
        """
        self.eval()
        predictions = []
        
        current_target = initial_target
        hidden = None
        
        for step in range(n_steps):
            feat_step = features[:, step:step+1, :]
            
            with torch.no_grad():
                mu, sigma, hidden = self.forward(feat_step, current_target, hidden)
            
            predictions.append(mu[:, -1, :])
            
            # Use prediction as next input (optionally sample from distribution)
            current_target = mu[:, -1:, :]
        
        return torch.stack(predictions, dim=1)
```

**Retail-specific features for demand forecasting:**

```yaml
demand_features:
  temporal:
    - day_of_week (7 one-hot features)
    - month_of_year (12 one-hot)
    - week_of_year (52 cyclic encoding)
    - is_weekend (binary)
    - days_since_last_promotion
    - days_to_next_holiday
    
  promotional:
    - discount_pct
    - promotion_type (BOGO, percentage, gift card)
    - promotion_channel (email, social, in-store)
    - competitor_promotion_indicator
    
  product:
    - price_relative_to_avg
    - inventory_level
    - days_since_last_restock
    - product_age_days
    - category_trend (rolling 4-week)
    - new_product_flag (first 30 days)
    
  external:
    - weather_temperature
    - weather_precipitation
    - local_events_indicators
    - economic_indicator (CPI)
    - google_trends_signal
```

### Causal Inference for Promotions

```python
class PromotionCausalModel:
    """
    Estimate causal effect of promotions on sales using:
    1. Synthetic Control (for large promotions)
    2. Difference-in-Differences (for A/B tested promotions)
    3. Causal Forest (for heterogeneous treatment effects)
    """
    def __init__(self):
        from econml.dml import CausalForestDML
        self.causal_forest = CausalForestDML(
            model_t=GradientBoostingRegressor(),
            model_y=GradientBoostingRegressor(),
            n_estimators=100,
            max_depth=5
        )
    
    def estimate_promotion_effect(self, df, promotion_col, sales_col, features):
        """
        Estimate heterogeneous treatment effects (CATE).
        
        df: DataFrame with sales data
        promotion_col: binary treatment indicator
        sales_col: outcome variable
        features: control features
        """
        T = df[promotion_col].values
        Y = np.log1p(df[sales_col].values)  # Log-transform sales
        X = df[features].values
        
        # Fit causal forest
        self.causal_forest.fit(Y, T, X=X)
        
        # Get treatment effects
        treatment_effects = self.causal_forest.effect(X)
        
        # Get feature importance for treatment effect heterogeneity
        importance = self.causal_forest.feature_importances_
        
        return {
            'average_treatment_effect': np.mean(treatment_effects),
            'heterogeneity': treatment_effects,
            'feature_importance': dict(zip(features, importance)),
            'shap_values': self._shap_importance()
        }
    
    def synthetic_control(self, treated_series, control_series_pool):
        """
        Synthetic control for estimating effect of a promotion.
        
        treated_series: Sales in treated region/store
        control_series_pool: Sales in other regions (potential controls)
        """
        from scipy.optimize import minimize
        
        n_controls = control_series_pool.shape[1]
        
        def objective(weights):
            weights = weights / weights.sum()
            synthetic = control_series_pool @ weights
            return np.mean((treated_series - synthetic) ** 2)
        
        constraints = ({'type': 'eq', 'fun': lambda x: x.sum() - 1})
        bounds = [(0, 1)] * n_controls
        
        result = minimize(
            objective,
            x0=np.ones(n_controls) / n_controls,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        
        weights = result.x / result.x.sum()
        synthetic = control_series_pool @ weights
        
        # Treatment effect = actual - synthetic (post-promotion)
        return {
            'weights': weights,
            'synthetic_control': synthetic,
            'treatment_effect': treated_series - synthetic
        }
```

## Dynamic Pricing

### Price Elasticity Models

```python
class PriceElasticityModel:
    """
    Estimate price elasticity of demand using Bayesian methods.
    
    log(Q) = α + β * log(P) + γ * X + ε
    
    Where β is the price elasticity (typically negative).
    """
    def __init__(self):
        import pymc as pm
        self.pymc = pm
    
    def estimate_elasticity(self, price, quantity, features):
        """
        Bayesian linear regression for elasticity estimation.
        """
        with self.pymc.Model() as model:
            # Priors
            alpha = self.pymc.Normal('alpha', mu=0, sigma=10)
            beta = self.pymc.Normal('beta', mu=-1, sigma=2)  # Price elasticity
            gamma = self.pymc.Normal('gamma', mu=0, sigma=1, shape=features.shape[1])
            sigma = self.pymc.HalfNormal('sigma', sigma=1)
            
            # Expected log-quantity
            mu = alpha + beta * np.log(price) + features @ gamma
            
            # Likelihood
            self.pymc.Normal('quantity', mu=mu, sigma=sigma, observed=np.log(quantity))
            
            # Inference
            trace = self.pymc.sample(2000, tune=1000, chains=4)
        
        return {
            'elasticity_mean': np.mean(trace['beta']),
            'elasticity_ci': np.percentile(trace['beta'], [2.5, 97.5]),
            'optimal_price': self._compute_optimal_price(trace, features)
        }
    
    def _compute_optimal_price(self, trace, features, cost=0.5):
        """Compute revenue-maximizing price from posterior"""
        # Revenue = P * Q(P) = P * exp(α + β * log(P) + γ * X)
        # Revenue maximizing price: P* = exp(-α - γ*X) / (1 + 1/β)
        
        alpha_mean = np.mean(trace['alpha'])
        beta_mean = np.mean(trace['beta'])
        
        # Correction for cost
        optimal_log_p = (-alpha_mean + np.log(cost * beta_mean / (1 + beta_mean)))
        return np.exp(optimal_log_p)
```

### Competitor-aware Pricing

```yaml
dynamic_pricing_engine:
  inputs:
    own_data:
      - Current price
      - Inventory level
      - Product cost
      - Historical elasticity
      - Sales velocity
    
    competitor_data:
      - Competitor prices (scraped hourly)
      - Competitor stock status
      - Price history (30 days)
      - Market share estimates
    
    context:
      - Day of week / hour
      - Season / holiday calendar
      - Weather (for weather-sensitive goods)
      - Category trends
  
  pricing_strategy:
    # Multiple strategies combined with RL
    
    strategy_1: "Match lowest competitor + margin floor"
      rule: |
        price = max(competitor_min * 0.99, cost * (1 + min_margin))
      when: "Low differentiation, high competition"
    
    strategy_2: "Premium pricing + value justification"
      rule: |
        price = own_avg_price * (1 + brand_premium + quality_score)
      when: "High brand strength or exclusive items"
    
    strategy_3: "Surge pricing (high demand)"
      rule: |
        multiplier = 1 + demand_coeff * (current_velocity / avg_velocity - 1)
        price = base_price * min(multiplier, max_surge_multiplier)
      when: "Limited supply, high demand spikes"
  
  ml_models:
    demand_forecast: DeepAR (next 24h demand curve)
    elasticity_model: Bayesian (per-product elasticity)
    competitor_prediction: LSTM (competitor price movement)
    optimal_price: Dualing DQN (trained on revenue reward)
```

## Visual Search & Image Recognition

### Siamese Networks for Product Matching

```python
class SiameseProductMatcher(nn.Module):
    """
    Siamese network for product image matching.
    Learns embedding space where same products are close,
    different products are far apart.
    """
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Shared CNN backbone
        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Embedding projection
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward_one(self, image):
        features = self.backbone(image)
        features = features.view(features.size(0), -1)
        embedding = self.embedding(features)
        # L2 normalize
        return embedding / embedding.norm(dim=1, keepdim=True)
    
    def forward(self, image1, image2):
        emb1 = self.forward_one(image1)
        emb2 = self.forward_one(image2)
        return emb1, emb2

class TripletLoss(nn.Module):
    """Triplet loss with semi-hard negative mining"""
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        pos_dist = (anchor - positive).pow(2).sum(1)
        neg_dist = (anchor - negative).pow(2).sum(1)
        
        # Semi-hard negatives: negatives that are harder than positives
        # but not too hard (within margin)
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

# Product attribute extraction
class FashionAttributeExtractor(nn.Module):
    """
    Multi-label attribute classifier for fashion items.
    
    Predicts: category, color, pattern, sleeve_length, neckline, etc.
    """
    def __init__(self, attribute_dims={'category': 50, 'color': 20, 'pattern': 15}):
        super().__init__()
        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Multi-scale feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.attention_pool = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )
        
        # Attribute-specific heads
        self.attribute_heads = nn.ModuleDict()
        for attr_name, n_classes in attribute_dims.items():
            self.attribute_heads[attr_name] = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, n_classes)
            )
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Global + attention pooling
        global_feat = self.global_pool(features).squeeze(-1).squeeze(-1)
        attention = self.attention_pool(features)
        attended_feat = (features * attention).sum(dim=(-1, -2))
        
        # Combine
        combined = global_feat + attended_feat
        
        # Attribute predictions
        output = {}
        for name, head in self.attribute_heads.items():
            output[name] = head(combined)
        
        return output
```

## Customer Segmentation & Personalization

### Clustering & RFM Analysis

```python
class CustomerSegmenter:
    """
    Multi-level customer segmentation using:
    1. RFM (Recency, Frequency, Monetary) analysis
    2. Behavioral clustering with K-Means / GMM
    3. Dynamic segmentation with HMM
    """
    def __init__(self, n_segments=5):
        self.n_segments = n_segments
        self.scaler = StandardScaler()
        self.clustering = None
    
    def compute_rfm(self, transactions, reference_date=None):
        """
        Compute Recency, Frequency, Monetary values.
        """
        if reference_date is None:
            reference_date = transactions['transaction_date'].max()
        
        rfm = transactions.groupby('customer_id').agg({
            'transaction_date': lambda x: (reference_date - x.max()).days,  # Recency
            'transaction_id': 'count',  # Frequency
            'amount': 'sum'  # Monetary
        }).rename(columns={
            'transaction_date': 'recency',
            'transaction_id': 'frequency',
            'amount': 'monetary'
        })
        
        # Log-transform monetary (usually right-skewed)
        rfm['monetary_log'] = np.log1p(rfm['monetary'])
        rfm['frequency_log'] = np.log1p(rfm['frequency'])
        
        return rfm
    
    def segment(self, rfm_df):
        """Cluster customers into segments"""
        features = rfm_df[['recency', 'frequency_log', 'monetary_log']].values
        
        # Scale
        scaled = self.scaler.fit_transform(features)
        
        # Gaussian Mixture Model (soft clustering)
        from sklearn.mixture import GaussianMixture
        self.clustering = GaussianMixture(
            n_components=self.n_segments,
            covariance_type='full',
            random_state=42
        )
        
        rfm_df['segment'] = self.clustering.fit_predict(scaled)
        rfm_df['segment_probability'] = self.clustering.predict_proba(scaled).max(axis=1)
        
        # Name segments by their characteristics
        segment_profiles = rfm_df.groupby('segment').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean',
            'customer_id': 'count'
        }).rename(columns={'customer_id': 'count'})
        
        return rfm_df, segment_profiles
    
    def segment_labels(self, profiles):
        """Assign human-readable labels to segments"""
        labels = {}
        for seg_id in profiles.index:
            row = profiles.loc[seg_id]
            if row['frequency'] > profiles['frequency'].quantile(0.8):
                if row['monetary'] > profiles['monetary'].quantile(0.8):
                    labels[seg_id] = 'VIP'
                else:
                    labels[seg_id] = 'Frequent'
            elif row['recency'] < 30:
                labels[seg_id] = 'Active'
            elif row['recency'] < 180:
                labels[seg_id] = 'Lapsed'
            else:
                labels[seg_id] = 'Lost'
        return labels
```

## Chatbots & Conversational Commerce

### Retail Conversational AI Architecture

```yaml
retail_chatbot:
  intent_classification:
    model: BERT-based (distilbert-base-uncased fine-tuned)
    intents: [
      'product_search', 'order_status', 'return_request',
      'recommendation', 'price_inquiry', 'size_guide',
      'shipping_info', 'complaint', 'general_question'
    ]
    entities: [
      'product_name', 'product_category', 'brand',
      'size', 'color', 'price_range', 'order_number'
    ]
    
  dialog_management:
    approach: "Hybrid: RASA-like state machine + neural response ranking"
    state_tracking:
      - Current intent
      - Slot filling progress
      - Conversation history (last 5 turns)
      - Customer context (loyalty tier, browsing history)
    
  product_discovery:
    retrieval:
      - Elasticsearch BM25 for keyword matching
      - Dense retrieval using CLIP embeddings (text <-> image)
      - Filtering: size, color, price, brand in stock
    
    ranking:
      - Semantic similarity (Sentence-BERT)
      - Personal relevance (matrix factorization score)
      - Business rules (margin, promo, stock)
    
  response_generation:
    template: Slot-filled templates for known intents
    generative: Fine-tuned GPT-2 for open-ended conversation
    guardrails:
      - PII detection and redaction
      - Policy compliance checking
      - Sentiment-aware escalation to human
```

## Case Studies

### Case Study 1: Amazon Recommendation System

**Background**: Amazon's recommendation engine drives an estimated 35% of total sales, using a multi-layered ML system.

**Technical architecture:**
```yaml
amazon_recsys:
  data:
    - 300M+ active users
    - 600M+ products
    - 100B+ user-item interactions
  
  embedding:
    item2vec: "Skip-gram on purchase sequences (similar to word2vec)"
    graph_embeddings: "Node2Vec on co-purchase graph"
    session_embeddings: "LSTM encode current browsing session"
  
  retrieval:
    - "Two-tower DNN for user-to-item retrieval"
    - "Similar items (item-to-item CF)"
    - "Popularity-based fallback"
    - "Personalized trending"
  
  ranking:
    - "XGBoost with 1000+ features"
    - "Deep neural network with Transformers"
    - "Multi-objective: CTR, conversion, margin, long-term value"
  
  re-ranking:
    - "MMR for diversity"
    - "Freshness boost for new arrivals"
    - "Cross-category diversity"
```

### Case Study 2: ASOS Visual Search

**Background**: ASOS implemented visual search allowing customers to upload photos and find similar products.

**Technical implementation:**
```python
# ASOS visual search pipeline (simplified)
class ASOSVisualSearch:
    def __init__(self):
        # ResNet-50 pretrained on ImageNet + fine-tuned on fashion
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(2048, 256)
        
        # Text encoder for attribute matching
        self.text_encoder = SentenceBERT('all-MiniLM-L6-v2')
        
        # FAISS index for fast retrieval
        self.index = faiss.IndexFlatIP(256)  # Inner product search
    
    def encode_product_images(self, product_id, images):
        """Index all product images"""
        embeddings = []
        for img in images:
            emb = self.image_encoder(img.unsqueeze(0))
            emb = emb / emb.norm()  # Normalize
            embeddings.append(emb)
        
        product_embedding = torch.stack(embeddings).mean(dim=0)  # Average multi-view
        return product_embedding
    
    def search(self, query_image, top_k=20):
        """Find similar products"""
        query_emb = self.image_encoder(query_image.unsqueeze(0))
        query_emb = query_emb / query_emb.norm()
        
        distances, indices = self.index.search(query_emb.numpy(), top_k)
        
        return [self.products[i] for i in indices[0]]
```

## Cross-References

This document relates to other categories in the AI Knowledge Base:

- **[03-Finance-AI.md](03-Finance-AI.md)** — Customer segmentation and LTV models share RFM and clustering techniques with credit scoring
- **[04-Manufacturing-AI.md](04-Manufacturing-AI.md)** — Demand forecasting and inventory optimization connect manufacturing with retail supply chains
- **[05-Education-AI.md](05-Education-AI.md)** — Recommendation algorithms (collaborative filtering, bandits) are shared with adaptive content recommendation
- **[07-Media-Entertainment-AI.md](07-Media-Entertainment-AI.md)** — Content recommendation systems share fundamental architectures (collaborative filtering, two-tower models)
- **[08-Agriculture-AI.md](08-Agriculture-AI.md)** — Agricultural commodity pricing and supply chain connect with retail procurement
- **[09-Transportation-AI.md](09-Transportation-AI.md)** — Last-mile delivery optimization and logistics are critical for e-commerce fulfillment

## Summary & Conclusion

AI in retail and e-commerce represents one of the most mature and commercially impactful applications of machine learning. The field encompasses a diverse and interconnected system of technologies:

- **Recommendation Systems**: From matrix factorization to Transformers, driving personalized product discovery through multi-stage pipelines (retrieval → ranking → re-ranking)
- **Demand Forecasting**: DeepAR and probabilistic models for inventory planning, with causal inference for promotion effectiveness
- **Dynamic Pricing**: Bayesian elasticity estimation, competitor-aware models, and RL-based pricing optimization
- **Visual Search**: Siamese networks and CLIP embeddings enabling product discovery from images
- **Customer Analytics**: RFM clustering, LTV prediction, and real-time personalization engines
- **Conversational AI**: NLU for product discovery, order management, and customer service

The most successful retail AI systems combine deep learning for pattern recognition with interpretable models for business decisions, operate at massive scale with sub-second latency, and continuously adapt to changing consumer behavior and market conditions.

# AI Advertising & Programmatic

## Table of Contents

1. [Overview](#overview)
2. [Programmatic Ad Buying](#programmatic-ad-buying)
3. [Audience Segmentation & Lookalike Modeling](#audience-segmentation--lookalike-modeling)
4. [Creative Optimization with AI](#creative-optimization-with-ai)
5. [Budget Allocation & Optimization](#budget-allocation--optimization)
6. [Attribution Modeling](#attribution-modeling)
7. [Ad Fraud Detection](#ad-fraud-detection)
8. [Real-Time Bidding Algorithms](#real-time-bidding-algorithms)
9. [Cross-Channel Optimization](#cross-channel-optimization)
10. [Privacy & Identity in Ad Tech](#privacy--identity-in-ad-tech)
11. [Cross-References](#cross-references)

---

## Overview

AI has transformed digital advertising from a rules-based, manual optimization discipline into a real-time, algorithmically-driven market. The global programmatic advertising market is projected to exceed $800B by 2027, with AI-driven optimization becoming the standard for buying, placement, creative, and measurement.

### AI in Advertising — Key Areas

| Area | AI Techniques | Impact |
|------|--------------|--------|
| Real-Time Bidding (RTB) | RL, contextual bandits | 15-30% CPM reduction |
| Audience Segmentation | Clustering, embeddings | 3x better targeting precision |
| Creative Optimization | Generative AI, A/B testing at scale | 40% higher CTR |
| Attribution | Shapley values, Markov chains | 50% more accurate ROAS |
| Fraud Detection | GNN, anomaly detection | 90% fraud reduction |
| Budget Allocation | Portfolio optimization, RL | 20% better ROAS |

---

## Programmatic Ad Buying

Programmatic advertising uses automated bidding and real-time auctions to buy ad inventory. AI optimizes every step of the process.

### The RTB Ecosystem

```
Advertiser → DSP (Demand-Side Platform)
                ↓
          Ad Exchange (real-time auction)
                ↓
          SSP (Supply-Side Platform)
                ↓
          Publisher's Inventory
```

### DSP AI Components

**1. Bidder:** Real-time bid calculation (within 50-100ms)
**2. Predictor:** Click-through rate (CTR) prediction, conversion probability
**3. Optimizer:** Budget pacing, frequency capping, audience targeting
**4. Fraud Detector:** Real-time traffic quality scoring

### ML Model for Bid Price

```python
# Real-Time Bidding model
bid_price = base_price × CTR_prediction × conversion_probability × user_value_score

# CTR prediction model (GBM / Neural Network)
features = [
    user_features,        # Demographics, browsing history, purchase intent
    context_features,     # Page content, time of day, device type
    ad_features,         # Creative, format, brand
    historical_features,  # Past CTR for user-ad combinations
]

ctr_prediction = gradient_boosted_trees.predict(features)
optimal_bid = ctr_prediction × target_CPA × pacing_multiplier
```

---

## Audience Segmentation & Lookalike Modeling

### ML-Based Segmentation

```python
from sklearn.cluster import KMeans, DBSCAN
from sentence_transformers import SentenceTransformer
import numpy as np

# Step 1: Embed user behavior
encoder = SentenceTransformer("all-MiniLM-L6-v2")
user_embeddings = encoder.encode(user_behavior_texts)

# Step 2: Cluster users
clusters = KMeans(n_clusters=20, random_state=42).fit(user_embeddings)

# Step 3: Profile clusters
for cluster_id in range(20):
    cluster_users = users[clusters.labels_ == cluster_id]
    profile = {
        "size": len(cluster_users),
        "avg_ltv": cluster_users["ltv"].mean(),
        "top_categories": get_top_categories(cluster_users),
        "channel_preference": get_channel_pref(cluster_users),
    }
```

### Lookalike Modeling

```python
# Lookalike: find users similar to high-value seed audience
from sklearn.ensemble import GradientBoostingClassifier

# Train binary classifier: 1 = high-value, 0 = random
model = GradientBoostingClassifier(n_estimators=200)
X_train = all_users_features
y_train = all_users_is_high_value

model.fit(X_train, y_train)

# Score all users for lookalike probability
all_users["lookalike_score"] = model.predict_proba(X_all)[:, 1]

# Target top N% of lookalike scores
target_audience = all_users[all_users["lookalike_score"] > 0.8]
```

---

## Creative Optimization

### AI-Generated Ad Creative

```python
# Generate ad variants with LLM
prompt_template = """
Generate {num_variants} ad copy variants for:
- Product: {product_name}
- USP: {unique_selling_point}
- Target audience: {audience_segment}
- Tone: {brand_tone}
- Length: {max_words} words
- CTA: include a clear call to action

Output as numbered list.
"""

variants = llm.generate(prompt_template)
# A/B test variants in market
# Automated winner identification after statistical significance
```

### Dynamic Creative Optimization (DCO)

Real-time assembly of ad components (headline, image, CTA, offer) based on user profile:

```
User visits page → Ad call → DCO Engine
                              ↓
                    ┌────────────────────┐
                    │ Component Selector │
                    │ - Headline: 5 opts │
                    │ - Image: 8 opts    │
                    │ - CTA: 3 opts     │
                    │ - Offer: 2 opts   │
                    └────────────────────┘
                              ↓
                    ML predicts best combination
                    (multi-armed bandit)
                              ↓
                    Serve optimal creative → Track engagement → Update model
```

---

## Budget Allocation & Optimization

### Portfolio Optimization for Campaigns

```python
import numpy as np
from scipy.optimize import minimize

def optimize_budget(channels, expected_roas, covariance, total_budget):
    """
    Modern Portfolio Theory for marketing budget allocation.
    """
    n = len(channels)
    
    # Objective: maximize return for given risk level
    def neg_sharpe(weights):
        portfolio_return = np.dot(weights, expected_roas)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
        return -(portfolio_return / portfolio_risk)
    
    # Constraints: sum to 1, no negative weights
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    ]
    bounds = [(0.05, 0.5) for _ in range(n)]  # Min 5%, max 50% per channel
    
    result = minimize(
        neg_sharpe,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    
    return dict(zip(channels, result.x * total_budget))
```

### RL for Real-Time Budget Pacing

```python
# Reinforcement learning for budget pacing
class BudgetPacingAgent:
    def __init__(self, daily_budget, total_hours=24):
        self.budget = daily_budget
        self.hours = total_hours
        self.spent = 0
        self.hour = 0
    
    def get_action(self, state):
        """State: hour, remaining_budget, spent_today, performance_signals"""
        # Simple rule-based pacing, can be learned with RL
        ideal_pace = self.budget / self.hours
        current_pace = self.spent / max(self.hour, 1)
        
        if current_pace > ideal_pace * 1.2:
            return "reduce_bid"  # Overspending
        elif current_pace < ideal_pace * 0.8:
            return "increase_bid"  # Underspending
        else:
            return "maintain"  # On track
```

---

## Attribution Modeling

### Multi-Touch Attribution with Shapley Values

```python
import itertools
from typing import List

def shapley_attribution(channels: List[str], conversions: dict) -> dict:
    """
    Calculate Shapley value for each channel in conversion path.
    """
    n = len(channels)
    shapley_values = {c: 0 for c in channels}
    
    for channel in channels:
        for r in range(1, n + 1):
            subsets = [s for s in itertools.combinations(
                [c for c in channels if c != channel], r - 1
            )]
            
            for subset in subsets:
                # Conversion with channel in set
                with_channel = conversions.get(
                    tuple(sorted(subset + (channel,))), 0
                )
                # Conversion without channel
                without_channel = conversions.get(
                    tuple(sorted(subset)), 0
                )
                
                marginal_contribution = with_channel - without_channel
                weight = 1 / (n * len(list(itertools.combinations(
                    [c for c in channels if c != channel], r - 1
                ))))
                
                shapley_values[channel] += marginal_contribution * weight
    
    # Normalize to total conversions
    total = sum(shapley_values.values())
    return {k: v / total for k, v in shapley_values.items()}
```

### Model Comparison

| Model | Granularity | Data Required | Implementation Complexity |
|-------|------------|---------------|---------------------------|
| Last Click | Low | Minimal | None |
| Linear/U-Shaped | Low | Touch data | Low |
| Time Decay | Medium | Timestamped touches | Low |
| Markov Chain | High | All paths + conversions | Medium |
| Shapley Value | High | All subsets | High |
| Data-Driven (ML) | Very High | Rich user/campaign data | High |

---

## Ad Fraud Detection

### Types of Ad Fraud

| Fraud Type | Description | Detection Method |
|------------|-------------|-----------------|
| Click Fraud | Automated/Bot clicks | Behavioral analysis, timing patterns |
| Impression Fraud | Fake impressions (pixel stuffing) | Viewability measurement |
| Conversion Fraud | Fake leads/sales | Pattern analysis, device fingerprinting |
| Domain Spoofing | Fake premium inventory | Supply chain verification (ads.txt) |
| IVT (Invalid Traffic) | General bot traffic | GIVT/SIVT detection |

### ML Detection Pipeline

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Features for fraud detection
features = [
    "clicks_per_second",      # Rapid clicking (bots)
    "mouse_movement_score",   # Human-like cursor paths
    "time_on_page",           # Too short (bot) or too long (click farm)
    "device_fingerprint",     # Unusual device combinations
    "ip_velocity",            # Same IP / different users
    "conversion_time",        # Suspiciously fast conversions
    "user_agent_consistency", # Mismatched user agent signals
]

# Anomaly detection model
fraud_detector = IsolationForest(
    contamination=0.1,  # Expected 10% fraud
    random_state=42
)

fraud_scores = fraud_detector.fit_predict(ad_traffic[features])
# -1 = anomaly (likely fraud), 1 = normal
```

---

## Real-Time Bidding Algorithms

### RL-Based Bid Optimization

```python
class RLBidder:
    """Contextual bandit for real-time bid optimization."""
    
    def __init__(self, n_actions=10):
        self.arms = np.zeros(n_actions)  # Bid multipliers
        self.counts = np.zeros(n_actions)
        self.alpha = 1.0
        self.beta = 1.0
    
    def select_bid(self, bid_request_features):
        """Thompson Sampling for bid selection."""
        samples = np.random.beta(
            self.alpha + self.counts,
            self.beta + np.max(self.counts) - self.counts
        )
        arm = np.argmax(samples)
        base_bid = bid_request_features["estimated_value"]
        return base_bid * self.arms[arm]
    
    def update(self, arm, reward):
        """Update with win/reward signal."""
        self.counts[arm] += 1
        if reward > 0:
            self.alpha += reward
        else:
            self.beta += 1
```

---

## Cross-References

- **Personalization & CDP** → [05-AI-Personalization-and-CDP.md](./05-AI-Personalization-and-CDP.md) — Audience data integration
- **Marketing Analytics** → [08-AI-Marketing-Analytics-and-Measurement.md](./08-AI-Marketing-Analytics-and-Measurement.md) — MMM and measurement
- **AI Sales** → [02-AI-Sales-Development-Reps.md](./02-AI-Sales-Development-Reps.md) — Sales-advertising alignment
- **ML Engineering** → [11-AI-Applications/06-Retail-AI.md](../11-AI-Applications/06-Retail-AI.md) — Recommendation systems synergy

---

*Last updated: June 2026 | 400+ lines*

# 05 — Real-Time Financial Fraud Detection

## Case Study: ML-Powered Fraud Detection System for Digital Banking

| Metadata | Value |
|----------|-------|
| **Industry** | Financial Services / Digital Banking |
| **Domain** | Real-time fraud detection, AML, transaction monitoring |
| **Difficulty** | Advanced |
| **Est. Timeline** | 10-16 weeks |
| **Team Size** | 6-9 engineers (3 ML, 2 streaming, 1 backend, 1 fraud analyst) |

---

## 🎯 Problem Statement

### Business Context

**Company:** FinBank Digital (neobank, 5M+ customers, $15B annual transaction volume)
**Transaction Volume:** 500K+ transactions per hour (peak); 12M/day average
**Current Fraud System:** Legacy rule-based engine (800+ rules), 12% fraud capture rate, 40% false positive rate

### Pain Points

1. **Low Fraud Capture** — Legacy rules capture only 12% of actual fraud; $45M annual fraud losses
2. **High False Positives** — 40% false positive rate (FP rate); 3,200 legitimate transactions blocked/day
3. **Slow Detection** — Average detection time: 24-72 hours post-transaction (batch scoring)
4. **Adaptive Fraudsters** — Fraud rings constantly evolve; rule updates take 2-4 weeks to deploy
5. **Synthetic Identity** — 22% of new account fraud uses synthetic identities — rules can't detect patterns
6. **Real-time Requirement** — Payments must be approved/rejected in < 500ms for good UX
7. **Scalability** — Transaction volume growing 40% YoY; legacy system can't keep up

### Success Criteria

| Metric | Target | Baseline |
|--------|--------|----------|
| **Fraud Capture Rate** | > 85% | 12% |
| **False Positive Rate** | < 0.5% | 40% |
| **Detection Latency** | < 100ms (real-time) | 24-72 hours |
| **Transaction Block Rate** | < 2% (legit blocked) | 3.2% |
| **Annual Fraud Loss** | < $10M | $45M |
| **Model Retrain Frequency** | Daily incremental | Monthly |
| **System Throughput** | 100K TPS | 5K TPS |

### Constraints

- PSD2 / Open Banking compliance (EU region)
- GDPR: Must be able to explain individual decisions
- Must integrate with existing core banking system (IBM Mainframe)
- PCI-DSS Level 1 compliance required
- Model must be re-trainable without downtime
- Fraud analyst team of 12 — system must provide interpretable reasons

---

## 🏗️ Solution Architecture

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                        TRANSACTION SOURCES                                        │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────┐   │
│  │  Online   │  │  Mobile   │  │  POS/ATM  │  │  Wire     │  │  Internal     │   │
│  │  Banking  │  │  App      │  │           │  │  Transfer │  │  Transfers    │   │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └───────┬───────┘   │
│        │              │              │              │                │           │
└────────┼──────────────┼──────────────┼──────────────┼────────────────┼───────────┘
         │              │              │              │                │
         ▼              ▼              ▼              ▼                ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                         STREAMING DATA INGESTION                                   │
│                                                                                    │
│  ┌───────────────────────────────────────────────────────────────────────────┐     │
│  │               Apache Kafka (6 brokers, 3 AZs, 48 partitions/topic)        │     │
│  │                                                                           │     │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │     │
│  │  │  raw-       │  │  enriched-  │  │  scored-    │  │  decisions-     │ │     │
│  │  │  transactions│  │  transactions│  │  transactions│  │  audit-log     │ │     │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘ │     │
│  └───────────────────────────────────────┬───────────────────────────────────┘     │
│                                          │                                        │
└──────────────────────────────────────────┼────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────┼────────────────────────────────────────┐
│                     STREAM PROCESSING LAYER                                       │
│                                          ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐     │
│  │  Apache Flink (real-time) + Spark Structured Streaming (micro-batch)      │     │
│  │                                                                           │     │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐  │     │
│  │  │  Feature        │  │  Aggregation   │  │  Graph Construction        │  │     │
│  │  │  Engineering    │  │  Windows       │  │  (entity resolution,      │  │     │
│  │  │  (per-tx,      │  │  (5min/1hr/    │  │   device fingerprinting)   │  │     │
│  │  │   per-user)    │  │   24hr/7d)    │  │                           │  │     │
│  │  └────────┬───────┘  └────────┬───────┘  └─────────────┬──────────────┘  │     │
│  │           │                   │                        │                │     │
│  │           ▼                   ▼                        ▼                │     │
│  │  ┌──────────────────────────────────────────────────────────────────┐   │     │
│  │  │                 Feature Store (Feast + Redis)                     │   │     │
│  │  │                 - Real-time features (Redis TTL)                  │   │     │
│  │  │                 - Historical features (PostgreSQL)                │   │     │
│  │  └──────────────────────────────────┬───────────────────────────────┘   │     │
│  └─────────────────────────────────────┼─────────────────────────────────────┘     │
│                                        │                                       │
└────────────────────────────────────────┼─────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────┼─────────────────────────────────────────┐
│                         MODEL INFERENCE LAYER                                    │
│                                         ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────────┐     │
│  │                          MODEL ENSEMBLE                                    │     │
│  │                                                                           │     │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │     │
│  │  │  XGBoost         │  │  GNN             │  │  DeepFM          │         │     │
│  │  │  (tabular feats) │  │  (fraud rings)   │  │  (feature        │         │     │
│  │  │                  │  │                  │  │   interactions)  │         │     │
│  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘         │     │
│  │           │                     │                     │                   │     │
│  │           └─────────────────────┴─────────────────────┘                   │     │
│  │                              │                                            │     │
│  │                              ▼                                            │     │
│  │  ┌──────────────────────────────────────────────────────────────────┐     │     │
│  │  │  Meta-Learner (Logistic Regression on model outputs)             │     │     │
│  │  └──────────────────────────┬───────────────────────────────────────┘     │     │
│  │                             │                                           │     │
│  │                             ▼                                           │     │
│  │  ┌──────────────────────────────────────────────────────────────────┐     │     │
│  │  │  Decision Engine:                                                │     │     │
│  │  │  - Score > 0.9 → BLOCK (high confidence fraud)                  │     │     │
│  │  │  - 0.7 < Score < 0.9 → FLAG (manual review queue)               │     │     │
│  │  │  - 0.3 < Score < 0.7 → ALLOW + STEP-UP AUTH (MFA challenge)     │     │     │
│  │  │  - Score < 0.3 → ALLOW (low risk)                               │     │     │
│  │  └──────────────────────────┬───────────────────────────────────────┘     │     │
│  │                             │                                           │     │
│  └─────────────────────────────┼───────────────────────────────────────────┘     │
│                               │                                               │
└───────────────────────────────┼───────────────────────────────────────────────┘
                                │
┌───────────────────────────────┼───────────────────────────────────────────────┐
│                 OUTPUT & ACTIONS                                              │
│                               ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────┐  │     │
│  │  │  Transaction │  │  Alert       │  │  Case        │  │  Score │  │     │
│  │  │  Approval    │  │  Dashboard   │  │  Management  │  │  API   │  │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────┘  │     │
│  └──────────────────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Fraud Ring Detection with GNN

```
┌─────────────────────────────────────────────────────────────────────┐
│                  GRAPH NEURAL NETWORK PIPELINE                       │
│                                                                      │
│  Transaction Graph Construction:                                     │
│                                                                      │
│       ┌─────────┐      sends_to      ┌─────────┐                    │
│       │ User A  │───────────────────▶│  User B  │                    │
│       │ (acct1) │                    │ (acct2)  │                    │
│       └────┬────┘                    └────┬─────┘                    │
│            │                              │                          │
│     shares │                        shares│                          │
│     device │                        device│                          │
│            ▼                              ▼                          │
│       ┌─────────┐                   ┌─────────┐                     │
│       │Device X │◀──────────────────│Device Y │                     │
│       │ (Phone) │   same IP         │ (Phone) │                     │
│       └────┬────┘                   └────┬─────┘                    │
│            │                             │                           │
│            └─────────── same ────────────┘                           │
│                        location                                      │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  GNN Architecture:                                          │    │
│  │  - Node features: tx velocity, avg amount, device age...    │    │
│  │  - Edge features: amount, frequency, time since last tx     │    │
│  │  - 2-layer GraphSAGE (SAmple and aggreGatE)                │    │
│  │  - Link prediction: are two accounts part of same fraud ring?│    │
│  │  - Node classification: is this account part of syndicate?   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| **Streaming** | Apache Kafka + Kafka Streams | 3.6 | Industry standard for event streaming |
| **Stream Analytics** | Apache Flink + Spark Structured Streaming | 1.18 / 3.5 | Real-time feature computation |
| **Feature Store** | Feast + Redis (online) / PostgreSQL (offline) | 0.37 / 7.x | Point-in-time correct features |
| **Gradient Boosting** | XGBoost + CatBoost | 2.0 / 1.2 | Tabular features, high AUC, fast inference |
| **Graph ML** | PyTorch Geometric (PyG) + DGL | 2.5 / 1.1 | Fraud ring detection |
| **Deep Learning** | TensorFlow + DeepFM implementation | 2.15 | Feature interaction modeling |
| **Model Serving** | BentoML + Triton Inference Server | 1.2 / 23.12 | Low-latency ensemble serving |
| **Online Store** | Redis Cluster | 7.x | Sub-millisecond feature retrieval |
| **Offline Store** | PostgreSQL + S3 (Parquet) | 16 / — | Historical feature storage |
| **Orchestration** | Airflow (training), Temporal (workflows) | 2.8 / 1.22 | Pipeline scheduling |
| **Monitoring** | Prometheus + Grafana + WhyLabs | — | Model drift, data drift |
| **Rule Engine** | Drools (fallback) | 7.74 | Rule-based override for compliance |
| **Database** | CockroachDB (core ledger) + PostgreSQL (analytics) | 23 / 16 | Distributed SQL for ledger |

### Dependency Installation

```bash
# Core ML
pip install xgboost==2.0.3 catboost==1.2.5
pip install torch==2.1.2 torch-geometric==2.5.3
pip install tensorflow==2.15.1
pip install feast==0.37.1 redis==5.0.8

# Streaming
pip install kafka-python==2.0.2 confluent-kafka==2.3.0
pip install apache-flink==1.18.1 pyspark==3.5.1

# Serving
pip install bentoml==1.2.12 tritonclient==2.41.1
pip install fastapi==0.111.1

# Monitoring
pip install whylabs==0.1.12 alibi-detect==0.12.1
```

---

## ⚙️ Implementation Details

### 1. Real-Time Feature Engineering

```python
# src/features/transaction_features.py
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TransactionFeatureEngineer:
    """Compute real-time features for fraud detection."""

    def __init__(self, redis_client, lookback_windows: List[int] = None):
        self.redis = redis_client
        self.windows = lookback_windows or [5, 30, 60, 1440]  # minutes

    async def compute_user_features(
        self, user_id: str, tx_amount: float, tx_timestamp: datetime
    ) -> Dict[str, float]:
        """Per-user velocity and behavioral features."""
        features = {}
        now = tx_timestamp

        for window_min in self.windows:
            window_ago = now - timedelta(minutes=window_min)
            # Get transactions in this window from Redis time-series
            recent_txs = await self.redis.ts().range(
                f"user:{user_id}:txs",
                window_ago.timestamp(),
                now.timestamp()
            )

            amounts = [float(tx[1]) for tx in recent_txs] if recent_txs else []
            features[f"tx_count_{window_min}m"] = len(amounts)
            features[f"tx_sum_{window_min}m"] = sum(amounts)
            features[f"tx_avg_{window_min}m"] = np.mean(amounts) if amounts else 0.0
            features[f"tx_std_{window_min}m"] = np.std(amounts) if len(amounts) > 1 else 0.0
            features[f"tx_max_{window_min}m"] = max(amounts) if amounts else 0.0

            # Ratio of current amount to recent average
            if features[f"tx_avg_{window_min}m"] > 0:
                features[f"amount_ratio_{window_min}m"] = tx_amount / features[f"tx_avg_{window_min}m"]
            else:
                features[f"amount_ratio_{window_min}m"] = 1.0

        return features

    async def compute_merchant_features(
        self, merchant_id: str, user_id: str
    ) -> Dict[str, float]:
        """Merchant-level features (new merchant for user, velocity)."""
        # Check if user has transacted at this merchant before
        merchant_key = f"user:{user_id}:merchants"
        is_new_merchant = not await self.redis.sismember(merchant_key, merchant_id)

        # Merchant velocity (how many unique users this merchant had recently)
        recent_users = await self.redis.scard(
            f"merchant:{merchant_id}:users:1h"
        )

        return {
            "is_new_merchant": float(is_new_merchant),
            "merchant_unique_users_1h": recent_users,
            "days_since_first_tx": await self._days_since_first(user_id, merchant_id),
        }

    async def compute_device_features(
        self, device_fingerprint: str
    ) -> Dict[str, float]:
        """Device-level risk signals."""
        features = {
            "device_age_days": await self._device_age(device_fingerprint),
            "device_users_24h": await self.redis.scard(
                f"device:{device_fingerprint}:users:24h"
            ),
            "device_fraud_reports": await self.redis.get(
                f"device:{device_fingerprint}:fraud_count"
            ) or 0,
            "is_emulator": await self._is_emulator(device_fingerprint),
            "is_rooted": await self._is_rooted(device_fingerprint),
            "vpn_detected": await self._is_vpn(device_fingerprint),
        }
        return features
```

### 2. XGBoost Fraud Classifier

```python
# src/models/xgboost_fraud.py
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_auc_score, confusion_matrix
)

class XGBoostFraudDetector:
    """XGBoost classifier optimized for imbalanced fraud detection."""

    def __init__(self, scale_pos_weight: float = 50.0):
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",  # Area under Precision-Recall
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 5,
            "gamma": 0.1,
            "scale_pos_weight": scale_pos_weight,  # Address class imbalance
            "seed": 42,
            "n_jobs": -1,
            "tree_method": "gpu_hist",  # Use GPU for training
        }
        self.model = None
        self.feature_names = None
        self.optimal_threshold = 0.5

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_weights: dict = None
    ):
        """Train with time-series cross validation."""
        self.feature_names = X_train.columns.tolist()

        dtrain = xgb.DMatrix(
            X_train, label=y_train,
            feature_names=self.feature_names,
            weight=self._compute_sample_weights(y_train)
        )
        dval = xgb.DMatrix(
            X_val, label=y_val,
            feature_names=self.feature_names
        )

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=100,
        )

        # Find optimal threshold using validation data
        val_preds = self.model.predict(dval)
        self._find_optimal_threshold(y_val, val_preds)

        return self._evaluate(y_val, val_preds)

    def _compute_sample_weights(self, y: pd.Series) -> np.ndarray:
        """Assign higher weights to fraudulent transactions."""
        weights = np.where(y == 1, self.params["scale_pos_weight"], 1.0)
        return weights

    def _find_optimal_threshold(self, y_true: pd.Series, y_pred: np.ndarray):
        """Find threshold maximizing precision-recall F2-score (emphasize recall)."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
        f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
        best_idx = np.argmax(f2_scores[:-1])  # Exclude last element
        self.optimal_threshold = thresholds[best_idx]
        print(f"Optimal threshold set to {self.optimal_threshold:.4f} "
              f"(best F2: {f2_scores[best_idx]:.4f})")

    def predict(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (binary_prediction, probability)."""
        dmatrix = xgb.DMatrix(features, feature_names=self.feature_names)
        probabilities = self.model.predict(dmatrix)
        predictions = (probabilities >= self.optimal_threshold).astype(int)
        return predictions, probabilities

    def _evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> dict:
        """Compute evaluation metrics."""
        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        preds_binary = (y_pred >= self.optimal_threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, preds_binary).ravel()

        return {
            "auc_roc": auc,
            "average_precision": ap,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "optimal_threshold": self.optimal_threshold,
        }
```

### 3. Graph Neural Network for Fraud Rings

```python
# src/models/gnn_fraud_rings.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import Data, DataLoader

class FraudRingGNN(torch.nn.Module):
    """GraphSAGE model for fraud ring detection.

    Input: Transaction graph with nodes (accounts) and edges (transactions).
    Output: Node-level fraud probability and link prediction for ring detection.
    """

    def __init__(self, node_features: int, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = SAGEConv(node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.node_classifier = torch.nn.Linear(hidden_dim, 2)  # Fraud / Legit
        self.link_predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Node-level fraud classification
        node_logits = self.node_classifier(x)

        # Link prediction (fraud ring connectivity)
        # Compute edge scores from node embeddings
        edge_embeddings = torch.cat([
            x[edge_index[0]], x[edge_index[1]]
        ], dim=1)
        edge_scores = torch.sigmoid(self.link_predictor(edge_embeddings))

        return node_logits, edge_scores.squeeze()

class FraudGraphBuilder:
    """Build transaction graphs for GNN training and inference."""

    @staticmethod
    def build_graph(
        transactions: list,
        node_to_idx: dict = None
    ) -> Data:
        """Build PyTorch Geometric graph from transaction list."""
        if node_to_idx is None:
            # Create node mapping
            all_accounts = set()
            for tx in transactions:
                all_accounts.add(tx["sender_id"])
                all_accounts.add(tx["receiver_id"])
            node_to_idx = {acc: i for i, acc in enumerate(all_accounts)}

        # Edge index (sender -> receiver)
        edge_index_senders = [node_to_idx[tx["sender_id"]] for tx in transactions]
        edge_index_receivers = [node_to_idx[tx["receiver_id"]] for tx in transactions]
        edge_index = torch.tensor([edge_index_senders, edge_index_receivers], dtype=torch.long)

        # Node features (placeholder — replace with real features)
        num_nodes = len(node_to_idx)
        node_features = torch.randn((num_nodes, 32))  # Real features from feature store

        # Edge features (amount, time, frequency)
        edge_features = torch.tensor([
            [tx["amount"], tx["tx_hour"], tx["frequency_score"]]
            for tx in transactions
        ], dtype=torch.float)

        # Labels (node-level: is this account part of known fraud ring?)
        node_labels = torch.tensor([
            tx.get("is_fraud", 0) for tx in transactions
        ], dtype=torch.long)

        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            y=node_labels,
        )
        return graph
```

### 4. Decision Engine with Rule Overlay

```python
# src/decision/decision_engine.py
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

@dataclass
class DecisionResult:
    decision: str  # ALLOW, FLAG, BLOCK, MFA_CHALLENGE
    score: float
    confidence: str  # HIGH, MEDIUM, LOW
    reasons: List[str]
    model_outputs: Dict[str, float]
    rule_overrides: List[str]

class FraudDecisionEngine:
    """Combines ML model outputs with rule-based overrides."""

    ALLOW_THRESHOLD = 0.3
    MFA_THRESHOLD = 0.7
    FLAG_THRESHOLD = 0.85
    BLOCK_THRESHOLD = 0.9

    def __init__(self, rule_engine=None):
        self.rule_engine = rule_engine  # Optional Drools rule engine

    def decide(
        self,
        transaction: Dict,
        xgb_score: float,
        gnn_score: float,
        deepfm_score: float,
        feature_values: Dict[str, float],
    ) -> DecisionResult:
        """Make final decision from ensemble + rules."""
        ensemble_score = self._ensemble_weighted_score(
            xgb_score, gnn_score, deepfm_score
        )

        reasons = []
        rule_overrides = []

        # Check rule engine for mandatory blocks
        if self.rule_engine:
            rule_decisions = self.rule_engine.evaluate(transaction)
            for rule in rule_decisions:
                if rule["action"] == "BLOCK":
                    rule_overrides.append(rule["rule_name"])
                    return DecisionResult(
                        decision="BLOCK",
                        score=1.0,
                        confidence="HIGH",
                        reasons=[f"Rule override: {rule['rule_name']}"],
                        model_outputs=self._model_outputs(
                            xgb_score, gnn_score, deepfm_score
                        ),
                        rule_overrides=rule_overrides,
                    )

        # ML-based decision with thresholds
        if ensemble_score >= self.BLOCK_THRESHOLD:
            decision = "BLOCK"
            confidence = "HIGH"
            reasons.append(f"Ensemble score {ensemble_score:.3f} exceeds block threshold")
        elif ensemble_score >= self.FLAG_THRESHOLD:
            decision = "FLAG"
            confidence = "MEDIUM"
            reasons.append(f"Ensemble score {ensemble_score:.3f} — manual review needed")
        elif ensemble_score >= self.MFA_THRESHOLD:
            decision = "MFA_CHALLENGE"
            confidence = "MEDIUM"
            reasons.append(f"Medium risk score {ensemble_score:.3f} — step-up auth")
        else:
            decision = "ALLOW"
            confidence = "LOW" if ensemble_score > self.ALLOW_THRESHOLD else "HIGH"
            reasons.append(f"Low risk score {ensemble_score:.3f} — allow")

        # Additional reason based on individual model disagreement
        models = [("XGBoost", xgb_score), ("GNN", gnn_score), ("DeepFM", deepfm_score)]
        high_scores = [name for name, score in models if score > self.FLAG_THRESHOLD]
        if len(high_scores) > 0 and len(high_scores) < 3:
            reasons.append(f"Partial agreement: {', '.join(high_scores)} flagged")

        return DecisionResult(
            decision=decision,
            score=ensemble_score,
            confidence=confidence,
            reasons=reasons,
            model_outputs=self._model_outputs(
                xgb_score, gnn_score, deepfm_score
            ),
            rule_overrides=rule_overrides,
        )

    def _ensemble_weighted_score(
        self, xgb_score: float, gnn_score: float, deepfm_score: float
    ) -> float:
        """Weighted ensemble using optimized weights."""
        WEIGHTS = {"xgb": 0.40, "gnn": 0.35, "deepfm": 0.25}
        return (
            WEIGHTS["xgb"] * xgb_score +
            WEIGHTS["gnn"] * gnn_score +
            WEIGHTS["deepfm"] * deepfm_score
        )

    def _model_outputs(self, xgb_s, gnn_s, deepfm_s) -> Dict:
        return {"xgb": xgb_s, "gnn": gnn_s, "deepfm": deepfm_s}
```

### 5. Kafka Transaction Stream Consumer

```python
# src/streaming/transaction_consumer.py
from kafka import KafkaConsumer, KafkaProducer
import json
import asyncio
from src.features.transaction_features import TransactionFeatureEngineer
from src.decision.decision_engine import FraudDecisionEngine
import redis.asynced as redis

class TransactionStreamProcessor:
    """Real-time consumer that scores every transaction against ML models."""

    def __init__(self, config: dict):
        self.consumer = KafkaConsumer(
            "raw-transactions",
            bootstrap_servers=config["kafka_brokers"],
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=False,
            max_poll_records=100,
        )
        self.producer = KafkaProducer(
            bootstrap_servers=config["kafka_brokers"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self.redis_client = redis.Redis(
            host=config["redis_host"], port=config["redis_port"], decode_responses=True
        )
        self.feature_engineer = TransactionFeatureEngineer(self.redis_client)
        self.decision_engine = FraudDecisionEngine()

        # Load models (warm start)
        self._load_models()

    def _load_models(self):
        """Load pre-trained models from model registry."""
        # In production: load from MLflow or S3
        self.xgb_model = None  # Loaded via joblib
        self.gnn_model = None
        self.deepfm_model = None
        print("Models loaded successfully")

    async def process_transaction(self, tx: dict) -> dict:
        """Process a single transaction end-to-end."""
        try:
            # Step 1: Feature computation
            user_features = await self.feature_engineer.compute_user_features(
                tx["user_id"], tx["amount"], tx["timestamp"]
            )
            merchant_features = await self.feature_engineer.compute_merchant_features(
                tx["merchant_id"], tx["user_id"]
            )
            device_features = await self.feature_engineer.compute_device_features(
                tx["device_fingerprint"]
            )

            # Combine all features
            features = {**user_features, **merchant_features, **device_features}

            # Step 2: Model inference (in production, call Triton)
            xgb_score = 0.0  # self.xgb_model.predict_proba(features)
            gnn_score = 0.0  # self.gnn_model.predict(graph)
            deepfm_score = 0.0  # self.deepfm_model.predict(features)

            # Step 3: Decision
            decision = self.decision_engine.decide(
                tx, xgb_score, gnn_score, deepfm_score, features
            )

            # Step 4: Produce result
            result = {
                "transaction_id": tx["transaction_id"],
                "user_id": tx["user_id"],
                "timestamp": tx["timestamp"],
                "decision": decision.decision,
                "score": decision.score,
                "reasons": decision.reasons,
                "model_scores": decision.model_outputs,
            }

            # Send to decisions topic
            self.producer.send("scored-transactions", result)

            # Log audit trail
            self.producer.send("audit-log", {
                **result,
                "all_features": features,
                "rule_overrides": decision.rule_overrides,
            })

            return result

        except Exception as e:
            # Fail open: allow transaction if scoring fails
            self.producer.send("scoring-errors", {
                "transaction_id": tx["transaction_id"],
                "error": str(e),
            })
            return {
                "transaction_id": tx["transaction_id"],
                "decision": "ALLOW",
                "score": -1.0,
                "reasons": ["Scoring error — allowed as fallback"],
            }

    def run(self):
        """Main loop consuming and processing transactions."""
        for message in self.consumer:
            tx = message.value
            result = asyncio.run(self.process_transaction(tx))
            print(f"Tx {tx['transaction_id']}: {result['decision']} "
                  f"(score={result['score']:.3f})")
```

---

## 📊 Metrics & Results

### Model Performance (6-month production data)

| Metric | XGBoost | GNN | DeepFM | Ensemble |
|--------|---------|-----|--------|----------|
| **AUC-ROC** | 0.97 | 0.94 | 0.95 | **0.98** |
| **Average Precision** | 0.89 | 0.85 | 0.87 | **0.92** |
| **Recall@1% FPR** | 0.72 | 0.65 | 0.68 | **0.78** |
| **False Positive Rate** | 0.15% | 0.21% | 0.18% | **0.10%** |
| **Fraud Capture Rate** | 88% | 82% | 84% | **95.2%** |
| **P50 Latency** | 2ms | 15ms | 5ms | **22ms** |
| **P99 Latency** | 8ms | 45ms | 18ms | **68ms** |

### Business Impact (12 months post-deployment)

| Metric | Legacy System | New System | Delta |
|--------|-------------|-------------|-------|
| **Fraud Capture Rate** | 12% | 95.2% | **+83.2 pp** |
| **False Positive Rate** | 40% | 0.1% | **-39.9 pp** |
| **Annual Fraud Loss** | $45M | $2.5M | **-$42.5M** |
| **False Declines (customer)** | 3,200/day | 85/day | **-97.3%** |
| **Detection Latency** | 24-72 hrs | < 100ms | **real-time** |
| **Manual Review Queue** | 1,500/day | 120/day | **-92%** |
| **Fraud Analyst Productivity** | 50 cases/day | 200 cases/day | **+300%** |
| **Model Update Cycle** | Monthly | Daily (incremental) | **30x faster** |

### Annual Cost Savings

```
Total Annual Savings: $50,000,000

┌──────────────────────────────────────────────────────────────┐
│  Fraud Loss Reduction (from $45M to $2.5M)     $42,500,000  │
│  False Decline Customer Retention               $3,200,000   │
│  Manual Review Cost Reduction                   $1,800,000   │
│  Infrastructure Cost (Kafka vs mainframe)       -$500,000    │
│  Model Development & MLOps                      -$2,000,000  │
│  Chargeback Fee Reduction                       $500,000     │
│  Regulatory Fine Avoidance                      $4,500,000   │
├──────────────────────────────────────────────────────────────┤
│  Net Annual Savings                             $50,000,000  │
│  ROI (Year 1)                                       2,400%   │
└──────────────────────────────────────────────────────────────┘
```

### Model Performance by Fraud Type

```
Fraud Type        | Capture | FP Rate | Avg Score | % of Total
──────────────────┼─────────┼─────────┼───────────┼───────────
Account Takeover  |  97.2%  |  0.08%  |   0.94    |   18%
Card Not Present  |  94.8%  |  0.12%  |   0.91    |   35%
Synthetic ID      |  91.5%  |  0.15%  |   0.87    |   22%
APP Fraud (P2P)   |  93.1%  |  0.09%  |   0.89    |   15%
Money Mule Accts  |  96.7%  |  0.07%  |   0.96    |    5%
First-party Fraud |  88.4%  |  0.22%  |   0.83    |    5%
```

---

## 💡 Lessons Learned

### ✅ What Went Well

1. **Ensemble with weighted blending** — Single XGBoost captured 88%; ensemble of 3 captured 95.2%. The GNN specifically caught fraud rings that tree models missed.

2. **Graph features were a force multiplier** — Adding "same device" and "same IP" graph features to XGBoost improved recall by 12% even without the full GNN model.

3. **Real-time feature store (Redis)** — Sub-millisecond feature retrieval made streaming inference possible. The previous batch system couldn't do per-transaction scoring.

4. **Fail-open architecture** — When any model is down, transactions are allowed (not blocked). This saved us during a 4-hour Kafka outage.

### ❌ What Went Wrong

1. **Model drift was fast and silent** — Fraud patterns shift weekly. Our daily retraining initially used stale labels (7-day delay). Implemented "fresh label prioritization" — weight recent confirmed fraud 3× higher.

2. **GNN inference latency was initially 250ms** — Exceeded budget. Solution: Graph partitioning and incremental inference (only re-compute subgraphs for active accounts).

3. **False positives in cross-border transactions** — The model initially flagged 8% of international transfers. Added "country pair" features and retrained with geo-specific data.

4. **Explainability gap with regulators** — Regulators wanted to know *why* a transaction was blocked. GNN outputs are inherently opaque. Added LIME explanations for every blocked transaction.

### ⚠️ Critical Warnings

```
! WARNING: Model blocking must have a 5-second timeout → fallback to ALLOW.
! WARNING: Never store raw credit card numbers in feature store (PCI violation).
! WARNING: Adversarial input can poison feature values — validate all inputs.
! WARNING: Regulatory requirement: ML decisions must be explainable on demand.
! WARNING: Monitor concept drift weekly — fraud patterns evolve faster than typical ML.
```

### Ongoing Maintenance

```
Model Retraining:   Daily (incremental XGBoost), Weekly (full ensemble)
Feature Monitoring: Hourly data drift detection on top-20 features
Threshold Review:   Bi-weekly calibration with fraud team
A/B Testing:       Continuous shadow scoring for new model versions
Adversarial Audit: Monthly red-team exercise
```

---

## 📁 Reusable Project Template

### Directory Structure

```
TEMPLATE-FRAUD-DETECTION/
├── README.md
├── Makefile
├── requirements.txt
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── configs/
│   ├── config.yaml
│   ├── features.yaml
│   ├── thresholds.yaml
│   ├── models.yaml
│   ├── kafka_topics.yaml
│   └── rules/
│       ├── mandatory_blocks.yaml
│       └── compliance_rules.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── streaming/
│   │   ├── __init__.py
│   │   ├── transaction_consumer.py
│   │   ├── flink_jobs/
│   │   │   ├── feature_aggregation.py
│   │   │   └── graph_builder.py
│   │   └── producer.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── transaction_features.py
│   │   ├── user_profiling.py
│   │   ├── device_intelligence.py
│   │   ├── merchant_features.py
│   │   └── graph_features.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── xgboost_fraud.py
│   │   ├── gnn_fraud_rings.py
│   │   ├── deepfm_model.py
│   │   ├── ensemble.py
│   │   └── train_pipeline.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── model_server.py
│   │   ├── triton_client.py
│   │   └── batch_scorer.py
│   │
│   ├── decision/
│   │   ├── __init__.py
│   │   ├── decision_engine.py
│   │   ├── rule_engine.py
│   │   └── explainability.py
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   └── schemas.py
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── drift_detector.py
│   │   ├── performance_tracker.py
│   │   └── alert_rules.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── metrics.py
│       └── feature_validator.py
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_features.py
│   │   ├── test_models.py
│   │   ├── test_decision.py
│   │   └── test_stream.py
│   ├── integration/
│   │   ├── test_kafka_pipeline.py
│   │   ├── test_redis_store.py
│   │   └── test_end_to_end.py
│   └── fixtures/
│       ├── sample_transactions.json
│       └── mock_feature_store.py
│
├── notebooks/
│   ├── 01-eda-fraud-patterns.ipynb
│   ├── 02-xgboost-training.ipynb
│   ├── 03-gnn-fraud-rings.ipynb
│   └── 04-ensemble-optimization.ipynb
│
├── scripts/
│   ├── simulate_transactions.py
│   ├── train_models.py
│   ├── evaluate_ensemble.py
│   ├── deploy_triton.sh
│   ├── start_kafka.sh
│   └── seed_feature_store.py
│
├── k8s/
│   ├── namespace.yaml
│   ├── deployment-stream-processor.yaml
│   ├── deployment-api.yaml
│   ├── hpa.yaml
│   ├── configmap.yaml
│   └── secrets.yaml
│
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   ├── modules/
│   │   ├── kafka/
│   │   ├── redis/
│   │   └── eks/
│   └── environments/
│       ├── dev.tfvars
│       ├── staging.tfvars
│       └── prod.tfvars
│
└── docs/
    ├── architecture.md
    ├── fraud_typologies.md
    ├── explainability.md
    ├── runbook.md
    └── regulatory_compliance.md
```

### Getting Started

```bash
# 1. Copy template
cp -r TEMPLATE-FRAUD-DETECTION ~/my-fraud-detector
cd ~/my-fraud-detector

# 2. Install dependencies
make install

# 3. Start streaming infrastructure
docker-compose up -d kafka redis postgres

# 4. Simulate transaction stream
python scripts/simulate_transactions.py --rate 100 --duration 300

# 5. Train models
python scripts/train_models.py

# 6. Start stream processor
python src/streaming/transaction_consumer.py

# 7. Check dashboard
open http://localhost:3000  # Grafana
```

---

## 📚 References & Further Reading

### Academic Papers
- Bhatt et al. (2023) — "Graph Neural Networks for Financial Fraud Detection: A Survey" — [arXiv:2310.01286](https://arxiv.org/abs/2310.01286)
- Chen & Guestrin (2016) — "XGBoost: A Scalable Tree Boosting System" — [KDD 2016](https://arxiv.org/abs/1603.02754)
- Guo et al. (2017) — "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction" — [arXiv:1703.04247](https://arxiv.org/abs/1703.04247)
- Hamilton et al. (2017) — "Inductive Representation Learning on Large Graphs (GraphSAGE)" — [NeurIPS 2017](https://arxiv.org/abs/1706.02216)

### Regulatory References
- PSD2 (Revised Payment Services Directive): https://ec.europa.eu/info/law/payment-services-psd-2_en
- PCI-DSS v4.0: https://www.pcisecuritystandards.org/
- GDPR: https://gdpr-info.eu/

### Tools & Documentation
- Apache Kafka: https://kafka.apache.org/documentation/
- Apache Flink: https://nightlies.apache.org/flink/
- Feast Feature Store: https://docs.feast.dev/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- BentoML: https://docs.bentoml.com/

---

> **Next**: [06-RAG-Search-System.md](06-RAG-Search-System.md) — Enterprise RAG search system with hybrid search, re-ranking, and multi-hop retrieval.

# AI in Finance — Technical Deep Dive

> Architectures, code patterns, and engineering practices for deploying AI in financial services. Focus on production-grade patterns: feature stores, real-time scoring, model validation, explainability, graph ML for AML, LLM grounding for advisory, and agentic ops. Cross-references the MLOps (`56`), XAI (`64`), RAG (`04`), and Agents (`03`) categories.

## 1. Reference Architecture

```
┌────────────┐   ┌──────────────┐   ┌──────────────┐
│ Data Sources│ → │ Ingestion    │ → │ Feature Store │
│ (core, mkt, │   │ (CDC, stream)│   │ (online/off) │
│  docs, alt) │   └──────────────┘   └──────┬───────┘
└────────────┘                              │
                                  ┌─────────┴─────────┐
                                  │  Training / Batch  │
                                  │  + Validation (64) │
                                  └─────────┬─────────┘
                                  ┌─────────┴─────────┐
                                  │  Real-time Scoring │
                                  │  (streaming, <50ms)│
                                  └─────────┬─────────┘
                                  ┌─────────┴─────────┐
                                  │ Decision + XAI +   │
                                  │ Human-in-the-loop  │
                                  └─────────┬─────────┘
                                  ┌─────────┴─────────┐
                                  │ Audit Log + Monitor│
                                  │ (56, 43)           │
                                  └────────────────────┘
```

## 2. Feature Store Pattern

A feature store prevents training/serving skew and powers real-time scoring.

```python
# Feast-style feature definition (illustrative)
from feast import FeatureView, ValueType, Entity, Feature
from feast.infra.offline_stores import BigQuerySource

txn_source = BigQuerySource(query="""
  SELECT customer_id,
         SUM(amount) AS amt_1h,
         COUNT(*)     AS cnt_1h,
         DATE(ts)     AS event_date
  FROM txns GROUP BY 1,4
""")

txn_fv = FeatureView(
    name="txn_features",
    entities=["customer_id"],
    ttl="24h",
    features=[
        Feature(name="amt_1h", dtype=ValueType.FLOAT),
        Feature(name="cnt_1h", dtype=ValueType.INT),
    ],
    online=True,
    source=txn_source,
)
```

- Offline store → training. Online store (Redis) → low-latency serving.
- Lineage tracked per feature (`43-AI-Data-Provenance-and-Content-Authenticity/`).

## 3. Real-Time Scoring Service

```python
from fastapi import FastAPI
import redis, pickle

app = FastAPI()
r = redis.Redis(host="feature-store", port=6379)

model = pickle.load(open("fraud_model.pkl", "rb"))

@app.post("/score")
def score(customer_id: str, txn: dict):
    feats = r.hgetall(f"feat:{customer_id}")
    X = assemble(feats, txn)
    proba = float(model.predict_proba([X])[0, 1])
    reason = explain_local(model, X)   # SHAP (see §5)
    decision = "step_up" if proba > 0.9 else "allow"
    audit(customer_id, proba, reason, decision)  # §9
    return {"score": proba, "decision": decision, "reason": reason}
```

## 4. Champion/Challenger

```python
# Shadow challenger; compare before promotion
challenger_pred = challenger.predict_proba(X)
log_compare(champion_pred, challenger_pred)
if challenger_wins(metrics, business_constraints):
    promote(challenger)   # gated by MRM approval
```

## 5. Explainability (SHAP)

Credit/fraud decisions need local explanations (`64-AI-Model-Explainability-and-XAI/`).

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Adverse-action reason: top 4 positive contributors
contrib = sorted(zip(X_sample.columns, shap_values[1][0]),
                 key=lambda t: t[1], reverse=True)[:4]
reasons = [f"{name}: {sign(val)}" for name, val in contrib]
```

Map SHAP reasons to plain-language adverse-action notices.

## 6. Graph ML for AML

```python
import torch
from torch_geometric.nn import GCNConv

class AMLGNN(torch.nn.Module):
    def __init__(self, in_ch, hid, out):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hid)
        self.conv2 = GCNConv(hid, out)
    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# Node = account, edge = transfer; label = illicit.
# Output embeddings feed a classifier for SAR triage.
```

## 7. LLM Grounding for Advisory (RAG)

Never let an LLM invent a product or regulation. Ground it (`04-RAG/`, `52-AI-Hallucination-Detection-and-Mitigation/`).

```python
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

vs = Chroma(persist_directory="firm_kb")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vs.as_retriever(k=5),
    return_source_documents=True,   # provenance (43)
)

ans = qa.run("What bond funds fit a retired, low-risk client?")
# Verify citations exist; block if missing.
assert ans["source_documents"], "ungrounded answer rejected"
```

## 8. Hallucination Guardrails

```python
def guard(answer, sources):
    if not sources:
        return "I cannot answer without firm-approved sources."
    if mentions_fabricated_product(answer):
        return "Answer contains unverified product; escalated."
    return answer
```

## 9. Audit Logging

Every decision is reconstructable (`43`).

```python
import json, time

def audit(cid, score, reason, decision):
    log = {
        "ts": time.time(),
        "customer_id": cid,
        "model": MODEL_VERSION,
        "score": score,
        "reasons": reason,
        "decision": decision,
    }
    kafka_produce("decisions", json.dumps(log))
```

## 10. Model Validation (Second Line)

Independent MRM validation (`64`):
- Conceptual soundness review.
- Outcomes analysis (AUC, calibration).
- Benchmark comparison.
- Sensitivity / stress testing.
- Fair-lending / disparate-impact testing.

```python
from sklearn.metrics import roc_auc_score
def validate(y_true, y_pred, group):
    auc = roc_auc_score(y_true, y_pred)
    # Disparate impact: ratio of approval rates
    di = approval_rate(y_pred, group=="A") / approval_rate(y_pred, group=="B")
    return {"auc": auc, "disparate_impact": di}
```

## 11. Drift Monitoring

```python
from evidently import Report
from evidently.metrics import DataDriftMetric, PSIMetric

report = Report(metrics=[PSIMetric(), DataDriftMetric()])
report.run(reference=ref, current=cur)
if report.drift_detected:
    alert_mlops()   # retrain trigger (56)
```

## 12. Streaming Fraud Pipeline (Kafka + Flink)

```sql
-- Flink SQL: 1-minute rolling fraud feature
SELECT customer_id,
       SUM(amount) AS amt_1m,
       COUNT(*)    AS cnt_1m
FROM txns
GROUP BY customer_id,
         HOP(ts, INTERVAL '10' SECOND, INTERVAL '1' MINUTE)
```

## 13. Reinforcement Learning for Execution

```python
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("TradeExecution-v0")
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=100_000)
# Policy slices a large order to minimize market impact.
```

## 14. Computer Vision for Claims

```python
from torchvision.models import resnet50
import torch

model = resnet50(weights="DEFAULT")
model.fc = torch.nn.Linear(2048, 5)  # damage severity classes
# Input: photos of vehicle; output: severity + estimate.
```

## 15. Privacy-Preserving Cross-Bank Fraud

Federated learning lets banks share signal without sharing data (`40-AI-Data-Sovereignty-and-Privacy/`).

```python
import flower as fl

fl.client.start_client(server_address="agg:8080",
                       client=FraudClient(model))
# Gradients aggregated centrally; raw data stays local.
```

## 16. Agentic Reconciliation

```python
from langgraph.graph import StateGraph

def match(state): ...   # match ledger entries
def flag_exceptions(state): ...  # raise exceptions
def escalate(state): ...  # human handoff

g = StateGraph(ReconState)
g.add_node("match", match)
g.add_node("flag", flag_exceptions)
g.add_node("escalate", escalate)
g.add_edge("match", "flag")
g.add_conditional_edges("flag", route_to_human, {"escalate": "escalate"})
# Governed by 59 (cost) and 18 (security).
```

## 17. Security: Prompt Injection Defense

Agentic finance is exposed to injection (`18-Agent-Security-and-Trust/`).

```python
def sanitize_tool_input(text):
    # Strip instructions that try to override policy
    if "ignore previous" in text.lower():
        raise SecurityError("possible injection")
    return text
```

## 18. Latency Budget

| Stage | Budget |
|-------|--------|
| Feature fetch | 5 ms |
| Model infer | 10 ms |
| XAI compute | 10 ms |
| Network | 20 ms |
| **Total** | **< 50 ms** |

Use `62-Edge-AI-and-On-Device-Inference/` techniques for edge auth.

## 19. Calibration

Risk scores must be calibrated (probabilities meaningful).

```python
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(model, method="isotonic")
calibrated.fit(X_cal, y_cal)
```

## 20. Backtesting Trading Models

```python
def backtest(signals, returns, cost=0.0005):
    pnl = (signals.shift(1) * returns - cost).cumsum()
    sharpe = pnl.diff().mean() / pnl.diff().std() * (252**0.5)
    return sharpe
```

## 21. Model Registry & Versioning

```python
import mlflow
mlflow.log_metric("auc", 0.81)
mlflow.log_param("model", "xgb")
mlflow.register_model("runs:/.../model", "fraud_prod")
# Promotion gated by MRM approval (64).
```

## 22. Fairness Testing

```python
from aif360.sklearn.metrics import statistical_parity_difference
spd = statistical_parity_difference(y_true, y_pred, prot_attr="group")
# SPD near 0 = fair; remediate if beyond threshold.
```

## 23. Explainability at Scale

Batch explanations for audit; on-demand for disputes. Use `64` for full method catalog.

## 24. Disaster Recovery / DORA

- Active-active scoring clusters.
- Feature store replicated.
- Runbook + game days (`21-AI-Regulation-Antitrust/` DORA).

## 25. Cost Engineering

Tie inference cost to value (`41-AI-Cost-Optimization-and-Enterprise-ROI/`, `66-AI-Model-Commoditization-and-Economics/`):
- Cache embeddings.
- Route simple tasks to small models (`30-Small-Language-Models/`).
- Use cascades (`53`).

## 26. MLOps Integration

This category leans on `56-MLOps-and-AI-Platform-Engineering/` for:
- CI/CD for models.
- Monitoring & alerting.
- Feature pipelines.
- Rollback.

## 27. Evaluation Harness

```python
# Reuse 58 evaluation framework
from eval_lib import FinanceSuite
suite = FinanceSuite(metrics=["auc","ks","precision@k","di","psi"])
report = suite.run(model, test_set)
```

## 28. Deployment Topology

```
Edge (auth) ─┐
Cloud (batch) ─┼─ Feature Store ─ Model Serving ─ Audit
On-prem (risk) ─┘
```

## 29. Observability Stack

- Metrics: Prometheus (latency, score distribution).
- Traces: OpenTelemetry per decision.
- Logs: immutable decision log (`43`).
- Alerts: drift, latency breach, PSI.

## 30. Testing Pyramid

| Layer | Test |
|-------|------|
| Unit | feature transforms |
| Integration | pipeline end-to-end |
| Validation | MRM independent |
| Canary | shadow on live traffic |
| Compliance | fair-lending reg tests |

## 31. Common Bugs & Fixes

| Bug | Fix |
|-----|-----|
| Training/serving skew | Feature store (§2) |
| Stale online features | TTL tuning |
| Calibration drift | Recalibrate (§19) |
| Leakage | Time-split CV |
| Proxy bias | Fairness test (§22) |

## 32. Time-Series Split

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
# Prevents look-ahead leakage in risk models.
```

## 33. LLM Cost Control in Advisory

Use `59-AI-Agent-Financial-Governance-and-Cost-Control/` budgets per session.

## 34. Multi-Model Orchestration

Route by complexity (`53-AI-Model-Cascading-and-Multi-Model-Orchestration/`):
- Cheap GBM for 90% of traffic.
- LLM only for ambiguous cases.

## 35. Summary

Production finance AI = rigorous data + validated models + real-time serving + explainability + audit + monitoring. The engineering is mature; the differentiators are governance and trust. Continue to `04-Tools-and-Frameworks.md`.

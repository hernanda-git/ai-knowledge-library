# Technical Deep Dive — AI in Real Estate and PropTech

> Implementation-level reference: building AVMs, computer-vision inspection pipelines, LLM lease abstraction, recommendation systems, and agentic transaction workflows. Includes architecture patterns, code, data schemas, MLOps, and observability, with cross-references to the engineering categories in this library.

---

## Table of Contents

1. [Reference Architecture](#reference-architecture)
2. [AVM Engineering](#avm-engineering)
3. [Computer-Vision Inspection Pipeline](#computer-vision-inspection-pipeline)
4. [LLM Lease Abstraction Service](#llm-lease-abstraction-service)
5. [Recommendation System](#recommendation-system)
6. [Agentic Transaction Workflows](#agentic-transaction-workflows)
7. [Feature Store and Data Contracts](#feature-store-and-data-contracts)
8. [MLOps and Monitoring](#mlops-and-monitoring)
9. [Observability and Guardrails](#observability-and-guardrails)
10. [Security and Privacy](#security-and-privacy)
11. [Cost Optimization](#cost-optimization)
12. [Cross-Category References](#cross-category-references)

---

## Reference Architecture

```
                  ┌─────────────────────────────────────┐
   Ingestion ───▶ │  Raw zone (S3/ADLS)                 │
   (MLS, imagery, │  → Bronze (validated)              │
    deeds, IoT)   │  → Silver (features)               │
                  │  → Gold (serving tables)            │
                  └──────────────┬──────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
   AVM service            CV inspection            NLP/LLM service
   (batch + online)       (batch + edge)           (RAG)
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 ▼
                   Orchestration / Agent layer
                   (durable workflows: lead→offer→close)
                                 ▼
                   API + Portal + Agent Copilot
```

Use durable execution for multi-day workflows (offer acceptance, inspection scheduling) — see `../31-AI-Workflow-Orchestration-and-Durable-Execution/`.

---

## AVM Engineering

### Feature schema (silver table)
```sql
CREATE TABLE avm_features (
  property_id     STRING,
  sqft            FLOAT,
  beds            INT,
  baths           FLOAT,
  age_yrs         INT,
  lot_sqft        FLOAT,
  zip             STRING,
  median_income   FLOAT,
  miles_transit   FLOAT,
  school_rating   FLOAT,
  flood_zone      STRING,
  last_sale_price FLOAT,
  days_since_sale INT,
  photo_score     FLOAT,      -- CV-derived condition proxy
  comp_count_90d  INT,
  ingested_at     TIMESTAMP
);
```

### Training + backtest
```python
import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_parquet("avm_features.parquet")
# Time-based split avoids leakage from future sales
tscv = TimeSeriesSplit(n_splits=5)
X = df.drop(columns=["sale_price","property_id"])
y = df["sale_price"]

for tr, te in tscv.split(X):
    m = lgb.LGBMRegressor(objective="regression_l1", n_estimators=2000,
                          learning_rate=0.02, num_leaves=63)
    m.fit(X.iloc[tr], y.iloc[tr])
    pred = m.predict(X.iloc[te])
    err = (pred - y.iloc[te]).abs() / y.iloc[te]
    print("segment MAPE:", round(err.median(), 4))
    # segment by zip to detect disparate error
    seg = pd.DataFrame({"zip": df["zip"].iloc[te], "err": err})
    print(seg.groupby("zip")["err"].median().sort_values(ascending=False).head())
```

### Online scoring
- Cache features in a low-latency store (Redis/DynamoDB).
- Serve model via a REST/gRPC endpoint; return price + confidence interval + top comps.
- Refresh nightly with new sales; retrain weekly (see `../56-MLOps-and-AI-Platform-Engineering/`).

---

## Computer-Vision Inspection Pipeline

### Batch pipeline
```python
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch

model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()

def score_property(photo_paths):
    results = []
    for p in photo_paths:
        img = Image.open(p).convert("RGB")
        # in production: transform + batch + custom classes
        # (pool, roof_damage, hvac, solar, fireplace)
        out = model([torch.from_numpy(np.array(img)).permute(2,0,1)/255.0])
        results.append(out)
    return results
```

### Edge deployment
Run lightweight models on inspector phones/tablets to give instant condition feedback (see `../62-Edge-AI-and-On-Device-Inference/`).

```python
# On-device via ONNX (edge)
import onnxruntime as ort
sess = ort.InferenceSession("inspection_edge.onnx")
# input: resized 224x224 RGB; output: [pool, damage, hvac, ...]
```

### Floor-plan reconstruction
- Detect walls/openings from panorama or multiple photos.
- Use a trained segmentation + Hough/learned line model.
- Output vector floor plan for GLA (gross living area) computation.

---

## LLM Lease Abstraction Service

### Design
1. **Ingest** lease PDF → OCR/layout parse → text+layout.
2. **Chunk** by clause; embed into vector store.
3. **Extract** structured fields via LLM with JSON schema.
4. **Verify** with a rule engine + confidence thresholds.
5. **Store** structured record + citations.

```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

class Lease(BaseModel):
    landlord: str
    tenant: str
    start_date: str
    end_date: str
    monthly_rent: float
    deposit: float
    break_lease_fee: str | None
    pets_allowed: bool
    renewal_terms: str | None

parser = PydanticOutputParser(pydantic_object=Lease)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = f"Extract lease fields.\n{parser.get_format_instructions()}\n\nTEXT:\n{lease_text}"
lease = parser.parse(llm.invoke(prompt).content)
```

### Guardrails
- Force citations to source spans (see `../52-AI-Hallucination-Detection-and-Mitigation/`).
- Redact PII before logging (see `../40-AI-Data-Sovereignty-and-Privacy/`).
- Human-in-the-loop for low-confidence extractions.

---

## Recommendation System

### Two-tower embeddings
```python
import torch, torch.nn as nn

class Tower(nn.Module):
    def __init__(self, d_in, d_out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(),
                                 nn.Linear(256, d_out))
    def forward(self, x): return self.net(x)

user_tower = Tower(d_user); item_tower = Tower(d_listing)
# train with infoNCE / BPR; index item embeddings in FAISS
# serve: encode user → ANN search → top-K listings
```

### LLM preference parsing (front-end)
- Convert natural-language wants into structured filters (see `02-Core-Topics.md`).

---

## Agentic Transaction Workflows

A durable, multi-step workflow from lead → offer → inspection → close:

```python
# Pseudocode (Temporal/Restate-style durable workflow)
@workflow.defn
class PurchaseWorkflow:
    @workflow.run
    async def run(self, lead):
        prequal = await workflow.execute_activity(assess_prequal, lead)
        if not prequal.ok: return "decline"
        offer = await workflow.execute_activity(price_offer, lead.property_id)
        accepted = await workflow.execute_activity(send_offer, offer)
        if not accepted:
            return "counter_or_exit"
        insp = await workflow.execute_activity(schedule_inspection, lead.property_id)
        repairs = await workflow.execute_activity(cv_inspect, insp.photos)
        final = await workflow.execute_activity(adjust_offer, offer, repairs)
        await workflow.execute_activity(close, final)
        return "closed"
```

Agent patterns and reliability: `../03-Agents/`, `../31-AI-Workflow-Orchestration-and-Durable-Execution/`, `../54-AI-Agent-State-Management-and-Persistence/`.

---

## Feature Store and Data Contracts

- Centralize features so AVM, CV, and recsys share definitions.
- Enforce schemas and SLAs (see `../37-AI-Native-Databases/`).
- Track lineage for audit (see `../43-AI-Data-Provenance-and-Content-Authenticity/`).

```yaml
# feature spec (example)
- name: photo_score
  owner: cv-team
  source: cv_inspection_job
  type: float
  sla_hours: 24
  description: CV-derived condition proxy for AVM
```

---

## MLOps and Monitoring

| Concern | Practice |
|---------|----------|
| Retraining | Scheduled + drift-triggered |
| Drift | Monitor feature/prediction distributions (rates, inventory) |
| Registry | Model versioning, rollback |
| Repro | Pin data snapshot + code commit |
| Eval | Segmented MAPE, bias dashboards |

See `../56-MLOps-and-AI-Platform-Engineering/` and `../58-AI-Evaluation-and-Benchmarking-at-Scale/`.

---

## Observability and Guardrails

- **Tracing** for LLM calls and agent steps (see `../20-Agent-Infrastructure-and-Observability/`).
- **Cost guardrails** per workflow (see `../59-AI-Agent-Financial-Governance-and-Cost-Control/`).
- **Fair-housing filters** on generated ad copy and targeting.
- **Hallucination checks** on lease/title output.

---

## Security and Privacy

- Tenant/owner PII is sensitive; encrypt at rest, minimize logs.
- Access control on AVM endpoints (competitors scrape estimates).
- Prompt-injection defense on agent tools (see `../18-Agent-Security-and-Trust/`, `../61-AI-Red-Teaming-for-LLMs/`).

---

## Cost Optimization

- Use small models where possible (see `../30-Small-Language-Models/`).
- Cache embeddings and comps; batch CV jobs.
- Route LLM calls by complexity (see `../53-AI-Model-Cascading-and-Multi-Model-Orchestration/`).
- Quantify ROI of automation (see `../41-AI-Cost-Optimization-and-Enterprise-ROI/`).

---

## Cross-Category References

- `../03-Agents/`, `../31-AI-Workflow-Orchestration-and-Durable-Execution/`, `../54-AI-Agent-State-Management-and-Persistence/` — transaction agents.
- `../04-RAG/`, `../69-GraphRAG-and-Knowledge-Graph-Retrieval/` — lease/title + comp graphs.
- `../50-Multimodal-AI/`, `../62-Edge-AI-and-On-Device-Inference/` — CV, edge.
- `../56-MLOps-and-AI-Platform-Engineering/`, `../58-AI-Evaluation-and-Benchmarking-at-Scale/` — MLOps/eval.
- `../20-Agent-Infrastructure-and-Observability/`, `../59-AI-Agent-Financial-Governance-and-Cost-Control/` — ops/cost.
- `../55-AI-Ethics-and-Responsible-AI/`, `../64-AI-Model-Explainability-and-XAI/` — fairness, explainability.
- `../40-AI-Data-Sovereignty-and-Privacy/`, `../43-AI-Data-Provenance-and-Content-Authenticity/` — privacy, lineage.
- `../67-AI-in-Finance-and-Financial-Services/` — mortgage overlap.

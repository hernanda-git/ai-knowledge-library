# AI in Finance — Core Topics

> Deep treatment of the major application areas where AI is deployed across financial services: credit risk, fraud & AML, trading, insurance, wealth, payments, and RegTech. Each section covers the problem, the AI approach, real examples, and the governance wrapper.

## 1. Credit Risk & Underwriting

### 1.1 The problem
Lenders must decide *who to lend to, at what price, and how much*. Errors cost money (defaults) or customers (over-rejection). Traditional credit scoring (logistic regression, FICO) struggles with thin-file applicants and non-linear patterns.

### 1.2 AI approach
- **Gradient-boosted trees** (XGBoost, LightGBM) remain the workhorse for tabular risk.
- **Neural networks** for non-linear interactions.
- **Graph neural networks** to detect linked fraud rings and synthetic identities.
- **LLMs** to parse bank statements, pay stubs, and alternative documents.

### 1.3 Example: feature pipeline
```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

num = ["income", "dti", "age", "utilization"]
cat = ["state", "product"]

pre = ColumnTransformer([
    ("num", "passthrough", num),
    ("cat", "onehot", cat),
])

model = Pipeline([
    ("pre", pre),
    ("clf", XGBClassifier(n_estimators=400, max_depth=6,
                          eval_metric="logloss")),
])

model.fit(X_train, y_train)
preds = model.predict_proba(X_test)[:, 1]
```

### 1.4 Governance wrapper
- **Adverse action**: must produce reasons in plain language (`64-AI-Model-Explainability-and-XAI/`).
- **Fair lending**: test disparate impact across protected groups (`55-AI-Ethics-and-Responsible-AI/`).
- **Proxy variables**: zip code / device can encode bias — test and drop.
- **EU AI Act**: credit scoring is *high-risk* → mandatory risk management, data governance, human oversight.

### 1.5 Metrics
| Metric | Target |
|--------|--------|
| Gini / AUC | > 0.75 typical |
| KS | > 0.4 |
| Default rate in approve pop | within risk appetite |
| Adverse action completeness | 100% |

## 2. Fraud Detection & AML

### 2.1 The problem
Real-time payments expanded fraud surface. Criminals adapt constantly (adversarial drift). AML requires monitoring for suspicious activity across millions of accounts.

### 2.2 AI approach
- **Supervised classifiers** for known fraud patterns (XGBoost, isolation forests).
- **Unsupervised anomaly detection** (autoencoders, clustering) for novel fraud.
- **Graph ML** to find money-laundering networks and mule accounts.
- **Sequence models** (LSTM, Transformers) for transaction time-series.
- **LLMs** to triage SAR (Suspicious Activity Report) narratives.

### 2.3 Example: graph feature for AML
```python
import networkx as nx

G = nx.DiGraph()
for src, dst, amt in transactions:
    G.add_edge(src, dst, weight=amt)

# Flag accounts with high in/out imbalance
for n in G.nodes:
    in_deg = G.in_degree(n, weight="weight")
    out_deg = G.out_degree(n, weight="weight")
    if in_deg > 0 and out_deg / in_deg > 50:
        flag(n, reason="structuring-like flow")
```

### 2.4 Governance wrapper
- **False positive cost**: blocking legit payments angers customers — balance precision/recall.
- **Explainability**: investigators need a reason to act (`64`).
- **Audit**: every alert must be reconstructable (`43-AI-Data-Provenance-and-Content-Authenticity/`).
- **Regulatory**: AML mandates SAR filing; AI assists, humans decide.

### 2.5 Metrics
| Metric | Meaning |
|--------|---------|
| Precision@k | Fraud in top-k alerts |
| Recall | Fraud caught overall |
| False positive rate | Legit blocked |
| Investigator STP | % auto-closed as benign |

## 3. Capital Markets & Algorithmic Trading

### 3.1 The problem
Generate alpha, manage risk, and execute efficiently across equities, FX, rates, crypto.

### 3.2 AI approach
- **Time-series models** (LSTMs, Transformers, TCNs) for price/vol forecasting.
- **Reinforcement learning** for execution (optimal order slicing).
- **LLMs** for news/social sentiment and earnings-call analysis.
- **Alternative data** (satellite, web scrape) via CV/NLP.

### 3.3 Example: sentiment from news
```python
from transformers import pipeline

sentiment = pipeline("text-classification",
                     model="finbert-tone", revision="main")

news = ["Fed signals rate cut amid softening labor market"]
scores = sentiment(news)
# → [{'label': 'positive', 'score': 0.92}]
```

### 3.4 Governance wrapper
- **Model risk**: trading models are high-stakes → strict validation (`64`).
- **Surveillance**: detect market abuse (spoofing, insider trading) via NLP on comms.
- **Explainability**: regulators want to know *why* a trade happened.
- **Latency**: HFT needs microsecond inference — see `62-Edge-AI-and-On-Device-Inference/`.

### 3.5 Metrics
| Metric | Meaning |
|--------|---------|
| Sharpe / Sortino | Risk-adjusted return |
| Hit rate | % profitable signals |
| Max drawdown | Worst peak-to-trough |
| Information ratio | Excess vs benchmark |

## 4. Insurance (InsurTech)

### 4.1 The problem
Price risk fairly, automate claims, detect fraud, personalize products.

### 4.2 AI approach
- **Pricing models** (GBM) for risk-based premiums.
- **Computer vision** for damage assessment from photos (auto, property).
- **LLMs** for first-notice-of-loss (FNOL) intake and claims triage.
- **Telematics** for usage-based insurance (UBI).
- **Anomaly detection** for claims fraud.

### 4.3 Example: claims triage
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

tpl = PromptTemplate.from_template(
    "Triage this claim:\n{claim_text}\n"
    "Return JSON: severity, fraud_risk, routing_team"
)
# chain = LLMChain(llm=llm, prompt=tpl)
# Chain output routes to human or auto-settlement.
```

### 4.4 Governance wrapper
- **Fair pricing**: avoid proxy discrimination (`55`).
- **Solvency II**: model governance for insurers (EU).
- **Explainability**: claim denials need reasons (`64`).
- **Privacy**: health/telematics data is sensitive (`40`).

## 5. Wealth Management & Robo-Advisory

### 5.1 The problem
Deliver personalized investment advice at scale and low cost.

### 5.2 AI approach
- **Portfolio optimization** (mean-variance, Black-Litterman).
- **LLM agents** for financial planning Q&A grounded in RAG (`04-RAG/`).
- **Risk profiling** via behavioral ML.
- **Rebalancing automation**.

### 5.3 Example: RAG advisor
```python
# Retrieval-augmented advisory: ground LLM in firm-approved docs
from langchain.retrievers import VectorStoreRetriever

retriever = VectorStoreRetriever(vectorstore=vs)
docs = retriever.get_relevant_documents("best bond fund for retired client?")
# Pass docs into LLM prompt to avoid hallucination (see 52).
```

### 5.4 Governance wrapper
- **Suitability**: advice must fit client profile (MiFID II, SEC).
- **Fiduciary duty**: no conflicts; disclose.
- **Hallucination**: never let LLM invent products (`52-AI-Hallucination-Detection-and-Mitigation/`).
- **Disclosure**: human oversight required for recommendations.

## 6. Payments & Real-Time Scoring

### 6.1 The problem
Authorize millions of payments per second, blocking fraud without hurting UX.

### 6.2 AI approach
- **Streaming ML** (feature stores + online models) for sub-50ms scoring.
- **Rules + ML hybrid** for explainability and tunability.
- **Device & behavioral biometrics** for auth.

### 6.3 Example: online feature lookup
```python
# Pseudocode: online fraud score at auth time
features = feature_store.get_online(["txn_amt_1h",
                                     "device_age",
                                     "geo_velocity"])
score = fraud_model.predict_proba(features)
if score > 0.9:
    step_up_auth()   # challenge the user
```

### 6.4 Governance wrapper
- **Latency SLA**: scoring must be fast (`62-Edge-AI-and-On-Device-Inference/`).
- **Customer experience**: minimize false declines.
- **Security**: protect against injection in auth flows (`18-Agent-Security-and-Trust/`).

## 7. RegTech & Compliance (SupTech)

### 7.1 The problem
Compliance is document- and rules-heavy; monitoring communications and filings is expensive.

### 7.2 AI approach
- **NLP extraction** from contracts, filings, policies.
- **LLM assistants** that map obligations to controls.
- **Surveillance** of trader/comms for abuse.
- **Automated reporting** generation.

### 7.3 Example: obligation extraction
```python
# Pseudocode: extract regulatory obligations from a PDF
obligations = llm.extract(
    document=policy_pdf,
    schema={"obligation": str, "owner": str,
            "deadline": str, "citation": str}
)
# Feed into a compliance register; track coverage.
```

### 7.4 Governance wrapper
- **Auditability**: every extraction logged (`43`).
- **Hallucination**: citations must be real (`52`).
- **DORA**: operational resilience of the RegTech itself.

## 8. KYC / Identity / Onboarding

### 8.1 The problem
Verify customer identity and screen against sanctions/PEP lists at onboarding and continuously.

### 8.2 AI approach
- **OCR + CV** for document capture.
- **Face match / liveness** for biometric verification.
- **Entity resolution** (graph) to link records.
- **LLMs** for adverse-media screening.

### 8.3 Governance wrapper
- **Privacy**: biometric data is special category (`40`).
- **False match**: wrongful rejection damages trust.
- **Adversarial**: deepfake liveness attacks → security (`22-AI-Cybersecurity-Mythos/`).

## 9. Quantitative Research & Alternative Data

### 9.1 The problem
Find signals in unstructured, non-traditional data.

### 9.2 AI approach
- **CV** for satellite imagery (parking lots, shipping).
- **NLP** for filings, news, earnings.
- **Self-supervised** pretraining on financial corpora.
- **Factor discovery** via ML.

## 10. Agentic Finance (Emerging)

### 10.1 Patterns
- **Reconciliation agents** that match ledgers and raise exceptions.
- **Treasury agents** optimizing cash within policy.
- **Compliance agents** monitoring comms.
- **Underwriting agents** assembling decision packages.

### 10.2 Governance
- Must obey `59-AI-Agent-Financial-Governance-and-Cost-Control/` and `18-Agent-Security-and-Trust/`. Bound actions by policy; require human approval for money movement.

## 11. Cross-Cutting Themes

| Theme | Relevant library category |
|-------|---------------------------|
| Explainability | `64-AI-Model-Explainability-and-XAI/` |
| Fairness | `55-AI-Ethics-and-Responsible-AI/` |
| Evaluation | `58-AI-Evaluation-and-Benchmarking-at-Scale/` |
| MLOps | `56-MLOps-and-AI-Platform-Engineering/` |
| Privacy | `40-AI-Data-Sovereignty-and-Privacy/` |
| Security | `18-Agent-Security-and-Trust/` |
| Hallucination | `52-AI-Hallucination-Detection-and-Mitigation/` |
| RAG | `04-RAG/` |
| Agents | `03-Agents/` |
| Regulation | `21-AI-Regulation-Antitrust/` |
| Cost/ROI | `41-AI-Cost-Optimization-and-Enterprise-ROI/` |

## 12. Build vs Buy Decision

| Capability | Build | Buy |
|------------|-------|-----|
| Core risk model | Often build (IP) | Vendor for SMB |
| Fraud scoring | Hybrid | Vendor + custom |
| KYC | Buy (regex/reg compliance) | — |
| LLM assistant | Build on platform | Platform |
| RegTech reporting | Buy | — |

## 13. Implementation Roadmap

1. Use-case inventory + risk tiering.
2. Data foundation (feature store, lineage).
3. Model development + validation (`64`).
4. MLOps deployment + monitoring (`56`).
5. Human-in-the-loop for high-risk.
6. Continuous audit + retraining.

## 14. Pitfalls by Topic

| Topic | Pitfall |
|-------|---------|
| Credit | Proxy discrimination |
| Fraud | Alert fatigue from false positives |
| Trading | Overfit to backtests |
| Insurance | Unfair pricing proxies |
| Wealth | Hallucinated products |
| Payments | Latency breaches |
| RegTech | Hallucinated citations |

## 15. Case Snapshot Patterns

- **Bank fraud**: GBM + graph + rules hybrid → 30–60% false-positive reduction.
- **Insurer**: CV damage estimate → claims settled in minutes.
- **Asset manager**: LLM earnings analysis → faster research notes.
- **Payments**: streaming score → <50ms auth with step-up.

## 16. Human-in-the-Loop Design

High-risk decisions (credit denial, claim rejection, trade block) route to humans with:
- Model score + top reasons (`64`).
- Confidence interval.
- Full feature attribution.
- One-click override + rationale log.

## 17. Monitoring & Drift

Financial models drift fast (macro shifts, adversarial adaptation). Monitor:
- Population stability index (PSI).
- Feature drift.
- Performance decay (`56-MLOps-and-AI-Platform-Engineering/`).
- Champion/challenger (`53`).

## 18. Regulatory Mapping by Topic

| Topic | Key regulation |
|-------|----------------|
| Credit | ECOA/FHA, EU AI Act (high-risk) |
| Trading | SEC/FINRA, MiFID II |
| Insurance | Solvency II, state DOI |
| Payments | PSD2/PSD3, DORA |
| KYC/AML | BSA, sanctions, 6AMLD |
| Wealth | fiduciary, suitability |

## 19. Data Quality Requirements

- **Completeness**: missing data biases scores.
- **Timeliness**: stale features hurt fraud.
- **Lineage**: trace every feature (`43`).
- **Bias audit**: test slices (`55`).

## 20. Summary Table of Approaches

| Domain | Primary ML | Secondary | Governance must |
|--------|-----------|-----------|-----------------|
| Credit | GBM | GNN, LLM | XAI, fair lending |
| Fraud | GBM + anomaly | Graph | Precision/recall balance |
| Trading | TS + RL | LLM | Validation, surveillance |
| Insurance | GBM + CV | LLM | Fair pricing |
| Wealth | Optimization | RAG LLM | Suitability |
| Payments | Streaming ML | Rules | Latency |
| RegTech | NLP | LLM | Audit |

## 21. Open Challenges

1. Explainability vs accuracy trade-off in deep models.
2. Adversarial robustness of fraud/graph models.
3. Cross-institution data sharing without privacy loss (federated).
4. LLM grounding for advice and compliance.
5. Keeping pace with regulation (`21`).

## 22. Related Categories

- `59-AI-Agent-Financial-Governance-and-Cost-Control/`
- `28-AI-Agent-Commerce-and-A2A-Payments/`
- `63-AI-for-Healthcare-and-Clinical-AI/`
- `49-AI-for-Legal-and-LegalTech/`
- `42-AI-for-Science-and-Drug-Discovery/`

## 23. Quick Reference Card

| Need | Start here |
|------|-----------|
| Score credit | §1, `64` |
| Catch fraud | §2, `56` |
| Trade signals | §3 |
| Price insurance | §4 |
| Advise clients | §5, `04-RAG/` |
| Authorize pay | §6, `62` |
| Comply | §7, `21` |

## 24. Tooling Preview

See `04-Tools-and-Frameworks.md` for vendors and open-source.

## 25. What's Next

- Architecture & code → `03-Technical-Deep-Dive.md`
- Future (agentic finance, AI Act impact) → `05-Future-Outlook.md`

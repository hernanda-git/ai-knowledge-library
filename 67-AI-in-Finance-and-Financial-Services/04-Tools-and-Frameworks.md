# AI in Finance — Tools and Frameworks

> A practitioner's catalog of the platforms, vendors, and open-source libraries used to build AI in financial services. Organized by layer: data/feature, modeling, MLOps, LLM/agent, and domain-specific RegTech/KYC vendors. Cross-references `56-MLOps-and-AI-Platform-Engineering/`, `04-RAG/`, `03-Agents/`.

## 1. Layered Tooling Map

```
┌─ Application ────────────────┐
│ KYC, Fraud, Trading, RegTech │
├─ LLM / Agent Platform ───────┤
│ RAG, copilots, agents (03,04)│
├─ MLOps & Serving ────────────┤
│ MLflow, KServe, Feast (56)   │
├─ Modeling Libraries ─────────┤
│ sklearn, XGBoost, torch      │
├─ Data & Feature ─────────────┤
│ Kafka, Spark, warehouse      │
└──────────────────────────────┘
```

## 2. Open-Source Modeling Libraries

| Tool | Use in finance |
|------|----------------|
| **scikit-learn** | Baselines, preprocessing |
| **XGBoost / LightGBM / CatBoost** | Tabular risk & fraud (workhorse) |
| **PyTorch / TensorFlow** | Deep models, CV, GNN |
| **PyTorch Geometric** | AML graph models |
| **Stable-Baselines3** | RL execution |
| **SHAP / LIME** | Explainability (`64`) |
| **Evidently** | Drift monitoring (`56`) |
| **Alibi / AI Fairness 360** | XAI & fairness (`55`) |
| **Flower** | Federated learning (`40`) |
| **Optuna** | Hyperparameter tuning |

## 3. Feature Stores

| Tool | Notes |
|------|-------|
| **Feast** | Open-source, online/offline |
| **Tecton** | Managed feature platform |
| **Hopsworks** | Feature + model registry |
| **AWS SageMaker Feature Store** | Managed |

Feature stores prevent skew (see `03-Technical-Deep-Dive.md` §2).

## 4. MLOps & Serving

| Tool | Role |
|------|------|
| **MLflow** | Tracking, registry |
| **KServe / Seldon** | Model serving (k8s) |
| **BentoML** | Packaging |
| **Kubeflow** | Pipelines |
| **Evidently / WhyLabs** | Monitoring |
| **Great Expectations** | Data quality |
| **Prometheus / Grafana** | Observability |

Covered broadly in `56-MLOps-and-AI-Platform-Engineering/`.

## 5. LLM & RAG Frameworks

| Tool | Use |
|------|-----|
| **LangChain / LlamaIndex** | RAG pipelines (`04-RAG/`) |
| **LangGraph** | Agent orchestration (`03-Agents/`) |
| **Haystack** | Retrieval |
| **vLLM / TGI** | LLM inference |
| **Chroma / Weaviate / pgvector** | Vector DBs |
| **Guardrails / NeMo Guardrails** | Hallucination control (`52`) |

## 6. Financial LLMs / Models

| Model | Notes |
|-------|-------|
| **FinBERT** | Financial sentiment |
| **BloombergGPT** | Finance-specific LLM |
| **FinGPT** | Open finance LLM |
| **General LLMs (GPT/Claude/Llama)** | Used via RAG for advisory |

> Use general models grounded in firm-approved docs — never ungrounded (`04-RAG/`, `52`).

## 7. KYC / Identity Vendors

| Vendor | Capability |
|--------|-----------|
| **Jumio** | Document + biometric verification |
| **Onfido** | Identity verification |
| **Persona** | KYC orchestration |
| **Alloy** | KYC/CDD platform |
| **ComplyAdvantage** | AML/sanctions screening |
| **Refinitiv World-Check** | PEP/sanctions data |

## 8. Fraud & Risk Vendors

| Vendor | Capability |
|--------|-----------|
| **Feedzai** | Real-time fraud & risk |
| **FICO** | Scores + fraud |
| **SAS** | Risk analytics |
| **DataVisor** | Unsupervised fraud |
| **Sift** | Digital fraud |
| **Featurespace** | ARIC adaptive behavioral analytics |

## 9. RegTech / Compliance

| Vendor | Capability |
|--------|-----------|
| **Ayasdi** | Model risk (ML) |
| **Ascent** | Regulatory intelligence |
| **ComplyAdvantage** | Screening |
| **Beacon** | Compliance automation |
| **Nasdaq Verafin** | AML/fraud for FIs |
| **CUBE** | Regulatory change |

## 10. Trading / Capital Markets

| Vendor | Capability |
|--------|-----------|
| **Numerai** | Crowdsourced hedge fund |
| **Kensho (S&P)** | Analytics/NLP |
| **AlphaSense** | Market intelligence |
| **Bloomberg Terminal** | Data + GPT |
| **QuantConnect** | Algo trading platform |

## 11. Insurance (InsurTech)

| Vendor | Capability |
|--------|-----------|
| **Tractable** | CV damage assessment |
| **Lemonade** | AI-native insurer |
| **Shift Technology** | Claims fraud |
| **Cape Analytics** | Property CV |

## 12. Wealth / Robo-Advisory

| Vendor | Capability |
|--------|-----------|
| **Betterment / Wealthfront** | Robo-advisors |
| **Vanguard / Schwab** | Hybrid advisory |
| **SigFig** | B2B robo |

## 13. Cloud AI Platforms (Finance-grade)

| Cloud | Service |
|-------|---------|
| **AWS** | SageMaker, Fraud Detector, Kendra |
| **Azure** | Azure AI, Purview (governance) |
| **GCP** | Vertex AI, Document AI |
| **Snowflake** | Data cloud + Cortex ML |

## 14. Vector / Graph Databases

| Tool | Use |
|------|-----|
| **Neo4j** | AML graph (`03` §6) |
| **TigerGraph** | Real-time graph |
| **pgvector / Chroma / Weaviate** | RAG (`04`) |

## 15. Explainability Tooling

| Tool | Use |
|------|-----|
| **SHAP** | Local/global attributions |
| **LIME** | Local surrogate |
| **Alibi Explain** | Permutation, anchors |
| **Captum** | PyTorch attributions |

Full catalog in `64-AI-Model-Explainability-and-XAI/`.

## 16. Evaluation & Benchmarking

| Tool | Use |
|------|-----|
| **Evidently** | Drift |
| **Deepchecks** | Data/模型校验 |
| **Great Expectations** | Data contracts |
| **Custom suites** | Finance metrics (`58`) |

## 17. Privacy & Security

| Tool | Use |
|------|-----|
| **OpenMined PySyft** | Federated (`40`) |
| **Google Differential Privacy** | DP |
| **Vault / KMS** | Secret mgmt |
| **Guardrails** | LLM injection (`18`) |

## 18. Build vs Buy Matrix

| Capability | Build | Buy | Rationale |
|-----------|-------|-----|-----------|
| Risk model | ✅ | | Core IP |
| Fraud | ✅/⬜ | ✅ | Hybrid common |
| KYC | | ✅ | Compliance burden |
| LLM advisor | ✅ | ⬜ | On platform |
| RegTech | | ✅ | Specialized |
| Feature store | ⬜ | ✅ | Mature OSS |

## 19. Selection Criteria

1. **Regulatory fit** — audit, explainability, data residency (`40`).
2. **Latency** — real-time scoring needs <50ms (`62`).
3. **Explainability** — native SHAP/audit (`64`).
4. **Security** — encryption, access control (`22`).
5. **TCO** — balance build vs buy (`41`, `66`).

## 20. Integration Pattern

```text
Vendor KYC → Event → Feature Store → In-house Risk Model
                                    → Scoring API → Decision + XAI
                                    → Audit (43) → Monitoring (56)
```

Keep core IP in-house; buy commoditized capabilities.

## 21. Open-Source Starter Stack

```text
Kafka + Spark        → data
Feast                → features
XGBoost + PyTorch    → models
MLflow + KServe      → MLOps (56)
SHAP + Evidently     → XAI + drift (64,56)
LangChain + pgvector → RAG advisory (04)
Guardrails           → safety (52)
```

## 22. LLM Cost Control

Use `59-AI-Agent-Financial-Governance-and-Cost-Control/` to cap token spend per advisory session; route to small models (`30`) for triage.

## 23. Emerging Tooling

- **Agentic platforms** (`44-Agentic-Platforms-and-Enterprise-Collaboration/`) for finance ops.
- **Small models** (`30-Small-Language-Models/`) on edge for auth (`62`).
- **Commodity models** (`66`) lowering build cost.

## 24. Vendor Risk

Every vendor adds third-party risk:
- Data residency (`40`).
- SOC 2 / ISO 27001.
- Model transparency (`64`).
- Exit / portability.

## 25. Reference Implementations

- Feast fraud example: feature store + online scoring.
- PyG AML: graph neural network for SAR triage.
- LangChain + pgvector: grounded advisor.

## 26. Tooling by Sub-field

| Sub-field | Primary tools |
|-----------|---------------|
| Credit | XGBoost, SHAP, MLflow |
| Fraud | Feedzai/DataVisor, Feast, graph |
| Trading | PyTorch TS, SB3, AlphaSense |
| Insurance | Tractable CV, Shift |
| Wealth | RAG LLM, optimization |
| KYC | Jumio/Onfido, ComplyAdvantage |
| RegTech | Ascent, Verafin |

## 27. Cloud Comparison

| Need | AWS | Azure | GCP |
|------|-----|-------|-----|
| Model train | SageMaker | Azure ML | Vertex |
| Doc AI | Textract | Form Recognizer | Document AI |
| Search | Kendra | Cognitive | Vertex Search |
| Governance | Macie | Purview | DLP |

## 28. Deployment Checklist

- [ ] Feature store live (online+offline)
- [ ] Model in registry (MLflow)
- [ ] SHAP reasons in decision payload
- [ ] Audit log to immutable store (`43`)
- [ ] Drift monitor active (`56`)
- [ ] Fair-lending tests passing (`55`)
- [ ] Rollback runbook tested
- [ ] Cost budget set (`59`)

## 29. Cost Levers

- Batch where possible.
- Cache embeddings.
- Model cascade (`53`).
- Smaller models for triage (`30`).
- Spot/preemptible training.

## 30. Community & Learning

- Kaggle finance competitions (credit, fraud).
- PyData / FinML conferences.
- Open datasets: LendingClub, IEEE fraud, Sparkov.

## 31. Anti-Patterns

| Anti-pattern | Consequence |
|-------------|-------------|
| No feature store | Skew, incidents |
| Ungrounded LLM | Hallucinated advice (`52`) |
| No audit log | Unreconstructable decisions (`43`) |
| Skip validation | Model risk (`64`) |
| Shadow AI | Compliance breach (`55`) |

## 32. Summary

Finance AI tooling is mature and layered. Win by integrating:
- In-house core models (IP).
- Best-of-breed vendors (KYC, RegTech).
- Strong MLOps + XAI + audit backbone.

Continue to `05-Future-Outlook.md`.

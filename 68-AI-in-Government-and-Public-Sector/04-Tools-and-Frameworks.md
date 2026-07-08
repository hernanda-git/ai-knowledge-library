# AI in Government — Tools and Frameworks

> A vendor- and OSS-neutral map of the tools, platforms, and frameworks used to build, govern, and operate public-sector AI — organized by layer, with sovereign-cloud options and evaluation harnesses. Cross-referenced with `56-MLOps`, `04-RAG`, `64-XAI`, `40-Sovereignty`, `58-Evaluation`.

---

## 1. Layer Map

```
┌──────────────────────────────────────────────────────────┐
│  Citizen Experience   (chat, portal, voice, kiosk)       │
├──────────────────────────────────────────────────────────┤
│  AI Application Layer  (RAG, scoring, CV, agents)        │
├──────────────────────────────────────────────────────────┤
│  Model Layer  (LLM, GBT, CV, multimodal)                │
├──────────────────────────────────────────────────────────┤
│  Data & Vector Layer  (DB, lakehouse, vector store)     │
├──────────────────────────────────────────────────────────┤
│  Governance Layer  (registry, AIA, fairness, logging)   │
├──────────────────────────────────────────────────────────┤
│  Infrastructure Layer  (sovereign cloud, on-prem, edge) │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Citizen Experience Layer

| Tool | Type | Notes |
|------|------|-------|
| **Gov.uk Notify / Pay** | Gov service APIs (UK) | Transactional messaging & payments |
| **Twilio / Vonage** | Comms | Multichannel citizen contact |
| **Microsoft Power Platform** | Low-code | Agency-built assistants |
| **Streamlit / Gradio** | Prototyping | Rapid internal demos |
| **WCAG audit tools** | Accessibility | Must meet AA |

> UI must be multilingual & accessible (cf. `50-Multimodal`).

---

## 3. AI Application Layer (RAG & Agents)

| Tool | Purpose | Cross-ref |
|------|---------|-----------|
| **LangChain** | RAG orchestration | `04-RAG` |
| **LlamaIndex** | Indexing / retrieval | `04-RAG` |
| **Haystack** | Pipeline framework | `04-RAG` |
| **CrewAI / AutoGen** | Multi-agent casework | `03-Agents` |
| **Haystack Agents** | Tool-using assistants | `03-Agents` |
| **DSPy** | Prompt optimization | `02-LLMs` |

### Minimal GovLLM stack example

```text
LlamaIndex (ingest statutes)
   → Qdrant (vector store)
   → GovLLM (RAG-grounded)
   → Guardrails (refusal + citation)
   → Decision log (immutable)
```

---

## 4. Model Layer

### LLMs (sovereign-friendly)
| Model | Deployment | Notes |
|-------|-----------|-------|
| **Llama 3 / 4 (Meta)** | Self-host (`23-Local-AI`) | Open-ish weights |
| **Mistral / Mixtral** | Self-host | EU-origin, sovereignty-friendly |
| **Qwen** | Self-host | Multilingual |
| **Command R+ (Cohere)** | API / private | RAG-tuned |
| **GPT / Claude / Gemini** | API | Use with sovereign clauses (`40`) |

### Scoring / Tabular
| Tool | Use |
|------|-----|
| **XGBoost** | Benefits, tax scoring |
| **LightGBM** | Large-scale tabular |
| **sklearn** | Baselines |
| **PyTorch / JAX** | Custom DL |

### Computer Vision
| Tool | Use |
|------|-----|
| **YOLO / Ultralytics** | Inspection, object detection |
| **Detectron2** | Segmentation |
| **OpenCV** | Pre/post-processing |

---

## 5. Data & Vector Layer

Cross-ref `37-AI-Native-Databases`.

| Tool | Role |
|------|------|
| **Qdrant** | Vector search (sovereign-deployable) |
| **Weaviate** | Vector + hybrid |
| **pgvector (Postgres)** | SQL + vectors |
| **Milvus** | Large-scale vectors |
| **Chroma** | Lightweight RAG dev |
| **Delta Lake / Iceberg** | Lakehouse governance |
| **dbt** | Transformation & lineage |

---

## 6. Governance Layer (the differentiator)

| Capability | Tools / Standards |
|------------|-------------------|
| **AI Registry** | Internal catalog; EU database for high-risk |
| **AIA** | Canada DADM templates; custom |
| **Fairness** | AIF360, Fairlearn, What-If Tool |
| **XAI** | SHAP, LIME, Captum (`64`) |
| **Provenance** | OpenLineage, custom hash-chain (`43`) |
| **Policy** | OPA (Open Policy Agent) |
| **Transparency** | Algorithmic transparency records (`21`) |

```python
# Fairlearn demographic parity check
from fairlearn.metrics import demographic_parity_difference
dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=group)
print(f"Demographic parity difference: {dpd:.3f}  (target ≤ 0.1)")
```

---

## 7. Infrastructure & Sovereign Cloud

Cross-ref `40-Data-Sovereignty`, `25-Multi-Cloud-AI`, `23-Local-AI`.

| Provider | Posture |
|----------|---------|
| **AWS GovCloud** | US public-sector isolated |
| **Azure Government** | US FedRAMP High |
| **Google Cloud Public Sector** | US / EU |
| **EU sovereign clouds** (e.g., national) | Data-residency guaranteed |
| **On-prem (K8s + vLLM/Ollama)** | Maximum control (`23`) |

### Decision: where to host?

```
Data sensitivity HIGH → sovereign / on-prem
Data sensitivity LOW  → shared gov cloud (cost-efficient)
```

---

## 8. MLOps & Observability

Cross-ref `56-MLOps-and-AI-Platform-Engineering`.

| Tool | Role |
|------|------|
| **MLflow** | Experiment + model registry |
| **Kubeflow / ZenML** | Pipelines |
| **Evidently** | Drift & data-quality |
| **Great Expectations** | Data validation |
| **OpenTelemetry** | Tracing + metrics |
| **Grafana / Prometheus** | Dashboards & alerts |
| **Arize / WhyLabs** | LLM observability |

---

## 9. Evaluation Harnesses

Cross-ref `58-AI-Evaluation-and-Benchmarking-at-Scale`.

| Tool | Focus |
|------|-------|
| **Ragas** | RAG faithfulness / answer quality |
| **DeepEval** | LLM unit tests |
| **Giskard** | ML testing & bias scan |
| **LangSmith / Langfuse** | Tracing & evals |
| **HELM / HELM-Lite** | Holistic model eval |

### Example CI eval gate

```yaml
eval-gate:
  steps:
    - ragas faithfulness >= 0.95
    - fairlearn dpd <= 0.10
    - evidently psi < 0.10
    - redteam leak == 0
  on_fail: block_promotion
```

---

## 10. Security & Red-Teaming Tooling

Cross-ref `18-Agent-Security`, `61-AI-Red-Teaming`, `22-AI-Cybersecurity`.

| Tool | Use |
|------|-----|
| **PyRIT** (Microsoft) | Red-team LLM attacks |
| **Garak** | LLM vulnerability scanner |
| **OWASP LLM Top 10** | Threat taxonomy |
| **Vault** | Secrets |
| **Falco** | Runtime security |

---

## 11. Identity & Access

| Standard | Use |
|----------|-----|
| **OIDC / OAuth2** | Citizen & officer auth |
| **W3C VCs / DIDs** | Verifiable credentials |
| **FIDO2 / passkeys** | Phishing-resistant login |
| **RBAC / ABAC** | Authorization |

---

## 12. Document & Case Management

| Tool | Use |
|------|-----|
| **SharePoint / Records Mgmt** | Case files |
| **OpenText / Box Gov** | Document governance |
| **Camunda / Temporal** | Durable case workflows (`31-Workflow-Orchestration`) |

---

## 13. Vendor Landscape (illustrative, non-endorsement)

| Segment | Examples |
|---------|----------|
| Gov AI platforms | Palantir, C3 AI, Snowflake for Gov |
| Chatbot SaaS | custom LangChain + GovLLM |
| Analytics | SAS, Databricks Gov |
| Cloud | AWS/Azure/Google Gov clouds |
| OSS stack | LlamaIndex + Qdrant + vLLM + MLflow |

> Prefer **right-to-audit** and **data-residency** clauses in any contract (`21-Regulation`).

---

## 14. Build vs. Buy Matrix

| Component | Build | Buy |
|-----------|-------|-----|
| Foundation model | Rarely | Usually (API or hosted) |
| RAG over law | ✅ (agency owns corpus) | Partial |
| Scoring models | ✅ (agency data) | Sometimes |
| Fairness/XAI | ✅ (policy-specific) | Tooling buy |
| Cloud | — | ✅ (sovereign) |
| UI | ✅ (citizen-specific) | Low-code buy |

---

## 15. Reference OSS Stack (fully self-hostable)

```
Frontend:   Streamlit / custom (WCAG AA)
Assistant:  LlamaIndex + GovLLM (Llama/Mistral) + Guardrails
Vectors:    Qdrant (in-region)
Scoring:    XGBoost + SHAP
Fairness:   Fairlearn + AIF360
MLOps:      MLflow + Evidently + OTel
Logging:    Immutable append-only store
Red-team:   PyRIT + Garak
Deploy:     Kubernetes (sovereign namespace)
```

---

## 16. Cost Levers

| Lever | Effect |
|-------|--------|
| Model cascading (`53`) | Route easy queries to small model |
| Caching | Cut LLM calls |
| Sovereign on-prem | Higher capex, lower per-call |
| Batch scoring | Cheaper than real-time |
| Open-weight models | No per-token fee |

> Tie spend to outcomes — see `41-AI-Cost-Optimization`, `59-Agent-Financial-Governance`.

---

## 17. Interoperability Standards

- **NIST AI RMF 1.0**
- **ISO/IEC 42001** (AI MS)
- **EU AI Act** conformity
- **Schema.org / open data** for publishing
- **OpenTelemetry** for observability

---

## 18. Tooling Selection Checklist

- [ ] Runs in required region (sovereignty)
- [ ] Supports right-to-audit
- [ ] Emits provenance (`43`)
- [ ] Integrates fairness scanner (`64`)
- [ ] Has eval hooks (`58`)
- [ ] Red-team compatible (`61`)
- [ ] Accessible UI (WCAG)

---

## 19. Anti-Patterns in Tooling

- Buying a black-box SaaS with no audit rights.
- Using a general chatbot without legal grounding → hallucinations (`52`).
- Skipping fairness tooling to "move fast."
- Centralizing all inference off-region (privacy breach).

---

## 20. Summary

The public-sector toolchain is **ordinary ML tooling wrapped in extraordinary governance**. The winning stack is open, sovereign-deployable, audit-friendly, and evaluation-gated. Vendor lock-in is the top procurement risk — mitigate with standards and right-to-audit.

> Next: `05-Future-Outlook.md` — agentic government, AI constituents, and the 2030 trajectory.

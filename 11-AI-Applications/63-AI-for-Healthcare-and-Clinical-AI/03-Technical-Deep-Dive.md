# AI for Healthcare and Clinical AI — Technical Deep Dive

> Implementation patterns, data engineering, modeling, validation, and MLOps for AI that touches patients. Assumes familiarity with ML basics (see [01-Foundations](../01-Foundations/01-Overview.md)) and agents ([03-Agents](../03-Agents/01-Overview.md)).

## 1. Data Foundations

### 1.1 Sources & Standards
| Data | Standard | Library/format |
|---|---|---|
| EHR | **FHIR** (HL7) | JSON bundles, REST |
| Imaging | **DICOM** | Pixel data + tags |
| Labs/terms | LOINC, SNOMED-CT, RxNorm | Code maps |
| Genomics | HL7 FHIR Genomics, VCF | Variant records |
| Notes | CDA / plain text | NLP |

### 1.2 Extracting Structured Data from the EHR
```python
from fhirclient import client

smart = client.Server("https://ehr.example.org/fhir", app_id="healthai")
bundle = smart.get("Patient/{id}/Observation?category=laboratory")
labs = [obs["valueQuantity"]["value"] for obs in bundle["entry"]]
```

### 1.3 De-identification (HIPAA Safe Harbor / Expert)
Before any modeling on PHI:
```python
# Pseudonymize
df["patient_id"] = df["patient_id"].map(hash_map)   # stable salted hash
# Strip 18 HIPAA identifiers (names, dates, geo, etc.)
text = redact_phi(note_text, phi_model)             # Presidio / custom
```
See [40-Privacy](../40-AI-Data-Sovereignty-and-Privacy/01-Overview.md).

### 1.4 Label Scarcity
Clinical labels are expensive. Common strategies:
- **Weak labels** from reports (NLP-extracted positive/negative).
- **Distant supervision** from orders/billing codes.
- **Active learning** — prioritize uncertain cases for expert review.
- **Foundation models** pre-trained on large unlabeled imaging/physics corpuses, then fine-tuned.

## 2. Modeling Patterns

### 2.1 Tabular Risk Models (CDS)
Gradient-boosted trees (XGBoost/LightGBM) still dominate structured EHR risk scores.

```python
import lightgbm as lgb
train = lgb.Dataset(X_tr, label=y_tr)
model = lgb.train({"objective":"binary","metric":"auc"},
                  train, num_boost_round=500,
                  valid_sets=[lgb.Dataset(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30)])
```

### 2.2 Calibration Matters More Than AUC
A risk score used for thresholds must be **calibrated** (predicted probability ≈ true frequency).

```python
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(model, method="isotonic")
calibrated.fit(X_val, y_val)         # recalibrate on site-specific data
```

### 2.3 Imaging Models
CNN / ViT backbones on DICOM volumes. Key tactic: **test-time augmentation + uncertainty**.

```python
import torch
logits = torch.stack([model(tta_img(i)) for i in range(8)])
prob = torch.softmax(logits, -1).mean(0)
uncert = prob.max(-1).values.std()      # high uncertainty -> human review
```

### 2.4 Clinical LLMs (scribe, summarization)
Use RAG over the patient chart + strict output schema.

```python
note = clinical_llm.generate(
    system="You are a scribe. Output a SOAP note. Never invent findings.",
    context=retrieve_chart(patient_id, top_k=20),   # FHIR-derived
    audio_transcript=asr_text,
    schema=SOAP_SCHEMA,                  # structured, validated
)
assert faithfulness_check(note, asr_text)   # see 52
```

### 2.5 Time-Series / Vital Models
ICU deterioration uses recurrent or transformer models over multivariate vital time series.

```python
# conceptual: sliding-window transformer over vitals
window = vitals.rolling("1h").tensor()        # [T, features]
score = deterioration_transformer(window)     # hourly risk
```

## 3. Explainability & Evidence

Clinicians will not trust a number without a reason.

```python
import shap
expl = shap.Explainer(model)
sv = expl(X_sample)
# Surface top contributing features per patient
top_feats = sv[patient_idx].abs().argsort()[::-1][:5]
```

Show: contributing vitals/labs, contraindications, similar past cases. Use [18-Agent-Security](../18-Agent-Security-and-Trust/01-Overview.md) patterns for trust.

## 4. Evaluation — Clinically Rigorous

Go beyond test AUC. See [58-Evaluation](../58-AI-Evaluation-and-Benchmarking-at-Scale/01-Overview.md).

| Dimension | Method |
|---|---|
| Discrimination | AUC, AUPRC |
| Calibration | Reliability curve, Brier |
| Subgroup fairness | Per-demographic metrics (see 55) |
| External validation | Holdout site(s) not in training |
| Prospective | Silent trial → pilot → RCT |
| Drift | Continuous monitoring of inputs + scores |

```python
from sklearn.metrics import roc_auc_score, average_precision_score
auc = roc_auc_score(y_test, proba)
auprc = average_precision_score(y_test, proba)   # imbalanced-friendly
```

**Silent trial:** run the model in production feeding a worklist *without* clinicians seeing it; compare decisions to measure impact safely before go-live.

## 5. MLOps for Clinical AI

Treat models as medical-adjacent software.

- **Versioning:** data + code + model + labels (reproducibility). See [56-MLOps](../56-MLOps-and-AI-Platform-Engineering/01-Overview.md).
- **Monitoring:** input drift, score drift, label drift, performance decay.
- **Change control:** for FDA PCCP devices, document every allowed update.
- **Audit trail:** every prediction logged with inputs, version, clinician action.
- **Rollback:** one-click revert to prior model version.

```yaml
# example drift monitor
monitor:
  input_drift: psi_threshold 0.2
  score_drift: ks_threshold 0.1
  performance: auc_floor 0.85
  alert: page_oncall
```

## 6. Agentic Clinical Workflows — Engineering

For prior-auth / referral / care-plan agents, apply [03-Agents](../03-Agents/01-Overview.md) patterns with extra guards:

```python
class ClinicalPriorAuthAgent:
    def __init__(self, ehr, policy_kb, reviewer):
        self.ehr = ehr; self.policy_kb = policy_kb; self.reviewer = reviewer
    def run(self, case):
        facts = self.ehr.extract(case)                 # FHIR read-only
        matched = self.policy_kb.match(facts)          # policy retrieval
        draft = llm_draft(facts, matched)              # structured request
        return self.reviewer.human_signoff(draft)      # NEVER auto-submit
```

Guardrails: read-only EHR by default, no autonomous orders, mandatory human sign-off, full trace (see [18](../18-Agent-Security-and-Trust/01-Overview.md), [20](../20-Agent-Infrastructure-and-Observability/01-Overview.md)). Add durable execution from [31](../31-AI-Workflow-Orchestration-and-Durable-Execution/01-Overview.md) for long-running prior-auth.

## 7. Privacy-Preserving Training

When data cannot leave an institution:
- **Federated learning** across hospital sites.
- **Differential privacy** for shared model updates.
- **Synthetic data** for development (see [51-Synthetic-Data](../51-Synthetic-Data-Generation/01-Overview.md)).

```python
# conceptual federated round
global_model = average([site.train(local_copy(global_model)) for site in sites])
```

## 8. Safety Patterns Checklist

- [ ] Calibrated + externally validated
- [ ] Explainability surfaced
- [ ] Human-in-the-loop for high stakes
- [ ] Alert budget tuned (combat fatigue)
- [ ] PHI de-identified / access-controlled
- [ ] Drift + performance monitored
- [ ] Audit log immutable
- [ ] Rollback path
- [ ] Bias checked across subgroups
- [ ] Failure-mode playbook
- [ ] Faithfulness check for any generated text

## 9. Reference Architecture (End-to-End)

```
[EHR/FHIR] [DICOM PACS] [Wearables]
        │  ingest + de-identify
        ▼
[Feature Store] ──► [Model Registry] ──► [Inference Svc]
                                        │  + evidence/shap
                                        ▼
[CDS Worklist / Scribe UI / Alert] ──► Clinician
        │                                      │
        ▼                                      │
[Audit + Monitoring + Drift] ◄────────────────┘
```

## 10. Common Pitfalls (Postmortem-style)

1. **Trained on one hospital, shipped to another** → silent performance collapse. Fix: external validation + drift monitors.
2. **Optimized AUC, ignored calibration** → thresholds meaningless. Fix: isotonic calibration.
3. **Alert on every prediction** → fatigue, muted. Fix: precision-tuned worklist.
4. **LLM scribe invented a symptom** → patient harm. Fix: faithfulness check ([52](../52-AI-Hallucination-Detection-and-Mitigation/01-Overview.md)).
5. **No audit trail** → cannot investigate adverse event. Fix: log everything.
6. **Ignored equity** → fails underrepresented group. Fix: subgroup eval ([55](../55-AI-Ethics-and-Responsible-AI/01-Overview.md)).
7. **No rollback** → bad model stuck in prod. Fix: versioned serving.

## 11. Tooling Snippets Recap
- FHIR client for EHR — §1.2
- PHI redaction — §1.3
- LightGBM risk — §2.1
- Calibration — §2.2
- Imaging uncertainty — §2.3
- Clinical LLM schema — §2.4
- Vital transformer — §2.5
- SHAP — §3
- Evaluation — §4
- Drift YAML — §5
- Federated — §7

## 12. Reproducibility & Audit
Every prediction should be reproducible: store model version, feature version, input hash, output, and clinician action. This supports both regulatory audits and [43-Provenance](../43-AI-Data-Provenance-and-Content-Authenticity/01-Overview.md) requirements.

## What's Next
- **04-Tools-and-Frameworks.md** — the vendor/open-source landscape.
- **05-Future-Outlook.md** — agents, multimodal clinics, regulation.

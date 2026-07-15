# Technical Deep Dive: Building and Validating Clinical AI

> Engineering rigor for healthcare AI — dataset construction, training methodology, the validation science that separates lab wins from clinical wins, deployment architecture, and the monitoring that keeps models honest after launch.

## 1. The Validation Ladder

```
Level 0  Retrospective, single-site, random split      (weakest)
Level 1  Retrospective, multi-site, group split
Level 2  Retrospective, temporal split (future data)
Level 3  Prospective, silent (model runs, clinician blind)
Level 4  Prospective, randomized controlled deployment   (strongest)
```

Most published "AI beats doctors" claims live at Level 0–1. Regulatory and clinical trust require Level 2–4. **Always state which level your evidence is.**

## 2. Dataset Construction

### 2.1 De-identification first

Never build on raw PHI. Use:
- **DICOM:** `pydicom` + `deid` libraries; strip private tags, burn-in pixel text (annotations on images!).
- **Free text:** `nlpier` / `Philter` / MITRE's `Presidio` for PHI scrubbing.
- **Structured:** OMOP + date shifting per patient (preserve intervals, hide absolute dates).

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def deid_note(text: str) -> str:
    results = analyzer.analyze(text=text, language="en")
    return anonymizer.anonymize(text=text, analyzer_results=results).text
```

### 2.2 Label provenance

| Label source | Quality | Risk |
|---|---|---|
| Expert annotation | High | Cost, rarity |
| Radiology report NLP | Medium | Noisy, needs confirmation |
| Billing codes | Low-med | Upcoding, leakage |
| Proxy outcomes | Variable | Hidden confounding |

### 2.3 Handling class imbalance

Clinical events are rare (readmission ~15%, sepsis ~5%). Use:
- AUPRC as primary metric (not AUC alone).
- Class weights / focal loss.
- Stratified + group-aware resampling.

---

## 3. Training Methodology

### 3.1 Imaging

- **Transfer learning:** ImageNet → pathology/radiology pretrained backbones (DenseNet, ConvNeXt, ViT).
- **Self-supervised:** SimCLR / MAE on unlabeled scans to exploit the huge unlabeled pool.
- **Monai workflows:** `monai.apps` + `monai.data.CacheDataset` for GPU-fed pipelines.

```python
from monai.data import CacheDataset, DataLoader
from monai.transforms import RandFlipd, RandScaleIntensityd, Compose

val_tfms = Compose([
    LoadImaged(keys=["img"]), EnsureChannelFirstd(keys=["img"]),
    ScaleIntensityd(keys=["img"]), ToTensord(keys=["img"]),
])
ds = CacheDataset(data=val_files, transform=val_tfms, cache_rate=0.4, num_workers=4)
loader = DataLoader(ds, batch_size=4, shuffle=False)
```

### 3.2 Clinical language models

- Domain-adaptive pretraining (DAPT) on clinical corpora (MIMIC, ClinicalTrials).
- Instruction-tune for SOAP, discharge summary, Q&A over guidelines.
- **Retrieval-grounding** is essential: pair the LLM with `04-RAG` over UpToDate/guidelines to avoid hallucinated citations.

### 3.3 Multimodal fusion

Late fusion (separate encoders → concat) is more robust and debuggable than early fusion. Use cross-attention only when data is tightly aligned.

---

## 4. The Evaluation Science

### 4.1 Beyond a single number

Report:
- Sensitivity / Specificity at the **chosen operating point**.
- AUC and AUPRC with 95% bootstrap CIs.
- **Subgroup tables** (by age band, sex, ethnicity, insurance, comorbidity).
- Calibration (Brier score, reliability curve).
- Decision-curve analysis (net benefit vs treat-all / treat-none).

### 4.2 Subgroup evaluation skeleton

```python
import pandas as pd
def subgroup_auc(df, subgroup_col, score="pred", y="y"):
    rows = []
    for val, g in df.groupby(subgroup_col):
        if g[y].nunique() < 2:
            continue
        rows.append({
            subgroup_col: val,
            "n": len(g),
            "auc": roc_auc_score(g[y], g[score]),
        })
    return pd.DataFrame(rows)
```

### 4.3 Reader studies

Head-to-head with clinicians needs:
- **Enough readers** (≥3) for CI on reader performance.
- **Counterbalanced** (reader sees AI suggestion on randomized half).
- Report **AI assistance effect** (does the human get better *with* the model?).

### 4.4 Prospective silent trial design

```text
Production traffic ──▶ Copy of request ──▶ Shadow model
                              │                  │
                              └──► Clinician sees NO model output
                                                 │
                              Outcomes recorded ──► Compare model vs actual care
```

This measures real-world performance without changing care.

---

## 5. Deployment Architecture

### 5.1 Reference topology

```
EHR/FHIR ──▶ Feature store (OMOP-derived)
                  │
            Inference service (container, GPU pool)
                  │  model card + version pinned
            CDS hook / PACS plugin / scribe service
                  │
            Audit log + drift monitor ──▶ Alerting
```

### 5.2 Inference service sketch

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("models/readmit_v3.joblib")   # pinned version

class Req(BaseModel):
    features: dict
    patient_id: str
    model_version: str = "readmit_v3"

@app.post("/score")
def score(req: Req):
    p = float(model.predict_proba([req.features])[0, 1])
    return {"patient_id": req.patient_id, "risk": p,
            "model_version": req.model_version,
            "threshold": 0.18}
```

### 5.3 Model cards (required)

A clinical model card must include: intended use, training data source/sites, performance by subgroup, known failures, and monitoring plan. Template aligns with `55-AI-Ethics-and-Responsible-AI`.

---

## 6. Monitoring & Drift (the part everyone skips)

### 6.1 Three drift types

| Type | Example | Detection |
|---|---|---|
| Data drift | New scanner model | Feature distribution test (KS) |
| Concept drift | Protocol change | Labeled-sample AUC tracking |
| Population drift | New patient mix | Cohort demographic monitoring |

### 6.2 Drift monitor skeleton

```python
from scipy.stats import ks_2samp

def feature_drift(ref: list[float], live: list[float], thr=0.1) -> bool:
    stat, p = ks_2samp(ref, live)
    return p < 0.05 and stat > thr
```

### 6.3 Triggers for revalidation

- Performance drop beyond CI band.
- Significant data/concept drift.
- New site onboarding.
- Regulatory change.

---

## 7. Privacy-Preserving Training

| Technique | Use | Trade-off |
|---|---|---|
| De-identification | Baseline | Still re-identifiable if linked |
| Federated learning | Multi-hospital | Comms overhead, heterogeneous data |
| Differential privacy | Strong guarantee | Utility loss |
| Synthetic data | Share-able dev sets | Fidelity vs privacy (`51-Synthetic-Data-Generation`) |

Federated example concept:

```text
Hospital A ─┐
Hospital B ─┼──▶ aggregate gradients (secure) ──▶ global model
Hospital C ─┘        (no raw data leaves site)
```

---

## 8. Explainability in the Clinic

Clinicians trust what they understand. Use:
- **Grad-CAM / attention maps** for imaging.
- **SHAP / permutation importance** for tabular.
- **Retrieved guideline snippets** for LLM CDS (grounding = explainability).
- See `07-xai-explainability` (referenced in cron reports) and `52-AI-Hallucination-Detection-and-Mitigation`.

---

## 9. Common Engineering Anti-Patterns

- Training on `train.csv` that includes the test patient's earlier visits (leakage).
- Reporting AUC only, hiding that AUPRC is near base rate.
- No version pin → silently changed model.
- "Launch then monitor" with no monitor built.
- Treating a wellness-model threshold as clinical.

---

## 10. MLOps for Regulated AI

- Immutable model registry with approval gate.
- Reproducible environments (lockfiles, containers).
- Audit trail for every inference (who/what/version/output).
- Rollback path (shadow → canary → full).
- Aligns with `20-Agent-Infrastructure-and-Observability` and `31-AI-Workflow-Orchestration-and-Durable-Execution`.

---

## 11. Security & Adversarial Robustness

- Medical imaging is susceptible to **adversarial patches** (e.g., a sticker confusing a classifier). Validate with adversarial testing (`65-AI-for-Cybersecurity`).
- Prompt injection in scribe LLMs (malicious instruction in patient speech) — mitigate per `18-Agent-Security-and-Trust`.
- PHI exfiltration via model memorization — differential privacy / unlearning.

---

## 12. Putting It Together: A 12-Week Plan

| Week | Work |
|---|---|
| 1–2 | De-id data, define label, OMOP normalize |
| 3–4 | Baseline (XGBoost / MONAI), group splits |
| 5–6 | Subgroup eval, calibration, threshold pick |
| 7–8 | Shadow service, audit logging |
| 9–10 | Silent prospective window |
| 11–12 | Model card, monitoring dashboards, go/no-go |

---

## 13. Metrics Cheat-Sheet

| Metric | Use when |
|---|---|
| AUC | Ranking quality, balanced view |
| AUPRC | Rare events (most clinical) |
| Sensitivity | Missed-disease cost is high |
| Specificity | False-alarm cost is high |
| Brier / ECE | Need calibrated probabilities |
| Net benefit | Choosing whether to act on score |

---

## 14. Summary

Clinical AI engineering is 30% modeling and 70% everything-around-the-model: provenance, validation science, deployment, and relentless monitoring. The bar is prospective, subgroup-aware, and auditable.

## 15. Cross-References

- `01-Overview.md`, `02-Core-Topics.md`
- `69-AI-Evaluation-and-LLM-Testing` — eval depth
- `55-AI-Ethics-and-Responsible-AI` — model cards, fairness
- `52-AI-Hallucination-Detection-and-Mitigation` — scribe safety
- `40-AI-Data-Sovereignty-and-Privacy` — PHI, federated
- `20-Agent-Infrastructure-and-Observability` — MLOps
- `51-Synthetic-Data-Generation` — synthetic dev sets

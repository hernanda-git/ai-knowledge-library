# Core Topics in Healthcare & Clinical AI

> The five workhorse application areas of clinical AI — medical imaging, clinical decision support, clinical language/documentation, remote monitoring, and operational AI — each with the data shapes, modeling approaches, and Python examples you need to be productive.

## 1. Topic Map

1. Medical Imaging & Diagnostics
2. Clinical Decision Support & Risk Prediction
3. Clinical Language & Documentation (Ambient AI)
4. Remote Monitoring & Digital Biomarkers
5. Operational / Revenue-Cycle AI

Each section includes a minimal, runnable sketch. These are **illustrative**, not production clinical pipelines.

---

## 2. Medical Imaging & Diagnostics

### 2.1 Where AI helps most

| Modality | Typical task | Why AI fits |
|---|---|---|
| Radiography (X-ray) | Abnormality detection, triage | High volume, clear labels from radiology reports |
| CT/MRI | Lesion detection, segmentation | 3D, time-consuming for humans |
| Pathology (WSI) | Mitosis counting, grading | Gigapixel, repetitive |
| Dermatology | Lesion classification | Consumer + clinic images |
| Ophthalmology | DR, glaucoma screening | Standardized fundus photos |
| Endoscopy | Polyp detection | Real-time, miss-rate reduction |

### 2.2 The DICOM reality

Medical images are not PNGs. They are DICOM files carrying both pixels and a massive metadata header (patient, study, series, modality, windowing). You almost always convert to NumPy arrays + a sidecar metadata table.

```python
import pydicom
import numpy as np

def load_dicom_volume(folder: str) -> tuple[np.ndarray, dict]:
    slices = []
    meta = {}
    for f in sorted(glob(folder + "/*.dcm")):
        ds = pydicom.dcmread(f)
        slices.append(ds.pixel_array.astype(np.float32))
        if not meta:
            meta = {
                "Modality": ds.Modality,
                "PatientID": str(ds.PatientID),
                "StudyDate": str(ds.StudyDate),
                "Rows": ds.Rows, "Cols": ds.Columns,
            }
    vol = np.stack(slices, axis=0)            # (Z, H, W)
    # window/level normalization (HU for CT)
    return vol, meta
```

### 2.3 Minimal segmentation with MONAI

```python
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandRotate90d, ToTensord,
)

train_tfms = Compose([
    LoadImaged(keys=["img", "seg"]),
    EnsureChannelFirstd(keys=["img", "seg"]),
    ScaleIntensityd(keys=["img"]),
    RandRotate90d(keys=["img", "seg"], prob=0.5),
    ToTensord(keys=["img", "seg"]),
])

model = UNet(
    spatial_dims=2, in_channels=1, out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2), num_res_units=2,
)
loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
```

### 2.4 Evaluation nuance

Imaging models must be evaluated with **patient-level splits** (no same-patient leakage between train/val/test) and reported with **sensitivity/specificity at an operating point**, plus confidence intervals. See `69-AI-Evaluation-and-LLM-Testing`.

---

## 3. Clinical Decision Support & Risk Prediction

### 3.1 Classic targets

- Sepsis onset prediction (e.g., SAPS, MEWS, and learned early-warning scores)
- ICU mortality / deterioration
- 30-day readmission
- Length-of-stay
- No-show / cancellation

### 3.2 Tabular is still king

For structured EHR, gradient-boosted trees (XGBoost/LightGBM) frequently beat deep nets. Time-series vitals call for TCNs or transformers.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

df = pd.read_parquet("features/ehr_admissions.parquet")  # OMOP-derived
X = df.drop(columns=["label_readmit_30d", "patient_id"])
y = df["label_readmit_30d"]

# GROUP split by patient to avoid leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
tr, te = next(gss.split(X, y, groups=df["patient_id"]))
X_tr, X_te = X.iloc[tr], X.iloc[te]
y_tr, y_te = y.iloc[tr], y.iloc[te]

clf = xgb.XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
)
clf.fit(X_tr, y_tr)
p = clf.predict_proba(X_te)[:, 1]
print("AUC", roc_auc_score(y_te, p), "AUPRC", average_precision_score(y_te, p))
```

### 3.3 Operating point selection

A risk score with no chosen threshold is not a decision. Pick the threshold from a **clinical utility curve** (net benefit) or a target sensitivity, not raw accuracy.

```python
import numpy as np
def net_benefit(y_true, proba, threshold, prevalence):
    pred = (proba >= threshold).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    n = len(y_true)
    return (tp / n) - (fp / n) * (threshold / (1 - threshold))
```

### 3.4 Calibration matters more than you think

A poorly calibrated probability misleads triage. Use isotonic/Platt scaling and reliability curves; monitor calibration drift post-deployment.

---

## 4. Clinical Language & Documentation (Ambient AI)

### 4.1 The ambient scribe pipeline

```
[Microphone] → ASR (diarized) → Speaker-separated transcript
   → LLM note generator (SOAP) → Structured sections → EHR write-back (FHIR)
```

### 4.2 Sketch: structured note from transcript

```python
from openai import OpenAI  # or any LLM client
client = OpenAI()

SYSTEM = """You are a clinical documentation assistant. Convert the
supplied clinician–patient transcript into a SOAP note. Output valid JSON
with keys: subjective, objective, assessment, plan. Only use information
present in the transcript. If ambiguous, write 'not specified'."""

def draft_soap(transcript: str) -> dict:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": transcript},
        ],
        temperature=0.0,
    )
    return json.loads(r.choices[0].message.content)
```

### 4.3 Why this needs hallucination controls

A scribe that *invents* a medication or allergy is dangerous. Pair generation with:
- **Grounding**: verify drug names against an RxNorm terminology.
- **Constraint decoding / schema**: force structured output.
- **Human review queue** for low-confidence spans.
- See `52-AI-Hallucination-Detection-and-Mitigation`.

### 4.4 Coding & billing (downstream)

Generated notes feed auto-ICD-10/CPT coding. This is operational but clinically consequential — wrong codes mean denied claims or fraud exposure.

---

## 5. Remote Monitoring & Digital Biomarkers

### 5.1 Data shapes

- Continuous: wearables (HR, SpO2, steps), CGM (glucose), ECG patches.
- Behavioral: smartphone keyboard dynamics, gait, voice.
- Patient-reported: ePRO surveys.

### 5.2 Streaming anomaly detection

```python
import numpy as np

class EWMAAlarm:
    """Exponentially weighted mean/variance alarm for a vital sign."""
    def __init__(self, alpha=0.05, k=4.0):
        self.alpha, self.k = alpha, k
        self.mu = None; self.var = 1.0
    def update(self, x: float) -> bool:
        if self.mu is None:
            self.mu = x; return False
        self.mu = self.alpha * x + (1 - self.alpha) * self.mu
        self.var = (1 - self.alpha) * (self.var + self.alpha * (x - self.mu) ** 2)
        z = (x - self.mu) / max(np.sqrt(self.var), 1e-6)
        return abs(z) > self.k
```

### 5.3 Digital biomarkers in practice

- Parkinson's: finger-tapping frequency from phone.
- Heart failure: daily weight + impedance trends.
- Depression: voice prosody + activity entropy.

These feed both **care** (alert clinician) and **trials** (endpoint measurement) — bridging into `42-AI-for-Science-and-Drug-Discovery`.

---

## 6. Operational / Revenue-Cycle AI

| Process | AI role | Caveat |
|---|---|---|
| Scheduling | No-show prediction, slot optimization | Fairness across neighborhoods |
| Prior-auth | Auto-eligibility checks | Payer rule fragility |
| Coding | ICD/CPT suggestion | Audit trail required |
| Length-of-stay | Capacity planning | Don't punish sick patients |
| Supply chain | Demand forecasting | External shocks |

Operational AI is lower-risk than diagnostic AI but touches equity and cost directly. Keep fairness audits (see `55-AI-Ethics-and-Responsible-AI`).

---

## 7. Cross-Cutting Data Engineering

### 7.1 OMOP Common Data Model

Analytics-ready EHR should be normalized to OMOP so models are portable across sites.

```sql
-- Example: 30-day readmission label in OMOP
SELECT p.person_id,
       1 AS label_readmit_30d
FROM omop.visit_occurrence v1
JOIN omop.visit_occurrence v2
  ON v2.person_id = v1.person_id
 AND v2.visit_start_date BETWEEN v1.visit_end_date
     AND v1.visit_end_date + INTERVAL '30' DAY;
```

### 7.2 FHIR for apps

```python
import requests
base = "https://fhir.example.org"
r = requests.get(f"{base}/Observation?patient=Patient/123&code=http://loinc.org|8867-4",
                 headers={"Authorization": "Bearer <token>"})
obs = r.json()["entry"]  # heart rate Observations
```

---

## 8. Choosing a First Project

| If you have… | Start with… |
|---|---|
| A labeled imaging set | MONAI segmentation/classification |
| EHR tables | XGBoost risk model + OMOP |
| Transcript corpus | Ambient SOAP generator + grounding |
| Wearable stream | EWMA/alarm + drift monitor |
| Billing data | Coding assistant under audit |

---

## 9. Pitfalls Specific to Health Data

- **Label leakage:** `discharge_disposition` leaks readmission; `diagnosis` recorded at discharge leaks outcome.
- **Left-censoring:** patients who die before follow-up have no label.
- **Selection bias:** model trained on referred patients won't generalize to screening.
- **Concept drift:** new protocols, new scanners, new ICD codes.

---

## 10. Evaluation Checklist (must-have)

- [ ] Patient/group-level splits (no leakage)
- [ ] Prospective or at least temporal validation
- [ ] Subgroup performance (age, sex, ethnicity, comorbidities)
- [ ] Calibration reported
- [ ] Confidence intervals
- [ ] Silent failure / drift monitoring plan
- [ ] Human-in-the-loop defined for high-risk outputs

---

## 11. Relationship to Agents

Autonomous *clinical* agents (ordering tests, messaging patients) are high-risk and must sit behind the guardrails in `03-Agents` and `18-Agent-Security-and-Trust`. Most deployments today keep the agent in **assistive** mode: draft, suggest, route — human signs.

---

## 12. Summary

The five core topics share one lesson: **narrow, validated, monitored, and trustworthy beats broad and clever.** Code skeletons above are starting points; the deep-dive file covers training/eval/deployment rigor, and the tools file lists production-grade libraries.

## 13. Cross-References

- `01-Overview.md` — definitions, stack, safety culture
- `03-Technical-Deep-Dive.md` — methodology depth
- `04-Tools-and-Frameworks.md` — MONAI, FHIR, cloud HLS
- `69-AI-Evaluation-and-LLM-Testing` — clinical eval specifics
- `52-AI-Hallucination-Detection-and-Mitigation` — scribe safety
- `40-AI-Data-Sovereignty-and-Privacy` — PHI handling

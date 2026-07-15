# Tools and Frameworks for Healthcare AI

> A practictioner's catalog of the open-source and commercial tooling that powers clinical AI — imaging (MONAI), health data standards (FHIR/DICOM/OMOP), language (MedLM-class models), viewers, MLOps, and cloud health platforms — with when to use each.

## 1. Tool Landscape at a Glance

| Category | Open-source | Commercial / Cloud |
|---|---|---|
| Imaging DL | MONAI, NiBabel, SimpleITK | Azure Health Imaging, AWS HLS |
| Health data | FHIR servers (HAPI), OHDSI/OMOP, PyDicom | Snowflake Healthcare, Databricks |
| Language | Med-PaLM (research), local LLMs + RAG | Azure Health Bot, AWS HealthScribe |
| Viewers | OHIF, 3D Slicer, Insight Toolkit | Vendor PACS plugins |
| MLOps | MLflow, Evidently, ClearML | SageMaker, Vertex AI, Azure ML |
| Privacy | Opacus (DP), Substra (FL), PySyft | Cloud confidential compute |
| Eval | MONAI Label, MedEval | Vendor validation suites |

---

## 2. MONAI — Medical Open Network for AI

The de-facto PyTorch ecosystem for medical imaging.

- `monai.networks` — UNet, SegResNet, VISTA3D, SwinUNETR.
- `monai.transforms` — 100+ medical-specific augmentations.
- `monai.apps` — datasets (Medical Segmentation Decathlon), pretrained weights.
- `monai.label` — active learning annotation.
- `monai.deploy` — package models as clinical-grade apps.

```bash
pip install monai[all]
```

```python
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet

model = SegResNet(spatial_dims=3, in_channels=1, out_channels=3)
# 3D sliding-window inference over a full CT volume
out = sliding_window_inference(
    ct_volume, roi_size=(96, 96, 96),
    sw_batch_size=4, predictor=model, overlap=0.5,
)
```

---

## 3. Health Data Standards Tooling

### 3.1 DICOM / imaging

- **pydicom** — read/write DICOM.
- **highdicom** — annotate DICOM with AI results (segmentation, measurements).
- **dcm2niix** — convert to NIfTI for research.

```python
import highdicom as hd
# attach model segmentation as a DICOM Segmentation object
seg = hd.seg.Segmentation(
    source_images=source_sop,
    pixel_array=mask,  # (H, W) or (Z, H, W)
    segmentation_type=hd.seg.SegmentationType.BINARY,
    segment_descriptions=[seg_desc],
)
```

### 3.2 FHIR / structured

- **HAPI FHIR** (Java server), **Firely** (.NET), **Python `fhirclient`**.
- **SMART on FHIR** — authorize apps inside the EHR (EHR-launch).
- **Bulk Data (NDJSON export)** for analytics.

```python
from fhirclient import client
smart = client.FHIRClient(settings={"app_id": "hcare-ai",
                                     "api_base": "https://fhir.example.org"})
from fhirclient.models.observation import Observation
obs = Observation.where(struct={'patient': 'Patient/123'}).perform(smart)
```

### 3.3 OMOP / analytics

- **OHDSI ATLAS** — cohort design, characterization.
- **WhiteRabbit / Rabbit-In-a-Hat** — ETL design.
- **SqlRender** — write once, run on any SQL dialect.

---

## 4. Clinical Language Models

| Model | Notes |
|---|---|
| Med-PaLM / Med-PaLM 2 (research) | Benchmarked on USMLE-style Q&A |
| LLaVA-Med | Multimodal biomedical VLM |
| BioBERT / ClinicalBERT / PubMedBERT | Encoder models for NER/classification |
| Mistral/LLaMA + DAPT | Private, on-prem clinical assistant |
| GPT-class + RAG | Scribe/summarizer with guideline grounding |

> Most production *clinical* deployments avoid sending PHI to third-party APIs; prefer on-prem or BAA-covered hosted models (Azure OpenAI with HIPAA BAA, AWS with BAA).

### 4.1 Grounded clinical Q&A sketch

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Guideline chunks indexed; retrieve before answering
db = FAISS.load_local("guidelines_index", HuggingFaceEmbeddings())
docs = db.similarity_search("first-line tx community-acquired pneumonia", k=3)
context = "\n".join(d.page_content for d in docs)
# pass `context` into the LLM prompt + cite chunk IDs
```

Cross-ref: `04-RAG`, `68-Context-Engineering`.

---

## 5. Viewers & Annotation

- **OHIF** — web DICOM viewer; embed AI results via DICOM SR / Segmentation.
- **3D Slicer** — research segmentation, extensions for MONAI.
- **ITK / SimpleITK** — image processing primitives.

---

## 6. MLOps & Monitoring

| Need | Tool |
|---|---|
| Experiment tracking | MLflow, ClearML, Weights & Biases |
| Drift / data quality | Evidently, NannyML, WhyLabs |
| Registry + approval | MLflow Model Registry, SageMaker Model Registry |
| Serving | Triton, TorchServe, BentoML, FastAPI |
| Orchestration | `31-AI-Workflow-Orchestration-and-Durable-Execution` |
| Observability | `20-Agent-Infrastructure-and-Observability` |

Example drift dashboard hook:

```python
import evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_df, current_data=live_df)
report.save_html("drift.html")
```

---

## 7. Privacy & Federated Tooling

- **PySyft** (OpenMined) — federated learning, DP.
- **Opacus** — differential privacy for PyTorch.
- **Substra** — federated ML for regulated orgs.
- **NVIDIA Flare** — federated learning for healthcare imaging.

```python
import torch
from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private(
    module=model, optimizer=optimizer, data_loader=train_loader,
    noise_multiplier=1.1, max_grad_norm=1.0,
)
```

---

## 8. Cloud Health Platforms

| Cloud | Offering | Use |
|---|---|---|
| AWS | HealthLake, HealthScribe, HealthImaging | FHIR store, scribing, imaging |
| Azure | Azure Health Data Services, Health Bot | FHIR, bots, BAA ML |
| GCP | Healthcare API, Vertex AI | FHIR + autoML |

These provide BAA-covered, HIPAA-eligible infrastructure — important when you cannot run fully on-prem.

```python
# AWS HealthScribe (concept): transcribe + clinical note
import boto3
hs = boto3.client("healthscribe", region_name="us-east-1")
job = hs.start_clinical_note_generation_job(
    SourceBucket="calls", OutputBucket="notes",
)
```

---

## 9. Evaluation Tooling

- **MONAI Label** — human-in-the-loop annotation + active learning.
- **MedEval / MedQA harnesses** — benchmark clinical LLMs.
- **Scikit-learn / scipy** — subgroup/AUC/AUPRC (see `03-Technical-Deep-Dive.md`).
- **Shap / Captum** — explainability.

---

## 10. Selection Guide

| Goal | Start with |
|---|---|
| Train a segmentation model | MONAI + 3D Slicer |
| Serve EHR features | FHIR (HAPI) + OMOP |
| Build a scribe | ASR + LLM + grounding (RAG) |
| Multisite without sharing data | NVIDIA Flare / PySyft |
| Monitor in production | Evidently + Triton + audit log |
| Stay HIPAA-compliant | BAA cloud (Azure/AWS) or on-prem |

---

## 11. Integration Pattern: AI result into PACS

```
Model ──▶ highdicom Segmentation ──▶ DICOM store
                                        │
                                   OHIF viewer shows overlay
                                        │
                                   Radiologist accepts/edits
```

This keeps AI inside radiologist workflow rather than a side portal (higher adoption).

---

## 12. Cost & Scale Notes

- Imaging inference is GPU-heavy; batch off-peak, use mixed precision.
- LLM scribing is token-heavy; cache templates, summarize incrementally.
- Store only what you need; raw video/audio needs retention policy (PHI).
- See `41-AI-Cost-Optimization-and-Enterprise-ROI`.

---

## 13. Tooling Caveats

- Open-source models still need validation before clinical use.
- FHIR implementations vary wildly by EHR vendor — budget integration time.
- DICOM "burned-in" text must be removed before training/sharing.
- Cloud health APIs change fast — pin versions, read release notes.

---

## 14. Summary

The tooling is mature enough that a small team can stand up a credible clinical-AI pipeline in weeks — MONAI for imaging, FHIR/OMOP for data, RAG-grounded LLMs for language, Evidently for monitoring, and BAA clouds for compliance. Tooling is not the bottleneck; **validation and trust are.**

## 15. Cross-References

- `01-Overview.md`, `02-Core-Topics.md`, `03-Technical-Deep-Dive.md`
- `04-RAG`, `68-Context-Engineering` — grounding
- `51-Synthetic-Data-Generation` — synthetic dev data
- `40-AI-Data-Sovereignty-and-Privacy` — PHI, federated
- `20-Agent-Infrastructure-and-Observability` — serving/monitoring
- `41-AI-Cost-Optimization-and-Enterprise-ROI` — cost

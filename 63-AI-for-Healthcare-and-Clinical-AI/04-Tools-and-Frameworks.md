# AI for Healthcare and Clinical AI — Tools and Frameworks

> The practical landscape: open-source libraries, cloud healthcare AI services, imaging toolkits, clinical LLM platforms, and standards bodies. Not an endorsement — a map.

## 1. Open-Source Foundations

| Tool | Purpose | Notes |
|---|---|---|
| **FHIR** (HL7) | EHR data interchange | Standard, not a tool; use fhirclient / HAPI |
| **DICOM** / pydicom | Imaging read/write | Metadata + pixels |
| **MONAI** | Medical imaging deep learning | PyTorch-based, DICOM-native |
| **MedMNIST** | Benchmark datasets | 2D/3D medical image sets |
| **Microsoft Presidio** | PHI de-identification | PII/PHI analyzer + anonymizer |
| **Synthia / SDV** | Synthetic health data | See [51](../51-Synthetic-Data-Generation/01-Overview.md) |
| **PyHealth / EHRShot** | EHR ML frameworks | Embeddings, benchmarks |
| **MedCAT** | Clinical concept annotation | UMLS/SNOMED linking |
| **ClinicalBERT / Gatito** | Clinical NLP models | Note understanding |
| **scikit-learn / LightGBM** | Tabular risk models | Workhorse for CDS |
| **SHAP / Captum** | Explainability | Evidence for clinicians |
| **Evidently / NannyML** | Drift monitoring | See [56](../56-MLOps-and-AI-Platform-Engineering/01-Overview.md) |

### 1.1 MONAI quick start
```python
from monai.networks.nets import DenseNet121
model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=2)
# train on DICOM-derived tensors; MONAI handles transforms/augmentation
```

### 1.2 Presidio for PHI
```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
analyzer, anonymizer = AnalyzerEngine(), AnonymizerEngine()
results = analyzer.analyze(text=note, language="en",
                           entities=["PERSON","LOCATION","DATE","PHONE_NUMBER"])
anon = anonymizer.anonymize(text=note, analyzer_results=results)
```

## 2. Cloud Healthcare AI Platforms

| Provider | Service | Role |
|---|---|---|
| AWS | HealthLake, Comprehend Medical | FHIR store + PHI NLP |
| Azure | Azure Health Bot, Health Data Services | Bot + FHIR |
| GCP | Healthcare API, Vertex AI | FHIR/Imaging store + ML |
| Google | Med-PaLM / AMIE research | Clinical LLM research |
| OpenAI | GPT-4 class (via API) | General LLM for scribe (not a device) |
| Anthropic | Claude | Long-context note summarization |

> Note: General LLMs are **not medical devices**. Use them as components under your own cleared/validated workflow.

## 3. Imaging & Diagnostics Vendors (cleared devices)

A non-exhaustive set of FDA-cleared imaging AI categories:
- Radiology triage (CT stroke, CXR, mammography)
- Pathology (digital slide analysis)
- Cardiology (echo/ECG)
- Ophthalmology (retinopathy screening)

(Product names change fast; verify current clearances on the FDA 510(k)/De Novo database.)

## 4. Ambient Scribe / Clinical Documentation

Vendors offer EHR-integrated scribes. Engineering pattern (see 03-§2.4):
```
mic → ASR → clinical LLM → structured SOAP → clinician edit → EHR
```
Build vs buy depends on PHI posture, EHR integration depth, and faithfulness SLAs (see [52](../52-AI-Hallucination-Detection-and-Mitigation/01-Overview.md)).

## 5. Agent Frameworks for Clinical Workflows

Reuse general agent stacks ([03-Agents](../03-Agents/01-Overview.md), [20-Observability](../20-Agent-Infrastructure-and-Observability/01-Overview.md)) with healthcare guards:
- **LangGraph / CrewAI / AutoGen** — orchestration
- **MCP** ([48-MCP](../48-MCP-Cloud-Infrastructure-Agent-as-a-Service/01-Overview.md)) — tool connectors to EHR
- **Tracing** — mandatory for audit

```python
# MCP tool exposing read-only EHR
@mcp.tool()
def get_labs(patient_id: str) -> dict:
    return fhir.read(patient_id, resource="Observation")   # read-only
```

## 6. Standards & Regulatory Bodies (bookmark these)

| Body | What |
|---|---|
| **FDA** (CDRH / Digital Health) | SaMD, PCCP, TPLC guidance |
| **EMA** | EU device regulation |
| **HL7** | FHIR, CDA standards |
| **DICOM** | Imaging standard |
| **ISO** | 42001 (AI mgmt), 13485, IEC 62304 |
| **ONC** | US interoperability rules |
| **EU AI Act** | High-risk AI obligations |
| **NIH / NCATS** | Research datasets (MIMIC, etc.) |

## 7. Public Datasets (research, de-identified)

| Dataset | Modality |
|---|---|
| **MIMIC-IV** | ICU EHR (credentialed) |
| **CheXpert / MIMIC-CXR** | Chest X-ray |
| **NIH ChestX-ray14** | CXR |
| **BraTS** | Brain MRI |
| **ISIC** | Skin lesions |
| **PhysioNet** | Multi signals |

> Always check data use agreements and credentialing (MIMIC requires training).

## 8. Evaluation Harnesses

- **Scikit-learn / torchmetrics** — metrics
- **Evidently / NannyML** — drift
- **Subgroup analysis notebooks** — fairness (see [55](../55-AI-Ethics-and-Responsible-AI/01-Overview.md))
- Internal "silent trial" infra (03-§4)

## 9. Build vs Buy Decision Matrix

| Need | Build when | Buy when |
|---|---|---|
| Imaging triage | Unique modality/population | Standard modality, vendor cleared |
| Scribe | Strong PHI/EHR control needs | Want fast ROI, EHR-integrated |
| CDS risk | Novel score, internal data | Standard score exists |
| Prior-auth agent | Have policy KB + dev team | Workflow complexity high |

## 10. Reference Stack (one opinionated combo)

```
Ingest:    FHIR client + pydicom
De-ID:     Presidio
Store:     HealthLake / Healthcare API / Postgres+FHIR
Features:  PyHealth + Feature Store
Train:     LightGBM (tabular) + MONAI (imaging) + clinical LLM (scribe)
Explain:   SHAP
Serve:     FastAPI inference + evidence
Monitor:   Evidently
Agents:    LangGraph + MCP (read-only EHR)
Audit:     Immutable log (blockchain-optional) + tracing
```

## 11. Vendor Evaluation Checklist

- [ ] Cleared/registered where required (FDA/CE)
- [ ] Data residency / PHI controls (HIPAA BAAs)
- [ ] External validation evidence
- [ ] Calibration + subgroup performance published
- [ ] Explainability for clinicians
- [ ] Drift/performance monitoring provided
- [ ] Audit/rollback support
- [ ] Integration with your EHR (FHIR/DICOM)

## 12. Interoperability Pitfalls
- FHIR versions differ across vendors (R4 vs R5). Pin a version.
- DICOM private tags vary; normalize before training.
- Terminology mismatches (local codes vs SNOMED/LOINC) require mapping tables.

## 13. Cost & ROI Tooling
Tie spend tracking to [41](../41-AI-Cost-Optimization-and-Enterprise-ROI/01-Overview.md) and [59](../59-AI-Agent-Financial-Governance-and-Cost-Control/01-Overview.md): measure clinician time saved, avoided readmissions, denied-claim recovery.

## What's Next
- **05-Future-Outlook.md** — where healthcare AI is heading.

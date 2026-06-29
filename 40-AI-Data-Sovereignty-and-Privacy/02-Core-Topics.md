# AI Data Sovereignty — Core Topics

> **Description:** Deep dive into the core technical and policy topics that define AI data sovereignty, including data residency requirements, cross-border transfer mechanisms, privacy-preserving AI techniques, and governance frameworks.

---

## Table of Contents

1. [Data Residency Requirements](#1-data-residency-requirements)
2. [Cross-Border Data Transfer Mechanisms](#2-cross-border-data-transfer-mechanisms)
3. [Privacy-Preserving AI Techniques](#3-privacy-preserving-ai-techniques)
4. [AI Training Data Governance](#4-ai-training-data-governance)
5. [Inference-Time Data Sovereignty](#5-inference-time-data-sovereignty)
6. [Model Governance and Provenance](#6-model-governance-and-provenance)
7. [Cloud Sovereignty Patterns](#7-cloud-sovereignty-patterns)
8. [Compliance Frameworks](#8-compliance-frameworks)
9. [Vendor Management for Sovereign AI](#9-vendor-management-for-sovereign-ai)
10. [Audit and Compliance Verification](#10-audit-and-compliance-verification)

---

## 1. Data Residency Requirements

### What Constitutes "Residency"?

Data residency for AI goes beyond simple storage location. It encompasses every stage of the AI data lifecycle:

```
AI Data Lifecycle Stages:
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│Collection│ → │ Storage  │ → │Processing│ → │ Training │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
                                            ↓
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  archival│ ← │  Backup  │ ← │ Inference│ ← │  Model   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘

Each stage may have different residency requirements.
```

### Residency Requirements by Data Type

| Data Type | Storage Location | Processing Location | Transfer Restrictions |
|-----------|-----------------|--------------------|-----------------------|
| **Personal Data (EU)** | EU/EEA or adequate country | EU/EEA (SCCs required otherwise) | Strict consent + purpose limitation |
| **Health Data (US)** | US (HIPAA compliance) | US (BAA required) | No cross-border without BAA |
| **Financial Data (EU)** | EU (PSD2/PSD3) | EU preferred | Transfer via adequacy/SCCs |
| **Government Data** | National borders | National borders | No cross-border transfer |
| **Classified Data** | Secure facility | Secure facility | Air-gapped only |
| **AI Training Data** | Depends on source jurisdiction | Depends on model jurisdiction | Source + model jurisdiction rules |

### Technical Implementation of Residency

```python
from enum import Enum
from typing import Optional, List

class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    CLASSIFIED = "classified"

class ResidencyRequirement:
    def __init__(self, classification: DataClassification, allowed_regions: List[str],
                 requires_encryption: bool = True, max_cross_border_transfers: int = 0,
                 requires_dpo_approval: bool = False):
        self.classification = classification
        self.allowed_regions = allowed_regions
        self.requires_encryption = requires_encryption
        self.max_cross_border_transfers = max_cross_border_transfers
        self.requires_dpo_approval = requires_dpo_approval

class DataResidencyEnforcer:
    def __init__(self):
        self.requirements = {
            DataClassification.PUBLIC: ResidencyRequirement(
                DataClassification.PUBLIC, allowed_regions=["*"], requires_encryption=False),
            DataClassification.CONFIDENTIAL: ResidencyRequirement(
                DataClassification.CONFIDENTIAL,
                allowed_regions=["eu-west-1", "eu-central-1", "eu-north-1"],
                requires_encryption=True, max_cross_border_transfers=1,
                requires_dpo_approval=True),
            DataClassification.RESTRICTED: ResidencyRequirement(
                DataClassification.RESTRICTED, allowed_regions=["eu-west-1"],
                requires_encryption=True, max_cross_border_transfers=0,
                requires_dpo_approval=True),
            DataClassification.CLASSIFIED: ResidencyRequirement(
                DataClassification.CLASSIFIED, allowed_regions=["on-prem-secure-1"],
                requires_encryption=True, max_cross_border_transfers=0,
                requires_dpo_approval=True),
        }

    def validate_storage(self, data: bytes, classification: DataClassification,
                         target_region: str, transfer_count: int = 0) -> bool:
        req = self.requirements[classification]
        if "*" not in req.allowed_regions and target_region not in req.allowed_regions:
            raise ResidencyViolation(
                f"Cannot store {classification.value} data in {target_region}. "
                f"Allowed regions: {req.allowed_regions}")
        if transfer_count > req.max_cross_border_transfers:
            raise ResidencyViolation(
                f"Data has been transferred {transfer_count} times, exceeding limit of "
                f"{req.max_cross_border_transfers}")
        if req.requires_encryption and not self._is_encrypted(data):
            raise EncryptionRequired(f"Data must be encrypted for {classification.value}")
        return True

class ResidencyViolation(Exception): pass
class EncryptionRequired(Exception): pass
```

---

## 2. Cross-Border Data Transfer Mechanisms

### Transfer Mechanisms Comparison

| Mechanism | Applicability | Protection Level | Complexity | Cost |
|-----------|--------------|-----------------|------------|------|
| **Adequacy Decision** | EU → Adequate countries | High | Low | Low |
| **Standard Contractual Clauses (SCCs)** | EU → Non-adequate | High | Medium | Medium |
| **Binding Corporate Rules (BCRs)** | Intra-group transfers | High | High | High |
| **Consent** | Individual-specific | Low-Medium | Low | Low |
| **EU-US Data Privacy Framework** | EU → US (certified) | High | Low | Low |
| **Code of Conduct** | Sector-specific | Medium | Medium | Low |

### SCCs for AI Workloads

Standard Contractual Clauses require specific adaptations for AI:

```
AI-Specific SCC Addendum:

Module 1: Controller to Controller (C2C)
├── Training data sharing between entities
├── Model weight transfers
└── Fine-tuning dataset exchanges

Module 2: Controller to Processor (C2P)
├── Data sent to cloud AI provider
├── API-based inference requests
└── Managed AI service data flows

Module 3: Processor to Processor (P2P)
├── Data processing sub-processors
├── Multi-cloud AI training
└── Distributed training data flows

Module 4: Processor to Controller (P2C)
├── AI output delivery
├── Model hosting arrangements
└── Managed inference services
```

### Technical Transfer Safeguards

```python
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class TransferRecord:
    data_id: str
    source_region: str
    destination_region: str
    transfer_mechanism: str
    timestamp: datetime
    purpose: str
    data_classification: str
    encryption_applied: bool
    dpo_approved: bool

class CrossBorderTransferManager:
    def __init__(self):
        self.transfer_log: list[TransferRecord] = []
        self.blocked_transfers: list[dict] = []

    def initiate_transfer(self, data_id: str, source: str, destination: str,
                          classification: str, purpose: str) -> dict:
        mechanism = self._determine_mechanism(source, destination, classification)
        if mechanism is None:
            return {"status": "blocked",
                    "reason": f"No valid transfer mechanism for {source} → {destination}"}
        if not self._validate_mechanism_docs(mechanism, source, destination):
            return {"status": "blocked",
                    "reason": f"Transfer mechanism {mechanism} documentation incomplete"}
        record = TransferRecord(
            data_id=data_id, source_region=source, destination_region=destination,
            transfer_mechanism=mechanism, timestamp=datetime.utcnow(), purpose=purpose,
            data_classification=classification, encryption_applied=True, dpo_approved=True)
        self.transfer_log.append(record)
        return {"status": "approved", "mechanism": mechanism,
                "record_id": len(self.transfer_log)}

    def generate_transfer_report(self) -> str:
        report = {"total_transfers": len(self.transfer_log),
                  "by_mechanism": {}, "by_region": {}}
        for record in self.transfer_log[-100:]:
            m = record.transfer_mechanism
            report["by_mechanism"][m] = report["by_mechanism"].get(m, 0) + 1
            key = f"{record.source_region} → {record.destination_region}"
            report["by_region"][key] = report["by_region"].get(key, 0) + 1
        return json.dumps(report, indent=2, default=str)
```

---

## 3. Privacy-Preserving AI Techniques

### Technique Comparison Matrix

| Technique | Privacy Level | AI Capability Impact | Maturity | Compute Overhead |
|-----------|:------------:|:-------------------:|:--------:|:----------------:|
| **Federated Learning** | High | Low-Medium | Production | 2-5x |
| **Differential Privacy** | Very High | Medium-High | Production | 1.5-3x |
| **Homomorphic Encryption** | Maximum | High | Emerging | 10-1000x |
| **Secure Multi-Party Computation** | Very High | Medium | Production | 5-50x |
| **Trusted Execution Environments** | High | Low | Production | 1.2-2x |
| **Synthetic Data Generation** | High | Variable | Production | 2-5x |
| **Data Anonymization** | Medium | Low | Production | 1.1-1.5x |

### Federated Learning for Sovereign AI

```
Federated Learning Architecture:

┌─────────────────────────────────────────────────────┐
│                  Global Coordinator                  │
│              (Neutral jurisdiction)                   │
│                                                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐             │
│  │ Model   │  │Gradient │  │Aggregation│             │
│  │ Server  │←→│Aggregator│←→│Engine    │             │
│  └─────────┘  └─────────┘  └─────────┘             │
└──────────────┬──────────┬──────────┬────────────────┘
               │          │          │
    ┌──────────┘    ┌─────┘    ┌─────┘
    ↓               ↓          ↓
┌─────────┐  ┌─────────┐  ┌─────────┐
│ EU Hub  │  │ US Hub  │  │Asia Hub │
│Local    │  │Local    │  │Local    │
│Data     │  │Data     │  │Data     │
│Local    │  │Local    │  │Local    │
│Training │  │Training │  │Training │
│Gradients│  │Gradients│  │Gradients│
│(No raw  │  │(No raw  │  │(No raw  │
│ data)   │  │ data)   │  │ data)   │
└─────────┘  └─────────┘  └─────────┘
Only model updates cross borders.
Raw data never leaves its jurisdiction.
```

### Differential Privacy Implementation

```python
import numpy as np

class DifferentialPrivacyMechanism:
    """Implements epsilon-differential privacy for AI training data."""

    def __init__(self, epsilon: float, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.total_epsilon_used = 0.0

    def add_laplace_noise(self, value: float, sensitivity: float) -> float:
        if self.total_epsilon_used >= self.epsilon:
            raise PrivacyBudgetExhausted(
                f"Privacy budget exhausted: {self.total_epsilon_used}/{self.epsilon}")
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        self.total_epsilon_used += self.epsilon
        return value + noise

    def clip_gradients(self, gradients: np.ndarray, max_norm: float) -> np.ndarray:
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > max_norm:
            gradients = gradients * (max_norm / grad_norm)
        return gradients

    def private_sgd_step(self, model_params: np.ndarray, gradients: np.ndarray,
                         learning_rate: float, batch_size: int,
                         max_grad_norm: float) -> np.ndarray:
        clipped = self.clip_gradients(gradients, max_grad_norm)
        noise_scale = (max_grad_norm * np.sqrt(2 * np.log(1.25 / self.delta))
                       / (self.epsilon * batch_size))
        noisy_gradients = clipped + np.random.normal(0, noise_scale, size=clipped.shape)
        return model_params - learning_rate * noisy_gradients

class PrivacyBudgetExhausted(Exception): pass
```

### Homomorphic Encryption for AI Inference

```python
class HomomorphicAIInference:
    """Conceptual homomorphic encryption for AI inference.
    In practice, use TenSEAL (Python bindings for Microsoft SEAL)."""

    def __init__(self, security_level: int = 128):
        self.security_level = security_level
        self.poly_modulus_degree = 8192
        self.coeff_modulus_bits = [60, 40, 40, 60]
        self.scale = 2**40

    def encrypt_input(self, plaintext: list[float]) -> "Ciphertext":
        return {"type": "encrypted", "data": "..."}

    def encrypted_linear_layer(self, ciphertext: "Ciphertext",
                               weights: "EncryptedWeights") -> "Ciphertext":
        return {"type": "encrypted", "data": "..."}

    def encrypted_relu(self, ciphertext: "Ciphertext") -> "Ciphertext":
        # ReLU approximated as: 0.5*x + 0.5*sqrt(x^2 + epsilon)
        return {"type": "encrypted", "data": "..."}

    def private_inference(self, input_data: list[float]) -> list[float]:
        encrypted_input = self.encrypt_input(input_data)
        h1 = self.encrypted_linear_layer(encrypted_input, self._weights[0])
        h1 = self.encrypted_relu(h1)
        h2 = self.encrypted_linear_layer(h1, self._weights[1])
        h2 = self.encrypted_relu(h2)
        output = self.encrypted_linear_layer(h2, self._weights[2])
        return self.decrypt_output(output)
```

---

## 4. AI Training Data Governance

### Data Provenance Framework

```
Training Data Provenance Chain:

Source Collection
├── Where: [Country/Region]
├── When: [Date range]
├── Consent: [Legal basis]
├── Purpose: [Original collection purpose]
├── License: [Data license type]
└── Quality: [Data quality assessment]

Data Processing
├── Cleaning: [What was removed/modified]
├── Augmentation: [What was added]
├── Transformation: [How data was transformed]
├── Filtering: [What was filtered out and why]
└── PII Handling: [How PII was detected and handled]

Training Usage
├── Model: [Which model was trained]
├── Epoch: [Training epoch/step]
├── Influence: [Data influence score]
└── Memorization: [Whether model memorized specific data]

Post-Training
├── Audit: [When last audited]
├── Retention: [How long data is retained]
├── Deletion: [Deletion rights handling]
└── Update: [When data was last updated]
```

### Training Data Consent Management

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

class ConsentType(Enum):
    CONSENT = "consent"
    LEGITIMATE_INTEREST = "legitimate_interest"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    PUBLIC_INTEREST = "public_interest"

class AITrainingConsent:
    def __init__(self):
        self.consent_records: dict[str, dict] = {}

    def record_consent(self, data_id: str, data_subject_id: str,
                       consent_type: ConsentType, purpose: str,
                       expires_at: Optional[datetime] = None,
                       jurisdiction: str = "eu",
                       explicit_for_ai: bool = True) -> dict:
        record = {
            "data_id": data_id, "data_subject_id": data_subject_id,
            "consent_type": consent_type.value, "purpose": purpose,
            "explicit_for_ai": explicit_for_ai,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "jurisdiction": jurisdiction,
            "recorded_at": datetime.utcnow().isoformat(),
            "withdrawn": False, "valid": True
        }
        if jurisdiction == "eu" and consent_type == ConsentType.CONSENT:
            if not explicit_for_ai:
                record["valid"] = False
                record["invalid_reason"] = "EU requires explicit consent for AI training"
        self.consent_records[data_id] = record
        return record

    def withdraw_consent(self, data_id: str) -> dict:
        if data_id not in self.consent_records:
            return {"success": False, "reason": "No consent record found"}
        record = self.consent_records[data_id]
        record["withdrawn"] = True
        record["withdrawn_at"] = datetime.utcnow().isoformat()
        record["valid"] = False
        return {
            "success": True,
            "action_required": [
                "Remove data from training set",
                "Trigger model retraining without this data",
                "Verify model no longer contains memorized data",
                "Update data lineage records"
            ]
        }
```

---

## 5. Inference-Time Data Sovereignty

### API Data Flow Control

```
Sovereign Inference Architecture:

┌──────────────┐     ┌──────────────────────────┐
│   User App   │────→│   Sovereign API Gateway   │
│  (EU Region) │     │                            │
└──────────────┘     │  1. Classify request data  │
                     │  2. Strip/mask PII         │
                     │  3. Check destination       │
                     │  4. Route to sovereign      │
                     │     endpoint                │
                     │  5. Audit log               │
                     └──────────┬─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    ↓           ↓           ↓
            ┌──────────┐ ┌──────────┐ ┌──────────┐
            │ EU Model │ │ US Model │ │ On-Prem  │
            │ (Mistral)│ │ (GPT-4o) │ │ (Llama)  │
            │ Sovereign│ │ Non-     │ │ Air-gapped│
            └──────────┘ └──────────┘ └──────────┘

Routing Rules:
├── PII detected → Route to EU model or on-premise
├── No PII + Low sensitivity → Can route to any model
├── Government data → Route to on-premise only
└── Financial data → Route to EU/US regulated model
```

### Request Classification Engine

```python
from dataclasses import dataclass
from enum import Enum
import re

class SensitivityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ClassifiedRequest:
    original_request: str
    sensitivity: SensitivityLevel
    contains_pii: bool
    pii_types: list[str]
    jurisdiction: str
    requires_sovereignty: bool

class RequestClassifier:
    PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    }
    SENSITIVE_KEYWORDS = [
        "patient", "diagnosis", "medical", "health",
        "financial", "account", "balance", "transaction",
        "government", "classified", "confidential",
        "employee", "salary", "performance review"
    ]

    def classify(self, request: str, user_jurisdiction: str = "eu") -> ClassifiedRequest:
        pii_found = [t for t, p in self.PII_PATTERNS.items() if re.search(p, request)]
        sensitive_count = sum(1 for kw in self.SENSITIVE_KEYWORDS if kw in request.lower())
        if sensitive_count >= 3 or len(pii_found) >= 2:
            sensitivity = SensitivityLevel.CRITICAL
        elif sensitive_count >= 2 or len(pii_found) >= 1:
            sensitivity = SensitivityLevel.HIGH
        elif sensitive_count >= 1:
            sensitivity = SensitivityLevel.MEDIUM
        else:
            sensitivity = SensitivityLevel.LOW
        requires_sovereignty = (
            len(pii_found) > 0 or
            sensitivity in [SensitivityLevel.HIGH, SensitivityLevel.CRITICAL] or
            user_jurisdiction in ["eu", "cn", "ru", "in"])
        return ClassifiedRequest(
            original_request=request, sensitivity=sensitivity,
            contains_pii=len(pii_found) > 0, pii_types=pii_found,
            jurisdiction=user_jurisdiction, requires_sovereignty=requires_sovereignty)
```

---

## 6. Model Governance and Provenance

### Model Card for Sovereignty

```yaml
model_name: "enterprise-llm-v2"
version: "2.3.1"
training_data:
  total_tokens: 2_000_000_000_000
  source_jurisdictions:
    - region: "us"
      percentage: 35
      data_types: ["web_crawl", "books", "code"]
    - region: "eu"
      percentage: 25
      data_types: ["web_crawl", "academic", "news"]
    - region: "asia"
      percentage: 20
      data_types: ["web_crawl", "code", "academic"]
training_compute:
  location: "us-west-2"
  provider: "aws"
  gpu_type: "nvidia-h100"
  gpu_count: 256
  training_duration_days: 45
compliance:
  certifications: ["soc2-type2", "iso27001", "gdpr-compliant"]
  audit_date: "2026-03-15"
  data_governance:
    right_to_erasure: "implemented"
    data_minimization: "applied"
    purpose_limitation: "ai_research_and_commercial"
deployment_restrictions:
  allowed_regions: ["eu", "us", "uk", "ca", "au", "jp", "sg"]
  blocked_regions: ["cn", "ru", "ir", "kp"]
  requires_encryption: true
```

---

## 7. Cloud Sovereignty Patterns

### Pattern Comparison

| Pattern | Control Level | Cost | Complexity | Best For |
|---------|:------------:|:----:|:----------:|----------|
| **Global Cloud** | Low | Low | Low | Startups, non-regulated |
| **Regional Cloud** | Medium | Medium | Medium | Regional enterprises |
| **Sovereign Cloud** | High | High | High | Regulated industries |
| **Hybrid Cloud** | High | Medium-High | High | Multi-jurisdiction |
| **On-Premise** | Maximum | Very High | Very High | Government, defense |
| **Air-Gapped** | Complete | Extreme | Extreme | Classified, military |

### Sovereign Cloud Deployment Checklist

```
Sovereign Cloud Deployment Checklist:

Infrastructure
├── [ ] GPU compute in target jurisdiction
├── [ ] Storage in target jurisdiction
├── [ ] Network routing stays within jurisdiction
├── [ ] DNS resolution within jurisdiction
├── [ ] No third-party telemetry leaving jurisdiction
└── [ ] Encrypted backups in jurisdiction

Data Governance
├── [ ] Data classification system implemented
├── [ ] PII detection and handling pipeline
├── [ ] Data retention policies configured
├── [ ] Data deletion capabilities verified
├── [ ] Cross-border transfer logging enabled
└── [ ] Consent management system integrated

Model Governance
├── [ ] Model provenance documented
├── [ ] Training data sources audited
├── [ ] Model card with sovereignty metadata
├── [ ] Fine-tuning data jurisdiction verified
├── [ ] Model update pipeline sovereignty-checked
└── [ ] Inference logging for compliance

Security
├── [ ] Encryption at rest (AES-256)
├── [ ] Encryption in transit (TLS 1.3)
├── [ ] Key management within jurisdiction
├── [ ] Access controls and RBAC
├── [ ] Audit logging enabled
└── [ ] Incident response plan

Compliance
├── [ ] DPO/DPA review completed
├── [ ] DPIA completed
├── [ ] Transfer Impact Assessment completed
├── [ ] SCCs or BCRs in place
├── [ ] Regular audit schedule established
└── [ ] Documentation maintained
```

---

## 8. Compliance Frameworks

### Framework Mapping

| Framework | Jurisdiction | AI-Specific? | Data Sovereignty Focus |
|-----------|-------------|:------------:|:----------------------:|
| **GDPR** | EU | Indirect | High |
| **EU AI Act** | EU | Yes | Medium |
| **CCPA/CPRA** | California | Indirect | Medium |
| **DPDP Act** | India | Indirect | High |
| **PIPL** | China | Indirect | Very High |
| **NIST AI RMF** | US | Yes | Low |
| **ISO/IEC 42001** | Global | Yes | Medium |

### Compliance Automation

```python
class ComplianceChecker:
    def __init__(self):
        self.checks = {
            "gdpr": self._check_gdpr,
            "ai_act": self._check_ai_act,
            "ccpa": self._check_ccpa,
            "dpdp": self._check_dpdp,
            "iso42001": self._check_iso42001,
        }

    def run_compliance_check(self, ai_system: dict, jurisdiction: str) -> dict:
        results = {}
        for framework, check_fn in self.checks.items():
            if self._applies_to(framework, jurisdiction):
                results[framework] = check_fn(ai_system)
        overall = all(r.get("compliant", False) for r in results.values())
        return {"overall_compliant": overall, "details": results,
                "recommendations": self._generate_recommendations(results)}

    def _check_gdpr(self, ai_system: dict) -> dict:
        issues = []
        if not ai_system.get("legal_basis"):
            issues.append("No legal basis documented for data processing")
        if not ai_system.get("data_minimized"):
            issues.append("Data minimization principle not verified")
        if not ai_system.get("erasure_capability"):
            issues.append("Right to erasure not implemented for AI outputs")
        return {"compliant": len(issues) == 0, "issues": issues, "framework": "GDPR"}

    def _check_ai_act(self, ai_system: dict) -> dict:
        issues = []
        risk_level = ai_system.get("risk_level", "unknown")
        if risk_level == "unknown":
            issues.append("AI system risk level not classified under EU AI Act")
        if risk_level == "high":
            if not ai_system.get("data_governance"):
                issues.append("High-risk AI system lacks data governance documentation")
            if not ai_system.get("bias_testing"):
                issues.append("High-risk AI system lacks bias testing")
        return {"compliant": len(issues) == 0, "issues": issues, "framework": "EU AI Act"}

    def _check_ccpa(self, ai_system: dict) -> dict:
        issues = []
        if not ai_system.get("opt_out_mechanism"):
            issues.append("No opt-out mechanism for AI data sales/sharing")
        return {"compliant": len(issues) == 0, "issues": issues, "framework": "CCPA"}

    def _check_dpdp(self, ai_system: dict) -> dict:
        issues = []
        if ai_system.get("requires_localization") and not ai_system.get("data_localized"):
            issues.append("Data localization requirement not met")
        return {"compliant": len(issues) == 0, "issues": issues, "framework": "DPDP"}

    def _check_iso42001(self, ai_system: dict) -> dict:
        issues = []
        if not ai_system.get("ai_management_system"):
            issues.append("AI Management System (AIMS) not established")
        return {"compliant": len(issues) == 0, "issues": issues, "framework": "ISO 42001"}

    def _applies_to(self, framework: str, jurisdiction: str) -> bool:
        mapping = {"gdpr": ["eu", "eea", "uk"], "ai_act": ["eu"],
                   "ccpa": ["us-ca", "us"], "dpdp": ["in"], "iso42001": ["*"]}
        return "*" in mapping.get(framework, []) or jurisdiction in mapping.get(framework, [])

    def _generate_recommendations(self, results: dict) -> list[str]:
        recs = []
        for fw, r in results.items():
            for issue in r.get("issues", []):
                recs.append(f"[{fw}] {issue}")
        return recs
```

---

## 9. Vendor Management for Sovereign AI

### Vendor Evaluation Matrix

| Criteria | Weight | Scoring Guide |
|----------|:------:|---------------|
| **Data residency options** | 25% | 5=multi-region, 1=single region only |
| **Compliance certifications** | 20% | 5=ISO 27001+SOC2+GDPR, 1=none |
| **On-premise options** | 15% | 5=full on-prem, 1=cloud only |
| **Contractual sovereignty guarantees** | 15% | 5=legally binding SLA, 1=none |
| **Transparency (training data)** | 10% | 5=full disclosure, 1=opaque |
| **Exit strategy** | 10% | 5=data portability guaranteed, 1=lock-in |
| **Support jurisdiction** | 5% | 5=in-region support, 1=offshore only |

### Sovereignty SLA Template

```
SOVEREIGN AI SERVICE LEVEL AGREEMENT

1. DATA RESIDENCE
   1.1 Provider guarantees all Customer Data will be stored within [REGION]
   1.2 No Customer Data will be transferred outside [REGION] without consent
   1.3 Provider will notify Customer within 24 hours of any residency breach

2. PROCESSING LOCATION
   2.1 All AI inference for Customer will be performed within [REGION]
   2.2 Training data (if applicable) will remain within [REGION]
   2.3 No Customer data will be used for model training without consent

3. COMPLIANCE
   3.1 Provider maintains [CERTIFICATIONS] at all times
   3.2 Provider will cooperate with Customer's compliance audits
   3.3 Provider will provide quarterly compliance reports

4. SECURITY
   4.1 All data encrypted at rest with AES-256
   4.2 All data encrypted in transit with TLS 1.3
   4.3 Encryption keys managed within [REGION]

5. TERMINATION
   5.1 Upon termination, all Customer Data deleted within 30 days
   5.2 Deletion certified in writing
   5.3 Customer data export available in standard formats
```

---

## 10. Audit and Compliance Verification

### AI Data Sovereignty Audit Checklist

```
AUDIT SCOPE: AI Data Sovereignty

Phase 1: Data Inventory
├── [ ] All AI training data sources identified
├── [ ] Data jurisdiction of origin documented
├── [ ] Legal basis for each data source verified
├── [ ] Data retention policies documented
└── [ ] Data deletion capabilities tested

Phase 2: Infrastructure Audit
├── [ ] All compute locations verified
├── [ ] All storage locations verified
├── [ ] Network traffic analysis for cross-border flows
├── [ ] Backup locations verified
└── [ ] Third-party sub-processor locations verified

Phase 3: Model Audit
├── [ ] Model training data provenance documented
├── [ ] Model card with sovereignty metadata verified
├── [ ] Fine-tuning data sources audited
├── [ ] Model memorization risk assessed
└── [ ] Model deployment restrictions validated

Phase 4: Inference Audit
├── [ ] API request routing verified
├── [ ] Cross-border transfer logging reviewed
├── [ ] PII detection accuracy tested
├── [ ] Sovereign endpoint availability confirmed
└── [ ] Inference logging completeness verified

Phase 5: Governance Audit
├── [ ] Consent management system reviewed
├── [ ] DPO/DPA consultation records reviewed
├── [ ] Transfer Impact Assessments reviewed
├── [ ] Vendor SLAs reviewed
└── [ ] Incident response plan tested
```

### Automated Audit Pipeline

```python
from datetime import datetime
from typing import Dict, Any

class SovereigntyAuditPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.results: Dict[str, Any] = {}

    def run_full_audit(self) -> dict:
        phases = [
            ("data_inventory", self._audit_data_inventory),
            ("infrastructure", self._audit_infrastructure),
            ("model_provenance", self._audit_model_provenance),
            ("inference_routing", self._audit_inference_routing),
            ("governance", self._audit_governance),
        ]
        for name, fn in phases:
            try:
                self.results[name] = fn()
            except Exception as e:
                self.results[name] = {"status": "error", "error": str(e)}
        return self._generate_report()

    def _audit_data_inventory(self) -> dict:
        return {"status": "completed", "findings": [], "checks": {
            "training_data_sources_documented": True,
            "data_jurisdiction_mapped": True,
            "legal_basis_recorded": True,
            "retention_policies_defined": True,
            "deletion_capabilities_verified": True}}

    def _audit_infrastructure(self) -> dict:
        return {"status": "completed", "findings": [], "checks": {}}

    def _audit_model_provenance(self) -> dict:
        return {"status": "completed", "findings": [], "checks": {}}

    def _audit_inference_routing(self) -> dict:
        return {"status": "completed", "findings": [], "checks": {}}

    def _audit_governance(self) -> dict:
        return {"status": "completed", "findings": [], "checks": {}}

    def _generate_report(self) -> dict:
        total, passed = 0, 0
        for r in self.results.values():
            checks = r.get("checks", {})
            total += len(checks)
            passed += sum(1 for v in checks.values() if v)
        return {"audit_date": datetime.utcnow().isoformat(),
                "total_checks": total, "passed_checks": passed,
                "compliance_rate": passed / max(total, 1),
                "overall_status": "COMPLIANT" if passed == total else "NON_COMPLIANT",
                "phase_results": self.results}
```

---

## Summary

AI data sovereignty requires coordinated technical, legal, and organizational responses. The core topics covered here — data residency, cross-border transfers, privacy-preserving AI, compliance frameworks, vendor management, and audit processes — form the foundation for any sovereign AI strategy.

Key takeaways:
1. **Data sovereignty is not optional** — regulations are tightening globally
2. **Technical solutions exist** — federated learning, differential privacy, sovereign clouds
3. **Costs are real but manageable** — sovereignty premiums range from 1.5x to 10x
4. **Vendor management is critical** — SLAs must explicitly address sovereignty
5. **Automation is essential** — manual compliance doesn't scale

---

*This document is part of the [AiBaseKnowledge](../README.md) library. See [01-Overview.md](01-Overview.md) for the big picture, [03-Technical-Deep-Dive.md](03-Technical-Deep-Dive.md) for implementation details, [04-Tools-and-Frameworks.md](04-Tools-and-Frameworks.md) for the technology landscape, and [05-Future-Outlook.md](05-Future-Outlook.md) for predictions.*

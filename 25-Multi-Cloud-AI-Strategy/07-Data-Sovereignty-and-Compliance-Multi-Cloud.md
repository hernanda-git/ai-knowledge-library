# Data Sovereignty & Compliance in Multi-Cloud AI

## Overview

Data sovereignty — the principle that data is subject to the laws of the country where it's collected — is one of the most complex challenges in multi-cloud AI strategy. Each major cloud provider operates data centers globally, but AI workloads introduce unique complications: training data residency, inference data processing, model weight storage, and cross-border data flows for multi-model pipelines.

### The Compliance Challenge

| Cloud Provider | Regions | Data Residency Controls | AI-Specific Compliance |
|---------------|---------|------------------------|----------------------|
| AWS | 30+ | AWS Control Tower, S3 Object Lock, Outposts | Bedrock data processing opt-out |
| Azure | 60+ | Azure Policy, Data Residency, Sovereign Cloud | OpenAI data processing options |
| GCP | 40+ | VPC Service Controls, Data Classifier | Vertex AI data governance |

## Key Regulations

### GDPR (EU)
Applies to any AI processing personal data of EU residents. Key requirements:
- Data minimization in AI training
- Right to explanation for automated decisions (Art. 22)
- Data Protection Impact Assessment (DPIA) for AI systems
- Cross-border transfer restrictions (SCCs, BCRs)
- 4% global turnover penalties

### HIPAA (US Healthcare)
- AI processing PHI requires Business Associate Agreements (BAAs)
- All three clouds offer HIPAA-eligible services
- Encryption at rest and in transit mandatory
- Audit logging required for all AI model access

### EU AI Act
- High-risk AI systems require conformity assessment
- Training data governance requirements
- Human oversight mandates
- Up to 7% global turnover penalties
- Cloud providers as "deployers" in certain configurations

## Architecture for Compliant Multi-Cloud AI

```
Region A (EU)                    Region B (US)                    Region C (APAC)
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Training Data    │    │ Inference Only   │    │ Inference Only   │
│ Model Training   │───→│ Model Deployment │───→│ Model Deployment │
│ (stays in EU)    │    │ (no training)    │    │ (no training)    │
└──────────────────┘    └──────────────────┘    └──────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │  Central Governance   │
                    │  - Policy enforcement  │
                    │  - Audit logging      │
                    │  - Data lineage       │
                    └───────────────────────┘
```

**Key principle:** Training data and model training happen within the data's home region. Inference can be distributed globally but must not use training data across borders.

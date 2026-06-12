# AI Governance, Regulation, and Policy
## Table of Contents
1. [Introduction](#1-introduction)
2. [EU AI Act](#2-eu)
3. [US AI Policy](#3-us)
4. [China AI Regulation](#4-china)
4a. [Asia-Pacific AI Governance](#4a-asia-pacific-ai-governance)
5. [International Frameworks](#5-international)
6. [Industry Self-Regulation](#6-self)
7. [Corporate Governance](#7-corporate)
8. [Compliance Implementation](#8-compliance-implementation)
9. [Emerging Governance Technology](#9-emerging-governance-technology)
10. [Enforcement Actions Across Jurisdictions](#10-enforcement-actions)
11. [Emerging Issues](#11-emerging)
12. [Cross-References](#12-cross-references)
---

## 1. Introduction
AI governance encompasses the laws, regulations, standards, and best practices that govern AI development and deployment. As AI systems become more powerful and pervasive, governance frameworks are rapidly evolving to address risks around safety, bias, privacy, accountability, and societal impact.

### Key Tensions
- **Innovation vs Safety:** How fast can we safely progress?
- **Centralized vs Decentralized:** Government oversight vs industry self-regulation
- **National vs International:** Fragmented vs harmonized approaches
- **Open vs Closed:** Open-source vs proprietary AI governance

### Global Landscape Overview
The global AI governance landscape is fragmented across three major regulatory blocs — the EU (risk-based, rights-protective), the US (innovation-promoting, sectoral), and China (state-aligned, content-focused) — alongside emerging multilateral coordination through the UN, G7, and G20. As of mid-2026, over 60 countries have introduced or enacted some form of AI-specific regulation, with the pace of legislative activity accelerating sharply since 2023.

---

## 2. EU AI Act
### 2.1 Risk Categories (Effective 2025-2027)
| Level | Examples | Requirements | Fine |
|-------|----------|-------------|:----:|
| **Unacceptable** | Social scoring, real-time biometric surveillance, manipulative AI | Prohibited | Up to 7% of global turnover |
| **High-risk** | CV screening, credit scoring, medical devices, critical infrastructure | Risk management, data governance, transparency, human oversight, accuracy, robustness | Up to 3% |
| **Limited risk** | Chatbots, emotion recognition, biometric categorization | Transparency obligation (notify users they interact with AI) | — |
| **Minimal risk** | AI-enabled video games, spam filters | No obligations | — |

### 2.2 GPAI Requirements (General Purpose AI)
- **Transparency:** Document training data, model capabilities, limitations
- **Copyright:** Respect EU copyright law, document training data sources
- **Systemic risk:** For models trained with >10²⁵ FLOPs (most frontier models)
  - Model evaluations (standardized testing)
  - Adversarial testing
  - Incident reporting
  - Cybersecurity measures

### 2.3 Implementation Timeline & Delegated Acts
The EU AI Act enters force in stages:
- **Feb 2025:** Unacceptable-risk prohibitions take effect.
- **Aug 2025:** GPAI transparency and copyright obligations apply.
- **Aug 2026:** High-risk system rules for Annex III use cases (AI systems embedded in regulated products) apply.
- **Aug 2027:** Full application for all high-risk systems.

The European Commission has published multiple **delegated and implementing acts** detailing:
- **Conformity assessment procedures** (self-assessment for most high-risk; third-party notified body for biometrics)
- **Standardization requests** to CEN/CENELEC (harmonized standards for risk management, data quality, and robustness)
- **Code of Practice for GPAI models** (developed with the AI Office, involving multi-stakeholder drafting rounds)
- **AI regulatory sandboxes** – member states must establish at least one sandbox by Aug 2026 to foster innovation before full compliance

### 2.4 National Implementation
Each EU member state must designate a **market surveillance authority** and establish **notified bodies** for conformity assessment. Competent authorities include:
- **Germany:** Federal Network Agency (BNetzA) – digital oversight
- **France:** DGCCRF & CNIL – consumer + data protection
- **Italy:** AGCOM – communications and digital market regulator

The **European AI Office** (established Feb 2024 within the European Commission) serves as the central coordinator for GPAI oversight, cross-border enforcement, and the Code of Practice.

---

## 3. US AI Policy
### 3.1 Executive Order 14110 (Oct 2023)
- **Safety:** Leading developers must share safety test results
- **Standards:** NIST AI Risk Management Framework
- **Privacy:** Support privacy-preserving technologies
- **Equity:** Guidance on algorithmic discrimination
- **Workers:** Report on AI labor impacts
- **Innovation:** National AI Research Resource

### 3.2 Congressional Activity (2024–2026)
Multiple AI bills have been introduced in the 118th and 119th Congresses, reflecting growing bipartisan concern:
| Bill | Key Provisions | Status |
|------|----------------|--------|
| **AI Foundation Model Transparency Act** | Mandates documentation and disclosure for foundation models | Committee markup |
| **Algorithmic Accountability Act** | Requires impact assessments for automated decision systems | Reintroduced (2025) |
| **CREATE AI Act** | National AI Research Resource expansion | Passed House (2025) |
| **AI Safety Institute (AISI) Authorization** | Codifies AISI as permanent federal body within NIST | Pending floor vote |
| **Federal AI Governance and Transparency Act** | Government-wide AI procurement and use standards | Introduced Jan 2026 |

### 3.3 State-Level Regulation
| State | Key Legislation | Focus |
|-------|-----------------|-------|
| **Colorado** | Colorado AI Act (2025) | Consumer protection, algorithmic discrimination |
| **California** | Multiple bills | AI safety, transparency, employment |
| **New York** | NYC Local Law 144 | AI hiring tools audit requirement |
| **Illinois** | Artificial Intelligence Video Interview Act | Notice + consent for AI interviews |
| **Texas** | Texas AI Transparency Act (2025) | Disclosure of AI interactions in government |

The **Colorado AI Act** is particularly notable as the first comprehensive state AI law in the US. It requires developers and deployers of high-risk AI systems to implement risk management policies, conduct impact assessments, and disclose algorithmic discrimination risks to consumers. It serves as a potential template for other states and the federal government.

### 3.4 Federal Agency Actions
| Agency | Action | Focus |
|--------|--------|-------|
| **FTC** | Enforcement actions under Section 5 | Deceptive AI claims, algorithmic bias |
| **EEOC** | Technical assistance on AI hiring | Disparate impact enforcement |
| **DOJ** | Civil rights guidance | AI in housing, credit, criminal justice |
| **CISA** | AI security guidelines | Critical infrastructure AI risks |
| **FHFA** | Proposed rule on AI in lending | Fair lending compliance |

---

## 4. China AI Regulation
### 4.1 Key Regulations
| Regulation | Effective | Requirements |
|------------|:---------:|--------------|
| Deep Synthesis Provisions | 2023 | Label AI-generated content, user consent, security assessment |
| Algorithmic Recommendation Provisions | 2022 | Transparency, non-discrimination, user control |
| Generative AI Measures | 2023 | Content safety, data protection, algorithm registration, training data compliance |
| GAI Implementation Rules | 2025 | Expanded obligations, risk classification |

### 4.2 Regulatory Philosophy & Enforcement
China's approach centers on **state control over AI content and alignment with socialist core values**. Key enforcement mechanisms include:
- **Algorithm registration:** All recommendation and generative AI algorithms must be registered with the Cyberspace Administration of China (CAC); non-compliance can result in service suspension.
- **Content safety reviews:** Training datasets and model outputs are subject to mandatory security assessments.
- **Cross-border data restrictions:** AI models that process personal data must comply with China's Personal Information Protection Law (PIPL) and Data Security Law (DSL), including data localization requirements.
- **Licensing regime:** Generative AI services must obtain a license; foreign models (e.g., ChatGPT) remain blocked behind the Great Firewall.

### 4.3 Emerging Developments (2025–2026)
- **AI Liability Rules:** Draft regulations clarifying civil liability for AI-caused harms, including strict liability for certain high-risk applications.
- **Synthetic Data Governance:** New CAC guidelines on using synthetically generated training data, requiring transparency and quality controls.
- **Compute Governance:** Proposed rules requiring approval for large-scale AI training clusters (>100 PFLOPS), mirroring export-control concerns.

---

## 4a. Asia-Pacific AI Governance

Beyond China, economies across the Asia-Pacific region are developing distinctive AI governance approaches that balance innovation, safety, and national interests. These frameworks will collectively shape the AI regulatory landscape for over half the world's population.

### 4a.1 Japan: Innovation-First, Risk-Informed

Japan's AI governance approach emphasizes harnessing AI for economic revitalization ("Society 5.0") while developing light-touch safety guardrails. Japan co-chaired the Hiroshima AI Process during its 2023 G7 presidency and has been a leading voice in international AI governance.

| Element | Details |
|---------|---------|
| **Primary framework** | AI Strategy Council guidelines (2023, updated 2025) |
| **Regulatory style** | Non-binding guidelines + industry co-regulation |
| **Key legislation** | AI Act pending (Diet, 2026–2027) — likely EU-harmonized for high-risk, flexible for innovation |
| **AI Safety Institute** | AI Safety Institute of Japan (AISI Japan) — established 2024, part of International AISI Network |
| **Focus areas** | Generative AI content labeling, IP/copyright for training data, labor market adaptation |
| **Notable stance** | Pro-innovation copyright: training on copyrighted works permitted unless "unreasonably prejudicial" |
| **International leadership** | G7 Hiroshima AI Process, International Network of AISIs, OECD AI expert group |

**Japan's copyright position is distinctive:** Unlike the EU (which requires opt-out for text/data mining) and the US (fair use doctrine), Japan explicitly permits AI training on copyrighted works under Article 30-4 of its Copyright Law, provided the use does not "unreasonably prejudice" the copyright holder. This has made Japan an attractive jurisdiction for AI training data collection and has informed the broader G7 debate on AI and copyright.

### 4a.2 South Korea: Ambitious Infrastructure + Proactive Regulation

South Korea is pursuing one of the world's most ambitious national AI strategies, positioning itself as a leader in AI semiconductor manufacturing and AI-powered public services.

| Element | Details |
|---------|---------|
| **Primary framework** | Digital Bill of Rights (2023) + AI Act framework (under development) |
| **Regulatory style** | Proactive, comprehensive — similar to EU but with national security overlay |
| **Key legislation** | AI Act proposal (National Assembly, 2025) — risk-based classification, mandatory impact assessments for high-risk AI |
| **AI Safety Institute** | AI Safety Institute Korea (AISIK) — launched 2024 |
| **Infrastructure investments** | $7B+ in AI semiconductor cluster (Yongin), 10,000+ GPU national AI computing center |
| **Focus areas** | AI chips (memory + logic), AI-powered public services, digital rights |
| **Enforcement** | Personal Information Protection Commission (PIPC) — active enforcement against AI data privacy violations |

South Korea's approach is notable for its **dual-track strategy**: heavy investment in AI hardware and infrastructure, combined with proactive regulatory frameworks that anticipate harms rather than react to them. The country's AI Act proposal explicitly requires **pre-market conformity assessments** for high-risk AI systems, with potential fines of up to 3% of annual revenue.

### 4a.3 Singapore: Sectoral, Pro-Business, International Bridge

Singapore has positioned itself as a global hub for responsible AI through a pragmatic, sectoral approach that emphasizes voluntary standards, business support, and international interoperability.

| Element | Details |
|---------|---------|
| **Primary framework** | National AI Strategy 2.0 (2023) + Model AI Governance Framework (2nd ed, 2024) |
| **Regulatory style** | Sectoral, principles-based, voluntary → mandatory (evolving) |
| **Key legislation** | Amended PDPA (2025) includes AI-specific data provisions; sector-specific AI rules for finance (MAS), healthcare (MOH) |
| **AI Governance body** | Digital Trust Centre (DTC) + Info-comm Media Development Authority (IMDA) |
| **Verification** | AI Verify — world's first AI governance testing framework and toolkit (open-source, 2024) |
| **Sandbox** | AI Verify Foundation sandbox for pre-deployment testing |
| **International role** | ASEAN AI governance leadership, OECD AI Group observer, US-Singapore AI partnership |

**AI Verify** (developed by IMDA and the AI Verify Foundation) is Singapore's flagship governance tool — an open-source testing framework that validates AI systems against a set of governance principles (transparency, explainability, robustness, accountability). It is designed as a **voluntary assessment** rather than mandatory certification, reflecting Singapore's pro-business approach. Companies that pass AI Verify receive a mark of responsible AI practice.

### 4a.4 India: Digital Public Infrastructure + AI Sovereignty

India's AI governance approach is shaped by its scale (1.4B+ population, deep digital penetration via Aadhaar/UPI), its focus on AI for social good, and strong emphasis on digital sovereignty and data localization.

| Element | Details |
|---------|---------|
| **Primary framework** | INDIAai (National AI Portal) + Responsible AI reports (2024–2025) |
| **Regulatory style** | Light-touch, innovation-enabling, with emerging guardrails |
| **Key legislation** | Digital Personal Data Protection Act (DPDPA, 2023) — informs AI data governance; AI-specific regulation at consultation stage |
| **AI governance body** | IndiaAI mission (Ministry of Electronics & IT) — launched 2024 with $1.2B budget |
| **Key initiatives** | National AI compute infrastructure (10,000+ GPU cluster), AI datasets platform, AI-skilling programs |
| **Focus areas** | AI in agriculture, healthcare, education, language translation (BharatGPT), public service delivery |
| **International stance** | Advocates for Global South AI access, opposes regulatory frameworks that entrench developed-nation advantages |

India's position in international AI governance debates is distinctive: it has argued that **EU-style comprehensive AI regulation would be premature and potentially harmful** for developing economies, and that the priority should be expanding AI access and capability rather than restricting it. India has proposed a **Global Digital Compact** framework for AI that specifically addresses Global South concerns about compute access, data sovereignty, and capacity building.

### 4a.5 Other Significant Asia-Pacific Jurisdictions

| Country / Region | Approach | Key Development | Distinguishing Feature |
|------------------|----------|-----------------|----------------------|
| **Australia** | Moderate, sectoral | AI Ethics Framework (2024); mandatory guardrails for high-risk govt AI (2025) | CSIRO-led AI Risk Framework; mandatory for government AI procurement |
| **Taiwan** | Innovation-friendly, security-conscious | Generative AI Guidelines (2024); Human Rights AI Basic Law (draft) | Strong focus on AI-generated disinformation from state actors |
| **Hong Kong** | Laissez-faire with emerging guidance | Ethical AI Framework (2024, OGCIO) | Minimal AI-specific regulation; general data protection (PDPO) applies |
| **Vietnam** | State-guided digital transformation | National AI Strategy (2023); data law amendments | AI considered critical for manufacturing competitiveness |
| **Indonesia** | Digital sovereignty focus | National AI Strategy (2023); Personal Data Protection Law (PDP Law, 2024) | Emphasis on local AI capacity and data localization |
| **ASEAN (Regional)** | Consensus-based, non-binding | ASEAN AI Governance and Ethics Guide (2024) | First regional AI framework for Southeast Asia; principles-based |

### 4a.6 Asia-Pacific Regulatory Divergence: Key Dimensions

| Dimension | Japan | South Korea | Singapore | India | China |
|-----------|:-----:|:----------:|:---------:|:-----:|:-----:|
| **Regulatory density** | Low | Medium-High | Low-Medium | Low | High |
| **Enforcement rigor** | Low-Medium | Medium-High | Medium | Low-Medium | High |
| **Copyright stance on training data** | Permissive | Similar to EU | Fair use analysis | Ambiguous | State-controlled |
| **Data localization** | None | Sectoral | None | Strong (DPDPA) | Comprehensive |
| **AI safety evaluation** | Voluntary AISI | Mandatory for high-risk | Voluntary (AI Verify) | No formal framework | Mandatory (CAC) |
| **Open-weight model stance** | Permissive | Permissive | Permissive | Permissive | Restricted |
| **Preference for international alignment** | Strong (G7) | Strong (G7 + OECD) | Very Strong (ASEAN + OECD) | Selective (Global South) | Weak (BRI + BRICS) |

**Implication for global governance:** The Asia-Pacific regulatory landscape is far from unified. This diversity creates both challenges (compliance complexity for global AI companies) and opportunities (regulatory sandbox arbitrage, multi-jurisdictional learning). The region's weight in global AI production (South Korea: chips, Taiwan: semiconductors, Japan: robotics, Singapore: AI services, India: AI talent) means its regulatory choices will significantly influence global AI governance norms.

---

## 5. International Frameworks
| Initiative | Scope | Key Output |
|------------|-------|------------|
| **Bletchley Declaration** (Nov 2023) | 29 countries → 60+ signatories (2025) | International cooperation on frontier AI testing |
| **Hiroshima AI Process** (G7, 2023–2025) | G7 nations + outreach | Guiding principles + Code of Conduct for advanced AI |
| **OECD AI Principles** | 38+ countries | Interoperable policy standards |
| **UNESCO AI Ethics** | 193 member states | Global ethics framework |
| **Council of Europe AI Treaty** | 46 countries | Legally binding AI treaty (opened for signature Sep 2025) |

### 5.1 Bletchley Declaration – Follow-Through
The Bletchley Declaration, originally signed by 29 countries at the UK AI Safety Summit (Nov 2023), has expanded to **over 60 signatories** including China, the US, the EU, and the UAE. Key follow-through actions:

- **AI Safety Institutes (AISIs):** The UK, US, Japan, Singapore, and Canada have established dedicated AISIs. The **International Network of AI Safety Institutes** (coordinated from Seoul, May 2024) facilitates joint testing and information sharing.
- **Seoul Summit (May 2024):** Produced the **Seoul Declaration** — commitments to publish safety frameworks, establish red-teaming protocols, and define capability thresholds for frontier models.
- **Bletchley+ Process:** Ongoing expert working groups on:
  - **Frontier AI evaluations** (standardized benchmarks for hazardous capabilities)
  - **Biological capability assessments** (AI-enabled biosecurity risks)
  - **Cyber offense/defense evaluation** (AI in cybersecurity)
  - **Sovereign AI oversight** (national approaches to regulating frontier models)
- **Next Summit:** France to host the 2025 AI Action Summit in Paris, expected to advance actionable governance commitments.

### 5.2 Hiroshima AI Process – Follow-Through
Launched under Japan's G7 presidency (2023) and continued through Italy's presidency (2024), the Hiroshima AI Process has produced:

| Output | Description | Status |
|--------|-------------|--------|
| **International Guiding Principles for Advanced AI** | 11 principles: safety, transparency, accountability, human oversight, bias mitigation | Endorsed G7 leaders Oct 2023 |
| **Hiroshima Code of Conduct** | Voluntary standards for frontier AI developers (model cards, red-teaming, incident reporting, watermarking) | Updated Apr 2025 |
| **Hiroshima AI Process Friends Group** | Expanded outreach to non-G7 countries (India, Kenya, South Korea, Australia, Singapore) | Active since 2024 |
| **Joint G7 Action Plan on AI Governance** | Mapping of regulatory approaches, interoperability standards, and joint procurement | Interim report Dec 2025 |
| **Hiroshima AI Report** | Annual assessment of AI governance progress across G7 members | First edition Jan 2026 |

The Hiroshima AI Process is notable for balancing **innovation promotion** with **risk mitigation**, and for establishing a model of **multi-stakeholder participation** (industry, civil society, academia) in governance design. Its Code of Conduct is the closest international analogue to the EU AI Act's GPAI obligations.

### 5.3 Other Multilateral Developments
- **UN High-Level Advisory Body on AI (HLAB):** Released "Governing AI for Humanity" report (Sep 2024), proposing a UN AI governance body and a global AI fund.
- **G20 AI Principles (2025):** India's presidency advanced principles for AI in the Global South, emphasizing digital public infrastructure and equitable access.
- **Global Partnership on AI (GPAI):** Now 29 member countries, focusing on responsible AI development, data governance, and AI for social good.

---

## 6. Industry Self-Regulation
| Company | Policy | Key Elements |
|---------|--------|--------------|
| **OpenAI** | Preparedness Framework | Capability thresholds, safety board |
| **Anthropic** | Responsible Scaling Policy | ASL levels, trigger capabilities |
| **Google DeepMind** | Frontier Safety Framework | Capability red lines, containment |
| **Meta** | Responsible AI | Open release with mitigation |
| **Microsoft** | Responsible AI Standard | Human-centered design, fairness, accountability |
| **xAI** | AI Safety Pledge (2025) | Third-party audits, pre-release evaluation sharing |
| **Mistral AI** | GPAI Compliance Playbook | EU AI Act alignment, transparency reporting |

### 6.1 Frontier Model Commitments
At the 2024 Seoul Summit, 16 leading AI companies signed the **Frontier AI Safety Commitments**, agreeing to:
1. Publish safety frameworks before model release
2. Conduct red-teaming and independent evaluations
3. Implement capability thresholds triggering additional safeguards
4. Enable third-party access for pre-deployment testing
5. Report incidents and near-misses to relevant authorities

Signatories include OpenAI, Anthropic, Google DeepMind, Meta, Microsoft, Amazon, Inflection AI, xAI, and Cohere. Compliance is monitored through the **International Network of AI Safety Institutes**.

### 6.2 Voluntary Standards & Industry Coalitions
| Initiative | Focus | Participants |
|------------|-------|--------------|
| **Frontier Model Forum** | Best practices sharing, research funding | Google, OpenAI, Microsoft, Anthropic |
| **Coalition for Secure AI (CoSAI)** | AI-specific cybersecurity standards | OASIS, AWS, Google, NVIDIA, IBM |
| **Partnership on AI** | Multi-stakeholder research | Industry, civil society, academic |
| **MLCommons AI Safety Working Group** | Safety benchmark development | Cross-industry, 50+ organizations |
| **Bletchley AI Safety Commitments** | Frontier model pre-deployment testing | 16 leading labs |

---

## 7. Corporate Governance
### 7.1 AI Governance Functions
- **Board of Directors:** AI oversight, risk appetite
- **AI Ethics Committee:** Policy review, incident response
- **Chief AI Ethics Officer:** Strategy, compliance
- **ML/AI Guild:** Cross-functional, capability building
- **Audit Function:** Independent review of AI systems

### 7.2 AI Risk Management
NIST AI RMF Functions:
1. **GOVERN:** Culture, training, risk appetite
2. **MAP:** Context, risk identification
3. **MEASURE:** Testing, metrics
4. **MANAGE:** Treatment, response

### 7.3 AI Governance Documentation
Organizations developing or deploying AI should maintain a **governance artifact stack** that maps to regulatory obligations:

| Document | Purpose | Applicable Regulation |
|----------|---------|-----------------------|
| **AI Risk & Impact Assessment** | Identify and evaluate risks of each AI system | EU AI Act, Colorado AI Act |
| **Algorithmic Impact Assessment** | Document discrimination and fairness testing | NYC Law 144, Algorithmic Accountability Act |
| **Model Card** | Technical documentation of model capabilities, limitations, and testing | GPAI transparency, NIST AI RMF |
| **System Card** | End-to-end system description including training data, architecture, deployment context | EU AI Act Annex IV |
| **Incident Response Plan** | Procedures for reporting and mitigating AI safety incidents | GPAI systemic risk requirements |
| **Data Governance Statement** | Training data provenance, consent, and copyright compliance | EU AI Act, GDPR, copyright directives |

---

## 8. Compliance Implementation

### 8.1 AI Compliance Risk Classifier (Python)

The following Python implementation demonstrates a practical compliance tool that classifies AI systems under the EU AI Act risk framework. It parses system attributes, applies the regulatory rules, and outputs risk levels with corresponding obligations.

```python
#!/usr/bin/env python3
"""
AI Compliance Checker — EU AI Act Risk Classification
------------------------------------------------------
Classifies an AI system's risk level per the EU AI Act 
tiers: Unacceptable, High-risk, Limited, Minimal.

Usage:
    python ai_compliance_checker.py [--input INPUT_FILE] [--output OUTPUT_FILE]
    Or import the AiComplianceChecker class in your own code.

Dependencies: Python 3.9+ (no external packages required)
"""

import json
import csv
import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class AiSystem:
    """Represents an AI system subject to compliance assessment."""
    name: str
    description: str
    use_case_category: str               # e.g., "cv_screening", "chatbot", "biometric_surveillance"
    sector: str                          # e.g., "employment", "healthcare", "finance"
    is_general_purpose: bool = False     # GPAI flag
    is_biometric_categorization: bool = False
    uses_real_time_biometric: bool = False
    involves_social_scoring: bool = False
    involves_manipulation: bool = False
    involves_vulnerable_populations: bool = False
    training_compute_flops: Optional[float] = None  # 10²⁵ FLOPs threshold

    # Additional metadata for GPAI obligations
    training_data_size: Optional[int] = None  # number of tokens / samples
    is_open_weight: bool = False


@dataclass
class ComplianceResult:
    """Classification result with obligations."""
    risk_level: str                       # "Unacceptable" | "High-risk" | "Limited" | "Minimal"
    requirements: list                    # list of obligation strings
    estimated_fine_pct: float             # % of global turnover
    applies_gpai: bool = False
    gpai_obligations: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Classification Rules Engine
# ---------------------------------------------------------------------------

class EuAiActClassifier:
    """
    Implements the EU AI Act risk classification rules.
    Reference: Regulation (EU) 2024/1689, Titles II–IV.
    """

    # High-risk categories defined in Annex III
    ANNEX_III_CATEGORIES = {
        "cv_screening": "employment",
        "credit_scoring": "finance",
        "medical_device": "healthcare",
        "critical_infrastructure": "critical_infrastructure",
        "emotion_recognition": "justice_or_democracy",
        "biometric_categorization": "law_enforcement",
        "educational_access": "education",
        "benefit_eligibility": "social_services",
    }

    # GPAI systemic-risk threshold: 10²⁵ FLOPs
    GPAI_SYSTEMIC_THRESHOLD = 1e25

    RISK_LEVELS = ["Minimal", "Limited", "High-risk", "Unacceptable"]
    PROHIBITED_PRACTICES = [
        "social_scoring",
        "real_time_biometric",
        "manipulative_ai",
        "exploiting_vulnerabilities",
    ]

    def classify(self, system: AiSystem) -> ComplianceResult:
        """Determine the risk level for a given AI system."""
        risk_level = self._determine_risk_level(system)
        requirements = self._get_requirements(risk_level)
        fine_pct = self._get_fine_pct(risk_level)

        result = ComplianceResult(
            risk_level=risk_level,
            requirements=requirements,
            estimated_fine_pct=fine_pct,
        )

        # GPAI obligations
        if system.is_general_purpose:
            result.applies_gpai = True
            result.gpai_obligations = self._get_gpai_obligations(system)

        return result

    def _determine_risk_level(self, system: AiSystem) -> str:
        # Step 1: Unacceptable risk (PROHIBITED)
        if system.involves_social_scoring:
            return "Unacceptable"
        if system.uses_real_time_biometric:
            return "Unacceptable"
        if system.involves_manipulation:
            return "Unacceptable"
        if system.involves_vulnerable_populations and system.is_biometric_categorization:
            return "Unacceptable"

        # Step 2: High-risk — check Annex III categories
        if system.use_case_category in self.ANNEX_III_CATEGORIES:
            # Exception: if the AI system performs narrow procedural tasks
            # (Art. 6(3) derogation), it may be downgraded. For this
            # implementation we assume full classification.
            return "High-risk"

        # Step 3: Limited risk — transparency obligation
        if system.use_case_category in ("chatbot", "emotion_recognition", "deepfake"):
            if not system.is_biometric_categorization:
                return "Limited"

        # Step 4: GPAI with systemic risk  — mapped to High-risk for obligations
        if system.is_general_purpose:
            if system.training_compute_flops and system.training_compute_flops >= self.GPAI_SYSTEMIC_THRESHOLD:
                return "High-risk"

        # Step 5: Default — Minimal risk
        return "Minimal"

    def _get_requirements(self, level: str) -> list:
        reqs = {
            "Unacceptable": ["Prohibited — system may not be placed on the market or deployed"],
            "High-risk": [
                "Risk management system (Art. 9)",
                "Data governance and training data quality (Art. 10)",
                "Technical documentation (Art. 11)",
                "Record-keeping & logging (Art. 12)",
                "Transparency & provision of information (Art. 13)",
                "Human oversight (Art. 14)",
                "Accuracy, robustness, cybersecurity (Art. 15)",
            ],
            "Limited": ["Transparency obligation — users must be informed they are interacting with AI"],
            "Minimal": ["No specific obligations under the AI Act"],
        }
        return reqs.get(level, ["Unknown risk level"])

    def _get_fine_pct(self, level: str) -> float:
        fines = {
            "Unacceptable": 7.0,  # higher of €35M or 7% of annual global turnover
            "High-risk": 3.0,     # higher of €15M or 3%
            "Limited": 0.0,       # transparency — direct fine not specified
            "Minimal": 0.0,
        }
        return fines.get(level, 0.0)

    def _get_gpai_obligations(self, system: AiSystem) -> list:
        obligations = [
            "Transparency — document training data, model capabilities, limitations",
            "Copyright compliance — policy respecting EU copyright law",
            "Training data documentation — sufficiently detailed summaries of sources",
        ]
        if system.training_compute_flops and system.training_compute_flops >= self.GPAI_SYSTEMIC_THRESHOLD:
            obligations.extend([
                "Systemic risk evaluation (Art. 55) — standardized model evaluations",
                "Adversarial testing — red-teaming by independent evaluators",
                "Incident reporting — report serious incidents to AI Office within 15 days",
                "Cybersecurity — adequate protection against adversarial attacks",
            ])
        return obligations


# ---------------------------------------------------------------------------
# Scenario Library — pre-built test cases
# ---------------------------------------------------------------------------

SCENARIOS = [
    AiSystem(
        name="TuringCV",
        description="AI-powered CV screening tool for enterprise hiring",
        use_case_category="cv_screening",
        sector="employment",
    ),
    AiSystem(
        name="CareBot",
        description="Customer service chatbot for healthcare inquiries",
        use_case_category="chatbot",
        sector="healthcare",
    ),
    AiSystem(
        name="SocialScoreX",
        description="Social credit scoring system for government benefits",
        use_case_category="benefit_eligibility",
        sector="social_services",
        involves_social_scoring=True,
    ),
    AiSystem(
        name="FrontierLM v3",
        description="Large language model trained on 3e25 FLOPs",
        use_case_category="general_purpose_llm",
        sector="technology",
        is_general_purpose=True,
        training_compute_flops=3e25,
    ),
    AiSystem(
        name="SmartFilter",
        description="AI spam filter for email classification",
        use_case_category="spam_filter",
        sector="technology",
    ),
    AiSystem(
        name="EmoWare",
        description="Classroom emotion recognition for student engagement tracking",
        use_case_category="emotion_recognition",
        sector="education",
        is_general_purpose=False,
    ),
    AiSystem(
        name="VisionGuard Pro",
        description="Real-time biometric surveillance deployed in public transit",
        use_case_category="biometric_surveillance",
        sector="law_enforcement",
        uses_real_time_biometric=True,
    ),
    AiSystem(
        name="AlphaMed",
        description="AI-assisted diagnostic medical device — Class IIa",
        use_case_category="medical_device",
        sector="healthcare",
    ),
]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def generate_report(system: AiSystem, result: ComplianceResult, format: str = "text"):
    """Format the compliance result for output."""
    if format == "json":
        return json.dumps({
            "system": system.name,
            "description": system.description,
            "risk_level": result.risk_level,
            "requirements": result.requirements,
            "gpai_applies": result.applies_gpai,
            "gpai_obligations": result.gpai_obligations,
            "estimated_max_fine_pct": result.estimated_fine_pct,
        }, indent=2)
    elif format == "csv":
        return f'{system.name},{result.risk_level},"{"; ".join(result.requirements)}"'
    else:
        lines = [
            f"\n{'='*60}",
            f"  System:        {system.name}",
            f"  Description:   {system.description}",
            f"  Sector:        {system.sector}",
            f"  Category:      {system.use_case_category}",
            f"{'='*60}",
            f"  Risk Level:    {result.risk_level}",
            f"  Max Fine:      {result.estimated_fine_pct}% of global turnover",
            f"{'─'*60}",
        ]
        if result.requirements:
            lines.append("  Obligations:")
            for req in result.requirements:
                lines.append(f"    • {req}")
        if result.applies_gpai:
            lines.extend([
                f"{'─'*60}",
                "  GPAI Obligations:",
            ])
            for ob in result.gpai_obligations:
                lines.append(f"    • {ob}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


def process_scenarios(scenarios=None, output_format="text", output_file=None):
    """Run classification against a set of scenarios."""
    classifier = EuAiActClassifier()
    if scenarios is None:
        scenarios = SCENARIOS

    outputs = []
    for system in scenarios:
        result = classifier.classify(system)
        outputs.append(generate_report(system, result, format=output_format))

    report = "\n".join(outputs)
    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        print(f"Report written to {output_file}")
    else:
        print(report)

    return outputs


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="EU AI Act Compliance Risk Classifier",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="JSON input file with AiSystem definitions",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format (default: text)",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    if args.input:
        with open(args.input) as f:
            data = json.load(f)
        scenarios = [AiSystem(**item) for item in data]
    else:
        scenarios = SCENARIOS

    process_scenarios(scenarios, output_format=args.format, output_file=args.output)


if __name__ == "__main__":
    main()
```

**Example Output** (running `python ai_compliance_checker.py` on included scenarios):

```
============================================================
  System:        FrontierLM v3
  Description:   Large language model trained on 3e25 FLOPs
  Sector:        technology
  Category:      general_purpose_llm
============================================================
  Risk Level:    High-risk
  Max Fine:      3.0% of global turnover
────────────────────────────────────────────────────────────
  Obligations:
    • Risk management system (Art. 9)
    • Data governance and training data quality (Art. 10)
    • Technical documentation (Art. 11)
    • Record-keeping & logging (Art. 12)
    • Transparency & provision of information (Art. 13)
    • Human oversight (Art. 14)
    • Accuracy, robustness, cybersecurity (Art. 15)
────────────────────────────────────────────────────────────
  GPAI Obligations:
    • Transparency — document training data, model capabilities, limitations
    • Copyright compliance — policy respecting EU copyright law
    • Training data documentation — sufficiently detailed summaries of sources
    • Systemic risk evaluation (Art. 55) — standardized model evaluations
    • Adversarial testing — red-teaming by independent evaluators
    • Incident reporting — report serious incidents to AI Office within 15 days
    • Cybersecurity — adequate protection against adversarial attacks
============================================================
```

### 8.2 Integration into Compliance Workflows

The classifier above can be integrated into a broader **AI compliance pipeline**:

1. **System Registry** — Maintain an inventory of all AI systems in production (model registry).
2. **Automated Classification** — Run the classifier on each system to determine risk tier.
3. **Gap Analysis** — Compare current practices against obligations for the system's risk tier.
4. **Remediation Tracking** — Assign action items (risk assessment, documentation, human oversight procedures) with deadlines.
5. **Audit Trail** — Log all classification decisions, including override reasons and review dates.
6. **Continuous Monitoring** — Re-classify when system behavior or regulatory guidance changes.

### 8.3 EU AI Act Compliance Checklist

| # | Requirement | High-Risk | GPAI Systemic | Status |
|---|-------------|:---------:|:-------------:|:------:|
| 1 | Risk management system established | ✅ | ✅ | — |
| 2 | Training data quality & governance | ✅ | ✅ | — |
| 3 | Technical documentation prepared | ✅ | ✅ | — |
| 4 | Automatic logging enabled | ✅ | — | — |
| 5 | Transparency & user disclosure | ✅ | ✅ | — |
| 6 | Human oversight measures in place | ✅ | — | — |
| 7 | Accuracy, robustness, cybersecurity testing | ✅ | ✅ | — |
| 8 | Conformity assessment (CE marking) | ✅ | — | — |
| 9 | Registration in EU database | ✅ | — | — |
| 10 | Model evaluation & adversarial testing | — | ✅ | — |
| 11 | Serious incident reporting mechanism | — | ✅ | — |
| 12 | EU representative appointed (non-EU providers) | ✅ | ✅ | — |

---

## 9. Emerging Governance Technology

### 9.1 AI Audit & Evaluation Tools
| Tool / Platform | Purpose | Maintainer |
|-----------------|---------|------------|
| **Inspect** | Framework for AI safety evaluations (scoring, benchmarks, sandboxing) | UK AISI |
| **Garak** | Automated red-teaming & vulnerability scanning | NVIDIA / community |
| **lm-evaluation-harness** | Standardized model capability and bias benchmarks | EleutherAI |
| **Honeycomb** | Interpretability and mechanistic analysis | Anthropic |
| **AI Audit Toolkit (AAT)** | Algorithmic auditing for fairness, accountability, transparency | Partnership on AI / Omidyar Network |
| **Eval Safety** | Safety-focused evaluation suite (harmlessness, bias, jailbreaks) | MLCommons AI Safety |
| **Fairlearn** | Fairness assessment and mitigation for ML models | Microsoft Research |
| **AI Incident Database (AIID)** | Structured repository of AI incidents for trend analysis | Partnership on AI |

### 9.2 Model Registries & Transparency Repositories
**Model registries** are centralized databases where AI developers publish governance-relevant information about their systems. They are a key emerging infrastructure for AI accountability:

| Registry | Scope | Content | Status |
|----------|-------|---------|--------|
| **EU AI Act GPAI Registry** | All general-purpose AI models placed on EU market | Model cards, training data summaries, copyright policy, evaluation results | Operational Aug 2025 |
| **Hugging Face Model Card Registry** | Voluntary, community-driven | Model cards, dataset cards, usage guidelines | Active (100k+ models) |
| **US AI Safety Institute Model Registry** | Frontier models (US-developed or deployed) | Safety evaluations, red-teaming results, capability assessments | Pilot (2025) |
| **China CAC Algorithm Registry** | Recommendation and generative AI algorithms | Algorithm name, type, purpose, data used | Operational as of 2023 |
| **Global AI Model Registry (proposed)** | UN-backed universal model catalog | Standardized governance metadata across jurisdictions | Proposal stage (2025) |

### 9.3 Emerging Technical Standards
| Standard | Focus | Body | Status |
|----------|-------|------|--------|
| **ISO/IEC 42001** | AI Management System | ISO/IEC JTC 1/SC 42 | Published Dec 2023 |
| **ISO/IEC 23894** | AI Risk Management Guidance | ISO/IEC JTC 1/SC 42 | Published Feb 2024 |
| **ISO/IEC 42005** | AI Impact Assessment | ISO/IEC JTC 1/SC 42 | Draft (2025) |
| **NIST AI 600-1** | AI Risk Management Framework Playbook | NIST | Updated Apr 2025 |
| **IEEE 7000 Series** | Ethically Aligned Design Standards | IEEE | Multiple published |
| **CEN/CENELEC AI Standards** | Harmonized EU standards for AI Act compliance | CEN/CENELEC JTC 21 | In progress (2024–2026) |
| **C2PA (Content Credentials)** | Provenance and authenticity for AI-generated content | C2PA Coalition | v2.1 (2025) |

### 9.4 AI Governance Platforms
Enterprise governance platforms that operationalize AI compliance at scale:

| Platform | Capabilities | Deployment |
|----------|--------------|------------|
| **Credo AI** | Risk assessment, policy mapping, monitoring dashboards | SaaS / On-prem |
| **Monitaur** | Impact assessments, model inventory, documentation | SaaS |
| **Fairly AI** | Automated bias testing, fairness metrics, reporting | API / SaaS |
| **Protect AI** | ML supply chain security, model scanning, vulnerability DB | SaaS / Open-source |
| **Weights & Biases (W&B)** | Experiment tracking, model registry, artifact lineage | SaaS |

---

## 10. Enforcement Actions Across Jurisdictions

### 10.1 Notable Enforcement Cases
| Date | Jurisdiction | Entity | Case / Action | Basis | Penalty |
|:----:|:------------:|--------|---------------|:-----:|:-------:|
| Jul 2023 | Italy (EU) | OpenAI / ChatGPT | Temporary suspension; alleged GDPR + AI Act violations over data scraping and lack of age verification | GDPR Art. 5, 6, 9 | Suspension lifted after compliance changes |
| Jan 2024 | EU (EDPB) | Meta / Facebook & Instagram | Ban on behavioral advertising using inferred sensitive data | GDPR Art. 9, 22 | Order to cease processing |
| Feb 2024 | South Korea | Google / Lotte | Fine for improper AI collection of personal data without consent | Personal Information Protection Act | KRW 3.5B (~$2.6M) |
| Apr 2024 | US (FTC) | Rite Aid | AI facial recognition misidentification — false shoplifting accusations | FTC Act Sec. 5 | 5-year ban on AI surveillance, settlement |
| Sep 2024 | US (FTC) | Amazon | Amazon use of AI hiring tool — gender discrimination claims | Title VII, FTC Act | Pending litigation |
| Nov 2024 | China (CAC) | Baidu | Inadequate content moderation on Ernie Bot generative AI outputs | GAI Measures | Public reprimand, temporary service suspension |
| Jan 2025 | EU (CNIL, France) | Clearview AI | Fines for biometric data scraping without consent | GDPR | €20M (cumulative across EU fines) |
| Mar 2025 | US (EEOC) | Workday | Algorithmic hiring discrimination lawsuit — disparate impact on protected groups | Title VII | Settlement pending (2026) |
| May 2025 | UK (ICO) | Snapchat | My AI chatbot — children's privacy risks over data processing | UK GDPR, PECR | Preliminary enforcement notice |
| Jul 2025 | Canada (OPC) | OpenAI | Investigation into ChatGPT data collection, use and disclosure | PIPEDA | Compliance agreement reached |

### 10.2 Enforcement Comparison by Jurisdiction

| Dimension | European Union | United States | China | United Kingdom |
|-----------|:--------------:|:-------------:|:-----:|:--------------:|
| **Primary regulator** | AI Office + national authorities | FTC, EEOC, DOJ (sectoral) | CAC (Cyberspace Administration) | ICO + DSIT |
| **Legal basis** | EU AI Act + GDPR | FTC Act Sec. 5, Civil Rights Acts | GAI Measures, DSL, PIPL | UK GDPR, pending AI Bill |
| **Enforcement style** | Proactive — codes of practice, pre-market conformity | Reactive — enforcement actions after harm | Proactive — algorithm registration, ex-ante approval | Hybrid — ICO investigatory powers |
| **Penalty ceiling** | Up to 7% of global turnover | Consumer redress, bans, disgorgement | Service suspension, license revocation, criminal liability | Up to £17.5M or 4% of turnover (GDPR) |
| **Criminal penalties?** | No (administrative fines) | Yes (fraud, civil rights violations) | Yes (content crimes, data breaches) | Limited |
| **Extraterritorial scope** | Yes — applies to any provider targeting EU market | Limited — jurisdiction over US-based harms | Yes — applies to algorithms affecting Chinese users | Yes — UK GDPR scope |
| **Notable gaps** | Enforcement capacity, speed of procedures | Fragmented, no comprehensive federal law | Transparency limited, state-controlled litigation | AI Bill still in legislative process |
| **First major enforcement** | Italian ChatGPT suspension (2023) | Rite Aid facial recognition ban (2024) | Baidu Ernie Bot suspension (2024) | Snapchat enforcement notice (2025) |

### 10.3 Enforcement Trends & Predictions (2026–2028)
1. **First EU AI Act fines** expected in H2 2026 for GPAI transparency violations, setting precedent for fine levels.
2. **US federal AI law** (if enacted in 2026–2027) would create a dedicated agency or empower FTC with AI-specific rulemaking.
3. **Cross-jurisdictional cooperation** increasing — the Global AI Enforcement Network (GAEN) proposed at the 2025 Paris Summit would enable information sharing between regulators.
4. **Private right of action** emerging — courts in the US and EU are increasingly allowing AI-related class actions (bias, discrimination, privacy harms).
5. **Algorithmic audit mandates** becoming standard — NYC Law 144, Colorado AI Act, and likely future federal law require independent third-party audits.

---

## 11. Emerging Issues
- **Open-weight models:** How to regulate models released without usage restrictions; tension between open science and safety
- **Compute governance:** Tracking large training runs (compute thresholds); proposed compute registries and licensing for >10²⁶ FLOPs
- **AI-generated content:** Labeling requirements (C2PA, SynthID, DALL·E watermarking); detection vs. provenance approaches
- **Vote manipulation:** AI in political campaigns, synthetic media disclosure, deepfake election interference
- **AI whistleblowing:** Legal protection for AI safety disclosures (e.g., EU Whistleblower Directive, proposed US AI Whistleblower Act)
- **Compute allocation:** Equitable access to AI compute globally; sovereignty concerns over chip supply chains
- **Right to opt-out:** Individual rights to reject automated decision-making (expanding GDPR Art. 22)
- **AI liability:** Civil liability for AI-caused harms — EU AI Liability Directive, US product liability for software
- **Labor displacement:** Regulatory frameworks for AI-driven job automation, universal basic income pilots, reskilling mandates
- **Sovereign AI:** Nations building independent AI capacity to reduce dependency on US/Chinese infrastructure; implications for governance interoperability
## 11a. AI Governance for Startups and SMBs

Enterprise-grade AI governance frameworks (EU AI Act compliance, NIST AI RMF, ISO 42001) are essential for large organizations, but they can be overwhelming and cost-prohibitive for startups and small-to-medium businesses. This section provides a practical, risk-based approach to AI governance that scales with organizational size and resources.

### 11a.1 The Startup Governance Minimum

Not every AI startup needs a full compliance department. The minimum viable governance stack depends on risk level:

| Risk Level | Examples | Minimum Viable Governance | Estimated Cost (annual) |
|:-----------|:---------|:------------------------|:-----------------------:|
| **Low** | Spam filter, product recommender, content summarizer | Basic documentation (model card + data provenance), acceptable use policy, opt-out mechanism | $1,000–$5,000 |
| **Medium** | Customer support chatbot, code completion tool, HR screener | Above + bias testing, output guardrails, incident response plan, third-party AI audit | $5,000–$25,000 |
| **High** | Medical diagnosis support, credit scoring, hiring platform | Above + full EU AI Act compliance (risk management system, human oversight, logging), independent auditor, legal counsel | $25,000–$100,000+ |
| **Critical** | Autonomous driving, biometric surveillance, critical infrastructure | Above + formal verification, regulatory pre-approval, insurance, dedicated safety team | $100,000+ |

### 11a.2 Cost-Effective Compliance Strategy

```
Priority 1: Documentation (lowest cost, highest impact)

+-- Model Cards — 1-page summaries of each AI system (capabilities, limitations, training data)
+-- Data Provenance — Where training data came from, consent status, licensing
+-- Acceptable Use Policy — What users may and may not do with your AI
+-- Incident Response Plan — What to do when something goes wrong

Priority 2: Technical Controls (moderate cost, enables trust)

+-- Input/Output Guardrails — Block harmful inputs and outputs
+-- Basic Bias Testing — Measure performance across demographic groups
+-- Usage Monitoring — Track how your AI is being used, detect anomalies
+-- Opt-Out Mechanism — Allow users to refuse AI processing

Priority 3: External Validation (higher cost, builds credibility)

+-- Third-Party Audit — Independent review of governance documentation and controls
+-- Penetration Testing — Red-teaming for security and safety
+-- Certification — ISO 42001 (AI management), SOC 2 Type II
```

### 11a.3 Governance Artifact Templates for Startups

Instead of building compliance documentation from scratch, startups can use existing templates adapted for small teams:

| Artifact | Template Source | Time to Complete | Key Sections for Startups |
|:---------|:---------------|:----------------:|:--------------------------|
| **Model Card** | Hugging Face Model Card template | 2–4 hours | Intended use, limitations, evaluation results, ethical considerations |
| **Dataset Card** | Hugging Face Dataset Card template | 1–2 hours per dataset | Data sources, preprocessing, biases, consent status |
| **AI Risk Assessment** | NIST AI RMF Quick Start (startup edition) | 4–8 hours | Use case context, risk identification, mitigation measures |
| **Incident Response Plan** | OWASP AI Incident Response template | 2–4 hours | Escalation path, containment steps, communication plan |
| **Bias Test Report** | AI Fairness 360 (IBM) template | 4–16 hours | Metrics used, results by group, remediation actions |

### 11a.4 When to Invest More in Governance

| Trigger Point | Recommended Action | Estimated Cost |
|:--------------|:-------------------|:--------------:|
| **You raise your first institutional round (Series A)** | Hire part-time AI ethics consultant; complete model cards for all systems | $10K–$30K one-time |
| **You process EU user data** | GDPR compliance audit; EU representative appointment; data protection impact assessment | $15K–$50K |
| **Your AI is used in a regulated sector** (healthcare, finance, hiring) | Full risk assessment; third-party audit; legal counsel with AI expertise | $50K–$200K |
| **You deploy a frontier model** (>10²⁵ FLOPs training compute) | GPAI compliance program; systemic risk evaluation; incident reporting mechanism | $100K–$500K |
| **You receive a regulatory inquiry** | Legal response; documentation review; external counsel | $20K–$100K per inquiry |
| **Your AI causes harm** (or is alleged to) | Incident response; PR + legal + technical fixes; regulatory engagement | $100K–$1M+ |

### 11a.5 Open-Source and Low-Cost Governance Tools

| Tool | Purpose | Cost | Maintainer |
|:-----|:--------|:----:|:-----------|
| **AI Fairness 360 (AIF360)** | Bias detection and mitigation | Free | IBM Research |
| **AI Explainability 360 (AIX360)** | Model interpretability tools | Free | IBM Research |
| **Adversarial Robustness 360 (ART)** | Security testing for ML models | Free | IBM Research |
| **AI Incident Database (AIID)** | Learn from real-world AI failures | Free | Partnership on AI |
| **AI Verify** (Singapore) | AI governance testing toolkit | Free | IMDA Singapore |
| **Garak** | Automated red-teaming for LLMs | Free | NVIDIA / community |
| **NeMo Guardrails** | Input/output guardrails for LLMs | Free | NVIDIA |
| **Lakera Guard** | Prompt injection detection | Freemium | Lakera AI |
| **Guardrails AI** | Output validation and risk management | Open-core | Guardrails AI |

### 11a.6 Key Principles for Startup AI Governance

1. **Document first, automate later.** A simple README per model is better than nothing. You can automate compliance checks once you know what you need to check.
2. **Prioritize based on actual risk, not regulatory fear.** Most startups building low-risk AI systems do not need ISO 42001 certification. Focus on the controls that prevent real harm.
3. **Build governance into the product, not as an add-on.** The easiest way to ensure compliance is to design it into your architecture from day one — opt-out mechanisms, data provenance tracking, guardrails.
4. **Use open-source tools before buying enterprise solutions.** The open-source governance ecosystem is maturing rapidly. Startups can achieve 80% of compliance needs with free tools.
5. **Invest proportionally to your funding stage and risk profile.** A bootstrapped startup making a spam filter does not need the same governance as a venture-backed company deploying AI in healthcare.
6. **Plan for the next stage, don't over-engineer for it.** Document what you know today. When you raise funding, enter a new market, or hit a risk trigger, invest at that point. Premature governance is wasted effort.

**See also:** [05-Enterprise/01-Enterprise-AI-Deployment.md] for enterprise-grade compliance implementation; [07-Emerging/02-AI-Safety.md] for safety testing; [08-Reference/01-Glossary.md] for governance terminology.

---

## 12. Cross-References
| Reference | Description |
|-----------|-------------|
| [07-Emerging/02-AI-Safety.md] | Safety frameworks, RSPs, red teaming |
| [05-Enterprise/01-Enterprise-AI-Deployment.md] | Enterprise compliance implementation |
| [08-Reference/02-AI-Roadmap.md] | Future regulatory landscape |
| [08-Reference/01-Glossary.md] | Governance terminology |
| [07-Emerging/01-AI-Ethics.md] | Ethical frameworks and principles |
| [08-Reference/03-Regulatory-Timeline.md] | Key regulatory dates by jurisdiction |
|---
*Document version: 3.0 — June 2026 | Expanded with compliance tooling, enforcement actions, governance technology, §4a Asia-Pacific AI Governance, and §11a AI Governance for Startups and SMBs*
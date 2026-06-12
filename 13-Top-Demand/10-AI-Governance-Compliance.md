# AI Governance & Compliance

> **Last Updated:** June 2026  
> **Category:** Top Demand — Current Market Snapshot  
> **Cross-References:** 05-AI-Safety-Alignment.md, 07-Fine-Tuning-Custom-Models.md, 09-AI-Automation.md, 11-Real-Time-AI-Systems.md

---

## Table of Contents

1. [Market Context & Global Regulatory Landscape](#1-market-context--global-regulatory-landscape)
2. [EU AI Act (Full Analysis)](#2-eu-ai-act-full-analysis)
   - 2.1 Tier Classification System
   - 2.2 Obligations by Tier
   - 2.3 Penalties & Enforcement
   - 2.4 Compliance Checklist
3. [US AI Executive Orders & Federal Regulation](#3-us-ai-executive-orders--federal-regulation)
   - 3.1 Executive Order 14110 (October 2023)
   - 3.2 AI Executive Order 2025 Expansion
   - 3.3 State-Level Regulation (Colorado, California, New York)
   - 3.4 Compliance Checklist
4. [China AI Regulations](#4-china-ai-regulations)
   - 4.1 Algorithm Registration Framework
   - 4.2 Content Moderation Requirements
   - 4.3 Generative AI Rules (Deep Synthesis Provisions)
   - 4.4 Compliance Checklist
5. [NIST AI Risk Management Framework](#5-nist-ai-risk-management-framework)
   - 5.1 Core Functions (Govern, Map, Measure, Manage)
   - 5.2 Categories & Subcategories
   - 5.3 Playbook Implementation
   - 5.4 Compliance Checklist
6. [ISO 42001 — AI Management System](#6-iso-42001--ai-management-system)
   - 6.1 Structure & Clauses
   - 6.2 Annex A Controls
   - 6.3 Certification Process
   - 6.4 Compliance Checklist
7. [Sectoral Regulations](#7-sectoral-regulations)
   - 7.1 FDA — Medical AI / SaMD
   - 7.2 SEC — Algorithmic Trading & AI in Finance
   - 7.3 FCC — AI Robocalls & Communications
   - 7.4 EU Sectoral: MDR, GDPR, DSA Intersections
   - 7.5 Compliance Checklists
8. [Model Auditing Requirements](#8-model-auditing-requirements)
   - 8.1 Internal vs External Audits
   - 8.2 Audit Methodology
   - 8.3 Automated Auditing Tools
   - 8.4 Red-Teaming Mandates
9. [Bias Testing Methodologies](#9-bias-testing-methodologies)
   - 9.1 Statistical Parity Metrics
   - 9.2 Disparate Impact Analysis
   - 9.3 Intersectional Bias Testing
   - 9.4 Dataset Bias Audits
   - 9.5 Post-Deployment Monitoring
10. [Explainability Requirements & XAI Techniques](#10-explainability-requirements--xai-techniques)
    - 10.1 Intrinsic vs Post-Hoc Methods
    - 10.2 SHAP & SHAPley Values
    - 10.3 LIME (Local Interpretable Model-Agnostic Explanations)
    - 10.4 Integrated Gradients
    - 10.5 Attention-Based Explanations
    - 10.6 Concept Activation Vectors
    - 10.7 Regulatory Acceptability of XAI Methods
11. [Documentation Standards](#11-documentation-standards)
    - 11.1 Model Cards (Google/Mitchell et al.)
    - 11.2 Data Sheets (Gebru et al.)
    - 11.3 System Cards (OpenAI)
    - 11.4 AI Facts Sheets (IBM)
    - 11.5 EU AI Act Technical Documentation Template
    - 11.6 Template Examples
12. [International Alignment Efforts](#12-international-alignment-efforts)
    - 13.1 OECD AI Principles
    - 13.2 GPAI (Global Partnership on AI)
    - 13.3 Council of Europe AI Convention
    - 13.4 G7 Hiroshima AI Process
    - 13.5 UNESCO AI Ethics Recommendation
    - 13.6 Bletchley Declaration & AI Safety Summits
13. [Enforcement Trends & Notable Cases](#13-enforcement-trends--notable-cases)
14. [Future Outlook](#14-future-outlook)

---

## 1. Market Context & Global Regulatory Landscape

As of June 2026, AI governance and compliance has become one of the most rapidly evolving regulatory domains worldwide. The convergence of binding regulation (EU AI Act enforcement beginning August 2026), executive action (US AI Executive Orders), and national security concerns (China's comprehensive AI laws) has created a complex compliance environment for organizations deploying AI globally.

**Key market signals:**
- Global spending on AI governance, risk management, and compliance (GRC) tooling: $6.8B in 2026, projected $18B by 2030
- 73% of enterprises with AI in production report compliance as their top operational concern (Gartner AI Governance Survey, Q1 2026)
- AI compliance officer roles have grown 450% since 2023
- 38% of AI deployments in the EU have been delayed or modified due to AI Act compliance concerns
- Non-compliance penalties globally have exceeded €2.1B in 2025–2026 across all AI-related regulations

**Why now?**
- **EU AI Act enforcement** begins August 2, 2026 — prohibitions on unacceptable risk AI systems take effect immediately
- **US federal AI regulation** is advancing through multiple executive orders and proposed legislation (AI Foundation Model Transparency Act, NO FAKES Act)
- **China's AI regulations** now cover all stages of the AI lifecycle from training data to deployment
- **ISO 42001 certification** is becoming a de facto requirement for government AI procurement
- **Cross-border AI deployments** must satisfy multiple, sometimes conflicting, regulatory regimes simultaneously
- **High-profile enforcement actions** (Italian Garante against ChatGPT, Dutch DPA against bias-scoring AI) have demonstrated regulators' willingness to act

**The compliance stack for modern AI systems:**

```
┌─────────────────────────────────────────────────┐
│           REGULATORY LAYERS                      │
│  ┌─────────────────────────────────────────┐    │
│  │  Horizontal: EU AI Act, ISO 42001,      │    │
│  │  NIST AI RMF, OECD Principles           │    │
│  ├─────────────────────────────────────────┤    │
│  │  Sectoral: FDA, SEC, FCC, HIPAA,       │    │
│  │  GDPR, CCPA, MDR, DSA                  │    │
│  ├─────────────────────────────────────────┤    │
│  │  National: US EO, China Regulations,   │    │
│  │  UK AI Principles, Canada AIDA         │    │
│  ├─────────────────────────────────────────┤    │
│  │  Technical: Model Cards, Data Sheets,  │    │
│  │  XAI, Bias Audits, Red-Teaming         │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

---

## 2. EU AI Act (Full Analysis)

The EU AI Act (Regulation (EU) 2024/1689) is the world's first comprehensive horizontal AI regulation. Adopted in May 2024, its enforcement begins in phases culminating in full applicability by August 2027. As of June 2026, the **unacceptable risk prohibitions** take effect in just weeks (August 2, 2026).

### 2.1 Tier Classification System

The AI Act uses a **risk-based pyramid** with four tiers:

```
                    ┌────────────┐
                    │ UNACCEPTABLE│ ← Banned outright
                    │   Risk      │
                    ├────────────┤
                    │   HIGH      │ ← Most obligations
                    │   Risk      │
                    ├────────────┤
                    │ LIMITED     │ ← Transparency only
                    │   Risk      │
                    ├────────────┤
                    │ MINIMAL     │ ← No obligations
                    │   Risk      │
                    └────────────┘
```

**Tier 1 — Unacceptable Risk (Prohibited AI Practices, Art. 5):**
- **Subliminal manipulation** — AI systems deploying subliminal techniques to distort behavior causing physical or psychological harm
- **Vulnerability exploitation** — Systems exploiting vulnerabilities of persons due to age, disability, or socioeconomic situation
- **Social scoring** — AI-based social scoring by public authorities (China-style systems)
- **Real-time biometric identification** — Law enforcement use of real-time remote biometric identification in publicly accessible spaces (with limited exceptions)
- **Predictive policing** — Individual risk assessment based solely on profiling or personality traits
- **Emotion inference in workplaces/education** — AI systems inferring emotions in workplace or educational settings
- **Biometric categorization** — Systems categorizing people based on biometric data to infer race, political opinions, religion, etc.
- **Untargeted facial image scraping** — Creating or expanding facial recognition databases via untargeted scraping of images from the internet or CCTV footage

**Tier 2 — High Risk (Art. 6 + Annex III):**
AI systems that pose significant risk to health, safety, or fundamental rights. Divided into two categories:

*Category A: Products covered by EU harmonization legislation (Toys Directive, Medical Devices Regulation, Machinery Regulation, etc.) that require third-party conformity assessment*

*Category B: Standalone AI systems in listed areas:*
1. Biometric identification and categorization of natural persons
2. Management and operation of critical infrastructure (transport, energy, water)
3. Education and vocational training (access to education, assessment)
4. Employment, workers management, and access to self-employment
5. Access to essential services (credit scoring, insurance pricing)
6. Law enforcement (evidence evaluation, risk assessment)
7. Migration, asylum, and border control management
8. Administration of justice and democratic processes

**Tier 3 — Limited Risk (Art. 50):**
- **Chatbots and conversational AI** — Must disclose that users are interacting with an AI system
- **Deepfakes / AI-generated content** — Must be labeled as artificially generated or manipulated
- **Emotion recognition or biometric categorization systems** — Must inform data subjects

**Tier 4 — Minimal Risk:**
- All other AI systems not falling into the above categories
- Voluntary codes of conduct encouraged (Art. 69)

**General-purpose AI (GPAI) / Foundation Models (Art. 51–56):**

The AI Act also introduces specific rules for general-purpose AI models, categorized by systemic risk:

```
┌──────────────────────────────────────────────┐
│         GPAI MODEL CLASSIFICATION            │
├──────────────────┬───────────────────────────┤
│  GPAI (Standard) │  GPAI with Systemic Risk  │
│                  │  (≥10^25 FLOPs training)   │
├──────────────────┼───────────────────────────┤
│ Technical        │ Full technical            │
│ documentation +  │ documentation +           │
│ Copyright policy │ Model evaluation +        │
│ Training data    │ Adversarial testing +     │
│ summary          │ Incident reporting +      │
│                  │ Cybersecurity measures    │
└──────────────────┴───────────────────────────┘
```

The European Commission designates systemic risk based on computational threshold (10^25 FLOPs for training) or by designation based on capabilities.

### 2.2 Obligations by Tier

**High-Risk AI System Obligations (Art. 8–29):**

1. **Risk management system** (Art. 9):
   - Continuous, iterative process throughout the entire AI system lifecycle
   - Identify and analyze known and foreseeable risks
   - Estimate and evaluate risks that emerge when deployed in accordance with intended purpose
   - Implement risk management measures
   - Testing must consider reasonably foreseeable misuse
   - Residual risk must be deemed acceptable

2. **Data governance** (Art. 10):
   - Training, validation, and testing datasets must be relevant, representative, and free from errors
   - Appropriate data governance practices: examination of bias, data adequacy, completeness
   - Special categories of personal data processed only when strictly necessary
   - Data provenance documentation required

3. **Technical documentation** (Art. 11 + Annex IV):
   - General description of the AI system
   - Detailed system design methodology
   - Development process including system architecture
   - Training methodology and evaluation results
   - Performance metrics and intended purpose
   - Detailed description of risk management system

4. **Record-keeping & logging** (Art. 12):
   - Automatic logging of events during operation
   - Logs must cover: system activation/deactivation, input data timing, operator identification
   - Logs retained for period appropriate to intended purpose (minimum 6 months)
   - Enable traceability and post-hoc auditing

5. **Transparency & provision of information** (Art. 13):
   - AI systems must be designed to enable operators to understand system outputs
   - Instructions for use: characteristics, limitations, performance specifications
   - Information about accuracy, robustness, and cybersecurity levels

6. **Human oversight** (Art. 14):
   - Must enable operators to: fully understand system capabilities and limitations, remain aware of automation bias, correctly interpret system output, override or stop the system
   - Measures can be: human-in-the-loop, human-on-the-loop (oversight during operation), human-in-command (strategic supervision)

7. **Accuracy, robustness & cybersecurity** (Art. 15):
   - Appropriate levels of accuracy as declared in instructions for use
   - Resilience to errors, faults, and inconsistencies
   - Robustness against unauthorized modification or adversarial attacks
   - Cybersecurity measures appropriate to risk level

**GPAI Obligations:**

| Obligation | Standard GPAI | Systemic Risk GPAI |
|---|---|---|
| Technical documentation + Annex XI | ✓ | ✓ |
| Training data policy (copyright compliance) | ✓ | ✓ |
| Model evaluation (standardized protocols) | | ✓ |
| Adversarial testing / red-teaming | | ✓ |
| Systemic risk assessment & mitigation | | ✓ |
| Incident reporting to AI Office | | ✓ |
| Cybersecurity adequacy | | ✓ |
| Energy consumption reporting | | ✓ |
| Code of practice adherence | Optional | Expected |

### 2.3 Penalties & Enforcement

The EU AI Act establishes a graduated penalty structure with global reach:

| Violation Type | Maximum Fine | EU Revenue-Based Cap |
|---|---|---|
| Prohibited AI practices (Art. 5) | €35,000,000 or 7% of worldwide annual turnover | Whichever higher |
| Non-compliance with high-risk obligations | €15,000,000 or 3% of worldwide annual turnover | Whichever higher |
| Provision of incorrect/misleading info to authorities | €7,500,000 or 1.5% of worldwide annual turnover | Whichever higher |
| GPAI violations | €15,000,000 or 3% of worldwide annual turnover | Whichever higher |

**Enforcement architecture:**
- **European AI Office** (within European Commission) — Oversees GPAI enforcement, codes of practice
- **European Artificial Intelligence Board (EAIB)** — Advisory body of member state representatives
- **National competent authorities** — Each EU member state designates a notifying authority
- **Market surveillance authorities** — Systematic monitoring, market checks, and post-market surveillance
- **Notified bodies** — Third-party conformity assessment for certain high-risk systems (e.g., biometrics, critical infrastructure)

**Timeline:**
- **May 2024** — Regulation adopted
- **February 2025** — GPAI rules effective (codes of practice)
- **August 2, 2026** — Prohibitions on unacceptable risk AI practices apply
- **August 2, 2027** — High-risk rules apply (Annex III), GPAI rules fully applicable
- **August 2, 2028** — High-risk rules apply (AI systems that are products or safety components)

### 2.4 Compliance Checklist

```
EU AI ACT COMPLIANCE CHECKLIST
═══════════════════════════════

□ 1. AI System Inventory
   □ Catalog all AI systems in development or production
   □ Classify each system by risk tier (Unacceptable / High / Limited / Minimal)
   □ Document classification rationale for each system
   □ Update inventory whenever systems are modified

□ 2. Unacceptable Risk (Art. 5) — Immediate Action
   □ Audit all systems for prohibited AI practices
   □ Remove or redesign any systems that violate Art. 5
   □ Document removal/redesign process
   □ Implement safeguards against future prohibited uses
   □ Deadline: August 2, 2026

□ 3. High-Risk Compliance Program
   □ Establish risk management system (Art. 9)
   □ Implement data governance practices (Art. 10)
   □ Draft technical documentation per Annex IV (Art. 11)
   □ Configure automatic logging and record-keeping (Art. 12)
   □ Prepare transparency documentation for deployers (Art. 13)
   □ Design and document human oversight measures (Art. 14)
   □ Implement accuracy, robustness, and cybersecurity (Art. 15)
   □ Register high-risk AI system in EU database (Art. 49)

□ 4. Conformity Assessment
   □ Determine if third-party assessment is required
   □ If self-assessment: prepare Declaration of Conformity (Art. 47)
   □ If third-party: engage Notified Body
   □ Affix CE marking (Art. 48)
   □ Maintain conformity documentation for 10 years

□ 5. GPAI Compliance
   □ Determine if GPAI model qualifies as systemic risk
   □ Prepare technical documentation per Annex XI
   □ Implement copyright policy for training data
   □ Document training data sources and selection methodology
   □ Adhere to Code of Practice (if systemic risk)

□ 6. Transparency Obligations
   □ Add AI disclosure to all chatbot/ conversational interfaces
   □ Implement content labeling for AI-generated content
   □ Label emotional state inferences
   □ Keep records of interactions with limited-risk systems

□ 7. Governance Structure
   □ Appoint AI compliance officer or team
   □ Establish AI ethics board / review committee
   □ Implement AI compliance monitoring program
   □ Create incident reporting mechanism
   □ Schedule regular compliance reviews

□ 8. Post-Market Monitoring
   □ Implement post-market surveillance system (Art. 61)
   □ Report serious incidents to market surveillance authority
   □ Continuously monitor for systemic risks
   □ Update risk assessment when system is modified
   □ Retain logs and documentation per retention schedule
```

---

## 3. US AI Executive Orders & Federal Regulation

The United States has taken a multi-pronged approach combining executive action, agency-level regulation, and proposed legislation. As of June 2026, there is no comprehensive federal AI law (similar to the EU AI Act), but the regulatory environment is rapidly evolving.

### 3.1 Executive Order 14110 (October 2023)

Signed by President Biden on October 30, 2023, this remains the foundational US AI policy document.

**Key provisions:**

1. **AI Safety and Security:**
   - Developers of powerful AI systems (dual-use foundation models) must share safety test results with the federal government
   - NIST to develop standards for AI safety testing (red-teaming guidelines, watermarking)
   - Department of Homeland Security to establish AI Safety and Security Board
   - Critical infrastructure operators must notify government of AI safety incidents

2. **Privacy:**
   - Congress urged to pass bipartisan data privacy legislation
   - Federal agencies to evaluate how AI systems collect and use commercially available information
   - Development of privacy-preserving techniques (differential privacy, federated learning)

3. **Equity and Civil Rights:**
   - Guidance to prevent AI-driven discrimination in housing, federal benefits, and criminal justice
   - DOJ to develop enforcement of algorithmic discrimination
   - Fair housing AI audits

4. **Consumer Protection:**
   - HHS to develop AI safety programs in healthcare
   - DOT to address AI in transportation (aviation, autonomous vehicles)
   - CFPB to protect consumers from AI-based financial discrimination

5. **Innovation:**
   - National AI Research Resource (NAIRR) pilot launched
   - Expansion of AI talent in government
   - Visa modernization for AI experts

### 3.2 AI Executive Order 2025 Expansion

The 2025 Executive Order (signed by President Smith in January 2025) expanded on EO 14110 with binding obligations:

**New requirements:**
- **Mandatory safety reporting** — All foundation models trained on 10^26+ FLOPs must report training details, evaluation results, and safety testing to Department of Commerce
- **Algorithmic impact assessments** — Required for AI used in housing, employment, credit, healthcare, and criminal justice
- **Watermarking mandate** — AI-generated content from major platforms must include imperceptible watermarks
- **Federal procurement ban** — Federal agencies prohibited from procuring AI systems without bias testing
- **AI liability framework** — Proposed framework holding developers liable for foreseeable misuse

### 3.3 State-Level Regulation

In the absence of comprehensive federal legislation, states have become active AI regulators:

**Colorado AI Act (SB 24-205) — Effective 2026:**
- First comprehensive state AI law
- Regulates "high-risk" AI systems used for consequential decisions
- Requires risk assessments, impact assessments, and consumer disclosures
- Annual third-party audits for high-risk systems
- Creates private right of action for consumers harmed by discriminatory AI

**California AI Regulation (Multiple Bills):**
- **AB 2930 (AI Training Data Transparency)** — Requires disclosure of training data sources
- **SB 1047 (Frontier AI Safety)** — Requires safety testing for large-scale AI models
- **CPRA (California Privacy Rights Act)** — Expanded to cover AI-related data processing
- **Automated Decision Systems** — Regulations governing ADM in state government

**New York:**
- **NYC Local Law 144** — AI-based hiring tools must undergo bias audits (took effect 2024)
- **NY AI Child Safety Act** — Protections for minors interacting with AI systems
- **NY DFS Circular** — Insurance companies must certify AI models are free from unfair discrimination

**Other States with Active AI Legislation (2025–2026):**
- Connecticut (CT AI Act — risk management + impact assessments)
- Texas (AI in Government — transparency + procurement requirements)
- Illinois (AI in Employment — notice + consent requirements)
- Washington (WA My Health My Data Act — health data + AI intersection)
- Virginia (Consumer Data Protection Act amendments — AI profiling)

### 3.4 Compliance Checklist

```
US AI REGULATORY COMPLIANCE CHECKLIST
═══════════════════════════════════════

□ 1. Federal Executive Order Compliance
   □ If developing dual-use foundation models: prepare safety test results
   □ Establish red-teaming program per NIST standards
   □ Register for NAIRR if applicable
   □ Implement watermarking for AI-generated content
   □ Notify relevant agencies of AI safety incidents

□ 2. Department-Level Compliance
   □ EOF/NIST: Submit AI safety testing documentation
   □ DOC: Report foundation model training details (if >10^26 FLOPs)
   □ DHS: Coordinate with AI Safety and Security Board
   □ HHS: Implement healthcare AI safety program
   □ CFPB: Ensure AI financial decisions are non-discriminatory

□ 3. State-Level Compliance (check applicable states)
   □ Colorado: Register high-risk AI systems, conduct impact assessments
   □ California: Disclose training data sources, safety test frontier models
   □ New York: Audit AI hiring tools per LLC 144
   □ Connecticut: Implement risk management framework
   □ Texas: Comply with government AI procurement rules
   □ Illinois: Provide notice/consent for AI in employment
   □ Track emerging state requirements (updated quarterly)

□ 4. Algorithmic Impact Assessments
   □ Identify all AI systems used in consequential decisions
   □ Conduct bias testing (annual minimum)
   □ Document algorithm design, training data, and performance metrics
   □ Prepare consumer-facing disclosures
   □ File impact assessments with appropriate agency
   □ Third-party audit (where required)

□ 5. Sectoral Compliance
   □ Healthcare: FDA premarket review if AI is a medical device
   □ Finance: SEC, FINRA, CFPB compliance for AI-driven decisions
   □ Employment: EEOC guidance on AI discrimination
   □ Housing: HUD disparate impact analysis
   □ Criminal justice: DOJ guidance on AI in sentencing/risk assessment
```

---

## 4. China AI Regulations

China has established one of the world's most comprehensive AI regulatory regimes, covering the entire AI lifecycle from algorithm development to content moderation. The regulatory framework is evolving rapidly, with overlapping requirements from multiple agencies.

### 4.1 Algorithm Registration Framework

The **Algorithmic Recommendation Regulation** (effective March 1, 2022) and subsequent **Internet Information Service Algorithm Registration System** require:

**Registration requirements:**
- All algorithms providing recommendation, search, or content generation services to Chinese users must be registered
- Registration covers: algorithm type, purpose, training data, logic, and risk mitigation measures
- Public algorithm registration catalog maintained by CAC (Cyberspace Administration of China)
- Updates required when algorithms are significantly modified

**Algorithmic Governance Principles:**
- **Transparency** — Algorithms must not be "black box"; basic operational logic must be explainable
- **Fairness** — No price discrimination based on user data (algorithmic pricing bans)
- **User control** — Users must be able to disable algorithm-based recommendations
- **Content safety** — Algorithms must not propagate illegal or harmful information
- **Data minimization** — Only necessary data may be used for algorithm training and operation

**Platform Obligations:**
- Establish algorithm ethics review committees
- File annual algorithm compliance reports
- Maintain algorithm logs for at least 6 months
- Enable government access to algorithm verification and auditing

### 4.2 Content Moderation Requirements

China's content moderation regime for AI systems is governed by multiple overlapping regulations:

**Core requirements:**
- AI systems must not generate content that: subverts state power, endangers national security, damages national honor, promotes terrorism, undermines ethnic unity, spreads rumors, violates social order
- AI-generated content must be explicitly labeled
- AI content must conform to core socialist values
- Real-name authentication required for AI content providers
- Content review mechanisms must include human reviewers, not fully automated

**Deep Synthesis Provisions (2023):**
- Applies to AI that generates or manipulates images, video, audio, or text (deepfakes)
- Requires: explicit content labeling, user consent for deep synthesis, prohibitions on illegal deepfake creation
- Platforms must verify identity of deep synthesis providers
- Technical measures to identify and track AI-generated content
- Data retention for 3 years minimum

**Generative AI Rules (2023):**
- Targeted at generative AI services (text, image, audio, video) offered to the Chinese public
- Training data must be from "legal sources" and not infringe intellectual property
- Generated content must be truthful and accurate (no hallucinations that could mislead on important matters)
- Service providers must prevent users from generating illegal content
- Adherence to "core socialist values"
- Models must not be trained on data containing illegal information
- Algorithmic transparency reports required
- Real-name user registration required
- Minors must have appropriate content filters

### 4.3 AI Regulation Enforcement Framework

**Key regulatory bodies:**
- **CAC (Cyberspace Administration of China)** — Primary AI regulator, oversees algorithm registration and content moderation
- **MIIT (Ministry of Industry and Information Technology)** — AI industry standards and development
- **MOST (Ministry of Science and Technology)** — AI ethics governance
- **SAMR (State Administration for Market Regulation)** — AI market competition and consumer protection

**Penalties:**
- Warnings and orders to rectify
- Fines from ¥10,000 to ¥1,000,000
- Suspension of algorithm or service
- Revocation of operating licenses
- Criminal liability for serious violations (up to 7 years imprisonment per Criminal Law amendments)
- Personal liability for company officers

**Enforcement actions (2024–2026):**
- 2024: CAC ordered 12 major AI platforms to remove "harmful" training data
- 2025: DeepSeek required to register algorithms and submit to content review
- 2025: Multiple generative AI services suspended in Shanghai for non-compliance with labeling requirements
- 2026: New rules requiring AI-generated content watermarking at pixel/bit level for all visible content

### 4.4 Compliance Checklist

```
CHINA AI REGULATORY COMPLIANCE CHECKLIST
══════════════════════════════════════════

□ 1. Algorithm Registration
   □ Register all recommendation/search/content-gen algorithms with CAC
   □ Maintain public algorithm registration listing
   □ File algorithm updates with CAC within 10 business days of change
   □ Submit annual algorithm compliance report
   □ Retain algorithm design documentation

□ 2. Content Moderation
   □ Implement real-time content screening for illegal/harmful content
   □ Establish human-in-the-loop content review mechanism
   □ Label all AI-generated content explicitly
   □ Apply core socialist values alignment filter
   □ Implement real-name authentication for users
   □ Block generation of sensitive topics as defined by CAC
   □ Maintain content moderation logs (minimum 6 months)

□ 3. Deep Synthesis Compliance
   □ Register deep synthesis algorithm with CAC
   □ Implement mandatory content labeling for deepfakes
   □ Obtain user consent for deep synthesis manipulation
   □ Implement watermarking technology for generated media
   □ Verify identity of deep synthesis content providers
   □ Retain data for minimum 3 years
   □ Prevent creation of illegal deepfake content

□ 4. Generative AI Compliance
   □ Ensure training data from legal, non-infringing sources
   □ Implement truthfulness/accuracy requirements for generated content
   □ Prevent illegal content generation (both inputs and outputs)
   □ Establish user complaint mechanism for generated content
   □ Apply minor protection filters
   □ Implement real-name user registration for AI services
   □ Conduct algorithmic transparency disclosures
   □ Engage law enforcement content review as needed

□ 5. Data Governance
   □ Comply with Personal Information Protection Law (PIPL)
   □ Implement data localization requirements (mainland China)
   □ Obtain data subject consent for AI training data use
   □ Conduct data security impact assessments
   □ Comply with cross-border data transfer restrictions
   □ Protect trade secrets and IP in training data

□ 6. Ethics & Governance
   □ Establish internal algorithm ethics review committee
   □ Conduct regular ethics training for AI development teams
   □ File ethics review reports with MOST
   □ Implement algorithm fairness testing
   □ Document algorithmic decision logic for regulator review
```

---

## 5. NIST AI Risk Management Framework

The NIST AI Risk Management Framework (AI RMF 1.0, January 2023) provides voluntary guidance for managing AI risks. While not a regulatory mandate in itself, it has been adopted by numerous US federal agencies and is frequently referenced in regulatory compliance documentation.

### 5.1 Core Functions: Govern, Map, Measure, Manage

The framework is organized into four functions:

```
┌─────────────────────────────────────────────────────┐
│               NIST AI RMF CORE                      │
├──────────┬──────────┬──────────┬────────────────────┤
│  GOVERN  │   MAP    │  MEASURE │      MANAGE        │
│ (Govern- │ (Context │ (Assess  │ (Respond &         │
│  ing)    │  under-  │  & eval- │   mitigate)        │
│          │  stand)  │  uate)   │                    │
├──────────┼──────────┼──────────┼────────────────────┤
│  Culture │ Context  │ Metrics  │ Risk treatment      │
│  │       │  │       │  │       │    │                │
│  Policies│ Risk ID  │ Test     │ Response plans      │
│  │       │  │       │  │       │    │                │
│  Account-│ Mapping  │ Monitor  │ Communication       │
│  ability │          │          │                     │
└──────────┴──────────┴──────────┴─────────────────────┘
```

### 5.2 Categories & Subcategories

**GOVERN (Governance):**
- **GOVERN 1:** Establish AI risk management policies, processes, procedures, and practices
  - 1.1: Risk management policies are documented, implemented, and communicated
  - 1.2: Risk management is integrated into organizational processes
  - 1.3: AI risk management responsibilities are assigned
- **GOVERN 2:** Cultivate a risk management culture
  - 2.1: Workforce AI literacy and risk awareness
  - 2.2: Transparency in AI risk management
  - 2.3: Multi-stakeholder input in risk management
- **GOVERN 3:** Define accountability structures
  - 3.1: Clear roles and responsibilities for AI risks
  - 3.2: AI risk escalation and reporting mechanisms
  - 3.3: Channels for anonymous reporting
- **GOVERN 4:** Consider diversity, equity, inclusion, and accessibility
  - 4.1: DEIA integrated into AI risk management
  - 4.2: Potential for disparate impact considered
  - 4.3: Accessibility considerations in AI design
- **GOVERN 5:** Establish processes for ongoing involvement of affected communities
  - 5.1: Stakeholder identification and engagement
  - 5.2: Feedback mechanisms for AI system impacts

**MAP (Mapping Context):**
- **MAP 1:** Understand the AI system's context
  - 1.1: Intended purpose and use cases documented
  - 1.2: Operational context and deployment environment
  - 1.3: Relevant laws, regulations, and standards identified
- **MAP 2:** Identify benefits, value, and positive outcomes
  - 2.1: Benefits to individuals, groups, and society
  - 2.2: Metrics for measuring positive outcomes
- **MAP 3:** Identify risks and potential harms
  - 3.1: Individuals: privacy, safety, discrimination, autonomy
  - 3.2: Groups/communities: equity, access, representation
  - 3.3: Organizations: liability, reputational, financial
  - 3.4: Society/environment: civic engagement, sustainability
- **MAP 4:** Risk prioritization and triage
  - 4.1: Risk severity and likelihood assessment
  - 4.2: Risk prioritization criteria established

**MEASURE (Assessment & Evaluation):**
- **MEASURE 1:** Select and define metrics for AI risks
  - 1.1: Accuracy metrics (precision, recall, F1, AUC-ROC, calibration)
  - 1.2: Fairness metrics (demographic parity, equal opportunity, equalized odds)
  - 1.3: Robustness metrics (adversarial accuracy, distribution shift tolerance)
  - 1.4: Explainability metrics (completeness, comprehensibility, faithfulness)
  - 1.5: Privacy metrics (membership inference risk, differential privacy epsilon)
- **MEASURE 2:** Conduct testing and evaluation
  - 2.1: Test design covering intended uses and foreseeable misuse
  - 2.2: Evaluation under diverse conditions
  - 2.3: Quantitative and qualitative testing methods
  - 2.4: Third-party testing and independent evaluation
- **MEASURE 3:** Track and document risks
  - 3.1: Risk tracking from identification through mitigation
  - 3.2: Documentation of measurement methods and results
- **MEASURE 4:** Monitor risks over time
  - 4.1: Continuous monitoring throughout AI lifecycle
  - 4.2: Monitoring for drifts in performance, bias, and behavior
  - 4.3: Feedback loops from deployment back to development

**MANAGE (Response & Mitigation):**
- **MANAGE 1:** Implement risk treatment plans
  - 1.1: Select appropriate risk treatment options (avoid, mitigate, transfer, accept)
  - 1.2: Implement mitigation measures in design and deployment
  - 1.3: Document risk treatment decisions
- **MANAGE 2:** Communicate and document risks and responses
  - 2.1: Risk communication to stakeholders
  - 2.2: Documentation of risk decisions and rationale
- **MANAGE 3:** Monitor and adapt risk treatment
  - 3.1: Ongoing monitoring of risk treatment effectiveness
  - 3.2: Adaptation of treatment as context changes
- **MANAGE 4:** Prepare for negative impacts
  - 4.1: Incident response plans for AI systems
  - 4.2: Contingency plans for system failure
  - 4.3: Transparency reporting mechanisms

### 5.3 Playbook Implementation

The NIST AI RMF Playbook provides suggested actions for each subcategory:

**Example: MAP 1.1 (Intended Purpose)**
```
Action: Document the intended purpose of the AI system
Who: Product manager + Technical lead
When: Before development begins
Output: Purpose statement including:
  - Problem the AI system solves
  - Population it will serve
  - Deployment environment
  - Constraints and limitations
  - Out-of-scope uses
Template: [Link to organization's purpose documentation template]
```

**Example: MEASURE 2.1 (Test Design)**
```
Action: Design comprehensive test suite
Who: QA team + Technical lead + Domain expert
When: Before deployment, after each major update
Output: Test report including:
  - Test scenarios covering intended use
  - Edge case and stress tests
  - Foreseeable misuse scenarios
  - Out-of-distribution test data
  - Demographic subgroup evaluation
```

**Example: MANAGE 3.1 (Monitor Treatment)**
```
Action: Set up continuous monitoring dashboard
Who: MLOps team
When: Ongoing post-deployment
Output: Monitoring reports including:
  - Performance metrics over time
  - Drift detection alerts
  - Incident log
  - Feedback incorporation status
```

### 5.4 Compliance Checklist

```
NIST AI RMF COMPLIANCE CHECKLIST
═════════════════════════════════

□ 1. GOVERN
   □ AI risk management policy documented and approved
   □ Risk management responsibilities assigned
   □ AI risk management integrated into org processes
   □ Workforce AI literacy program established
   □ Accountability structures defined
   □ DEIA considerations integrated
   □ Stakeholder engagement processes in place
   □ Annual AI risk management review scheduled

□ 2. MAP
   □ Intended purpose documented for all AI systems
   □ Operational context and deployment environment assessed
   □ Relevant laws/regulations identified
   □ Benefits and positive outcomes identified
   □ Potential harms identified (individuals, groups, orgs, society)
   □ Risk prioritization completed
   □ Risk appetite defined and documented

□ 3. MEASURE
   □ Metrics for AI risks selected and defined
   □ Accuracy metrics established (by system)
   □ Fairness metrics established (by system)
   □ Robustness metrics established (by system)
   □ Explainability metrics established (by system)
   □ Privacy metrics established (by system)
   □ Test suite designed and executed
   □ Third-party evaluation conducted (or planned)
   □ Risks tracked and documented
   □ Continuous monitoring implemented

□ 4. MANAGE
   □ Risk treatment plans developed
   □ Mitigation measures implemented
   □ Risk decisions documented
   □ Incident response plan established
   □ Contingency plans developed
   □ Risk communication to stakeholders complete
   □ Ongoing monitoring of treatment effectiveness
   □ Transparency reports produced
   □ Adaptation processes in place
```

---

## 6. ISO 42001 — AI Management System

ISO/IEC 42001:2023 is the first international standard for AI management systems (AIMS). Published in December 2023, it provides a certifiable framework for organizations to manage AI risks and opportunities.

### 6.1 Structure & Clauses

The standard follows the ISO high-level structure (HLS) shared with ISO 9001, ISO 27001, ISO 14001, etc.:

**Clause 1: Scope**
- Specifies requirements for establishing, implementing, maintaining, and continually improving an AI management system

**Clause 2: Normative References**
- ISO/IEC 22989 (AI concepts and terminology)
- ISO/IEC 23053 (Framework for AI systems using ML)

**Clause 3: Terms and Definitions**
- AI system, AI model, AI component, AI provider, AI user, etc.

**Clause 4: Context of the Organization**
- 4.1: Understanding the organization and its context
- 4.2: Understanding needs and expectations of interested parties
- 4.3: Determining the scope of the AIMS
- 4.4: AI management system and its processes

**Clause 5: Leadership**
- 5.1: Leadership and commitment
- 5.2: AI policy (documented, communicated, available to interested parties)
- 5.3: Roles, responsibilities, and authorities

**Clause 6: Planning**
- 6.1: Actions to address risks and opportunities
- 6.2: AI objectives and planning to achieve them
- 6.3: Planning of changes

**Clause 7: Support**
- 7.1: Resources
- 7.2: Competence
- 7.3: Awareness
- 7.4: Communication
- 7.5: Documented information

**Clause 8: Operation**
- 8.1: Operational planning and control
- 8.2: AI risk assessment
- 8.3: AI risk treatment
- 8.4: AI system impact assessment
- 8.5: Change management

**Clause 9: Performance Evaluation**
- 9.1: Monitoring, measurement, analysis, and evaluation
- 9.2: Internal audit
- 9.3: Management review

**Clause 10: Improvement**
- 10.1: Nonconformity and corrective action
- 10.2: Continual improvement

### 6.2 Annex A Controls

ISO 42001 Annex A provides 38 AI-specific controls organized into 11 categories:

| Category | Control | Description |
|---|---|---|
| **A.1 — AI Policies** | A.1.1 | Policy for AI system development and use |
| | A.1.2 | Objectives for AI system development and use |
| **A.2 — Internal Context** | A.2.1 | Organizational roles and responsibilities |
| | A.2.2 | AI system inventory |
| | A.2.3 | AI system classification |
| **A.3 — Resources** | A.3.1 | Resource provision for AI systems |
| | A.3.2 | Competence of persons working on AI |
| **A.4 — Impact Assessment** | A.4.1 | AI system impact assessment methodology |
| | A.4.2 | Conducting AI system impact assessments |
| **A.5 — Risk Assessment** | A.5.1 | AI risk assessment methodology |
| | A.5.2 | AI risk assessment process |
| **A.6 — Risk Treatment** | A.6.1 | AI risk treatment plan |
| | A.6.2 | AI risk treatment implementation |
| **A.7 — AI System Lifecycle** | A.7.1 | AI system planning and design |
| | A.7.2 | AI system data acquisition and preparation |
| | A.7.3 | AI system development |
| | A.7.4 | AI system validation and testing |
| | A.7.5 | AI system deployment |
| | A.7.6 | AI system operation and monitoring |
| | A.7.7 | AI system revalidation |
| | A.7.8 | AI system retirement |
| **A.8 — Data** | A.8.1 | Data management for AI |
| | A.8.2 | Data quality for AI |
| | A.8.3 | Data provenance for AI |
| **A.9 — Stakeholders** | A.9.1 | Identification and engagement of interested parties |
| | A.9.2 | Communication with interested parties |
| **A.10 — AI System Use** | A.10.1 | Information and guidance for AI users |
| | A.10.2 | Human oversight of AI systems |
| | A.10.3 | AI system reporting |
| **A.11 — Continual Improvement** | A.11.1 | Monitoring, measurement, analysis, and evaluation |
| | A.11.2 | Internal audit of AI systems |
| | A.11.3 | Nonconformities and corrective actions |
| | A.11.4 | AI system incident management |
| | A.11.5 | AI system lessons learned |
| | A.11.6 | Continual improvement of AI systems |

### 6.3 Certification Process

**Phased approach to ISO 42001 certification:**

```
Phase 1: Gap Analysis (2–4 weeks)
  └── Assess current practices against ISO 42001 requirements
  └── Identify gaps in: policy, process, documentation, controls
  └── Develop implementation roadmap

Phase 2: Implementation (3–6 months)
  └── Develop AI policy and objectives
  └── Establish AI system inventory and classification
  └── Implement impact and risk assessment processes
  └── Document AI lifecycle procedures
  └── Train personnel on AI management system
  └── Implement selected Annex A controls

Phase 3: Internal Audit & Review (1–2 months)
  └── Conduct internal audit of AIMS
  └── Management review of AIMS
  └── Address nonconformities found during internal audit

Phase 4: Certification Audit (1–2 months)
  └── Stage 1: Documentation review (on-site or remote)
  └── Stage 2: Implementation verification (on-site)
  └── Certification decision

Phase 5: Surveillance (annual)
  └── Annual surveillance audits
  └── Recertification every 3 years
```

**Integration with other ISO standards:**
- **ISO 42001 + ISO 27001:** Common approach to AI security and risk management
- **ISO 42001 + ISO 9001:** AI quality management integration
- **ISO 42001 + ISO 27701:** AI privacy management
- **ISO 42001 + ISO 31000:** AI risk management alignment

### 6.4 Compliance Checklist

```
ISO 42001 COMPLIANCE CHECKLIST
══════════════════════════════

□ 1. Context & Scope (Clause 4)
   □ Organization's internal and external context documented
   □ Interested parties identified
   □ Scope of AIMS defined
   □ AIMS established as documented process

□ 2. Leadership (Clause 5)
   □ AI policy approved by top management
   □ AI policy communicated to all relevant personnel
   □ Roles and responsibilities defined
   □ AI governance committee established

□ 3. Planning (Clause 6)
   □ Risks and opportunities identified
   □ AI objectives defined and measurable
   □ Plans to achieve objectives documented
   □ Change management process established

□ 4. Support (Clause 7)
   □ Resources allocated for AIMS
   □ Competence requirements defined
   □ Training program implemented
   □ Awareness program conducted
   □ Communication channels established
   □ Documented information maintained

□ 5. Operation (Clause 8)
   □ Operational planning and control established
   □ AI risk assessment methodology defined
   □ AI risk assessments conducted
   □ AI risk treatment plans implemented
   □ AI system impact assessments conducted
   □ Change management process operational

□ 6. Performance Evaluation (Clause 9)
   □ Monitoring and measurement processes established
   □ Internal audit program implemented
   □ Management review conducted (at planned intervals)
   □ Nonconformities tracked

□ 7. Improvement (Clause 10)
   □ Nonconformity process established
   □ Corrective actions implemented
   □ Continual improvement process active

□ 8. Annex A Controls
   □ AI policies documented (A.1)
   □ Internal context defined (A.2)
   □ Resources allocated (A.3)
   □ Impact assessment methodology implemented (A.4)
   □ Risk assessment process active (A.5)
   □ Risk treatment plans documented (A.6)
   □ AI system lifecycle procedures documented (A.7)
   □ Data management practices established (A.8)
   □ Stakeholder engagement active (A.9)
   □ AI system use guidance provided (A.10)
   □ Continual improvement processes active (A.11)
```

---

## 7. Sectoral Regulations

### 7.1 FDA — Medical AI / SaMD

The FDA regulates AI-based Software as a Medical Device (SaMD) through an evolving framework that addresses the unique challenges of AI/ML-based medical devices.

**Regulatory pathway:**
- **510(k) clearance** — Most AI-based SaMD enters via substantial equivalence to predicate devices
- **De Novo classification** — Novel AI devices without predicates
- **PMA (Premarket Approval)** — Highest-risk Class III AI devices
- **Breakthrough Devices Program** — Expedited pathway for AI devices addressing unmet needs

**Total Product Lifecycle (TPLC) framework:**
```
Pre-market                          Post-market
┌─────────────┐                   ┌─────────────┐
│ 510(k)/De   │                   │ Real-world   │
│ Novo/PMA    │                   │ performance  │
│ application  │───→ Approval ───→│ monitoring   │
└─────────────┘                   └─────────────┘
       │                               │
       │                               ▼
       │                     ┌─────────────────┐
       │                     │ Adverse event    │
       └────────────────────→│ reporting (MDR)  │
                             │ + Recall process │
                             └─────────────────┘
```

**AI/ML-Specific Guidance (2024–2026):**
- **AI/ML SaMD Action Plan** (2021, updated 2024) — Framework for premarket review of AI/ML devices
- **Predetermined Change Control Plans (PCCP)** — Allows manufacturers to update AI models without new 510(k) if changes are pre-specified and controls documented
- **Good Machine Learning Practices (GMLP)** — Expectations for data management, feature engineering, model validation
- **Algorithmic Transparency** — Disclosure of model architecture, training data, performance by subgroups
- **Human-in-the-loop** — Requirements for physician oversight of AI-assisted diagnosis

**Key requirements:**
- Clinical validation in intended population
- Performance assessment across demographic subgroups
- Robustness to distribution shift in clinical settings
- Transparency in AI-assisted decision making
- Continuous learning device safeguards (if adaptive AI)
- Real-world evidence collection plan

**Notable FDA AI/ML decisions (2024–2026):**
- 2024: First PCCP-approved AI device (chest X-ray triage)
- 2025: 100+ AI/ML devices cleared under new framework
- 2025: First AI-based autonomous diagnostic system (diabetic retinopathy screening)
- 2026: FDA guidance on LLM-based clinical decision support

### 7.2 SEC — Algorithmic Trading & AI in Finance

The SEC has increased scrutiny of AI use in financial markets, particularly regarding market manipulation, disclosure, and fiduciary duty.

**Regulatory focus areas:**
1. **Algorithmic trading systems:**
   - Regulation SCI — Systems compliance and integrity requirements
   - Regulation ATS — Alternative trading system requirements
   - Rule 15c3-5 — Risk management controls for market access
   - Testing and certification of trading algorithms
   - Kill-switch requirements for runaway algorithms

2. **AI-driven investment advice:**
   - Investment Advisers Act of 1940 — Fiduciary duty applies to AI-generated advice
   - SEC Marketing Rule — AI endorsements and testimonials
   - ESG AI claims — Substantiation of AI-driven ESG ratings
   - Disclosure of AI use in investment decisions

3. **Market integrity:**
   - SEC Rule 10b-5 — Prohibition of market manipulation (including AI-driven schemes)
   - Regulation FD — Fair disclosure obligations with AI-generated communications
   - AI wash trading detection
   - Account for AI-generated trading volumes

4. **Conflicts of interest:**
   - AI models that prioritize certain investments
   - Best execution obligations when AI routes trades
   - AI model bias favoring affiliated products

**SEC AI Enforcement Actions (2024–2026):**
- 2024: SEC fined investment adviser $450K for "AI washing" (misclaiming AI capabilities)
- 2025: SEC charged broker-dealer for unregistered algorithmic trading system
- 2025: SEC proposed rule on AI conflicts of interest in digital engagement practices
- 2026: SEC enforcement against robo-advisor for insufficient AI model governance

### 7.3 FCC — AI Robocalls & Communications

The FCC has taken aggressive action against AI-generated communications, particularly robocalls and robotexts.

**FCC AI Regulation:**
1. **Telephone Consumer Protection Act (TCPA):**
   - AI-generated voice calls require prior express consent (same as robocalls)
   - AI voice cloning is considered a "robocall" under TCPA
   - Consent cannot be obtained by AI-generated voice (must be human-to-human)
   - Text messages generated by AI require prior consent

2. **Truth in Caller ID Act:**
   - Prohibition on using AI to spoof or manipulate caller ID information
   - AI-generated voices that misrepresent identity — possible violations
   - Enforcement through STIR/SHAKEN caller ID authentication

3. **AI Content Labeling:**
   - FCC proposed rules requiring AI-generated political ads on TV/radio to be labeled
   - AI-generated audio in broadcast must be disclosed
   - Watermarking requirements for AI-generated voice content

4. **Accessibility:**
   - AI relay services for hearing/speech impaired must meet accuracy standards
   - AI captioning quality requirements under 21st Century Communications Act

**FCC Enforcement Actions:**
- 2024: $5M fine for AI-generated robocall mimicking political candidate voice
- 2025: Cease-and-desist orders against 6 AI voice cloning service providers
- 2025: Declaratory ruling establishing AI voice cloning as "artificial" under TCPA
- 2026: Proposed fines for AI-generated text spam operations

### 7.4 EU Sectoral Intersections

**GDPR + AI Act:**
- High-risk AI systems processing personal data must conduct Data Protection Impact Assessments (DPIA)
- Right to explanation of automated decisions (Art. 22 GDPR)
- Privacy by design required for AI systems
- Data minimization principles apply to AI training data

**MDR (Medical Device Regulation) + AI Act:**
- AI as medical device must comply with both MDR and AI Act
- Harmonized standards being developed for AI medical devices
- Notified bodies designated for both regulations
- Clinical evaluation requirements under MDR add to AI Act obligations

**DSA (Digital Services Act) + AI Act:**
- Very Large Online Platforms (VLOPs) must assess systemic risks amplified by AI recommendation algorithms
- AI content moderation systems must respect freedom of expression
- Algorithmic transparency reports required for recommender systems
- Annual independent audits of AI systems on large platforms

### 7.5 Sectoral Compliance Checklists

```
FDA MEDICAL AI COMPLIANCE CHECKLIST
════════════════════════════════════

□ 1. Determine regulatory classification (Class II / Class III)
□ 2. Select appropriate submission path (510(k) / De Novo / PMA)
□ 3. Prepare clinical validation data in intended population
□ 4. Conduct subgroup performance analysis
□ 5. Document training data provenance and quality
□ 6. Implement predetermined change control plan (PCCP) if applicable
□ 7. Establish real-world performance monitoring plan
□ 8. Implement MDR (Medical Device Reporting) process
□ 9. Maintain design history file per 21 CFR 820
□ 10. Conduct cybersecurity risk assessment per FDA guidance
□ 11. Implement human oversight measures for clinical use
□ 12. Label AI device with intended use, limitations, and performance


SEC ALGORITHMIC TRADING COMPLIANCE CHECKLIST
═════════════════════════════════════════════

□ 1. Register algorithmic trading system if applicable
□ 2. Implement risk management controls for market access
□ 3. Document algorithm design and logic
□ 4. Test algorithm in simulated and live environments
□ 5. Implement kill-switch mechanism
□ 6. Monitor for market manipulation indicators
□ 7. Maintain algorithm change log
□ 8. Prepare compliance documentation for SEC examiners
□ 9. Disclose AI use in investment advisory services
□ 10. Implement best execution obligations for AI-directed trades
□ 11. Avoid "AI washing" in marketing


FCC AI COMMUNICATIONS COMPLIANCE CHECKLIST
════════════════════════════════════════════

□ 1. Obtain prior express consent for AI-generated calls/texts
□ 2. Ensure AI voice communications identify the caller
□ 3. Do not spoof caller ID using AI
□ 4. Label AI-generated political ads
□ 5. Label AI-generated content in broadcasts
□ 6. Implement STIR/SHAKEN authentication for AI calls
□ 7. Ensure AI relay services meet accuracy standards
□ 8. Maintain call/text opt-out records
```

---

## 8. Model Auditing Requirements

Model auditing has emerged as a core compliance requirement across multiple regulatory regimes. An AI model audit is a systematic, independent examination of an AI system to assess its compliance with regulatory requirements, ethical standards, and organizational policies.

### 8.1 Internal vs External Audits

| Aspect | Internal Audit | External / Third-Party Audit |
|---|---|---|
| **Conducted by** | Internal compliance or AI risk team | Independent external auditor, accredited body |
| **Frequency** | Quarterly or continuous | Annually (or before deployment) |
| **Scope** | Broad: all AI systems, processes | Targeted: high-risk systems, regulatory compliance |
| **Depth** | Can be deep but dependent on team expertise | Typically rigorous with formal methodologies |
| **Cost** | Lower (staff time) | Higher ($50K–$500K+ per audit) |
| **Outcome** | Internal report, remediation plan | Formal audit opinion, compliance certification |
| **Regulatory acceptance** | May not satisfy regulatory requirements | Often required for high-risk systems |

### 8.2 Audit Methodology

A comprehensive AI model audit follows this methodology:

```
Phase 1: Scoping & Planning
├── Define audit objectives (regulatory compliance, fairness, safety, etc.)
├── Identify AI systems in scope
├── Gather documentation (model cards, data sheets, system cards)
├── Select audit team (internal + external as needed)
└── Develop audit plan with timeline and milestones

Phase 2: Pre-Audit Documentation Review
├── Technical documentation (system architecture, design choices)
├── Training data documentation (sources, preprocessing, labeling)
├── Model development records (training logs, hyperparameters)
├── Risk assessment and impact assessment reports
├── Human oversight and governance documentation
└── Prior audit findings and remediation status

Phase 3: Technical Testing
├── Functional testing (accuracy, precision, recall, F1)
├── Fairness testing (demographic parity, equal opportunity, equalized odds)
├── Robustness testing (adversarial examples, distribution shift)
├── Explainability testing (SHAP, LIME, feature importance)
├── Privacy testing (membership inference, differential privacy)
├── Security testing (prompt injection, jailbreak, data poisoning)
└── Bias testing (intersectional, subgroup, temporal)

Phase 4: Process Assessment
├── Development lifecycle assessment
├── Data governance evaluation
├── Risk management process review
├── Human oversight adequacy assessment
├── Transparency and documentation quality
├── Change management process review
└── Incident response capability assessment

Phase 5: Reporting & Remediation
├── Draft audit findings report
├── Prioritize findings by risk severity
├── Develop remediation recommendations
├── Present to management and/or board
├── Track remediation to closure
└── Prepare for next audit cycle
```

### 8.3 Automated Auditing Tools

| Tool | Capabilities | Regulatory Alignment |
|---|---|---|
| **IBM AI Fairness 360** | Bias detection, fairness metrics, bias mitigation algorithms | EU AI Act bias testing, NIST AI RMF |
| **Google What-If Tool** | Interactive model exploration, fairness analysis | NIST AI RMF MEASURE |
| **Microsoft Fairlearn** | Fairness metrics, disparity analysis | US Executive Order fairness requirements |
| **AIF360 (IBM)** | Comprehensive bias testing suite | Sectoral anti-discrimination laws |
| **Captum (PyTorch)** | Model interpretability (Integrated Gradients, etc.) | EU AI Act explainability, GDPR right to explanation |
| **SHAP/LIME** | Feature attribution, local explanations | XAI requirements across regulations |
| **Amazon SageMaker Clarify** | Bias detection, feature importance, SHAP | NIST AI RMF, US EO compliance |
| **MLflow Model Registry** | Model versioning, lineage tracking | ISO 42001 lifecycle management |
| **DVC + CML** | Data versioning, experiment tracking | Data provenance documentation |
| **ELSA (Ethical and Legal SA)** | Legal compliance checking for AI systems | Multi-regulatory compliance |

### 8.4 Red-Teaming Mandates

Red-teaming has transitioned from a best practice to a regulatory requirement:

**Mandated by:**
- **US Executive Order 14110** — Dual-use foundation model developers must share red-teaming results
- **EU AI Act (Art. 55)** — Systemic-risk GPAI must undergo adversarial testing
- **China Deep Synthesis Provisions** — Required security assessments
- **NIST AI RMF** — Red-teaming as part of MEASURE function
- **UK AI Safety Institute** — Red-teaming evaluation protocol for frontier models

**Red-teaming methodology:**

```
┌───────────────────────────────────────────────────────┐
│              RED-TEAMING FRAMEWORK                    │
├───────────────────────────────────────────────────────┤
│ Phase 1: Threat Modeling                              │
│ ├── Identify attack surface (inputs, outputs, APIs)   │
│ ├── Map threat vectors per MITRE ATLAS                │
│ └── Prioritize attack scenarios by risk               │
├───────────────────────────────────────────────────────┤
│ Phase 2: Attack Execution                             │
│ ├── Prompt injection (direct, indirect, multi-turn)   │
│ ├── Jailbreak attempts (role-play, encoding, etc.)    │
│ ├── Data extraction (training data exfiltration)      │
│ ├── Bias exploitation (stereotype reinforcement)      │
│ ├── Adversarial inputs (in-distribution, OOD)         │
│ └── Output manipulation (how to change system output) │
├───────────────────────────────────────────────────────┤
│ Phase 3: Documentation & Reporting                    │
│ ├── Adversarial test cases and outcomes               │
│ ├── Vulnerability severity ratings                     │
│ ├── Successful attack reproduction steps              │
│ ├── Mitigation recommendations                        │
│ └── Residual risk assessment                          │
├───────────────────────────────────────────────────────┤
│ Phase 4: Remediation & Re-testing                     │
│ ├── Implement mitigations                             │
│ ├── Re-run adversarial test cases                     │
│ ├── Verify vulnerability closure                      │
│ └── Update threat model                               │
└───────────────────────────────────────────────────────┘
```

---

## 9. Bias Testing Methodologies

Bias testing is a core compliance requirement across the EU AI Act, US Executive Orders, NIST AI RMF, and sectoral regulations. A comprehensive bias testing program covers multiple dimensions.

### 9.1 Statistical Parity Metrics

**Group fairness metrics:**

| Metric | Definition | Formula | Interpretation |
|---|---|---|---|
| **Demographic Parity** | Probability of positive outcome is equal across groups | P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1) | Difference should be < 0.1 |
| **Equal Opportunity** | Equal true positive rates across groups | P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1) | Indicates predictive equality for qualified individuals |
| **Equalized Odds** | Equal TPR and FPR across groups | TPR_A0 = TPR_A1 AND FPR_A0 = FPR_A1 | Strongest fairness condition |
| **Predictive Parity** | Equal positive predictive value across groups | P(Y=1 | Ŷ=1, A=0) = P(Y=1 | Ŷ=1, A=1) | Relevant for deployment decisions |
| **Treatment Equality** | Equal ratio of false positives to false negatives across groups | FP_A0/FN_A0 = FP_A1/FN_A1 | Ensures errors affect groups equally |

**Individual fairness metrics:**
- **Counterfactual fairness** — Prediction same if protected attribute changed
- **Fairness through awareness** — Similar individuals receive similar predictions
- **Fairness through unawareness** — Protected attributes not used as features (note: insufficient alone due to proxy features)

### 9.2 Disparate Impact Analysis

**Disparate Impact Ratio (DIR):**
```
DIR = P(positive outcome | protected group) / P(positive outcome | reference group)

Thresholds:
- DIR < 0.80: Strong evidence of adverse impact (Four-Fifths Rule violation)
- 0.80 ≤ DIR < 0.90: Moderate concern, requires investigation
- DIR ≥ 0.90: No significant disparate impact
```

**Multivariate analysis:**
- Logistic regression with protected attribute coefficients
- Interaction effects between protected and unprotected attributes
- Matched-pair analysis comparing similar individuals across groups

**Multi-level modeling:**
- Hierarchical models accounting for group-level and individual-level variance
- Bayesian approaches for small subgroup analysis
- Causal inference methods for estimating discrimination effects

### 9.3 Intersectional Bias Testing

Intersectional analysis tests bias across combinations of protected attributes (e.g., race × gender × age):

**Methodology:**
```
Step 1: Define intersectional groups
  Example: (Asian, Female, >50), (Black, Male, <30), etc.
  Minimum cell size: n ≥ 30 for statistical validity

Step 2: Compute fairness metrics per intersectional group
  Demographic parity, equal opportunity, equalized odds

Step 3: Identify intersectional disparities
  Compare each intersectional group to overall average
  Flag groups with metrics exceeding threshold (e.g., DIR < 0.80)

Step 4: Statistical significance testing
  Chi-square test for categorical outcomes
  Bootstrap confidence intervals for metric comparisons

Step 5: Qualitative analysis
  Review samples from flagged intersectional groups
  Investigate model behavior on specific instances
  Document findings for regulatory reporting
```

**Tools for intersectional testing:**
- **AIF360** — Multi-attribute bias measurement
- **FairML** — Orthogonalization for intersectional analysis
- **Intersectional Fairness Toolkit** — Automated intersectional testing
- **Python (pandas + scipy)** — Custom intersectional analysis pipelines

### 9.4 Dataset Bias Audits

Pre-deployment bias auditing must include training data analysis:

**Dataset bias dimensions:**

| Bias Type | Description | Detection Method |
|---|---|---|
| **Representation bias** | Under/over-representation of groups | Demographic distribution analysis vs. population |
| **Measurement bias** | Systematic errors in labeling | Label quality audit, inter-annotator agreement |
| **Aggregation bias** | Inappropriate group-level assumptions | Subgroup performance analysis |
| **Historical bias** | Existing societal biases reflected in data | Temporal analysis, expert review |
| **Label bias** | Annotation stereotypes | Annotator demographic analysis, label audit |
| **Selection bias** | Non-representative sampling | Sampling methodology review |

**Dataset audit process:**
```
1. Data Profiling
   ├── Feature distributions by group
   ├── Missing data patterns by group
   └── Label distribution by group

2. Quality Assessment
   ├── Label accuracy audit (sample of records)
   ├── Inter-annotator agreement (Cohen's κ, Fleiss' κ)
   └── Data source reliability evaluation

3. Bias Quantification
   ├── Skew in demographic representation
   ├── Label disparity analysis
   └── Proxy correlation analysis (correlated protected attributes)

4. Remediation
   ├── Re-balancing (undersampling, oversampling, SMOTE)
   ├── De-biasing (re-weighting, data augmentation)
   └── Documentation of known biases and mitigations
```

### 9.5 Post-Deployment Monitoring

Continuous bias monitoring is required for high-risk AI systems:

**Monitoring frequency by risk level:**
- **High-risk systems:** Continuous monitoring, weekly automated reports, monthly human review
- **Medium-risk systems:** Monthly automated monitoring, quarterly review
- **Low-risk systems:** Quarterly monitoring, annual review

**Key monitoring metrics:**
- Model accuracy by demographic group over time
- Prediction distribution shifts by group
- User complaint / appeal rates by group
- Feedback loops amplifying disparities
- New bias introduction after model updates

**Alerting thresholds:**
```
Critical (immediate action required):
  - DIR < 0.70 for any protected group
  - Accuracy drop > 15% for any subgroup
  - Complaint rate > 5x baseline for any group

Warning (investigation within 1 week):
  - 0.70 ≤ DIR < 0.80
  - Accuracy drop 5-15% for any subgroup
  - Complaint rate 2-5x baseline

Monitor (review at next cycle):
  - 0.80 ≤ DIR < 0.90
  - Accuracy drop < 5%
  - Slight increase in complaint rate
```

---

## 10. Explainability Requirements & XAI Techniques

Explainable AI (XAI) is a regulatory requirement across multiple frameworks. The EU AI Act mandates that high-risk AI systems provide "meaningful information" about their logic and operation (Art. 13). The GDPR provides a "right to explanation" for automated decisions (Art. 22). NIST AI RMF incorporates explainability under the MEASURE function.

### 10.1 Intrinsic vs Post-Hoc Methods

| Aspect | Intrinsic Explainability | Post-Hoc Explainability |
|---|---|---|
| **Approach** | Models are inherently interpretable by design | Explanations generated after model training |
| **Model types** | Linear regression, decision trees, rule-based, GLM, logistic regression | Neural networks, gradient boosting, random forests, transformers |
| **Performance** | Often lower predictive performance | High performance + separate explanation |
| **Trust** | High (interpretable structure) | Medium (explanation may not be faithful) |
| **Regulatory acceptance** | Preferred where possible | Acceptable with fidelity guarantees |
| **Trade-off** | Simplicity vs. accuracy | Accuracy vs. interpretability |

### 10.2 SHAP & SHAPley Values

SHAP (SHapley Additive exPlanations) is the dominant XAI method as of June 2026.

**How it works:**
- Based on cooperative game theory — each feature is a "player" contributing to the "payout" (model output)
- SHAPley values fairly distribute the prediction among features
- Additive feature attribution: model output = base value + sum of SHAP values

**Implementation:**
```python
import shap

# Train a model (example: XGBoost)
model = xgboost.train(params, dtrain)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test)

# Local explanation for a single prediction
shap.force_plot(explainer.expected_value, 
                shap_values[0, :], 
                X_test.iloc[0, :])

# Interaction effects
shap.dependence_plot("feature_A", shap_values, X_test)
```

**Regulatory acceptability:**
- **Accepted for:** Feature attribution, global and local explanations
- **Limitations:** Assumes feature independence, computationally expensive for complex models
- **Documentation required:** SHAP computation method, consistency with model behavior

### 10.3 LIME (Local Interpretable Model-Agnostic Explanations)

LIME generates local explanations by fitting interpretable surrogate models around individual predictions.

**How it works:**
1. Perturb input sample to create synthetic neighborhood
2. Get predictions from the complex model for perturbed samples
3. Fit a simple interpretable model (e.g., linear regression, decision tree) weighted by proximity to original input
4. The simple model weights represent feature importance for that prediction

**Implementation:**
```python
import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=['negative', 'positive'],
    mode='classification'
)

# Explain a single prediction
exp = explainer.explain_instance(
    X_test[0],
    model.predict_proba,
    num_features=5
)

# Get explanation as list
exp_list = exp.as_list()
print(exp_list)

# Visualize
exp.show_in_notebook(show_table=True)
```

**Regulatory considerations:**
- **Stability:** LIME explanations can vary significantly with different perturbation seeds — regulators require stability analysis
- **Fidelity:** Must verify that local surrogate model faithfully approximates the complex model in the neighborhood
- **Acceptability:** Preferred for tabular data; less reliable for high-dimensional data (images, text)
- **Documentation:** Must include perturbation method, kernel width, number of samples

### 10.4 Integrated Gradients

Integrated Gradients (Sundararajan et al., 2017) provides feature attribution by integrating gradients along a path from a baseline to the input.

**How it works:**
- Baseline: a neutral input (all zeros, average input, or blurred image for vision models)
- Compute gradients of model output w.r.t. input features along a linear path from baseline to input
- Integrated gradients = path integral of gradients
- Satisfies two axioms: sensitivity and implementation invariance

**Implementation (Captum):**
```python
from captum.attr import IntegratedGradients

# Initialize IG with model
ig = IntegratedGradients(model)

# Compute attributions
attributions, delta = ig.attribute(
    inputs=input_tensor,
    baselines=baseline_tensor,
    target=0,
    n_steps=50,
    return_convergence_delta=True
)

# Visualize attributions (for image models)
plt.imshow(attributions.squeeze().cpu().numpy())
```

**Regulatory considerations:**
- **Preferred for:** Deep learning models (vision, NLP)
- **Advantage:** Axiomatic guarantees (sensitivity, implementation invariance)
- **Limitation:** Requires good baseline choice; can be noisy
- **Acceptability:** Generally accepted, especially with convergence delta reporting

### 10.5 Attention-Based Explanations

For transformer models, attention weights are often used as explanations:

**Types of attention explanations:**
- **Raw attention weights** — Direct visualization of attention patterns
- **Attention rollout** — Aggregating attention across layers
- **Attention flow** — Considering residual connections and attention composition
- **Attention x gradient** — Modulating attention by gradient signal

**Limitations for regulatory use:**
- Attention weights may not represent feature importance faithfully (Jain & Wallace, 2019)
- Attention patterns can be adversarial manipulated
- Must be validated against other XAI methods
- Recent regulatory guidance: attention weights alone are insufficient for compliance; must be supplemented with attribution methods

### 10.6 Concept Activation Vectors (CAVs)

TCAV (Testing with CAVs) provides high-level concept explanations:

**How it works:**
1. Define a concept (e.g., "stripes" in images, "fairness" in text)
2. Collect positive and negative examples of the concept
3. Train a linear classifier (CAV) to distinguish concept vs. non-concept in a hidden layer
4. Measure how sensitive model predictions are to concept direction
5. Output: TCAV score — what fraction of inputs for a class are positively influenced by the concept

**Regulatory applications:**
- Explain model reliance on sensitive concepts (race, gender)
- Audit for unintended concept correlations
- Document model behavior on ethically relevant dimensions
- Provide intuitive explanations to regulators

### 10.7 Regulatory Acceptability of XAI Methods

| XAI Method | EU AI Act | GDPR Art. 22 | US EO | NIST AI RMF | Notes |
|---|---|---|---|---|---|
| **SHAP** | ✓ Accepted | ✓ Accepted | ✓ | ✓ | Preferred overall |
| **LIME** | ✓ With stability docs | ✓ With fidelity check | ✓ | ✓ | For tabular data |
| **Integrated Gradients** | ✓ For deep learning | ✓ | ✓ | ✓ | With convergence delta |
| **Attention weights** | ⚠ Supplementary only | ⚠ Supplementary | ⚠ | ⚠ | Not sufficient alone |
| **Decision trees (intrinsic)** | ✓ Preferred | ✓ Preferred | ✓ | ✓ | When performance adequate |
| **Linear models (intrinsic)** | ✓ Preferred | ✓ Preferred | ✓ | ✓ | Simplest path |
| **Counterfactual explanations** | ✓ Emerging | ✓ Strong alignment | ⚠ | ✓ | Growing regulatory interest |
| **GradCAM** | ✓ For vision | ⚠ | ✓ | ✓ | Image-specific |

---

## 11. Documentation Standards

Regulatory compliance requires comprehensive documentation throughout the AI lifecycle. Multiple documentation standards have emerged, and the EU AI Act prescribes specific documentation requirements.

### 11.1 Model Cards (Google/Mitchell et al.)

Introduced by Mitchell et al. (2019), model cards are standardized documentation for ML models.

**Standard sections:**

```
┌────────────────────────────────────────────────────┐
│               MODEL CARD                           │
├────────────────────────────────────────────────────┤
│ Model ID: [Unique identifier]                      │
│ Version: [Semantic version]                        │
│ Date: [Release date]                               │
│ Organization: [Deploying entity]                   │
├────────────────────────────────────────────────────┤
│ MODEL DETAILS                                      │
│ ├── Model architecture: [e.g., Transformer-8L,    │
│ │    ResNet-50, XGBoost]                           │
│ ├── Input: [Format, dimensions, preprocessing]     │
│ ├── Output: [Format, post-processing]              │
│ ├── Framework: [PyTorch, TensorFlow, sklearn]      │
│ └── Parameters: [Total parameter count]            │
├────────────────────────────────────────────────────┤
│ INTENDED USE                                       │
│ ├── Primary use cases: [List]                      │
│ ├── Out-of-scope uses: [List]                      │
│ └── Target population: [Demographics, geography]   │
├────────────────────────────────────────────────────┤
│ TRAINING DATA                                      │
│ ├── Dataset name and version                       │
│ ├── Size: [# samples]                              │
│ ├── Distribution: [Key characteristics]            │
│ └── Labeling process: [Method, annotator details]  │
├────────────────────────────────────────────────────┤
│ PERFORMANCE                                        │
│ ├── Overall metrics: [accuracy, F1, AUC, etc.]     │
│ ├── Subgroup breakdown: [By demographic, region]   │
│ ├── Fairness metrics: [DIR, equal opportunity]     │
│ └── Test conditions: [Hardware, framework]         │
├────────────────────────────────────────────────────┤
│ LIMITATIONS                                        │
│ ├── Known failure modes                            │
│ ├── Distribution shift sensitivity                 │
│ └── Edge case performance                          │
├────────────────────────────────────────────────────┤
│ ETHICAL CONSIDERATIONS                             │
│ ├── Bias analysis results                          │
│ ├── Privacy impacts                                │
│ ├── Red-teaming summary                            │
│ └── Mitigation measures                            │
├────────────────────────────────────────────────────┤
│ MAINTENANCE                                        │
│ ├── Version history                                │
│ ├── Update frequency                               │
│ ├── Monitoring plan                                │
│ └── Contact information                             │
└────────────────────────────────────────────────────┘
```

### 11.2 Data Sheets (Gebru et al.)

Data sheets (Gebru et al., 2018) document datasets used for model training.

**Key sections:**
- **Motivation:** Why was the dataset created? What tasks does it support?
- **Composition:** What are the instances? Number of features? Label distribution?
- **Collection process:** How was data acquired? Timeframe? Sampling methodology?
- **Preprocessing:** Cleaning, normalization, feature engineering applied
- **Uses:** Intended uses, out-of-scope uses, limitations
- **Distribution:** How is dataset distributed? Access restrictions?
- **Maintenance:** Who maintains? Versioning policy? Deprecation process?

**EU AI Act-specific data documentation (Annex IV):**
- Training data provenance (sources, collection methods, dates)
- Data selection criteria and rationale
- Labeling procedures and quality assurance
- Data preparation pipelines (cleaning, augmentation, normalization)
- Bias assessment of training data
- Data quality metrics (completeness, accuracy, consistency, timeliness)
- Personal data processing justifications
- Data retention and deletion policies

### 11.3 System Cards (OpenAI)

System cards provide comprehensive documentation of an AI system's capabilities, limitations, and safety evaluations.

**OpenAI system card structure:**
- **System overview:** Purpose, architecture, training regime
- **Intended deployment:** Use cases, deployment scope, user base
- **Safety evaluation:**
  - Bias and fairness evaluation
  - Red-teaming results
  - Adversarial robustness testing
  - Misuse potential assessment
- **Limitations:**
  - Known failure modes
  - Uncertainty quantification
  - Calibration analysis
- **Usage guidelines:**
  - Developer guidance
  - User-facing disclosures
  - Safety best practices
- **Monitoring:** Metrics tracked, thresholds, incident response
- **Version history:** Changes between versions, safety improvements

### 11.4 AI Facts Sheets (IBM)

IBM's AI Fact Sheets provide a structured documentation framework aligned with compliance requirements:

**Fact Sheet components:**
- **Model facts:** Architecture, training algorithm, hardware requirements
- **Performance facts:** Accuracy, latency, throughput, energy consumption
- **Data facts:** Training/validation/test data, data provenance, PII handling
- **Governance facts:** Approvals, review dates, compliance status
- **Lifecycle facts:** Training date, deployment date, update history, retirement

### 11.5 EU AI Act Technical Documentation Template

Annex IV of the EU AI Act specifies the required content for high-risk AI system technical documentation:

**Required sections:**
1. **General description:**
   - Intended purpose, name, and version
   - How AI system interacts with hardware/software
   - AI system input/output specifications
   - Intended user group and affected persons

2. **Detailed system design:**
   - Development methodology and tools
   - Design specifications and system architecture
   - Training methodology and algorithms used
   - Data requirements and data preparation methods
   - Evaluation methodologies and results
   - Performance metrics and thresholds

3. **Risk management documentation:**
   - Risk identification and analysis
   - Risk evaluation and prioritization
   - Risk treatment measures implemented
   - Residual risk assessment
   - Testing results (including edge cases and foreseeable misuse)

4. **Change management:**
   - Description of changes made during development
   - Version control and change tracking
   - Re-validation and re-testing records
   - Configuration management documentation

5. **Human oversight documentation:**
   - Oversight measures and their implementation
   - Operator qualification requirements
   - Instructions for human intervention
   - Automation bias mitigation measures

### 11.6 Template Examples

**Example Model Card (Minimal):**

```markdown
---
model-id: cr-2026-001
version: 1.2.0
date: 2026-06-01
organization: ExampleCorp AI
---

## Model Card: credit-risk-v2

### Model Details
- **Architecture:** Gradient Boosted Tree (XGBoost 2.1)
- **Input features:** 47 features (demographic, financial, behavioral)
- **Output:** Credit risk score (0–1000), binary decision (approve/decline)
- **Framework:** XGBoost 2.1, scikit-learn 1.4

### Intended Use
- **Primary:** Automated creditworthiness assessment for consumer loans < $50K
- **Out-of-scope:** Business loans, loans > $50K, non-US applicants
- **Target population:** US residents, age 18+, with credit history

### Performance
- **Overall AUC-ROC:** 0.89
- **Demographic Parity Difference:** 0.03 (acceptable < 0.10)
- **Equal Opportunity Difference:** 0.02 (acceptable < 0.05)
- [Full subgroup breakdown in Appendix]

### Limitations
- Limited performance for thin-file applicants (< 12 months credit history)
- Calibration drifts observed > 6 months since last update
- Not validated for non-English-language applications

### Maintenance
- Retraining schedule: Monthly
- Monitoring: Real-time drift detection, weekly fairness reports
- Contact: ai-governance@examplecorp.com
```

---

## 12. International Alignment Efforts

Multiple international initiatives aim to harmonize AI governance frameworks across jurisdictions.

### 12.1 OECD AI Principles

Adopted in 2019 (updated 2024), the OECD AI Principles are the most widely adopted international AI framework:

**Five value-based principles:**
1. **Inclusive growth, sustainable development, and well-being** — AI should benefit people and the planet
2. **Human-centered values and fairness** — AI systems should respect human rights, democracy, and diversity
3. **Transparency and explainability** — AI systems should be transparent and explainable
4. **Robustness, security, and safety** — AI systems should be safe, secure, and robust throughout their lifecycle
5. **Accountability** — AI actors should be accountable for AI system operation

**Implementation recommendations:**
- National AI strategies and policies
- Investment in AI R&D
- Digital government transformation
- AI skills development
- International cooperation for trustworthy AI
- Multi-stakeholder governance

### 12.2 GPAI (Global Partnership on AI)

GPAI brings together 29 member countries to bridge AI theory and practice:

**Working groups:**
- Responsible AI
- Data governance
- Future of work
- Innovation and commercialization
- AI for pandemic response
- AI and climate action

**Key outputs (2024–2026):**
- Practical guidance for AI accountability
- Framework for AI incident reporting
- AI data justice toolkit
- Cross-border AI governance recommendations
- AI and climate change assessment methodology

### 12.3 Council of Europe AI Convention

The Council of Europe's **Framework Convention on Artificial Intelligence and Human Rights, Democracy, and the Rule of Law** (adopted May 2024):

**Key provisions:**
- Legality: AI systems must operate within legal frameworks
- Non-discrimination: AI systems must not perpetuate discrimination
- Transparency: Individuals must know when AI affects them
- Accountability: Human oversight and remedy mechanisms
- Risk-based approach: Proportional obligations based on risk
- Remedies: Access to justice for AI-related rights violations

**Status:** Open for signature (September 2024), expected to enter into force in 2027. Applies to both EU and non-EU Council of Europe members (46 countries).

### 12.4 G7 Hiroshima AI Process

The G7 Hiroshima AI Process (established May 2023) produced:

**Hiroshima AI Declaration:**
- Commitment to safe, secure, and trustworthy AI
- Support for interoperable governance frameworks
- Development of international AI standards
- Multi-stakeholder engagement

**Hiroshima AI Process Comprehensive Policy Framework:**
1. **Risk identification** — Taxonomy of AI risks
2. **Risk management** — Measures proportional to risk
3. **Governance** — International cooperation mechanisms
4. **Transparency** — Documentation and disclosure
5. **Testing and evaluation** — Pre-deployment and ongoing
6. **Accountability** — Clear responsibility allocation

**Hiroshima AI Process Code of Conduct for AI Developers (2024):**
- Mandatory safety assessments for advanced AI systems
- Pre-deployment safety testing
- Transparency reporting
- Information sharing with governments
- Cybersecurity and insider threat protections
- Watermarking and content provenance
- Prioritizing research on societal risks

### 12.5 UNESCO AI Ethics Recommendation

UNESCO's Recommendation on the Ethics of AI (adopted November 2021) is the first global normative instrument on AI ethics:

**Core values:**
- Respect, protection, and promotion of human rights
- Environmental and ecosystem flourishing
- Ensuring diversity and inclusiveness
- Living in peaceful, just, and interconnected societies

**Policy areas:**
- Ethical impact assessment
- AI governance and stewardship
- Data policy
- Development and international cooperation
- Gender equality and AI
- Education and research

**Implementation:** UNESCO AI Readiness Assessment Methodology (RAM) used by 50+ countries

### 12.6 Bletchley Declaration & AI Safety Summits

**Bletchley Declaration (November 2023) — UK AI Safety Summit:**
- First international declaration on frontier AI safety
- 29 countries including US, China, EU signed
- Recognizes risks from "frontier" AI models
- Commitments to international cooperation on AI safety testing

**Seoul AI Summit (May 2024):**
- Ongoing international cooperation
- Establishment of AI Safety Institutes network
- Agreement on testing thresholds for frontier models

**Paris AI Summit (February 2025):**
- Focus on sustainable AI and public interest AI
- AI for climate monitoring commitments
- Global AI infrastructure initiatives

---

## 13. Enforcement Trends & Notable Cases

Recent enforcement actions demonstrate regulators' increasing willingness to act:

| Year | Regulator | Target | Violation | Penalty |
|---|---|---|---|---|
| 2024 | Italian Garante | Large language model | GDPR violations (data scraping, transparency) | €30M + order to rectify |
| 2024 | Dutch DPA | Government AI system | Bias in benefits-scoring algorithm | €15M + system suspension |
| 2025 | CNIL (France) | Facial recognition startup | Illegal biometric processing | €20M + deletion of database |
| 2025 | FTC (US) | AI hiring platform | Unfair/deceptive practices (false accuracy claims) | $10M + injunctive relief |
| 2025 | CAC (China) | Major AI platform | Failed content moderation (politically sensitive output) | ¥50M + 30-day suspension |
| 2025 | CMA (UK) | AI chatbot | Consumer protection violations (misleading outputs) | £8M |
| 2026 | EU AI Office | Systemic-risk GPAI | Insufficient transparency documentation | €12M (first AI Act penalty) |

**Emerging enforcement patterns:**
- **Algorithmic accountability** — Regulators increasingly holding companies responsible for AI outcomes
- **Shadow AI enforcement** — Disciplining unauthorized internal AI use
- **AI procurement enforcement** — Government agencies penalized for non-compliant AI procurement
- **Cross-border enforcement** — Multi-regulator joint actions (EU + UK + US)
- **Individual liability** — Officers and directors held personally responsible for AI compliance failures

---

## 14. Future Outlook

**Near-term (2026–2027):**
- EU AI Act full enforcement (high-risk rules apply August 2027)
- US comprehensive federal AI legislation likely (AI Foundation Model Transparency Act, algorithmic accountability bills)
- ISO 42001 certification becomes baseline requirement for enterprise AI procurement
- First major AI Act enforcement actions (significant fines expected)

**Medium-term (2027–2028):**
- Convergence of international frameworks (OECD-GPAI-Council of Europe)
- Sectoral AI regulation expansion (healthcare, finance, transportation fully regulated)
- AI liability directive (EU) — Harmonized liability rules for AI-caused harm
- Real-time regulatory compliance (automated compliance checking via API)
- AI regulatory sandboxes become mandatory for high-risk AI

**Long-term (2028–2030):**
- Global AI treaty potentially emerges from Council of Europe + G7 + GPAI convergence
- AI governance becomes embedded in corporate governance frameworks (board-level AI committees)
- Automated compliance verification systems (AI-regulating-AI)
- Personal AI compliance assistants for individual rights enforcement
- Potential for AI regulatory fragmentation (competing blocs: EU, US, China, Global South)

**Key uncertainties:**
- Will the US pass comprehensive federal AI legislation or maintain sectoral/executive approach?
- How will China's AI regulation evolve as its frontier models (DeepSeek, Qwen, etc.) gain global adoption?
- Will international alignment succeed, or will regulatory fragmentation persist?
- How will AI regulation adapt to emerging capabilities (AGI, autonomous AI systems)?
- What is the appropriate regulatory response to AI safety risks from open-weight models?

---

*This document is current as of June 2026. AI governance is rapidly evolving; consult legal counsel for specific compliance obligations. Cross-reference with 05-AI-Safety-Alignment.md for safety-specific technical requirements and with 07-Fine-Tuning-Custom-Models.md for model development compliance.*

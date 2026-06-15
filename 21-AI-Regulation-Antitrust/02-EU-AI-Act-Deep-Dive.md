# 02 — EU AI Act: Complete Technical & Regulatory Deep Dive

## 1. Introduction: The World's First Comprehensive AI Regulation

The EU Artificial Intelligence Act (Regulation 2024/1689 of the European Parliament and of the Council) was adopted on June 13, 2024, and entered into force on August 1, 2024. It represents the culmination of over four years of legislative work, building on the European Commission's April 2021 proposal and the remarkable acceleration prompted by ChatGPT's launch in November 2022.

This regulation establishes a unified legal framework for the development, placement on the market, and use of AI systems across the European Union. It is the first horizontal, risk-based AI regulatory framework in any major economy, and its extraterritorial reach means it governs AI activity far beyond EU borders.

### 1.1 Legal Basis and Framework

The AI Act is a regulation — not a directive — meaning it is directly applicable and binding in all 27 EU member states without national transposition. It sits within the EU's broader digital strategy alongside:

- **General Data Protection Regulation (GDPR)** (Regulation 2016/679): Data protection and privacy
- **Digital Services Act (DSA)** (Regulation 2022/2065): Platform content moderation
- **Digital Markets Act (DMA)** (Regulation 2022/1925): Competition in digital markets
- **Data Act** (Regulation 2023/2854): Data access and sharing
- **Product Liability Directive (revision)** : Liability for defective AI products
- **AI Liability Directive** (proposed): Civil liability for AI-caused harm

The AI Act does not replace these instruments but operates alongside them. Compliance with the AI Act does not imply compliance with GDPR or other EU laws.

### 1.2 Extraterritorial Scope — The Brussels Effect

Article 2 of the AI Act defines its scope broadly:

1. **Providers** established in the EU placing AI systems on the market or putting them into service
2. **Deployers** established in the EU using AI systems
3. **Providers and deployers established outside the EU** where the output of the AI system is used in the EU
4. **Importers and distributors** of AI systems in the EU
5. **Product manufacturers** placing AI systems on the market under their own name
6. **Authorized representatives** of non-EU providers

This extraterritorial scope means any company worldwide whose AI system produces outputs used by EU residents must comply. This is the "Brussels Effect" — the EU regulatory framework becomes de facto global standard because of market size and enforcement credibility.

### 1.3 Exemptions

Certain activities are exempted:
- AI systems used exclusively for military, defense, or national security purposes
- AI systems used exclusively for research and development
- Non-professional personal use (unless high-risk)
- Free and open-source AI components released for research (unless placed on the market as high-risk or subject to GPAI rules)
- AI systems used by third-country public authorities under bilateral international agreements (certain conditions)

## 2. The Risk-Based Classification System

The AI Act's core innovation is its four-tier risk classification. Every AI system must be classified into exactly one tier based on its intended purpose.

### 2.1 Unacceptable Risk (Prohibited Practices) — Article 5

**Effective**: February 2, 2025
**Penalty**: Up to €35 million or 7% of global annual turnover

Eight categories of AI practice are prohibited outright:

1. **Subliminal manipulation**: AI systems that deploy subliminal techniques to distort behavior, causing or likely to causing physical or psychological harm
2. **Vulnerability exploitation**: AI systems exploiting vulnerabilities of persons due to age, disability, or socio-economic situation to distort behavior
3. **Social scoring**: AI-based social scoring by public or private entities leading to detrimental or unfavorable treatment in unrelated contexts
4. **Predictive policing (individualized)**: Risk assessment of natural persons to predict criminal offenses based solely on profiling or personality traits (except where used for evidence-based crime analytics supporting human assessment)
5. **Non-consensual facial image scraping**: Creating or expanding facial recognition databases via untargeted scraping from the internet or CCTV footage
6. **Emotion recognition in workplaces and education**: AI systems inferring emotions in workplace or educational settings (except for medical or safety reasons)
7. **Biometric categorization (sensitive characteristics)** : Categorizing individuals based on biometric data to deduce race, political opinions, trade union membership, religious/philosophical beliefs, or sexual orientation
8. **Real-time remote biometric identification (RBI) by law enforcement**: Real-time facial recognition in publicly accessible spaces for law enforcement — **severely restricted** with narrow exceptions:
   - Search for victims of kidnapping, trafficking, or sexual exploitation
   - Prevention of specific, substantial, and imminent terrorist threat
   - Localization/identification of perpetrators of specific serious criminal offenses (as listed in Annex II)
   - Requires prior judicial authorization, time/scope limits, notification to national authorities

### 2.2 High-Risk AI Systems — Title III, Articles 6-51

High-risk systems face the most extensive obligations. A system is high-risk if:

1. It is intended to be used as a safety component of a product covered by EU product safety legislation (Annex II list), **or**
2. It is a stand-alone AI system falling into one of eight categories listed in Annex III, **and**
3. It does **not** perform a narrow procedural task, improve a previously completed human activity, detect decision patterns without replacing human assessment, or prepare decisions without material impact

#### Annex III High-Risk Categories:

1. **Biometric identification and categorization of natural persons** (excluding prohibited categories)
2. **Critical infrastructure management and operation** (road traffic, water, gas, electricity, digital infrastructure)
3. **Education and vocational training**: Access, admission, evaluation, assessment of learning outcomes, monitoring student behavior
4. **Employment, workers management, and access to self-employment**: Recruitment, performance evaluation, promotion/termination decisions
5. **Essential services**: Creditworthiness assessment, insurance pricing and risk assessment, essential public benefits eligibility
6. **Law enforcement**: Individual risk assessment, polygraph/lie detector use, evaluation of evidence reliability, predicting re-offense risk, profiling in investigations, crime analytics
7. **Migration, asylum, and border control management**: Polygraph testing, risk assessment, credibility evaluation, visa/travel document authenticity verification, border monitoring
8. **Administration of justice and democratic processes**: AI assisting judicial decision-making, applying law to facts, alternative dispute resolution; AI influencing election outcomes or voter behavior

#### Obligations for High-Risk Systems:

**For providers (developers):**

1. **Risk management system** (Article 9): Continuous, iterative process throughout AI system lifecycle. Identify known and foreseeable risks, estimate probability and severity, implement risk management measures, test with appropriate datasets
2. **Data governance** (Article 10): Training, validation, and testing datasets must be relevant, representative, free from errors, and appropriate for intended purpose. Special categories of personal data may be processed only under strict conditions. Bias detection and correction required
3. **Technical documentation** (Article 11): Comprehensive documentation including development methodology, design specifications, training datasets, evaluation results, accuracy metrics, intended purpose, and risk assessment
4. **Record-keeping and logging** (Article 12): Automatic logging of events during operation. Logs must capture system operation, training data periods, input-output pairs, and human oversight actions
5. **Transparency and provision of information to deployers** (Article 13): Clear, appropriate information about system capabilities, limitations, accuracy, risks, and intended purpose
6. **Human oversight** (Article 14): Appropriate human-machine interface enabling deployers to oversee system operation. Measures include: understanding system capabilities, monitoring for anomalies, deciding not to use system, overriding/intervening
7. **Accuracy, robustness, and cybersecurity** (Article 15): Stated accuracy levels must be achievable. Resilient against errors, faults, and inconsistencies. Robust to adversarial manipulation. Cybersecurity protection against third-party exploitation

**For deployers (users):**

1. Implement technical and organizational measures to use system per provider instructions
2. Monitor operation for risks
3. Keep logs automatically generated
4. Inform workers' representatives when deploying high-risk AI in workplace
5. Conduct Data Protection Impact Assessment (DPIA) if required by GDPR
6. Facilitate human oversight

**For notified bodies (conformity assessment):**

High-risk systems must undergo conformity assessment procedures:
- **Self-assessment** (Annex III category systems): Provider conducts assessment based on harmonized standards
- **Third-party assessment** (biometric systems, Annex II safety components): Notified body involvement

### 2.3 Limited-Risk AI Systems — Title IV, Article 50

**Transparency obligations** apply when:
1. Interacting with an AI system (except obviously): Users must be informed they are interacting with AI
2. AI-generated content (deep fakes must be labeled)
3. AI-generated text published for public interest (AI-generated text must be disclosed)
4. Emotion recognition or biometric categorization systems: Natural persons must be informed when such systems are operating

### 2.4 Minimal-Risk AI Systems

No additional obligations beyond existing law. Voluntary application of codes of conduct encouraged.

## 3. General Purpose AI (GPAI) — A Late-Stage Addition

The rapid emergence of foundation models (GPT-4, Claude, Gemini, Llama) during the legislative process forced the addition of Title V (Articles 51-56) on General Purpose AI.

### 3.1 GPAI Definition

A GPAI model is an AI model that:
- Displays significant generality (competent across wide range of distinct tasks)
- Can integrate into various downstream systems
- Is trained using large-scale self-supervision
- Has significant parametric complexity (>1 billion parameters — indicative threshold)

### 3.2 Two-Tier Obligations for GPAI

**All GPAI providers must:**

1. **Technical documentation**: Detailed documentation including training data, compute used, energy consumption, model architecture, evaluation results
2. **Training data policy**: Detailed summary of copyrighted data used in training (transparency for copyright holders)
3. **Model card**: Publicly available documentation of model capabilities, limitations, biases, safety evaluation results
4. **Copyright policy**: Implement a policy to respect EU copyright law
5. **Information for downstream providers**: Sufficient information for downstream providers to understand capabilities and limitations

**GPAI with systemic risk (additional obligations):**

A GPAI model has systemic risk if it either:
- Has cumulative training compute exceeding 10^25 FLOPs (presumed systemic risk)
- Is designated by the Commission as having high-impact capabilities equivalent to or exceeding the compute threshold

Systemic risk providers must:
1. Conduct **model evaluations** (adversarial testing, red-teaming, capability assessment)  
2. Assess and mitigate **systemic risks** at EU level
3. Track, document, and report **serious incidents** and corrective measures
4. Ensure **adequate cybersecurity protection** against unauthorized access
5. Participate in development of **codes of practice** (Article 56)

### 3.3 Open-Source GPAI

Free and open-source AI models (where parameters are published) are partially exempt:
- Exempt from transparency obligations for downstream providers
- Exempt from copyright training data summary (unless systemic risk)
- Still subject to systemic risk obligations if above the compute threshold

## 4. Conformity Assessment Procedures

### 4.1 Self-Assessment (Most High-Risk Systems)

For high-risk systems falling under Annex III (not biometrics):
1. Establish and implement risk management system
2. Ensure data governance compliance
3. Prepare technical documentation
4. Design for automatic logging
5. Ensure transparency and information provision
6. Design for human oversight
7. Implement accuracy, robustness, cybersecurity measures
8. Draw up EU declaration of conformity
9. Affix CE marking
10. Register in EU database (Annex III systems must register in EU AI database)

### 4.2 Third-Party Assessment (Biometric Systems, Safety Components)

For biometric systems (Annex III, category 1) and systems used as safety components of regulated products:
- Provider submits technical documentation to **notified body**
- Notified body examines documentation, assesses conformity
- Issues certificate of conformity (valid for 5 years, subject to surveillance)
- Provider affixes CE marking and registers in database

### 4.3 Role of Harmonized Standards

The Commission has issued standardization request to CEN/CENELEC (JTC 21) to develop harmonized standards. Conformity with harmonized standards provides presumption of conformity with AI Act requirements. Standards cover:
- Risk management system
- Data quality and data governance
- Transparency and information provision
- Human oversight
- Accuracy, robustness, cybersecurity
- Conformity assessment procedures

As of mid-2026, approximately 60% of harmonized standards are finalized. In areas without standards, compliance with the state of the art is required.

## 5. Governance and Enforcement Architecture

### 5.1 European AI Office

Located within the European Commission (DG CONNECT):
- Coordinates implementation and enforcement
- Monitors GPAI codes of practice
- Conducts evaluations of GPAI models
- Maintains EU AI database
- Provides guidance and interpretation
- Imposes fines on GPAI providers (direct enforcement power)

### 5.2 European Artificial Intelligence Board (EAIB)

Composition: One representative per member state, plus European Data Protection Supervisor, chaired by AI Office.
Functions:
- Advises Commission and member states
- Issues opinions and recommendations
- Contributes to uniform application
- Maintains market surveillance coordination

### 5.3 Scientific Panel of Independent Experts

Independent experts advising AI Office and national authorities on:
- Systemic risk designation
- Evaluation methodologies
- Safety assessment
- Emerging risks

### 5.4 National Competent Authorities

Each member state must designate:
- **Notifying authority**: Responsible for establishing and operating notified bodies
- **Market surveillance authority**: Enforcement at national level
Each authority must have at least one **AI testing facility** available.

### 5.5 Penalties

| Violation | Maximum Fine |
|-----------|-------------|
| Prohibited practices | €35M or 7% of global annual turnover |
| Non-compliance with high-risk obligations | €15M or 3% of global annual turnover |
| Incorrect information to authorities | €7.5M or 1% of global annual turnover |
| GPAI non-compliance | €15M or 3% of global annual turnover |
| SME/Startup penalties | Lower amounts (more proportionate) |

## 6. Implementation Timeline

| Date | Milestone |
|------|-----------|
| 13 Jun 2024 | Regulation adopted |
| 1 Aug 2024 | Entry into force |
| 2 Feb 2025 | Prohibited practices effective |
| 2 May 2025 | GPAI rules effective (codes of practice period started) |
| 2 Aug 2025 | GPAI transparency obligations, codes of practice |
| 2 Aug 2026 | Annex III high-risk rules effective |
| 2 Aug 2027 | Full application (Annex II high-risk) |
| 31 Dec 2030 | Annex III legacy systems (already placed on market) compliance |

## 7. Interplay with GDPR

The AI Act and GDPR are complementary but create complex interactions:

| Area | Interaction |
|------|-------------|
| **Data processing for bias detection** | High-risk AI may process special categories of data for bias detection (Article 10 AI Act) — explicit legal basis |
| **Data Protection Impact Assessment** | DPIA required under Article 35 GDPR for high-risk AI; may be combined with AI Act risk assessment |
| **Right to explanation** | Article 13-14 AI Act + Article 22 GDPR (automated individual decision-making) — dual transparency obligations |
| **Cross-border data processing** | GDPR governs data flows; AI Act governs AI system deployment |
| **Enforcement coordination** | National data protection authorities (DPAs) will also enforce AI Act in many member states |
| **Consent** | GDPR consent standards apply when AI systems process personal data; AI Act's risk obligations are additional |
| **Privacy by design** | AI Act Article 15 (accuracy, robustness, cybersecurity) complements GDPR Article 25 (data protection by design/default) |

**Key tension**: The AI Act allows special category data processing for bias detection; GDPR Article 9 generally prohibits such processing without specific consent. The AI Act provides the explicit legal basis needed. However, organizations must still comply with GDPR principles of data minimization and purpose limitation.

## 8. EU AI Act Compliance Checklist

Below is a comprehensive compliance checklist organized by function. Use this for gap analysis and audit preparation.

### 8.1 Classification (Pre-Compliance Phase)

- [ ] Determine if your system falls within scope (EU-established provider, or output used in EU)
- [ ] Determine if exempted (military, research, personal use, open-source)
- [ ] Classify AI system into risk tier:
  - [ ] Unacceptable risk (Article 5) — if yes, **do not place on market**
  - [ ] High-risk (Article 6 + Annex III) — most obligations apply
  - [ ] GPAI model — Title V obligations
  - [ ] Limited risk (Article 50) — transparency only
  - [ ] Minimal risk — voluntary codes of conduct only
- [ ] If high-risk, determine whether it falls under Annex II (safety components) or Annex III
- [ ] If GPAI, determine if systemic risk threshold is exceeded (compute >10^25 FLOPs or designated)

### 8.2 Risk Management System (Article 9)

- [ ] Establish documented risk management system
- [ ] Risk management is continuous, iterative, throughout AI lifecycle
- [ ] Identify known and foreseeable risks under intended use and reasonably foreseeable misuse
- [ ] Estimate and evaluate risks (probability × severity)
- [ ] Implement risk management measures (test with pre-defined datasets)
- [ ] Residual risk evaluation acceptable
- [ ] Documentation maintained throughout system's lifetime
- [ ] Regular review and update of risk assessment

### 8.3 Data Governance (Article 10)

- [ ] Training, validation, and testing datasets examined for relevance
- [ ] Datasets are representative for intended purpose
- [ ] Datasets are free from biases where possible (bias detection and mitigation)
- [ ] Special categories of personal data processed only under AI Act Article 10(5) conditions
- [ ] Data governance and management practices in place
- [ ] Data collection methods documented
- [ ] Data labeling protocols documented
- [ ] Data provenance, purpose, and suitability documented

### 8.4 Technical Documentation (Article 11)

- [ ] General description of AI system (intended purpose, development methodology, architecture)
- [ ] Detailed description of system elements:
  - [ ] Development methodology, design choices
  - [ ] System architecture and data flow
  - [ ] Training datasets (sources, size, preprocessing)
  - [ ] Evaluation methodology and metrics
  - [ ] Accuracy, robustness, cybersecurity measures
  - [ ] Human oversight measures
  - [ ] Risk assessment documentation
  - [ ] Conformity assessment planning
- [ ] Version history and change log

### 8.5 Record-Keeping and Logging (Article 12)

- [ ] Automatic logging of events during system operation
- [ ] Logs capture:
  - [ ] System operation periods
  - [ ] Training dataset periods (if relevant)
  - [ ] Input-output pairs
  - [ ] Human oversight actions
  - [ ] Identification of specific human overseer
  - [ ] System anomalies and incidents
- [ ] Logs kept for appropriate duration (proportionate to system lifecycle)
- [ ] Logs accessible to deployers and authorities

### 8.6 Transparency and Information (Article 13)

- [ ] Provider information (name, contact details) on system
- [ ] System characteristics, limitations, capabilities documented
- [ ] Intended purpose documented
- [ ] Level of accuracy, robustness, performance documented
- [ ] Foreseeable risks documented
- [ ] Instructions for use provided
- [ ] Maintenance and care instructions provided
- [ ] Information provided in clear, intelligible language

### 8.7 Human Oversight (Article 14)

- [ ] Human oversight measures identified and implemented
- [ ] Oversight measures enable deployer to:
  - [ ] Understand system capabilities and limitations
  - [ ] Monitor for anomalies, dysfunctions, unexpected performance
  - [ ] Remain aware of automation bias
  - [ ] Correctly interpret system output
  - [ ] Decide not to use system in any situation
  - [ ] Intervene or stop system operation
- [ ] Human-machine interface designed appropriately
- [ ] Oversight measures proportionate to risk

### 8.8 Accuracy, Robustness, Cybersecurity (Article 15)

- [ ] Accuracy levels stated and achievable
- [ ] Accuracy metrics appropriate for intended purpose
- [ ] System robust to errors, faults, inconsistencies
- [ ] Resilience to adversarial inputs
- [ ] Cybersecurity measures against third-party exploitation
- [ ] Reproducibility of results (where applicable)
- [ ] System performs consistently under normal and abnormal conditions

### 8.9 Conformity Assessment

- [ ] Internal (self) assessment conducted (for Annex III, non-biometric)
- [ ] Or third-party notified body assessment (biometric systems, Annex II products)
- [ ] EU declaration of conformity drafted
- [ ] CE marking affixed
- [ ] System registered in EU database (for Annex III, high-risk)
- [ ] Technical documentation maintained for regulator inspection

### 8.10 Post-Market Monitoring

- [ ] Post-market monitoring system established
- [ ] Serious incident reporting mechanism (to national authority + AI Office)
- [ ] Corrective action plan for non-conforming systems
- [ ] Periodic review of risk assessment
- [ ] Cooperation with competent authorities on incident investigations
- [ ] Ongoing update of technical documentation

### 8.11 GPAI-Specific (If Applicable)

- [ ] Training data policy (copyright compliance, Article 53(1)(c))
- [ ] Technical documentation drawn up and maintained
- [ ] Model card publicly available
- [ ] Information provided to downstream providers
- [ ] Copyright policy implemented
- [ ] **If systemic risk**: Model evaluations conducted
- [ ] **If systemic risk**: Systemic risk assessment and mitigation
- [ ] **If systemic risk**: Serious incident reporting
- [ ] **If systemic risk**: Adequate cybersecurity
- [ ] **If systemic risk**: Participation in codes of practice development

### 8.12 Deployer Obligations (If Applicable)

- [ ] Use system in accordance with provider instructions
- [ ] Monitor operation for risks (human oversight)
- [ ] Inform workers' representatives if deployed in workplace (Article 29(5))
- [ ] Conduct DPIA if required by GDPR Article 35
- [ ] Keep automated logs
- [ ] Cooperate with market surveillance authorities
- [ ] Implement human oversight measures

## 9. Practical Challenges and Open Questions (2026)

1. **Harmonized standards gap**: With ~40% of standards still in development, providers must rely on "state of the art" interpretation — creating legal uncertainty
2. **GPAI definition scope**: What counts as a GPAI model vs. a purpose-built model? Compressed models, fine-tuned models, ensemble models
3. **Open-source exemption boundary**: When does an open-source model become "placed on the market" as commercial? Ambiguity around model weights distribution via APIs
4. **Systemic risk threshold adequacy**: The 10^25 FLOPs threshold was set in 2023 context. By 2026, many models exceed this. Potential for threshold escalation
5. **Notified body capacity**: Insufficient number of designated notified bodies creates bottlenecks for third-party conformity assessment
6. **International coherence**: How will EU enforcement interact with US, UK, China regulation? Mutual recognition? Enforcement coordination?
7. **SME burden**: Startups and SMEs face disproportionate compliance costs. The AI Act includes innovation-facilitating provisions (regulatory sandboxes), but practical impact unclear
8. **Extraterritorial enforcement**: Can the EU effectively enforce against US and Chinese companies without international cooperation?

## 10. Enforcement Outlook (2026-2027)

The AI Office has announced priority enforcement areas for the first year of robust enforcement:
- Prohibited practices (particularly emotion recognition and social scoring)
- GPAI transparency obligations (training data disclosure, copyright compliance)
- High-risk systems in employment and creditworthiness assessment
- Real-time biometric identification by law enforcement (narrow exceptions)

Expected: 5-10 major enforcement actions by end of 2027, including at least one against a major foundation model provider, with fines in the €100M+ range.

---

**Document metadata**: Created June 2026. Part of the AI Regulation & Antitrust knowledge base. For complementary information on cross-jurisdictional compliance, see documents 01 (Overview), 03 (US Regulation), and 04 (China Governance).

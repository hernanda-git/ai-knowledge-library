# 06 — AI Regulation by Sector: Healthcare, Finance, Legal, Education, HR, Insurance, Defense, and Transportation

## 1. Introduction: The Sectoral Approach to AI Regulation

While the EU AI Act provides horizontal regulation across all sectors, many jurisdictions — particularly the United States — regulate AI through existing sectoral regulatory frameworks. This approach applies AI-specific interpretations of existing laws and sometimes creates new AI-specific rules within sectoral domains.

This document provides comprehensive analysis of AI regulation across eight key sectors: healthcare, financial services, legal, education, human resources, insurance, defense/national security, and transportation (autonomous vehicles and aviation).

**Core principle**: Sectoral AI regulation applies the "same activity, same risk, same rules" principle — AI systems performing regulated activities face the same (or analogous) regulatory requirements as traditional tools performing those activities.

### 1.1 Regulatory Patterns Across Sectors

| Sector | Primary Regulator(s) | AI-Specific Rules | Enforcement Approach |
|--------|---------------------|-------------------|---------------------|
| Healthcare | FDA, HHS, OCR | Yes (SaMD, PCCP) | Pre-market review + post-market |
| Financial Services | SEC, CFPB, OCC, Fed | Proposed rules + guidance | Examination + enforcement |
| Legal | State bar associations | Ethics opinions (emerging) | Self-regulation + bar discipline |
| Education | ED, FERPA Office | Guidance (limited) | Consent decrees + investigations |
| HR/Employment | EEOC, DOL, OFCCP | Guidance + litigation | Enforcement actions + private lawsuits |
| Insurance | State insurance depts | NAIC model law (emerging) | Rate filing + market conduct |
| Defense | DoD, CDAO | Policy directives + NDAA | Contract compliance + oversight |
| Autonomous Vehicles | NHTSA, DOT | Guidance + standing orders | Defect investigations + recalls |
| Aviation | FAA, EASA | Certification specs + guidance | Type certification + airworthiness |

## 2. Healthcare AI Regulation

### 2.1 FDA Regulation of AI/ML Medical Devices

The FDA treats AI/ML-enabled software as medical devices when it is intended to diagnose, treat, cure, mitigate, or prevent disease. Key regulatory pathways:

**Software as a Medical Device (SaMD) Classification:**

| Class | Risk Level | Examples | Regulatory Pathway |
|-------|-----------|----------|-------------------|
| Class I | Low | AI-powered appointment scheduling | Generally exempt from 510(k) |
| Class II | Moderate | AI radiology triage, AI ECG interpretation | 510(k) clearance with special controls |
| Class III | High | AI critical care monitoring, AI cancer diagnosis | Premarket Approval (PMA) or De Novo |

**Predetermined Change Control Plan (PCCP) — revolutionary framework:**

The PCCP, finalized in April 2025 guidance, allows AI/ML devices that continuously learn to receive pre-authorization for planned changes without requiring new 510(k)/PMA submissions. Framework:

1. **Description of modifications**: Types of anticipated changes (algorithm updates, performance improvements, adaptation to new data)
2. **Protocol for implementing changes**: How changes will be developed, validated, and implemented
3. **Impact assessment**: Methodology for assessing impact on safety and effectiveness
4. **Performance monitoring**: Ongoing real-world monitoring to detect degradation

**Requirements for AI medical devices:**

1. **Clinical validation**: Evidence that the AI device performs safely and effectively in the intended patient population
2. **Bias assessment**: Evaluation of performance across demographic subgroups (age, race, ethnicity, sex, comorbidities)
3. **Generalizability**: Testing on diverse datasets representative of intended use population
4. **Interpretability**: For high-risk decisions, reasonable interpretability of AI outputs
5. **Transparency**: Clear labeling of AI-assisted vs. AI-autonomous decisions
6. **Cybersecurity**: Robust security against adversarial attacks (especially for imaging AI)
7. **Real-world performance monitoring**: Post-market surveillance confirming continued safety and effectiveness

**Key FDA AI Publications:**

| Document | Date | Significance |
|----------|------|-------------|
| AI/ML SaMD Action Plan | Jan 2021 | Framework for AI/ML device regulation |
| Good Machine Learning Practice (GMLP) | Oct 2021 (with Health Canada, MHRA) | Guiding principles for ML medical devices |
| Clinical Decision Support Software Guidance | Sep 2024 | Clarifies when CDS is regulated as device |
| PCCP Guidance | Apr 2025 | Enables iterative AI without re-submission |
| AI Transparency Framework | Draft 2025 (final pending) | Transparency requirements for AI devices |
| Cybersecurity of AI Medical Devices | Draft 2026 | Security requirements for connected AI devices |

### 2.2 HIPAA and AI

The Health Insurance Portability and Accountability Act (HIPAA) governs AI processing of protected health information (PHI):

- **Covered entities**: Healthcare providers, health plans, clearinghouses using AI must comply with HIPAA Privacy, Security, and Breach Notification Rules
- **Business associates**: AI vendors processing PHI are business associates requiring HIPAA-compliant BAAs
- **De-identification**: AI training on de-identified data is not subject to HIPAA; but re-identification risk must be assessed
- **AI and authorization**: Using PHI for AI training requires individual authorization unless IRB waiver or de-identified
- **Minimum necessary**: AI systems should access minimum PHI necessary for their function
- **Security rule**: AI systems must implement administrative, physical, and technical safeguards for ePHI

**Enforcement**: HHS Office for Civil Rights (OCR) has issued guidance on AI and HIPAA. 2025-2026 OCR priorities include AI-related PHI disclosures and AI model training data privacy.

### 2.3 Healthcare AI State Regulation

State regulation of AI in healthcare varies, with notable activity in:
- **California**: CCPA/CPRA applies to AI training on consumer health data
- **New York**: State hospital AI oversight requirements (2025)
- **Colorado**: AI in healthcare decisions covered by Colorado AI Act (high-risk category)

### 2.4 Key Compliance Areas for Healthcare AI

- [ ] Determine if AI is a regulated medical device (SaMD)
- [ ] If SaMD, identify appropriate regulatory pathway (510(k)/PMA/De Novo)
- [ ] For continuously learning AI, develop PCCP
- [ ] Conduct clinical validation with diverse populations
- [ ] Implement bias testing across demographic groups
- [ ] Ensure HIPAA compliance for AI processing PHI
- [ ] Execute BAAs with AI vendors
- [ ] Establish real-world performance monitoring
- [ ] Implement cybersecurity measures for AI systems
- [ ] Prepare FDA inspection and post-market surveillance readiness

## 3. Financial Services AI Regulation

### 3.1 SEC Regulation of AI in Capital Markets

**SEC Proposed Rule on Predictive Data Analytics (July 2023, revised 2024, final 2025):**

This rule addresses conflicts of interest arising from AI use by broker-dealers and investment advisers:

1. **Scope**: Covers use of "covered technology" (including AI) for investor interactions, recommendations, and order execution
2. **Conflict identification**: Firms must identify AI-related conflicts that place firm interests ahead of investor interests
3. **Conflict elimination or neutralization**: AI conflicts must be eliminated or neutralized — cannot be merely disclosed
4. **Recordkeeping**: Detailed records of AI model development, testing, validation, and monitoring
5. **Annual review**: Independent review of AI compliance program
6. **Testing**: Pre-implementation testing, ongoing monitoring, and periodic re-testing

**SEC Regulation Best Execution (Rule 605/606)** : AI trading algorithms must achieve best execution for client orders considering price, speed, likelihood of execution, and other factors.

**SEC Market Access Rule (Rule 15c3-5):** AI algorithmic trading systems must have risk controls:
- Pre-trade credit/position limits
- Automated kill switch for aberrant behavior
- Real-time monitoring of AI trading decisions

**SEC AI Washing Enforcement:**
- Settled cases against companies falsely claiming AI capabilities in SEC filings
- Focus on material misstatements about AI use in investment strategies
- Enhanced disclosure requirements for AI use in compliance functions (2025)

### 3.2 CFPB AI Regulation

**Equal Credit Opportunity Act (ECOA) / Regulation B:**
- AI credit scoring models must comply with adverse action notification requirements
- Specific reasons for credit denial must be provided — even if AI model cannot easily explain decisions
- CFPB circular (2022): ECOA requirements apply regardless of technology used

**Fair Credit Reporting Act (FCRA):**
- AI credit scoring models may constitute "credit reporting systems" if they evaluate creditworthiness using consumer report information
- Accuracy requirements apply to AI model outputs
- Consumer dispute rights apply to AI-based consumer reports

**Unfair, Deceptive, or Abusive Acts or Practices (UDAAP):**
- AI chatbots providing false or misleading information to consumers may violate UDAAP
- AI bias in lending (race, gender, age discrimination) as UDAAP violation
- AI pricing algorithms that result in discriminatory pricing (e.g., higher rates for minorities) as UDAAP

**CFPB Section 1033 Open Banking Rule (final October 2024):**
- Consumer data rights extend to data used by AI financial advisory services
- AI developers must provide consumer data access and portability

### 3.3 Banking Prudential Regulation

**Federal Reserve, OCC, FDIC:**
- **Model Risk Management Guidance (SR 11-7/OCC 2011-12)** : Applies to AI/ML models used in banking (credit risk, capital calculations, fraud detection)
- **AI model validation expectations**: Independent validation, ongoing monitoring, governance, and documentation
- **AI explainability**: Challenge of black-box AI models for regulatory capital calculations (must be auditable)
- **Fair lending**: AI credit models tested for disparate impact under ECOA and Fair Housing Act

**Recent developments:**
- **2024**: OCC pilot program for AI in credit underwriting
- **2025**: Interagency guidance on AI model risk management (proposed)
- **2026**: FDIC proposed rule on AI bias testing in lending

### 3.4 Financial AI Compliance Checklist

- [ ] Identify all AI systems used in financial decision-making
- [ ] For investment advice AI: Comply with predictive data analytics rule
- [ ] For trading AI: Ensure best execution, risk controls, and compliance supervision
- [ ] For lending AI: Ensure ECOA adverse action compliance and fair lending testing
- [ ] For fraud detection AI: Ensure accuracy, avoid disparate impact
- [ ] For AI chatbots: Ensure accurate consumer information, UDAAP compliance
- [ ] Implement model risk management framework (SR 11-7 standards)
- [ ] Conduct annual AI model validation (independent third-party)
- [ ] Maintain AI records for regulatory examination
- [ ] Test for AI bias across protected classes

## 4. Legal Profession AI Regulation

### 4.1 ABA Regulation and State Ethics Rules

The American Bar Association (ABA) and state bar associations govern lawyer use of AI:

**ABA Model Rules (as interpreted for AI):**

- **Rule 1.1 (Competence)** : Lawyers must understand AI tools they use; competence includes understanding AI benefits and risks (ABA Formal Opinion 512, 2024)
- **Rule 1.6 (Confidentiality)** : Inputting client information into AI tools may waive attorney-client privilege; lawyers must ensure AI providers maintain confidentiality
- **Rule 5.3 (Non-lawyer Assistance)** : AI systems used by lawyers are akin to non-lawyer assistants; supervision and confidentiality obligations apply
- **Rule 7.1-7.3 (Advertising/Solicitation)** : AI-generated legal advertising must not be false or misleading
- **Rule 8.4 (Misconduct)** : Using AI to harass, discriminate, or mislead constitutes professional misconduct

**State bar AI opinions (2024-2025):**
- **California**: Formal Opinion 2023-207 — AI confidentiality and competence requirements
- **New York**: NYBA Ethics Opinion 2024-1 — AI use in legal practice
- **Florida**: Proposed ethics guidelines for AI in litigation (2025)
- **Texas**: Ethics guidance on AI-generated legal documents
- **DC**: Bar opinion on AI-assisted legal research

**Key compliance areas:**

1. **AI-generated legal documents**: Must be reviewed by human attorney; lawyer responsible for accuracy
2. **AI legal research**: Must verify AI output; AI hallucination of cases is professional misconduct (multiple sanctions issued 2024-2025)
3. **Client consent**: Some states require disclosure to clients when AI is used in legal services
4. **Billing**: AI efficiency cannot justify double billing; time saved by AI must be reflected in fees
5. **Court filings**: Increasing number of courts require AI disclosure (filing was AI-assisted); judges' standing orders

### 4.2 AI in Judicial Decision-Making

- AI use in sentencing, bail, and parole decisions: Subject to due process and equal protection challenges
- **COMPAS/risk assessment**: Constitutional challenges to AI in criminal justice (State v. Loomis, 2016; ongoing challenges)
- **EU AI Act**: High-risk classification for AI used in judicial decision-making
- **US**: No comprehensive federal rules; state-by-state approach

## 5. Education AI Regulation

### 5.1 FERPA and AI

Family Educational Rights and Privacy Act (FERPA) governs AI processing of student education records:

- **Student records**: AI analyzing student performance data must have legitimate educational interest basis
- **Directory information**: May be used for AI systems with opt-out
- **Third-party AI vendors**: Must be under direct control of educational institution; annual audit requirements
- **De-identified data**: De-identified student data may be used for AI training without FERPA restrictions
- **AI grading**: May be used but must be accurate; right to human review for consequential decisions

**ED FERPA AI guidance:**
- **2024 guidance**: AI use in schools — data privacy, transparency, and equity considerations
- **2025 draft guidance**: AI tools in the classroom — data collection limitations and student notice
- **2026 proposed rule**: AI and FERPA — updating definitions for AI era

### 5.2 Students with Disabilities

**IDEA (Individuals with Disabilities Education Act):**
- AI used to identify or evaluate students for special education must be valid and reliable
- AI must not discriminate against students with disabilities
- Human judgment in AI-assisted IEP (Individualized Education Program) decisions

**Section 504 / ADA:**
- AI educational tools must be accessible to students with disabilities
- AI must provide reasonable accommodations

### 5.3 Admissions AI

- **Test-optional admissions**: AI analysis of applicant records may introduce bias
- **Algorithmic admissions systems**: Under investigation by ED Office for Civil Rights
- **California**: AI in admissions covered by SB 1047 protections (consequential decisions)
- **Colorado AI Act**: Educational admissions and evaluations as high-risk AI

## 6. Human Resources and Employment AI

### 6.1 EEOC Framework

The EEOC enforces federal anti-discrimination laws for AI in employment:

**Title VII of Civil Rights Act (1964):**
- AI hiring, promotion, termination tools must not discriminate on race, color, religion, sex, or national origin
- **Disparate impact**: AI that has adverse impact on protected groups violates Title VII even without discriminatory intent
- **Selection procedures**: AI assessments considered "selection procedures" under Uniform Guidelines on Employee Selection Procedures (UGESP)
- **Four-fifths rule**: Selection rate for protected groups must be at least 80% of highest group — applies to AI systems

**Americans with Disabilities Act (ADA):**
- AI employment testing must accommodate disabilities
- AI that screens out people with disabilities may violate ADA
- Reasonable accommodation requirement applies to AI assessments

**Age Discrimination in Employment Act (ADEA):**
- AI must not discriminate against workers aged 40+
- AI resume screening favoring recent graduates may have disparate impact on older workers

**EEOC AI guidance summary:**
1. **Assessment vendors**: Employers are liable for AI vendor discrimination — cannot outsource compliance
2. **Audit requirements**: Employers should conduct self-audits for AI adverse impact
3. **Alternative procedures**: If AI has adverse impact, employers must evaluate alternative AI systems with less impact
4. **Job-relatedness**: AI assessment must be job-related and consistent with business necessity
5. **Accommodations**: AI assessments must provide reasonable accommodation

### 6.2 State HR AI Laws

| State | Requirement | Effective |
|-------|-------------|-----------|
| **New York** (NYC LL 144) | AI hiring tool bias audit, results published | July 2023 (expanded 2025) |
| **Illinois** | Notice + consent for AI video interviews | Updated 2025 |
| **Maryland** | Employment AI bias testing required | 2026 |
| **Colorado** | High-risk AI in employment covered | 2026 |
| **California** | SB 1047 covers employment AI | 2026 |

### 6.3 Worker Surveillance AI

- **National Labor Relations Act (NLRA):** AI surveillance of union activity may be unlawful
- **State laws**: Connecticut, New York, and California require employer disclosure of AI monitoring
- **EU**: GDPR Article 22 limits automated decision-making about workers
- **EU AI Act**: High-risk classification for AI in employment; emotion recognition in workplace is prohibited

## 7. Insurance AI Regulation

### 7.1 State Insurance Regulation (US)

Insurance is primarily state-regulated under the McCarran-Ferguson Act. State insurance departments regulate AI:

**National Association of Insurance Commissioners (NAIC):**

**NAIC Principles on AI (2020, updated 2024):**
- Fairness: AI must not unfairly discriminate based on protected characteristics
- Transparency: Insurers must disclose AI use in underwriting, pricing, claims
- Accountability: Insurers are responsible for AI decisions by their vendors
- Compliance: AI must comply with state insurance laws

**NAIC Model Bulletin on AI (2024):**
- Insurers must develop AI governance framework
- AI used in ratemaking must comply with actuarial standards
- AI claims processing must comply with unfair claims settlement practices
- AI underwriting must not use prohibited factors (race, religion, national origin)
- Annual reporting on AI use to state insurance departments

**State actions:**
- **California**: Proposition 103 — AI ratemaking must demonstrate no unfair discrimination
- **New York**: DFS Circular Letter 2024 on AI in insurance — governance, bias testing, transparency
- **Connecticut**: AI in insurance fairness requirements (2025 statute)
- **Colorado**: Insurance AI covered under Colorado AI Act

### 7.2 EU Insurance AI

- **Solvency II**: AI risk models for capital calculation must be validated
- **IDD (Insurance Distribution Directive)**: AI-based insurance advice must be suitable for customer
- **AI Act (High-Risk)**: Insurance pricing and risk assessment for health, life, and property insurance
- **Gender Directive**: AI must not use gender as risk factor in insurance pricing

## 8. Defense and National Security AI

### 8.1 DoD AI Governance

**DoD Responsible AI (RAI) Strategy and Implementation Pathway (2022, updated 2025):**
- Ethical principles: Responsible, equitable, traceable, reliable, governable
- AI Ethics Framework: Embed RAI in AI development lifecycle
- AI Assurance: Testing and evaluation of AI systems before deployment
- AI Workforce: Training programs for AI literacy across DoD

**DoD Directive 3000.09 (Autonomy in Weapon Systems):**
- **2023 update**: Expanded scope to cover AI-enabled autonomous and semi-autonomous weapons
- Human control requirement: Meaningful human control over lethal decisions
- Testing requirements: Rigorous testing before operational use
- Review requirements: Senior-level review for autonomous weapons systems

**Key DoD AI Programs and Regulation:**
- **Project Maven**: AI for intelligence analysis; subject to RAI Framework
- **Air Combat Evolution (ACE)** : AI fighter pilot; certification for autonomous flight
- **Joint All-Domain Command and Control (JADC2)** : AI decision support for military operations
- **Autonomous Systems**: Ground, maritime, and aerial autonomous vehicles

### 8.2 International Humanitarian Law (IHL) and AI

- **Distinction**: AI weapons must distinguish combatants from civilians
- **Proportionality**: AI must not cause excessive civilian harm
- **Necessity**: Military necessity must govern AI targeting
- **Human control**: Emerging norm of "meaningful human control" over lethal AI

**UN CCW (Convention on Conventional Weapons) GGE on LAWS:**
- Group of Governmental Experts on Lethal Autonomous Weapons Systems
- No binding treaty as of 2026 (negotiations ongoing since 2014)
- Divergent positions: US prefers non-binding norms; Russia blocks negotiations; EU advocates binding treaty

### 8.3 Defense Export Controls for AI

- **ITAR (International Traffic in Arms Regulations)** : AI systems specifically designed for military use
- **Export controls on AI**: Dual-use AI export controls (see Document 07)
- **Security clearance**: AI systems handling classified information require accreditation

## 9. Autonomous Vehicle and Transportation AI

### 9.1 NHTSA Regulation

**Current NHTSA framework for autonomous vehicles:**

**Standing General Order (June 2021, updated 2024):**
- Manufacturers/operators of SAE Level 2+ ADS-equipped vehicles must report crashes
- Reporting within 10 days for incidents involving injury, death, or airbag deployment
- Monthly reports for less severe incidents
- Data collected: Vehicle speed, ADS engaged/not, crash location, injury severity

**Automated Vehicle Comprehensive Plan (AVCP — 2023):**
- Safety framework for ADS (Automated Driving Systems)
- ADS safety self-assessments (voluntary, but expected)
- NHTSA authority to investigate and recall defective ADS software

**FMVSS (Federal Motor Vehicle Safety Standards) for AVs:**
- **NPRM (April 2025)** : Proposed rulemaking on AV compliance with existing FMVSS
- Key issues: Steering wheel requirement, driver monitoring, crashworthiness with novel seating
- NHTSA exemption authority allows limited AV deployment without full FMVSS compliance

**NHTSA Enforcement:**
- **Tesla Autopilot/FSD investigations** (ongoing since 2021): Multiple crashes involving Autopilot; recall of 2M+ vehicles (December 2023, expanded 2024)
- **Cruise investigation**: Pedestrian crash in San Francisco (October 2023); Cruise recalled fleet, halted operations, fined $1.5M
- **Waymo**: Investigation into unexpected ADS behavior (2024); no enforcement action to date
- **Zoox**: Investigation into self-certification accuracy (2024)

### 9.2 State AV Regulation

- **California**: DMV permits for AV testing and deployment; CPUC for commercial passenger service
- **Arizona**: Laissez-faire approach; AV testing allowed statewide
- **Texas**: AV-friendly framework; testing and deployment permitted
- **Nevada**: Early adopter of AV legislation
- **New York**: More restrictive; testing permits required

**State patchwork problem**: AV developers must navigate 50+ state regulatory frameworks for nationwide deployment. Federal preemption of AV regulation remains contested.

### 9.3 Aviation AI Regulation (FAA/EASA)

**FAA AI Roadmap (2024):**
- AI for aircraft design, maintenance, and operations
- Certification framework for AI in safety-critical aviation functions
- AI in air traffic management (NextGen)
- EASA (European Union Aviation Safety Agency) AI roadmap (2023, updated 2025)

**EASA AI Framework (2024):**
- **Level 1**: AI assisting human (training, maintenance support) — standard certification
- **Level 2**: AI collaborating with human (pilot advisory systems, crew assistance) — enhanced certification
- **Level 3**: AI with full autonomy (pilotless aircraft) — novel certification, not yet approved
- **Trustworthiness requirements**: Ethical, safe, secure, transparent, auditable

**Key aviation AI regulatory challenges:**
- Certification of AI systems that continue learning post-deployment
- Verification and validation of AI in safety-critical functions
- Human factors: Pilot-AI teaming, automation bias, workload
- Cybersecurity: AI attack surfaces in connected aircraft
- International harmonization: FAA-EASA certification coordination for global aviation

### 9.4 Maritime and Rail AI

- **Maritime**: IMO (International Maritime Organization) framework for Maritime Autonomous Surface Ships (MASS)
  - Interim guidelines for MASS trials (2024)
  - MASS Code development (expected 2028)
- **Rail**: FRA (Federal Railroad Administration) — AI for positive train control and automation
  - Automatic train operation (ATO) frameworks

## 10. Cross-Sectoral Regulatory Themes

### 10.1 AI Bias and Fairness Testing Across Sectors

| Sector | Fairness Standard | Testing Requirement | Regulator |
|--------|------------------|-------------------|-----------|
| Healthcare | Demographic parity in outcomes | Clinical validation subgroup analysis | FDA |
| Lending | Equal credit opportunity | Adverse impact analysis, 4/5ths rule | CFPB, DOJ |
| Employment | Title VII disparate impact | Four-fifths rule, selection procedure analysis | EEOC |
| Insurance | Non-discrimination | Rate filing review, market conduct exams | State insurance depts |
| Criminal justice | Equal protection, due process | Validation studies, bias testing | DOJ, courts |
| Education | Equal access | FERPA reviews, OCR investigations | ED, OCR |

### 10.2 Automation Bias and Human Oversight

A consistent theme across sectoral regulation: AI systems must include meaningful human oversight. Key components:

1. **Understandable output**: Humans must be able to interpret AI recommendations
2. **Override capability**: Humans must be able to override or reject AI decisions
3. **Monitoring**: Humans must monitor AI performance and detect failures
4. **Training**: Human overseers must be trained on AI capabilities and limitations
5. **Accountability**: Humans remain accountable for decisions, even if AI-assisted

### 10.3 Explainability and Transparency

Different sectors have different explainability requirements:

- **High (lending, criminal justice)**: Must provide specific, understandable reasons for decisions
- **Medium (medical diagnosis, hiring)**: Must provide relevant factors considered
- **Low (recommendation systems, content moderation)**: Basic disclosure sufficient

### 10.4 Third-Party AI Vendor Management

Across sectors, regulators increasingly hold the organization using AI (not just the AI developer) responsible. Key requirements:
- **Due diligence**: Evaluate vendor AI systems before adoption
- **Contracts**: Ensure vendor agreements include compliance obligations
- **Monitoring**: Continuously monitor vendor AI performance
- **Audits**: Right to audit vendor AI systems
- **Liability**: Organization remains liable for vendor AI failures

## 11. Sectoral AI Regulation by Jurisdiction: Comparative Table

| Sector | EU | US | UK | China |
|--------|----|----|----|-------|
| **Healthcare** | MDR/IVDR + AI Act high-risk | FDA SaMD framework | MHRA (AI as medical device pathway) | NMPA device registration |
| **Financial** | DORA, AI Act, MiFID II | SEC, CFPB, OCC sectoral | FCA AI guidance | PBOC, CSRC sectoral |
| **Legal** | AI Act high-risk admin of justice | State ethics rules | SRA guidance | Ministry of Justice controls |
| **Education** | AI Act high-risk access/admissions | FERPA, state laws | DfE guidance | MOE AI education policies |
| **HR/Employment** | AI Act high-risk + GDPR Article 22 | EEOC, state laws | ACAS guidance | Labor law + social credit |
| **Insurance** | Solvency II, IDD, AI Act | State depts, NAIC | PRA/FCA | CIRC (insurance regulator) |
| **Defense** | Member state + NATO | DoD Directive 3000.09, NDAA | MOD AI Centre | PLA regulations |
| **Transport (AV)** | UN Regulation R157 (L3), national | NHTSA + states | CAV legislation | MIIT AV classification |
| **Aviation** | EASA AI roadmap | FAA AI roadmap | CAA AI guidance | CAAC rules |

## 12. Emerging Sectoral AI Issues (2026)

1. **AI in legal services**: Generative AI for legal document creation raises unauthorized practice of law concerns
2. **AI in clinical trials**: AI-designed drugs and clinical trial protocols — regulatory pathways evolving
3. **AI in elections**: Regulation of AI-generated political ads (FEC, state laws, EU Code of Practice)
4. **AI in journalism**: Disclosure requirements for AI-generated news (EU AI Act, various state laws)
5. **AI in policing**: Facial recognition, predictive policing, body camera AI analytics — civil liberties oversight
6. **AI in child safety**: COPPA updates for AI toys and education apps
7. **AI in social media moderation**: Transparency requirements for AI content moderation decisions (EU DSA)

---

**Document metadata**: Created June 2026. Part of the AI Regulation & Antitrust knowledge base. For horizontal regulatory frameworks, see Documents 02 (EU AI Act), 03 (US Regulation), and 04 (China Governance).

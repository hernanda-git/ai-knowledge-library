# 03 — US AI Regulation: Executive Action, State Laws, and the Sectoral Patchwork

## 1. Introduction: The American Approach to AI Governance

The United States has not enacted comprehensive federal AI legislation comparable to the EU AI Act. Instead, AI governance in the US proceeds along three parallel tracks: (1) executive branch action leveraging existing statutory authority, (2) state-level legislative activism filling the federal vacuum, and (3) sectoral regulation by federal agencies within their existing mandates.

This approach reflects deliberate policy choices rooted in American political economy: skepticism of centralized bureaucratic control, preference for market-led innovation, separation of powers between federal and state governments, and a regulatory culture that emphasizes ex-post enforcement over ex-ante licensing.

As of mid-2026, this framework is showing signs of strain. The patchwork of state laws creates significant compliance burdens for multistate operators. Federal agencies' existing legal authorities are stretched thin when applied to novel AI challenges. And the absence of national standards leaves US AI governance fragmented compared to the EU and China.

### 1.1 The Three-Track System

```
Track 1: Executive Branch Action
├── Executive Order 14110 (Oct 2023)
├── OMB Memoranda (agency AI governance)
├── National AI Initiative Act (statutory, 2020)
└── 2025-2026 EO expansions

Track 2: State Legislation
├── Colorado AI Act (SB 24-205)
├── California SB 1047
├── New York AI Laws
├── Texas AI Safety Act
└── 15+ other states with pending/completed AI laws

Track 3: Sectoral Federal Regulation
├── FTC (consumer protection, competition)
├── DOJ (civil rights, antitrust)
├── FDA (medical AI / SaMD)
├── SEC (algorithmic trading, AI disclosures)
├── FCC (AI in communications, robocalls)
├── EEOC (employment AI, Title VII)
├── CFPB (fintech AI, fair lending)
├── HHS (healthcare AI, HIPAA)
├── DoD/NIST (defense, standards)
├── NHTSA (autonomous vehicles)
├── FAA (aviation AI)
├── FEC (AI in political ads)
└── HUD (housing AI, Fair Housing Act)
```

## 2. Executive Action: Setting Federal AI Policy

### 2.1 Executive Order 14110 (October 30, 2023)

"Safe, Secure, and Trustworthy Development and Use of Artificial Intelligence"

This landmark executive order established Biden administration AI policy across eight priority areas:

**2.1.1 Safety and Security Standards**
- Required developers of dual-use foundation models (threshold: training compute >10^26 FLOPs, or models with biological/cyber capabilities) to share safety test results with the US government
- Directed NIST to develop standards for AI safety testing (red-teaming, adversarial testing)
- Required cloud providers to report foreign access to AI training compute
- Initiated development of AI watermarking and content provenance standards (C2PA, DHS Center for Homeland Security)
- Established the AI Safety and Security Board at DHS

**2.1.2 Privacy Protection**
- Congress recommendations for privacy legislation
- Evaluation of how agencies collect and use commercially available information (including AI training data)
- Federal privacy-preserving AI research (differential privacy, federated learning)

**2.1.3 Equity and Civil Rights**
- DOJ guidance on AI discrimination (housing, employment, criminal justice)
- Federal agency assessment of AI bias in benefit programs
- Training for federal civil rights offices on AI impacts

**2.1.4 Consumer Protection, Healthcare, Education**
- HHS AI task force (healthcare AI safety, bias)
- VA AI strategy for veterans' benefits
- Education Department AI report (algorithmic fairness, student privacy)
- FTC consumer protection guidance (AI in commerce)

**2.1.5 Support for Workers**
- Report on AI's labor market impacts
- Principles for AI in the workplace (worker surveillance, job displacement)
- Federal contractor AI transparency

**2.1.6 Innovation**
- National AI Research Resource (NAIRR) pilot — government-funded compute access for researchers
- AI talent surge (streamlined hiring for AI expertise)
- Expansion of AI research grant programs (NSF, NIH, DARPA)

**2.1.7 Federal Agency AI Use**
- OMB Memorandum M-24-10: Agency AI use requirements
- Chief AI Officers in every agency
- AI governance boards for risk management
- Inventory of AI use cases published publicly

**2.1.8 International Leadership**
- Bilateral AI agreements (with UK, Japan, South Korea)
- Multilateral engagement (G7, OECD, UN)
- AI export controls coordination

### 2.2 Post-Biden Developments (2025-2026)

The 2025-2026 period saw significant evolution of the EO framework:

- **January 2025 EO expansion**: Extended reporting requirements to AI models exceeding 10^25 FLOPs (lowering threshold); added AI-generated content labeling requirements for federal procurement
- **June 2025 EO on AI and National Security**: Mandated DHS AI infrastructure protection; expanded CFIUS AI reviews; established AI National Security Council
- **January 2026 EO on AI Safety**: Required federal agencies to only deploy AI systems that have passed NIST-validated safety testing; established independent AI Incident Reporting Database
- **March 2026 agency rulemaking**: DHS final rule on AI in immigration decisions; DOJ final rule on Title VII AI discrimination

### 2.3 Limitations of Executive Action

Executive orders cannot appropriate funds, create permanent statutory agencies, or establish binding standards that survive administration changes. Key vulnerabilities:
- **Reversibility**: A new administration could rescind EOs on Day 1
- **Funding constraints**: NAIRR, AI Safety Institute, and other programs depend on annual appropriations
- **No private right of action**: Individuals harmed by AI cannot sue under EO provisions alone
- **Limited extraterritorial effect**: Federal AI rules apply to federal agencies and contractors, not to private sector broadly

## 3. State-Level AI Legislation: The Federalism Patchwork

With federal comprehensive legislation stalled, states have become the primary laboratories of AI regulation.

### 3.1 Colorado AI Act (SB 24-205)

**Enacted**: May 17, 2024
**Effective**: February 1, 2026 (delayed from original date)
**First major US comprehensive AI law**

**Scope**: Applies to developers and deployers of "high-risk AI systems" making consequential decisions in:
- Employment (hiring, promotion, termination, compensation)
- Education (admissions, evaluation, disciplinary)
- Housing (rental decisions, eviction)
- Healthcare (treatment decisions, insurance determinations)
- Financial services (credit, lending, insurance underwriting)
- Essential government services (benefits, law enforcement)

**Key requirements:**

1. **Risk management framework**: Deployers must implement a risk management program (may follow NIST AI RMF or equivalent)
2. **Impact assessments**: Must conduct annual AI impact assessments addressing:
   - Purpose, intended use, and benefits
   - Data governance and privacy protections
   - Bias testing and mitigation measures
   - Transparency and human oversight
   - Risk analysis (probability and severity of algorithmic discrimination)
   - Mitigation measures implemented
3. **Consumer notice**: Individuals must receive notice when a high-risk AI system makes a consequential decision, including:
   - Purpose of the AI system
   - Contact information for deployer
   - Right to opt out (where feasible)
   - Right to correct inaccurate data
   - Right to appeal (challenge adverse decisions)
4. **Transparency**: Public disclosure of AI systems used for consequential decisions

**Enforcement**: 
- Colorado AG has exclusive enforcement authority (no private right of action)
- Penalties: Injunctive relief, civil penalties up to $10,000 per violation
- 90-day cure period before penalties apply

**Significance**: First state comprehensive AI law; model for other states; establishes core framework (risk assessments + transparency + consumer rights)

### 3.2 California SB 1047 (Safe and Secure Innovation for Frontier AI Models Act)

**Passed**: September 4, 2024
**Effective**: January 1, 2026 (with phased implementation)

**Scope**: Developers of "covered models" — AI models trained using >10^26 FLOPs of compute (or models that exceed this threshold through fine-tuning/merging). Also covers "covered model derivatives" (fine-tuned versions).

**Key requirements:**

1. **Pre-deployment safety testing**: Before initial training of a covered model, developer must:
   - Implement a safety and security protocol
   - Conduct capability evaluations (red-teaming, adversarial testing)
   - Assess "critical harms" including CBRN weapons, cyberattacks, mass casualties, massive property destruction
2. **Shutdown capability**: Developer must ensure the model can be fully shut down
3. **Emergency response**: Must report safety incidents to CA AG within 72 hours
4. **Annual third-party audits**: Independent auditors must assess safety protocol compliance
5. **Whistleblower protections**: Legal protections for employees who report AI safety violations
6. **Public transparency**: Summary safety reports publicly accessible

**CalCompute** (Section 107):
- Public cloud computing cluster for AI research
- 50,000+ GPU equivalent capacity
- Access for California researchers, startups, nonprofits
- Prioritizes safety research, fairness, and transparency

**Enforcement:**
- California Attorney General enforcement
- Civil penalties: $10M - $50M per violation for first offense; higher for subsequent
- Injunctive relief (can halt training or deployment)
- Whistleblower retaliation: treble damages + attorney fees
- Private right of action for whistleblower retaliation only

**Implementation timeline:**
- January 1, 2026: Core provisions effective
- January 1, 2027: CalCompute operational
- January 1, 2027: Third-party audit requirements effective

### 3.3 New York AI Regulation

**Executive Order 30 (October 2024)**:
- NY state agency AI use governance
- AI policy for state procurement
- Algorithmic fairness assessment for state-used AI

**NY AI Employment Law (2025)**:
- Automated employment decision tools must undergo bias audit before use
- Results of bias audit must be published
- Notice to candidates/employees when AI used in hiring
- Independent auditor requirement for bias audits
- NYC Local Law 144 expanded statewide

**NY Biometric Privacy Act (2025, pending)**:
- Consent requirements for biometric data collection
- Private right of action ($1,000 - $5,000 per violation)
- Similar to Illinois BIPA

### 3.4 Texas AI Safety Act (2025)

- High-risk AI in critical infrastructure (grid, water, telecom, healthcare)
- Mandatory safety testing before deployment
- State licensing system for high-risk AI operators
- Penalties up to $10,000 per day

### 3.5 Other Notable State Laws

| State | Law | Key Provisions | Effective |
|-------|-----|---------------|-----------|
| **Illinois** | AI Transparency Act | AI-generated content labeling; prohibition on deceptive AI in political ads | 2025 |
| **Illinois** | Artificial Intelligence Video Interview Act | Notice + consent for AI hiring interviews | Updated 2025 |
| **Maryland** | AI Bias Testing Law | Employment AI bias testing; adverse impact analysis | 2026 |
| **Connecticut** | CT AI Act (SB 1103) | Risk assessments for high-risk AI; consumer rights | 2026 |
| **Washington** | WA AI Transparency Act | AI in government; deepfake labeling | 2025 |
| **Massachusetts** | MA AI Leadership Act | State AI commission; procurement rules | 2025 |
| **Vermont** | VT AI Consumer Protection Act | AI in insurance pricing; disclosure requirements | 2026 |
| **Hawaii** | HI AI Ethics Act | State AI use framework; procurement standards | 2025 |

### 3.6 State Preemption Debate

A critical question: Can federal law preempt state AI regulation? The current legal framework:

- **No automatic preemption**: States have broad police powers to regulate health, safety, and welfare
- **Conflict preemption**: If federal law directly conflicts with state law, federal law prevails
- **Field preemption**: If Congress occupies the entire regulatory field, states are preempted

As of 2026, no federal comprehensive AI law exists to preempt state laws. However, sector-specific federal laws already preempt certain state AI regulation:
- **Banking**: National Bank Act preempts state lending AI regulation for national banks
- **Insurance**: McCarran-Ferguson Act leaves insurance regulation to states (so state AI rules for insurance are actually *preserved*)
- **Employment**: Federal employment laws (Title VII, ADA, ADEA) do not preempt state law — they set floor, not ceiling

**Proposed federal AI preemption** (in draft legislative frameworks): Some proposals preempt state AI laws that "directly and materially conflict" with federal standards, while preserving state unfair competition and consumer protection authority. This is hotly contested.

## 4. Federal Agency Sectoral Regulation

### 4.1 Federal Trade Commission (FTC)

The FTC is the most active federal enforcer on AI, using Section 5 (unfair/deceptive practices) authority:

**FTC AI Enforcement Actions (2023-2026):**
- **DoNotPay**: Settled charges that AI-powered "robot lawyer" misrepresented legal capabilities ($193,000 penalty)
- **Rite Aid**: Biased facial recognition in stores; FTC order requiring 5-year AI compliance monitoring, deletion of improperly collected biometric data
- **Amazon "Alexa"**: Children's privacy (COPPA) — AI voice recordings kept indefinitely; $25M penalty
- **X (Twitter)**: Deceptive AI claims in advertising targeting; FTC consent order compliance
- **OpenAI**: Investigation into ChatGPT's ability to generate false/misleading information about individuals
- **Automated hiring AI providers**: Multiple investigations into deceptive AI hiring claims

**FTC Guidance (2023-2026):**
- "Keep Your AI Claims in Check" (Feb 2023): Advertising/enforcement guidance
- "AI and the FTC" (ongoing blog series)
- "Combatting Voice Cloning Fraud" (2024): Telemarketing AI restrictions
- Staff report on AI in commerce (2025)

**FTC Rulemaking:**
- Proposed rule on commercial surveillance (2024, includes AI data practices)
- Trade regulation rule on AI-generated impersonation (2025)
- COPPA amendments for AI children's products (2025)

**FTC Competition Authority:**
- Section 5 (unfair methods of competition) used for AI market investigations
- Antitrust enforcement in AI markets (see Document 05)

### 4.2 National Institute of Standards and Technology (NIST)

NIST is the primary federal standards body for AI trustworthiness:

**AI Risk Management Framework (AI RMF 1.0, January 2023):**
- **Govern**: Establish AI governance structures, risk culture
- **Map**: Contextualize AI risks in specific use cases
- **Measure**: Quantitative and qualitative AI risk assessment
- **Manage**: Prioritize, respond to, and treat AI risks

**AI Safety Institute (AISI, established by EO 14110):**
- Research on AI safety testing methodology
- Model evaluation standards (red-teaming, capability testing)
- International coordination on safety standards
- Pre-deployment testing of frontier models (voluntary, then mandatory under 2026 EO)

**Major NIST AI Publications:**
- AI RMF Generative AI Profile (July 2024)
- AI RMF Playbook (living document)
- AI Accountability Framework
- Synthetic Content Authentication Standards
- Adversarial Machine Learning: Prevention and Detection (February 2024)
- AI Bias Testing Standards (September 2025)
- AI Safety Evaluation Guidelines (January 2026)

### 4.3 Equal Employment Opportunity Commission (EEOC)

AI in employment decisions is a major enforcement priority:

**EEOC AI Guidance:**
- "The Americans with Disabilities Act and the Use of Software, Algorithms, and Artificial Intelligence to Assess Job Applicants and Employees" (May 2022)
- "The Use of Artificial Intelligence in Employment Selection" (May 2023)
- Technical assistance on algorithmic fairness under Title VII (ongoing)

**Key Enforcement Theory:**
- Employers using AI for hiring, promotion, or termination are liable for disparate impact discrimination under Title VII (Civil Rights Act of 1964)
- "Fourth industrial revolution" initiative targeting AI hiring discrimination
- Investigating third-party AI vendors as "employment agencies" under Title VII
- Focus areas: resume screening, video interview analysis, personality tests, worker surveillance

**Pending cases (2026):**
- EEOC v. [Hiring AI Platform] — alleged age discrimination in resume filtering
- EEOC v. [Large Employer] — alleged race discrimination in AI-powered performance evaluation
- Multiple investigations into AI-driven worker surveillance tools (ADA reasonable accommodation claims)

### 4.4 Food and Drug Administration (FDA)

Regulation of AI in medical devices and healthcare:

**FDA AI/ML Framework:**
- "Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD) Action Plan" (January 2021, updated 2024)
- "Clinical Decision Support Software" final guidance (September 2024)
- "Marketing Submission Recommendations for a Predetermined Change Control Plan for AI/ML-Enabled Device Software Functions" (April 2025)

**Key Provisions:**
- AI/ML-based SaMD requires premarket approval (PMA) or premarket notification (510(k))
- Predetermined Change Control Plans (PCCP): Pre-authorize iterative AI model updates without re-submission — revolutionary framework for continuously learning AI
- Total Product Lifecycle (TPL) approach: Monitoring AI device performance post-market
- Real-world performance monitoring requirements
- Cybersecurity requirements for AI medical devices

**Approved AI/ML Medical Devices (cumulative):**
- ~1,000+ AI/ML-enabled devices cleared/approved by FDA as of 2026
- Majority in radiology (77%), followed by cardiology (12%), neurology (3%)
- Notable approvals: AI stroke detection (Viz.ai), AI mammography screening, AI sepsis prediction

**Enforcement:**
- FDA warning letters for unapproved AI-based clinical decision support
- Pre-market review of AI bias and generalization across demographic groups
- Post-market surveillance orders for AI devices with safety signals

### 4.5 Securities and Exchange Commission (SEC)

AI in financial markets and corporate disclosures:

**SEC AI Regulation:**
- **Proposed Rule (2023)**: "Predictive Data Analytics" — requirements for broker-dealers and investment advisers using AI for investor interaction
  - Evaluate conflicts of interest in AI-driven recommendations
  - Disclosure of AI use in investment advice
  - Recordkeeping and testing of AI models
- **Final Rule "AI Conflicts" (2025)**: Eliminates or neutralizes conflicts in AI-driven investment advice
- **Regulation Best Execution**: AI trading algorithms must achieve best execution for clients
- **Market Access Rule**: Risk controls for AI algorithmic trading

**SEC Enforcement:**
- Enforcement actions against AI-washing (fake AI claims in securities filings) — multiple settled cases (2024-2026)
- Investigation into AI-driven trading malfunctions (flash crashes, algo failures)
- Disclosure enforcement: Public companies must disclose material AI risks (per existing disclosure obligations)
- Examination priority: AI use in compliance and oversight functions

### 4.6 Consumer Financial Protection Bureau (CFPB)

AI in lending, credit, and financial services:

**CFPB AI Guidance and Enforcement:**
- **Circular 2022-03**: Adverse action notification requirements apply to AI/credit decisions (ECOA, Regulation B)
- **Circular 2023-03**: Unfair, deceptive, or abusive AI practices (UDAAP)
- **Section 1033 Rulemaking**: Open banking, consumer data rights — implications for AI personalization
- **Fair Credit Reporting Act**: AI credit scoring models must maintain reasonable procedures for accuracy
- **Algorithmic auditing**: Expectations for model validation, bias testing, and fairness monitoring

**Key enforcement areas:**
- AI-based credit discrimination (alleged race discrimination in AI credit scoring)
- Black-box AI underwriting models (unable to provide adverse action reasons)
- AI chatbot failures in consumer financial services
- AI tenant screening and housing discrimination
- AI rent-setting algorithms (alleged price fixing)

### 4.7 Department of Defense (DoD)

AI in defense and national security:

**DoD AI Strategy and Policy:**
- **DoD Data, Analytics, and AI Adoption Strategy** (2023, revision): Accelerated AI adoption, responsible AI principles
- **JAIC** (Joint Artificial Intelligence Center) → **CDAO** (Chief Digital and AI Office)
- **Responsible AI (RAI) Toolkit and Guidelines**: DoD AI ethical framework (based on 2020 AI Ethical Principles)
- **Autonomous Weapons Systems Policy**: DoD Directive 3000.09 (2012, updated 2023): Human oversight of lethal autonomous weapons
- **AI Safety**: DoD AI incident reporting system

**Key programs:**
- Project Maven (computer vision for ISR)
- Air Combat Evolution (ACE) — AI-piloted F-16
- Project Convergence (joint force AI command and control)
- CDAO algorithmic warfare competitions

**Congressional oversight:**
- National Defense Authorization Act (NDAA) AI provisions annually
- House Armed Services Committee AI Panel
- Section 227 requirements: AI safety, human control, and bias testing for DoD AI
- Tracking AI in lethal autonomous weapons (LAWS)

### 4.8 National Highway Traffic Safety Administration (NHTSA)

Autonomous vehicle regulation:

**NHTSA AV Framework:**
- **Standing General Order (2021, updated)**: Reporting requirements for AV crashes
- **AVCC (Automated Vehicles Comprehensive Plan)** : Safety framework for AVs
- **FMVSS (Federal Motor Vehicle Safety Standards)**: Rulemaking on AV compliance (April 2025 NPRM)
- **ADS (Automated Driving Systems) safety framework**: Voluntary safety self-assessments
- **Enforcement**: Defect investigations, recall authority for AV software

**Key issues (2026):**
- Level 4 AV deployment (Waymo, Cruise, Zoox, Amazon Zoox) in limited geographies
- NHTSA investigations into AV crashes (software failures, unexpected behavior)
- Preemption of state AV laws for federally-regulated vehicles
- Cybersecurity standards for AV AI systems
- AI-enabled driver monitoring systems (regulation/reliability)

## 5. Federal Comprehensive AI Legislation: Prospects

### 5.1 Key Bills Under Consideration

| Bill | Status | Key Provisions |
|------|--------|----------------|
| Bipartisan Framework for AI Act | Discussion draft (2024-2025) | Risk-based (NIST-aligned), preemption limited, sectoral agency enforcement, safe harbor for research |
| SAFE Innovation for Frontier AI Act | Introduced 2024, stalled | Safety testing for frontier models, AISI statutory authority, mandatory incident reporting |
| Algorithmic Accountability Act | Reintroduced 2024 | Impact assessments, bias testing, FTC enforcement, private right of action |
| CREATE AI Act (NAIRR statutory) | Pending | National AI Research Resource, $2.5B authorization |
| AI Transparency and Disclosure Act | Introduced 2025 | AI labeling, deepfake disclosure, political ad AI disclosure |
| National AI Commission Act | Passed House (2025) | Establish blue-ribbon commission on comprehensive AI regulation |

### 5.2 Political Dynamics

- **Friction points**: Content moderation liability (Section 230); facial recognition; worker surveillance; antitrust; preemption scope
- **Industry positions**: Tech industry prefers federal preemption of state laws (to reduce patchwork compliance); civil society prefers stronger baseline protections
- **Election cycle**: 2026 midterms, 2028 presidential election — comprehensive AI legislation unlikely before 2029 unless a major AI incident creates legislative urgency

## 6. Comparative Analysis: US vs. EU vs. China

| Dimension | US Approach | EU Approach | China Approach |
|-----------|-------------|-------------|----------------|
| **Philosophy** | Innovation-first, ex-post enforcement | Precautionary principle, ex-ante compliance | State control, social stability |
| **Scope** | Sectoral patchwork | Horizontal regulation | Comprehensive + sectoral |
| **Risk classification** | Agency-specific, NIST-informed | Four-tier statutory | Content-based, state security |
| **Penalties** | Agency-specific (up to millions) | Up to 7%/€35M global | Variable (revocation, criminal) |
| **Extraterritorial** | Limited | Broad (outputs used in EU) | Extensive (data localization) |
| **Standards** | NIST AI RMF (voluntary) | Harmonized standards (mandatory via presumption) | National standards (GB) |
| **Enforcement** | Fragmented (multiple agencies) | Centralized (AI Office) + national | Centralized (CAC, MIIT, SAMR) |
| **Private right of action** | Limited (agency-dependent) | Limited (aimed at GDPR model) | None |
| **Preemption** | No comprehensive preemption | Full EU-wide harmonization | National unity |

## 7. US AI Regulation Compliance Matrix

For organizations operating in the US, the following matrix maps regulatory obligations across common AI use cases:

| AI Use Case | Primary Federal Regulator | State Requirements | Key Obligations |
|-------------|--------------------------|-------------------|-----------------|
| Hiring AI | EEOC, DOJ | CO, CA, NY, IL, MD | Bias audit, adverse impact analysis, transparency |
| Medical AI (SaMD) | FDA | None (preempted) | Pre-market approval/clearance, PCCP, post-market |
| Credit AI (lending) | CFPB, OCC, Fed | CA, CO, NY | Adverse action reason, ECOA, model validation |
| Insurance AI | State insurance depts | NAIC model, state laws | Fairness testing, rate filing, discrimination prohibition |
| Content AI (GenAI) | FTC | CO, CA, WA, IL | Transparency, IP disclosure, consumer protection |
| Education AI | ED, DOJ | CA, CO, WA | FERPA compliance, differential impact analysis |
| Autonomous Vehicles | NHTSA | Varies by state | Crash reporting, safety assessment, FMVSS |
| Gov't benefits AI | Agency-specific | Various state | Fairness, civil rights, procedural due process |
| Algorithmic trading | SEC, CFTC | None (preempted) | Risk controls, best execution, conflicts disclosure |
| Law enforcement AI | DOJ | Various | Due process, Brady compliance, bias testing |

## 8. NIST AI RMF Implementation Guide

### 8.1 GOVERN Function

- **1.1**: Embed responsible AI into organizational policies
- **1.2**: Establish AI risk management roles (AI governance committee, C-suite accountability)
- **1.3**: Workforce diversity for AI risk assessment
- **1.4**: Document AI system boundaries, data flows, stakeholders
- **1.5**: Establish AI transparency policies
- **1.6**: Create mechanisms for AI risk reporting and escalation

### 8.2 MAP Function

- **2.1**: Develop context (intended use, deployment domain, users)
- **2.2**: Determine AI system capabilities and limitations
- **2.3**: Identify relevant laws, regulations, and standards
- **2.4**: Map system boundaries (inputs, outputs, environment)
- **2.5**: Engage stakeholders (affected communities, civil society)

### 8.3 MEASURE Function

- **3.1**: Identify relevant risk metrics (accuracy, fairness, robustness, transparency)
- **3.2**: Conduct bias testing (demographic parity, equal opportunity, predictive parity)
- **3.3**: Evaluate system performance under stress (adversarial inputs, distribution shift)
- **3.4**: Document risk measurement results
- **3.5**: Assess potential for negative individual and societal impacts

### 8.4 MANAGE Function

- **4.1**: Prioritize risks based on likelihood and impact
- **4.2**: Develop risk response plan (avoid, mitigate, transfer, accept)
- **4.3**: Implement risk mitigation measures
- **4.4**: Monitor risk continuously
- **4.5**: Maintain incident response plan
- **4.6**: Document risk management decisions

## 9. Preparing for US AI Regulation: Practical Steps

### 9.1 Federal Contractor Requirements (Immediate)

If your organization contracts with the US federal government:
- Chief AI Officer appointed
- AI use case inventory published
- NIST AI RMF implementation aligned with OMB M-24-10
- AI incident reporting mechanism established
- EO 14110/2025 expanded compliance (model reporting if applicable)

### 9.2 Colorado AI Act Compliance (Deadline: February 2026)

- [ ] Identify all high-risk AI systems making consequential decisions
- [ ] Implement NIST AI RMF-based risk management program
- [ ] Document annual impact assessments (first due February 2027)
- [ ] Prepare consumer notice templates
- [ ] Establish complaint/appeal mechanism for adverse decisions
- [ ] Training for personnel on AI governance obligations
- [ ] Review vendor/third-party AI provider compliance

### 9.3 California SB 1047 Compliance (If Covered)

- [ ] Determine if model compute exceeds 10^26 FLOPs threshold
- [ ] Prepare safety and security protocol
- [ ] Conduct pre-training capability evaluations
- [ ] Establish emergency shutdown capability
- [ ] Set up 72-hour incident reporting process
- [ ] Contract annual third-party auditor
- [ ] Implement whistleblower protection policy
- [ ] Prepare public safety summary reports

### 9.4 EEOC Compliance (Ongoing)

- [ ] Audit all AI used for employment decisions (hiring, promotion, termination, compensation)
- [ ] Conduct adverse impact analysis (4/5ths rule for disparate impact)
- [ ] Document alternative selection procedures evaluated
- [ ] Ensure ADA reasonable accommodation process for AI assessments
- [ ] Train HR on AI fairness obligations
- [ ] Review third-party AI vendor compliance with Title VII

## 10. The Road Ahead: US AI Regulation 2027-2029

Key developments to monitor:
- **2026 midterm elections**: Shift in congressional control could change federal AI legislative prospects
- **Major AI incident**: A significant AI-caused harm (financial crash, safety failure, discrimination scandal) could catalyze federal legislation
- **State regulatory maturation**: Colorado and California enforcement actions will set precedents adopted by other states
- **Federal preemption litigation**: Challenges to state AI laws on preemption grounds will shape the regulatory landscape
- **International influence**: EU enforcement actions and UK regulatory developments will inform US approaches through regulatory dialogue
- **AI model capability jumps**: New capabilities (general-purpose robotics, AI-driven science, autonomous agents) will test current legal frameworks

---

**Document metadata**: Created June 2026. Part of the AI Regulation & Antitrust knowledge base. For sector-specific agency regulation, see Document 06 (AI Regulation by Sector). For antitrust, see Document 05.

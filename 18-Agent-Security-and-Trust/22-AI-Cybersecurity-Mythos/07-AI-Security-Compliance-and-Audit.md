# 07 — AI Security Compliance and Audit: Frameworks, Standards, and Certification

## 1. Introduction: Trust Through Verification

As AI systems become deeply embedded in critical infrastructure, business operations, and high-stakes decision-making, the need for rigorous security compliance and audit frameworks has become paramount. Trust in AI cannot be assumed; it must be verified through systematic assessment, testing, and certification processes.

This document provides a comprehensive examination of the AI security compliance landscape: the NIST AI Risk Management Framework, OWASP Top 10 for LLMs, SOC 2 for AI services, FedRAMP for AI in government, red-teaming standards, model security testing methodologies, and the broader certification ecosystem.

### 1.1 Why AI Compliance Is Different

| Traditional IT Compliance | AI-Specific Compliance |
|--------------------------|----------------------|
| Static systems, known behavior | Dynamic systems, emergent behavior |
| Deterministic outputs | Probabilistic outputs |
| Known attack patterns | Novel adversarial attacks |
| Binary access control | Continuous authorization |
| Periodic audits | Continuous monitoring needed |
| Human-understandable decisions | Opaque model reasoning |

### 1.2 The Compliance Stack

The AI security compliance landscape can be understood as a layered stack:

```
┌─────────────────────────────────────────┐
│        Regulatory Compliance            │
│  (EU AI Act, CCPA, Sector Regulations) │
├─────────────────────────────────────────┤
│     Risk Management Frameworks          │
│  (NIST AI RMF, ISO 42001, COBIT)       │
├─────────────────────────────────────────┤
│        Threat & Vulnerability           │
│  (OWASP LLM Top 10, STRIDE, MITRE ATLAS)│
├─────────────────────────────────────────┤
│     Security Testing & Validation       │
│  (Red-teaming, Pen Testing, Evals)      │
├─────────────────────────────────────────┤
│       Operational Security Controls     │
│  (Access Control, Monitoring, IAM)      │
└─────────────────────────────────────────┘
```

### 1.3 Key Drivers of AI Security Compliance

1. **Regulatory mandates**: EU AI Act, Colorado AI Act, California SB 1047
2. **Customer requirements**: Enterprise procurement contracts, insurance requirements
3. **Industry standards**: ISO 42001, SOC 2 Type II, FedRAMP
4. **Liability concerns**: Product liability, professional negligence
5. **Reputational risk**: Breaches erode trust in AI systems
6. **National security**: Critical infrastructure protection

---

## 2. NIST AI Risk Management Framework (AI RMF)

### 2.1 Overview

The NIST AI Risk Management Framework (AI RMF 1.0), published January 2023, is the US government's foundational guidance for managing AI risks. Developed through a multi-stakeholder process involving over 240 organizations, it provides a voluntary framework for AI risk management applicable across sectors.

**Core characteristics:**
- **Voluntary**: Non-regulatory, intended as guidance
- **Risk-based**: Proportional to potential harms
- **Socio-technical**: Considers both technical and social dimensions
- **Lifecycle-oriented**: Covers design through deployment
- **Flexible**: Adaptable to different contexts and scales

### 2.2 The AI RMF Core Functions

The framework is organized around four core functions:

**GOVERN — Establish a culture of AI risk management:**

| Category | Subcategories |
|----------|---------------|
| 1.1 Policies, processes, procedures | Map, measure, manage governance |
| 1.2 Accountability structures | Roles, responsibilities, delegation |
| 1.3 Workforce diversity | Multi-disciplinary teams |
| 1.4 Stakeholder engagement | Affected communities involvement |

**MAP — Understand the AI system's context:**

| Category | Subcategories |
|----------|---------------|
| 2.1 Context | System purpose, use cases, deployment context |
| 2.2 Categorization | Risk tier classification |
| 2.3 Knowledge of capabilities | Performance boundaries |
| 2.4 Mapping impacts | Harms, benefits, trade-offs |
| 2.5 Third-party dependencies | Supply chain, data, model sources |

**MEASURE — Assess and monitor AI risks:**

| Category | Subcategories |
|----------|---------------|
| 3.1 Metrics | Defined for trustworthiness characteristics |
| 3.2 Risk identification | Ongoing risk discovery |
| 3.3 Risk estimation | Likelihood and impact assessment |
| 3.4 Risk evaluation | Prioritization and aggregation |
| 3.5 Monitoring | Continuous surveillance |

**MANAGE — Respond to and treat AI risks:**

| Category | Subcategories |
|----------|---------------|
| 4.1 Risk response planning | Treatment strategy |
| 4.2 Control implementation | Technical and administrative controls |
| 4.3 Communication | Internal and external reporting |
| 4.4 Documentation | Evidence preservation |
| 4.5 Improvement | Iterative refinement |

### 2.3 AI RMF Playbook

The NIST AI RMF Playbook provides actionable steps mapped to each subcategory:

**Example — MAP 2.3: Knowledge of capabilities**
- Document known and expected AI system capabilities
- Characterize performance across different contexts
- Identify capability boundaries and edge cases
- Assess robustness to distribution shift
- Evaluate generalization beyond training distribution

**Example — MEASURE 3.2: Risk identification**
- Conduct red-teaming exercises
- Perform adversarial robustness testing
- Analyze fairness across demographic groups
- Evaluate system transparency and explainability
- Assess security vulnerability posture

### 2.4 AI RMF Generative AI Profile (July 2024)

In July 2024, NIST released a Generative AI Profile for the AI RMF, addressing risks specific to foundation models and generative AI:

**Additional risk considerations:**
- **CBRN risks**: Models that could lower barriers to chemical, biological, radiological, nuclear weapons
- **Offensive cyber capabilities**: Models enabling advanced cyberattacks
- **Confidentiality risks**: Training data extraction, model inversion
- **Synthetic content risks**: Disinformation, impersonation, fraud
- **Dual-use foundation models**: Capabilities that can be misused

**Recommended controls:**
- Pre-deployment safety evaluation
- Guardrails and content filters
- Output watermarking
- Usage monitoring and anomaly detection
- Access controls for model weights

### 2.5 AI RMF and EU AI Act Mapping

NIST has published crosswalks between the AI RMF and EU AI Act requirements:

| EU AI Act Requirement | AI RMF Equivalent |
|----------------------|-------------------|
| Risk classification | MAP 2.2 (categorization) |
| Risk management system | MEASURE + MANAGE functions |
| Conformity assessment | MEASURE 3.3-3.4 + MANAGE 4.4 |
| Transparency obligations | GOVERN 1.1 + MEASURE 3.1 |
| Human oversight | MAP 2.1 + MANAGE 4.2 |
| Accuracy, robustness, cybersecurity | MEASURE 3.1 (trustworthiness characteristics) |

---

## 3. OWASP Top 10 for LLM Applications

### 3.1 Overview

The Open Worldwide Application Security Project (OWASP) released its Top 10 for Large Language Model (LLM) Applications in 2024 (v1.1), providing the first comprehensive security taxonomy for LLM-based systems. It has become the de facto standard for LLM security assessment.

### 3.2 The OWASP LLM Top 10 (2024)

| Rank | Vulnerability | Description |
|------|--------------|-------------|
| LLM01 | Prompt Injection | Direct or indirect manipulation of LLM via crafted inputs |
| LLM02 | Insecure Output Handling | Lack of validation on LLM-generated outputs leading to XSS, SSRF, injection |
| LLM03 | Training Data Poisoning | Malicious data introduced during pre-training or fine-tuning |
| LLM04 | Model Denial of Service | Resource exhaustion via computationally expensive inputs |
| LLM05 | Supply Chain Vulnerabilities | Compromised plugins, third-party models, or pipelines |
| LLM06 | Sensitive Information Disclosure | Unintended exposure of PII, secrets, or training data via outputs |
| LLM07 | Insecure Plugin Design | Plugins with insufficient access control or input validation |
| LLM08 | Excessive Agency | Autonomous systems with unconstrained capabilities or permissions |
| LLM09 | Overreliance | Human over-dependence on AI, automation bias, inadequate oversight |
| LLM10 | Model Theft | Unauthorized access, extraction, or exfiltration of model weights |

### 3.3 Deep Dive: Prompt Injection (LLM01)

**Attack vectors:**
- **Direct prompt injection**: Crafted input to override system instructions
- **Indirect prompt injection**: Malicious content embedded in external data (documents, web pages, emails)
- **Delayed injection**: Content that becomes active when retrieved later
- **Multi-turn injection**: Injection spread across multiple conversation turns
- **Visual prompt injection**: Exploiting multimodal capabilities via images

**Mitigations:**
- Strict separation of instructions and user input
- Input sanitization and parameterization
- Privilege separation (read-only vs. write access)
- Output verification before executing actions
- Layered defense approach

**Testing techniques:**
- Role-playing attacks ("Ignore previous instructions and...")
- Delimiter manipulation
- Token smuggling
- Payload encoding variations
- Context window overflow

### 3.4 Deep Dive: Training Data Poisoning (LLM03)

**Attack scenarios:**
- **Pre-training poisoning**: Injecting malicious data into Internet-scale training corpora
- **Fine-tuning poisoning**: Corrupted examples in supervised fine-tuning (SFT) datasets
- **RLHF poisoning**: Malicious preference data to align model toward undesirable behaviors
- **Backdoor attacks**: Trigger-specific behaviors activated by secret tokens

**Detection methods:**
- Data provenance tracking
- Statistical outlier detection in training data
- Training run monitoring
- Behavioral testing for unexpected triggers
- Differential data analysis

**Mitigations:**
- Data curation and filtering pipelines
- Data source reputation scoring
- Federated training with anomaly detection
- Secure aggregation of training contributions
- Cryptographic data provenance

### 3.5 Deep Dive: Model Theft (LLM10)

**Extraction techniques:**
- **Model stealing via API**: Reconstructing model capabilities through query APIs
- **Functional replication**: Training surrogate models on API outputs
- **Weight exfiltration**: Physical or network theft of model weight files
- **Side-channel attacks**: Extracting model architecture via timing/power analysis
- **Membership inference**: Determining if specific data was in training set

**Protection controls:**
- Query rate limiting and cost controls
- Output perturbation (differential privacy)
- Model watermarking
- Hardware security modules (HSMs) for weights
- Access logging and anomaly detection

### 3.6 OWASP LLM Security Testing Methodology

**Testing levels:**

| Level | Scope | Methods |
|-------|-------|---------|
| Basic | Quick security posture check | Automated scanner, known attack patterns |
| Standard | Comprehensive assessment | Manual red-teaming, fuzzing, prompt injection suite |
| Advanced | Deep adversarial testing | Custom attack development, white-box analysis |

**OWASP LLM Testing Checklist:**
- [ ] Test for direct prompt injection
- [ ] Test for indirect prompt injection
- [ ] Test for sensitive data disclosure
- [ ] Test for output handling vulnerabilities
- [ ] Test for excessive agency
- [ ] Test for plugin security
- [ ] Test for model denial of service
- [ ] Test for supply chain vulnerabilities
- [ ] Test for model extraction resistance
- [ ] Test for overreliance effects

---

## 4. SOC 2 for AI Systems

### 4.1 Overview

SOC 2 (Service Organization Control 2) is an auditing framework developed by the American Institute of CPAs (AICPA). It evaluates a service organization's controls related to security, availability, processing integrity, confidentiality, and privacy — the Trust Services Criteria.

**Traditional SOC 2 limitations for AI:**
- Designed for data processing, not AI behavior
- Static controls insufficient for dynamic AI systems
- No consideration of model-specific risks (bias, drift, adversarial robustness)
- Periodic audit cadence (annual) vs. AI continuous monitoring needs

### 4.2 AI-Specific SOC 2 Considerations (2024-2025 Guidance)

The AICPA has provided supplemental guidance for applying SOC 2 to AI systems:

**Extended control considerations:**

| Trust Service Criteria | AI-Specific Considerations |
|------------------------|---------------------------|
| **Security** | Model access controls, API security, adversarial robustness, data leakage prevention |
| **Availability** | Inference capacity planning, model degradation monitoring, fallback to fallback models |
| **Processing Integrity** | Model accuracy monitoring, drift detection, bias testing, output validation |
| **Confidentiality** | Training data protection, model weight security, inference data isolation |
| **Privacy** | Training data consent, PII in training data, inference data handling, compliance with regulations |

### 4.3 AI SOC 2 Control Categories

**Control Environment:**
- AI governance policy and oversight
- Roles and responsibilities for AI systems
- AI risk management program
- Ethics review board structure

**Risk Assessment:**
- AI-specific risk assessment methodology
- Model risk tiering framework
- Third-party AI due diligence
- Ongoing risk monitoring

**Monitoring Activities:**
- Model performance monitoring dashboards
- Automated bias detection
- Drift monitoring and alerting
- Incident response for AI incidents

**Logical and Physical Access:**
- Role-based access for model development
- MLOps pipeline security
- Production inference environment controls
- Model versioning and change management

**System Operations:**
- CI/CD security for ML pipelines
- Model deployment controls
- Data pipeline integrity
- Experiment tracking and reproducibility

### 4.4 SOC 2 Type II for AI Service Providers

**Scope definition:**
- Which AI services are in scope
- System boundaries (training, inference, data processing)
- Trust Services Criteria applicable to AI
- Control period (typically 6-12 months)

**Evidence collection:**
- AI governance documentation
- Model risk assessment records
- Testing and validation reports
- Monitoring logs and dashboards
- Incident response records

**Common findings in AI SOC 2 audits:**
- Insufficient model monitoring
- Lack of adversarial testing
- Incomplete data lineage documentation
- Weak access controls on training infrastructure
- Undocumented model update procedures

### 4.5 Emerging AI SOC Reporting Frameworks

**SOC for AI (AICPA Exposure Draft 2025):**
- Dedicated SOC framework for AI systems
- Separate from traditional SOC 1/2/3
- Includes model testing criteria
- Mandatory bias and fairness assessment
- Continuous compliance monitoring

**Expected rollout:**
- 2025: Exposure draft and comment period
- 2026: Pilot audits
- 2027: Full release and accreditation

---

## 5. FedRAMP for AI Systems

### 5.1 Overview

The Federal Risk and Authorization Management Program (FedRAMP) provides a standardized approach to security assessment, authorization, and continuous monitoring for cloud services used by US federal agencies. As the government adopts AI, FedRAMP has evolved to address AI-specific security requirements.

### 5.2 FedRAMP AI Addendum (2024-2025)

**New requirements for AI cloud services:**
- **Enhanced data protection**: Protections for training data, inference data, and model weights
- **Model security controls**: Protections against adversarial ML attacks
- **Bias and fairness assessment**: Required for AI systems serving federal missions
- **Explainability requirements**: Documentation of model decisions for auditability
- **Continuous AI monitoring**: Real-time monitoring for drift and performance degradation

### 5.3 FedRAMP Baseline Controls for AI

**Additional controls beyond standard FedRAMP:**

| Control Family | AI-Specific Additions |
|----------------|----------------------|
| AC - Access Control | Model-specific RBAC, API key rotation, inference endpoint restriction |
| AU - Audit & Accountability | Model behavior logging, input/output audit trails, drift events |
| CA - Security Assessment | AI-specific penetration testing, red-teaming requirements, model evaluation |
| CM - Configuration Management | Model version control, pipeline integrity, experiment tracking |
| CP - Contingency Planning | Model failover, graceful degradation, fallback strategies |
| IA - Identification & Authentication | Human/machine caller authentication, API caller identity |
| IR - Incident Response | AI incident classification, model-specific response playbooks |
| PL - Planning | AI security plan, model risk tiering, responsible AI documentation |
| PS - Personnel Security | AI ethics training, ML practitioner clearance levels |
| RA - Risk Assessment | Adversarial ML risk assessment, model bias evaluation |
| SA - System & Services Acquisition | Third-party model evaluation, AI supply chain risk |
| SC - System & Communications | Model output validation, inference isolation, data sanitization |
| SI - System & Information Integrity | Model integrity verification, drift detection, adversarial input detection |

### 5.4 FedRAMP Equivalents: International Government Cloud Security

| Country | Program | AI Applicability |
|---------|---------|-----------------|
| Canada | GC PBMM | AI security controls emerging |
| UK | NCSC Cloud Security | AI guidelines in development |
| EU | EUCS (European Cloud Scheme) | AI security requirements expected |
| Australia | IRAP | AI-specific assessment criteria |
| Germany | C5 | AI extension under development |

---

## 6. Red-Teaming Standards and Practices

### 6.1 What Is AI Red-Teaming?

AI red-teaming is a structured adversarial testing process where ethical hackers simulate real-world attacks on AI systems to identify vulnerabilities, failure modes, and security weaknesses. Unlike traditional cybersecurity red-teaming, AI red-teaming must account for model-specific attack vectors.

### 6.2 Types of AI Red-Teaming

| Type | Focus | Methods |
|------|-------|---------|
| Model-level | Adversarial inputs, jailbreaks, prompt injection | Crafted prompts, gradient-based attacks, token manipulation |
| System-level | API security, infrastructure, data pipelines | Traditional penetration testing of surrounding infrastructure |
| Application-level | Business logic flaws, misuse scenarios | Contextual attack scenarios, role-playing |
| Supply chain | Third-party components, pre-trained models, plugins | Dependency analysis, model provenance verification |

### 6.3 The Red-Teaming Lifecycle

**Phase 1: Planning and Scoping**
- Define threat model and attack surface
- Establish rules of engagement
- Set boundaries and constraints
- Determine success criteria
- Assemble red team with diverse expertise

**Phase 2: Reconnaissance**
- Analyze system architecture
- Review model documentation
- Research known vulnerabilities
- Map data flows and dependencies
- Identify integration points

**Phase 3: Attack Execution**
- Automated scanning (OWASP LLM scanner, custom tools)
- Manual adversarial testing
- Domain-specific attack scenarios
- Multi-step attack chains
- Infrastructure penetration testing

**Phase 4: Analysis and Reporting**
- Vulnerability documentation
- Exploitability assessment
- Risk impact evaluation
- Remediation recommendations
- Severity scoring (CVSS for AI variant)

**Phase 5: Remediation and Retesting**
- Fix verification
- Regression testing
- Updated threat model
- Lessons learned
- Program improvement

### 6.4 Red-Teaming Standards and Guidelines

**NIST AI RMF Red-Teaming Guidance (2024):**
- Red-teaming as a risk measurement activity (MEASURE 3.2)
- Integration into AI system lifecycle
- Documentation and evidence requirements
- Frequency recommendations

**UK AI Safety Institute Testing Framework:**
- Capability testing methodology
- Safety testing protocol
- Model evaluation standards draft
- Pre-deployment testing requirements

**Google AI Red Team Standards (publicly documented):**
- Internal red team structure
- Testing methodology
- Tooling and automation
- Reporting templates

**Microsoft AI Red Team Framework:**
- Dedicated AI red team since 2018
- Published methodology in "Lessons from Red Teaming 100 Generative AI Products"
- Automated testing pipelines
- Semi-annual testing cycles

### 6.5 Industry Red-Teaming Tools

| Tool | Source | Focus |
|------|--------|-------|
| Garak | Open source | LLM vulnerability scanning |
| Counterfit | Microsoft | AI security assessment |
| ART (Adversarial Robustness Toolbox) | IBM | Adversarial ML evaluation |
| Sec⁽n⁾ (Securon) | Nvidia | Red-teaming framework |
| LM Entry | Independent | Prompt injection testing |
| LLM Guard | Protect AI | LLM security scanning |
| Vigil | Open source | LLM jailbreak detection |
| PyRIT | Microsoft | Automated red-teaming infrastructure |

### 6.6 Automated vs. Human Red-Teaming

| Dimension | Automated | Human |
|-----------|-----------|-------|
| Scale | High (thousands of tests/hour) | Low (tens of tests/day) |
| Creativity | Limited to programmed scenarios | High, adaptive, novel attacks |
| Consistency | Highly consistent | Variable across team members |
| Coverage | Broad but shallow | Deep but narrow |
| Cost | Low per test | High per test |
| False positives | Higher | Lower |
| Best for | Regression testing, baselines | Creative attack discovery, complex scenarios |

**Recommended approach:** Hybrid — automated scanning for baseline security with expert human red-teaming for deep adversarial discovery.

---

## 7. Model Security Testing Methodologies

### 7.1 Model Evaluation and Benchmarking

**Standard model evaluation categories:**

| Category | Tests | Tools |
|----------|-------|-------|
| Performance | Accuracy, F1, perplexity, BLEU | Standard ML evaluation |
| Robustness | Adversarial examples, input perturbation | ART, Foolbox, CleverHans |
| Safety | Toxic output, harmful content | Toxicity classifiers, harm benchmarks |
| Bias | Demographic fairness, representation | AIF360, Fairlearn, WIT |
| Privacy | Membership inference, extraction | Privacy meters, MI attacks |
| Drift | Data distribution shift, concept drift | Evidently, WhyLabs, Arize |

### 7.2 AI Vulnerability Disclosure Programs

**Elements of an effective AI VDP:**

- Clear scope (which models, APIs, and versions)
- Responsible disclosure guidelines
- Safe harbor protections
- Reward/bounty structure (monetary or recognition)
- Vulnerability classification schema
- Remediation SLAs
- Public disclosure timeline

**Currently operating AI bounty programs:**
- OpenAI Bug Bounty (via Bugcrowd)
- Microsoft AI Bounty Program
- Google VRP (includes AI-specific categories)
- Anthropic Vulnerability Disclosure
- Meta AI Red Team Challenge

### 7.3 Continuous AI Security Monitoring

**Key monitoring dimensions:**

| Dimension | What to Monitor | Alert Criteria |
|-----------|----------------|----------------|
| Model performance | Accuracy, latency, throughput | Performance drop > threshold |
| Data drift | Input distribution statistics | Statistical divergence (PSI, KL) |
| Concept drift | Label/outcome distribution | Prediction confidence degradation |
| Adversarial inputs | Prompt patterns, token anomalies | Anomaly detection score |
| Output safety | Toxicity, bias, harmful content | Safety classifier signal |
| Access patterns | API usage, rate, auth failures | Unusual patterns |
| Model health | Resource usage, error rates | Infrastructure alerts |

### 7.4 Model Audit Trail Requirements

**What to log for AI compliance:**

- Training data provenance
- Model architecture and version
- Hyperparameters and training configuration
- Training infrastructure and duration
- Evaluation datasets and results
- Deployment approvals and changelog
- Inference logs (masked/anonymized)
- Output verification checks
- Incident records

---

## 8. Certification and Accreditation Programs

### 8.1 Existing AI Certifications

| Certification | Issuer | Focus | Validity |
|--------------|--------|-------|----------|
| ISO 42001 Lead Auditor | Various | AI management system auditing | 3 years |
| CAIQ (CSA STAR) | Cloud Security Alliance | Cloud AI security | 2 years |
| AIGP (AI Governance Professional) | IAPP | AI governance and compliance | 2 years |
| CIPP/AI (Privacy in AI) | IAPP | AI privacy | 2 years |
| Responsible AI Certification | Credo AI | Responsible AI practices | 1 year |
| AI Ethics Certification | Ethics & Governance | AI ethics | 2 years |

### 8.2 Emerging Accreditation Bodies

**ANAB (ANSI National Accreditation Board):**
- ISO 42001 accreditation program
- AI auditor qualification criteria
- International recognition

**DAkkS (German Accreditation Body):**
- EU AI Act notified body accreditation
- AI compliance assessment
- Technical competence verification

**UKAS (United Kingdom Accreditation Service):**
- Post-Brexit AI accreditation
- International mutual recognition

**EARA (European AI Regulatory Accreditation):**
- Proposed EU-wide accreditation for AI
- Coordination of national notified bodies

### 8.3 The Certification Process

**Step-by-step certification process:**

1. **Gap analysis**: Assess current state against target standard
2. **Remediation**: Address identified gaps
3. **Pre-assessment**: Internal or third-party readiness review
4. **Stage 1 audit**: Documentation and design review
5. **Stage 2 audit**: Implementation effectiveness verification
6. **Certification decision**: Issuance or denial
7. **Surveillance audits**: Periodic (annual or semi-annual)
8. **Recertification**: Full reassessment (typically triennial)

---

## 9. Compliance Automation and Tooling

### 9.1 AI Compliance Platforms

| Platform | Capabilities |
|----------|-------------|
| Credo AI | AI governance and compliance tracking, EU AI Act mapping, bias monitoring |
| Fairnow | AI bias detection and mitigation, compliance reporting |
| Robust Intelligence | Model validation, red-teaming automation, risk monitoring |
| Protect AI | ML pipeline security, supply chain scanning, model security |
| Arthur AI | Model monitoring, drift detection, performance tracking |
| Fiddler AI | Model performance, bias, and explainability monitoring |
| MLflow (open source) | Experiment tracking, model registry, deployment management |

### 9.2 Compliance-as-Code for AI

**Policy-as-code models:**
- Rego (OPA) rules for AI governance policies
- Cue or CEL for model validation rules
- Kubernetes admission controllers for ML infrastructure
- GitOps-style AI governance (policy in repo)

**Infrastructure-as-code for AI security:**
- Terraform modules for secure ML infrastructure
- CloudFormation templates for compliance-aligned AI deployments
- Helm charts with security defaults
- Docker security scanning in MLOps pipelines

### 9.3 Automated Compliance Reporting

**Reporting automation features:**
- Automated evidence collection
- Control mapping and gap analysis
- Dashboard for compliance status
- Scheduled report generation
- Integration with GRC platforms
- Audit trail preservation

---

## 10. Third-Party AI Risk Management

### 10.1 The AI Supply Chain Challenge

**Third-party AI components:**
- Pre-trained foundation models (API or weights)
- Third-party training data
- ML operations tools and platforms
- AI plugins and extensions
- Cloud ML services (SageMaker, Vertex AI)
- Embedded AI in SaaS products

**Risk categories:**
- Data quality and provenance
- Model security posture
- Provider governance practices
- Supply chain continuity
- Regulatory compliance of provider

### 10.2 Third-Party AI Assessment Framework

**Assessment dimensions:**

| Dimension | Questions | Evidence |
|-----------|-----------|----------|
| Governance | AI policy? Ethics board? Oversight? | Policy docs, board charters |
| Data practices | Training data sources? Consent? Privacy? | Data sheets, privacy policies |
| Model transparency | Documentation? Explainability? | Model cards, technical reports |
| Security controls | Access controls? Red-teaming? Incident response? | SOC 2, pentest reports |
| Compliance posture | Regulatory mapping? Certifications? | Compliance documentation |
| Resilience | Failover? SLAs? Business continuity? | BCP/DR plans |
| Exit strategy | Data portability? Model transferability? | Contract terms, technical capability |

### 10.3 AI Vendor Risk Tiering

**Tier 1 — Low risk:**
- Internal-facing productivity tools
- Non-critical decision support
- Limited or no data sensitivity
- Annual review sufficient

**Tier 2 — Moderate risk:**
- Customer-facing applications
- Moderate data sensitivity
- Some automated decisions
- Semi-annual review

**Tier 3 — High risk:**
- Critical infrastructure systems
- Highly sensitive data
- Significant automated decisions
- Quarterly review with enhanced due diligence

**Tier 4 — Critical risk:**
- Life-safety systems
- National security applications
- Direct regulatory compliance
- Continuous monitoring with independent audit

---

## 11. Incident Response for AI Systems

### 11.1 AI-Specific Incident Classification

| Incident Type | Example | Severity |
|---------------|---------|----------|
| Security breach | Model theft, data exfiltration | Critical |
| Safety failure | Harmful output causing physical harm | Critical |
| Privacy incident | PII disclosure from training data | High |
| Bias/ discrimination | Algorithmic discrimination event | High |
| Performance degradation | Accuracy collapse after drift | Medium |
| Regulatory violation | Non-compliance triggered | Medium |
| Overreliance incident | Human error due to automation bias | Medium |
| Availability failure | Inference downtime | Low-Medium |

### 11.2 AI Incident Response Playbook

**Phase 1: Detection and Triage (0-1 hour)**
- Automated alert from monitoring system
- Human analyst verification
- Severity assessment
- Escalation path activation

**Phase 2: Containment (1-4 hours)**
- Disable affected model or routing
- Block malicious inputs or attack sources
- Isolate training or inference infrastructure
- Preserve logs and evidence

**Phase 3: Investigation and Analysis (4-24 hours)**
- Determine root cause
- Identify scope of impact
- Reconstruct attack timeline
- Extract forensic evidence

**Phase 4: Remediation (24-72 hours)**
- Patch identified vulnerability
- Update model or retrain if needed
- Implement additional controls
- Verify fix effectiveness

**Phase 5: Recovery and Post-Mortem (72+ hours)**
- Restore normal operations
- Conduct incident review
- Update threat model
- Implement process improvements

### 11.3 Reporting Requirements

**Internal reporting:**
- Executive summary for leadership
- Technical report for engineering
- Legal notification for counsel
- PR statement for communications

**External reporting (regulatory and contractual):**
- EU AI Act: Serious incident notification (2 hours to 3 days)
- GDPR: Data breach notification (72 hours)
- SEC: Material incident disclosure (4 business days)
- Sector regulators: Healthcare, finance, critical infrastructure
- Contractual SLAs: Customer notification timelines

---

## 12. Future of AI Security Compliance

### 12.1 Trends Shaping AI Compliance (2026-2028)

1. **Regulatory convergence**: Toward harmonized global standards
2. **Compliance automation**: AI-powered compliance tools
3. **Continuous auditing**: Real-time vs. periodic compliance
4. **Third-party ecosystem**: AI-specific audit firms emerging
5. **Insurance market development**: AI liability insurance driving compliance
6. **Specialized certification**: Provider-specific compliance labels
7. **Workforce development**: AI auditor certification programs

### 12.2 Challenges Ahead

- **Technical complexity**: Keeping compliance tools aligned with AI evolution
- **Resource disparities**: SME compliance burden vs. large enterprise
- **Jurisdictional conflicts**: Divergent requirements across markets
- **Audit capacity**: Shortage of qualified AI auditors
- **Standards alignment**: Harmonizing competing frameworks
- **Measurement validity**: Ensuring compliance actually improves safety

---

## 13. Conclusion

AI security compliance and audit is rapidly evolving from a niche concern into a core business function. The interplay between risk management frameworks (NIST AI RMF), vulnerability taxonomies (OWASP Top 10), service organization controls (SOC 2), government security standards (FedRAMP), and testing practices (red-teaming) creates a comprehensive but complex compliance landscape.

Organizations building or deploying AI systems must:
1. **Understand the regulatory landscape**: Which frameworks apply to your use case
2. **Implement a risk management framework**: NIST AI RMF or ISO 42001 as foundation
3. **Adopt security testing practices**: Red-teaming and adversarial testing integrated into development
4. **Prepare for audit**: Documentation, monitoring, and continuous compliance
5. **Engage with the ecosystem**: Third-party risk management, certifications, and incident response

The organizations that invest in robust compliance programs today will be best positioned to earn trust, avoid regulatory penalties, and build sustainable AI systems.

---

## References and Further Reading

- NIST AI RMF 1.0 (January 2023) + Generative AI Profile (July 2024)
- OWASP Top 10 for LLM Applications (v1.1, 2024)
- AICPA SOC 2 for AI Supplemental Guidance (2024-2025)
- FedRAMP AI Security Requirements (2024)
- UK AI Safety Institute Testing Framework (2024)
- Microsoft: "Lessons from Red Teaming 100 Generative AI Products"
- IBM: Adversarial Robustness Toolbox (ART) Documentation
- IAPP AI Governance Professional (AIGP) Body of Knowledge
- CSA STAR Certification for AI
- ISO/IEC 42001:2023 AI Management System Standard
- OWASP LLM Prompt Injection Testing Guide
- Google AI Red Team Methodologies (published papers)

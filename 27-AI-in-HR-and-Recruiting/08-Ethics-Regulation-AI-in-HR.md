# 08 — Ethics, Regulation, and Governance of AI in HR

## Overview

The use of AI in HR operates at the intersection of powerful technology, sensitive personal data, high-stakes decisions about people's livelihoods, and a rapidly evolving regulatory landscape. This document provides a comprehensive examination of the ethical frameworks, legal regulations, bias detection methodologies, transparency requirements, and governance best practices that apply to AI in human resources.

From the US EEOC's guidance on algorithmic fairness to New York City's Local Law 144, the EU AI Act, and Illinois' AI hiring law, this document covers the full regulatory spectrum and provides actionable guidance for compliance and ethical practice.

---

## 1. The Ethical Foundations of AI in HR

### 1.1 Why HR AI Is High-Stakes

AI decisions in HR affect fundamental life opportunities:
- **Hiring** — Whether someone gets a job (income, career trajectory)
- **Performance** — How someone is evaluated (promotion, compensation, role)
- **Retention** — Whether someone is flagged for intervention (or for exit)
- **Compensation** — What someone earns (equity, fairness)
- **Termination** — Whether someone loses their livelihood

Because HR decisions are inherently about people, the ethical stakes are qualitatively different from AI in logistics, marketing, or even finance.

### 1.2 Core Ethical Principles

| Principle | Definition | HR AI Implication |
|---|---|---|
| **Beneficence** | AI should benefit people | HR AI should improve outcomes for employees and candidates, not just employer efficiency |
| **Non-maleficence** | Do no harm | AI should not discriminate, invade privacy, or create surveillance cultures |
| **Autonomy** | Respect individual agency | People should have choice, consent, and the right to opt out of AI-driven processes |
| **Justice** | Fair distribution of benefits and burdens | AI should not perpetuate or amplify existing inequalities |
| **Transparency** | Operations should be visible and explainable | Decisions driven by AI must be auditable and explainable to affected individuals |
| **Accountability** | Responsibility for outcomes must be assignable | Organisations must be responsible for their AI systems' impacts |
| **Dignity** | Respect inherent human worth | AI should treat people as ends, not merely means to organisational efficiency |

### 1.3 Key Ethical Tensions

| Tension | Example | Ethical Question |
|---|---|---|
| Efficiency vs Fairness | Automated resume screening filters 90% of applicants | Does the efficiency gain justify potential false negatives that exclude qualified candidates? |
| Personalisation vs Privacy | Monitoring Slack messages for sentiment analysis | Does personalised retention support justify continuous communication monitoring? |
| Prediction vs Stigma | Flagging an employee as "flight risk" | Does knowing their risk score change how managers treat them (self-fulfilling prophecy)? |
| Objectivity vs Context | Algorithmic performance scores | Can any algorithm fully capture context (personal circumstances, team dynamics)? |
| Scale vs Individuality | One-size-fits-all engagement model | Do aggregated insights miss individual differences that matter? |

---

## 2. Regulatory Landscape

### 2.1 United States Federal Regulations

**EEOC (Equal Employment Opportunity Commission) Guidance:**

The EEOC has issued technical guidance on AI and algorithmic fairness in employment decisions, focusing on disparate impact under Title VII of the Civil Rights Act of 1964.

| Key Document | Year | Summary |
|---|---|---|
| "The Americans with Disabilities Act and the Use of Software, Algorithms, and Artificial Intelligence to Assess Job Applicants and Employees" | 2022 | AI assessments must not discriminate against people with disabilities; candidates can request reasonable accommodation |
| "Selecting AI: The EEOC's Guidance on the Use of Artificial Intelligence in Employment Decisions" | 2023 | AI hiring tools must comply with adverse impact standards; employers are liable even if using third-party AI vendors |
| "Artificial Intelligence and Algorithmic Fairness" | 2023 | Employers must monitor AI for disparate impact, even if the AI vendor claims fairness |

**Key EEOC Requirements:**
1. **Adverse impact analysis**: If AI screens out a protected group at a rate < 4/5ths of the majority group, it triggers investigation
2. **Alternative employment practice**: If a less discriminatory AI or process can achieve the same business goals, it must be used
3. **Vendor responsibility**: Employers cannot outsource liability — you are responsible for your AI vendor's tool's compliance
4. **Reasonable accommodation**: Candidates with disabilities have the right to alternative assessment methods

**OFCCP (Office of Federal Contract Compliance Programs):**
- Federal contractors must maintain records of AI tool usage
- Must conduct adverse impact analyses of AI-driven employment decisions
- Must provide data and methodology to government auditors upon request

### 2.2 State and Local Laws

**New York City Local Law 144 (NYC LL 144):**

The first US law specifically regulating AI hiring tools, effective July 2023.

| Requirement | Detail |
|---|---|
| Scope | Any "automated employment decision tool" (AEDT) used in NYC to screen candidates or employees |
| Mandatory bias audit | Annual independent bias audit required |
| Audit content | Disparate impact analysis by race/ethnicity and gender; selection rate ratios |
| Public disclosure | Audit results published on the employer's website |
| Notice to candidates | Candidates informed an AEDT is being used; what data is collected; how it's used |
| Opt-out | Candidates can request an alternative selection process |
| Penalties | $500–$1,500 per violation per day |

**Audit requirements:**
- Independent auditor (can be internal if independent of the AI vendor and hiring team)
- Calculation of selection rate and impact ratio for each demographic group
- For each group: "Has this group been subject to disparate impact?"
- Historical data (minimum 1 year) required
- Scored candidate data must be retained for audit

**Illinois Artificial Intelligence Video Interview Act (HB 2557):**

| Requirement | Detail |
|---|---|
| Notice | Candidates must be informed before AI analysis of video interviews |
| Consent | Candidates must consent to AI analysis |
| Explanation | Candidates must be told how AI works and what traits it evaluates |
| Data retention | Video recordings must be deleted within 30 days of request |
| Reporting | Annual demographic impact report required |
| Opt-out | Must provide non-AI interview option |

**Other State Activity:**
| State | Status | Key Provisions |
|---|---|---|
| California | Proposed (AB 2930, AB 331) | Bias audit, transparency, opt-out for AI hiring tools |
| Maryland | Proposed | Prohibits AI that uses facial recognition in hiring |
| Washington | Proposed | Bias audit and transparency requirements |
| New Jersey | Proposed | AI hiring transparency law, vendor liability |
| Massachusetts | Proposed | Algorithmic accountability for employment AI |
| Connecticut | NY LL 144-style law (HB 5600) | 2025 effective date, similar to NYC |

### 2.3 European Union

**EU AI Act (Regulation (EU) 2024/1689):**

The world's first comprehensive AI regulation, with employment-related AI classified as **high-risk**.

| Classification | AI in HR is "high-risk" under Annex III, Article 6 |
|---|---|
| Requirements | Risk management system, technical documentation, transparency, human oversight, accuracy, robustness, cybersecurity |
| Conformity assessment | Self-assessment for most HR AI (third-party for biometric/facial analysis) |
| Fundamental rights impact assessment | Required before deployment of high-risk HR AI |
| Transparency to workers | Must inform employees/candidates that AI is being used |
| Human oversight | HR AI decisions must be reviewable by a human |
| Record-keeping | Automatic logs of AI system operation for traceability |
| Penalties | Up to 35M EUR or 7% of global annual turnover, whichever is higher |

**Timeline:**
| Date | Implementation |
|---|---|
| 2024 | Regulation adopted |
| 2025–2027 | High-risk provisions phased in |
| 2026 | HR AI provisions fully enforceable |

**GDPR (General Data Protection Regulation) Article 22:**
- Right to not be subject to a decision based solely on automated processing
- If AI is used in a way that produces "legal effects or similarly significant effects" (employment decisions qualify), the data subject has the right to human intervention
- Right to explanation of the logic involved
- Right to contest the decision

**EU Pay Transparency Directive (2023):**
- Right for employees to know individual and average pay levels
- Gender pay gap reporting required
- AI-driven pay determination must be transparent and auditable

### 2.4 Other Jurisdictions

| Jurisdiction | Regulation | Key Provisions |
|---|---|---|
| **Canada** | PIPEDA + Algorithmic Transparency Act (proposed) | Consent for AI data use; transparency in AI decision-making |
| **UK** | Equalities Act 2010 + AI regulation consultation (2023) | Disparate impact from AI covered under existing equality law |
| **Australia** | Workplace Gender Equality Act + AI Ethics Framework | Pay equity + voluntary AI ethics standards |
| **Brazil** | LGPD + AI Bill (PL 2338/2023) | Data protection + high-risk AI classification |
| **India** | Digital Personal Data Protection Act (2023) | Consent for personal data in AI systems |
| **China** | Deep Synthesis Provisions + Algorithmic Recommendation Provisions | Transparency in algorithmic hiring; ban on some AI interview analysis |
| **Singapore** | Model AI Governance Framework + PDPA | Voluntary framework for ethical AI in HR |

---

## 3. Bias Detection Methodologies

### 3.1 Statistical Measures

**Disparate Impact / 4/5ths Rule:**
```
Selection rate for protected group
───────────────────────────────────── < 0.80 = Adverse impact
Selection rate for majority group

Example:
Hired: 20 of 100 women (20%) vs 50 of 100 men (50%)
Impact ratio: 20% / 50% = 0.40 → Adverse impact present
```

**Standardised Mean Difference (Cohen's d):**
```
d = (μ_protected - μ_majority) / σ_pooled
|d| < 0.20 = negligible
|d| 0.20–0.50 = small
|d| 0.50–0.80 = medium
|d| > 0.80 = large
```

**Statistical Significance Tests:**

| Test | Use Case | When Appropriate |
|---|---|---|
| Chi-squared | 2×2 contingency table (demographic × hire/no-hire) | Large sample sizes (n > 100) |
| Fisher's exact | 2×2 table, small sample | Small sample sizes |
| Mann-Whitney U | Score distribution comparison | Non-normal score distributions |
| t-test | Mean score comparison | Normal distributions, large n |
| ANOVA | Comparison across >2 groups | Multiple demographic categories |
| Logistic regression | Controlling for covariates | Need to isolate group effect |

### 3.2 Fairness Metrics

| Metric | Definition | When to Use |
|---|---|---|
| **Demographic parity** | P(Ŷ=1|A=0) = P(Ŷ=1|A=1) | When base rates differ, may be too strict |
| **Equal opportunity** | P(Ŷ=1|Y=1,A=0) = P(Ŷ=1|Y=1,A=1) | Focus on false negatives (missed qualified candidates) |
| **Equalised odds** | P(Ŷ=1|Y=y,A=0) = P(Ŷ=1|Y=y,A=1) for y in {0,1} | Both false positives and false negatives matter |
| **Predictive parity** | P(Y=1|Ŷ=1,A=0) = P(Y=1|Ŷ=1,A=1) | Focus on precision — are positive predictions equally reliable? |
| **Individual fairness** | Similar individuals get similar predictions | Requires good similarity metric |
| **Counterfactual fairness** | Prediction unchanged if sensitive attribute changed | Requires causal model |

### 3.3 Intersectional Bias Analysis

Bias may be invisible in single-axis analysis but visible at intersections:

```
Single-axis (gender only):
Female selection rate: 22%
Male selection rate: 26%
Impact ratio: 0.85 → borderline, may not be flagged

Intersectional (gender × race):
White women: 24% (0.92) → fine
Asian women: 25% (0.96) → fine
Black women: 14% (0.54) → ADVERSE IMPACT
Hispanic women: 16% (0.62) → ADVERSE IMPACT

→ Intersectional analysis reveals disparity that single-axis analysis misses.
```

### 3.4 Causal Fairness Analysis

Beyond correlation, causal methods ask: "What if we could change the protected attribute?"

**Directed Acyclic Graph (DAG):**
```
Skill → Score → Hire Decision
  ↑        ↑
Experience  Protected Attribute (Gender/Race)

Question: Does protected attribute directly affect Score (bias)
or only indirectly through legitimate factors (Experience, Skill)?
```

**Path-specific effects:**
- Direct effect: Protected attribute → Score (bias, illegal)
- Indirect effect: Protected attribute → Opportunity difference → Score (historical discrimination)
- Total effect: Direct + Indirect = Overall disparity to remediate

---

## 4. Transparency and Explainability

### 4.1 Right to Explanation

Under GDPR Article 22 and emerging laws (NYC LL 144, Illinois AI law, EU AI Act), individuals have the right to know:

1. **That AI is being used** — Notification before or during the process
2. **How the AI works** — General logic, not necessarily full source code
3. **What data is used** — Categories of data, sources
4. **Why a decision was made** — Specific reasons for this individual's outcome
5. **How to contest** — Appeal mechanism

### 4.2 Explanation Methods

**Global explanations (model-level):**
| Method | What It Shows | Example |
|---|---|---|
| Feature importance | Which factors matter most overall | "Skills match contributes 40% to resume score" |
| Partial dependence plots | How outcome changes with feature | "Score increases with experience years until 15yr plateau" |
| Decision tree surrogate | Simplified model of complex model | "If skills ≥ 0.8 AND experience ≥ 3yr → high score" |

**Local explanations (individual-level):**
| Method | What It Shows | Example |
|---|---|---|
| SHAP values | Feature contributions to this prediction | "Your score is 78/100 (+5 from skills, -10 from missing certification)" |
| LIME | Local linear approximation | "Your resume is most similar to candidates who scored 75-82" |
| Counterfactual | What would need to change | "If you had 'Kubernetes' in your skills, your score would increase to 85" |
| Anchors | Decision rules for this instance | "Skills match > 0.7 → score above threshold" |

### 4.3 Transparency Requirements by Jurisdiction

| Jurisdiction | Notice Required | Explanation Required | Opt-Out Available | Audit Required |
|---|---|---|---|---|
| NYC (LL 144) | ✓ | ✓ (on request) | ✓ | ✓ (annual) |
| Illinois | ✓ | ✓ | ✓ | ✗ |
| EU (AI Act) | ✓ | ✓ | ✓ (some cases) | ✓ (conformity) |
| EU (GDPR) | ✓ | ✓ | ✓ (Art. 22) | ✗ (unless DPO required) |
| California (proposed) | ✓ | ✓ | ✓ | ✓ |
| UK (Equality Act) | ✗ | ✗ | ✗ | ✓ (if challenged) |

### 4.4 Sample Candidate Notice

```
NOTICE OF AI USE IN HIRING PROCESS

[Company Name] uses an automated resume screening system as part of our hiring process.
This tool helps us fairly and efficiently evaluate all applications.

What the AI evaluates:
- Your resume is analysed for skills, experience, education, and certifications
- A matching score is computed against the job requirements
- The AI does not use your name, photo, age, gender, race, or address in scoring

How your data is used:
- Your resume text is processed by the AI model
- Data is stored securely and deleted 6 months after the hiring decision
- No facial recognition or emotion analysis is used

Your rights:
- You can request a human-only review of your application
- You can request an explanation of why your application was scored as it was
- You can access, correct, or delete your data at any time

To exercise any of these rights, contact: [email/phone]
```

---

## 5. Governance and Accountability

### 5.1 AI Ethics Committee

Organisations should establish an AI Ethics Committee for HR AI:

| Role | Responsibilities |
|---|---|
| Chief HR Officer | Business ownership of HR AI decisions |
| Data Science Lead | Technical oversight, model validation |
| Legal Counsel | Compliance with regulations |
| Ethics Officer | Ethical framework, stakeholder consultation |
| HR Ops Lead | Implementation, training, process integration |
| Employee Representative | Worker perspective (union, works council) |
| Independent Advisor | External ethics/academic perspective |

**Committee functions:**
1. Approve new HR AI use cases before deployment
2. Review bias audit results
3. Handle escalated complaints
4. Oversee transparency and communication
5. Approve material changes to existing AI systems
6. Review vendor AI tools for compliance

### 5.2 AI Risk Classification

Organisations should classify each HR AI use case by risk level:

| Tier | Definition | Examples | Requirements |
|---|---|---|---|
| **Low risk** | No significant employment decision | FAQ chatbot, scheduling tool, learning recommendations | Basic transparency |
| **Medium risk** | Decision-support for employment decisions | Resume scoring for recruiter review, sentiment dashboards | Bias audit, transparency, human review option |
| **High risk** | Automated employment decisions | Automated rejection, AI-only video interview scoring, auto-generated performance rating | Full audit, regulatory compliance, human override, appeal process |

### 5.3 Vendor Due Diligence

When buying AI HR tools, organisations must assess:

```
1. VENDOR ASSESSMENT CHECKLIST

☐ Does the vendor disclose their model's training data demographics?
☐ Has the vendor conducted a bias audit? Will they share results?
☐ Is the model retrained on customer data? How is it protected?
☐ Is the model regularly updated? How often?
☐ Is the decision-making process explainable? To what level?
☐ Does the vendor provide individual-level explanations?
☐ What data does the vendor retain? For how long?
☐ Is the vendor SOC 2 / ISO 27001 certified?
☐ Can you export your data? (Right to data portability)
☐ Is there a human review pathway?
☐ Has the vendor faced any regulatory action related to bias?
☐ Does the vendor indemnify against bias-related claims?
```

### 5.4 Incident Response

When an AI HR system produces a harmful outcome:

```
AI HR Incident Response Plan:

1. IDENTIFY (24 hours)
   - Is there a credible report of AI-caused harm?
   - Document: what happened, who was affected, which AI system

2. CONTAIN (48 hours)
   - Pause the AI system if it's actively causing harm
   - Notify affected individuals
   - Preserve all logs and data

3. INVESTIGATE (1–2 weeks)
   - Internal review with data science + legal
   - Root cause analysis: bias? data error? model drift? implementation error?
   - Determine scope: individual incident or systemic?

4. REMEDIATE (1–4 weeks)
   - Fix root cause
   - Remediate affected individuals (re-evaluate, adjust outcomes)
   - If systemic: retrain model, adjust thresholds, or replace tool

5. COMMUNICATE
   - Internal: affected individuals + AI ethics committee
   - External: regulator (if required), public (if high-profile)
   - Transparency report

6. PREVENT
   - Update validation/testing protocol to catch similar issues
   - Add ongoing monitoring for identified risk
   - Document lessons learned for future incidents
```

---

## 6. Algorithmic Auditing

### 6.1 The Auditing Process

An algorithmic bias audit follows a structured methodology:

**Audit Steps:**
1. **Scope definition** — Which AI tool, which protected characteristics, which metrics?
2. **Data collection** — Input data, model scores, outcomes, demographic data
3. **Data validation** — Ensure demographic data is accurate and sufficient
4. **Metric calculation** — Selection rates, impact ratios, statistical significance
5. **Covariate adjustment** — Regression analysis controlling for legitimate factors
6. **Intersectional analysis** — Examine multiple protected characteristics together
7. **Threshold analysis** — Test different thresholds for bias impact
8. **Alternative model comparison** — Would a different model produce fairer outcomes?
9. **Findings report** — Document disparities, root causes, recommendations
10. **Remediation plan** — Specific actions with timeline and accountability

### 6.2 Audit Frequency

| Context | Frequency |
|---|---|
| Pre-deployment | Mandatory before any AI tool is used in production |
| Annual | Required by NYC LL 144, recommended best practice |
| Post-significant change | Retrain, threshold change, new features |
| Post-incident | After any complaint or adverse event |
| Vendor model update | When vendor releases new version |

### 6.3 Audit Tools

| Tool | Type | Features |
|---|---|---|
| **AI Fairness 360** (IBM) | Open source | 70+ fairness metrics, 10+ bias mitigation algorithms |
| **What-If Tool** (Google) | Open source | Interactive exploration, fairness visualisation |
| **Fairlearn** (Microsoft) | Open source | Fairness metrics, mitigation algorithms |
| **Aequitas** (UChicago) | Open source | Bias audit toolkit, reporting |
| **SHAP** | Open source | Explainability, feature attribution |
| **LIME** | Open source | Local explanations |
| **Carvaggio AI Fair** | Commercial | End-to-end bias audit, regulatory reporting |
| **Be Informed** | Commercial | AI governance platform |

---

## 7. Consent and Data Rights

### 7.1 Consent Requirements by Jurisdiction

| Jurisdiction | Consent Type | Specifics |
|---|---|---|
| GDPR | Explicit, freely given, specific, informed, unambiguous | Opt-in required for processing special category data (race, health, etc.) |
| Illinois | Written consent | For AI video interview analysis |
| NYC LL 144 | Notice (not explicit consent) | Notification that AEDT is used; opt-out available |
| CCPA | Notice + opt-out (for sale of data) | Right to know, right to delete |
| PIPEDA | Meaningful consent | Consent for collection, use, disclosure |
| LGPD | Specific, unambiguous | Opt-in required for sensitive data |

### 7.2 Data Retention Policies

| Data Type | Recommended Retention | Rationale |
|---|---|---|
| Resume / application | 6 months post-decision | Enough for adverse impact analysis |
| Video interview | 30–90 days post-decision | Delete after decision or candidate request |
| Assessment scores | 2 years | Audit trail for bias analysis |
| Demographic data (if collected) | Aggregate only | Do not retain at individual level |
| AI model inputs | 1 year | Reproduce predictions if challenged |
| Audit records | Permanently | Legal defence |
| Consent records | Until data is deleted | Proof of compliance |

### 7.3 Right to Delete / Right to be Forgotten

Under GDPR Article 17 and CCPA, candidates and employees can request deletion of their data. HR AI systems must support:
- Bulk deletion on request
- Automated removal after retention period
- Anonymisation where deletion impossible (e.g., model weights)
- Cascade deletion to third-party AI vendors

---

## 8. Enforcement and Penalties

### 8.1 Penalty Regimes

| Regulation | Maximum Penalty | Recent Enforcement |
|---|---|---|
| GDPR | 20M EUR or 4% of global turnover | Italian DPA: $27M fine for AI screening (2023) |
| EU AI Act | 35M EUR or 7% of global turnover | First enforcement expected 2026 |
| NYC LL 144 | $1,500/violation/day | First fines issued 2024 ($15K+) |
| Illinois AI Law | $2,000/violation | Private right of action |
| EEOC enforcement | Back pay, punitive damages | 2023: $365K settlement for AI-driven age discrimination |
| CCPA | $2,500–$7,500/violation | No specific HR AI enforcement yet |

### 8.2 Notable Enforcement Actions

| Case | Year | Summary |
|---|---|---|
| EEOC v. iTutorGroup | 2023 | Age discrimination via automated resume screening; $365K settlement |
| Italian DPA (Garante) | 2023 | Rejected AI-based hiring tool for lack of transparency and consent |
| EU Data Protection Authorities | 2024 | Multiple investigations into hiring AI compliance with GDPR Art. 22 |
| NYC DCWP | 2024 | First LL 144 enforcement actions against employers using unaudited AEDTs |
| Amazon (abandoned) | 2018 | AI recruiting tool biased against women; project disbanded after internal audit |

---

## 9. Best Practice Framework

### 9.1 Pre-Deployment Checklist

- [ ] What is the specific business problem this AI solves?
- [ ] Is AI the right solution? Could a simpler non-AI process achieve the same?
- [ ] What data will be collected? Is it necessary and minimally sufficient?
- [ ] Have we identified potential sources of bias?
- [ ] Have we conducted a pre-deployment fairness audit?
- [ ] Is the model explainable to the affected population?
- [ ] Is there a human review pathway for significant decisions?
- [ ] Have we obtained legal review for regulatory compliance?
- [ ] Have we communicated with employees/candidates about the AI?
- [ ] Have we established consent/opt-out mechanisms?

### 9.2 Ongoing Monitoring

- [ ] Automated monitoring dashboard (selection rates, score distributions)
- [ ] Monthly drift detection (data drift, concept drift)
- [ ] Quarterly bias audit (disparate impact analysis)
- [ ] Incident reporting mechanism (for both employees and candidates)
- [ ] Model performance monitoring (accuracy, calibration)
- [ ] Regulatory change tracking (new laws, guidance, enforcement actions)

### 9.3 Key Performance Indicators for Ethical AI

| Metric | Target | Measurement |
|---|---|---|
| Candidate/employee complaints about AI | < 0.1% of interactions | Count / total interactions |
| Unexplained demographic score gap | < 0.1 standard deviation | Score mean difference / pooled std dev |
| Audit findings resolved on time | 100% | Resolved within 30 days |
| Human override rate for AI decisions | 5–15% | Overrides / total decisions |
| AI model explainability score | > 4/5 on user test | Survey of internal stakeholders |
| Regulatory compliance score | 100% | Self-assessment against requirements |

---

## 10. The Future of AI HR Ethics and Regulation

### 10.1 Emerging Trends

1. **Global convergence on bias audit requirements** — NYC LL 144 model being adopted globally
2. **Worker representative involvement** — Union/works council rights to approve HR AI
3. **Real-time bias monitoring** — Continuous rather than periodic auditing
4. **AI ethics by design** — Fairness embedded in model development, not as afterthought
5. **Cross-jurisdictional compliance** — Global companies navigating patchwork of laws
6. **Private right of action expansion** — More laws allowing individuals to sue over AI discrimination
7. **Facial/emotion analysis bans** — Growing consensus against these high-risk modalities in HR

### 10.2 Calls for a "Right to Work with a Human"

- Advocates for a fundamental right to human decision-making in employment
- AI as decision-support, not decision-maker
- Particularly important for: rejection decisions, firing decisions, compensation reductions

### 10.3 The Role of International Standards

| Standard | Focus | Status |
|---|---|---|
| ISO/IEC 42001 | AI Management System | Published 2023 |
| ISO/IEC TR 24027 | Bias in AI systems | Published 2021 |
| IEEE P7003 | Algorithmic bias considerations | Published 2023 |
| NIST AI Risk Management Framework | AI risk management | Published 2023 |
| OECD AI Principles | International AI governance | Adopted 2019 |
| UNESCO AI Ethics Recommendation | Global ethics framework | Adopted 2021 |

---

## References

- EEOC, "The Americans with Disabilities Act and the Use of Software, Algorithms, and Artificial Intelligence to Assess Job Applicants and Employees," 2022.
- EEOC, "Selecting AI: The EEOC's Guidance on the Use of Artificial Intelligence in Employment Decisions," 2023.
- NYC Department of Consumer and Worker Protection, "Local Law 144: Automated Employment Decision Tools," Final Rule, 2023.
- Illinois HB 2557, "Artificial Intelligence Video Interview Act," 2022.
- European Commission, "EU AI Act (Regulation 2024/1689)," Official Journal of the European Union, 2024.
- European Commission, "GDPR Article 22: Automated Individual Decision-Making," 2018.
- EU Pay Transparency Directive (EU) 2023/970, Official Journal of the European Union, 2023.
- R. K. Bellamy et al., "AI Fairness 360: An Extensible Toolkit for Detecting and Mitigating Algorithmic Bias," IBM Journal of R&D, 2019.
- S. Barocas and A. D. Selbst, "Big Data's Disparate Impact," California Law Review, 2016.
- M. Raghavan et al., "Mitigating Bias in Algorithmic Hiring: A Critical Review," ACM FAccT, 2020.
- NIST, "AI Risk Management Framework (AI RMF 1.0)," National Institute of Standards and Technology, 2023.
- IEEE, "P7003: Standard for Algorithmic Bias Considerations," 2023.
- J. Buolamwini and T. Gebru, "Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification," ACM FAccT, 2018.
- C. O'Neil, "Weapons of Math Destruction: How Big Data Increases Inequality and Threatens Democracy," Crown, 2016.
- OECD, "OECD Principles on Artificial Intelligence," OECD Publishing, 2019.
- UNESCO, "Recommendation on the Ethics of Artificial Intelligence," UNESCO, 2021.
- N. Mehrabi et al., "A Survey on Bias and Fairness in Machine Learning," ACM Computing Surveys, 2021.
- D. Roselli et al., "Managing Bias in AI," Companion Proceedings of The Web Conference, 2019.
- World Economic Forum, "Responsible AI in HR: A Framework for Action," WEF, 2023.
- Deloitte, "Trustworthy AI in HR: From Principles to Practice," Deloitte Insights, 2024.

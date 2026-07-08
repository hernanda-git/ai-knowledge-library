# AI in Government — Core Topics

> Deep dive into the functional domains where public-sector AI is deployed: social services, tax, justice, law enforcement, immigration, transport, defense, health, education, and general citizen services. Each domain is mapped to data needs, AI techniques, risk tier, and required controls.

---

## 1. Functional Domain Map

```
Public Sector
├── Citizen-Facing Services
│   ├── Virtual assistants / copilots (limited risk)
│   ├── Multilingual intake & translation
│   └── Permit & license automation
├── Social Protection
│   ├── Benefits eligibility (high risk)
│   ├── Fraud & improper-payment detection
│   └── Case prioritization / triage
├── Revenue (Tax & Customs)
│   ├── Audit selection
│   ├── Document & invoice extraction
│   └── Evasion / smuggling detection
├── Justice & Corrections
│   ├── Risk / recidivism assessment (contested)
│   ├── Parole & sentencing support
│   └── Legal document analysis
├── Law Enforcement & Security
│   ├── Biometric identification (high/banned)
│   ├── Predictive policing (heavily restricted)
│   └── Evidence & media analysis
├── Immigration & Asylum
│   ├── Document verification
│   └── Credibility / claim assessment (high risk)
├── Infrastructure & Transport
│   ├── Traffic & transit optimization
│   └── Asset inspection (CV/drones)
├── Defense & National Security
│   ├── Intelligence fusion
│   ├── ISR (intelligence, surveillance, recon)
│   └── Cyber defense
├── Public Health
│   └── (see 63-AI-for-Healthcare)
└── Cross-Cutting
    ├── Procurement AI
    ├── Workforce / HR (see 27)
    └── Finance (see 67)
```

---

## 2. Domain 1 — Social Protection & Benefits

### Capabilities
- **Eligibility determination**: Classify applicants against statute (RAG over law + structured rules).
- **Fraud / improper-payment detection**: Anomaly detection on claim patterns.
- **Case prioritization**: Rank high-need cases to caseworkers.
- **Overpayment recovery**: Estimate and notify.

### Data needs
- Demographics, income, employment, household composition, historic claims.
- Must be **privacy-preserving** (see `40-Data-Sovereignty`).

### Risk & controls
| Control | Reference |
|---------|-----------|
| High-risk classification | `21-Regulation` |
| Fairness testing | `64-XAI`, `55-Ethics` |
| Human review of denials | `18-Agent-Security` |
| Appeal logging | `43-Provenance` |

### Pitfalls
- **Wrongful denial** → economic harm, legal challenge.
- **Automation bias** → caseworkers defer to model.

---

## 3. Domain 2 — Tax & Customs

### Capabilities
- **Audit selection**: Score returns by evasion probability.
- **Document extraction**: Invoice/ receipt OCR + NER.
- **Cross-border smuggling**: Graph analytics on shipment networks.

### Techniques
- Gradient-boosted trees (XGBoost, LightGBM) for scoring.
- GNNs for network detection (cf. `67-Finance` AML).
- LLM extraction pipelines (cf. `33-AI-Native-Software-Dev`).

### Controls
- Explainability of audit flags (taxpayer due process).
- Right-to-human-review.

---

## 4. Domain 3 — Justice, Corrections & Policing

> **Caution:** These are the most contested GovAI uses. Many jurisdictions restrict or ban predictive policing and social scoring.

### Capabilities (where permitted)
- **Risk assessment** (pretrial, parole): actuarial + ML augmentation.
- **Legal research & document review**: LLM-assisted (cf. `49-LegalTech`).
- **Evidence media analysis**: CV for forensic imagery.

### Controversies
- Proven racial/disparate impact in legacy tools (e.g., COMPAS-style).
- "Accuracy" ≠ "fairness" — must report **per-group** metrics (`64-XAI`).

### Recommendation
- Use AI strictly as **decision-support**, never autonomous sentencing.
- Mandatory impact assessment + independent audit.

---

## 5. Domain 4 — Immigration & Asylum

### Capabilities
- Document authenticity verification (CV + forensic).
- Claim credibility support (NLP on testimony vs. country reports).
- Case backlog triage.

### Risk
- **High-risk** under EU AI Act when affecting individual rights.
- Credibility scoring is ethically fraught → human adjudication mandatory.

---

## 6. Domain 5 — Infrastructure, Transport & Environment

### Capabilities
- Traffic signal optimization (reinforcement learning).
- Bridge/road inspection via drone CV.
- Disaster response routing.
- (Environmental intelligence → `45-AI-for-Climate`.)

### Risk
- Mostly **limited/standard** risk; safety-critical subsets need testing.

---

## 7. Domain 6 — Defense & National Security

### Capabilities
- Intelligence fusion (multi-source ML).
- ISR automation (object detection in imagery).
- Cyber threat detection (cf. `22-AI-Cybersecurity-Mythos`).
- Autonomous logistics.

### Controls
- Strict chain-of-custody, red-teaming (`61`), human command authority.
- Lethal autonomous weapons: internationally contested — out of scope for domestic GovAI.

---

## 8. Domain 7 — Citizen-Facing Services (Virtual Agents)

### Capabilities
- **GovLLM**: RAG-grounded assistant answering policy questions (cf. `04-RAG`).
- **Multilingual**: real-time translation (cf. `50-Multimodal`).
- **Form intake**: structured extraction from free text.

### Risk
- **Limited risk** → transparency disclosure ("You are talking to AI").
- Must escalate to humans on low-confidence / high-stakes queries.

### Example prompt pattern (RAG over legislation)

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Index statutes & policy docs
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vs = FAISS.load_local("gov_corpus_faiss", embeddings,
                      allow_dangerous_deserialization=True)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a government services assistant. Answer ONLY from the "
     "provided policy excerpts. If unsure, say 'Please contact a human "
     "caseworker.' Never invent entitlements."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

def ask(question: str):
    docs = vs.similarity_search(question, k=5)
    context = "\n---\n".join(d.page_content for d in docs)
    return prompt.format(context=context, question=question)
```

> Rule: **never fabricate entitlements** — hallucination in GovAI can create false legal expectations. See `52-AI-Hallucination-Detection`.

---

## 9. Domain 8 — Procurement, HR & Finance (cross-cutting)

- **Procurement AI**: vendor risk scoring, contract clause extraction.
- **HR** → `27-AI-in-HR-and-Recruiting`.
- **Finance** → `67-AI-in-Finance`.

---

## 10. Data Governance Baseline

| Principle | Practice |
|-----------|----------|
| Lawful basis | GDPR / national privacy law |
| Minimization | Collect only needed attributes |
| Residency | Store in-region (cf. `40`) |
| Provenance | Lineage of every training/decision input (cf. `43`) |
| Retention | Time-boxed deletion |
| Access | Role-based, audited |

---

## 11. Algorithmic Impact Assessment (AIA) Template

```
1. System description & purpose
2. Affected populations
3. Data sources & representativeness
4. Intended & out-of-scope uses
5. Risk tier (EU AI Act)
6. Fairness metrics & thresholds
7. Human-oversight design
8. Explainability approach (cf. 64)
9. Monitoring & drift plan (cf. 56)
10. Appeal & redress pathway
11. Sign-off (DPO, ethics board)
```

> Canada's DADM and the EU AI Act both require AIA-style artifacts before deployment.

---

## 12. Equity & Fairness Checklist

- [ ] Disaggregated error rates by sex, age, ethnicity, disability
- [ ] Proxy-variable audit (zip code ≈ race?)
- [ ] Calibration across groups
- [ ] Qualitative community impact review
- [ ] Published bias-mitigation plan

---

## 13. Human-in-the-Loop Patterns

| Pattern | When |
|---------|------|
| **Human-on-the-loop** (monitor) | Low-risk, high-volume |
| **Human-in-the-loop** (approve) | High-risk decisions |
| **Human-over-the-loop** (override) | Ambiguous edge cases |
| **No-human** | Banned for rights-affecting uses |

---

## 14. Common Architectural Building Blocks

- **AI Registry** — catalog of all systems (`21`).
- **Policy RAG corpus** — statutes, manuals (`04`).
- **Decision log** — immutable audit trail (`43`).
- **Fairness harness** — automated bias scans (`58`, `64`).
- **Red-team pipeline** — adversarial testing (`61`).
- **Sovereign inference** — in-region model serving (`40`, `23-Local-AI`).

---

## 15. Case Snapshot — Benefits Triage

```
Input: applicant record + claim text
  → Eligibility RAG (law grounding)
  → Fraud anomaly score
  → Caseworker queue with priority + reason codes
  → Decision + explanation + appeal path
  → Logged to decision registry
```

Each step emits a **provenance record** and a **fairness check**.

---

## 16. Technology Adoption by Maturity

| Maturity | Examples |
|----------|----------|
| Production at scale | Chatbots, OCR, fraud detection |
| Pilot / scaling | Benefits eligibility, tax audit ML |
| Contested / restricted | Predictive policing, credibility scoring |
| Emerging | Agentic casework (`03-Agents`), autonomous inspection |

---

## 17. Stakeholders

- **Citizens** — rights-holders, beneficiaries.
- **Caseworkers** — human reviewers.
- **Agency CIO/CDO** — platform owners.
- **DPO** — privacy oversight.
- **Audit bodies** — compliance.
- **Elected officials** — accountability.
- **Civil society** — watchdogs.

---

## 18. Summary Table — Domains at a Glance

| Domain | Primary AI | Risk Tier | Human Review |
|--------|-----------|-----------|--------------|
| Benefits | RAG + scoring | High | Mandatory |
| Tax audit | GBT + GNN | High | Mandatory |
| Justice | Risk models | High/contested | Mandatory |
| Policing | CV/biometrics | High/banned(some) | Mandatory |
| Immigration | CV + NLP | High | Mandatory |
| Transport | RL/CV | Variable | Varies |
| Defense | Fusion/ML | High | Command authority |
| Chatbots | LLM RAG | Limited | Escalation |

---

## 19. Cross-Domain Patterns

- All high-risk uses share: **registry → AIA → fairness → HITL → log → monitor**.
- All citizen-facing uses share: **disclosure + escalation**.
- All data uses share: **privacy + provenance**.

---

## 20. Looking Ahead

`03-Technical-Deep-Dive.md` covers the *how*: RAG over law, XAI for decisions, secure MLOps, and agentic casework architectures.

> Next: technical implementation patterns, code, and reference architectures.

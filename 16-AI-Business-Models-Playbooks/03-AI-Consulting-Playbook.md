# 03 — AI Consulting Playbook: Building and Scaling an AI Consulting Practice

## 1. Executive Summary

AI consulting is one of the fastest-growing segments in professional services, with the market expanding at 30%+ CAGR. This playbook provides a complete framework for building, pricing, and scaling an AI consulting practice — from solo practitioner to multi-practice agency.

The AI consulting market in 2025 is bifurcated:
- **Boutique AI specialists** (2–20 people): $500–$5K/day, deep technical expertise
- **Big 4 / Accenture / McKinsey**: $3K–$10K/day, broad enterprise relationships, AI embedded in larger engagements
- **Niche vertical AI firms**: $2K–$8K/day, deep domain expertise (legal, healthcare, finance)
- **Independent consultants** (solo): $1K–$3K/day, specialized skill sets

## 2. Engagement Models

### 2.1 Fixed-Bid (Fixed Price)

The consultant commits to delivering a defined scope at a fixed price.

| Aspect | Details |
| --- | --- |
| Best for | Well-defined projects with clear requirements |
| Typical size | $20K–$200K total |
| Duration | 4–16 weeks |
| Risk bearer | Consultant (if scope creeps) |
| Client preference | High (budget certainty) |
| Margin | 30–50% (if well-managed) |

**When to use**: MVP development, model evaluation, POC, specific integration
**When to avoid**: Ambiguous requirements, research-heavy projects, changing scope

**Fixed-bid pricing formula**:
```
Estimated hours × Blended rate × 1.3 (buffer) × 1.2 (risk premium)
= Project price
```

### 2.2 Time & Materials (T&M)

Client pays for actual time spent at agreed rates.

| Aspect | Details |
| --- | --- |
| Best for | Exploratory, evolving scope, ongoing work |
| Typical rate | $150–$500/hour (depending on seniority/region) |
| Duration | Open-ended or sprint-based |
| Risk bearer | Client (scope changes = more hours) |
| Client preference | Medium (but accepted for consulting) |
| Margin | 40–60% (lower risk for consultant) |

**When to use**: R&D, model exploration, advisory, ongoing optimization
**When to avoid**: Budget-constrained clients, procurement-driven orgs

### 2.3 Retainer

Monthly recurring fee for a block of hours or ongoing advisory.

| Aspect | Details |
| --- | --- |
| Best for | Long-term relationships, ongoing support |
| Typical pricing | $5K–$50K/month |
| Hours included | 20–160 hours/month |
| Duration | 3–12+ months |
| Risk bearer | Shared (consultant commits capacity) |
| Client preference | High (predictable costs, priority access) |
| Margin | 50–70% (highest of all models) |

**Retainer pricing formula**:
```
Monthly retainer = (Hours/month × Rate) × 0.85 (retainer discount)
```

**Retainer tiers**:

| Tier | Hours/Month | Price | Best For |
| --- | --- | --- | --- |
| Advisory | 10 hours | $4K–$8K/mo | Strategic guidance |
| Standard | 40 hours | $15K–$25K/mo | Regular project work |
| Premium | 80 hours | $30K–$50K/mo | Dedicated team member |
| Strategic | 160 hours | $50K–$80K/mo | Fractional CTO/AI leader |

### 2.4 Value-Based Pricing

Price is tied to the value delivered rather than time spent.

| Aspect | Details |
| --- | --- |
| Best for | Measurable outcomes (cost savings, revenue increase) |
| Typical pricing | 10–30% of value delivered |
| Risk bearer | Consultant (must deliver value to get paid) |
| Client preference | Very high (pay for results) |
| Margin | 60–200%+ (huge upside) |

**Value-based pricing examples**:

| Engagement | Value Delivered | Fee | % of Value |
| --- | --- | --- | --- |
| AI chatbot reducing support tickets by 40% | $500K/yr savings | $100K one-time + $5K/mo | 15–20% |
| ML model improving inventory accuracy by 25% | $2M/yr savings | $300K + 10% of savings year 1 | 25% |
| Sales AI increasing conversion by 15% | $5M incremental revenue | $500K + 5% of uplift | 15% |

## 3. Organizational Structures

### 3.1 The Pod Model (Recommended for Scale)

Each pod is a self-contained unit with all necessary skills:

```
                     ┌─────────────┐
                     │  Pod Lead    │
                     │  (Sr. Cons.) │
                     └──────┬──────┘
            ┌───────────────┼───────────────┐
            │               │               │
     ┌──────┴──────┐ ┌──────┴──────┐ ┌──────┴──────┐
     │ ML Engineer │ │ Data Eng.   │ │ Domain Exp. │
     │ (1–2)       │ │ (1)         │ │ (0.5)       │
     └─────────────┘ └─────────────┘ └─────────────┘
```

**Pod composition by engagement type**:

| Engagement Type | Pod Size | Roles | Typical Rate |
| --- | --- | --- | --- |
| Strategy & advisory | 1–2 people | Sr. Consultant + Domain Expert | $2K–$5K/day |
| POC / Prototype | 2–3 people | ML Engineer + Data Eng + PM | $3K–$6K/day |
| Full product build | 3–6 people | 2 ML Eng, Data Eng, FE/BE, PM, Design | $5K–$12K/day |
| MLOps / Infrastructure | 2–4 people | ML Eng + DevOps + Data Eng | $4K–$8K/day |
| AI Training / Workshops | 1–2 people | Sr. Consultant + Curriculum designer | $8K–$25K/week |

### 3.2 Hub-and-Spoke Model

Centralized expertise (hub) supports distributed delivery (spokes).

```
              ┌───────────────────┐
              │    CENTRAL HUB    │
              │  - AI Research    │
              │  - Model Training │
              │  - IP Development │
              │  - Best Practices │
              └────────┬──────────┘
        ┌──────────────┼──────────────┐
        │              │              │
  ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐
  │  Spoke 1   │  │  Spoke 2   │  │  Spoke 3   │
  │ (Client A) │  │ (Client B) │  │ (Client C) │
  │ - Delivery │  │ - Delivery │  │ - Delivery │
  │ - Support  │  │ - Support  │  │ - Support  │
  └───────────┘  └───────────┘  └───────────┘
```

**Best for**: Geographic expansion (hub in a tech hub city, spokes in client locations)

### 3.3 Solo Practitioner Structure

```
┌──────────────────────────────────────┐
│        YOU (Solo Consultant)         │
├──────────────────────────────────────┤
│ Capabilities:                        │
│ • Strategy & advisory (40% of time)  │
│ • Technical delivery (30% of time)   │
│ • Business dev & sales (20% of time) │
│ • Admin & learning (10% of time)     │
├──────────────────────────────────────┤
│ Network (Subcontractors):            │
│ • ML Engineers (2–3)                 │
│ • Data Engineers (1–2)               │
│ • UI/UX Designer (1)                 │
│ • Domain experts (per project)       │
└──────────────────────────────────────┘
```

## 4. Delivery Frameworks

### 4.1 CRISP-DM for AI Projects

The industry standard data science process, adapted for modern AI:

| Phase | Activities | Deliverables | Time |
| --- | --- | --- | --- |
| 1. Business Understanding | Define objectives, assess situation, determine goals | Project charter, success criteria | 1–2 weeks |
| 2. Data Understanding | Collect data, describe data, explore, verify quality | Data report, data dictionary | 2–4 weeks |
| 3. Data Preparation | Clean, transform, feature engineer, split | Cleaned dataset, preprocessing pipeline | 2–6 weeks |
| 4. Modeling | Select techniques, build models, hyperparameter tune | Model card, experiment tracker | 2–8 weeks |
| 5. Evaluation | Evaluate results, review process, determine next steps | Evaluation report, go/no-go decision | 1–2 weeks |
| 6. Deployment | Plan deployment, monitor, maintain, document | Production model, monitoring dashboard | 2–6 weeks |

### 4.2 TDSP (Team Data Science Process) — Microsoft

Microsoft's agile data science lifecycle, good for larger teams:

```
┌─────────────────────────────────────────────────┐
│                 BUSINESS UNDERSTANDING           │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│                DATA ACQUISITION                  │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│    │ Pipeline │──│ Ingest   │──│ Validate │    │
│    └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│                MODELING (Iterative)              │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│    │ Feature  │──│ Train    │──│ Evaluate │    │
│    │ Eng.     │  │          │  │          │    │
│    └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│                DEPLOYMENT                        │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│    │ Package  │──│ Deploy   │──│ Monitor  │    │
│    └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────┘
```

### 4.3 AI Project Delivery Checklist

**Discovery Phase**:
- [ ] Client stakeholder interviews (min 5)
- [ ] Current state assessment
- [ ] Data availability & quality assessment
- [ ] Infrastructure audit
- [ ] Regulatory/compliance requirements identified
- [ ] Success criteria defined (quantitative)
- [ ] Risk register created
- [ ] Timeline and budget agreed

**Build Phase**:
- [ ] Data pipeline established
- [ ] Baseline model built
- [ ] Evaluation framework in place
- [ ] Experiment tracking implemented (MLflow, Weights & Biases)
- [ ] Model versioning and reproducibility ensured
- [ ] Unit tests for data and model code
- [ ] CI/CD pipeline for ML
- [ ] Security review conducted

**Deploy Phase**:
- [ ] Model serving infrastructure set up
- [ ] Monitoring and alerting configured
- [ ] Performance baselines established
- [ ] Documentation delivered
- [ ] Client team trained
- [ ] Handoff/knowledge transfer completed
- [ ] Success metrics review
- [ ] Ongoing support plan agreed

## 5. IP Development: How to Productize Consulting

### 5.1 The IP Ladder

```
Level 5: Product (SaaS product, licenseable)
Level 4: Framework (Methodology, playbook, certification)
Level 3: Tools (Internal automation, templates, accelerators)
Level 2: Knowledge (Whitepapers, talks, case studies)
Level 1: Experience (What you learn from client work)
```

**How to move up the ladder**:
1. **Level 1→2**: Document client outcomes → write case studies, speak at conferences
2. **Level 2→3**: Build reusable templates, code libraries, assessment tools
3. **Level 3→4**: Package methodology into training, certification programs
4. **Level 4→5**: Identify productizable components → spin off as SaaS

### 5.2 Productizable Consulting Assets

| Asset Type | Description | Revenue Model | Example |
| --- | --- | --- | --- |
| AI Maturity Assessment | Scorecard to evaluate org readiness | $5K–$15K per engagement | Like Deloitte's AI Maturity model |
| Data Quality Framework | Audit tools + remediation playbook | $10K–$30K per engagement | |
| Model Evaluation Suite | Standardized eval + benchmark toolkit | $20K–$50K license/yr | MLPerf, but simplified |
| Prompt Engineering Toolkit | Templates, guardrails, eval framework | $5K–$20K training | |
| AI Ethics Framework | Governance model, bias detection | $15K–$40K per engagement | |
| Custom Fine-tuning Pipeline | Data prep → training → deployment | $50K–$200K per project | |
| MLOps Accelerator | Pre-built CI/CD for ML | $20K–$60K deployment | |
| Industry-specific Model | Trained on industry data with expertise | $100K–$500K+ | LegalAI, MedAI |

### 5.3 IP Commercialization Strategy

```
Year 1: Build expertise through client work
  └─ Document everything in playbooks
  └─ Develop repeatable assessment frameworks

Year 2: Package IP
  └─ Create 3–4 "productized" offerings
  └─ Build templates and accelerators
  └─ Develop training curriculum

Year 3: Scale through IP
  └─ License frameworks to other consultancies
  └─ Launch certification program
  └─ Spin off SaaS product from most popular IP
```

## 6. Pricing (Daily Rates by Region and Expertise)

### 6.1 Global Rate Card (USD/day)

| Expertise Level | North America | Western Europe | UK | Australia | Middle East | SE Asia | India |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Junior (0–2 yrs) | $800–$1,500 | $700–$1,200 | $600–$1,000 | $700–$1,200 | $600–$1,000 | $400–$800 | $200–$500 |
| Mid (3–5 yrs) | $1,500–$2,500 | $1,200–$2,000 | $1,000–$1,800 | $1,200–$2,000 | $1,000–$1,800 | $800–$1,500 | $500–$1,000 |
| Senior (6–10 yrs) | $2,500–$4,000 | $2,000–$3,500 | $1,800–$3,000 | $2,000–$3,500 | $1,500–$3,000 | $1,200–$2,500 | $800–$1,500 |
| Principal/Partner | $4,000–$7,000 | $3,500–$6,000 | $3,000–$5,000 | $3,500–$6,000 | $3,000–$5,000 | $2,000–$4,000 | $1,500–$3,000 |

### 6.2 AI Specialization Premiums

| Specialization | Premium Over General Consulting |
| --- | --- |
| LLM / Generative AI | +25–40% |
| Computer Vision | +15–25% |
| NLP / NLU | +15–25% |
| MLOps / Infrastructure | +20–30% |
| AI Security / Red Teaming | +30–50% |
| AI Ethics / Governance | +10–20% |
| Reinforcement Learning | +25–35% |
| Edge AI / Embedded ML | +20–35% |

### 6.3 Rate Setting Formula

```
Target daily rate = 
  (Target annual income + Overhead + Profit) / Billable days

Target annual income: $100K–$300K+ (your desired take-home)
Overhead: 30–50% of salary (insurance, tools, office, marketing)
Profit margin: 20–40% (for agency)
Billable days: 180–220 days/year (rest is sales, admin, training)
```

**Example (Solo consultant, North America)**:
```
Target income: $200K
Overhead (40%): $80K
Total: $280K
Billable days: 190
Daily rate: $280K / 190 = $1,474/day
Hourly equivalent (8hr day): $184/hr
```

## 7. Scaling from Solo to Agency

### 7.1 Growth Stages

| Stage | Revenue | Team Size | Focus |
| --- | --- | --- | --- |
| **Solo** | $100K–$300K | 1 | Technical delivery, direct sales |
| **Boutique** | $300K–$1M | 2–5 | Hire delivery, founder sells |
| **Established** | $1M–$5M | 5–20 | Sales hire, repeatable offerings |
| **Agency** | $5M–$20M | 20–80 | Multiple practices, IP products |
| **Scale firm** | $20M+ | 80+ | Vertical specializations, SaaS spin-offs |

### 7.2 Key Hiring Milestones

| Revenue Milestone | First Hire | Second Hire | Third Hire |
| --- | --- | --- | --- |
| $200K | Subcontractor for overflow | — | — |
| $400K | Full-time ML Engineer | — | — |
| $750K | Operations/Admin | Junior Consultant | — |
| $1.2M | Sales/Business Dev | Senior Consultant | PM |
| $2M+ | Practice Lead | 2 more engineers | Marketing |

### 7.3 Scaling Ratios

| Metric | Solo | 5-person | 20-person |
| --- | --- | --- | --- |
| Utilization target | 60–70% | 70–75% | 70–80% |
| Billable vs non-billable | 70/30 | 65/35 | 60/40 |
| Revenue per consultant | $150K–$250K | $180K–$250K | $200K–$300K |
| Gross margin | 50–70% | 45–60% | 40–55% |
| Sales as % of revenue | 15–25% | 12–20% | 10–15% |

## 8. SOW Template

### MASTER SERVICES AGREEMENT (MSA) — Statement of Work Template

```
───────────────────────────────────────────
STATEMENT OF WORK NO. [X]
───────────────────────────────────────────

Project Name:   [Project Name]
Client:         [Client Company Name]
Consultant:     [Your Company Name]
Date:           [Date]
SOW Type:       [Fixed-Bid / T&M / Retainer / Value-Based]

───────────────────────────────────────────
1. PROJECT OVERVIEW
───────────────────────────────────────────

1.1 Executive Summary
[2–3 sentences describing the project]

1.2 Business Objectives
- Objective 1: [Measurable outcome]
- Objective 2: [Measurable outcome]
- Objective 3: [Measurable outcome]

1.3 Success Criteria
| # | Criterion | Measurement | Target |
|---|-----------|-------------|--------|
| 1 | [Criterion] | [How measured] | [Target value] |
| 2 | [Criterion] | [How measured] | [Target value] |

───────────────────────────────────────────
2. SCOPE OF SERVICES
───────────────────────────────────────────

2.1 In-Scope
- [Activity 1]
- [Activity 2]
- [Activity 3]
- [Deliverable 1]
- [Deliverable 2]

2.2 Out-of-Scope
- [Activity not included]
- [Activity not included]

2.3 Assumptions
- Client will provide access to [systems/data/people]
- Client technical contact: [Name]
- Decision-making authority: [Name]

───────────────────────────────────────────
3. DELIVERABLES & MILESTONES
───────────────────────────────────────────

| Phase | Milestone | Deliverable | Due Date | Acceptance Criteria |
|-------|-----------|-------------|----------|---------------------|
| 1. Discovery | Kickoff completed | Assessment report | Week 2 | Report approved |
| 2. Build | POC complete | Working prototype | Week 6 | Demo accepted |
| 3. Deploy | Production launch | Deployed model | Week 12 | KPIs met |
| 4. Handoff | Knowledge transfer | Training + docs | Week 14 | Team trained |

───────────────────────────────────────────
4. PRICING & PAYMENT SCHEDULE
───────────────────────────────────────────

4.1 Fee Structure: [Fixed-Bid / T&M / Retainer / Value-Based]

4.2 Total Fee: $[Amount]

4.3 Payment Schedule:
| Milestone | Deliverable | Amount | Due |
|-----------|-------------|--------|-----|
| [1. Signed SOW] | [—] | $[Amount] (X%) | Upon signing |
| [2. Phase 1 completion] | [Deliverable] | $[Amount] (X%) | Approval |
| [3. Phase 2 completion] | [Deliverable] | $[Amount] (X%) | Approval |
| [4. Final delivery] | [Deliverable] | $[Amount] (X%) | Acceptance |

4.4 Expenses: [Included / Reimbursed at cost (up to $X/mo)]

4.5 Late Payment: [1–1.5% / month interest after 30 days]

───────────────────────────────────────────
5. TEAM & RESOURCES
───────────────────────────────────────────

| Role | Name | Rate | Allocation |
|------|------|------|------------|
| Sr. AI Consultant | [Name] | $[rate]/day | X days |
| ML Engineer | [Name] | $[rate]/day | X days |
| PM | [Name] | $[rate]/day | X days |

───────────────────────────────────────────
6. TIMELINE
───────────────────────────────────────────

Start date:   [Date]
End date:     [Date]
Duration:     [X weeks]
Key milestones: [See Section 3]

───────────────────────────────────────────
7. TERMS & CONDITIONS
───────────────────────────────────────────

7.1 IP Ownership: [Consultant retains / Client owns / Joint ownership]
7.2 Confidentiality: [Standard MSA terms apply]
7.3 Warranty: [90 days defect correction]
7.4 Termination: [30 days written notice, work delivered paid for]
7.5 Change Requests: [Tracked, quoted separately, signed as SOW amendment]

───────────────────────────────────────────
8. APPROVALS
───────────────────────────────────────────

Client: ___________________   Date: _______
Name: [Name], [Title]

Consultant: ________________   Date: _______
Name: [Name], [Title]

───────────────────────────────────────────
```

## 9. AI Consulting Practice Financial Model

### Revenue Projection Template

| Month | Billable Consultants | Avg Rate/Day | Billable Days | Revenue | Expenses | Profit |
|-------|---------------------|-------------|-------------|---------|----------|--------|
| 1 | 1 | $1,500 | 15 | $22,500 | $15,000 | $7,500 |
| 2 | 1 | $1,500 | 18 | $27,000 | $15,000 | $12,000 |
| 3 | 2 | $1,400 | 16 | $44,800 | $25,000 | $19,800 |
| Q1 | — | — | — | $94,300 | $55,000 | $39,300 |
| 4 | 3 | $1,400 | 17 | $71,400 | $35,000 | $36,400 |
| 5 | 3 | $1,500 | 18 | $81,000 | $35,000 | $46,000 |
| 6 | 4 | $1,400 | 16 | $89,600 | $45,000 | $44,600 |
| Q2 | — | — | — | $242,000 | $115,000 | $127,000 |
| Year 1 Total | — | — | — | ~$500K–$800K | ~$300K–$500K | ~$200K–$300K |

### Expense Budget Template

| Category | Monthly | Annual | % of Revenue |
|----------|---------|--------|-------------|
| Salaries & benefits | $X | $X | 40–50% |
| Contractor/subcontractor | $X | $X | 10–20% |
| Office / coworking | $X | $X | 3–5% |
| Software & tools | $X | $X | 3–5% |
| Cloud infrastructure | $X | $X | 5–10% |
| Marketing & sales | $X | $X | 5–10% |
| Travel & entertainment | $X | $X | 2–5% |
| Insurance & legal | $X | $X | 1–3% |
| Professional development | $X | $X | 2–4% |
| Other | $X | $X | 3–5% |
| **Total** | **$X** | **$X** | **100%** |

## 10. Case Studies

### Case Study 1: Boutique GenAI Consultancy (15 people)

| Metric | Data |
| --- | --- |
| Founded | 2023 |
| Team | 15 (6 senior, 4 mid, 3 junior, 2 ops/sales) |
| Focus | Enterprise GenAI strategy and implementation |
| Avg engagement | $150K (3-month POC → $50K/mo retainer) |
| Day rate range | $1,200–$4,500 |
| Annual revenue | $4.2M |
| Gross margin | 52% |
| Utilization | 72% |

**Key success factors**:
- Published thought leadership early (newsletter, conference talks)
- Developed AI Maturity Assessment framework (productized IP)
- Focused on 2 verticals: healthcare and financial services
- Partnered with cloud providers for lead generation

### Case Study 2: Solo AI Consultant → Agency

| Stage | Year 1 | Year 2 | Year 3 |
| --- | --- | --- | --- |
| Revenue | $180K | $420K | $1.1M |
| Team | Solo | 3 people | 8 people |
| Avg rate | $1,800/day | $2,200/day | $2,500/day |
| Clients | 8 | 12 | 18 |
| Retainers | 0 | 3 | 7 |

**Scaling playbook**:
- Year 1: High-touch, delivery-focused, built portfolio
- Year 2: Hired senior ML engineer, developed IP (fine-tuning toolkit)
- Year 3: Added sales capacity, launched 2 productized offerings

## 11. AI Consulting Business Development

### Lead Generation Channels (ranked by effectiveness)

| Channel | Effort | Conversion Rate | Avg Deal Size | Time to Close |
| --- | --- | --- | --- | --- |
| Existing network referrals | Low | 30–50% | $50K–$150K | 2–6 weeks |
| Speaking engagements | Medium | 5–15% | $50K–$200K | 4–12 weeks |
| Content marketing | High | 2–5% | $20K–$100K | 8–24 weeks |
| Partnerships (cloud providers) | Medium | 10–25% | $100K–$500K | 4–16 weeks |
| Cold outbound | High | 1–3% | $30K–$100K | 8–20 weeks |
| Community (Discord, GitHub) | Medium | 5–10% | $10K–$50K | 4–12 weeks |
| Conference sponsorship | High | 3–8% | $50K–$200K | 8–16 weeks |

### Proposal-to-Win Ratios

| Stage | Ratio | Action |
| --- | --- | --- |
| Initial conversation → Proposal | 40–60% | Need more qualifying |
| Proposal → Shortlist | 30–50% | Improve proposal quality |
| Shortlist → Won | 40–60% (for top 2) | Differentiate, build relationships |
| Overall win rate | 15–25% | Industry average |

## 12. Risks & Mitigation

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Scope creep on fixed-bid projects | Margin erosion 10–40% | Clear SOW, change control process |
| Low utilization (under 60%) | Revenue shortfall 20–30% | Mix delivery with IP development |
| Single client dependency (>30% revenue) | Catastrophic if lost | Cap at 30%, diversify pipeline |
| Consultant burnout | Attrition, quality decline | Utilization cap at 80%, mental health support |
| Skills obsolescence (new models) | Losing competitive edge | R&D budget (10% of time), continuous learning |
| Difficulty hiring AI talent | Can't scale | Remote-first, competitive pay, equity |
| Client procurement delays | Cash flow crunch | Retainers, milestone billing, deposits |
| AI commoditization | Clients question high rates | Specialize in niche, create IP moat |

---
*Next: Playbook 04 — AI Product Ideas for market gap identification.*

# 03 — AI Candidate Sourcing and Outreach

## Overview

AI-powered candidate sourcing has transformed recruiting from a reactive, posting-and-waiting model to a proactive, intelligence-driven approach. Instead of relying solely on applicants who happen to find a job posting, AI sourcing systems continuously scan millions of professional profiles across platforms (LinkedIn, GitHub, Indeed, Stack Overflow, AngelList, etc.), identify candidates who match target profiles, score and rank them, and even initiate personalised outreach automatically.

This document provides a comprehensive technical and strategic overview of AI candidate sourcing and outreach, covering the tools, architectures, algorithms, and ethical considerations involved.

---

## 1. The AI Sourcing Landscape

### 1.1 Why AI Sourcing Matters

| Challenge | Manual Sourcing | AI-Powered Sourcing |
|---|---|---|
| Profiles reviewed per hour | 20–50 | 500–5,000 |
| Time to fill a specialised role | 45–60 days | 25–40 days |
| Candidates sourced per req | 50–150 | 200–800+ |
| Diversity of pipeline | Limited by recruiter network | Algorithmically broadened |
| Personalisation at scale | Not feasible | Thousands of customised messages |
| Passive candidate discovery | Low | Systematic |

### 1.2 Types of AI Sourcing

1. **Boolean search augmentation** — AI generates optimal Boolean/LinkedIn search strings from a job description
2. **Profile discovery** — Crawl and index professional profiles from multiple platforms
3. **Lookalike modelling** — Find candidates similar to known high-performing employees
4. **Intent-based sourcing** — Identify candidates who are likely open to new opportunities
5. **Skills-based sourcing** — Build pipelines around skills, not job titles
6. **Predictive sourcing** — Anticipate future hiring needs based on company growth patterns

---

## 2. Automated Sourcing from Professional Platforms

### 2.1 Platform-Specific Strategies

**LinkedIn Sourcing:**
- LinkedIn Recruiter API (eligible organisations) — structured profile access
- LinkedIn Sales Navigator — broader filters (company size, growth signals, seniority changes)
- LinkedIn Talent Insights — aggregate market data, talent pool analytics
- LinkedIn open-to-work signals (green banner, #opentowork, recruiter outreach preferences)

**GitHub Sourcing:**
- GitHub Search API — search by language, repository, location, contribution frequency
- GitHub Archive / GH Archive — historical event data for contribution pattern analysis
- Stars, forks, and pull requests as quality signals
- Dormant but highly skilled contributors (last active 3–6 months ago = likely open)

**Other Platforms:**
| Platform | Key Signals | API Available |
|---|---|---|
| Stack Overflow | Reputation score, tags, answers/year | Yes (Stack Exchange API) |
| Indeed | Resume database, job seeker intent | Yes (Indeed Publisher API) |
| AngelList / Wellfound | Startup experience, investor signals | Yes |
| Google Scholar | Publication count, citation index, h-index | Partial |
| Kaggle | Competition rank, kernels, datasets | Yes |
| Behance / Dribbble | Portfolio quality, project diversity | Partial |
| Meetup | Event attendance, group membership | Yes |

### 2.2 Web Scraping vs API Access

| Aspect | API Access | Web Scraping |
|---|---|---|
| Legality | Authorised, ToS-compliant | Grey area (subject to CFAA, GDPR, platform ToS) |
| Rate limits | 100–10K requests/day | Can bypass with proxies but risky |
| Data freshness | Real-time | Depends on crawl frequency |
| Structure | Clean JSON | Requires HTML parsing |
| Scalability | Can scale (up to plan limits) | Proxy management, CAPTCHAs, IP blocking |
| Cost | $50–$500/month per platform | Proxy costs + dev time |

**Ethical sourcing best practice**: Use official APIs wherever possible. Scraping public profiles for passive candidate discovery is legally contested in many jurisdictions (HiQ v. LinkedIn case in the US, GDPR restrictions in Europe).

### 2.3 Building a Multi-Platform Sourcing Pipeline

```
                    ┌─────────────────────┐
                    │  Job Description     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Requirement Extractor│
                    │ (skill, title, exp,  │
                    │  location, industry) │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │ LinkedIn     │   │ GitHub       │   │ Indeed       │
    │ Source Engine│   │ Source Engine│   │ Source Engine│
    └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Profile Unification  │
                    │ (dedup across        │
                    │  platforms)          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Scoring & Ranking    │
                    │ (ML model)           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ CRM / ATS Import    │
                    └─────────────────────┘
```

### 2.4 Profile Unification

Candidates often have profiles on multiple platforms. Unification involves:

- **Identity resolution**: Match profiles by email hash, name + location, LinkedIn URL, GitHub handle
- **Profile merging**: Union of skills (each platform reveals different facets), most recent experience from LinkedIn, most accurate technical skills from GitHub
- **Deduplication**: Keep the richest profile, discard duplicates
- **Signal aggregation**: Combine signals — "active on GitHub last week" + "LinkedIn open to work" = high intent

---

## 3. Lookalike Candidate Identification

### 3.1 The Lookalike Model

Lookalike modelling finds candidates who resemble a set of "seed" candidates (e.g., your top-performing engineers).

**Input:**
- Seed set: 50–500 profiles of known high-performing employees
- Candidate set: Millions of profiles from sourcing pipeline

**Output:**
- Ranked list of candidates with lookalike score (0–1)

### 3.2 Feature Engineering for Lookalike Models

For each profile, construct a feature vector:

**Explicit features:**
- Years of experience (total and relevant)
- Number of positions held
- Seniority level (junior, mid, senior, staff, principal)
- Educational attainment (highest degree, field, institution tier)
- Skill overlap with JD skills (count, weighted)
- Industry match
- Company size (current and previous)
- Leadership experience (managerial titles, team sizes)
- Certification count

**Derived / behavioural features:**
- Career velocity (promotion rate in years per promotion)
- Job hopping frequency (average tenure, jobs in last 5 years)
- Skill breadth (number of distinct skill categories)
- Skill depth (max proficiency level in any skill)
- Company prestige score (based on external rankings, market cap)
- Network size (LinkedIn connections, GitHub followers/stars)
- Content contribution (blog posts, conference talks, open source PRs)
- Education-to-career alignment (is the degree relevant to current role?)

### 3.3 Model Architectures

| Model | Pros | Cons | Typical AUC |
|---|---|---|---|
| Logistic Regression | Interpretable, fast | Limited expressiveness | 0.72–0.78 |
| Random Forest | Handles non-linearity, feature importance | Overfits on small seed sets | 0.78–0.85 |
| XGBoost / LightGBM | State-of-the-art for tabular data | Requires hyperparameter tuning | 0.82–0.90 |
| Neural network (MLP) | Can learn complex interactions | Requires large seed set (>500) | 0.80–0.88 |
| Siamese Network | Learns similarity directly | Complex training setup | 0.83–0.91 |
| k-NN with learned embeddings | Simple, interpretable | Scales poorly | 0.75–0.82 |

### 3.4 Handling Small Seed Sets

When you have < 50 known high performers:

- **Transfer learning**: Use a model pre-trained on industry-wide lookalike data, fine-tune on your small seed set
- **Rule-based hybrid**: Start with weighted rules, use seed set only for calibration (weight tuning)
- **Synthetic data augmentation**: Generate synthetic candidate profiles by interpolating between seed pairs
- **Active learning**: Label additional candidates from uncertain predictions to grow the seed set

### 3.5 Cold Start for New Roles

For roles you've never hired before:

1. Use public job description text as the seed — embed the JD, find candidates with similar skill embeddings
2. Use market benchmarks — what do other companies look for in this role? (inferred from similar JDs)
3. Use adjacent roles — candidates who are a good fit for "Senior ML Engineer" are likely similar to "Senior Data Scientist" and "ML Infrastructure Engineer"
4. Use occupation mapping — O*NET provides standardised occupation descriptors with required skills, knowledge, and abilities

---

## 4. Personalised Outreach Generation

### 4.1 The AI SDR (Sales Development Representative) for Recruiting

AI can generate highly personalised outreach messages at scale, mimicking the best practices of human recruiters.

**Outreach types:**
| Type | Channel | Personalisation Level | Conversion Rate |
|---|---|---|---|
| InMail (LinkedIn) | LinkedIn Messaging | Medium | 15–25% |
| Email | Direct email | High | 20–35% |
| SMS/WhatsApp | Mobile | High (risky) | 30–50% |
| GitHub message | GitHub | Low | 10–15% |
| Multi-channel sequence | Email → LinkedIn → SMS | Very High | 35–50% |

### 4.2 Personalisation Dimensions

AI-generated outreach is personalised along these axes:

**1. Professional context:**
- Refer to specific current role and company
- Mention specific project, skill, or achievement from their profile
- Reference shared experience (same previous company, same university)
- Mention mutual connections (if any)

**2. Job relevance:**
- Explain why their profile caught attention (specific skills, experience)
- Connect their work to the role's responsibilities
- Avoid generic "we're hiring" messages

**3. Timing:**
- Send when likely to check (LinkedIn: weekday morning; Email: Tuesday–Thursday 10am–12pm)
- Avoid holiday periods
- Trigger on relevant events: "Saw you contributed to [project] last week"

**4. Tone:**
- Match industry norms (startup: casual; finance: professional)
- Adapt to seniority (individual contributor vs VP)
- Avoid overly familiar language for senior candidates

### 4.3 Message Generation Architecture

```
[Candidate Profile]
        │
        ▼
┌──────────────────────────┐
│ Profile Summariser       │
│ (LLM: extract key facts) │
│ Role, skills, projects,  │
│ achievements, interests  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ JD Summariser            │
│ (LLM: extract key points)│
│ Role, requirements,      │
│ unique selling points,   │
│ company pitch            │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ Personalisation Engine   │
│ Match candidate + JD,    │
│ find hooks, connection   │
│ points, talking points   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ Message Generator (LLM)  │
│ Compose subject line +   │
│ body + CTA               │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ Quality & Compliance     │
│ Check (rules + LLM)     │
│ - No gender bias         │
│ - No exaggerated claims  │
│ - Legal compliance       │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ A/B Test Assignment      │
│ (variant A vs B per      │
│  segment)                │
└──────────────────────────┘
```

### 4.4 Prompt Engineering for Outreach

**System Prompt Template:**
```
You are an AI recruiting assistant. Generate a personalised outreach message 
for a candidate based on their profile and the job description.

Guidelines:
- Keep message under 150 words
- Be specific about why this candidate is a good fit
- Reference one specific achievement or skill from their profile
- Explain what makes the role/company compelling
- End with a clear, low-friction call to action
- Use [Company] as placeholder for company name
- Do not use gender-coded language
- Do not make promises about compensation or career growth
- Tone: professional but warm
```

**Few-shot examples:**
```
Candidate: Senior ML Engineer at Google, 8 years experience, led tensor processing 
unit optimisation team, PhD in CS from Stanford.
Role: Staff ML Engineer at [Company], working on LLM inference optimisation.

Message:
Hi [Name],

Your work leading TPU optimisation at Google caught my attention — the 3x latency 
reduction you achieved on large-scale inference is exactly the kind of impact we 
need at [Company]. We're building the next generation of LLM serving infrastructure 
and facing the same challenges you've been solving at scale.

Would you be open to a brief chat about what we're working on? No pressure, just 
curious if it aligns with your interests.

Best,
[Recruiter name]
```

### 4.5 A/B Testing Outreach

| Variable | Test | Control | Metric |
|---|---|---|---|
| Subject line | Specific ("Your work on TPU optimisation") | Generic ("Exciting opportunity") | Open rate |
| CTA | "Brief chat?" | "Apply here" | Response rate |
| Length | Short (< 100 words) | Long (200+ words) | Reply rate |
| Personalisation depth | 3 personalisation points | 1 point | Conversion to screening |
| Channel | LinkedIn first | Email first | Response time |
| Day of week | Tuesday 10am | Friday 3pm | Response rate |
| Sender name | Recruiter name | AI sender | Reply rate |

### 4.6 Sequence Automation

Multi-touch sequences increase conversion:

| Step | Channel | Delay | Content |
|---|---|---|---|
| 1 | LinkedIn InMail | Day 0 | First outreach, personalised |
| 2 | Email | Day 3 | Follow-up, additional context |
| 3 | LinkedIn InMail | Day 7 | "Just wanted to circle back" |
| 4 | Email | Day 14 | Case study / team culture content |
| 5 | LinkedIn InMail | Day 21 | Final message, "closing the role soon" |
| 6 | Auto-archive | Day 35 | Move to nurture campaign |

---

## 5. Candidate Scoring and Ranking

### 5.1 Scoring Models

Beyond basic fit, AI sourcing scores candidates on multiple dimensions:

**Fit Score (0–100):**
How well the candidate matches the specific job requirements.
- Skill match: 40%
- Experience relevance: 25%
- Education alignment: 10%
- Industry familiarity: 10%
- Location (willingness to relocate): 10%
- Work authorization: 5%

**Interest Score (0–100):**
How likely the candidate is to respond positively.
- Open-to-work signal: +30
- Years at current company > 2: +10
- Years at current company > 4: +20
- Recent promotion: -10 (likely less open)
- Recently laid off (from news/layoff data): +40
- Active GitHub contributions: +5
- Endorsements from recruiters: -5 (being heavily sourced = may be overwhelmed)

**Risk Score (0–100):**
How likely the candidate is to be lost or decline if hired.
- Counteroffer risk: high if highly specialised + currently well-compensated
- Relocation discomfort: high if strong local ties
- Competing offers: infer from recruiter interaction patterns

### 5.2 Combined Ranking

```
Overall Score = α × FitScore + β × InterestScore - γ × RiskScore
```

Weights (α, β, γ) can be:
- **Fixed** — Set by recruiting operations
- **Learned** — Optimised via logistic regression on historical outreach → hire conversion
- **Dynamic** — Adjusted per role type (e.g., hard-to-fill roles weight interest higher)

### 5.3 Tiered Pipeline Management

| Tier | Score Range | Action |
|---|---|---|
| A (Hot) | 85–100 | Immediate personalised outreach |
| B (Warm) | 70–84 | Automated personalised email sequence |
| C (Nurture) | 50–69 | Add to nurture campaign (monthly newsletter, relevant content) |
| D (Archive) | < 50 | Archive, review quarterly for pipeline refresh |

---

## 6. Tools and Platforms

### 6.1 Commercial AI Sourcing Tools

| Tool | Key Features | Pricing |
|---|---|---|
| **Gem** (gem.com) | AI sourcing, CRM, multi-channel outreach, unified inbox, analytics | $99–$299/seat/month |
| **Hiretual** (hiretual.com) | AI sourcing from 750M+ profiles, contact data enrichment, Chrome extension | $99–$199/seat/month |
| **SeekOut** | Advanced search filters, AI scoring, diversity sourcing, 800M+ profiles | $99–$299/seat/month |
| **Entelo** | Passive candidate discovery, predictive analytics, diversity tools | Custom pricing |
| **Ideal** | AI matching, resume screening, video interview analysis | Custom pricing |
| **Eightfold** | AI talent intelligence platform, skills inference, career path mapping | Enterprise pricing |
| **Pymetrics** | Gamified assessments + AI matching (emphasis on diversity) | Custom pricing |
| **Beamery** | Talent CRM, AI sourcing, candidate engagement, analytics | Custom pricing |
| **Talent Intelligence** (LinkedIn) | Talent pool analytics, workforce planning | Enterprise add-on |

### 6.2 Open Source / DIY Components

| Function | Tool |
|---|---|
| Profile scraping (LinkedIn) | Scrapy, Selenium (use with caution re: ToS) |
| Profile scraping (GitHub) | GitHub API + PyGithub |
| Embedding model | Sentence-Transformers (all-MiniLM-L6-v2) |
| Vector database | Qdrant, Weaviate, Pinecone |
| Search & Boolean generation | PySpark NLP, custom regex engine |
| Message generation | LangChain + OpenAI/Anthropic API |
| Email automation | SendGrid API, AWS SES |
| CRM / ATS integration | Greenhouse API, Lever API, Workday API |
| Workflow orchestration | Airflow, Prefect, n8n |
| Analytics | Metabase, Superset, Tableau |

### 6.3 Build vs Buy Decision Matrix

| Factor | Buy | Build |
|---|---|---|
| Speed to market | Weeks | Months |
| Data coverage (profile counts) | 750M+ | Limited by API access/scraping |
| Maintenance cost | Subscription ($10K–$100K/yr) | Engineering team ($150K–$500K/yr) |
| Customisation | Limited | Full control |
| Compliance risk | Vendor-managed | Self-managed |
| Competitive differentiation | Low | High (if done well) |
| Integration depth | API-level | Deep ATS/CRM integration |

---

## 7. Diversity Sourcing

### 7.1 AI for Diversity

AI sourcing can be explicitly designed to broaden, not narrow, candidate pipelines.

**Techniques:**
1. **Blind sourcing** — Remove demographic indicators from search criteria
2. **Skills-first search** — Prioritise skills over titles, companies, or degrees
3. **Bias-aware lookalike models** — Ensure seed sets are themselves diverse; otherwise you propagate existing homogeneity
4. **Source diversity tracking** — Monitor pipeline diversity by source channel
5. **Expansive Boolean generation** — Intentionally include organisations, universities, and communities known for underrepresented groups
6. **Name-blind outreach** — Avoid implicit gender/ethnicity bias in initial contact

### 7.2 Diversity Sourcing Platforms

| Platform | Focus | Features |
|---|---|---|
| **SeekOut Diversity** | Underrepresented tech talent | Filters for HBCU grads, women in tech, veterans, people with disabilities |
| **Jopwell** | Black, Latine, Native American professionals | Career community + job matching |
| **PowerToFly** | Women in tech | Community + job board + sourcing |
| **HireTalent** | Neurodiverse candidates | Accessibility-first platform |
| **Inclusively** | People with disabilities | Accommodation-based matching |
| **Veteran-focused** (RecruitMilitary, Hirepurpose) | Military veterans | Skill translation from MOS to civilian roles |

### 7.3 Measuring Diversity Impact

| Metric | Formula | Target |
|---|---|---|
| Pipeline diversity rate | (Underrepresented candidates / Total candidates) × 100 | > 25% |
| Source diversity index | 1 - Σ(p_i²) where p_i = proportion from each source | > 0.7 |
| Conversion parity | (URM → interview rate) / (non-URM → interview rate) | 0.8–1.2 |
| Offer acceptance parity | (URM offer accept rate) / (non-URM offer accept rate) | 0.9–1.1 |
| New hire diversity | URM hires / total hires | Industry benchmark + 5% |

---

## 8. Compliance and Ethics

### 8.1 Legal Landscape

| Jurisdiction | Key Regulation | Impact on Sourcing |
|---|---|---|
| US (Federal) | EEOC Title VII, ADEA, ADA | Disparate impact from AI sourcing is unlawful |
| New York City | Local Law 144 | Mandatory bias audit of AI hiring tools |
| Illinois | AI Hiring Law (HB 2557) | Requires notice of AI use in hiring, opt-out option |
| California | CCPA / CPRA | Candidate data privacy rights |
| EU | GDPR | Requires consent for data collection, right to explanation for automated decisions |
| EU | AI Act (proposed) | High-risk classification for hiring AI |
| Canada | PIPEDA | Consent for profile data collection |

### 8.2 Ethical Sourcing Principles

1. **Transparency** — Candidates should know if they were sourced through AI and have the right to opt out of AI-driven outreach
2. **Consent** — Respect "do not contact" signals; honour GDPR right to erasure and data portability
3. **Fairness** — Actively monitor sourcing for demographic disparities; adjust models to broaden rather than narrow pipelines
4. **Privacy** — Only collect data relevant to job fit; do not infer protected characteristics (race, religion, sexual orientation)
5. **Human oversight** — Final decisions (interview invitations, offers) require human review
6. **Accuracy** — Regularly audit profile data accuracy; a sourced candidate should not be contacted based on outdated or incorrect information

### 8.3 Sourcing Opt-Out Mechanisms

- Unsubscribe link in every automated outreach message
- Centralised preference centre (Beamery, Gem)
- Right to deletion of sourced profile data (GDPR Art. 17)
- Minimum 30-day retention of "do not contact" flag (permanent in some jurisdictions)

---

## 9. Implementation Guide

### 9.1 Phased Rollout

**Phase 1: Basic automation (Weeks 1–4)**
- Boolean search generation from JD
- LinkedIn/GitHub profile discovery (API-based)
- CSV export of candidate profiles

**Phase 2: Scoring & ranking (Weeks 5–10)**
- Build or configure fit/interest scoring
- ATS integration (push scored candidates)
- Manual review of AI-ranked candidates

**Phase 3: Outreach automation (Weeks 11–16)**
- Personalised email/InMail generation
- Multi-touch sequence automation
- A/B testing framework

**Phase 4: Advanced (Weeks 17–24)**
- Lookalike modelling
- Intent prediction
- Predictive hiring needs forecasting
- Full multi-channel orchestration

### 9.2 Metrics to Track

| Metric | Definition | Good | Excellent |
|---|---|---|---|
| Response rate | Responses / outreaches | > 20% | > 35% |
| Conversion to screen | Screens scheduled / responses | > 30% | > 50% |
| Pipeline fill rate | Sourced candidate pipeline / open reqs | 3× | 5× |
| Time to first contact | JD posted → first sourced candidate contacted | < 24h | < 4h |
| Diversity of sourced pipeline | % underrepresented | > 25% | > 35% |
| Offer rate from sourced | Offers / sourced candidates screened | > 10% | > 20% |
| Cost per sourced hire | Total sourcing cost / sourced hires | < $5K | < $2K |

---

## 10. Future Directions

### 10.1 Agentic Sourcing

AI agents that autonomously:
- Negotiate initial conversations with candidates
- Answer questions about role, team, and company
- Schedule screening calls directly on the recruiter's calendar
- Gather preliminary information (availability, salary expectations)
- Escalate promising candidates to human recruiters

### 10.2 Predictive Talent Pipelines

- Predict which skills will be in demand 6–12 months ahead (based on job posting trends, technology adoption curves, company growth plans)
- Proactively source candidates with those skills
- Build relationships before the requisition is opened

### 10.3 Hyper-Personalisation with Multimodal Input

- Analyse a candidate's blog posts, conference talks, open source PR comments
- Generate outreach that references specific technical opinions or contributions
- "I read your post about using Ray for distributed training — that's exactly what our ML platform team is building."

### 10.4 Ethical Scoring

Third-party certifications and standards for ethical AI sourcing:
- **EEOC AI Hiring Guidelines** — Compliance certification
- **NYC Local Law 144** — Mandatory bias audits
- **IEEE P7003** — Algorithmic bias considerations
- **EU AI Act Conformity** — High-risk AI certification

---

## References

- K. Datta et al., "AI-Powered Talent Sourcing: A Systematic Review," Journal of Applied Artificial Intelligence, 2023.
- D. Lewis, "The Effectiveness of Personalised Outreach in Tech Recruiting," Harvard Business Review, 2022.
- EEOC, "Selecting AI: The EEOC's Guidance on AI Hiring Tools," Technical Report, 2023.
- LinkedIn Talent Solutions, "Global Talent Trends 2024: AI in Recruiting," LinkedIn Research, 2024.
- Gem.com, "The State of AI in Talent Sourcing," Industry Report, 2024.
- M. Raghavan et al., "Mitigating Bias in Algorithmic Hiring," ACM Conference on Fairness, Accountability, and Transparency (FAccT), 2020.
- HiQ Labs v. LinkedIn Corp., 938 F.3d 985 (9th Cir. 2019) — Web scraping public data.
- EU AI Act, Proposal for a Regulation laying down harmonised rules on artificial intelligence, 2021/0106(COD).

# 02 — AI SaaS Playbook: Complete SaaS Playbook for AI-Native Companies

## 1. Executive Summary

The AI SaaS model combines traditional SaaS recurring revenue with AI-specific cost structures (inference, fine-tuning, model hosting). This playbook covers everything from pricing models to deployment options, with real case studies and ready-to-use calculator templates.

AI SaaS in 2025–2026 is characterized by:
- **Hybrid pricing**: Most companies combine usage-based + subscription models
- **PLG dominance**: Product-led growth is the primary channel for AI SaaS
- **High gross margins**: 65–85% achievable with careful cost management
- **NRR > 110%**: AI products typically see strong expansion revenue
- **Agentic add-ons**: New revenue streams from autonomous agent capabilities

## 2. AI SaaS Pricing Models

### 2.1 Per-Seat Pricing

Standard per-user pricing, familiar to enterprise buyers.

| Tier | Monthly Price | Features |
| --- | --- | --- |
| Individual | $19–$29/mo | Basic AI features, limited queries |
| Team | $39–$59/user/mo | Advanced features, team management |
| Business | $79–$99/user/mo | Custom models, priority support |
| Enterprise | Custom ($100–$250/user/mo) | Dedicated infra, SLA, SSO |

**Pros**: Simple, predictable revenue, easy to compare
**Cons**: Leaves money on table from heavy users, caps usage, doesn't reflect value

**Best for**: Collaboration tools, workflow automation, internal tools

### 2.2 Usage-Based Pricing (Consumption)

Pay-as-you-go based on tokens, queries, API calls, or compute consumed.

| Component | Example Pricing |
| --- | --- |
| Tokens (input) | $0.003–$0.015/1K tokens |
| Tokens (output) | $0.015–$0.06/1K tokens |
| API calls | $0.001–$0.10/call |
| Processing time | $0.01–$0.50/minute |
| Storage/vector DB | $0.50–$2.00/GB/mo |
| Active users/mo | $0.20–$1.00/user |

**Pros**: Scales with customer success, aligns cost to revenue, no waste
**Cons**: Revenue unpredictability, bill shock, harder to forecast

**Best for**: API products, infrastructure, high-variance usage patterns

### 2.3 Tiered Pricing (Recommended)

Bundles features and usage into distinct tiers. Most common AI SaaS model.

| Tier | Price | Features | Usage Limit |
| --- | --- | --- | --- |
| Free | $0 | 50 queries/mo, basic model | 10K tokens/mo |
| Starter | $19/mo | 500 queries/mo, standard model | 100K tokens/mo |
| Pro | $49/mo | 5,000 queries, advanced model, priority | 1M tokens/mo |
| Business | $149/mo | 50,000 queries, custom model, team features | 10M tokens/mo |
| Enterprise | Custom | Unlimited, dedicated, SLA, on-prem | Unlimited |

### 2.4 Freemium Pricing

Free tier as top-of-funnel. See also Playbook 07.

| Free Tier Limit | Conversion Trigger | Conversion Rate Benchmark |
| --- | --- | --- |
| Monthly usage cap (e.g., 100 queries) | Hits limit, sees upgrade screen | 3–8% |
| Feature gating (e.g., no API access) | Needs integration | 5–12% |
| Time-limited (14–30 day trial) | Trial expires | 10–25% |
| Limited users (1 user free) | Team collaboration need | 8–15% |

**Key metrics for freemium**:
- Free-to-paid conversion rate: 4–10% (good)
- Time-to-conversion: 30–90 days average
- Viral coefficient from free users: >0.3 is strong
- Free user CAC (usually $0–$2): effectively free acquisition

### 2.5 Hybrid Pricing (Emerging Standard)

Most successful AI SaaS companies use a combination:

| Component | Example (GitHub Copilot) | Example (Notion AI) |
| --- | --- | --- |
| Base subscription | $10–$39/user/mo | $10/user/mo |
| Usage add-on | Limited included, overage at $X | Included in subscription |
| Premium features | Code review, security | Unlimited AI queries |
| Enterprise | Custom pricing, on-prem | Custom, SAML, SSO |

### 2.6 Outcome-Based Pricing (Emerging)

Charge based on results delivered rather than inputs used.

| Model | How It Works | Example |
| --- | --- | --- |
| Per-task | $0.01–$0.50 per completed task | AI customer support ticket resolution |
| Per-transaction | % of transaction value | AI sales assistant: 1–3% of deal |
| Cost savings share | 20–30% of savings achieved | AI procurement optimization |
| Performance bonus | Base + bonus for hitting KPIs | AI marketing: $ base + $ per conversion |

## 3. AI SaaS Pricing Calculator Template

Use this template to model your pricing:

```python
# AI SaaS Pricing Calculator
# Replace inputs with your data

# ——— Inputs ———
avg_tokens_per_query = 2000
cost_per_thousand_input_tokens = 0.003   # GPT-4o mini
cost_per_thousand_output_tokens = 0.015  # GPT-4o mini
overhead_multiplier = 1.3  # Hosting, vector DB, API management
target_gross_margin = 0.75  # 75%
monthly_active_users = 10000
queries_per_user_per_month = 500

# ——— Cost Calculation ———
cost_per_query = (
    (avg_tokens_per_query * 0.7 * cost_per_thousand_input_tokens / 1000) +
    (avg_tokens_per_query * 0.3 * cost_per_thousand_output_tokens / 1000)
) * overhead_multiplier

monthly_inference_cost = cost_per_query * queries_per_user_per_month * monthly_active_users
monthly_cogs = monthly_inference_cost * 1.15  # 15% overhead

# ——— Price Calculation ———
target_revenue_per_user = cost_per_query * queries_per_user_per_month / (1 - target_gross_margin)
monthly_price = round(target_revenue_per_user, 2)

# ——— Output ———
print(f"Cost per query: ${cost_per_query:.4f}")
print(f"Monthly COGS per user: ${cost_per_query * queries_per_user_per_month:.2f}")
print(f"Price per user/mo (75% margin): ${monthly_price:.2f}")
print(f"Monthly COGS total: ${monthly_cogs:,.0f}")
print(f"Target Monthly Revenue: ${monthly_price * monthly_active_users:,.0f}")
```

### Ready-to-Use Pricing Grid (Based on Model)

| AI Model | Cost/1K Input | Cost/1K Output | Recommend Price/1K Queries | Margins at $0.02/query |
| --- | --- | --- | --- | --- |
| GPT-4o | $0.0025 | $0.010 | $0.05–$0.15 | 70–80% |
| GPT-4o mini | $0.00015 | $0.0006 | $0.01–$0.03 | 75–85% |
| Claude 3.5 Sonnet | $0.003 | $0.015 | $0.05–$0.20 | 65–78% |
| Claude 3 Haiku | $0.00025 | $0.00125 | $0.005–$0.02 | 70–82% |
| Gemini Pro 1.5 | $0.00125 | $0.005 | $0.02–$0.08 | 65–80% |
| Llama 3 70B (self-host) | $0.0001 | $0.0005 | $0.005–$0.01 | 80–90% |
| Mistral Large 2 | $0.002 | $0.006 | $0.03–$0.10 | 70–85% |

## 4. Key SaaS Metrics for AI Companies

### 4.1 Revenue Metrics

| Metric | Formula | AI SaaS Benchmark | Notes |
| --- | --- | --- | --- |
| ARR | MRR × 12 | $1M+ seed, $10M+ Series A | Standard SaaS metric |
| MRR | Total recurring monthly revenue | Growing 10–20% MoM (good) | Track by cohort |
| NRR | (Starting MRR + Expansion - Churn) / Starting MRR | 110–130% median AI SaaS | Key AI SaaS metric |
| GRR | (Starting MRR - Churn) / Starting MRR | 90–95% | True retention |
| ARPU | MRR / Total Customers | $50–$500/mo | Depends on segment |
| LTV | ARPU × Gross Margin × (1/Churn Rate) | 24–48 months | Higher for AI |
| CAC | Total Sales & Marketing / New Customers | PLG: $500–$2K; Ent: $15K–$50K | Channel specific |
| LTV/CAC | LTV / CAC | >3x healthy, >5x great | AI companies target 5–7x |
| Magic Number | (Q MRR - Q-1 MRR) × 4 / Q-1 S&M | >0.75 great, >1.0 exceptional | Efficiency metric |

### 4.2 AI-Specific Metrics

| Metric | Definition | Benchmark | Why It Matters |
| --- | --- | --- | --- |
| Inference GM | (Revenue - Inference Cost) / Revenue | >75% | Core AI profitability |
| Model Training ROI | (Revenue Attributed - Training Cost) / Training Cost | >10x over 2 years | Validate R&D spend |
| Compute per User | Avg compute cost / Monthly active user | Decreasing 20–30% YoY | Efficiency trend |
| Latency Budget | P95 API response time | <500ms for chat, <2s for gen | Affects conversion |
| Context Utilization | % of context window actually used | 40–70% | Optimization signal |
| Hallucination Rate | % of outputs with errors | <3% production, <1% regulated | Quality metric |
| Agent Task Success | % of autonomous tasks completed | >85% | Agentic product health |
| Time-to-Value | Days from signup to first "aha" moment | <1 day ideal, <7 days good | Activation metric |

### 4.3 Growth Metrics for AI PLG

| Metric | Definition | Target |
| --- | --- | --- |
| Activation Rate | % of signups who reach key moment | >40% within 24h |
| DAU/MAU | Daily active / monthly active | >30% good, >50% great |
| Stickiness | (DAU/MAU) × 100 | >30% |
| Viral Coefficient K | Invites × Conversion Rate | >1.0 is viral |
| Time-to-AHA | Days to first value moment | <1 day |
| PQL Rate | % of users becoming product-qualified | >10% |
| Free to Paid | Conversion rate from free to paid | >5% good, >10% great |
| Expansion MRR | Revenue from upgrades | 10–20% of MRR monthly |

## 5. PLG vs. Sales-Led for AI SaaS

### 5.1 Product-Led Growth (PLG) Motion

**Best for**: Developer tools, API products, horizontal AI tools, low ACV ($10–$500/mo)

| Stage | PLG Activity | Metrics |
| --- | --- | --- |
| Awareness | Content, SEO, open-source, social | Organic traffic, forks, stars |
| Acquisition | Self-serve signup, freemium, free trial | Signups, activation rate |
| Activation | First query, first API call, first output | Time-to-AHA, completion rate |
| Revenue | Upgrade prompt, usage limits, feature gates | PQL → paid, conversion rate |
| Expansion | Team invites, usage growth, department rollout | NRR, expansion MRR |
| Advocacy | Reviews, referrals, community contributions | NPS, referrals, community posts |

**PLG Tech Stack for AI SaaS**:

| Category | Tools |
| --- | --- |
| Analytics | Amplitude, Mixpanel, Heap |
| Product tours | Appcues, Pendo |
| Billing/Usage | Stripe, Chargebee, Metronome |
| Feature gating | LaunchDarkly, Statsig |
| CS platform | Intercom, Zendesk, Drift |
| Data warehouse | Snowflake, BigQuery |
| Reverse ETL | Hightouch, Census |

### 5.2 Sales-Led Motion

**Best for**: Enterprise products, high ACV ($20K+/yr), regulated industries, custom deployments

| Stage | Sales-Led Activity | Metrics |
| --- | --- | --- |
| Lead generation | ABM, events, outbound, partnerships | MQLs, SQLs |
| Qualification | BANT, MEDDIC, CHAMP | SQL → Opportunity rate |
| Demo/POC | Custom demo, proof-of-value | Demo-to-pilot rate |
| Pilot | Time-boxed deployment with success criteria | Pilot-to-paid rate |
| Procurement | Security review, legal, compliance | Sales cycle length |
| Deployment | Onboarding, training, migration | Time-to-go-live |
| Expansion | Account growth, multi-department | NRR, upsell rate |

**Enterprise Sales Tech Stack**:

| Category | Tools |
| --- | --- |
| CRM | Salesforce, HubSpot |
| Sales engagement | Outreach, SalesLoft |
| Demo creation | Loom, Walnut |
| CPQ | Zuora, DealHub |
| Contracting | Ironclad, PandaDoc |
| Security/compliance | Vanta, Drata |
| Customer success | Gainsight, Catalyst |

### 5.3 Hybrid: PLG + Sales-Led (Recommended for AI SaaS)

Most successful AI companies use both:

| ACV Range | Motion | Example |
| --- | --- | --- |
| $0–$500/mo | Self-serve PLG | OpenAI ChatGPT Plus |
| $500–$5K/mo | PLG + Inside sales | GitHub Copilot Teams |
| $5K–$50K/mo | Sales-led + PLG enablement | Notion AI Enterprise |
| $50K+/mo | Enterprise sales + solutions | Custom enterprise AI platforms |

## 6. AI SaaS Deployment Options

### 6.1 Cloud (SaaS Multi-Tenant)

| Factor | Details |
| --- | --- |
| Best for | B2B horizontal, SMB, mid-market |
| Infrastructure | AWS, GCP, Azure, shared GPU pools |
| Model hosting | Cloud APIs (OpenAI, Anthropic) or self-hosted |
| Security | SOC 2 Type II minimum, encryption at rest/transit |
| Gross margin | 70–85% (highest of deployment options) |
| Maintenance | Provider handles all maintenance |
| Time to deploy | Days to weeks |
| Example companies | Notion AI, Grammarly, Jasper |

### 6.2 Hybrid (Partially Dedicated)

| Factor | Details |
| --- | --- |
| Best for | Mid-market, regulated data sensitivity |
| Infrastructure | Shared control plane, dedicated model instances |
| Model hosting | Dedicated GPU instances per customer or per cluster |
| Security | SOC 2 + HIPAA/BAA, CCPA, GDPR compliance |
| Gross margin | 60–75% |
| Maintenance | Provider manages shared infra, customer manages config |
| Time to deploy | Weeks to months |
| Example companies | Copy.ai Enterprise, Writer.com, Typeface |

### 6.3 On-Premises / VPC

| Factor | Details |
| --- | --- |
| Best for | Enterprise, government, defense, regulated verticals |
| Infrastructure | Customer's own cloud VPC or data center |
| Model hosting | Packaged model containers, often compressed/quantized |
| Security | Full customer control, air-gapped possible |
| Gross margin | 50–65% (higher overhead, but premium pricing) |
| Maintenance | Customer handles infrastructure, provider handles software |
| Time to deploy | 1–6 months |
| Pricing | 2–5x cloud pricing, typically annual contracts |
| Example companies | Glean, Moveworks (early), enterprise deployments |

## 7. Case Studies

### 7.1 OpenAI API

| Metric | Data |
| --- | --- |
| Product | OpenAI API (GPT-4, GPT-4o, DALL-E, Whisper) |
| Pricing Model | Usage-based (per-token, per-image, per-minute) |
| Pricing Range | $0.00015–$0.06/1K tokens depending on model |
| Estimated ARR | $4B+ (2024) |
| Gross Margin | 50–70% (model dependent, inference-heavy) |
| Key Metric | API revenue growing 200%+ YoY (2023–2024) |
| PLG Motion | Self-serve signup, $5 in free credits, then usage billing |
| Enterprise | Custom pricing, dedicated capacity, SLAs |
| Lesson | Usage-based pricing scales with customer success but creates revenue volatility |

### 7.2 Notion AI

| Metric | Data |
| --- | --- |
| Product | AI writing, summarization, Q&A embedded in Notion |
| Pricing Model | Per-seat add-on to existing Notion subscription |
| Pricing | $10/user/mo (annual), $12/user/mo (monthly) |
| Estimated ARR | $200M+ (AI add-on, 2024) |
| Gross Margin | 75–85% |
| NRR | 120%+ (driven by team expansion) |
| PLG Motion | Viral team collaboration: one user invites their team |
| Key Insight | AI as an add-on to existing product converts at higher rates than standalone |
| Lesson | Bundling AI into existing workflows reduces friction and increases adoption |

### 7.3 GitHub Copilot

| Metric | Data |
| --- | --- |
| Product | AI pair programmer for code |
| Pricing Model | Per-seat monthly subscription |
| Pricing | Individual: $10/mo, Team: $19/user/mo, Enterprise: $39/user/mo |
| Estimated ARR | $300M+ (2024) |
| Gross Margin | 75–85% (inference costs offset by massive scale) |
| Adoption | 1.8M+ paid users (2024) |
| NRR | 115%+ |
| PLG Motion | Free trial (30 day), then upgrade. Org-wide deployment. |
| Key Insight | Developer tools have high willingness to pay — developers buy their own licenses |
| Lesson | Price to the end-user persona, not IT procurement |

### 7.4 Jasper AI

| Metric | Data |
| --- | --- |
| Product | AI content creation for marketing teams |
| Pricing Model | Tiered per-seat with usage limits |
| Pricing | Creator: $49/mo, Pro: $69/mo, Business: Custom |
| Estimated ARR | ~$75M (2024) |
| Gross Margin | 70–80% |
| Challenge | Competition from ChatGPT Plus ($20/mo broad AI) compressed pricing |
| Pivot | Expanded from pure content to enterprise marketing suite |
| Lesson | Horizontal AI platforms can disrupt vertical AI apps — build defensibility through workflow integration |

## 8. AI SaaS Unit Economics Deep Dive

### 8.1 Cost Structure Breakdown

| Cost Category | % of Revenue | Notes |
| --- | --- | --- |
| Model inference | 15–30% | Largest single COGS item |
| Cloud infrastructure | 5–10% | Hosting, DB, networking |
| R&D (engineering) | 20–35% | Model fine-tuning, product dev |
| Sales & marketing | 25–40% | Varies by motion (PLG lower) |
| G&A | 8–15% | Legal, finance, HR |
| **Net margin** | **–20% to +25%** | Most AI SaaS unprofitable early |

### 8.2 Improving Unit Economics

| Strategy | Impact | Timeline |
| --- | --- | --- |
| Model optimization (distillation) | Reduce inference cost 30–60% | 1–3 months |
| Caching common queries | Reduce inference cost 20–50% | 2–4 weeks |
| Batch processing | Reduce cost 40% for non-real-time | 1–2 months |
| Quantization (FP16 → INT8) | Reduce cost 50% | 2–4 weeks |
| Multi-model routing | Use cheap model for simple queries | 20–30% savings |
| Vertical fine-tuning | Smaller model, same accuracy | 3–6 months |
| Student-teacher distillation | Compress model 10x | 3–9 months |

## 9. AI SaaS Revenue Optimization Playbook

### 9.1 Pricing Optimization Process

```
Month 1: Research & Benchmark
  └─ Analyze competitors → create pricing grid
  └─ Customer interviews on willingness to pay (Van Westendorp)
  └─ Build pricing model (see calculator above)

Month 2: Test & Validate
  └─ A/B test pricing on landing page
  └─ Run willingness-to-pay survey to 100+ prospects
  └─ Pilot with 5–10 customers

Month 3: Launch & Monitor
  └─ Public pricing page
  └─ Monitor conversion by tier
  └─ Track usage patterns

Month 4+: Iterate
  └─ Adjust tiers based on data
  └─ Introduce new features at higher tiers
  └─ Annual plan discounts (15–25% off)
```

### 9.2 Psychological Pricing Tactics for AI

| Tactic | Application | Example |
| --- | --- | --- |
| Decoy pricing | Add "premium" tier to make "pro" look better | $49/$99/$199 → $99 tier gets most sales |
| Anchoring | Show high price first, then discount | "Was $199, now $99/mo" |
| Penny gap | $49 vs $50: small difference, big perception | Always .99 or .00 |
| Feature elimination | Remove features to lower tier, don't cap usage | Limits feel better than caps |
| Usage empathy | Show usage data, then suggest upgrade | "You've used 80% of your queries" |
| Social proof | "Join 10,000+ companies" | Builds trust and FOMO |
| Risk reversal | "Cancel anytime", "14-day money back" | Reduces purchase anxiety |

## 10. AI SaaS Go-to-Market Checklist

### Pre-Launch (8–12 weeks before)

- [ ] Define target persona and ICP
- [ ] Build pricing page with comparison table
- [ ] Set up Stripe/Chargebee billing
- [ ] Implement usage tracking and metering
- [ ] Create self-serve onboarding flow
- [ ] Build waitlist / early access program
- [ ] Prepare documentation + API references
- [ ] Configure analytics (Mixpanel/Amplitude)
- [ ] Set up customer support (Intercom/Zendesk)
- [ ] Establish SLAs for uptime and latency

### Launch (Week 1–4)

- [ ] Launch on Product Hunt
- [ ] Post in relevant AI/tech communities (Hacker News, Reddit, Discord)
- [ ] Activate early adopter program
- [ ] Monitor pricing page conversion
- [ ] Track activation metrics daily
- [ ] Collect testimonials and case studies
- [ ] Deploy PQL scoring
- [ ] Start outbound to ideal tier-1 accounts

### Post-Launch (Month 2–6)

- [ ] Analyze pricing tier performance
- [ ] A/B test pricing page
- [ ] Launch referral program
- [ ] Build self-serve enterprise upgrade flow
- [ ] Expand to 2nd use case / vertical
- [ ] Hire first sales person (if ACV > $10K)
- [ ] Start paid acquisition (if LTV/CAC > 3x)
- [ ] Publish AI benchmarks and ROI studies

## 11. Risks & Mitigation

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Model provider price cuts compress margins | Revenue drops 20–50% | Own the model (fine-tune), multi-model routing |
| Competitor undercuts pricing | Customer churn | Differentiate on output quality, integration depth |
| Infererence cost spikes (GPU shortage) | Margins shrink 10–15% | Contract GPU capacity, use spot instances |
| Open-source model disruption | Premium pricing questioned | Build proprietary data moats, fine-tuned expertise |
| Bill shock for customers | Customer complaints, churn | Usage alerts, caps, predictable pricing tiers |
| AI regulation limiting use cases | Market size shrinks | Build compliant features, target less regulated verticals |
| Model degradation (drift) | Quality drops, churn increases | Continuous evaluation, automated retraining |
| Security breach (prompt injection, data leak) | Trust destroyed | Red teaming, guardrails, third-party audits |

## 12. AI SaaS Pricing Model Decision Matrix

| Factor | Per-Seat | Usage | Tiered | Outcome |
| --- | --- | --- | --- | --- |
| Predictable revenue | ★★★★★ | ★★☆☆☆ | ★★★★☆ | ★★☆☆☆ |
| Aligned with value | ★★☆☆☆ | ★★★★★ | ★★★★☆ | ★★★★★ |
| Easy to understand | ★★★★★ | ★★★★☆ | ★★★★☆ | ★☆☆☆☆ |
| Customer preferred | ★★★★☆ | ★★★☆☆ | ★★★★★ | ★★★☆☆ |
| Upsell potential | ★★☆☆☆ | ★★★★★ | ★★★★☆ | ★★★★★ |
| Enterprise friendly | ★★★★★ | ★★★☆☆ | ★★★★☆ | ★★☆☆☆ |
| PLG compatible | ★★★★☆ | ★★★★★ | ★★★★★ | ★★☆☆☆ |
| Gross margin control | ★★★★★ | ★★★☆☆ | ★★★★☆ | ★★★★☆ |
| **Overall** | **Good for SW** | **Best for API** | **Best for SaaS** | **Emerging** |

## 13. Templates & Tools

### AI SaaS Monthly Business Review (MBR) Template

```
1. Revenue Summary
   - MRR: $___ (MoM growth: ___%)
   - ARR: $___ (YoY growth: ___%)
   - Net New ARR: $___
   - Logo count: ___ (Net new: ___)

2. Growth Metrics
   - New signups: ___ (MoM: ___%)
   - Activation rate: ___%
   - Free → Paid conversion: ___%
   - PQLs generated: ___
   - Sales accepted leads: ___

3. Retention & Expansion
   - Logo churn: ___%
   - Revenue churn: ___%
   - NRR: ___%
   - Expansion MRR: $___
   - Contraction MRR: $___

4. Unit Economics
   - Avg ACV: $___
   - CAC (PLG): $___
   - CAC (Sales): $___
   - LTV/CAC: ___
   - Payback period: ___ months
   - Gross margin: ___%

5. AI-Specific
   - Avg inference cost/revenue: ___%
   - Model latency (P95): ___ms
   - Hallucination rate: ___%
   - Top 3 features requested: ___

6. Action Items
   - [ ] Item 1
   - [ ] Item 2
   - [ ] Item 3
```

### Pricing Page A/B Testing Framework

| Variable | Variant A | Variant B | Expected Impact |
| --- | --- | --- | --- |
| Anchor price | Show highest tier first | Show middle tier first | Variant B: +15% mid-tier |
| Annual discount | 15% off | 25% off | Variant B: +20% annual |
| Feature comparison | Table layout | Card layout | Variant A: +10% conversion |
| Free tier prominence | Show free below paid | Show free first | Variant B: +25% signups, –10% paid |
| Usage limits | "500 queries" | "Unlimited for 500 tasks" | Variant B: +8% conversion |

---
*Next: 03-AI-Consulting-Playbook.md for consulting/agency models.*

# 07 — Freemium to Enterprise: The AI Monetization Ladder

## 1. Executive Summary

The journey from free to enterprise is the most critical growth path for AI SaaS products. This playbook provides a complete framework for designing your monetization ladder — from free tier through to enterprise contracts — with specific guidance on usage limits, feature gates, expansion strategies, and enterprise pricing.

### Why the Monetization Ladder Matters

| Segment | % of Users | % of Revenue | Conversion Path |
| --- | --- | --- | --- |
| Free users | 70–90% | 0% | Top of funnel |
| Paid individual | 5–15% | 5–15% | Self-serve upgrade |
| Team/Pro | 2–8% | 20–40% | Team expansion |
| Business | 1–3% | 30–50% | Sales-assisted |
| Enterprise | <1% | 10–30% | Full sales cycle |

## 2. Free Tier Design

### 2.1 Types of Free Tiers

| Type | Description | Best For | Example |
| --- | --- | --- | --- |
| **Freemium** | Free forever with limited features/usage | PLG, high volume | Canva, Slack, Notion |
| **Free trial** | Full access, time-limited (14–30 days) | Higher ACV products | GitHub Copilot (30d) |
| **Usage-limited trial** | Full access, usage-capped | API products | OpenAI ($5 free credits) |
| **Open-source** | Free code, paid hosting/managed | Dev tools | GitLab, Grafana |

### 2.2 Free Tier Design Principles

1. **Deliver real value immediately** — Free users must experience the "aha moment" within minutes
2. **Create friction at the right points** — Free tier should be good enough to love, limited enough to upgrade
3. **Collect data for upgrades** — Monitor usage patterns to identify upgrade triggers
4. **Make upgrade seamless** — One click to upgrade, no data loss, no disruption
5. **Build network effects** — Free users invite others (viral loop) or create content that attracts more users

### 2.3 Free Tier Limits Comparison

| Limit Type | Free | Pro (Paid) | Comments |
| --- | --- | --- | --- |
| **Usage cap** | 100 queries/mo | 5,000 queries/mo | Common, easy to understand |
| **Feature gating** | Basic model only | Advanced model | Good for quality differentiation |
| **User limit** | 1 user | 5 users | Team-upsell trigger |
| **Data limit** | 10 docs | 1,000 docs | Storage-based gating |
| **Integration limit** | No API | API access | Developer upgrade driver |
| **Quality limit** | Standard speed | Priority speed | Latency-based tiering |
| **Output limit** | 500 words/output | 4,000 words/output | Content creators upgrade |
| **History limit** | 7-day history | Unlimited history | Retention value |

### 2.4 Free Tier Cost Management

Free tier is a cost center — manage it carefully:

| Cost Type | Cost/Free User/Month | Management Strategy |
| --- | --- | --- |
| Inference | $0.10–$2.00 | Strict limits, use cheaper models for free tier |
| Storage | $0.01–$0.50 | Limit storage capacity, archive old data |
| Support | $0.05–$0.50 | Chatbot-only for free tier |
| Infrastructure | $0.10–$1.00 | Shared infra, lower priority queue |
| **Total** | **$0.26–$4.00** | Target: <$1/user/mo for free tier |

### 2.5 Free Tier KPIs

| Metric | Target | Red Flag |
| --- | --- | --- |
| Free → Paid conversion | >5% | <3% |
| Time-to-conversion | 30–90 days | >180 days |
| Free user activation rate | >40% | <25% |
| Free user DAU/MAU | >20% | <10% |
| Free user NPS | >30 | <10 |
| Cost per free user | <$1/mo | >$3/mo |
| Viral coefficient (K) | >0.3 | <0.1 |

## 3. Growth Levers: Usage Limits & Feature Gates

### 3.1 Usage Limit Types

| Limit Type | How It Works | User Perception | Best For |
| --- | --- | --- | --- |
| **Hard cap** | Physically cannot exceed limit | Negative (feels restrictive) | Cost-sensitive products |
| **Soft cap** | Warn at limit but allow overage (billable) | Neutral to positive | API products |
| **Throttling** | Slow down after limit | Negative (frustrating) | Resource-intensive features |
| **Quality degradation** | Use cheaper model after limit | Mixed (not obvious) | High-volume applications |
| **Time-based** | Lock for X hours after limit | Negative | Usage-sensitive apps |

### 3.2 Recommended Usage Limit Strategy

```
Free Tier: 100 queries/month (hard cap)
  └─ "You've reached your free limit. Upgrade for unlimited queries."
  
Pro Tier: 5,000 queries/month (soft cap → overage)
  └─ "You've used 80% of your queries. You have 500 remaining."
  └─ At limit: "You're out of queries. Auto-add 1,000 more for $10?"
  
Business Tier: 50,000 queries/month (soft cap → overage)
  └─ "Your team used 45,000 queries this month. Usage growing 20% MoM."
  └─ "Consider our $149/mo unlimited plan."
  
Enterprise Tier: Unlimited (within reason)
  └─ Custom pricing based on expected volume
```

### 3.3 Feature Gates (What to Gate vs. Not)

**DO gate (upgrade triggers)**:
- Model quality (free = cheap model, paid = best model)
- Advanced features (custom instructions, fine-tuning)
- Collaboration (team workspaces, shared resources)
- API access
- Priority support
- SSO/SAML
- Audit logs
- Advanced analytics
- Custom branding

**DON'T gate (table stakes)**:
- Core functionality (must work for free)
- Security basics (encryption, basic compliance)
- Basic export
- Account management
- Documentation and learning resources

### 3.4 Upgrade Trigger Points (When People Convert)

Based on industry data, these are the most common upgrade triggers:

| Trigger | Conversion Rate | Example |
| --- | --- | --- |
| Hits usage limit | 8–15% | "You've used 100% of your queries" |
| Wants to invite team | 12–20% | "Upgrade to Pro to add team members" |
| Needs API access | 15–25% | "Connect your tools with our API" |
| Hits storage limit | 5–10% | "Your workspace is full" |
| Needs priority support | 3–8% | "Get answers in under 1 hour" |
| SSO/compliance requirement | 8–15% | "Your company requires SSO" |
| Wants analytics/reporting | 5–10% | "See usage and impact dashboard" |

## 4. Expansion Revenue Strategies

### 4.1 Types of Expansion Revenue

| Type | Description | Typical Uplift | Example |
| --- | --- | --- | --- |
| **User expansion** | More users within the account | 2–5x over time | Slack: 1 user → 500 users |
| **Seat upgrade** | Free → paid within same account | 1x → $/user/mo | GitHub Copilot |
| **Usage expansion** | More usage drives higher bill | 2–10x over time | AWS, OpenAI API |
| **Feature/Module upsell** | Add-on features | 20–50% increase | Notion AI add-on |
| **Tier upgrade** | Basic → Pro → Enterprise | 2–5x per tier | Zoom, Intercom |
| **Cross-sell** | Different product line | 30–100% increase | HubSpot CRM → Marketing Hub |

### 4.2 Expansion Playbook

**Phase 1: Land (Acquire)**
```
Goal: Get first user or small team on paid plan
Tactics:
- Low-entry pricing ($10–$20/user/mo)
- Free trial with full features
- Self-serve upgrade
- Credit card not required (for some)
```

**Phase 2: Expand (Grow)**
```
Goal: Grow within account
Tactics:
- Usage-based upselling ("Your team is growing!")
- Department expansion ("Need more seats?")
- Feature adoption emails ("Try our new AI feature")
- Usage analytics shared with admin
- Quarterly business review (enterprise)
```

**Phase 3: Deepen (Entrench)**
```
Goal: Make product indispensable
Tactics:
- Custom integrations
- API access and developer tools
- Data portability costs (stickiness)
- Multi-product bundling
- Executive sponsorship
- Training and certification
```

### 4.3 Expansion Metrics

| Metric | Definition | Target |
| --- | --- | --- |
| **NRR (Net Revenue Retention)** | (Starting MRR + Expansion - Churn - Contraction) / Starting MRR | >110% |
| **Expansion MRR** | Revenue growth from existing customers | >15% of MRR |
| **Logo expansion rate** | % of accounts that increased spend in a year | >30% |
| **Seat expansion rate** | Growth in paid seats per account | >20% YoY |
| **Module adoption rate** | % of customers using 2+ features | >40% |
| **Time to first expansion** | Days from first payment to first upsell | <180 days |

## 5. Enterprise Pricing

### 5.1 When to Offer Enterprise Pricing

| Signal | Description |
| --- | --- |
| Customer asks for quote | They've outgrown self-serve tiers |
| >100 users | Volume discount needed |
| SSO/SAML required | Enterprise security requirement |
| Custom compliance needs | SOC 2, HIPAA, GDPR specifics |
| Dedicated infrastructure | On-prem/VPC deployment |
| Multi-year commitment | Annual or multi-year preferred |
| Custom integrations | Needs bespoke work |

### 5.2 Enterprise Pricing Models

| Model | Description | Best For |
| --- | --- | --- |
| **Volume discount** | $/user decreases with volume | Seat-based products |
| **Annual contract** | Discount for annual commitment | Predictable revenue |
| **Usage commitment** | Commit to $X/mo usage for discount | API products |
| **Platform fee + usage** | Base fee + variable usage | Enterprise platforms |
| **SLA-tiered** | Higher SLA = higher price | Mission-critical apps |
| **Per-workspace** | Price by workspace/org | Multi-team products |

### 5.3 Enterprise Pricing Template

```
┌──────────────────────────────────────────────────────────┐
│                   ENTERPRISE QUOTE                        │
│                      [Company Name]                       │
│                    Valid through: [Date]                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│ PLAN: Enterprise Cloud                                   │
│                                                          │
│ Annual Platform Fee: $24,000 ($2,000/mo)                 │
│   - Includes first 50 users                             │
│                                                          │
│ Additional Users: $15/user/mo (billed annually)          │
│   - 50–200 users                                         │
│   - Volume discount available for 200+                   │
│                                                          │
│ Estimated Total Year 1:                                  │
│   50 users included: $24,000                             │
│   150 additional × $15 × 12: $27,000                     │
│   Total: $51,000 ($4,250/mo)                             │
│                                                          │
│ What's Included:                                         │
│  ✓ Unlimited queries                                     │
│  ✓ Best model access                                     │
│  ✓ SSO/SAML / SCIM                                       │
│  ✓ SLA: 99.9% uptime                                     │
│  ✓ Dedicated CSM                                         │
│  ✓ Priority support (1-hour response)                    │
│  ✓ Audit logs & analytics                                │
│  ✓ Custom integrations (up to 5)                         │
│  ✓ Security review support                               │
│  ✓ Quarterly business review                             │
│                                                          │
│ Optional Add-ons:                                        │
│  • On-prem deployment: +50%                              │
│  • Premium SLA (99.99%): +20%                            │
│  • AI Agent add-on: $25/user/mo                          │
│  • Advanced analytics: $5K/mo                            │
│                                                          │
│ Payment Terms: Net 30                                    │
│ Contract: Annual, auto-renew                             │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 5.4 SLA Pricing

| Uptime SLA | Typical Price Premium | Credit for Downtime |
| --- | --- | --- |
| 99.0% (standard) | Included in base | 5% credit per 1% below |
| 99.5% | +10–15% | 10% per 0.5% below |
| 99.9% (enterprise) | +20–30% | 10% per 30 min down |
| 99.99% (mission-critical) | +50–100% | 25% per 5 min down |

### 5.5 Enterprise Discounting Playbook

| Scenario | Discount Range | Conditions |
| --- | --- | --- |
| Annual contract vs monthly | 15–25% off | Standard |
| Multi-year (2–3 year) | 20–30% off year 1 | Only for large deals |
| Volume >500 users | 15–30% off list | Volume tiers |
| Competitive displacement | 20–40% off | Designed to win |
| Non-profit / education | 25–50% off | Mission-driven |
| Early adopter / strategic | 30–50% off | Reference-able account |

**Discounting rules**:
- Never discount without something in return (term, commitment, reference)
- Always show list price first, then discount (anchoring)
- Cap discounts at 40% without VP approval
- Have standard "tier 1, 2, 3" discount levels

## 6. Negotiation Playbook

### 6.1 Common Enterprise Objections

| Objection | Response |
| --- | --- |
| "It's too expensive" | "Compared to what? What would it cost to build this internally? Let's calculate the ROI together." |
| "We need a discount" | "What would help us justify that? Longer term? Larger commitment? Being a reference account?" |
| "We're evaluating competitors" | "That's smart. What criteria are you using? Let's make sure we're evaluated on the dimensions that matter most to you." |
| "We can't sign annual" | "We offer monthly at 20% premium. But annual aligns best with the value you'll get." |
| "Security needs to review" | "Absolutely. Here's our trust center, SOC 2 report, and security whitepaper. Let's schedule a call with your security team." |
| "We need a pilot first" | "Great. Let's scope a 30-day pilot with these success criteria. If we meet them, we proceed." |
| "Budget is frozen until Q3" | "Let's start a pilot now so you have the data to make the case. We can invoice in Q3." |
| "We already have a vendor" | "What's working well? What isn't? Many of our customers came from [competitor] because [reason]." |

### 6.2 The Negotiation Process

```
Step 1: Establish value
  └─ ROI calculator, case studies, benchmarks
  └─ "You're spending $XXX on [current solution]. We can reduce that by Y%."

Step 2: Anchor high
  └─ Start with list price
  └─ "Our standard enterprise pricing is $XX/user/mo."

Step 3: Trade concessions
  └─ "If we do X% discount, can you commit to Y?"
  └─ Never give without getting

Step 4: Trial close
  └─ "If we can get to $X, do we have a deal?"
  └─ "What else do we need to address?"

Step 5: Legal/Procurement
  └─ Have standard terms ready
  └─ Involve legal early for large deals

Step 6: Close
  └─ Send Docusign immediately after verbal agreement
  └─ "We can start onboarding as soon as this is signed."
```

### 6.3 Negotiation Concession Matrix

| Their Ask | Our Give | Our Get |
| --- | --- | --- |
| 20% discount | 20% off first year | 2-year commitment |
| 30% discount | 30% off | Public case study + reference calls |
| Additional users | Free users first 3 months | Annual commitment for all users |
| Custom feature | Build it on roadmap | Naming rights, reference |
| Faster payment terms | Net 15 instead of Net 30 | Monthly billing (reduces risk) |
| Pilot discount | 50% off first 3 months | Auto-convert to full price |

## 7. Monetization Ladder: Complete Example (Notion AI)

### 7.1 Ladder Structure

| Tier | Price | Users | AI Features | Limits |
| --- | --- | --- | --- | --- |
| **Free** | $0 | 1 | Limited AI | 50 AI queries/mo |
| **Plus** | $10/mo | Unlimited | Full AI | Unlimited AI queries |
| **Business** | $18/user/mo | Team | Full AI + admin | +Custom model training |
| **Enterprise** | Custom | Org-wide | Full AI + dedicated | +SSO, SLA, audit logs |

### 7.2 Expansion Paths

```
Free → Plus
  └─ Hits 50 AI query limit → "Upgrade for unlimited AI"

Plus → Business
  └─ Invites 2–5 team members → "Business includes admin controls"
  └─ "Your team has grown — manage them with Business"

Business → Enterprise
  └─ Hits 50+ users → "Volume pricing available"
  └─ SSO requirement → "You need Enterprise for SSO"
  └─ Needs audit logs → "Upgrade to Enterprise"

Expansion within Plus:
  └─ Notion AI add-on: +$10/user/mo
```

### 7.3 Performance (Estimated)

| Metric | Value |
| --- | --- |
| Free users | 50M+ |
| Paid users (Plus + Business) | 4M+ |
| Enterprise logos | 1,000+ |
| NRR | 120%+ |
| Conversion rate (free → paid) | ~5–8% |

## 8. Tier Design Framework

### 8.1 Determining Number of Tiers

| # of Tiers | Best For | Pros | Cons |
| --- | --- | --- | --- |
| 2 (Free + Pro) | Developer tools, simple products | Simple, easy decision | Limited upsell path |
| 3 (Free + Pro + Business) | Most SaaS (recommended) | Good balance | Middle tier can confuse |
| 4 (Free + Starter + Pro + Enterprise) | Complex products | Clear upgrade path | Feature overload |
| 1 (Paid only) | Niche enterprise | Simple | No free funnel |

### 8.2 Tier Design Rules

1. **Feature gap between tiers should widen** — Not incremental, but meaningful jumps
2. **Each tier targets a different persona** — Individual vs. team vs. org
3. **Price differences should follow value differences** — 2–5x between tiers
4. **Enterprise tier is always "contact us"** — Custom pricing signals flexibility
5. **Annual discount = 15–25%** — Enough to convert, not enough to devalue monthly

### 8.3 Tier Pricing Multiples

```
Free: $0
  └─ 10% of features, 0% of usage capacity
Pro: $19–$29/mo
  └─ 2–3x the price of the tier below
Business: $49–$99/user/mo
  └─ 2–3x the price of the tier below
Enterprise: Custom ($100–$250/user/mo)
  └─ 2–5x business tier (feature premium + SLA premium)
```

## 9. Monetization Ladder Optimization Playbook

### 9.1 A/B Testing the Ladder

| Test | Variant A | Variant B | Expected Impact |
| --- | --- | --- | --- |
| Free tier generosity | 50 queries/mo | 100 queries/mo | Variant B: +20% signups, –5% paid conversion |
| Pro price point | $19/mo | $29/mo | Variant A: +10% conversion, Variant B: +15% revenue per user |
| Annual discount | 15% off | 25% off | Variant B: +25% annual adoption |
| Feature placement | Advanced model in Pro | Advanced model in Business | Variant B: +15% business tier adoption |
| Usage limit display | Show exact number remaining | Show percentage | Variant A: +8% upgrade when <10% remaining |

### 9.2 Pricing Page Optimization Checklist

- [ ] Most popular tier highlighted (badge, different color)
- [ ] Comparison table with clear "✓" vs "—"
- [ ] Annual/monthly toggle with savings shown
- [ ] Social proof ("Join 10,000+ teams on Pro")
- [ ] Money-back guarantee prominently displayed
- [ ] "Enterprise" section with features + contact form
- [ ] FAQ section addressing common pricing objections
- [ ] Customer logos/testimonials near pricing
- [ ] Feature comparisons are concrete, not abstract
- [ ] Call-to-action buttons clearly visible on every tier

### 9.3 Quarterly Revenue Review

Each quarter, analyze:
1. **Tier distribution**: % of customers on each tier
2. **Conversion rate**: Free → paid, trial → paid by tier
3. **Expansion rate**: % of accounts that upgraded
4. **Contraction rate**: % of accounts that downgraded
5. **Churn by tier**: Which tier has lowest retention?
6. **Cost to serve**: COGS by tier (model costs, support)
7. **NRR**: Are we expanding or contracting?
8. **Payback period**: Months to recover CAC

**Optimization cycle**:
```
Month 1: Analyze data → identify opportunities
Month 2: Design tests (3–5 experiments)
Month 3: Run A/B tests → analyze results
Quarter 2: Implement winning variants
Repeat quarterly
```

## 10. Risks & Anti-Patterns

| Anti-Pattern | Why It Fails | Better Approach |
| --- | --- | --- |
| Free tier is too generous | Low conversion, high cost | Cut features, usage, or time |
| Free tier is useless | No one signs up | Make free valuable enough to attract users |
| Too many tiers | Decision paralysis | Max 4 tiers (Free + 3 paid) |
| No annual discount | Lower NRR, less cash | 15–25% annual discount |
| Hiding pricing | Frustrated buyers | Public pricing (within limits) |
| One-size pricing | Leaves money on table | Segment and optimize per segment |
| No usage data sharing | Users don't know they're close to limit | Show usage proactively |
| Enterprise pricing too low | Undermines deal, low margin | 50–100% premium over highest tier |
| Perpetual discounts | Trains customers to ask | Time-box discounts, linked to commitments |

---
*Next: Playbook 08 — Agentic Services Pricing (NEW category).*

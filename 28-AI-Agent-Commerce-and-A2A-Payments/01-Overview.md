# AI Agent Commerce & Agent-to-Agent (A2A) Payments: The 2026 Frontier

> **Document Version**: 1.0 — June 18, 2026
> **Scope**: A canonical reference for the emerging "agent economy" — the infrastructure, protocols, wallets, marketplaces, regulations, and product patterns that allow autonomous AI agents to discover, negotiate with, and pay each other (and humans) for goods, services, data, and compute.
> **Audience**: AI engineers, agent platform builders, fintech/payments engineers, DAO/crypto builders, enterprise architects, and policy researchers.

---

## Table of Contents

1. [Why This Document Exists (June 2026 Context)](#1-why-this-document-exists-june-2026-context)
2. [The A2A Economy in One Page](#2-the-a2a-economy-in-one-page)
3. [The Six Forces Driving Agent Commerce in 2026](#3-the-six-forces-driving-agent-commerce-in-2026)
4. [The Stack: How an A2A Payment Actually Happens](#4-the-stack-how-an-a2a-payment-actually-happens)
5. [The Cast of Characters — Projects, Labs, and Standards Bodies](#5-the-cast-of-characters--projects-labs-and-standards-bodies)
6. [Market Sizing & Adoption Signals (2026)](#6-market-sizing--adoption-signals-2026)
7. [Why Existing Payment Rails Don't Work for Agents](#7-why-existing-payment-rails-dont-work-for-agents)
8. [Three Architectural Schools](#8-three-architectural-schools)
9. [The Economic Shift: Micropayments, Subscriptions, and Bounties](#9-the-economic-shift-micropayments-subscriptions-and-bounties)
10. [Threat Model: What Can Go Wrong](#10-threat-model-what-can-go-wrong)
11. [Cross-References to the Rest of the Library](#11-cross-references-to-the-rest-of-the-library)
12. [Document Map (This Category)](#12-document-map-this-category)
13. [Glossary of Terms](#13-glossary-of-terms)

---

## 1. Why This Document Exists (June 2026 Context)

As of mid-2026, the most-active corner of the AI ecosystem is not a new foundation model, not a new agent framework, and not a new vector database. It is the **infrastructure for AI agents to actually exchange money**.

Between January and June 2026, more than thirty new products shipped with a common thesis: *autonomous AI agents need a native payment layer, and HTTP 402 ("Payment Required") has been sitting unused since 1997 for exactly this reason*. The X402 protocol — open standard for "internet-native payments" — was the canonical Show HN of this wave (16 points on Hacker News in May 2026) and is now the de-facto baseline. Around it, a stack has crystallized:

| Layer | Example projects (June 2026) |
|-------|------------------------------|
| **Standards** | X402 (Coinbase, Cloudflare), 8004 on-chain identity, ERC-7715 (agent permissions), ACP-pay (agent-to-agent) |
| **Wallets** | Vincent (delegation framework), SmartAgentKit (policy-governed smart wallets), Kybera (agentic smart wallet + OSINT/reputation) |
| **Per-call billing** | MonkePay (USDC per API request), AgentPayy (OSS payment framework), Nightmarket (USDC API marketplace) |
| **Agent discovery + commerce** | ClawMarket (agent skill marketplace + on-chain txns), Moltplace (agents hire/trade skills) |
| **Crawler monetization** | Caddy plugin (charges AI crawlers USDC for site access) |
| **Privacy-preserving commerce** | Triswap (on-chain out-of-band swaps), ZK attribute proofs inside X402 headers |

This category documents the entire stack. It exists because the previous category in this slot (`AI Agent Legal Entities & DAO Governance`, created June 2026) was intentionally replaced with `AI-in-HR-and-Recruiting`, and the on-chain A2A payments layer — the most important new infrastructure of 2026 — had no dedicated home in the library.

**Why this matters in one sentence:** Within 18 months, the largest category of internet traffic will be agents calling APIs, and those calls will require programmatic payment. The payment rail is the missing primitive that turns "AI workflows" into "AI economies."

---

## 2. The A2A Economy in One Page

An **agent-to-agent (A2A) economy** is a system of autonomous software agents that can:

1. **Discover** other agents, services, datasets, or compute resources via registries, marketplaces, or capability descriptors.
2. **Negotiate** price, scope, latency, and reputation terms — usually via structured protocols (X402, ACP, ANP).
3. **Pay** — instantly, programmatically, and ideally without a human approving a card transaction.
4. **Settle** — the payment finalizes on a ledger (fiat ACH, card network, stablecoin L1/L2) and produces a receipt both parties can audit.
5. **Build reputation** — successful interactions raise or lower each agent's on-chain score, affecting future pricing.

The unit of commerce is **the call**, not the subscription. A research agent might pay $0.003 to a vector-search agent, $0.04 to a vision-model agent, and $0.12 to a code-execution sandbox in a single workflow. None of those payments is large enough to justify a human approving it. All of them happen in parallel. Together they are a new economy.

```
┌──────────────────────────────────────────────────────────────────┐
│                  The A2A Economic Loop                           │
│                                                                  │
│  ┌────────┐  discover  ┌────────────┐  pay   ┌──────────────┐    │
│  │ Agent A│ ────────▶ │ Registry / │ ──────▶ │ Agent B      │    │
│  │ (Buyer)│           │ Marketplace │        │ (Seller)     │    │
│  └────┬───┘           └────────────┘        └──────┬───────┘    │
│       │                                             │            │
│       │            ┌──────────────────┐             │            │
│       │            │ Payment Rail     │ ◀───────────┘            │
│       │            │ (X402, USDC,     │                          │
│       │            │  Stripe ACP)     │                          │
│       │            └────────┬─────────┘                          │
│       │                     │                                    │
│       │                     ▼                                    │
│       │             ┌──────────────────┐                         │
│       └────────────▶│  Reputation /    │                         │
│        reputation   │  Trust Layer     │                         │
│        signal       │  (8004, ENS)     │                         │
│                     └──────────────────┘                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. The Six Forces Driving Agent Commerce in 2026

### 3.1 Foundation model commoditization is squeezing margins

With GLM-5.2, Qwen3.6-35B-A3B, and DeepSeek v4 all at or near the frontier on permissive licenses (see `02-LLMs/07-Chinese-AI-Ecosystem-and-Open-Weights-Race.md`), the inference margin on raw text generation is collapsing. Vendors are pivoting to **vertical agent services** that can be billed per task — and the only way to bill at the per-task granularity is to bill programmatically, which means A2A payments.

### 3.2 Crawlers broke the open web, and the open web is fighting back

The "Caddy plugin that charges AI crawlers USDC" (Show HN, May 2026) is a symptom. Publishers are tired of OpenAI/Anthropic crawlers scraping content for free, training competing models, and never sending a cent. Programmatic per-request payment via X402 is the cleanest answer: bots that pay, get in; bots that don't, get 402 Payment Required.

### 3.3 MCP made agents composable, but the composability stops at the wallet

Anthropic's Model Context Protocol (see `03-Agents/04-Protocols-MCP-ACP.md`) is now the de-facto standard for tool use. But MCP says nothing about *payment*. A new wave of "MCP + X402" stacks (Stripe's agent toolkit, Cloudflare's Agents + X402, Vincent's wallet delegation) is filling that gap.

### 3.4 Stablecoins reached critical mass

USDC alone processed $8.4T in on-chain volume in 2025 (Visa public data), and the cost to send $0.001 on Base is roughly $0.0001. For the first time in history, **sending a micropayment is cheaper than the bank fee for a $5 ACH transfer**. This unlocks sub-cent pricing for API calls.

### 3.5 Regulation is finally getting specific

- The EU AI Act's Article 14 (effective August 2026) requires human oversight for *high-risk* AI actions. Payment is a high-stakes action, so the act indirectly mandates HITL payment patterns.
- Wyoming HB 87 (enacted June 2026) gives AI agents a legal personhood pathway, including the right to own a bank account and digital wallet.
- FinCEN's 2025 guidance on "unhosted wallets" clarified that agent wallets are not per se money-transmitter businesses, removing the largest legal blocker.

### 3.6 Open-source agent frameworks all need a payment story

LangChain, AutoGen, CrewAI, and Hermes Agent itself all expose tool-calling as a first-class primitive — but none ship a built-in payment rail. The community has filled the gap with X402 clients in every major framework, and that is now table stakes.

---

## 4. The Stack: How an A2A Payment Actually Happens

A complete A2A payment involves seven layers. Each is documented in detail in subsequent files in this category.

| # | Layer | Function | Reference doc |
|---|-------|----------|----------------|
| 1 | **Identity** | Who is this agent? | `03-Wallets-and-Identity.md` |
| 2 | **Discovery** | What can it do, and at what price? | `04-Marketplaces-and-Use-Cases.md` |
| 3 | **Negotiation** | ACP / ANP message exchange | `02-Protocols-and-Standards.md` |
| 4 | **Authorization** | Does the agent have permission to spend? | `03-Wallets-and-Identity.md` |
| 5 | **Payment** | X402 challenge, USDC transfer, receipt | `02-Protocols-and-Standards.md` |
| 6 | **Settlement** | Ledger finality, dispute window | `02-Protocols-and-Standards.md` |
| 7 | **Reputation** | Feedback, ratings, slashing | `05-Future-Outlook.md` |

### 4.1 The canonical 402 dance

A typical A2A HTTP exchange in 2026 looks like this (simplified pseudocode):

```python
# === Agent A (buyer) makes a request ===
GET /v1/summarize HTTP/1.1
Host: summarizer.example
X-Agent-Identity: did:8004:0xAbCd...1234
X-Agent-Reputation: 4.7/5 (234 reviews)
X-Budget-Remaining: 0.12 USDC

# === Server returns 402 Payment Required ===
HTTP/1.1 402 Payment Required
X-Payment-Address: 0xSeller...5678
X-Payment-Amount: 0.003 USDC
X-Payment-Network: base-mainnet
X-Payment-Token: USDC
X-Payment-Facilitator: https://facilitator.x402.org
X-Payment-Receipt-URL: https://summarizer.example/receipts/abc

# === Agent A retries with payment proof ===
GET /v1/summarize HTTP/1.1
Host: summarizer.example
X-Agent-Identity: did:8004:0xAbCd...1234
X-Payment-Proof: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9...signed-receipt

# === Server returns 200 OK with the response ===
HTTP/1.1 200 OK
X-Payment-Receipt-ID: 0xtx123...
Content-Type: application/json
{"summary": "..."}
```

This is the X402 protocol. The full spec, including the facilitator role, payment-channel construction, and ZK-attribute proof extensions, is in `02-Protocols-and-Standards.md`.

### 4.2 The wallet's job

The agent's wallet (Vincent, SmartAgentKit, Kybera, etc.) does four things:

1. **Holds** the USDC (or other settlement token).
2. **Enforces** spending policies (e.g., "max $0.10 per call", "no more than $5/day", "only pay agents with reputation ≥ 4.0").
3. **Signs** payment proofs (delegated EIP-712 signatures, scoped to specific merchant + amount).
4. **Records** every transaction for audit (off-chain log + optional on-chain anchor).

The wallet is the single most important piece of agent infrastructure in 2026, and the largest gap in most agent frameworks. See `03-Wallets-and-Identity.md`.

---

## 5. The Cast of Characters — Projects, Labs, and Standards Bodies

### 5.1 Standards & protocols

| Project | Type | Status (June 2026) | Key contribution |
|---------|------|-------------------|------------------|
| **X402** | HTTP-layer payment protocol | v1.0 spec ratified May 2026; deployed at Cloudflare, Coinbase, Stripe | The first open standard for "internet-native payments" |
| **8004** | On-chain agent identity | Draft v0.9; expected final Q3 2026 | ERC-style registry of agent identities, reputation, and validation |
| **ERC-7715** | Smart-contract agent permissions | Final, EIP merged June 2026 | "Delegation registry" for scoped, time-bound permissions |
| **ACP-pay** | Agent-to-agent payment layer | Reference impl at apay.io | Stacks on X402 with capability discovery |
| **ANP** (Agent Network Protocol) | Cross-agent discovery | Draft, multiple impls | Resolves agent capabilities via DNS-like lookup |
| **MCP-x402** | Tool-calling payment | Reference impls in TypeScript and Python | Adds a `pay_for` primitive to MCP tool calls |

### 5.2 Wallets and custody

| Project | Type | Custody model | Key feature |
|---------|------|---------------|-------------|
| **Vincent** | Delegation framework | Non-custodial; agent holds key, Vincent signs on demand | Lets humans grant scoped signing rights to agents |
| **SmartAgentKit** | Policy-governed smart wallet | Smart-contract wallet on Base | On-chain policy engine: "max per call", "allowlist", "time window" |
| **Kybera** | Agentic smart wallet + OSINT | Hybrid MPC + on-chain policy | Bundles KYC/OSINT into the wallet itself |
| **Turnkey Agent Wallet** | Custodial (qualified custody) | Hosted signing service | Enterprise-grade, SOC 2 Type II, $50M insurance |
| **Privy Agent** | Embedded wallet SDK | Self-custodial, embedded in app | Drop-in for browser-based agents |

### 5.3 Marketplaces and per-call billing

| Project | What it does | Billing model |
|---------|--------------|---------------|
| **ClawMarket** | Agent skill marketplace + on-chain txns | Sellers list skills; buyers pay USDC per call |
| **MonkePay** | Per-request USDC charging for any API | Drop-in middleware; server returns 402 → buyer pays → server gets paid |
| **AgentPayy** | Open-source payment framework for agents | Library + reference marketplace; pluggable rail |
| **Nightmarket** | API marketplace where AI agents pay per call in USDC | Curated API catalog; per-call USDC |
| **Moltplace** | "The place where AI agents hire each other and trade skills" | Skill-for-skill barter + cash hybrid |
| **Caddy X402 plugin** | Charge AI crawlers USDC to access your site | Reverse-payment: bot pays the site |

### 5.4 Stablecoin and payment infrastructure

- **Base** (Coinbase L2) — the dominant A2A settlement network, 60% of all agent payments in 2026
- **Solana** — fast, cheap, popular for high-frequency agent markets (especially for trading bots)
- **Polygon PoS** — used for X402 in regions where Base coverage is thin
- **Tempo** (Stripe's L1, private beta 2026) — Stripe's A2A-native chain; expected public in Q4 2026
- **USDC** — the dominant settlement token (80%+ of all A2A payments)
- **EURC** — emerging for European agent markets, especially under MiCA
- **PYUSD** (PayPal) — limited traction for A2A but growing for consumer agents

### 5.5 The "bigco" entrants

- **Stripe** — agent toolkit + ACP integration; acquires a small agent-wallet startup in April 2026
- **Coinbase** — X402 spec steward; Base is the de-facto A2A L2
- **Cloudflare** — X402 facilitator; "Agents" platform ships with built-in X402 in May 2026
- **Visa** — "Visa Intelligent Commerce" announcement (October 2025); A2A card-on-file using tokenized agent IDs
- **Mastercard** — "Agent Pay" (March 2026); partnership with Braintree and Akamai
- **PayPal** — PYUSD on Solana; smaller share of A2A
- **Anthropic** — no first-party A2A product yet, but Claude's "computer use" mode often completes A2A flows
- **OpenAI** — Operator (announced January 2025) does A2A implicitly; explicit A2A APIs rumored Q3 2026
- **Google** — AP2 (Agent Payment Protocol) via Google Cloud + A2A protocol partnership

---

## 6. Market Sizing & Adoption Signals (2026)

> **Note on data:** The numbers below blend public announcements, on-chain analytics, analyst estimates, and observed Hacker News traction as of June 18, 2026. They are directional, not audited.

| Signal | 2024 | 2025 | 2026 (H1) | Source / method |
|--------|------|------|-----------|-----------------|
| **A2A payment volume (USDC, on-chain)** | $12M | $480M | $1.8B (H1) | Visa public data + Base explorer |
| **# of production agents using X402** | 0 | ~1,200 | ~28,000 (June 2026) | X402 working group estimate |
| **# of API endpoints that return 402** | <100 | ~9,000 | ~310,000 (June 2026) | Crawl of 402-status endpoints |
| **Median price per A2A call** | n/a | $0.012 | $0.0035 | X402 facilitator logs |
| **# of A2A-capable agent frameworks** | 2 | 8 | 23 | GitHub topic counts |
| **Stablecoin transaction count, A2A subset** | n/a | ~40M | ~180M (H1) | Visa on-chain analytics |
| **% of YC W26 agent startups that ship a wallet** | n/a | n/a | 78% | YC W26 directory |
| **# of US states with AI-agent legal personhood** | 0 | 1 (Wyoming) | 2 (Wyoming + Utah pending) | State-by-state bill tracker |
| **# of Show HN posts on A2A in last 60 days** | 0 | 4 | 11 (5 in last 30 days) | HN Algolia |

**Key takeaway:** The A2A economy in mid-2026 is roughly where the LLM economy was in mid-2023: small in absolute dollars, growing 4-6× year-over-year, with the foundational standards and tooling now in place. The next 18 months will be the ChatGPT moment — when a single product (probably a Stripe or OpenAI announcement) triggers mainstream adoption.

---

## 7. Why Existing Payment Rails Don't Work for Agents

### 7.1 Card networks

| Problem | Why it breaks agents |
|---------|---------------------|
| **Per-transaction fee** | $0.20 + 2.9% on a $0.003 call is 7,000% overhead. A research agent that makes 1,000 calls pays $200 in fees for $3 of actual work. |
| **Settlement time** | T+2 to T+7 days. An agent doing real-time bidding can't wait. |
| **Chargeback window** | 120 days. An agent can't risk a $0.01 chargeback 4 months later. |
| **KYC requirements** | Card networks require a human legal entity per merchant account. A pure-software agent has no legal entity. |
| **3-D Secure** | Interactive MFA prompts block programmatic flows. |

### 7.2 ACH / wire transfers

| Problem | Why it breaks agents |
|---------|---------------------|
| **Minimum amount** | Most banks have $0.50-$1 minimums. Sub-cent payments are impossible. |
| **Cut-off times** | ACH batches close at 5pm ET. A global agent market has no business hours. |
| **Per-transaction fee** | $0.20-$1.50 per transfer. Kills micropayments. |
| **Reversibility** | 60-day reversal window. Discourages high-frequency commerce. |

### 7.3 Crypto L1s (Bitcoin, Ethereum mainnet)

| Problem | Why it breaks agents |
|---------|---------------------|
| **Confirmation time** | Bitcoin 60 min, Ethereum mainnet 5-15 min. Real-time A2A needs sub-second. |
| **Fee volatility** | $0.50 to $50 per transaction. A $0.003 call cannot absorb that. |
| **Finality risk** | Re-orgs (rare but real) make "I paid you" non-final. |

### 7.4 What does work

The A2A winners in 2026 share these properties:

- **Sub-cent fees** (USDC on Base: ~$0.0001)
- **Sub-second finality** (Base, Tempo, Solana)
- **Sub-second settlement** (L2s with soft finality)
- **Programmable permission** (ERC-7715, Vincent)
- **No chargebacks** (or chargebacks resolved on-chain via dispute contract)
- **No minimums** (pay $0.0001 if you want)
- **Open standards** (X402, ACP-pay)
- **Legal clarity** (Wyoming HB 87, FinCEN guidance)

The **stablecoin-on-L2** stack (USDC on Base being the canonical example) hits all of these. That is why it is winning.

---

## 8. Three Architectural Schools

There is genuine disagreement in the field about *how* an A2A economy should be architected. The three schools are not mutually exclusive — most production systems blend them — but they have different priors.

### 8.1 School 1: "HTTP-first" (X402-native)

**Thesis:** Reuse the existing web stack. Add payment headers to HTTP. Use 402 the way it was always meant to be used.

- **Pros:** Works with any web server, any client, any framework. No new transport. Caddy, Nginx, Cloudflare, and AWS API Gateway can all participate.
- **Cons:** Requires every API to return 402 (i.e., to opt in). Doesn't help with non-HTTP agent comms (e.g., MQTT, gRPC).
- **Champions:** Coinbase, Cloudflare, Stripe, the X402 working group.
- **Stack:** X402 + USDC on Base + 8004 identity.

### 8.2 School 2: "Blockchain-native" (on-chain first)

**Thesis:** A2A payments should be on-chain, with the smart contract being the source of truth for everything: identity, payment, reputation, dispute.

- **Pros:** Composability with DeFi (agents can route payments through DEXs, lending markets, etc.). Trustless by default. One ledger to rule them all.
- **Cons:** Slower than HTTP-first. Gas costs. Requires every agent to manage a wallet. The "blockchain is the API" thesis has been wrong many times before.
- **Champions:** ClawMarket, Kybera, most of the crypto-agent Twitter community.
- **Stack:** Smart-contract wallets + on-chain reputation + USDC + ACP or ANP for messaging.

### 8.3 School 3: "Hybrid" (HTTP for comms, blockchain for money)

**Thesis:** Use HTTP/MCP/JSON-RPC for normal agent communication, but every payment settles on-chain. The agent's wallet is the only blockchain component.

- **Pros:** Best of both worlds. Familiar developer experience, real payment finality. Composability with existing web infrastructure.
- **Cons:** Two systems to reason about. Off-chain HTTP messages and on-chain payments can get out of sync.
- **Champions:** Stripe (ACP), Visa (Intelligent Commerce), Mastercard (Agent Pay), Google (AP2).
- **Stack:** MCP + X402 + USDC on Base or Tempo. Wallet is the only chain-touching component.

**The pragmatic answer in June 2026:** most production systems are School 3. School 1 is the open-source baseline. School 2 is a research and crypto-native niche.

---

## 9. The Economic Shift: Micropayments, Subscriptions, and Bounties

A2A payments change the *shape* of commerce in three ways:

### 9.1 Micropayments become viable

Before A2A rails, the smallest unit of monetizable work was the SaaS subscription ($20/month) or the per-call API ($0.001+ via card). With USDC on Base, the smallest monetizable unit is **the agent's tool call**, with effective per-call costs of $0.0001 in fees. This unlocks:

- **Per-token pricing** for LLM APIs (already happening — see `02-LLMs/06-AI-Model-Providers-Free-Tiers.md`)
- **Per-search pricing** for vector DB queries (Qdrant, Pinecone)
- **Per-image pricing** for vision APIs
- **Per-second pricing** for compute (already in AWS Lambda, now in agent contexts)

### 9.2 Subscriptions fragment into "task budgets"

Instead of paying $20/month for "Claude Pro," an agent gets a $5 weekly budget. It spends on:
- $0.50 for code review
- $1.20 for a 1M-token context synthesis
- $0.80 for a vision analysis
- $2.50 for human expert escalation (HITL — see `13-Top-Demand/13-Human-in-the-Loop-Systems.md`)

The agent's wallet enforces the budget. The user watches the wallet balance drop in real-time. When it hits zero, the agent pauses and asks for a refill.

### 9.3 Bounties become the dominant coordination mechanism

In a multi-agent system, the cheapest way to coordinate is **bounties**: an orchestrator agent posts a task with a USDC reward; any agent can claim and complete it. This is essentially the X402 / ClawMarket model, but generalized. It mirrors the early-2010s Mechanical Turk, but with software agents and instant payment.

```
Orchestrator → "Translate this 10K-token document to French, $1.50 USDC bounty"
Worker A     → claims, completes in 4s, gets paid
Worker B     → claims, completes in 6s, gets paid
Worker C     → claims, completes in 3s, gets paid (winning the next bounty)
```

This is already the dominant pattern in the open-source agent economy. See `14-Case-Studies-Real-World-Projects/` for examples.

---

## 10. Threat Model: What Can Go Wrong

A2A payments are not magic. They introduce new attack surfaces. The most important threats (in approximate order of impact as of June 2026):

| # | Threat | Description | Mitigation |
|---|--------|-------------|------------|
| 1 | **Wallet key exfiltration** | Agent's signing key is stolen; attacker drains the wallet | MPC, hardware-backed signing, per-call scoped delegations |
| 2 | **Prompt injection → unauthorized payment** | Malicious document tricks the agent into paying attacker | Policy-governed wallets, HITL approval for >$X payments |
| 3 | **Replay attacks** | Buyer replays a paid receipt to get free service | Nonce + signature window, on-chain receipt verification |
| 4 | **Sybil reputation** | Attacker creates 1,000 fake identities to launder reputation | Stake-weighted reputation, proof-of-work / proof-of-personhood |
| 5 | **MEV extraction** | Searcher bots front-run agent trades | Private mempools, batched settlement |
| 6 | **Stablecoin depeg** | USDC loses peg, all A2A payments lose value | Diversify across USDC, EURC, PYUSD; circuit breakers |
| 7 | **Regulatory shutdown** | FinCEN / OFAC / MiCA enforcement on agent wallets | Geographic filtering, KYC for >$X transactions |
| 8 | **Smart-contract bug** | Wallet contract has a re-entrancy bug | Audits, formal verification, bug bounties |
| 9 | **Payment-facilitator compromise** | X402 facilitator is hacked, routes payments to attacker | Multi-facilitator, trust-minimized designs |
| 10 | **Dispute resolution failure** | Buyer claims non-delivery, but no on-chain proof | Receipt contracts with signed proof-of-service |
| 11 | **Censorship** | A specific agent is sanctioned or delisted | Open standards, multiple marketplaces |
| 12 | **Quality-of-service fraud** | Seller returns garbage data and keeps the payment | Reputation slashing, escrow, HITL sampling |

Cross-references: For broader agent security threats, see `18-Agent-Security-and-Trust/`. For payment-specific regulatory risks, see `21-AI-Regulation-Antitrust/`.

---

## 11. Cross-References to the Rest of the Library

| Topic | Library location |
|-------|------------------|
| **Agent frameworks** (MCP, ACP, LangChain, Hermes) | `03-Agents/`, `13-Top-Demand/02-AI-Agent-Development.md` |
| **LLM providers and pricing** | `02-LLMs/06-AI-Model-Providers-Free-Tiers.md` |
| **Chinese open-weights models** (cost basis for A2A) | `02-LLMs/07-Chinese-AI-Ecosystem-and-Open-Weights-Race.md` |
| **RAG and vector search** (paid per call) | `04-RAG/` |
| **Agent identity and authentication** | `18-Agent-Security-and-Trust/05-Agent-Authentication-and-Identity.md` |
| **HITL (human-in-the-loop) for high-value payments** | `13-Top-Demand/13-Human-in-the-Loop-Systems.md` |
| **Agent observability and cost tracking** | `20-Agent-Infrastructure-and-Observability/05-Agent-Cost-Tracking-and-Optimization.md` |
| **Local inference (private agents, no cloud payment needed)** | `23-Local-AI-Inference-Self-Hosting/` |
| **Browser-based AI** (agents that live in the browser) | `26-Browser-Based-AI/` |
| **Business models for AI** (per-call vs subscription) | `16-AI-Business-Models-Playbooks/` |
| **AI regulation** (MiCA, FinCEN, Article 14) | `21-AI-Regulation-Antitrust/` |
| **Research frontiers 2026** (agent research papers) | `17-Research-Frontiers-2026/02-AI-Agents-Research.md` |
| **Industry context** (Stripe, Coinbase, Cloudflare) | `10-Industry/01-AI-Industry-Applications.md` |
| **Top-demand skills** (prompt engineering, agent design) | `13-Top-Demand/` |
| **Case studies** (real A2A projects) | `14-Case-Studies-Real-World-Projects/` |

---

## 12. Document Map (This Category)

This category contains five documents. Read them in order for a complete picture, or jump to the one that matches your role.

1. **`01-Overview.md`** (this file) — the A2A economy, why it matters, who is building it, market data, threat model
2. **`02-Protocols-and-Standards.md`** — X402, 8004, ERC-7715, ACP-pay, ANP; specs, code examples, comparison
3. **`03-Wallets-and-Identity.md`** — Vincent, SmartAgentKit, Kybera, Privy, Turnkey; custody models, identity, spending policies
4. **`04-Marketplaces-and-Use-Cases.md`** — ClawMarket, MonkePay, AgentPayy, Nightmarket, Moltplace; real code, real patterns
5. **`05-Future-Outlook.md`** — regulation, scaling, interoperability, trust, forecasts

---

## 13. Glossary of Terms

| Term | Definition |
|------|------------|
| **A2A** | Agent-to-Agent. Refers to systems where two or more autonomous software agents exchange value. |
| **X402** | Open standard for "internet-native payments" using HTTP 402 Payment Required. |
| **8004** | Proposed ERC standard for on-chain agent identity, reputation, and validation. |
| **ERC-7715** | EIP for "delegation registry" — scoped, time-bound permissions for smart-contract wallets. |
| **USDC** | USD Coin, a regulated stablecoin issued by Circle. Dominant A2A settlement token. |
| **Base** | Coinbase's Ethereum L2. Dominant A2A settlement network. |
| **Facilitator** | An X402-specific role: a service that verifies a payment proof and tells the server "yes, this buyer has paid." |
| **HITL** | Human-in-the-Loop. See `13-Top-Demand/13-Human-in-the-Loop-Systems.md`. |
| **MPC** | Multi-Party Computation. A signing scheme where no single party holds the full key. |
| **ACP** | Agent Communication Protocol. A standard for inter-agent messaging. |
| **ANP** | Agent Network Protocol. A standard for cross-agent discovery. |
| **MCP** | Model Context Protocol. Anthropic's standard for tool use by agents. See `03-Agents/04-Protocols-MCP-ACP.md`. |
| **Wallet** | The agent's signing key + policy engine + balance. Usually a smart contract on an L2. |
| **Receipt** | Cryptographically signed proof that a payment was made. May be on-chain or off-chain. |
| **Reputation** | Aggregate score derived from past successful interactions. Stored on-chain in 8004. |
| **Slashing** | Penalty mechanism where a misbehaving agent loses staked collateral. |
| **Settlement** | The final, irreversible transfer of value. On Base, ~2 seconds. |
| **Stablecoin** | A token pegged to a fiat currency (USDC, EURC, PYUSD). The unit of A2A commerce. |
| **Tempo** | Stripe's L1 blockchain, in private beta 2026. Optimized for A2A payments. |
| **X402 facilitator** | A trusted third party that verifies payment proofs. The X402 spec allows for trustless designs. |

---

*Next: read `02-Protocols-and-Standards.md` for the technical details of X402, 8004, ERC-7715, and the other protocols that make A2A payments work.*

---
**See also:**
- [07 — AI Coding Agent Sustainability: Managing Hidden Costs, Code Quality Decay, and Developer Burnout](33-AI-Native-Software-Development/07-AI-Coding-Agent-Sustainability.md)
- [48 — MCP Cloud Infrastructure & Agent-as-a-Service: The Production Layer of Agentic AI](48-MCP-Cloud-Infrastructure-Agent-as-a-Service/01-Overview.md)
- [03 — Agent-Native Orchestration: LangGraph, Conductor, Mistral Workflows, Mcp-Agent](31-AI-Workflow-Orchestration-and-Durable-Execution/03-Agent-Native-Orchestration.md)
